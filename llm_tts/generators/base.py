import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from llm_tts.utils.flops import FLOPCalculator

log = logging.getLogger(__name__)


@dataclass
class StepCandidate:
    """Represents a candidate next step in trajectory"""

    def __init__(
        self,
        text: str,
        token_ids: List[int],
        is_complete: bool,
        is_trajectory_complete: bool,
        generation_scores: Optional[torch.Tensor] = None,
        raw_text: str = None,
        other_data: Dict[str, any] = None,
    ):
        self.text = text
        self.token_ids = token_ids
        self.is_complete = is_complete
        self.is_trajectory_complete = is_trajectory_complete
        self.generation_scores = generation_scores
        self.raw_text = raw_text or text
        self.other_data = other_data

    def __str__(self):
        return f"StepCandidate(text='{self.text[:50]}...', complete={self.is_complete})"


def convert_trajectory_to_string(trajectory: List[StepCandidate]) -> str:
    """Convert trajectory to string.

    Each step.text should already end with newline, so we just concatenate.
    """
    return "".join([step.text for step in trajectory])


class StepCandidateGeneratorBase:
    """Base class for step candidate generator.

    Provides token and FLOP tracking for all generator implementations.
    """

    def __init__(
        self,
        generation_batch_size: int,
        flop_calculator: Optional["FLOPCalculator"] = None,
    ):
        self.generation_batch_size = generation_batch_size
        self.flop_calculator = flop_calculator

        # Per-sample statistics (reset at start of each sample)
        self._sample_input_tokens: int = 0  # Context tokens (prompt + trajectory)
        self._sample_output_tokens: int = 0  # Generated tokens (all candidates)
        self._sample_generation_count: int = 0  # Number of generate calls

        # Cumulative statistics (across all samples)
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_samples: int = 0

    def reset_sample_stats(self) -> None:
        """Reset per-sample statistics. Call at start of each new sample."""
        self._sample_input_tokens = 0
        self._sample_output_tokens = 0
        self._sample_generation_count = 0

    def _record_generation(
        self,
        candidates: List[StepCandidate],
        context_tokens: int = 0,
    ) -> None:
        """Record token counts from generated candidates.

        Args:
            candidates: List of generated candidates
            context_tokens: Number of context tokens (prompt + trajectory).
                           With prefix caching, this is processed once per step.

        Called automatically after each generation. Subclasses can override
        to add custom tracking.
        """
        if not candidates:
            return

        output_tokens = sum(len(c.token_ids) for c in candidates)
        self._sample_input_tokens += context_tokens  # Context processed once per step
        self._sample_output_tokens += output_tokens
        self._sample_generation_count += 1

        log.debug(
            f"Recorded generation: context={context_tokens}, output={output_tokens} "
            f"from {len(candidates)} candidates "
            f"(sample total: input={self._sample_input_tokens}, output={self._sample_output_tokens})"
        )

    def finalize_sample_stats(self, num_samples: int = 1) -> None:
        """Finalize sample statistics. Call at end of each sample."""
        self._total_input_tokens += self._sample_input_tokens
        self._total_output_tokens += self._sample_output_tokens
        self._total_samples += num_samples

    def get_sample_stats(self) -> Dict[str, any]:
        """Get statistics for current sample.

        Returns:
            Dictionary with token counts and FLOP estimates.
            - input_tokens: Context tokens processed (prompt + trajectory)
            - output_tokens: Generated tokens (all candidates)
            - total_tokens_this_sample: Sum of input and output tokens
            - generation_count: Number of generation calls (num_steps * candidates_per_step)
            - tflops: Estimated TFLOPs based on total tokens
        """
        total_tokens = self._sample_input_tokens + self._sample_output_tokens
        stats = {
            "input_tokens": self._sample_input_tokens,
            "output_tokens": self._sample_output_tokens,
            "total_tokens_this_sample": total_tokens,
            "generation_count": self._sample_generation_count,
        }

        if self.flop_calculator is not None:
            stats["tflops"] = self.flop_calculator.compute_tflops(total_tokens)
        else:
            stats["tflops"] = None

        return stats

    def get_total_stats(self) -> Dict[str, any]:
        """Get cumulative statistics across all samples.

        Returns:
            Dictionary with total token counts and FLOP estimates.
        """
        total_tokens = self._total_input_tokens + self._total_output_tokens
        stats = {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": total_tokens,
            "total_samples": self._total_samples,
            "avg_tokens_per_sample": (
                total_tokens / self._total_samples if self._total_samples > 0 else 0
            ),
        }

        if self.flop_calculator is not None:
            stats["total_tflops"] = self.flop_calculator.compute_tflops(total_tokens)
        else:
            stats["total_tflops"] = None

        return stats

    @abstractmethod
    def generate_step_candidates(
        self, request: List[Dict[str, str]], trajectory: List[StepCandidate]
    ) -> List[StepCandidate]:
        """Generate N candidate next steps for a given trajectory."""
        pass

    @abstractmethod
    def generate_answer_candidates(
        self, request: List[Dict[str, str]], trajectory: List[StepCandidate]
    ) -> List[StepCandidate]:
        """Generate answer for a given trajectory"""
        pass

    def __call__(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate candidates for a given trajectory.

        Automatically records token statistics after generation.
        """
        if self.generation_batch_size < candidates_per_step:
            candidates = self._generate_step_candidates_in_batches(
                request,
                trajectory=trajectory,
                candidates_per_step=candidates_per_step,
            )
        else:
            candidates = self.generate_step_candidates(
                request,
                trajectory=trajectory,
                candidates_per_step=candidates_per_step,
            )
            # Record tokens (batch generation records per-batch)
            self._record_generation(candidates)

        return candidates

    def _generate_step_candidates_in_batches(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List:
        """Generate step candidates in smaller batches to avoid OOM.

        Records token statistics for each batch.
        """
        all_candidates = []

        # Calculate number of batches needed
        num_batches = (
            candidates_per_step + self.generation_batch_size - 1
        ) // self.generation_batch_size

        for batch_idx in range(num_batches):
            # Calculate batch size for this iteration
            start_idx = batch_idx * self.generation_batch_size
            end_idx = min(
                (batch_idx + 1) * self.generation_batch_size,
                candidates_per_step,
            )
            batch_size = end_idx - start_idx

            log.info(
                f"Generating batch {batch_idx+1}/{num_batches} ({batch_size} candidates)"
            )

            # Generate batch
            batch_candidates = self.generate_step_candidates(
                request, trajectory=trajectory, candidates_per_step=batch_size
            )
            if batch_candidates:
                all_candidates.extend(batch_candidates)
                # Record tokens for this batch
                self._record_generation(batch_candidates)

            # Clear GPU cache after each batch
            torch.cuda.empty_cache()

        return all_candidates
