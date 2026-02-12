import logging
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from llm_tts.utils.flops import FLOPCalculator

log = logging.getLogger(__name__)


class CompletionReason(str, Enum):
    """Reason why a trajectory was marked as complete."""

    THINKING_COMPLETE = "thinking_complete"  # </think> found in thinking mode
    EOS_PATTERN = "eos_pattern"  # <end of response> pattern matched
    ANSWER_PATTERN = "answer_pattern"  # <Answer>: or similar pattern matched
    CONTEXT_LIMIT = "context_limit"  # Not enough context for next step + answer


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
        other_data: Dict[str, Any] = None,
        is_thinking_complete: bool = False,
    ):
        self.text = text
        self.token_ids = token_ids
        self.is_complete = is_complete
        self.is_trajectory_complete = is_trajectory_complete
        self.is_thinking_complete = is_thinking_complete
        self.generation_scores = generation_scores
        self.raw_text = raw_text or text
        self.other_data = other_data

    def __str__(self):
        return f"StepCandidate(text='{self.text[:50]}...', complete={self.is_complete})"


def convert_trajectory_to_string(trajectory: List[StepCandidate]) -> str:
    """Convert trajectory to string.

    Each step.text should already end with newline, so we just concatenate.
    Uses raw_text if available to preserve original model output.
    """
    return "".join([step.raw_text or step.text for step in trajectory])


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

        # Per-sample statistics for batch strategies (keyed by sample_id)
        self._per_sample_stats: Dict[Any, Dict[str, int]] = {}

        # Cumulative statistics (across all samples)
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_samples: int = 0

    def reset_sample_stats(self) -> None:
        """Reset per-sample statistics. Call at start of each new sample."""
        self._sample_input_tokens = 0
        self._sample_output_tokens = 0
        self._sample_generation_count = 0

    def reset_per_sample_stats(self) -> None:
        """Reset per-sample statistics dict. Call at start of a batch."""
        self._per_sample_stats.clear()

    def record_sample_tokens(
        self, sample_id: Any, candidates: List[StepCandidate], context_tokens: int = 0
    ) -> None:
        """Record tokens for a specific sample. Accumulates across calls.

        Args:
            sample_id: Identifier for the sample (e.g., index in batch).
            candidates: List of generated candidates for this sample.
            context_tokens: Number of context tokens (prompt + trajectory).
        """
        if sample_id not in self._per_sample_stats:
            self._per_sample_stats[sample_id] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "generation_count": 0,
            }
        out = sum(
            (
                c.other_data.get("original_token_count", len(c.token_ids))
                if c.other_data
                else len(c.token_ids)
            )
            for c in candidates
        )
        self._per_sample_stats[sample_id]["input_tokens"] += context_tokens
        self._per_sample_stats[sample_id]["output_tokens"] += out
        self._per_sample_stats[sample_id]["generation_count"] += 1

    def get_sample_stats_for(self, sample_id: Any) -> Dict[str, Any]:
        """Get token_stats dict for a specific sample (same schema as get_sample_stats).

        Args:
            sample_id: Identifier for the sample.

        Returns:
            Dictionary with token counts and FLOP estimates for the given sample.
        """
        raw = self._per_sample_stats.get(
            sample_id, {"input_tokens": 0, "output_tokens": 0, "generation_count": 0}
        )
        total = raw["input_tokens"] + raw["output_tokens"]
        stats = {
            "input_tokens": raw["input_tokens"],
            "output_tokens": raw["output_tokens"],
            "total_tokens_this_sample": total,
            "generation_count": raw["generation_count"],
            "tflops": (
                self.flop_calculator.compute_tflops(total)
                if self.flop_calculator
                else None
            ),
        }
        return stats

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

    def get_sample_stats(self) -> Dict[str, Any]:
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

    def get_total_stats(self) -> Dict[str, Any]:
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
    def generate_step_candidates_batch(
        self,
        requests: List[List[Dict[str, str]]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
        stop_tokens_override=None,
        max_tokens=None,
        compute_uncertainty: bool = True,
        sample_ids=None,
        beam_ids=None,
    ) -> List[List[StepCandidate]]:
        """Generate N candidate next steps for each trajectory.

        Primary generation method. Each trajectory can have its own request.

        Args:
            requests: Per-trajectory chat messages.
            trajectories: List of trajectories (each a list of StepCandidates).
            candidates_per_step: Number of candidates per trajectory.
            stop_tokens_override: Override stop tokens (None = use defaults).
            max_tokens: Override max tokens (None = use defaults).
            compute_uncertainty: Whether to compute uncertainty scores.
            sample_ids: Optional per-trajectory sample IDs for token tracking.
            beam_ids: Optional per-trajectory beam IDs for logging.

        Returns:
            List of candidate lists, one per trajectory.
        """
        pass

    def generate_step_candidates(
        self,
        request: List[Dict[str, str]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
        compute_uncertainty: bool = True,
    ) -> List[List[StepCandidate]]:
        """Convenience wrapper: same request for all trajectories.

        Delegates to generate_step_candidates_batch with broadcast request.
        """
        requests = [request] * len(trajectories)
        return self.generate_step_candidates_batch(
            requests,
            trajectories,
            candidates_per_step,
            compute_uncertainty=compute_uncertainty,
        )

    @abstractmethod
    def generate_answer_candidates_batch(
        self,
        requests: List[List[Dict[str, str]]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
    ) -> List[List[StepCandidate]]:
        """Generate answer candidates for multiple trajectories.

        Primary answer generation method. Each trajectory can have its own request.

        Args:
            requests: Per-trajectory chat messages.
            trajectories: Per-trajectory step lists.
            candidates_per_step: Number of answer candidates per trajectory.

        Returns:
            List of candidate lists, one per trajectory.
        """
        pass

    def generate_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Convenience wrapper: single trajectory answer generation.

        Delegates to generate_answer_candidates_batch with single item.
        """
        results = self.generate_answer_candidates_batch(
            [request], [trajectory], candidates_per_step
        )
        return results[0] if results else []

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
