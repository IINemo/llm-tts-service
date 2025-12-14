import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

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
    """Convert trajectory to string"""
    return "\n".join([step.text for step in trajectory])


class StepCandidateGeneratorBase:
    """Base class for step candidate generator"""

    def __init__(self, generation_batch_size: int):
        self.generation_batch_size = generation_batch_size

    @abstractmethod
    def generate_candidates(
        self, request: List[Dict[str, str]], trajectory: List[StepCandidate]
    ) -> List[StepCandidate]:
        """Generate candidates for a given trajectory"""
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
        """Generate candidates for a given trajectory"""

        if self.generation_batch_size < candidates_per_step:
            candidates = self._generate_candidates_in_batches(
                request,
                trajectory=trajectory,
                candidates_per_step=candidates_per_step,
            )
        else:
            candidates = self.generate_candidates(
                request,
                trajectory=trajectory,
                candidates_per_step=candidates_per_step,
            )

        return candidates

    def _generate_candidates_in_batches(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List:
        """Generate candidates in smaller batches to avoid OOM"""

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
            batch_candidates = self.generate_candidates(
                request, trajectory=trajectory, candidates_per_step=batch_size
            )
            if batch_candidates:
                all_candidates.extend(batch_candidates)

            # Clear GPU cache after each batch
            torch.cuda.empty_cache()

        return all_candidates
