from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


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


def covert_trajectory_to_string(trajectory: List[StepCandidate]) -> str:
    """Convert trajectory to string"""
    return "\n".join([step.text for step in trajectory])

class StepCandidateGeneratorBase:
    """Base class for step candidate generator"""

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
