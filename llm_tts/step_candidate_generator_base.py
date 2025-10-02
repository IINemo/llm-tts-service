from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
from abc import abstractmethod


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
    ):
        self.text = text
        self.token_ids = token_ids
        self.is_complete = is_complete
        self.is_trajectory_complete = is_trajectory_complete
        self.generation_scores = generation_scores
        self.raw_text = raw_text or text

    def __str__(self):
        return f"StepCandidate(text='{self.text[:50]}...', complete={self.is_complete})"



class StepCandidateGeneratorBase:
    """Base class for step candidate generator"""

    @abstractmethod
    def generate_candidates(self, request: List[Dict[str, str]]) -> List[StepCandidate]:
        """Generate candidates for a given trajectory"""
        pass
