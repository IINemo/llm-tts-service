"""
Base scoring interface for online best-of-n step evaluation
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    """Detailed scoring information for a single candidate step"""

    candidate_text: str
    claim_scores: List[float]  # Individual claim/sub-step scores
    aggregate_scores: Dict[str, float]  # Different aggregation methods
    metadata: Dict[str, Any] = None  # Additional scoring info

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        if self.claim_scores:
            self.aggregate_scores.update(
                {
                    "mean": np.mean(self.claim_scores),
                    "max": np.max(self.claim_scores),
                    "min": np.min(self.claim_scores),
                    "median": np.median(self.claim_scores),
                    "std": np.std(self.claim_scores),
                    "count": len(self.claim_scores),
                }
            )
        else:
            self.aggregate_scores.update(
                {
                    "mean": 0.5,
                    "max": 0.5,
                    "min": 0.5,
                    "median": 0.5,
                    "std": 0.0,
                    "count": 0,
                }
            )

    def get_score(self, method: str = "mean") -> float:
        """Get aggregated score using specified method"""
        return self.aggregate_scores.get(method, 0.5)


class StepScorerBase(ABC):
    """Abstract base class for scoring step candidates in real-time"""

    def __init__(self, name: str = "base_scorer"):
        """Initialize with scorer name."""
        self.name = name

    @abstractmethod
    def score_candidates_detailed(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        """
        Score candidates with detailed information

        Args:
            chat: Current chat
            candidates: List of candidate next step texts
            **kwargs: Additional scoring parameters

        Returns:
            List of CandidateScore objects with detailed scoring info
        """
        pass

    def score_candidates(
        self,
        chat,
        candidates: List[str],
        aggregation: str = "mean",
        **kwargs,
    ) -> List[float]:
        """
        Score candidates and return simple aggregated scores

        Args:
            trajectory: Current trajectory text
            candidates: List of candidate next step texts
            aggregation: How to aggregate claim scores ('mean', 'max', 'min', etc.)
            **kwargs: Additional scoring parameters

        Returns:
            List of scores (higher = better) for each candidate
        """
        detailed_scores = self.score_candidates_detailed(chat, candidates, **kwargs)
        return [score.get_score(aggregation) for score in detailed_scores]

    def score_trajectory(
        self,
        chat: List[Dict[str, str]],
        trajectory: List,
        **kwargs,
    ) -> List[float]:
        """
        Score a complete trajectory and return scores for all steps.

        Must be implemented by subclasses that support trajectory scoring
        (e.g., PRM scorer for offline Best-of-N).

        Args:
            chat: Chat messages (contains the question)
            trajectory: List of step candidates

        Returns:
            List of scores, one per step
        """
        raise NotImplementedError("score_trajectory must be implemented by subclass")

    def score_trajectories_batch(
        self,
        chats: List[List[Dict[str, str]]],
        trajectories: List[List],
        sample_ids: List[int] = None,
        trajectory_ids: List[int] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Score multiple trajectories in a single batch call for efficiency.

        Default implementation falls back to sequential scoring.
        Subclasses (e.g., PRM scorer with vLLM) can override for true batching.

        Args:
            chats: List of chat messages (one per trajectory, contains question)
            trajectories: List of trajectories (each is a list of steps)
            sample_ids: Optional list of sample indices for logging
            trajectory_ids: Optional list of trajectory indices within each sample

        Returns:
            List of score lists, one per trajectory
        """
        # Default: sequential fallback
        return [
            self.score_trajectory(chat, traj, **kwargs)
            for chat, traj in zip(chats, trajectories)
        ]

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
