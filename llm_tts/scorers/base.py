"""
Base scoring interface for online best-of-n step evaluation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import logging

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


class StepScorer(ABC):
    """Abstract base class for scoring step candidates in real-time"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def score_candidates_detailed(
        self, trajectory: str, candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        """
        Score candidates with detailed information

        Args:
            trajectory: Current trajectory text
            candidates: List of candidate next step texts
            **kwargs: Additional scoring parameters

        Returns:
            List of CandidateScore objects with detailed scoring info
        """
        pass

    def score_candidates(
        self,
        trajectory: str,
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
        detailed_scores = self.score_candidates_detailed(
            trajectory, candidates, **kwargs
        )
        return [score.get_score(aggregation) for score in detailed_scores]

    @abstractmethod
    def prepare_model(self):
        """Prepare/load the scoring model if needed"""
        pass

    def cleanup(self):
        """Cleanup resources (default no-op)"""
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class RewardBasedScorer(StepScorer):
    """Base class for reward-based step scoring (higher reward = higher score)"""

    @abstractmethod
    def compute_claim_rewards(
        self, trajectory: str, candidates: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Compute reward scores for individual claims in each candidate

        Args:
            trajectory: Current trajectory text
            candidates: List of candidate next step texts

        Returns:
            List of lists - for each candidate, list of reward values for its claims
        """
        pass

    def score_candidates_detailed(
        self, trajectory: str, candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        """Score candidates with detailed claim-level information"""
        claim_rewards_list = self.compute_claim_rewards(
            trajectory, candidates, **kwargs
        )

        detailed_scores = []
        for i, candidate in enumerate(candidates):
            claim_rewards = (
                claim_rewards_list[i] if i < len(claim_rewards_list) else [0.0]
            )

            candidate_score = CandidateScore(
                candidate_text=candidate,
                claim_scores=claim_rewards,  # Rewards are already scores (higher = better)
                aggregate_scores={},
                metadata={"scorer_type": "reward", "raw_rewards": claim_rewards},
            )
            detailed_scores.append(candidate_score)

        return detailed_scores
