from abc import abstractmethod
from typing import List

from .step_scorer_base import CandidateScore, StepScorerBase


class UncertaintyBasedScorer(StepScorerBase):
    """Base class for uncertainty-based step scoring (lower uncertainty = higher score)"""

    def __init__(self, name: str, invert_scores: bool = True):
        super().__init__(name)
        self.invert_scores = invert_scores  # Convert uncertainty to score

    @abstractmethod
    def compute_claim_uncertainties(
        self, trajectory: str, candidates: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Compute uncertainty scores for individual claims in each candidate

        Args:
            trajectory: Current trajectory text
            candidates: List of candidate next step texts

        Returns:
            List of lists - for each candidate, list of uncertainty values for its claims
        """
        pass

    def score_candidates_detailed(
        self, trajectory: str, candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        """Score candidates with detailed claim-level information"""
        claim_uncertainties_list = self.compute_claim_uncertainties(
            trajectory, candidates, **kwargs
        )

        detailed_scores = []
        for i, candidate in enumerate(candidates):
            claim_uncertainties = (
                claim_uncertainties_list[i]
                if i < len(claim_uncertainties_list)
                else [0.5]
            )

            # Convert uncertainties to confidence scores if requested
            if self.invert_scores:
                claim_scores = [1.0 - u for u in claim_uncertainties]
            else:
                claim_scores = claim_uncertainties

            # Create aggregate scores dict
            aggregate_scores = {}

            candidate_score = CandidateScore(
                candidate_text=candidate,
                claim_scores=claim_scores,
                aggregate_scores=aggregate_scores,
                metadata={
                    "scorer_type": "uncertainty",
                    "raw_uncertainties": claim_uncertainties,
                },
            )
            detailed_scores.append(candidate_score)

        return detailed_scores
