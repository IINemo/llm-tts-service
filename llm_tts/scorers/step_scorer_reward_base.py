from .step_scorer_base import StepScorerBase, CandidateScore

from typing import List
from abc import abstractmethod


class StepScorerRewardBase(StepScorerBase):
    """Base class for reward-based step scoring (higher reward = higher score)"""
    
    @abstractmethod
    def compute_claim_rewards(
        self,
        trajectory: str,
        candidates: List[str], 
        **kwargs
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
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[CandidateScore]:
        """Score candidates with detailed claim-level information"""
        claim_rewards_list = self.compute_claim_rewards(trajectory, candidates, **kwargs)
        
        detailed_scores = []
        for i, candidate in enumerate(candidates):
            claim_rewards = claim_rewards_list[i] if i < len(claim_rewards_list) else [0.0]
            
            candidate_score = CandidateScore(
                candidate_text=candidate,
                claim_scores=claim_rewards,  # Rewards are already scores (higher = better)
                aggregate_scores={},
                metadata={'scorer_type': 'reward', 'raw_rewards': claim_rewards}
            )
            detailed_scores.append(candidate_score)
            
        return detailed_scores
