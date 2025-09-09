"""
Base scoring interface for online best-of-n step evaluation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
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
            self.aggregate_scores.update({
                'mean': np.mean(self.claim_scores),
                'max': np.max(self.claim_scores),
                'min': np.min(self.claim_scores),
                'median': np.median(self.claim_scores),
                'std': np.std(self.claim_scores),
                'count': len(self.claim_scores)
            })
        else:
            self.aggregate_scores.update({
                'mean': 0.5, 'max': 0.5, 'min': 0.5, 
                'median': 0.5, 'std': 0.0, 'count': 0
            })
    
    def get_score(self, method: str = 'mean') -> float:
        """Get aggregated score using specified method"""
        return self.aggregate_scores.get(method, 0.5)


class StepScorer(ABC):
    """Abstract base class for scoring step candidates in real-time"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def score_candidates_detailed(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
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
        aggregation: str = 'mean',
        **kwargs
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
        detailed_scores = self.score_candidates_detailed(trajectory, candidates, **kwargs)
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


class UncertaintyBasedScorer(StepScorer):
    """Base class for uncertainty-based step scoring (lower uncertainty = higher score)"""
    
    def __init__(self, name: str, invert_scores: bool = True):
        super().__init__(name)
        self.invert_scores = invert_scores  # Convert uncertainty to score
        
    @abstractmethod 
    def compute_claim_uncertainties(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
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
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[CandidateScore]:
        """Score candidates with detailed claim-level information"""
        claim_uncertainties_list = self.compute_claim_uncertainties(trajectory, candidates, **kwargs)
        
        detailed_scores = []
        for i, candidate in enumerate(candidates):
            claim_uncertainties = claim_uncertainties_list[i] if i < len(claim_uncertainties_list) else [0.5]
            
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
                metadata={'scorer_type': 'uncertainty', 'raw_uncertainties': claim_uncertainties}
            )
            detailed_scores.append(candidate_score)
            
        return detailed_scores


class RewardBasedScorer(StepScorer):
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


class EnsembleScorer(StepScorer):
    """Ensemble multiple scorers with weighted combination"""
    
    def __init__(
        self, 
        scorers: List[StepScorer],
        weights: Optional[List[float]] = None,
        aggregation: str = "weighted_mean"
    ):
        self.scorers = scorers
        self.weights = weights or [1.0] * len(scorers)
        self.aggregation = aggregation
        
        if len(self.weights) != len(self.scorers):
            raise ValueError("Number of weights must match number of scorers")
            
        super().__init__(f"Ensemble({', '.join(s.name for s in scorers)})")
        
    def prepare_model(self):
        """Prepare all constituent models"""
        for scorer in self.scorers:
            scorer.prepare_model()
            
    def cleanup(self):
        """Cleanup all constituent models"""
        for scorer in self.scorers:
            scorer.cleanup()
            
    def score_candidates_detailed(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[CandidateScore]:
        """Combine detailed scores from multiple scorers"""
        all_detailed_scores = []
        
        for scorer in self.scorers:
            try:
                detailed_scores = scorer.score_candidates_detailed(trajectory, candidates, **kwargs)
                all_detailed_scores.append(detailed_scores)
            except Exception as e:
                log.warning(f"Scorer {scorer.name} failed: {e}")
                fallback_scores = [
                    CandidateScore(
                        candidate_text=candidate,
                        claim_scores=[0.5],
                        aggregate_scores={},
                        metadata={'scorer_type': 'fallback', 'error': str(e)}
                    )
                    for candidate in candidates
                ]
                all_detailed_scores.append(fallback_scores)
                
        if not all_detailed_scores:
            return [
                CandidateScore(candidate_text=candidate, claim_scores=[0.5], aggregate_scores={})
                for candidate in candidates
            ]
            
        # Combine detailed scores
        combined_detailed_scores = []
        for i in range(len(candidates)):
            # Get all scorer results for this candidate
            candidate_scores_from_all_scorers = [
                detailed_scores[i] for detailed_scores in all_detailed_scores
            ]
            
            # Combine scores using weighted aggregation
            combined_aggregate_score = 0.0
            total_weight = 0.0
            
            for weight, candidate_score in zip(self.weights, candidate_scores_from_all_scorers):
                score = candidate_score.get_score('mean')  # Use mean aggregation
                combined_aggregate_score += weight * score
                total_weight += weight
                
            if total_weight > 0:
                combined_aggregate_score /= total_weight
            
            # Create ensemble candidate score
            ensemble_score = CandidateScore(
                candidate_text=candidates[i],
                claim_scores=[combined_aggregate_score],
                aggregate_scores={},
                metadata={
                    'scorer_type': 'ensemble',
                    'component_scores': [cs.get_score('mean') for cs in candidate_scores_from_all_scorers],
                    'weights': self.weights
                }
            )
            combined_detailed_scores.append(ensemble_score)
            
        return combined_detailed_scores