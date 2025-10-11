from typing import Dict, List
import numpy as np
from collections.abc import Iterable

from .step_scorer_reward_base import CandidateScore, StepScorerBase


class StepScorerUncertainty(StepScorerBase):
    def __init__(self):
        super().__init__()

    def score_candidates_detailed(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        result = []
        for candidate in candidates:
            uncertainty_score = candidate.other_data["uncertainty_score"]
            
            if not isinstance(uncertainty_score, Iterable):
                uncertainty_score = [uncertainty_score]

            claim_scores = 1.0 - np.asarray(uncertainty_score)
            result.append(
                CandidateScore(
                    candidate_text=candidate,
                    claim_scores=claim_scores,
                    aggregate_scores={},
                    metadata={"scorer_type": "uncertainty"},
                )
            )
            
        return result
