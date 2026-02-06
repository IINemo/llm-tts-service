from collections.abc import Iterable
from typing import Dict, List

import numpy as np

from .step_scorer_reward_base import CandidateScore, StepScorerBase


class StepScorerConfidence(StepScorerBase):
    def __init__(self):
        super().__init__()

    def score_candidates_detailed(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        result = []
        for candidate in candidates:
            # Handle case where other_data might be None or missing validity_score
            if candidate.other_data is None:
                validity_score = 0.0
            else:
                validity_score = candidate.other_data.get("validity_score", 0.0)

            if not isinstance(validity_score, Iterable):
                validity_score = [validity_score]

            claim_scores = np.asarray(validity_score)
            result.append(
                CandidateScore(
                    candidate_text=candidate,
                    claim_scores=claim_scores,
                    aggregate_scores={},
                    metadata={"scorer_type": "uncertainty"},
                )
            )

        return result
