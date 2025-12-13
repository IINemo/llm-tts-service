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
            if hasattr(candidate, "other_data") and candidate.other_data:
                uncertainty_score = candidate.other_data["uncertainty_score"]
            else:
                # in vllm, the uncertainty score is stored in the generation_scores dictionary
                # TODO: need to reimplement
                uncertainty_score = candidate.generation_scores['perplexity']
            if not isinstance(uncertainty_score, Iterable):
                uncertainty_score = [uncertainty_score]

            claim_scores = np.asarray(uncertainty_score)
            result.append(
                CandidateScore(
                    candidate_text=candidate,
                    claim_scores=claim_scores,
                    aggregate_scores={},
                    metadata={"scorer_type": "uncertainty"},
                )
            )

        return result
