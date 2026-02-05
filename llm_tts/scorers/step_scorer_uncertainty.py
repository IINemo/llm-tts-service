import logging
from collections.abc import Iterable
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

from .step_scorer_reward_base import CandidateScore, StepScorerBase


class StepScorerUncertainty(StepScorerBase):
    def __init__(self):
        super().__init__()

    def score_candidates_detailed(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        result = []
        for candidate in candidates:
            if not candidate.other_data or "validity_score" not in candidate.other_data:
                raise ValueError(
                    f"Candidate missing 'validity_score' in other_data. "
                    f"Ensure generator is configured with an uncertainty estimator."
                )
            validity_score = candidate.other_data["validity_score"]

            if validity_score is None:
                log.warning(
                    "Candidate has validity_score=None — no estimator configured? "
                    "Using neutral score 0.5."
                )
                validity_score = [0.5]
            elif not isinstance(validity_score, Iterable):
                validity_score = [validity_score]

            # since we're using validity score, we need to convert it to uncertainty score
            claim_scores = np.asarray(validity_score, dtype=float)
            result.append(
                CandidateScore(
                    candidate_text=candidate,
                    claim_scores=claim_scores,
                    aggregate_scores={},
                    metadata={"scorer_type": "uncertainty"},
                )
            )

        return result
