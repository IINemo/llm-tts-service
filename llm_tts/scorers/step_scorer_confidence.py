from collections.abc import Iterable
import logging
from typing import Dict, List

import numpy as np

from .step_scorer_reward_base import CandidateScore, StepScorerBase

log = logging.getLogger(__name__)


class StepScorerConfidence(StepScorerBase):
    def __init__(self):
        super().__init__()

    def score_candidates_detailed(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        result = []
        for candidate in candidates:
            validity_score = candidate.other_data["validity_score"]

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

    def score_trajectory(
        self,
        chat: List[Dict[str, str]],
        trajectory: List,
        **kwargs,
    ) -> List[float]:
        if not trajectory:
            return []

        step_scores = []
        for step in trajectory:
            if hasattr(step, "other_data") and step.other_data:
                if "validity_score" in step.other_data:
                    step_scores.append(float(step.other_data["validity_score"]))

        if step_scores:
            return step_scores

        trajectory_validity_score = kwargs.get("trajectory_validity_score")
        if trajectory_validity_score is not None:
            return [float(trajectory_validity_score)]

        trajectory_uncertainty_score = kwargs.get("trajectory_uncertainty_score")
        if trajectory_uncertainty_score is not None:
            return [1.0 / (1.0 + float(trajectory_uncertainty_score))]

        log.warning(
            "No validity or uncertainty score provided for trajectory; defaulting to 0.5"
        )
        return [0.5]
