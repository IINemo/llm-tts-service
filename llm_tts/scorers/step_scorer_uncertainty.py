from typing import Dict, List

from .step_scorer_reward_base import CandidateScore, StepScorerBase


class StepScorerUncertainty(StepScorerBase):
    def __init__(self):
        super().__init__()

    def score_candidates_detailed(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        return [
            CandidateScore(
                candidate_text=candidate,
                claim_scores=(1. - candidate.other_data["uncertainty_score"]),
                aggregate_scores={},
                metadata={"scorer_type": "uncertainty"},
            )
            for candidate in candidates
        ]
