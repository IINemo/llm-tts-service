"""
Step scorer that uses pre-computed generation scores.

Works with vLLM step generators that compute perplexity/entropy during generation.
"""

from typing import Dict, List

import numpy as np

from llm_tts.generators.base import StepCandidate
from .step_scorer_reward_base import CandidateScore, StepScorerBase


class StepScorerGeneration(StepScorerBase):
    """
    Scorer that uses generation_scores from StepCandidate.

    Supports:
    - perplexity: Lower is better (less uncertain)
    - mean_entropy: Lower is better (less uncertain)

    Converts to validity score where higher = better.
    """

    def __init__(self, score_type: str = "mean_entropy"):
        """
        Args:
            score_type: Which generation score to use ("perplexity" or "mean_entropy")
        """
        super().__init__()
        self.score_type = score_type

    def score_candidates(
        self, chat: List[Dict[str, str]], candidates: List[StepCandidate], **kwargs
    ) -> List[float]:
        """
        Score candidates using their pre-computed generation scores.

        Returns validity scores where higher = better (more confident).
        """
        scores = []
        for candidate in candidates:
            gen_scores = getattr(candidate, "generation_scores", {})
            raw_score = gen_scores.get(self.score_type, 0.0)

            # Convert to validity: lower uncertainty = higher validity
            # Use negative exponential to map to [0, 1] range
            # perplexity/entropy typically ranges from 0 to ~10+
            validity = np.exp(-raw_score / 5.0)  # Scale factor of 5
            scores.append(float(validity))

        return scores

    def score_candidates_detailed(
        self, chat: List[Dict[str, str]], candidates: List[StepCandidate], **kwargs
    ) -> List[CandidateScore]:
        """Score candidates with detailed output."""
        result = []
        for candidate in candidates:
            gen_scores = getattr(candidate, "generation_scores", {})
            raw_score = gen_scores.get(self.score_type, 0.0)

            # Convert to validity
            validity = np.exp(-raw_score / 5.0)

            result.append(
                CandidateScore(
                    candidate_text=candidate.text,
                    claim_scores=np.array([validity]),
                    aggregate_scores={
                        self.score_type: raw_score,
                        "validity": validity,
                    },
                    metadata={
                        "scorer_type": "generation",
                        "score_type": self.score_type,
                    },
                )
            )

        return result

    def cleanup(self):
        """No resources to clean up."""
        pass
