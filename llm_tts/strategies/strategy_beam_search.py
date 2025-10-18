import logging
from typing import Any, Dict, List

import numpy as np

from llm_tts.step_candidate_generator_base import covert_trajectory_to_string
from llm_tts.step_candidate_generator_through_api import (
    StepCandidateGeneratorThroughAPI,
)
from llm_tts.step_candidate_generator_through_huggingface import (
    StepCandidateGeneratorThroughHuggingface,
)

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyBeamSearch(StrategyBase):
    """
    Beam Search strategy for LLM reasoning.
    ---------------------------------------

    Keeps a beam of top-N reasoning chains at each step based on a scoring function.
    This balances exploration (multiple reasoning branches) and exploitation
    (keeping only the highest-scoring paths).

    Compared to:
      - ChainOfThought: explores multiple paths, not just one.
      - SelfConsistency: more efficient by pruning low-score paths early.
      - OnlineBestOfN: generalizes it by tracking multiple beams instead of one.

    Args:
        step_generator: Generator for candidate steps (API or HuggingFace).
        scorer: Scoring function for ranking step candidates.
        beam_size: Number of top beams to keep at each step.
        candidates_per_beam: Number of candidates to generate for each beam per step.
        max_steps: Maximum reasoning steps.
        aggregation: How to aggregate scores across steps ("avg", "sum", "min", "product").
    """

    def __init__(
        self,
        step_generator: (
            StepCandidateGeneratorThroughAPI | StepCandidateGeneratorThroughHuggingface
        ),
        scorer: Any,
        beam_size: int = 5,
        candidates_per_beam: int = 3,
        max_steps: int = 10,
        aggregation: str = "mean",
    ):
        self.step_generator = step_generator
        self.scorer = scorer
        self.beam_size = beam_size
        self.candidates_per_beam = candidates_per_beam
        self.max_steps = max_steps
        self.aggregation = aggregation

    def generate_trajectory(self, request: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate a reasoning trajectory using beam search.

        Args:
            request: Input chat or prompt context.

        Returns:
            Dictionary with trajectory, steps, score, and metadata.
        """

        # Initialize beams with empty trajectory
        beams = [{"steps": [], "scores": []}]
        completed_beams = []

        for step in range(self.max_steps):
            log.info(f"\n=== Beam Search Step {step} ===")
            new_beams = []

            # Expand each current beam
            for beam_idx, beam in enumerate(beams):
                log.info(
                    f"Expanding beam {beam_idx} with score "
                    f"{self._aggregate_scores(beam['scores']):.3f}"
                )

                candidates = self.step_generator(
                    request,
                    trajectory=beam["steps"],
                    candidates_per_step=self.candidates_per_beam,
                )

                if not candidates:
                    log.info(f"  No candidates for beam {beam_idx}, skipping")
                    continue

                scores = self.scorer.score_candidates(request, candidates)

                # Expand with new candidates
                for cand, score in zip(candidates, scores):
                    scores = beam["scores"] + [score]
                    new_beams.append(
                        {"steps": beam["steps"] + [cand], "scores": scores}
                    )

                    log.info(
                        f"    Candidate: score={score:.3f}, aggregated score={self._aggregate_scores(scores):.3f}, text='{cand.text[:80]}'"
                    )

            if not new_beams:
                log.info("No new beams generated, stopping early.")
                break

            # Sort and prune to top-k beams
            new_beams.sort(
                key=lambda b: self._aggregate_scores(b["scores"]),
                reverse=True,
            )
            beams = new_beams[: self.beam_size]
            log.info(f"Kept top {len(beams)} beams for next step")

            # Separate completed beams
            done, active = self._split_completed(beams)
            completed_beams.extend(done)
            beams = active

            # Stop if all beams are completed
            if not beams:
                log.info("All beams completed early.")
                break

        # Choose best final beam
        best_beam = self._select_best_beam(completed_beams or beams)

        return {
            "trajectory": covert_trajectory_to_string(best_beam["steps"]),
            "steps": best_beam["steps"],
            "validity_scores": best_beam["scores"],
            "completed": len(completed_beams) > 0,
        }

    def _aggregate_scores(self, scores: list[float]) -> float:
        """Aggregate scores across steps according to selected strategy."""
        if len(scores) == 0:
            return 0
        if self.aggregation == "sum":
            return sum(scores)
        elif self.aggregation == "mean":
            return np.mean(scores).item()
        elif self.aggregation == "product":
            return np.prod(scores).item()
        elif self.aggregation == "max":
            return np.max(scores).item()
        elif self.aggregation == "min":
            return np.min(scores).item()
        else:
            raise Exception(f"Unknown aggregation {self.aggregation}")

    def _split_completed(self, beams: List[Dict]) -> tuple:
        """Split beams into completed and active."""
        completed = []
        active = []
        for b in beams:
            if b["steps"] and b["steps"][-1].is_trajectory_complete:
                completed.append(b)
            else:
                active.append(b)
        return completed, active

    def _select_best_beam(self, beams: List[Dict]) -> Dict:
        """Select the highest scoring beam."""
        if not beams:
            return {"steps": [], "scores": 0.0}
        return max(beams, key=lambda b: self._aggregate_scores(b["scores"]))

    def cleanup(self):
        """Clean up scorer resources if necessary."""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
