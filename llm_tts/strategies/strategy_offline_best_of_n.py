import logging
from typing import Dict, List, Tuple

from llm_tts.step_candidate_generator_base import (
    StepCandidate,
    covert_trajectory_to_string,
)
from llm_tts.step_candidate_generator_through_api import (
    StepCandidateGeneratorThroughAPI,
)
from llm_tts.step_candidate_generator_through_huggingface import (
    StepCandidateGeneratorThroughHuggingface,
)

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyOfflineBestOfN(StrategyBase):
    """
    Offline best-of-n: sample complete trajectories, then select the best final answer.
    """

    def __init__(
        self,
        scorer,
        trajectories: int,
        max_steps: int,
        step_generator: (
            StepCandidateGeneratorThroughAPI | StepCandidateGeneratorThroughHuggingface
        ),
    ):
        self.num_trajectories = trajectories
        self.max_steps = max_steps
        self.scorer = scorer
        self.step_generator = step_generator

    def generate_trajectory(self, request: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Generate multiple complete trajectories, score their final answers,
        and return the best trajectory.

        Returns a dict with keys:
          - trajectory: stringified best trajectory
          - steps: list of `StepCandidate` for the best trajectory
          - validity_scores: list of scores for each final answer candidate
          - completed: whether at least one trajectory was generated
        """

        all_trajectories: List[List[StepCandidate]] = []
        final_answers: List[StepCandidate] = []

        # Sample full trajectories
        for traj_idx in range(self.num_trajectories):
            log.info(f"\n=== Trajectory {traj_idx} ===")

            trajectory: List[StepCandidate] = []
            selected_candidate: StepCandidate | None = None

            for step_num in range(self.max_steps):
                log.info(f"Step {step_num}")

                # Sample exactly one candidate to extend this trajectory
                candidates = self.step_generator(
                    request,
                    trajectory=trajectory,
                    candidates_per_step=1,
                )

                if not candidates:
                    log.info("No candidate generated; stopping this trajectory")
                    break

                selected_candidate = candidates[0]
                trajectory.append(selected_candidate)

                if selected_candidate.is_trajectory_complete:
                    log.info("Answer pattern detected - trajectory complete")
                    break

            # If not complete, explicitly generate answer segment
            if not trajectory or not trajectory[-1].is_trajectory_complete:
                answer_candidates = self.step_generator.generate_answer_candidates(
                    request,
                    trajectory=trajectory,
                    candidates_per_step=1,
                )
                if answer_candidates:
                    trajectory.append(answer_candidates[0])
                    selected_candidate = answer_candidates[0]

            # Only keep non-empty trajectories
            if trajectory:
                all_trajectories.append(trajectory)
                # The final step in the trajectory is treated as the final answer
                final_answers.append(trajectory[-1])

        if not final_answers:
            log.info("No trajectories generated")
            return {
                "trajectory": "",
                "steps": [],
                "validity_scores": [],
                "completed": False,
            }

        # Score final answers for each trajectory and select the best
        answer_scores = self.scorer.score_candidates(request, final_answers)

        best_idx = max(range(len(answer_scores)), key=lambda i: answer_scores[i])
        best_trajectory = all_trajectories[best_idx]

        log.info(
            f"Generated {len(all_trajectories)} trajectories; selected #{best_idx} with score {answer_scores[best_idx]:.3f}"
        )

        return {
            "trajectory": covert_trajectory_to_string(best_trajectory),
            "steps": best_trajectory,
            "validity_scores": answer_scores,
            "completed": len(best_trajectory) > 0,
        }

    def cleanup(self):
        """Clean up resources"""
        self.scorer.cleanup()
