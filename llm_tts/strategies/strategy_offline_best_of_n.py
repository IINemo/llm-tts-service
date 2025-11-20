import logging
from typing import Any, Dict, List

from lm_polygraph import BlackboxModel

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyOfflineBestOfN(StrategyBase):
    """
    Offline best-of-n: generate n complete trajectories directly, then select the best one.
    This is different from online best-of-n which generates step-by-step.
    """

    def __init__(
        self,
        model: BlackboxModel,
        trajectories: int,
        scorer: Any = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ):
        self.model = model
        self.num_trajectories = trajectories
        self.scorer = scorer
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate_trajectory(self, request: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Generate multiple complete trajectories directly, then select the best one.

        Returns a dict with keys:
          - trajectory: stringified best trajectory
          - steps: list containing the best trajectory
          - validity_scores: list of scores for each trajectory
          - completed: whether at least one trajectory was generated
        """

        all_trajectories: List[str] = []

        # Generate complete trajectories directly using the model
        for traj_idx in range(self.num_trajectories):
            log.info(f"\n=== Trajectory {traj_idx} ===")

            try:
                # Generate complete trajectory in one go using the model
                results = self.model.generate_texts(
                    chats=[request],
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

                if results and len(results) > 0:
                    trajectory_text = results[0].get("text", "")
                    if trajectory_text.strip():
                        all_trajectories.append(trajectory_text)
                        log.info(
                            f"Generated trajectory {traj_idx}: {trajectory_text[:100]}..."
                        )
                    else:
                        log.warning(f"Empty trajectory generated for {traj_idx}")
                else:
                    log.warning(f"No result returned for trajectory {traj_idx}")

            except Exception as e:
                log.error(f"Error generating trajectory {traj_idx}: {e}")
                continue

        if not all_trajectories:
            log.info("No trajectories generated")
            return {
                "trajectory": "",
                "steps": [],
                "validity_scores": [],
                "completed": False,
            }

        # Score trajectories using provided scorer if available, otherwise fallback
        # to a simple heuristic (trajectory length).
        trajectory_scores: List[float]
        if self.scorer is not None:
            try:
                # Prepare full chains by concatenating the chat prompt with generated text
                try:
                    prompt = "".join([m.get("content", "") for m in request])
                except Exception:
                    prompt = ""

                chains = [prompt + traj for traj in all_trajectories]

                # Prefer scorer API that directly scores complete chains if available
                if hasattr(self.scorer, "score_complete_chains"):
                    trajectory_scores = self.scorer.score_complete_chains(chains)
                else:
                    # Fall back to generic candidate scorer (passes chat + candidates)
                    trajectory_scores = self.scorer.score_candidates(
                        request, all_trajectories
                    )
            except Exception as e:
                log.error(
                    f"Scoring trajectories failed: {e}. Falling back to length heuristic"
                )
                trajectory_scores = [len(traj) for traj in all_trajectories]
        else:
            # For now, use trajectory length as a proxy for quality
            trajectory_scores = [len(traj) for traj in all_trajectories]

        # Select the best trajectory based on scores
        best_idx = max(
            range(len(trajectory_scores)), key=lambda i: trajectory_scores[i]
        )
        best_trajectory = all_trajectories[best_idx]

        log.info(
            f"Generated {len(all_trajectories)} trajectories; selected #{best_idx} with score {trajectory_scores[best_idx]}"
        )

        return {
            "trajectory": best_trajectory,
            "steps": [best_trajectory],  # Single complete trajectory
            "validity_scores": trajectory_scores,
            "completed": len(best_trajectory) > 0,
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, "scorer") and self.scorer is not None:
            if hasattr(self.scorer, "cleanup"):
                try:
                    self.scorer.cleanup()
                except Exception:
                    log.exception("Error during scorer cleanup")
