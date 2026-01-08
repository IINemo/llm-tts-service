"""
Offline Best-of-N strategy - Stepwise generation.

Generates N complete trajectories step-by-step, scores each step,
then selects the best trajectory based on aggregated step scores.

Key difference from online best-of-n:
- Online: greedy step selection at each iteration (selects best step, continues)
- Offline: generates all N trajectories independently, then picks best complete solution
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from llm_tts.generators import StepCandidate, StepCandidateGeneratorBase
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase, count_thinking_and_response_steps

log = logging.getLogger(__name__)


class StrategyOfflineBestOfN(StrategyBase):
    """
    Offline Best-of-N strategy with stepwise generation.

    Generates N complete trajectories step-by-step, scores each step,
    then selects the best trajectory based on aggregated step scores.
    """

    def __init__(
        self,
        scorer,
        num_trajectories: int,
        max_steps: int,
        step_generator: StepCandidateGeneratorBase,
        score_aggregation: str = "mean",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize offline best-of-n strategy.

        Args:
            scorer: Scorer for evaluating steps (PRM, entropy, etc.)
            num_trajectories: Number of complete trajectories to generate
            max_steps: Maximum steps per trajectory
            step_generator: Generator for step candidates
            score_aggregation: How to aggregate step scores ('mean', 'min', 'max', 'product', 'last')
            output_dir: Directory for saving logs
        """
        self.scorer = scorer
        self.num_trajectories = num_trajectories
        self.max_steps = max_steps
        self.step_generator = step_generator
        self.score_aggregation = score_aggregation
        self.output_dir = output_dir
        self._current_sample_idx = 0

        log.info(
            f"StrategyOfflineBestOfN initialized: "
            f"{num_trajectories} trajectories, max_steps={max_steps}, "
            f"aggregation={score_aggregation}"
        )

    def _get_uncertainty_score(self, candidate: StepCandidate) -> float:
        """Get uncertainty score from candidate's other_data."""
        if candidate.other_data and "uncertainty_score" in candidate.other_data:
            return candidate.other_data["uncertainty_score"]
        return 0.0

    def _aggregate_scores(self, step_scores: List[float]) -> float:
        """
        Aggregate step scores into a single trajectory score.

        Args:
            step_scores: List of scores for each step

        Returns:
            Aggregated score (higher = better)
        """
        if not step_scores:
            return 0.0

        if self.score_aggregation == "mean":
            return float(np.mean(step_scores))
        elif self.score_aggregation == "min":
            # Conservative: trajectory is only as good as its weakest step
            return float(np.min(step_scores))
        elif self.score_aggregation == "max":
            # Optimistic: best step determines trajectory score
            return float(np.max(step_scores))
        elif self.score_aggregation == "product":
            return float(np.prod(step_scores))
        elif self.score_aggregation == "last":
            return step_scores[-1]
        else:
            log.warning(f"Unknown aggregation '{self.score_aggregation}', using mean")
            return float(np.mean(step_scores))

    def _generate_single_trajectory(
        self,
        request: List[Dict[str, str]],
        trajectory_idx: int,
    ) -> Dict[str, any]:
        """
        Generate a single complete trajectory step-by-step.

        Args:
            request: Chat messages for the request
            trajectory_idx: Index of this trajectory (for logging)

        Returns:
            Dictionary with trajectory info, steps, and scores
        """
        trajectory = []  # List of StepCandidate
        step_scores = []
        step_texts = []

        log.info(f"\n--- Trajectory {trajectory_idx + 1}/{self.num_trajectories} ---")

        for step_num in range(self.max_steps):
            # Generate single candidate for this step
            candidates = self.step_generator(
                request,
                trajectory=trajectory,
                candidates_per_step=1,  # One candidate per step for each trajectory
            )

            if not candidates:
                log.info(f"  Step {step_num + 1}: No candidates, stopping")
                break

            candidate = candidates[0]
            step_texts.append(candidate.text)

            # Score this step
            scores = self.scorer.score_candidates(request, [candidate])
            step_score = scores[0] if scores else 0.0
            step_scores.append(step_score)

            # Get token count
            num_tokens = len(candidate.token_ids) if candidate.token_ids else 0

            log.info(
                f"  Step {step_num + 1}: score={step_score:.3f}, "
                f"tokens={num_tokens}, complete={candidate.is_trajectory_complete}"
            )

            # Add to trajectory
            trajectory.append(candidate)

            # Check if trajectory is complete
            if candidate.is_trajectory_complete:
                log.info(f"  Trajectory complete at step {step_num + 1}")
                break

        # Aggregate step scores
        aggregated_score = self._aggregate_scores(step_scores)

        # Build full trajectory text
        full_text = "".join(step_texts)

        return {
            "trajectory": trajectory,
            "step_scores": step_scores,
            "aggregated_score": aggregated_score,
            "full_text": full_text,
            "num_steps": len(trajectory),
            "is_complete": (
                trajectory[-1].is_trajectory_complete if trajectory else False
            ),
        }

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, any]:
        """
        Generate N complete trajectories and return the best one.

        Args:
            request: Chat messages for the request
            sample_idx: Index of current sample (for logging)

        Returns:
            Dictionary with best trajectory and metadata
        """
        self._current_sample_idx = sample_idx

        # Reset token tracking
        self.step_generator.reset_sample_stats()

        log.info(f"\n{'='*60}")
        log.info(f"Generating {self.num_trajectories} trajectories (stepwise offline)")
        log.info(f"Score aggregation: {self.score_aggregation}")
        log.info(f"{'='*60}")

        # Generate all trajectories
        all_trajectory_results = []
        for i in range(self.num_trajectories):
            result = self._generate_single_trajectory(request, i)
            all_trajectory_results.append(result)

        # Log summary
        log.info("\n--- Trajectory Scores Summary ---")
        for i, result in enumerate(all_trajectory_results):
            log.info(
                f"Trajectory {i + 1}: "
                f"aggregated={result['aggregated_score']:.4f}, "
                f"steps={result['num_steps']}, "
                f"complete={result['is_complete']}, "
                f"step_scores={[f'{s:.3f}' for s in result['step_scores']]}"
            )

        # Select best trajectory
        aggregated_scores = [r["aggregated_score"] for r in all_trajectory_results]
        best_idx = int(np.argmax(aggregated_scores))
        best_result = all_trajectory_results[best_idx]

        log.info(f"\n{'='*60}")
        log.info(
            f"Selected trajectory {best_idx + 1} "
            f"with aggregated score {best_result['aggregated_score']:.4f}"
        )
        log.info(f"{'='*60}")

        # Extract answer from best trajectory
        extracted = extract_answer(best_result["full_text"])

        # Get token stats
        token_stats = self.step_generator.get_sample_stats()

        # Count thinking and response steps
        thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
            best_result["trajectory"]
        )

        # Save logs if output_dir provided
        if self.output_dir:
            self._save_trajectories_log(all_trajectory_results, best_idx)

        return {
            "trajectory": best_result["full_text"],
            "extracted_answer": extracted,
            "steps": best_result["trajectory"],
            "thinking_num_steps": thinking_num_steps,
            "response_num_steps": response_num_steps,
            "validity_scores": best_result["step_scores"],
            "aggregated_score": best_result["aggregated_score"],
            "all_trajectories": [r["full_text"] for r in all_trajectory_results],
            "all_scores": aggregated_scores,
            "all_step_scores": [r["step_scores"] for r in all_trajectory_results],
            "best_idx": best_idx,
            "completed": best_result["is_complete"],
            "token_stats": token_stats,
        }

    def _save_trajectories_log(
        self,
        all_results: List[Dict[str, any]],
        best_idx: int,
    ):
        """Save all trajectories to JSON for analysis."""
        if not self.output_dir:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(
            self.output_dir, f"trajectories_sample_{self._current_sample_idx}.json"
        )

        log_data = {
            "sample_idx": self._current_sample_idx,
            "num_trajectories": len(all_results),
            "score_aggregation": self.score_aggregation,
            "best_idx": best_idx,
            "best_score": all_results[best_idx]["aggregated_score"],
            "trajectories": [
                {
                    "idx": i,
                    "aggregated_score": r["aggregated_score"],
                    "step_scores": r["step_scores"],
                    "num_steps": r["num_steps"],
                    "is_complete": r["is_complete"],
                    "text": r["full_text"],
                    "is_best": i == best_idx,
                }
                for i, r in enumerate(all_results)
            ],
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        log.info(f"Saved trajectories log to {log_path}")

    def get_token_stats(self) -> Dict[str, any]:
        """Get token statistics from the generator."""
        return self.step_generator.get_sample_stats()

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
