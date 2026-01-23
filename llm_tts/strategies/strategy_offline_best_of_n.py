"""
Offline Best-of-N strategy - Single-call trajectory generation with post-hoc step splitting.

Generates N complete trajectories in ONE vLLM call, splits them into steps post-hoc
using the same stop tokens, then scores each trajectory with PRM.

Key features:
- Single vLLM call: All N trajectories generated in parallel (n=num_trajectories)
- Post-hoc step detection: Splits trajectories using same stop tokens as step-by-step
- Efficient PRM scoring: Each trajectory scored once after generation
- Maximum throughput: No stopping at step boundaries during generation

Key difference from online best-of-n:
- Online: greedy step selection at each iteration (selects best step, continues)
- Offline: generates all N trajectories independently, then picks best complete solution
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from llm_tts.generators.base import StepCandidate, StepCandidateGeneratorBase
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyOfflineBestOfN(StrategyBase):
    """
    Offline Best-of-N strategy with single-call trajectory generation.

    Generates N complete trajectories in ONE vLLM call (n=num_trajectories),
    splits them into steps post-hoc using the same stop tokens,
    then scores each trajectory with PRM and selects the best one.
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

    def _generate_all_trajectories_single_call(
        self,
        request: List[Dict[str, str]],
    ) -> List[Dict[str, any]]:
        """
        Generate all N trajectories in a SINGLE vLLM call.

        Uses generate_full_trajectories() which:
        1. Generates N complete trajectories with n=num_trajectories
        2. Splits each into steps post-hoc using the same stop tokens

        This is much faster than step-by-step generation because there's
        no stopping at step boundaries during generation.

        Args:
            request: Chat messages for the request

        Returns:
            List of trajectory dictionaries (scores added later)
        """
        log.info(
            f"\n--- Generating {self.num_trajectories} trajectories (single call) ---"
        )

        # Single vLLM call generates all N trajectories
        raw_results = self.step_generator.generate_full_trajectories(
            request=request,
            num_trajectories=self.num_trajectories,
        )

        # Convert to expected format
        results = []
        for i, raw in enumerate(raw_results):
            results.append(
                {
                    "steps": raw["steps"],  # List of step strings
                    "step_scores": [],  # Will be filled after scoring
                    "aggregated_score": 0.0,  # Will be filled after scoring
                    "full_text": raw["full_text"],
                    "num_steps": len(raw["steps"]),
                    "is_complete": raw["is_complete"],
                }
            )

        return results

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, any]:
        """
        Generate N complete trajectories and return the best one.

        Uses single-call vLLM generation for maximum throughput - all N trajectories
        are generated in ONE vLLM call with n=num_trajectories.

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
        log.info(
            f"Generating {self.num_trajectories} trajectories (single-call offline)"
        )
        log.info(f"Score aggregation: {self.score_aggregation}")
        log.info(f"{'='*60}")

        # Step 1: Generate all trajectories in ONE vLLM call
        all_trajectory_results = self._generate_all_trajectories_single_call(request)

        # Step 2: Score all trajectories efficiently (one PRM call per trajectory)
        log.info(f"\n{'='*60}")
        log.info(f"Scoring {self.num_trajectories} trajectories")
        log.info(f"{'='*60}")

        for i, result in enumerate(all_trajectory_results):
            if result["steps"]:
                # Score entire trajectory in single forward pass
                # steps is List[str] - PRM scorer handles this via hasattr check
                step_scores = self.scorer.score_trajectory(
                    request,
                    result["steps"],
                    trajectory_validity_score=result.get("validity_score"),
                    trajectory_uncertainty_score=result.get("uncertainty_score"),
                )
                result["step_scores"] = step_scores
                result["aggregated_score"] = self._aggregate_scores(step_scores)
            else:
                result["step_scores"] = []
                result["aggregated_score"] = 0.0

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

        # Save logs if output_dir provided
        if self.output_dir:
            self._save_trajectories_log(all_trajectory_results, best_idx)

        return {
            "trajectory": best_result["full_text"],
            "extracted_answer": extracted,
            "steps": best_result["steps"],  # List of step strings
            "thinking_num_steps": 0,  # Not tracked in single-call mode
            "response_num_steps": best_result["num_steps"],
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
                    "steps": r["steps"],  # Individual step texts
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
