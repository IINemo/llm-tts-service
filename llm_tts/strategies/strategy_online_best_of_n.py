import json
import logging
import os
from typing import Dict, List, Optional

import torch

from llm_tts.generators import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.strategies.deepconf.utils import extract_answer

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyOnlineBestOfN(StrategyBase):
    """
    Greedy online best-of-n strategy.

    Works with any step generator (HuggingFace, API, or vLLM).
    """

    def __init__(
        self,
        scorer,
        candidates_per_step: int,
        max_steps: int,
        step_generator: StepCandidateGeneratorBase,
        output_dir: Optional[str] = None,
    ):
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.scorer = scorer
        self.step_generator = step_generator
        self.output_dir = output_dir
        self._current_sample_idx = 0
        self._steps_log = []

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using specified criterion.

        Args:
            request: Chat messages for the request
            sample_idx: Index of current sample (for logging)

        Returns:
            Dictionary with:
                - trajectory: Final generated trajectory
                - steps: List of selected steps
                - completed: Whether trajectory reached completion
        """
        self._current_sample_idx = sample_idx
        self._steps_log = []

        trajectory = []
        selected_steps = []
        validity_scores = []
        for step_num in range(self.max_steps):
            log.info(f"\n=== Step {step_num} ===")

            candidates = self.step_generator(
                request,
                trajectory=trajectory,
                candidates_per_step=self.candidates_per_step,
            )

            if not candidates:
                log.info("\nNo candidates generated, stopping")
                break

            # Score candidates
            candidate_validity_scores = self.scorer.score_candidates(
                request, candidates
            )

            # Log all candidates
            log.info(f"\nGenerated {len(candidates)} candidates:")
            for i, (candidate, val_score) in enumerate(
                zip(candidates, candidate_validity_scores)
            ):
                log.info(f"\n[{i}] Validity: {val_score:.3f}\nText:\n{candidate.text}")

            # Select best candidate
            best_idx, selected_candidate = self._select_best_candidate(
                candidates, candidate_validity_scores
            )
            log.info(f"\nSelected candidate {best_idx}\nText:\n{selected_candidate.text}")

            # Update trajectory
            trajectory.append(selected_candidate)
            selected_steps.append(selected_candidate)
            validity_scores.append(candidate_validity_scores[best_idx])

            # Get full trajectory for logging
            full_trajectory = convert_trajectory_to_string(trajectory)

            # Log step to JSON (save every 5 steps)
            self._log_step(
                step_num=step_num,
                candidates=candidates,
                scores=candidate_validity_scores,
                selected_idx=best_idx,
                trajectory_so_far=full_trajectory,
            )
            if (step_num + 1) % 5 == 0:
                self._save_steps_log()
                self._save_trajectory_log(full_trajectory)

            # Clear CUDA cache to reduce OOM risk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check if trajectory is complete
            if selected_candidate.is_trajectory_complete:
                log.info("\nAnswer pattern detected in step")
                # Check if answer content is present after the pattern
                # When using HuggingFace/vLLM with stopping criteria, generation
                # stops at "<Answer>:" without generating the actual answer content
                if not self._has_answer_content(selected_candidate):
                    log.info("\nAnswer content missing, generating final answer")
                    # Remove the incomplete step and generate proper answer
                    trajectory.pop()
                    selected_steps.pop()
                    final_answer, final_validity = self._generate_final_answer(
                        request, trajectory
                    )
                    trajectory.append(final_answer)
                    selected_steps.append(final_answer)
                    validity_scores.append(final_validity)
                break

        if not selected_candidate.is_trajectory_complete:
            final_answer, final_validity = self._generate_final_answer(
                request, trajectory
            )
            trajectory.append(final_answer)
            selected_steps.append(final_answer)
            validity_scores.append(final_validity)

        # Save steps log and final trajectory to JSON
        final_trajectory = convert_trajectory_to_string(trajectory)
        self._save_steps_log()
        self._save_trajectory_log(final_trajectory)

        # Extract answer from trajectory (e.g., content between <Answer>: and <end of response>)
        extracted = extract_answer(final_trajectory)

        return {
            "trajectory": final_trajectory,
            "extracted_answer": extracted,
            "steps": selected_steps,
            "validity_scores": validity_scores,
            "completed": len(selected_steps) > 0,
        }

    def _select_best_candidate(self, candidates: List, scores: List[float]) -> tuple:
        """Select the best candidate based on scores"""

        # Higher validity is better
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, candidates[best_idx]

    def _generate_final_answer(
        self, chat: List[Dict[str, str]], trajectory: List[StepCandidate]
    ) -> tuple:
        """Generate and select best final answer based on criterion"""

        # Generate answer candidates in batches if needed
        answer_candidates = self.step_generator.generate_answer_candidates(
            chat, trajectory=trajectory, candidates_per_step=self.candidates_per_step
        )

        # Score answer candidates
        answer_validity_scores = self.scorer.score_candidates(chat, answer_candidates)

        # Select best answer based on criterion
        best_idx, _ = self._select_best_candidate(
            answer_candidates, answer_validity_scores
        )

        log.info(
            f"\nGenerated {len(answer_candidates)} answer candidates\n"
            f"Selected answer {best_idx}\n"
            f"Validity: {answer_validity_scores[best_idx]:.3f}\n"
            f"Answer text:\n{answer_candidates[best_idx].text}"
        )

        return answer_candidates[best_idx], answer_validity_scores[best_idx]

    def cleanup(self):
        """Clean up resources"""

        self.scorer.cleanup()

    def _log_step(
        self,
        step_num: int,
        candidates: List[StepCandidate],
        scores: List[float],
        selected_idx: int,
        trajectory_so_far: str,
    ):
        """Log step information for JSON output."""
        step_data = {
            "step_num": step_num,
            "candidates": [
                {
                    "idx": i,
                    "text": c.text,
                    "validity_score": float(scores[i]),  # Convert to Python float
                    "is_complete": c.is_complete,
                    "is_trajectory_complete": c.is_trajectory_complete,
                }
                for i, c in enumerate(candidates)
            ],
            "selected_idx": selected_idx,
            "selected_text": candidates[selected_idx].text,
            "selected_score": float(scores[selected_idx]),  # Convert to Python float
            "trajectory_so_far": trajectory_so_far,
        }
        self._steps_log.append(step_data)

    def _save_steps_log(self):
        """Save steps log to JSON file."""
        if not self.output_dir:
            return

        log_file = os.path.join(
            self.output_dir, f"steps_sample_{self._current_sample_idx}.json"
        )
        try:
            with open(log_file, "w") as f:
                json.dump(
                    {
                        "sample_idx": self._current_sample_idx,
                        "total_steps": len(self._steps_log),
                        "steps": self._steps_log,
                    },
                    f,
                    indent=2,
                )
            log.info(f"\nSaved steps log to {log_file}")
        except Exception as e:
            log.warning(f"\nFailed to save steps log: {e}")

    def _save_trajectory_log(self, trajectory_text: str):
        """Save concatenated trajectory to a separate JSON file."""
        if not self.output_dir:
            return

        log_file = os.path.join(
            self.output_dir, f"trajectory_sample_{self._current_sample_idx}.json"
        )
        try:
            with open(log_file, "w") as f:
                json.dump(
                    {
                        "sample_idx": self._current_sample_idx,
                        "total_steps": len(self._steps_log),
                        "trajectory": trajectory_text,
                    },
                    f,
                    indent=2,
                )
            log.info(f"\nSaved trajectory to {log_file}")
        except Exception as e:
            log.warning(f"\nFailed to save trajectory log: {e}")
