import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_tts.generators import (
    StepCandidate,
    StepCandidateGeneratorThroughAPI,
    StepCandidateGeneratorThroughHuggingface,
    convert_trajectory_to_string,
)
from llm_tts.generators.base import CompletionReason

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


def compute_softmax(x: List[float]) -> List[float]:
    """
    Compute softmax of a list of values.

    Args:
        x: List of values

    Returns:
        List of softmax values
    """
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PhiDecoding(StrategyBase):
    """
    PhiDecoding: PhiDecoding for Large Language Models
    """

    def __init__(
        self,
        step_generator: (
            StepCandidateGeneratorThroughAPI | StepCandidateGeneratorThroughHuggingface
        ),
        scorer,
        max_steps: int,
        candidates_per_step: int = 4,
        cluster_num: int = 2,
    ):
        self.max_steps = max_steps
        self.candidates_per_step = candidates_per_step
        self.step_generator = step_generator
        self.scorer = scorer
        self.cluster_num = cluster_num

    def generate_trajectory(self, request: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using specified criterion.

        Args:
            input: Initial prompt/question or a conversation

        Returns:
            Dictionary with:
                - trajectory: Final generated trajectory
                - steps: List of selected steps
                - completed: Whether trajectory reached completion
        """

        trajectory = []
        selected_steps = []
        validity_scores = []

        for step_num in range(self.max_steps):
            log.info(f"\n=== Step {step_num} ===")

            candidates = self.step_generator(
                request,
                trajectory=trajectory,
                candidates_per_step=1,
            )
            if not candidates:
                log.info("No candidates generated, stopping")
                break

            selected_candidate = candidates[0]

            # Phi scale
            candidates = self.step_generator(
                request,
                trajectory=trajectory,
                candidates_per_step=self.candidates_per_step,
            )
            scores_phi = self.scorer.score_candidates(request, candidates)
            # Select best candidate
            best_idx, _ = self.foresight_rerank(
                request, candidates, trajectory, self.cluster_num, step_num
            )

            selected_candidate = candidates[best_idx]
            cur_signal = scores_phi[best_idx]
            validity_scores.append(cur_signal)

            # Update trajectory
            trajectory.append(selected_candidate)
            selected_steps.append(selected_candidate)

            # Check if trajectory is complete
            if selected_candidate.is_trajectory_complete:
                # Get completion reason from candidate
                completion_reason = None
                if selected_candidate.other_data:
                    completion_reason = selected_candidate.other_data.get(
                        "completion_reason"
                    )

                # If stopped at EOS, response is already complete (e.g., Qwen2.5-Math with \boxed{})
                if completion_reason == CompletionReason.EOS_PATTERN:
                    log.info("Stopped at EOS, response already complete")
                    break

                log.info("Answer pattern detected in step")
                if not self._has_answer_content(selected_candidate):
                    log.info("Answer content missing, generating final answer")
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

        return {
            "trajectory": convert_trajectory_to_string(trajectory),
            "steps": selected_steps,
            "validity_scores": validity_scores,
            "completed": len(selected_steps) > 0,
        }

    # Simulate the future trajectory and rerank the candidates
    def foresight_rerank(
        self, request, step_candidates, trajectory, cluster_num, step_num
    ):
        foresight_texts, foresight_scores = [], []
        for i in range(len(step_candidates)):
            new_trajectory = trajectory + [step_candidates[i]]
            candidate = self.step_generator(
                request,
                trajectory=new_trajectory,
                candidates_per_step=1,
            )
            foresight_texts.append(candidate[0].text)
            scores_simulate = self.scorer.score_candidates(request, candidate)
            foresight_scores.append(scores_simulate[0] - 1.0)
        try:
            X = TfidfVectorizer().fit_transform(foresight_texts)
            kmeans = KMeans(n_clusters=cluster_num, n_init="auto").fit(X)
            labels = kmeans.labels_

            cluster_sizes = [list[Any](labels).count(i) for i in labels]
            cluster_probs = compute_softmax(cluster_sizes)
            foresight_probs = compute_softmax(foresight_scores)
            combined_probs = [
                (foresight_probs[i] + cluster_probs[i]) / 2
                for i in range(len(foresight_scores))
            ]

            best_idx = np.random.choice(range(len(foresight_texts)), p=combined_probs)
            return best_idx, foresight_texts[best_idx]
        except Exception as e:
            print("Clustering failed:", e)
            fallback_probs = compute_softmax(foresight_scores)
            best_idx = np.random.choice(range(len(foresight_scores)), p=fallback_probs)
            return best_idx, foresight_texts[best_idx]

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

        log.info(f"Generated {len(answer_candidates)} answer candidates")
        log.info(f"Selected answer {best_idx}")
        log.info(f"Validity: {answer_validity_scores[best_idx]:.3f}")
        log.info(f"Text: {answer_candidates[best_idx].text}")

        return answer_candidates[best_idx], answer_validity_scores[best_idx]

    def cleanup(self):
        """Clean up resources"""

        self.scorer.cleanup()
