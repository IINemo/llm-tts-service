import logging
from typing import TYPE_CHECKING, Dict, List, Union

from llm_tts.generators import (
    StepCandidate,
    StepCandidateGeneratorThroughAPI,
    StepCandidateGeneratorThroughHuggingface,
    convert_trajectory_to_string,
)
from llm_tts.generators.vllm import CompletionReason
from llm_tts.utils import extract_answer

if TYPE_CHECKING:
    from llm_tts.generators import VLLMStepGenerator
from llm_tts.scale_discriminator import ScaleDiscriminator

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class AdaptiveScalingBestOfN(StrategyBase):
    """
    Adaptive scaling online best-of-n strategy.
    """

    def __init__(
        self,
        scorer,
        candidates_per_step: int,
        max_steps: int,
        step_generator: Union[
            StepCandidateGeneratorThroughAPI,
            StepCandidateGeneratorThroughHuggingface,
            "VLLMStepGenerator",
        ],
        scaling_rate: float = 0.9,
        momentum_rate: float = 0.9,
        adaptive_scaling_method: str = "momentum",
    ):
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.scorer = scorer
        self.step_generator = step_generator
        self.scaling_rate = scaling_rate
        self.momentum_rate = momentum_rate
        kwargs = {}
        kwargs["momentum_rate"] = momentum_rate
        kwargs["scaling_rate"] = scaling_rate
        self.scale_discriminator = ScaleDiscriminator(
            criterion=adaptive_scaling_method, **kwargs
        )

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using specified criterion.

        Args:
            prompt: Initial prompt/question

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

            # Score candidates
            candidate_validity_scores = self.scorer.score_candidates(
                request, candidates
            )
            selected_candidate = candidates[0]
            cur_signal = candidate_validity_scores[0]

            log.info(f"Current signal: {cur_signal}")
            log.info(f"Current candidate: {selected_candidate.text}")

            if self.scale_discriminator.should_scale(cur_signal):
                log.info("Scaling step - generating new candidates")
                candidates = self.step_generator(
                    request,
                    trajectory=trajectory,
                    candidates_per_step=self.candidates_per_step,
                )
                all_candidate_scores = self.scorer.score_candidates(request, candidates)
                # Select best candidate
                best_idx, selected_candidate = self._select_best_candidate(
                    candidates, all_candidate_scores
                )
                cur_signal = all_candidate_scores[best_idx]

            self.scale_discriminator.update(cur_signal)

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
        self.scale_discriminator.reset()

        # Extract answer from trajectory
        final_trajectory = convert_trajectory_to_string(trajectory)
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
        best_idx = min(range(len(scores)), key=lambda i: scores[i])
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
        self.scale_discriminator.reset()
