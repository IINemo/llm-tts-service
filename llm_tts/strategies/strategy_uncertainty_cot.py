import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

log = logging.getLogger(__name__)


class StrategyUncertaintyCoT:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = "",
        candidates_per_step: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_empty_steps: Optional[int] = None,
        uncertainty_threshold: Optional[float] = None,
        step_generator: Optional[object] = None,
        uncertainty_sampling: Optional[str] = None,
    ):
        """
        Initialize Uncertainty-Guided CoT with Predictive Distribution (PD).

        Args:
            api_client: Unified API client (TogetherAIModel, OpenRouterModel, or compatible)
            candidates_per_step: Number of candidate paths to generate at each step
            max_steps: Maximum number of reasoning steps
            max_empty_steps: Maximum number of empty generations before stopping
            uncertainty_threshold: Threshold for triggering CoT branching
        """
        if step_generator is None:
            raise RuntimeError("Provide either step_generator or api_client (legacy).")
        if uncertainty_sampling.lower() not in {"sequence", "token"}:
            raise ValueError(
                f"uncertainty_sampling must be 'sequence' or 'token', got {uncertainty_sampling}"
            )
        self.step_generator = step_generator
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.max_empty_steps = max_empty_steps
        self.max_new_tokens = step_generator.max_new_tokens
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_sampling_mode = uncertainty_sampling.lower()

    def generate_trajectory(
        self, prompt_or_chat: Union[str, List[Dict[str, str]]]
    ) -> Dict[str, any]:
        """
        Uncertainty-guided decoding pipeline.

        Preferred path: use step_generator to obtain both generations and
        uncertainty scores directly from the model wrapper. Falls back to the
        legacy api_client path if no step_generator is provided.
        """
        if self.step_generator is not None:
            request_chat = self._normalize_to_chat(prompt_or_chat)
            prompt_text = self._normalize_to_prompt(prompt_or_chat)

            log_prompt = prompt_text.replace("\n", "\\n")
            log.info(f"Initial prompt: {log_prompt}")

            trajectory_text = ""
            trajectory_steps = []
            uncertainties = []
            validity_scores = []
            empty_gen_count = 0
            num_multi_path_steps = 0
            num_greedy_steps = 0

            for step_num in range(self.max_steps):
                log.info(f"\n=== PD step {step_num+1} ===")

                # 1) Get initial uncertainty score
                initial_candidate = None
                # if we are sampling at the token level, we need to probe 1 token uncertainty
                # and use it to determine if we should use CoT, but we still need to generate a sequence
                if self.uncertainty_sampling_mode == "token":
                    initial_uncertainty = self._probe_token_uncertainty(
                        request_chat, trajectory_steps
                    )

                # if we are sampling at the sequence level, we can just generate a sequence
                # and if uncertainty is low, we keep it
                elif self.uncertainty_sampling_mode == "sequence":
                    self.step_generator.max_new_tokens = self.max_new_tokens
                    initial_candidate = self.step_generator.generate_candidates(
                        request_chat,
                        trajectory_steps,
                        candidates_per_step=1,
                    )[0]
                    if not initial_candidate:
                        raise RuntimeError("Initial generation returned no candidates")
                    initial_uncertainty = initial_candidate.other_data["validity_score"]

                log.info(
                    f"[initial] Uncertainty ({self.uncertainty_sampling_mode}): {initial_uncertainty}"
                )

                # 2) Branch based on uncertainty
                use_cot = bool(initial_uncertainty > self.uncertainty_threshold)
                self.step_generator.max_new_tokens = self.max_new_tokens
                if use_cot:
                    log.info("Using multi-path completion")
                    cand_list = self.step_generator.generate_candidates(
                        request_chat,
                        trajectory_steps,
                        candidates_per_step=self.candidates_per_step,
                    )
                    if not cand_list:
                        raise RuntimeError("No candidates returned for CoT branch")

                    cand_scores = np.array(
                        [cand.other_data["validity_score"] for cand in cand_list]
                    )
                    # we want to choose the candidate with the lowest uncertainty
                    chosen = cand_list[np.argmin(cand_scores)]
                    for cand_idx, cand in enumerate(cand_list):
                        log.info(
                            f"[{cand_idx}] Uncertainty: {cand_scores[cand_idx]:.3f} | Text: {cand.text}"
                        )

                    num_multi_path_steps += 1
                    extra = {
                        "uncert_cot_metadata": {
                            "branch": "multi-path",
                            "trigger_uncertainty": initial_uncertainty,
                            "sampling_mode": self.uncertainty_sampling_mode,
                            "num_candidates": len(cand_list),
                            "all_candidates": [cand.text for cand in cand_list],
                            "all_uncertainties": [
                                cand.other_data["validity_score"] for cand in cand_list
                            ],
                        }
                    }

                else:
                    log.info("Using single-path completion")
                    # if we used the token level probe and uncertainty is low,
                    # we need to generate a sequence
                    if initial_candidate is None:
                        initial_candidate = self.step_generator.generate_candidates(
                            request_chat,
                            trajectory_steps,
                            candidates_per_step=1,
                        )[0]
                        if not initial_candidate:
                            raise RuntimeError(
                                "No candidate returned for greedy completion"
                            )
                    chosen = initial_candidate
                    num_greedy_steps += 1
                    extra = {
                        "uncert_cot_metadata": {
                            "branch": "greedy",
                            "trigger_uncertainty": initial_uncertainty,
                            "sampling_mode": self.uncertainty_sampling_mode,
                        }
                    }

                # 3) Append and check for answer
                chosen_text = chosen.text
                chosen_uncert = chosen.other_data["validity_score"]
                chosen.other_data.update(extra)

                trajectory_steps.append(chosen)
                trajectory_text += ("\n" if trajectory_text != "" else "") + chosen_text

                uncertainties.append(initial_uncertainty)
                validity_scores.append(1 - chosen_uncert)

                if chosen_text == "":
                    empty_gen_count += 1
                    if empty_gen_count > self.max_empty_steps:
                        log.warning(
                            f"No generation found within last {self.max_empty_steps} steps."
                            "Stopping generation."
                        )
                        break

                if self.step_generator.detector.is_trajectory_complete(trajectory_text):
                    log.info(
                        f"Trajectory complete at step {step_num+1}; Generating answer candidates"
                    )

                    answer_cands = self.step_generator.generate_answer_candidates(
                        request_chat,
                        trajectory_steps,
                        candidates_per_step=self.candidates_per_step,
                    )
                    if answer_cands:
                        log.info("Answer candidates generated")
                        answer_scores = np.array(
                            [
                                candidate.other_data["validity_score"]
                                for candidate in answer_cands
                            ]
                        )
                        chosen = answer_cands[np.argmin(answer_scores)]

                        for cand_idx, cand in enumerate(answer_cands):
                            log.info(
                                f"[{cand_idx}] Uncertainty: {answer_scores[cand_idx]:.3f} | Text: {cand.text}"
                            )

                        trajectory_steps.append(chosen)
                        trajectory_text += chosen.text
                        uncertainties.append(chosen.other_data["validity_score"])
                        validity_scores.append(1 - chosen.other_data["validity_score"])

                        break

            return {
                "trajectory": trajectory_text,
                "steps": trajectory_steps,
                "uncertainties": uncertainties,
                "validity_scores": validity_scores,
                "completed": self.step_generator.detector.contains_answer_pattern(
                    trajectory_text
                ),
                "metadata": {
                    "uncert_cot_threshold": self.uncertainty_threshold,
                    "uncert_sampling_mode": self.uncertainty_sampling_mode,
                    "num_steps": step_num + 1,
                    "num_greedy_steps": num_greedy_steps,
                    "num_multi_path_steps": num_multi_path_steps,
                },
            }

    def _normalize_to_chat(
        self, prompt_or_chat: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        if isinstance(prompt_or_chat, list):
            return prompt_or_chat
        if isinstance(prompt_or_chat, str):
            return [{"role": "user", "content": prompt_or_chat}]
        raise TypeError(
            "generate_trajectory expects a string or chat-style List[Dict[str, str]]"
        )

    def _probe_token_uncertainty(
        self, request_chat: List[Dict[str, str]], trajectory_steps: List[Any]
    ) -> Optional[float]:
        self.step_generator.max_new_tokens = 1
        probe = self.step_generator.generate_candidates(
            request_chat, trajectory_steps, candidates_per_step=1
        )
        if not probe:
            raise RuntimeError("Token-level probe generation returned no candidates")
        return probe[0].other_data.get("validity_score")

    def _normalize_to_prompt(
        self, prompt_or_chat: Union[str, List[Dict[str, str]]]
    ) -> str:
        """
        Accept either a plain prompt string or a chat-style message list and
        return a single prompt string used by the uncertainty-guided pipeline.

        Chat format expected: List[{"role": str, "content": str}]. We
        concatenate non-empty system messages (if any) and the user messages,
        preserving order, separated by blank lines. If only one of them is
        present, it is used as-is.
        """
        if isinstance(prompt_or_chat, str):
            return prompt_or_chat

        if isinstance(prompt_or_chat, list):
            try:
                system_parts: List[str] = []
                user_parts: List[str] = []
                for m in prompt_or_chat:
                    role = m.get("role")
                    content = m.get("content")
                    if not content:
                        continue
                    if role == "system":
                        system_parts.append(str(content))
                    elif role == "user":
                        user_parts.append(str(content))
                    else:
                        # Fallback: include any other roles in order
                        user_parts.append(str(content))

                system_text = "\n".join(p for p in system_parts if p)
                user_text = "\n".join(p for p in user_parts if p)

                if system_text and user_text:
                    return f"{system_text}\n\n{user_text}"
                return system_text or user_text or ""
            except Exception:
                # As a last resort, try a naive stringify
                return "\n".join(
                    str(m.get("content", ""))
                    for m in prompt_or_chat
                    if isinstance(m, dict)
                )

        raise TypeError(
            "generate_trajectory expects a string or chat-style List[Dict[str, str]]"
        )
