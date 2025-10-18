import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from llm_tts.step_candidate_generator_base import StepCandidate

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
        score_aggregation: Optional[str] = "mean",
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
        # Accept either a step generator (preferred) or legacy api_client
        if step_generator is None:
            raise RuntimeError("Provide either step_generator or api_client (legacy).")
        self.step_generator = step_generator
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.max_empty_steps = max_empty_steps
        self.max_new_tokens = step_generator.max_new_tokens
        self.uncertainty_threshold = uncertainty_threshold

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
            # New path: drive everything through the step generator
            request_chat = self._normalize_to_chat(prompt_or_chat)
            prompt_text = self._normalize_to_prompt(prompt_or_chat)

            log_prompt = prompt_text.replace("\n", "\\n")
            log.info(f"Initial prompt: {log_prompt}")

            trajectory_text = prompt_text
            trajectory_steps = []
            selected_steps = []
            uncertainties = []
            decision_trace = []
            validity_scores = []
            empty_gen_count = 0

            for step_num in range(self.max_steps):
                log.info(f"\n=== PD step {step_num+1} ===")

                # 1) Probe: get one candidate and read its uncertainty score
                self.step_generator.max_new_tokens = 1
                probe_token = self.step_generator.generate_candidates(
                    request_chat, trajectory_steps, candidates_per_step=1
                )
                if not probe_token:
                    raise RuntimeError("Probe generation returned no candidates")

                probe_uncertainty = probe_token[0].other_data["uncertainty_score"]
                uncertainties.append({"uncertainty": probe_uncertainty})

                use_cot = bool(probe_uncertainty > self.uncertainty_threshold)

                # 2) Branch based on uncertainty
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
                        [cand.other_data["uncertainty_score"] for cand in cand_list]
                    )
                    best_idx = np.argmin(cand_scores)
                    chosen = cand_list[best_idx]

                    chosen_text = chosen.text
                    chosen_uncert = chosen.other_data["uncertainty_score"]

                    branch = "cot"
                    extra = {
                        "candidates": [
                            {
                                "text": c.text,
                                "uncertainty": float(cand_scores[i]),
                                "selected": i == best_idx,
                            }
                            for i, c in enumerate(cand_list)
                        ]
                    }

                else:
                    log.info("Using greedy completion (generator)")
                    chosen = self.step_generator.generate_candidates(
                        request_chat,
                        trajectory_steps,
                        candidates_per_step=1,
                    )[0]
                    chosen_text = chosen.text
                    chosen_uncert = chosen.other_data["uncertainty_score"]
                    branch = "greedy"
                    extra = {}

                # 3) Append and check for answer
                trajectory_steps.append(chosen)
                trajectory_text += chosen_text
                log.info(
                    "Step %d generation: %s",
                    step_num + 1,
                    chosen_text.replace("\n", "\\n"),
                )
                selected_steps.append(chosen)
                validity_scores.append(chosen_uncert)

                decision = {
                    "step": step_num + 1,
                    "branch": branch,
                    "uncertainty": float(chosen_uncert),
                    "threshold": float(self.uncertainty_threshold),
                }
                decision.update(extra)
                decision_trace.append(decision)

                if chosen.is_trajectory_complete:
                    log.info(f"Answer found in step {step_num+1}")
                    break

                if chosen_text == "":
                    empty_gen_count += 1
                    if empty_gen_count > self.max_empty_steps:
                        log.warning(
                            f"No generation found within last {self.max_empty_steps} steps."
                            "Stopping generation."
                        )
                        break

            # If no explicit <Answer>:, try to elicit one via generator
            if not self.step_generator.detector.contains_answer_pattern(
                trajectory_text
            ):
                answer_cands = self.step_generator.generate_answer_candidates(
                    request_chat,
                    trajectory_steps,
                    candidates_per_step=self.candidates_per_step,
                )
                if answer_cands:
                    ans_scores = np.array(
                        [
                            candidate.other_data["uncertainty_score"]
                            for candidate in answer_cands
                        ]
                    )
                    chosen = answer_cands[np.argmin(ans_scores)]
                    trajectory_steps.append(chosen)
                    trajectory_text += chosen.text
                    selected_steps.append(chosen.text)
                    validity_scores.append(chosen.other_data["uncertainty_score"])

            return {
                "trajectory": trajectory_text,
                "steps": selected_steps,
                "uncertainties": uncertainties,
                "completed": self.step_generator.detector.contains_answer_pattern(
                    trajectory_text
                ),
                "decision_trace": decision_trace,
                "validity_scores": validity_scores,
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

    def _extract_uncertainty_score(self, candidate: StepCandidate) -> float:
        data = getattr(candidate, "other_data", None)
        if isinstance(data, dict) and "uncertainty_score" in data:
            try:
                return float(data["uncertainty_score"])
            except Exception:
                return np.nan
        return np.nan

    def _extract_uncertainty_probe_score(self, candidate: StepCandidate) -> float:
        """Probe should use the first-token uncertainty if available; fallback to scalar."""
        data = getattr(candidate, "other_data", None)
        if isinstance(data, dict) and "uncertainty_score" in data:
            s = data["uncertainty_score"]
            try:
                # If it's an array-like per-token series, use first token
                if hasattr(s, "__len__") and not isinstance(s, (str, bytes)):
                    return float(s[0]) if len(s) > 0 else np.nan
                return float(s)
            except Exception:
                return np.nan
        return np.nan
