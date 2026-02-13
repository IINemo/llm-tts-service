import logging
from typing import Any, Dict, List, Optional

import numpy as np

from llm_tts.generators.base import convert_trajectory_to_string
from llm_tts.utils import extract_answer

from .strategy_base import StrategyBase, count_reasoning_steps

log = logging.getLogger(__name__)


class StrategyUncertaintyCoT(StrategyBase):
    def __init__(
        self,
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
            candidates_per_step: Number of candidate paths to generate at each step
            max_steps: Maximum number of reasoning steps
            max_empty_steps: Maximum number of empty generations before stopping
            uncertainty_threshold: Threshold for triggering CoT branching
            step_generator: Step candidate generator instance
            uncertainty_sampling: Sampling mode ('sequence' or 'token')
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
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_sampling_mode = uncertainty_sampling.lower()

    def generate_trajectories_batch(
        self, requests: List[List[Dict[str, str]]], sample_indices: List[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate trajectories for a batch of samples.

        Uncertainty-CoT is inherently sequential per-sample (each step depends
        on the previous step's uncertainty), so we loop over samples.
        """
        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        self.step_generator.reset_per_sample_stats()

        results = []
        for idx, (request, sample_idx) in enumerate(zip(requests, sample_indices)):
            result = self._generate_single(request, sample_idx, stats_idx=idx)
            results.append(result)

        self.step_generator.finalize_sample_stats(num_samples=len(requests))
        return results

    def _generate_single(
        self,
        request_chat: List[Dict[str, str]],
        sample_idx: int,
        stats_idx: int,
    ) -> Dict[str, Any]:
        """Core uncertainty-guided decoding pipeline for a single sample."""
        log_prompt = str(request_chat[-1]["content"])[:200].replace("\n", "\\n")
        log.info(f"Initial prompt: {log_prompt}")

        trajectory_steps = []  # reasoning steps only (StepCandidate objects)
        trajectory_all = []  # all steps incl. answer (for text building)
        uncertainties = []
        validity_scores = []
        empty_gen_count = 0
        num_multi_path_steps = 0
        num_greedy_steps = 0
        answer_step_text = None

        for step_num in range(self.max_steps):
            log.info(f"=== PD step {step_num+1} ===")

            # 1) Get initial uncertainty score
            initial_candidate = None
            if self.uncertainty_sampling_mode == "token":
                initial_uncertainty = self._probe_token_uncertainty(
                    request_chat, trajectory_steps
                )
            elif self.uncertainty_sampling_mode == "sequence":
                initial_candidate = self.step_generator(
                    request_chat,
                    trajectory_steps,
                    candidates_per_step=1,
                )[0]
                if not initial_candidate:
                    raise RuntimeError("Initial generation returned no candidates")
                initial_uncertainty = initial_candidate.other_data["uncertainty_score"]

            log.info(
                f"[initial] Uncertainty ({self.uncertainty_sampling_mode}): {initial_uncertainty}"
            )

            # 2) Branch based on uncertainty
            use_cot = bool(initial_uncertainty > self.uncertainty_threshold)
            if use_cot:
                log.info("Using multi-path completion")
                cand_list = self.step_generator(
                    request_chat,
                    trajectory_steps,
                    candidates_per_step=self.candidates_per_step,
                )
                if not cand_list:
                    raise RuntimeError("No candidates returned for CoT branch")

                cand_uncertainties = np.array(
                    [cand.other_data["uncertainty_score"] for cand in cand_list]
                )
                chosen = cand_list[np.argmin(cand_uncertainties)]
                for cand_idx, cand in enumerate(cand_list):
                    log.info(
                        f"[{cand_idx}] Uncertainty: {cand_uncertainties[cand_idx]:.3f} | Text: {cand.text}"
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
                            cand.other_data["uncertainty_score"] for cand in cand_list
                        ],
                    }
                }

            else:
                log.info("Using single-path completion")
                if initial_candidate is None:
                    initial_candidate = self.step_generator(
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

            # 3) Append reasoning step
            chosen.other_data.update(extra)

            trajectory_steps.append(chosen)
            trajectory_all.append(chosen)

            uncertainties.append(float(initial_uncertainty))
            validity_scores.append(float(chosen.other_data["validity_score"]))

            if chosen.text == "":
                empty_gen_count += 1
                if empty_gen_count > self.max_empty_steps:
                    log.warning(
                        f"No generation found within last {self.max_empty_steps} steps. "
                        "Stopping generation."
                    )
                    break

            # 4) Check for answer / max steps
            if chosen.is_trajectory_complete and self._has_answer_content(chosen):
                # Answer is already embedded in the last reasoning step
                answer_step_text = chosen.raw_text if chosen.raw_text else chosen.text
                break
            if (
                chosen.is_trajectory_complete
                or chosen.is_thinking_complete
                or step_num == self.max_steps - 1
            ):
                # Generate answer only in thinking mode — non-thinking mode
                # produces the answer naturally in the last reasoning step.
                if getattr(self.step_generator, "thinking_mode", False):
                    answer_step_text = self._generate_answer_step(
                        request_chat, trajectory_steps, trajectory_all
                    )
                break

        # Finalize and capture token stats
        token_stats = self.step_generator.get_sample_stats_for(stats_idx)

        # Build trajectory text from all steps (reasoning + answer)
        trajectory_text = convert_trajectory_to_string(trajectory_all)

        extracted = extract_answer(trajectory_text)

        # reasoning_steps count: steps list has only reasoning steps
        thinking_mode = getattr(self.step_generator, "thinking_mode", False)
        reasoning_steps = count_reasoning_steps(trajectory_steps, thinking_mode)

        return {
            "trajectory": trajectory_text,
            "extracted_answer": extracted,
            "steps": trajectory_steps,
            "reasoning_steps": reasoning_steps,
            "validity_scores": validity_scores,
            "completed": self.step_generator.detector.contains_answer_pattern(
                trajectory_text
            ),
            "token_stats": token_stats,
            "answer_step": answer_step_text,
            "metadata": {
                "uncert_cot_threshold": self.uncertainty_threshold,
                "uncert_sampling_mode": self.uncertainty_sampling_mode,
                "num_steps": len(trajectory_steps),
                "num_greedy_steps": num_greedy_steps,
                "num_multi_path_steps": num_multi_path_steps,
            },
        }

    def _generate_answer_step(
        self,
        request_chat: List[Dict[str, str]],
        trajectory_steps: List,
        trajectory_all: List,
    ) -> Optional[str]:
        """Generate answer candidates and select the most confident one.

        The chosen answer is appended to trajectory_all (for text building)
        but NOT to trajectory_steps (which holds reasoning steps only).

        Returns the answer text, or None if no candidates were generated.
        """
        log.info("Generating answer candidates")
        answer_cands = self.step_generator.generate_answer_candidates(
            request_chat,
            trajectory_steps,
            candidates_per_step=self.candidates_per_step,
        )
        if not answer_cands:
            return None

        log.info(f"Generated {len(answer_cands)} answer candidates")
        answer_uncertainties = [
            cand.other_data["uncertainty_score"] for cand in answer_cands
        ]
        # Handle None uncertainties (e.g. from already-complete trajectories)
        if all(u is None for u in answer_uncertainties):
            chosen_answer = answer_cands[0]
        else:
            chosen_answer = answer_cands[
                np.argmin(
                    np.array(
                        [
                            u if u is not None else float("inf")
                            for u in answer_uncertainties
                        ]
                    )
                )
            ]

        for cand_idx, cand in enumerate(answer_cands):
            u = answer_uncertainties[cand_idx]
            u_str = f"{u:.3f}" if u is not None else "None"
            log.info(f"[{cand_idx}] Uncertainty: {u_str} | Text: {cand.text}")

        trajectory_all.append(chosen_answer)
        return chosen_answer.raw_text if chosen_answer.raw_text else chosen_answer.text

    def _probe_token_uncertainty(
        self, request_chat: List[Dict[str, str]], trajectory_steps: List[Any]
    ) -> Optional[float]:
        saved_limit = self.step_generator.generation_limit
        self.step_generator.generation_limit = 1
        try:
            probe = self.step_generator(
                request_chat, trajectory_steps, candidates_per_step=1
            )
        finally:
            self.step_generator.generation_limit = saved_limit
        if not probe:
            raise RuntimeError("Token-level probe generation returned no candidates")
        return probe[0].other_data.get("uncertainty_score")
