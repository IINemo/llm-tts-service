"""
Extended thinking strategy (s1-style "Wait" injection).

In thinking mode, models sometimes produce </think> prematurely before fully
reasoning through a problem. This strategy injects continuation tokens (like
"Wait") when the model tries to stop thinking, forcing longer and more thorough
reasoning chains before allowing the final answer.

Based on the s1 paper: https://arxiv.org/abs/2501.19393
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from llm_tts.generators.base import convert_trajectory_to_string
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase, count_reasoning_steps

log = logging.getLogger(__name__)


class StrategyExtendedThinking(StrategyBase):
    """
    Extended thinking strategy — forces longer reasoning by injecting
    continuation tokens when the model tries to close </think> prematurely.

    Thinking mode only. Uses batch generation via generate_step_candidates_batch.

    Algorithm:
        1. Generate thinking steps in batch for all samples.
        2. When </think> appears and continuations < max_continuations:
           strip </think> from the step, append the continuation token,
           and keep generating.
        3. When continuations >= max_continuations: allow </think> and
           generate the final answer via generate_answer_candidates_batch.
        4. Non-thinking completions (EOS, answer pattern) mark the sample
           as done immediately.
    """

    def __init__(
        self,
        step_generator,
        max_continuations: int = 3,
        continuation_token: str = "\nWait, ",
        max_steps: int = 50,
        output_dir: str = "./outputs",
        eos_patterns: List[str] = None,
        stop_token_ids: List[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.step_generator = step_generator
        self.max_continuations = max_continuations
        self.continuation_token = continuation_token
        self.max_steps = max_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eos_patterns = eos_patterns or ["<end of response>"]
        self.stop_token_ids = stop_token_ids

        if not getattr(self.step_generator, "thinking_mode", False):
            log.warning(
                "ExtendedThinking strategy is designed for thinking mode, "
                "but the step generator does not have thinking_mode=True. "
                "The strategy will still run but Wait injection will have no effect."
            )

        log.info(
            f"ExtendedThinking strategy initialized: "
            f"max_continuations={max_continuations}, "
            f"continuation_token={continuation_token!r}, "
            f"max_steps={max_steps}"
        )

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int] = None,
        save_callback=None,
    ) -> List[Dict[str, Any]]:
        M = len(requests)
        if sample_indices is None:
            sample_indices = list(range(M))

        log.info(
            f"ExtendedThinking: batch generating {M} responses "
            f"(max_continuations={self.max_continuations}, "
            f"continuation_token={self.continuation_token!r}, "
            f"max_steps={self.max_steps})"
        )

        # Reset per-sample tracking
        self.step_generator.reset_per_sample_stats()

        # Build stop tokens list
        stop_tokens = list(self.eos_patterns)
        if "</think>" not in stop_tokens:
            stop_tokens.append("</think>")
        log.info(f"Stop tokens: {stop_tokens}")

        # Per-sample state
        trajectories = [[] for _ in range(M)]  # list of StepCandidate lists
        continuations = [0] * M  # how many times we injected Wait
        completed = [False] * M  # sample is fully done
        needs_final_answer = [False] * M  # thinking done, needs answer generation

        for step_num in range(self.max_steps):
            # Collect active (not completed, not waiting for answer) sample indices
            active_indices = [
                i for i in range(M) if not completed[i] and not needs_final_answer[i]
            ]
            if not active_indices:
                log.info(f"All samples done after {step_num} steps, exiting loop")
                break

            log.info(
                f"--- Step {step_num + 1}/{self.max_steps} --- "
                f"{len(active_indices)}/{M} active samples"
            )
            for i in active_indices:
                log.info(
                    f"  Sample {sample_indices[i]}: "
                    f"continuations={continuations[i]}/{self.max_continuations}, "
                    f"trajectory_steps={len(trajectories[i])}"
                )

            # Build batch request for active samples only
            active_requests = [requests[i] for i in active_indices]
            active_trajectories = [trajectories[i] for i in active_indices]

            # We use original indices as sample_ids so stats accumulate correctly
            batch_sample_ids = list(active_indices)

            batch_results = self.step_generator.generate_step_candidates_batch(
                requests=active_requests,
                trajectories=active_trajectories,
                candidates_per_step=1,
                stop_tokens_override=stop_tokens,
                max_tokens=self.step_generator.generation_limit,
                compute_uncertainty=False,
                sample_ids=batch_sample_ids,
            )

            # Process each active sample's result
            for batch_idx, orig_idx in enumerate(active_indices):
                candidates = batch_results[batch_idx]
                if not candidates:
                    log.warning(
                        f"Sample {sample_indices[orig_idx]}: "
                        f"no candidates at step {step_num + 1}, marking completed"
                    )
                    completed[orig_idx] = True
                    continue

                candidate = candidates[0]
                n_tokens = len(candidate.token_ids) if candidate.token_ids else 0
                text_preview = (candidate.text or "")[:200]

                log.info(
                    f"Sample {sample_indices[orig_idx]} step {step_num + 1} result: "
                    f"tokens={n_tokens}, "
                    f"is_thinking_complete={candidate.is_thinking_complete}, "
                    f"is_trajectory_complete={candidate.is_trajectory_complete}, "
                    f"is_complete={candidate.is_complete}"
                )
                log.debug(
                    f"Sample {sample_indices[orig_idx]} step {step_num + 1} "
                    f"text preview: {text_preview!r}"
                )

                if (
                    candidate.is_thinking_complete
                    and not candidate.is_trajectory_complete
                ):
                    # Model wants to close </think>
                    if continuations[orig_idx] < self.max_continuations:
                        # Strip </think> and inject continuation
                        old_text_tail = (candidate.text or "")[-80:]
                        candidate.text = self._strip_think_close(candidate.text)
                        candidate.text += self.continuation_token
                        candidate.raw_text = self._strip_think_close(candidate.raw_text)
                        candidate.raw_text += self.continuation_token
                        candidate.is_thinking_complete = False
                        candidate.is_complete = False
                        continuations[orig_idx] += 1
                        new_text_tail = candidate.text[-80:]
                        log.info(
                            f">>> FORCED CONTINUATION #{continuations[orig_idx]}/{self.max_continuations} "
                            f"for sample {sample_indices[orig_idx]} at step {step_num + 1}: "
                            f"stripped </think>, appended {self.continuation_token!r}"
                        )
                        log.info(f"    Text tail BEFORE: ...{old_text_tail!r}")
                        log.info(f"    Text tail AFTER:  ...{new_text_tail!r}")
                    else:
                        # Allow </think>, need final answer
                        needs_final_answer[orig_idx] = True
                        log.info(
                            f"<<< THINKING COMPLETE for sample {sample_indices[orig_idx]} "
                            f"at step {step_num + 1} after "
                            f"{continuations[orig_idx]} forced continuations. "
                            f"Allowing </think>, will generate final answer."
                        )
                elif candidate.is_trajectory_complete:
                    # EOS or answer pattern — fully done
                    completed[orig_idx] = True
                    completion_reason = (
                        candidate.other_data.get("completion_reason", "unknown")
                        if candidate.other_data
                        else "unknown"
                    )
                    log.info(
                        f"Sample {sample_indices[orig_idx]}: "
                        f"trajectory complete at step {step_num + 1} "
                        f"(reason={completion_reason}, no Wait injection needed)"
                    )
                else:
                    # Generation stopped but neither thinking_complete nor trajectory_complete
                    # (e.g., hit max_tokens mid-thinking). Just keep going.
                    log.info(
                        f"Sample {sample_indices[orig_idx]}: step {step_num + 1} "
                        f"generation ended without completion markers, continuing"
                    )

                trajectories[orig_idx].append(candidate)

            # Check if all samples are done
            if all(completed[i] or needs_final_answer[i] for i in range(M)):
                log.info(
                    f"All {M} samples reached terminal state after step {step_num + 1}"
                )
                break

        # Handle samples that hit max_steps without completing
        for i in range(M):
            if not completed[i] and not needs_final_answer[i]:
                log.warning(
                    f"Sample {sample_indices[i]}: hit max_steps={self.max_steps} "
                    f"without completing thinking "
                    f"(continuations={continuations[i]}, "
                    f"trajectory_steps={len(trajectories[i])})"
                )
                if trajectories[i]:
                    needs_final_answer[i] = True
                else:
                    completed[i] = True

        # Batch generate answers for all samples that need them
        answer_indices = [i for i in range(M) if needs_final_answer[i]]
        answer_map = {}  # orig_idx -> answer_step

        if answer_indices:
            log.info(
                f"=== ANSWER GENERATION PHASE === "
                f"Generating final answers for {len(answer_indices)}/{M} samples "
                f"in batched call"
            )
            for i in answer_indices:
                thinking_tokens = sum(
                    len(s.token_ids) for s in trajectories[i] if s.token_ids
                )
                log.info(
                    f"  Sample {sample_indices[i]}: "
                    f"{len(trajectories[i])} thinking steps, "
                    f"~{thinking_tokens} thinking tokens, "
                    f"{continuations[i]} forced continuations"
                )
            batch_answer_reqs = [requests[i] for i in answer_indices]
            batch_answer_trajs = [trajectories[i] for i in answer_indices]
            answer_results = self.step_generator.generate_answer_candidates_batch(
                batch_answer_reqs,
                batch_answer_trajs,
                candidates_per_step=1,
            )
            for batch_idx, orig_idx in enumerate(answer_indices):
                if answer_results[batch_idx]:
                    answer_step = answer_results[batch_idx][0]
                    answer_step.is_trajectory_complete = True
                    answer_map[orig_idx] = answer_step
                    completed[orig_idx] = True
                    answer_preview = (answer_step.raw_text or answer_step.text or "")[
                        :300
                    ]
                    log.info(
                        f"  Sample {sample_indices[orig_idx]} answer generated: "
                        f"{answer_preview!r}"
                    )
                else:
                    log.warning(
                        f"  Sample {sample_indices[orig_idx]}: "
                        f"answer generation returned empty!"
                    )
        else:
            log.info("No samples need answer generation (all completed directly)")

        # Build result dicts
        log.info("=== BUILDING RESULTS ===")
        results = []
        for idx in range(M):
            if not trajectories[idx]:
                log.warning(f"Sample {sample_indices[idx]}: empty trajectory")
                results.append(
                    {
                        "trajectory": "",
                        "extracted_answer": "",
                        "steps": [],
                        "answer_step": None,
                        "reasoning_steps": 0,
                        "validity_scores": [],
                        "completed": False,
                        "token_stats": {},
                    }
                )
                continue

            trajectory = list(trajectories[idx])
            answer_text = None

            if idx in answer_map:
                answer_step = answer_map[idx]
                trajectory.append(answer_step)
                answer_text = answer_step.raw_text or answer_step.text

            final_trajectory = convert_trajectory_to_string(trajectory)
            extracted = extract_answer(final_trajectory)
            token_stats = self.step_generator.get_sample_stats_for(idx)

            reasoning_steps = count_reasoning_steps(
                trajectory,
                getattr(self.step_generator, "thinking_mode", False),
            )

            total_tokens = sum(len(s.token_ids) for s in trajectory if s.token_ids)
            log.info(
                f"Sample {sample_indices[idx]} summary: "
                f"completed={completed[idx]}, "
                f"continuations={continuations[idx]}/{self.max_continuations}, "
                f"steps={len(trajectory)}, "
                f"reasoning_steps={reasoning_steps}, "
                f"total_tokens={total_tokens}, "
                f"extracted_answer={extracted!r}"
            )

            results.append(
                {
                    "trajectory": final_trajectory,
                    "extracted_answer": extracted,
                    "steps": trajectory,
                    "answer_step": answer_text,
                    "reasoning_steps": reasoning_steps,
                    "validity_scores": [],
                    "completed": completed[idx],
                    "token_stats": token_stats,
                    "continuations": continuations[idx],
                }
            )

        completed_count = sum(1 for r in results if r["completed"])
        total_continuations = sum(continuations)
        avg_continuations = total_continuations / M if M > 0 else 0
        log.info(
            f"=== ExtendedThinking DONE === "
            f"{completed_count}/{M} completed, "
            f"total forced continuations: {total_continuations}, "
            f"avg continuations per sample: {avg_continuations:.1f}"
        )

        return results

    @staticmethod
    def _strip_think_close(text: str) -> str:
        """Strip the trailing </think> tag from text."""
        if text.endswith("</think>"):
            return text[: -len("</think>")]
        # Handle case where </think> might be followed by whitespace
        idx = text.rfind("</think>")
        if idx != -1:
            return text[:idx] + text[idx + len("</think>") :]
        return text

    def cleanup(self):
        """Cleanup resources."""
        pass
