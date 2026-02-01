import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

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

from .strategy_base import StrategyBase, count_thinking_and_response_steps

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
        batch_size: int = 1000,
    ):
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.scorer = scorer
        self.step_generator = step_generator
        self.scaling_rate = scaling_rate
        self.momentum_rate = momentum_rate
        self.adaptive_scaling_method = adaptive_scaling_method
        kwargs = {}
        kwargs["momentum_rate"] = momentum_rate
        kwargs["scaling_rate"] = scaling_rate
        self.scale_discriminator = ScaleDiscriminator(
            criterion=adaptive_scaling_method, **kwargs
        )
        self.batch_size = batch_size

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

        # Reset token tracking for this sample
        self.step_generator.reset_sample_stats()

        trajectory = []
        selected_steps = []
        validity_scores = []
        selected_candidate = None
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
                request, candidates, trajectory=trajectory
            )
            selected_candidate = candidates[0]
            cur_signal = candidate_validity_scores[0]

            log.info(f"Current signal: {cur_signal}")
            log.info(f"Current candidate: {selected_candidate.text}")

            # Decide whether to scale at this step
            if self.scale_discriminator.should_scale(cur_signal):
                log.info("Scaling step - generating new candidates")
                candidates = self.step_generator(
                    request,
                    trajectory=trajectory,
                    candidates_per_step=self.candidates_per_step,
                )
                all_candidate_scores = self.scorer.score_candidates(request, candidates, trajectory=trajectory)
                # Select best candidate
                best_idx, selected_candidate = self._select_best_candidate(
                    candidates, all_candidate_scores
                )
                cur_signal = all_candidate_scores[best_idx]

            self.scale_discriminator.update(1 - cur_signal)
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
                    validity_scores.pop()
                    final_answer, final_validity = self._generate_final_answer(
                        request, trajectory
                    )
                    trajectory.append(final_answer)
                    selected_steps.append(final_answer)
                    validity_scores.append(final_validity)
                break

        if selected_candidate is not None and not selected_candidate.is_trajectory_complete:
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

        # Finalize and get token statistics
        self.step_generator.finalize_sample_stats()
        token_stats = self.step_generator.get_sample_stats()
        log.info(
            f"Sample token stats: "
            f"total_tokens={token_stats['total_tokens_this_sample']:,}, "
            f"input_tokens={token_stats.get('input_tokens', 0):,}, "
            f"output_tokens={token_stats.get('output_tokens', 0):,}, "
            f"generations={token_stats['generation_count']}"
            + (f", tflops={token_stats['tflops']:.3f}" if token_stats["tflops"] else "")
        )
        # Count thinking and response steps separately
        thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
            selected_steps
        )

        return {
            "trajectory": final_trajectory,
            "extracted_answer": extracted,
            "steps": selected_steps,
            "validity_scores": validity_scores,
            "completed": len(selected_steps) > 0,
            "thinking_num_steps": thinking_num_steps,
            "response_num_steps": response_num_steps,
            "token_stats": token_stats,
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

        log.info(f"Generated {len(answer_candidates)} answer candidates")
        log.info(f"Selected answer {best_idx}")
        log.info(f"Validity: {answer_validity_scores[best_idx]:.3f}")
        log.info(f"Text: {answer_candidates[best_idx].text}")

        return answer_candidates[best_idx], answer_validity_scores[best_idx]

    def generate_trajectory_mini_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_idxs: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batched version of generate_trajectory that runs multiple samples "online" in parallel.

        Key idea for vLLM speed:
        - At each step, we only call the step_generator ONCE for all active samples.
        - This lets vLLM batch prompts efficiently.

        Args:
            requests: list of chat message lists, one per sample
            sample_idxs: list of sample ids for logging/output file names; defaults to 0..B-1

        Returns:
            List of per-sample outputs (same schema as generate_trajectory()).
        """
        num_samples = len(requests)
        if sample_idxs is None:
            sample_idxs = list(range(num_samples))
        assert (
            len(sample_idxs) == num_samples
        ), "sample_idxs must have same length as requests"

        # --- Per-sample state ---
        trajectories: List[List[StepCandidate]] = [[] for _ in range(num_samples)]
        selected_steps: List[List[StepCandidate]] = [[] for _ in range(num_samples)]
        validity_scores: List[List[float]] = [[] for _ in range(num_samples)]
        completed: List[bool] = [False for _ in range(num_samples)]
        last_selected: List[Optional[StepCandidate]] = [
            None for _ in range(num_samples)
        ]

        # Per-sample token tracking (like beam search)
        sample_token_stats: Dict[int, Dict[str, int]] = {
            i: {"input_tokens": 0, "output_tokens": 0, "generation_count": 0}
            for i in range(num_samples)
        }

        # Reset batch-wide token tracking
        self.step_generator.reset_sample_stats()

        def _count_context_tokens(
            request: List[Dict[str, str]], trajectory: List[StepCandidate]
        ) -> int:
            """Count context tokens for a (request, trajectory) pair."""
            traj_text = convert_trajectory_to_string(trajectory)
            if getattr(self.step_generator, "disable_thinking_mode", False):
                prompt = self.step_generator._apply_chat_template(
                    request, enable_thinking=False
                )
            else:
                prompt = self.step_generator.tokenizer.apply_chat_template(
                    request, tokenize=False, add_generation_prompt=True
                )
            if traj_text:
                prompt = prompt + traj_text
            return len(self.step_generator.tokenizer.encode(prompt))

        def _track_generation_tokens(
            sample_indices_for_call: List[int],
            reqs: List[List[Dict[str, str]]],
            trajs: List[List[StepCandidate]],
            candidates_batch: List[List[StepCandidate]],
        ) -> None:
            """Record per-sample input/output tokens from a generation call."""
            for pos, sample_idx in enumerate(sample_indices_for_call):
                ctx_tokens = _count_context_tokens(reqs[pos], trajs[pos])
                sample_token_stats[sample_idx]["input_tokens"] += ctx_tokens
                for candidate in candidates_batch[pos]:
                    out_tokens = candidate.other_data.get(
                        "original_token_count", len(candidate.token_ids)
                    )
                    sample_token_stats[sample_idx]["output_tokens"] += out_tokens
                sample_token_stats[sample_idx]["generation_count"] += 1

        # ---- Batched helpers (duck-typing) ----
        def _gen_step_candidates_batch(
            active_reqs: List[List[Dict[str, str]]],
            active_trajs: List[List[StepCandidate]],
            candidates_per_step: int,
        ) -> List[List[StepCandidate]]:
            """
            Returns list-of-list of candidates aligned to active samples.
            Prefer true batched generator method for vLLM throughput.
            """
            if hasattr(self.step_generator, "generate_step_candidates_batch"):
                return self.step_generator.generate_step_candidates_batch(
                    active_reqs,
                    trajectories=active_trajs,
                    candidates_per_step=candidates_per_step,
                )

            try:
                out = self.step_generator(
                    active_reqs,
                    trajectory=active_trajs,
                    candidates_per_step=candidates_per_step,
                )
                if out and isinstance(out[0], list):
                    return out
            except Exception:
                log.warning(
                    "Batch generation failed, falling back to sequential",
                    exc_info=True,
                )

            # Fallback: loop (no vLLM batching)
            return [
                self.step_generator(
                    req,
                    trajectory=traj,
                    candidates_per_step=candidates_per_step,
                )
                for req, traj in zip(active_reqs, active_trajs)
            ]

        def _score_candidates_batch(
            active_reqs: List[List[Dict[str, str]]],
            active_cands: List[List[StepCandidate]],
            active_trajs: List[List[StepCandidate]],
        ) -> List[List[float]]:
            """Returns list-of-list of scores aligned to active samples."""
            if hasattr(self.scorer, "score_candidates_batch"):
                return self.scorer.score_candidates_batch(
                    active_reqs, active_cands, trajectories=active_trajs
                )

            # Fallback: loop per sample
            scores = []
            for req, cands, traj in zip(active_reqs, active_cands, active_trajs):
                scores.append(self.scorer.score_candidates(req, cands, trajectory=traj))
            return scores

        def _gen_answer_candidates_batch(
            active_reqs: List[List[Dict[str, str]]],
            active_trajs: List[List[StepCandidate]],
        ) -> List[List[StepCandidate]]:
            """Batched final answer generation."""
            if hasattr(self.step_generator, "generate_answer_candidates_batch"):
                return self.step_generator.generate_answer_candidates_batch(
                    active_reqs,
                    trajectories=active_trajs,
                    candidates_per_step=self.candidates_per_step,
                )

            # Fallback: loop
            return [
                self.step_generator.generate_answer_candidates(
                    req, trajectory=traj, candidates_per_step=self.candidates_per_step
                )
                for req, traj in zip(active_reqs, active_trajs)
            ]

        def _select_best(scores: List[float]) -> int:
            return max(range(len(scores)), key=lambda i: scores[i])

        def _new_scale_discriminator() -> ScaleDiscriminator:
            return ScaleDiscriminator(
                criterion=self.adaptive_scaling_method,
                momentum_rate=self.momentum_rate,
                scaling_rate=self.scaling_rate,
            )

        # ---- Main online loop (batched across samples) ----
        scale_discriminators: List[ScaleDiscriminator] = [
            _new_scale_discriminator() for _ in range(num_samples)
        ]
        needs_final_answer: List[bool] = [False for _ in range(num_samples)]

        for step_num in range(self.max_steps):
            # Which samples are still active?
            active_indices = [idx for idx in range(num_samples) if not completed[idx]]
            if not active_indices:
                log.info(f"All {num_samples} samples completed at step {step_num}")
                break

            log.info(
                f"\n=== Step {step_num} === "
                f"({len(active_indices)}/{num_samples} active samples)"
            )

            active_reqs = [requests[idx] for idx in active_indices]
            active_trajs = [trajectories[idx] for idx in active_indices]

            # 1) Generate a single candidate first for each active sample
            step_candidates_batch = _gen_step_candidates_batch(
                active_reqs, active_trajs, candidates_per_step=1
            )

            # Track tokens for the initial 1-candidate generation
            _track_generation_tokens(
                active_indices, active_reqs, active_trajs, step_candidates_batch
            )

            # Handle samples that produced no candidates
            filtered_indices = []
            filtered_reqs = []
            filtered_trajs = []
            filtered_cands = []
            for pos, sample_idx in enumerate(active_indices):
                cands = step_candidates_batch[pos]
                if not cands:
                    completed[sample_idx] = True
                    last_selected[sample_idx] = None
                else:
                    filtered_indices.append(sample_idx)
                    filtered_reqs.append(requests[sample_idx])
                    filtered_trajs.append(trajectories[sample_idx])
                    filtered_cands.append(cands)

            if not filtered_indices:
                break

            # 2) Score the single candidates (batched if possible)
            scores_batch = _score_candidates_batch(
                filtered_reqs, filtered_cands, filtered_trajs
            )

            # 3) Decide which samples should scale
            scale_indices = []
            scale_reqs = []
            scale_trajs = []
            for pos, sample_idx in enumerate(filtered_indices):
                cur_signal = scores_batch[pos][0]
                if scale_discriminators[sample_idx].should_scale(cur_signal):
                    scale_indices.append(sample_idx)
                    scale_reqs.append(requests[sample_idx])
                    scale_trajs.append(trajectories[sample_idx])

            if scale_indices:
                log.info(
                    f"Scaling {len(scale_indices)}/{len(filtered_indices)} samples "
                    f"(samples: {[sample_idxs[i] for i in scale_indices]})"
                )

            # 4) For samples that scale, generate N more candidates and rescore
            scaled_candidates = {}
            scaled_scores = {}
            if scale_indices:
                scale_cands_batch = _gen_step_candidates_batch(
                    scale_reqs,
                    scale_trajs,
                    candidates_per_step=self.candidates_per_step,
                )
                # Track tokens for the scaling generation
                _track_generation_tokens(
                    scale_indices, scale_reqs, scale_trajs, scale_cands_batch
                )
                scale_scores_batch = _score_candidates_batch(
                    scale_reqs, scale_cands_batch, scale_trajs
                )
                for pos, sample_idx in enumerate(scale_indices):
                    scaled_candidates[sample_idx] = scale_cands_batch[pos]
                    scaled_scores[sample_idx] = scale_scores_batch[pos]

            # 5) Select candidate per sample and update states
            for pos, sample_idx in enumerate(filtered_indices):
                if sample_idx in scaled_candidates and scaled_candidates[sample_idx]:
                    cands = scaled_candidates[sample_idx]
                    scores = scaled_scores[sample_idx]
                    best_idx = _select_best(scores)
                    chosen = cands[best_idx]
                    cur_signal = scores[best_idx]
                else:
                    chosen = filtered_cands[pos][0]
                    cur_signal = scores_batch[pos][0]

                scale_discriminators[sample_idx].update(1 - cur_signal)
                trajectories[sample_idx].append(chosen)
                selected_steps[sample_idx].append(chosen)
                validity_scores[sample_idx].append(cur_signal)
                last_selected[sample_idx] = chosen

                # Completion checks (mirror single-sample behavior)
                if chosen.is_trajectory_complete:
                    completion_reason = None
                    if chosen.other_data:
                        completion_reason = chosen.other_data.get("completion_reason")

                    if completion_reason == CompletionReason.EOS_PATTERN:
                        completed[sample_idx] = True
                        scores_str = ", ".join(
                            f"{s:.3f}" for s in validity_scores[sample_idx]
                        )
                        log.info(
                            f"Sample {sample_idxs[sample_idx]}: Completed (EOS) at step {step_num} "
                            f"with {len(selected_steps[sample_idx])} steps, "
                            f"scores=[{scores_str}]"
                        )
                        continue

                    if not self._has_answer_content(chosen):
                        trajectories[sample_idx].pop()
                        selected_steps[sample_idx].pop()
                        validity_scores[sample_idx].pop()
                        needs_final_answer[sample_idx] = True
                    completed[sample_idx] = True
                    scores_str = ", ".join(
                        f"{s:.3f}" for s in validity_scores[sample_idx]
                    )
                    log.info(
                        f"Sample {sample_idxs[sample_idx]}: Completed (answer pattern) at step {step_num} "
                        f"with {len(selected_steps[sample_idx])} steps, "
                        f"needs_final_answer={needs_final_answer[sample_idx]}, "
                        f"scores=[{scores_str}]"
                    )

        # Log samples that hit max_steps without completing
        for idx in range(num_samples):
            if not completed[idx]:
                scores_str = ", ".join(f"{s:.3f}" for s in validity_scores[idx])
                log.info(
                    f"Sample {sample_idxs[idx]}: Reached max_steps ({self.max_steps}) "
                    f"with {len(selected_steps[idx])} steps, "
                    f"scores=[{scores_str}]"
                )

        # ---- Final answer for samples that need it (batched) ----
        to_finalize: List[int] = []
        for idx in range(num_samples):
            if len(selected_steps[idx]) == 0:
                to_finalize.append(idx)
                continue
            if last_selected[idx] is None:
                to_finalize.append(idx)
                continue
            if not last_selected[idx].is_trajectory_complete:
                to_finalize.append(idx)
                continue
            if needs_final_answer[idx]:
                to_finalize.append(idx)

        if to_finalize:
            log.info(
                f"Generating final answers for {len(to_finalize)} samples "
                f"(samples: {[sample_idxs[i] for i in to_finalize]})"
            )
            fin_reqs = [requests[idx] for idx in to_finalize]
            fin_trajs = [trajectories[idx] for idx in to_finalize]

            answer_cands_batch = _gen_answer_candidates_batch(fin_reqs, fin_trajs)

            # Track tokens for final answer generation
            _track_generation_tokens(
                to_finalize, fin_reqs, fin_trajs, answer_cands_batch
            )

            answer_scores_batch = _score_candidates_batch(
                fin_reqs, answer_cands_batch, fin_trajs
            )

            for pos, sample_idx in enumerate(to_finalize):
                a_cands = answer_cands_batch[pos]
                a_scores = answer_scores_batch[pos]
                if not a_cands:
                    log.info(
                        f"Sample {sample_idxs[sample_idx]}: No final answer candidates generated"
                    )
                    continue
                best_idx = _select_best(a_scores)
                chosen = a_cands[best_idx]
                trajectories[sample_idx].append(chosen)
                selected_steps[sample_idx].append(chosen)
                validity_scores[sample_idx].append(a_scores[best_idx])
                last_selected[sample_idx] = chosen

        # ---- Finalize stats & build outputs ----
        self.step_generator.finalize_sample_stats(num_samples=num_samples)

        # Compute batch totals from per-sample stats
        total_input = sum(s["input_tokens"] for s in sample_token_stats.values())
        total_output = sum(s["output_tokens"] for s in sample_token_stats.values())
        total_tokens = total_input + total_output
        total_gens = sum(s["generation_count"] for s in sample_token_stats.values())
        batch_tflops = (
            self.step_generator.flop_calculator.compute_tflops(total_tokens)
            if hasattr(self.step_generator, "flop_calculator")
            and self.step_generator.flop_calculator
            else None
        )

        log.info(
            f"\n{'='*60}\n"
            f"Mini-batch complete: {num_samples} samples\n"
            f"Token stats (batch total): "
            f"total_tokens={total_tokens:,}, "
            f"input_tokens={total_input:,}, "
            f"output_tokens={total_output:,}, "
            f"generations={total_gens}"
            + (f", tflops={batch_tflops:.3f}" if batch_tflops else "")
            + f"\n{'='*60}"
        )

        outputs: List[Dict[str, Any]] = []
        for idx in range(num_samples):
            final_trajectory = convert_trajectory_to_string(trajectories[idx])
            extracted = extract_answer(final_trajectory)

            thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
                selected_steps[idx]
            )

            # Build per-sample token stats (like beam search _build_token_stats)
            raw = sample_token_stats[idx]
            sample_total = raw["input_tokens"] + raw["output_tokens"]
            token_stats = {
                "input_tokens": raw["input_tokens"],
                "output_tokens": raw["output_tokens"],
                "total_tokens_this_sample": sample_total,
                "generation_count": raw["generation_count"],
                "tflops": (
                    self.step_generator.flop_calculator.compute_tflops(sample_total)
                    if hasattr(self.step_generator, "flop_calculator")
                    and self.step_generator.flop_calculator
                    else None
                ),
            }

            scores_str = ", ".join(f"{s:.3f}" for s in validity_scores[idx])
            log.info(
                f"Sample {sample_idxs[idx]}: "
                f"{len(selected_steps[idx])} steps "
                f"({thinking_num_steps} thinking, {response_num_steps} response), "
                f"tokens={sample_total:,}, "
                f"scores=[{scores_str}], "
                f"answer={extracted!r}"
            )

            outputs.append(
                {
                    "trajectory": final_trajectory,
                    "extracted_answer": extracted,
                    "steps": selected_steps[idx],
                    "thinking_num_steps": thinking_num_steps,
                    "response_num_steps": response_num_steps,
                    "validity_scores": validity_scores[idx],
                    "completed": len(selected_steps[idx]) > 0,
                    "token_stats": token_stats,
                }
            )

        return outputs

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_idxs: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batched version of generate_trajectory that runs multiple samples "online" in parallel.
        """
        # split requests into mini-batches
        if sample_idxs is None:
            sample_idxs = list(range(len(requests)))
        mini_batches = [
            requests[i : min(i + self.batch_size, len(requests))]
            for i in range(0, len(requests), self.batch_size)
        ]
        sample_idxs = [
            sample_idxs[i : min(i + self.batch_size, len(sample_idxs))]
            for i in range(0, len(sample_idxs), self.batch_size)
        ]
        outputs = []
        for i, mini_batch in enumerate(mini_batches):
            log.info(f"Generating mini-batch {i+1} of {len(mini_batches)}")
            batch_outputs = self.generate_trajectory_mini_batch(
                mini_batch, sample_idxs[i]
            )
            outputs.extend(batch_outputs)
            log.info(
                f"Mini-batch {i+1} done, "
                f"{sum(1 for o in batch_outputs if o['completed'])}/{len(batch_outputs)} completed"
            )

        return outputs

    def cleanup(self):
        """Clean up resources"""

        self.scorer.cleanup()
        self.scale_discriminator.reset()
