import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from llm_tts.generators import (
    StepCandidate,
    StepCandidateGeneratorThroughAPI,
    StepCandidateGeneratorThroughHuggingface,
    convert_trajectory_to_string,
)
from llm_tts.generators.base import CompletionReason
from llm_tts.utils import extract_answer

if TYPE_CHECKING:
    from llm_tts.generators import VLLMStepGenerator

from llm_tts.scale_discriminator import ScaleDiscriminator

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


def calculate_perplexity_score(candidate: StepCandidate) -> float:
    """Calculate perplexity score from candidate's other_data."""
    logprobs = candidate.other_data.get("logprobs", []) if candidate.other_data else []
    if not logprobs:
        return 0.0
    return -np.mean(logprobs)


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

    def _select_best_candidate(self, candidates: List, scores: List[float]) -> tuple:
        """Select the best candidate based on scores"""

        # Higher validity is better
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, candidates[best_idx]

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
        answer_steps: List[Optional[str]] = [None for _ in range(num_samples)]

        # Reset batch-wide and per-sample token tracking
        self.step_generator.reset_sample_stats()
        self.step_generator.reset_per_sample_stats()

        # ---- Batched helpers (duck-typing) ----
        def _gen_step_candidates_batch(
            active_reqs: List[List[Dict[str, str]]],
            active_trajs: List[List[StepCandidate]],
            candidates_per_step: int,
            sample_ids: Optional[List[int]] = None,
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
                    sample_ids=sample_ids,
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
            batch_results: List[List[StepCandidate]],
            batch_requests: List[List[Dict[str, str]]] = None,
            batch_trajectories: List[List[StepCandidate]] = None,
            batch_sample_ids: List[int] = None,
            use_prm=False,
        ) -> List[List[float]]:
            """Returns list-of-list of scores aligned to active samples."""
            if not use_prm:
                all_scores = []
                for candidates in batch_results:
                    scores = [
                        c.other_data.get("validity_score", 0.0) if c.other_data else 0.0
                        for c in candidates
                    ]
                    all_scores.append(scores)
                return all_scores
            else:
                # PRM needs full trajectory context (parent steps + candidate)
                # Build flat lists for batch scoring
                flat_chats = []
                flat_trajectories = []
                flat_sample_ids = []  # Track sample IDs for PRM token accounting
                candidate_map = []  # (batch_idx, cand_idx)

                for batch_idx in range(len(batch_results)):
                    req = batch_requests[batch_idx]
                    parent_traj = batch_trajectories[batch_idx]
                    sample_id = batch_sample_ids[batch_idx]
                    for cand_idx, cand in enumerate(batch_results[batch_idx]):
                        full_traj = parent_traj + [cand]
                        flat_chats.append(req)
                        flat_trajectories.append(full_traj)
                        flat_sample_ids.append(sample_id)
                        candidate_map.append((batch_idx, cand_idx))

                # Score all trajectories in batch
                if flat_trajectories and hasattr(
                    self.scorer, "score_trajectories_batch"
                ):
                    all_traj_scores = self.scorer.score_trajectories_batch(
                        flat_chats,
                        flat_trajectories,
                        sample_ids=flat_sample_ids,
                    )
                    flat_scores = [
                        traj_scores[-1] if traj_scores else 0.0
                        for traj_scores in all_traj_scores
                    ]
                elif flat_trajectories:
                    # Fallback: score one by one
                    flat_scores = []
                    for chat, traj in zip(flat_chats, flat_trajectories):
                        score_list = self.scorer.score_trajectory(chat, traj)
                        flat_scores.append(score_list[-1] if score_list else 0.0)
                else:
                    flat_scores = []

                # Map flat scores back to per-sample lists
                all_scores = [[] for _ in range(len(batch_results))]
                for (batch_idx, cand_idx), score in zip(candidate_map, flat_scores):
                    all_scores[batch_idx].append(score)
                return all_scores

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

        use_prm_scorer = (
            hasattr(self.scorer, "prm_model") and self.scorer.prm_model is not None
        )
        if hasattr(self.scorer, "reset_prm_stats"):
            self.scorer.reset_prm_stats()
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
                active_reqs,
                active_trajs,
                candidates_per_step=1,
                sample_ids=active_indices,
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

            # 2) Decide which samples should scale (using perplexity signal)
            scale_indices = []
            scale_reqs = []
            scale_trajs = []
            for pos, sample_idx in enumerate(filtered_indices):
                if scale_discriminators[sample_idx].should_scale(
                    calculate_perplexity_score(filtered_cands[pos][0])
                ):
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
                    sample_ids=scale_indices,
                )

                scale_scores_batch = _score_candidates_batch(
                    scale_cands_batch,
                    batch_requests=scale_reqs,
                    batch_trajectories=scale_trajs,
                    batch_sample_ids=scale_indices,
                    use_prm=use_prm_scorer,
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
                else:
                    chosen = filtered_cands[pos][0]

                scale_discriminators[sample_idx].update(
                    calculate_perplexity_score(chosen)
                )
                trajectories[sample_idx].append(chosen)
                selected_steps[sample_idx].append(chosen)
                validity_scores[sample_idx].append(
                    chosen.other_data.get("validity_score", 0.0)
                )
                last_selected[sample_idx] = chosen

                # Check for thinking mode completion
                if (
                    getattr(self.step_generator, "thinking_mode", False)
                    and chosen.is_thinking_complete
                ):
                    # Thinking phase complete: mark for final answer generation.
                    if chosen.is_trajectory_complete:
                        reason = chosen.other_data.get("completion_reason") if chosen.other_data else None
                        log.warning(
                            f"Sample {sample_idxs[sample_idx]}: "
                            f"thinking complete but is_trajectory_complete was set "
                            f"(reason={reason}), resetting for answer generation"
                        )
                        chosen.is_trajectory_complete = False
                    completed[sample_idx] = True
                    needs_final_answer[sample_idx] = True
                    scores_str = ", ".join(
                        f"{s:.3f}" for s in validity_scores[sample_idx]
                    )
                    log.info(
                        f"Sample {sample_idxs[sample_idx]}: "
                        f"thinking complete (`` detected) at step {step_num}, "
                        f"marking for answer generation, "
                        f"scores=[{scores_str}]"
                    )
                    continue

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

        # Generate final answers only in thinking mode — non-thinking mode
        # produces the answer naturally in the last reasoning step.
        thinking_mode = getattr(self.step_generator, "thinking_mode", False)
        if to_finalize and thinking_mode:
            log.info(
                f"Generating final answers for {len(to_finalize)} samples "
                f"(samples: {[sample_idxs[i] for i in to_finalize]})"
            )
            fin_reqs = [requests[idx] for idx in to_finalize]
            fin_trajs = [trajectories[idx] for idx in to_finalize]

            answer_cands_batch = _gen_answer_candidates_batch(fin_reqs, fin_trajs)

            # Record tokens for final answer generation (not going through batch API)
            for pos, sample_idx in enumerate(to_finalize):
                if answer_cands_batch[pos]:
                    ctx_tokens = self.step_generator.count_context_tokens(
                        fin_reqs[pos], fin_trajs[pos]
                    )
                    self.step_generator.record_sample_tokens(
                        sample_idx, answer_cands_batch[pos], context_tokens=ctx_tokens
                    )

            answer_scores_batch = _score_candidates_batch(
                answer_cands_batch,
                fin_reqs,
                fin_trajs,
                batch_sample_ids=to_finalize,
                use_prm=use_prm_scorer,
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
                # Don't append to selected_steps/validity_scores —
                # answer is stored separately, same as offline BoN
                last_selected[sample_idx] = chosen
                # Store answer_step text for thinking mode
                answer_steps[sample_idx] = (
                    chosen.raw_text if chosen.raw_text else chosen.text
                )

        # ---- Finalize stats & build outputs ----
        self.step_generator.finalize_sample_stats(num_samples=num_samples)

        # Compute batch totals from per-sample stats
        total_input = 0
        total_output = 0
        total_gens = 0
        for idx in range(num_samples):
            s = self.step_generator.get_sample_stats_for(idx)
            total_input += s["input_tokens"]
            total_output += s["output_tokens"]
            total_gens += s["generation_count"]
        total_tokens = total_input + total_output
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

            reasoning_steps = len(selected_steps[idx])

            # Get per-sample token stats from generator's tracking
            token_stats = self.step_generator.get_sample_stats_for(idx)
            sample_total = token_stats["total_tokens_this_sample"]

            # Merge PRM scorer stats if available
            if hasattr(self.scorer, "get_prm_total_stats"):
                prm_stats = self.scorer.get_prm_stats_for(idx)
                token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
                token_stats["prm_tflops"] = prm_stats["prm_tflops"]
                token_stats["tflops"] = (token_stats.get("tflops") or 0) + (
                    prm_stats["prm_tflops"] or 0
                )
            scores_str = ", ".join(f"{s:.3f}" for s in validity_scores[idx])
            log.info(
                f"Sample {sample_idxs[idx]}: "
                f"{len(selected_steps[idx])} steps "
                f"({reasoning_steps} reasoning steps), "
                f"tokens={sample_total:,}, "
                f"scores=[{scores_str}], "
                f"answer={extracted!r}"
            )

            outputs.append(
                {
                    "trajectory": final_trajectory,
                    "extracted_answer": extracted,
                    "steps": selected_steps[idx],
                    "reasoning_steps": reasoning_steps,
                    "validity_scores": validity_scores[idx],
                    "completed": len(selected_steps[idx]) > 0,
                    "token_stats": token_stats,
                    "answer_step": answer_steps[idx],
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
