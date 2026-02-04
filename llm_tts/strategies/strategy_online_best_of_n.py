import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import torch

from llm_tts.generators import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.generators.base import CompletionReason
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase, count_thinking_and_response_steps

log = logging.getLogger(__name__)

# Pattern to detect garbage/degenerate output
# Matches: emojis, CJK characters, unusual unicode, repeated nonsense patterns
_GARBAGE_PATTERN = re.compile(
    r"[\U0001F300-\U0001F9FF]"  # Emojis
    r"|[\u4E00-\u9FFF]"  # CJK Unified Ideographs (Chinese)
    r"|[\u3040-\u309F\u30A0-\u30FF]"  # Japanese Hiragana/Katakana
    r"|[\uFF01-\uFF60]"  # Fullwidth punctuation
    r"|[\u0100-\u024F]{2,}"  # Extended Latin with diacritics - 2+ consecutive
)


def _detect_garbage(text: str, threshold: int = 2) -> bool:
    """Detect garbage/degenerate output (emojis, CJK chars, unusual unicode)."""
    matches = _GARBAGE_PATTERN.findall(text)
    return len(matches) >= threshold


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
        batch_generation: bool = True,
        prompt_buffer: int = 500,
    ):
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.scorer = scorer
        self.step_generator = step_generator
        self.output_dir = output_dir
        self.batch_generation = batch_generation
        self.prompt_buffer = prompt_buffer
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
                - token_stats: Token and TFLOP statistics from generation
        """
        self._current_sample_idx = sample_idx
        self._steps_log = []

        # Reset token tracking for this sample
        self.step_generator.reset_sample_stats()
        if hasattr(self.scorer, "reset_prm_stats"):
            self.scorer.reset_prm_stats()

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

            # Score candidates - pass trajectory directly for PRM
            # Each trajectory step is one PRM step (not split by newlines)
            candidate_validity_scores = self.scorer.score_candidates(
                request, candidates, trajectory=trajectory
            )

            # Log all candidates with token stats
            log.info(f"\nGenerated {len(candidates)} candidates:")
            for i, (candidate, val_score) in enumerate(
                zip(candidates, candidate_validity_scores)
            ):
                # Get uncertainty score from other_data
                uncertainty = self._get_uncertainty_score(candidate)
                # Count tokens: generated (before truncation) vs truncated (in token_ids)
                truncated_tokens = (
                    len(candidate.token_ids) if candidate.token_ids else 0
                )
                # Original generated count stored in other_data, fallback to truncated
                generated_tokens = (
                    candidate.other_data.get("original_token_count", truncated_tokens)
                    if candidate.other_data
                    else truncated_tokens
                )
                tflops = (
                    self.step_generator.flop_calculator.compute_tflops(truncated_tokens)
                    if self.step_generator.flop_calculator
                    else 0
                )
                log.info(
                    f"\n[{i}] Validity: {val_score:.3f} | Uncertainty: {uncertainty:.3f} | "
                    f"Tokens (generated: {generated_tokens}, truncated: {truncated_tokens}) | "
                    f"TFLOPs: {tflops:.3f}\nText:\n{candidate.text}"
                )

            # Select best candidate
            best_idx, selected_candidate = self._select_best_candidate(
                candidates, candidate_validity_scores
            )
            all_scores_str = ", ".join(
                f"c{i}={s:.3f}" for i, s in enumerate(candidate_validity_scores)
            )
            log.info(
                f"\nSelected candidate {best_idx} "
                f"(score={candidate_validity_scores[best_idx]:.3f}), "
                f"all scores=[{all_scores_str}]"
            )

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
                # Get completion reason from candidate
                completion_reason = None
                if selected_candidate.other_data:
                    completion_reason = selected_candidate.other_data.get(
                        "completion_reason"
                    )

                # If stopped at EOS, response is already complete (e.g., Qwen2.5-Math with \boxed{})
                # No need to generate final answer
                if completion_reason == CompletionReason.EOS_PATTERN:
                    log.info("\nStopped at EOS, response already complete")
                    break

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

        # Finalize and get token statistics
        self.step_generator.finalize_sample_stats()
        token_stats = self.step_generator.get_sample_stats()

        # Merge PRM scorer stats if available
        if hasattr(self.scorer, "get_prm_total_stats"):
            prm_stats = self.scorer.get_prm_total_stats()
            token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
            token_stats["prm_tflops"] = prm_stats["prm_tflops"]
            token_stats["tflops"] = (token_stats.get("tflops") or 0) + (
                prm_stats["prm_tflops"] or 0
            )

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
            "thinking_num_steps": thinking_num_steps,
            "response_num_steps": response_num_steps,
            "validity_scores": validity_scores,
            "completed": len(selected_steps) > 0,
            "token_stats": token_stats,
        }

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate trajectories for ALL samples in parallel using batched online BoN.

        At each step, collects all active samples and makes ONE vLLM call
        to generate candidates for all of them, then scores and selects the best
        candidate per sample (greedy).

        This reduces vLLM calls from O(samples × steps) to O(steps).

        Args:
            requests: List of M chat message lists (each is a sample's request)
            sample_indices: Optional list of sample indices for logging

        Returns:
            List of M result dictionaries (one per sample)
        """
        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        M = len(requests)
        log.info(
            f"Batched Online BoN: {M} samples, candidates_per_step={self.candidates_per_step}, "
            f"max_steps={self.max_steps}"
        )

        # Check if scorer is a PRM model (separate model) or uses uncertainty from generation
        use_prm_scorer = (
            hasattr(self.scorer, "prm_model") and self.scorer.prm_model is not None
        )
        log.info(f"Using PRM scorer: {use_prm_scorer}")

        # Dispatch to pipelined path for API + non-PRM (entropy/perplexity scoring)
        from llm_tts.generators.api import StepCandidateGeneratorThroughAPI

        is_api_generator = isinstance(
            self.step_generator, StepCandidateGeneratorThroughAPI
        )
        if not use_prm_scorer and is_api_generator:
            return self._generate_trajectories_pipelined(requests, sample_indices)

        # Reset per-sample token tracking in generator
        self.step_generator.reset_per_sample_stats()
        if hasattr(self.scorer, "reset_prm_stats"):
            self.scorer.reset_prm_stats()

        # Context limit for trajectories
        max_context_budget = getattr(self.step_generator, "max_context_budget", 4096)
        max_step_tokens = getattr(self.step_generator, "max_step_tokens", 256)
        max_trajectory_tokens = min(
            max_context_budget - self.prompt_buffer,
            self.max_steps * max_step_tokens,
        )
        log.info(
            f"Max trajectory tokens: {max_trajectory_tokens} "
            f"(max_context_budget={max_context_budget}, prompt_buffer={self.prompt_buffer}, "
            f"max_steps={self.max_steps}, max_step_tokens={max_step_tokens})"
        )

        # Per-sample state
        trajectories: List[List[StepCandidate]] = [[] for _ in range(M)]
        selected_steps: List[List[StepCandidate]] = [[] for _ in range(M)]
        validity_scores: List[List[float]] = [[] for _ in range(M)]
        completed: List[bool] = [False] * M
        needs_final_answer: List[bool] = [False] * M
        total_tokens: List[int] = [0] * M

        for step_num in range(self.max_steps):
            # 1. Collect active sample indices
            active_sample_ids = [i for i in range(M) if not completed[i]]
            if not active_sample_ids:
                log.info(f"All samples completed at step {step_num}")
                break

            log.info(
                f"\n{'=' * 60}\n"
                f"Online BoN Step {step_num}: {len(active_sample_ids)} active samples\n"
                f"{'=' * 60}"
            )

            # 2. Skip samples whose trajectory exceeds context limit
            batch_sample_ids = []
            for i in active_sample_ids:
                if total_tokens[i] >= max_trajectory_tokens - 200:
                    log.info(
                        f"Sample {sample_indices[i]}: Context limit reached "
                        f"(tokens: {total_tokens[i]} >= {max_trajectory_tokens - 200}), "
                        f"marking for final answer"
                    )
                    completed[i] = True
                    needs_final_answer[i] = True
                else:
                    batch_sample_ids.append(i)

            if not batch_sample_ids:
                log.info("No active samples to process after context limit check")
                break

            # 3. Build batch requests/trajectories for remaining active samples
            batch_requests = [requests[i] for i in batch_sample_ids]
            batch_trajectories = [trajectories[i] for i in batch_sample_ids]

            log.info(
                f"Batched generation: {len(batch_requests)} samples "
                f"× {self.candidates_per_step} candidates"
            )

            # 4. ONE vLLM call: generate candidates for all active samples
            batch_results = self.step_generator.generate_step_candidates_batch(
                requests=batch_requests,
                trajectories=batch_trajectories,
                candidates_per_step=self.candidates_per_step,
                compute_uncertainty=not use_prm_scorer,
                sample_ids=batch_sample_ids,
            )

            # 5. Score candidates
            if use_prm_scorer:
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
            else:
                # Use validity scores from generation
                all_scores = []
                for candidates in batch_results:
                    scores = [
                        c.other_data.get("validity_score", 0.0) if c.other_data else 0.0
                        for c in candidates
                    ]
                    all_scores.append(scores)

            # 6. Select best candidate per sample (greedy max score)
            for batch_idx, sample_id in enumerate(batch_sample_ids):
                candidates = batch_results[batch_idx]
                scores = all_scores[batch_idx]

                if not candidates:
                    log.info(
                        f"Sample {sample_indices[sample_id]}: No candidates generated, "
                        f"marking complete"
                    )
                    completed[sample_id] = True
                    needs_final_answer[sample_id] = True
                    continue

                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                selected = candidates[best_idx]

                # Additional completion checks (matching beam search):
                # Track whether we forced completion via boxed/garbage detection
                forced_complete = False

                # Check if full trajectory contains a boxed answer
                if not selected.is_trajectory_complete:
                    full_traj_text = (
                        convert_trajectory_to_string(trajectories[sample_id])
                        + selected.text
                    )
                    has_boxed = bool(extract_answer(full_traj_text, "boxed"))
                    if has_boxed:
                        selected.is_trajectory_complete = True
                        forced_complete = True
                        log.info(
                            f"Sample {sample_indices[sample_id]}: Boxed answer detected"
                        )

                # Detect garbage/degenerate output
                if not selected.is_trajectory_complete and _detect_garbage(
                    selected.text
                ):
                    selected.is_trajectory_complete = True
                    forced_complete = True
                    log.info(
                        f"Sample {sample_indices[sample_id]}: Garbage output detected, "
                        f"marking complete"
                    )

                all_scores_str = ", ".join(
                    f"c{i}={s:.3f}" for i, s in enumerate(scores)
                )
                log.info(
                    f"Sample {sample_indices[sample_id]}: Selected candidate {best_idx} "
                    f"(score={scores[best_idx]:.3f}), all scores=[{all_scores_str}]"
                )

                # Track token count
                new_tokens = len(selected.token_ids) if selected.token_ids else 0
                total_tokens[sample_id] += new_tokens

                # 7. Append to trajectory, record validity score
                trajectories[sample_id].append(selected)
                selected_steps[sample_id].append(selected)
                validity_scores[sample_id].append(scores[best_idx])

                # 8. Completion checks
                if forced_complete:
                    # Boxed answer or garbage: keep the step, just mark done
                    completed[sample_id] = True
                elif selected.is_trajectory_complete:
                    completion_reason = None
                    if selected.other_data:
                        completion_reason = selected.other_data.get("completion_reason")

                    if completion_reason == CompletionReason.EOS_PATTERN:
                        log.info(f"Sample {sample_indices[sample_id]}: Stopped at EOS")
                        completed[sample_id] = True
                    elif not self._has_answer_content(selected):
                        log.info(
                            f"Sample {sample_indices[sample_id]}: Answer pattern without content, "
                            f"removing step and marking for final answer"
                        )
                        trajectories[sample_id].pop()
                        selected_steps[sample_id].pop()
                        validity_scores[sample_id].pop()
                        completed[sample_id] = True
                        needs_final_answer[sample_id] = True
                    else:
                        log.info(
                            f"Sample {sample_indices[sample_id]}: Answer pattern with content, done"
                        )
                        completed[sample_id] = True

                # Context limit check after appending
                if (
                    not completed[sample_id]
                    and total_tokens[sample_id] >= max_trajectory_tokens
                ):
                    log.info(
                        f"Sample {sample_indices[sample_id]}: Context limit reached after step "
                        f"(tokens: {total_tokens[sample_id]})"
                    )
                    completed[sample_id] = True
                    needs_final_answer[sample_id] = True

        # 9. Collect samples needing final answer (incomplete + needs_final_answer)
        to_finalize = []
        for i in range(M):
            if not completed[i]:
                # Reached max_steps without completing
                needs_final_answer[i] = True
            if needs_final_answer[i]:
                to_finalize.append(i)

        # 10. Batch generate final answers
        if to_finalize:
            log.info(
                f"Generating final answers for {len(to_finalize)} samples "
                f"(samples: {[sample_indices[i] for i in to_finalize]})"
            )

            fin_reqs = [requests[i] for i in to_finalize]
            fin_trajs = [trajectories[i] for i in to_finalize]

            # Generate answer candidates per sample (no batch API available)
            answer_cands_batch = [
                self.step_generator.generate_answer_candidates(
                    req,
                    trajectory=traj,
                    candidates_per_step=self.candidates_per_step,
                )
                for req, traj in zip(fin_reqs, fin_trajs)
            ]

            # 11. Record tokens for final answer generation
            for pos, sample_id in enumerate(to_finalize):
                if answer_cands_batch[pos]:
                    ctx_tokens = self.step_generator.count_context_tokens(
                        fin_reqs[pos], fin_trajs[pos]
                    )
                    self.step_generator.record_sample_tokens(
                        sample_id, answer_cands_batch[pos], context_tokens=ctx_tokens
                    )

            # 12. Score and select best final answer per sample
            for pos, sample_id in enumerate(to_finalize):
                a_cands = answer_cands_batch[pos]
                if not a_cands:
                    log.info(
                        f"Sample {sample_indices[sample_id]}: No final answer candidates"
                    )
                    continue

                # Score answer candidates
                a_scores = self.scorer.score_candidates(
                    fin_reqs[pos], a_cands, trajectory=fin_trajs[pos]
                )
                best_idx = max(range(len(a_scores)), key=lambda i: a_scores[i])

                log.info(
                    f"Sample {sample_indices[sample_id]}: Final answer selected "
                    f"(score={a_scores[best_idx]:.3f})"
                )

                trajectories[sample_id].append(a_cands[best_idx])
                selected_steps[sample_id].append(a_cands[best_idx])
                validity_scores[sample_id].append(a_scores[best_idx])

        # Finalize stats
        self.step_generator.finalize_sample_stats(num_samples=M)

        # Compute batch totals for logging
        total_input = 0
        total_output = 0
        total_gens = 0
        for idx in range(M):
            s = self.step_generator.get_sample_stats_for(idx)
            total_input += s["input_tokens"]
            total_output += s["output_tokens"]
            total_gens += s["generation_count"]
        batch_total_tokens = total_input + total_output
        batch_tflops = (
            self.step_generator.flop_calculator.compute_tflops(batch_total_tokens)
            if hasattr(self.step_generator, "flop_calculator")
            and self.step_generator.flop_calculator
            else None
        )
        log.info(
            f"\n{'='*60}\n"
            f"Batch complete: {M} samples\n"
            f"Token stats (batch total): "
            f"total_tokens={batch_total_tokens:,}, "
            f"input_tokens={total_input:,}, "
            f"output_tokens={total_output:,}, "
            f"generations={total_gens}"
            + (f", tflops={batch_tflops:.3f}" if batch_tflops else "")
            + f"\n{'='*60}"
        )

        # 13. Build result dicts
        outputs: List[Dict[str, Any]] = []
        for idx in range(M):
            final_trajectory = convert_trajectory_to_string(trajectories[idx])
            extracted = extract_answer(final_trajectory)

            thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
                selected_steps[idx]
            )

            token_stats = self.step_generator.get_sample_stats_for(idx)

            # Merge PRM scorer stats if available
            if hasattr(self.scorer, "get_prm_stats_for"):
                prm_stats = self.scorer.get_prm_stats_for(idx)
                token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
                token_stats["prm_tflops"] = prm_stats["prm_tflops"]
                token_stats["tflops"] = (token_stats.get("tflops") or 0) + (
                    prm_stats["prm_tflops"] or 0
                )

            scores_str = ", ".join(f"{s:.3f}" for s in validity_scores[idx])
            log.info(
                f"Sample {sample_indices[idx]}: "
                f"{len(selected_steps[idx])} steps "
                f"({thinking_num_steps} thinking, {response_num_steps} response), "
                f"tokens={token_stats['total_tokens_this_sample']:,}, "
                f"scores=[{scores_str}], "
                f"answer={extracted!r}"
            )
            for step_idx, step in enumerate(selected_steps[idx]):
                score = (
                    validity_scores[idx][step_idx]
                    if step_idx < len(validity_scores[idx])
                    else 0.0
                )
                log.info(f"  Step {step_idx + 1} (score={score:.3f}):\n{step.text}")

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

    def _generate_trajectories_pipelined(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Pipelined trajectory generation for API + non-PRM scoring.

        Each sample runs its full step loop independently. A shared semaphore
        limits concurrent API calls so total connections stay within budget.
        """
        M = len(requests)
        log.info(
            f"Pipelined Online BoN: {M} samples, candidates_per_step={self.candidates_per_step}, "
            f"max_steps={self.max_steps}"
        )

        # Reset per-sample token tracking
        self.step_generator.reset_per_sample_stats()
        # Pre-initialize per-sample stats keys so threads don't race on dict creation
        for i in range(M):
            self.step_generator._per_sample_stats[i] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "generation_count": 0,
            }

        # Semaphore: each generate call uses candidates_per_step concurrent connections
        max_concurrent = getattr(self.step_generator, "max_concurrent_requests", 256)
        sem_slots = max(1, max_concurrent // self.candidates_per_step)
        semaphore = threading.Semaphore(sem_slots)
        log.info(
            f"Pipelined concurrency: semaphore={sem_slots} slots "
            f"(max_concurrent_requests={max_concurrent}, "
            f"candidates_per_step={self.candidates_per_step})"
        )

        # Context limit
        max_context_budget = getattr(self.step_generator, "max_context_budget", 4096)
        max_step_tokens = getattr(self.step_generator, "max_step_tokens", 256)
        max_trajectory_tokens = min(
            max_context_budget - self.prompt_buffer,
            self.max_steps * max_step_tokens,
        )

        results: List[Optional[Dict[str, Any]]] = [None] * M
        completed_count = [0]
        completed_lock = threading.Lock()

        def process_sample(sample_id: int) -> Dict[str, Any]:
            return self._process_single_sample(
                sample_id=sample_id,
                sample_idx=sample_indices[sample_id],
                request=requests[sample_id],
                semaphore=semaphore,
                max_trajectory_tokens=max_trajectory_tokens,
            )

        with ThreadPoolExecutor(max_workers=M) as executor:
            future_to_id = {executor.submit(process_sample, i): i for i in range(M)}
            for future in as_completed(future_to_id):
                sid = future_to_id[future]
                try:
                    results[sid] = future.result()
                except Exception:
                    log.exception(
                        f"Sample {sample_indices[sid]}: Unhandled exception in pipelined worker"
                    )
                    results[sid] = {
                        "trajectory": "",
                        "extracted_answer": None,
                        "steps": [],
                        "thinking_num_steps": 0,
                        "response_num_steps": 0,
                        "validity_scores": [],
                        "completed": False,
                        "token_stats": self.step_generator.get_sample_stats_for(sid),
                    }
                with completed_lock:
                    completed_count[0] += 1
                    n_done = completed_count[0]
                r = results[sid]
                n_steps = len(r["steps"]) if r else 0
                log.info(
                    f"Sample {sample_indices[sid]} done ({n_steps} steps) — "
                    f"{n_done}/{M} completed, {M - n_done} active"
                )

        # Finalize stats
        self.step_generator.finalize_sample_stats(num_samples=M)

        # Batch-level logging
        total_input = 0
        total_output = 0
        total_gens = 0
        for idx in range(M):
            s = self.step_generator.get_sample_stats_for(idx)
            total_input += s["input_tokens"]
            total_output += s["output_tokens"]
            total_gens += s["generation_count"]
        batch_total_tokens = total_input + total_output
        batch_tflops = (
            self.step_generator.flop_calculator.compute_tflops(batch_total_tokens)
            if hasattr(self.step_generator, "flop_calculator")
            and self.step_generator.flop_calculator
            else None
        )
        log.info(
            f"\n{'='*60}\n"
            f"Pipelined batch complete: {M} samples\n"
            f"Token stats (batch total): "
            f"total_tokens={batch_total_tokens:,}, "
            f"input_tokens={total_input:,}, "
            f"output_tokens={total_output:,}, "
            f"generations={total_gens}"
            + (f", tflops={batch_tflops:.3f}" if batch_tflops else "")
            + f"\n{'='*60}"
        )

        # Per-sample logging
        for idx in range(M):
            r = results[idx]
            token_stats = r["token_stats"]
            scores_str = ", ".join(f"{s:.3f}" for s in r["validity_scores"])
            log.info(
                f"Sample {sample_indices[idx]}: "
                f"{len(r['steps'])} steps "
                f"({r['thinking_num_steps']} thinking, {r['response_num_steps']} response), "
                f"tokens={token_stats['total_tokens_this_sample']:,}, "
                f"scores=[{scores_str}], "
                f"answer={r['extracted_answer']!r}"
            )
            for step_idx, step in enumerate(r["steps"]):
                score = (
                    r["validity_scores"][step_idx]
                    if step_idx < len(r["validity_scores"])
                    else 0.0
                )
                log.info(f"  Step {step_idx + 1} (score={score:.3f}):\n{step.text}")

        return results

    def _process_single_sample(
        self,
        sample_id: int,
        sample_idx: int,
        request: List[Dict[str, str]],
        semaphore: threading.Semaphore,
        max_trajectory_tokens: int,
    ) -> Dict[str, Any]:
        """Run the full step loop for a single sample (called from a worker thread)."""
        trajectory: List[StepCandidate] = []
        selected_steps: List[StepCandidate] = []
        validity_scores: List[float] = []
        total_toks = 0

        for step_num in range(self.max_steps):
            # Context limit pre-check
            if total_toks >= max_trajectory_tokens - 200:
                log.info(
                    f"Sample {sample_idx}: Context limit reached "
                    f"(tokens: {total_toks} >= {max_trajectory_tokens - 200}), "
                    f"generating final answer"
                )
                break

            log.info(
                f"Sample {sample_idx}: Step {step_num} "
                f"(trajectory tokens: {total_toks})"
            )

            # Generate candidates (acquire semaphore for API budget)
            semaphore.acquire()
            try:
                batch_results = self.step_generator.generate_step_candidates_batch(
                    requests=[request],
                    trajectories=[trajectory],
                    candidates_per_step=self.candidates_per_step,
                    compute_uncertainty=True,
                    sample_ids=[sample_id],
                )
            finally:
                semaphore.release()

            candidates = batch_results[0] if batch_results else []
            if not candidates:
                log.info(f"Sample {sample_idx}: No candidates generated, stopping")
                break

            # Score from validity_score (instant, no external scorer call)
            scores = [
                c.other_data.get("validity_score", 0.0) if c.other_data else 0.0
                for c in candidates
            ]

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            selected = candidates[best_idx]

            # Additional completion checks
            forced_complete = False

            if not selected.is_trajectory_complete:
                full_traj_text = (
                    convert_trajectory_to_string(trajectory) + selected.text
                )
                has_boxed = bool(extract_answer(full_traj_text, "boxed"))
                if has_boxed:
                    selected.is_trajectory_complete = True
                    forced_complete = True
                    log.info(f"Sample {sample_idx}: Boxed answer detected")

            if not selected.is_trajectory_complete and _detect_garbage(selected.text):
                selected.is_trajectory_complete = True
                forced_complete = True
                log.info(
                    f"Sample {sample_idx}: Garbage output detected, marking complete"
                )

            all_scores_str = ", ".join(f"c{i}={s:.3f}" for i, s in enumerate(scores))
            log.info(
                f"Sample {sample_idx}: Selected candidate {best_idx} "
                f"(score={scores[best_idx]:.3f}), all scores=[{all_scores_str}]"
            )

            new_tokens = len(selected.token_ids) if selected.token_ids else 0
            total_toks += new_tokens

            trajectory.append(selected)
            selected_steps.append(selected)
            validity_scores.append(scores[best_idx])

            # Completion checks
            if forced_complete:
                break

            if selected.is_trajectory_complete:
                completion_reason = None
                if selected.other_data:
                    completion_reason = selected.other_data.get("completion_reason")

                if completion_reason == CompletionReason.EOS_PATTERN:
                    log.info(f"Sample {sample_idx}: Stopped at EOS")
                    break
                elif not self._has_answer_content(selected):
                    log.info(
                        f"Sample {sample_idx}: Answer pattern without content, "
                        f"removing step and generating final answer"
                    )
                    trajectory.pop()
                    selected_steps.pop()
                    validity_scores.pop()
                    break
                else:
                    log.info(f"Sample {sample_idx}: Answer pattern with content, done")
                    break

            # Context limit after appending
            if total_toks >= max_trajectory_tokens:
                log.info(
                    f"Sample {sample_idx}: Context limit reached after step "
                    f"(tokens: {total_toks})"
                )
                break

        # Check if we need a final answer
        needs_final = False
        if not selected_steps:
            needs_final = True
        elif not selected_steps[-1].is_trajectory_complete:
            needs_final = True

        if needs_final:
            log.info(f"Sample {sample_idx}: Generating final answer")
            semaphore.acquire()
            try:
                answer_cands = self.step_generator.generate_answer_candidates(
                    request,
                    trajectory=trajectory,
                    candidates_per_step=self.candidates_per_step,
                )
            finally:
                semaphore.release()

            if answer_cands:
                # Record tokens for the final answer generation
                ctx_tokens = self.step_generator.count_context_tokens(
                    request, trajectory
                )
                self.step_generator.record_sample_tokens(
                    sample_id, answer_cands, context_tokens=ctx_tokens
                )

                a_scores = [
                    c.other_data.get("validity_score", 0.0) if c.other_data else 0.0
                    for c in answer_cands
                ]
                best_a = max(range(len(a_scores)), key=lambda i: a_scores[i])

                log.info(
                    f"Sample {sample_idx}: Final answer selected "
                    f"(score={a_scores[best_a]:.3f})"
                )
                trajectory.append(answer_cands[best_a])
                selected_steps.append(answer_cands[best_a])
                validity_scores.append(a_scores[best_a])

        # Build result
        final_trajectory = convert_trajectory_to_string(trajectory)
        extracted = extract_answer(final_trajectory)
        thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
            selected_steps
        )
        token_stats = self.step_generator.get_sample_stats_for(sample_id)

        return {
            "trajectory": final_trajectory,
            "extracted_answer": extracted,
            "steps": selected_steps,
            "thinking_num_steps": thinking_num_steps,
            "response_num_steps": response_num_steps,
            "validity_scores": validity_scores,
            "completed": len(selected_steps) > 0,
            "token_stats": token_stats,
        }

    def _get_uncertainty_score(self, candidate: "StepCandidate") -> float:
        """Get uncertainty_score from candidate, logging error if missing."""
        if candidate.other_data is None:
            log.error(f"Candidate has no other_data! Text: {candidate.text[:100]}...")
            return 0.0
        if "uncertainty_score" not in candidate.other_data:
            log.error(
                f"uncertainty_score missing from candidate.other_data! "
                f"Keys: {list(candidate.other_data.keys())}, "
                f"Text: {candidate.text[:100]}..."
            )
            return 0.0
        return candidate.other_data["uncertainty_score"]

    def _select_best_candidate(self, candidates: List, scores: List[float]) -> tuple:
        """Select the best candidate based on scores"""

        # Higher validity is better
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, candidates[best_idx]

    def _generate_final_answer(
        self, chat: List[Dict[str, str]], trajectory: List[StepCandidate]
    ) -> tuple:
        """Generate and select best final answer based on criterion"""

        # Generate answer candidates (token recording handled by generator)
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
        # Build candidate data with token stats
        candidates_data = []
        for i, c in enumerate(candidates):
            num_tokens = len(c.token_ids) if c.token_ids else 0
            tflops = (
                self.step_generator.flop_calculator.compute_tflops(num_tokens)
                if self.step_generator.flop_calculator
                else None
            )
            candidates_data.append(
                {
                    "idx": i,
                    "text": c.text,
                    "validity_score": float(scores[i]),
                    "num_tokens": num_tokens,
                    "tflops": tflops,
                    "is_complete": c.is_complete,
                    "is_trajectory_complete": c.is_trajectory_complete,
                    "token_ids": list(c.token_ids) if c.token_ids else [],
                    "logprobs": (
                        c.other_data.get("logprobs", []) if c.other_data else []
                    ),
                }
            )

        # Get selected candidate's token stats
        selected_tokens = candidates_data[selected_idx]["num_tokens"]
        selected_tflops = candidates_data[selected_idx]["tflops"]

        step_data = {
            "step_num": step_num,
            "candidates": candidates_data,
            "selected_idx": selected_idx,
            "selected_text": candidates[selected_idx].text,
            "selected_score": float(scores[selected_idx]),
            "selected_tokens": selected_tokens,
            "selected_tflops": selected_tflops,
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
