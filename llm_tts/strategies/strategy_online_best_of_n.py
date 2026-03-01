import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from llm_tts.generators import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.generators.base import CompletionReason, get_completion_info
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase

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

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: Optional[List[int]] = None,
        save_callback=None,
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
        max_context_budget = getattr(self.step_generator, "context_budget", 4096)
        max_step_tokens = getattr(self.step_generator, "step_token_limit", 256)
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
        answer_steps: List[Optional[str]] = [None] * M
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
                    flat_scores = []
                    for i, traj_scores in enumerate(all_traj_scores):
                        score = traj_scores[-1] if traj_scores else None
                        if score is None:
                            batch_idx, cand_idx = candidate_map[i]
                            n_steps = len(traj_scores) if traj_scores else 0
                            n_null = (
                                sum(1 for s in traj_scores if s is None)
                                if traj_scores
                                else 0
                            )
                            log.warning(
                                f"PRM returned no valid score for "
                                f"candidate {cand_idx} (batch {batch_idx}): "
                                f"{n_null}/{n_steps} steps are null "
                                f"(likely a very short candidate). "
                                f"This candidate will be skipped "
                                f"during selection."
                            )
                        flat_scores.append(score)
                elif flat_trajectories:
                    # Fallback: score one by one
                    flat_scores = []
                    for i, (chat, traj) in enumerate(
                        zip(flat_chats, flat_trajectories)
                    ):
                        score_list = self.scorer.score_trajectory(chat, traj)
                        score = score_list[-1] if score_list else None
                        if score is None:
                            batch_idx, cand_idx = candidate_map[i]
                            log.warning(
                                f"PRM returned no valid score for "
                                f"candidate {cand_idx} (batch {batch_idx}). "
                                f"This candidate will be skipped "
                                f"during selection."
                            )
                        flat_scores.append(score)
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
                    scores = []
                    for c in candidates:
                        data = c.other_data if c.other_data else {}
                        v = data.get("validity_score")
                        if v is None:
                            log.warning(
                                "Missing 'validity_score' in candidate other_data"
                            )
                            v = 0.0
                        scores.append(v)
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

                if not scores:
                    log.warning(
                        f"Sample {sample_indices[sample_id]}: Empty scores for "
                        f"{len(candidates)} candidates, marking complete"
                    )
                    completed[sample_id] = True
                    needs_final_answer[sample_id] = True
                    continue

                valid_indices = [i for i, s in enumerate(scores) if s is not None]
                if not valid_indices:
                    log.warning(
                        f"Sample {sample_indices[sample_id]}: All scores are None "
                        f"for {len(candidates)} candidates, selecting index 0"
                    )
                    best_idx = 0
                else:
                    best_idx = max(valid_indices, key=lambda i: scores[i])
                selected = candidates[best_idx]

                # Additional completion checks (matching beam search):
                # Track whether we forced completion via boxed/garbage detection
                forced_complete = False

                # Check if full trajectory contains a boxed answer
                # (skip for thinking-mode steps — boxed inside <think> is not final)
                if (
                    not selected.is_trajectory_complete
                    and not selected.is_thinking_complete
                ):
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
                # (skip for thinking-mode steps — answer phase still needed)
                if (
                    not selected.is_trajectory_complete
                    and not selected.is_thinking_complete
                    and _detect_garbage(selected.text)
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
                if (
                    getattr(self.step_generator, "thinking_mode", False)
                    and selected.is_thinking_complete
                ):
                    # Thinking phase complete: generate answer via
                    # generate_answer_candidates (proper stop tokens, not
                    # step-level splitting).
                    completed[sample_id] = True
                    if selected.is_trajectory_complete:
                        reason = (
                            selected.other_data.get("completion_reason")
                            if selected.other_data
                            else None
                        )
                        log.warning(
                            f"Sample {sample_indices[sample_id]}: "
                            f"thinking complete but is_trajectory_complete was set "
                            f"(reason={reason}), skipping answer generation"
                        )
                    else:
                        needs_final_answer[sample_id] = True
                        log.info(
                            f"Sample {sample_indices[sample_id]}: "
                            f"thinking complete, marking for answer generation"
                        )
                elif forced_complete:
                    # Boxed answer or garbage: keep the step, mark done.
                    # In thinking mode, still need proper answer generation
                    # after closing </think>.
                    completed[sample_id] = True
                    if getattr(self.step_generator, "thinking_mode", False):
                        needs_final_answer[sample_id] = True
                elif selected.is_trajectory_complete:
                    completion_reason = None
                    if selected.other_data:
                        completion_reason = selected.other_data.get("completion_reason")

                    if completion_reason == CompletionReason.EOS_PATTERN:
                        log.info(f"Sample {sample_indices[sample_id]}: Stopped at EOS")
                    else:
                        log.info(
                            f"Sample {sample_indices[sample_id]}: Answer pattern detected"
                        )
                        # In thinking mode, answer pattern before </think> means
                        # the model put \boxed{} inside reasoning — still need
                        # proper answer generation after closing </think>.
                        if getattr(self.step_generator, "thinking_mode", False):
                            needs_final_answer[sample_id] = True
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
                log.warning(f"Sample {i}: Reached max_steps without completing")
                needs_final_answer[i] = True
            if needs_final_answer[i]:
                to_finalize.append(i)

        # 10. Batch generate final answers for samples that need them
        if to_finalize:
            log.info(
                f"Generating final answers for {len(to_finalize)} samples "
                f"(samples: {[sample_indices[i] for i in to_finalize]})"
            )

            fin_reqs = [requests[i] for i in to_finalize]
            fin_trajs = [trajectories[i] for i in to_finalize]

            # Batch generate answer candidates in single call
            answer_cands_batch = self.step_generator.generate_answer_candidates_batch(
                fin_reqs,
                trajectories=fin_trajs,
                candidates_per_step=self.candidates_per_step,
            )

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
                if not a_scores:
                    log.warning(
                        f"Sample {sample_indices[sample_id]}: Empty scores for "
                        f"{len(a_cands)} final answer candidates, skipping"
                    )
                    continue
                valid_a_indices = [i for i, s in enumerate(a_scores) if s is not None]
                if not valid_a_indices:
                    log.warning(
                        f"Sample {sample_indices[sample_id]}: All final answer "
                        f"scores are None, selecting index 0"
                    )
                    best_idx = 0
                else:
                    best_idx = max(valid_a_indices, key=lambda i: a_scores[i])

                score_str = (
                    f"{a_scores[best_idx]:.3f}"
                    if a_scores[best_idx] is not None
                    else "None"
                )
                log.info(
                    f"Sample {sample_indices[sample_id]}: Final answer selected "
                    f"(score={score_str})"
                )

                trajectories[sample_id].append(a_cands[best_idx])
                # Don't append to selected_steps/validity_scores —
                # answer is stored separately, same as offline BoN
                answer_steps[sample_id] = (
                    a_cands[best_idx].raw_text or a_cands[best_idx].text
                )

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

            # Answer step is stored separately (not in selected_steps),
            # so selected_steps contains only reasoning steps
            reasoning_steps = len(selected_steps[idx])

            token_stats = self.step_generator.get_sample_stats_for(idx)

            # Merge PRM scorer stats if available
            if hasattr(self.scorer, "get_prm_stats_for"):
                prm_stats = self.scorer.get_prm_stats_for(idx)
                token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
                token_stats["prm_tflops"] = prm_stats["prm_tflops"]
                gen_tflops = token_stats.get("tflops")
                if gen_tflops is None:
                    log.warning(
                        f"Sample {sample_indices[idx]}: missing 'tflops' in token_stats when merging PRM stats"
                    )
                    gen_tflops = 0
                prm_tflops = prm_stats["prm_tflops"]
                if prm_tflops is None:
                    log.warning(
                        f"Sample {sample_indices[idx]}: missing 'prm_tflops' in PRM stats"
                    )
                    prm_tflops = 0
                token_stats["tflops"] = gen_tflops + prm_tflops

            scores_str = ", ".join(f"{s:.3f}" for s in validity_scores[idx])
            log.info(
                f"Sample {sample_indices[idx]}: "
                f"{len(selected_steps[idx])} steps "
                f"({reasoning_steps} reasoning steps), "
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

            result = {
                "trajectory": final_trajectory,
                "extracted_answer": extracted,
                "steps": selected_steps[idx],
                "answer_step": answer_steps[idx],
                "reasoning_steps": reasoning_steps,
                "validity_scores": validity_scores[idx],
                "completed": bool(selected_steps[idx])
                and selected_steps[idx][-1].is_trajectory_complete,
                "token_stats": token_stats,
            }
            result.update(get_completion_info(selected_steps[idx]))
            outputs.append(result)

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
        max_context_budget = getattr(self.step_generator, "context_budget", 4096)
        max_step_tokens = getattr(self.step_generator, "step_token_limit", 256)
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
                        "reasoning_steps": 0,
                        "validity_scores": [],
                        "completed": False,
                        "token_stats": self.step_generator.get_sample_stats_for(sid),
                        **get_completion_info([]),
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
                f"({r['reasoning_steps']} reasoning steps), "
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
        needs_thinking_answer = False

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
            scores = []
            for c in candidates:
                data = c.other_data if c.other_data else {}
                v = data.get("validity_score")
                if v is None:
                    log.warning(
                        f"Sample {sample_idx}: missing 'validity_score' in candidate other_data"
                    )
                    v = 0.0
                scores.append(v)

            if not scores:
                log.warning(
                    f"Sample {sample_idx}: Empty scores for "
                    f"{len(candidates)} candidates, stopping"
                )
                break

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            selected = candidates[best_idx]

            # Additional completion checks
            forced_complete = False

            # Skip boxed/garbage checks for thinking-mode steps (answer phase still needed)
            if (
                not selected.is_trajectory_complete
                and not selected.is_thinking_complete
            ):
                full_traj_text = (
                    convert_trajectory_to_string(trajectory) + selected.text
                )
                has_boxed = bool(extract_answer(full_traj_text, "boxed"))
                if has_boxed:
                    selected.is_trajectory_complete = True
                    forced_complete = True
                    log.info(f"Sample {sample_idx}: Boxed answer detected")

            if (
                not selected.is_trajectory_complete
                and not selected.is_thinking_complete
                and _detect_garbage(selected.text)
            ):
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
            if (
                getattr(self.step_generator, "thinking_mode", False)
                and selected.is_thinking_complete
            ):
                if selected.is_trajectory_complete:
                    reason = (
                        selected.other_data.get("completion_reason")
                        if selected.other_data
                        else None
                    )
                    log.warning(
                        f"Sample {sample_idx}: "
                        f"thinking complete but is_trajectory_complete was set "
                        f"(reason={reason}), skipping answer generation"
                    )
                else:
                    log.info(
                        f"Sample {sample_idx}: "
                        f"thinking complete, marking for answer generation"
                    )
                break

            if forced_complete:
                # In thinking mode, still need proper answer generation
                # after closing </think>.
                if getattr(self.step_generator, "thinking_mode", False):
                    needs_thinking_answer = True
                break

            if selected.is_trajectory_complete:
                completion_reason = None
                if selected.other_data:
                    completion_reason = selected.other_data.get("completion_reason")

                if completion_reason == CompletionReason.EOS_PATTERN:
                    log.info(f"Sample {sample_idx}: Stopped at EOS")
                else:
                    log.info(f"Sample {sample_idx}: Answer pattern detected")
                    # In thinking mode, answer pattern before </think> means
                    # the model put \boxed{} inside reasoning — still need
                    # proper answer generation after closing </think>.
                    if getattr(self.step_generator, "thinking_mode", False):
                        needs_thinking_answer = True
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
        answer_text = None
        if not selected_steps:
            needs_final = True
        elif not selected_steps[-1].is_trajectory_complete:
            needs_final = True
        elif needs_thinking_answer:
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

                a_scores = []
                for c in answer_cands:
                    data = c.other_data if c.other_data else {}
                    v = data.get("validity_score")
                    if v is None:
                        log.warning(
                            f"Sample {sample_idx}: missing 'validity_score' in answer candidate other_data"
                        )
                        v = 0.0
                    a_scores.append(v)

                if not a_scores:
                    log.warning(
                        f"Sample {sample_idx}: Empty scores for "
                        f"{len(answer_cands)} answer candidates, skipping final answer"
                    )
                else:
                    best_a = max(range(len(a_scores)), key=lambda i: a_scores[i])

                    log.info(
                        f"Sample {sample_idx}: Final answer selected "
                        f"(score={a_scores[best_a]:.3f})"
                    )
                    trajectory.append(answer_cands[best_a])
                    # Don't append to selected_steps/validity_scores —
                    # answer is stored separately, same as offline BoN
                    answer_text = (
                        answer_cands[best_a].raw_text or answer_cands[best_a].text
                    )

        # Build result
        final_trajectory = convert_trajectory_to_string(trajectory)
        extracted = extract_answer(final_trajectory)
        # Answer step is stored separately (not in selected_steps),
        # so selected_steps contains only reasoning steps
        reasoning_steps = len(selected_steps)
        token_stats = self.step_generator.get_sample_stats_for(sample_id)

        result = {
            "trajectory": final_trajectory,
            "extracted_answer": extracted,
            "steps": selected_steps,
            "answer_step": answer_text,
            "reasoning_steps": reasoning_steps,
            "validity_scores": validity_scores,
            "completed": bool(selected_steps)
            and selected_steps[-1].is_trajectory_complete,
            "token_stats": token_stats,
        }
        result.update(get_completion_info(selected_steps))
        return result

    def cleanup(self):
        """Clean up resources"""

        self.scorer.cleanup()
