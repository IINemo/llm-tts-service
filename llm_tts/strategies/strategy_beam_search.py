"""
Beam Search strategy for LLM reasoning with batched generation.

Uses batched generation (generate_trajectories_batch):
- Processes ALL samples in parallel
- One vLLM call per step for all samples x beams
- Reduces calls from O(samples x steps x beams) to O(steps)
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from llm_tts.generators import (
    StepCandidateGeneratorThroughAPI,
    StepCandidateGeneratorThroughHuggingface,
    convert_trajectory_to_string,
)
from llm_tts.generators.base import StepCandidate
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)

# Pattern to detect garbage/degenerate output
# Matches: emojis, CJK characters, unusual unicode, repeated nonsense patterns
_GARBAGE_PATTERN = re.compile(
    r"[\U0001F300-\U0001F9FF]"  # Emojis
    r"|[\u4E00-\u9FFF]"  # CJK Unified Ideographs (Chinese)
    r"|[\u3040-\u309F\u30A0-\u30FF]"  # Japanese Hiragana/Katakana
    r"|[\uFF01-\uFF60]"  # Fullwidth punctuation (！」etc)
    r"|[\u0100-\u024F]{2,}"  # Extended Latin with diacritics (mę etc) - 2+ consecutive
)


def _detect_garbage(text: str, threshold: int = 2) -> bool:
    """
    Detect garbage/degenerate output in text.

    Returns True if text contains suspicious patterns that indicate
    the model is generating garbage (emojis, CJK chars, unusual unicode).

    Args:
        text: Text to check
        threshold: Minimum number of garbage matches to trigger detection

    Returns:
        True if garbage is detected
    """
    matches = _GARBAGE_PATTERN.findall(text)
    return len(matches) >= threshold


class StrategyBeamSearch(StrategyBase):
    """
    Beam Search strategy for LLM reasoning.
    ---------------------------------------

    Keeps a beam of top-N reasoning chains at each step based on a scoring function.
    This balances exploration (multiple reasoning branches) and exploitation
    (keeping only the highest-scoring paths).

    Uses batched generation (generate_trajectories_batch):
    - Processes ALL samples in parallel
    - One vLLM call per step for all samples × beams
    - Reduces calls from O(samples × steps × beams) to O(steps)

    Args:
        step_generator: Generator for candidate steps (vLLM, API, or HuggingFace).
        scorer: Scoring function for ranking step candidates (PRM, entropy, perplexity).
        beam_size: Number of top beams to keep at each step.
        candidates_per_beam: Number of candidates to generate for each beam per step.
        max_steps: Maximum reasoning steps.
        aggregation: How to aggregate scores across steps ("last", "mean", "sum", "min", "max", "product").
        batch_generation: If True, use batched mode in generate_trajectories_batch.
        scoring_window: If set, only use the last N step scores for aggregation.
            None means use all steps (default).
    """

    def __init__(
        self,
        step_generator: (
            StepCandidateGeneratorThroughAPI | StepCandidateGeneratorThroughHuggingface
        ),
        scorer: Any,
        beam_size: int = 5,
        candidates_per_beam: int = 3,
        max_steps: int = 10,
        aggregation: str = "mean",
        batch_generation: bool = True,
        prompt_buffer: int = 500,
        scoring_window: Optional[int] = None,
    ):
        self.step_generator = step_generator
        self.scorer = scorer
        self.beam_size = beam_size
        self.candidates_per_beam = candidates_per_beam
        self.max_steps = max_steps
        self.aggregation = aggregation
        self.batch_generation = batch_generation
        self.prompt_buffer = prompt_buffer
        self.scoring_window = scoring_window

        if beam_size <= 0:
            raise ValueError(f"beam_size must be > 0, got {beam_size}")
        if candidates_per_beam <= 0:
            raise ValueError(
                f"candidates_per_beam must be > 0, got {candidates_per_beam}"
            )
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {max_steps}")

        # Get stop tokens info for logging
        stop_tokens = getattr(step_generator, "stop_tokens", [])
        eos_token_ids = getattr(step_generator, "eos_token_ids", [151645, 151643])
        min_step_tokens = getattr(step_generator, "min_step_tokens", 50)

        log.info(
            f"StrategyBeamSearch initialized: beam_size={beam_size}, "
            f"candidates_per_beam={candidates_per_beam}, max_steps={max_steps}, "
            f"aggregation={aggregation}, batch_generation={batch_generation}, "
            f"scoring_window={scoring_window}, "
            f"stop_tokens={len(stop_tokens)}, min_step_tokens={min_step_tokens}, "
            f"eos_token_ids={eos_token_ids}"
        )

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: Optional[List[int]] = None,
        save_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Generate trajectories for ALL samples in parallel using batched beam search.

        At each step, collects all (sample, beam) pairs and makes ONE vLLM call
        to generate candidates for all of them, then ONE scorer call to score them.

        This reduces vLLM calls from O(samples × steps × beams) to O(steps).

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
            f"Batched Beam Search: {M} samples, beam_size={self.beam_size}, "
            f"candidates_per_beam={self.candidates_per_beam}, max_steps={self.max_steps}"
        )

        # Check if scorer is a PRM model (separate model) or uses uncertainty from generation
        # PRM scorer (StepScorerPRM) has 'prm_model' attribute.
        # If using PRM, we'll batch score with PRM. Otherwise use uncertainty scores.
        use_prm_scorer = (
            hasattr(self.scorer, "prm_model") and self.scorer.prm_model is not None
        )
        # LLM critic scorer does its own LLM calls, doesn't need uncertainty logprobs
        use_llm_critic = hasattr(self.scorer, "score_candidates_batch")
        log.info(f"Using PRM scorer: {use_prm_scorer}, LLM critic: {use_llm_critic}")

        # Reset per-sample token tracking in generator
        self.step_generator.reset_per_sample_stats()
        if hasattr(self.scorer, "reset_prm_stats"):
            self.scorer.reset_prm_stats()
        if hasattr(self.scorer, "reset_stats"):
            self.scorer.reset_stats()

        # sample_beams[sample_id] = list of ACTIVE beams only
        # Each beam has: steps, scores, total_tokens, beam_idx, unique_id, ancestors
        sample_beams = {
            i: [
                {
                    "steps": [],
                    "scores": [],
                    "total_tokens": 0,
                    "beam_idx": 0,
                    "unique_id": 0,
                    "ancestors": [0],  # Track lineage: list of unique_ids of ancestors
                }
            ]
            for i in range(M)
        }

        # Global counter for unique beam IDs (per sample for clarity)
        self._beam_id_counter = {i: 0 for i in range(M)}

        # Store COMPLETED beams separately so they are never expanded
        completed_beams_by_sample: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(M)
        }

        # Context limit for trajectories
        # Calculated as min of:
        #   1. max_context_budget - prompt_buffer (leave room for prompt)
        #   2. max_steps * max_step_tokens (theoretical max from step limits)
        max_context_budget = getattr(self.step_generator, "context_budget", 4096)
        max_step_tokens = getattr(self.step_generator, "step_token_limit", 256)
        max_trajectory_tokens = max(
            0,
            min(
                max_context_budget - self.prompt_buffer,
                self.max_steps * max_step_tokens,
            ),
        )
        if max_trajectory_tokens == 0:
            log.warning(
                f"max_trajectory_tokens is 0 (max_context_budget={max_context_budget}, "
                f"prompt_buffer={self.prompt_buffer}) — all samples will complete immediately"
            )
        log.info(
            f"Max trajectory tokens: {max_trajectory_tokens} "
            f"(max_context_budget={max_context_budget}, prompt_buffer={self.prompt_buffer}, "
            f"max_steps={self.max_steps}, max_step_tokens={max_step_tokens})"
        )
        completed_results: Dict[int, Dict[str, Any]] = {}
        active_samples = set(range(M))

        # Check if thinking mode is enabled (for answer generation after thinking)
        thinking_mode = getattr(
            self.step_generator, "thinking_mode", False
        ) and not getattr(self.step_generator, "disable_thinking_mode", True)

        for step_num in range(self.max_steps):
            if not active_samples:
                log.info(f"All samples completed at step {step_num}")
                break

            log.info(
                f"\n{'=' * 60}\n"
                f"Beam Search Step {step_num}: {len(active_samples)} active samples\n"
                f"{'=' * 60}"
            )

            # 1) Build list of (sample, beam) pairs to process
            #    Skip:
            #    - beams already marked complete
            #    - beams too long to continue (context limit)
            prompt_metadata = []  # (sample_id, beam_idx, beam) for each prompt
            skipped_beams = (
                []
            )  # beams that are too long to continue (treated as completed)

            for sample_id in active_samples:
                for beam_idx, beam in enumerate(sample_beams[sample_id]):

                    # Never expand a completed beam (trajectory done or thinking done)
                    if beam["steps"] and (
                        beam["steps"][-1].is_trajectory_complete
                        or beam["steps"][-1].is_thinking_complete
                    ):
                        continue

                    beam_tokens = beam.get("total_tokens", 0)
                    beam_uid = beam.get("unique_id", beam_idx)
                    if beam_tokens >= max_trajectory_tokens - 200:
                        skipped_beams.append((sample_id, beam_idx, beam))
                        log.info(
                            f"Sample {sample_id}: Skipping beam id={beam_uid} "
                            f"(orig_idx={beam_idx}, tokens: {beam_tokens} >= limit)"
                        )
                    else:
                        prompt_metadata.append((sample_id, beam_idx, beam))

            if not prompt_metadata and not skipped_beams:
                log.info("No prompts to process, stopping")
                break

            log.info(
                f"Generating candidates for {len(prompt_metadata)} (sample, beam) pairs, {len(skipped_beams)} skipped due to length"
            )

            # Initialize containers for this step
            all_candidates_data = []

            # Only generate if there are prompts to process
            if prompt_metadata:
                # 2. Build per-trajectory requests and trajectories for batched generation
                batch_requests = [
                    requests[sample_id] for sample_id, _, _ in prompt_metadata
                ]
                batch_trajectories = [
                    parent_beam["steps"] for _, _, parent_beam in prompt_metadata
                ]

                log.info(
                    f"Batched generation: {len(batch_requests)} (sample, beam) pairs "
                    f"× {self.candidates_per_beam} candidates"
                )

                # 3. Single call via generate_step_candidates_batch (handles FLOP tracking)
                # Pass sample_ids and beam_ids so generator logs them
                batch_sample_ids = [sample_id for sample_id, _, _ in prompt_metadata]
                batch_beam_ids = [beam_idx for _, beam_idx, _ in prompt_metadata]
                batch_results = self.step_generator.generate_step_candidates_batch(
                    requests=batch_requests,
                    trajectories=batch_trajectories,
                    candidates_per_step=self.candidates_per_beam,
                    compute_uncertainty=not use_prm_scorer and not use_llm_critic,
                    sample_ids=batch_sample_ids,
                    beam_ids=batch_beam_ids,
                )

                # 4. Process StepCandidates into candidate data
                for prompt_idx, (
                    candidates,
                    (sample_id, beam_idx, parent_beam),
                ) in enumerate(zip(batch_results, prompt_metadata)):
                    candidates_for_beam = []

                    for candidate in candidates:
                        text = candidate.text
                        token_ids = list(candidate.token_ids)
                        is_trajectory_complete = candidate.is_trajectory_complete
                        is_thinking_complete = candidate.is_thinking_complete

                        # Additional beam search checks not in generator:
                        # Check if we can extract a valid boxed answer from FULL trajectory
                        # (skip for thinking-mode steps — boxed inside <think> is not final)
                        if not is_thinking_complete:
                            full_traj_text = (
                                convert_trajectory_to_string(parent_beam["steps"])
                                + text
                            )
                            has_boxed = bool(extract_answer(full_traj_text, "boxed"))
                            if has_boxed:
                                is_trajectory_complete = True

                            # Detect garbage/degenerate output (emojis, CJK, unusual unicode)
                            if _detect_garbage(text):
                                is_trajectory_complete = True

                        data = candidate.other_data if candidate.other_data else {}
                        uncertainty = data.get("uncertainty_score", 0.0)
                        validity = data.get("validity_score", 0.0)

                        candidates_for_beam.append(
                            {
                                "text": text,
                                "token_ids": token_ids,
                                "uncertainty": uncertainty,
                                "validity": validity,
                                "is_complete": is_trajectory_complete,
                                "is_thinking_complete": is_thinking_complete,
                                "sample_id": sample_id,
                                "beam_idx": beam_idx,
                                "parent_beam": parent_beam,
                            }
                        )

                    all_candidates_data.append(candidates_for_beam)

                # 5) Scoring (optional)
                if use_prm_scorer:
                    # Use PRM scorer - need to batch score all candidates
                    all_candidates_data = self._batch_score_with_prm(
                        requests, all_candidates_data, prompt_metadata
                    )
                elif hasattr(self.scorer, "score_candidates_batch"):
                    # Use scorer-provided batch scoring (e.g., LLM critic)
                    all_candidates_data = self._batch_score_with_scorer(
                        requests, all_candidates_data, prompt_metadata
                    )

            # 7) Update beams for each sample
            new_sample_beams = {i: [] for i in active_samples}

            for prompt_idx, candidates in enumerate(all_candidates_data):
                sample_id, beam_idx, parent_beam = prompt_metadata[prompt_idx]

                for cand_data in candidates:
                    step_candidate = StepCandidate(
                        text=cand_data["text"],
                        token_ids=cand_data["token_ids"],
                        is_complete=True,
                        is_trajectory_complete=cand_data["is_complete"],
                        is_thinking_complete=cand_data.get(
                            "is_thinking_complete", False
                        ),
                        other_data={
                            "validity_score": cand_data["validity"],
                            "uncertainty_score": cand_data["uncertainty"],
                        },
                    )

                    score = cand_data.get("prm_score", cand_data["validity"])

                    new_tokens = len(cand_data["token_ids"])
                    new_total_tokens = parent_beam.get("total_tokens", 0) + new_tokens

                    # Assign unique ID and track lineage
                    self._beam_id_counter[sample_id] += 1
                    new_unique_id = self._beam_id_counter[sample_id]
                    parent_ancestors = parent_beam.get("ancestors", [])

                    # Mark complete if exceeding context limit
                    if (
                        new_total_tokens >= max_trajectory_tokens
                        and not step_candidate.is_trajectory_complete
                    ):
                        step_candidate.is_trajectory_complete = True
                        log.info(
                            f"Sample {sample_id}: Trajectory marked complete "
                            f"(tokens: {new_total_tokens} >= {max_trajectory_tokens})"
                        )

                    new_beam = {
                        "steps": parent_beam["steps"] + [step_candidate],
                        "scores": parent_beam["scores"] + [score],
                        "total_tokens": new_total_tokens,
                        "beam_idx": beam_idx,  # Track original beam index
                        "unique_id": new_unique_id,
                        "ancestors": parent_ancestors + [new_unique_id],
                    }
                    new_sample_beams[sample_id].append(new_beam)

            # 7b) Add skipped beams as completed (hit context limit)
            for sample_id, beam_idx, skipped_beam in skipped_beams:
                if skipped_beam["steps"]:
                    skipped_beam["steps"][-1].is_trajectory_complete = True
                new_sample_beams[sample_id].append(skipped_beam)

            # 8) Prune to top-k ACTIVE beams per sample and record completions
            samples_to_remove = []
            samples_pending_finalization = []  # (sample_id, best_beam)

            for sample_id in active_samples:
                beams = new_sample_beams[sample_id]

                if not beams:
                    # No beams generated, mark as complete with empty result
                    log.warning(
                        f"Sample {sample_indices[sample_id]}: No beams generated"
                    )
                    completed_results[sample_id] = self._finalize_sample(
                        [],
                        [],
                        token_stats=self.step_generator.get_sample_stats_for(sample_id),
                        sample_id=sample_id,
                    )
                    samples_to_remove.append(sample_id)
                    continue

                # Sort by aggregated score (descending)
                beams.sort(
                    key=lambda b: self._aggregate_scores(b["scores"]),
                    reverse=True,
                )

                # Enhanced logging with explicit aggregation details
                log.info(
                    f"Sample {sample_indices[sample_id]}: Step {step_num} - Beam Selection"
                    f" (aggregation={self.aggregation}, "
                    f"{'lower aggregated score = better' if self.aggregation in ['min', 'product'] else 'higher aggregated score = better'})"
                )

                # Find the min/max step index for each beam to mark it
                all_scores_parts = []
                for i, b in enumerate(beams):
                    scores = b["scores"]
                    agg_score = self._aggregate_scores(scores)

                    # Find which step is the aggregated value
                    agg_idx = 0
                    if self.aggregation == "min":
                        agg_idx = scores.index(min(scores)) if scores else 0
                    elif self.aggregation == "max":
                        agg_idx = scores.index(max(scores)) if scores else 0

                    # Mark the aggregated step in the list
                    score_list_with_mark = [
                        f"{s:.3f}↓" if idx == agg_idx else f"{s:.3f}"
                        for idx, s in enumerate(scores)
                    ]
                    marked_steps_str = ", ".join(score_list_with_mark)

                    # Get lineage info
                    beam_uid = b.get("unique_id", i)
                    ancestors = b.get("ancestors", [])
                    orig_idx = b.get("beam_idx", i)

                    # Format ancestors: show full chain or abbreviated if long
                    if len(ancestors) <= 4:
                        ancestors_str = str(ancestors)
                    else:
                        ancestors_str = f"[{ancestors[0]}, {ancestors[1]}, ..., {ancestors[-2]}, {ancestors[-1]}]"

                    beam_status = (
                        "[BEST]"
                        if i == 0
                        else "[KEPT]" if i < self.beam_size else "[PRUNED]"
                    )
                    all_scores_parts.append(
                        f"  {beam_status} id={beam_uid} (sorted={i}, orig={orig_idx}, lineage={ancestors_str}): "
                        f"agg={agg_score:.3f}, steps=[{marked_steps_str}]"
                    )

                all_scores_str = "\n".join(all_scores_parts)
                total_candidates = len(beams)
                log.info(
                    f"Total candidates: {total_candidates} (from {len(sample_beams.get(sample_id, []))} active beams × {self.candidates_per_beam} candidates)"
                )
                log.info(f"Sorted beams:\n{all_scores_str}")

                # Split into completed and active
                completed, active = self._split_completed(beams)

                # ✅ FIX: store completed beams separately; never keep them in active beam list
                if completed:
                    completed_beams_by_sample[sample_id].extend(completed)

                # Keep only top beam_size ACTIVE beams for next step
                active = active[: self.beam_size]
                sample_beams[sample_id] = active

                # Log beam search state after selection (before next generation)
                if active:
                    active_beams_parts = []
                    for i, b in enumerate(active):
                        score = self._aggregate_scores(b["scores"])
                        n_steps = len(b["steps"])
                        tokens = b.get("total_tokens", 0)
                        beam_uid = b.get("unique_id", i)
                        ancestors = b.get("ancestors", [])
                        orig_idx = b.get("beam_idx", i)

                        # Format ancestors
                        if len(ancestors) <= 4:
                            ancestors_str = str(ancestors)
                        else:
                            ancestors_str = f"[{ancestors[0]}, {ancestors[1]}, ..., {ancestors[-2]}, {ancestors[-1]}]"

                        active_beams_parts.append(
                            f"  id={beam_uid} (sorted={i}, orig={orig_idx}, lineage={ancestors_str}): "
                            f"{n_steps} steps, {tokens} tokens, agg={score:.3f}"
                        )
                    active_beams_str = "\n".join(active_beams_parts)
                    log.info(
                        f"Active beams for next step ({len(active)}/{total_candidates} kept):\n{active_beams_str}"
                    )
                else:
                    log.info(
                        "Active beams for next step: (none - all beams completed or pruned)"
                    )

                log.info("")  # Empty line for readability

                if not active:
                    # No active beams remain; select best beam for finalization
                    best_beam = self._select_best_beam(
                        completed_beams_by_sample[sample_id]
                    )
                    samples_pending_finalization.append((sample_id, best_beam))
                    samples_to_remove.append(sample_id)

            # Generate answer phases for thinking-complete beams (batched)
            if thinking_mode and samples_pending_finalization:
                self._generate_beam_answers(samples_pending_finalization, requests)

            # Finalize completed samples
            for sample_id, best_beam in samples_pending_finalization:
                completed_results[sample_id] = self._finalize_sample(
                    best_beam["steps"],
                    best_beam["scores"],
                    token_stats=self.step_generator.get_sample_stats_for(sample_id),
                    sample_id=sample_id,
                )
                # Log chosen trajectory details - show each step separately
                scores_str = ", ".join(f"{s:.3f}" for s in best_beam["scores"])
                beam_uid = best_beam.get("unique_id", "N/A")
                ancestors = best_beam.get("ancestors", [])
                lineage_str = (
                    str(ancestors)
                    if len(ancestors) <= 4
                    else f"[{ancestors[0]}, {ancestors[1]}, ..., {ancestors[-1]}]"
                )
                log.info(
                    f"Sample {sample_indices[sample_id]}: Completed with "
                    f"beam_id={beam_uid}, lineage={lineage_str}, "
                    f"{len(best_beam['steps'])} steps, "
                    f"score={self._aggregate_scores(best_beam['scores']):.3f}, "
                    f"scores=[{scores_str}]"
                )
                for step_idx, (step, score) in enumerate(
                    zip(best_beam["steps"], best_beam["scores"])
                ):
                    log.info(f"  Step {step_idx + 1} (score={score:.3f}):\n{step.text}")

            # Remove completed samples from active set
            for sample_id in samples_to_remove:
                active_samples.discard(sample_id)

            log.info(
                f"Step {step_num} complete: {len(active_samples)} samples still active, "
                f"{len(completed_results)} completed"
            )

        # Finalize any remaining active samples (prefer completed if any exist)
        remaining_pending = []
        for sample_id in active_samples:
            active = sample_beams[sample_id]
            candidates = completed_beams_by_sample[sample_id] or active
            best_beam = self._select_best_beam(candidates)
            remaining_pending.append((sample_id, best_beam))

        # Generate answer phases for remaining thinking-complete beams
        if thinking_mode and remaining_pending:
            self._generate_beam_answers(remaining_pending, requests)

        for sample_id, best_beam in remaining_pending:
            completed_results[sample_id] = self._finalize_sample(
                best_beam["steps"],
                best_beam["scores"],
                token_stats=self.step_generator.get_sample_stats_for(sample_id),
                sample_id=sample_id,
            )
            # Log chosen trajectory details - show each step separately
            scores_str = ", ".join(f"{s:.3f}" for s in best_beam["scores"])
            beam_uid = best_beam.get("unique_id", "N/A")
            ancestors = best_beam.get("ancestors", [])
            lineage_str = (
                str(ancestors)
                if len(ancestors) <= 4
                else f"[{ancestors[0]}, {ancestors[1]}, ..., {ancestors[-1]}]"
            )
            log.warning(
                f"Sample {sample_indices[sample_id]}: Reached max_steps with "
                f"beam_id={beam_uid}, lineage={lineage_str}, "
                f"{len(best_beam['steps'])} steps, "
                f"score={self._aggregate_scores(best_beam['scores']):.3f}, "
                f"scores=[{scores_str}]"
            )
            for step_idx, (step, score) in enumerate(
                zip(best_beam["steps"], best_beam["scores"])
            ):
                log.info(f"  Step {step_idx + 1} (score={score:.3f}):\n{step.text}")

        # Return results in original order
        results = [completed_results[i] for i in range(M)]
        return results

    def _build_prompt_with_history(
        self,
        request: List[Dict[str, str]],
        trajectory_text: str,
        max_prompt_tokens: int = 3500,
    ) -> str:
        """Build a prompt string with chat template and trajectory history.

        Args:
            request: Chat messages
            trajectory_text: Previous steps as text
            max_prompt_tokens: Maximum tokens for prompt (leave room for generation)
        """
        # Apply chat template to get base prompt
        if getattr(self.step_generator, "disable_thinking_mode", False):
            prompt = self.step_generator._apply_chat_template(
                request, enable_thinking=False
            )
        else:
            prompt = self.step_generator.tokenizer.apply_chat_template(
                request, tokenize=False, add_generation_prompt=True
            )

        # Append trajectory history if any
        if trajectory_text:
            full_prompt = prompt + trajectory_text

            # Check if prompt is too long and truncate trajectory if needed
            tokenizer = self.step_generator.tokenizer
            prompt_tokens = len(tokenizer.encode(full_prompt))

            if prompt_tokens > max_prompt_tokens:
                # Truncate trajectory from the beginning (keep recent steps)
                base_tokens = len(tokenizer.encode(prompt))
                available_for_traj = max_prompt_tokens - base_tokens - 100  # Buffer

                if available_for_traj > 0:
                    # Truncate trajectory text to fit
                    traj_tokens = tokenizer.encode(trajectory_text)
                    if len(traj_tokens) > available_for_traj:
                        # Keep last N tokens of trajectory
                        truncated_tokens = traj_tokens[-available_for_traj:]
                        trajectory_text = tokenizer.decode(truncated_tokens)
                        log.warning(
                            f"Truncated trajectory from {len(traj_tokens)} to "
                            f"{len(truncated_tokens)} tokens to fit context"
                        )

                full_prompt = prompt + trajectory_text

            return full_prompt

        return prompt

    def _batch_score_with_prm(
        self,
        requests: List[List[Dict[str, str]]],
        all_candidates_data: List[List[Dict]],
        prompt_metadata: List[tuple],
    ) -> List[List[Dict]]:
        """
        Batch score all candidates using PRM scorer.

        Args:
            requests: Original requests for each sample
            all_candidates_data: List of candidate lists (one per prompt)
            prompt_metadata: List of (sample_id, beam_idx, parent_beam) tuples

        Returns:
            all_candidates_data with 'prm_score' added to each candidate
        """
        # Build full trajectories (parent steps + new candidate) for batch scoring
        full_trajectories = []
        flat_chats = []
        candidate_map = []  # (prompt_idx, cand_idx)
        sample_ids_for_scoring = []  # Track sample IDs for logging
        traj_ids_for_scoring = []  # Track trajectory IDs within each sample

        for prompt_idx, candidates in enumerate(all_candidates_data):
            sample_id, beam_idx, parent_beam = prompt_metadata[prompt_idx]
            request = requests[sample_id]

            for cand_idx, cand_data in enumerate(candidates):
                # Create StepCandidate for the new candidate
                new_step = StepCandidate(
                    text=cand_data["text"],
                    token_ids=cand_data.get("token_ids", []),
                    is_complete=True,
                    is_trajectory_complete=cand_data.get("is_complete", False),
                )
                # Build full trajectory: parent steps + new step
                full_traj = parent_beam["steps"] + [new_step]
                full_trajectories.append(full_traj)
                flat_chats.append(request)
                candidate_map.append((prompt_idx, cand_idx))
                sample_ids_for_scoring.append(sample_id)
                traj_ids_for_scoring.append(len(candidate_map) - 1)

        if not full_trajectories:
            return all_candidates_data

        log.info(f"Batch PRM scoring {len(full_trajectories)} candidates")

        # Score all trajectories in batch using score_trajectories_batch
        if hasattr(self.scorer, "score_trajectories_batch"):
            # Returns list of score lists (one per trajectory)
            all_scores = self.scorer.score_trajectories_batch(
                flat_chats,
                full_trajectories,
                sample_ids=sample_ids_for_scoring,
                trajectory_ids=traj_ids_for_scoring,
            )
            # Extract the last step score (the new candidate's score)
            scores = []
            for traj_scores in all_scores:
                if traj_scores:
                    scores.append(traj_scores[-1])  # Last step is the new candidate
                else:
                    scores.append(0.0)
        else:
            # Fallback: score one by one (less efficient)
            scores = []
            for chat, traj in zip(flat_chats, full_trajectories):
                score_list = self.scorer.score_trajectory(chat, traj)
                if score_list:
                    scores.append(score_list[-1])  # Last step score
                else:
                    scores.append(0.0)

        # Map scores back to candidates
        for (prompt_idx, cand_idx), score in zip(candidate_map, scores):
            all_candidates_data[prompt_idx][cand_idx]["prm_score"] = score

        return all_candidates_data

    def _generate_beam_answers(
        self,
        beams_needing_answers: List[tuple],  # [(sample_id, beam_dict)]
        requests: List[List[Dict[str, str]]],
    ) -> None:
        """Generate answer phases for beams where thinking is complete but trajectory is not.

        After beam search selects the best thinking trajectory, this method generates
        the answer phase (text after </think> with \\boxed{}) in one batched call.
        Modifies beams in-place by appending answer StepCandidates.

        Args:
            beams_needing_answers: List of (sample_id, beam_dict) pairs
            requests: Original requests for each sample
        """
        # Filter beams where thinking is done but trajectory is not
        to_generate = []
        for sample_id, beam in beams_needing_answers:
            if (
                beam["steps"]
                and beam["steps"][-1].is_thinking_complete
                and not beam["steps"][-1].is_trajectory_complete
            ):
                to_generate.append((sample_id, beam))

        if not to_generate:
            return

        log.info(
            f"Generating {len(to_generate)} answer phases for thinking-complete beams"
        )

        batch_requests = [requests[sample_id] for sample_id, _ in to_generate]
        batch_trajectories = [beam["steps"] for _, beam in to_generate]

        answer_results = self.step_generator.generate_answer_candidates_batch(
            batch_requests,
            batch_trajectories,
            candidates_per_step=1,
        )

        for batch_idx, (sample_id, beam) in enumerate(to_generate):
            candidates = (
                answer_results[batch_idx] if batch_idx < len(answer_results) else []
            )
            if candidates:
                answer_step = candidates[0]
                answer_step.is_trajectory_complete = True
                beam["steps"].append(answer_step)
                # Answer steps are NOT scored (following offline BoN pattern)
                answer_tokens = (
                    len(answer_step.token_ids) if answer_step.token_ids else 0
                )
                log.info(
                    f"Sample {sample_id}: Answer phase appended ({answer_tokens} tokens)"
                )
            else:
                log.warning(f"Sample {sample_id}: No answer candidates generated")

    def _batch_score_with_scorer(
        self,
        requests: List[List[Dict[str, str]]],
        all_candidates_data: List[List[Dict]],
        prompt_metadata: List[tuple],
    ) -> List[List[Dict]]:
        """
        Batch score all candidates using a scorer that supports score_candidates_batch.

        Works with any scorer that implements score_candidates_batch (e.g.,
        StepScorerLLMCritic, StepScorerConfidence).

        Args:
            requests: Original requests for each sample
            all_candidates_data: List of candidate lists (one per prompt)
            prompt_metadata: List of (sample_id, beam_idx, parent_beam) tuples

        Returns:
            all_candidates_data with 'validity' updated from scorer
        """
        chats = []
        candidates_list = []
        trajectories = []
        scorer_sample_ids = []

        for prompt_idx, candidates in enumerate(all_candidates_data):
            sample_id, _, parent_beam = prompt_metadata[prompt_idx]
            chats.append(requests[sample_id])
            candidates_list.append([c["text"] for c in candidates])
            trajectories.append(parent_beam["steps"])
            scorer_sample_ids.append(sample_id)

        if not candidates_list:
            return all_candidates_data

        log.info(f"Batch scorer evaluation for {len(candidates_list)} prompt groups")

        all_scores = self.scorer.score_candidates_batch(
            chats,
            candidates_list,
            trajectories=trajectories,
            sample_ids=scorer_sample_ids,
        )

        for prompt_idx, scores in enumerate(all_scores):
            for cand_idx, score in enumerate(scores):
                all_candidates_data[prompt_idx][cand_idx]["validity"] = score

        return all_candidates_data

    def _finalize_sample(
        self,
        steps: List[StepCandidate],
        scores: List[float],
        token_stats: Optional[Dict[str, Any]] = None,
        sample_id: Any = None,
    ) -> Dict[str, Any]:
        """Create final result dict for a completed sample."""
        trajectory_text = convert_trajectory_to_string(steps)
        extracted = extract_answer(trajectory_text)

        # Merge PRM scorer stats if available
        if (
            token_stats is not None
            and hasattr(self.scorer, "get_prm_stats_for")
            and sample_id is not None
        ):
            prm_stats = self.scorer.get_prm_stats_for(sample_id)
            token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
            token_stats["prm_tflops"] = prm_stats["prm_tflops"]
            gen_tflops = token_stats.get("tflops") or 0
            prm_tflops = prm_stats["prm_tflops"] or 0
            token_stats["tflops"] = gen_tflops + prm_tflops

        # Merge LLM critic scorer stats if available
        if (
            token_stats is not None
            and hasattr(self.scorer, "get_stats_for")
            and sample_id is not None
        ):
            sv_stats = self.scorer.get_stats_for(sample_id)
            token_stats.update(sv_stats)
            gen_tflops = token_stats.get("tflops") or 0
            sv_tflops = sv_stats.get("llm_critic_tflops") or 0
            token_stats["tflops"] = gen_tflops + sv_tflops

        is_completed = bool(steps and steps[-1].is_trajectory_complete)
        is_completed = bool(steps and steps[-1].is_trajectory_complete)

        result = {
            "trajectory": trajectory_text,
            "steps": steps,
            "validity_scores": scores,
            "completed": is_completed,
            "extracted_answer": extracted,
        }

        # If answer step was appended (more steps than scores), record it separately
        # so the eval script logs it as "Generated Answer" instead of a numbered step
        if len(steps) > len(scores) and steps:
            answer_step = steps[-1]
            result["answer_step"] = answer_step.raw_text or answer_step.text

        if token_stats is not None:
            result["token_stats"] = token_stats
        return result

    def _aggregate_scores(self, scores: list[float]) -> float:
        """Aggregate scores across steps according to selected strategy."""
        if len(scores) == 0:
            return 0.0
        # Filter out None (skipped PRM steps) and non-finite values (NaN, inf, -inf)
        clean = [s for s in scores if s is not None and np.isfinite(s)]
        if not clean:
            log.warning(f"All {len(scores)} scores are non-finite, returning 0.0")
            return 0.0
        if len(clean) < len(scores):
            log.warning(
                f"Dropped {len(scores) - len(clean)} non-finite scores out of {len(scores)}"
            )
        # Apply sliding window: only use the last N valid scores
        if self.scoring_window is not None and len(clean) > self.scoring_window:
            clean = clean[-self.scoring_window :]
        if self.aggregation == "last":
            return clean[-1]
        elif self.aggregation == "sum":
            return sum(clean)
        elif self.aggregation == "mean":
            return np.mean(clean).item()
        elif self.aggregation == "product":
            return np.prod(clean).item()
        elif self.aggregation == "max":
            return np.max(clean).item()
        elif self.aggregation == "min":
            return np.min(clean).item()
        else:
            raise Exception(f"Unknown aggregation {self.aggregation}")

    def _split_completed(self, beams: List[Dict]) -> tuple:
        """Split beams into completed and active."""
        completed = []
        active = []
        for b in beams:
            if b["steps"] and (
                b["steps"][-1].is_trajectory_complete
                or b["steps"][-1].is_thinking_complete
            ):
                completed.append(b)
            else:
                active.append(b)
        return completed, active

    def _select_best_beam(self, beams: List[Dict]) -> Dict:
        """Select the highest scoring beam."""
        if not beams:
            return {"steps": [], "scores": []}
        return max(beams, key=lambda b: self._aggregate_scores(b["scores"]))

    def cleanup(self):
        """Clean up scorer resources if necessary."""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
