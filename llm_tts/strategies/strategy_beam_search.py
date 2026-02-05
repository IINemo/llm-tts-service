"""
Beam Search strategy for LLM reasoning with optional batched generation.

Two modes:
- Sequential (default): Process one sample at a time, good for debugging
- Batched: Process ALL samples in parallel with synchronized beam search steps
  - One vLLM call per step (not per sample×beam)
  - Reduces calls from O(samples × steps × beams) to O(steps)
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

# Self-verification scorer (optional import)
try:
    from llm_tts.scorers.step_scorer_self_verification import StepScorerSelfVerification

    SELF_VERIFICATION_AVAILABLE = True
except ImportError:
    SELF_VERIFICATION_AVAILABLE = False
    StepScorerSelfVerification = None

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

    Supports two modes:
    - Sequential: Process samples one at a time (generate_trajectory)
    - Batched: Process ALL samples in parallel (generate_trajectories_batch)
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
    ):
        self.step_generator = step_generator
        self.scorer = scorer
        self.beam_size = beam_size
        self.candidates_per_beam = candidates_per_beam
        self.max_steps = max_steps
        self.aggregation = aggregation
        self.batch_generation = batch_generation
        self.prompt_buffer = prompt_buffer

        if beam_size <= 0:
            raise ValueError(f"beam_size must be > 0, got {beam_size}")
        if candidates_per_beam <= 0:
            raise ValueError(f"candidates_per_beam must be > 0, got {candidates_per_beam}")
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
            f"stop_tokens={len(stop_tokens)}, min_step_tokens={min_step_tokens}, "
            f"eos_token_ids={eos_token_ids}"
        )

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate a reasoning trajectory using beam search.

        Args:
            request: Input chat or prompt context.
            sample_idx: Index of current sample (for logging).

        Returns:
            Dictionary with trajectory, steps, score, and metadata.
        """

        # Reset token tracking for this sample
        self.step_generator.reset_sample_stats()
        if hasattr(self.scorer, "reset_prm_stats"):
            self.scorer.reset_prm_stats()

        # Initialize beams with empty trajectory
        beams = [{"steps": [], "scores": []}]
        completed_beams = []

        for step in range(self.max_steps):
            log.info(f"\n=== Beam Search Step {step} ===")
            new_beams = []

            # Expand each current beam
            for beam_idx, beam in enumerate(beams):
                log.info(
                    f"Expanding beam {beam_idx} with score "
                    f"{self._aggregate_scores(beam['scores']):.3f}"
                )

                candidates = self.step_generator(
                    request,
                    trajectory=beam["steps"],
                    candidates_per_step=self.candidates_per_beam,
                )

                if not candidates:
                    log.info(f"  No candidates for beam {beam_idx}, skipping")
                    continue

                # Pass trajectory context so PRM can score candidate in context of previous steps
                scores = self.scorer.score_candidates(
                    request, candidates, trajectory=beam["steps"]
                )

                # Expand with new candidates
                for cand, score in zip(candidates, scores):
                    updated_scores = beam["scores"] + [score]
                    new_beams.append(
                        {"steps": beam["steps"] + [cand], "scores": updated_scores}
                    )

                    log.info(
                        f"    Candidate: score={score:.3f}, aggregated score={self._aggregate_scores(updated_scores):.3f}, text='{cand.text[:80]}'"
                    )

            if not new_beams:
                log.info("No new beams generated, stopping early.")
                break

            # Sort and prune to top-k beams
            new_beams.sort(
                key=lambda b: self._aggregate_scores(b["scores"]),
                reverse=True,
            )
            beams = new_beams[: self.beam_size]
            log.info(f"Kept top {len(beams)} beams for next step")

            # Separate completed beams
            done, active = self._split_completed(beams)
            completed_beams.extend(done)
            beams = active

            # Stop if all beams are completed
            if not beams:
                log.info("All beams completed early.")
                break

        # Choose best final beam
        best_beam = self._select_best_beam(completed_beams or beams)
        trajectory_text = convert_trajectory_to_string(best_beam["steps"])

        # Extract answer from trajectory (e.g., from \boxed{})
        extracted = extract_answer(trajectory_text)

        # Get token stats from generator
        self.step_generator.finalize_sample_stats()
        token_stats = self.step_generator.get_sample_stats()

        # Merge PRM scorer stats if available
        if hasattr(self.scorer, "get_prm_total_stats"):
            prm_stats = self.scorer.get_prm_total_stats()
            token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
            token_stats["prm_tflops"] = prm_stats["prm_tflops"]
            gen_tflops = token_stats.get("tflops")
            if gen_tflops is None:
                log.warning("Missing 'tflops' in token_stats when merging PRM stats")
                gen_tflops = 0
            prm_tflops = prm_stats["prm_tflops"]
            if prm_tflops is None:
                log.warning("Missing 'prm_tflops' in PRM stats")
                prm_tflops = 0
            token_stats["tflops"] = gen_tflops + prm_tflops

        return {
            "trajectory": trajectory_text,
            "steps": best_beam["steps"],
            "validity_scores": best_beam["scores"],
            "completed": len(completed_beams) > 0,
            "extracted_answer": extracted,
            "token_stats": token_stats,
        }

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: Optional[List[int]] = None,
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
        log.info(f"Using PRM scorer: {use_prm_scorer}")

        # Check if scorer is a Self-Verification scorer (Tree of Thoughts style)
        use_self_verification_scorer = SELF_VERIFICATION_AVAILABLE and isinstance(
            self.scorer, StepScorerSelfVerification
        )
        log.info(f"Using Self-Verification scorer: {use_self_verification_scorer}")

        # Reset per-sample token tracking in generator
        self.step_generator.reset_per_sample_stats()
        if hasattr(self.scorer, "reset_prm_stats"):
            self.scorer.reset_prm_stats()

        # sample_beams[sample_id] = list of ACTIVE beams only
        sample_beams = {
            i: [{"steps": [], "scores": [], "total_tokens": 0}] for i in range(M)
        }

        # Store COMPLETED beams separately so they are never expanded
        completed_beams_by_sample: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(M)
        }

        # Context limit for trajectories
        # Calculated as min of:
        #   1. max_context_budget - prompt_buffer (leave room for prompt)
        #   2. max_steps * max_step_tokens (theoretical max from step limits)
        max_context_budget = getattr(self.step_generator, "max_context_budget", 4096)
        max_step_tokens = getattr(self.step_generator, "max_step_tokens", 256)
        max_trajectory_tokens = max(0, min(
            max_context_budget - self.prompt_buffer,
            self.max_steps * max_step_tokens,
        ))
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

                    # Never expand a completed beam
                    if beam["steps"] and beam["steps"][-1].is_trajectory_complete:
                        continue

                    beam_tokens = beam.get("total_tokens", 0)
                    if beam_tokens >= max_trajectory_tokens - 200:
                        skipped_beams.append((sample_id, beam_idx, beam))
                        log.info(
                            f"Sample {sample_id}: Skipping beam {beam_idx} (tokens: {beam_tokens} >= limit)"
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
                    compute_uncertainty=not use_prm_scorer,
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

                        # Additional beam search checks not in generator:
                        # Check if we can extract a valid boxed answer from FULL trajectory
                        full_traj_text = (
                            convert_trajectory_to_string(parent_beam["steps"]) + text
                        )
                        has_boxed = bool(extract_answer(full_traj_text, "boxed"))
                        if has_boxed:
                            is_trajectory_complete = True

                        # Detect garbage/degenerate output (emojis, CJK, unusual unicode)
                        if _detect_garbage(text):
                            is_trajectory_complete = True

                        data = candidate.other_data if candidate.other_data else {}
                        uncertainty = data.get("uncertainty_score")
                        if uncertainty is None:
                            log.warning(f"Sample {sample_id}, beam {beam_idx}: missing 'uncertainty_score' in candidate other_data")
                            uncertainty = 0.0
                        validity = data.get("validity_score")
                        if validity is None:
                            log.warning(f"Sample {sample_id}, beam {beam_idx}: missing 'validity_score' in candidate other_data")
                            validity = 0.0

                        candidates_for_beam.append(
                            {
                                "text": text,
                                "token_ids": token_ids,
                                "uncertainty": uncertainty,
                                "validity": validity,
                                "is_complete": is_trajectory_complete,
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

                # 6) Self-Verification scoring (optional, Tree of Thoughts style)
                if use_self_verification_scorer:
                    all_candidates_data = self._batch_score_with_self_verification(
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
                        other_data={
                            "validity_score": cand_data["validity"],
                            "uncertainty_score": cand_data["uncertainty"],
                        },
                    )

                    # Priority: PRM score > Self-verification score > Validity score
                    score = cand_data.get(
                        "prm_score",
                        cand_data.get("self_verification_score", cand_data["validity"]),
                    )

                    new_tokens = len(cand_data["token_ids"])
                    new_total_tokens = parent_beam.get("total_tokens", 0) + new_tokens

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
                    }
                    new_sample_beams[sample_id].append(new_beam)

            # 7b) Add skipped beams as completed (hit context limit)
            for sample_id, beam_idx, skipped_beam in skipped_beams:
                if skipped_beam["steps"]:
                    skipped_beam["steps"][-1].is_trajectory_complete = True
                new_sample_beams[sample_id].append(skipped_beam)

            # 8) Prune to top-k ACTIVE beams per sample and record completions
            samples_to_remove = []

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

                # Split into completed and active
                completed, active = self._split_completed(beams)

                # ✅ FIX: store completed beams separately; never keep them in active beam list
                if completed:
                    completed_beams_by_sample[sample_id].extend(completed)

                # Keep only top beam_size ACTIVE beams for next step
                active = active[: self.beam_size]
                sample_beams[sample_id] = active

                if not active:
                    # No active beams remain; finalize using best completed beam (accumulated)
                    best_beam = self._select_best_beam(
                        completed_beams_by_sample[sample_id]
                    )
                    completed_results[sample_id] = self._finalize_sample(
                        best_beam["steps"],
                        best_beam["scores"],
                        token_stats=self.step_generator.get_sample_stats_for(sample_id),
                        sample_id=sample_id,
                    )
                    samples_to_remove.append(sample_id)
                    # Log chosen trajectory details - show each step separately
                    scores_str = ", ".join(f"{s:.3f}" for s in best_beam["scores"])
                    log.info(
                        f"Sample {sample_indices[sample_id]}: Completed with "
                        f"{len(best_beam['steps'])} steps, "
                        f"score={self._aggregate_scores(best_beam['scores']):.3f}, "
                        f"scores=[{scores_str}]"
                    )
                    for step_idx, (step, score) in enumerate(
                        zip(best_beam["steps"], best_beam["scores"])
                    ):
                        log.info(
                            f"  Step {step_idx + 1} (score={score:.3f}):\n{step.text}"
                        )

            # Remove completed samples from active set
            for sample_id in samples_to_remove:
                active_samples.discard(sample_id)

            log.info(
                f"Step {step_num} complete: {len(active_samples)} samples still active, "
                f"{len(completed_results)} completed"
            )

        # Finalize any remaining active samples (prefer completed if any exist)
        for sample_id in active_samples:
            active = sample_beams[sample_id]
            candidates = completed_beams_by_sample[sample_id] or active
            best_beam = self._select_best_beam(candidates)

            completed_results[sample_id] = self._finalize_sample(
                best_beam["steps"],
                best_beam["scores"],
                token_stats=self.step_generator.get_sample_stats_for(sample_id),
                sample_id=sample_id,
            )
            # Log chosen trajectory details - show each step separately
            scores_str = ", ".join(f"{s:.3f}" for s in best_beam["scores"])
            log.info(
                f"Sample {sample_indices[sample_id]}: Reached max_steps with "
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

    def _batch_score_with_self_verification(
        self,
        requests: List[List[Dict[str, str]]],
        all_candidates_data: List[List[Dict]],
        prompt_metadata: List[tuple],
    ) -> List[List[Dict]]:
        """
        Batch score all candidates using Self-Verification scorer (Tree of Thoughts).

        Groups candidates by (sample_id, beam) and calls the scorer's
        score_candidates_detailed method for each group.

        Args:
            requests: Original requests for each sample
            all_candidates_data: List of candidate lists (one per prompt)
            prompt_metadata: List of (sample_id, beam_idx, parent_beam) tuples

        Returns:
            all_candidates_data with 'self_verification_score' added to each candidate
        """
        total_candidates = sum(len(cands) for cands in all_candidates_data)
        log.info(f"Batch Self-Verification scoring {total_candidates} candidates")

        # Process each (sample, beam) group
        for prompt_idx, candidates in enumerate(all_candidates_data):
            if not candidates:
                continue

            sample_id, beam_idx, parent_beam = prompt_metadata[prompt_idx]
            request = requests[sample_id]
            trajectory = parent_beam["steps"]

            # Create StepCandidate objects for scoring
            step_candidates = []
            for cand_data in candidates:
                step_candidate = StepCandidate(
                    text=cand_data["text"],
                    token_ids=cand_data.get("token_ids", []),
                    is_complete=True,
                    is_trajectory_complete=cand_data.get("is_complete", False),
                )
                step_candidates.append(step_candidate)

            # Call self-verification scorer
            try:
                scored_results = self.scorer.score_candidates_detailed(
                    chat=request,
                    candidates=step_candidates,
                    trajectory=trajectory,
                )

                # Extract scores and update candidate data
                for cand_idx, (cand_data, score_result) in enumerate(
                    zip(candidates, scored_results)
                ):
                    # Get the aggregate score (value method returns 'value', vote returns 'votes')
                    if hasattr(score_result, "aggregate_scores"):
                        agg_scores = score_result.aggregate_scores
                        score = agg_scores.get("value", agg_scores.get("votes", 1.0))
                    else:
                        score = 1.0

                    cand_data["self_verification_score"] = score

                    log.debug(
                        f"Sample {sample_id}, beam {beam_idx}, cand {cand_idx}: "
                        f"self_verification_score={score:.3f}"
                    )

            except Exception as e:
                log.warning(
                    f"Self-verification scoring failed for sample {sample_id}, "
                    f"beam {beam_idx}: {e}. Using default score 1.0"
                )
                for cand_data in candidates:
                    cand_data["self_verification_score"] = 1.0

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
            gen_tflops = token_stats.get("tflops")
            if gen_tflops is None:
                log.warning(f"Sample {sample_id}: missing 'tflops' in token_stats when merging PRM stats")
                gen_tflops = 0
            prm_tflops = prm_stats["prm_tflops"]
            if prm_tflops is None:
                log.warning(f"Sample {sample_id}: missing 'prm_tflops' in PRM stats")
                prm_tflops = 0
            token_stats["tflops"] = gen_tflops + prm_tflops

        # Determine actual completion status from beam steps
        is_completed = bool(
            steps and steps[-1].is_trajectory_complete
        )

        result = {
            "trajectory": trajectory_text,
            "steps": steps,
            "validity_scores": scores,
            "completed": is_completed,
            "extracted_answer": extracted,
        }
        if token_stats is not None:
            result["token_stats"] = token_stats
        return result

    def _aggregate_scores(self, scores: list[float]) -> float:
        """Aggregate scores across steps according to selected strategy."""
        if len(scores) == 0:
            return 0.0
        # Filter out non-finite values (NaN, inf, -inf)
        clean = [s for s in scores if np.isfinite(s)]
        if not clean:
            log.warning(f"All {len(scores)} scores are non-finite, returning 0.0")
            return 0.0
        if len(clean) < len(scores):
            log.warning(f"Dropped {len(scores) - len(clean)} non-finite scores out of {len(scores)}")
        if self.aggregation == "last":
            return scores[-1]
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
            if b["steps"] and b["steps"][-1].is_trajectory_complete:
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
