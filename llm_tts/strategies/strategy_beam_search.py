"""
Beam Search strategy for LLM reasoning with optional batched generation.

Two modes:
- Sequential (default): Process one sample at a time, good for debugging
- Batched: Process ALL samples in parallel with synchronized beam search steps
  - One vLLM call per step (not per sample×beam)
  - Reduces calls from O(samples × steps × beams) to O(steps)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from vllm import SamplingParams

from llm_tts.generators import (
    StepCandidateGeneratorThroughAPI,
    StepCandidateGeneratorThroughHuggingface,
    convert_trajectory_to_string,
)
from llm_tts.generators.base import StepCandidate
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


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
        aggregation: How to aggregate scores across steps ("mean", "sum", "min", "max", "product").
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
    ):
        self.step_generator = step_generator
        self.scorer = scorer
        self.beam_size = beam_size
        self.candidates_per_beam = candidates_per_beam
        self.max_steps = max_steps
        self.aggregation = aggregation
        self.batch_generation = batch_generation

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
                    scores = beam["scores"] + [score]
                    new_beams.append(
                        {"steps": beam["steps"] + [cand], "scores": scores}
                    )

                    log.info(
                        f"    Candidate: score={score:.3f}, aggregated score={self._aggregate_scores(scores):.3f}, text='{cand.text[:80]}'"
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

        return {
            "trajectory": trajectory_text,
            "steps": best_beam["steps"],
            "validity_scores": best_beam["scores"],
            "completed": len(completed_beams) > 0,
            "extracted_answer": extracted,
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

        # Initialize beams for all samples
        # sample_beams[sample_id] = list of beams, each beam = {"steps": [...], "scores": [...], "total_tokens": int}
        sample_beams = {
            i: [{"steps": [], "scores": [], "total_tokens": 0}] for i in range(M)
        }

        # Context limit for trajectories (leave room for prompt ~800 tokens + min generation)
        # Model has 4096 context, so 4096 - 800 (prompt) - 200 (buffer) = 3000
        max_trajectory_tokens = 3000
        completed_results = {}  # sample_id -> result dict
        active_samples = set(range(M))

        for step_num in range(self.max_steps):
            if not active_samples:
                log.info(f"All samples completed at step {step_num}")
                break

            log.info(
                f"\n{'='*60}\n"
                f"Beam Search Step {step_num}: {len(active_samples)} active samples\n"
                f"{'='*60}"
            )

            # 1. Build list of (sample, beam) pairs to process
            # Skip beams that are already too long (would exceed context limit)
            prompt_metadata = []  # Track (sample_id, beam_idx, beam) for each prompt
            skipped_beams = []  # Beams that are too long to continue

            for sample_id in active_samples:
                for beam_idx, beam in enumerate(sample_beams[sample_id]):
                    beam_tokens = beam.get("total_tokens", 0)
                    # Skip if beam is close to context limit (leave room for prompt ~800 tokens + generation)
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
            outputs = []

            # Only generate if there are prompts to process
            if prompt_metadata:
                # 2. Build all prompts for batched generation (like offline best-of-n)
                prompts = []
                for sample_id, beam_idx, parent_beam in prompt_metadata:
                    request = requests[sample_id]
                    trajectory = parent_beam["steps"]

                    # Build prompt using step_generator's template
                    prompt = self.step_generator._apply_chat_template(
                        request, enable_thinking=False
                    )
                    if trajectory:
                        trajectory_text = convert_trajectory_to_string(trajectory)
                        prompt = prompt + trajectory_text
                    prompts.append(prompt)

                # 3. Create sampling params with step-level stop tokens (same as step_generator)
                sampling_params = SamplingParams(
                    n=self.candidates_per_beam,
                    max_tokens=self.step_generator.max_step_tokens,
                    min_tokens=self.step_generator.min_step_tokens,
                    temperature=self.step_generator.temperature,
                    top_p=self.step_generator.top_p,
                    top_k=getattr(self.step_generator, "top_k", -1),
                    stop=self.step_generator.stop_tokens,  # 433 step boundary tokens
                    stop_token_ids=self.step_generator.stop_token_ids,  # EOS tokens [151645, 151643]
                    logprobs=20 if not use_prm_scorer else None,
                )

                log.info(
                    f"Batched generation: {len(prompts)} prompts × {self.candidates_per_beam} candidates, "
                    f"stop={len(self.step_generator.stop_tokens)} tokens, "
                    f"min={sampling_params.min_tokens}, max={sampling_params.max_tokens}"
                )

                # 4. Single vLLM call for ALL prompts
                # Use raw vLLM for PRM scorer (compute_uncertainty=False)
                if use_prm_scorer:
                    raw_llm = getattr(
                        self.step_generator.model, "llm", self.step_generator.model
                    )
                    outputs = raw_llm.generate(prompts, sampling_params)
                else:
                    # Use VLLMWithUncertainty to get uncertainty scores
                    outputs = self.step_generator.model.generate(
                        prompts, sampling_params, compute_uncertainty=True
                    )
                # Sort by request_id to maintain order
                outputs = sorted(outputs, key=lambda x: int(x.request_id))

                # 5. Process outputs into candidate data (same logic as step_generator)
                for prompt_idx, (
                    request_output,
                    (sample_id, beam_idx, parent_beam),
                ) in enumerate(zip(outputs, prompt_metadata)):
                    candidates_for_beam = []

                    for cand_idx, output in enumerate(request_output.outputs):
                        raw_text = output.text
                        token_ids = list(output.token_ids)
                        stop_reason = getattr(output, "stop_reason", None)

                        # Clean up text (same as step_generator)
                        text = raw_text.strip()

                        # Detect trajectory completion (same logic as step_generator)
                        # Stopped at EOS token ID
                        stopped_at_eos = (
                            stop_reason in self.step_generator.stop_token_ids
                        )
                        # Hit max tokens
                        hit_max_tokens = stop_reason == "length" or (
                            stop_reason is None
                            and len(token_ids) >= self.step_generator.max_step_tokens
                        )

                        # Check for repetition and truncate if needed (same as step_generator)
                        repetition_detected = False
                        if hasattr(self.step_generator, "_detect_line_repetitions"):
                            repetition_detected = (
                                self.step_generator._detect_line_repetitions(text)
                            )
                        if hit_max_tokens and hasattr(
                            self.step_generator, "_truncate_repetitions"
                        ):
                            text, was_truncated = (
                                self.step_generator._truncate_repetitions(
                                    text, len(token_ids)
                                )
                            )
                            if was_truncated:
                                repetition_detected = True

                        # Check for boxed answer (marks trajectory complete, no truncation)
                        has_boxed = "\\boxed{" in text or "\\boxed " in text

                        # Use detector's is_trajectory_complete for full pattern detection
                        # (checks for </think>, answer patterns, balanced \boxed{}, etc.)
                        detector_complete = False
                        if hasattr(self.step_generator, "detector"):
                            detector_complete = (
                                self.step_generator.detector.is_trajectory_complete(
                                    text, reached_eos=stopped_at_eos
                                )
                            )

                        is_trajectory_complete = (
                            stopped_at_eos
                            or has_boxed
                            or repetition_detected
                            or detector_complete
                        )

                        # Get uncertainty from output if available
                        uncertainty = 0.0
                        if (
                            hasattr(output, "uncertainty")
                            and output.uncertainty is not None
                        ):
                            uncertainty = output.uncertainty
                        validity = 1.0 / (1.0 + uncertainty) if uncertainty > 0 else 0.0

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

                # 4. Score all candidates (PRM or uncertainty)
                if use_prm_scorer:
                    # Use PRM scorer - need to batch score all candidates
                    all_candidates_data = self._batch_score_with_prm(
                        requests, all_candidates_data, prompt_metadata
                    )

            # 5. Update beams for each sample
            new_sample_beams = {i: [] for i in active_samples}

            for prompt_idx, candidates in enumerate(all_candidates_data):
                sample_id, beam_idx, parent_beam = prompt_metadata[prompt_idx]

                for cand_data in candidates:
                    # Create StepCandidate
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

                    # Get score (validity for uncertainty, PRM score for PRM)
                    score = cand_data.get("prm_score", cand_data["validity"])

                    # Calculate total tokens for new beam
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

                    # Create new beam with this candidate
                    new_beam = {
                        "steps": parent_beam["steps"] + [step_candidate],
                        "scores": parent_beam["scores"] + [score],
                        "total_tokens": new_total_tokens,
                    }
                    new_sample_beams[sample_id].append(new_beam)

            # 5b. Add skipped beams as completed (they hit context limit)
            for sample_id, beam_idx, skipped_beam in skipped_beams:
                # Mark the last step as complete due to context limit
                if skipped_beam["steps"]:
                    skipped_beam["steps"][-1].is_trajectory_complete = True
                new_sample_beams[sample_id].append(skipped_beam)

            # 6. Prune to top-k beams per sample and check completions
            samples_to_remove = []

            for sample_id in active_samples:
                beams = new_sample_beams[sample_id]

                if not beams:
                    # No beams generated, mark as complete with empty result
                    log.warning(
                        f"Sample {sample_indices[sample_id]}: No beams generated"
                    )
                    completed_results[sample_id] = self._finalize_sample([], [])
                    samples_to_remove.append(sample_id)
                    continue

                # Sort by aggregated score (descending)
                beams.sort(
                    key=lambda b: self._aggregate_scores(b["scores"]),
                    reverse=True,
                )

                # Split into completed and active
                completed, active = self._split_completed(beams)

                # Keep top beam_size active beams
                active = active[: self.beam_size]

                if not active:
                    # All beams completed - select best from completed
                    best_beam = self._select_best_beam(completed)
                    completed_results[sample_id] = self._finalize_sample(
                        best_beam["steps"], best_beam["scores"]
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
                else:
                    # Store completed beams and continue with active
                    sample_beams[sample_id] = (
                        active + completed[: self.beam_size - len(active)]
                    )

            # Remove completed samples from active set
            for sample_id in samples_to_remove:
                active_samples.discard(sample_id)

            log.info(
                f"Step {step_num} complete: {len(active_samples)} samples still active, "
                f"{len(completed_results)} completed"
            )

        # Finalize any remaining active samples
        for sample_id in active_samples:
            beams = sample_beams[sample_id]
            best_beam = self._select_best_beam(beams)
            completed_results[sample_id] = self._finalize_sample(
                best_beam["steps"], best_beam["scores"]
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

    def _finalize_sample(
        self, steps: List[StepCandidate], scores: List[float]
    ) -> Dict[str, Any]:
        """Create final result dict for a completed sample."""
        trajectory_text = convert_trajectory_to_string(steps)
        extracted = extract_answer(trajectory_text)

        return {
            "trajectory": trajectory_text,
            "steps": steps,
            "validity_scores": scores,
            "completed": True,
            "extracted_answer": extracted,
        }

    def _aggregate_scores(self, scores: list[float]) -> float:
        """Aggregate scores across steps according to selected strategy."""
        if len(scores) == 0:
            return 0
        if self.aggregation == "sum":
            return sum(scores)
        elif self.aggregation == "mean":
            return np.mean(scores).item()
        elif self.aggregation == "product":
            return np.prod(scores).item()
        elif self.aggregation == "max":
            return np.max(scores).item()
        elif self.aggregation == "min":
            return np.min(scores).item()
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
            return {"steps": [], "scores": 0.0}
        return max(beams, key=lambda b: self._aggregate_scores(b["scores"]))

    def cleanup(self):
        """Clean up scorer resources if necessary."""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
