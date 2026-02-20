"""
Offline Best-of-N strategy - Single-call trajectory generation with post-hoc step splitting.

Generates N complete trajectories in ONE vLLM call, splits them into steps post-hoc
using the same stop tokens, then scores each trajectory with PRM.

Key features:
- Single vLLM call: All N trajectories generated in parallel (n=num_trajectories)
- Post-hoc step detection: Splits trajectories using same stop tokens as step-by-step
- Efficient PRM scoring: Each trajectory scored once after generation
- Maximum throughput: No stopping at step boundaries during generation
- Batch generation: M samples × N trajectories in ONE vLLM call (generate_trajectories_batch)

Key difference from online best-of-n:
- Online: greedy step selection at each iteration (selects best step, continues)
- Offline: generates all N trajectories independently, then picks best complete solution
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.scorers.multi_scorer import (
    compute_logprob_scores,
    compute_logprob_scores_per_step,
)
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyOfflineBestOfN(StrategyBase):
    """
    Offline Best-of-N strategy with single-call trajectory generation.

    Generates N complete trajectories in ONE vLLM call (n=num_trajectories),
    splits them into steps post-hoc using the same stop tokens,
    then scores each trajectory with PRM and selects the best one.
    """

    def __init__(
        self,
        scorer,
        num_trajectories: int,
        max_steps: int,
        step_generator: StepCandidateGeneratorBase,
        score_aggregation: str = "mean",
        output_dir: Optional[str] = None,
        batch_generation: bool = True,
        calculate_entropy_score: bool = False,
        calculate_perplexity_score: bool = False,
        calculate_sequence_prob_score: bool = False,
        calculate_pd_gap_score: bool = False,
        calculate_prm_score: bool = False,
        prm_scorer=None,
        scoring_window: Optional[int] = None,
    ):
        """
        Initialize offline best-of-n strategy.

        Args:
            scorer: Scorer for evaluating steps (PRM, entropy, etc.)
            num_trajectories: Number of complete trajectories to generate
            max_steps: Maximum steps per trajectory
            step_generator: Generator for step candidates
            score_aggregation: How to aggregate step scores ('mean', 'min', 'max', 'product', 'last')
            output_dir: Directory for saving logs
            batch_generation: If True, use fully batched M×N generation in single vLLM call
            calculate_entropy_score: Compute entropy scores for all candidates
            calculate_perplexity_score: Compute perplexity scores for all candidates
            calculate_sequence_prob_score: Compute sequence probability scores for all candidates
            calculate_pd_gap_score: Compute PD-gap scores for all candidates
            calculate_prm_score: Compute PRM scores for all candidates (requires prm_scorer)
            prm_scorer: StepScorerPRM instance for PRM scoring (only if calculate_prm_score=True)
            scoring_window: If set, only use the last N steps for score aggregation.
                None means use all steps (default).
        """
        self.scorer = scorer
        self.num_trajectories = num_trajectories
        self.max_steps = max_steps
        self.step_generator = step_generator
        self.score_aggregation = score_aggregation
        self.output_dir = output_dir
        self.batch_generation = batch_generation
        self.scoring_window = scoring_window
        self._current_sample_idx = 0

        # Per-scorer flags for multi-scoring
        self.calculate_entropy_score = calculate_entropy_score
        self.calculate_perplexity_score = calculate_perplexity_score
        self.calculate_sequence_prob_score = calculate_sequence_prob_score
        self.calculate_pd_gap_score = calculate_pd_gap_score
        self.calculate_prm_score = calculate_prm_score
        self.prm_scorer = prm_scorer

        # Build list of enabled extra logprob metrics
        self._extra_metrics = []
        if calculate_entropy_score:
            self._extra_metrics.append("entropy")
        if calculate_perplexity_score:
            self._extra_metrics.append("perplexity")
        if calculate_sequence_prob_score:
            self._extra_metrics.append("sequence_prob")
        if calculate_pd_gap_score:
            self._extra_metrics.append("pd_gap")

        self._multi_scoring_enabled = bool(self._extra_metrics) or calculate_prm_score

        window_str = f", scoring_window={scoring_window}" if scoring_window else ""
        log.info(
            f"StrategyOfflineBestOfN initialized: "
            f"{num_trajectories} trajectories, max_steps={max_steps}, "
            f"aggregation={score_aggregation}, batch_generation={batch_generation}"
            f"{window_str}"
        )
        if self._multi_scoring_enabled:
            extra_info = []
            if self._extra_metrics:
                extra_info.append(f"logprob_metrics={self._extra_metrics}")
            if calculate_prm_score:
                extra_info.append("prm_scoring=True")
            log.info(f"Multi-scoring enabled: {', '.join(extra_info)}")

    def _get_uncertainty_score(self, candidate: StepCandidate) -> float:
        """Get uncertainty score from candidate's other_data."""
        if candidate.other_data and "uncertainty_score" in candidate.other_data:
            score = candidate.other_data["uncertainty_score"]
            if score is None:
                log.warning(
                    "Candidate has uncertainty_score=None — no estimator configured?"
                )
                return 0.0
            return score
        return 0.0

    def _compute_per_step_uncertainty(
        self,
        steps: List[str],
        token_ids: List[int],
        logprobs: List,
        uncertainty_wrapper,
    ) -> List[float]:
        """
        Compute uncertainty score for each step independently.

        Maps step text boundaries to token boundaries, then scores each step's
        tokens using VLLMWithUncertainty.score() or APIUncertaintyScorer.score().
        Works with any uncertainty estimator (Perplexity, MeanTokenEntropy, etc.).

        Args:
            steps: List of step text strings
            token_ids: Full trajectory token IDs
            logprobs: Full trajectory logprobs from vLLM or API
            uncertainty_wrapper: VLLMWithUncertainty or APIUncertaintyScorer instance

        Returns:
            List of validity scores (1/(1+uncertainty)) for each step
        """
        if not steps or not token_ids or not logprobs:
            return [0.0] * len(steps) if steps else []

        tokenizer = uncertainty_wrapper.get_tokenizer()

        # For API pseudo-tokenizer, set the trajectory context for positional lookup
        if hasattr(tokenizer, "set_context"):
            tokenizer.set_context(token_ids, logprobs)

        # Decode tokens incrementally to find step boundaries
        # We need to match step text to token positions
        step_scores = []
        current_token_idx = 0

        for step_idx, step_text in enumerate(steps):
            # Find how many tokens this step uses
            # Strategy: decode tokens incrementally until we cover this step's text
            step_start_idx = current_token_idx
            accumulated_text = ""

            # Find end token index for this step
            while current_token_idx < len(token_ids):
                # Decode from start to current position
                accumulated_text = tokenizer.decode(
                    token_ids[step_start_idx : current_token_idx + 1],
                    skip_special_tokens=False,
                )
                current_token_idx += 1

                # Check if we've covered this step's text
                # Use startswith to handle potential whitespace differences
                if len(accumulated_text.strip()) >= len(step_text.strip()):
                    break

            # Get token_ids and logprobs for this step
            step_token_ids = token_ids[step_start_idx:current_token_idx]
            step_logprobs = logprobs[step_start_idx:current_token_idx]

            # Score this step's tokens
            if step_token_ids and step_logprobs:
                try:
                    uncertainty = uncertainty_wrapper.score(
                        step_token_ids, step_logprobs
                    )
                    # Convert to validity score: higher = better (lower uncertainty)
                    validity_score = 1.0 / (1.0 + uncertainty)
                except Exception as e:
                    log.warning(f"Failed to score step {step_idx}: {e}")
                    validity_score = 0.5  # Neutral score on error
            else:
                validity_score = 0.5  # Neutral score for empty steps

            step_scores.append(validity_score)

        return step_scores

    def _aggregate_scores(self, step_scores: List[float]) -> float:
        """
        Aggregate step scores into a single trajectory score.

        Args:
            step_scores: List of scores for each step (may contain None for skipped steps)

        Returns:
            Aggregated score (higher = better)
        """
        if not step_scores:
            return 0.0

        # Filter out None values (skipped steps from PRM tail truncation)
        valid_scores = [s for s in step_scores if s is not None]
        num_none = len(step_scores) - len(valid_scores)
        if num_none > 0:
            log.debug(
                f"Aggregation: filtered {num_none}/{len(step_scores)} None values "
                f"(from tail truncation), {len(valid_scores)} valid scores remain"
            )
        if not valid_scores:
            return 0.0

        # Apply sliding window: only use the last N valid scores
        if self.scoring_window is not None and len(valid_scores) > self.scoring_window:
            valid_scores = valid_scores[-self.scoring_window :]

        if self.score_aggregation == "mean":
            return float(np.mean(valid_scores))
        elif self.score_aggregation == "min":
            # Conservative: trajectory is only as good as its weakest step
            return float(np.min(valid_scores))
        elif self.score_aggregation == "max":
            # Optimistic: best step determines trajectory score
            return float(np.max(valid_scores))
        elif self.score_aggregation == "product":
            return float(np.prod(valid_scores))
        elif self.score_aggregation == "last":
            return valid_scores[-1]
        else:
            log.warning(f"Unknown aggregation '{self.score_aggregation}', using mean")
            return float(np.mean(valid_scores))

    def _split_thinking_candidate(
        self,
        candidate: StepCandidate,
    ) -> Dict[str, Any]:
        """
        Split a candidate into steps with thinking-mode awareness (no generation).

        For thinking-mode candidates (containing </think>):
        - Splits thinking text into steps via detector
        - Marks as needs_answer=True so answer can be batched later

        For non-thinking candidates: split full text via detector as before.

        Args:
            candidate: StepCandidate from generation

        Returns:
            Dict with steps, full_text, num_steps, is_complete, num_tokens,
            reasoning_steps, needs_answer, candidate
        """
        raw_text = candidate.raw_text or candidate.text
        num_tokens = (
            candidate.other_data.get("original_token_count", len(candidate.token_ids))
            if candidate.other_data
            else len(candidate.token_ids)
        )

        # Thinking mode: stopped at </think>, needs answer generation
        if (
            getattr(self.step_generator, "thinking_mode", False)
            and candidate.is_thinking_complete
            and not candidate.is_trajectory_complete
        ):
            # Use candidate.text for splitting — it has </think> appended back
            # (candidate.raw_text is vLLM output which strips stop strings)
            thinking_text = candidate.text
            if hasattr(self.step_generator, "detector"):
                thinking_steps = self.step_generator.detector.detect_steps(
                    thinking_text, use_stop_tokens=True
                )
            else:
                thinking_steps = [thinking_text]

            return {
                "steps": thinking_steps,  # Only thinking steps (for scoring)
                "full_text": raw_text,  # Will be updated after answer generation
                "num_steps": len(thinking_steps),
                "is_complete": False,  # Not yet — answer still needed
                "num_tokens": num_tokens,
                "reasoning_steps": len(thinking_steps),
                "needs_answer": True,
                "candidate": candidate,
            }

        # Non-thinking mode: split full text via detector
        if hasattr(self.step_generator, "detector"):
            steps = self.step_generator.detector.detect_steps(
                raw_text, use_stop_tokens=True
            )
        else:
            steps = [raw_text]

        return {
            "steps": steps,
            "full_text": raw_text,
            "num_steps": len(steps),
            "is_complete": candidate.is_trajectory_complete,
            "num_tokens": num_tokens,
            "reasoning_steps": len(steps),
            "needs_answer": False,
            "candidate": candidate,
        }

    def _generate_answers(
        self,
        traj_datas: List[Dict[str, Any]],
        requests: List[List[Dict[str, str]]],
    ) -> None:
        """
        Generate answer phases for all thinking candidates that need them.

        Uses generate_answer_candidates (the proper answer generation API) which
        handles </think> closing and answer pattern appending. Only thinking
        steps are scored — answer phase is appended to full_text but excluded
        from scoring.

        Args:
            traj_datas: List of trajectory dicts from _split_thinking_candidate.
                        Modified in-place: full_text, is_complete, num_tokens
                        are updated for candidates needing answers.
            requests: Corresponding request for each traj_data.
        """
        # Collect candidates needing answer generation
        answer_indices = []
        for i, traj_data in enumerate(traj_datas):
            if traj_data.get("needs_answer"):
                answer_indices.append(i)

        if not answer_indices:
            return

        log.info(
            f"Generating {len(answer_indices)} answer phases in batched call "
            f"(only thinking steps are scored, answers excluded from scoring)"
        )

        # Batch generate all answers in one vLLM call
        batch_requests = [requests[i] for i in answer_indices]
        batch_trajectories = [[traj_datas[i]["candidate"]] for i in answer_indices]
        answer_results = self.step_generator.generate_answer_candidates_batch(
            batch_requests,
            batch_trajectories,
            candidates_per_step=1,
        )

        # Distribute answers back to trajectory data
        for batch_idx, traj_idx in enumerate(answer_indices):
            traj_data = traj_datas[traj_idx]
            candidates = (
                answer_results[batch_idx] if batch_idx < len(answer_results) else []
            )

            if candidates:
                answer_step = candidates[0]
                answer_step.is_trajectory_complete = True
                thinking_step = traj_data["candidate"]
                trajectory = [thinking_step, answer_step]
                traj_data["full_text"] = convert_trajectory_to_string(trajectory)
                answer_tokens = (
                    len(answer_step.token_ids) if answer_step.token_ids else 0
                )
                traj_data["num_tokens"] += answer_tokens
                traj_data["is_complete"] = True
                # Store answer text for logging
                answer_text = answer_step.raw_text or answer_step.text
                traj_data["answer_step"] = answer_text
            else:
                log.warning(f"No answer generated for trajectory {traj_idx}")
                traj_data["is_complete"] = False

    def get_token_stats(self) -> Dict[str, Any]:
        """Get token statistics from the generator."""
        return self.step_generator.get_sample_stats()

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: Optional[List[int]] = None,
        save_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Generate N trajectories for each of M samples using generate_step_candidates_batch.

        All M×N trajectories are generated with proper FLOP tracking,
        then each trajectory is scored with PRM and the best one per sample is selected.

        Args:
            requests: List of M chat message lists (each is a sample's request)
            sample_indices: Optional list of sample indices for logging
            save_callback: Optional callback(results, phase=str) for progressive saves

        Returns:
            List of M result dictionaries (one per sample)
        """
        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        M = len(requests)
        N = self.num_trajectories

        log.info(
            f"Offline Best-of-N batch: generating {M} samples × {N} trajectories = "
            f"{M * N} total via generate_step_candidates_batch"
        )

        # Check if scorer is a PRM model (separate model) or uses uncertainty from generation.
        use_prm_scorer = (
            hasattr(self.scorer, "prm_model") and self.scorer.prm_model is not None
        )
        use_uncertainty_wrapper = (
            hasattr(self.step_generator.model, "estimator") and not use_prm_scorer
        )
        log.info(
            f"Using PRM scorer: {use_prm_scorer}, uncertainty wrapper: {use_uncertainty_wrapper}"
        )

        # Reset per-sample tracking and generate all M×N trajectories
        self.step_generator.reset_per_sample_stats()
        if hasattr(self.scorer, "reset_prm_stats"):
            self.scorer.reset_prm_stats()
        # Build stop tokens list, including </think> for thinking mode
        stop_tokens = ["<end of response>"]
        if (
            getattr(self.step_generator, "thinking_mode", False)
            and "</think>" not in stop_tokens
        ):
            stop_tokens.append("</think>")

        batch_results = self.step_generator.generate_step_candidates_batch(
            requests=requests,
            trajectories=[[] for _ in range(M)],
            candidates_per_step=N,
            stop_tokens_override=stop_tokens,
            max_tokens=self.step_generator.generation_limit,
            compute_uncertainty=use_uncertainty_wrapper,
            sample_ids=list(range(M)),
        )

        # Phase 1: Parse StepCandidates and build trajectory data for ALL samples
        sample_data = []  # List of per-sample data
        all_chats_for_scoring = []  # Flat list for batch scoring
        all_trajectories_for_scoring = []  # Flat list for batch scoring
        all_sample_ids_for_scoring = []  # Sample IDs for logging
        all_trajectory_ids_for_scoring = []  # Trajectory IDs within sample for logging
        trajectory_to_sample_map = []  # (sample_data_idx, traj_idx_within_sample)

        # Phase 1a: Split all candidates into steps (no answer generation yet)
        all_traj_datas = []  # Flat list of all traj_datas across all samples
        all_traj_requests = []  # Corresponding request for each traj_data

        for idx, (candidates, request, sample_idx) in enumerate(
            zip(batch_results, requests, sample_indices)
        ):
            if not candidates:
                log.error(f"No output generated for sample {sample_idx}")
                sample_data.append(
                    {
                        "sample_idx": sample_idx,
                        "request": request,
                        "trajectories": [],
                        "failed": True,
                    }
                )
                continue

            trajectories = []
            for traj_idx, candidate in enumerate(candidates):
                traj_data = self._split_thinking_candidate(candidate)
                traj_data["step_scores"] = []
                traj_data["aggregated_score"] = 0.0
                trajectories.append(traj_data)
                all_traj_datas.append(traj_data)
                all_traj_requests.append(request)

            sample_data.append(
                {
                    "sample_idx": sample_idx,
                    "request": request,
                    "trajectories": trajectories,
                    "failed": False,
                }
            )

        # Phase 1b: Batch-generate answer phases for ALL thinking candidates in one vLLM call
        self._generate_answers(all_traj_datas, all_traj_requests)

        # Progressive save: checkpoint after generation, before scoring
        if save_callback is not None:
            pre_scoring_results = self._build_results_from_sample_data(sample_data, N)
            save_callback(pre_scoring_results, phase="post_generation")

        # Phase 1c: Compute uncertainty scores and build scoring lists
        for sample_data_idx, data in enumerate(sample_data):
            if data["failed"]:
                continue
            sample_idx = data["sample_idx"]
            request = data["request"]
            for traj_idx, traj_data in enumerate(data["trajectories"]):
                candidate = traj_data["candidate"]

                # Compute per-step uncertainty if using uncertainty wrapper
                # Score ONLY thinking steps using original candidate's token_ids/logprobs
                if use_uncertainty_wrapper and candidate.other_data.get("raw_logprobs"):
                    step_scores = self._compute_per_step_uncertainty(
                        steps=traj_data["steps"],
                        token_ids=list(candidate.token_ids),
                        logprobs=candidate.other_data["raw_logprobs"],
                        uncertainty_wrapper=self.step_generator.model,
                    )
                    aggregated = self._aggregate_scores(step_scores)
                    traj_data["step_scores"] = step_scores
                    traj_data["aggregated_score"] = aggregated

                # Add to flat lists for batch scoring (only if has steps AND no pre-computed scores)
                if traj_data["steps"] and not use_uncertainty_wrapper:
                    all_chats_for_scoring.append(request)
                    all_trajectories_for_scoring.append(traj_data["steps"])
                    all_sample_ids_for_scoring.append(sample_idx)
                    all_trajectory_ids_for_scoring.append(traj_idx)
                    trajectory_to_sample_map.append((sample_data_idx, traj_idx))

        # Phase 1d: Compute extra logprob-based metrics if multi-scoring enabled
        if self._extra_metrics:
            self._compute_extra_logprob_scores(sample_data)

        # Phase 2: Batch score ALL trajectories in single call
        # Skip if using uncertainty wrapper (scores already computed during generation)
        if use_uncertainty_wrapper:
            log.info(
                "Skipping batch scoring - using uncertainty scores "
                f"from generation ({sum(len(d['trajectories']) for d in sample_data)} trajectories)"
            )
            # Log uncertainty scores
            log.info("--- Uncertainty Scoring Results ---")
            for data in sample_data:
                if data["failed"]:
                    continue
                sample_idx = data["sample_idx"]
                trajectories = data["trajectories"]
                log.info(f"Sample {sample_idx}: {len(trajectories)} trajectories")
                for traj_idx, traj in enumerate(trajectories):
                    log.info(
                        f"  Trajectory {traj_idx + 1}: "
                        f"steps={traj['num_steps']}, "
                        f"aggregated_validity={traj['aggregated_score']:.4f}, "
                        f"step_validities={[f'{s:.3f}' for s in traj['step_scores']]}"
                    )
        elif all_trajectories_for_scoring:
            # Use batch scoring if available, otherwise falls back to sequential
            log.info(
                f"Batch scoring {len(all_trajectories_for_scoring)} trajectories "
                f"from {len(sample_data)} samples"
            )
            all_scores = self.scorer.score_trajectories_batch(
                all_chats_for_scoring,
                all_trajectories_for_scoring,
                sample_ids=all_sample_ids_for_scoring,
                trajectory_ids=all_trajectory_ids_for_scoring,
            )

            # Distribute scores back to trajectories
            for flat_idx, (sample_data_idx, traj_idx) in enumerate(
                trajectory_to_sample_map
            ):
                step_scores = all_scores[flat_idx]
                sample_data[sample_data_idx]["trajectories"][traj_idx][
                    "step_scores"
                ] = step_scores
                agg = self._aggregate_scores(step_scores)
                sample_data[sample_data_idx]["trajectories"][traj_idx][
                    "aggregated_score"
                ] = agg

            # Log PRM scoring summary per sample
            log.info("--- PRM Score Distribution Summary ---")
            for data in sample_data:
                if data["failed"]:
                    continue
                sample_idx = data["sample_idx"]
                trajectories = data["trajectories"]
                for traj_idx, traj in enumerate(trajectories):
                    scores = traj["step_scores"]
                    num_none = sum(1 for s in scores if s is None)
                    valid = [s for s in scores if s is not None]
                    score_strs = [
                        f"{s:.3f}" if s is not None else "null" for s in scores
                    ]
                    valid_summary = (
                        f"min={min(valid):.3f}, max={max(valid):.3f}, mean={sum(valid)/len(valid):.3f}"
                        if valid
                        else "no valid scores"
                    )
                    log.info(
                        f"  Sample {sample_idx}, Traj {traj_idx}: "
                        f"agg({self.score_aggregation})={traj['aggregated_score']:.4f}, "
                        f"{len(scores)} steps ({num_none} null), "
                        f"valid: {valid_summary}, "
                        f"scores=[{', '.join(score_strs)}]"
                    )

        # Phase 2b: Compute extra PRM scores if enabled
        if self.calculate_prm_score:
            self._compute_extra_prm_scores(sample_data, use_prm_scorer)

        # Progressive save: checkpoint after scoring
        if save_callback is not None:
            scored_results = self._build_results_from_sample_data(sample_data, N)
            save_callback(scored_results, phase="post_scoring")

        # Phase 3: Select best trajectory per sample and build results
        results = self._build_results_from_sample_data(sample_data, N)

        log.info(
            f"Offline Best-of-N batch: completed {len(results)} samples, "
            f"total {M * N} trajectories generated and scored"
        )
        return results

    def _get_tokenizer(self):
        """Get tokenizer from step generator for token boundary mapping."""
        if hasattr(self.step_generator, "model") and hasattr(
            self.step_generator.model, "get_tokenizer"
        ):
            return self.step_generator.model.get_tokenizer()
        if hasattr(self.step_generator, "tokenizer"):
            return self.step_generator.tokenizer
        return None

    def _compute_extra_logprob_scores(self, sample_data: List[Dict]) -> None:
        """Compute extra logprob-based metrics for all candidates.

        Stores results in traj_data["extra_scores"] for each trajectory.
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            log.warning("Cannot compute extra logprob scores: no tokenizer available")
            return

        total_computed = 0
        for data in sample_data:
            if data["failed"]:
                continue
            for traj_data in data["trajectories"]:
                candidate = traj_data["candidate"]
                if not candidate.other_data or not candidate.other_data.get("raw_logprobs"):
                    continue

                token_ids = list(candidate.token_ids)
                raw_logprobs = candidate.other_data["raw_logprobs"]

                # Compute per-step scores
                per_step = compute_logprob_scores_per_step(
                    steps=traj_data["steps"],
                    token_ids=token_ids,
                    raw_logprobs=raw_logprobs,
                    tokenizer=tokenizer,
                    metrics=self._extra_metrics,
                )

                # Compute trajectory-level scores
                trajectory_scores = compute_logprob_scores(
                    token_ids=token_ids,
                    raw_logprobs=raw_logprobs,
                    metrics=self._extra_metrics,
                )

                # Store in extra_scores
                extra_scores = traj_data.get("extra_scores", {})
                for metric in self._extra_metrics:
                    extra_scores[metric] = {
                        "per_step": per_step.get(metric, []),
                        "trajectory": trajectory_scores.get(metric, float("nan")),
                    }
                traj_data["extra_scores"] = extra_scores
                total_computed += 1

        log.info(
            f"Computed extra logprob scores ({self._extra_metrics}) "
            f"for {total_computed} trajectories"
        )

    def _compute_extra_prm_scores(
        self, sample_data: List[Dict], primary_is_prm: bool
    ) -> None:
        """Compute PRM scores for all candidates if calculate_prm_score is enabled.

        If the primary scorer is already PRM, reuses those scores.
        Otherwise uses self.prm_scorer for a separate PRM pass.
        """
        if primary_is_prm:
            # Primary scorer is PRM — reuse step_scores as PRM scores
            for data in sample_data:
                if data["failed"]:
                    continue
                for traj_data in data["trajectories"]:
                    extra_scores = traj_data.get("extra_scores", {})
                    step_scores = traj_data.get("step_scores", [])
                    valid_scores = [s for s in step_scores if s is not None]
                    extra_scores["prm"] = {
                        "per_step": step_scores,
                        "trajectory": float(np.mean(valid_scores)) if valid_scores else float("nan"),
                    }
                    traj_data["extra_scores"] = extra_scores
            log.info("Reused primary PRM scores for multi-scoring candidates_data")
            return

        if self.prm_scorer is None:
            log.warning("calculate_prm_score=True but no prm_scorer provided, skipping")
            return

        # Batch all trajectories for PRM scoring
        all_chats = []
        all_trajectories = []
        all_indices = []  # (sample_data_idx, traj_idx)
        for sample_data_idx, data in enumerate(sample_data):
            if data["failed"]:
                continue
            for traj_idx, traj_data in enumerate(data["trajectories"]):
                if traj_data["steps"]:
                    all_chats.append(data["request"])
                    all_trajectories.append(traj_data["steps"])
                    all_indices.append((sample_data_idx, traj_idx))

        if not all_trajectories:
            return

        log.info(f"Computing extra PRM scores for {len(all_trajectories)} trajectories")
        all_prm_scores = self.prm_scorer.score_trajectories_batch(
            all_chats,
            all_trajectories,
            sample_ids=[i for i, _ in all_indices],
            trajectory_ids=[j for _, j in all_indices],
        )

        # Distribute PRM scores back
        for flat_idx, (sample_data_idx, traj_idx) in enumerate(all_indices):
            traj_data = sample_data[sample_data_idx]["trajectories"][traj_idx]
            prm_step_scores = all_prm_scores[flat_idx]
            valid = [s for s in prm_step_scores if s is not None]
            extra_scores = traj_data.get("extra_scores", {})
            extra_scores["prm"] = {
                "per_step": prm_step_scores,
                "trajectory": float(np.mean(valid)) if valid else float("nan"),
            }
            traj_data["extra_scores"] = extra_scores

        log.info(f"Extra PRM scoring complete for {len(all_trajectories)} trajectories")

    def _build_results_from_sample_data(
        self, sample_data: List[Dict], N: int
    ) -> List[Dict[str, Any]]:
        """Build result dicts from sample_data by selecting best trajectory per sample.

        Handles missing PRM stats gracefully (they won't exist during pre-scoring save).

        Args:
            sample_data: List of per-sample dicts with trajectories and metadata
            N: Number of trajectories per sample

        Returns:
            List of result dicts (one per sample)
        """
        results = []
        for sample_data_idx, data in enumerate(sample_data):
            if data["failed"]:
                results.append(self._empty_result())
                continue

            trajectories = data["trajectories"]
            sample_idx = data["sample_idx"]

            # Select best trajectory
            aggregated_scores = [t["aggregated_score"] for t in trajectories]
            best_idx = int(np.argmax(aggregated_scores)) if aggregated_scores else 0
            best_result = (
                trajectories[best_idx] if trajectories else self._empty_result()
            )

            # Extract answer from best trajectory
            answer_text = best_result.get("answer_step") or best_result.get(
                "full_text", ""
            )
            extracted = extract_answer(answer_text)

            # Token stats from generator's per-sample tracking
            token_stats = self.step_generator.get_sample_stats_for(sample_data_idx)
            token_stats["generation_count"] = N

            # Merge PRM scorer stats if available
            if hasattr(self.scorer, "get_prm_stats_for"):
                prm_stats = self.scorer.get_prm_stats_for(sample_idx)
                if prm_stats.get("prm_input_tokens"):
                    token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
                    token_stats["prm_tflops"] = prm_stats["prm_tflops"]
                    gen_tflops = token_stats.get("tflops")
                    if gen_tflops is None:
                        log.warning(
                            f"Sample {sample_idx}: missing 'tflops' in token_stats when merging PRM stats"
                        )
                        gen_tflops = 0
                    prm_tflops = prm_stats["prm_tflops"]
                    if prm_tflops is None:
                        log.warning(
                            f"Sample {sample_idx}: missing 'prm_tflops' in PRM stats"
                        )
                        prm_tflops = 0
                    token_stats["tflops"] = gen_tflops + prm_tflops

            log.info(
                f"Sample {sample_idx}: Selected trajectory {best_idx + 1}/{N} "
                f"(aggregation={self.score_aggregation}, "
                f"score={best_result.get('aggregated_score', 0.0):.4f}, "
                f"steps={best_result.get('num_steps', 0)})"
            )

            result_dict = {
                "trajectory": best_result.get("full_text", ""),
                "extracted_answer": extracted,
                "steps": best_result.get("steps", []),
                "answer_step": best_result.get("answer_step", None),
                "reasoning_steps": best_result.get("reasoning_steps", 0),
                "validity_scores": best_result.get("step_scores", []),
                "aggregated_score": best_result.get("aggregated_score", 0.0),
                "all_trajectories": [t["full_text"] for t in trajectories],
                "all_scores": aggregated_scores,
                "all_step_scores": [t["step_scores"] for t in trajectories],
                "best_idx": best_idx,
                "completed": best_result.get("is_complete", False),
                "token_stats": token_stats,
            }

            # Add candidates_data for multi-scoring analysis
            if self._multi_scoring_enabled:
                result_dict["candidates_data"] = [
                    {
                        "trajectory": traj["full_text"],
                        "extracted_answer": extract_answer(
                            traj.get("answer_step") or traj["full_text"]
                        ),
                        "steps": traj["steps"],
                        "num_tokens": traj["num_tokens"],
                        "scores": traj.get("extra_scores", {}),
                    }
                    for traj in trajectories
                ]

            results.append(result_dict)

        return results

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for failed generation."""
        return {
            "trajectory": "",
            "extracted_answer": "",
            "steps": [],
            "answer_step": None,
            "reasoning_steps": 0,
            "validity_scores": [],
            "aggregated_score": 0.0,
            "all_trajectories": [],
            "all_scores": [],
            "all_step_scores": [],
            "best_idx": 0,
            "completed": False,
            "token_stats": {},
        }

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
