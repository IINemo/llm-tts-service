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

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from llm_tts.generators.base import StepCandidate, StepCandidateGeneratorBase
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
        """
        self.scorer = scorer
        self.num_trajectories = num_trajectories
        self.max_steps = max_steps
        self.step_generator = step_generator
        self.score_aggregation = score_aggregation
        self.output_dir = output_dir
        self.batch_generation = batch_generation
        self._current_sample_idx = 0

        log.info(
            f"StrategyOfflineBestOfN initialized: "
            f"{num_trajectories} trajectories, max_steps={max_steps}, "
            f"aggregation={score_aggregation}, batch_generation={batch_generation}"
        )

    def _get_uncertainty_score(self, candidate: StepCandidate) -> float:
        """Get uncertainty score from candidate's other_data."""
        if candidate.other_data and "uncertainty_score" in candidate.other_data:
            return candidate.other_data["uncertainty_score"]
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
            step_scores: List of scores for each step

        Returns:
            Aggregated score (higher = better)
        """
        if not step_scores:
            return 0.0

        if self.score_aggregation == "mean":
            return float(np.mean(step_scores))
        elif self.score_aggregation == "min":
            # Conservative: trajectory is only as good as its weakest step
            return float(np.min(step_scores))
        elif self.score_aggregation == "max":
            # Optimistic: best step determines trajectory score
            return float(np.max(step_scores))
        elif self.score_aggregation == "product":
            return float(np.prod(step_scores))
        elif self.score_aggregation == "last":
            return step_scores[-1]
        else:
            log.warning(f"Unknown aggregation '{self.score_aggregation}', using mean")
            return float(np.mean(step_scores))

    def _generate_all_trajectories_single_call(
        self,
        request: List[Dict[str, str]],
    ) -> List[Dict[str, any]]:
        """
        Generate all N trajectories in a SINGLE vLLM call.

        Uses generate_full_trajectories() which:
        1. Generates N complete trajectories with n=num_trajectories
        2. Splits each into steps post-hoc using the same stop tokens

        This is much faster than step-by-step generation because there's
        no stopping at step boundaries during generation.

        Args:
            request: Chat messages for the request

        Returns:
            List of trajectory dictionaries (scores added later)
        """
        log.info(
            f"\n--- Generating {self.num_trajectories} trajectories (single call) ---"
        )

        # Single vLLM call generates all N trajectories
        raw_results = self.step_generator.generate_full_trajectories(
            request=request,
            num_trajectories=self.num_trajectories,
        )

        # Convert to expected format
        results = []
        for i, raw in enumerate(raw_results):
            results.append(
                {
                    "steps": raw["steps"],  # List of step strings
                    "step_scores": [],  # Will be filled after scoring
                    "aggregated_score": 0.0,  # Will be filled after scoring
                    "full_text": raw["full_text"],
                    "num_steps": len(raw["steps"]),
                    "is_complete": raw["is_complete"],
                }
            )

        return results

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, any]:
        """
        Generate N complete trajectories and return the best one.

        Uses single-call vLLM generation for maximum throughput - all N trajectories
        are generated in ONE vLLM call with n=num_trajectories.

        Args:
            request: Chat messages for the request
            sample_idx: Index of current sample (for logging)

        Returns:
            Dictionary with best trajectory and metadata
        """
        self._current_sample_idx = sample_idx

        # Reset token tracking
        self.step_generator.reset_sample_stats()
        if hasattr(self.scorer, "reset_prm_stats"):
            self.scorer.reset_prm_stats()

        log.info(f"\n{'='*60}")
        log.info(
            f"Generating {self.num_trajectories} trajectories (single-call offline)"
        )
        log.info(f"Score aggregation: {self.score_aggregation}")
        log.info(f"{'='*60}")

        # Step 1: Generate all trajectories in ONE vLLM call
        all_trajectory_results = self._generate_all_trajectories_single_call(request)

        # Step 2: Score all trajectories efficiently (one PRM call per trajectory)
        log.info(f"\n{'='*60}")
        log.info(f"Scoring {self.num_trajectories} trajectories")
        log.info(f"{'='*60}")

        for i, result in enumerate(all_trajectory_results):
            if result["steps"]:
                # Score entire trajectory in single forward pass
                # steps is List[str] - PRM scorer handles this via hasattr check
                step_scores = self.scorer.score_trajectory(request, result["steps"])
                result["step_scores"] = step_scores
                result["aggregated_score"] = self._aggregate_scores(step_scores)
            else:
                result["step_scores"] = []
                result["aggregated_score"] = 0.0

        # Log summary
        log.info("\n--- Trajectory Scores Summary ---")
        for i, result in enumerate(all_trajectory_results):
            log.info(
                f"Trajectory {i + 1}: "
                f"aggregated={result['aggregated_score']:.4f}, "
                f"steps={result['num_steps']}, "
                f"complete={result['is_complete']}, "
                f"step_scores={[f'{s:.3f}' for s in result['step_scores']]}"
            )

        # Select best trajectory
        aggregated_scores = [r["aggregated_score"] for r in all_trajectory_results]
        best_idx = int(np.argmax(aggregated_scores))
        best_result = all_trajectory_results[best_idx]

        log.info(f"\n{'='*60}")
        log.info(
            f"Selected trajectory {best_idx + 1} "
            f"with aggregated score {best_result['aggregated_score']:.4f}"
        )
        log.info(f"{'='*60}")

        # Extract answer from best trajectory
        extracted = extract_answer(best_result["full_text"])

        # Get token stats
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

        # Save logs if output_dir provided
        if self.output_dir:
            self._save_trajectories_log(all_trajectory_results, best_idx)

        return {
            "trajectory": best_result["full_text"],
            "extracted_answer": extracted,
            "steps": best_result["steps"],  # List of step strings
            "thinking_num_steps": 0,  # Not tracked in single-call mode
            "response_num_steps": best_result["num_steps"],
            "validity_scores": best_result["step_scores"],
            "aggregated_score": best_result["aggregated_score"],
            "all_trajectories": [r["full_text"] for r in all_trajectory_results],
            "all_scores": aggregated_scores,
            "all_step_scores": [r["step_scores"] for r in all_trajectory_results],
            "best_idx": best_idx,
            "completed": best_result["is_complete"],
            "token_stats": token_stats,
        }

    def _save_trajectories_log(
        self,
        all_results: List[Dict[str, any]],
        best_idx: int,
    ):
        """Save all trajectories to JSON for analysis."""
        if not self.output_dir:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(
            self.output_dir, f"trajectories_sample_{self._current_sample_idx}.json"
        )

        log_data = {
            "sample_idx": self._current_sample_idx,
            "num_trajectories": len(all_results),
            "score_aggregation": self.score_aggregation,
            "best_idx": best_idx,
            "best_score": all_results[best_idx]["aggregated_score"],
            "trajectories": [
                {
                    "idx": i,
                    "aggregated_score": r["aggregated_score"],
                    "step_scores": r["step_scores"],
                    "num_steps": r["num_steps"],
                    "is_complete": r["is_complete"],
                    "text": r["full_text"],
                    "steps": r["steps"],  # Individual step texts
                    "is_best": i == best_idx,
                }
                for i, r in enumerate(all_results)
            ],
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        log.info(f"Saved trajectories log to {log_path}")

    def get_token_stats(self) -> Dict[str, any]:
        """Get token statistics from the generator."""
        return self.step_generator.get_sample_stats()

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, any]]:
        """
        Generate N trajectories for each of M samples using generate_step_candidates_batch.

        All M×N trajectories are generated with proper FLOP tracking,
        then each trajectory is scored with PRM and the best one per sample is selected.

        Args:
            requests: List of M chat message lists (each is a sample's request)
            sample_indices: Optional list of sample indices for logging

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
        batch_results = self.step_generator.generate_step_candidates_batch(
            requests=requests,
            trajectories=[[]] * M,
            candidates_per_step=N,
            stop_tokens_override=["<end of response>"],
            max_tokens_override=self.step_generator.max_new_tokens,
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

            # Build trajectory results from the N StepCandidates
            trajectories = []

            for traj_idx, candidate in enumerate(candidates):
                raw_text = candidate.raw_text or candidate.text
                num_tokens = candidate.other_data.get(
                    "original_token_count", len(candidate.token_ids)
                )

                # Split into steps post-hoc using step generator's detector
                if hasattr(self.step_generator, "detector"):
                    steps = self.step_generator.detector.detect_steps(
                        raw_text, use_stop_tokens=True
                    )
                else:
                    steps = [raw_text]  # Fallback: treat as single step

                # Compute per-step uncertainty if using uncertainty wrapper
                # Score each step independently using its token logprobs
                if use_uncertainty_wrapper and candidate.other_data.get("raw_logprobs"):
                    step_scores = self._compute_per_step_uncertainty(
                        steps=steps,
                        token_ids=list(candidate.token_ids),
                        logprobs=candidate.other_data["raw_logprobs"],
                        uncertainty_wrapper=self.step_generator.model,
                    )
                    aggregated = self._aggregate_scores(step_scores)
                else:
                    step_scores = []
                    aggregated = 0.0

                traj_data = {
                    "steps": steps,
                    "step_scores": step_scores,
                    "aggregated_score": aggregated,
                    "full_text": raw_text,
                    "num_steps": len(steps),
                    "is_complete": candidate.is_trajectory_complete,
                    "num_tokens": num_tokens,
                }
                trajectories.append(traj_data)

                # Add to flat lists for batch scoring (only if has steps AND no pre-computed scores)
                if steps and not use_uncertainty_wrapper:
                    all_chats_for_scoring.append(request)
                    all_trajectories_for_scoring.append(steps)
                    all_sample_ids_for_scoring.append(sample_idx)
                    all_trajectory_ids_for_scoring.append(traj_idx)
                    trajectory_to_sample_map.append((len(sample_data), traj_idx))

            sample_data.append(
                {
                    "sample_idx": sample_idx,
                    "request": request,
                    "trajectories": trajectories,
                    "failed": False,
                }
            )

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
                sample_data[sample_data_idx]["trajectories"][traj_idx][
                    "aggregated_score"
                ] = self._aggregate_scores(step_scores)

        # Phase 3: Select best trajectory per sample and build results
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
            extracted = extract_answer(best_result.get("full_text", ""))

            # Token stats from generator's per-sample tracking
            token_stats = self.step_generator.get_sample_stats_for(sample_data_idx)
            token_stats["generation_count"] = N  # N candidates in one vLLM call

            # Merge PRM scorer stats if available
            # PRM scorer tracks tokens keyed by dataset sample_idx (passed via score_trajectories_batch)
            if hasattr(self.scorer, "get_prm_stats_for"):
                prm_stats = self.scorer.get_prm_stats_for(sample_idx)
                token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
                token_stats["prm_tflops"] = prm_stats["prm_tflops"]
                gen_tflops = token_stats.get("tflops")
                if gen_tflops is None:
                    log.warning(f"Sample {sample_idx}: missing 'tflops' in token_stats when merging PRM stats")
                    gen_tflops = 0
                prm_tflops = prm_stats["prm_tflops"]
                if prm_tflops is None:
                    log.warning(f"Sample {sample_idx}: missing 'prm_tflops' in PRM stats")
                    prm_tflops = 0
                token_stats["tflops"] = gen_tflops + prm_tflops

            log.info(
                f"Sample {sample_idx}: best trajectory {best_idx + 1}/{N}, "
                f"score={best_result.get('aggregated_score', 0.0):.4f}, "
                f"steps={best_result.get('num_steps', 0)}"
            )

            results.append(
                {
                    "trajectory": best_result.get("full_text", ""),
                    "extracted_answer": extracted,
                    "steps": best_result.get("steps", []),
                    "thinking_num_steps": 0,
                    "response_num_steps": best_result.get("num_steps", 0),
                    "validity_scores": best_result.get("step_scores", []),
                    "aggregated_score": best_result.get("aggregated_score", 0.0),
                    "all_trajectories": [t["full_text"] for t in trajectories],
                    "all_scores": aggregated_scores,
                    "all_step_scores": [t["step_scores"] for t in trajectories],
                    "best_idx": best_idx,
                    "completed": best_result.get("is_complete", False),
                    "token_stats": token_stats,
                }
            )

        log.info(
            f"Offline Best-of-N batch: completed {len(results)} samples, "
            f"total {M * N} trajectories generated and scored"
        )
        return results

    def _empty_result(self) -> Dict[str, any]:
        """Return empty result for failed generation."""
        return {
            "trajectory": "",
            "extracted_answer": "",
            "steps": [],
            "thinking_num_steps": 0,
            "response_num_steps": 0,
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
