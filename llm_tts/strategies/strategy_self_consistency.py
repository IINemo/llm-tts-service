"""
Self-consistency strategy for LLM reasoning.

Based on "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
by Wang et al. (2022). Generates multiple diverse reasoning paths and selects
the most consistent answer via majority voting.

Uses VLLMStepGenerator for step-by-step generation with uncertainty scoring
and FLOP tracking.
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from llm_tts.generators.base import convert_trajectory_to_string
from llm_tts.scorers.majority_voting import ChainMajorityVotingScorer
from llm_tts.utils import extract_answer

from .metadata_builder import StrategyMetadataBuilder
from .strategy_base import StrategyBase, count_thinking_and_response_steps

if TYPE_CHECKING:
    from llm_tts.generators import StepCandidateGeneratorBase

log = logging.getLogger(__name__)


class StrategySelfConsistency(StrategyBase):
    """
    Self-consistency strategy that generates multiple reasoning paths
    and selects the most consistent answer via majority voting.

    Uses step generator for step-by-step generation with uncertainty scoring.
    """

    def __init__(
        self,
        step_generator: "StepCandidateGeneratorBase",
        num_paths: int = 10,
        max_steps: int = 250,
        scorer: Optional[Any] = None,
        parallel: bool = True,
    ):
        """
        Initialize self-consistency strategy.

        Args:
            step_generator: Step generator (VLLMStepGenerator) for step-by-step
                           generation with uncertainty scoring.
            num_paths: Number of reasoning paths to generate
            max_steps: Maximum steps per trace (default 250)
            scorer: Custom scorer for answer selection (defaults to majority voting)
            parallel: If True (default), generate all paths in parallel using batch
                     generation. This is ~num_paths times faster than sequential.
        """
        self.step_generator = step_generator
        self.num_paths = num_paths
        self.max_steps = max_steps
        self.parallel = parallel

        # Use majority voting scorer by default
        self.scorer = scorer or ChainMajorityVotingScorer()
        if hasattr(self.scorer, "prepare_model"):
            self.scorer.prepare_model()

    def _generate_single_trace(
        self,
        request: List[Dict[str, str]],
        path_idx: int,
    ) -> Dict[str, Any]:
        """
        Generate a single complete trace step-by-step using the step generator.

        Args:
            request: Chat messages for the request
            path_idx: Index of this path (for logging)

        Returns:
            Dictionary with:
                - text: Complete trace text
                - num_tokens: Total tokens generated
                - steps: List of step texts
                - validity_scores: Validity score per step
        """
        trajectory = []
        total_tokens = 0
        validity_scores = []

        for step_num in range(self.max_steps):
            # Generate single candidate (no selection in self-consistency)
            candidates = self.step_generator(
                request,
                trajectory=trajectory,
                candidates_per_step=1,
            )

            if not candidates:
                log.warning(f"  Path {path_idx + 1}: No candidates at step {step_num}")
                break

            candidate = candidates[0]
            trajectory.append(candidate)

            # Track tokens
            step_tokens = len(candidate.token_ids) if candidate.token_ids else 0
            total_tokens += step_tokens

            # Get validity score from other_data
            validity = 1.0
            if candidate.other_data and "validity_score" in candidate.other_data:
                validity = candidate.other_data["validity_score"]
            validity_scores.append(validity)

            # Check if trajectory is complete
            if candidate.is_trajectory_complete:
                log.info(
                    f"  Path {path_idx + 1}: Completed at step {step_num + 1}, "
                    f"tokens={total_tokens}"
                )
                break

        # Convert trajectory to text
        trace_text = convert_trajectory_to_string(trajectory)

        # Count thinking vs response steps
        thinking_steps, response_steps = count_thinking_and_response_steps(trajectory)

        return {
            "text": trace_text,
            "num_tokens": total_tokens,
            "steps": [s.text for s in trajectory],
            "validity_scores": validity_scores,
            "thinking_steps": thinking_steps,
            "response_steps": response_steps,
            "avg_validity": (
                sum(validity_scores) / len(validity_scores) if validity_scores else 0
            ),
        }

    def _generate_paths_parallel(
        self, request: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple reasoning paths in parallel using batch generation.

        Instead of generating paths sequentially (path1 complete, then path2, etc.),
        this generates the next step for ALL active paths in a single vLLM batch call.
        This is ~num_paths times faster.

        Architecture:
            Step 1: [path1_step1, path2_step1, ..., pathN_step1]  ← single batch
            Step 2: [path1_step2, path2_step2, ..., pathN_step2]  ← single batch
            ...
            (paths complete at different times, batch shrinks)

        Args:
            request: Chat messages in OpenAI format

        Returns:
            List of dicts with text, num_tokens, steps, validity_scores per path
        """
        log.info(
            f"Generating {self.num_paths} reasoning paths in PARALLEL (batch mode)..."
        )

        # Initialize all paths
        trajectories = [[] for _ in range(self.num_paths)]
        path_tokens = [0 for _ in range(self.num_paths)]
        path_validity_scores = [[] for _ in range(self.num_paths)]
        active_paths = set(range(self.num_paths))

        for step_num in range(self.max_steps):
            if not active_paths:
                break

            # Get trajectories for active paths only
            active_indices = sorted(active_paths)
            active_trajectories = [trajectories[i] for i in active_indices]

            # Batch generate next step for all active paths
            step_candidates_nested = self.step_generator.generate_step_candidates(
                request, active_trajectories, candidates_per_step=1
            )
            # Flatten: each trajectory gets 1 candidate
            step_candidates = [cands[0] for cands in step_candidates_nested]

            # Process results and update trajectories
            newly_completed = []
            for idx, path_idx in enumerate(active_indices):
                candidate = step_candidates[idx]
                trajectories[path_idx].append(candidate)

                # Track tokens and validity
                step_tokens = len(candidate.token_ids) if candidate.token_ids else 0
                path_tokens[path_idx] += step_tokens

                validity = 1.0
                if candidate.other_data and "validity_score" in candidate.other_data:
                    validity = candidate.other_data["validity_score"]
                path_validity_scores[path_idx].append(validity)

                # Check if path is complete
                if candidate.is_trajectory_complete:
                    newly_completed.append(path_idx)

            # Remove completed paths from active set
            for path_idx in newly_completed:
                active_paths.discard(path_idx)
                log.info(
                    f"  Path {path_idx + 1}: Completed at step {len(trajectories[path_idx])}, "
                    f"tokens={path_tokens[path_idx]}"
                )

            # Log progress every 10 steps
            if (step_num + 1) % 10 == 0:
                log.info(
                    f"  Step {step_num + 1}: {len(active_paths)} paths still active"
                )

        # Build result dicts for all paths
        paths = []
        for i in range(self.num_paths):
            trajectory = trajectories[i]
            trace_text = convert_trajectory_to_string(trajectory)
            thinking_steps, response_steps = count_thinking_and_response_steps(
                trajectory
            )
            validity_scores = path_validity_scores[i]

            answer = extract_answer(trace_text, answer_format="auto") or "no_answer"
            avg_validity = (
                sum(validity_scores) / len(validity_scores) if validity_scores else 0
            )

            log.info(
                f"  Path {i + 1}/{self.num_paths}: "
                f"steps={len(trajectory)}, tokens={path_tokens[i]}, "
                f"avg_validity={avg_validity:.3f}, answer={answer}"
            )

            paths.append(
                {
                    "text": trace_text,
                    "num_tokens": path_tokens[i],
                    "steps": [s.text for s in trajectory],
                    "validity_scores": validity_scores,
                    "thinking_steps": thinking_steps,
                    "response_steps": response_steps,
                    "avg_validity": avg_validity,
                }
            )

        total_tokens = sum(path_tokens)
        log.info(
            f"Generated {self.num_paths} paths in parallel, total tokens: {total_tokens}"
        )

        return paths

    def generate_reasoning_paths(
        self, request: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple reasoning paths using step-by-step generation.

        Each path is generated independently using the step generator.
        This provides uncertainty scores per step and accurate FLOP tracking.

        Args:
            request: Chat messages in OpenAI format

        Returns:
            List of dicts with text, num_tokens, steps, validity_scores per path
        """
        # Reset stats for this sample
        self.step_generator.reset_sample_stats()

        # Use parallel or sequential generation based on config
        if self.parallel:
            paths = self._generate_paths_parallel(request)
        else:
            log.info(f"Generating {self.num_paths} reasoning paths SEQUENTIALLY...")
            paths = []
            total_tokens = 0

            for i in range(self.num_paths):
                trace = self._generate_single_trace(request, i)
                paths.append(trace)
                total_tokens += trace["num_tokens"]

                # Extract answer for logging
                answer = (
                    extract_answer(trace["text"], answer_format="auto") or "no_answer"
                )
                log.info(
                    f"  Path {i + 1}/{self.num_paths}: "
                    f"steps={len(trace['steps'])}, tokens={trace['num_tokens']}, "
                    f"avg_validity={trace['avg_validity']:.3f}, answer={answer}"
                )

            log.info(f"Generated {len(paths)}/{self.num_paths} paths sequentially")
            log.info(f"  Total tokens: {total_tokens}")

        # Finalize stats
        self.step_generator.finalize_sample_stats()
        token_stats = self.step_generator.get_sample_stats()

        log.info(f"Token stats - TFLOPs: {token_stats.get('tflops', 0):.3f}")

        return paths

    def select_best_answer(self, reasoning_paths: List[Dict]) -> Dict[str, Any]:
        """
        Select the best answer using majority voting across reasoning paths.

        Args:
            reasoning_paths: List of path dicts with 'text' and 'num_tokens'

        Returns:
            Dictionary containing:
                - best_path: The reasoning path with the most consistent answer
                - best_answer: The extracted answer
                - consensus_score: Confidence based on answer frequency
                - all_answers: All extracted answers for debugging
                - answer_distribution: Answer frequency distribution
                - all_traces: List of dicts with text, num_tokens, answer for each path
        """
        if not reasoning_paths:
            return {
                "best_path": "",
                "best_answer": "no_answer",
                "consensus_score": 0.0,
                "all_answers": [],
                "answer_distribution": {},
                "all_traces": [],
                "total_tokens": 0,
            }

        # Extract texts and tokens from path dicts
        path_texts = [p["text"] for p in reasoning_paths]
        path_tokens = [p["num_tokens"] for p in reasoning_paths]

        # Use the scorer to get consensus scores
        scores = self.scorer.score_complete_chains(path_texts)

        # Find the path with highest consensus
        best_idx = int(np.argmax(scores))
        best_path = path_texts[best_idx]
        best_score = float(scores[best_idx])

        # Extract the best answer
        best_answer = self.scorer.extract_answer(best_path)

        # Get all answers for analysis
        all_answers = [self.scorer.extract_answer(path) for path in path_texts]

        # Calculate answer distribution
        answer_counts = Counter(all_answers)

        log.info(
            f"Selected reasoning path {best_idx} with consensus score {best_score:.3f}"
        )
        log.info(f"Best answer: {best_answer}")
        log.info(f"Answer distribution: {dict(answer_counts)}")

        # Build all_traces with token info and step details
        all_traces = []
        for i, (path_data, answer) in enumerate(zip(reasoning_paths, all_answers)):
            all_traces.append(
                {
                    "text": path_data["text"],
                    "num_tokens": path_data["num_tokens"],
                    "num_steps": len(path_data.get("steps", [])),
                    "thinking_steps": path_data.get("thinking_steps", 0),
                    "response_steps": path_data.get("response_steps", 0),
                    "avg_validity": path_data.get("avg_validity", 0),
                    "answer": answer,
                    "score": float(scores[i]),
                    "selected": i == best_idx,
                }
            )

        total_tokens = sum(path_tokens)
        log.info(f"Total tokens across all paths: {total_tokens}")

        return {
            "best_path": best_path,
            "best_answer": best_answer,
            "consensus_score": best_score,
            "all_answers": all_answers,
            "answer_distribution": dict(answer_counts),
            "all_paths": path_texts,
            "all_scores": [float(s) for s in scores],
            "all_traces": all_traces,
            "total_tokens": total_tokens,
        }

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Main entry point for self-consistency reasoning.

        Args:
            request: Chat messages in OpenAI format
            sample_idx: Index of current sample (for logging)

        Returns:
            Dictionary with trajectory information compatible with evaluation framework
        """
        prompt_preview = request[0]["content"][:100] if request else ""
        log.info(f"Starting self-consistency reasoning for: {prompt_preview}...")

        # Generate multiple reasoning paths
        reasoning_paths = self.generate_reasoning_paths(request)

        # Select best answer via majority voting
        result = self.select_best_answer(reasoning_paths)

        # Get token stats from step generator
        token_stats = self.step_generator.get_sample_stats()

        # Build metadata using StrategyMetadataBuilder
        builder = StrategyMetadataBuilder("self_consistency")

        # Add configuration
        builder.add_config(
            num_paths=len(reasoning_paths),
            max_steps=self.max_steps,
        )

        # Add results
        builder.add_results(
            selected_answer=result["best_answer"],
            consensus_score=result["consensus_score"],
            answer_distribution=result["answer_distribution"],
        )

        # Check if we have valid paths (generation might have failed)
        if reasoning_paths and "all_paths" in result and "all_scores" in result:
            # Create path summaries for detailed analysis
            best_idx = result["all_scores"].index(max(result["all_scores"]))
            path_summaries = builder.create_path_summaries(
                paths=result["all_paths"],
                scores=result["all_scores"],
                answers=result["all_answers"],
                selected_index=best_idx,
            )

            # Add generation details
            builder.add_generation_details(
                all_paths=result["all_paths"],
                all_scores=result["all_scores"],
                all_answers=result["all_answers"],
                path_summaries=path_summaries,
            )
        else:
            # No valid paths generated - log error
            log.error(
                f"Failed to generate any valid reasoning paths "
                f"({len(reasoning_paths)} successful out of {self.num_paths})"
            )

        # Log summary to console
        builder.log_summary(log)

        # Calculate average thinking/response steps
        avg_thinking_steps = sum(
            t.get("thinking_steps", 0) for t in result.get("all_traces", [])
        ) / max(len(result.get("all_traces", [])), 1)
        avg_response_steps = sum(
            t.get("response_steps", 0) for t in result.get("all_traces", [])
        ) / max(len(result.get("all_traces", [])), 1)

        # Format output to match expected interface
        return {
            "trajectory": result["best_path"],
            "steps": [result["best_path"]],  # Single step containing full reasoning
            "validity_scores": [result["consensus_score"]],  # Consensus as validity
            "completed": bool(reasoning_paths),
            "strategy": "self_consistency",
            "extracted_answer": result["best_answer"],
            "metadata": builder.build(),
            "all_traces": result.get("all_traces", []),
            "total_tokens": result.get("total_tokens", 0),
            "token_stats": token_stats,
            "thinking_num_steps": avg_thinking_steps,
            "response_num_steps": avg_response_steps,
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
