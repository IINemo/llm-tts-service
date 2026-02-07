"""
Self-consistency strategy for LLM reasoning.

Based on "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
by Wang et al. (2022). Generates multiple diverse reasoning paths and selects
the most consistent answer via majority voting.

Key feature: Generates ALL trajectories in a SINGLE vLLM call using n=num_paths,
then does majority voting on the final answers. This is much faster than
step-by-step generation.
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from llm_tts.generators.base import convert_trajectory_to_string
from llm_tts.scorers.majority_voting import ChainMajorityVotingScorer
from llm_tts.utils import extract_answer

from .metadata_builder import StrategyMetadataBuilder
from .strategy_base import StrategyBase, count_reasoning_steps

if TYPE_CHECKING:
    from llm_tts.generators import StepCandidateGeneratorBase

log = logging.getLogger(__name__)


def _get_answer_step_from_traces(all_traces: List[Dict]) -> Optional[str]:
    """Extract answer_step from the selected (best) trace."""
    if all_traces:
        for trace in all_traces:
            if trace.get("selected"):
                return trace.get("answer_step")
    return None


class StrategySelfConsistency(StrategyBase):
    """
    Self-consistency strategy that generates multiple reasoning paths
    and selects the most consistent answer via majority voting.

    Uses single-call batch generation for maximum efficiency - all N trajectories
    are generated in ONE vLLM call with n=num_paths parameter.
    """

    def __init__(
        self,
        step_generator: "StepCandidateGeneratorBase",
        num_paths: int = 10,
        scorer: Optional[Any] = None,
        batch_generation: bool = True,
        data_name: Optional[str] = None,
    ):
        """
        Initialize self-consistency strategy.

        Args:
            step_generator: Step generator (VLLMStepGenerator) for generation.
            num_paths: Number of reasoning paths to generate
            scorer: Custom scorer for answer selection (defaults to majority voting)
            batch_generation: If True (default), use fully batched generation for all samples.
                            If False, generate per-sample (M separate vLLM calls).
            data_name: Dataset name for official answer extraction (e.g., "minerva_math", "math500").
                      This ensures consistency between running accuracy and final evaluation.
        """
        self.step_generator = step_generator
        self.num_paths = num_paths
        self.batch_generation = batch_generation
        self.data_name = data_name

        # Use majority voting scorer by default, pass data_name for official extraction
        self.scorer = scorer or ChainMajorityVotingScorer(data_name=data_name)
        if hasattr(self.scorer, "prepare_model"):
            self.scorer.prepare_model()

        mode = "fully batched (single vLLM call)" if batch_generation else "per-sample"
        log.info(
            f"Self-consistency strategy initialized: {num_paths} paths, {mode} mode"
        )

    def _complete_thinking_paths(
        self,
        request: List[Dict[str, str]],
        candidates: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Complete thinking-mode candidates by generating answer phases.

        For each candidate that stopped at </think>, generates the answer phase
        via generate_answer_candidates, producing a proper two-step trajectory.

        Args:
            request: Chat messages for the request
            candidates: List of StepCandidate objects from generation

        Returns:
            List of path dictionaries with text, tokens, steps info
        """
        # Identify which candidates need answer generation
        thinking_indices = []
        for i, candidate in enumerate(candidates):
            if (
                getattr(self.step_generator, "thinking_mode", False)
                and "</think>" in candidate.text
                and not candidate.is_trajectory_complete
            ):
                thinking_indices.append(i)

        # Batch generate all answer phases in one call
        answer_map = {}  # candidate index -> answer_step
        if thinking_indices:
            log.info(
                f"Generating {len(thinking_indices)} answer phases in batched call"
            )
            batch_requests = [request] * len(thinking_indices)
            batch_trajectories = [[candidates[i]] for i in thinking_indices]
            answer_results = self.step_generator.generate_answer_candidates_batch(
                batch_requests,
                batch_trajectories,
                candidates_per_step=1,
            )
            for batch_idx, cand_idx in enumerate(thinking_indices):
                if answer_results[batch_idx]:
                    answer_map[cand_idx] = answer_results[batch_idx][0]

        # Build paths from candidates + answers
        paths = []
        for i, candidate in enumerate(candidates):
            text = candidate.raw_text or candidate.text
            num_tokens = candidate.other_data.get(
                "original_token_count", len(candidate.token_ids)
            )

            if i in answer_map:
                answer_step = answer_map[i]
                answer_step.is_trajectory_complete = True
                trajectory = [candidate, answer_step]
                full_text = convert_trajectory_to_string(trajectory)
                answer_tokens = (
                    len(answer_step.token_ids) if answer_step.token_ids else 0
                )
                num_tokens += answer_tokens
                reasoning_steps = count_reasoning_steps(
                    trajectory,
                    getattr(self.step_generator, "thinking_mode", False),
                )
                paths.append(
                    {
                        "text": full_text,
                        "num_tokens": num_tokens,
                        "steps": [candidate.text, answer_step.text],
                        "is_complete": True,
                        "reasoning_steps": reasoning_steps,
                        "validity_scores": [],
                        "avg_validity": 0.0,
                        "answer_step": answer_step.raw_text or answer_step.text,
                    }
                )
                continue

            # Non-thinking or no </think>: single-step path
            paths.append(
                {
                    "text": text,
                    "num_tokens": num_tokens,
                    "steps": [text],
                    "is_complete": candidate.is_trajectory_complete,
                    "reasoning_steps": count_reasoning_steps(
                        [candidate],
                        getattr(self.step_generator, "thinking_mode", False),
                    ),
                    "validity_scores": [],
                    "avg_validity": 0.0,
                    "answer_step": None,
                }
            )

        return paths

    def _generate_paths_batch(
        self, request: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate all N trajectories in a SINGLE vLLM call.

        Uses generate_step_candidates_batch with proper stop tokens (including
        </think> for thinking mode). For thinking-mode candidates that stop at
        </think>, generates the answer phase separately.

        Args:
            request: Chat messages for the request

        Returns:
            List of path dictionaries with text, tokens, steps info
        """
        log.info(
            f"Generating {self.num_paths} trajectories in SINGLE vLLM call (batch mode)..."
        )

        # Build stop tokens list, including </think> for thinking mode
        stop_tokens = ["<end of response>"]
        if (
            getattr(self.step_generator, "thinking_mode", False)
            and "</think>" not in stop_tokens
        ):
            stop_tokens.append("</think>")

        # Single vLLM call generates all N trajectories
        batch_results = self.step_generator.generate_step_candidates_batch(
            requests=[request],
            trajectories=[[]],
            candidates_per_step=self.num_paths,
            stop_tokens_override=stop_tokens,
            max_tokens_override=self.step_generator.max_new_tokens,
            compute_uncertainty=False,
            sample_ids=[0],
        )

        candidates = batch_results[0] if batch_results else []

        # Complete thinking paths (generate answer phase for </think> candidates)
        paths = self._complete_thinking_paths(request, candidates)

        # Log summary
        total_tokens = sum(p["num_tokens"] for p in paths)
        for i, path in enumerate(paths):
            answer = extract_answer(path["text"], answer_format="auto") or "no_answer"
            log.info(
                f"  Path {i + 1}/{self.num_paths}: "
                f"tokens={path['num_tokens']}, steps={len(path['steps'])}, "
                f"complete={path['is_complete']}, answer={answer}"
            )

        log.info(f"Generated {len(paths)} paths, total tokens: {total_tokens}")

        return paths

    def generate_reasoning_paths(
        self, request: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple reasoning paths using batch generation.

        Args:
            request: Chat messages in OpenAI format

        Returns:
            List of dicts with text, num_tokens, steps per path
        """
        # Reset stats for this sample
        self.step_generator.reset_sample_stats()

        # Use batch generation (single vLLM call)
        paths = self._generate_paths_batch(request)

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
        """
        if not reasoning_paths:
            return {
                "best_path": "",
                "best_steps": [],
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
        best_steps = reasoning_paths[best_idx].get("steps", [best_path])

        # Extract the best answer
        best_answer = self.scorer.extract_answer(best_path)

        # Get all answers for analysis
        all_answers = [self.scorer.extract_answer(path) for path in path_texts]

        # Calculate answer distribution
        answer_counts = Counter(all_answers)

        log.info(
            f"Selected reasoning path {best_idx + 1} with consensus score {best_score:.3f}"
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
                    "reasoning_steps": path_data.get("reasoning_steps", 0),
                    "avg_validity": path_data.get("avg_validity", 0),
                    "answer": answer,
                    "score": float(scores[i]),
                    "selected": i == best_idx,
                    "answer_step": path_data.get("answer_step"),  # Include answer_step in traces
                }
            )

        total_tokens = sum(path_tokens)
        log.info(f"Total tokens across all paths: {total_tokens}")

        return {
            "best_path": best_path,
            "best_steps": best_steps,
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

        # Generate multiple reasoning paths (single vLLM call)
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
        )

        # Add results
        builder.add_results(
            selected_answer=result["best_answer"],
            consensus_score=result["consensus_score"],
            answer_distribution=result["answer_distribution"],
        )

        # Check if we have valid paths
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
            log.error(
                f"Failed to generate any valid reasoning paths "
                f"({len(reasoning_paths)} successful out of {self.num_paths})"
            )

        # Log summary to console
        builder.log_summary(log)

        # Calculate average reasoning steps
        avg_reasoning_steps = sum(
            t.get("reasoning_steps", 0) for t in result.get("all_traces", [])
        ) / max(len(result.get("all_traces", [])), 1)

        # Format output to match expected interface
        return {
            "trajectory": result["best_path"],
            "steps": result["best_steps"],
            "validity_scores": [result["consensus_score"]],  # Consensus as validity
            "completed": bool(reasoning_paths),
            "strategy": "self_consistency",
            "extracted_answer": result["best_answer"],
            "metadata": builder.build(),
            "all_traces": result.get("all_traces", []),
            "total_tokens": result.get("total_tokens", 0),
            "token_stats": token_stats,
            "reasoning_steps": avg_reasoning_steps,
            "answer_step": _get_answer_step_from_traces(result.get("all_traces", [])),
        }

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate N paths for each of M samples using generate_step_candidates_batch.

        All M×N trajectories are generated in one call with proper FLOP tracking,
        then grouped by sample for majority voting.

        Args:
            requests: List of M chat message lists (each is a sample's request)
            sample_indices: Optional list of sample indices for logging

        Returns:
            List of M result dictionaries (one per sample)
        """
        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        M = len(requests)
        N = self.num_paths

        log.info(
            f"Self-consistency batch: generating {M} samples × {N} paths = {M * N} "
            f"trajectories via generate_step_candidates_batch"
        )

        # Build stop tokens list, including </think> for thinking mode
        stop_tokens = ["<end of response>"]
        if (
            getattr(self.step_generator, "thinking_mode", False)
            and "</think>" not in stop_tokens
        ):
            stop_tokens.append("</think>")

        # Reset per-sample tracking and generate all M×N trajectories
        self.step_generator.reset_per_sample_stats()
        batch_results = self.step_generator.generate_step_candidates_batch(
            requests=requests,
            trajectories=[[]] * M,
            candidates_per_step=N,
            stop_tokens_override=stop_tokens,
            max_tokens_override=self.step_generator.max_new_tokens,
            compute_uncertainty=False,
            sample_ids=list(range(M)),
        )

        # Process results - each entry has N candidates
        results = []
        for idx, (candidates, sample_idx) in enumerate(
            zip(batch_results, sample_indices)
        ):
            if not candidates:
                log.error(f"No output generated for sample {sample_idx}")
                results.append(self._empty_result())
                continue

            # Build paths from the N StepCandidates, completing thinking phases
            paths = self._complete_thinking_paths(requests[idx], candidates)

            # Do majority voting for this sample
            result = self.select_best_answer(paths)

            # Token stats from generator's per-sample tracking
            token_stats = self.step_generator.get_sample_stats_for(idx)
            token_stats["generation_count"] = N  # N candidates in one vLLM call

            # Build metadata
            builder = StrategyMetadataBuilder("self_consistency")
            builder.add_config(num_paths=N)
            builder.add_results(
                selected_answer=result["best_answer"],
                consensus_score=result["consensus_score"],
                answer_distribution=result["answer_distribution"],
            )

            # Calculate avg reasoning steps
            avg_reasoning_steps = sum(
                t.get("reasoning_steps", 0) for t in result.get("all_traces", [])
            ) / max(len(result.get("all_traces", [])), 1)

            results.append(
                {
                    "trajectory": result["best_path"],
                    "steps": result["best_steps"],
                    "validity_scores": [result["consensus_score"]],
                    "completed": bool(paths),
                    "strategy": "self_consistency",
                    "extracted_answer": result["best_answer"],
                    "metadata": builder.build(),
                    "all_traces": result.get("all_traces", []),
                    "total_tokens": result.get("total_tokens", 0),
                    "token_stats": token_stats,
                    "reasoning_steps": avg_reasoning_steps,
                    "answer_step": _get_answer_step_from_traces(
                        result.get("all_traces", [])
                    ),
                }
            )

        log.info(
            f"Self-consistency batch: completed {len(results)} samples, "
            f"total {M * N} trajectories generated"
        )
        return results

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for failed generation."""
        return {
            "trajectory": "",
            "steps": [],
            "validity_scores": [],
            "completed": False,
            "strategy": "self_consistency",
            "extracted_answer": "",
            "metadata": {},
            "all_traces": [],
            "total_tokens": 0,
            "token_stats": {},
            "reasoning_steps": 0,
            "answer_step": None,
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
