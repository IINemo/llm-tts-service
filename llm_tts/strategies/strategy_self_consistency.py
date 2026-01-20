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

from llm_tts.scorers.majority_voting import ChainMajorityVotingScorer
from llm_tts.utils import extract_answer

from .metadata_builder import StrategyMetadataBuilder
from .strategy_base import StrategyBase

if TYPE_CHECKING:
    from llm_tts.generators import StepCandidateGeneratorBase

log = logging.getLogger(__name__)


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
        log.info(f"Self-consistency strategy initialized: {num_paths} paths, {mode} mode")

    def _generate_paths_batch(
        self, request: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate all N trajectories in a SINGLE vLLM call.

        Uses generate_full_trajectories() which generates N complete trajectories
        with n=num_paths in one call. This is much faster than step-by-step.

        Args:
            request: Chat messages for the request

        Returns:
            List of path dictionaries with text, tokens, steps info
        """
        log.info(
            f"Generating {self.num_paths} trajectories in SINGLE vLLM call (batch mode)..."
        )

        # Single vLLM call generates all N trajectories
        # Skip step splitting - self-consistency only needs full text for majority voting
        raw_results = self.step_generator.generate_full_trajectories(
            request=request,
            num_trajectories=self.num_paths,
            split_steps=False,
        )

        # Convert to expected format
        paths = []
        total_tokens = 0

        for i, raw in enumerate(raw_results):
            num_tokens = len(raw.get("token_ids", []))
            total_tokens += num_tokens

            # Extract answer for logging
            answer = (
                extract_answer(raw["full_text"], answer_format="auto") or "no_answer"
            )

            log.info(
                f"  Path {i + 1}/{self.num_paths}: "
                f"tokens={num_tokens}, complete={raw['is_complete']}, answer={answer}"
            )

            paths.append(
                {
                    "text": raw["full_text"],
                    "num_tokens": num_tokens,
                    "steps": raw["steps"],
                    "is_complete": raw["is_complete"],
                    "thinking_steps": 0,
                    "response_steps": len(raw["steps"]),
                    "validity_scores": [],
                    "avg_validity": 0.0,
                }
            )

        log.info(f"Generated {self.num_paths} paths, total tokens: {total_tokens}")

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

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate N paths for each of M samples in a SINGLE vLLM call.

        This is the fully batched version - all M×N trajectories are generated
        in one vLLM call, then grouped by sample for majority voting.

        Args:
            requests: List of M chat message lists (each is a sample's request)
            sample_indices: Optional list of sample indices for logging

        Returns:
            List of M result dictionaries (one per sample)
        """
        from vllm import SamplingParams

        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        M = len(requests)
        N = self.num_paths

        log.info(
            f"Self-consistency batch: generating {M} samples × {N} paths = {M * N} "
            f"trajectories in SINGLE vLLM call"
        )

        # Build all prompts using step generator's chat template
        prompts = []
        for request in requests:
            if getattr(self.step_generator, "disable_thinking_mode", False):
                prompt = self.step_generator._apply_chat_template(
                    request, enable_thinking=False
                )
            else:
                prompt = self.step_generator.tokenizer.apply_chat_template(
                    request, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)

        # Create sampling params with n=num_paths for diversity
        sampling_params = SamplingParams(
            n=N,  # Generate N outputs per prompt
            max_tokens=self.step_generator.max_new_tokens,
            temperature=self.step_generator.temperature,
            top_p=self.step_generator.top_p,
            top_k=getattr(self.step_generator, "top_k", -1),
            stop=["<end of response>"],
            stop_token_ids=getattr(self.step_generator, "eos_token_ids", [151645, 151643]),
        )

        log.info(
            f"Self-consistency batch: temp={sampling_params.temperature}, "
            f"n={N}, max_tokens={sampling_params.max_tokens}"
        )

        # Get the raw vLLM LLM (bypass wrapper)
        raw_llm = getattr(self.step_generator.model, "llm", self.step_generator.model)

        # Single vLLM call for all M prompts × N paths
        outputs = raw_llm.generate(prompts, sampling_params)

        # Sort outputs by request_id to maintain order
        outputs = sorted(outputs, key=lambda x: int(x.request_id))

        # Process results - each output has N completions
        results = []
        for request_output, prompt, sample_idx in zip(
            outputs, prompts, sample_indices
        ):
            context_tokens = len(self.step_generator.tokenizer.encode(prompt))

            if not request_output.outputs:
                log.error(f"No output generated for sample {sample_idx}")
                results.append(self._empty_result())
                continue

            # Build paths from the N outputs
            paths = []
            total_output_tokens = 0

            for output in request_output.outputs:
                raw_text = output.text
                num_tokens = len(output.token_ids)
                total_output_tokens += num_tokens

                paths.append({
                    "text": raw_text,
                    "num_tokens": num_tokens,
                    "steps": [raw_text],
                    "is_complete": True,
                    "thinking_steps": 0,
                    "response_steps": 1,
                    "validity_scores": [],
                    "avg_validity": 0.0,
                })

            # Do majority voting for this sample
            result = self.select_best_answer(paths)

            # Token stats for this sample
            total_tokens = context_tokens * N + total_output_tokens  # N prompts worth of context
            token_stats = {
                "total_tokens_this_sample": total_tokens,
                "input_tokens": context_tokens * N,
                "output_tokens": total_output_tokens,
                "generation_count": N,
            }

            # Add TFLOPs if calculator available
            if (
                hasattr(self.step_generator, "flop_calculator")
                and self.step_generator.flop_calculator
            ):
                tflops = self.step_generator.flop_calculator.compute_tflops(total_tokens)
                token_stats["tflops"] = tflops

            # Build metadata
            builder = StrategyMetadataBuilder("self_consistency")
            builder.add_config(num_paths=N)
            builder.add_results(
                selected_answer=result["best_answer"],
                consensus_score=result["consensus_score"],
                answer_distribution=result["answer_distribution"],
            )

            # Calculate avg steps
            avg_thinking_steps = sum(
                t.get("thinking_steps", 0) for t in result.get("all_traces", [])
            ) / max(len(result.get("all_traces", [])), 1)
            avg_response_steps = sum(
                t.get("response_steps", 0) for t in result.get("all_traces", [])
            ) / max(len(result.get("all_traces", [])), 1)

            results.append({
                "trajectory": result["best_path"],
                "steps": [result["best_path"]],
                "validity_scores": [result["consensus_score"]],
                "completed": bool(paths),
                "strategy": "self_consistency",
                "extracted_answer": result["best_answer"],
                "metadata": builder.build(),
                "all_traces": result.get("all_traces", []),
                "total_tokens": result.get("total_tokens", 0),
                "token_stats": token_stats,
                "thinking_num_steps": avg_thinking_steps,
                "response_num_steps": avg_response_steps,
            })

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
            "thinking_num_steps": 0,
            "response_num_steps": 0,
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
