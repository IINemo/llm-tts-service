"""
Baseline strategy for single-shot generation.

Generates a complete response in one call without iterative step-by-step selection.
Stop tokens are configurable via eos_patterns parameter.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from llm_tts.generators.base import StepCandidate, convert_trajectory_to_string
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase, count_thinking_and_response_steps

log = logging.getLogger(__name__)


class StrategyBaseline(StrategyBase):
    """
    Baseline strategy - single-shot generation without iterative refinement.

    Simply calls the step generator once with <end of response> as the only stop token.
    No best-of-N selection, no iterative step generation.

    Supports two modes:
    - batch_generation=True: Fast batch generation using raw vLLM (no uncertainty scores)
    - batch_generation=False: Sequential generation with uncertainty wrapper (like online BoN)
    """

    def __init__(
        self,
        step_generator,
        output_dir: str = "./outputs",
        eos_patterns: List[str] = None,
        stop_token_ids: List[int] = None,
        batch_generation: bool = True,
        **kwargs,
    ):
        """
        Initialize baseline strategy.

        Args:
            step_generator: Generator to use for single-shot generation
            output_dir: Directory to save outputs
            eos_patterns: Stop tokens/patterns for generation (default: ["<end of response>"])
            stop_token_ids: Additional stop token IDs (e.g., [151645, 151643] for Qwen2)
            batch_generation: If True (default), use fast batch generation with raw vLLM.
                             If False, use sequential generation with uncertainty wrapper.
        """
        super().__init__()
        self.step_generator = step_generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eos_patterns = eos_patterns or ["<end of response>"]
        self.stop_token_ids = stop_token_ids
        self.batch_generation = batch_generation

        mode = (
            "batch (raw vLLM)" if batch_generation else "sequential (with uncertainty)"
        )
        log.info(f"Baseline strategy initialized in {mode} mode")

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate a complete response in a single call.

        Args:
            request: Chat messages (list of dicts with 'role' and 'content')
            sample_idx: Sample index for logging

        Returns:
            Dictionary with trajectory, extracted answer, and metadata
        """
        if self.batch_generation:
            return self._generate_trajectory_batch_mode(request, sample_idx)
        else:
            return self._generate_trajectory_with_uncertainty(request, sample_idx)

    def _generate_trajectory_with_uncertainty(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate response using uncertainty wrapper (like online BoN).

        Uses step_generator which goes through VLLMWithUncertainty for
        proper uncertainty scoring with the same stop tokens as batch mode.
        """
        log.info("Baseline strategy: generating with uncertainty wrapper")

        # Reset token tracking for this sample
        self.step_generator.reset_sample_stats()

        # Override step_generator's stop tokens to use baseline's eos_patterns
        # This ensures full response generation (not step-by-step)
        original_stop = self.step_generator.sampling_params.stop
        original_stop_token_ids = self.step_generator.sampling_params.stop_token_ids
        self.step_generator.sampling_params.stop = self.eos_patterns
        self.step_generator.sampling_params.stop_token_ids = self.stop_token_ids

        try:
            # Use step_generator to generate candidates (goes through VLLMWithUncertainty)
            # This computes uncertainty scores during generation
            candidates = self.step_generator(
                request,
                trajectory=[],
                candidates_per_step=1,
            )
        finally:
            # Restore original stop tokens
            self.step_generator.sampling_params.stop = original_stop
            self.step_generator.sampling_params.stop_token_ids = original_stop_token_ids

        if not candidates:
            log.error("No candidates generated")
            return {
                "trajectory": "",
                "extracted_answer": "",
                "steps": [],
                "thinking_num_steps": 0,
                "response_num_steps": 0,
                "validity_scores": [],
                "uncertainty_scores": [],
                "completed": False,
                "token_stats": {},
            }

        # Take the single generated candidate
        candidate = candidates[0]

        # Get uncertainty score from candidate
        uncertainty_score = candidate.other_data.get("uncertainty_score", 0.0)
        validity_score = candidate.other_data.get("validity_score", 1.0)

        # Build trajectory from the single candidate
        trajectory = [candidate]
        final_trajectory = convert_trajectory_to_string(trajectory)

        log.info(
            f"Generated response ({len(candidate.token_ids) if candidate.token_ids else 0} tokens)"
        )
        log.info(
            f"Uncertainty: {uncertainty_score:.4f}, Validity: {validity_score:.4f}"
        )
        log.info(f"Response:\n{final_trajectory}")

        # Extract answer from trajectory
        extracted = extract_answer(final_trajectory)

        # Get token statistics
        self.step_generator.finalize_sample_stats()
        token_stats = self.step_generator.get_sample_stats()

        log.info(
            f"Sample token stats: "
            f"total_tokens={token_stats['total_tokens_this_sample']:,}, "
            f"input_tokens={token_stats.get('input_tokens', 0):,}, "
            f"output_tokens={token_stats.get('output_tokens', 0):,}, "
            f"generations={token_stats['generation_count']}"
            + (
                f", tflops={token_stats['tflops']:.3f}"
                if token_stats.get("tflops")
                else ""
            )
        )

        # Count thinking and response steps
        thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
            trajectory
        )

        return {
            "trajectory": final_trajectory,
            "extracted_answer": extracted,
            "steps": trajectory,
            "thinking_num_steps": thinking_num_steps,
            "response_num_steps": response_num_steps,
            "validity_scores": [validity_score],
            "uncertainty_scores": [uncertainty_score],
            "completed": candidate.is_trajectory_complete,
            "token_stats": token_stats,
        }

    def _generate_trajectory_batch_mode(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate response using raw vLLM (fast batch mode, no uncertainty).

        Bypasses VLLMWithUncertainty wrapper for maximum speed.
        Matches official Qwen2.5-Math eval behavior.
        """
        log.info("Baseline strategy: generating with raw vLLM (batch mode)")

        from vllm import SamplingParams

        # Build prompt using step generator's chat template
        prompt = self.step_generator.tokenizer.apply_chat_template(
            request, tokenize=False, add_generation_prompt=True
        )

        # Create sampling params - match official Qwen3/Qwen2.5-Math eval exactly
        sampling_params = SamplingParams(
            n=1,
            max_tokens=self.step_generator.max_new_tokens,
            temperature=self.step_generator.temperature,
            top_p=self.step_generator.top_p,
            top_k=getattr(self.step_generator, "top_k", -1),
            presence_penalty=getattr(self.step_generator, "presence_penalty", 0.0),
            stop=self.eos_patterns,
            stop_token_ids=self.stop_token_ids,
        )

        log.info(
            f"Baseline: generating with stop={self.eos_patterns}, stop_token_ids={self.stop_token_ids}"
        )

        # Track context tokens
        context_tokens = len(self.step_generator.tokenizer.encode(prompt))

        # Get the raw vLLM LLM (bypass VLLMWithUncertainty wrapper entirely)
        raw_llm = getattr(self.step_generator.model, "llm", self.step_generator.model)

        # Generate directly using raw vLLM (no wrapper)
        outputs = raw_llm.generate([prompt], sampling_params)
        request_output = outputs[0]

        if not request_output.outputs:
            log.error("No output generated")
            return {
                "trajectory": "",
                "extracted_answer": "",
                "steps": [],
                "thinking_num_steps": 0,
                "response_num_steps": 0,
                "validity_scores": [],
                "completed": False,
                "token_stats": {},
            }

        output = request_output.outputs[0]
        raw_text = output.text
        stop_reason = getattr(output, "stop_reason", None)

        # Build full text - append stop token if generation stopped at one
        final_text = raw_text
        is_trajectory_complete = (
            stop_reason in self.eos_patterns if stop_reason else False
        )
        if is_trajectory_complete and stop_reason:
            final_text = final_text + stop_reason

        # Create StepCandidate
        candidate = StepCandidate(
            text=final_text,
            token_ids=output.token_ids,
            is_complete=True,
            is_trajectory_complete=is_trajectory_complete,
            other_data={
                "uncertainty_score": 0.0,
                "validity_score": 1.0,
            },
            raw_text=raw_text,
        )

        # Record generation stats
        self.step_generator._record_generation(
            [candidate],
            context_tokens=context_tokens,
        )

        # Build trajectory from the single candidate
        trajectory = [candidate]
        final_trajectory = convert_trajectory_to_string(trajectory)

        log.info(f"Generated response ({len(candidate.token_ids)} tokens)")
        log.info(f"Response preview: {final_trajectory[:200]}...")

        # Extract answer from trajectory
        extracted = extract_answer(final_trajectory)

        # Get token statistics
        self.step_generator.finalize_sample_stats()
        token_stats = self.step_generator.get_sample_stats()

        log.info(
            f"Sample token stats: "
            f"total_tokens={token_stats['total_tokens_this_sample']:,}, "
            f"input_tokens={token_stats.get('input_tokens', 0):,}, "
            f"output_tokens={token_stats.get('output_tokens', 0):,}, "
            f"generations={token_stats['generation_count']}"
            + (
                f", tflops={token_stats['tflops']:.3f}"
                if token_stats.get("tflops")
                else ""
            )
        )

        # Count thinking and response steps
        thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
            trajectory
        )

        return {
            "trajectory": final_trajectory,
            "extracted_answer": extracted,
            "steps": trajectory,
            "thinking_num_steps": thinking_num_steps,
            "response_num_steps": response_num_steps,
            "validity_scores": [1.0],  # No scoring in batch mode
            "completed": candidate.is_trajectory_complete,
            "token_stats": token_stats,
        }

    def generate_trajectories_batch(
        self, requests: List[List[Dict[str, str]]], sample_indices: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple requests.

        In batch_generation mode: Uses raw vLLM batch call (fast, no uncertainty).
        In sequential mode: Uses uncertainty wrapper for each request.

        Args:
            requests: List of chat messages (each is a list of dicts with 'role' and 'content')
            sample_indices: Optional list of sample indices for logging

        Returns:
            List of dictionaries with trajectory, extracted answer, and metadata for each request
        """
        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        if not self.batch_generation:
            # Sequential mode with uncertainty - process each request individually
            log.info(
                f"Baseline strategy: sequential generating {len(requests)} responses with uncertainty"
            )
            results = []
            for i, (request, sample_idx) in enumerate(zip(requests, sample_indices)):
                log.info(f"[{i+1}/{len(requests)}] Processing sample idx={sample_idx}")
                result = self._generate_trajectory_with_uncertainty(request, sample_idx)
                results.append(result)
            log.info(
                f"Baseline sequential: completed {len(results)} generations with uncertainty"
            )
            return results

        # Batch mode - use raw vLLM for maximum speed
        return self._generate_trajectories_batch_mode(requests, sample_indices)

    def _generate_trajectories_batch_mode(
        self, requests: List[List[Dict[str, str]]], sample_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple requests using raw vLLM batch call.

        Fast batch mode without uncertainty scoring.
        """
        from vllm import SamplingParams

        log.info(
            f"Baseline strategy: batch generating {len(requests)} responses (raw vLLM)"
        )

        # Build all prompts using step generator's chat template
        prompts = []
        for request in requests:
            if getattr(self.step_generator, "disable_thinking_mode", False):
                # Qwen3 models: explicitly disable thinking
                prompt = self.step_generator._apply_chat_template(
                    request, enable_thinking=False
                )
            else:
                # Qwen2.5-Math models: use default template without enable_thinking
                prompt = self.step_generator.tokenizer.apply_chat_template(
                    request, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)

        # Create sampling params - match official Qwen3/Qwen2.5-Math eval exactly
        sampling_params = SamplingParams(
            n=1,
            max_tokens=self.step_generator.max_new_tokens,
            temperature=self.step_generator.temperature,
            top_p=self.step_generator.top_p,
            top_k=getattr(self.step_generator, "top_k", -1),
            presence_penalty=getattr(self.step_generator, "presence_penalty", 0.0),
            stop=self.eos_patterns,
            stop_token_ids=self.stop_token_ids,
        )

        log.info(
            f"Baseline batch: generating with stop={self.eos_patterns}, stop_token_ids={self.stop_token_ids}"
        )

        # Get the raw vLLM LLM (bypass VLLMWithUncertainty wrapper entirely)
        raw_llm = getattr(self.step_generator.model, "llm", self.step_generator.model)

        # Generate all responses in a single call using raw vLLM
        outputs = raw_llm.generate(prompts, sampling_params)

        # Sort outputs by request_id to maintain order
        outputs = sorted(outputs, key=lambda x: int(x.request_id))

        # Process all outputs
        results = []
        for idx, (request_output, prompt, sample_idx) in enumerate(
            zip(outputs, prompts, sample_indices)
        ):
            context_tokens = len(self.step_generator.tokenizer.encode(prompt))

            if not request_output.outputs:
                log.error(f"No output generated for sample {sample_idx}")
                results.append(
                    {
                        "trajectory": "",
                        "extracted_answer": "",
                        "steps": [],
                        "thinking_num_steps": 0,
                        "response_num_steps": 0,
                        "validity_scores": [],
                        "completed": False,
                        "token_stats": {},
                    }
                )
                continue

            output = request_output.outputs[0]
            raw_text = output.text
            stop_reason = getattr(output, "stop_reason", None)

            # Build full text - append stop token if generation stopped at one
            final_text = raw_text
            is_trajectory_complete = (
                stop_reason in self.eos_patterns if stop_reason else False
            )
            if is_trajectory_complete and stop_reason:
                final_text = final_text + stop_reason

            # Create StepCandidate
            candidate = StepCandidate(
                text=final_text,
                token_ids=output.token_ids,
                is_complete=True,
                is_trajectory_complete=is_trajectory_complete,
                other_data={
                    "uncertainty_score": 0.0,
                    "validity_score": 1.0,
                },
                raw_text=raw_text,
            )

            # Build trajectory from the single candidate
            trajectory = [candidate]
            final_trajectory = convert_trajectory_to_string(trajectory)

            # Extract answer from trajectory
            extracted = extract_answer(final_trajectory)

            # Token stats for this sample
            output_tokens = len(output.token_ids)
            total_tokens = context_tokens + output_tokens
            token_stats = {
                "total_tokens_this_sample": total_tokens,
                "input_tokens": context_tokens,
                "output_tokens": output_tokens,
                "generation_count": 1,
            }

            # Add TFLOPs if calculator available
            if (
                hasattr(self.step_generator, "flop_calculator")
                and self.step_generator.flop_calculator
            ):
                tflops = self.step_generator.flop_calculator.compute_tflops(
                    total_tokens
                )
                token_stats["tflops"] = tflops

            # Count thinking and response steps
            thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
                trajectory
            )

            results.append(
                {
                    "trajectory": final_trajectory,
                    "extracted_answer": extracted,
                    "steps": trajectory,
                    "thinking_num_steps": thinking_num_steps,
                    "response_num_steps": response_num_steps,
                    "validity_scores": [1.0],  # No scoring in batch mode
                    "completed": candidate.is_trajectory_complete,
                    "token_stats": token_stats,
                }
            )

        log.info(f"Baseline batch: completed {len(results)} generations")
        return results

    def cleanup(self):
        """Cleanup resources."""
        pass
