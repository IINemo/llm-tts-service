"""
Baseline strategy for single-shot generation.

Generates a complete response in one call without iterative step-by-step selection.
Uses only <end of response> as stop token.
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
    """

    def __init__(
        self,
        step_generator,
        output_dir: str = "./outputs",
        **kwargs,
    ):
        """
        Initialize baseline strategy.

        Args:
            step_generator: Generator to use for single-shot generation
            output_dir: Directory to save outputs
        """
        super().__init__()
        self.step_generator = step_generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        log.info(f"Baseline strategy: generating single-shot response")

        # For baseline, we generate directly without step boundaries
        # Use the step generator's internal generation but with custom stop tokens
        from vllm import SamplingParams

        # Build prompt using step generator's chat template
        prompt = self.step_generator._apply_chat_template(request, enable_thinking=False)

        # Create sampling params with only <end of response> as stop token
        sampling_params = SamplingParams(
            n=1,
            max_tokens=self.step_generator.max_new_tokens,
            min_tokens=1,
            temperature=self.step_generator.temperature,
            top_p=self.step_generator.top_p,
            top_k=self.step_generator.top_k,
            logprobs=20,
            stop=["<end of response>"],  # Only stop at end of response
        )

        log.info(f"Baseline: generating with stop=['<end of response>']")

        # Track context tokens
        context_tokens = len(self.step_generator.tokenizer.encode(prompt))

        # Generate directly using vLLM
        outputs = self.step_generator.model.generate([prompt], sampling_params)
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

        # Build full text
        final_text = raw_text
        if stop_reason == "<end of response>":
            final_text = final_text + "<end of response>"

        # Create StepCandidate
        candidate = StepCandidate(
            text=final_text,
            token_ids=output.token_ids,
            is_complete=True,
            is_trajectory_complete=(stop_reason == "<end of response>"),
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
            + (f", tflops={token_stats['tflops']:.3f}" if token_stats.get("tflops") else "")
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
            "validity_scores": [1.0],  # No scoring in baseline
            "completed": candidate.is_trajectory_complete,
            "token_stats": token_stats,
        }

    def cleanup(self):
        """Cleanup resources."""
        pass
