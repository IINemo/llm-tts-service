"""
Baseline strategy for single-shot generation.

Generates a complete response in one call without iterative step-by-step selection.
Stop tokens are configurable via eos_patterns parameter.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from llm_tts.generators.base import convert_trajectory_to_string
from llm_tts.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase, count_reasoning_steps

log = logging.getLogger(__name__)


class StrategyBaseline(StrategyBase):
    """
    Baseline strategy - single-shot generation without iterative refinement.

    Simply calls the step generator once with <end of response> as the only stop token.
    No best-of-N selection, no iterative step generation.

    Supports two modes:
    - batch_generation=True: Fast batch generation using generate_step_candidates_batch
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
            batch_generation: If True (default), use fast batch generation.
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

    def _generate_trajectory_with_uncertainty(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate response using uncertainty wrapper (like online BoN).

        Uses step_generator which goes through VLLMWithUncertainty for
        proper uncertainty scoring. Stop tokens are configured via config,
        not overridden at runtime.
        """
        log.info("Baseline strategy: generating with uncertainty wrapper")

        # Reset token tracking for this sample
        self.step_generator.reset_sample_stats()

        # Use step_generator to generate candidates (goes through VLLMWithUncertainty)
        # Stop tokens are already configured from config (no runtime override needed)
        candidates = self.step_generator(
            request,
            trajectory=[],
            candidates_per_step=1,
        )

        if not candidates:
            log.error("No candidates generated")
            return {
                "trajectory": "",
                "extracted_answer": "",
                "steps": [],
                "answer_step": None,
                "reasoning_steps": 0,
                "validity_scores": [],
                "uncertainty_scores": [],
                "completed": False,
                "token_stats": {},
            }

        # Take the single generated candidate
        candidate = candidates[0]

        # Thinking mode: step 1 produces <think>...</think>, step 2 generates final answer
        if (
            getattr(self.step_generator, "thinking_mode", False)
            and candidate.is_thinking_complete
            and not candidate.is_trajectory_complete
        ):
            log.info("Thinking phase complete, generating final answer (step 2)")
            thinking_step = candidate
            answer_candidates = self.step_generator.generate_answer_candidates(
                request,
                [thinking_step],
                candidates_per_step=1,
            )
            if answer_candidates:
                answer_step = answer_candidates[0]
                answer_step.is_trajectory_complete = True
                # Return both steps in trajectory
                trajectory = [thinking_step, answer_step]
                final_trajectory = convert_trajectory_to_string(trajectory)
                extracted = extract_answer(final_trajectory)
                # Store answer text separately for logging
                answer_text = answer_step.raw_text or answer_step.text

                self.step_generator.finalize_sample_stats()
                token_stats = self.step_generator.get_sample_stats()
                reasoning_steps = count_reasoning_steps(
                    trajectory,
                    getattr(self.step_generator, "thinking_mode", False),
                )

                log.info(f"Response:\n{final_trajectory}")
                return {
                    "trajectory": final_trajectory,
                    "extracted_answer": extracted,
                    "steps": trajectory,
                    "answer_step": answer_text,  # Final answer text (thinking mode only)
                    "reasoning_steps": reasoning_steps,
                    "validity_scores": [
                        candidate.other_data.get("validity_score", 1.0)
                    ],
                    "uncertainty_scores": [
                        candidate.other_data.get("uncertainty_score", 0.0)
                    ],
                    "completed": True,
                    "token_stats": token_stats,
                }

        # Get uncertainty score from candidate
        uncertainty_score = candidate.other_data.get("uncertainty_score")
        if uncertainty_score is None:
            log.warning(
                f"Sample {sample_idx}: missing 'uncertainty_score' in candidate other_data"
            )
            uncertainty_score = 0.0
        validity_score = candidate.other_data.get("validity_score")
        if validity_score is None:
            log.warning(
                f"Sample {sample_idx}: missing 'validity_score' in candidate other_data"
            )
            validity_score = 1.0

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

        # Count reasoning steps
        reasoning_steps = count_reasoning_steps(
            trajectory,
            getattr(self.step_generator, "thinking_mode", False),
        )

        return {
            "trajectory": final_trajectory,
            "extracted_answer": extracted,
            "steps": trajectory,
            "answer_step": None,
            "reasoning_steps": reasoning_steps,
            "validity_scores": [validity_score],
            "uncertainty_scores": [uncertainty_score],
            "completed": candidate.is_trajectory_complete,
            "token_stats": token_stats,
        }

    def generate_trajectories_batch(
        self, requests: List[List[Dict[str, str]]], sample_indices: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple requests.

        In batch_generation mode: Uses generate_step_candidates_batch (proper FLOP tracking).
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

        # Batch mode
        return self._generate_trajectories_batch_mode(requests, sample_indices)

    def _generate_trajectories_batch_mode(
        self, requests: List[List[Dict[str, str]]], sample_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple requests using generate_step_candidates_batch.

        Uses the generator's batched method for proper FLOP tracking.
        """
        M = len(requests)

        log.info(
            f"Baseline strategy: batch generating {M} responses "
            f"via generate_step_candidates_batch"
        )

        # Reset per-sample tracking and generate all responses
        self.step_generator.reset_per_sample_stats()
        stop_tokens = list(self.eos_patterns)
        if (
            getattr(self.step_generator, "thinking_mode", False)
            and "</think>" not in stop_tokens
        ):
            stop_tokens.append("</think>")
        batch_results = self.step_generator.generate_step_candidates_batch(
            requests=requests,
            trajectories=[[]] * M,
            candidates_per_step=1,
            stop_tokens_override=stop_tokens,
            max_tokens=self.step_generator.generation_limit,
            compute_uncertainty=False,
            sample_ids=list(range(M)),
        )

        # Identify thinking-mode candidates that need answer generation
        thinking_indices = []  # indices into batch_results
        for idx, candidates in enumerate(batch_results):
            if candidates:
                candidate = candidates[0]
                if (
                    getattr(self.step_generator, "thinking_mode", False)
                    and candidate.is_thinking_complete
                    and not candidate.is_trajectory_complete
                ):
                    thinking_indices.append(idx)

        # Batch generate all answer phases in one call
        answer_map = {}  # idx -> answer_step
        if thinking_indices:
            log.info(
                f"Generating {len(thinking_indices)} answer phases in batched call"
            )
            batch_answer_reqs = [requests[i] for i in thinking_indices]
            batch_answer_trajs = [[batch_results[i][0]] for i in thinking_indices]
            answer_results = self.step_generator.generate_answer_candidates_batch(
                batch_answer_reqs,
                batch_answer_trajs,
                candidates_per_step=1,
            )
            for batch_idx, orig_idx in enumerate(thinking_indices):
                if answer_results[batch_idx]:
                    answer_map[orig_idx] = answer_results[batch_idx][0]

        # Process StepCandidates into result dicts
        results = []
        for idx, (candidates, sample_idx) in enumerate(
            zip(batch_results, sample_indices)
        ):
            if not candidates:
                log.error(f"No output generated for sample {sample_idx}")
                results.append(
                    {
                        "trajectory": "",
                        "extracted_answer": "",
                        "steps": [],
                        "answer_step": None,
                        "reasoning_steps": 0,
                        "validity_scores": [],
                        "completed": False,
                        "token_stats": {},
                    }
                )
                continue

            candidate = candidates[0]  # candidates_per_step=1

            if idx in answer_map:
                answer_step = answer_map[idx]
                answer_step.is_trajectory_complete = True
                trajectory = [candidate, answer_step]
                answer_text = answer_step.raw_text or answer_step.text
            else:
                trajectory = [candidate]
                answer_text = None
            final_trajectory = convert_trajectory_to_string(trajectory)

            # Extract answer from trajectory
            extracted = extract_answer(final_trajectory)

            # Token stats from generator's per-sample tracking
            token_stats = self.step_generator.get_sample_stats_for(idx)

            # Count reasoning steps
            reasoning_steps = count_reasoning_steps(
                trajectory,
                getattr(self.step_generator, "thinking_mode", False),
            )

            results.append(
                {
                    "trajectory": final_trajectory,
                    "extracted_answer": extracted,
                    "steps": trajectory,
                    "answer_step": answer_text,
                    "reasoning_steps": reasoning_steps,
                    "validity_scores": [
                        self._get_validity_score(candidate, sample_idx)
                    ],
                    "completed": trajectory[-1].is_trajectory_complete,
                    "token_stats": token_stats,
                }
            )

        log.info(f"Baseline batch: completed {len(results)} generations")
        return results

    @staticmethod
    def _get_validity_score(candidate, sample_idx: int) -> float:
        data = candidate.other_data if candidate.other_data else {}
        score = data.get("validity_score")
        if score is None:
            log.warning(
                f"Sample {sample_idx}: missing 'validity_score' in candidate other_data"
            )
            return 0.0
        return score

    def cleanup(self):
        """Cleanup resources."""
        pass
