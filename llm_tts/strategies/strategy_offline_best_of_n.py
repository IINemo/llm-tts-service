"""
Offline Best-of-N strategy for vLLM thinking mode.

Generates N complete trajectories, then selects the best one based on scoring.
Each trajectory includes:
1. Thinking phase: <think>...</think>
2. Response phase: <start of response>...<end of response>

Unlike online best-of-n which selects at each step, this generates full
trajectories first, then picks the best complete solution.

Uses ThinkingStepGeneratorVLLM with no intermediate stop tokens for unified
token tracking and FLOP calculation.
"""

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from vllm import LLM

from llm_tts.generators import StepCandidate
from llm_tts.generators.vllm.thinking import ThinkingStepGeneratorVLLM
from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector
from llm_tts.strategies.deepconf.utils import extract_answer

from .strategy_base import StrategyBase

if TYPE_CHECKING:
    from llm_tts.utils.flops import FLOPCalculator

log = logging.getLogger(__name__)


class StrategyOfflineBestOfN(StrategyBase):
    """
    Offline Best-of-N strategy for thinking mode.

    Generates N complete trajectories in batches, scores them,
    and returns the best one.
    """

    def __init__(
        self,
        model: LLM,
        scorer,
        num_trajectories: int = 4,
        max_thinking_tokens: int = 4096,
        max_response_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        answer_patterns: Optional[List[str]] = None,
        disable_thinking_mode: bool = False,
        output_dir: Optional[str] = None,
        # Step boundary detector settings (same as online mode)
        min_step_chars: int = 200,
        max_step_chars: int = 1200,
        use_sequence: bool = True,
        use_conclusion: bool = True,
        use_thinking: bool = True,
        use_verification: bool = True,
        use_reasoning: bool = False,
        use_correction: bool = False,
        use_structure: bool = False,
        # FLOP calculator for token tracking
        flop_calculator: Optional["FLOPCalculator"] = None,
    ):
        """
        Initialize offline best-of-n strategy.

        Args:
            model: vLLM model instance
            scorer: Scorer for ranking trajectories
            num_trajectories: Number of complete trajectories to generate
            max_thinking_tokens: Max tokens for thinking phase
            max_response_tokens: Max tokens for response phase
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            answer_patterns: Patterns that mark end of response (default: ["<end of response>"])
            disable_thinking_mode: If True, skip thinking phase
            output_dir: Directory for saving logs
            min_step_chars: Minimum characters per step (for detector)
            max_step_chars: Maximum characters per step (for detector)
            use_*: Marker categories for step boundary detection
            flop_calculator: Optional FLOP calculator for token tracking
        """
        self.model = model
        self.tokenizer = model.get_tokenizer()
        self.scorer = scorer
        self.num_trajectories = num_trajectories
        self.max_thinking_tokens = max_thinking_tokens
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.answer_patterns = (
            list(answer_patterns) if answer_patterns else ["<end of response>"]
        )
        self.disable_thinking_mode = disable_thinking_mode
        self.output_dir = output_dir
        self._current_sample_idx = 0

        # Create step generator with NO intermediate stop tokens
        # This generates full thinking in one shot (stops only at </think>)
        self.generator = ThinkingStepGeneratorVLLM(
            model=model,
            min_step_chars=1,  # No min - generate full thinking
            max_step_chars=999999,  # No max - generate full thinking
            max_new_tokens=max_thinking_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # Disable ALL intermediate stop tokens for full generation
            use_sequence=False,
            use_conclusion=False,
            use_thinking=False,
            use_verification=False,
            use_reasoning=False,
            use_correction=False,
            use_structure=False,
            custom_words=[],
            answer_patterns=self.answer_patterns,
            disable_thinking_mode=disable_thinking_mode,
            flop_calculator=flop_calculator,
        )

        # Create step boundary detector for splitting thinking into steps (post-hoc)
        self.detector = ThinkingMarkerDetector(
            min_step_chars=min_step_chars,
            max_step_chars=max_step_chars,
            use_sequence=use_sequence,
            use_conclusion=use_conclusion,
            use_thinking=use_thinking,
            use_verification=use_verification,
            use_reasoning=use_reasoning,
            use_correction=use_correction,
            use_structure=use_structure,
        )

        log.info(
            f"StrategyOfflineBestOfN initialized: "
            f"{num_trajectories} trajectories, "
            f"answer_patterns={self.answer_patterns}, "
            f"using step generator for token tracking"
        )

    def _split_thinking_into_steps(self, thinking_text: str) -> List[str]:
        """
        Split thinking content into steps using the marker detector.

        Args:
            thinking_text: The thinking text (may include <think>...</think> tags)

        Returns:
            List of step strings
        """
        # Extract raw content for comparison
        think_match = re.search(r"<think>(.*?)</think>", thinking_text, re.DOTALL)
        raw_content = (
            think_match.group(1).strip() if think_match else thinking_text.strip()
        )

        steps = self.detector.detect_steps(thinking_text)
        if not steps:
            # Return with tags if single step
            return [f"<think>{thinking_text}</think>"]

        # Verify no text is lost
        steps_combined = "\n".join(steps)
        original_len = len(raw_content)
        steps_len = len(steps_combined)

        if steps_len < original_len * 0.95:  # Allow 5% tolerance for whitespace
            log.warning(
                f"Possible text loss in step splitting: "
                f"original={original_len} chars, steps={steps_len} chars "
                f"({100*steps_len/original_len:.1f}%)"
            )

        # Add <think> to first step and </think> to last step
        if len(steps) > 0:
            steps[0] = f"<think>{steps[0]}"
            steps[-1] = f"{steps[-1]}</think>"

        return steps

    def _build_prompt(self, request: List[Dict[str, str]]) -> str:
        """Build prompt from request using tokenizer's chat template."""
        import inspect

        tokenizer_signature = inspect.signature(self.tokenizer.apply_chat_template)
        has_enable_thinking = "enable_thinking" in tokenizer_signature.parameters

        if has_enable_thinking:
            prompt = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(not self.disable_thinking_mode),
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
            )
            if self.disable_thinking_mode:
                prompt += "<think>\n\n</think>\n\n"

        return prompt

    def _generate_thinking_phase(
        self, request: List[Dict[str, str]], n: int
    ) -> List[Dict[str, any]]:
        """
        Generate N thinking phases in parallel using step generator.

        Uses generator.generate_full_thinking() for efficient batched generation.
        Stops at </think> token.

        Returns list of dicts with:
            - text: Generated thinking text (including </think>)
            - token_ids: Token IDs
            - logprobs: Log probabilities
            - generation_scores: Perplexity and entropy scores
        """
        # Use step generator's batched full thinking method
        candidates = self.generator.generate_full_thinking(
            request=request,
            num_candidates=n,
            max_tokens=self.max_thinking_tokens,
        )

        # Calculate context tokens for FLOP tracking
        prompt = self.generator._build_prompt(request, [])
        context_tokens = len(self.tokenizer.encode(prompt))

        # Record generation for FLOP tracking
        self.generator._record_generation(candidates, context_tokens=context_tokens)

        results = []
        for i, candidate in enumerate(candidates):
            num_tokens = len(candidate.token_ids) if candidate.token_ids else 0

            # Check if truncated at max_tokens
            if num_tokens >= self.max_thinking_tokens - 10:
                log.warning(
                    f"Thinking {i+1} likely truncated: {num_tokens} tokens "
                    f"(max={self.max_thinking_tokens})"
                )

            # Check for </think> token
            has_think_close = "</think>" in candidate.text
            if has_think_close:
                log.info(
                    f"Thinking {i+1}: {num_tokens} tokens (</think>=True, complete)"
                )
            else:
                log.info(
                    f"Thinking {i+1}: {num_tokens} tokens (</think>=False, appended)"
                )

            results.append(
                {
                    "text": candidate.text,
                    "token_ids": candidate.token_ids,
                    "logprobs": candidate.other_data.get("logprobs", []),
                    "generation_scores": candidate.generation_scores,
                }
            )

        return results

    def _generate_response_phase(
        self,
        request: List[Dict[str, str]],
        thinking_results: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """
        Generate response phase for each thinking trajectory using step generator.

        Uses step generator's generate_response for each trajectory.
        Responses are generated with proper token tracking.

        Args:
            request: Original request messages
            thinking_results: List of thinking phase results

        Returns:
            List of dicts with response text, token_ids, generation_scores
        """
        results = []
        total_context_tokens = 0
        all_candidates = []

        for i, thinking in enumerate(thinking_results):
            # Create trajectory from thinking text
            thinking_candidate = StepCandidate(
                text=thinking["text"],
                token_ids=thinking.get("token_ids", []),
                is_complete=True,
                is_trajectory_complete=True,
            )

            # Generate response using step generator
            response_candidates = self.generator.generate_response(
                request=request,
                trajectory=[thinking_candidate],
                candidates_per_step=1,
            )

            if not response_candidates:
                log.warning(f"Response {i+1}: No candidates generated")
                results.append(
                    {
                        "text": self.answer_patterns[0],
                        "token_ids": [],
                        "generation_scores": {},
                    }
                )
                continue

            candidate = response_candidates[0]
            all_candidates.append(candidate)

            # Calculate context tokens for this response
            prompt = self.generator._build_prompt(request, [thinking_candidate])
            total_context_tokens += len(self.tokenizer.encode(prompt))

            text = candidate.text
            # Ensure answer pattern is included
            has_pattern = any(p in text for p in self.answer_patterns)
            if not has_pattern:
                text = text + self.answer_patterns[0]

            results.append(
                {
                    "text": text,
                    "token_ids": candidate.token_ids,
                    "logprobs": candidate.other_data.get("logprobs", []),
                    "generation_scores": candidate.generation_scores,
                }
            )

        # Record generation for FLOP tracking
        if all_candidates:
            self.generator._record_generation(
                all_candidates, context_tokens=total_context_tokens
            )

        return results

    def get_token_stats(self) -> Dict[str, any]:
        """Get token statistics from the generator."""
        return self.generator.get_sample_stats()

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, any]:
        """
        Generate N complete trajectories and return the best one.

        Args:
            request: Chat messages for the request
            sample_idx: Index of current sample (for logging)

        Returns:
            Dictionary with:
                - trajectory: Best trajectory text
                - extracted_answer: Extracted answer from trajectory
                - steps: List containing the trajectory as single step
                - validity_scores: Scores for all trajectories
                - all_trajectories: All generated trajectories (for analysis)
                - completed: Whether generation completed successfully
        """
        self._current_sample_idx = sample_idx

        # Reset generator stats for this sample
        self.generator.reset_sample_stats()

        log.info(f"\n{'='*60}")
        log.info(f"Generating {self.num_trajectories} trajectories (offline best-of-n)")
        log.info(f"{'='*60}")

        # Phase 1: Generate N thinking phases using step generator
        log.info("\n--- Phase 1: Generating thinking ---")
        thinking_results = self._generate_thinking_phase(request, self.num_trajectories)
        log.info(f"Generated {len(thinking_results)} thinking phases")

        # Split each thinking phase into steps and log
        all_thinking_steps = []
        for i, thinking in enumerate(thinking_results):
            steps = self._split_thinking_into_steps(thinking["text"])
            all_thinking_steps.append(steps)
            log.info(f"\n[Thinking {i+1}] ({len(steps)} steps):")
            for j, step in enumerate(steps):
                log.info(f"\n  Step {j+1}: {step}")

        # Phase 2: Generate response for each thinking using step generator
        log.info("\n--- Phase 2: Generating responses ---")
        response_results = self._generate_response_phase(request, thinking_results)
        log.info(f"Generated {len(response_results)} responses")

        # Log each response
        for i, response in enumerate(response_results):
            log.info(f"\n[Response {i+1}]: {response['text']}")

        # Combine into complete trajectories
        all_trajectories = []
        all_scores = []
        all_token_counts = []

        # Score trajectories
        log.info("\n--- Phase 3: Scoring trajectories ---")
        for i, (thinking, response) in enumerate(
            zip(thinking_results, response_results)
        ):
            full_text = thinking["text"] + "\n\n" + response["text"]
            all_trajectories.append(full_text)

            # Get token counts
            thinking_tokens = len(thinking.get("token_ids", []))
            response_tokens = len(response.get("token_ids", []))
            total_tokens = thinking_tokens + response_tokens
            all_token_counts.append(total_tokens)

            # Use generation_scores from step generator if available
            thinking_scores = thinking.get("generation_scores", {})
            response_scores = response.get("generation_scores", {})

            # Average the entropy scores
            thinking_entropy = thinking_scores.get("mean_entropy", 0)
            response_entropy = response_scores.get("mean_entropy", 0)
            avg_entropy = (thinking_entropy + response_entropy) / 2

            # Create combined generation scores
            generation_scores = {
                "perplexity": (
                    thinking_scores.get("perplexity", 0)
                    + response_scores.get("perplexity", 0)
                )
                / 2,
                "mean_entropy": avg_entropy,
            }

            # Create a StepCandidate for scoring
            candidate = StepCandidate(
                text=full_text,
                token_ids=thinking.get("token_ids", []) + response.get("token_ids", []),
                is_complete=True,
                is_trajectory_complete=True,
                generation_scores=generation_scores,
                other_data={
                    "uncertainty_score": 1.0 / (1.0 + avg_entropy),
                },
            )

            # Use scorer to get validity/quality score
            scores = self.scorer.score_candidates(request, [candidate])
            score = scores[0] if scores else 0.0
            all_scores.append(score)

            log.info(
                f"[Trajectory {i+1}] Score: {score:.4f}, "
                f"tokens: {total_tokens} (thinking: {thinking_tokens}, response: {response_tokens})"
            )

        # Select best trajectory
        best_idx = int(
            np.argmax(all_scores)
        )  # Convert to Python int for JSON serialization
        best_trajectory = all_trajectories[best_idx]
        best_score = float(all_scores[best_idx])  # Convert to Python float

        log.info(f"\n{'='*60}")
        log.info(f"Selected trajectory {best_idx + 1} with score {best_score:.4f}")
        log.info(f"{'='*60}")

        # Extract answer
        extracted = extract_answer(best_trajectory)

        # Create StepCandidate objects preserving token_ids and logprobs
        best_thinking = thinking_results[best_idx]
        best_response = response_results[best_idx]

        # Build step candidates with full token_ids and logprobs
        # First: thinking phase (full, with token_ids)
        thinking_candidate = StepCandidate(
            text=best_thinking["text"],
            token_ids=best_thinking.get("token_ids", []),
            is_complete=True,
            is_trajectory_complete=False,
            generation_scores=best_thinking.get("generation_scores", {}),
            other_data={
                "logprobs": best_thinking.get("logprobs", []),
                "phase": "thinking",
            },
        )

        # Second: response phase (with token_ids)
        response_candidate = StepCandidate(
            text=best_response["text"],
            token_ids=best_response.get("token_ids", []),
            is_complete=True,
            is_trajectory_complete=True,
            generation_scores=best_response.get("generation_scores", {}),
            other_data={
                "logprobs": best_response.get("logprobs", []),
                "phase": "response",
                "validity": best_score,
            },
        )

        step_candidates = [thinking_candidate, response_candidate]

        # Also keep split steps for logging (text only)
        best_thinking_steps = all_thinking_steps[best_idx]

        # Log steps from best trajectory
        log.info(f"\n--- Best trajectory steps ({len(step_candidates)}) ---")
        for i, step in enumerate(step_candidates):
            log.info(f"\nStep {i+1}: {step.text}")

        # Log token stats from generator
        token_stats = self.generator.get_sample_stats()
        total_tokens = sum(all_token_counts)
        log.info(
            f"\nToken stats: {total_tokens} total tokens across {len(all_trajectories)} trajectories"
        )
        log.info(
            f"  Generator stats: input={token_stats.get('input_tokens', 0)}, "
            f"output={token_stats.get('output_tokens', 0)}, "
            f"generations={token_stats.get('generation_count', 0)}"
        )
        if token_stats.get("tflops"):
            log.info(f"  TFLOPs: {token_stats['tflops']:.3f}")

        # Save logs if output_dir provided
        if self.output_dir:
            self._save_trajectories_log(
                all_trajectories, all_scores, best_idx, all_thinking_steps
            )

        return {
            "trajectory": best_trajectory,
            "extracted_answer": extracted,
            "steps": step_candidates,
            "validity_scores": [best_score] * len(step_candidates),
            "all_trajectories": all_trajectories,
            "all_scores": all_scores,
            "all_token_counts": all_token_counts,
            "best_idx": best_idx,
            "completed": True,
            "token_stats": token_stats,
        }

    def _save_trajectories_log(
        self,
        trajectories: List[str],
        scores: List[float],
        best_idx: int,
        all_thinking_steps: Optional[List[List[str]]] = None,
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
            "num_trajectories": len(trajectories),
            "best_idx": best_idx,
            "best_score": scores[best_idx],
            "trajectories": [
                {
                    "idx": i,
                    "score": scores[i],
                    "text": traj,
                    "is_best": i == best_idx,
                    "num_steps": (
                        len(all_thinking_steps[i]) if all_thinking_steps else 1
                    ),
                    "steps": all_thinking_steps[i] if all_thinking_steps else [traj],
                }
                for i, traj in enumerate(trajectories)
            ],
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        log.info(f"Saved trajectories log to {log_path}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
