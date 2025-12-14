"""
Offline Best-of-N strategy for vLLM thinking mode.

Generates N complete trajectories, then selects the best one based on scoring.
Each trajectory includes:
1. Thinking phase: <think>...</think>
2. Response phase: <start of response>...<end of response>

Unlike online best-of-n which selects at each step, this generates full
trajectories first, then picks the best complete solution.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
from vllm import LLM, SamplingParams

from llm_tts.generators import StepCandidate, convert_trajectory_to_string
from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector
from llm_tts.strategies.deepconf.utils import extract_answer

from .strategy_base import StrategyBase

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

        # Create step boundary detector for splitting thinking into steps
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
            f"answer_patterns={self.answer_patterns}"
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

    def _generate_thinking_phase(self, prompt: str, n: int) -> List[Dict[str, any]]:
        """
        Generate N thinking phases in parallel.

        Returns list of dicts with:
            - text: Generated thinking text (including </think>)
            - token_ids: Token IDs
            - logprobs: Log probabilities
        """
        sampling_params = SamplingParams(
            n=n,
            max_tokens=self.max_thinking_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            logprobs=20,
            stop=["</think>"],
            include_stop_str_in_output=True,  # Include </think> in output
        )

        outputs = self.model.generate([prompt], sampling_params)
        results = []

        for i, output in enumerate(outputs[0].outputs):
            text = output.text
            num_tokens = len(output.token_ids) if output.token_ids else 0

            # Check for </think> token (note: <think> is in prompt, not generated)
            has_think_close = "</think>" in text

            # Check if truncated at max_tokens
            if num_tokens >= self.max_thinking_tokens - 10:  # Allow small margin
                log.warning(
                    f"Thinking {i+1} likely truncated: {num_tokens} tokens "
                    f"(max={self.max_thinking_tokens})"
                )

            # Ensure </think> is included (should be present with include_stop_str_in_output=True)
            if not has_think_close:
                text = text + "</think>"
                log.info(
                    f"Thinking {i+1}: {num_tokens} tokens (</think>=False, appended)"
                )
            else:
                log.info(
                    f"Thinking {i+1}: {num_tokens} tokens (</think>=True, complete)"
                )

            results.append(
                {
                    "text": text,
                    "token_ids": output.token_ids,
                    "logprobs": output.logprobs,
                }
            )

        return results

    def _generate_response_phase(self, prompts: List[str]) -> List[Dict[str, any]]:
        """
        Generate response phase for each thinking trajectory.

        Args:
            prompts: List of prompts (base prompt + thinking text for each)

        Returns:
            List of dicts with response text, token_ids, logprobs
        """
        sampling_params = SamplingParams(
            n=1,
            max_tokens=self.max_response_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            logprobs=20,
            stop=self.answer_patterns,
        )

        outputs = self.model.generate(prompts, sampling_params)
        results = []

        for output in outputs:
            text = output.outputs[0].text
            # Ensure answer pattern is included
            has_pattern = any(p in text for p in self.answer_patterns)
            if not has_pattern:
                text = text + self.answer_patterns[0]

            results.append(
                {
                    "text": text,
                    "token_ids": output.outputs[0].token_ids,
                    "logprobs": output.outputs[0].logprobs,
                }
            )

        return results

    def _calculate_perplexity(self, logprobs: List, token_ids: List) -> float:
        """Calculate perplexity from logprobs."""
        if not logprobs or not token_ids:
            return 0.0

        total = 0.0
        for token, logprob_dict in zip(token_ids, logprobs):
            if logprob_dict and token in logprob_dict:
                total += logprob_dict[token].logprob

        return -1.0 * total / max(len(token_ids), 1)

    def _calculate_mean_entropy(self, logprobs: List) -> float:
        """Calculate mean token entropy from logprobs."""
        if not logprobs:
            return 0.0

        total = 0.0
        for logprob_dict in logprobs:
            if logprob_dict:
                for v in logprob_dict.values():
                    prob = np.exp(v.logprob)
                    total += v.logprob * prob

        return -1.0 * total / max(len(logprobs), 1)

    def _calculate_trajectory_score(
        self,
        trajectory_text: str,
        request: List[Dict[str, str]],
        thinking_logprobs: List = None,
        thinking_token_ids: List = None,
        response_logprobs: List = None,
        response_token_ids: List = None,
    ) -> float:
        """Calculate score for a complete trajectory."""
        # Combine logprobs from thinking and response
        all_logprobs = (thinking_logprobs or []) + (response_logprobs or [])
        all_token_ids = (thinking_token_ids or []) + (response_token_ids or [])

        # Compute generation scores
        generation_scores = {
            "perplexity": self._calculate_perplexity(all_logprobs, all_token_ids),
            "mean_entropy": self._calculate_mean_entropy(all_logprobs),
        }

        # Create a StepCandidate for scoring
        candidate = StepCandidate(
            text=trajectory_text,
            token_ids=all_token_ids,
            is_complete=True,
            is_trajectory_complete=True,
            generation_scores=generation_scores,
        )

        # Use scorer to get validity/quality score
        scores = self.scorer.score_candidates(request, [candidate])
        return scores[0] if scores else 0.0

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
        base_prompt = self._build_prompt(request)

        log.info(f"\n{'='*60}")
        log.info(f"Generating {self.num_trajectories} trajectories (offline best-of-n)")
        log.info(f"{'='*60}")

        # Phase 1: Generate N thinking phases
        log.info("\n--- Phase 1: Generating thinking ---")
        thinking_results = self._generate_thinking_phase(
            base_prompt, self.num_trajectories
        )
        log.info(f"Generated {len(thinking_results)} thinking phases")

        # Split each thinking phase into steps and log
        all_thinking_steps = []
        for i, thinking in enumerate(thinking_results):
            steps = self._split_thinking_into_steps(thinking["text"])
            all_thinking_steps.append(steps)
            log.info(f"\n[Thinking {i+1}] ({len(steps)} steps):")
            for j, step in enumerate(steps):
                log.info(f"\n  Step {j+1}: {step}")

        # Phase 2: Generate response for each thinking
        log.info("\n--- Phase 2: Generating responses ---")
        response_prompts = [
            base_prompt + result["text"] + "\n\n" for result in thinking_results
        ]
        response_results = self._generate_response_phase(response_prompts)
        log.info(f"Generated {len(response_results)} responses")

        # Log each response
        for i, response in enumerate(response_results):
            log.info(f"\n[Response {i+1}]: {response['text']}")

        # Combine into complete trajectories
        all_trajectories = []
        all_scores = []

        # Score trajectories
        log.info("\n--- Phase 3: Scoring trajectories ---")
        for i, (thinking, response) in enumerate(
            zip(thinking_results, response_results)
        ):
            full_text = thinking["text"] + "\n\n" + response["text"]
            all_trajectories.append(full_text)

            # Score trajectory (with logprobs from generation)
            score = self._calculate_trajectory_score(
                full_text,
                request,
                thinking_logprobs=thinking.get("logprobs"),
                thinking_token_ids=thinking.get("token_ids"),
                response_logprobs=response.get("logprobs"),
                response_token_ids=response.get("token_ids"),
            )
            all_scores.append(score)

            log.info(f"[Trajectory {i+1}] Score: {score:.4f}")

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

        # Create StepCandidate objects for each step in the best trajectory
        best_thinking_steps = all_thinking_steps[best_idx]
        best_response = response_results[best_idx]["text"]

        # Build step candidates list
        step_candidates = []
        for step_text in best_thinking_steps:
            step_candidates.append(
                StepCandidate(
                    text=step_text,
                    token_ids=[],
                    is_complete=True,
                    is_trajectory_complete=False,
                )
            )

        # Add response as final step
        step_candidates.append(
            StepCandidate(
                text=best_response,
                token_ids=[],
                is_complete=True,
                is_trajectory_complete=True,
                generation_scores={"validity": best_score},
            )
        )

        # Log steps from best trajectory
        log.info(f"\n--- Best trajectory steps ({len(step_candidates)}) ---")
        for i, step in enumerate(step_candidates):
            log.info(f"\nStep {i+1}: {step.text}")

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
            "best_idx": best_idx,
            "completed": True,
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
