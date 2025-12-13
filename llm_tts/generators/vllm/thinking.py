"""
vLLM step candidate generator for thinking mode.

Two-phase generation:
1. generate_thinking_step() - generates thinking content inside <think>...</think>
   - Uses semantic stop tokens for step boundaries
   - Stops at </think> to end thinking phase
2. generate_response() - generates response after thinking
   - Generates <start of response>...<end of response>
   - Stops at <end of response>
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput

from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    covert_trajectory_to_string,
)
from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector
from llm_tts.step_boundary_detectors.thinking.vllm import (
    ANSWER_TOKENS,
    get_stop_tokens_compact,
)

log = logging.getLogger(__name__)


class ThinkingStepGeneratorVLLM(StepCandidateGeneratorBase):
    """
    vLLM step generator for thinking mode with two-phase generation.

    Phase 1 (Thinking): Generates content inside <think>...</think>
    - Uses semantic stop tokens for step boundaries
    - Stops when </think> is encountered

    Phase 2 (Response): Generates <start of response>...<end of response>
    - Called after thinking phase completes
    - Stops at <end of response>
    """

    def __init__(
        self,
        model: LLM,
        min_step_chars: int = 200,
        max_step_chars: int = 1200,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        # Stop token configuration (matches ThinkingMarkerDetector)
        use_sequence: bool = True,
        use_conclusion: bool = True,
        use_thinking: bool = True,
        use_verification: bool = True,
        use_reasoning: bool = False,
        use_correction: bool = False,
        use_structure: bool = False,
        custom_words: Optional[List[str]] = None,
        # Answer patterns for response completion
        answer_patterns: Optional[List[str]] = None,
        # Thinking mode control
        disable_thinking_mode: bool = False,
    ):
        self.model = model
        self.tokenizer = model.get_tokenizer()
        self.disable_thinking_mode = disable_thinking_mode
        self.min_step_chars = min_step_chars
        self.max_step_chars = max_step_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # Answer patterns for response phase (default: <end of response>)
        # Convert to regular list in case it's OmegaConf ListConfig
        self.answer_patterns = list(answer_patterns) if answer_patterns else ["<end of response>"]

        # Build stop tokens for THINKING phase (step boundaries + </think>)
        self.thinking_stop_tokens = get_stop_tokens_compact(
            use_sequence=use_sequence,
            use_conclusion=use_conclusion,
            use_thinking=use_thinking,
            use_verification=use_verification,
            use_reasoning=use_reasoning,
            use_correction=use_correction,
            use_structure=use_structure,
            custom_words=custom_words,
        )
        # Add </think> to stop thinking phase
        self.thinking_stop_tokens = list(set(self.thinking_stop_tokens + ["</think>"]))

        # Stop tokens for RESPONSE phase (answer patterns only)
        self.response_stop_tokens = self.answer_patterns.copy()

        # Create detector for validation logic
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
        self.detector.answer_patterns = self.answer_patterns

        log.info(
            f"ThinkingStepGeneratorVLLM initialized: "
            f"{len(self.thinking_stop_tokens)} thinking stops, "
            f"{len(self.response_stop_tokens)} response stops"
        )

    def _create_sampling_params(
        self,
        stop_tokens: List[str],
        n: int = 1,
        max_tokens: Optional[int] = None,
        min_tokens: int = 0,
    ) -> SamplingParams:
        """Create SamplingParams with specified stop tokens."""
        return SamplingParams(
            n=n,
            max_tokens=max_tokens or self.max_new_tokens,
            min_tokens=min_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            logprobs=20,
            stop=stop_tokens,
        )

    def _calculate_perplexity(self, candidate: CompletionOutput) -> float:
        """Calculate perplexity of the response."""
        if not candidate.logprobs or not candidate.token_ids:
            return 0.0

        total = 0.0
        for token, logprob_dict in zip(candidate.token_ids, candidate.logprobs):
            if token in logprob_dict:
                total += logprob_dict[token].logprob

        return -1.0 * total / max(len(candidate.token_ids), 1)

    def _calculate_mean_entropy(self, candidate: CompletionOutput) -> float:
        """Calculate mean token entropy of the response."""
        if not candidate.logprobs:
            return 0.0

        total = 0.0
        for logprob_dict in candidate.logprobs:
            for v in logprob_dict.values():
                prob = np.exp(v.logprob)
                total += v.logprob * prob

        return -1.0 * total / max(len(candidate.logprobs), 1)

    def _is_valid_step_boundary(self, text: str) -> bool:
        """
        Check if text represents a valid step boundary.

        Validates:
        - min_step_chars requirement
        - Ends at sentence boundary (. ! ? or newline)
        """
        text = text.strip()

        # Check minimum length
        if len(text) < self.min_step_chars:
            return False

        # Check sentence boundary
        if text and text[-1] in ".!?\n":
            return True

        return False

    def _is_thinking_complete(self, text: str) -> bool:
        """Check if thinking phase is complete (contains </think>)."""
        return "</think>" in text

    def _build_prompt(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> str:
        """Build prompt from request and trajectory using tokenizer's chat template."""
        import inspect

        # Check if tokenizer supports enable_thinking parameter
        tokenizer_signature = inspect.signature(self.tokenizer.apply_chat_template)
        has_enable_thinking = "enable_thinking" in tokenizer_signature.parameters

        # For FIRST generation (no trajectory): use apply_chat_template normally
        if not trajectory:
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
                # Fallback: add empty think block to skip thinking if disabled
                if self.disable_thinking_mode:
                    prompt += "<think>\n\n</think>\n\n"
            return prompt

        # For CONTINUATION (has trajectory): build base prompt then append trajectory
        if has_enable_thinking:
            base_prompt = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(not self.disable_thinking_mode),
            )
        else:
            base_prompt = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
            )
            if self.disable_thinking_mode:
                base_prompt += "<think>\n\n</think>\n\n"

        # Append trajectory directly (model continues from here)
        trajectory_text = covert_trajectory_to_string(trajectory)
        return base_prompt + trajectory_text

    def generate_thinking_step(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """
        Generate thinking step candidates (inside <think> block).

        Stops at:
        - Semantic step boundaries (e.g., "\\nSo ", "\\nLet me ")
        - </think> tag (end of thinking phase)

        Returns candidates with is_trajectory_complete=True if </think> was reached.
        """
        trajectory = trajectory or []
        candidates = []

        for i in range(candidates_per_step):
            candidate = self._generate_single_thinking_step(request, trajectory)
            candidates.append(candidate)

        return candidates

    def _generate_single_thinking_step(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> StepCandidate:
        """
        Generate a single thinking step with boundary validation.

        Continues generation if initial stop is not a valid boundary.
        Marks is_trajectory_complete=True when </think> is reached.
        """
        prompt = self._build_prompt(request, trajectory)
        accumulated_text = ""
        accumulated_tokens = []
        accumulated_logprobs = []
        max_continuation_attempts = 5

        # Minimum tokens before stop tokens apply
        min_tokens = max(1, self.min_step_chars // 4)

        for attempt in range(max_continuation_attempts):
            remaining_tokens = self.max_new_tokens - len(accumulated_tokens)
            if remaining_tokens <= 0:
                break

            sampling_params = self._create_sampling_params(
                stop_tokens=self.thinking_stop_tokens,
                n=1,
                max_tokens=remaining_tokens,
                min_tokens=min_tokens if attempt == 0 else 0,
            )

            current_prompt = prompt + accumulated_text
            outputs = self.model.generate([current_prompt], sampling_params)
            output = outputs[0].outputs[0]

            accumulated_text += output.text
            accumulated_tokens.extend(output.token_ids)
            if output.logprobs:
                accumulated_logprobs.extend(output.logprobs)

            # Check if thinking phase is complete
            if self._is_thinking_complete(accumulated_text):
                log.debug(f"Thinking complete (</think>) after {attempt + 1} attempts")
                # Truncate at </think>
                think_pos = accumulated_text.find("</think>")
                accumulated_text = accumulated_text[: think_pos + len("</think>")]
                break

            # Validate step boundary
            if self._is_valid_step_boundary(accumulated_text):
                log.debug(f"Valid thinking step boundary after {attempt + 1} attempts")
                break

            # Check max step chars
            if len(accumulated_text.strip()) >= self.max_step_chars:
                log.debug(f"Max step chars reached after {attempt + 1} attempts")
                break

            log.debug(
                f"Attempt {attempt + 1}: Invalid boundary "
                f"(len={len(accumulated_text)}, min={self.min_step_chars}), continuing..."
            )

        # Check if thinking is complete
        thinking_complete = self._is_thinking_complete(accumulated_text)

        # Calculate scores
        generation_scores = {}
        if accumulated_logprobs and accumulated_tokens:

            class MockOutput:
                def __init__(self, tokens, logprobs):
                    self.token_ids = tokens
                    self.logprobs = logprobs

            mock = MockOutput(accumulated_tokens, accumulated_logprobs)
            generation_scores = {
                "perplexity": self._calculate_perplexity(mock),
                "mean_entropy": self._calculate_mean_entropy(mock),
            }

        return StepCandidate(
            text=accumulated_text.strip(),
            token_ids=accumulated_tokens,
            is_complete=True,
            is_trajectory_complete=thinking_complete,  # True when </think> reached
            generation_scores=generation_scores,
            raw_text=accumulated_text,
        )

    def generate_response(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """
        Generate response after thinking phase.

        Generates <start of response>...<end of response>.
        Should only be called after thinking phase is complete (trajectory contains </think>).

        Args:
            request: Original user request
            trajectory: Completed thinking trajectory (must contain </think>)
            candidates_per_step: Number of response candidates to generate

        Returns:
            List of response candidates, each with is_trajectory_complete=True
        """
        candidates = []
        prompt = self._build_prompt(request, trajectory)

        sampling_params = self._create_sampling_params(
            stop_tokens=self.response_stop_tokens,
            n=candidates_per_step,
            max_tokens=self.max_new_tokens,
            min_tokens=0,  # No minimum for response phase
        )

        outputs = self.model.generate([prompt], sampling_params)

        for output in outputs[0].outputs:
            text = output.text

            # Ensure we include the stop token in the output
            # vLLM may or may not include it depending on settings
            for pattern in self.answer_patterns:
                if pattern not in text:
                    text = text + pattern
                    break

            generation_scores = {
                "perplexity": self._calculate_perplexity(output),
                "mean_entropy": self._calculate_mean_entropy(output),
            }

            candidate = StepCandidate(
                text=text.strip(),
                token_ids=output.token_ids,
                is_complete=True,
                is_trajectory_complete=True,  # Response phase always completes trajectory
                generation_scores=generation_scores,
                raw_text=output.text,
            )
            candidates.append(candidate)

        return candidates

    def generate_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """
        Generate N candidate next steps from current trajectory.

        Automatically determines phase:
        - If trajectory doesn't contain </think>: generate thinking step
        - If trajectory contains </think>: generate response
        """
        trajectory = trajectory or []

        # Check if thinking phase is complete
        full_trajectory = covert_trajectory_to_string(trajectory)
        thinking_complete = "</think>" in full_trajectory

        if thinking_complete:
            # Phase 2: Generate response
            log.info("Thinking complete, generating response")
            return self.generate_response(request, trajectory, candidates_per_step)
        else:
            # Phase 1: Generate thinking step
            return self.generate_thinking_step(request, trajectory, candidates_per_step)

    def generate_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Alias for generate_response (for strategy compatibility)."""
        return self.generate_response(request, trajectory, candidates_per_step)

    def __call__(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Callable interface for step generation."""
        return self.generate_candidates(request, trajectory, candidates_per_step)


if __name__ == "__main__":
    # Test the generator
    model = LLM(
        model="Qwen/Qwen3-8B",
        gpu_memory_utilization=0.9,
        max_model_len=32768,
    )

    generator = ThinkingStepGeneratorVLLM(
        model=model,
        min_step_chars=200,
        max_step_chars=1200,
        answer_patterns=["<end of response>"],
    )

    request = [{"role": "user", "content": "What is the sum of 2+2?"}]

    # Test thinking phase
    print("=== THINKING PHASE ===")
    trajectory = []
    for step in range(10):
        candidates = generator.generate_thinking_step(request, trajectory)
        best = candidates[0]
        trajectory.append(best)
        print(f"\nStep {step}: {best.text[:100]}...")
        print(f"  Thinking complete: {best.is_trajectory_complete}")
        if best.is_trajectory_complete:
            break

    # Test response phase
    print("\n=== RESPONSE PHASE ===")
    response_candidates = generator.generate_response(request, trajectory)
    print(f"Response: {response_candidates[0].text}")
