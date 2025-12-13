"""
vLLM step candidate generator for thinking mode.

Uses semantic stop tokens derived from ThinkingMarkerDetector patterns
with post-stop validation for min_step_chars and sentence boundaries.
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
    vLLM step generator for thinking mode (<think> tags).

    Uses semantic stop tokens (e.g., "\\nSo ", "\\nLet me ") derived from
    ThinkingMarkerDetector patterns. After each stop, validates:
    - min_step_chars: Minimum characters per step
    - Sentence boundary: Step should end at sentence punctuation

    If validation fails, continues generation until a valid boundary.
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
        # Answer patterns for trajectory completion
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

        # Answer patterns for trajectory completion
        self.answer_patterns = answer_patterns or ANSWER_TOKENS

        # Build stop tokens from marker categories
        self.stop_tokens = get_stop_tokens_compact(
            use_sequence=use_sequence,
            use_conclusion=use_conclusion,
            use_thinking=use_thinking,
            use_verification=use_verification,
            use_reasoning=use_reasoning,
            use_correction=use_correction,
            use_structure=use_structure,
            custom_words=custom_words,
        )

        # Add answer patterns to stop tokens so generation stops at answer boundaries
        self.stop_tokens = list(set(self.stop_tokens + self.answer_patterns))

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

        log.info(f"ThinkingStepGeneratorVLLM initialized with {len(self.stop_tokens)} stop tokens")

    def _create_sampling_params(
        self,
        n: int = 1,
        include_stop_tokens: bool = True,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
    ) -> SamplingParams:
        """Create SamplingParams with appropriate stop tokens."""
        stop = self.stop_tokens if include_stop_tokens else self.answer_patterns

        # Use min_tokens to force minimum generation before stop tokens apply
        # This prevents stop tokens from triggering at the very start of generation
        # Approximate: 4 chars per token, so min_step_chars / 4
        if min_tokens is None and include_stop_tokens:
            min_tokens = max(1, self.min_step_chars // 4)

        return SamplingParams(
            n=n,
            max_tokens=max_tokens or self.max_new_tokens,
            min_tokens=min_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            logprobs=20,
            stop=stop,
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

    def _is_trajectory_complete(self, text: str) -> bool:
        """Check if trajectory is complete (answer pattern found)."""
        for pattern in self.answer_patterns:
            if pattern in text:
                return True
        return False

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
        # Don't use apply_chat_template with trajectory - it mangles the thinking blocks
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

    def generate_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """
        Generate N candidate next steps from current trajectory.

        Uses stop tokens to detect potential boundaries, then validates.
        If validation fails, continues generation.
        """
        trajectory = trajectory or []
        candidates = []

        for i in range(candidates_per_step):
            candidate = self._generate_single_step(request, trajectory)
            candidates.append(candidate)

        return candidates

    def _generate_single_step(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> StepCandidate:
        """
        Generate a single step with boundary validation.

        Continues generation if initial stop is not a valid boundary.
        """
        # Build initial prompt
        prompt = self._build_prompt(request, trajectory)
        accumulated_text = ""
        accumulated_tokens = []
        accumulated_logprobs = []
        max_continuation_attempts = 5

        for attempt in range(max_continuation_attempts):
            # Create sampling params
            remaining_tokens = self.max_new_tokens - len(accumulated_tokens)
            if remaining_tokens <= 0:
                break

            sampling_params = self._create_sampling_params(
                n=1,
                include_stop_tokens=True,
                max_tokens=remaining_tokens,
            )

            # Generate
            current_prompt = prompt + accumulated_text
            outputs = self.model.generate([current_prompt], sampling_params)
            output = outputs[0].outputs[0]

            # Accumulate
            accumulated_text += output.text
            accumulated_tokens.extend(output.token_ids)
            if output.logprobs:
                accumulated_logprobs.extend(output.logprobs)

            # Check trajectory completion first
            if self._is_trajectory_complete(accumulated_text):
                log.debug(f"Trajectory complete after {attempt + 1} attempts")
                break

            # Validate boundary
            if self._is_valid_step_boundary(accumulated_text):
                log.debug(f"Valid step boundary after {attempt + 1} attempts")
                break

            # Check max step chars
            if len(accumulated_text.strip()) >= self.max_step_chars:
                log.debug(f"Max step chars reached after {attempt + 1} attempts")
                break

            log.debug(
                f"Attempt {attempt + 1}: Invalid boundary "
                f"(len={len(accumulated_text)}, min={self.min_step_chars}), continuing..."
            )

        # Build candidate
        is_trajectory_complete = self._is_trajectory_complete(accumulated_text)

        # Calculate scores from accumulated logprobs
        generation_scores = {}
        if accumulated_logprobs and accumulated_tokens:
            # Create mock output for score calculation
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
            is_trajectory_complete=is_trajectory_complete,
            generation_scores=generation_scores,
            raw_text=accumulated_text,
        )

    def generate_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """
        Generate final answer candidates.

        Uses only answer patterns as stop tokens, not step boundary tokens.
        """
        candidates = []
        prompt = self._build_prompt(request, trajectory)

        # Create sampling params with only answer tokens as stops
        sampling_params = self._create_sampling_params(
            n=candidates_per_step,
            include_stop_tokens=False,  # Only use answer patterns
            max_tokens=self.max_new_tokens,
        )

        outputs = self.model.generate([prompt], sampling_params)

        for output in outputs[0].outputs:
            generation_scores = {
                "perplexity": self._calculate_perplexity(output),
                "mean_entropy": self._calculate_mean_entropy(output),
            }

            candidate = StepCandidate(
                text=output.text.strip(),
                token_ids=output.token_ids,
                is_complete=True,
                is_trajectory_complete=True,
                generation_scores=generation_scores,
                raw_text=output.text,
            )
            candidates.append(candidate)

        return candidates

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
        use_sequence=True,
        use_conclusion=True,
        use_thinking=True,
        use_verification=True,
    )

    request = [{"role": "user", "content": "What is the sum of 2+2?"}]
    candidates = generator.generate_candidates(request, candidates_per_step=2)

    for i, c in enumerate(candidates):
        print(f"\n=== Candidate {i} ===")
        print(f"Text: {c.text[:200]}...")
        print(f"Complete: {c.is_trajectory_complete}")
        print(f"Scores: {c.generation_scores}")
