"""
Unified vLLM step candidate generator supporting both structured and thinking modes.

Two modes:
1. thinking_mode=True (default): Two-phase generation with <think>...</think>
   - Uses semantic stop tokens for step boundaries
   - Phase 1: Generate thinking content inside <think>...</think>
   - Phase 2: Generate response after thinking

2. thinking_mode=False: Simple structured generation
   - Uses StructuredStepDetector with explicit step patterns
   - Single-phase generation
"""

import inspect
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput

from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.step_boundary_detectors import StructuredStepDetector
from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector
from llm_tts.step_boundary_detectors.thinking.vllm import get_stop_tokens

if TYPE_CHECKING:
    from llm_tts.utils.flops import FLOPCalculator

log = logging.getLogger(__name__)


class VLLMStepGenerator(StepCandidateGeneratorBase):
    """
    Unified vLLM step generator supporting both thinking and structured modes.

    Args:
        model: vLLM model instance
        thinking_mode: If True, use two-phase thinking generation (default).
                      If False, use simple structured generation.
        detector: StructuredStepDetector for non-thinking mode (optional)
        sampling_params: SamplingParams for non-thinking mode (optional)

    Thinking mode parameters:
        min_step_chars: Minimum characters per thinking step
        max_step_chars: Maximum characters per thinking step
        max_new_tokens: Maximum tokens per generation
        temperature, top_p, top_k: Sampling parameters
        use_sequence, use_conclusion, etc.: Stop token configuration
        answer_patterns: Patterns marking end of response
        disable_thinking_mode: If True, skip <think> block even in thinking mode
        max_model_len: Maximum context length for truncation
    """

    def __init__(
        self,
        model: LLM,
        thinking_mode: bool = True,
        # Non-thinking mode parameters
        detector: Optional[StructuredStepDetector] = None,
        sampling_params: Optional[SamplingParams] = None,
        # Thinking mode parameters
        min_step_chars: int = 200,
        max_step_chars: int = 1200,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        # Stop token configuration (for thinking mode)
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
        # Context length limit
        max_model_len: int = 32768,
        # FLOP calculator for token tracking
        flop_calculator: Optional["FLOPCalculator"] = None,
    ):
        # vLLM handles batching internally, so generation_batch_size is large
        super().__init__(generation_batch_size=1024, flop_calculator=flop_calculator)

        self.model = model
        self.thinking_mode = thinking_mode
        self.tokenizer = model.get_tokenizer()

        # Store parameters for both modes
        self.disable_thinking_mode = disable_thinking_mode
        self.min_step_chars = min_step_chars
        self.max_step_chars = max_step_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_model_len = max_model_len

        # Answer patterns for response phase (default: <end of response>)
        self.answer_patterns = (
            list(answer_patterns) if answer_patterns else ["<end of response>"]
        )

        if thinking_mode:
            self._init_thinking_mode(
                use_sequence=use_sequence,
                use_conclusion=use_conclusion,
                use_thinking=use_thinking,
                use_verification=use_verification,
                use_reasoning=use_reasoning,
                use_correction=use_correction,
                use_structure=use_structure,
                custom_words=custom_words,
            )
        else:
            self._init_structured_mode(detector, sampling_params)

        mode_str = "thinking" if thinking_mode else "structured"
        log.info(f"VLLMStepGenerator initialized in {mode_str} mode")

    def _init_thinking_mode(
        self,
        use_sequence: bool,
        use_conclusion: bool,
        use_thinking: bool,
        use_verification: bool,
        use_reasoning: bool,
        use_correction: bool,
        use_structure: bool,
        custom_words: Optional[List[str]],
    ):
        """Initialize thinking mode specific components."""
        # Build stop tokens for THINKING phase (step boundaries + </think>)
        self.thinking_stop_tokens = get_stop_tokens(
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
            min_step_chars=self.min_step_chars,
            max_step_chars=self.max_step_chars,
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
            f"Thinking mode: {len(self.thinking_stop_tokens)} thinking stops, "
            f"{len(self.response_stop_tokens)} response stops"
        )

    def _init_structured_mode(
        self,
        detector: Optional[StructuredStepDetector],
        sampling_params: Optional[SamplingParams],
    ):
        """Initialize structured mode specific components."""
        self.detector = detector or StructuredStepDetector()
        self.sampling_params = sampling_params or SamplingParams(
            min_tokens=100,
            max_tokens=self.max_new_tokens,
            logprobs=20,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

    # =========================================================================
    # Common utility methods
    # =========================================================================

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

    def _extract_logprobs(
        self, token_ids: List[int], logprobs: List[Dict]
    ) -> List[float]:
        """Extract logprobs for the generated tokens as a flat list."""
        if not logprobs or not token_ids:
            return []

        result = []
        for token_id, logprob_dict in zip(token_ids, logprobs):
            if token_id in logprob_dict:
                result.append(logprob_dict[token_id].logprob)
            else:
                result.append(-100.0)
        return result

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

    # =========================================================================
    # Thinking mode methods
    # =========================================================================

    def _is_valid_step_boundary(self, text: str) -> bool:
        """Check if text represents a valid step boundary (thinking mode)."""
        text = text.strip()

        if len(text) < self.min_step_chars:
            return False

        if text and text[-1] in ".!?\n":
            return True

        return False

    def _is_thinking_complete(self, text: str) -> bool:
        """Check if thinking phase is complete (contains </think>)."""
        return "</think>" in text

    def _truncate_trajectory(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        reserved_tokens: int = 4096,
    ) -> List[StepCandidate]:
        """Truncate trajectory to fit within max_model_len."""
        if not trajectory:
            return trajectory

        # Calculate base prompt tokens (without trajectory)
        tokenizer_signature = inspect.signature(self.tokenizer.apply_chat_template)
        has_enable_thinking = "enable_thinking" in tokenizer_signature.parameters

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

        base_tokens = len(self.tokenizer.encode(base_prompt))
        available_tokens = self.max_model_len - base_tokens - reserved_tokens

        if available_tokens <= 0:
            log.warning(
                f"Base prompt ({base_tokens} tokens) + reserved ({reserved_tokens}) "
                f"exceeds max_model_len ({self.max_model_len}). Returning empty trajectory."
            )
            return []

        # Calculate tokens for each step
        step_tokens = []
        for step in trajectory:
            tokens = len(self.tokenizer.encode(step.text))
            step_tokens.append(tokens)

        total_tokens = sum(step_tokens)

        if total_tokens <= available_tokens:
            return trajectory

        # Truncate from the beginning (keep recent steps)
        truncated = list(trajectory)
        truncated_tokens = list(step_tokens)

        while sum(truncated_tokens) > available_tokens and truncated:
            truncated.pop(0)
            removed_tokens = truncated_tokens.pop(0)
            log.info(
                f"Truncating trajectory: removed step with {removed_tokens} tokens "
                f"(remaining: {len(truncated)} steps, {sum(truncated_tokens)} tokens)"
            )

        if not truncated:
            log.warning("All trajectory steps truncated due to context length limit")

        return truncated

    def _build_prompt(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> str:
        """Build prompt from request and trajectory using tokenizer's chat template."""
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

        trajectory_text = convert_trajectory_to_string(trajectory)
        return base_prompt + trajectory_text

    def _generate_single_thinking_step(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> StepCandidate:
        """Generate a single thinking step with boundary validation."""
        prompt = self._build_prompt(request, trajectory)
        accumulated_text = ""
        accumulated_tokens = []
        accumulated_logprobs = []
        max_continuation_attempts = 5

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

            if self._is_thinking_complete(accumulated_text):
                log.debug(f"Thinking complete (</think>) after {attempt + 1} attempts")
                think_pos = accumulated_text.find("</think>")
                accumulated_text = accumulated_text[: think_pos + len("</think>")]
                break

            if self._is_valid_step_boundary(accumulated_text):
                log.debug(f"Valid thinking step boundary after {attempt + 1} attempts")
                break

            if len(accumulated_text.strip()) >= self.max_step_chars:
                log.debug(f"Max step chars reached after {attempt + 1} attempts")
                break

            log.debug(
                f"Attempt {attempt + 1}: Invalid boundary "
                f"(len={len(accumulated_text)}, min={self.min_step_chars}), continuing..."
            )

        thinking_complete = self._is_thinking_complete(accumulated_text)

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
            is_trajectory_complete=thinking_complete,
            generation_scores=generation_scores,
            other_data={
                "uncertainty_score": 1.0
                / (1.0 + generation_scores.get("mean_entropy", 0)),
                "logprobs": self._extract_logprobs(
                    accumulated_tokens, accumulated_logprobs
                ),
            },
            raw_text=accumulated_text,
        )

    def generate_thinking_step(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate thinking step candidates (inside <think> block)."""
        if not self.thinking_mode:
            raise RuntimeError("generate_thinking_step() requires thinking_mode=True")

        trajectory = trajectory or []
        trajectory = self._truncate_trajectory(
            request, trajectory, reserved_tokens=self.max_new_tokens
        )

        candidates = []
        for i in range(candidates_per_step):
            candidate = self._generate_single_thinking_step(request, trajectory)
            candidates.append(candidate)

        return candidates

    def generate_response(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate response after thinking phase."""
        if not self.thinking_mode:
            raise RuntimeError("generate_response() requires thinking_mode=True")

        candidates = []
        prompt = self._build_prompt(request, trajectory)

        sampling_params = self._create_sampling_params(
            stop_tokens=self.response_stop_tokens,
            n=candidates_per_step,
            max_tokens=self.max_new_tokens,
            min_tokens=0,
        )

        outputs = self.model.generate([prompt], sampling_params)

        for output in outputs[0].outputs:
            text = output.text

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
                is_trajectory_complete=True,
                generation_scores=generation_scores,
                other_data={
                    "uncertainty_score": 1.0
                    / (1.0 + generation_scores.get("mean_entropy", 0)),
                    "logprobs": self._extract_logprobs(
                        output.token_ids, output.logprobs
                    ),
                },
                raw_text=output.text,
            )
            candidates.append(candidate)

        return candidates

    def generate_full_thinking(
        self,
        request: List[Dict[str, str]],
        num_candidates: int = 1,
        max_tokens: Optional[int] = None,
    ) -> List[StepCandidate]:
        """Generate N complete thinking phases in batch (no step boundaries)."""
        if not self.thinking_mode:
            raise RuntimeError("generate_full_thinking() requires thinking_mode=True")

        prompt = self._build_prompt(request, [])
        max_tokens = max_tokens or self.max_new_tokens

        sampling_params = self._create_sampling_params(
            stop_tokens=["</think>"],
            n=num_candidates,
            max_tokens=max_tokens,
            min_tokens=0,
        )

        outputs = self.model.generate([prompt], sampling_params)
        candidates = []

        for output in outputs[0].outputs:
            text = output.text

            if "</think>" not in text:
                text = text + "</think>"

            generation_scores = {
                "perplexity": self._calculate_perplexity(output),
                "mean_entropy": self._calculate_mean_entropy(output),
            }

            candidate = StepCandidate(
                text=text.strip(),
                token_ids=output.token_ids,
                is_complete=True,
                is_trajectory_complete=True,
                generation_scores=generation_scores,
                other_data={
                    "uncertainty_score": 1.0
                    / (1.0 + generation_scores.get("mean_entropy", 0)),
                    "logprobs": self._extract_logprobs(
                        output.token_ids, output.logprobs
                    ),
                },
                raw_text=output.text,
            )
            candidates.append(candidate)

        return candidates

    # =========================================================================
    # Structured mode methods
    # =========================================================================

    def _generate_structured_candidates(
        self, request, candidates_per_step: int = 1
    ) -> List[StepCandidate]:
        """Generate candidates using structured mode (simple generation)."""
        candidates = []

        self.sampling_params.n = candidates_per_step
        answers = self.model.generate(request, sampling_params=self.sampling_params)[
            0
        ].outputs

        for i in range(candidates_per_step):
            step_text = answers[i].text
            is_complete = True

            is_trajectory_complete = self.detector.is_trajectory_complete(
                answers[i].text
            )
            token_ids = answers[i].token_ids
            generation_scores = {
                "perplexity": self._calculate_perplexity(answers[i]),
                "mean_entropy": self._calculate_mean_entropy(answers[i]),
            }

            candidate = StepCandidate(
                text=step_text,
                token_ids=token_ids,
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                generation_scores=generation_scores,
                raw_text=answers[i].text,
            )
            candidates.append(candidate)

        return candidates

    # =========================================================================
    # Common interface methods
    # =========================================================================

    def generate_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory."""
        if self.thinking_mode:
            trajectory = trajectory or []
            full_trajectory = convert_trajectory_to_string(trajectory)
            thinking_complete = "</think>" in full_trajectory

            if thinking_complete:
                log.info("Thinking complete, generating response")
                return self.generate_response(request, trajectory, candidates_per_step)
            else:
                return self.generate_thinking_step(
                    request, trajectory, candidates_per_step
                )
        else:
            return self._generate_structured_candidates(request, candidates_per_step)

    def generate_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate answer candidates with token tracking."""
        if self.thinking_mode:
            # Check if thinking phase is complete
            full_trajectory = convert_trajectory_to_string(trajectory)
            if "</think>" not in full_trajectory:
                log.warning(
                    "generate_answer_candidates called without </think> in trajectory. "
                    "Adding </think> to close thinking phase."
                )
                close_thinking_step = StepCandidate(
                    text="\n</think>\n\n<start of response>\nReasoning Steps:\n",
                    token_ids=[],
                    is_complete=True,
                    is_trajectory_complete=True,
                )
                trajectory.append(close_thinking_step)

            prompt = self._build_prompt(request, trajectory)
            context_tokens = len(self.tokenizer.encode(prompt))

            candidates = self.generate_response(
                request, trajectory, candidates_per_step
            )
            self._record_generation(candidates, context_tokens=context_tokens)
            return candidates
        else:
            return self._generate_structured_candidates(request, candidates_per_step)

    def generate_answer(
        self, request, candidates_per_step: int, more_information=False
    ) -> List[StepCandidate]:
        """Generate final answer (structured mode compatibility)."""
        return self.generate_candidates(request, None, candidates_per_step)

    def generate_batch(
        self, requests: List[str], candidates_per_step: int = 1
    ) -> List[StepCandidate]:
        """Generate batch of completions from requests (structured mode)."""
        if self.thinking_mode:
            raise RuntimeError("generate_batch() requires thinking_mode=False")

        self.sampling_params.n = candidates_per_step
        vllm_outputs = self.model.generate(
            requests, sampling_params=self.sampling_params
        )
        formated_vllm_output = [
            vllm_outputs[i].outputs[0] for i in range(len(requests))
        ]
        result = []
        for i in range(len(requests)):
            step_text = formated_vllm_output[i].text
            is_complete = True

            is_trajectory_complete = self.detector.is_trajectory_complete(
                formated_vllm_output[i].text
            )
            token_ids = formated_vllm_output[i].token_ids
            generation_scores = {
                "perplexity": self._calculate_perplexity(formated_vllm_output[i]),
                "mean_entropy": self._calculate_mean_entropy(formated_vllm_output[i]),
            }

            candidate = StepCandidate(
                text=step_text,
                token_ids=token_ids,
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                generation_scores=generation_scores,
                raw_text=formated_vllm_output[i].text,
            )
            result.append(candidate)

        return result

    def __call__(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Callable interface for step generation with token tracking."""
        trajectory = trajectory or []

        if self.thinking_mode:
            prompt = self._build_prompt(request, trajectory)
            context_tokens = len(self.tokenizer.encode(prompt))
        else:
            context_tokens = 0

        candidates = self.generate_candidates(request, trajectory, candidates_per_step)

        if self.thinking_mode:
            self._record_generation(candidates, context_tokens=context_tokens)

        return candidates
