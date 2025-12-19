"""
Unified vLLM step candidate generator supporting both structured and thinking modes.

Two modes:
1. thinking_mode=True (default): Two-phase generation with <think>...</think>
   - Uses semantic stop tokens for step boundaries (e.g., "Wait,", "Hmm,", etc.)
   - Phase 1: Generate thinking content inside <think>...</think>
   - Phase 2: Generate response after thinking

2. thinking_mode=False: Simple structured generation
   - Uses StructuredStepDetector with explicit step patterns
   - Single-phase generation

Uncertainty scoring:
- Uses VLLMWithUncertainty wrapper from lm-polygraph for scoring
- Pass VLLMWithUncertainty as the model parameter

Architecture Notes:
-------------------

Both modes use vLLM's native batch generation (n=candidates_per_step) for efficiency.
Step boundaries are enforced via stop tokens and min_tokens parameter.

Method naming convention:
- PUBLIC API (no underscore prefix):
  - generate_step_candidates()   - Generate N candidate next steps
  - generate_answer_candidates() - Generate N final answer candidates

- PRIVATE IMPLEMENTATION (underscore prefix):
  - Thinking mode:
    - _generate_thinking_step_candidates()  - Generate N thinking step candidates
    - _generate_thinking_answer_candidates()- Generate N answer candidates after </think>
  - Structured mode:
    - _generate_structured_step_candidates()  - Generate N structured step candidates
    - _generate_structured_answer_candidates()- Generate N answer candidates

Token tracking (_record_generation):
- Called at LEAF methods that invoke model.generate() to ensure accurate FLOP calculation
- Each leaf method tracks: context_tokens (input) and output_tokens (generated)
"""

import inspect
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from vllm import SamplingParams

# Optional lm-polygraph imports for uncertainty computation
try:
    from lm_polygraph.utils import VLLMWithUncertainty

    POLYGRAPH_AVAILABLE = True
except ImportError:
    POLYGRAPH_AVAILABLE = False
    VLLMWithUncertainty = None

from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.step_boundary_detectors import StructuredStepDetector
from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector

if TYPE_CHECKING:
    from llm_tts.utils.flops import FLOPCalculator

log = logging.getLogger(__name__)


class VLLMStepGenerator(StepCandidateGeneratorBase):
    """
    Unified vLLM step generator supporting both thinking and structured modes.

    Uses VLLMWithUncertainty wrapper from lm-polygraph for both generation and
    uncertainty scoring. The wrapper's score() method computes uncertainty on
    (possibly truncated) tokens.

    Args:
        model: VLLMWithUncertainty wrapper instance (wraps vLLM LLM with scoring)
        thinking_mode: If True, use two-phase thinking generation (default).
                      If False, use simple structured generation.
        detector: Step boundary detector (StructuredStepDetector or ThinkingMarkerDetector).
                 If None, creates default based on thinking_mode.
        sampling_params: SamplingParams for generation (structured mode only)
        answer_patterns: Patterns marking end of response
        max_new_tokens: Maximum tokens per generation
        temperature, top_p, top_k: Sampling parameters
        max_model_len: Maximum context length for truncation
        flop_calculator: Optional FLOP calculator for token tracking
    """

    def __init__(
        self,
        model: "VLLMWithUncertainty",
        thinking_mode: bool = True,
        detector: Optional[
            Union[StructuredStepDetector, ThinkingMarkerDetector]
        ] = None,
        sampling_params: Optional[SamplingParams] = None,
        answer_patterns: Optional[List[str]] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_model_len: int = 32768,
        flop_calculator: Optional["FLOPCalculator"] = None,
    ):
        super().__init__(generation_batch_size=1024, flop_calculator=flop_calculator)

        self.model = model  # VLLMWithUncertainty wrapper
        self.thinking_mode = thinking_mode
        self.tokenizer = model.get_tokenizer()

        # Store common parameters
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
            self._init_thinking_mode(detector)
        else:
            self._init_structured_mode(detector, sampling_params)

        mode_str = "thinking" if thinking_mode else "structured"
        log.info(f"VLLMStepGenerator initialized in {mode_str} mode")

    def _init_thinking_mode(
        self,
        detector: Optional[ThinkingMarkerDetector],
    ):
        """Initialize thinking mode specific components."""
        # Create default detector if not provided
        if detector is None:
            detector = ThinkingMarkerDetector()

        self.detector = detector
        self.detector.answer_patterns = self.answer_patterns

        # Get min/max step tokens from detector (required)
        self.min_step_tokens = detector.min_step_tokens
        self.max_step_tokens = detector.max_step_tokens

        # Derive stop tokens from detector's configuration
        self.thinking_stop_tokens = detector.get_vllm_stop_tokens()

        # Add </think> to stop thinking phase
        if "</think>" not in self.thinking_stop_tokens:
            self.thinking_stop_tokens.append("</think>")

        # Stop tokens for RESPONSE phase (answer patterns only)
        self.response_stop_tokens = self.answer_patterns.copy()

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
            min_tokens=0,  # No minimum - stop tokens can trigger immediately
            max_tokens=self.max_new_tokens,
            logprobs=20,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

    # =========================================================================
    # Common utility methods
    # =========================================================================

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

    def score_truncated_sequence(
        self,
        prompt: str,
        truncated_text: str,
    ) -> Dict[str, float]:
        """Score a truncated sequence using vLLM's prompt_logprobs (forward pass).

        This is used after step boundary truncation to re-compute uncertainty
        on the actual truncated text rather than the original generation.

        Args:
            prompt: The original prompt (context)
            truncated_text: The truncated generated text to score

        Returns:
            Dict with 'uncertainty_score' key (from configured estimator)
        """
        # Combine prompt and truncated text
        full_text = prompt + truncated_text

        # Use vLLM to get logprobs for the truncated portion
        # prompt_logprobs returns logprobs for input tokens (teacher forcing)
        scoring_params = SamplingParams(
            max_tokens=1,  # Generate minimal tokens
            temperature=0.0,  # Deterministic
            prompt_logprobs=20,  # Get logprobs for prompt tokens
        )

        outputs = self.model.generate([full_text], scoring_params)
        request_output = outputs[0]

        if not request_output.prompt_logprobs:
            log.warning("No prompt_logprobs returned, returning default metrics")
            return {"uncertainty_score": 0.0, "validity_score": 1.0}

        # Extract logprobs for the truncated_text portion only
        prompt_tokens = len(self.tokenizer.encode(prompt))
        truncated_logprobs = request_output.prompt_logprobs[prompt_tokens:]

        if not truncated_logprobs:
            return {"uncertainty_score": 0.0, "validity_score": 1.0}

        # Convert prompt_logprobs to format expected by wrapper.score()
        # prompt_logprobs is List[Optional[Dict[int, Logprob]]]
        # Get token IDs from the truncated portion
        full_token_ids = self.tokenizer.encode(full_text)
        truncated_token_ids = full_token_ids[prompt_tokens:]

        # Filter out None entries and create logprob dicts
        valid_logprobs = []
        valid_token_ids = []
        for i, lp in enumerate(truncated_logprobs):
            if lp is not None and i < len(truncated_token_ids):
                valid_logprobs.append(lp)
                valid_token_ids.append(truncated_token_ids[i])

        if not valid_logprobs:
            return {"uncertainty_score": 0.0, "validity_score": 1.0}

        # Compute uncertainty using wrapper
        uncertainty = self.model.score(valid_token_ids, valid_logprobs)
        return {
            "uncertainty_score": uncertainty,
            "validity_score": 1.0 / (1.0 + uncertainty),
        }

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
            repetition_penalty=1.05,  # Penalize repetitive text
        )

    # =========================================================================
    # Thinking mode methods
    # =========================================================================

    def _is_thinking_complete(self, text: str) -> bool:
        """Check if thinking phase is complete (contains </think>)."""
        return "</think>" in text

    def _truncate_trajectory(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        reserved_tokens: int = 4096,
    ) -> tuple:
        """Truncate trajectory to fit within max_model_len.

        Returns:
            Tuple of (truncated_trajectory, was_truncated)
        """
        if not trajectory:
            return trajectory, False

        # Calculate base prompt tokens (without trajectory)
        tokenizer_signature = inspect.signature(self.tokenizer.apply_chat_template)
        has_enable_thinking = "enable_thinking" in tokenizer_signature.parameters

        if has_enable_thinking:
            base_prompt = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
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
            return [], True

        # Calculate tokens for each step
        step_tokens = []
        for step in trajectory:
            tokens = len(self.tokenizer.encode(step.text))
            step_tokens.append(tokens)

        total_tokens = sum(step_tokens)

        if total_tokens <= available_tokens:
            return trajectory, False

        # Truncate from the beginning (keep recent steps)
        truncated = list(trajectory)
        truncated_tokens = list(step_tokens)
        removed_count = 0

        while sum(truncated_tokens) > available_tokens and truncated:
            truncated.pop(0)
            truncated_tokens.pop(0)
            removed_count += 1

        log.warning(
            f"Context limit reached: removed {removed_count} old steps "
            f"(remaining: {len(truncated)} steps, {sum(truncated_tokens)} tokens). "
            f"Marking trajectory complete."
        )

        if not truncated:
            log.warning("All trajectory steps truncated due to context length limit")

        return truncated, True

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
                    enable_thinking=True,
                )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    request,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return prompt

        # For CONTINUATION (has trajectory): build base prompt then append trajectory
        if has_enable_thinking:
            base_prompt = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            base_prompt = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
            )

        trajectory_text = convert_trajectory_to_string(trajectory)
        return base_prompt + trajectory_text

    def _apply_chat_template(
        self,
        request: List[Dict[str, str]],
        enable_thinking: bool = True,
    ) -> str:
        """
        Apply chat template to request with enable_thinking support.

        Args:
            request: Chat messages in OpenAI format
            enable_thinking: Whether to enable thinking mode (if tokenizer supports it)

        Returns:
            Formatted prompt string
        """
        tokenizer_signature = inspect.signature(self.tokenizer.apply_chat_template)
        has_enable_thinking = "enable_thinking" in tokenizer_signature.parameters

        if has_enable_thinking:
            result = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            result = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Force-close thinking when disabled but tokenizer doesn't support enable_thinking
        if not enable_thinking and not has_enable_thinking:
            result += "<think>\n\n</think>\n\n"

        return result

    def _generate_thinking_step_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate N thinking step candidates using vLLM batch generation.

        Uses vLLM's native batching (n=candidates_per_step) with stop tokens
        to define step boundaries. No continuation logic needed since semantic
        markers like "Wait,", "Hmm," etc. are valid reasoning step boundaries.

        Records token usage for FLOP calculation.
        """
        if not self.thinking_mode:
            raise RuntimeError(
                "_generate_thinking_step_candidates() requires thinking_mode=True"
            )

        trajectory = trajectory or []
        trajectory, context_limit_reached = self._truncate_trajectory(
            request, trajectory, reserved_tokens=self.max_new_tokens
        )

        # If context limit reached, return a closing step with </think>
        if context_limit_reached:
            log.warning("Context limit reached, closing thinking phase with </think>")
            closing_step = StepCandidate(
                text="</think>",
                token_ids=[],
                is_complete=True,
                is_trajectory_complete=True,
                other_data={
                    "uncertainty_score": 0.0,
                    "validity_score": 1.0,
                    "context_limit_reached": True,
                },
                raw_text="</think>",
            )
            return [closing_step] * candidates_per_step

        prompt = self._build_prompt(request, trajectory)
        context_tokens = len(self.tokenizer.encode(prompt))

        # Use vLLM batch generation with n=candidates_per_step
        sampling_params = self._create_sampling_params(
            stop_tokens=self.thinking_stop_tokens,
            n=candidates_per_step,
            max_tokens=self.max_step_tokens,
            min_tokens=self.min_step_tokens,
        )

        outputs = self.model.generate([prompt], sampling_params)
        request_output = outputs[0]

        candidates = []
        for idx, output in enumerate(request_output.outputs):
            text = output.text
            token_ids = output.token_ids
            logprobs = output.logprobs
            stop_reason = getattr(output, "stop_reason", None)

            # Check if thinking phase is complete
            thinking_complete = "</think>" in text
            if thinking_complete:
                # Truncate at </think>
                think_pos = text.find("</think>")
                text = text[: think_pos + len("</think>")]

            # If hit max_tokens, truncate at sentence boundary to avoid mid-word cuts
            hit_max_tokens = stop_reason is None or stop_reason == "length"
            if hit_max_tokens and not thinking_complete:
                text = self._truncate_at_sentence_boundary(text)

            # Check for repetitions
            repetition_detected = self._detect_line_repetitions(text)
            if repetition_detected:
                log.warning(
                    f"Repetition detected in thinking step [{idx}], "
                    "marking trajectory complete"
                )

            # Compute uncertainty
            uncertainty_score = self.model.score(token_ids, logprobs)

            candidate = StepCandidate(
                text=text.strip(),
                token_ids=token_ids,
                is_complete=True,
                is_trajectory_complete=thinking_complete or repetition_detected,
                other_data={
                    "uncertainty_score": uncertainty_score,
                    "validity_score": 1.0 / (1.0 + uncertainty_score),
                    "logprobs": self._extract_logprobs(token_ids, logprobs),
                },
                raw_text=output.text,
            )
            candidates.append(candidate)

        # Record token usage for FLOP calculation
        self._record_generation(candidates, context_tokens=context_tokens)

        return candidates

    def _generate_thinking_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate N answer candidates after thinking phase (after </think>).

        This is a leaf method that calls model.generate() directly using
        vLLM's native batching (n=candidates_per_step).
        Records token usage for FLOP calculation.
        """
        if not self.thinking_mode:
            raise RuntimeError(
                "_generate_thinking_answer_candidates() requires thinking_mode=True"
            )

        candidates = []
        prompt = self._build_prompt(request, trajectory)
        context_tokens = len(self.tokenizer.encode(prompt))

        sampling_params = self._create_sampling_params(
            stop_tokens=self.response_stop_tokens,
            n=candidates_per_step,
            max_tokens=self.max_new_tokens,
            min_tokens=0,
        )

        outputs = self.model.generate([prompt], sampling_params)
        request_output = outputs[0]

        for idx, output in enumerate(request_output.outputs):
            text = output.text

            for pattern in self.answer_patterns:
                if pattern not in text:
                    text = text + pattern
                    break

            # Compute uncertainty using wrapper
            uncertainty_score = self.model.score(output.token_ids, output.logprobs)

            candidate = StepCandidate(
                text=text.strip(),
                token_ids=output.token_ids,
                is_complete=True,
                is_trajectory_complete=True,
                other_data={
                    "uncertainty_score": uncertainty_score,
                    "validity_score": 1.0 / (1.0 + uncertainty_score),
                    "logprobs": self._extract_logprobs(
                        output.token_ids, output.logprobs
                    ),
                },
                raw_text=output.text,
            )
            candidates.append(candidate)

        # Record token usage for FLOP calculation
        self._record_generation(candidates, context_tokens=context_tokens)

        return candidates

    def generate_full_thinking(
        self,
        request: List[Dict[str, str]],
        num_candidates: int = 1,
        max_tokens: Optional[int] = None,
    ) -> List[StepCandidate]:
        """Generate N complete thinking phases in batch (no step boundaries).

        Records token usage for FLOP calculation.
        """
        if not self.thinking_mode:
            raise RuntimeError("generate_full_thinking() requires thinking_mode=True")

        prompt = self._build_prompt(request, [])
        context_tokens = len(self.tokenizer.encode(prompt))
        max_tokens = max_tokens or self.max_new_tokens

        sampling_params = self._create_sampling_params(
            stop_tokens=["</think>"],
            n=num_candidates,
            max_tokens=max_tokens,
            min_tokens=0,
        )

        outputs = self.model.generate([prompt], sampling_params)
        request_output = outputs[0]
        candidates = []

        for idx, output in enumerate(request_output.outputs):
            text = output.text

            if "</think>" not in text:
                text = text + "</think>"

            # Compute uncertainty using wrapper
            uncertainty_score = self.model.score(output.token_ids, output.logprobs)

            candidate = StepCandidate(
                text=text.strip(),
                token_ids=output.token_ids,
                is_complete=True,
                is_trajectory_complete=True,
                other_data={
                    "uncertainty_score": uncertainty_score,
                    "validity_score": 1.0 / (1.0 + uncertainty_score),
                    "logprobs": self._extract_logprobs(
                        output.token_ids, output.logprobs
                    ),
                },
                raw_text=output.text,
            )
            candidates.append(candidate)

        # Record token usage for FLOP calculation
        self._record_generation(candidates, context_tokens=context_tokens)

        return candidates

    # =========================================================================
    # Structured mode methods
    # =========================================================================

    def _truncate_at_step_boundary(self, text: str) -> str:
        """Truncate text at the SECOND occurrence of '- Step' pattern.

        We want to keep the first step but remove any subsequent steps.
        Example: '- Step 1: foo\n- Step 2: bar' -> '- Step 1: foo\n'
        """
        if not hasattr(self, "detector"):
            return text

        # Only truncate at "- Step" marker
        step_marker = "- Step"
        first_pos = text.find(step_marker)
        if first_pos == -1:
            return text  # No step marker found

        # Find second occurrence
        second_pos = text.find(step_marker, first_pos + len(step_marker))
        if second_pos > 0:
            return text[:second_pos]
        return text

    def _detect_line_repetitions(
        self,
        text: str,
        min_lines: int = 4,
        max_unique_ratio: float = 0.3,
    ) -> bool:
        """Detect if text contains excessive line-by-line repetitions.

        Args:
            text: Generated text to check
            min_lines: Minimum number of lines to check (avoid false positives on short text)
            max_unique_ratio: If unique_lines/total_lines < this ratio, it's repetition

        Returns:
            True if repetition detected, False otherwise
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if len(lines) < min_lines:
            return False

        unique_lines = set(lines)
        unique_ratio = len(unique_lines) / len(lines)

        if unique_ratio <= max_unique_ratio:
            log.warning(
                f"Detected line repetitions: {len(unique_lines)} unique out of "
                f"{len(lines)} lines (ratio {unique_ratio:.2f} <= {max_unique_ratio})"
            )
            return True

        return False

    def _truncate_repetitions(
        self,
        text: str,
        token_count: int,
        min_tokens_for_check: int = 1000,
        min_sentences_per_1k_tokens: int = 2,
    ) -> tuple:
        """Detect and truncate repetitive text when model hits max tokens.

        Simple heuristic: if we generated many tokens but have very few sentences,
        the model is likely stuck in a repetition loop.

        Args:
            text: Generated text to check
            token_count: Number of tokens generated
            min_tokens_for_check: Only check for repetition if token count exceeds this
            min_sentences_per_1k_tokens: Expected minimum sentences per 1000 tokens

        Returns:
            Tuple of (truncated_text, was_truncated)
        """
        # First check for line-by-line repetitions (works even for short texts)
        if self._detect_line_repetitions(text):
            return text + "<end of response>", True

        # Only check sentence-count heuristic if we generated many tokens
        if token_count < min_tokens_for_check:
            return text, False

        # Count sentences (by newlines)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        num_sentences = len(lines)

        # Calculate expected minimum sentences based on token count
        expected_min = (token_count / 1000) * min_sentences_per_1k_tokens

        if num_sentences < expected_min:
            # Too few sentences for this many tokens - likely repetition
            # Append <end of response> to force trajectory completion
            log.warning(
                f"Detected repetition: only {num_sentences} sentences for "
                f"{token_count} tokens (expected >= {expected_min:.0f}), "
                f"forcing end of response"
            )
            return text + "<end of response>", True

        return text, False

    def _truncate_at_sentence_boundary(self, text: str) -> str:
        """Truncate text at the last sentence boundary (period, newline, etc.).

        Used when hitting max_step_tokens to avoid cutting mid-sentence.
        """
        # Find last sentence boundary
        boundaries = [". ", ".\n", "?\n", "? ", "!\n", "! ", "\n\n"]
        last_boundary_pos = -1
        last_boundary = None

        for boundary in boundaries:
            pos = text.rfind(boundary)
            if pos > last_boundary_pos:
                last_boundary_pos = pos
                last_boundary = boundary

        if last_boundary_pos > 0:
            # Include the boundary character (period, etc.) but not trailing space/newline
            truncated = text[: last_boundary_pos + 1]
            log.debug(
                f"Truncated at sentence boundary '{repr(last_boundary)}' "
                f"pos {last_boundary_pos}, kept {len(truncated)}/{len(text)} chars"
            )
            return truncated

        # No sentence boundary found - return as-is
        return text

    def _generate_structured_step_candidates(
        self,
        request,
        candidates_per_step: int = 1,
        trajectory: Optional[List[StepCandidate]] = None,
    ) -> List[StepCandidate]:
        """Generate N step candidates using structured mode with step prefix injection.

        This is a leaf method that calls model.generate() directly using
        vLLM's native batching (n=candidates_per_step).

        Key approach (from PR #57):
        1. Prepend "- Step N: " to prompt BEFORE generation
        2. Model only generates step CONTENT (not the step marker)
        3. "- Step" stop token triggers only if model tries to start NEXT step
        4. No truncation needed - cleaner token counting

        Unlike thinking mode, structured mode doesn't need a separate
        _generate_structured_step() method because vLLM handles batching natively
        and step boundaries are enforced via stop tokens without iterative continuation.
        """
        candidates = []
        trajectory = trajectory or []

        # Build prompt with enable_thinking=False to disable thinking mode
        prompt = self._apply_chat_template(request, enable_thinking=False)

        # Append trajectory if present (continue from previous steps)
        if trajectory:
            trajectory_text = convert_trajectory_to_string(trajectory)
            prompt = prompt + trajectory_text

        # Calculate step number and create prefix
        # Step 0 = first step = "- Step 1: " (1-indexed for display)
        step_number = len(trajectory) + 1

        # No leading newline - previous step ends with newline
        # No trailing space - model will generate space if needed
        if step_number == 1:
            step_prefix = "<start of response>\nReasoning Steps:\n- Step 1:"
        else:
            step_prefix = f"- Step {step_number}:"

        # Append step prefix to prompt - model generates only the step content
        prompt_with_prefix = prompt + step_prefix

        self.sampling_params.n = candidates_per_step
        log.info(
            f"Structured mode: step {step_number}, "
            f"stop={self.sampling_params.stop}, min_tokens={self.sampling_params.min_tokens}"
        )

        context_tokens = len(self.tokenizer.encode(prompt_with_prefix))
        outputs = self.model.generate(
            prompt_with_prefix, sampling_params=self.sampling_params
        )
        request_output = outputs[0]
        answers = request_output.outputs

        for i in range(candidates_per_step):
            raw_text = answers[i].text
            generated_token_ids = answers[i].token_ids
            generated_logprobs = answers[i].logprobs

            # Log raw vLLM output
            stop_reason = getattr(answers[i], "stop_reason", None)
            log.debug(
                f"vLLM output [{i}]: {len(generated_token_ids)} tokens, stop={repr(stop_reason)}\n"
                f"  Raw text: {repr(raw_text[-80:] if len(raw_text) > 80 else raw_text)}"
            )

            # Build full step text by prepending the step prefix we injected
            # Model generates \n before "- Step", so raw_text ends with \n - keep it
            step_text = step_prefix + raw_text

            # Check if model hit max tokens (potential repetition loop)
            hit_max_tokens = stop_reason is None or stop_reason == "length"
            if hit_max_tokens:
                token_count = len(generated_token_ids)
                truncated_text, was_truncated = self._truncate_repetitions(
                    raw_text, token_count
                )
                if was_truncated:
                    raw_text = truncated_text
                    step_text = step_prefix + raw_text

            # Check if stopped at end-of-response marker
            # vLLM strips stop tokens from output, so check stop_reason
            stopped_at_eos = stop_reason == "<end of response>"
            if stopped_at_eos:
                # Append the EOS marker back since vLLM stripped it
                # raw_text already ends with \n, so just append the marker
                step_text = step_text + "<end of response>"
                log.info(
                    f"Candidate [{i}] stopped at <end of response>, appending marker"
                )

            is_complete = True

            # Check if trajectory is complete:
            # Either stop_reason is <end of response>, or text contains it
            is_trajectory_complete = (
                stopped_at_eos or self.detector.is_trajectory_complete(raw_text)
            )

            # Log scoring details with full repr to see newlines
            # Decode full token_ids to see what triggered the stop (includes stop token)
            token_count = len(generated_token_ids)
            full_decoded = self.tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )
            log.info(
                f"Scoring [{i}]: {token_count} tokens, stop={repr(stop_reason)}\n"
                f"  Full tokens decoded: {repr(full_decoded)}\n"
                f"  Raw text (stripped): {repr(raw_text)}\n"
                f"  Step text:           {repr(step_text)}"
            )

            # Compute uncertainty on generated tokens (no truncation needed)
            uncertainty_score = self.model.score(
                generated_token_ids, generated_logprobs
            )

            candidate = StepCandidate(
                text=step_text,
                token_ids=generated_token_ids,
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                other_data={
                    "uncertainty_score": uncertainty_score,
                    "validity_score": 1.0 / (1.0 + uncertainty_score),
                    "logprobs": self._extract_logprobs(
                        generated_token_ids, generated_logprobs
                    ),
                },
                raw_text=raw_text,
            )
            candidates.append(candidate)

        # IMPORTANT: VLLMStepGenerator overrides __call__ and therefore does NOT
        # inherit the base class' automatic token tracking for structured mode.
        # Record tokens here so per-sample token/FLOP metrics are non-zero.
        self._record_generation(candidates, context_tokens=context_tokens)

        return candidates

    # =========================================================================
    # Common interface methods
    # =========================================================================

    def generate_step_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory.

        PUBLIC API - Main entry point for step generation.
        Routes to appropriate private method based on mode and trajectory state.
        """
        if self.thinking_mode:
            trajectory = trajectory or []
            full_trajectory = convert_trajectory_to_string(trajectory)
            thinking_complete = "</think>" in full_trajectory

            if thinking_complete:
                log.info("Thinking complete, generating response")
                return self._generate_thinking_answer_candidates(
                    request, trajectory, candidates_per_step
                )
            else:
                return self._generate_thinking_step_candidates(
                    request, trajectory, candidates_per_step
                )
        else:
            return self._generate_structured_step_candidates(
                request, candidates_per_step, trajectory
            )

    def generate_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate N final answer candidates.

        PUBLIC API - Force generation of final answer even if model hasn't
        naturally reached end-of-response marker.
        """
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

            return self._generate_thinking_answer_candidates(
                request, trajectory, candidates_per_step
            )
        else:
            # Structured mode: force generation of <Answer>: instead of another step
            return self._generate_structured_answer_candidates(
                request, candidates_per_step, trajectory
            )

    def _generate_structured_answer_candidates(
        self,
        request: List[Dict[str, str]],
        candidates_per_step: int = 1,
        trajectory: Optional[List[StepCandidate]] = None,
    ) -> List[StepCandidate]:
        """Generate N final answer candidates in structured mode.

        This is a leaf method that calls model.generate() directly using
        vLLM's native batching (n=candidates_per_step).
        Records token usage for FLOP calculation.

        Called when max_steps is reached without the model naturally generating <Answer>:.
        Forces the model to provide a final answer by injecting <Answer>: prefix.
        """
        candidates = []
        trajectory = trajectory or []

        # Build prompt with enable_thinking=False
        prompt = self._apply_chat_template(request, enable_thinking=False)

        # Append trajectory if present
        if trajectory:
            trajectory_text = convert_trajectory_to_string(trajectory)
            prompt = prompt + trajectory_text

        # Inject <Answer>: prefix to force answer generation
        answer_prefix = "<Answer>:"
        prompt_with_prefix = prompt + answer_prefix
        context_tokens = len(self.tokenizer.encode(prompt_with_prefix))

        # Only stop at <end of response>
        answer_sampling_params = SamplingParams(
            n=candidates_per_step,
            max_tokens=512,  # Answer should be short
            min_tokens=1,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            logprobs=20,
            stop=["<end of response>"],
        )

        log.info(
            f"Structured mode: generating final answer with prefix '{answer_prefix}'"
        )

        outputs = self.model.generate(
            prompt_with_prefix, sampling_params=answer_sampling_params
        )
        request_output = outputs[0]
        answers = request_output.outputs

        for i in range(candidates_per_step):
            raw_text = answers[i].text
            generated_token_ids = answers[i].token_ids
            generated_logprobs = answers[i].logprobs
            stop_reason = getattr(answers[i], "stop_reason", None)

            # Build answer text with prefix
            answer_text = answer_prefix + raw_text

            # Append <end of response> if stopped there
            if stop_reason == "<end of response>":
                answer_text = answer_text + "<end of response>"

            log.info(
                f"Answer candidate [{i}]: stop={repr(stop_reason)}, "
                f"text={repr(answer_text[:100])}"
            )

            # Compute uncertainty
            uncertainty_score = self.model.score(
                generated_token_ids, generated_logprobs
            )

            candidate = StepCandidate(
                text=answer_text,
                token_ids=generated_token_ids,
                is_complete=True,
                is_trajectory_complete=True,  # Answer is always trajectory complete
                other_data={
                    "uncertainty_score": uncertainty_score,
                    "validity_score": 1.0 / (1.0 + uncertainty_score),
                    "logprobs": self._extract_logprobs(
                        generated_token_ids, generated_logprobs
                    ),
                },
                raw_text=raw_text,
            )
            candidates.append(candidate)

        # Record token usage for FLOP calculation
        self._record_generation(candidates, context_tokens=context_tokens)

        return candidates

    def generate_answer(
        self, request, candidates_per_step: int, more_information=False
    ) -> List[StepCandidate]:
        """Generate final answer (structured mode compatibility)."""
        return self.generate_step_candidates(request, None, candidates_per_step)

    def generate_batch(
        self, requests: List[str], candidates_per_step: int = 1
    ) -> List[StepCandidate]:
        """Generate batch of completions from requests (structured mode).

        Records token usage for FLOP calculation.
        """
        if self.thinking_mode:
            raise RuntimeError("generate_batch() requires thinking_mode=False")

        # Calculate total context tokens across all requests
        context_tokens = sum(len(self.tokenizer.encode(req)) for req in requests)

        self.sampling_params.n = candidates_per_step
        vllm_outputs = self.model.generate(
            requests, sampling_params=self.sampling_params
        )
        result = []
        for i in range(len(requests)):
            request_output = vllm_outputs[i]
            output = request_output.outputs[0]
            step_text = output.text
            is_complete = True

            is_trajectory_complete = self.detector.is_trajectory_complete(output.text)
            token_ids = output.token_ids

            # Compute uncertainty using wrapper
            uncertainty_score = self.model.score(token_ids, output.logprobs)

            candidate = StepCandidate(
                text=step_text,
                token_ids=token_ids,
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                other_data={
                    "uncertainty_score": uncertainty_score,
                    "validity_score": 1.0 / (1.0 + uncertainty_score),
                    "logprobs": self._extract_logprobs(
                        output.token_ids,
                        output.logprobs,
                    ),
                },
                raw_text=output.text,
            )
            result.append(candidate)

        # Record token usage for FLOP calculation
        self._record_generation(result, context_tokens=context_tokens)

        return result

    def __call__(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Callable interface for step generation.

        Delegates to generate_step_candidates(). Token tracking is handled
        at leaf methods (_generate_thinking_step, _generate_structured_step_candidates, etc.)
        """
        return self.generate_step_candidates(request, trajectory, candidates_per_step)
