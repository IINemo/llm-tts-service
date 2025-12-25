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
  - _generate_step_candidates_impl() - Unified step generation for both modes
  - _generate_answer_candidates_impl() - Unified answer generation for both modes
  - _process_generation_output() - Common output processing helper

Token tracking (_record_generation):
- Called at LEAF methods that invoke model.generate() to ensure accurate FLOP calculation
- Each leaf method tracks: context_tokens (input) and output_tokens (generated)
"""

import inspect
import logging
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

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


class CompletionReason(str, Enum):
    """Reason why a trajectory was marked as complete."""

    THINKING_COMPLETE = "thinking_complete"  # </think> found in thinking mode
    EOS_PATTERN = "eos_pattern"  # <end of response> pattern matched
    ANSWER_PATTERN = "answer_pattern"  # <Answer>: or similar pattern matched
    CONTEXT_LIMIT = "context_limit"  # Not enough context for next step + answer


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
        max_answer_tokens: int = 512,
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
        self.max_answer_tokens = max_answer_tokens
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

        # Get min/max step tokens from detector (like thinking mode)
        self.min_step_tokens = getattr(self.detector, "min_step_tokens", 0)
        self.max_step_tokens = getattr(self.detector, "max_step_tokens", 300)

        self.sampling_params = sampling_params or SamplingParams(
            min_tokens=self.min_step_tokens,
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

    def _log_step_scoring(
        self,
        token_ids: List[int],
        stop_reason: Optional[str],
        raw_text: Optional[str] = None,
        step_text: Optional[str] = None,
        scoring_token_count: Optional[int] = None,
        path_idx: Optional[int] = None,
        candidate_idx: Optional[int] = None,
    ) -> None:
        """Log scoring details for a generated step.

        Handles both structured mode (raw_text + step_text) and thinking mode
        (with optional token truncation) logging patterns.

        Args:
            token_ids: Full list of generated token IDs
            stop_reason: vLLM stop reason (e.g., 'length', '<end of response>')
            raw_text: Raw text from model output (structured mode)
            step_text: Full step text with prefix (structured mode)
            scoring_token_count: Number of tokens used for scoring (after truncation)
            path_idx: Path index for batch generation (0-indexed, displayed as 1-indexed)
            candidate_idx: Candidate index for single-path generation
        """
        original_token_count = len(token_ids)
        full_decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        effective_token_count = scoring_token_count or original_token_count
        is_truncated = (
            scoring_token_count is not None
            and scoring_token_count < original_token_count
        )

        # Build prefix for log message
        if path_idx is not None:
            prefix = f"Scoring [path {path_idx + 1}]"
        elif candidate_idx is not None:
            prefix = f"Scoring [{candidate_idx}]"
        else:
            prefix = "Scoring"

        # Build token count string
        if is_truncated:
            token_str = (
                f"{effective_token_count}/{original_token_count} tokens "
                f"(truncated {original_token_count - effective_token_count})"
            )
        else:
            token_str = f"{effective_token_count} tokens"

        # Structured mode: show raw_text and step_text
        if raw_text is not None and step_text is not None:
            log.info(
                f"{prefix}: {token_str}, stop={repr(stop_reason)}\n"
                f"  Full tokens decoded: {repr(full_decoded)}\n"
                f"  Raw text (stripped): {repr(raw_text)}\n"
                f"  Step text:           {repr(step_text)}"
            )
        # Thinking mode with truncation
        elif is_truncated:
            scoring_text = self.tokenizer.decode(
                token_ids[:scoring_token_count], skip_special_tokens=True
            )
            log.info(
                f"{prefix}: {token_str}\n"
                f"  Stop reason: {repr(stop_reason)}\n"
                f"  Full tokens decoded:      {repr(full_decoded)}\n"
                f"  Truncated tokens decoded: {repr(scoring_text)}"
            )
        # Thinking mode without truncation
        else:
            log.info(
                f"{prefix}: {token_str} (full, no truncation)\n"
                f"  Stop reason: {repr(stop_reason)}\n"
                f"  Text: {repr(full_decoded)}"
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
            repetition_penalty=1.05,  # Penalize repetitive text
        )

    # =========================================================================
    # Thinking mode methods
    # =========================================================================

    def _is_thinking_complete(self, text: str) -> bool:
        """Check if thinking phase is complete (contains </think>)."""
        return "</think>" in text

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

    def _find_scoring_token_prefix(
        self,
        token_ids: List[int],
        target_text: str,
    ) -> int:
        """Find the best token prefix length that matches target text.

        vLLM includes stop tokens in token_ids but strips them from output.text.
        This finds the longest token prefix that decodes to match the target text.

        Args:
            token_ids: Full list of generated token IDs
            target_text: The text to match (without stop tokens)

        Returns:
            Best prefix length for scoring (excludes stop tokens)
        """
        original_token_count = len(token_ids)
        best_prefix_len = original_token_count

        if original_token_count > 0:
            target_stripped = target_text.strip()
            for prefix_len in range(original_token_count, 0, -1):
                prefix_tokens = token_ids[:prefix_len]
                prefix_text = self.tokenizer.decode(
                    prefix_tokens, skip_special_tokens=True
                )
                if prefix_text.strip() == target_stripped or target_stripped.startswith(
                    prefix_text.strip()
                ):
                    best_prefix_len = prefix_len
                    break

        return best_prefix_len

    def _create_step_candidate(
        self,
        text: str,
        token_ids: List[int],
        logprobs: List[Dict],
        best_prefix_len: int,
        is_trajectory_complete: bool,
        raw_text: str,
    ) -> StepCandidate:
        """Create a StepCandidate with uncertainty scoring.

        Args:
            text: Final step text (with prefix for structured mode)
            token_ids: Full list of generated token IDs
            logprobs: Logprobs for generated tokens
            best_prefix_len: Number of tokens to use for scoring
            is_trajectory_complete: Whether this completes the trajectory
            raw_text: Original raw text from model output

        Returns:
            StepCandidate with uncertainty and validity scores
        """
        original_token_count = len(token_ids)

        # Compute uncertainty on content tokens only (excluding stop token)
        uncertainty_score = self.model.score(
            token_ids[:best_prefix_len],
            logprobs[:best_prefix_len],
        )

        return StepCandidate(
            text=text,
            token_ids=token_ids[:best_prefix_len],
            is_complete=True,
            is_trajectory_complete=is_trajectory_complete,
            other_data={
                "uncertainty_score": uncertainty_score,
                "validity_score": 1.0 / (1.0 + uncertainty_score),
                "logprobs": self._extract_logprobs(
                    token_ids[:best_prefix_len],
                    logprobs[:best_prefix_len],
                ),
                "original_token_count": original_token_count,
            },
            raw_text=raw_text,
        )

    def _process_generation_output(
        self,
        output,
        final_text: str,
        is_trajectory_complete: bool,
        idx: int,
        target_text: Optional[str] = None,
        raw_text_for_log: Optional[str] = None,
        step_text_for_log: Optional[str] = None,
        path_idx: Optional[int] = None,
    ) -> StepCandidate:
        """Process a single generation output into StepCandidate.

        Combines prefix finding, logging, and candidate creation - the common
        pattern used in both step and answer generation methods.

        Args:
            output: vLLM CompletionOutput object
            final_text: Final text for the candidate (after post-processing)
            is_trajectory_complete: Whether this completes the trajectory
            idx: Candidate index for logging
            target_text: Text to match for token prefix finding (default: output.text)
            raw_text_for_log: Raw text for structured mode logging
            step_text_for_log: Step text for structured mode logging
            path_idx: Path index for batch generation logging

        Returns:
            StepCandidate with uncertainty scoring on truncated tokens
        """
        token_ids = output.token_ids
        logprobs = output.logprobs
        stop_reason = getattr(output, "stop_reason", None)

        # Find best token prefix (exclude stop tokens from scoring)
        scoring_target = target_text if target_text is not None else output.text
        best_prefix_len = self._find_scoring_token_prefix(token_ids, scoring_target)

        # Log scoring details
        self._log_step_scoring(
            token_ids=token_ids,
            stop_reason=stop_reason,
            raw_text=raw_text_for_log,
            step_text=step_text_for_log,
            scoring_token_count=best_prefix_len,
            path_idx=path_idx,
            candidate_idx=idx if path_idx is None else None,
        )

        # Create candidate with truncated tokens
        return self._create_step_candidate(
            text=final_text,
            token_ids=token_ids,
            logprobs=logprobs,
            best_prefix_len=best_prefix_len,
            is_trajectory_complete=is_trajectory_complete,
            raw_text=output.text,
        )

    def _generate_step_candidates_impl(
        self,
        request: List[Dict[str, str]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
    ) -> List[List[StepCandidate]]:
        """Unified step candidate generation for both thinking and structured modes.

        Handles both single-trajectory (best-of-n) and multi-trajectory (self-consistency)
        in a single method using vLLM's batch generation capabilities.

        Args:
            request: Chat messages (same for all trajectories)
            trajectories: List of trajectories. Each trajectory is a list of StepCandidates.
            candidates_per_step: Number of candidates to generate per trajectory.

        Returns:
            List of candidate lists, one per trajectory. Each inner list contains
            candidates_per_step candidates.
        """
        if not trajectories:
            return []

        # Build prompts for all trajectories
        prompts = []
        step_prefixes = []  # Only used for structured mode
        traj_indices = []  # Maps prompt index to trajectory index
        already_complete = {}  # Trajectories that are already complete

        for traj_idx, trajectory in enumerate(trajectories):
            # Check if trajectory is already complete
            if trajectory and trajectory[-1].is_trajectory_complete:
                log.warning(
                    f"Path {traj_idx}: trajectory already complete, skipping. "
                    "Strategy should not include completed trajectories."
                )
                already_complete[traj_idx] = [
                    StepCandidate(
                        text="",
                        token_ids=[],
                        is_complete=True,
                        is_trajectory_complete=True,
                        other_data={"uncertainty_score": 0.0, "validity_score": 1.0},
                        raw_text="",
                    )
                    for _ in range(candidates_per_step)
                ]
                continue

            # Build prompt (mode-specific)
            if self.thinking_mode:
                prompt = self._build_prompt(request, trajectory)
                step_prefix = None
            else:
                prompt = self._apply_chat_template(request, enable_thinking=False)
                if trajectory:
                    trajectory_text = convert_trajectory_to_string(trajectory)
                    prompt = prompt + trajectory_text

                step_number = len(trajectory) + 1

                # Add instruction for longer steps if min_step_tokens is set
                if self.min_step_tokens > 0 and step_number == 1:
                    step_prefix = (
                        f"<start of response>\n"
                        f"(Generate detailed reasoning steps, each step should be at least {self.min_step_tokens} tokens)\n"
                        f"Reasoning Steps:\n- Step 1:"
                    )
                elif step_number == 1:
                    step_prefix = "<start of response>\nReasoning Steps:\n- Step 1:"
                else:
                    step_prefix = f"- Step {step_number}:"
                prompt = prompt + step_prefix

            prompts.append(prompt)
            step_prefixes.append(step_prefix)
            traj_indices.append(traj_idx)

        # If all trajectories complete, return early
        if not prompts:
            return [already_complete[i] for i in range(len(trajectories))]

        total_context_tokens = sum(len(self.tokenizer.encode(p)) for p in prompts)

        # Create sampling params (mode-specific stop tokens)
        if self.thinking_mode:
            sampling_params = self._create_sampling_params(
                stop_tokens=self.thinking_stop_tokens,
                n=candidates_per_step,
                max_tokens=self.max_step_tokens,
                min_tokens=self.min_step_tokens,
            )
        else:
            sampling_params = SamplingParams(
                n=candidates_per_step,
                max_tokens=self.sampling_params.max_tokens,
                min_tokens=self.sampling_params.min_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                logprobs=20,
                stop=self.sampling_params.stop,
                repetition_penalty=1.05,
            )
            if len(trajectories) == 1:
                log.info(
                    f"Structured mode: step {len(trajectories[0]) + 1}, "
                    f"stop={sampling_params.stop}, min_tokens={sampling_params.min_tokens}"
                )
            else:
                log.info(f"Batch generating for {len(prompts)} paths")

        # Generate for all prompts
        outputs = self.model.generate(prompts, sampling_params)

        # Process outputs and organize by trajectory
        candidates_by_traj = {}

        for prompt_idx, (traj_idx, step_prefix, request_output) in enumerate(
            zip(traj_indices, step_prefixes, outputs)
        ):
            trajectory = trajectories[traj_idx]
            candidates = []

            for cand_idx, output in enumerate(request_output.outputs):
                raw_text = output.text
                token_ids = output.token_ids
                stop_reason = getattr(output, "stop_reason", None)

                # Mode-specific text processing and completion detection
                if self.thinking_mode:
                    text = raw_text

                    # Check if thinking phase is complete
                    thinking_complete = "</think>" in text
                    if thinking_complete:
                        think_pos = text.find("</think>")
                        text = text[: think_pos + len("</think>")]

                    # Handle max tokens
                    hit_max_tokens = stop_reason is None or stop_reason == "length"
                    if hit_max_tokens and not thinking_complete:
                        text = self._truncate_at_sentence_boundary(text)

                    # Check for repetitions
                    repetition_detected = self._detect_line_repetitions(text)
                    if repetition_detected:
                        log.warning(
                            f"Path {traj_idx} cand {cand_idx}: repetition detected"
                        )

                    is_trajectory_complete = thinking_complete or repetition_detected
                    step_text = text.strip()
                    target_text = text.strip()

                else:
                    # Structured mode
                    log.debug(
                        f"vLLM output [path {traj_idx}, cand {cand_idx}]: "
                        f"{len(token_ids)} tokens, stop={repr(stop_reason)}"
                    )

                    step_text = step_prefix + raw_text

                    # Handle max tokens / repetition
                    hit_max_tokens = stop_reason is None or stop_reason == "length"
                    if hit_max_tokens:
                        token_count = len(token_ids)
                        truncated_text, was_truncated = self._truncate_repetitions(
                            raw_text, token_count
                        )
                        if was_truncated:
                            raw_text = truncated_text
                            step_text = step_prefix + raw_text

                    # Check for EOS marker
                    stopped_at_eos = stop_reason == "<end of response>"
                    if stopped_at_eos:
                        step_text = step_text + "<end of response>"

                    # Check if stopped at answer pattern
                    stopped_at_answer = False
                    if hasattr(self.detector, "answer_patterns"):
                        for pattern in self.detector.answer_patterns:
                            if stop_reason and pattern in stop_reason:
                                stopped_at_answer = True
                                log.info(
                                    f"Path {traj_idx}: stopped at answer pattern "
                                    f"'{stop_reason}', reasoning complete"
                                )
                                break

                    is_trajectory_complete = (
                        stopped_at_eos
                        or stopped_at_answer
                        or self.detector.is_trajectory_complete(raw_text)
                    )
                    target_text = raw_text

                # Determine completion reason
                completion_reason = None
                if is_trajectory_complete:
                    if self.thinking_mode and "</think>" in step_text:
                        completion_reason = CompletionReason.THINKING_COMPLETE
                    elif not self.thinking_mode:
                        if stopped_at_eos:
                            completion_reason = CompletionReason.EOS_PATTERN
                        elif stopped_at_answer:
                            completion_reason = CompletionReason.ANSWER_PATTERN

                # Process output into candidate
                candidate = self._process_generation_output(
                    output=output,
                    final_text=step_text,
                    is_trajectory_complete=is_trajectory_complete,
                    idx=cand_idx,
                    target_text=target_text,
                    raw_text_for_log=raw_text if not self.thinking_mode else None,
                    step_text_for_log=step_text if not self.thinking_mode else None,
                    path_idx=traj_idx if len(trajectories) > 1 else None,
                )

                if completion_reason:
                    candidate.other_data["completion_reason"] = completion_reason

                candidates.append(candidate)

            # Post-generation context limit check for this trajectory
            if candidates and not candidates[0].is_trajectory_complete:
                context_tokens = len(self.tokenizer.encode(prompts[prompt_idx]))
                max_gen = max(len(c.token_ids) for c in candidates)
                total_tokens = context_tokens + max_gen

                max_step = getattr(self, "max_step_tokens", 300)
                tokens_needed = max_step + self.max_answer_tokens
                remaining = self.max_model_len - total_tokens

                if remaining < tokens_needed:
                    log.warning(
                        f"Path {traj_idx}: context limit, "
                        f"only {remaining} remaining (need {tokens_needed})"
                    )
                    for c in candidates:
                        c.is_trajectory_complete = True
                        c.other_data["completion_reason"] = (
                            CompletionReason.CONTEXT_LIMIT
                        )

            candidates_by_traj[traj_idx] = candidates

        # Merge with already-complete results
        candidates_by_traj.update(already_complete)

        # Build result in original trajectory order
        result = [candidates_by_traj[i] for i in range(len(trajectories))]

        # Flatten for token recording
        all_candidates = [c for cands in result for c in cands]
        self._record_generation(all_candidates, context_tokens=total_context_tokens)

        return result

    def _generate_answer_candidates_impl(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Unified answer candidate generation for both thinking and structured modes.

        Generates N final answer candidates after reasoning is complete.
        Uses self.thinking_mode to branch on mode-specific behavior.

        Args:
            request: Chat messages
            trajectory: Current trajectory (reasoning steps so far)
            candidates_per_step: Number of answer candidates to generate

        Returns:
            List of answer candidates
        """
        trajectory = trajectory or []
        candidates = []

        # Build prompt and sampling params (mode-specific)
        if self.thinking_mode:
            prompt = self._build_prompt(request, trajectory)
            sampling_params = self._create_sampling_params(
                stop_tokens=self.response_stop_tokens,
                n=candidates_per_step,
                max_tokens=self.max_new_tokens,
                min_tokens=0,
            )
            answer_prefix = None
        else:
            # Structured mode: inject <Answer>: prefix
            prompt = self._apply_chat_template(request, enable_thinking=False)
            if trajectory:
                trajectory_text = convert_trajectory_to_string(trajectory)
                prompt = prompt + trajectory_text

            answer_prefix = "<Answer>:"
            prompt = prompt + answer_prefix

            sampling_params = SamplingParams(
                n=candidates_per_step,
                max_tokens=self.max_answer_tokens,
                min_tokens=1,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                logprobs=20,
                stop=["<end of response>"],
            )
            log.info(f"Generating final answer with prefix '{answer_prefix}'")

        context_tokens = len(self.tokenizer.encode(prompt))

        # Generate
        outputs = self.model.generate([prompt], sampling_params)
        request_output = outputs[0]

        # Process outputs
        for idx, output in enumerate(request_output.outputs):
            raw_text = output.text
            stop_reason = getattr(output, "stop_reason", None)

            if self.thinking_mode:
                # Append answer pattern if not present
                text = raw_text
                for pattern in self.answer_patterns:
                    if pattern not in text:
                        text = text + pattern
                        break
                final_text = text.strip()
                target_text = None
            else:
                # Structured mode: prepend <Answer>: prefix
                final_text = answer_prefix + raw_text
                if stop_reason == "<end of response>":
                    final_text = final_text + "<end of response>"
                target_text = raw_text

            candidate = self._process_generation_output(
                output=output,
                final_text=final_text,
                is_trajectory_complete=True,
                idx=idx,
                target_text=target_text,
                raw_text_for_log=raw_text if not self.thinking_mode else None,
                step_text_for_log=final_text if not self.thinking_mode else None,
            )
            candidates.append(candidate)

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

            # Process output into candidate
            candidate = self._process_generation_output(
                output=output,
                final_text=text.strip(),
                is_trajectory_complete=True,
                idx=idx,
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

    # =========================================================================
    # Common interface methods
    # =========================================================================

    def generate_step_candidates(
        self,
        request: List[Dict[str, str]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
    ) -> List[List[StepCandidate]]:
        """Generate N candidate next steps for each trajectory.

        PUBLIC API - Main entry point for step generation. Supports both:
        - Single trajectory (best-of-n): pass [trajectory], get [[cand1, cand2, ...]]
        - Multiple trajectories (self-consistency): pass [traj1, traj2, ...], get [[c1], [c2], ...]

        Args:
            request: Chat messages (same for all trajectories)
            trajectories: List of trajectories. Each trajectory is a list of StepCandidates.
            candidates_per_step: Number of candidates to generate per trajectory.

        Returns:
            List of candidate lists, one per trajectory.

        Note: Strategy should check is_trajectory_complete on returned candidates
        and call generate_answer_candidates() when reasoning phase is done.
        """
        return self._generate_step_candidates_impl(
            request, trajectories, candidates_per_step
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

        In thinking mode, ensures </think> is present before generating answer.
        In structured mode, injects <Answer>: prefix to force answer generation.
        """
        trajectory = trajectory or []

        # Thinking mode: ensure </think> is present
        if self.thinking_mode:
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
                trajectory = trajectory + [close_thinking_step]

        return self._generate_answer_candidates_impl(
            request, trajectory, candidates_per_step
        )

    def generate_answer(
        self, request, candidates_per_step: int, more_information=False
    ) -> List[StepCandidate]:
        """Generate final answer (structured mode compatibility)."""
        result = self.generate_step_candidates(request, [[]], candidates_per_step)
        return result[0] if result else []

    def __call__(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Callable interface for step generation.

        Convenience wrapper that accepts a single trajectory and returns flat list.
        Internally calls generate_step_candidates with [trajectory].
        """
        trajectory = trajectory or []
        result = self.generate_step_candidates(
            request, [trajectory], candidates_per_step
        )
        return result[0] if result else []
