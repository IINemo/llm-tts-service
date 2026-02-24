"""
Unified vLLM step candidate generator with ThinkingMarkerDetector.

Modes:
1. thinking_mode=True: Two-phase generation with <think>...</think>
   - For models with thinking support (e.g., Qwen3)
   - Phase 1: Generate thinking content inside <think>...</think>
   - Phase 2: Generate response after thinking

2. thinking_mode=False: Single-phase generation
   - For models without thinking support (e.g., Qwen2.5-Math)
   - Uses same semantic stop tokens for step boundaries

Both modes use ThinkingMarkerDetector for step boundary detection via semantic
stop tokens (e.g., "Wait,", "Thus,", "First,", etc.).

Architecture Notes:
-------------------
- Uses vLLM's native batch generation (n=candidates_per_step) for efficiency
- Step boundaries enforced via stop tokens and min_tokens parameter
- Uses VLLMWithUncertainty wrapper from lm-polygraph for uncertainty scoring

Token tracking (_record_generation):
- Called at LEAF methods that invoke model.generate() to ensure accurate FLOP calculation
- Each leaf method tracks: context_tokens (input) and output_tokens (generated)
"""

import inspect
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

# Optional vLLM import (not available in CI)
try:
    from vllm import SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    SamplingParams = None

# Optional lm-polygraph imports for uncertainty computation
try:
    from lm_polygraph.utils import VLLMWithUncertainty

    POLYGRAPH_AVAILABLE = True
except ImportError:
    POLYGRAPH_AVAILABLE = False
    VLLMWithUncertainty = None

from llm_tts.generators.base import (
    CompletionReason,
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector

if TYPE_CHECKING:
    from llm_tts.utils.flops import FLOPCalculator

log = logging.getLogger(__name__)


class VLLMStepGenerator(StepCandidateGeneratorBase):
    """
    Unified vLLM step generator supporting both thinking and non-thinking modes.

    Uses VLLMWithUncertainty wrapper from lm-polygraph for both generation and
    uncertainty scoring. The wrapper's score() method computes uncertainty on
    (possibly truncated) tokens.

    Args:
        model: VLLMWithUncertainty wrapper instance (wraps vLLM LLM with scoring)
        thinking_mode: If True, use two-phase thinking generation (for models like Qwen3).
                      If False, use single-phase generation (for models like Qwen2.5-Math).
        detector: ThinkingMarkerDetector for step boundary detection. If None, creates default.
        answer_patterns: Patterns marking end of response
        max_new_tokens: Maximum tokens per generation
        temperature, top_p, top_k: Sampling parameters
        presence_penalty: Penalty for tokens that have already appeared (default 0.0)
        max_context_budget: Maximum context length for truncation
        flop_calculator: Optional FLOP calculator for token tracking
        disable_thinking_mode: If set (not None), controls whether to pass enable_thinking
                               to the chat template. True = enable_thinking=False, False = enable_thinking=True.
                               If None, enable_thinking is not passed at all.
    """

    def __init__(
        self,
        model: "VLLMWithUncertainty",
        thinking_mode: bool = True,
        detector: Optional[ThinkingMarkerDetector] = None,
        answer_patterns: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 0.0,
        max_context_budget: int = 32768,
        flop_calculator: Optional["FLOPCalculator"] = None,
        disable_thinking_mode: Optional[bool] = None,
    ):
        super().__init__(generation_batch_size=1024, flop_calculator=flop_calculator)

        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required but not installed. Install it with: pip install vllm"
            )

        self.model = model  # VLLMWithUncertainty wrapper
        self.thinking_mode = thinking_mode
        self.disable_thinking_mode = disable_thinking_mode
        self.tokenizer = model.get_tokenizer()

        # Store common parameters (internal names differ from config keys)
        self.generation_limit = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.context_budget = max_context_budget

        # Stop token IDs (e.g., [151645, 151643] for Qwen EOS)
        self.stop_token_ids = list(stop_token_ids) if stop_token_ids else None

        # Answer patterns for response phase (default: <end of response>)
        self.answer_patterns = (
            list(answer_patterns) if answer_patterns else ["<end of response>"]
        )

        self._init_detector(detector)

        log.info(
            f"VLLMStepGenerator initialized: thinking_mode={thinking_mode}, "
            f"{len(self.stop_tokens)} stop tokens"
        )
        self._log_stop_tokens_with_strings()
        log.info(
            f"Generation parameters: temperature={self.temperature}, "
            f"top_p={self.top_p}, top_k={self.top_k}, "
            f"presence_penalty={self.presence_penalty}, "
            f"generation_limit={self.generation_limit}, "
            f"context_budget={self.context_budget}"
        )

    def _init_detector(self, detector: Optional[ThinkingMarkerDetector]):
        """Initialize detector and derive stop tokens from it.

        Args:
            detector: ThinkingMarkerDetector for step boundary detection.
                     Stop tokens are derived from detector's use_* flags.
        """
        # Create default detector if not provided
        if detector is None:
            detector = ThinkingMarkerDetector()

        self.detector = detector
        self.detector.answer_patterns = self.answer_patterns

        # Get min/step token limit from detector
        if not hasattr(detector, "min_step_tokens"):
            log.warning("Detector does not have min_step_tokens set, defaulting to 50")
        if not hasattr(detector, "max_step_tokens"):
            log.warning("Detector does not have max_step_tokens set, defaulting to 300")
        self.min_step_tokens = getattr(detector, "min_step_tokens", 50)
        self.step_token_limit = getattr(detector, "max_step_tokens", 300)

        # Derive stop tokens from detector's use_* flags
        self.stop_tokens = detector.get_vllm_stop_tokens()

        # Add </think> for thinking mode
        if self.thinking_mode and "</think>" not in self.stop_tokens:
            self.stop_tokens.append("</think>")

        # Response stop tokens for answer phase (thinking mode only)
        self.response_stop_tokens = self.answer_patterns.copy()

    # =========================================================================
    # Common utility methods
    # =========================================================================

    def _log_stop_tokens_with_strings(self) -> None:
        """Log all stop tokens used for generation, grouped by category."""
        if not self.stop_tokens and not self.stop_token_ids:
            log.info("No stop tokens configured")
            return

        num_ids = len(self.stop_token_ids) if self.stop_token_ids else 0
        log.info(f"Stop tokens: {len(self.stop_tokens)} strings, {num_ids} token IDs")

        # Try to categorize string stop tokens by detector category
        detector = getattr(self, "detector", None)
        if detector and hasattr(detector, "use_sequence"):
            from llm_tts.step_boundary_detectors.thinking.vllm import (
                get_stop_tokens_sentence_start,
            )
            from llm_tts.step_boundary_detectors.thinking.vllm.stop_tokens import (
                ANSWER_TOKENS,
                CONCLUSION_WORDS,
                CORRECTION_WORDS,
                REASONING_WORDS,
                SEQUENCE_WORDS,
                STRUCTURE_TOKENS,
                THINKING_WORDS,
                VERIFICATION_WORDS,
            )

            # Map flag names to word lists
            flag_categories = [
                ("Sequence", "use_sequence", SEQUENCE_WORDS),
                ("Conclusion", "use_conclusion", CONCLUSION_WORDS),
                ("Thinking", "use_thinking", THINKING_WORDS),
                ("Verification", "use_verification", VERIFICATION_WORDS),
                ("Reasoning", "use_reasoning", REASONING_WORDS),
                ("Correction", "use_correction", CORRECTION_WORDS),
                ("Structure", "use_structure", STRUCTURE_TOKENS),
            ]

            # For each enabled category, get its tokens using the same expansion
            # as detector.get_vllm_stop_tokens() (get_stop_tokens_sentence_start)
            all_flags = [
                "use_sequence",
                "use_conclusion",
                "use_thinking",
                "use_verification",
                "use_reasoning",
                "use_correction",
                "use_structure",
            ]
            # ANSWER_TOKENS are always added by get_stop_tokens_sentence_start,
            # exclude them from category matching to avoid misattribution
            answer_tokens = set(ANSWER_TOKENS)
            stop_tokens_set = set(self.stop_tokens)
            accounted = set()

            for cat_name, flag_name, word_list in flag_categories:
                if not getattr(detector, flag_name, False):
                    continue

                # Generate tokens for just this category using the same
                # expansion as detector.get_vllm_stop_tokens()
                kwargs = {f: False for f in all_flags}
                kwargs[flag_name] = True
                cat_tokens = (
                    set(get_stop_tokens_sentence_start(**kwargs)) - answer_tokens
                )

                matched = sorted(stop_tokens_set & cat_tokens)
                accounted.update(matched)

                words_str = ", ".join(word_list)
                log.info(
                    f"  {cat_name}: {len(matched)} tokens "
                    f"from {len(word_list)} words [{words_str}]"
                )

            # Tokens not in any category (e.g., </think>, answer patterns)
            uncategorized = sorted(stop_tokens_set - accounted)
            if uncategorized:
                tokens_str = ", ".join(repr(t) for t in uncategorized)
                log.info(f"  Other: {len(uncategorized)} tokens [{tokens_str}]")
        else:
            # No category info, list all tokens
            for i, token in enumerate(self.stop_tokens):
                log.info(f"  [{i+1}] {repr(token)}")

        # Log token ID stop tokens with decoded strings
        if self.stop_token_ids:
            parts = []
            for token_id in self.stop_token_ids:
                try:
                    decoded = self.tokenizer.decode([token_id])
                    parts.append(f"{token_id}={repr(decoded)}")
                except Exception:
                    parts.append(str(token_id))
            log.info(f"  Token IDs: {', '.join(parts)}")

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

        Handles both non-thinking mode (raw_text + step_text) and thinking mode
        (with optional token truncation) logging patterns.

        Args:
            token_ids: Full list of generated token IDs
            stop_reason: vLLM stop reason (e.g., 'length', '\\n\\n')
            raw_text: Raw text from model output (non-thinking mode)
            step_text: Full step text with prefix (non-thinking mode)
            scoring_token_count: Number of tokens used for scoring (after truncation)
            path_idx: Path index for batch generation (0-indexed, displayed as 1-indexed)
            candidate_idx: Candidate index for single-path generation
        """
        original_token_count = len(token_ids)
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

        # non-thinking mode: show step_text only
        if raw_text is not None and step_text is not None:
            log.info(
                f"{prefix}: {token_str}, stop={repr(stop_reason)}\n"
                f"  Step text: {repr(step_text)}"
            )
        # Thinking mode with truncation
        elif is_truncated:
            scoring_text = self.tokenizer.decode(
                token_ids[:scoring_token_count], skip_special_tokens=True
            )
            log.info(
                f"{prefix}: {token_str}, stop={repr(stop_reason)}\n"
                f"  Step text: {repr(scoring_text)}"
            )
        # Thinking mode without truncation
        else:
            full_decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            log.info(
                f"{prefix}: {token_str}, stop={repr(stop_reason)}\n"
                f"  Step text: {repr(full_decoded)}"
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
            max_tokens=max_tokens or self.generation_limit,
            min_tokens=min_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            logprobs=20,
            stop=stop_tokens,
            stop_token_ids=self.stop_token_ids,
            presence_penalty=self.presence_penalty,
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
        # Only pass enable_thinking if disable_thinking_mode is explicitly set (not None)
        # This avoids issues with models that don't expect this parameter
        if self.disable_thinking_mode is not None:
            # Pass enable_thinking = not disable_thinking_mode
            result = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not self.disable_thinking_mode,
            )
        else:
            # Default: don't pass enable_thinking parameter
            result = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
            )

        return result

    def _find_scoring_token_prefix(
        self,
        token_ids: List[int],
        target_text: str,
    ) -> int:
        """Find the best token prefix length that matches target text.

        vLLM includes stop tokens in token_ids but strips them from output.text.
        This finds the longest token prefix that decodes to match the target text.

        The algorithm scans from full length down, looking for either:
        1. Exact match: decoded prefix == target
        2. Prefix contains target: decoded prefix starts with target (has extra stop tokens)

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
                prefix_stripped = prefix_text.strip()
                # Check: exact match OR prefix contains target (has stop tokens at end)
                if prefix_stripped == target_stripped or prefix_stripped.startswith(
                    target_stripped
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
            text: Final step text (with prefix for non-thinking mode)
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
        # Check for VLLMWithUncertainty wrapper by looking for 'estimator' attribute
        # (raw vLLM has a different score() method for pooling models)
        if hasattr(self.model, "estimator"):
            try:
                uncertainty_score = self.model.score(
                    token_ids[:best_prefix_len],
                    logprobs[:best_prefix_len],
                )
                validity_score = 1.0 / (1.0 + uncertainty_score)
            except Exception as e:
                log.warning(
                    f"Uncertainty scoring failed ({best_prefix_len} tokens): {e}"
                )
                uncertainty_score = None
                validity_score = None
        else:
            uncertainty_score = None
            validity_score = None

        return StepCandidate(
            text=text,
            token_ids=token_ids[:best_prefix_len],
            is_complete=True,
            is_trajectory_complete=is_trajectory_complete,
            other_data={
                "uncertainty_score": uncertainty_score,
                "validity_score": validity_score,
                "logprobs": self._extract_logprobs(
                    token_ids[:best_prefix_len],
                    logprobs[:best_prefix_len],
                ),
                "raw_logprobs": logprobs[:best_prefix_len],
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
        raw_text: Optional[str] = None,
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
            raw_text: Raw text to store in candidate (default: output.text)
            raw_text_for_log: Raw text for non-thinking mode logging
            step_text_for_log: Step text for non-thinking mode logging
            path_idx: Path index for batch generation logging

        Returns:
            StepCandidate with uncertainty scoring on truncated tokens
        """
        token_ids = output.token_ids
        logprobs = output.logprobs
        stop_reason = getattr(output, "stop_reason", None)

        if raw_text is None:
            raw_text = output.text

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
            raw_text=raw_text,
        )

    def _generate_step_candidates_impl(
        self,
        requests: List[List[Dict[str, str]]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
        stop_tokens_override: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        compute_uncertainty: bool = True,
        sample_ids: Optional[List] = None,
        beam_ids: Optional[List] = None,
    ) -> List[List[StepCandidate]]:
        """Unified step candidate generation for both thinking and non-thinking modes.

        Handles both single-trajectory (best-of-n) and multi-trajectory (self-consistency)
        in a single method using vLLM's batch generation capabilities.

        Args:
            requests: Per-trajectory chat messages. Must have len(requests) == len(trajectories).
            trajectories: List of trajectories. Each trajectory is a list of StepCandidates.
            candidates_per_step: Number of candidates to generate per trajectory.
            stop_tokens_override: Override stop tokens (for full trajectory generation).
                                  If None, uses self.stop_tokens.
            max_tokens: Override max tokens. If None, uses self.step_token_limit.
            compute_uncertainty: If True and model supports it, compute uncertainty scores.
                Set to False when using PRM scorer (scores computed separately).
            beam_ids: Optional list mapping each trajectory index to a beam_id.
                Used for logging to identify which beam each trajectory belongs to.

        Returns:
            List of candidate lists, one per trajectory. Each inner list contains
            candidates_per_step candidates.
        """
        if not trajectories:
            return []

        # Build prompts for all trajectories
        prompts = []
        step_prefixes = []  # Only used for non-thinking mode
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
                        other_data={"uncertainty_score": None, "validity_score": None},
                        raw_text="",
                    )
                    for _ in range(candidates_per_step)
                ]
                continue

            # Build prompt (mode-specific)
            if self.thinking_mode:
                prompt = self._build_prompt(requests[traj_idx], trajectory)
            else:
                prompt = self._apply_chat_template(
                    requests[traj_idx], enable_thinking=False
                )
                if trajectory:
                    trajectory_text = convert_trajectory_to_string(trajectory)
                    prompt = prompt + trajectory_text

            step_prefix = None
            prompts.append(prompt)
            step_prefixes.append(step_prefix)
            traj_indices.append(traj_idx)

        # If all trajectories complete, return early
        if not prompts:
            return [already_complete[i] for i in range(len(trajectories))]

        per_prompt_context_tokens = [len(self.tokenizer.encode(p)) for p in prompts]
        total_context_tokens = sum(per_prompt_context_tokens)

        # Use override parameters if provided, otherwise use defaults
        effective_stop_tokens = (
            stop_tokens_override
            if stop_tokens_override is not None
            else self.stop_tokens
        )
        max_tokens = max_tokens if max_tokens is not None else self.step_token_limit

        # Create sampling params with stop tokens
        sampling_params = self._create_sampling_params(
            stop_tokens=effective_stop_tokens,
            n=candidates_per_step,
            max_tokens=max_tokens,
            min_tokens=self.min_step_tokens,
        )

        for traj_i, traj in enumerate(trajectories):
            raw_traj = convert_trajectory_to_string(traj)
            sid = sample_ids[traj_i] if sample_ids else traj_i
            bid = beam_ids[traj_i] if beam_ids else None
            label = f"Sample {sid}" + (f"/Beam {bid}" if bid is not None else "")
            log.info(
                f"{label}: Step {len(traj) + 1}, "
                f"stop={len(effective_stop_tokens)} tokens, min={self.min_step_tokens}\n"
                f"  Raw trajectory ({len(traj)} steps): {repr(raw_traj)}"
            )

        # Generate for all prompts
        # Check if model supports uncertainty computation (VLLMWithUncertainty wrapper)
        use_uncertainty_wrapper = hasattr(self.model, "estimator")
        try:
            if use_uncertainty_wrapper and compute_uncertainty:
                outputs = self.model.generate(
                    prompts, sampling_params, compute_uncertainty=True
                )
            else:
                # Use raw vLLM for PRM scorer or when uncertainty not needed
                raw_llm = getattr(self.model, "llm", self.model)
                outputs = raw_llm.generate(prompts, sampling_params)
        except Exception as e:
            log.error(
                f"vLLM generate failed for {len(prompts)} prompts, "
                f"n={candidates_per_step}, max_tokens={max_tokens}: {e}"
            )
            raise

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
                is_thinking_complete = False
                if self.thinking_mode:
                    text = raw_text

                    # Check if thinking phase is complete
                    # vLLM strips stop strings from output, so check stop_reason too
                    thinking_complete = "</think>" in text or stop_reason == "</think>"
                    # Don't mark trajectory complete — answer phase still needs to run.
                    # Append </think> back to text so downstream can detect it
                    if stop_reason == "</think>" and "</think>" not in text:
                        text = text + "</think>"

                    # Truncate at </think> if it appeared mid-step.
                    # This happens when min_step_tokens prevents vLLM from stopping
                    # at </think> — the model continues into answer phase, leaking
                    # answer tokens into the step. Discard everything after </think>.
                    if thinking_complete and stop_reason != "</think>":
                        think_pos = text.find("</think>")
                        if think_pos >= 0:
                            leaked = len(text) - (think_pos + len("</think>"))
                            if leaked > 0:
                                text = text[: think_pos + len("</think>")]
                                log.info(
                                    f"Path {traj_idx} cand {cand_idx}: "
                                    f"truncated {leaked} chars after </think> "
                                    f"(min_step_tokens bypassed stop)"
                                )

                    if thinking_complete:
                        is_thinking_complete = True

                    # Handle max tokens - only truncate if we actually hit the limit
                    # stop_reason=None can mean EOS token ID stop (complete) or max tokens
                    # Check token count to distinguish
                    actual_hit_max_tokens = stop_reason == "length" or (
                        stop_reason is None and len(token_ids) >= max_tokens
                    )
                    if actual_hit_max_tokens and not thinking_complete:
                        log.warning(
                            f"Path {traj_idx} cand {cand_idx}: hit max tokens "
                            f"({len(token_ids)} >= {max_tokens}), "
                            f"truncating at sentence boundary"
                        )
                        text = self._truncate_at_sentence_boundary(text)

                    # Check for repetitions
                    repetition_detected = self._detect_line_repetitions(text)
                    if repetition_detected:
                        log.warning(
                            f"Path {traj_idx} cand {cand_idx}: repetition detected"
                        )

                    # Stopped at EOS token ID (stop_reason=None but didn't hit max)
                    stopped_at_eos = stop_reason is None and len(token_ids) < max_tokens

                    is_trajectory_complete = repetition_detected or stopped_at_eos
                    step_text = text.strip()
                    target_text = text.strip()

                else:
                    # non-thinking mode
                    log.debug(
                        f"vLLM output [path {traj_idx}, cand {cand_idx}]: "
                        f"{len(token_ids)} tokens, stop={repr(stop_reason)}"
                    )

                    step_text = raw_text

                    # Handle max tokens / repetition
                    hit_max_tokens = stop_reason is None or stop_reason == "length"
                    if hit_max_tokens:
                        token_count = len(token_ids)
                        truncated_text, was_truncated = self._truncate_repetitions(
                            raw_text, token_count
                        )
                        if was_truncated:
                            log.warning(
                                f"Path {traj_idx} cand {cand_idx}: truncated "
                                f"repetitions (hit max tokens: {token_count})"
                            )
                            raw_text = truncated_text
                            step_text = raw_text

                    # Check for natural EOS (stop_reason is None AND didn't hit max tokens)
                    stopped_at_eos = stop_reason is None and len(token_ids) < max_tokens

                    # Check if stopped at answer pattern
                    stopped_at_answer = False
                    if hasattr(self.detector, "answer_patterns"):
                        for pattern in self.detector.answer_patterns:
                            # stop_reason can be string or int (token ID)
                            if (
                                stop_reason
                                and isinstance(stop_reason, str)
                                and pattern in stop_reason
                            ):
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
                if is_thinking_complete:
                    completion_reason = CompletionReason.THINKING_COMPLETE
                elif is_trajectory_complete:
                    if self.thinking_mode:
                        if repetition_detected:
                            completion_reason = CompletionReason.REPETITION_DETECTED
                        elif stopped_at_eos:
                            completion_reason = CompletionReason.EOS_PATTERN
                    else:
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
                    raw_text=text if self.thinking_mode else None,
                    raw_text_for_log=raw_text if not self.thinking_mode else None,
                    step_text_for_log=step_text if not self.thinking_mode else None,
                    path_idx=traj_idx if len(trajectories) > 1 else None,
                )

                if completion_reason:
                    candidate.other_data["completion_reason"] = completion_reason

                candidate.is_thinking_complete = is_thinking_complete
                candidates.append(candidate)

            # Post-generation context limit check for this trajectory
            if candidates and not candidates[0].is_trajectory_complete:
                context_tokens = len(self.tokenizer.encode(prompts[prompt_idx]))
                max_gen = max(
                    c.other_data.get("original_token_count", len(c.token_ids))
                    for c in candidates
                )
                total_tokens = context_tokens + max_gen

                # After thinking is complete, we only need room for the answer
                thinking_done = self.thinking_mode and any(
                    c.is_thinking_complete
                    or c.other_data.get("completion_reason")
                    == CompletionReason.THINKING_COMPLETE
                    for c in candidates
                )
                if self.thinking_mode:
                    # Thinking mode: reserve room for the answer phase.
                    # Answer after </think> is typically short (boxed result),
                    # so 512 tokens is sufficient.
                    answer_reserve = 512
                    tokens_needed = (
                        answer_reserve
                        if thinking_done
                        else self.step_token_limit + answer_reserve
                    )
                else:
                    # Non-thinking: no separate answer phase
                    tokens_needed = self.step_token_limit
                remaining = self.context_budget - total_tokens

                if remaining < tokens_needed:
                    log.warning(
                        f"Path {traj_idx}: context limit reached — "
                        f"used {total_tokens}/{self.context_budget} tokens "
                        f"(prompt={context_tokens}, generated={max_gen}), "
                        f"remaining={remaining}, needed={tokens_needed} "
                        f"({'answer_reserve' if thinking_done else 'step+answer_reserve'})"
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

        # Flatten for aggregate token recording
        all_candidates = [c for cands in result for c in cands]
        self._record_generation(all_candidates, context_tokens=total_context_tokens)

        # Per-sample token recording (when sample_ids provided by strategy)
        if sample_ids is not None:
            for prompt_idx, traj_idx in enumerate(traj_indices):
                sid = sample_ids[traj_idx]
                ctx = per_prompt_context_tokens[prompt_idx]
                self.record_sample_tokens(
                    sid, candidates_by_traj[traj_idx], context_tokens=ctx
                )

        return result

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
        max_tokens = max_tokens or self.generation_limit

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
                is_trajectory_complete=False,
                idx=idx,
            )
            candidate.is_thinking_complete = True
            candidates.append(candidate)

        # Record token usage for FLOP calculation
        self._record_generation(candidates, context_tokens=context_tokens)

        return candidates

    # =========================================================================
    # Full trajectory generation (for offline best-of-n)
    # =========================================================================

    def _split_trajectory_into_steps(self, text: str) -> List[str]:
        """Split a complete trajectory into steps using stop tokens.

        Uses the same stop tokens that would be used during step-by-step generation,
        ensuring consistent step boundaries whether generating incrementally or
        splitting post-hoc.

        Stop tokens mark the START of a new step, so we split BEFORE them.

        Args:
            text: Complete trajectory text

        Returns:
            List of step texts
        """
        import re

        if not self.stop_tokens:
            # No stop tokens defined, return whole text as single step
            return [text] if text.strip() else []

        # Escape special regex characters in stop tokens
        escaped_tokens = [re.escape(tok) for tok in self.stop_tokens]

        # Build pattern that splits BEFORE each stop token (positive lookahead)
        # This keeps the stop token at the start of the next step
        pattern = r"(?=" + "|".join(escaped_tokens) + ")"

        # Split the text
        steps = re.split(pattern, text)

        # Filter empty steps and strip whitespace
        steps = [s for s in steps if s.strip()]

        log.debug(
            f"Split trajectory into {len(steps)} steps using {len(self.stop_tokens)} stop tokens"
        )

        return steps

    # =========================================================================
    # non-thinking mode methods
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

    def count_context_tokens(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> int:
        """Count context tokens for a (request, trajectory) pair.

        Builds the full prompt (chat template + trajectory text) and tokenizes it.

        Args:
            request: Chat messages for the request.
            trajectory: Current trajectory steps.

        Returns:
            Number of context tokens.
        """
        traj_text = convert_trajectory_to_string(trajectory)
        if self.disable_thinking_mode:
            prompt = self._apply_chat_template(request, enable_thinking=False)
        else:
            prompt = self.tokenizer.apply_chat_template(
                request, tokenize=False, add_generation_prompt=True
            )
        if traj_text:
            prompt = prompt + traj_text
        return len(self.tokenizer.encode(prompt))

    def generate_step_candidates_batch(
        self,
        requests: List[List[Dict[str, str]]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
        stop_tokens_override: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        compute_uncertainty: bool = True,
        sample_ids: Optional[List] = None,
        beam_ids: Optional[List] = None,
    ) -> List[List[StepCandidate]]:
        """Generate step candidates with per-trajectory requests.

        PUBLIC API - Like generate_step_candidates but accepts a different request
        per trajectory. Use this when each trajectory may have a different prompt
        (e.g. offline best-of-n, self-consistency with varied prompts).

        Args:
            requests: Per-trajectory chat messages. len(requests) must equal len(trajectories).
            trajectories: List of trajectories. Each trajectory is a list of StepCandidates.
            candidates_per_step: Number of candidates to generate per trajectory.
            stop_tokens_override: Override stop tokens. If None, uses self.stop_tokens.
            max_tokens: Override max tokens. If None, uses self.step_token_limit.
            compute_uncertainty: If True and model supports it, compute uncertainty scores.
            sample_ids: Optional list mapping each trajectory index to a sample_id.
                If provided, per-sample token stats are recorded via record_sample_tokens().
                Multiple trajectories can map to the same sample_id (e.g., beam search).
            beam_ids: Optional list mapping each trajectory index to a beam_id.
                Used for logging to identify which beam each trajectory belongs to.

        Returns:
            List of candidate lists, one per trajectory.
        """
        if len(requests) != len(trajectories):
            raise ValueError(
                f"requests and trajectories must have the same length, "
                f"got {len(requests)} and {len(trajectories)}"
            )
        return self._generate_step_candidates_impl(
            requests,
            trajectories,
            candidates_per_step,
            stop_tokens_override=stop_tokens_override,
            max_tokens=max_tokens,
            compute_uncertainty=compute_uncertainty,
            sample_ids=sample_ids,
            beam_ids=beam_ids,
        )

    def generate_answer_candidates_batch(
        self,
        requests: List[List[Dict[str, str]]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
        sample_ids: Optional[List] = None,
    ) -> List[List[StepCandidate]]:
        """Generate answer candidates for multiple trajectories in one call."""
        if len(requests) != len(trajectories):
            raise ValueError(
                f"requests and trajectories must have same length, "
                f"got {len(requests)} and {len(trajectories)}"
            )

        if not requests:
            return []

        # Pre-process: close </think> if needed
        model_supports_thinking = self.disable_thinking_mode is False
        processed_trajectories = []
        for trajectory in trajectories:
            trajectory = trajectory or []
            if self.thinking_mode and model_supports_thinking:
                full_trajectory = convert_trajectory_to_string(trajectory)
                if "</think>" not in full_trajectory:
                    log.warning(
                        "generate_answer_candidates_batch: trajectory missing "
                        "</think>. Adding closing step."
                    )
                    close_thinking_step = StepCandidate(
                        text="</think>",
                        token_ids=[],
                        is_complete=True,
                        is_trajectory_complete=False,
                        is_thinking_complete=True,
                    )
                    trajectory = trajectory + [close_thinking_step]
            processed_trajectories.append(trajectory)

        answer_max_tokens = self.generation_limit  # Only called in thinking mode
        results = self._generate_step_candidates_impl(
            requests=requests,
            trajectories=processed_trajectories,
            candidates_per_step=candidates_per_step,
            stop_tokens_override=self.response_stop_tokens,
            max_tokens=answer_max_tokens,
            sample_ids=sample_ids,
        )

        # Post-process: mark complete and append answer patterns for thinking mode
        for candidates in results:
            for c in candidates:
                c.is_trajectory_complete = True
                c.is_thinking_complete = True
                if self.thinking_mode:
                    for pattern in self.answer_patterns:
                        if pattern not in c.text:
                            c.text = c.text + pattern
                            break

        return results

    def __call__(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
        compute_uncertainty: bool = True,
    ) -> List[StepCandidate]:
        """Callable interface for step generation.

        Convenience wrapper that accepts a single trajectory and returns flat list.
        """
        trajectory = trajectory or []
        result = self.generate_step_candidates_batch(
            [request],
            [trajectory],
            candidates_per_step,
            compute_uncertainty=compute_uncertainty,
        )
        return result[0] if result else []
