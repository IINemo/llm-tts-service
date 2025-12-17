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
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

# lm-polygraph imports for uncertainty computation
from lm_polygraph.estimators.perplexity import Perplexity
from lm_polygraph.estimators.token_entropy import MeanTokenEntropy
from vllm import LLM, SamplingParams

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

    Uncertainty estimation is done locally using perplexity/entropy from logprobs,
    or optionally via an external scorer for truncated sequences.

    Args:
        model: vLLM model instance (LLM)
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
        scorer: Optional scorer for truncated sequence scoring
    """

    def __init__(
        self,
        model: LLM,
        thinking_mode: bool = True,
        # Detector (works for both modes)
        detector: Optional[
            Union[StructuredStepDetector, ThinkingMarkerDetector]
        ] = None,
        # Structured mode parameters
        sampling_params: Optional[SamplingParams] = None,
        # Common parameters
        answer_patterns: Optional[List[str]] = None,
        # Common generation parameters
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        # Context length limit
        max_model_len: int = 32768,
        # FLOP calculator for token tracking
        flop_calculator: Optional["FLOPCalculator"] = None,
        # Optional scorer for truncated sequence scoring
        scorer=None,
    ):
        # vLLM handles batching internally, so generation_batch_size is large
        super().__init__(generation_batch_size=1024, flop_calculator=flop_calculator)

        self.model = model
        self.thinking_mode = thinking_mode
        self.tokenizer = model.get_tokenizer()
        self.scorer = scorer  # Optional scorer for truncated sequences

        # Initialize lm-polygraph estimators once (avoid repeated initialization)
        self._perplexity_estimator = Perplexity()
        self._entropy_estimator = MeanTokenEntropy()

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

    def _compute_uncertainty_metrics(
        self, token_ids: List[int], logprobs: List[Dict]
    ) -> Dict[str, float]:
        """Compute uncertainty metrics using lm-polygraph estimators.

        Note: vLLM only provides top-k logprobs (configured via logprobs parameter).
        - Perplexity: accurate (uses selected token logprobs only)
        - Mean entropy: approximate (uses top-k logprobs, not full vocabulary)

        For full-vocabulary uncertainty, use HybridStepGenerator or HuggingFace.

        Args:
            token_ids: Generated token IDs
            logprobs: List of logprob dicts from vLLM output

        Returns:
            Dict with 'perplexity' and 'mean_entropy' keys
        """
        if not logprobs or not token_ids:
            return {"perplexity": 0.0, "mean_entropy": 0.0}

        # Extract selected token log probs for Perplexity
        log_likelihoods = []
        for token, logprob_dict in zip(token_ids, logprobs):
            if token in logprob_dict:
                log_likelihoods.append(logprob_dict[token].logprob)
            else:
                log_likelihoods.append(-100.0)  # Very unlikely token

        # Compute perplexity using cached lm-polygraph estimator
        stats_ppl = {"greedy_log_likelihoods": [log_likelihoods]}
        neg_mean_ll = float(self._perplexity_estimator(stats_ppl)[0])
        perplexity = float(np.exp(neg_mean_ll))

        # Compute entropy from top-k logprobs (approximate)
        # This is approximate because vLLM only gives top-k logprobs
        entropies = []
        for logprob_dict in logprobs:
            # Get all log probs from the dict
            lps = np.array([v.logprob for v in logprob_dict.values()])
            # Compute entropy from available logprobs
            mask = ~np.isinf(lps)
            if mask.any():
                entropies.append(-np.sum(lps[mask] * np.exp(lps[mask])))
            else:
                entropies.append(0.0)

        # Compute mean entropy using cached lm-polygraph estimator
        stats_entropy = {"entropy": [entropies]}
        mean_entropy = float(self._entropy_estimator(stats_entropy)[0])

        return {"perplexity": perplexity, "mean_entropy": mean_entropy}

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
            Dict with 'perplexity', 'mean_entropy', and 'uncertainty_score' keys
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
            return {"perplexity": 1.0, "mean_entropy": 0.0, "uncertainty_score": 1.0}

        # Extract logprobs for the truncated_text portion only
        prompt_tokens = len(self.tokenizer.encode(prompt))
        truncated_logprobs = request_output.prompt_logprobs[prompt_tokens:]

        if not truncated_logprobs:
            return {"perplexity": 1.0, "mean_entropy": 0.0, "uncertainty_score": 1.0}

        # Convert prompt_logprobs to format expected by _compute_uncertainty_metrics
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
            return {"perplexity": 1.0, "mean_entropy": 0.0, "uncertainty_score": 1.0}

        # Compute uncertainty metrics
        metrics = self._compute_uncertainty_metrics(valid_token_ids, valid_logprobs)
        metrics["uncertainty_score"] = 1.0 / (1.0 + metrics["mean_entropy"])

        return metrics

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

    def _is_valid_step_boundary(self, text: str, num_tokens: int) -> bool:
        """Check if text represents a valid step boundary (thinking mode)."""
        if num_tokens < self.min_step_tokens:
            return False

        text = text.strip()
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
        last_request_output = None  # Track for uncertainty extraction

        for attempt in range(max_continuation_attempts):
            remaining_tokens = self.max_new_tokens - len(accumulated_tokens)
            if remaining_tokens <= 0:
                break

            sampling_params = self._create_sampling_params(
                stop_tokens=self.thinking_stop_tokens,
                n=1,
                max_tokens=remaining_tokens,
                min_tokens=self.min_step_tokens if attempt == 0 else 0,
            )

            current_prompt = prompt + accumulated_text
            outputs = self.model.generate([current_prompt], sampling_params)
            last_request_output = outputs[0]  # Store for uncertainty
            output = last_request_output.outputs[0]

            # Log raw vLLM output
            stop_reason = getattr(output, "stop_reason", None)
            log.debug(
                f"vLLM output: {len(output.token_ids)} tokens, stop={repr(stop_reason)}\n"
                f"  Raw text: {repr(output.text[-80:] if len(output.text) > 80 else output.text)}"
            )

            accumulated_text += output.text
            accumulated_tokens.extend(output.token_ids)
            if output.logprobs:
                accumulated_logprobs.extend(output.logprobs)

            if self._is_thinking_complete(accumulated_text):
                log.debug(f"Thinking complete (</think>) after {attempt + 1} attempts")
                think_pos = accumulated_text.find("</think>")
                accumulated_text = accumulated_text[: think_pos + len("</think>")]
                break

            if self._is_valid_step_boundary(accumulated_text, len(accumulated_tokens)):
                log.debug(f"Valid thinking step boundary after {attempt + 1} attempts")
                break

            if len(accumulated_tokens) >= self.max_step_tokens:
                log.debug(f"Max step tokens reached after {attempt + 1} attempts")
                break

            log.debug(
                f"Attempt {attempt + 1}: Invalid boundary "
                f"(tokens={len(accumulated_tokens)}, min={self.min_step_tokens}), continuing..."
            )

        thinking_complete = self._is_thinking_complete(accumulated_text)

        # Get stop reason from last output (what triggered the stop)
        stop_reason = getattr(output, "stop_reason", None) if output else None

        # Check if text was truncated (e.g., at </think> boundary)
        # If truncated, find how many tokens to keep by decoding prefixes
        truncated_text = accumulated_text.strip()
        original_token_count = len(accumulated_tokens) if accumulated_tokens else 0

        if original_token_count > 0:
            # Find the longest token prefix that decodes to a prefix of truncated_text
            # This handles whitespace stripping and mid-token truncation correctly
            best_prefix_len = original_token_count  # Default: use all tokens

            for prefix_len in range(original_token_count, 0, -1):
                prefix_tokens = accumulated_tokens[:prefix_len]
                prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)

                # Check if this prefix is contained in the truncated text
                # (allowing for whitespace differences)
                if prefix_text.strip() == truncated_text or truncated_text.startswith(
                    prefix_text.strip()
                ):
                    best_prefix_len = prefix_len
                    break

            # Log scoring details - decode both full tokens and truncated for comparison
            full_decoded = self.tokenizer.decode(
                accumulated_tokens, skip_special_tokens=True
            )
            scoring_text = self.tokenizer.decode(
                accumulated_tokens[:best_prefix_len], skip_special_tokens=True
            )
            if best_prefix_len < original_token_count:
                log.info(
                    f"Scoring: {best_prefix_len}/{original_token_count} tokens (truncated {original_token_count - best_prefix_len})\n"
                    f"  Full tokens decoded:      {repr(full_decoded)}\n"
                    f"  Truncated tokens decoded: {repr(scoring_text)}"
                )
            else:
                log.info(
                    f"Scoring: {best_prefix_len} tokens (full, no truncation)\n"
                    f"  Text: {repr(full_decoded)}"
                )

            uncertainty_metrics = self._compute_uncertainty_metrics(
                accumulated_tokens[:best_prefix_len],
                accumulated_logprobs[:best_prefix_len],
            )
        else:
            # No tokens - return default metrics
            log.warning("No tokens to score, using default metrics")
            uncertainty_metrics = {"perplexity": 1.0, "mean_entropy": 0.0}

        return StepCandidate(
            text=accumulated_text.strip(),
            token_ids=accumulated_tokens,
            is_complete=True,
            is_trajectory_complete=thinking_complete,
            other_data={
                # Convert mean_entropy to validity score (lower entropy = higher score)
                "uncertainty_score": 1.0 / (1.0 + uncertainty_metrics["mean_entropy"]),
                "perplexity": uncertainty_metrics["perplexity"],
                "mean_entropy": uncertainty_metrics["mean_entropy"],
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
        request_output = outputs[0]

        for idx, output in enumerate(request_output.outputs):
            text = output.text

            for pattern in self.answer_patterns:
                if pattern not in text:
                    text = text + pattern
                    break

            # Compute uncertainty metrics using lm-polygraph
            uncertainty_metrics = self._compute_uncertainty_metrics(
                output.token_ids, output.logprobs
            )

            candidate = StepCandidate(
                text=text.strip(),
                token_ids=output.token_ids,
                is_complete=True,
                is_trajectory_complete=True,
                other_data={
                    # Convert mean_entropy to validity score (lower entropy = higher score)
                    "uncertainty_score": 1.0
                    / (1.0 + uncertainty_metrics["mean_entropy"]),
                    "perplexity": uncertainty_metrics["perplexity"],
                    "mean_entropy": uncertainty_metrics["mean_entropy"],
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
        request_output = outputs[0]
        candidates = []

        for idx, output in enumerate(request_output.outputs):
            text = output.text

            if "</think>" not in text:
                text = text + "</think>"

            # Compute uncertainty metrics using lm-polygraph
            uncertainty_metrics = self._compute_uncertainty_metrics(
                output.token_ids, output.logprobs
            )

            candidate = StepCandidate(
                text=text.strip(),
                token_ids=output.token_ids,
                is_complete=True,
                is_trajectory_complete=True,
                other_data={
                    # Convert mean_entropy to validity score (lower entropy = higher score)
                    "uncertainty_score": 1.0
                    / (1.0 + uncertainty_metrics["mean_entropy"]),
                    "perplexity": uncertainty_metrics["perplexity"],
                    "mean_entropy": uncertainty_metrics["mean_entropy"],
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
        outputs = self.model.generate(request, sampling_params=self.sampling_params)
        request_output = outputs[0]
        answers = request_output.outputs

        for i in range(candidates_per_step):
            step_text = answers[i].text
            is_complete = True

            is_trajectory_complete = self.detector.is_trajectory_complete(
                answers[i].text
            )
            token_ids = answers[i].token_ids

            # Compute uncertainty metrics using lm-polygraph
            uncertainty_metrics = self._compute_uncertainty_metrics(
                token_ids, answers[i].logprobs
            )

            candidate = StepCandidate(
                text=step_text,
                token_ids=token_ids,
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                other_data={
                    # Convert mean_entropy to validity score (lower entropy = higher score)
                    "uncertainty_score": 1.0
                    / (1.0 + uncertainty_metrics["mean_entropy"]),
                    "perplexity": uncertainty_metrics["perplexity"],
                    "mean_entropy": uncertainty_metrics["mean_entropy"],
                    "logprobs": self._extract_logprobs(
                        answers[i].token_ids, answers[i].logprobs
                    ),
                },
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
        result = []
        for i in range(len(requests)):
            request_output = vllm_outputs[i]
            output = request_output.outputs[0]
            step_text = output.text
            is_complete = True

            is_trajectory_complete = self.detector.is_trajectory_complete(output.text)
            token_ids = output.token_ids

            # Compute uncertainty metrics using lm-polygraph
            uncertainty_metrics = self._compute_uncertainty_metrics(
                token_ids, output.logprobs
            )

            candidate = StepCandidate(
                text=step_text,
                token_ids=token_ids,
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                other_data={
                    # Convert mean_entropy to validity score (lower entropy = higher score)
                    "uncertainty_score": 1.0
                    / (1.0 + uncertainty_metrics["mean_entropy"]),
                    "perplexity": uncertainty_metrics["perplexity"],
                    "mean_entropy": uncertainty_metrics["mean_entropy"],
                    "logprobs": self._extract_logprobs(
                        output.token_ids,
                        output.logprobs,
                    ),
                },
                raw_text=output.text,
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
