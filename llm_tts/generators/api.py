"""
API-based step candidate generator with full VLLMStepGenerator parity.

Supports all strategies: baseline, online BoN, offline BoN, beam search,
self-consistency, adaptive scaling — using OpenAI-compatible API backends.

Architecture:
- Uses BlackboxModelWithStreaming for API calls (streaming for n=1, batch for n>1)
- Stop tokens derived from ThinkingMarkerDetector (same as vLLM)
- Logprob conversion from API format to lm-polygraph format for uncertainty scoring
- ThreadPoolExecutor for concurrent API calls across trajectories
"""

import copy
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llm_tts.generators.base import (
    CompletionReason,
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector

if TYPE_CHECKING:
    from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
    from llm_tts.utils.flops import FLOPCalculator

log = logging.getLogger(__name__)


# =========================================================================
# Logprob conversion and uncertainty scoring from lm-polygraph
# =========================================================================

from lm_polygraph.utils.api_with_uncertainty import (
    APILogprobData,
    APIWithUncertainty,
    convert_api_logprobs,
)

# Backward compatibility alias
APIUncertaintyScorer = APIWithUncertainty


# =========================================================================
# Main generator class
# =========================================================================


class StepCandidateGeneratorThroughAPI(StepCandidateGeneratorBase):
    """Generates step candidates using OpenAI-compatible API with full strategy support.

    Mirrors VLLMStepGenerator's public API so all strategies (baseline, online BoN,
    offline BoN, beam search, self-consistency, adaptive scaling) work identically.

    Key differences from vLLM:
    - Token counting via tiktoken (no local tokenizer)
    - API calls via ThreadPoolExecutor for parallelism
    - Logprob conversion from API format for uncertainty scoring
    - Streaming (n=1) uses BoundaryEarlyStopping on the model
    - Non-streaming (n>1) uses stop parameter + post-hoc splitting

    Args:
        model: BlackboxModelWithStreaming instance.
        thinking_mode: If True, expect <think>...</think> patterns.
        detector: ThinkingMarkerDetector for step boundary detection.
        answer_patterns: Patterns marking end of response.
        max_new_tokens: Maximum tokens per generation.
        max_answer_tokens: Maximum tokens for answer generation.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter (note: not all API providers support this).
        presence_penalty: Presence penalty.
        max_context_budget: Maximum context length for truncation checks.
        flop_calculator: Optional FLOP calculator.
        prefill_mode: If True, use assistant prefill for trajectory continuation.
        disable_thinking_mode: Controls enable_thinking in chat template.
        supports_logprobs: Whether the API supports logprobs.
    """

    def __init__(
        self,
        model,
        thinking_mode: bool = False,
        detector: Optional[ThinkingMarkerDetector] = None,
        answer_patterns: Optional[List[str]] = None,
        max_new_tokens: int = 4096,
        max_answer_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 0.0,
        max_context_budget: int = 32768,
        flop_calculator: Optional["FLOPCalculator"] = None,
        prefill_mode: bool = False,
        disable_thinking_mode: Optional[bool] = None,
        supports_logprobs: bool = True,
        max_concurrent_requests: int = 256,
    ):
        super().__init__(generation_batch_size=1024, flop_calculator=flop_calculator)

        self.model = model
        self.thinking_mode = thinking_mode
        self.disable_thinking_mode = disable_thinking_mode
        self.prefill_mode = prefill_mode
        self.supports_logprobs = supports_logprobs
        self.max_concurrent_requests = max_concurrent_requests

        # Store generation parameters
        self.max_new_tokens = max_new_tokens
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.max_context_budget = max_context_budget

        # Answer patterns for response phase
        self.answer_patterns = (
            list(answer_patterns) if answer_patterns else ["<end of response>"]
        )

        # Initialize detector and derive stop tokens
        self._init_detector(detector)

        # Initialize tokenizer for token counting
        self._init_tokenizer(getattr(model, "model_path", None))

        log.info(
            f"StepCandidateGeneratorThroughAPI initialized: thinking_mode={thinking_mode}, "
            f"{len(self.stop_tokens)} stop tokens, "
            f"supports_logprobs={supports_logprobs}, "
            f"max_concurrent_requests={max_concurrent_requests}"
        )
        log.info(
            f"Generation parameters: temperature={self.temperature}, "
            f"top_p={self.top_p}, top_k={self.top_k}, "
            f"presence_penalty={self.presence_penalty}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"max_answer_tokens={self.max_answer_tokens}, "
            f"max_context_budget={self.max_context_budget}"
        )

    # =========================================================================
    # Initialization helpers
    # =========================================================================

    def _init_detector(self, detector: Optional[ThinkingMarkerDetector]):
        """Initialize detector and derive stop tokens from it.

        Mirrors VLLMStepGenerator._init_detector().
        """
        if detector is None:
            detector = ThinkingMarkerDetector()

        self.detector = detector
        self.detector.answer_patterns = self.answer_patterns

        # Get min/max step tokens from detector
        self.min_step_tokens = getattr(detector, "min_step_tokens", 0)
        self.max_step_tokens = getattr(detector, "max_step_tokens", 300)

        # Derive stop tokens from detector's use_* flags
        self.stop_tokens = detector.get_vllm_stop_tokens()

        # Add </think> for thinking mode
        if self.thinking_mode and "</think>" not in self.stop_tokens:
            self.stop_tokens.append("</think>")

        # Response stop tokens for answer phase
        self.response_stop_tokens = self.answer_patterns.copy()

    def _init_tokenizer(self, model_name: Optional[str]):
        """Initialize tiktoken tokenizer for token counting."""
        try:
            import tiktoken

            try:
                self._tokenizer = tiktoken.encoding_for_model(model_name or "gpt-4")
            except KeyError:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            self._count_tokens = lambda text: len(self._tokenizer.encode(text))
            log.info(f"Using tiktoken tokenizer for model: {model_name}")
        except ImportError:
            log.warning(
                "tiktoken not available, using approximate token counting (chars/4)"
            )
            self._tokenizer = None
            self._count_tokens = lambda text: len(text) // 4

    # =========================================================================
    # Request preparation
    # =========================================================================

    def _prepare_request(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> List[Dict[str, str]]:
        """Build API request messages from request + trajectory.

        Uses assistant prefill mode when supported, otherwise appends
        trajectory as continuation prompt.
        """
        if not trajectory:
            return list(request)

        request_with_trajectory = copy.deepcopy(request)

        if self.prefill_mode:
            request_with_trajectory.append(
                {
                    "role": "assistant",
                    "content": convert_trajectory_to_string(trajectory),
                    "prefix": True,
                }
            )
        else:
            # Append trajectory text as continuation in assistant role
            trajectory_text = convert_trajectory_to_string(trajectory)
            request_with_trajectory.append(
                {
                    "role": "assistant",
                    "content": trajectory_text,
                }
            )

        return request_with_trajectory

    # =========================================================================
    # Utility methods (ported from VLLMStepGenerator)
    # =========================================================================

    def _detect_line_repetitions(
        self,
        text: str,
        min_lines: int = 4,
        max_unique_ratio: float = 0.3,
    ) -> bool:
        """Detect if text contains excessive line-by-line repetitions."""
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
        """Detect and truncate repetitive text when model hits max tokens."""
        if self._detect_line_repetitions(text):
            return text + "<end of response>", True

        if token_count < min_tokens_for_check:
            return text, False

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        num_sentences = len(lines)
        expected_min = (token_count / 1000) * min_sentences_per_1k_tokens

        if num_sentences < expected_min:
            log.warning(
                f"Detected repetition: only {num_sentences} sentences for "
                f"{token_count} tokens (expected >= {expected_min:.0f}), "
                f"forcing end of response"
            )
            return text + "<end of response>", True

        return text, False

    def _truncate_at_sentence_boundary(self, text: str) -> str:
        """Truncate text at the last sentence boundary."""
        boundaries = [". ", ".\n", "?\n", "? ", "!\n", "! ", "\n\n"]
        last_boundary_pos = -1

        for boundary in boundaries:
            pos = text.rfind(boundary)
            if pos > last_boundary_pos:
                last_boundary_pos = pos

        if last_boundary_pos > 0:
            return text[: last_boundary_pos + 1]
        return text

    def _split_trajectory_into_steps(self, text: str) -> List[str]:
        """Split a complete trajectory into steps using stop tokens.

        Mirrors VLLMStepGenerator._split_trajectory_into_steps().
        """
        if not self.stop_tokens:
            return [text] if text.strip() else []

        escaped_tokens = [re.escape(tok) for tok in self.stop_tokens]
        pattern = r"(?=" + "|".join(escaped_tokens) + ")"
        steps = re.split(pattern, text)
        steps = [s for s in steps if s.strip()]

        log.debug(
            f"Split trajectory into {len(steps)} steps "
            f"using {len(self.stop_tokens)} stop tokens"
        )
        return steps

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

        Mirrors VLLMStepGenerator._log_step_scoring() but uses text directly
        instead of tokenizer.decode() (API doesn't have a local tokenizer).

        Args:
            token_ids: List of generated token IDs (pseudo-IDs for API).
            stop_reason: Stop reason string.
            raw_text: Raw text from API output.
            step_text: Processed step text (after boundary detection).
            scoring_token_count: Number of tokens used for scoring.
            path_idx: Path index for batch generation (0-indexed, displayed as 1-indexed).
            candidate_idx: Candidate index for single-path generation.
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

        # non-thinking mode: show raw_text and step_text
        if raw_text is not None and step_text is not None:
            log.info(
                f"{prefix}: {token_str}, stop={repr(stop_reason)}\n"
                f"  Raw text (stripped): {repr(raw_text[:200])}\n"
                f"  Step text:           {repr(step_text[:200])}"
            )
        # Thinking mode with truncation
        elif is_truncated:
            log.info(
                f"{prefix}: {token_str}\n"
                f"  Stop reason: {repr(stop_reason)}\n"
                f"  Text: {repr((raw_text or '')[:200])}"
            )
        # Default
        else:
            log.info(
                f"{prefix}: {token_str} (full, no truncation)\n"
                f"  Stop reason: {repr(stop_reason)}\n"
                f"  Text: {repr((raw_text or '')[:200])}"
            )

    # =========================================================================
    # Core generation implementation
    # =========================================================================

    def _generate_single_streaming(
        self,
        request_messages: List[Dict[str, str]],
        max_tokens: int,
        call_id: str = "",
    ) -> Dict[str, Any]:
        """Generate a single response via streaming (n=1).

        Uses a fresh BoundaryEarlyStopping instance for step boundary detection
        during streaming. This is safe for concurrent calls because each call
        gets its own early stopping state.

        The detector's full marker set (30+ stop tokens) is used for boundary
        detection — unlike the API's 4-stop limit.

        Args:
            request_messages: Chat messages for the API call.
            max_tokens: Maximum tokens to generate.
            call_id: Context string for logging (e.g. "sample=12 cand=3").

        Returns:
            Dict with text, logprobs, and metadata.
        """
        from llm_tts.early_stopping import BoundaryEarlyStopping

        # Create a fresh early stopping instance per call to avoid shared state
        # across concurrent threads. The detector is stateless and safe to share.
        early_stopping = BoundaryEarlyStopping(detector=self.detector)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                results = self.model.generate_texts(
                    [request_messages],
                    max_new_tokens=max_tokens,
                    temperature=self.temperature,
                    n=1,
                    output_scores=self.supports_logprobs,
                    early_stopping=early_stopping,
                    timeout=300,
                    call_id=call_id,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    log.warning(f"[{call_id}] Streaming call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                    import time
                    time.sleep(wait)
                else:
                    raise

        if not results or len(results) == 0:
            raise ValueError("No result returned from streaming generation")

        result = results[0]

        # Normalize result format
        if isinstance(result, str):
            return {"text": result, "logprobs": [], "finish_reason": "stop"}

        return {
            "text": result.get("text", result.get("raw_collected", "")),
            "logprobs": result.get("logprobs", []),
            "raw_collected": result.get("raw_collected", ""),
            "step_text": result.get("step_text", ""),
            "trajectory_complete": result.get("trajectory_complete", False),
            "finish_reason": result.get("reason", "stop"),
        }

    def _generate_batch(
        self,
        request_messages: List[Dict[str, str]],
        n: int,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        call_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Generate n responses without step-boundary early stopping.

        For n>1: uses the model's non-streaming batch path.
        For n=1: uses the model's streaming path but with early_stopping=None,
        so generation continues until the API's stop sequences or max_tokens.

        Args:
            request_messages: Chat messages for the API call.
            n: Number of completions to generate.
            max_tokens: Maximum tokens to generate.
            stop: Optional stop sequences (passed to API, max 4).
            call_id: Context string for logging (e.g. "sample=12").

        Returns:
            List of dicts, each with text, logprobs, and metadata.
        """
        # Pass early_stopping=None to disable model-level BoundaryEarlyStopping.
        # The batch path relies on the API's stop parameter for stopping, not
        # client-side early stopping. Without this, n=1 calls would take the
        # model's streaming path and BoundaryEarlyStopping would cut generation
        # at the first step boundary — breaking baseline, answer generation, etc.
        max_retries = 5
        for attempt in range(max_retries):
            try:
                results = self.model.generate_texts(
                    [request_messages],
                    max_new_tokens=max_tokens,
                    temperature=self.temperature,
                    n=n,
                    output_scores=self.supports_logprobs,
                    stop=stop,
                    early_stopping=None,
                    timeout=300,
                    call_id=call_id,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    log.warning(f"[{call_id}] Batch call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                    import time
                    time.sleep(wait)
                else:
                    raise

        if not results or len(results) == 0:
            raise ValueError("No result returned from batch generation")

        # For n>1, results is List[List[dict]] — one list per chat
        chat_results = results[0]
        if isinstance(chat_results, dict):
            # Single result wrapped
            chat_results = [chat_results]

        return chat_results

    def _score_candidate(
        self,
        text: str,
        api_logprobs: List[Dict],
    ) -> Dict[str, Any]:
        """Score a candidate using API logprobs and uncertainty scorer.

        Args:
            text: Generated text.
            api_logprobs: Logprob data from API.

        Returns:
            Dict with uncertainty_score, validity_score, token_ids, logprobs.
        """
        has_scorer = hasattr(self.model, "estimator")
        if api_logprobs and has_scorer:
            token_ids, logprobs = convert_api_logprobs(api_logprobs)
            uncertainty_score = self.model.score(token_ids, logprobs)
            flat_logprobs = []
            for tid, lp_dict in zip(token_ids, logprobs):
                if tid in lp_dict:
                    flat_logprobs.append(lp_dict[tid].logprob)
                else:
                    flat_logprobs.append(-100.0)
            return {
                "uncertainty_score": uncertainty_score,
                "validity_score": 1.0 / (1.0 + uncertainty_score),
                "token_ids": token_ids,
                "logprobs": flat_logprobs,
                "raw_logprobs": logprobs,
                "original_token_count": len(token_ids),
            }
        elif api_logprobs:
            # Have logprobs but no scorer — still convert for storage
            token_ids, logprobs = convert_api_logprobs(api_logprobs)
            flat_logprobs = []
            for tid, lp_dict in zip(token_ids, logprobs):
                if tid in lp_dict:
                    flat_logprobs.append(lp_dict[tid].logprob)
                else:
                    flat_logprobs.append(-100.0)
            return {
                "uncertainty_score": 0.0,
                "validity_score": 1.0,
                "token_ids": token_ids,
                "logprobs": flat_logprobs,
                "raw_logprobs": logprobs,
                "original_token_count": len(token_ids),
            }
        else:
            # No logprobs available
            pseudo_ids = list(range(self._count_tokens(text)))
            return {
                "uncertainty_score": 0.0,
                "validity_score": 1.0,
                "token_ids": pseudo_ids,
                "logprobs": [],
                "raw_logprobs": [],
                "original_token_count": len(pseudo_ids),
            }

    def _process_candidate_text(
        self,
        raw_text: str,
        token_count: int,
        is_streaming: bool,
    ) -> tuple:
        """Process generated text: detect completion, handle repetitions.

        Args:
            raw_text: Raw text from API.
            token_count: Approximate token count.
            is_streaming: Whether this came from streaming (n=1) path.

        Returns:
            Tuple of (processed_text, is_trajectory_complete, completion_reason).
        """
        text = raw_text
        is_trajectory_complete = False
        completion_reason = None

        if self.thinking_mode:
            # Check if thinking phase is complete
            thinking_complete = "</think>" in text
            if thinking_complete:
                think_pos = text.find("</think>")
                text = text[: think_pos + len("</think>")]
                is_trajectory_complete = True
                completion_reason = CompletionReason.THINKING_COMPLETE

            # Handle repetitions
            if not thinking_complete and self._detect_line_repetitions(text):
                is_trajectory_complete = True
                completion_reason = CompletionReason.EOS_PATTERN

            # Truncate at sentence boundary if hit max tokens
            if not thinking_complete and token_count >= self.max_step_tokens:
                text = self._truncate_at_sentence_boundary(text)
        else:
            # Non-thinking mode
            truncated_text, was_truncated = self._truncate_repetitions(
                text, token_count
            )
            if was_truncated:
                text = truncated_text

            # Check for answer patterns
            stopped_at_answer = False
            if hasattr(self.detector, "answer_patterns"):
                for pattern in self.detector.answer_patterns:
                    if pattern in text:
                        stopped_at_answer = True
                        break

            is_trajectory_complete = (
                stopped_at_answer
                or self.detector.is_trajectory_complete(text)
            )

            if is_trajectory_complete:
                if stopped_at_answer:
                    completion_reason = CompletionReason.ANSWER_PATTERN
                else:
                    completion_reason = CompletionReason.EOS_PATTERN

        return text, is_trajectory_complete, completion_reason

    def _generate_step_candidates_impl(
        self,
        requests: List[List[Dict[str, str]]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
        stop_tokens_override: Optional[List[str]] = None,
        max_tokens_override: Optional[int] = None,
        compute_uncertainty: bool = True,
        sample_ids: Optional[List] = None,
    ) -> List[List[StepCandidate]]:
        """Unified step candidate generation — mirrors VLLMStepGenerator._generate_step_candidates_impl.

        Handles both single-trajectory and multi-trajectory generation.
        Uses streaming for n=1 (with BoundaryEarlyStopping) and batch API for n>1.

        Args:
            requests: Per-trajectory chat messages.
            trajectories: List of trajectories.
            candidates_per_step: Number of candidates per trajectory.
            stop_tokens_override: Override stop tokens. None = use self.stop_tokens.
            max_tokens_override: Override max tokens. None = use self.max_step_tokens.
            compute_uncertainty: If True, compute uncertainty scores.
            sample_ids: Optional sample IDs for per-sample token tracking.

        Returns:
            List of candidate lists, one per trajectory.
        """
        if not trajectories:
            return []

        effective_stop_tokens = (
            stop_tokens_override
            if stop_tokens_override is not None
            else self.stop_tokens
        )
        effective_max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else self.max_step_tokens
        )

        already_complete = {}
        active_indices = []

        for traj_idx, trajectory in enumerate(trajectories):
            if trajectory and trajectory[-1].is_trajectory_complete:
                log.warning(
                    f"Path {traj_idx}: trajectory already complete, skipping."
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
            active_indices.append(traj_idx)

        if not active_indices:
            return [already_complete[i] for i in range(len(trajectories))]

        # Prepare requests
        prepared_requests = {}
        context_token_counts = {}
        for traj_idx in active_indices:
            prepared = self._prepare_request(requests[traj_idx], trajectories[traj_idx])
            prepared_requests[traj_idx] = prepared
            # Estimate context tokens
            full_text = " ".join(
                m.get("content", "") for m in prepared if m.get("content")
            )
            context_token_counts[traj_idx] = self._count_tokens(full_text)

        total_context_tokens = sum(context_token_counts.values())

        if len(active_indices) == 1:
            traj = trajectories[active_indices[0]]
            log.info(
                f"Step {len(traj) + 1}, "
                f"stop={len(effective_stop_tokens)} tokens, "
                f"candidates={candidates_per_step}"
            )

        candidates_by_traj = {}

        # Routing decision: streaming vs batch API path.
        #
        # Streaming (N concurrent streaming calls with BoundaryEarlyStopping):
        #   Used when stop_tokens_override is None — i.e. step-by-step generation
        #   where the detector's full marker set (30+ tokens) is needed for correct
        #   step boundaries. The OpenAI API only allows 4 stop sequences, so we
        #   can't pass them all via the batch path. Each streaming call gets its
        #   own BoundaryEarlyStopping instance for thread safety.
        #
        # Batch (single API call with n=candidates_per_step):
        #   Used when stop_tokens_override is set — baseline (["<end of response>"]),
        #   offline BoN ([]), self-consistency (["<end of response>"]). These have
        #   0-1 stop tokens which fit within the API's 4-stop limit.
        use_streaming = stop_tokens_override is None

        # API stop parameter for the batch path (max 4 sequences)
        api_stop = None
        if not use_streaming and effective_stop_tokens:
            api_stop = effective_stop_tokens[:4]

        # Progress counter for concurrent generation
        import threading
        _completed_count = [0]
        _count_lock = threading.Lock()
        total_active = len(active_indices)

        def _generate_for_trajectory(traj_idx):
            """Generate candidates for a single trajectory."""
            messages = prepared_requests[traj_idx]
            sample_id = sample_ids[traj_idx] if sample_ids else traj_idx

            if use_streaming:
                # N concurrent streaming calls — each with its own
                # BoundaryEarlyStopping for correct step boundary detection
                results = []
                if candidates_per_step == 1:
                    results.append(self._generate_single_streaming(
                        messages, max_tokens=effective_max_tokens,
                        call_id=f"sample={sample_id} cand=0",
                    ))
                else:
                    # N concurrent streaming calls via ThreadPoolExecutor
                    with ThreadPoolExecutor(
                        max_workers=candidates_per_step
                    ) as cand_executor:
                        futures = [
                            cand_executor.submit(
                                self._generate_single_streaming,
                                messages, effective_max_tokens,
                                f"sample={sample_id} cand={ci}",
                            )
                            for ci in range(candidates_per_step)
                        ]
                        for future in as_completed(futures):
                            results.append(future.result())
            else:
                # Batch path — stop tokens passed directly to API
                results = self._generate_batch(
                    messages,
                    n=candidates_per_step,
                    max_tokens=effective_max_tokens,
                    stop=api_stop,
                    call_id=f"sample={sample_id}",
                )

            # Log progress and all candidates
            with _count_lock:
                _completed_count[0] += 1
                count = _completed_count[0]
            log.info(
                f"[{count}/{total_active}] Sample {sample_id} generated "
                f"{len(results)} candidate(s):"
            )
            for ci, r in enumerate(results):
                r_text = r.get("text", r.get("raw_collected", ""))
                r_tokens = len(r.get("logprobs", []))
                r_reason = r.get("finish_reason", r.get("reason", ""))
                log.info(
                    f"  candidate {ci}: {r_tokens} tokens, "
                    f"stop={repr(r_reason)}, "
                    f"text={repr(r_text)}"
                )
            return traj_idx, results

        # Execute concurrently across trajectories.
        # Cap outer workers so total concurrent connections stays within budget:
        #   streaming: outer × candidates_per_step concurrent API calls
        #   batch: outer × 1 concurrent API calls
        if use_streaming and candidates_per_step > 1:
            outer_workers = min(
                self.max_concurrent_requests // candidates_per_step,
                len(active_indices),
            )
            outer_workers = max(outer_workers, 1)
        else:
            outer_workers = min(self.max_concurrent_requests, len(active_indices))

        if len(active_indices) > 1:
            with ThreadPoolExecutor(max_workers=outer_workers) as executor:
                futures = {
                    executor.submit(_generate_for_trajectory, idx): idx
                    for idx in active_indices
                }
                raw_results = {}
                for future in as_completed(futures):
                    traj_idx, results = future.result()
                    raw_results[traj_idx] = results
        else:
            raw_results = {}
            traj_idx, results = _generate_for_trajectory(active_indices[0])
            raw_results[traj_idx] = results

        # Process raw results into StepCandidates
        for traj_idx in active_indices:
            raw_list = raw_results[traj_idx]
            candidates = []

            for cand_idx, raw in enumerate(raw_list):
                raw_text = raw.get("text", "")
                api_logprobs = raw.get("logprobs", [])
                token_count = len(api_logprobs) if api_logprobs else self._count_tokens(raw_text)

                # For streaming path, use pre-processed results from BoundaryEarlyStopping
                if use_streaming:
                    # Streaming result may have step_text from BoundaryEarlyStopping
                    step_text = raw.get("step_text", "")
                    raw_collected = raw.get("raw_collected", raw_text)
                    traj_complete = raw.get("trajectory_complete", False)

                    if step_text:
                        text = step_text
                    else:
                        text = raw_text

                    # Additional completion checks
                    if not traj_complete:
                        text, traj_complete, completion_reason = self._process_candidate_text(
                            raw_collected or text, token_count, is_streaming=True
                        )
                    else:
                        completion_reason = CompletionReason.EOS_PATTERN
                else:
                    # Batch path — process text for completion
                    text, traj_complete, completion_reason = self._process_candidate_text(
                        raw_text, token_count, is_streaming=False
                    )

                # Score candidate
                scoring_data = self._score_candidate(text, api_logprobs)

                # Determine stop reason string
                stop_reason = None
                if completion_reason:
                    stop_reason = completion_reason.value if hasattr(completion_reason, 'value') else str(completion_reason)
                elif raw.get("finish_reason"):
                    stop_reason = raw["finish_reason"]

                # Log scoring details per candidate
                sample_id = sample_ids[traj_idx] if sample_ids else traj_idx
                self._log_step_scoring(
                    token_ids=scoring_data["token_ids"],
                    stop_reason=stop_reason,
                    raw_text=raw_text,
                    step_text=text if text != raw_text else None,
                    scoring_token_count=scoring_data["original_token_count"],
                    path_idx=traj_idx if len(active_indices) > 1 else None,
                    candidate_idx=cand_idx if len(active_indices) <= 1 else cand_idx,
                )

                candidate = StepCandidate(
                    text=text,
                    token_ids=scoring_data["token_ids"],
                    is_complete=True,
                    is_trajectory_complete=traj_complete,
                    other_data={
                        "uncertainty_score": scoring_data["uncertainty_score"],
                        "validity_score": scoring_data["validity_score"],
                        "logprobs": scoring_data["logprobs"],
                        "raw_logprobs": scoring_data["raw_logprobs"],
                        "original_token_count": scoring_data["original_token_count"],
                    },
                    raw_text=raw_text,
                )

                if completion_reason:
                    candidate.other_data["completion_reason"] = completion_reason

                candidates.append(candidate)

            # Context limit check
            if candidates and not candidates[0].is_trajectory_complete:
                ctx_tokens = context_token_counts.get(traj_idx, 0)
                max_gen = max(
                    c.other_data.get("original_token_count", 0) for c in candidates
                )
                total_tokens = ctx_tokens + max_gen

                max_step = getattr(self, "max_step_tokens", 300)
                tokens_needed = max_step + self.max_answer_tokens
                remaining = self.max_context_budget - total_tokens

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

        # Record token stats
        all_candidates = [c for cands in result for c in cands]
        self._record_generation(all_candidates, context_tokens=total_context_tokens)

        if sample_ids is not None:
            for traj_idx in active_indices:
                sid = sample_ids[traj_idx]
                ctx = context_token_counts.get(traj_idx, 0)
                self.record_sample_tokens(
                    sid, candidates_by_traj[traj_idx], context_tokens=ctx
                )

        return result

    # =========================================================================
    # Public API methods (mirror VLLMStepGenerator)
    # =========================================================================

    def generate_step_candidates(
        self,
        request: List[Dict[str, str]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
        compute_uncertainty: bool = True,
    ) -> List[List[StepCandidate]]:
        """Generate N candidate next steps for each trajectory.

        Same request for all trajectories.
        """
        requests = [request] * len(trajectories)
        return self._generate_step_candidates_impl(
            requests,
            trajectories,
            candidates_per_step,
            compute_uncertainty=compute_uncertainty,
        )

    def generate_step_candidates_batch(
        self,
        requests: List[List[Dict[str, str]]],
        trajectories: List[List[StepCandidate]],
        candidates_per_step: int = 1,
        stop_tokens_override: Optional[List[str]] = None,
        max_tokens_override: Optional[int] = None,
        compute_uncertainty: bool = True,
        sample_ids: Optional[List] = None,
    ) -> List[List[StepCandidate]]:
        """Generate step candidates with per-trajectory requests."""
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
            max_tokens_override=max_tokens_override,
            compute_uncertainty=compute_uncertainty,
            sample_ids=sample_ids,
        )

    def generate_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int = 1,
    ) -> List[StepCandidate]:
        """Generate N final answer candidates.

        Uses response_stop_tokens to stop at answer boundaries.
        All returned candidates are marked as trajectory complete.
        """
        trajectory = trajectory or []

        # For thinking mode, ensure </think> is present
        model_supports_thinking = self.disable_thinking_mode is False
        if self.thinking_mode and model_supports_thinking:
            full_trajectory = convert_trajectory_to_string(trajectory)
            if "</think>" not in full_trajectory:
                log.warning(
                    "generate_answer_candidates called without </think>. Adding it."
                )
                close_thinking_step = StepCandidate(
                    text="\n</think>\n\n<start of response>\nReasoning Steps:\n",
                    token_ids=[],
                    is_complete=True,
                    is_trajectory_complete=True,
                )
                trajectory = trajectory + [close_thinking_step]

        # Generate with answer stop tokens
        # In thinking mode, use max_new_tokens (answer after </think> can be long)
        # In non-thinking mode, use max_answer_tokens (shorter direct answer)
        answer_max_tokens = self.max_new_tokens if self.thinking_mode else self.max_answer_tokens
        result = self._generate_step_candidates_impl(
            requests=[request],
            trajectories=[trajectory],
            candidates_per_step=candidates_per_step,
            stop_tokens_override=self.response_stop_tokens,
            max_tokens_override=answer_max_tokens,
        )

        candidates = result[0] if result else []

        # Mark all as trajectory complete
        for c in candidates:
            c.is_trajectory_complete = True

        return candidates

    def generate_full_trajectories(
        self,
        request: List[Dict[str, str]],
        num_trajectories: int,
        max_tokens: Optional[int] = None,
        split_steps: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate N complete trajectories in batch.

        Optimized for offline best-of-n: generates complete trajectories
        (no step boundaries) and optionally splits them post-hoc.

        Args:
            request: Chat messages for the request.
            num_trajectories: Number of trajectories to generate.
            max_tokens: Maximum tokens per trajectory.
            split_steps: If True, split trajectories into steps post-hoc.

        Returns:
            List of dicts with full_text, steps, token_ids, is_complete.
        """
        max_tokens = max_tokens or self.max_new_tokens

        log.info(
            f"Generating {num_trajectories} full trajectories "
            f"(max_tokens={max_tokens}, stop_tokens=[] (EOS only))"
        )

        raw_results = self._generate_step_candidates_impl(
            requests=[request],
            trajectories=[[]],
            candidates_per_step=num_trajectories,
            stop_tokens_override=[],  # No step boundaries, only EOS
            max_tokens_override=max_tokens,
        )

        candidates = raw_results[0] if raw_results else []
        results = []
        total_output_tokens = 0

        for idx, candidate in enumerate(candidates):
            text = candidate.text
            token_ids = list(candidate.token_ids) if candidate.token_ids else []

            if split_steps:
                steps = self._split_trajectory_into_steps(text)
            else:
                steps = [text]

            total_output_tokens += len(token_ids)

            log.info(
                f"  Trajectory {idx + 1}: {len(token_ids)} tokens, "
                f"complete={candidate.is_trajectory_complete}"
            )

            results.append(
                {
                    "full_text": text,
                    "steps": steps,
                    "token_ids": token_ids,
                    "is_complete": candidate.is_trajectory_complete,
                }
            )

        log.info(
            f"Generated {num_trajectories} trajectories, "
            f"total_output={total_output_tokens} tokens"
        )

        return results

    def count_context_tokens(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> int:
        """Count context tokens for a (request, trajectory) pair.

        Uses tiktoken for estimation.
        """
        traj_text = convert_trajectory_to_string(trajectory)
        prompt_text = " ".join(
            m.get("content", "") for m in request if m.get("content")
        )
        full_text = prompt_text + traj_text
        return self._count_tokens(full_text)

    def generate_answer(
        self, request, candidates_per_step: int, more_information=False
    ) -> List[StepCandidate]:
        """Generate final answer (compatibility method)."""
        result = self.generate_step_candidates(request, [[]], candidates_per_step)
        return result[0] if result else []

    def __call__(
        self,
        request: List[Dict[str, str]],
        trajectory: Optional[List[StepCandidate]] = None,
        candidates_per_step: int = 1,
        compute_uncertainty: bool = True,
    ) -> List[StepCandidate]:
        """Callable interface for step generation.

        Convenience wrapper: single trajectory in, flat candidate list out.
        """
        trajectory = trajectory or []
        result = self.generate_step_candidates(
            request, [trajectory], candidates_per_step, compute_uncertainty
        )
        return result[0] if result else []
