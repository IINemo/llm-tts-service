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

from lm_polygraph.utils.api_with_uncertainty import (
    APIWithUncertainty,
    convert_api_logprobs,
)

from llm_tts.generators.base import (
    CompletionReason,
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector

if TYPE_CHECKING:
    from llm_tts.models.blackboxmodel_with_streaming import (  # noqa: F401
        BlackboxModelWithStreaming,
    )
    from llm_tts.utils.flops import FLOPCalculator

log = logging.getLogger(__name__)

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

        # Store generation parameters (internal names differ from config keys)
        self.generation_limit = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.context_budget = max_context_budget

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
            f"generation_limit={self.generation_limit}, "
            f"context_budget={self.context_budget}"
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

    def _format_step_scoring(
        self,
        token_ids: List[int],
        stop_reason: Optional[str],
        raw_text: Optional[str] = None,
        step_text: Optional[str] = None,
        scoring_token_count: Optional[int] = None,
        path_idx: Optional[int] = None,
        candidate_idx: Optional[int] = None,
        sample_id: Optional[int] = None,
        validity_score: Optional[float] = None,
    ) -> str:
        """Format scoring details for a generated step into a single log line.

        Args:
            token_ids: List of generated token IDs (pseudo-IDs for API).
            stop_reason: Stop reason string.
            raw_text: Raw text from API output (full accumulated text).
            step_text: Processed step text (after boundary detection).
            scoring_token_count: Number of tokens used for scoring.
            path_idx: Path index for batch generation (0-indexed, displayed as 1-indexed).
            candidate_idx: Candidate index for single-path generation.
            sample_id: Sample ID for identification in logs.
            validity_score: Validity score from uncertainty scoring.
        """
        original_token_count = len(token_ids)
        effective_token_count = scoring_token_count or original_token_count
        is_truncated = (
            scoring_token_count is not None
            and scoring_token_count < original_token_count
        )

        # Build prefix for log message
        sample_tag = f"sample={sample_id} " if sample_id is not None else ""
        if path_idx is not None:
            prefix = f"Scoring [{sample_tag}path {path_idx + 1}]"
        elif candidate_idx is not None:
            prefix = f"Scoring [{sample_tag}cand={candidate_idx}]"
        else:
            prefix = f"Scoring [{sample_tag.rstrip()}]" if sample_tag else "Scoring"

        # Build token count string
        if is_truncated:
            token_str = (
                f"{effective_token_count}/{original_token_count} tokens "
                f"(truncated {original_token_count - effective_token_count})"
            )
        else:
            token_str = f"{effective_token_count} tokens"

        # Extract actual stop token from raw_text vs step_text difference
        stop_token_repr = repr(stop_reason)
        if raw_text and step_text and raw_text != step_text:
            actual_stop = raw_text[len(step_text) :]
            if actual_stop:
                stop_token_repr = repr(actual_stop)

        # Score string
        score_str = (
            f", score={validity_score:.3f}" if validity_score is not None else ""
        )

        return (
            f"  {prefix}: {token_str}, stop={stop_token_repr}{score_str}\n"
            f"    Step text: {repr(step_text or raw_text or '')}"
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
                    wait = 2**attempt
                    log.warning(
                        f"[{call_id}] Streaming call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s..."
                    )
                    import time

                    time.sleep(wait)
                else:
                    log.error(
                        f"[{call_id}] Streaming call failed after {max_retries} attempts: {e}"
                    )
                    raise

        if not results or len(results) == 0:
            raise ValueError(
                f"[{call_id}] No result returned from streaming generation"
            )

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
                    wait = 2**attempt
                    log.warning(
                        f"[{call_id}] Batch call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s..."
                    )
                    import time

                    time.sleep(wait)
                else:
                    log.error(
                        f"[{call_id}] Batch call failed after {max_retries} attempts (n={n}): {e}"
                    )
                    raise

        if not results or len(results) == 0:
            raise ValueError(
                f"[{call_id}] No result returned from batch generation (n={n})"
            )

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
        if api_logprobs:
            try:
                token_ids, logprobs = convert_api_logprobs(api_logprobs)
            except Exception as e:
                log.warning(
                    f"convert_api_logprobs failed ({len(api_logprobs)} entries): {e}"
                )
                pseudo_ids = list(range(self._count_tokens(text)))
                return {
                    "uncertainty_score": None,
                    "validity_score": None,
                    "token_ids": pseudo_ids,
                    "logprobs": [],
                    "raw_logprobs": [],
                    "original_token_count": len(pseudo_ids),
                }
            flat_logprobs = []
            for tid, lp_dict in zip(token_ids, logprobs):
                if tid in lp_dict:
                    flat_logprobs.append(lp_dict[tid].logprob)
                else:
                    flat_logprobs.append(-100.0)
            if has_scorer:
                try:
                    uncertainty_score = self.model.score(token_ids, logprobs)
                    validity_score = 1.0 / (1.0 + uncertainty_score)
                except Exception as e:
                    log.warning(
                        f"Uncertainty scoring failed ({len(token_ids)} tokens): {e}"
                    )
                    uncertainty_score = None
                    validity_score = None
            else:
                uncertainty_score = None
                validity_score = None
            return {
                "uncertainty_score": uncertainty_score,
                "validity_score": validity_score,
                "token_ids": token_ids,
                "logprobs": flat_logprobs,
                "raw_logprobs": logprobs,
                "original_token_count": len(token_ids),
            }
        else:
            # No logprobs available
            pseudo_ids = list(range(self._count_tokens(text)))
            return {
                "uncertainty_score": None,
                "validity_score": None,
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
            Tuple of (processed_text, is_trajectory_complete, completion_reason, is_thinking_complete).
        """
        text = raw_text
        is_trajectory_complete = False
        is_thinking_complete = False
        completion_reason = None

        if self.thinking_mode:
            # Check if thinking phase is complete
            thinking_complete = "</think>" in text
            if thinking_complete:
                think_pos = text.find("</think>")
                text = text[: think_pos + len("</think>")]
                is_thinking_complete = True
                completion_reason = CompletionReason.THINKING_COMPLETE

            # Handle repetitions
            if not thinking_complete and self._detect_line_repetitions(text):
                is_trajectory_complete = True
                completion_reason = CompletionReason.EOS_PATTERN

            # Truncate at sentence boundary if hit max tokens
            if not thinking_complete and token_count >= self.step_token_limit:
                log.warning(
                    f"API generation hit max tokens "
                    f"({token_count} >= {self.step_token_limit}), "
                    f"truncating at sentence boundary"
                )
                text = self._truncate_at_sentence_boundary(text)
                # Match vLLM behavior: mark trajectory complete when max tokens
                # exhausted without </think> (model ran out of budget)
                if not is_trajectory_complete:
                    is_trajectory_complete = True
                    completion_reason = CompletionReason.CONTEXT_LIMIT
        else:
            # Non-thinking mode
            truncated_text, was_truncated = self._truncate_repetitions(
                text, token_count
            )
            if was_truncated:
                log.warning(
                    f"API generation truncated repetitions (hit max tokens: {token_count})"
                )
                text = truncated_text

            # Check for answer patterns
            stopped_at_answer = False
            if hasattr(self.detector, "answer_patterns"):
                for pattern in self.detector.answer_patterns:
                    if pattern in text:
                        stopped_at_answer = True
                        break

            is_trajectory_complete = (
                stopped_at_answer or self.detector.is_trajectory_complete(text)
            )

            if is_trajectory_complete:
                if stopped_at_answer:
                    completion_reason = CompletionReason.ANSWER_PATTERN
                else:
                    completion_reason = CompletionReason.EOS_PATTERN

        return text, is_trajectory_complete, completion_reason, is_thinking_complete

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
        """Unified step candidate generation — mirrors VLLMStepGenerator._generate_step_candidates_impl.

        Handles both single-trajectory and multi-trajectory generation.
        Uses streaming for n=1 (with BoundaryEarlyStopping) and batch API for n>1.

        Args:
            requests: Per-trajectory chat messages.
            trajectories: List of trajectories.
            candidates_per_step: Number of candidates per trajectory.
            stop_tokens_override: Override stop tokens. None = use self.stop_tokens.
            max_tokens: Override max tokens. None = use self.step_token_limit.
            compute_uncertainty: If True, compute uncertainty scores.
            sample_ids: Optional sample IDs for per-sample token tracking.
            beam_ids: Optional list mapping each trajectory index to a beam_id.
                Used for logging to identify which beam each trajectory belongs to.

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
        max_tokens = max_tokens if max_tokens is not None else self.step_token_limit

        already_complete = {}
        active_indices = []

        for traj_idx, trajectory in enumerate(trajectories):
            if trajectory and trajectory[-1].is_trajectory_complete:
                log.warning(f"Path {traj_idx}: trajectory already complete, skipping.")
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
            traj_idx_0 = active_indices[0]
            traj = trajectories[traj_idx_0]
            sid = sample_ids[traj_idx_0] if sample_ids else traj_idx_0
            bid = beam_ids[traj_idx_0] if beam_ids else None
            label = f"Sample {sid}" + (f"/Beam {bid}" if bid is not None else "")
            raw_traj = convert_trajectory_to_string(traj)
            log.info(
                f"{label}: Step {len(traj) + 1}, "
                f"stop={len(effective_stop_tokens)} tokens, "
                f"candidates={candidates_per_step}\n"
                f"  Raw trajectory ({len(traj)} steps): {repr(raw_traj)}"
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
                    results.append(
                        self._generate_single_streaming(
                            messages,
                            max_tokens=max_tokens,
                            call_id=f"sample={sample_id} cand=0",
                        )
                    )
                else:
                    # N concurrent streaming calls via ThreadPoolExecutor
                    with ThreadPoolExecutor(
                        max_workers=candidates_per_step
                    ) as cand_executor:
                        futures = [
                            cand_executor.submit(
                                self._generate_single_streaming,
                                messages,
                                max_tokens,
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
                    max_tokens=max_tokens,
                    stop=api_stop,
                    call_id=f"sample={sample_id}",
                )

            # Log progress and all candidates
            with _count_lock:
                _completed_count[0] += 1
                count = _completed_count[0]
            log.info(
                f"[{count}/{total_active}] Sample {sample_id} generated "
                f"{len(results)} candidate(s)"
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
            scoring_log_lines = []
            sample_id = sample_ids[traj_idx] if sample_ids else traj_idx

            for cand_idx, raw in enumerate(raw_list):
                raw_text = raw.get("text", "")
                api_logprobs = raw.get("logprobs", [])
                token_count = (
                    len(api_logprobs) if api_logprobs else self._count_tokens(raw_text)
                )

                # For streaming path, use pre-processed results from BoundaryEarlyStopping.
                # step_text = text with stop token removed (like vLLM output.text)
                # raw_collected = full accumulated text including stop token
                if use_streaming:
                    step_text = raw.get("step_text", "")
                    raw_collected = raw.get("raw_collected", raw_text)
                    traj_complete = raw.get("trajectory_complete", False)

                    # Use step_text as the candidate text (stop token already
                    # removed, like vLLM's output.text)
                    text = step_text or raw_text

                    # Additional completion checks
                    is_thinking_complete = False
                    if not traj_complete:
                        _, traj_complete, completion_reason, is_thinking_complete = (
                            self._process_candidate_text(
                                text, token_count, is_streaming=True
                            )
                        )
                    else:
                        # BoundaryEarlyStopping set trajectory_complete=True.
                        # In thinking mode, </think> means thinking is done but
                        # the trajectory still needs an answer phase.
                        if self.thinking_mode and "</think>" in (raw_collected or text):
                            is_thinking_complete = True
                            traj_complete = False
                            completion_reason = CompletionReason.THINKING_COMPLETE
                        else:
                            completion_reason = CompletionReason.EOS_PATTERN
                else:
                    step_text = ""
                    is_thinking_complete = False
                    # Batch path — process text for completion
                    text, traj_complete, completion_reason, is_thinking_complete = (
                        self._process_candidate_text(
                            raw_text, token_count, is_streaming=False
                        )
                    )

                # Truncate logprobs to match step text (without stop token).
                # API logprobs cover all generated tokens including stop token,
                # but scoring should only use content tokens (like vLLM).
                original_logprob_count = len(api_logprobs) if api_logprobs else 0
                raw_for_comparison = raw_collected if use_streaming else raw_text
                if api_logprobs and text and text != raw_for_comparison:
                    # Use token text from logprobs to find the exact boundary
                    # rather than approximate _count_tokens which may mismatch
                    accumulated = ""
                    truncated_token_count = len(api_logprobs)
                    for lp_idx, lp_entry in enumerate(api_logprobs):
                        accumulated += lp_entry.get("token", "")
                        if len(accumulated) >= len(text):
                            truncated_token_count = lp_idx + 1
                            break
                    scoring_logprobs = api_logprobs[:truncated_token_count]
                else:
                    scoring_logprobs = api_logprobs

                # Score candidate using truncated logprobs
                scoring_data = self._score_candidate(text, scoring_logprobs)

                # Determine stop reason string
                stop_reason = None
                if completion_reason:
                    stop_reason = (
                        completion_reason.value
                        if hasattr(completion_reason, "value")
                        else str(completion_reason)
                    )
                elif raw.get("finish_reason"):
                    stop_reason = raw["finish_reason"]

                # Collect scoring log line for this candidate
                scoring_log_lines.append(
                    self._format_step_scoring(
                        token_ids=list(range(original_logprob_count)),
                        stop_reason=stop_reason,
                        raw_text=raw_collected if use_streaming else raw_text,
                        step_text=step_text if step_text else None,
                        scoring_token_count=scoring_data["original_token_count"],
                        path_idx=traj_idx if len(active_indices) > 1 else None,
                        candidate_idx=(
                            cand_idx if len(active_indices) <= 1 else cand_idx
                        ),
                        sample_id=sample_id,
                        validity_score=scoring_data["validity_score"],
                    )
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

                candidate.is_thinking_complete = is_thinking_complete
                candidates.append(candidate)

            # Emit all scoring lines for this sample as one log message
            if scoring_log_lines:
                log.info(
                    f"Sample {sample_id} scoring ({len(scoring_log_lines)} candidates):\n"
                    + "\n".join(scoring_log_lines)
                )

            # Context limit check
            if candidates and not candidates[0].is_trajectory_complete:
                ctx_tokens = context_token_counts.get(traj_idx, 0)
                max_gen = max(
                    c.other_data.get("original_token_count", 0) for c in candidates
                )
                total_tokens = ctx_tokens + max_gen

                # After thinking is complete, we only need room for the answer
                thinking_done = self.thinking_mode and any(
                    c.is_thinking_complete
                    or (
                        c.other_data
                        and c.other_data.get("completion_reason")
                        == CompletionReason.THINKING_COMPLETE
                    )
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
                        f"(prompt={ctx_tokens}, generated={max_gen}), "
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
        )

        # Mark all as trajectory complete
        for candidates in results:
            for c in candidates:
                c.is_trajectory_complete = True
                c.is_thinking_complete = True

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
        result = self.generate_step_candidates_batch(
            [request],
            [trajectory],
            candidates_per_step,
            compute_uncertainty=compute_uncertainty,
        )
        return result[0] if result else []
