"""
Early stopping conditions for streaming generation.

Provides a unified interface for different stopping strategies used during
text generation (confidence-based, boundary-based, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from llm_tts.step_boundary_detectors import (
    StructuredStepDetector,
    ThinkingMarkerDetector,
)


class EarlyStopping(ABC):
    """Abstract base class for early stopping conditions during streaming."""

    @abstractmethod
    def should_stop(self, state: Dict) -> bool:
        """
        Check if generation should stop based on current state.

        Args:
            state: Dictionary containing:
                - text: Accumulated text so far
                - token_count: Number of tokens generated
                - logprob: Current token's logprob (if available)
                - top_logprobs: Top-k logprobs (if available)

        Returns:
            True if generation should stop, False otherwise
        """
        pass

    @abstractmethod
    def get_reason(self) -> str:
        """Get a description of why generation was stopped."""
        pass

    def reset(self):
        """Reset any internal state (optional)."""
        pass


class ConfidenceEarlyStopping(EarlyStopping):
    """Stop generation when confidence drops below threshold (DeepConf style)."""

    def __init__(
        self,
        threshold: float,
        window_size: int = 5,
        top_k: int = 20,
        method: str = "mean_logprob",
    ):
        # Import here to avoid circular dependency
        from llm_tts.strategies.deepconf.utils import ConfidenceProcessor

        self.processor = ConfidenceProcessor(
            threshold=threshold, window_size=window_size, top_k=top_k, method=method
        )
        self._stopped = False

    def should_stop(self, state: Dict) -> bool:
        logprob = state.get("logprob")
        top_logprobs = state.get("top_logprobs", [])

        if logprob is not None and top_logprobs:
            self._stopped = self.processor.process_token(logprob, top_logprobs)
            return self._stopped
        return False

    def get_reason(self) -> str:
        return "confidence-threshold" if self._stopped else "not-stopped"

    def reset(self):
        self.processor.reset()
        self._stopped = False


class BoundaryEarlyStopping(EarlyStopping):
    """Stop generation when text boundary is detected (Best-of-N style)."""

    def __init__(self, detector: Optional[StructuredStepDetector] = None):
        self.detector = detector or StructuredStepDetector(
            step_patterns=None, answer_patterns=None, max_tokens_per_step=512
        )
        self._stop_reason = None

    def should_stop(self, state: Dict) -> bool:
        text = state.get("text", "")
        token_count = state.get("token_count", 0)

        if self.detector.is_step_complete(text, token_count):
            # Determine specific reason
            if self.detector.contains_answer_pattern(text):
                self._stop_reason = "answer-pattern"
            elif text.count("- Step") >= 2:
                self._stop_reason = "step-boundary"
            else:
                self._stop_reason = "boundary-detected"
            return True
        return False

    def get_reason(self) -> str:
        return self._stop_reason or "not-stopped"


class ThinkingStepEarlyStopping(EarlyStopping):
    """
    Stop generation when a reasoning step boundary is detected in thinking mode.

    Uses ThinkingMarkerDetector (marker_semantic_v2) to detect step boundaries
    during streaming generation. When a new step is completed, optionally evaluates
    it using a step scorer and stops if the score is below threshold.

    This enables Test-Time Scaling (TTS) strategies to:
    1. Detect when a reasoning step is complete
    2. Evaluate the step quality
    3. Decide whether to continue or backtrack/regenerate

    Args:
        detector: ThinkingMarkerDetector instance. If None, creates default
            marker_semantic_v2 configuration.
        step_scorer: Optional StepScorerBase instance to evaluate step quality.
            Uses score_candidates(chat, [step], aggregation) interface.
        score_threshold: Threshold below which a step is considered low quality.
            Only used if step_scorer is provided.
        score_aggregation: Aggregation method for scorer ('mean', 'min', 'max').
        stop_on_each_step: If True, stops at every step boundary (for step-by-step
            TTS). If False, only stops when step_scorer returns low score.
        min_chars_for_step: Minimum characters before considering step detection.
            Prevents false positives on very short accumulated text.
    """

    def __init__(
        self,
        detector: Optional[ThinkingMarkerDetector] = None,
        step_scorer=None,  # StepScorerBase
        score_threshold: float = 0.5,
        score_aggregation: str = "mean",
        stop_on_each_step: bool = True,
        min_chars_for_step: int = 100,
    ):
        # Default to marker_semantic_v2 configuration
        self.detector = detector or ThinkingMarkerDetector(
            use_sequence=True,
            use_conclusion=True,
            use_thinking=True,
            use_verification=True,
            use_reasoning=True,  # v2: for example, given that, similarly
            use_structure=False,
            use_sentence_start=False,
            use_correction=False,
            min_step_chars=100,
            max_step_chars=600,
        )
        self.step_scorer = step_scorer
        self.score_threshold = score_threshold
        self.score_aggregation = score_aggregation
        self.stop_on_each_step = stop_on_each_step
        self.min_chars_for_step = min_chars_for_step

        # Chat context for scorer (set via set_chat before generation)
        self._chat: Optional[List[Dict[str, str]]] = None

        # Internal state
        self._detected_steps: List[str] = []
        self._stop_reason: Optional[str] = None
        self._last_step_score: Optional[float] = None

    def set_chat(self, chat: List[Dict[str, str]]):
        """
        Set the chat context for step scoring.

        Must be called before generation if using a step_scorer, as the scorer
        needs the conversation context to evaluate steps.

        Args:
            chat: List of chat messages [{"role": "user", "content": "..."}]
        """
        self._chat = chat

    def should_stop(self, state: Dict) -> bool:
        text = state.get("text", "")

        # Skip detection if text too short
        if len(text) < self.min_chars_for_step:
            return False

        # Detect steps in accumulated text
        current_steps = self.detector.detect_steps(text)

        # Check if new step boundary detected
        if len(current_steps) > len(self._detected_steps):
            new_step = current_steps[-1]
            self._detected_steps = current_steps

            # Evaluate step if scorer provided
            if self.step_scorer is not None and self._chat is not None:
                # Use StepScorerBase interface: score_candidates(chat, candidates, aggregation)
                scores = self.step_scorer.score_candidates(
                    self._chat, [new_step], aggregation=self.score_aggregation
                )
                score = scores[0] if scores else 0.5
                self._last_step_score = score

                if score < self.score_threshold:
                    self._stop_reason = f"low_step_score:{score:.3f}"
                    return True

                # If scorer provided but score OK, continue unless stop_on_each_step
                if not self.stop_on_each_step:
                    return False

            # Stop at step boundary
            self._stop_reason = f"step_boundary:{len(current_steps)}"
            return True

        return False

    def get_reason(self) -> str:
        return self._stop_reason or "not-stopped"

    def get_detected_steps(self) -> List[str]:
        """Return list of detected steps so far."""
        return self._detected_steps.copy()

    def get_last_step_score(self) -> Optional[float]:
        """Return score of last evaluated step, if any."""
        return self._last_step_score

    def get_current_step_count(self) -> int:
        """Return number of steps detected so far."""
        return len(self._detected_steps)

    def reset(self):
        """Reset internal state for new generation."""
        self._detected_steps = []
        self._stop_reason = None
        self._last_step_score = None
        # Note: _chat is NOT reset - it should persist across retries


class CompositeEarlyStopping(EarlyStopping):
    """Combine multiple stopping conditions with AND/OR logic."""

    def __init__(self, conditions: List[EarlyStopping], logic: str = "OR"):
        """
        Args:
            conditions: List of stopping conditions to combine
            logic: "OR" (stop if ANY condition is met) or
                   "AND" (stop if ALL conditions are met)
        """
        self.conditions = conditions
        self.logic = logic.upper()
        if self.logic not in ["OR", "AND"]:
            raise ValueError("Logic must be 'OR' or 'AND'")
        self._triggered = []

    def should_stop(self, state: Dict) -> bool:
        self._triggered = []

        for condition in self.conditions:
            if condition.should_stop(state):
                self._triggered.append(condition)

        if self.logic == "OR":
            return len(self._triggered) > 0
        else:  # AND
            return len(self._triggered) == len(self.conditions)

    def get_reason(self) -> str:
        if not self._triggered:
            return "not-stopped"

        reasons = [c.get_reason() for c in self._triggered]
        return f"{self.logic}({','.join(reasons)})"

    def reset(self):
        for condition in self.conditions:
            condition.reset()
        self._triggered = []


class NoEarlyStopping(EarlyStopping):
    """No early stopping - generate until completion."""

    def should_stop(self, state: Dict) -> bool:
        return False

    def get_reason(self) -> str:
        return "completed"
