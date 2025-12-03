"""
Early stopping conditions for streaming generation.

Provides a unified interface for different stopping strategies used during
text generation (confidence-based, boundary-based, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from llm_tts.generators import StepBoundaryDetector


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

    def __init__(self, detector: Optional[StepBoundaryDetector] = None):
        self.detector = detector or StepBoundaryDetector(
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
