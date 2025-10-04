"""
Confidence-based stopping processor for streaming generation.
Similar to DeepConf's ConfPerReqLogitsProcessor but works with streaming API.
"""

import logging
import math
from collections import deque
from typing import List, Optional

log = logging.getLogger(__name__)


class ConfidenceProcessor:
    """
    Processes token-level confidence scores and triggers early stopping
    when confidence drops below threshold.
    """

    def __init__(
        self,
        threshold: float,
        window_size: int = 5,
        top_k: int = 20,
        method: str = "entropy",
    ):
        """
        Initialize confidence processor.

        Args:
            threshold: Confidence threshold below which to stop
            window_size: Size of sliding window for confidence averaging
            top_k: Number of top logprobs to consider
            method: Confidence computation method ('entropy', 'max_prob', 'margin')
        """
        self.threshold = threshold
        self.window_size = window_size
        self.top_k = top_k
        self.method = method

        # Sliding window of confidences
        self.conf_window: deque = deque(maxlen=window_size)
        self.conf_sum = 0.0

        # Track all confidences
        self.all_confidences: List[float] = []

        log.info(
            f"Initialized ConfidenceProcessor: threshold={threshold}, "
            f"window_size={window_size}, method={method}"
        )

    def compute_confidence(self, logprob: float, top_logprobs: List[dict]) -> float:
        """
        Compute confidence score from token logprobs.

        Args:
            logprob: Log probability of selected token
            top_logprobs: List of {'token': str, 'logprob': float} for top-k tokens

        Returns:
            Confidence score (higher is more confident)
        """
        if self.method == "mean_logprob":
            # Use mean of top-k logprobs (SAME AS WARMUP!)
            # This matches compute_token_confidence_from_logprobs()
            if not top_logprobs:
                return -logprob

            logprobs_values = [item["logprob"] for item in top_logprobs[: self.top_k]]
            mean_logprob = (
                sum(logprobs_values) / len(logprobs_values) if logprobs_values else 0.0
            )

            # Return negative mean (higher = more confident)
            return -mean_logprob

        elif self.method == "max_prob":
            # Simply use probability of selected token
            return math.exp(logprob)

        elif self.method == "entropy":
            # Compute normalized entropy from top-k distribution
            if not top_logprobs:
                return math.exp(logprob)

            # Convert logprobs to probabilities
            probs = [math.exp(item["logprob"]) for item in top_logprobs]

            # Normalize (should already be normalized but just in case)
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

            # Compute entropy: -sum(p * log(p))
            entropy = 0.0
            for p in probs:
                if p > 0:
                    entropy -= p * math.log(p)

            # Normalize by max possible entropy (log(k))
            max_entropy = math.log(min(len(probs), self.top_k))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Confidence is 1 - normalized_entropy (higher is more confident)
            return 1.0 - normalized_entropy

        elif self.method == "margin":
            # Margin between top-1 and top-2 probabilities
            if len(top_logprobs) < 2:
                return math.exp(logprob)

            top1_prob = math.exp(top_logprobs[0]["logprob"])
            top2_prob = math.exp(top_logprobs[1]["logprob"])

            return top1_prob - top2_prob

        else:
            raise ValueError(f"Unknown confidence method: {self.method}")

    def process_token(self, logprob: float, top_logprobs: List[dict]) -> bool:
        """
        Process a new token and check if generation should stop.

        Args:
            logprob: Log probability of generated token
            top_logprobs: Top-k alternative tokens with logprobs

        Returns:
            True if generation should stop (confidence below threshold)
        """
        # Compute confidence for this token
        conf = self.compute_confidence(logprob, top_logprobs)
        self.all_confidences.append(conf)

        # Update sliding window
        if len(self.conf_window) >= self.window_size:
            # Remove oldest value from sum
            self.conf_sum -= self.conf_window[0]

        self.conf_window.append(conf)
        self.conf_sum += conf

        # Check threshold once window is full
        if len(self.conf_window) >= self.window_size:
            avg_conf = self.conf_sum / len(self.conf_window)

            log.debug(
                f"Token conf: {conf:.4f}, Window avg: {avg_conf:.4f}, "
                f"Threshold: {self.threshold:.4f}"
            )

            if avg_conf < self.threshold:
                log.info(
                    f"Confidence dropped below threshold! "
                    f"avg={avg_conf:.4f} < {self.threshold:.4f}"
                )
                return True

        return False

    def get_min_confidence(self) -> Optional[float]:
        """Get minimum confidence across all tokens."""
        return min(self.all_confidences) if self.all_confidences else None

    def get_avg_confidence(self) -> Optional[float]:
        """Get average confidence across all tokens."""
        return (
            sum(self.all_confidences) / len(self.all_confidences)
            if self.all_confidences
            else None
        )

    def reset(self):
        """Reset processor state for new generation."""
        self.conf_window.clear()
        self.conf_sum = 0.0
        self.all_confidences.clear()
