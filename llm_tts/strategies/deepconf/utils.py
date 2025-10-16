"""
DeepConf-specific utilities for confidence computation and answer extraction.

This module consolidates DeepConf-specific functionality and leverages
lm-polygraph methods where possible for confidence computation.

Based on Facebook Research's DeepConf:
https://github.com/facebookresearch/deepconf
"""

import logging
from collections import deque
from typing import Dict, List, Optional

import torch
from torch.distributions.categorical import Categorical

log = logging.getLogger(__name__)


def compute_token_confidence_from_logprobs(
    top_logprobs: List[Dict[str, float]], topk: int = 20, method: str = "mean_logprob"
) -> float:
    """
    Compute confidence score for a single token from its top-k logprobs.

    Leverages lm-polygraph's approach for confidence computation.

    Args:
        top_logprobs: List of dicts with 'token' and 'logprob' keys
        topk: Number of top tokens considered
        method: Confidence computation method:
            - 'mean_logprob': Negative mean logprob (Perplexity-based)
            - 'max_prob': Negative max logprob (Maximum probability)
            - 'entropy': Shannon entropy over top-k distribution
            - 'margin': Difference between top-2 probabilities

    Returns:
        Confidence score (higher = more confident)
    """
    if not top_logprobs:
        return 0.0

    # Extract logprobs and convert to tensor
    logprobs_values = torch.tensor(
        [item.get("logprob", -100.0) for item in top_logprobs[:topk]],
        dtype=torch.float32,
    )

    if method == "mean_logprob":
        # Based on lm-polygraph's Perplexity estimator
        # perplexity.py: return np.array([-np.mean(ll) for ll in log_likelihoods])
        return float(-logprobs_values.mean())

    elif method == "max_prob":
        # Based on lm-polygraph's MaximumTokenProbability
        # max_probability.py: return [-np.array(log_likelihood[:-1])]
        return float(-logprobs_values.max())

    elif method == "entropy":
        # Based on lm-polygraph's token_restoration.py:96-105
        # Uses torch.distributions.categorical.Categorical
        probs = torch.exp(logprobs_values)
        probs = probs / probs.sum()  # Normalize
        return float(Categorical(probs=probs).entropy())

    elif method == "margin":
        # Top-2 probability margin
        if len(logprobs_values) < 2:
            return 0.0
        probs = torch.exp(logprobs_values)
        probs = probs / probs.sum()
        top2 = torch.topk(probs, min(2, len(probs))).values
        return float(top2[0] - top2[1]) if len(top2) >= 2 else float(top2[0])

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_sliding_window_confidence(
    token_confidences: List[float], window_size: int
) -> List[float]:
    """
    Compute sliding window average of token confidences.

    Uses fixed-size windows (returns fewer values than input).
    Based on deepconf/utils.py:52-61 (compute_least_grouped)

    Args:
        token_confidences: List of per-token confidence scores
        window_size: Size of sliding window

    Returns:
        List of window-averaged confidences (length = len(input) - window_size + 1)
    """
    if not token_confidences or window_size <= 0:
        return []

    # If sequence is shorter than window, return mean of all
    if len(token_confidences) < window_size:
        return [sum(token_confidences) / len(token_confidences)]

    # Fixed-size sliding windows only
    window_confs = []
    for i in range(len(token_confidences) - window_size + 1):
        window = token_confidences[i : i + window_size]
        window_conf = sum(window) / len(window)
        window_confs.append(window_conf)

    return window_confs


def extract_answer(text: str) -> str:
    """
    Extract answer from generated text (looks for \\boxed{answer} format).

    Handles nested braces correctly.
    Based on deepconf/utils.py:12-36

    Args:
        text: Generated text

    Returns:
        Extracted answer or empty string
    """
    if "boxed" not in text:
        return ""

    # Find the part after "boxed"
    ans = text.split("boxed")[-1]
    if len(ans) == 0:
        return ""

    # Handle \boxed{...} format with nested braces
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a.strip()

    # Handle \boxed answer$ format
    a = ans.split("$")[0].strip()
    return a.strip()


class ConfidenceProcessor:
    """
    Processes token-level confidence scores and triggers early stopping
    when confidence drops below threshold.

    Uses lm-polygraph-based confidence computation methods.
    """

    def __init__(
        self,
        threshold: float,
        window_size: int = 5,
        top_k: int = 20,
        method: str = "mean_logprob",
    ):
        """
        Initialize confidence processor.

        Args:
            threshold: Confidence threshold below which to stop
            window_size: Size of sliding window for confidence averaging
            top_k: Number of top logprobs to consider
            method: Confidence computation method (see compute_token_confidence_from_logprobs)
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

    def process_token(self, logprob: float, top_logprobs: List[dict]) -> bool:
        """
        Process a single token's logprobs and check if we should stop.

        Args:
            logprob: Token's logprob
            top_logprobs: List of top-k logprobs for this token

        Returns:
            True if confidence dropped below threshold (should stop)
        """
        # Compute confidence using lm-polygraph-based method
        confidence = compute_token_confidence_from_logprobs(
            top_logprobs, self.top_k, self.method
        )

        # Track confidence
        self.all_confidences.append(confidence)

        # Update sliding window
        if len(self.conf_window) == self.window_size:
            # Remove oldest from sum
            self.conf_sum -= self.conf_window[0]

        self.conf_window.append(confidence)
        self.conf_sum += confidence

        # Check threshold once window is full
        if len(self.conf_window) == self.window_size:
            avg_confidence = self.conf_sum / self.window_size
            if avg_confidence < self.threshold:
                log.debug(
                    f"Confidence dropped below threshold: {avg_confidence:.3f} < {self.threshold:.3f}"
                )
                return True

        return False

    def get_min_confidence(self) -> Optional[float]:
        """Get minimum confidence seen so far."""
        return min(self.all_confidences) if self.all_confidences else None

    def get_mean_confidence(self) -> Optional[float]:
        """Get mean confidence across all tokens."""
        return (
            sum(self.all_confidences) / len(self.all_confidences)
            if self.all_confidences
            else None
        )

    def reset(self):
        """Reset processor state."""
        self.conf_window.clear()
        self.conf_sum = 0.0
        self.all_confidences.clear()
