"""
DeepConf-specific utilities for confidence computation and answer extraction.

This module consolidates DeepConf-specific functionality and directly uses
lm-polygraph's uncertainty estimation methods for confidence computation.

Based on Facebook Research's DeepConf:
https://github.com/facebookresearch/deepconf

Leverages lm-polygraph's token-level uncertainty methods:
https://github.com/IINemo/lm-polygraph/tree/dev
See: lm_polygraph/utils/token_restoration.py:96-105
"""

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

# Use lm-polygraph's uncertainty estimation methods
from lm_polygraph.estimators import MaximumTokenProbability, Perplexity
from lm_polygraph.utils.token_restoration import Categorical

log = logging.getLogger(__name__)

# Reuse estimator instances to avoid creating new ones per token
_perplexity = Perplexity()
_max_token_prob = MaximumTokenProbability()


def compute_token_confidence_from_logprobs(
    top_logprobs: List[Dict[str, float]], topk: int = 20, method: str = "mean_logprob"
) -> float:
    """
    Compute confidence score for a single token using lm-polygraph methods.

    Args:
        top_logprobs: List of dicts with 'token' and 'logprob' keys
        topk: Number of top tokens considered
        method: 'entropy' | 'mean_logprob' | 'max_prob' | 'margin'

    Returns:
        Confidence score (higher = more confident)
    """
    if not top_logprobs:
        return 0.0

    logprobs_values = np.array(
        [item.get("logprob", -100.0) for item in top_logprobs[:topk]],
        dtype=np.float32,
    )

    if method == "entropy":
        posterior = torch.from_numpy(np.exp(logprobs_values))
        entropy = Categorical(posterior).entropy()
        return float(entropy.item())

    elif method == "mean_logprob":
        stats = {"greedy_log_likelihoods": [logprobs_values.tolist()]}
        result = _perplexity(stats)
        return float(result[0])

    elif method == "max_prob":
        stats = {"greedy_log_likelihoods": [logprobs_values.tolist()]}
        result = _max_token_prob(stats)
        return float(-np.mean(result[0]))

    elif method == "margin":
        if len(logprobs_values) < 2:
            return 0.0
        probs = np.exp(logprobs_values)
        probs = probs / probs.sum()
        top2_indices = np.argpartition(probs, -2)[-2:]
        top2_probs = probs[top2_indices]
        top2_sorted = np.sort(top2_probs)[::-1]
        return (
            float(top2_sorted[0] - top2_sorted[1])
            if len(top2_sorted) >= 2
            else float(top2_sorted[0])
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_sliding_window_confidence(
    token_confidences: List[float], window_size: int
) -> List[float]:
    """
    Compute sliding window average of token confidences.

    Args:
        token_confidences: Per-token confidence scores
        window_size: Window size

    Returns:
        Window-averaged confidences
    """
    if not token_confidences or window_size <= 0:
        return []

    if len(token_confidences) < window_size:
        return [sum(token_confidences) / len(token_confidences)]

    window_confs = []
    for i in range(len(token_confidences) - window_size + 1):
        window = token_confidences[i : i + window_size]
        window_conf = sum(window) / len(window)
        window_confs.append(window_conf)

    return window_confs


def extract_answer(text: str) -> str:
    """
    Extract answer from \\boxed{answer} format with nested braces support.

    Args:
        text: Generated text

    Returns:
        Extracted answer or empty string
    """
    if "boxed" not in text:
        return ""

    ans = text.split("boxed")[-1]
    if len(ans) == 0:
        return ""

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

    a = ans.split("$")[0].strip()
    return a.strip()


class ConfidenceProcessor:
    """Token-level confidence processor with early stopping."""

    def __init__(
        self,
        threshold: float,
        window_size: int = 5,
        top_k: int = 20,
        method: str = "mean_logprob",
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.top_k = top_k
        self.method = method

        self.conf_window: deque = deque(maxlen=window_size)
        self.conf_sum = 0.0
        self.all_confidences: List[float] = []

        log.info(
            f"Initialized ConfidenceProcessor: threshold={threshold}, "
            f"window_size={window_size}, method={method}"
        )

    def process_token(self, logprob: float, top_logprobs: List[dict]) -> bool:
        """Process token and check if confidence dropped below threshold."""
        confidence = compute_token_confidence_from_logprobs(
            top_logprobs, self.top_k, self.method
        )

        self.all_confidences.append(confidence)

        if len(self.conf_window) == self.window_size:
            self.conf_sum -= self.conf_window[0]

        self.conf_window.append(confidence)
        self.conf_sum += confidence

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
