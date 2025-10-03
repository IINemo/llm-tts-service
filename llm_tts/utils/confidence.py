"""
Confidence computation utilities for DeepConf strategy.

Based on Facebook Research's DeepConf implementation:
https://github.com/facebookresearch/deepconf
"""

import logging
from typing import Dict, List

log = logging.getLogger(__name__)


def compute_token_confidence_from_logprobs(
    top_logprobs: List[Dict[str, float]], topk: int = 20
) -> float:
    """
    Compute confidence score for a single token from its top-k logprobs.

    Uses mean of top-k logprobs (negative = confidence score).
    Based on deepconf/utils.py:40-48

    Args:
        top_logprobs: List of dicts with 'token' and 'logprob' keys
        topk: Number of top tokens considered

    Returns:
        Confidence score (negative mean logprob, higher = more confident)
    """
    if not top_logprobs:
        return 0.0

    # Compute mean of top-k logprobs
    logprobs = [item.get("logprob", -100) for item in top_logprobs[:topk]]
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0

    # Return negative mean (higher = more confident)
    return -mean_logprob


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
