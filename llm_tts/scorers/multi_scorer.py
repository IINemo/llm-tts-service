"""
Multi-scorer module for computing multiple logprob-based uncertainty metrics
from stored raw vLLM logprobs.

All metrics are computed from the same raw_logprobs data that is always stored
during vLLM generation (hardcoded logprobs=20). No re-generation needed.

Supported metrics:
- perplexity: -mean(greedy_log_likelihoods)
- sequence_prob: -sum(greedy_log_likelihoods)
- entropy: mean(token_entropy) from top-K logprobs
- pd_gap: mean(1 - (p1 - p2)) for top-2 token probabilities
"""

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Metric name -> (needs_matrix, needs_entropy)
_METRIC_REQUIREMENTS = {
    "perplexity": (False, False),
    "sequence_prob": (False, False),
    "entropy": (False, True),
    "pd_gap": (True, False),
}


def _run_logprobs_calculator(token_ids, raw_logprobs, output_matrix=False):
    """Run VLLMLogprobsCalculator on token_ids and logprobs."""
    from lm_polygraph.stat_calculators import VLLMLogprobsCalculator

    calc = VLLMLogprobsCalculator(output_matrix=output_matrix)
    deps = {"token_ids": token_ids, "logprobs": raw_logprobs}
    return calc(deps)


def _compute_entropy(greedy_log_probs):
    """Compute per-token entropy from greedy_log_probs (list of 1D arrays)."""
    from lm_polygraph.stat_calculators import EntropyCalculator

    calc = EntropyCalculator()
    return calc({"greedy_log_probs": greedy_log_probs})


def compute_logprob_scores(
    token_ids: List[int],
    raw_logprobs: List,
    metrics: List[str],
) -> Dict[str, float]:
    """Compute requested logprob-based uncertainty metrics from raw vLLM logprobs.

    Args:
        token_ids: Token IDs from vLLM generation
        raw_logprobs: Raw logprob dicts from vLLM (one per token)
        metrics: List of metric names to compute, e.g. ["perplexity", "entropy"]

    Returns:
        Dict with only the requested metrics, e.g. {"perplexity": 2.34, "entropy": 0.56}
    """
    if not token_ids or not raw_logprobs or not metrics:
        return {m: float("nan") for m in metrics}

    results = {}
    needs_basic = any(not _METRIC_REQUIREMENTS[m][0] for m in metrics if m in _METRIC_REQUIREMENTS)
    needs_matrix = any(_METRIC_REQUIREMENTS[m][0] for m in metrics if m in _METRIC_REQUIREMENTS)
    needs_entropy = any(_METRIC_REQUIREMENTS[m][1] for m in metrics if m in _METRIC_REQUIREMENTS)

    # Run basic calculator (for perplexity, sequence_prob, entropy)
    basic_stats = None
    if needs_basic or needs_entropy:
        basic_stats = _run_logprobs_calculator(token_ids, raw_logprobs, output_matrix=False)

    # Run matrix calculator (for pd_gap)
    matrix_stats = None
    if needs_matrix:
        matrix_stats = _run_logprobs_calculator(token_ids, raw_logprobs, output_matrix=True)

    # Compute entropy if needed
    entropy_stats = None
    if needs_entropy and basic_stats:
        entropy_stats = _compute_entropy(basic_stats["greedy_log_probs"])

    # Compute each requested metric
    for metric in metrics:
        if metric not in _METRIC_REQUIREMENTS:
            log.warning(f"Unknown metric '{metric}', skipping")
            results[metric] = float("nan")
            continue

        try:
            if metric == "perplexity":
                ll = basic_stats["greedy_log_likelihoods"][0]
                results[metric] = float(-np.mean(ll)) if ll else float("nan")

            elif metric == "sequence_prob":
                ll = basic_stats["greedy_log_likelihoods"][0]
                results[metric] = float(-np.sum(ll)) if ll else float("nan")

            elif metric == "entropy":
                ent = entropy_stats["entropy"][0]
                results[metric] = float(np.mean(ent)) if ent else float("nan")

            elif metric == "pd_gap":
                from llm_tts.scorers.estimator_uncertainty_pd import PDGap

                estimator = PDGap()
                unc = estimator({"greedy_log_probs": matrix_stats["greedy_log_probs"]})
                results[metric] = float(unc[0]) if len(unc) > 0 else float("nan")

        except Exception as e:
            log.warning(f"Failed to compute metric '{metric}': {e}")
            results[metric] = float("nan")

    return results


def compute_logprob_scores_per_step(
    steps: List[str],
    token_ids: List[int],
    raw_logprobs: List,
    tokenizer,
    metrics: List[str],
) -> Dict[str, List[float]]:
    """Compute requested metrics per-step by splitting token_ids at step boundaries.

    Uses text→token boundary mapping: decodes tokens incrementally to find where
    each step's text ends in the token sequence.

    Args:
        steps: List of step text strings
        token_ids: Full trajectory token IDs
        raw_logprobs: Full trajectory raw logprobs from vLLM
        tokenizer: Tokenizer for decoding tokens to find step boundaries
        metrics: List of metric names to compute

    Returns:
        Dict: {"perplexity": [step0_score, ...], "entropy": [...], ...}
    """
    if not steps or not token_ids or not raw_logprobs or not metrics:
        return {m: [] for m in metrics}

    # Find token boundaries for each step
    step_boundaries = _find_step_token_boundaries(steps, token_ids, tokenizer)

    # Compute metrics for each step
    per_step_results = {m: [] for m in metrics}

    for start_idx, end_idx in step_boundaries:
        step_token_ids = token_ids[start_idx:end_idx]
        step_logprobs = raw_logprobs[start_idx:end_idx]

        if step_token_ids and step_logprobs:
            step_scores = compute_logprob_scores(step_token_ids, step_logprobs, metrics)
        else:
            step_scores = {m: float("nan") for m in metrics}

        for m in metrics:
            per_step_results[m].append(step_scores.get(m, float("nan")))

    return per_step_results


def _find_step_token_boundaries(
    steps: List[str],
    token_ids: List[int],
    tokenizer,
) -> List[tuple]:
    """Map step text boundaries to token index boundaries.

    Decodes tokens incrementally to find where each step's text ends.

    Args:
        steps: List of step text strings
        token_ids: Full trajectory token IDs
        tokenizer: Tokenizer for decoding

    Returns:
        List of (start_idx, end_idx) tuples for each step
    """
    boundaries = []
    current_token_idx = 0

    for step_text in steps:
        step_start_idx = current_token_idx
        accumulated_text = ""

        while current_token_idx < len(token_ids):
            accumulated_text = tokenizer.decode(
                token_ids[step_start_idx : current_token_idx + 1],
                skip_special_tokens=False,
            )
            current_token_idx += 1

            if len(accumulated_text.strip()) >= len(step_text.strip()):
                break

        boundaries.append((step_start_idx, current_token_idx))

    return boundaries
