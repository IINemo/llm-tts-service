#!/usr/bin/env python3
"""
Post-hoc analysis of candidates.json from multi-scorer offline best-of-N runs.

Reads candidates.json, applies different aggregation heuristics to per-step scores,
selects the best candidate for each question, and computes accuracy for each
(scorer_type x aggregation) combination.

Usage:
    python scripts/analyze_candidates.py \
        --candidates-path outputs/.../candidates.json \
        --data-name math500

    # With all scoring windows:
    python scripts/analyze_candidates.py \
        --candidates-path outputs/.../candidates.json \
        --data-name math500 \
        --scoring-windows all
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root and scripts to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from llm_tts.evaluation.exact_match import EvaluatorExactMatch

log = logging.getLogger(__name__)


def load_candidates(path: str) -> List[Dict]:
    """Load candidates.json file."""
    with open(path, "r") as f:
        return json.load(f)


def aggregate_scores(
    per_step_scores: List[float],
    method: str,
    scoring_window: Optional[int] = None,
) -> float:
    """Aggregate per-step scores into a single trajectory score.

    Args:
        per_step_scores: List of per-step scores
        method: Aggregation method (mean, min, max, product)
        scoring_window: If set, only use the last N steps before aggregation

    Returns:
        Aggregated score (higher = better for PRM; lower = better for uncertainty metrics)
    """
    valid = [s for s in per_step_scores if s is not None and not np.isnan(s)]
    if not valid:
        return float("nan")

    # Apply scoring window: keep only last N valid scores
    if scoring_window is not None and len(valid) > scoring_window:
        valid = valid[-scoring_window:]

    if method == "mean":
        return float(np.mean(valid))
    elif method == "min":
        return float(np.min(valid))
    elif method == "max":
        return float(np.max(valid))
    elif method == "product":
        return float(np.prod(valid))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def select_best_candidate(
    candidates: List[Dict],
    scorer_type: str,
    aggregation: str,
    scoring_window: Optional[int] = None,
) -> Optional[int]:
    """Select the best candidate index using the given scorer and aggregation.

    For uncertainty metrics (perplexity, entropy, sequence_prob, pd_gap),
    lower is better (more certain = better).
    For PRM, higher is better (higher reward = better).

    Args:
        candidates: List of candidate dicts with 'scores' field
        scorer_type: Score type to use (e.g. "perplexity", "prm")
        aggregation: Aggregation method
        scoring_window: If set, only use the last N steps before aggregation

    Returns:
        Index of best candidate, or None if no valid scores
    """
    # PRM: higher is better; uncertainty metrics: lower is better
    higher_is_better = scorer_type == "prm"

    best_idx = None
    best_score = None

    for idx, candidate in enumerate(candidates):
        scores = candidate.get("scores", {})
        if scorer_type not in scores:
            continue

        scorer_data = scores[scorer_type]
        per_step = scorer_data.get("per_step", [])

        if not per_step:
            # Fall back to trajectory-level score
            agg_score = scorer_data.get("trajectory", float("nan"))
        else:
            agg_score = aggregate_scores(per_step, aggregation, scoring_window)

        if np.isnan(agg_score):
            continue

        if best_idx is None:
            best_idx = idx
            best_score = agg_score
        elif higher_is_better and agg_score > best_score:
            best_idx = idx
            best_score = agg_score
        elif not higher_is_better and agg_score < best_score:
            best_idx = idx
            best_score = agg_score

    return best_idx


def precompute_correctness(
    candidates_data: List[Dict],
    data_name: str,
    answer_format: str = "numeric",
) -> List[List[bool]]:
    """Pre-compute exact-match correctness of each candidate answer once.

    Returns:
        List of lists: correctness_labels[sample_idx][candidate_idx] = True/False
    """
    evaluator = EvaluatorExactMatch(
        dataset_answer_format=answer_format, data_name=data_name
    )

    correctness_labels = []
    for sample in candidates_data:
        gold_answer = str(sample.get("gold_answer", ""))
        candidates = sample.get("candidates", [])
        question = sample.get("question", "")
        sample_labels = []
        for candidate in candidates:
            extracted = candidate.get("extracted_answer", "")
            score = evaluator._score_single(
                (question, str(extracted), gold_answer),
                pre_extracted=True,
            )
            sample_labels.append(score > 0)
        correctness_labels.append(sample_labels)

    return correctness_labels


def analyze(
    candidates_data: List[Dict],
    correctness_labels: List[List[bool]],
    scoring_window: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Run post-hoc analysis using pre-computed correctness labels.

    Args:
        candidates_data: List of sample dicts from candidates.json
        correctness_labels: Pre-computed correctness per candidate
        scoring_window: If set, only use the last N steps before aggregation

    Returns:
        Nested dict: {scorer_type: {aggregation: accuracy}}
    """
    # Discover available scorer types from data
    scorer_types = set()
    for sample in candidates_data:
        for candidate in sample.get("candidates", []):
            scorer_types.update(candidate.get("scores", {}).keys())

    scorer_types = sorted(scorer_types)
    if not scorer_types:
        log.error("No scorer types found in candidates data")
        return {}

    aggregation_methods = ["mean", "min", "max", "product"]

    results = {}

    for scorer_type in scorer_types:
        results[scorer_type] = {}
        for agg_method in aggregation_methods:
            correct = 0
            total = 0

            for sample_idx, sample in enumerate(candidates_data):
                candidates = sample.get("candidates", [])
                if not candidates:
                    continue

                best_idx = select_best_candidate(
                    candidates, scorer_type, agg_method,
                    scoring_window=scoring_window,
                )

                if best_idx is None:
                    best_idx = 0

                if correctness_labels[sample_idx][best_idx]:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0.0
            results[scorer_type][agg_method] = accuracy

    return results


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print results as a formatted table."""
    if not results:
        print("No results to display")
        return

    first_scorer = next(iter(results.values()))
    agg_methods = list(first_scorer.keys())

    header = f"{'scorer':<18}" + "".join(f"{label:>10}" for label in agg_methods)

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for scorer_type in results:
        row = f"{scorer_type:<18}"
        for agg in agg_methods:
            row += f"{results[scorer_type].get(agg, 0.0):>10.4f}"
        print(row)

    print("=" * len(header))


def _save_csv(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    csv_path: Path,
    oracle_acc: float = 0.0,
):
    """Save results as CSV in long format.

    Format: scorer, aggregation, window, exact_match
    Sorted by scorer, then aggregation, then window (1, 2, ..., all).
    Oracle row at the top with empty aggregation/window fields.
    """
    if not all_results:
        return

    # Collect all scorers and aggregation methods
    first_window_results = next(iter(all_results.values()))
    scorer_types = sorted(first_window_results.keys())
    first_scorer = next(iter(first_window_results.values()))
    agg_methods = list(first_scorer.keys())

    # Build ordered window list: numeric sorted, then "all" at the end
    window_order = []
    for wl in all_results:
        val = wl.split("=", 1)[1]
        window_order.append(val)
    window_order.sort(key=lambda w: (0, int(w)) if w != "all" else (1, 0))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scorer", "aggregation", "window", "exact_match"])

        # Oracle row at the top
        writer.writerow(["oracle", "", "", f"{oracle_acc:.4f}"])

        for scorer_type in scorer_types:
            for agg in agg_methods:
                for window_val in window_order:
                    window_label = f"window={window_val}"
                    results = all_results.get(window_label, {})
                    acc = results.get(scorer_type, {}).get(agg, 0.0)
                    writer.writerow([scorer_type, agg, window_val, f"{acc:.4f}"])


def main():
    parser = argparse.ArgumentParser(
        description="Analyze candidates.json from multi-scorer offline best-of-N runs"
    )
    parser.add_argument(
        "--candidates-path",
        type=str,
        required=True,
        help="Path to candidates.json file",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        required=True,
        help="Dataset name for answer comparison (e.g., math500, gsm8k, minerva_math)",
    )
    parser.add_argument(
        "--answer-format",
        type=str,
        default="numeric",
        choices=["numeric", "boolean", "char", "string"],
        help="Answer format for exact match evaluation (default: numeric)",
    )
    parser.add_argument(
        "--scoring-windows",
        nargs="*",
        default=None,
        help="Scoring windows to compare. Use 'all' to auto-iterate 1..max_steps, "
        "or specify integers (e.g., --scoring-windows 2 3 5). "
        "If not set, uses all steps (no window).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file (optional)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load candidates data
    log.info(f"Loading candidates from {args.candidates_path}")
    candidates_data = load_candidates(args.candidates_path)
    log.info(f"Loaded {len(candidates_data)} samples")

    # Summarize
    total_candidates = sum(len(s.get("candidates", [])) for s in candidates_data)
    scorer_types = set()
    for sample in candidates_data:
        for candidate in sample.get("candidates", []):
            scorer_types.update(candidate.get("scores", {}).keys())
    log.info(
        f"Total candidates: {total_candidates}, "
        f"scorer types: {sorted(scorer_types)}"
    )

    # Find max step count across all candidates
    max_steps = 0
    for sample in candidates_data:
        for candidate in sample.get("candidates", []):
            num_steps = len(candidate.get("steps", []))
            max_steps = max(max_steps, num_steps)
    log.info(f"Max steps across all candidates: {max_steps}")

    # Pre-compute exact-match correctness labels
    log.info("Pre-computing exact-match correctness labels...")
    em_labels = precompute_correctness(
        candidates_data,
        data_name=args.data_name,
        answer_format=args.answer_format,
    )
    em_correct = sum(1 for labels in em_labels if any(labels))
    oracle_acc = em_correct / len(em_labels) if em_labels else 0.0
    log.info(f"Exact match: {em_correct}/{len(em_labels)} samples have at least one correct candidate (oracle={oracle_acc:.4f})")

    # Build list of windows to evaluate
    windows = [None]  # always include "all steps"
    if args.scoring_windows:
        if "all" in args.scoring_windows:
            windows.extend(range(1, max_steps + 1))
        else:
            windows.extend(int(w) for w in args.scoring_windows)

    all_results = {}
    for window in windows:
        window_label = f"window={window}" if window is not None else "window=all"
        log.info(f"Analyzing with {window_label}...")

        results = analyze(
            candidates_data,
            correctness_labels=em_labels,
            scoring_window=window,
        )
        all_results[window_label] = results

        print(f"\n>>> {window_label}")
        print_results_table(results)

    # Save CSV to same directory as candidates.json
    candidates_dir = Path(args.candidates_path).parent
    csv_path = candidates_dir / "scoring_analysis.csv"
    _save_csv(all_results, csv_path, oracle_acc=oracle_acc)
    log.info(f"CSV saved to {csv_path}")

    # Save to JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
