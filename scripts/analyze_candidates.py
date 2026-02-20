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

    # With custom last-k value:
    python scripts/analyze_candidates.py \
        --candidates-path outputs/.../candidates.json \
        --data-name math500 \
        --last-k 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    last_k: int = 3,
) -> float:
    """Aggregate per-step scores into a single trajectory score.

    Args:
        per_step_scores: List of per-step scores
        method: Aggregation method (mean, min, max, last, last_k, product)
        last_k: Number of last steps for 'last_k' method

    Returns:
        Aggregated score (higher = better for PRM; lower = better for uncertainty metrics)
    """
    valid = [s for s in per_step_scores if s is not None and not np.isnan(s)]
    if not valid:
        return float("nan")

    if method == "mean":
        return float(np.mean(valid))
    elif method == "min":
        return float(np.min(valid))
    elif method == "max":
        return float(np.max(valid))
    elif method == "last":
        return valid[-1]
    elif method == "last_k":
        return float(np.mean(valid[-last_k:])) if valid else float("nan")
    elif method == "product":
        return float(np.prod(valid))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def select_best_candidate(
    candidates: List[Dict],
    scorer_type: str,
    aggregation: str,
    last_k: int = 3,
) -> Optional[int]:
    """Select the best candidate index using the given scorer and aggregation.

    For uncertainty metrics (perplexity, entropy, sequence_prob, pd_gap),
    lower is better (more certain = better).
    For PRM, higher is better (higher reward = better).

    Args:
        candidates: List of candidate dicts with 'scores' field
        scorer_type: Score type to use (e.g. "perplexity", "prm")
        aggregation: Aggregation method
        last_k: K for last_k aggregation

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
            agg_score = aggregate_scores(per_step, aggregation, last_k)

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


def analyze(
    candidates_data: List[Dict],
    data_name: str,
    answer_format: str = "numeric",
    last_k: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Run post-hoc analysis on candidates data.

    Args:
        candidates_data: List of sample dicts from candidates.json
        data_name: Dataset name for answer comparison
        answer_format: Answer format for exact match evaluation
        last_k: K for last_k aggregation

    Returns:
        Nested dict: {scorer_type: {aggregation: accuracy}}
    """
    evaluator = EvaluatorExactMatch(
        dataset_answer_format=answer_format, data_name=data_name
    )

    # Discover available scorer types from data
    scorer_types = set()
    for sample in candidates_data:
        for candidate in sample.get("candidates", []):
            scorer_types.update(candidate.get("scores", {}).keys())

    scorer_types = sorted(scorer_types)
    if not scorer_types:
        log.error("No scorer types found in candidates data")
        return {}

    aggregation_methods = ["mean", "min", "max", "last", "last_k", "product"]

    # Also add oracle (best possible) and random baselines
    results = {}

    for scorer_type in scorer_types:
        results[scorer_type] = {}
        for agg_method in aggregation_methods:
            correct = 0
            total = 0

            for sample in candidates_data:
                gold_answer = str(sample.get("gold_answer", ""))
                candidates = sample.get("candidates", [])

                if not candidates:
                    continue

                best_idx = select_best_candidate(
                    candidates, scorer_type, agg_method, last_k
                )

                if best_idx is None:
                    # No valid scores — fall back to first candidate
                    best_idx = 0

                extracted = candidates[best_idx].get("extracted_answer", "")
                question = sample.get("question", "")

                score = evaluator._score_single(
                    (question, str(extracted), gold_answer),
                    pre_extracted=True,
                )
                if score > 0:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0.0
            results[scorer_type][agg_method] = accuracy

    # Compute oracle accuracy (any correct candidate exists)
    oracle_correct = 0
    oracle_total = 0
    for sample in candidates_data:
        gold_answer = str(sample.get("gold_answer", ""))
        candidates = sample.get("candidates", [])
        if not candidates:
            continue
        any_correct = False
        for candidate in candidates:
            extracted = candidate.get("extracted_answer", "")
            question = sample.get("question", "")
            score = evaluator._score_single(
                (question, str(extracted), gold_answer),
                pre_extracted=True,
            )
            if score > 0:
                any_correct = True
                break
        if any_correct:
            oracle_correct += 1
        oracle_total += 1

    results["_oracle"] = {
        agg: oracle_correct / oracle_total if oracle_total > 0 else 0.0
        for agg in aggregation_methods
    }

    # Compute random baseline (first candidate)
    random_correct = 0
    random_total = 0
    for sample in candidates_data:
        gold_answer = str(sample.get("gold_answer", ""))
        candidates = sample.get("candidates", [])
        if not candidates:
            continue
        extracted = candidates[0].get("extracted_answer", "")
        question = sample.get("question", "")
        score = evaluator._score_single(
            (question, str(extracted), gold_answer),
            pre_extracted=True,
        )
        if score > 0:
            random_correct += 1
        random_total += 1

    results["_first"] = {
        agg: random_correct / random_total if random_total > 0 else 0.0
        for agg in aggregation_methods
    }

    return results


def print_results_table(
    results: Dict[str, Dict[str, float]],
    last_k: int = 3,
):
    """Print results as a formatted table."""
    if not results:
        print("No results to display")
        return

    # Get aggregation methods from first scorer
    first_scorer = next(iter(results.values()))
    agg_methods = list(first_scorer.keys())

    # Format header
    agg_labels = []
    for agg in agg_methods:
        if agg == "last_k":
            agg_labels.append(f"last_{last_k}")
        else:
            agg_labels.append(agg)

    header = f"{'scorer':<18}" + "".join(f"{label:>10}" for label in agg_labels)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # Sort scorers: regular first, then baselines
    regular_scorers = [s for s in results if not s.startswith("_")]
    baseline_scorers = [s for s in results if s.startswith("_")]

    for scorer_type in regular_scorers:
        scores = results[scorer_type]
        row = f"{scorer_type:<18}"
        for agg in agg_methods:
            acc = scores.get(agg, 0.0)
            row += f"{acc:>10.4f}"
        print(row)

    if baseline_scorers:
        print("-" * len(header))
        for scorer_type in baseline_scorers:
            scores = results[scorer_type]
            label = scorer_type.lstrip("_")
            row = f"{label:<18}"
            for agg in agg_methods:
                acc = scores.get(agg, 0.0)
                row += f"{acc:>10.4f}"
            print(row)

    print("=" * len(header))


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
        "--last-k",
        type=int,
        default=3,
        help="K for last_k aggregation method (default: 3)",
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

    # Run analysis
    results = analyze(
        candidates_data,
        data_name=args.data_name,
        answer_format=args.answer_format,
        last_k=args.last_k,
    )

    # Print table
    print_results_table(results, last_k=args.last_k)

    # Save to JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
