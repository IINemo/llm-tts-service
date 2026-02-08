#!/usr/bin/env python3
"""
Script for aggregating statistics across multiple experiment runs with different seeds.
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def discover_runs(folder_path: str) -> List[str]:
    """Auto-discover all runs in the given folder."""
    runs = []

    # Find all directories containing metrics.json
    pattern = os.path.join(folder_path, "**", "metrics.json")
    metrics_files = glob.glob(pattern, recursive=True)

    if not metrics_files:
        # Also try to look for .json files directly (results files)
        pattern = os.path.join(folder_path, "*.json")
        json_files = glob.glob(pattern)
        for f in json_files:
            run_dir = os.path.dirname(f)
            if run_dir not in runs:
                runs.append(run_dir)
        return runs

    # Extract the parent directories (run directories)
    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        if run_dir not in runs:
            runs.append(run_dir)

    return sorted(runs)


def extract_seed_from_path(path: str) -> Optional[int]:
    """Extract seed number from directory path."""
    path_parts = os.path.basename(path).split("_")
    for part in path_parts:
        if part.startswith("seed"):
            try:
                return int(part[4:])
            except ValueError:
                continue
    return None


def load_results(run_path: str) -> Optional[Dict[str, Any]]:
    """Load results from a run's JSON file."""
    # Try metrics.json first
    metrics_file = os.path.join(run_path, "metrics.json")

    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading {metrics_file}: {e}")

    # Try any .json file in the directory
    json_files = glob.glob(os.path.join(run_path, "*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                # Check if this looks like a results file
                if isinstance(data, dict) and (
                    "metrics" in data or "eval" in data or "accuracy" in str(data)
                ):
                    return data
        except json.JSONDecodeError:
            continue

    return None


def extract_metrics(results_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from the results data."""
    extracted = {}

    # Handle nested metrics structure
    metrics = results_data.get("metrics", results_data)

    # Helper to get value - keys may contain '/' as part of the key name
    def get_value(key: str):
        # Try direct lookup first (keys like "exact_match/accuracy")
        if key in metrics:
            return metrics[key]
        # Try nested lookup (e.g., "compute", "total_tflops")
        parts = key.split("/")
        curr = metrics
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None
        return curr

    # Extract all metrics - just grab everything that's a number
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            extracted[key] = float(value)

    return extracted


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    arr = np.array(values)
    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "count": len(arr),
    }

    # 95% confidence interval
    if len(arr) > 1:
        sem = stats["std"] / np.sqrt(len(arr))
        stats["ci_95_low"] = stats["mean"] - 1.96 * sem
        stats["ci_95_high"] = stats["mean"] + 1.96 * sem

    return stats


def aggregate_statistics(runs: List[str]) -> Tuple[Dict[str, Any], List[Dict]]:
    """Aggregate statistics across all runs."""
    all_metrics = []

    for run_path in runs:
        results_data = load_results(run_path)
        if results_data is None:
            print(f"Warning: No valid results found in {run_path}")
            continue

        extracted = extract_metrics(results_data)
        extracted["run_path"] = run_path
        extracted["seed"] = extract_seed_from_path(run_path)

        all_metrics.append(extracted)

    if not all_metrics:
        raise ValueError("No valid metrics found in any runs")

    # Group by metric name
    metric_values = {}
    for metrics in all_metrics:
        for key, value in metrics.items():
            if key not in ["run_path", "seed"] and isinstance(value, (int, float)):
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(value)

    # Compute statistics for each metric
    aggregate_data = {}
    for metric, values in metric_values.items():
        if values:
            aggregate_data[metric] = compute_statistics(values)

    return aggregate_data, all_metrics


def display_aggregated_results(aggregate_data: Dict[str, Any], all_metrics: List[Dict]):
    """Display aggregated results in a clean table format."""
    print("\n" + "=" * 100)
    print(f"Aggregated Results across {len(all_metrics)} runs")
    print("=" * 100)

    if not aggregate_data:
        print("No metrics found to display")
        return

    # Define display order for common metrics (use actual key names from metrics.json)
    display_order = [
        "exact_match/accuracy",
        "llm_judge_gpt-5-mini/accuracy",
        "compute/avg_tflops_per_sample",
        "compute/avg_tokens_per_sample",
        "compute/avg_output_tokens_per_sample",
        "avg_validity_score",
        "avg_reasoning_steps_per_trajectory",
    ]

    # Filter to only metrics that exist
    display_metrics = [m for m in display_order if m in aggregate_data]
    # Add any remaining metrics
    for metric in aggregate_data:
        if metric not in display_metrics:
            display_metrics.append(metric)

    # Print table
    print(
        f"{'Metric':<35} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Median':>12} {'Count':>8}"
    )
    print("-" * 100)

    for metric in display_metrics:
        stats = aggregate_data[metric]
        name = metric.replace("_", " ").replace("/", " ").title()

        mean_str = f"{stats['mean']:.4f}"
        std_str = f"{stats['std']:.4f}"
        min_str = f"{stats['min']:.4f}"
        max_str = f"{stats['max']:.4f}"
        median_str = f"{stats['median']:.4f}"
        count_str = str(stats["count"])

        ci_str = ""
        if "ci_95_low" in stats:
            ci_str = f" [95% CI: {stats['ci_95_low']:.4f}, {stats['ci_95_high']:.4f}]"

        print(
            f"{name:<35} {mean_str:>12} {std_str:>12} {min_str:>12} {max_str:>12} {median_str:>12} {count_str:>8}"
        )
        if ci_str:
            print(f"{'':35} {'':>12} {'':>12} {ci_str}")

    print("\nRun details by seed:")
    for metrics in all_metrics:
        seed = metrics.get("seed", "N/A")
        path = metrics["run_path"]
        print(f"  Seed {seed}: {path}")


def save_aggregated_results(
    aggregate_data: Dict[str, Any], all_metrics: List[Dict], output_path: str
):
    """Save aggregated results to a JSON file."""
    output_data = {
        "summary": aggregate_data,
        "runs": all_metrics,
        "num_runs": len(all_metrics),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate statistics across multiple experiment runs with different seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover runs in a folder
  python scripts/aggregate_seed_results.py outputs/2026-02-08/aime24/baseline/

  # Specify specific runs
  python scripts/aggregate_seed_results.py \\
    outputs/2026-02-08/aime24/baseline/seed42_run1 \\
    outputs/2026-02-08/aime24/baseline/seed43_run2

  # Save to custom output file
  python scripts/aggregate_seed_results.py outputs/2026-02-08/aime24/baseline/ -o results.json
        """,
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="Either a folder path to auto-discover runs or a list of specific run paths",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="aggregated_results.json",
        help="Output file path for JSON results (default: aggregated_results.json)",
    )

    parser.add_argument(
        "--csv", action="store_true", help="Also save results to CSV format"
    )

    args = parser.parse_args()

    # Discover runs if a folder path is provided
    runs = []
    for input_path in args.input:
        if os.path.isdir(input_path):
            discovered = discover_runs(input_path)
            runs.extend(discovered)
            print(f"Discovered {len(discovered)} runs in {input_path}")
        elif os.path.exists(input_path):
            runs.append(input_path)
        else:
            print(f"Warning: Path does not exist: {input_path}")

    if not runs:
        print("No runs found to analyze")
        return

    print(f"\nAnalyzing {len(runs)} runs:")
    for run in runs[:5]:
        print(f"  - {run}")
    if len(runs) > 5:
        print(f"  ... and {len(runs) - 5} more")

    # Aggregate statistics
    try:
        aggregate_data, all_metrics = aggregate_statistics(runs)

        # Display results
        display_aggregated_results(aggregate_data, all_metrics)

        # Save aggregated results
        save_aggregated_results(aggregate_data, all_metrics, args.output)

        # Optionally save CSV
        if args.csv:
            import pandas as pd

            csv_path = args.output.replace(".json", "_seeds.csv")
            df = pd.DataFrame(all_metrics)
            df = df.sort_values("seed")
            df.to_csv(csv_path, index=False)
            print(f"Seed-level results saved to: {csv_path}")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
