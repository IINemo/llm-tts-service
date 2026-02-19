#!/usr/bin/env python3
"""
Log existing evaluation results to wandb.

Usage:
    python scripts/log_results_to_wandb.py outputs/2025-10-18/23-50-46
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import wandb
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_results(results_path: Path):
    """Load results from JSON file."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: Path):
    """Load Hydra config from YAML file."""
    return OmegaConf.load(config_path)


def compute_metrics(results):
    """Compute evaluation metrics from results."""
    total_samples = len(results)
    completed = sum(1 for r in results if "error" not in r)
    errors = sum(1 for r in results if "error" in r)

    # Collect evaluator statistics
    evaluator_stats = {}
    for result in results:
        if "eval" not in result:
            continue

        for eval_name, eval_data in result["eval"].items():
            if eval_name not in evaluator_stats:
                evaluator_stats[eval_name] = {"correct": 0, "incorrect": 0}

            if eval_data.get("is_correct"):
                evaluator_stats[eval_name]["correct"] += 1
            else:
                evaluator_stats[eval_name]["incorrect"] += 1

    # Collect step statistics
    all_steps = []
    all_validities = []
    for result in results:
        if "reasoning_steps" in result:
            all_steps.append(result["reasoning_steps"])
        if "validity_scores" in result and result["validity_scores"]:
            valid = [s for s in result["validity_scores"] if s is not None]
            if valid:
                all_validities.append(float(np.mean(valid)))

    # Build metrics dict
    metrics = {
        "total_samples": total_samples,
        "completed": completed,
        "completed_pct": completed / total_samples if total_samples > 0 else 0,
        "errors": errors,
        "errors_pct": errors / total_samples if total_samples > 0 else 0,
    }

    # Add per-evaluator metrics
    for eval_name, stats in evaluator_stats.items():
        correct = stats["correct"]
        incorrect = stats["incorrect"]
        metrics[f"{eval_name}/correct"] = correct
        metrics[f"{eval_name}/correct_pct"] = correct / total_samples
        metrics[f"{eval_name}/incorrect"] = incorrect
        metrics[f"{eval_name}/incorrect_pct"] = incorrect / total_samples
        metrics[f"{eval_name}/accuracy"] = correct / total_samples

    # Add step statistics
    if all_steps:
        metrics["avg_steps_per_trajectory"] = np.mean(all_steps)
    if all_validities:
        metrics["avg_validity_score"] = np.mean(all_validities)

    return metrics, evaluator_stats


def main():
    parser = argparse.ArgumentParser(
        description="Log existing evaluation results to wandb"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output directory (e.g., outputs/2025-10-18/23-50-46)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Wandb project name (default: from config or 'llm-tts-eval')",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Wandb group name (default: from config wandb_group)",
    )
    parser.add_argument(
        "--tags", type=str, nargs="+", default=None, help="Tags for the run"
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.exists():
        log.error(f"Output directory does not exist: {output_dir}")
        return 1

    # Load results
    results_path = output_dir / "results.json"
    if not results_path.exists():
        log.error(f"Results file not found: {results_path}")
        return 1

    log.info(f"Loading results from {results_path}")
    results = load_results(results_path)
    log.info(f"Loaded {len(results)} results")

    # Load config
    config_path = output_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        return 1

    log.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Use pre-computed metrics.json if available (has compute stats),
    # otherwise compute from results
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        log.info(f"Loading pre-computed metrics from {metrics_path}")
        with open(metrics_path, "r", encoding="utf-8") as f:
            precomputed = json.load(f)

    # Compute metrics from results (for evaluator stats and summary)
    log.info("Computing metrics...")
    metrics, evaluator_stats = compute_metrics(results)

    # Merge pre-computed metrics (compute stats, etc.) if available
    if metrics_path.exists():
        for key, value in precomputed.items():
            if key not in metrics:
                metrics[key] = value

    # Print summary
    log.info("=" * 80)
    log.info("Summary:")
    log.info(f"Total samples: {metrics['total_samples']}")
    log.info(f"Completed: {metrics['completed']} ({metrics['completed_pct']:.1%})")
    log.info(f"Errors: {metrics['errors']} ({metrics['errors_pct']:.1%})")

    for eval_name, stats in evaluator_stats.items():
        correct = stats["correct"]
        total = stats["correct"] + stats["incorrect"]
        accuracy = correct / total if total > 0 else 0
        log.info(f"[{eval_name}] Accuracy: {correct}/{total} ({accuracy:.1%})")

    if "avg_steps_per_trajectory" in metrics:
        log.info(f"Avg steps per trajectory: {metrics['avg_steps_per_trajectory']:.1f}")
    if "avg_validity_score" in metrics:
        log.info(f"Avg validity score: {metrics['avg_validity_score']:.3f}")

    log.info("=" * 80)

    # Initialize wandb
    project = args.project or getattr(config, "wandb_project", "llm-tts-eval")
    log.info(f"Initializing wandb (project: {project})...")

    # Convert config to dict for wandb (resolve=False to avoid Hydra interpolation issues)
    wandb_config = OmegaConf.to_container(config, resolve=False, throw_on_missing=False)

    # Add metadata
    wandb_config["HYDRA_CONFIG"] = str(config_path)
    wandb_config["retroactive_logging"] = True
    wandb_config["original_output_dir"] = str(output_dir)

    # Initialize wandb
    group = args.group or getattr(config, "wandb_group", None)
    if group:
        log.info(f"Wandb group: {group}")

    run = wandb.init(
        project=project,
        name=args.name,
        group=group,
        config=wandb_config,
        tags=args.tags,
        dir=output_dir,
    )

    # Log metrics
    log.info("Logging metrics to wandb...")
    wandb.log(metrics)

    # Upload results file as artifact
    log.info("Uploading results as artifact...")
    artifact = wandb.Artifact(
        name=f"results-{output_dir.parent.name}-{output_dir.name}",
        type="evaluation_results",
    )
    artifact.add_file(str(results_path))
    artifact.add_file(str(config_path))
    run.log_artifact(artifact)

    # Finish run
    wandb.finish()

    log.info("✓ Successfully logged to wandb!")
    log.info(f"View run at: {run.url}")

    return 0


if __name__ == "__main__":
    exit(main())
