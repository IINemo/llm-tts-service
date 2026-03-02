#!/usr/bin/env python3
"""Re-run evaluation phase on existing batch_results.jsonl.

Usage:
    python scripts/reeval.py <output_dir>

Loads the saved Hydra config and batch_results.jsonl from the output directory,
merges compute metrics from sample_metrics.jsonl (if available), runs
evaluate_results(), and saves metrics.json + results.json.
"""

import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add scripts/ to path for utils imports
sys.path.insert(0, str(Path(__file__).parent))

from run_tts_eval import evaluate_results  # noqa: E402

log = logging.getLogger(__name__)


def _load_sample_metrics(output_dir: Path) -> dict:
    """Load sample_metrics.jsonl and return a dict keyed by sample_index."""
    path = output_dir / "sample_metrics.jsonl"
    if not path.exists():
        return {}
    metrics_by_index = {}
    with open(path) as f:
        for line in f:
            m = json.loads(line)
            idx = m.get("sample_index")
            if idx is not None:
                metrics_by_index[idx] = m
    return metrics_by_index


def _build_token_stats(sample_metric: dict) -> dict:
    """Convert sample_metrics.jsonl record to the token_stats format
    expected by evaluate_results."""
    return {
        "total_tokens_this_sample": sample_metric.get("total_tokens_this_sample", 0),
        "input_tokens": sample_metric.get("input_tokens_this_sample", 0),
        "output_tokens": sample_metric.get("output_tokens_this_sample", 0),
        "generation_count": sample_metric.get("generations_this_sample", 0),
        "tflops": sample_metric.get("tflops_this_sample", 0),
        "prm_input_tokens": sample_metric.get("prm_tokens_this_sample", 0),
        "prm_tflops": sample_metric.get("prm_tflops_this_sample", 0),
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    batch_results_path = output_dir / "batch_results.jsonl"
    config_path = output_dir / ".hydra" / "config.yaml"

    if not batch_results_path.exists():
        print(f"ERROR: {batch_results_path} not found")
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    # Load config
    config = OmegaConf.load(config_path)
    log.info(f"Loaded config from {config_path}")

    # Load batch results
    results = []
    with open(batch_results_path) as f:
        for line in f:
            results.append(json.loads(line))
    log.info(f"Loaded {len(results)} results from {batch_results_path}")

    # Merge compute metrics from sample_metrics.jsonl
    sample_metrics = _load_sample_metrics(output_dir)
    merged = 0
    for result in results:
        idx = result.get("index")
        if idx in sample_metrics:
            result["token_stats"] = _build_token_stats(sample_metrics[idx])
            merged += 1
    if merged:
        log.info(f"Merged compute metrics for {merged}/{len(results)} samples")
    elif sample_metrics:
        log.warning("sample_metrics.jsonl found but no indices matched batch_results")
    else:
        log.warning("No sample_metrics.jsonl found — compute metrics will be zeros")

    # Run evaluation
    evaluate_results(
        config=config,
        results=results,
        save_path=str(output_dir),
    )
    log.info("Done. Check metrics.json and results.json in %s", output_dir)


if __name__ == "__main__":
    main()
