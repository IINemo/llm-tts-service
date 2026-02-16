#!/usr/bin/env python3
"""
Aggregate GPQA Diamond results from multiple parts into single seed folders.

For each seed:
1. Create a seedXX_aggregated subfolder
2. Move all part folders (seedXX_HH-MM-SS) into it
3. Aggregate metrics and results into metrics.json and results.json
4. Log aggregated metrics to wandb
"""

import json
import shutil
import wandb
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("outputs/2026-02-16/gpqa_diamond/vllm_thinking_qwen3_8b/offline_best_of_n")

def log_to_wandb(scorer: str, seed: str, metrics: dict, config_name: str):
    """Log aggregated metrics to wandb."""
    try:
        # wandb_group format: offline_bon_qwen3_8b_thinking_gpqa_diamond_{scorer}
        wandb_groups = {
            "entropy": "offline_bon_qwen3_8b_thinking_gpqa_diamond_entropy",
            "perplexity": "offline_bon_qwen3_8b_thinking_gpqa_diamond_perplexity",
            "sequence_prob": "offline_bon_qwen3_8b_thinking_gpqa_diamond_sequence_prob"
        }

        # Create run name
        timestamp = datetime.now().strftime("%H-%M-%S")
        run_name = f"2026-02-16_{config_name}_seed{seed}_{timestamp}"

        # Initialize wandb with correct group format
        run = wandb.init(
            project="llm-tts-eval-gpqa-diamond",
            group=wandb_groups[scorer],
            name=run_name,
            job_type="aggregated",
            config={"seed": int(seed), "scorer": scorer}
        )

        # Flatten metrics for wandb logging
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                flat_metrics[key] = value

        wandb.log(flat_metrics)
        wandb.finish()

        print(f"    - Logged to wandb: {run_name}")
        return True
    except Exception as e:
        print(f"    - Failed to log to wandb: {e}")
        return False

def aggregate_seed(scorer: str, seed: str):
    """Aggregate results for a specific scorer and seed."""
    scorer_dir = BASE_DIR / scorer
    if not scorer_dir.exists():
        print(f"  Scorer directory not found: {scorer}")
        return None

    # Find all part directories for this seed (exclude aggregated dirs)
    part_dirs = sorted([
        d for d in scorer_dir.glob(f"seed{seed}_*")
        if d.is_dir() and not d.name.endswith("_aggregated")
    ])

    if not part_dirs:
        print(f"  No parts found for seed{seed}")
        return None

    print(f"  Aggregating {len(part_dirs)} parts for seed{seed}")

    # Create aggregated directory
    agg_dir = scorer_dir / f"seed{seed}_aggregated"
    agg_dir.mkdir(exist_ok=True)

    # Move part directories into aggregated folder
    for part_dir in part_dirs:
        if part_dir.parent != agg_dir:
            new_path = agg_dir / part_dir.name
            if not new_path.exists():
                shutil.move(str(part_dir), str(new_path))
            print(f"    - Moved {part_dir.name} -> seed{seed}_aggregated/")

    # Now re-list part directories from the new location
    part_dirs = sorted(agg_dir.glob(f"seed{seed}_*"))
    part_dirs = [d for d in part_dirs if d.is_dir()]

    # Aggregate results
    all_results = []
    for part_dir in part_dirs:
        results_file = part_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                all_results.extend(results)

    # Save aggregated results
    if all_results:
        with open(agg_dir / "results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

    # Aggregate metrics
    all_metrics = {}
    total_samples = 0

    for part_dir in part_dirs:
        metrics_file = part_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

                part_samples = metrics.get("total_samples", 0)
                total_samples += part_samples

                # Aggregate metrics by category
                for key, value in metrics.items():
                    # Skip accuracy and percentage metrics - we'll recompute them
                    if any(x in key for x in ["accuracy", "pct", "_pct"]):
                        continue

                    if key == "total_samples":
                        all_metrics[key] = all_metrics.get(key, 0) + value
                    elif isinstance(value, (int, float)) and not any(x in key for x in ["avg_", "completed"]):
                        # Sum raw counts
                        all_metrics[key] = all_metrics.get(key, 0) + value
                    elif key not in all_metrics:
                        # Take first value for non-numeric fields
                        all_metrics[key] = value

    # Set total samples
    all_metrics["total_samples"] = total_samples

    # Compute Exact Match accuracy from summed correct/incorrect counts
    em_correct = all_metrics.get("exact_match/correct", 0)
    em_incorrect = all_metrics.get("exact_match/incorrect", 0)
    em_total = em_correct + em_incorrect
    if em_total > 0:
        all_metrics["exact_match/accuracy"] = em_correct / em_total
        all_metrics["exact_match/correct_pct"] = em_correct / em_total * 100
        all_metrics["exact_match/incorrect_pct"] = em_incorrect / em_total * 100

    # Compute LLM Judge accuracy from summed correct/incorrect counts
    llm_correct = all_metrics.get("llm_judge_openai_gpt-5-mini/correct", 0)
    llm_incorrect = all_metrics.get("llm_judge_openai_gpt-5-mini/incorrect", 0)
    llm_total = llm_correct + llm_incorrect
    if llm_total > 0:
        all_metrics["llm_judge_openai_gpt-5-mini/accuracy"] = llm_correct / llm_total
        all_metrics["llm_judge_openai_gpt-5-mini/correct_pct"] = llm_correct / llm_total * 100
        all_metrics["llm_judge_openai_gpt-5-mini/incorrect_pct"] = llm_incorrect / llm_total * 100

    # Compute average metrics from totals
    if "compute/total_output_tokens" in all_metrics and total_samples > 0:
        all_metrics["compute/avg_output_tokens_per_sample"] = all_metrics["compute/total_output_tokens"] / total_samples
    if "compute/total_tokens" in all_metrics and total_samples > 0:
        all_metrics["compute/avg_tokens_per_sample"] = all_metrics["compute/total_tokens"] / total_samples
    if "compute/total_tflops" in all_metrics and total_samples > 0:
        all_metrics["compute/avg_tflops_per_sample"] = all_metrics["compute/total_tflops"] / total_samples

    # completed_pct
    if total_samples > 0:
        all_metrics["completed_pct"] = all_metrics.get("completed", 0) / total_samples * 100

    # Save aggregated metrics
    with open(agg_dir / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Log to wandb
    config_name = f"offline_bon_vllm_thinking_qwen3_8b_gpqa_diamond_{scorer}"
    log_to_wandb(scorer, seed, all_metrics, config_name)

    return all_metrics

def compute_average_across_seeds(scorer: str, seeds: list):
    """Compute average metrics across all seeds for a scorer."""
    scorer_dir = BASE_DIR / scorer
    seed_metrics = {}

    for seed in seeds:
        metrics_file = scorer_dir / f"seed{seed}_aggregated" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                seed_metrics[seed] = json.load(f)

    if not seed_metrics:
        return None

    # Compute averages
    avg_metrics = {}
    num_seeds = len(seed_metrics)

    for key in seed_metrics[list(seed_metrics.keys())[0]].keys():
        values = [m.get(key) for m in seed_metrics.values() if m.get(key) is not None]

        if not values:
            continue

        # Skip string/non-numeric fields
        if not isinstance(values[0], (int, float)):
            if key not in avg_metrics:
                avg_metrics[key] = values[0]
            continue

        # Average ALL numeric values (including total_tflops)
        if "avg_" in key or key in ["completed", "errors"]:
            avg_metrics[key] = sum(values) / len(values)
        elif "compute/total_" in key:
            # Total compute metrics: average across seeds (each seed's total is sum of parts)
            avg_metrics[key] = sum(values) / len(values)
        elif key == "total_samples":
            # Keep total_samples as 198 (single seed size)
            avg_metrics[key] = values[0]
        else:
            # Average numeric values (will be recalculated for accuracy)
            avg_metrics[key] = sum(values) / len(values)

    # Recompute accuracy metrics from averaged correct/incorrect counts
    em_correct = avg_metrics.get("exact_match/correct", 0)
    em_incorrect = avg_metrics.get("exact_match/incorrect", 0)
    em_total = em_correct + em_incorrect
    if em_total > 0:
        avg_metrics["exact_match/accuracy"] = em_correct / em_total
        avg_metrics["exact_match/correct_pct"] = em_correct / em_total * 100
        avg_metrics["exact_match/incorrect_pct"] = em_incorrect / em_total * 100

    llm_correct = avg_metrics.get("llm_judge_openai_gpt-5-mini/correct", 0)
    llm_incorrect = avg_metrics.get("llm_judge_openai_gpt-5-mini/incorrect", 0)
    llm_total = llm_correct + llm_incorrect
    if llm_total > 0:
        avg_metrics["llm_judge_openai_gpt-5-mini/accuracy"] = llm_correct / llm_total
        avg_metrics["llm_judge_openai_gpt-5-mini/correct_pct"] = llm_correct / llm_total * 100
        avg_metrics["llm_judge_openai_gpt-5-mini/incorrect_pct"] = llm_incorrect / llm_total * 100

    # Recompute avg_tflops_per_sample from averaged total tflops
    avg_total_tflops = avg_metrics.get("compute/total_tflops", 0)
    total_samples = avg_metrics.get("total_samples", 0)
    if total_samples > 0:
        avg_metrics["compute/avg_tflops_per_sample"] = avg_total_tflops / total_samples

    # Recompute other averages from averaged totals
    if "compute/total_output_tokens" in avg_metrics and total_samples > 0:
        avg_metrics["compute/avg_output_tokens_per_sample"] = avg_metrics["compute/total_output_tokens"] / total_samples
    if "compute/total_tokens" in avg_metrics and total_samples > 0:
        avg_metrics["compute/avg_tokens_per_sample"] = avg_metrics["compute/total_tokens"] / total_samples

    # completed_pct
    if total_samples > 0:
        avg_metrics["completed_pct"] = avg_metrics.get("completed", 0) / total_samples * 100

    # Add num_seeds for reference
    avg_metrics["num_seeds"] = num_seeds

    return avg_metrics

def main():
    print("=" * 70)
    print("Aggregating GPQA Diamond Results")
    print("=" * 70)
    print(f"Base directory: {BASE_DIR}")
    print()

    scorers = ["entropy", "perplexity", "sequence_prob"]
    seeds = ["42", "43", "44"]

    # Aggregate all seeds
    for scorer in scorers:
        print("=" * 70)
        print(f"Scorer: {scorer}")
        print("=" * 70)
        for seed in seeds:
            metrics = aggregate_seed(scorer, seed)
            if metrics:
                em_acc = metrics.get("exact_match/accuracy", 0)
                llm_acc = metrics.get("llm_judge_openai_gpt-5-mini/accuracy", 0)
                total_tflops = metrics.get("compute/total_tflops", 0)
                print(f"  seed{seed}: EM={em_acc:.4f}, LLM={llm_acc:.4f}, TFLOPS={total_tflops:.0f}")
        print()

    # Compute averages across seeds
    print("=" * 70)
    print("Average Across Seeds")
    print("=" * 70)
    print()

    for scorer in scorers:
        avg_metrics = compute_average_across_seeds(scorer, seeds)
        if avg_metrics:
            em_acc = avg_metrics.get("exact_match/accuracy", 0)
            llm_acc = avg_metrics.get("llm_judge_openai_gpt-5-mini/accuracy", 0)
            em_correct = avg_metrics.get("exact_match/correct", 0)
            em_total = em_correct + avg_metrics.get("exact_match/incorrect", 0)
            llm_correct = avg_metrics.get("llm_judge_openai_gpt-5-mini/correct", 0)
            llm_total = llm_correct + avg_metrics.get("llm_judge_openai_gpt-5-mini/incorrect", 0)
            total_tflops = avg_metrics.get("compute/total_tflops", 0)
            avg_tflops_per_sample = avg_metrics.get("compute/avg_tflops_per_sample", 0)
            num_seeds = avg_metrics.get("num_seeds", 0)

            print(f"{scorer} ({num_seeds} seeds):")
            print(f"  Exact Match: {em_correct:.0f}/{em_total:.0f} = {em_acc:.4f} ({em_acc*100:.2f}%)")
            print(f"  LLM Judge:  {llm_correct:.0f}/{llm_total:.0f} = {llm_acc:.4f} ({llm_acc*100:.2f}%)")
            print(f"  Avg Total TFLOPS per seed: {total_tflops:.0f}")
            print(f"  Avg TFLOPS per sample: {avg_tflops_per_sample:.0f}")

            # Save average metrics
            scorer_dir = BASE_DIR / scorer
            with open(scorer_dir / "average_metrics.json", 'w') as f:
                json.dump(avg_metrics, f, indent=2)
        print()

    print("=" * 70)
    print("Aggregation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
