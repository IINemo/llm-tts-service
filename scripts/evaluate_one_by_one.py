#!/usr/bin/env python3
"""Evaluate GSM8K results by sending each answer pair to LLM judge one by one."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Use GPT-4o for fast evaluation via OpenRouter
MODEL = "openai/gpt-4o"
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"


def evaluate_single_answer(generated: str, gold: str, sample_id: int) -> bool:
    """Evaluate a single answer pair using LLM judge."""

    prompt = f"""You are evaluating a math problem solution.

The gold answer (correct answer) is: {gold}

The generated answer is:
{generated}

The generated answer may include step-by-step reasoning with the final answer in \\boxed{{}} format.
Check if the generated answer contains the correct final answer compared to the gold answer.

Consider the answer correct if:
- The final numeric answer in the generated text matches the gold answer
- Allow for mathematically equivalent forms (e.g., 1/2 = 0.5 = 50%)
- If there's a \\boxed{{}} in the generated answer, check the value inside it

Respond with ONLY 'correct' or 'incorrect' (no other text).
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "GSM8K Evaluation",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            verdict_text = data["choices"][0]["message"]["content"].strip().lower()
            is_correct = "correct" in verdict_text and "incorrect" not in verdict_text
            return is_correct
        else:
            print(f"  Sample {sample_id}: API error {response.status_code}")
            return False

    except Exception as e:
        print(f"  Sample {sample_id}: Error - {e}")
        return False


def evaluate_experiment(results_path: Path, exp_name: str) -> Dict:
    """Evaluate all samples in a results.json file one by one."""

    with open(results_path, "r") as f:
        results = json.load(f)

    print(f"\n{exp_name}:")
    print(f"  Processing {len(results)} samples...")

    correct = 0
    evaluations = []

    # Process each sample individually with progress bar
    for i, result in enumerate(tqdm(results, desc="  Evaluating", leave=True)):
        sample_id = i + 30  # Samples start from index 30
        generated = result.get("generated_answer", "").strip()
        gold = result.get("gold_answer", "").strip()

        if not generated:
            generated = "No answer"
        if not gold:
            gold = "No answer"

        # Evaluate this single answer
        is_correct = evaluate_single_answer(generated, gold, sample_id)

        if is_correct:
            correct += 1

        evaluations.append(
            {
                "sample_id": sample_id,
                "gold": gold,
                "generated": generated,  # Save full generated answer, no truncation
                "correct": is_correct,
            }
        )

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    accuracy = (correct / len(results) * 100) if len(results) > 0 else 0

    return {
        "total": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "evaluations": evaluations,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K results one by one")
    parser.add_argument(
        "--output-dir",
        default="/Users/karantonis/MBZUAI/courses/NLP/llm-tts-service/outputs/2025-11-23",
    )
    parser.add_argument(
        "--save-details", action="store_true", help="Save detailed evaluation results"
    )
    args = parser.parse_args()

    # Find all extension experiments
    experiments = [
        ("Baseline GPT-4o Ext", "gsm8k_baseline_4o_ext_21-11-26"),
        ("Baseline GPT-4o-mini Ext", "gsm8k_baseline_4omini_ext_20-58-47"),
        ("Self-Consistency GPT-4o Ext", "gsm8k_selfconsistency_gpt4o_n16_ext_20-58-47"),
        (
            "Self-Consistency GPT-4o-mini Ext",
            "gsm8k_selfconsistency_gpt4omini_n16_ext_20-58-47",
        ),
        ("DeepConf GPT-4o Ext", "gsm8k_deepconf_gpt4o_b16_ext_20-58-48"),
        ("DeepConf GPT-4o-mini Ext", "gsm8k_deepconf_gpt4omini_b16_ext_20-58-48"),
    ]

    print("=" * 80)
    print("GSM8K Extension Experiments - One-by-One Evaluation")
    print(f"Using {MODEL} as judge")
    print("=" * 80)

    all_results = {}

    for exp_name, exp_dir in experiments:
        results_path = Path(args.output_dir) / exp_dir / "results.json"

        if not results_path.exists():
            print(f"\n{exp_name}: results.json not found")
            continue

        try:
            exp_results = evaluate_experiment(results_path, exp_name)
            all_results[exp_name] = exp_results

            print(f"  ✓ Correct: {exp_results['correct']}/{exp_results['total']}")
            print(f"  ✓ Accuracy: {exp_results['accuracy']:.1f}%")

            # Save individual experiment results if requested
            if args.save_details:
                details_path = (
                    Path(args.output_dir) / exp_dir / "evaluation_details.json"
                )
                with open(details_path, "w") as f:
                    json.dump(exp_results, f, indent=2)
                print(f"  ✓ Details saved to: {details_path.name}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - GSM8K Extension Results (Samples 30-100)")
    print("=" * 80)

    print(f"\n{'Strategy':<30} {'Model':<15} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 65)

    for exp_name in sorted(all_results.keys()):
        result = all_results[exp_name]

        # Extract strategy and model from name
        if "Baseline" in exp_name:
            strategy = "Baseline (CoT)"
        elif "Self-Consistency" in exp_name:
            strategy = "Self-Consistency"
        elif "DeepConf" in exp_name:
            strategy = "DeepConf"
        else:
            strategy = "Unknown"

        if "4o-mini" in exp_name:
            model = "GPT-4o-mini"
        elif "4o" in exp_name:
            model = "GPT-4o"
        else:
            model = "Unknown"

        correct_str = f"{result['correct']}/{result['total']}"
        accuracy_str = f"{result['accuracy']:.1f}%"

        print(f"{strategy:<30} {model:<15} {correct_str:<10} {accuracy_str:<10}")

    # Save complete results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_dir) / f"evaluation_one_by_one_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Complete results saved to: {output_file}")


if __name__ == "__main__":
    main()
