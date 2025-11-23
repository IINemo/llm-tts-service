#!/usr/bin/env python3
"""Evaluate GSM8K results using LLM judge."""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

# Use GPT-4o for evaluation via OpenRouter
MODEL = "openai/gpt-4o"
BASE_URL = "https://openrouter.ai/api/v1"


def evaluate_single_answer(generated: str, gold: str, api_key: str) -> bool:
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
        "Authorization": f"Bearer {api_key}",
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
            f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            verdict_text = data["choices"][0]["message"]["content"].strip().lower()
            is_correct = "correct" in verdict_text and "incorrect" not in verdict_text
            return is_correct
        else:
            print(f"    API error {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K results with LLM judge")
    parser.add_argument("results_file", type=str, help="Path to results.json file")
    parser.add_argument("--api-key", type=str, required=True, help="OpenRouter API key")
    parser.add_argument("--save-details", action="store_true", help="Save detailed evaluation results")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        return

    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)

    print("=" * 80)
    print(f"GSM8K LLM Judge Evaluation")
    print(f"Using {MODEL} as judge")
    print(f"Results file: {results_path}")
    print(f"Total samples: {len(results)}")
    print("=" * 80)

    correct = 0
    evaluations = []

    # Process each sample
    for i, result in enumerate(tqdm(results, desc="Evaluating")):
        sample_id = result.get("index", i)
        generated = result.get("generated_answer", "").strip()
        gold = result.get("gold_answer", "").strip()
        question = result.get("question", "")

        if not generated:
            generated = "No answer"
        if not gold:
            gold = "No answer"

        # Evaluate this answer
        is_correct = evaluate_single_answer(generated, gold, args.api_key)

        if is_correct:
            correct += 1

        evaluations.append(
            {
                "sample_id": sample_id,
                "question": question,
                "gold": gold,
                "generated": generated[:500] + "..." if len(generated) > 500 else generated,
                "correct": is_correct,
            }
        )

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    accuracy = (correct / len(results) * 100) if len(results) > 0 else 0

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total samples: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {len(results) - correct}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Save results
    output_data = {
        "total": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "evaluations": evaluations if args.save_details else [],
    }

    output_file = results_path.parent / "llm_judge_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
