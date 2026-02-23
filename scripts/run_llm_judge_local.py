#!/usr/bin/env python3
"""
Run LLM-as-a-judge evaluation locally on results.json

Usage:
    python scripts/run_llm_judge_local.py outputs/.../results.json

API keys are loaded from .env file (OPENAI_API_KEY or OPENROUTER_API_KEY).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_tts.evaluation.llm_as_a_judge import (  # noqa: E402
    PROMPT_ANSWER_ONLY,
    PROMPT_FULL_SOLUTION,
)


def parse_reply(reply: str) -> int:
    """Parse a single reply and return label (1=correct, 0=incorrect, -1=unclear)."""
    if "<Grade>: Correct" in reply:
        return 1
    elif "<Grade>: Incorrect" in reply:
        return 0
    else:
        return -1


def evaluate_sample(client, model, sample, idx, budget=1, mode="full_solution"):
    """Evaluate a single sample with optional majority voting.

    Args:
        mode: "full_solution" - pass entire reasoning to judge
              "answer_only" - compare just extracted answer vs gold
    """
    if mode == "answer_only":
        # Try to get extracted answer, fall back to generated_answer field
        answer = (
            sample.get("extracted_answer")
            or sample.get("generated_answer")
            or sample.get("generated_trajectory", "")
        )
        prompt = PROMPT_ANSWER_ONLY.format(
            problem=sample["question"],
            answer=answer,
            gold_answer=sample["gold_answer"],
        )
    else:
        prompt = PROMPT_FULL_SOLUTION.format(
            problem=sample["question"],
            solution=sample["generated_trajectory"],
            gold_answer=sample["gold_answer"],
        )

    votes = []
    replies = []

    for i in range(budget):
        # Use unique prompt per vote to avoid caching identical responses
        if budget > 1:
            vote_prompt = f"{prompt}\n<!-- vote {i+1}/{budget} -->"
        else:
            vote_prompt = prompt

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an intelligent assistant."},
                    {"role": "user", "content": vote_prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            reply = response.choices[0].message.content
            label = parse_reply(reply)
        except Exception as e:
            reply = f"Error: {e}"
            label = -1

        votes.append(label)
        replies.append(reply)

    # Majority voting (exclude unclear votes)
    valid_votes = [v for v in votes if v >= 0]
    if valid_votes:
        correct_votes = sum(1 for v in valid_votes if v == 1)
        incorrect_votes = sum(1 for v in valid_votes if v == 0)
        final_label = 1 if correct_votes > incorrect_votes else 0
        consensus = max(correct_votes, incorrect_votes) / len(valid_votes)
        result_str = (
            f"Correct ({correct_votes}/{len(valid_votes)})"
            if final_label == 1
            else f"Incorrect ({incorrect_votes}/{len(valid_votes)})"
        )
    else:
        final_label = -1
        consensus = 0.0
        result_str = "Unclear"

    # Combine all replies for transparency
    if budget > 1:
        combined_reply = f"=== Majority Vote: {result_str} ===\n"
        for i, (reply, vote) in enumerate(zip(replies, votes)):
            vote_str = {1: "Correct", 0: "Incorrect", -1: "Unclear"}[vote]
            combined_reply += f"\n--- Vote {i+1}: {vote_str} ---\n{reply}\n"
    else:
        combined_reply = replies[0]

    return idx, final_label, combined_reply, consensus


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation")
    parser.add_argument("results_json", help="Path to results.json")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument(
        "--base-url", default="https://api.openai.com/v1", help="API base URL"
    )
    parser.add_argument(
        "--n-threads", type=int, default=50, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=3,
        help="Number of evaluations for majority voting (default: 3)",
    )
    parser.add_argument(
        "--mode",
        choices=["full_solution", "answer_only"],
        default="answer_only",
        help="Evaluation mode: 'full_solution' passes entire reasoning, 'answer_only' compares just answers (default)",
    )
    parser.add_argument(
        "--output", help="Output file (default: results_with_judge.json)"
    )
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable")
        return 1

    # Load results
    print(f"Loading {args.results_json}...")
    with open(args.results_json) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} samples")

    # Create client
    client = openai.OpenAI(api_key=api_key, base_url=args.base_url)

    # Evaluate in parallel
    budget_str = f", budget={args.budget}" if args.budget > 1 else ""
    print(
        f"Evaluating with {args.model} using {args.n_threads} threads (mode={args.mode}{budget_str})..."
    )
    judgments = {}

    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = {
            executor.submit(
                evaluate_sample, client, args.model, sample, idx, args.budget, args.mode
            ): idx
            for idx, sample in enumerate(results)
        }

        correct = 0
        incorrect = 0
        unclear = 0

        for future in tqdm(as_completed(futures), total=len(results), desc="Judging"):
            idx, label, reply, consensus = future.result()
            judgments[idx] = {"label": label, "reply": reply, "consensus": consensus}

            if label == 1:
                correct += 1
            elif label == 0:
                incorrect += 1
            else:
                unclear += 1

    # Add judgments to results
    for idx, sample in enumerate(results):
        j = judgments[idx]
        sample["llm_judge"] = {
            "is_correct": j["label"] == 1,
            "label": j["label"],
            "consensus": j["consensus"],
            "mode": args.mode,
            "response": j["reply"],
        }

    # Print summary
    total = len(results)
    print("\n=== LLM Judge Results ===")
    print(f"Correct:   {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Incorrect: {incorrect}/{total} ({100*incorrect/total:.1f}%)")
    if unclear > 0:
        print(f"Unclear:   {unclear}/{total} ({100*unclear/total:.1f}%)")

    # Compare with exact match
    exact_correct = sum(
        1
        for s in results
        if s.get("eval", {}).get("exact_match", {}).get("is_correct", False)
    )
    print(f"\nExact match: {exact_correct}/{total} ({100*exact_correct/total:.1f}%)")

    # Save results
    output_path = args.output or args.results_json.replace(".json", "_with_judge.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
