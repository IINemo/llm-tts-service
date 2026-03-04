#!/usr/bin/env python3
"""Generate cached_examples.json by running all strategy/scorer combos via the backend."""

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8080"
OUTPUT_PATH = Path("service_app/static/debugger/cached_examples.json")

QUESTIONS = [
    {
        "id": "modular_arithmetic",
        "title": "Modular Arithmetic",
        "description": "Find the remainder when 7^100 is divided by 13 — requires Fermat's Little Theorem.",
        "question": "Find the remainder when $7^{100}$ is divided by 13.",
        "ground_truth": "9",
    },
    {
        "id": "lattice_paths",
        "title": "Lattice Path Counting",
        "description": "Count shortest lattice paths through an intermediate point — requires combinatorics.",
        "question": "How many shortest lattice paths are there from (0,0) to (5,5) that pass through (2,2)?",
        "ground_truth": "120",
    },
    {
        "id": "conditional_probability",
        "title": "Conditional Probability",
        "description": "Conditional probability with combinatorics — requires Bayes' theorem and counting.",
        "question": "A box contains 5 red, 4 blue, and 3 green balls. Three balls are drawn without replacement. Given that at least one ball is red, what is the probability that all three balls are different colors? Express your answer as a fraction.",
        "ground_truth": "12/37",
    },
]

# Strategy/scorer combos to run — None means no scorer needed
COMBOS = [
    ("baseline", None),
    ("beam_search", "prm"),
    ("online_best_of_n", "prm"),
    ("offline_best_of_n", "prm"),
    ("self_consistency", None),
]

BUDGET = 8
SYSTEM_PROMPT = "Reason step-by-step. Return the final answer in \\boxed{}."


def run_single(provider: str, model_id: str, api_key: str,
               question: str, strategy_id: str, scorer_id: str | None) -> dict:
    """Run a single strategy via the streaming endpoint and return the payload."""
    body = {
        "question": question,
        "budget": BUDGET,
        "provider": provider,
        "model_id": model_id,
        "api_key": api_key,
        "strategy_id": strategy_id,
        "scorer_id": scorer_id,
        "advanced_config_yaml": f"prompt: \"{SYSTEM_PROMPT}\"\n",
    }

    resp = requests.post(
        f"{BASE_URL}/v1/debugger/demo/run-single-stream",
        json=body,
        stream=True,
        timeout=900,
    )
    resp.raise_for_status()

    payload = None
    error = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event.get("type") == "progress":
            print(f"    [{event.get('message', '')}]", flush=True)
        elif event.get("type") == "complete":
            payload = event["payload"]
        elif event.get("type") == "error":
            error = event.get("message", "unknown error")

    if error:
        raise RuntimeError(error)
    if not payload:
        raise RuntimeError("Stream ended without result")
    return payload


def save_progress(examples, adv_templates):
    """Save current state to disk after every successful run."""
    output = {"examples": examples}
    if adv_templates:
        merged = {}
        for key, tmpl in adv_templates.items():
            if tmpl.get("config"):
                merged[key] = tmpl["config"]
        if merged:
            output["advanced_config_templates"] = merged
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    [saved {OUTPUT_PATH} — {OUTPUT_PATH.stat().st_size:,} bytes]", flush=True)


def main():
    provider = sys.argv[1] if len(sys.argv) > 1 else "openrouter"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "anthropic/claude-sonnet-4"
    api_key = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        print("Error: no API key. Pass as arg or set OPENROUTER_API_KEY in .env")
        sys.exit(1)

    # Validate model first
    print(f"Validating {provider}:{model_id}...")
    val = requests.post(
        f"{BASE_URL}/v1/debugger/demo/validate-model",
        json={"provider": provider, "model_id": model_id, "api_key": api_key},
        timeout=60,
    ).json()
    if "strategies" not in val:
        print(f"Validation failed: {val}")
        sys.exit(1)
    avail_strategies = {s["id"] for s in val["strategies"]}
    avail_scorers = {s["id"] for s in val.get("scorers", [])}
    print(f"  strategies: {avail_strategies}")
    print(f"  scorers: {avail_scorers}")
    print(f"  logprobs={val['supports_logprobs']}, prefill={val['supports_prefill']}")

    # Fetch advanced config templates for reference
    adv_templates = {}
    for strat_id, scorer_id in COMBOS:
        if strat_id not in avail_strategies:
            continue
        if scorer_id and scorer_id not in avail_scorers:
            continue
        try:
            params = f"strategy_id={strat_id}"
            if scorer_id:
                params += f"&scorer_id={scorer_id}"
            tmpl = requests.get(
                f"{BASE_URL}/v1/debugger/demo/advanced-config/template?{params}",
                timeout=30,
            ).json()
            adv_templates[f"{strat_id}__{scorer_id or 'none'}"] = tmpl
        except Exception as e:
            print(f"  Warning: couldn't fetch template for {strat_id}/{scorer_id}: {e}")

    examples = []
    last_payload = None

    for q in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Question: {q['title']}")
        print(f"  {q['question']}")
        print(f"  Ground truth: {q['ground_truth']}")

        all_strategy_runs = []

        # Find or create the example entry for incremental updates
        example_entry = {
            "id": q["id"],
            "title": q["title"],
            "description": q["description"],
            "available_budgets": [BUDGET],
            "default_budget": BUDGET,
            "payloads": {str(BUDGET): None},
        }
        examples.append(example_entry)

        for strat_id, scorer_id in COMBOS:
            if strat_id not in avail_strategies:
                print(f"  SKIP {strat_id} (not available for this model)")
                continue
            if scorer_id and scorer_id not in avail_scorers:
                print(f"  SKIP {strat_id}+{scorer_id} (scorer not available)")
                continue

            combo_label = f"{strat_id}+{scorer_id}" if scorer_id else strat_id
            print(f"\n  Running {combo_label}...", flush=True)
            t0 = time.time()

            try:
                payload = run_single(
                    provider=provider,
                    model_id=model_id,
                    api_key=api_key,
                    question=q["question"],
                    strategy_id=strat_id,
                    scorer_id=scorer_id,
                )
                last_payload = payload
                elapsed = time.time() - t0

                runs = payload.get("strategies", [])
                if runs:
                    run_entry = runs[0]
                    final = run_entry.get("run", {}).get("final", {})
                    print(f"    Done in {elapsed:.1f}s — answer={final.get('answer', '?')}, "
                          f"correct={final.get('is_correct')}, "
                          f"confidence={final.get('confidence', 0):.4f}, "
                          f"tokens={run_entry.get('run', {}).get('tokens_used', '?')}")
                    all_strategy_runs.append(run_entry)
                else:
                    print(f"    WARNING: no strategy runs in payload")

            except Exception as e:
                print(f"    FAILED: {e}")

            # Re-rank and save after every successful run
            if all_strategy_runs:
                ranked = sorted(
                    all_strategy_runs,
                    key=lambda r: (
                        not r.get("run", {}).get("final", {}).get("is_correct", False),
                        -r.get("run", {}).get("final", {}).get("confidence", 0),
                    ),
                )
                for i, r in enumerate(ranked):
                    r["comparison_rank"] = i + 1

                scenario = {
                    "id": q["id"],
                    "title": q["title"],
                    "description": q["description"],
                    "prompt": f"{SYSTEM_PROMPT}\n\nQuestion: {q['question']}",
                    "ground_truth": q["ground_truth"],
                    "shared_prompt": SYSTEM_PROMPT,
                    "input_source": "cached_generation",
                    "model_config": {
                        "provider": provider,
                        "model_id": model_id,
                        "api_key_masked": val.get("api_key_masked", "***"),
                    },
                    "strategy_count": len(set(r["strategy_id"] for r in ranked)),
                    "scorer_count": len(set(r.get("scorer_id", "") for r in ranked if r.get("scorer_id"))),
                    "run_count": len(ranked),
                }

                ref = last_payload or {}
                example_entry["payloads"][str(BUDGET)] = {
                    "scenario": scenario,
                    "available_budgets": [BUDGET],
                    "selected_budget": BUDGET,
                    "strategy_catalog": ref.get("strategy_catalog", []),
                    "scorer_catalog": ref.get("scorer_catalog", []),
                    "strategies": ranked,
                }

                # Save after every run
                save_progress(
                    [e for e in examples if e["payloads"].get(str(BUDGET))],
                    adv_templates,
                )

    # Final summary
    valid_examples = [e for e in examples if e["payloads"].get(str(BUDGET))]
    print(f"\n{'='*60}")
    print(f"Done! {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)")
    print(f"Examples: {len(valid_examples)}")
    for ex in valid_examples:
        n_runs = len(ex['payloads'][str(BUDGET)]['strategies'])
        correct = sum(
            1 for r in ex['payloads'][str(BUDGET)]['strategies']
            if r.get('run', {}).get('final', {}).get('is_correct')
        )
        print(f"  {ex['id']}: {n_runs} runs, {correct} correct")


if __name__ == "__main__":
    main()
