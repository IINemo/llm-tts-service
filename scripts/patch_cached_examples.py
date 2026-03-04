#!/usr/bin/env python3
"""Patch cached_examples.json: re-run missing strategy/scorer combos for a given example."""

import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8080"
OUTPUT = "service_app/static/debugger/cached_examples.json"
SYSTEM_PROMPT = "Reason step-by-step. Return the final answer in \\boxed{}."


def run_single(api_key, provider, model_id, question, strat_id, scorer_id, budget):
    body = {
        "question": question, "budget": budget, "provider": provider,
        "model_id": model_id, "api_key": api_key, "strategy_id": strat_id,
        "scorer_id": scorer_id,
        "advanced_config_yaml": f'prompt: "{SYSTEM_PROMPT}"\n',
    }
    resp = requests.post(
        f"{BASE_URL}/v1/debugger/demo/run-single-stream",
        json=body, stream=True, timeout=900,
    )
    resp.raise_for_status()
    payload = None
    error = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "progress":
            print(f"    [{ev.get('message', '')}]", flush=True)
        elif ev.get("type") == "complete":
            payload = ev["payload"]
        elif ev.get("type") == "error":
            error = ev.get("message")
    if error:
        raise RuntimeError(error)
    if not payload:
        raise RuntimeError("No result")
    return payload


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: set OPENROUTER_API_KEY in .env")
        sys.exit(1)

    example_id = sys.argv[1] if len(sys.argv) > 1 else "lattice_paths"
    provider = "openrouter"
    model_id = "anthropic/claude-sonnet-4"

    with open(OUTPUT) as f:
        data = json.load(f)

    example = next((e for e in data["examples"] if e["id"] == example_id), None)
    if not example:
        print(f"Example '{example_id}' not found. Available: {[e['id'] for e in data['examples']]}")
        sys.exit(1)

    budget_key = str(example["default_budget"])
    bp = example["payloads"][budget_key]
    existing = bp["strategies"]
    existing_ids = {s["id"] for s in existing}
    question = bp["scenario"]["prompt"].split("Question: ", 1)[-1]

    # All expected combos
    all_combos = [
        ("baseline", None),
        ("beam_search", "prm"),
        ("online_best_of_n", "prm"),
        ("offline_best_of_n", "prm"),
        ("self_consistency", None),
    ]

    missing = []
    for strat_id, scorer_id in all_combos:
        run_id = f"{strat_id}__{scorer_id}" if scorer_id else strat_id
        # Check by strategy_id + scorer_id combo
        found = any(
            s.get("strategy_id") == strat_id and s.get("scorer_id") == scorer_id
            for s in existing
        )
        if not found:
            missing.append((strat_id, scorer_id))

    if not missing:
        print(f"All combos present for '{example_id}' ({len(existing)} runs). Nothing to do.")
        return

    print(f"Example: {example_id}")
    print(f"Question: {question[:100]}...")
    print(f"Existing: {len(existing)} runs")
    print(f"Missing: {missing}")

    for strat_id, scorer_id in missing:
        label = f"{strat_id}+{scorer_id}" if scorer_id else strat_id
        print(f"\n  Running {label}...", flush=True)
        t0 = time.time()
        try:
            payload = run_single(
                api_key, provider, model_id, question,
                strat_id, scorer_id, int(budget_key),
            )
            runs = payload.get("strategies", [])
            if runs:
                entry = runs[0]
                final = entry["run"]["final"]
                elapsed = time.time() - t0
                print(f"    Done in {elapsed:.1f}s — conf={final.get('confidence', 0):.4f}, "
                      f"tokens={entry['run'].get('tokens_used', '?')}")
                existing.append(entry)

                if not bp.get("strategy_catalog"):
                    bp["strategy_catalog"] = payload.get("strategy_catalog", [])
                if not bp.get("scorer_catalog"):
                    bp["scorer_catalog"] = payload.get("scorer_catalog", [])

                # Re-rank and save
                existing.sort(key=lambda r: (
                    not r.get("run", {}).get("final", {}).get("is_correct", False),
                    -r.get("run", {}).get("final", {}).get("confidence", 0),
                ))
                for i, r in enumerate(existing):
                    r["comparison_rank"] = i + 1
                bp["scenario"]["run_count"] = len(existing)

                with open(OUTPUT, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"    [saved — {len(existing)} runs total]", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nDone! '{example_id}' now has {len(existing)} runs")


if __name__ == "__main__":
    main()
