#!/usr/bin/env python3
"""Add CRT system example to cached_examples.json."""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8080"
OUTPUT = "service_app/static/debugger/cached_examples.json"
API_KEY = os.environ["OPENROUTER_API_KEY"]
PROVIDER = "openrouter"
MODEL = "anthropic/claude-sonnet-4"
BUDGET = 8
SYSTEM_PROMPT = "Reason step-by-step. Return the final answer in \\boxed{}."

Q = {
    "id": "chinese_remainder_theorem",
    "title": "Chinese Remainder Theorem",
    "description": "Solve a system of three modular congruences using the Chinese Remainder Theorem.",
    "question": (
        "Find the smallest positive integer n such that "
        "n ≡ 3 (mod 5), n ≡ 4 (mod 7), and n ≡ 2 (mod 9)."
    ),
    "ground_truth": "263",
}

COMBOS = [
    ("baseline", None),
    ("beam_search", "prm"),
    ("online_best_of_n", "prm"),
    ("offline_best_of_n", "prm"),
    ("self_consistency", None),
]


def run_single(strat_id, scorer_id):
    body = {
        "question": Q["question"],
        "budget": BUDGET,
        "provider": PROVIDER,
        "model_id": MODEL,
        "api_key": API_KEY,
        "strategy_id": strat_id,
        "scorer_id": scorer_id,
        "advanced_config_yaml": f'prompt: "{SYSTEM_PROMPT}"\n',
    }
    resp = requests.post(
        f"{BASE_URL}/v1/debugger/demo/run-single-stream",
        json=body, stream=True, timeout=900,
    )
    resp.raise_for_status()
    payload, error = None, None
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
    with open(OUTPUT) as f:
        data = json.load(f)

    data["examples"] = [e for e in data["examples"] if e["id"] != Q["id"]]

    all_runs = []
    last_payload = None

    print(f"Question: {Q['title']}")
    print(f"  {Q['question']}")
    print(f"  Ground truth: {Q['ground_truth']}")

    for strat_id, scorer_id in COMBOS:
        label = f"{strat_id}+{scorer_id}" if scorer_id else strat_id
        print(f"\n  Running {label}...", flush=True)
        t0 = time.time()
        try:
            payload = run_single(strat_id, scorer_id)
            last_payload = payload
            runs = payload.get("strategies", [])
            if runs:
                entry = runs[0]
                final = entry["run"]["final"]
                events = entry["run"].get("events", [])
                print(
                    f"    Done in {time.time()-t0:.1f}s — "
                    f"conf={final.get('confidence', 0):.4f}, "
                    f"tokens={entry['run'].get('tokens_used', '?')}, "
                    f"events={len(events)}"
                )
                all_runs.append(entry)
        except Exception as e:
            print(f"    FAILED: {e}")

        if all_runs:
            ranked = sorted(all_runs, key=lambda r: (
                not r.get("run", {}).get("final", {}).get("is_correct", False),
                -r.get("run", {}).get("final", {}).get("confidence", 0),
            ))
            for i, r in enumerate(ranked):
                r["comparison_rank"] = i + 1

            ref = last_payload or {}
            new_example = {
                "id": Q["id"],
                "title": Q["title"],
                "description": Q["description"],
                "available_budgets": [BUDGET],
                "default_budget": BUDGET,
                "payloads": {
                    str(BUDGET): {
                        "scenario": {
                            "id": Q["id"],
                            "title": Q["title"],
                            "description": Q["description"],
                            "prompt": f"{SYSTEM_PROMPT}\n\nQuestion: {Q['question']}",
                            "ground_truth": Q["ground_truth"],
                            "shared_prompt": SYSTEM_PROMPT,
                            "input_source": "cached_generation",
                            "model_config": {
                                "provider": PROVIDER,
                                "model_id": MODEL,
                                "api_key_masked": "sk-or...demo",
                            },
                            "strategy_count": len(set(r["strategy_id"] for r in ranked)),
                            "scorer_count": len(
                                set(r.get("scorer_id", "") for r in ranked if r.get("scorer_id"))
                            ),
                            "run_count": len(ranked),
                        },
                        "available_budgets": [BUDGET],
                        "selected_budget": BUDGET,
                        "strategy_catalog": ref.get("strategy_catalog", []),
                        "scorer_catalog": ref.get("scorer_catalog", []),
                        "strategies": ranked,
                    }
                },
            }

            idx = next(
                (i for i, e in enumerate(data["examples"]) if e["id"] == Q["id"]),
                None,
            )
            if idx is not None:
                data["examples"][idx] = new_example
            else:
                data["examples"].append(new_example)

            with open(OUTPUT, "w") as f:
                json.dump(data, f, indent=2)
            print(f"    [saved — {len(ranked)} runs total]", flush=True)

    print(f"\nDone! {len(all_runs)} runs for {Q['id']}")


if __name__ == "__main__":
    main()
