#!/usr/bin/env python3
"""
Discover which Claude models on OpenRouter *actually* honor assistant-prefill continuation.

What it does:
1) Fetches OpenRouter model list (/api/v1/models)
2) Filters to Anthropic Claude models (by model id)
3) For each model, sends a chat completion where the last message is role="assistant" (prefill)
4) Checks whether the returned assistant content starts with / contains the prefill prefix
5) Prints a results table + saves JSONL with raw outcomes

Install:
  pip install openai python-dotenv

Env:
  OPENROUTER_API_KEY=...
Optional:
  OPENROUTER_SITE_URL=https://example.com
  OPENROUTER_APP_NAME=PrefillProbe
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODELS_URL = f"{OPENROUTER_BASE}/models"

PROMPT = "Explain what a transformer model is in simple terms (1-2 short sentences)."
PREFILL = "A transformer model is a type of neural network that"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def fetch_models(api_key: str) -> List[Dict[str, Any]]:
    r = requests.get(
        MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    # OpenRouter returns {"data": [...]} typically
    models = data.get("data", [])
    if not isinstance(models, list):
        return []
    return models


def is_claude_model(model_id: str) -> bool:
    mid = model_id.lower()
    # Typical OpenRouter naming: anthropic/claude-...
    if not mid.startswith("anthropic/claude"):
        return False
    return True


def call_prefill(
    client: OpenAI,
    model_id: str,
    site_url: str | None,
    app_name: str | None,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    extra_headers = {}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if app_name:
        extra_headers["X-Title"] = app_name

    resp = client.chat.completions.create(
        model=model_id,
        extra_headers=extra_headers if extra_headers else None,
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": PREFILL},  # assistant prefill
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = (resp.choices[0].message.content or "").strip()
    # Keep a small raw snapshot for debugging (avoid huge dumps)
    raw = {
        "id": getattr(resp, "id", None),
        "model": model_id,
        "content": content,
        "finish_reason": getattr(resp.choices[0], "finish_reason", None),
        "usage": (
            getattr(resp, "usage", None).model_dump()
            if getattr(resp, "usage", None)
            else None
        ),
    }
    return content, raw


def score_prefill(output: str) -> Dict[str, bool]:
    return {
        "starts_with": output.startswith(PREFILL),
        "contains": (PREFILL in output),
    }


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of Claude models tested (0 = all).",
    )
    ap.add_argument(
        "--sleep", type=float, default=0.4, help="Seconds to sleep between requests."
    )
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=120)
    ap.add_argument(
        "--out",
        type=str,
        default="prefill_probe_results.jsonl",
        help="Write JSONL results here.",
    )
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        eprint("OPENROUTER_API_KEY is not set")
        sys.exit(1)

    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME", "PrefillProbe")

    # OpenAI SDK configured for OpenRouter
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)

    models = fetch_models(api_key)
    claude_ids = sorted(
        {
            m.get("id")
            for m in models
            if isinstance(m.get("id"), str) and is_claude_model(m["id"])
        }
    )
    if args.limit and args.limit > 0:
        claude_ids = claude_ids[: args.limit]

    if not claude_ids:
        eprint("No Claude models found in /models response.")
        sys.exit(2)

    print(f"Found {len(claude_ids)} Claude models. Testing assistant-prefill…")
    print(f"Prefill: {PREFILL!r}")
    print()

    results = []
    ok_starts = 0
    ok_contains = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for i, mid in enumerate(claude_ids, 1):
            status = "OK"
            try:
                out, raw = call_prefill(
                    client=client,
                    model_id=mid,
                    site_url=site_url,
                    app_name=app_name,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                sc = score_prefill(out)
                ok_starts += int(sc["starts_with"])
                ok_contains += int(sc["contains"])

                row = {
                    "model": mid,
                    "starts_with_prefill": sc["starts_with"],
                    "contains_prefill": sc["contains"],
                    "output_preview": out[:220],
                    "raw": raw,
                }
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                print(
                    f"[{i:>2}/{len(claude_ids)}] {mid:40} "
                    f"starts_with={str(sc['starts_with']):5}  contains={str(sc['contains']):5}"
                )
            except Exception as e:
                status = "ERR"
                row = {"model": mid, "error": str(e)}
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{i:>2}/{len(claude_ids)}] {mid:40} {status}: {e}")

            time.sleep(args.sleep)

    # Summary
    print("\nSummary:")
    print(f"  starts_with_prefill=True : {ok_starts}/{len(claude_ids)}")
    print(f"  contains_prefill=True    : {ok_contains}/{len(claude_ids)}")
    print(f"\nWrote detailed results to: {args.out}")

    # Print a short "supported" list (strict: starts_with)
    supported = [r["model"] for r in results if r.get("starts_with_prefill") is True]
    if supported:
        print("\nModels that STRICTLY continued the prefill (starts_with=True):")
        for m in supported:
            print(f"  - {m}")
    else:
        print(
            "\nNo models strictly continued the prefill in this run (starts_with=True)."
        )
        print(
            "Note: many providers treat assistant-prefill as context only; try a different prefix or prompt."
        )


if __name__ == "__main__":
    main()
