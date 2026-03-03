#!/usr/bin/env python3
"""Probe model capabilities: logprobs support (with actual return count) and prefill support.

Usage:
    # Test all models (keys from env OPENAI_API_KEY / OPENROUTER_API_KEY):
    python scripts/probe_top_logprobs.py

    # Test specific provider:
    python scripts/probe_top_logprobs.py --provider openrouter

    # Test a specific model:
    python scripts/probe_top_logprobs.py --provider openrouter --model anthropic/claude-sonnet-4

    # Override API key:
    python scripts/probe_top_logprobs.py --provider openai --api-key sk-...

    # Skip prefill test (faster):
    python scripts/probe_top_logprobs.py --skip-prefill

    # Skip logprobs test:
    python scripts/probe_top_logprobs.py --skip-logprobs

    # Disable cache:
    python scripts/probe_top_logprobs.py --no-cache

    # Custom cache path:
    python scripts/probe_top_logprobs.py --cache /tmp/probe_cache.json
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROVIDER_BASE_URLS = {
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
}

MODELS_BY_PROVIDER = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o4-mini",
        "o3-mini",
    ],
    "openrouter": [
        # OpenAI
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/o4-mini",
        # Anthropic
        "anthropic/claude-sonnet-4",
        "anthropic/claude-opus-4",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku",
        # DeepSeek
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat-v3-0324",
        # Qwen
        "qwen/qwen3.5-27b",
        "qwen/qwen3-235b-a22b",
        "qwen/qwen3-30b-a3b",
        # Google
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        # Meta
        "meta-llama/llama-4-maverick",
    ],
}

# top_logprobs values to test (descending); we find the highest that works
LOGPROBS_PROBE_VALUES = [20, 10, 5, 4, 3, 2, 1]

PREFILL_TEXT = "A transformer model is a type of neural network that"


@dataclass
class ProbeResult:
    provider: str
    model: str
    # Logprobs
    logprobs_supported: Optional[bool] = None
    logprobs_max_accepted: Optional[int] = None
    logprobs_max_returned: int = 0
    logprobs_note: str = ""
    # Prefill
    prefill_supported: Optional[bool] = None
    prefill_note: str = ""
    # General
    reachable: bool = True
    error: str = ""


def make_client(provider: str, api_key: str) -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 25.0, "max_retries": 0}
    base_url = PROVIDER_BASE_URLS.get(provider)
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def load_cache(cache_path: str) -> dict:
    """Load probe cache from JSON file."""
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict, cache_path: str):
    """Save probe cache to JSON file."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Probing functions
# ---------------------------------------------------------------------------


def check_reachable(
    client: OpenAI, model: str, model_cache: Optional[dict] = None
) -> tuple[Optional[str], bool]:
    """Quick check that the model responds at all.

    Returns (error_string_or_none, from_cache).
    """
    if model_cache is not None and "reachable" in model_cache:
        cached = model_cache["reachable"]
        if cached is True:
            return None, True
        return cached, True

    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply OK."}],
            max_tokens=4,
            temperature=0,
        )
        err = None
    except Exception as e:
        err = _compact_error(e)

    if model_cache is not None:
        model_cache["reachable"] = True if err is None else err

    return err, False


# ---------------------------------------------------------------------------
# Logprobs probing
# ---------------------------------------------------------------------------


def _try_logprobs(
    client: OpenAI, model: str, top_logprobs: int
) -> tuple[bool, int, Optional[str]]:
    """Returns (accepted, actual_returned_count, error_or_none)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is 2+2? Answer with one word."}
            ],
            max_tokens=5,
            temperature=0.7,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        max_returned = 0
        for choice in resp.choices or []:
            for token_info in getattr(choice.logprobs, "content", None) or []:
                tl = getattr(token_info, "top_logprobs", None) or []
                max_returned = max(max_returned, len(tl))
        return True, max_returned, None
    except Exception as e:
        return False, 0, _compact_error(e)


def probe_logprobs(
    client: OpenAI, model: str, model_cache: Optional[dict] = None
) -> tuple[Optional[bool], Optional[int], int, str, bool]:
    """Probe logprobs support and max returned count.

    Tests every value in LOGPROBS_PROBE_VALUES and reports the best one
    that actually returns logprobs (some models only work with specific values).

    Returns: (supported, max_accepted, max_returned, note, all_from_cache)
    """
    best_value = None
    best_returned = 0
    any_accepted = False
    max_accepted = None
    all_from_cache = True

    logprobs_cache: dict = {}
    if model_cache is not None:
        logprobs_cache = model_cache.setdefault("logprobs", {})

    for value in LOGPROBS_PROBE_VALUES:
        key = str(value)
        if key in logprobs_cache:
            accepted, returned = logprobs_cache[key]
        else:
            all_from_cache = False
            accepted, returned, _err = _try_logprobs(client, model, value)
            logprobs_cache[key] = [accepted, returned]

        if accepted:
            any_accepted = True
            if max_accepted is None:
                max_accepted = value
            if returned > best_returned:
                best_value = value
                best_returned = returned

    if best_returned > 0:
        return (
            True,
            best_value,
            best_returned,
            f"best={best_value}, returned={best_returned}",
            all_from_cache,
        )
    if any_accepted:
        return (
            False,
            max_accepted,
            0,
            f"accepted top_logprobs={max_accepted} but returned 0 logprobs",
            all_from_cache,
        )
    return False, None, 0, "logprobs rejected for all values", all_from_cache


# ---------------------------------------------------------------------------
# Prefill probing
# ---------------------------------------------------------------------------


def probe_prefill(
    client: OpenAI, model: str, model_cache: Optional[dict] = None
) -> tuple[bool, str, bool]:
    """Probe prefill (assistant message continuation) support.

    Returns: (supported, note, from_cache)
    """
    if model_cache is not None and "prefill" in model_cache:
        supported, note = model_cache["prefill"]
        return supported, note, True

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Explain what a transformer model is in simple terms.",
                },
                {"role": "assistant", "content": PREFILL_TEXT},
            ],
            max_tokens=60,
            temperature=0,
        )
    except Exception as e:
        err = _compact_error(e)
        lowered = err.lower()
        if any(
            kw in lowered
            for kw in ("prefix", "prefill", "not supported", "not allowed", "invalid")
        ):
            result = (False, f"rejected: {err[:100]}")
        else:
            result = (False, f"error: {err[:100]}")
        if model_cache is not None:
            model_cache["prefill"] = list(result)
        return result[0], result[1], False

    text = (
        (response.choices[0].message.content or "").strip() if response.choices else ""
    )
    if not text:
        result = (False, "empty response")
    elif text.startswith(PREFILL_TEXT):
        result = (True, "full echo (response starts with prefill)")
    else:
        first_char = text[0]
        if first_char in ("'", ",", ".", ";", " ", "-") or (
            first_char.isalpha() and first_char.islower()
        ):
            result = (True, f"continuation: {text[:50]!r}")
        else:
            result = (False, f"no continuation: {text[:50]!r}")

    if model_cache is not None:
        model_cache["prefill"] = list(result)
    return result[0], result[1], False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def probe_model(
    client: OpenAI,
    provider: str,
    model: str,
    skip_logprobs: bool = False,
    skip_prefill: bool = False,
    cache: Optional[dict] = None,
    cache_path: Optional[str] = None,
) -> ProbeResult:
    result = ProbeResult(provider=provider, model=model)
    cache_key = f"{provider}/{model}"
    model_cache = None
    if cache is not None:
        model_cache = cache.setdefault(cache_key, {})

    all_cached = True

    print(f"  {model} ... ", end="", flush=True)

    # Reachability check
    err, from_cache = check_reachable(client, model, model_cache)
    if not from_cache:
        all_cached = False
    if err:
        result.reachable = False
        result.error = err
        suffix = " (cached)" if from_cache else ""
        print(f"UNREACHABLE ({err[:60]}){suffix}")
        if cache is not None and cache_path:
            save_cache(cache, cache_path)
        return result

    parts = []

    # Logprobs
    if not skip_logprobs:
        supported, max_accepted, max_returned, note, lp_cached = probe_logprobs(
            client, model, model_cache
        )
        if not lp_cached:
            all_cached = False
        result.logprobs_supported = supported
        result.logprobs_max_accepted = max_accepted
        result.logprobs_max_returned = max_returned
        result.logprobs_note = note
        if supported:
            parts.append(f"logprobs={max_returned}/{max_accepted}")
        else:
            parts.append("logprobs=NO")

    # Prefill
    if not skip_prefill:
        supported, note, pf_cached = probe_prefill(client, model, model_cache)
        if not pf_cached:
            all_cached = False
        result.prefill_supported = supported
        result.prefill_note = note
        parts.append(f"prefill={'YES' if supported else 'NO'}")

    suffix = " (cached)" if all_cached else ""
    print(f"{', '.join(parts)}{suffix}")

    if cache is not None and cache_path:
        save_cache(cache, cache_path)

    return result


def print_summary(results: list[ProbeResult], skip_logprobs: bool, skip_prefill: bool):
    # Build header
    cols = [("Provider", 12), ("Model", 40)]
    if not skip_logprobs:
        cols += [("Logprobs", 10), ("Accepted", 10), ("Returned", 10)]
    if not skip_prefill:
        cols += [("Prefill", 10)]

    width = sum(w for _, w in cols) + len(cols) - 1
    header = " ".join(
        f"{name:<{w}}" if i == 0 or i == 1 else f"{name:>{w}}"
        for i, (name, w) in enumerate(cols)
    )

    print(f"\n{'=' * width}")
    print(header)
    print("-" * width)

    for r in results:
        parts = [f"{r.provider:<12}", f"{r.model:<40}"]

        if not skip_logprobs:
            if not r.reachable:
                parts += [f"{'--':>10}", f"{'--':>10}", f"{'--':>10}"]
            elif r.logprobs_supported is True:
                parts += [
                    f"{'YES':>10}",
                    f"{str(r.logprobs_max_accepted):>10}",
                    f"{str(r.logprobs_max_returned):>10}",
                ]
            elif r.logprobs_supported is False and r.logprobs_max_accepted is not None:
                parts += [
                    f"{'silent NO':>10}",
                    f"{str(r.logprobs_max_accepted):>10}",
                    f"{'0':>10}",
                ]
            else:
                parts += [f"{'NO':>10}", f"{'--':>10}", f"{'--':>10}"]

        if not skip_prefill:
            if not r.reachable:
                parts += [f"{'--':>10}"]
            elif r.prefill_supported is True:
                parts += [f"{'YES':>10}"]
            elif r.prefill_supported is False:
                parts += [f"{'NO':>10}"]
            else:
                parts += [f"{'--':>10}"]

        print(" ".join(parts))

    print("=" * width)


def _compact_error(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    status = getattr(exc, "status_code", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            msg = inner.get("message") or inner.get("msg")
            if msg:
                return f"Error {status}: {msg}" if status else str(msg)
        detail = body.get("detail")
        if detail:
            return f"Error {status}: {detail}" if status else str(detail)
    return str(exc)[:200]


def main():
    parser = argparse.ArgumentParser(
        description="Probe model capabilities: logprobs and prefill support"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "openrouter"],
        help="Test only this provider",
    )
    parser.add_argument(
        "--model", type=str, action="append", help="Test specific model(s) (repeatable)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set OPENAI_API_KEY / OPENROUTER_API_KEY env vars)",
    )
    parser.add_argument(
        "--skip-logprobs", action="store_true", help="Skip logprobs probing"
    )
    parser.add_argument(
        "--skip-prefill", action="store_true", help="Skip prefill probing"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="reports/probe_cache.json",
        help="Path to cache file (default: reports/probe_cache.json)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache, re-probe everything",
    )
    args = parser.parse_args()

    cache = None
    cache_path = None
    if not args.no_cache:
        cache_path = args.cache
        cache = load_cache(cache_path)

    if args.provider and args.model:
        test_plan = {args.provider: args.model}
    elif args.provider:
        test_plan = {args.provider: MODELS_BY_PROVIDER.get(args.provider, [])}
    else:
        test_plan = MODELS_BY_PROVIDER

    results: list[ProbeResult] = []

    for provider, models in test_plan.items():
        api_key = args.api_key
        if not api_key:
            if provider == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY", "")
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(f"\nSkipping {provider}: no API key (set --api-key or env var)")
            continue

        print(f"\n=== {provider.upper()} ===")
        client = make_client(provider, api_key)

        for model in models:
            result = probe_model(
                client,
                provider,
                model,
                args.skip_logprobs,
                args.skip_prefill,
                cache=cache,
                cache_path=cache_path,
            )
            results.append(result)

    print_summary(results, args.skip_logprobs, args.skip_prefill)


if __name__ == "__main__":
    main()
