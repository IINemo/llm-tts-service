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
"""

import argparse
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
LOGPROBS_PROBE_VALUES = [20, 10, 5, 3, 1]

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


def check_reachable(client: OpenAI, model: str) -> Optional[str]:
    """Quick check that the model responds at all. Returns error string or None."""
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply OK."}],
            max_tokens=4,
            temperature=0,
        )
        return None
    except Exception as e:
        return _compact_error(e)


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
    client: OpenAI, model: str
) -> tuple[Optional[bool], Optional[int], int, str]:
    """Probe logprobs support and max returned count.

    Returns: (supported, max_accepted, max_returned, note)
    """
    # Try each value from highest to lowest
    for value in LOGPROBS_PROBE_VALUES:
        accepted, returned, err = _try_logprobs(client, model, value)
        if accepted:
            if returned > 0:
                return True, value, returned, f"accepted={value}, returned={returned}"
            else:
                # Accepted but returned nothing — try lower values too
                # to confirm it's truly zero, not a fluke
                _, returned_1, _ = _try_logprobs(client, model, 1)
                if returned_1 > 0:
                    return (
                        True,
                        value,
                        returned_1,
                        f"accepted={value} but only returned {returned_1}",
                    )
                return (
                    False,
                    value,
                    0,
                    f"accepted top_logprobs={value} but returned 0 logprobs",
                )
        # If rejected, try next lower value
        continue

    # Nothing worked
    return False, None, 0, "logprobs rejected for all values"


# ---------------------------------------------------------------------------
# Prefill probing
# ---------------------------------------------------------------------------


def probe_prefill(client: OpenAI, model: str) -> tuple[bool, str]:
    """Probe prefill (assistant message continuation) support.

    Returns: (supported, note)
    """
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
            return False, f"rejected: {err[:100]}"
        return False, f"error: {err[:100]}"

    text = (
        (response.choices[0].message.content or "").strip() if response.choices else ""
    )
    if not text:
        return False, "empty response"

    # Check full echo: response includes prefix + continuation
    if text.startswith(PREFILL_TEXT):
        return True, "full echo (response starts with prefill)"

    # Check continuation-only: response continues mid-sentence
    first_char = text[0]
    if first_char in ("'", ",", ".", ";", " ", "-") or (
        first_char.isalpha() and first_char.islower()
    ):
        return True, f"continuation: {text[:50]!r}"

    return False, f"no continuation: {text[:50]!r}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def probe_model(
    client: OpenAI,
    provider: str,
    model: str,
    skip_logprobs: bool = False,
    skip_prefill: bool = False,
) -> ProbeResult:
    result = ProbeResult(provider=provider, model=model)

    print(f"  {model} ... ", end="", flush=True)

    # Reachability check
    err = check_reachable(client, model)
    if err:
        result.reachable = False
        result.error = err
        print(f"UNREACHABLE ({err[:60]})")
        return result

    parts = []

    # Logprobs
    if not skip_logprobs:
        supported, max_accepted, max_returned, note = probe_logprobs(client, model)
        result.logprobs_supported = supported
        result.logprobs_max_accepted = max_accepted
        result.logprobs_max_returned = max_returned
        result.logprobs_note = note
        if supported:
            parts.append(f"logprobs={max_returned}/{max_accepted}")
        else:
            parts.append(f"logprobs=NO")

    # Prefill
    if not skip_prefill:
        supported, note = probe_prefill(client, model)
        result.prefill_supported = supported
        result.prefill_note = note
        parts.append(f"prefill={'YES' if supported else 'NO'}")

    print(", ".join(parts))
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
    args = parser.parse_args()

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
                client, provider, model, args.skip_logprobs, args.skip_prefill
            )
            results.append(result)

    print_summary(results, args.skip_logprobs, args.skip_prefill)


if __name__ == "__main__":
    main()
