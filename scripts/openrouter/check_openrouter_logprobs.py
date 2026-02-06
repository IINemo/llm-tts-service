#!/usr/bin/env python3
"""
OpenRouter Logprobs Support Checker

This script tests which models on OpenRouter support logprobs.
Run with: python scripts/check_openrouter_logprobs.py

Requirements:
    - pip install requests
    - OPENROUTER_API_KEY in .env file or environment
"""

import json
import os
import sys
from typing import Any, Dict, List

import requests


def load_api_key() -> str:
    """Load OpenRouter API key from .env file or environment."""
    # Try environment variable first
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # Try loading from .env file
    env_path = os.path.join(os.path.dirname(__file__), "../.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()

    raise ValueError(
        "OPENROUTER_API_KEY not found. Set it in .env file or as environment variable."
    )


def test_model_logprobs(model: str, api_key: str) -> Dict[str, Any]:
    """
    Test if a model supports logprobs.

    Args:
        model: Model ID to test
        api_key: OpenRouter API key

    Returns:
        Dict with keys: 'model', 'supported', 'provider', 'error', 'content'
    """
    result = {
        "model": model,
        "supported": False,
        "provider": None,
        "error": None,
        "content": None,
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://llm-tts-service.local",
                "X-Title": "Logprobs Checker",
            },
            data=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "logprobs": True,
                    "top_logprobs": 3,
                    "max_tokens": 10,
                }
            ),
            timeout=15,
        )

        data = response.json()

        if "error" in data:
            result["error"] = data["error"].get("message", "Unknown error")
        elif "choices" in data and len(data["choices"]) > 0:
            result["provider"] = data.get("provider", "unknown")
            logprobs = data["choices"][0].get("logprobs")
            result["content"] = data["choices"][0].get("message", {}).get("content", "")

            # Check if logprobs are actually present
            if logprobs and logprobs is not None and logprobs != {}:
                result["supported"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


def get_all_models(api_key: str) -> List[str]:
    """Get list of all available models from OpenRouter."""
    response = requests.get(
        url="https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    return [m["id"] for m in response.json().get("data", [])]


def print_results(results: List[Dict[str, Any]]):
    """Print test results in a nice format."""
    supported = [r for r in results if r["supported"]]
    not_supported = [r for r in results if not r["supported"] and not r["error"]]
    errors = [r for r in results if r["error"]]

    print("\n" + "=" * 80)
    print(f"RESULTS: {len(results)} models tested")
    print("=" * 80)

    if supported:
        print(f"\n✅ MODELS SUPPORTING LOGPROBS ({len(supported)}):")
        print("-" * 80)
        for r in supported:
            print(f"  • {r['model']}")
            print(f"    Provider: {r['provider']}")
            if r["content"]:
                print(f"    Response: {r['content'][:60]}...")
            print()

    if not_supported:
        print(f"\n❌ MODELS NOT SUPPORTING LOGPROBS ({len(not_supported)}):")
        print("-" * 80)
        for r in not_supported:
            print(f"  • {r['model']} (provider: {r['provider']})")
        print()

    if errors:
        print(f"\n⚠️  ERRORS ({len(errors)}):")
        print("-" * 80)
        for r in errors:
            err = r["error"]
            if len(err) > 70:
                err = err[:67] + "..."
            print(f"  • {r['model']}: {err}")
        print()


# Default models to test
DEFAULT_MODELS = [
    # OpenAI - known to support logprobs
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/gpt-4o-2024-05-13",
    "openai/chatgpt-4o-latest",
    # Popular models that likely don't support logprobs
    "openai/gpt-oss-120b",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2512",
    "qwen/qwen-turbo",
    "deepseek/deepseek-r1",
    "nousresearch/hermes-4-70b",
]


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check which OpenRouter models support logprobs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (space-separated)",
        default=None,
    )
    parser.add_argument(
        "--all", action="store_true", help="Test all available models (slow!)"
    )
    parser.add_argument(
        "--category",
        choices=[
            "openai",
            "anthropic",
            "google",
            "meta",
            "mistral",
            "qwen",
            "deepseek",
        ],
        help="Test models from a specific provider",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine which models to test
    if args.all:
        print("Fetching all models from OpenRouter...")
        models_to_test = get_all_models(api_key)
        print(f"Found {len(models_to_test)} models. This may take a while...")
    elif args.category:
        all_models = get_all_models(api_key)
        prefix_map = {
            "openai": "openai/",
            "anthropic": "anthropic/",
            "google": "google/",
            "meta": "meta-llama/",
            "mistral": "mistralai/",
            "qwen": "qwen/",
            "deepseek": "deepseek/",
        }
        prefix = prefix_map[args.category]
        models_to_test = [m for m in all_models if m.startswith(prefix)]
    elif args.models:
        models_to_test = args.models
    else:
        models_to_test = DEFAULT_MODELS

    print(f"Testing {len(models_to_test)} models for logprobs support...\n")

    results = []
    for i, model in enumerate(models_to_test, 1):
        print(f"[{i}/{len(models_to_test)}] Testing: {model}", end=" ... ")
        sys.stdout.flush()

        result = test_model_logprobs(model, api_key)
        results.append(result)

        if result["supported"]:
            print(f"✅ YES (provider: {result['provider']})")
        elif result["error"]:
            print("⚠️  ERROR")
        else:
            print(f"❌ NO (provider: {result['provider']})")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)

    # Print summary
    print("=" * 80)
    print(
        f"SUMMARY: {len([r for r in results if r['supported']])} / {len(results)} models support logprobs"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
