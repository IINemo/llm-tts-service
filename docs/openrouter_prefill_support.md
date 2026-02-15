# OpenRouter Assistant Prefill Support

## Overview

Assistant prefill (also called "assistant message continuation") is a technique where the last message in a chat completion request has `role: "assistant"`, causing the model to continue from that text rather than generating a fresh response. This is critical for multi-step generation strategies (e.g., adaptive scaling) that build up a trajectory incrementally.

## Testing Script

Use [`scripts/test_prefill_models.py`](../scripts/test_prefill_models.py) to probe which models on OpenRouter support prefill:

```bash
# Requires OPENROUTER_API_KEY in .env or environment
python scripts/test_prefill_models.py
```

The script sends a prompt with an assistant prefill message and checks whether the model continues from it or generates a fresh response.

## Detection Caveat

OpenRouter strips the prefill text from Claude responses and returns **only the continuation tokens**. This means naive checks like `output.startswith(prefill)` will return `False` even when prefill works correctly. The reliable way to detect prefill support is to check whether the output is a mid-sentence continuation (e.g., starts with a lowercase word that grammatically follows the prefill).

## Results (2026-02-15)

### Claude Models (Anthropic)

| Model | Prefill Support | Notes |
|---|---|---|
| `anthropic/claude-3-haiku` | Yes | |
| `anthropic/claude-3.5-haiku` | Yes | |
| `anthropic/claude-3.5-sonnet` | Yes | |
| `anthropic/claude-3.7-sonnet` | Yes | |
| `anthropic/claude-3.7-sonnet:thinking` | Yes | |
| `anthropic/claude-haiku-4.5` | Yes | |
| `anthropic/claude-opus-4` | Yes | |
| `anthropic/claude-opus-4.1` | Yes | |
| `anthropic/claude-opus-4.5` | Yes | |
| `anthropic/claude-opus-4.6` | **No** | Explicitly blocked by provider: "This model does not support assistant message prefill" |
| `anthropic/claude-sonnet-4` | Yes | |
| `anthropic/claude-sonnet-4.5` | Yes | |

**Summary:** 11/12 Claude models support prefill. Only `claude-opus-4.6` blocks it.

### OpenAI Models

| Model | Prefill Support | Notes |
|---|---|---|
| `openai/gpt-4o-mini` | **No** | Treats assistant message as a complete prior turn; generates a fresh response |
| `openai/gpt-oss-120b` | **No** | Does not support logprobs or prefill |

### Implications for Multi-Step Strategies

- **Models with prefill support** (Claude): Can use `prefill_mode: true` in config. The model natively continues from the accumulated trajectory.
- **Models without prefill support** (OpenAI): Must use `continuation_prompt` workaround, which adds a user message after the trajectory instructing the model to continue. See [`docs/adaptive_scaling_code_prefill_problem.md`](./adaptive_scaling_code_prefill_problem.md) for details.
