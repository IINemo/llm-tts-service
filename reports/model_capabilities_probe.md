# Model Capabilities Probe Results

**Date:** 2026-03-03
**Script:** `scripts/probe_model_capabilities.py`

## Summary

- **Logprobs:** OpenAI models (gpt-4o, gpt-4.1-*) return up to 20 logprobs. Qwen3.5-27b returns up to 5. All other models via OpenRouter silently accept `top_logprobs` but return 0.
- **Prefill:** Supported by Claude, Qwen3-30b, Gemini. Not supported by OpenAI, DeepSeek, Qwen3.5-27b, Qwen3-235b, Llama 4.
- **Reasoning models** (o3-mini, o4-mini) don't support `max_tokens` parameter and are unreachable with current probe.

## Results

| Provider | Model | Logprobs | Accepted | Returned | Prefill |
|---|---|---|---|---|---|
| openai | gpt-4o-mini | YES | 20 | 20 | NO |
| openai | gpt-4o | YES | 20 | 20 | NO |
| openai | gpt-4.1-mini | YES | 20 | 20 | NO |
| openai | gpt-4.1-nano | YES | 20 | 20 | NO |
| openai | o4-mini | -- | -- | -- | -- |
| openai | o3-mini | -- | -- | -- | -- |
| openrouter | openai/gpt-4o-mini | YES | 20 | 20 | NO |
| openrouter | openai/gpt-4o | YES | 20 | 20 | NO |
| openrouter | openai/o4-mini | -- | -- | -- | -- |
| openrouter | anthropic/claude-sonnet-4 | silent NO | 20 | 0 | YES |
| openrouter | anthropic/claude-opus-4 | silent NO | 20 | 0 | YES |
| openrouter | anthropic/claude-3.5-sonnet | silent NO | 20 | 0 | YES |
| openrouter | anthropic/claude-3.5-haiku | silent NO | 20 | 0 | YES |
| openrouter | deepseek/deepseek-r1 | silent NO | 20 | 0 | NO |
| openrouter | deepseek/deepseek-chat-v3-0324 | silent NO | 20 | 0 | NO |
| openrouter | qwen/qwen3.5-27b | YES | 5 | 5 | NO |
| openrouter | qwen/qwen3-235b-a22b | silent NO | 20 | 0 | NO |
| openrouter | qwen/qwen3-30b-a3b | silent NO | 20 | 0 | YES |
| openrouter | google/gemini-2.5-flash | silent NO | 20 | 0 | YES |
| openrouter | google/gemini-2.5-pro | silent NO | 20 | 0 | YES |
| openrouter | meta-llama/llama-4-maverick | silent NO | 20 | 0 | NO |

## Key Takeaways

1. **Logprob-based scorers** (entropy, perplexity, sequence_prob) work with OpenAI models and Qwen3.5-27b (up to 5).
2. **OpenRouter silently ignores** `logprobs=True` for most non-OpenAI models — no error, just empty data.
3. **Prefill-dependent strategies** (online best-of-n, beam search, adaptive) work with Claude, Gemini, Qwen3-30b — but not with OpenAI, DeepSeek, Qwen3.5-27b, Qwen3-235b, or Llama 4.
4. **No model supports both** logprobs and prefill via these providers (Qwen3.5-27b has logprobs but no prefill).
