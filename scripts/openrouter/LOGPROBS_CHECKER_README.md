# OpenRouter Logprobs Checker

A Python script to empirically test which models on OpenRouter support logprobs.

## Installation

No special installation needed. Just requires Python 3 and requests:

```bash
pip install requests
```

## Setup

1. Set your OpenRouter API key in a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Or set it as an environment variable:

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Usage

### Test default models (recommended first run)

```bash
python scripts/check_openrouter_logprobs.py
```

This tests a curated list of popular models.

### Test specific models

```bash
python scripts/check_openrouter_logprobs.py \
  --models openai/gpt-4o deepseek/deepseek-r1 anthropic/claude-opus-4.6
```

### Test all models from a provider

```bash
python scripts/check_openrouter_logprobs.py --category openai
python scripts/check_openrouter_logprobs.py --category anthropic
python scripts/check_openrouter_logprobs.py --category meta
```

### Test ALL models on OpenRouter (slow!)

```bash
python scripts/check_openrouter_logprobs.py --all
```

### Get JSON output

```bash
python scripts/check_openrouter_logprobs.py --json > results.json
```

## Example Output

```
Testing 15 models for logprobs support...

[1/15] Testing: openai/gpt-4o ... ✅ YES (provider: OpenAI)
[2/15] Testing: openai/gpt-4o-mini ... ✅ YES (provider: OpenAI)
...
[6/15] Testing: openai/gpt-oss-120b ... ❌ NO (provider: BaseTen)

================================================================================
RESULTS: 15 models tested
================================================================================

✅ MODELS SUPPORTING LOGPROBS (5):
--------------------------------------------------------------------------------
  • openai/gpt-4o
    Provider: OpenAI
    Response: Hello! How can I assist you today?...

  • openai/gpt-4o-mini
    Provider: OpenAI
...

================================================================================
SUMMARY: 5 / 15 models support logprobs
================================================================================
```

## Current Findings

As of 2026-02-07, only **OpenAI GPT-4o family models** support logprobs on OpenRouter:

- ✅ `openai/gpt-4o`
- ✅ `openai/gpt-4o-mini`
- ✅ `openai/gpt-4-turbo`
- ✅ `openai/gpt-4o-2024-05-13`
- ✅ `openai/chatgpt-4o-latest`

**All other providers don't support logprobs:**
- ❌ All Anthropic/Claude models
- ❌ All Google/Gemini models
- ❌ All Meta/Llama models
- ❌ All Mistral models
- ❌ All Qwen models
- ❌ All DeepSeek models
- ❌ All Nous Research models
- ❌ All Cohere models

This is based on empirical testing of 50+ models. The situation may change over time, so use this script to verify.

## How It Works

The script sends a minimal API request with `logprobs: true` to each model and checks:
1. If the request succeeds
2. If the response contains a `logprobs` field with actual data (not `null` or `{}`)

## License

Same as parent project.
