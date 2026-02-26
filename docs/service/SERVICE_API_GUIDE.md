# Service API Guide: OpenAI-Compatible Interface

This document describes how the `service_app/` exposes TTS strategies through an OpenAI-compatible REST API, and explains the design decisions behind parameter passing.

## Overview

The service implements the `/v1/chat/completions` endpoint with the same schema as OpenAI's API. Any OpenAI-compatible client can call it by changing only the `base_url`. TTS-specific parameters (strategy, scorer, beam size, etc.) are passed as additional fields in the JSON request body.

```
┌──────────────────────────┐         POST /v1/chat/completions
│   OpenAI Python SDK      │ ─────────────────────────────────►  ┌──────────────────┐
│   or any HTTP client     │                                     │  service_app/    │
│                          │ ◄─────────────────────────────────  │  (FastAPI)       │
└──────────────────────────┘    OpenAI-compatible JSON response  └──────────────────┘
```

## Supported Strategies

| Strategy | Backend | Scorer | Description |
|---|---|---|---|
| `self_consistency` | OpenRouter / OpenAI API | majority voting | Generate N paths, select by answer agreement |
| `offline_bon` | Local vLLM | entropy / perplexity / sequence_prob / prm | Generate N full trajectories, pick best |
| `online_bon` | Local vLLM | same as above | Step-level best-of-N candidate selection |
| `beam_search` | Local vLLM | same as above | Beam search over reasoning steps |

## Usage with OpenAI Python SDK

### Basic Example (Self-Consistency)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="your-openrouter-key"
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Reason step by step, put answer in \\boxed{}."},
        {"role": "user", "content": "What is 15 * 7?"}
    ],
    extra_body={
        "tts_strategy": "self_consistency",
        "num_paths": 5
    }
)

print(response.choices[0].message.content)
```

### vLLM Strategy Example (Beam Search with PRM)

Requires `VLLM_MODEL_PATH` and `PRM_MODEL_PATH` environment variables.

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[...],
    extra_body={
        "tts_strategy": "beam_search",
        "tts_beam_size": 4,
        "tts_candidates_per_step": 4,
        "tts_scorer": "prm",
        "tts_score_aggregation": "mean",
        "tts_max_steps": 100,
    }
)
```

### Accessing TTS Metadata

The response includes TTS-specific metadata (consensus score, answer distribution, uncertainty scores, etc.) in an extra `tts_metadata` field on each choice. Since this field is not part of the standard OpenAI schema, the SDK's typed response object does not expose it as an attribute. To access it:

```python
raw = response.model_dump()
metadata = raw["choices"][0]["tts_metadata"]

# For self_consistency:
print(metadata["consensus_score"])       # 1.0 = all paths agree
print(metadata["uncertainty_score"])     # 0.0 = fully certain
print(metadata["answer_distribution"])   # {"105": 5}

# For vLLM strategies:
print(metadata["aggregated_score"])      # scorer's trajectory score
print(metadata["reasoning_steps"])       # number of steps generated
```

## How TTS Parameters Are Passed

### Why `extra_body`?

The OpenAI Python SDK provides three extension points for vendor-specific parameters:

| Mechanism | SDK Parameter | What It Does |
|---|---|---|
| **Request body** | `extra_body={...}` | Merges additional fields into the JSON POST body |
| **Query string** | `extra_query={...}` | Appends `?key=value` query parameters to the URL |
| **HTTP headers** | `extra_headers={...}` | Adds custom HTTP headers to the request |

All three are supported by the SDK (verified in `openai` v2.8.1). However, **`extra_body` is the correct choice** for TTS parameters because:

1. **Semantic correctness.** TTS parameters (`tts_strategy`, `tts_scorer`, `num_paths`, etc.) are part of the request payload — they describe *what* to generate, not *how* to route the request. The POST body is the standard location for request payload in REST APIs.

2. **Industry convention.** All major OpenAI-compatible services use `extra_body` for vendor extensions:
   - **vLLM**: `extra_body={"guided_json": ..., "best_of": 5}`
   - **Together AI**: `extra_body={"repetition_penalty": 1.1}`
   - **OpenRouter**: `extra_body={"transforms": ["middle-out"]}`
   - **Fireworks**: `extra_body={"context_length_exceeded_behavior": "truncate"}`

3. **Server-side simplicity.** FastAPI/Pydantic naturally validates extra body fields — we define them in the request model and they are parsed, type-checked, and documented in `/docs` automatically. Query parameters would require separate `Query(...)` declarations and would not appear in the same schema.

4. **Structured data.** Some TTS parameters may be nested or complex (e.g., future strategy configs). JSON body supports any structure; query strings are limited to flat key-value pairs.

### Why Not Query Parameters?

While `extra_query={"tts_strategy": "beam_search"}` would technically reach the server, it is the wrong abstraction:

- Query parameters on a POST endpoint are unconventional — POST body is the payload
- OpenAI's own API uses no query params on `/v1/chat/completions`
- No other OpenAI-compatible service uses query params for model/generation config
- FastAPI would need separate `Query(...)` parameter declarations, duplicating the schema

### Why Not HTTP Headers?

Headers are for transport-level metadata (auth, content-type, tracing IDs), not for application-level parameters. Encoding `tts_strategy` in a header like `X-TTS-Strategy: beam_search` would work but is semantically wrong and invisible to API documentation tools.

## Full Parameter Reference

### Standard OpenAI Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `openai/gpt-4o-mini` | Model name (OpenRouter format: `provider/model`) |
| `messages` | array | required | Chat messages (`role` + `content`) |
| `temperature` | float | 0.7 | Sampling temperature (0.0 – 2.0) |
| `max_tokens` | int | 4096 | Maximum tokens to generate |
| `stream` | bool | false | Streaming (not yet supported) |

### TTS Parameters (via `extra_body`)

| Parameter | Type | Default | Strategies | Description |
|---|---|---|---|---|
| `tts_strategy` | string | `self_consistency` | all | Strategy: `self_consistency`, `offline_bon`, `online_bon`, `beam_search` |
| `provider` | string | auto | all | API provider: `openrouter`, `openai`, `vllm`. Auto-detected from strategy if not set. |
| `num_paths` | int | 5 | self_consistency | Number of independent reasoning paths |
| `tts_scorer` | string | `entropy` | vLLM strategies | Scorer: `entropy`, `perplexity`, `sequence_prob`, `prm` |
| `tts_num_trajectories` | int | 8 | offline_bon | Number of full trajectories to generate |
| `tts_candidates_per_step` | int | 4 | online_bon, beam_search | Candidates generated per reasoning step |
| `tts_beam_size` | int | 4 | beam_search | Number of beams to maintain |
| `tts_max_steps` | int | 100 | vLLM strategies | Maximum reasoning steps |
| `tts_score_aggregation` | string | `min` | vLLM strategies | Score aggregation: `min`, `mean`, `max`, `product`, `last` |
| `tts_window_size` | int | None | vLLM strategies | Sliding window for scoring (last N steps) |

## Alternative: ChatTTS (LangChain Wrapper)

For LangChain users, we provide `ChatTTS` — a custom `BaseChatModel` that wraps the service and provides first-class access to TTS parameters and metadata. This avoids `extra_body` entirely.

```python
from llm_tts.integrations import ChatTTS

llm = ChatTTS(
    base_url="http://localhost:8001/v1",
    model="openai/gpt-4o-mini",
    tts_strategy="self_consistency",
    tts_budget=5,
)

msg = llm.invoke("What is 15 * 7?")

# Standard LangChain access
print(msg.content)

# TTS metadata in response_metadata (no model_dump() needed)
print(msg.response_metadata["tts_metadata"]["consensus_score"])
print(msg.response_metadata["tts_metadata"]["uncertainty_score"])
```

### Comparison

| Aspect | OpenAI SDK + `extra_body` | ChatTTS |
|---|---|---|
| Dependencies | `openai` (standard) | `langchain-core`, `httpx` |
| TTS params | via `extra_body={...}` | first-class constructor args |
| TTS metadata access | `response.model_dump()["choices"][0]["tts_metadata"]` | `msg.response_metadata["tts_metadata"]` |
| Works with LangChain chains | needs manual extraction | native |
| Async support | via `AsyncOpenAI` | built-in `ainvoke()` |
| Who should use it | anyone, any language | Python + LangChain users |

Both call the same `/v1/chat/completions` endpoint. Choose based on your stack.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | for self_consistency | OpenRouter API key |
| `OPENAI_API_KEY` | optional | Direct OpenAI API key |
| `VLLM_MODEL_PATH` | for vLLM strategies | Local model path (e.g., `Qwen/Qwen2.5-7B-Instruct`) |
| `PRM_MODEL_PATH` | for `tts_scorer=prm` | PRM model path (e.g., `Qwen/Qwen2.5-Math-PRM-7B`) |
| `PORT` | optional | Server port (default: 8001) |

## Running the Service

```bash
# Install
pip install -e ".[service]"

# Configure
cp service_app/.env.example .env
# Edit .env with your API keys

# Start
python service_app/main.py
# → http://localhost:8001
# → http://localhost:8001/docs (interactive API docs)
```
