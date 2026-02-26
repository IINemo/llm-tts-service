# Service API Guide: OpenAI-Compatible Interface

This document describes how the `service_app/` exposes TTS strategies through an OpenAI-compatible REST API, and explains the design decisions behind parameter passing.

## Overview

The service implements the `/v1/chat/completions` endpoint with the same schema as OpenAI's API. Any OpenAI-compatible client can call it by changing only the `base_url`.

TTS-specific parameters (strategy, scorer, beam size, etc.) can be specified in two ways:

1. **URL path** — encode strategy and scorer directly in `base_url`
2. **Request body** — pass via `extra_body` (standard OpenAI SDK pattern)

```
┌──────────────────────────┐
│   OpenAI Python SDK      │
│   or any HTTP client     │
└────────────┬─────────────┘
             │
             │  POST /v1/{strategy}/{scorer}/chat/completions
             │       or /v1/{strategy}/chat/completions
             │       or /v1/chat/completions
             ▼
┌──────────────────────────┐
│     service_app/         │
│     (FastAPI)            │
└──────────────────────────┘
```

## Supported Strategies

| Strategy | Backend | Scorer | Description |
|---|---|---|---|
| `self_consistency` | OpenRouter / OpenAI API | majority voting | Generate N paths, select by answer agreement |
| `offline_bon` | Local vLLM | entropy / perplexity / sequence_prob / prm | Generate N full trajectories, pick best |
| `online_bon` | Local vLLM | same as above | Step-level best-of-N candidate selection |
| `beam_search` | Local vLLM | same as above | Beam search over reasoning steps |

## Specifying Strategy and Scorer

### Option A: URL Path (Recommended)

The OpenAI SDK concatenates `base_url` with the hardcoded endpoint path `/chat/completions`. We exploit this by accepting strategy and scorer as **URL path segments**:

```
base_url                                                → SDK sends POST to
──────────────────────────────────────────────────────────────────────────────────────────
https://thinkbooster.ai/v1                              → /v1/chat/completions
https://thinkbooster.ai/v1/self_consistency              → /v1/self_consistency/chat/completions
https://thinkbooster.ai/v1/beam_search/prm               → /v1/beam_search/prm/chat/completions
```

> **Note:** Replace `thinkbooster.ai` with your actual deployment URL (e.g. `localhost:8001` for local development).

This is the cleanest approach — no `extra_body` needed for the two most common settings:

```python
from openai import OpenAI

# Strategy in the URL, budget in extra_body
client = OpenAI(
    base_url="https://thinkbooster.ai/v1/self_consistency",
    api_key="your-openrouter-key"
)

response = client.chat.completions.create(
    model="deepseek/deepseek-r1",
    messages=[
        {"role": "system", "content": "Think step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": (
            "Each of the 2001 students at a high school studies either Spanish or French, "
            "and some study both. The number who study Spanish is between 80 percent and "
            "85 percent of the school population, and the number who study French is between "
            "30 percent and 40 percent. Let m be the smallest number of students who could "
            "study both languages, and let M be the largest. Find M - m."
        )}
    ],
    extra_body={"num_paths": 8, "max_tokens": 4096}
)

print(response.choices[0].message.content)
```

With strategy **and** scorer:

```python
# Beam search with PRM scorer — both encoded in the URL
client = OpenAI(
    base_url="https://thinkbooster.ai/v1/beam_search/prm",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[
        {"role": "system", "content": "Think step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": (
            "Find the number of ordered pairs (x, y) of positive integers that satisfy "
            "x + 2y = 2xy."
        )}
    ],
    # Fine-grained params go through extra_body
    extra_body={
        "max_tokens": 8192,
        "tts_beam_size": 4,
        "tts_candidates_per_step": 4,
        "tts_max_steps": 50,
    }
)
```

Valid URL path values:

| Segment | Position | Valid values |
|---|---|---|
| `{strategy}` | 1st | `self_consistency`, `offline_bon`, `online_bon`, `beam_search` |
| `{scorer}` | 2nd (optional) | `entropy`, `perplexity`, `sequence_prob`, `prm` |

Invalid values return a `400 Bad Request` with an error message listing valid options.

**How it works internally:** The OpenAI SDK stores `base_url`, appends a trailing slash, and concatenates the endpoint path `/chat/completions` (stripping its leading slash). This is a documented behavior in the SDK's `_prepare_url` method (`openai/_base_client.py`). The server registers three FastAPI routes that all point to the same handler:

```
POST /v1/{strategy}/{scorer}/chat/completions
POST /v1/{strategy}/chat/completions
POST /v1/chat/completions
```

URL path segments take priority over body parameters when both are present.

### Option B: Request Body (`extra_body`)

The standard OpenAI SDK pattern for vendor extensions. All TTS parameters go into `extra_body`:

```python
client = OpenAI(
    base_url="https://thinkbooster.ai/v1",
    api_key="your-openrouter-key"
)

response = client.chat.completions.create(
    model="deepseek/deepseek-r1",
    messages=[
        {"role": "system", "content": "Think step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": "How many positive integers less than 1000 are divisible by 3 but not by 7?"}
    ],
    extra_body={
        "tts_strategy": "self_consistency",
        "num_paths": 16,
        "max_tokens": 4096,
    }
)
```

This is how all major OpenAI-compatible services pass vendor-specific parameters:
- **vLLM**: `extra_body={"guided_json": ..., "best_of": 5}`
- **Together AI**: `extra_body={"repetition_penalty": 1.1}`
- **OpenRouter**: `extra_body={"transforms": ["middle-out"]}`

### Combining Both

URL path and body parameters can be combined. URL path takes priority for strategy and scorer; fine-grained parameters always come from the body:

```python
# Strategy + scorer from URL, budget and aggregation from body
client = OpenAI(base_url="https://thinkbooster.ai/v1/offline_bon/prm", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[
        {"role": "system", "content": "Think step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": (
            "Let S be the set of integers from 1 to 2^40 that are divisible by 2^20. "
            "How many perfect squares does S contain?"
        )}
    ],
    extra_body={
        "max_tokens": 8192,
        "tts_num_trajectories": 16,
        "tts_score_aggregation": "mean",
    }
)
```

### Why Not Query Parameters or Headers?

The OpenAI SDK also supports `extra_query` and `extra_headers`, but these are not appropriate for TTS parameters:

- **`extra_query`** — query parameters on a POST endpoint are unconventional. OpenAI's own API uses no query params on `/v1/chat/completions`, and no other OpenAI-compatible service does either.
- **`extra_headers`** — headers are for transport-level metadata (auth, content-type, tracing IDs), not application-level parameters. Invisible to API documentation tools.

## Accessing TTS Metadata

The response includes TTS-specific metadata in an extra `tts_metadata` field on each choice. Since this field is not part of the standard OpenAI schema, the SDK's typed response object does not expose it as an attribute. To access it:

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

## Full Parameter Reference

### Standard OpenAI Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | — | Model name (OpenRouter format: `provider/model`, e.g. `deepseek/deepseek-r1`) |
| `messages` | array | required | Chat messages (`role` + `content`) |
| `temperature` | float | 0.7 | Sampling temperature (0.0 – 2.0) |
| `max_tokens` | int | 4096 | Maximum tokens to generate |
| `stream` | bool | false | Streaming (not yet supported) |

### TTS Parameters

These can be set via URL path (strategy, scorer) or `extra_body` (all params):

| Parameter | Type | Default | Strategies | Description |
|---|---|---|---|---|
| `tts_strategy` | string | `self_consistency` | all | Strategy (or use URL path) |
| `tts_scorer` | string | `entropy` | vLLM strategies | Scorer (or use URL path) |
| `provider` | string | auto | all | API provider: `openrouter`, `openai`, `vllm` |
| `num_paths` | int | 5 | self_consistency | Number of independent reasoning paths |
| `tts_num_trajectories` | int | 8 | offline_bon | Number of full trajectories to generate |
| `tts_candidates_per_step` | int | 4 | online_bon, beam_search | Candidates generated per reasoning step |
| `tts_beam_size` | int | 4 | beam_search | Number of beams to maintain |
| `tts_max_steps` | int | 100 | vLLM strategies | Maximum reasoning steps |
| `tts_score_aggregation` | string | `min` | vLLM strategies | Score aggregation: `min`, `mean`, `max`, `product`, `last` |
| `tts_window_size` | int | None | vLLM strategies | Sliding window for scoring (last N steps) |

## Alternative: ChatTTS (LangChain Wrapper)

For LangChain users, we provide `ChatTTS` — a custom `BaseChatModel` that wraps the service and provides first-class access to TTS parameters and metadata:

```python
from llm_tts.integrations import ChatTTS

llm = ChatTTS(
    base_url="https://thinkbooster.ai/v1",
    model="deepseek/deepseek-r1",
    tts_strategy="self_consistency",
    tts_budget=8,
)

msg = llm.invoke(
    "A function f satisfies f(x) + f(1 - 1/x) = arctan(x) for all x != 0. Find f(2)."
)

# Standard LangChain access
print(msg.content)

# TTS metadata in response_metadata (no model_dump() needed)
print(msg.response_metadata["tts_metadata"]["consensus_score"])
print(msg.response_metadata["tts_metadata"]["uncertainty_score"])
```

### Interface Comparison

| Aspect | URL Path | `extra_body` | ChatTTS |
|---|---|---|---|
| Dependencies | `openai` | `openai` | `langchain-core`, `httpx` |
| Strategy/scorer | in `base_url` | in `extra_body` | constructor args |
| Fine-grained params | `extra_body` | `extra_body` | constructor args |
| TTS metadata | `response.model_dump()` | `response.model_dump()` | `msg.response_metadata` |
| LangChain chains | manual | manual | native |
| Best for | clean config | full control | LangChain users |

All three approaches call the same endpoint. Choose based on your stack and use case.

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
