# LLM Test-Time Scaling Service

OpenAI-compatible REST API for Test-Time Scaling strategies.

## Overview

This service exposes TTS strategies through an **OpenAI-compatible API**. You can use it as a drop-in replacement for OpenAI's API, leveraging advanced test-time scaling with your existing OpenAI SDK code.

## Supported Strategies

| Strategy | Backend | Description |
|---|---|---|
| `self_consistency` | OpenRouter / OpenAI API | Majority voting over multiple reasoning paths |
| `offline_bon` | Local vLLM | Offline Best-of-N: generate N trajectories, pick best |
| `online_bon` | Local vLLM | Online Best-of-N: step-level candidate selection |
| `beam_search` | Local vLLM | Beam search over reasoning steps |

## Quick Start

### Automated Setup

```bash
# From repository root
./setup.sh          # Install dependencies
./start_service_app.sh  # Start the service
```

### Local Development

```bash
pip install -e ".[service]"
export OPENROUTER_API_KEY="your-key"
python service_app/main.py
```

The service starts on `http://localhost:8001`. Open http://localhost:8001/docs for interactive API documentation.

## Usage

### Self-Consistency (API models)

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
print(response.choices[0].tts_metadata)  # consensus_score, answer_distribution
```

### vLLM Strategies (local model)

Requires `VLLM_MODEL_PATH` to be set.

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[...],
    extra_body={
        "tts_strategy": "offline_bon",
        "tts_num_trajectories": 8,
        "tts_scorer": "entropy",        # entropy, perplexity, sequence_prob, prm
        "tts_score_aggregation": "min",  # min, mean, max, product, last
    }
)
```

### With cURL

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "Reason step by step, put answer in \\boxed{}."},
      {"role": "user", "content": "What is 15 * 7?"}
    ],
    "tts_strategy": "self_consistency",
    "num_paths": 5
  }'
```

## API Reference

### POST /v1/chat/completions

**Standard OpenAI Parameters:**
- `model` (string): Model to use (e.g., `openai/gpt-4o-mini`)
- `messages` (array): Chat messages
- `temperature` (float): Sampling temperature (0-2, default: 0.7)
- `max_tokens` (int): Maximum tokens (default: 4096)

**TTS Parameters (via `extra_body`):**
- `tts_strategy` (string): `self_consistency`, `offline_bon`, `online_bon`, `beam_search`
- `provider` (string): `openrouter`, `openai`, or `vllm` (auto-detected from strategy)
- `num_paths` (int): Reasoning paths for self_consistency (default: 5)
- `tts_scorer` (string): `entropy`, `perplexity`, `sequence_prob`, `prm` (vLLM strategies)
- `tts_num_trajectories` (int): Trajectories for offline_bon (default: 8)
- `tts_candidates_per_step` (int): Candidates per step for online_bon/beam_search (default: 4)
- `tts_beam_size` (int): Beam size for beam_search (default: 4)
- `tts_max_steps` (int): Max reasoning steps (default: 100)
- `tts_score_aggregation` (string): `min`, `mean`, `max`, `product`, `last`
- `tts_window_size` (int): Scoring window size (optional)

### GET /v1/models

List available models.

### GET /health

Health check endpoint.

## Configuration

### Environment Variables

See `service_app/.env.example` for all options:

```bash
# API Keys
OPENROUTER_API_KEY=your-key    # Required for self_consistency
OPENAI_API_KEY=your-key        # Optional

# Server
PORT=8001
HOST=0.0.0.0

# vLLM Backend (for offline_bon, online_bon, beam_search)
VLLM_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
VLLM_MAX_MODEL_LEN=32000
VLLM_GPU_MEMORY_UTILIZATION=0.5

# PRM Scorer (optional)
PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
PRM_USE_VLLM=true
PRM_GPU_MEMORY_UTILIZATION=0.3
```

## Project Structure

```
service_app/
├── api/
│   ├── routes/
│   │   ├── chat.py            # /v1/chat/completions
│   │   └── models.py          # /v1/models
│   └── models/
│       └── openai_compat.py   # OpenAI-compatible Pydantic schemas
├── core/
│   ├── config.py              # Configuration (pydantic-settings)
│   ├── strategy_manager.py    # Strategy creation and lifecycle
│   ├── prm_scorer_factory.py  # PRM scorer lazy initialization
│   └── logging_config.py      # Logging setup
├── main.py                    # FastAPI app entry point
├── Dockerfile
└── .env.example
```
