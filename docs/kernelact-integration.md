# KernelAct + LLM-TTS Service Integration

## Overview

This integration enables [KernelAct](https://github.com/ai-nikolai/KernelAct) (a CUDA kernel optimization benchmark) to use **llm-tts-service** as a remote backend for test-time scaling strategies. Instead of running plain vLLM inference locally, KernelAct can call the TTS service via an OpenAI-compatible API to apply uncertainty-guided reasoning strategies (Offline Best-of-N, Online Best-of-N, Beam Search) when generating CUDA kernel optimizations.

## Architecture

```
KernelAct (client)                          llm-tts-service (server)
┌─────────────────────┐                    ┌──────────────────────────────┐
│ run_inference_       │   OpenAI SDK      │  FastAPI service (port 8001) │
│ test_time_scaling.py │ ──────────────>   │  /v1/chat/completions        │
│                      │   POST request    │          │                   │
│ - Loads dataset      │   with extra_body │          v                   │
│ - Formats prompts    │                   │  StrategyManager             │
│ - Calls TTS service  │                   │    ├── vLLM backend (GPU)    │
│ - Collects results   │   <───────────    │    ├── Uncertainty scorer    │
│ - Evaluates kernels  │   response with   │    └── TTS strategies:       │
└─────────────────────┘   generated code   │        ├── Offline BoN       │
                                           │        ├── Online BoN        │
                                           │        └── Beam Search       │
                                           └──────────────────────────────┘
```

## What Changed

### llm-tts-service side (`feat/kernelact-vllm-service-integration` branch)

**4 files modified in `service_app/`:**

1. **`api/models/openai_compat.py`** — Extended `ChatCompletionRequest` with new fields for vLLM TTS strategies:
   - `tts_scorer` — uncertainty scorer type (`entropy`, `perplexity`, `sequence_prob`)
   - `tts_num_trajectories` — number of trajectories for Offline BoN
   - `tts_candidates_per_step` — candidates per step for Online BoN / Beam Search
   - `tts_beam_size` — beam width for Beam Search
   - `tts_max_steps` — maximum reasoning steps
   - `tts_score_aggregation` — how to aggregate step scores (`min`, `mean`, `max`, `product`, `last`)
   - `provider` field now accepts `"vllm"` in addition to `"openrouter"` and `"openai"`

2. **`api/routes/chat.py`** — Updated the `/v1/chat/completions` endpoint:
   - Routes vLLM strategies (`offline_bon`, `online_bon`, `beam_search`) to `generate_trajectories_batch()` instead of `generate_trajectory()`
   - Passes all TTS parameters from the request to the strategy config
   - Enriches response metadata with strategy-specific info (reasoning steps, validity scores, aggregated score)

3. **`core/config.py`** — Added `Settings` fields for vLLM backend configuration:
   - `vllm_model_path` — HuggingFace model ID (e.g., `"Qwen/Qwen2.5-Coder-7B-Instruct"`)
   - `vllm_max_model_len`, `vllm_gpu_memory_utilization`, `vllm_tensor_parallel_size` — vLLM engine parameters
   - `default_scorer`, `default_temperature`, `default_thinking_mode` — default TTS behavior

4. **`core/strategy_manager.py`** — Added vLLM backend management:
   - `_init_vllm_backend()` — lazily initializes vLLM model, wraps it with `VLLMWithUncertainty` (lm-polygraph), creates `VLLMStepGenerator` with `ThinkingMarkerDetector`, and a `StepScorerConfidence` scorer. All components are cached after first init.
   - `_create_vllm_strategy()` — factory for `StrategyOfflineBestOfN`, `StrategyOnlineBestOfN`, and `StrategyBeamSearch`, parameterized from the request config.
   - `clear_cache()` — now also cleans up vLLM resources.

### KernelAct side (`feat/tts-service-integration` branch)

**1 file modified: `kernelact/run_inference_test_time_scaling.py`**

The inference script was extended with a dual-path design:

- **Local vLLM path** (original): Loads model locally, generates with `SamplingParams`, processes `outputs` directly. Used when `--tts_service_url` is not provided.
- **TTS service path** (new): Skips local model loading entirely. Uses the OpenAI Python SDK to call the remote TTS service, passing strategy parameters via `extra_body`. Only requires network access to the service.

New CLI arguments:
```
--tts_service_url    URL of the TTS service (e.g., http://localhost:8001/v1)
--tts_strategy       offline_bon | online_bon | beam_search
--tts_num_trajectories  Number of trajectories for Offline BoN (default: 8)
--tts_scorer         entropy | perplexity | sequence_prob
--tts_candidates_per_step  Candidates per step (default: 4)
--tts_beam_size      Beam size (default: 4)
--tts_max_steps      Max reasoning steps (default: 100)
--tts_score_aggregation  min | mean | max | product | last
```

## Usage

### 1. Start the TTS service

```bash
# Set environment variables
export VLLM_MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
export VLLM_MAX_MODEL_LEN=32000
export VLLM_GPU_MEMORY_UTILIZATION=0.9

# Start the service
cd llm-tts-service
python service_app/main.py
# Service starts on http://localhost:8001
```

### 2. Run KernelAct with TTS

```bash
cd KernelAct

# Offline Best-of-N with entropy scoring
python kernelact/run_inference_test_time_scaling.py \
    --model_name "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --tts_service_url http://localhost:8001/v1 \
    --tts_strategy offline_bon \
    --tts_num_trajectories 8 \
    --tts_scorer entropy \
    --experiment_name "kernelact_offline_bon_entropy" \
    --level 1

# Beam Search with perplexity scoring
python kernelact/run_inference_test_time_scaling.py \
    --model_name "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --tts_service_url http://localhost:8001/v1 \
    --tts_strategy beam_search \
    --tts_beam_size 4 \
    --tts_candidates_per_step 4 \
    --tts_scorer perplexity \
    --experiment_name "kernelact_beam_search_perp" \
    --level 1
```

### 3. Run KernelAct without TTS (original local mode)

```bash
# Original behavior — local vLLM, no TTS strategies
python kernelact/run_inference_test_time_scaling.py \
    --model_name "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --experiment_name "kernelact_baseline" \
    --level 1
```

## Available Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| `offline_bon` | Generate N full trajectories, score each, pick the best | `tts_num_trajectories`, `tts_scorer` |
| `online_bon` | Step-by-step generation with N candidates per step, pick the best at each step | `tts_candidates_per_step`, `tts_max_steps`, `tts_scorer` |
| `beam_search` | Maintain K beams, expand each with N candidates per step, prune to top-K | `tts_beam_size`, `tts_candidates_per_step`, `tts_max_steps`, `tts_scorer` |

## Available Scorers

| Scorer | Description |
|--------|-------------|
| `entropy` | Mean token entropy — lower entropy = higher confidence |
| `perplexity` | Token-level perplexity — lower perplexity = higher confidence |
| `sequence_prob` | Sequence probability — higher probability = higher confidence |

All scorers are computed via the [lm-polygraph](https://github.com/IINemo/lm-polygraph) library's `VLLMWithUncertainty` wrapper, which extracts logprobs during vLLM generation and computes uncertainty estimates without additional model calls.

## API Request Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    messages=[
        {"role": "user", "content": "Optimize this CUDA kernel: ..."}
    ],
    temperature=0.6,
    max_tokens=16000,
    extra_body={
        "tts_strategy": "online_bon",
        "tts_candidates_per_step": 4,
        "tts_max_steps": 100,
        "tts_scorer": "entropy",
    },
)

print(response.choices[0].message.content)
```

## Branches

| Repository | Branch | Description |
|------------|--------|-------------|
| `llm-tts-service` | `feat/kernelact-vllm-service-integration` | Service-side: vLLM backend + strategy routing |
| `KernelAct` | `feat/tts-service-integration` | Client-side: OpenAI SDK client + CLI args |
