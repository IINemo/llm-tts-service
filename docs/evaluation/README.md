# Evaluation Protocol

This document defines the evaluation protocol for comparing test-time compute scaling strategies on mathematical reasoning tasks.

## Quick Links

- [Datasets](datasets.md) - AIME, MATH-500, SVAMP benchmark details
- [Models](models.md) - Model configurations and thinking mode settings
- [Metrics](metrics.md) - Accuracy, tokens, FLOPs calculation
- [WandB](wandb.md) - Logging conventions and upload workflow
- [Results](results/) - Experiment results by dataset

---

## Evaluation Algorithm

Follow these steps to run a reproducible evaluation:

### Step 1: Configure Reproducibility

Set `seed=42` for all experiments. The seed must be propagated to all components.

```yaml
# config/system/default.yaml
system:
  seed: 42  # REQUIRED for reproducibility
```

**Seed propagation checklist:**

| Component | Setting | Verified |
|-----------|---------|----------|
| System config | `system.seed: 42` | [ ] |
| vLLM SamplingParams | `seed=config.system.seed` | [ ] |
| Strategy (DeepConf/Self-Consistency) | `seed=config.system.seed` | [ ] |

### Step 2: Configure Model and Thinking Mode

For models with thinking mode (e.g., Qwen3), disable it for fair comparison:

```yaml
# config/model/vllm_qwen3.yaml
model:
  type: vllm
  model_path: "Qwen/Qwen3-8B"
  disable_thinking_mode: true  # REQUIRED for non-thinking evaluation
```

**Why disable thinking mode?** Thinking mode uses internal reasoning tokens that inflate compute costs. For fair comparison with other models, use non-thinking mode.

### Step 3: Configure Generation Parameters

Use model-specific recommended parameters. For Qwen3, parameters depend on thinking mode ([source](https://huggingface.co/Qwen/Qwen3-8B)):

| Mode | Temperature | top_p | top_k | Note |
|------|-------------|-------|-------|------|
| Non-thinking | 0.7 | 0.8 | 20 | DO NOT use greedy decoding |
| Thinking | 0.6 | 0.95 | 20 | For thinking mode |

```yaml
# config/generation/qwen3_nothink.yaml
generation:
  temperature: 0.7      # DO NOT use greedy (temp=0)
  top_p: 0.8
  top_k: 20
  max_new_tokens: 32768
```

**Important:** Different models require different parameters. See [models.md](models.md) for model-specific settings.

### Step 4: Configure Strategy

Set strategy-specific parameters:

```yaml
# DeepConf example
strategy:
  type: deepconf
  mode: offline
  budget: 16           # Number of traces
  filter_method: top10
  temperature: 0.7     # Must match generation.temperature

# Self-Consistency example
strategy:
  type: self_consistency
  num_paths: 16        # Number of reasoning paths
  temperature: 0.7     # Must match generation.temperature
```

### Step 5: Enable Logging

Enable WandB for experiment tracking:

```yaml
report_to: wandb
wandb_project: llm-tts-eval
run_name: "aime2025_deepconf_qwen3_n16"
```

### Step 6: Run Experiment

```bash
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_vllm_qwen3_aime2025_nothink
```

### Step 7: Verify Results

Check the output directory for:
- `results.json` - Raw results with all traces
- `run_tts_eval.log` - Execution log
- WandB dashboard - Metrics and visualizations

---

## Configuration Checklist

Before running any experiment, verify:

| Setting | Value | Why |
|---------|-------|-----|
| `system.seed` | `42` | Reproducibility |
| `model.disable_thinking_mode` | `true` | Fair comparison (for Qwen3) |
| `generation.temperature` | `0.7` | Non-greedy sampling |
| `generation.top_p` | `0.8` | Qwen3 recommended |
| `generation.top_k` | `20` | Qwen3 recommended |
| `strategy.temperature` | Same as generation | Consistency |
| `report_to` | `wandb` | Tracking |

---

## Inference Backends

### vLLM (Recommended)

Use vLLM for all local model evaluations:

```yaml
model:
  type: vllm
  model_path: "Qwen/Qwen3-8B"
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
  enable_prefix_caching: true
  max_model_len: 32768
```

**Why vLLM?**
- PagedAttention prevents OOM on long sequences
- Native batched generation for multiple traces
- Higher throughput than HuggingFace

**vLLM Limitations:**

> ⚠️ **Important**: vLLM does **not** support custom stopping criteria callbacks. It only supports:
> - `stop` - list of exact strings to stop on
> - `stop_token_ids` - list of token IDs to stop on
>
> **Example - why this is a limitation:**
>
> For **non-thinking mode** with explicit markers, stop tokens work fine:
> ```
> Model output: "- Step 1: Calculate x = 5\n- Step 2: ..."
>                                           ↑ stop=["- Step"] triggers here ✅
> ```
>
> For **thinking mode** with semantic markers, stop tokens fail:
> ```
> Model output: "The value is 5, so that means x = 10. Wait, let me reconsider..."
>                              ↑ stop=["so"] triggers here ❌
>                                (mid-sentence "so", not a step boundary!)
> ```
>
> With `StoppingCriteria`, we can use `ThinkingMarkerDetector` which:
> - Uses `min_step_chars` to merge short segments (avoids splitting on every "so")
> - Uses lookbehind patterns for some markers: `(?<=[.!?\n])\s*\bbut\b` (only after sentence end)
> - Prefers multi-word phrases like "for example", "let me" (rarely mid-sentence)
>
> vLLM can only do exact string matching - no such logic possible.
>
> This means vLLM **cannot** be used with strategies that require dynamic step boundary detection during generation (e.g., Online Best-of-N, Beam Search in thinking mode).
>
> **Impact on strategies:**
> | Strategy | vLLM Support | Notes |
> |----------|--------------|-------|
> | Self-Consistency | ✅ Full | Generates complete traces |
> | DeepConf | ✅ Full | Generates complete traces |
> | Online Best-of-N | ⚠️ Limited | Only works with explicit step markers (`"- Step N:"`), not semantic markers |
> | Beam Search | ⚠️ Limited | Same limitation as Best-of-N |
> | Thinking Mode TTS | ❌ Not supported | Requires semantic step detection during generation |
>
> **Workarounds:**
> 1. Use **HuggingFace** inference for step-by-step strategies (supports `StoppingCriteria` callbacks)
> 2. Use **post-hoc splitting** - generate full response, then split into steps with `ThinkingMarkerDetector`
>
> **References:**
> - [GitHub Issue #551](https://github.com/vllm-project/vllm/issues/551) - Feature request for custom stop functions
> - [vLLM Forums Discussion](https://discuss.vllm.ai/t/custom-function-based-stopping-criteria/1338) - Community discussion on this limitation

### API-Based Models

For OpenRouter/OpenAI:

```yaml
model:
  type: openai_api
  provider: openrouter
  model_name: "qwen/qwen3-8b"
```

---

## Strategies

| Strategy | Description | Paper |
|----------|-------------|-------|
| Self-Consistency | Multiple paths + majority voting | [Wang et al., 2022](https://arxiv.org/abs/2203.11171) |
| DeepConf | Confidence-filtered voting | [Yao et al., 2025](https://arxiv.org/abs/2508.15260) |
| Tree of Thoughts | Branching search with backtracking | [Yao et al., 2023](https://arxiv.org/abs/2305.10601) |
| CoT | Single chain-of-thought | [Wei et al., 2022](https://arxiv.org/abs/2201.11903) |
| Online Best-of-N | Step-by-step candidate selection | - |
| Beam Search | Beam search over reasoning steps | - |
| Phi Decoding | Phi-based step selection | [φ-Decoding](https://arxiv.org/abs/2503.13288) |

---

## Inference Framework Selection Guide

### By Strategy Type

| Strategy | Generation Type | Recommended Framework | Generator Class |
|----------|----------------|----------------------|-----------------|
| **Self-Consistency** | Full trace | vLLM | - (uses model directly) |
| **DeepConf** | Full trace | vLLM | - (uses model directly) |
| **CoT** | Full trace | vLLM | - (uses model directly) |
| **Tree of Thoughts** | Custom | vLLM / HuggingFace | - (custom implementation) |
| **Online Best-of-N** | Step-by-step | HuggingFace | `StepCandidateGeneratorThroughHuggingface` |
| **Beam Search** | Step-by-step | HuggingFace | `StepCandidateGeneratorThroughHuggingface` |
| **Phi Decoding** | Step-by-step | HuggingFace | `StepCandidateGeneratorThroughHuggingface` |
| **Adaptive Scaling** | Step-by-step | HuggingFace | `StepCandidateGeneratorThroughHuggingface` |

### By Mode (Thinking vs Non-Thinking)

| Mode | Step Detection | Framework | Stopping Criteria | Detector |
|------|---------------|-----------|-------------------|----------|
| **Non-thinking** (explicit markers) | `"- Step N:"` | vLLM ✅ | Stop tokens | `StructuredStepDetector` |
| **Non-thinking** (explicit markers) | `"- Step N:"` | HuggingFace ✅ | `BatchStepStoppingCriteria` | `StructuredStepDetector` |
| **Thinking mode** (semantic markers) | `"wait"`, `"so"`, `"let me"` | vLLM ❌ | Not supported | - |
| **Thinking mode** (semantic markers) | `"wait"`, `"so"`, `"let me"` | HuggingFace ✅ | `ThinkingStepStoppingCriteria` | `ThinkingMarkerDetector` |

### Complete Matrix: Strategy × Mode × Framework

| Strategy | Non-Thinking + vLLM | Non-Thinking + HF | Thinking + vLLM | Thinking + HF |
|----------|---------------------|-------------------|-----------------|---------------|
| Self-Consistency | ✅ | ✅ | ✅ (post-hoc split) | ✅ (post-hoc split) |
| DeepConf | ✅ | ✅ | ✅ (post-hoc split) | ✅ (post-hoc split) |
| CoT | ✅ | ✅ | ✅ | ✅ |
| Online Best-of-N | ✅ (stop tokens) | ✅ (`BatchStepStoppingCriteria`) | ❌ | ✅ (`ThinkingStepStoppingCriteria`) |
| Beam Search | ✅ (stop tokens) | ✅ (`BatchStepStoppingCriteria`) | ❌ | ✅ (`ThinkingStepStoppingCriteria`) |
| Phi Decoding | ✅ (stop tokens) | ✅ (`BatchStepStoppingCriteria`) | ❌ | ✅ (`ThinkingStepStoppingCriteria`) |

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BatchStepStoppingCriteria` | `llm_tts/generators/huggingface.py` | Stop at explicit step markers during HF generation |
| `ThinkingStepStoppingCriteria` | `llm_tts/generators/huggingface.py` | Stop at semantic markers during HF generation |
| `StructuredStepDetector` | `llm_tts/step_boundary_detectors/non_thinking/` | Detect `"- Step N:"` patterns |
| `ThinkingMarkerDetector` | `llm_tts/step_boundary_detectors/thinking/` | Detect semantic markers (`"wait"`, `"so"`, etc.) |
| `ThinkingStepEarlyStopping` | `llm_tts/early_stopping.py` | For API streaming (OpenAI-compatible) |

---

## Running Experiments

### Example Commands

```bash
# DeepConf on AIME 2025
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_vllm_qwen3_aime2025_nothink

# Self-Consistency on AIME 2025
python scripts/run_tts_eval.py \
    --config-name=experiments/self_consistency/sc_vllm_qwen3_aime2025

# Split dataset across GPUs (for parallel runs)
# GPU 0: first 15 samples
CUDA_VISIBLE_DEVICES=0 python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_vllm_qwen3_aime2025_nothink \
    dataset.subset=15 dataset.offset=0

# GPU 1: next 15 samples
CUDA_VISIBLE_DEVICES=1 python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_vllm_qwen3_aime2025_nothink \
    dataset.subset=15 dataset.offset=15
```

### Resume Interrupted Runs

```bash
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_vllm_qwen3_aime2025_nothink \
    --resume outputs/2025-12-04/aime2025_deepconf_part1
```
