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

Set `seed=42` and consistent `torch_dtype` for all experiments. These must be propagated to all components.

```yaml
# config/system/default.yaml
system:
  seed: 42                # REQUIRED for reproducibility
  torch_dtype: "bfloat16" # Options: float16, bfloat16, float32, auto
```

**Reproducibility checklist:**

| Component | Setting | Verified |
|-----------|---------|----------|
| System config | `system.seed: 42` | [ ] |
| System config | `system.torch_dtype: bfloat16` | [ ] |
| vLLM SamplingParams | `seed=config.system.seed` | [ ] |
| Strategy (DeepConf/Self-Consistency) | `seed=config.system.seed` | [ ] |

### Step 2: Configure Model and Thinking Mode

For models with thinking mode (e.g., Qwen3), set `disable_thinking_mode` based on your experiment:

```yaml
# config/model/vllm_qwen3.yaml
model:
  type: vllm
  model_path: "Qwen/Qwen3-8B"
  disable_thinking_mode: true  # Set to false for thinking mode experiments
```

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
| `system.torch_dtype` | `bfloat16` | Consistent precision |
| `model.disable_thinking_mode` | `true/false` | Depends on experiment type |
| `generation.temperature` | `0.7` / `0.6` | Non-thinking / Thinking mode |
| `generation.top_p` | `0.8` | Qwen3 recommended |
| `generation.top_k` | `20` | Qwen3 recommended |
| `strategy.temperature` | Same as generation | Consistency |
| `report_to` | `wandb` | Tracking |

---

## Inference Backends

### vLLM

High-performance inference engine optimized for throughput.

```yaml
model:
  type: vllm
  model_path: "Qwen/Qwen3-8B"
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
  enable_prefix_caching: true
  max_model_len: 32768
```

| Pros | Cons |
|------|------|
| PagedAttention prevents OOM on long sequences | No custom stopping criteria callbacks |
| Native batched generation for multiple traces | Only supports exact string stop tokens |
| Higher throughput than HuggingFace | Cannot do semantic step boundary detection |
| Prefix caching for repeated prompts | |

### HuggingFace

Standard inference with full control over generation process.

```yaml
model:
  type: huggingface
  model_path: "Qwen/Qwen3-8B"
  device_map: auto
  torch_dtype: bfloat16
```

| Pros | Cons |
|------|------|
| Full `StoppingCriteria` callback support | Lower throughput than vLLM |
| Custom step boundary detection during generation | Higher memory usage |
| Works with thinking mode semantic markers | May OOM on very long sequences |
| Fine-grained control over generation | |

### Framework Selection Guide

| Use Case | Framework | Why |
|----------|-----------|-----|
| Self-Consistency / DeepConf | vLLM | Full trace generation, high throughput |
| CoT (single trace) | vLLM | No step detection needed |
| Online Best-of-N (non-thinking) | vLLM or HF | Explicit markers work with stop tokens |
| Online Best-of-N (thinking mode) | HuggingFace | Requires `ThinkingStepStoppingCriteria` |
| Beam Search (thinking mode) | HuggingFace | Requires semantic step detection |
| Phi Decoding (thinking mode) | HuggingFace | Requires semantic step detection |

> **Note**: For thinking mode with step-by-step strategies, vLLM cannot detect semantic boundaries like "wait", "so", "let me" during generation. Use HuggingFace with `ThinkingMarkerDetector` instead.

**Why HuggingFace for step detection?**

HuggingFace's `StoppingCriteria` is more universal for step detection - it allows passing any custom detector that can inspect generated tokens and apply arbitrary logic (regex patterns, minimum character thresholds, lookbehind assertions, etc.):

```python
from llm_tts.step_boundary_detectors import ThinkingMarkerDetector
from llm_tts.generators.huggingface import ThinkingStepStoppingCriteria

# Create detector with semantic markers
detector = ThinkingMarkerDetector(min_step_chars=100)

# Pass detector to stopping criteria - inspects tokens during generation
stopping_criteria = ThinkingStepStoppingCriteria(
    tokenizer=tokenizer,
    detector=detector,  # ✅ Custom logic possible
    start_length=input_length,
    batch_size=batch_size,
)

# Generate with custom stopping
outputs = model.generate(
    input_ids,
    stopping_criteria=[stopping_criteria],  # ✅ Works
)
```

vLLM only supports exact string matching:

```python
from vllm import SamplingParams

params = SamplingParams(
    stop=["- Step"],     # ✅ Exact strings only
    stop_token_ids=[],   # ✅ Token IDs only
    # stopping_criteria=  # ❌ Not supported
)
```

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
