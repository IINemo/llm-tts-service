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

Use model-specific recommended parameters. For Qwen3 non-thinking mode:

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
