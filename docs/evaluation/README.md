# Evaluation Protocol

This document defines the evaluation protocol for comparing test-time compute scaling strategies on mathematical reasoning tasks.

## Table of Contents

- [Datasets](datasets.md) - AIME, MATH-500, SVAMP benchmark details
- [Models](models.md) - Paper-model matrix for strategy implementations
- [Metrics](metrics.md) - Accuracy, tokens, FLOPs calculation
- [WandB](wandb.md) - Logging conventions and upload workflow
- [Results](results/) - Experiment results by dataset
- [Reproducibility](#reproducibility) - Seed and determinism settings
- [Inference Backends](#inference-backends) - vLLM vs HuggingFace comparison
- [Model Configuration](#model-configuration) - Config files for experiments
- [Strategies](#strategies) - Test-time compute scaling methods

---

## Reproducibility

For reproducible experiments, always use `seed=42` in the system configuration. The seed is propagated to all components:

### Seed Propagation

| Component | How Seed is Used |
|-----------|------------------|
| Python `random` | `random.seed(seed)` |
| NumPy | `np.random.seed(seed)` |
| PyTorch | `torch.manual_seed(seed)` + CUDA seeds |
| vLLM SamplingParams | `seed=seed` parameter |
| Strategy (DeepConf) | Passed to `SamplingParams` for trace generation |
| Strategy (Self-Consistency) | Passed to `SamplingParams` for path generation |

### Configuration

```yaml
# config/system/default.yaml
system:
  seed: 42  # REQUIRED: Always use seed=42 for reproducibility
  device: cuda
  hf_cache: ~/.cache/huggingface
```

### Verification Checklist

Before running experiments, verify that:

1. **System seed is set**: `config.system.seed = 42`
2. **vLLM uses seed**: Check that `SamplingParams(seed=...)` is passed
3. **Strategy receives seed**: DeepConf and Self-Consistency strategies should have `seed` parameter

### Why Seed Matters

Without a fixed seed, strategies that use sampling (temperature > 0) will produce different results across runs. This makes it impossible to:
- Compare strategy performance fairly
- Debug issues in generation
- Reproduce reported results

> **IMPORTANT**: All experiments in the results tables were run with `seed=42`. If you need to run multiple trials, use seeds `[42, 43, 44, ...]` and report mean/std.

---

## Inference Backends

### vLLM (Recommended for Local Models)

**For all local model evaluations, use vLLM as the inference backend.** vLLM provides significant advantages over standard HuggingFace inference:

| Feature | vLLM | HuggingFace |
|---------|------|-------------|
| Memory Management | PagedAttention (efficient) | Standard (OOM-prone) |
| Long Sequences | Handles 32K+ tokens | Often OOM on long reasoning |
| Batched Generation | Native support | Limited |
| Throughput | High | Moderate |

#### Installation

```bash
pip install "llm-tts-service[vllm]"
```

#### vLLM Configuration

```yaml
# config/model/vllm_qwen3.yaml
model:
  type: "vllm"
  model_path: "Qwen/Qwen3-8B"
  device: cuda
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
  enable_prefix_caching: true
  trust_remote_code: true
  max_model_len: 32768
```

> **Why vLLM?** Test-time scaling strategies like DeepConf generate multiple long reasoning traces. vLLM's PagedAttention prevents out-of-memory errors on extended sequences that would crash standard HuggingFace inference.

### API-Based Models

For API-based models (OpenAI, OpenRouter, DeepSeek), use the standard `openai_api` model type. See [config/model/openrouter.yaml](../../config/model/openrouter.yaml) for examples.

---

## Model Configuration

### Configuration Files

- **vLLM config (recommended)**: [config/model/vllm_qwen3.yaml](../../config/model/vllm_qwen3.yaml)
- HuggingFace config: [config/model/hf_qwen3.yaml](../../config/model/hf_qwen3.yaml)
- Generation config: [config/generation/default.yaml](../../config/generation/default.yaml)
- Prompt template: [config/prompts/default.txt](../../config/prompts/default.txt)

> **IMPORTANT**: Do not modify model or generation configurations between experiments to ensure fair comparison. Always use vLLM for local model evaluations to avoid OOM issues on long reasoning sequences.

---

## Strategies

| Strategy | Description | Paper |
|----------|-------------|-------|
| CoT | Chain-of-thought prompting for step-by-step reasoning | [Wei et al., 2022](https://arxiv.org/abs/2201.11903) |
| Self-Consistency | Generates multiple reasoning paths and selects answer via majority voting | [Wang et al., 2022](https://arxiv.org/abs/2203.11171) |
| MUR | Momentum Uncertainty-guided Reasoning with adaptive scaling | [Hao et al., 2025](https://arxiv.org/abs/2507.14958) |
| DeepConf | Generates multiple traces with confidence scoring, filters by confidence, then votes. See [DeepConf.md](../deepconf/DeepConf.md) | [Yao et al., 2025](https://arxiv.org/abs/2508.15260) |
| Tree of Thoughts | Explores multiple reasoning branches with backtracking | [Yao et al., 2023](https://arxiv.org/abs/2305.10601) |
| Graph of Thoughts | Models reasoning as a graph structure with flexible exploration | [Besta et al., 2023](https://arxiv.org/abs/2308.09687) |
| ϕ-Decoding | Uncertainty-aware decoding with phi-based scoring | [Chen et al., 2025](https://arxiv.org/abs/2503.13288) |

---

## Running Experiments

### Local Models (vLLM)

```bash
# DeepConf on AIME 2025 with vLLM (recommended)
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_vllm_qwen3_aime2025

# Self-Consistency on AIME 2025 with vLLM
python scripts/run_tts_eval.py \
    --config-name=experiments/self_consistency/sc_vllm_qwen3_aime2025
```

### API-Based Models

```bash
# DeepConf via OpenRouter API
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_api_test

# Self-Consistency via API
python scripts/run_tts_eval.py \
    --config-name=experiments/self_consistency/self_consistency_api_test
```

> **Note**: For local model evaluations, always prefer vLLM configs (`*_vllm_*`) over HuggingFace configs (`*_hf_*`) to avoid OOM errors on long reasoning sequences.

