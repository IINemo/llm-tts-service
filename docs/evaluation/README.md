# Evaluation Protocol

This document defines the evaluation protocol for comparing test-time compute scaling strategies on mathematical reasoning tasks.

## Table of Contents

- [Datasets](datasets.md)
- [Models](models.md)
- [Metrics](metrics.md)
- [Results](results/) - Experiment results by dataset
- [Model Configuration](#model-configuration)
- [Strategies](#strategies)

---

## Model Configuration

### Base Model

- **Model**: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Mode**: Non-thinking mode (standard generation)
- **Precision**: float16

### Configuration Files

- Model config: [config/model/hf_qwen3.yaml](../../config/model/hf_qwen3.yaml)
- Generation config: [config/generation/default.yaml](../../config/generation/default.yaml)
- Prompt template: [config/prompts/default.txt](../../config/prompts/default.txt)

> **IMPORTANT**: Do not modify model or generation configurations between experiments to ensure fair comparison.

---

## Strategies

### 1. Self-Consistency (Baseline)

Generates multiple reasoning paths and selects answer via majority voting.

| Parameter | Value |
|-----------|-------|
| `num_paths` | 16 |
| `temperature` | 0.7 |
| `selection` | Majority voting |

**Config**: [`config/experiments/self_consistency/`](../../config/experiments/self_consistency/)

### 2. DeepConf (Offline)

Generates multiple traces with confidence scoring, filters by confidence, then votes.

| Parameter | Value |
|-----------|-------|
| `budget` | 16 |
| `filter_method` | top10 |
| `window_size` | 2048 |
| `temperature` | 0.7 |

**Config**: [`config/experiments/deepconf/`](../../config/experiments/deepconf/)

### 3. Best-of-N

Generates N completions and selects based on a verifier/reward model.

**Config**: [`config/experiments/best_of_n/`](../../config/experiments/best_of_n/)

### 4. Beam Search

Step-level beam search with scoring at each reasoning step.

**Config**: [`config/experiments/beam_search/`](../../config/experiments/beam_search/)

### 5. Tree of Thoughts

Explores multiple reasoning branches with backtracking.

**Config**: [`config/experiments/tree_of_thoughts/`](../../config/experiments/tree_of_thoughts/)

---

## Running Experiments

```bash
# DeepConf on AIME 2025
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_qwen3_aime2025

# Self-Consistency on AIME 2025
python scripts/run_tts_eval.py \
    --config-name=experiments/self_consistency/sc_qwen3_aime2025
```

---

## References

- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) - Snell et al., 2024
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023
