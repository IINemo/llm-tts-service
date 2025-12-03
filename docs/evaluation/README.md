# Evaluation Protocol

This document defines the evaluation protocol for comparing test-time compute scaling strategies on mathematical reasoning tasks.

## Table of Contents

- [Datasets](datasets.md) - AIME, MATH-500, SVAMP benchmark details
- [Models](models.md) - Paper-model matrix for strategy implementations
- [Metrics](metrics.md) - Accuracy, tokens, FLOPs calculation
- [Results](results/) - Experiment results by dataset
- [Model Configuration](#model-configuration) - Config files for experiments
- [Strategies](#strategies) - Test-time compute scaling methods

---

## Model Configuration

### Configuration Files

- Model config: [config/model/hf_qwen3.yaml](../../config/model/hf_qwen3.yaml)
- Generation config: [config/generation/default.yaml](../../config/generation/default.yaml)
- Prompt template: [config/prompts/default.txt](../../config/prompts/default.txt)

> **IMPORTANT**: Do not modify model or generation configurations between experiments to ensure fair comparison.

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

```bash
# DeepConf on AIME 2025
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_qwen3_aime2025

# Self-Consistency on AIME 2025
python scripts/run_tts_eval.py \
    --config-name=experiments/self_consistency/sc_qwen3_aime2025
```

