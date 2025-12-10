# Metrics

## Overview

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| **Accuracy** | Primary | % | Exact match with gold answer |
| **Total Tokens** | Primary | tokens | Sum of generated tokens across all traces |
| **FLOPs** | Primary | TFLOPs | Estimated compute cost |
| **Generation Time** | Secondary | seconds | Wall-clock time |
| **Token Efficiency** | Secondary | ratio | Actual vs max possible tokens |

---

## Primary Metrics

### 1. Accuracy (Exact Match)

Percentage of samples where the predicted answer exactly matches the gold answer.

```python
accuracy = correct_answers / total_samples * 100
```

- Uses string matching after normalization
- Implementation: [`llm_tts/evaluation/exact_match.py`](../../llm_tts/evaluation/exact_match.py)

**Reported as**: `XX.X%` (e.g., `26.7%`)

---

### 2. Token Count

Total number of tokens generated across all traces/paths for a sample.

```python
# Per sample
total_tokens = sum(trace["num_tokens"] for trace in all_traces)

# Dataset aggregation
avg_tokens_per_sample = mean([sample_total_tokens for all samples])
total_tokens_dataset = sum([sample_total_tokens for all samples])
```

**Reported as**:
- Per sample: total tokens (e.g., `45,230 tokens`)
- Dataset: average tokens per sample (e.g., `42,150 avg tokens/sample`)

---

### 3. Compute Cost (FLOPs)

Estimated floating-point operations required for autoregressive inference.

#### Approximate Formula

For transformer language models, FLOPs per generated token during **inference** can be approximated by:

```python
flops_per_token = 2 * num_parameters
```

This follows the compute model introduced in [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) ("Scaling Laws for Neural Language Models"), where inference requires approximately **one forward pass**, yielding an inference cost of **≈ 2N FLOPs**, with `N` equal to the number of non-embedding parameters. The interpretation of *≈2 FLOPs per parameter* aligns with standard FLOP accounting conventions (multiply + add).

#### Example for Qwen3-8B (~8B parameters)

```python
flops_per_token = 2 * 8e9 = 16e9  # 16 GFLOPs/token
```

This approximation is appropriate when comparing generation strategies on the **same model**, because attention-related terms are dominated by dense matrix multiplications in large transformer blocks.

#### Per-Sample and Dataset Compute

```python
# Per sample
total_flops = total_tokens * flops_per_token
total_tflops = total_flops / 1e12
```

**Reported as**: `XXX TFLOPs` (e.g., `672 TFLOPs avg/sample`)

#### Important Notes

- Actual FLOPs vary with sequence length due to attention costs.
- With **KV caching**, the attention computation for each new token is **linear** in sequence length rather than quadratic. This is described in [PaLM (2022)](https://arxiv.org/abs/2204.02311) Appendix B.2 and supported by [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135).
- The approximate **2N** rule remains accurate in practice unless the sequence length becomes very large.

#### More Precise FLOP Calculation

A full transformer forward pass can be decomposed using the formulation from [PaLM (2022)](https://arxiv.org/abs/2204.02311) Appendix B and engineering analyses such as [Kipply (2022)](https://kipp.ly/transformer-inference-arithmetic/):

```
FLOPs_per_layer = 24 * b * s * h²   +   4 * b * s² * h
                  \_____________/       \____________/
                  FFN + projections        attention
```

Where:
- `b` = batch size
- `s` = sequence length
- `h` = hidden dimension (`d_model`)

During autoregressive inference with a **KV cache**, the second term simplifies from `s²` to `s`, because keys and values for previous tokens are reused. This results in a practical per-token cost dominated by `O(h²)` dense compute.

#### Implementation

FLOP calculation is implemented in [`FLOPCalculator`](../../llm_tts/utils/flops.py):

```python
from llm_tts.utils import FLOPCalculator

# Initialize with model name (auto-loads architecture from HuggingFace)
calc = FLOPCalculator("Qwen/Qwen3-8B")

# Compute TFLOPs for token count
tflops = calc.compute_tflops(num_tokens=45000)

# Get TFLOPs per 1k tokens (useful for comparison)
print(f"{calc.tflops_per_1k_tokens:.3f} TFLOPs/1k tokens")
```

In `run_tts_eval.py`, FLOPCalculator is used to track compute cost per sample:
- Initialized after model loading with the model path
- `tflops_this_sample` logged per sample based on `tokens_this_sample`
- `running_total_tflops` accumulated across all samples
- `running_avg_tflops_per_sample` computed as running average

#### References

- [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) - "Scaling Laws for Neural Language Models" (OpenAI)
- [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) - "Training Compute-Optimal Large Language Models" (Chinchilla)
- [Chowdhery et al. (2022)](https://arxiv.org/abs/2204.02311) - "PaLM: Scaling Language Modeling with Pathways", Appendix B
- [Dao et al. (2022)](https://arxiv.org/abs/2205.14135) - "FlashAttention"
- [Kipply (2022)](https://kipp.ly/transformer-inference-arithmetic/) - "Transformer Inference Arithmetic"
- [Adam Casson (2023)](https://www.adamcasson.com/posts/transformer-flops) - "Transformer FLOPs"

---

## Secondary Metrics

### 4. Generation Time

Wall-clock time for generating all traces per sample.

```python
generation_time_sec = end_time - start_time
```

**Reported as**: `XXX sec/sample` (e.g., `165 sec avg`)

---

### 5. Token Efficiency

Measures how efficiently tokens are used, especially relevant for early stopping.

```python
# Per trace
token_efficiency = actual_tokens / max_tokens

# Early stopping rate
early_stop_rate = traces_stopped_early / total_traces * 100
```

**Reported as**:
- Efficiency ratio (e.g., `0.73`)
- Early stop rate (e.g., `87.5%`)

---

## Strategy-Specific Metrics

### Confidence Statistics (DeepConf)

```python
{
    "mean_confidence": mean(all_trace_confidences),
    "min_confidence": min(all_trace_confidences),
    "max_confidence": max(all_trace_confidences),
}
```

### Answer Distribution (Self-Consistency, DeepConf)

```python
{
    "unique_answers": len(set(all_answers)),
    "top_answer_votes": max(vote_counts),
    "consensus_strength": top_votes / total_traces,
}
```

---

## Results JSON Schema

Each sample result includes:

```json
{
    "index": 0,
    "question": "...",
    "gold_answer": "42",
    "generated_answer": "42",
    "all_traces": [
        {
            "trace_id": 0,
            "text": "...",
            "num_tokens": 1500,
            "min_conf": 0.85,
            "mean_conf": 0.91,
            "answer": "42",
            "selected": true
        }
    ],
    "metadata": {
        "strategy": "deepconf",
        "config": {...},
        "results": {...}
    }
}
```

---

## Reporting Format

### Example Results Table

| Strategy | Accuracy | Avg Tokens | Avg TFLOPs | Avg Time |
|----------|----------|------------|------------|----------|
| Self-Consistency (n=16) | 23.3% | 45,000 | 0.72 | 180s |
| DeepConf (b=16) | 26.7% | 42,000 | 0.67 | 165s |

### Required Information

When reporting results, include:

1. **Dataset**: Name, subset size
2. **Strategy**: Name, key hyperparameters (budget, temperature)
3. **Config**: Path to experiment config (e.g., `config/experiments/deepconf/deepconf_qwen3_aime2025.yaml`)
4. **Model**: Name, size, precision
5. **Hardware**: GPU type
6. **Metrics**: Accuracy, tokens, FLOPs, time
