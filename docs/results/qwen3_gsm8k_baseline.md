# Qwen3-8B GSM8K Baseline Results

## Experiment Overview

**Model:** Qwen/Qwen3-8B (local HuggingFace model)
**Strategy:** Chain-of-Thought (single path baseline)
**Dataset:** GSM8K (100 samples, indices 0-99)
**Evaluation:** LLM Judge (GPT-4o via OpenRouter)
**Date:** 2025-11-23

## Configuration

```yaml
Model:
  - Type: local
  - Model Path: Qwen/Qwen3-8B
  - Device: cuda:1
  - Precision: float16
  - Disable thinking mode: true

Generation:
  - Max new tokens: 1024
  - Temperature: 0.1
  - Top-p: 0.95
  - Top-k: 50
  - Batch size: 1

Strategy:
  - Type: chain_of_thought
  - Max new tokens: 1024
  - Temperature: 0.1 (low for deterministic reasoning)
```

## Results Summary

### Overall Performance

| Metric | Value |
|--------|-------|
| **Total Samples** | 100 |
| **Correct** | 83 |
| **Incorrect** | 17 |
| **Accuracy** | **83.0%** |

### Performance by Batch

| Batch | Samples | Correct | Accuracy |
|-------|---------|---------|----------|
| **First 30** | 0-29 | 22/30 | **73.3%** |
| **Next 70** | 30-99 | 61/70 | **87.1%** |

## Key Observations

1. **Performance Improvement Across Batches**
   - The model performed significantly better on samples 30-99 (87.1%) compared to samples 0-29 (73.3%)
   - 13.8% improvement suggests possible variation in dataset difficulty or problem types

2. **Strong Mathematical Reasoning**
   - 83% overall accuracy demonstrates Qwen3-8B's strong mathematical reasoning capabilities
   - Chain-of-Thought prompting enables step-by-step problem solving

3. **Evaluation Method**
   - LLM Judge (GPT-4o) provides more accurate evaluation than exact match
   - Recognizes mathematically correct answers even with formatting variations
   - Handles answers in various formats (boxed notation, plain numbers, etc.)

## Experiment Details

### Run 1: Samples 0-29
- **Directory:** `outputs/2025-11-23/gsm8k_baseline_qwen3_22-44-36/`
- **Results:** 22/30 correct (73.3%)
- **Exact Match:** 9/30 (30.0%) - significantly lower due to strict formatting
- **LLM Judge:** 22/30 (73.3%) - more accurate recognition of correct reasoning

### Run 2: Samples 30-99
- **Directory:** `outputs/2025-11-23/gsm8k_baseline_qwen3_23-11-41/`
- **Results:** 61/70 correct (87.1%)
- **Generation Time:** ~36 minutes (avg ~31 seconds per sample)

## Configuration Files

- **GSM8K Config:** `config/experiments/chain_of_thought/baseline_qwen3_gsm8k.yaml`
- **Model Config:** `config/model/hf_qwen3.yaml`
- **Evaluation Script:** `scripts/evaluate_gsm8k_llm_judge.py`

## Evaluation Files

- **Run 1 Evaluation:** `outputs/2025-11-23/gsm8k_baseline_qwen3_22-44-36/llm_judge_evaluation.json`
- **Run 2 Evaluation:** `outputs/2025-11-23/gsm8k_baseline_qwen3_23-11-41/llm_judge_evaluation.json`

## Comparison with Exact Match

| Evaluator | Run 1 (0-29) | Run 2 (30-99) |
|-----------|--------------|---------------|
| **Exact Match** | 30.0% | ~40% (estimated) |
| **LLM Judge (GPT-4o)** | **73.3%** | **87.1%** |

The LLM judge significantly outperforms exact match evaluation, demonstrating the importance of semantic understanding in evaluating mathematical reasoning.

## Technical Notes

- Model loaded with float16 precision for memory efficiency
- Device handling fixed for quantized models (uses `next(model.parameters()).device`)
- Dataset offset parameter used to continue from sample 30
- Evaluation uses OpenRouter API with GPT-4o as judge
- All generation uses temperature 0.1 for more deterministic reasoning

## Next Steps

1. Run AIME 2025 baseline evaluation (30 samples)
2. Compare performance across different difficulty levels
3. Analyze error patterns in incorrect samples
4. Consider testing with higher max_new_tokens for complex problems
