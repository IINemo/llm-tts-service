# Qwen3-8B GSM8K Results

## Overview
Evaluation of Qwen3-8B model on GSM8K dataset (first 100 samples) comparing baseline Chain-of-Thought vs Self-Consistency strategies.

## Results Summary

| Strategy | Samples | Correct | Accuracy | Improvement |
|----------|---------|---------|----------|-------------|
| **Baseline (CoT)** | 100 | 31 | 31.0% | - |
| **Self-Consistency (n=16)** | 100 | 41 | 41.0% | **+10.0pp** |

## Strategy Details

### Baseline (Chain-of-Thought)
- **Strategy**: Single reasoning path with step-by-step thinking
- **Temperature**: 0.1 (deterministic)
- **Max tokens**: 1024
- **Results file**: `baseline_cot_100samples.json`

### Self-Consistency (n=16)
- **Strategy**: Generate 16 diverse reasoning paths, majority voting
- **Temperature**: 0.7 (higher for diversity)
- **Max tokens**: 1024
- **Batch size**: 8 (for efficient generation)
- **Generation time**: ~89 seconds per sample
- **Results file**: `selfconsistency_n16_100samples.json`

## Key Findings

1. **Self-consistency improves accuracy by 10 percentage points** (31% → 41%)
2. **Batched generation** (batch_size=8) provides ~4x speedup over sequential generation
3. **Majority voting** effectively filters out incorrect reasoning paths
4. **Higher temperature** (0.7 vs 0.1) creates sufficient diversity in reasoning paths

## Model Configuration

- **Model**: Qwen/Qwen3-8B
- **Precision**: float16
- **Device**: CUDA (single GPU)
- **Prompt template**: Standard GSM8K baseline prompt

## Generation Details

### Batched Generation Implementation
The self-consistency strategy uses batched generation to improve efficiency:
- Generates 16 paths in 2 batches of 8
- Clears GPU cache after each batch
- Handles OOM gracefully with automatic retry

### Performance Metrics
- **Baseline**: ~30 seconds per sample
- **Self-consistency**: ~89 seconds per sample (16 paths)
- **Per-path overhead**: ~5.6 seconds (includes voting and filtering)

## Files

- `baseline_cot_100samples.json` - Full baseline results with generated reasoning
- `selfconsistency_n16_100samples.json` - Full self-consistency results with all 16 paths
- `README.md` - This summary file

## Date
November 24, 2025
