# GSM8K Dataset

## Overview

**GSM8K** (Grade School Math 8K) is a dataset of 8,500 grade school math word problems requiring multi-step reasoning.

**Dataset source**: [HuggingFace - openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)

**Why GSM8K for DeepConf:**
- Simple setup - auto-downloads from HuggingFace
- Low compute - short problems (~50 tokens), short answers (~200-400 tokens)
- Clear evaluation - numeric answer comparison
- Compatible format - uses `\boxed{answer}` notation
- Fast iteration - 100 samples complete in 10-30 minutes

## Quick Start

**Step 1: Setup API key**
```bash
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your-key-here
```

**Step 2: Run evaluation**
```bash
# Test on 1 sample
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf \
  dataset.subset=1

# Small benchmark (10 samples)
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf \
  dataset.subset=10

# Full benchmark (100 samples, ~30min)
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf \
  dataset.subset=100
```

## DeepConf Strategy Modes

**Offline mode (default)**: Generate all N traces, then filter and vote
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf \
  dataset.subset=10 \
  strategy.mode=offline \
  strategy.budget=8
```

**Online mode**: Warmup phase determines confidence threshold, then adaptive generation
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf \
  dataset.subset=10 \
  strategy.mode=online \
  strategy.warmup_traces=4 \
  strategy.total_budget=16
```

## Configuration

All configurations can be overridden via command line:

**Model settings:**
```bash
model.model_name="openai/gpt-4o"              # Change model
model=together_ai                              # Switch provider
model.top_logprobs=20                          # Top-k for confidence
```

**Strategy settings:**
```bash
strategy.budget=16                             # Number of traces (offline)
strategy.mode=online                           # Switch to online mode
strategy.warmup_traces=4                       # Warmup traces (online)
strategy.total_budget=16                       # Total budget (online)
strategy.filter_method=top10                   # Filter: top10, top5, threshold
strategy.confidence_threshold=0.85             # Threshold value (if using threshold)
strategy.window_size=2048                      # Sliding window size
```

**Generation settings:**
```bash
generation.temperature=0.8                     # Sampling temperature
generation.max_new_tokens=2048                 # Max tokens per trace
```

## Output

Results saved to: `outputs/YYYY-MM-DD/HH-MM-SS/`

### Files

- `results.pt` - PyTorch tensor with all results
- `.hydra/config.yaml` - Full configuration used
- `.hydra/hydra.log` - Execution logs

### Result Structure

```python
{
    "index": 0,
    "question": "Janet's ducks lay 16 eggs...",
    "gold_answer": "#### 18",
    "generated_trajectory": "Step 1: Calculate eggs used...",
    "generated_answer": "18",
    "validity_scores": [0.875],
    "completed": True,
    "is_correct": True,
    "metadata": {
        "mode": "offline",
        "total_traces": 8,
        "filtered_traces": 6,
        "confidence_score": 0.875,
        "selected_answer": "18",
        "vote_distribution": {"18": 0.875, "17": 0.125}
    }
}
```

## Expected Performance

Based on DeepConf paper benchmarks:

| Budget | Model           | Expected Accuracy | Time/Sample |
|--------|-----------------|-------------------|-------------|
| 8      | gpt-3.5-turbo   | ~70-75%          | ~10s        |
| 8      | gpt-4o-mini     | ~75-80%          | ~15s        |
| 16     | gpt-4o-mini     | ~80-85%          | ~30s        |
| 32     | gpt-4o          | ~85-90%          | ~60s        |

## Implementation Details

### Dataset Module (`llm_tts/datasets/gsm8k.py`)

- `load_gsm8k()` - Load from HuggingFace
- `extract_answer_from_gsm8k()` - Parse `#### X` format
- `format_gsm8k_for_deepconf()` - Convert to DeepConf format
- `evaluate_gsm8k_answer()` - Numeric comparison

### Prompt Template (`prompts/gsm8k_deepconf.txt`)

Instructs model to:
1. Solve step-by-step
2. Put final answer in `\boxed{answer}` format

### Configuration Files

- `config/experiments/deepconf/run_gsm8k_deepconf.yaml` - Main experiment config
- `config/dataset/gsm8k.yaml` - Dataset settings
- `config/strategy/deepconf.yaml` - Strategy defaults

## Troubleshooting

**No answers extracted:**
- Model isn't using `\boxed{}` format
- Check prompt template is loaded correctly
- Try stronger model (e.g., gpt-4o)

**API rate limits:**
- Reduce dataset size: `dataset.subset=10`
- Add delays between requests

**Generation cuts off:**
- Increase token limit: `generation.max_new_tokens=4096`

**Low confidence scores:**
- Increase logprobs: `model.top_logprobs=20`
- Check model supports logprobs properly
