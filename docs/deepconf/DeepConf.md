# DeepConf - Deep Think with Confidence

Complete guide for DeepConf implementation with OpenRouter API.

**Paper**: [Deep Think with Confidence](https://arxiv.org/abs/2508.15260)
**Original Code**: [github.com/facebookresearch/deepconf](https://github.com/facebookresearch/deepconf)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Algorithm Overview](#algorithm-overview)
3. [Implementation Details](#implementation-details)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
pip install openai
export OPENROUTER_API_KEY="your-key"
```

### Basic Usage

```python
from llm_tts.models import create_model
from llm_tts.strategies.deepconf_strategy import DeepConfStrategy

# Create model
model = create_model(
    provider="openrouter",
    model_name="openai/gpt-4o-mini",
    api_key="your-key",
    top_logprobs=20
)

# Create strategy
strategy = DeepConfStrategy(
    model=model,
    budget=8,              # Number of reasoning traces
    window_size=2048,      # Sliding window size
    filter_method="top10"  # Keep top 10% by confidence
)

# Run DeepConf
prompt = "What is 23 + 47? Put your answer in \\boxed{}."
result = strategy.generate_trajectory(prompt)

print(f"Answer: {result['metadata']['selected_answer']}")
print(f"Confidence: {result['metadata']['confidence_score']:.2%}")
```

### Running Experiments

**Offline mode** (generate all traces, then filter & vote):
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf \
  dataset.subset=10 \
  strategy.mode=offline \
  strategy.budget=8
```

**Online mode** (warmup phase + adaptive generation):
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf \
  dataset.subset=10 \
  strategy.mode=online \
  strategy.warmup_traces=4 \
  strategy.total_budget=16
```

**Key parameters:**
- `strategy.mode`: `offline` or `online`
- `strategy.budget`: Number of traces (offline mode)
- `strategy.warmup_traces`: Warmup traces (online mode)
- `strategy.total_budget`: Total traces including warmup (online mode)
- `strategy.filter_method`: `top10`, `top5`, or `threshold`
- `strategy.window_size`: Sliding window for confidence (default: 2048)
- `eval_method`: `simple` (numeric comparison) or `llm_judge` (LLM-based verification)

---

## Algorithm Overview

### Core Idea

DeepConf generates multiple reasoning traces, scores them by confidence using token probabilities, filters low-confidence traces, and performs weighted voting.

### Algorithm Steps

#### 1. Generate Multiple Traces
```python
for i in range(budget):
    trace_i = model.generate(prompt, temperature=0.7)
```

#### 2. Calculate Token Confidence
**Formula**: `C_i = -(1/k) * Σ_{j=1}^k log P_i(j)`

Where:
- `C_i` = confidence for token i
- `k` = top-k token probabilities (typically 20)
- `P_i(j)` = probability of j-th most likely token

Higher score = more confident.

#### 3. Sliding Window Group Confidence
**Formula**: `CG_i = (1/|G_i|) * Σ_{t∈G_i} C_t`

Where:
- `CG_i` = group confidence for window at position i
- `G_i` = sliding window of size w starting at i

**IMPORTANT: Windows overlap!**
- Windows slide by 1 token
- Each window shares (window_size - 1) tokens with the next
- Source: `deepconf/deepconf/utils.py:59-60`

**Trace confidence** = `min(all_window_confidences)`

#### 4. Filter Traces
- **top10**: Keep top 10% by confidence
- **threshold**: Keep traces above fixed threshold
- **none**: Use all traces

#### 5. Weighted Voting
```python
for trace in filtered_traces:
    weight[trace.answer] += trace.min_conf

selected = argmax(weight)
```

Each trace "votes" with weight = its confidence.

### Modes

**Offline Mode** (Simpler):
1. Generate all N traces
2. Calculate confidence for each
3. Filter by confidence
4. Weighted voting

**Online Mode** (Efficient):
1. Warmup: Generate 16 traces, calibrate threshold
2. Generation: Use early stopping if confidence drops below threshold
3. Filter and vote on high-confidence traces

---

## Implementation Details

### Key Files

```
llm_tts/
├── models/
│   ├── base.py              # Abstract model interface
│   ├── openrouter.py        # OpenRouter with logprobs
│   └── factory.py           # Model factory
├── strategies/
│   └── deepconf_strategy.py # Accurate DeepConf
└── confidence_scoring.py    # Confidence utilities

tests/deepconf/
├── test_deepconf_accurate.py  # Integration tests
└── test_deepconf_math.py      # Math problems
```

### Answer Extraction

DeepConf expects **LaTeX boxed format**:
```
\boxed{answer}
```

**Examples:**
- `\boxed{70}` → `"70"` ✅
- `\boxed{x+y}` → `"x+y"` ✅
- Plain text → `None` ❌

### Key Improvements Over Previous Version

| Issue | Before | After |
|-------|--------|-------|
| Answer extraction | "the final answer is 70" ❌ | "70" ✅ |
| Confidence | Heuristic text-based | Real token logprobs ✅ |
| Formula | Approximations | Exact DeepConf formula ✅ |
| Windows | Not clear | Confirmed overlapping ✅ |

---

## Configuration

### Strategy Parameters

```python
DeepConfStrategy(
    model=model,
    budget=8,              # Number of traces (8-512)
    window_size=2048,      # Sliding window (16, 32, 2048)
    temperature=0.7,       # Sampling temperature (0.6-0.8)
    top_p=0.95,           # Top-p sampling
    max_tokens=4096,      # Max tokens per trace
    top_logprobs=20,      # Top-k for confidence
    filter_method="top10" # none, top10, threshold
)
```

### Filter Methods

1. **`"none"`** - No filtering, use all traces
2. **`"top10"`** - Keep top 10% by confidence
3. **`"threshold"`** - Keep traces above `confidence_threshold`

### Example Configurations

**High Accuracy**:
```python
strategy = DeepConfStrategy(model, budget=16, filter_method="top10")
```

**Fast Prototyping**:
```python
strategy = DeepConfStrategy(model, budget=5, filter_method="none")
```

**Custom Threshold**:
```python
strategy = DeepConfStrategy(
    model,
    budget=8,
    filter_method="threshold",
    confidence_threshold=12.0
)
```

---

## Testing

### Integration Tests

```bash
python tests/deepconf/test_deepconf_accurate.py
```

Tests:
1. ✅ Answer Extraction - LaTeX parsing
2. ✅ Model Logprobs Support
3. ✅ DeepConf Generation
4. ✅ Weighted Voting

### Math Problems

```bash
# Standard run
python tests/deepconf/test_deepconf_math.py

# Verbose - see full reasoning paths
python tests/deepconf/test_deepconf_math.py --verbose

# More traces
python tests/deepconf/test_deepconf_math.py --budget 10
```

Problems:
- Difference of squares: 15² - 8²
- Rectangle area: 12 × 8
- Speed-distance: 120 km in 2 hours
- Arithmetic series: 1+2+...+10
- Order of operations: (3+4) × (5+6)

---

## Troubleshooting

### "Model does not support logprobs"

**Solution**: Use OpenRouter with OpenAI models:
- ✅ gpt-4o-mini
- ✅ gpt-4o
- ✅ gpt-3.5-turbo

### "No valid answers extracted"

**Solution**: Include `\boxed{}` in prompt:
```python
prompt = "Calculate X. Put answer in \\boxed{}."
```

### "All traces have same answer"

**Solution**: Increase temperature for diversity:
```python
strategy = DeepConfStrategy(model, temperature=0.9)
```

### Import Errors

**Solution**: Run from project root:
```bash
cd /path/to/llm-tts-service
python tests/deepconf/test_deepconf_accurate.py
```

---

## Performance

### OpenRouter + gpt-4o-mini

- Budget=8, max_tokens=500: ~15-20s, ~$0.02
- Budget=16, max_tokens=1000: ~40-50s, ~$0.08

### Cost Optimization

1. Reduce `budget` (fewer traces)
2. Reduce `max_tokens` (shorter reasoning)
3. Use `filter_method="top10"` (quality over quantity)

---

## References

- **Paper**: [arxiv.org/abs/2508.15260](https://arxiv.org/abs/2508.15260)
- **Original Code**: [github.com/facebookresearch/deepconf](https://github.com/facebookresearch/deepconf)
- **OpenRouter**: [openrouter.ai/docs](https://openrouter.ai/docs)
