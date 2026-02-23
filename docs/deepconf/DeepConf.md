# DeepConf - Deep Think with Confidence

Guide for the DeepConf strategy integrated with the framework's `step_generator`.

**Paper**: [Deep Think with Confidence](https://arxiv.org/abs/2508.15260)
**Original Code**: [github.com/facebookresearch/deepconf](https://github.com/facebookresearch/deepconf)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Algorithm Overview](#algorithm-overview)
3. [Implementation Details](#implementation-details)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Running an Experiment

```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/your_config \
  dataset.subset=10
```

### Minimal Config

Create a config YAML under `config/experiments/deepconf/`:

```yaml
# @package _global_
defaults:
  - /config
  - /dataset/math_500
  - /model/vllm_qwen3
  - /generation/default
  - /system/default
  - /evaluation/default
  - _self_

run_name: "deepconf_example_${now:%H-%M-%S}"

scorer: null  # DeepConf has built-in weighted voting

strategy:
  type: deepconf
  min_step_tokens: 50
  budget: 16
  window_size: 2048
  filter_method: "top10"
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `strategy.budget` | Number of traces to generate per sample | 8 |
| `strategy.window_size` | Sliding window size for confidence | 2048 |
| `strategy.filter_method` | `"top10"`, `"top5"`, `"none"`, or `"threshold"` | `"top10"` |
| `strategy.confidence_threshold` | Threshold value (only when `filter_method="threshold"`) | null |
| `strategy.data_name` | Dataset name for answer extraction | null |
| `generation.temperature` | Sampling temperature (controls diversity) | 0.7 |
| `generation.max_new_tokens` | Max tokens per trace | 32768 |

Generation parameters (`temperature`, `top_p`, `max_new_tokens`) are configured in the `generation` section and picked up by the step_generator automatically.

---

## Algorithm Overview

### Core Idea

DeepConf generates multiple reasoning traces in a single batched call, scores each trace by confidence using token-level logprobs, filters low-confidence traces, and performs weighted majority voting.

### Algorithm Steps

#### 1. Generate Multiple Traces

All N traces are generated in a **single** `generate_step_candidates_batch()` call via the framework's step_generator (vLLM or API backend). This is efficient because vLLM can batch all traces with shared prefix caching.

#### 2. Calculate Token Confidence

For each token position, extract top-k logprobs from `StepCandidate.other_data["raw_logprobs"]`:

**Formula**: `C_i = -(1/k) * sum_{j=1}^{k} log P_i(j)`

Where:
- `C_i` = confidence for token i
- `k` = number of top logprob values (up to 20)
- `P_i(j)` = probability of j-th most likely token

Higher score = more uncertain (lower confidence).

#### 3. Sliding Window Group Confidence

**Formula**: `CG_i = (1/|G_i|) * sum_{t in G_i} C_t`

Where:
- `CG_i` = group confidence for window at position i
- `G_i` = sliding window of size w starting at i

Windows overlap by sliding 1 token at a time. Each window shares `(window_size - 1)` tokens with the next.

**Trace confidence** = `min(all_window_confidences)`

#### 4. Filter Traces

- **top10** / **top5**: Keep top K traces by confidence (sorted by min_conf descending)
- **threshold**: Keep traces above a fixed confidence threshold
- **none**: Use all traces

If no traces pass the filter, all traces are used as fallback.

#### 5. Weighted Voting

```
for trace in filtered_traces:
    weight[trace.answer] += trace.min_conf

selected = argmax(weight)
```

Each trace "votes" with weight = its minimum window confidence.

---

## Implementation Details

### Architecture

DeepConf uses the framework's `step_generator` pattern (same as `StrategySelfConsistency`):

```
StrategyDeepConf
  └── step_generator.generate_step_candidates_batch()
        ├── VLLMStepGenerator (vLLM backend)
        └── StepCandidateGeneratorThroughAPI (API backend)
```

### Key Files

```
llm_tts/strategies/deepconf/
├── __init__.py       # Package exports
├── strategy.py       # Main strategy (~475 lines)
└── utils.py          # compute_sliding_window_confidence()

config/strategy/
└── deepconf.yaml     # Base strategy config
```

### Generation Flow

1. `generate_trajectories_batch(requests, sample_indices)` is called with M samples
2. Single `step_generator.generate_step_candidates_batch()` call generates M x N candidates
3. For thinking mode: `_complete_thinking_paths()` generates answer phases for candidates that stopped at `</think>`
4. For each candidate: extract `raw_logprobs` from `StepCandidate.other_data` and compute confidence via `_compute_trace_confidence()`
5. Per-sample `_filter_and_vote()` selects the best answer

### Logprob Format

The `raw_logprobs` from vLLM is `List[Dict[token_id -> Logprob]]` where each `Logprob` object has a `.logprob` attribute. For API models, `APILogprobData` objects provide the same interface. The confidence computation works identically for both backends.

### Thinking Mode Support

When `thinking_mode=True` (set via `model.disable_thinking_mode: false`):
- Generation stops at `</think>` (added to stop tokens)
- Answer phase is generated via `generate_answer_candidates_batch()`
- Full text = thinking + answer concatenated
- Confidence is computed from the thinking-phase logprobs

### Answer Extraction

Uses the framework's `llm_tts.utils.extract_answer()` with `answer_format="auto"`:
- Tries `<Answer>: ... <end of response>` format first
- Falls back to `\boxed{...}` format
- Supports nested braces

---

## Configuration

### Base Config (`config/strategy/deepconf.yaml`)

```yaml
type: deepconf
min_step_tokens: 50
budget: 8
window_size: 2048
filter_method: "top10"
confidence_threshold: null
```

### Filter Methods

| Method | Behavior |
|--------|----------|
| `"none"` | No filtering, use all traces |
| `"top10"` | Keep top 10 traces by confidence |
| `"top5"` | Keep top 5 traces by confidence |
| `"threshold"` | Keep traces with `min_conf >= confidence_threshold` |

### Example: vLLM with Qwen3

```yaml
model:
  type: "vllm"
  model_path: "Qwen/Qwen3-8B"
  disable_thinking_mode: true  # Non-thinking mode
  gpu_memory_utilization: 0.9
  enable_prefix_caching: true

generation:
  max_new_tokens: 32768
  temperature: 0.7
  top_p: 0.8
  top_k: 20

scorer: null

strategy:
  type: deepconf
  min_step_tokens: 50
  budget: 16
  window_size: 2048
  filter_method: "top10"
```

### Example: High Budget

```yaml
strategy:
  type: deepconf
  budget: 64
  filter_method: "top10"
  window_size: 2048
```

### Example: Custom Threshold

```yaml
strategy:
  type: deepconf
  budget: 16
  filter_method: "threshold"
  confidence_threshold: 12.0
  window_size: 2048
```

---

## Troubleshooting

### "No valid answers extracted"

Ensure the prompt instructs the model to use `\boxed{}` format:
```
Put your final answer in \boxed{}.
```

### "All traces have same answer"

Increase temperature for more diversity:
```yaml
generation:
  temperature: 0.9
```

### All min_conf values are 0.0 or inf

This means `raw_logprobs` is empty. Check:
- vLLM models always provide logprobs (hardcoded `logprobs=20`)
- API models need `supports_logprobs: true` in model config

### Low accuracy despite high budget

Try adjusting the filter method. `"top10"` keeps only the 10 most confident traces for voting. If budget < 10, all traces pass the filter anyway. Consider `"top5"` for smaller budgets or `"none"` to use all traces.

---

## References

- **Paper**: [arxiv.org/abs/2508.15260](https://arxiv.org/abs/2508.15260)
- **Original Code**: [github.com/facebookresearch/deepconf](https://github.com/facebookresearch/deepconf)
