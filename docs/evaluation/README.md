# Evaluation Protocol

This document defines the evaluation protocol for comparing test-time compute scaling strategies on mathematical reasoning tasks.

## Quick Links

- [Datasets](datasets.md) - AIME, MATH-500, SVAMP benchmark details
- [Models](models.md) - Model configurations and thinking mode settings
- [Metrics](metrics.md) - Accuracy, tokens, FLOPs calculation
- [WandB](wandb.md) - Logging conventions and upload workflow
- [Results](results/) - Experiment results by dataset
- [Architecture](../architecture.md) - Inference pipeline architecture

---

## Evaluation Algorithm

Follow these steps to run a reproducible evaluation:

### Step 1: Configure Reproducibility

Set `seed=42` and consistent `torch_dtype` for all experiments. These must be propagated to all components.

```yaml
# config/system/default.yaml
system:
  seed: 42                # REQUIRED for reproducibility
  torch_dtype: "bfloat16" # Options: float16, bfloat16, float32, auto
```

**Reproducibility checklist:**

| Component | Setting | Verified |
|-----------|---------|----------|
| System config | `system.seed: 42` | [ ] |
| System config | `system.torch_dtype: bfloat16` | [ ] |
| vLLM SamplingParams | `seed=config.system.seed` | [ ] |
| Strategy (DeepConf/Self-Consistency) | `seed=config.system.seed` | [ ] |

### Step 2: Configure Model and Thinking Mode

For models with thinking mode (e.g., Qwen3), set `disable_thinking_mode` based on your experiment:

```yaml
# config/model/vllm_qwen3.yaml
model:
  type: vllm
  model_path: "Qwen/Qwen3-8B"
  disable_thinking_mode: true  # Set to false for thinking mode experiments
```

### Step 3: Configure Generation Parameters

Use model-specific recommended parameters. For Qwen3, parameters depend on thinking mode ([source](https://huggingface.co/Qwen/Qwen3-8B)):

| Mode | Temperature | top_p | top_k | Note |
|------|-------------|-------|-------|------|
| Non-thinking | 0.7 | 0.8 | 20 | DO NOT use greedy decoding |
| Thinking | 0.6 | 0.95 | 20 | For thinking mode |

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
| `system.torch_dtype` | `bfloat16` | Consistent precision |
| `model.disable_thinking_mode` | `true/false` | Depends on experiment type |
| `generation.temperature` | `0.7` / `0.6` | Non-thinking / Thinking mode |
| `generation.top_p` | `0.8` | Qwen3 recommended |
| `generation.top_k` | `20` | Qwen3 recommended |
| `strategy.temperature` | Same as generation | Consistency |
| `report_to` | `wandb` | Tracking |

---

## Inference Backends

### Recommended Backend: vLLM

**vLLM is the recommended backend for all experiments.** It provides significantly higher throughput and better memory efficiency compared to HuggingFace.

#### vLLM vs HuggingFace Comparison

| Aspect | vLLM | HuggingFace |
|--------|------|-------------|
| **Throughput** | High (PagedAttention, batching) | Low |
| **Memory** | Efficient (no OOM on long sequences) | OOM-prone on long sequences |
| **Hidden states** | Not accessible | Full access |
| **Attention scores** | Not accessible | Full access |
| **Stopping criteria** | Stop tokens only | Custom callbacks |
| **Prefix caching** | Native support | Not available |

#### When to use each backend

- **Use vLLM (recommended)** for most experiments:
  - Self-Consistency, DeepConf, CoT
  - Online/Offline Best-of-N
  - Uncertainty estimators that use logprobs (MeanTokenEntropy, Perplexity, etc.)

- **Use HuggingFace** only when you need:
  - Hidden states access (required for UHead uncertainty quantification)
  - Attention scores access
  - Custom `StoppingCriteria` callbacks not achievable with stop tokens

> **Note**: HuggingFace is significantly slower and prone to OOM errors on long sequences. Use it only when vLLM cannot provide the required model internals.

---

### vLLM

High-performance inference engine optimized for throughput.

```yaml
model:
  type: vllm
  model_path: "Qwen/Qwen3-8B"
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
  enable_prefix_caching: true
  max_model_len: 32768
```

| Pros | Cons |
|------|------|
| PagedAttention prevents OOM on long sequences | No access to hidden states (UHead UQ not supported) |
| Native batched generation for multiple traces | No access to attention scores |
| Higher throughput than HuggingFace | No custom stopping criteria callbacks |
| Prefix caching for repeated prompts | Stop tokens require post-validation |
| Thinking mode step detection via stop tokens | |

#### vLLM Initialization in `run_tts_eval.py`

The `create_model()` function (`scripts/run_tts_eval.py:252`) initializes vLLM with the following pipeline:

1. **vLLM Engine**: Creates `vllm.LLM` with config parameters (gpu_memory_utilization, tensor_parallel_size, prefix_caching, max_model_len, seed)

2. **SamplingParams**: Creates default sampling parameters from generation config (temperature, top_p, max_tokens, logprobs)

3. **lm-polygraph Adapter**: Wraps vLLM with `WhiteboxModelvLLM` for compatibility with strategies that expect lm-polygraph model interface

4. **Uncertainty Wrapper** (for step-based strategies): Creates `VLLMWithUncertainty` that computes uncertainty scores from vLLM logprobs:
   - `scorer.type: entropy` → `MeanTokenEntropy` estimator (default)
   - `scorer.type: perplexity` → `Perplexity` estimator

5. **Step Generator** (for Online BoN, Beam Search, etc.): Creates `VLLMStepGenerator` with detector based on `strategy.detector_type`:
   - `thinking_marker` → `ThinkingMarkerDetector` for semantic step boundaries in `<think>` content
   - `structured` → `StructuredStepDetector` for explicit `- Step N` markers

```
┌─────────────────────────────────────────────────────────────┐
│                     run_tts_eval.py                         │
├─────────────────────────────────────────────────────────────┤
│  config.model.type == "vllm"                                │
│  ┌─────────────┐                                            │
│  │  vllm.LLM   │  ← GPU inference engine                    │
│  └──────┬──────┘                                            │
│         │                                                   │
│  ┌──────▼──────────────┐                                    │
│  │  WhiteboxModelvLLM  │  ← lm-polygraph adapter            │
│  └──────┬──────────────┘                                    │
│         │                                                   │
│  ┌──────▼──────────────────┐                                │
│  │  VLLMWithUncertainty    │  ← Uncertainty from logprobs   │
│  │  (MeanTokenEntropy /    │                                │
│  │   Perplexity)           │                                │
│  └──────┬──────────────────┘                                │
│         │                                                   │
│  ┌──────▼──────────────┐                                    │
│  │  VLLMStepGenerator  │  ← Step-by-step generation         │
│  │  + Detector         │     (thinking_marker / structured) │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

### HuggingFace

Standard inference with full control over generation process.

```yaml
model:
  type: huggingface
  model_path: "Qwen/Qwen3-8B"
  device_map: auto
  torch_dtype: bfloat16
```

| Pros | Cons |
|------|------|
| Access to hidden states (required for UHead UQ) | Lower throughput than vLLM |
| Access to attention scores | Higher memory usage |
| Full `StoppingCriteria` callback support | May OOM on very long sequences |
| Custom step boundary detection during generation | |
| Fine-grained control over generation | |

### vLLM Thinking Mode Step Detection

vLLM now supports thinking mode step detection using a stop-and-validate approach:

```python
from llm_tts.generators.vllm import StepCandidateGeneratorThroughVLLM
from llm_tts.step_boundary_detectors.thinking.vllm import get_stop_tokens_compact

# Generate stop tokens from semantic markers
stop_tokens = get_stop_tokens_compact(
    use_sequence=True,      # "first", "next", "then"
    use_conclusion=True,    # "so", "therefore", "thus"
    use_thinking=True,      # "let me", "wait", "hmm"
    use_verification=True,  # "to verify", "let's check"
)

# Create generator with post-validation
generator = StepCandidateGeneratorThroughVLLM(
    model=vllm_model,
    stop_tokens=stop_tokens,
    min_step_chars=150,     # Minimum chars before accepting boundary
    max_step_chars=600,     # Force split after this many chars
)
```

**How it works:**
1. Generate with stop tokens derived from semantic markers
2. When generation stops, validate if it's a real step boundary
3. If `len(text) < min_step_chars`, continue generating (false boundary)
4. If `len(text) > max_step_chars`, force accept as step boundary
5. Include stop string in output with `include_stop_str_in_output=True`

This approach works well for thinking mode and offers higher throughput than HuggingFace.

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
| Online Best-of-N | Step-by-step candidate selection | - |
| Offline Best-of-N | Generate N trajectories, select best | - |
| Beam Search | Beam search over reasoning steps | - |
| Phi Decoding | Phi-based step selection | [φ-Decoding](https://arxiv.org/abs/2503.13288) |

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

# Online Best-of-N with vLLM thinking mode
python scripts/run_tts_eval.py \
    --config-path ../config/experiments/thinking_mode \
    --config-name online_bon_vllm_thinking_aime2025 \
    strategy.min_step_chars=150 strategy.max_step_chars=600

# Offline Best-of-N with vLLM thinking mode
python scripts/run_tts_eval.py \
    --config-path ../config/experiments/thinking_mode \
    --config-name offline_bon_thinking_aime2025

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
