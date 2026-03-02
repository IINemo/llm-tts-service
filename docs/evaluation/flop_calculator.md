# FLOP Calculator: Compute Cost Tracking Pipeline

This document describes the full pipeline for estimating inference compute cost (FLOPs/TFLOPs) across all strategies and scorers. It covers the theoretical formulas, per-component token tracking, PRM-specific accounting, and how metrics are aggregated and reported.

**Implementation**: [`llm_tts/utils/flops.py`](../../llm_tts/utils/flops.py)

---

## 1. Theoretical Foundation

### 1.1 Simple Method (Default)

```
FLOPs = 2 × N × T
```

Where:
- `N` = number of model parameters (non-embedding approximation)
- `T` = total tokens processed (input context + generated output)

This follows [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361): inference requires ~2 FLOPs per parameter per token (one multiply + one add per weight).

**Example — Qwen3-8B:**

```
FLOPs per token = 2 × 8×10⁹ = 16 GFLOPs/token
1000 tokens → 16 TFLOPs
```

### 1.2 Precise Method (Architecture-Aware)

Per-layer cost with KV cache ([PaLM, 2022](https://arxiv.org/abs/2204.02311) Appendix B):

```
FLOPs_per_layer = 24 × b × h²        +  4 × b × s × h
                  ───────────────        ─────────────────
                  FFN + projections      attention (KV-cached)
```

Where:
- `b` = batch size
- `h` = hidden dimension
- `s` = sequence length (average)
- The first term covers: QKV projections (3h²) + output projection (h²) + FFN (8h²) = 12h², times 2 for multiply-add
- The second term is linear in `s` (not quadratic) because KV cache stores previous keys/values

**When to use which**: The simple method (`2N`) is the default and is accurate enough for comparing strategies on the same model. The precise method matters only when sequence lengths are very large (>32k) or when comparing across architectures.

### 1.3 Registered Model Architectures

| Model | Parameters | Hidden Size | Layers | Heads | TFLOPs/1k tokens |
|-------|-----------|-------------|--------|-------|------------------|
| `Qwen/Qwen3-8B` | 8B | 4096 | 36 | 32 | 16.0 |
| `Qwen/Qwen2.5-Math-7B-Instruct` | 7B | 3584 | 28 | 28 | 14.0 |
| `Qwen/Qwen2.5-Math-PRM-7B` | 7B | 3584 | 28 | 28 | 14.0 |

Models not in the table are auto-detected from HuggingFace configs at runtime.

---

## 2. Token Tracking Architecture

The system has two independent trackers that are merged at finalization:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Generator Tracker                             │
│                     (base.py → _per_sample_stats)                 │
│                                                                   │
│  Per generation call:                                             │
│    input_tokens += len(tokenizer.encode(prompt))   ← counted     │
│    output_tokens += Σ len(candidate.token_ids)       once per     │
│                                                      prompt       │
│                                                                   │
│  TFLOPs = 2 × N_gen × (input_tokens + output_tokens) / 10¹²     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │  merged in _finalize_sample()
                            │
┌───────────────────────────┴──────────────────────────────────────┐
│                      PRM Tracker                                  │
│                      (step_scorer_prm.py → _per_sample_prm_tokens)│
│                                                                   │
│  Per scoring call:                                                │
│    prm_input_tokens += len(prm_tokenizer.encode(prompt))          │
│                        for EACH trajectory scored                 │
│                                                                   │
│  PRM TFLOPs = 2 × N_prm × prm_input_tokens / 10¹²               │
│  (input-only: PRM is a reward model, no generation)               │
└──────────────────────────────────────────────────────────────────┘

Final: token_stats["tflops"] = gen_tflops + prm_tflops
```

---

## 3. Generator Token Counting

### 3.1 How Tokens Are Recorded

In `VLLMStepGenerator._generate_step_candidates_impl()`:

```python
# 1. Tokenize each prompt to count context tokens
per_prompt_context_tokens = [len(self.tokenizer.encode(p)) for p in prompts]

# 2. vLLM generates with n=candidates_per_step from each prompt
outputs = raw_llm.generate(prompts, sampling_params)

# 3. Record per-sample stats
for prompt_idx, traj_idx in enumerate(traj_indices):
    sid = sample_ids[traj_idx]
    ctx = per_prompt_context_tokens[prompt_idx]    # ← once per prompt
    self.record_sample_tokens(sid, candidates, context_tokens=ctx)
```

In `record_sample_tokens()`:

```python
# input_tokens: accumulated context tokens (prompt + trajectory)
self._per_sample_stats[sample_id]["input_tokens"] += context_tokens

# output_tokens: ALL candidates' tokens (n=5 → sum of 5 sequences)
out = sum(len(c.token_ids) for c in candidates)
self._per_sample_stats[sample_id]["output_tokens"] += out
```

### 3.2 Key Detail: vLLM `n` Parameter

When `candidates_per_step=5` (beam search default), vLLM generates 5 completions from one prompt:
- **Prefill**: done once per prompt (all 5 candidates share it)
- **Decode**: 5 independent sequences

Token accounting reflects this correctly:
- `context_tokens` counted **once per prompt** (not multiplied by 5)
- `output_tokens` counted for **all 5 candidates** (sum of 5 sequences)

### 3.3 What the Prompt Contains

Each generation call builds a prompt:

```
[system_prompt] + [question] + [trajectory_step_1] + ... + [trajectory_step_k]
```

The context grows with each beam search step. For a 20-step beam search with 3 active beams:

| Step | Context per beam (approx) | Beams | Total input tokens |
|------|---------------------------|-------|--------------------|
| 1 | 500 | 3 | 1,500 |
| 5 | 2,000 | 3 | 6,000 |
| 10 | 3,500 | 3 | 10,500 |
| 20 | 6,500 | 3 | 19,500 |

---

## 4. PRM Token Counting

### 4.1 PRM Scoring Flow in Beam Search

At each beam search step, PRM scores **all** candidate trajectories:

```
beam_size × candidates_per_beam = 3 × 5 = 15 trajectories per step
```

Each trajectory is converted to a PRM prompt:

```
question <extra_0> step_1 <extra_0> step_2 <extra_0> ... <extra_0> step_k+1
```

Where `step_k+1` is the new candidate being evaluated.

### 4.2 Token Recording

In `score_trajectories_batch()`:

```python
for traj_idx, (chat, trajectory) in enumerate(zip(chats, trajectories)):
    # Build prompt with tail truncation to prm_max_tokens (default: 4000)
    prompt, num_skipped = self._truncate_steps_from_tail(
        question, step_texts, self.prm_max_tokens
    )
    # Count ACTUAL prompt tokens (after truncation)
    num_prompt_tokens = len(self.prm_tokenizer.encode(prompt))
    # Record against the sample
    self._record_prm_tokens(num_prompt_tokens, sample_id=sample_id)
```

PRM counts **input tokens only** — it is a reward model that produces scores via forward pass, not generation. The formula `2 × 7B × input_tokens` is correct for a prefill-only workload.

### 4.3 Tail Truncation

When the trajectory exceeds `prm_max_tokens` (default 4000), **early steps are dropped** (tail truncation):

```
Full trajectory:  [step_1] [step_2] [step_3] ... [step_20] [new_candidate]
After truncation: [step_12] [step_13] ... [step_20] [new_candidate]  ← fits in 4000 tokens
```

The token count is measured **after** truncation, so FLOPs reflect actual processed tokens.

### 4.4 PRM FLOP Calculator Initialization

The PRM scorer gets its own `FLOPCalculator` instance initialized with the PRM model name:

```python
# In run_tts_eval.py
scorer.init_flop_calculator(config.scorer.model_path)
# → FLOPCalculator("Qwen/Qwen2.5-Math-PRM-7B", method="simple")
# → 2 × 7B = 14 GFLOPs/token
```

This is separate from the generator's calculator (2 × 8B = 16 GFLOPs/token).

---

## 5. Strategy-Specific Tracking

### 5.1 Beam Search

Per beam search step:

```
┌─ Generation ─────────────────────────────────────────────────────┐
│  3 prompts (one per beam) → vLLM generates n=5 from each        │
│  Input: 3 × context_tokens  (counted once per prompt)            │
│  Output: 3 × 5 × step_tokens  (all candidates)                  │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─ PRM Scoring (if enabled) ───────────────────────────────────────┐
│  15 trajectory prompts (3 beams × 5 candidates)                  │
│  Each prompt: question + full trajectory (up to 4000 tokens)     │
│  Input: 15 × prm_prompt_tokens                                   │
│  Output: none (reward model, forward pass only)                  │
└──────────────────────────────────────────────────────────────────┘
```

**Finalization** (`_finalize_sample`):

```python
gen_stats  = step_generator.get_sample_stats_for(sample_id)
prm_stats  = scorer.get_prm_stats_for(sample_id)

token_stats["tflops"]           = gen_tflops + prm_tflops
token_stats["prm_input_tokens"] = prm_stats["prm_input_tokens"]
token_stats["prm_tflops"]       = prm_stats["prm_tflops"]
```

### 5.2 Other Online Strategies (Best-of-N, Adaptive Scaling)

Same generator tracking, but:
- **Uncertainty scorers** (entropy, perplexity, sequence_prob): zero additional FLOPs — computed from generator logits during generation
- **PRM scorer**: same per-trajectory scoring as beam search

### 5.3 Offline Strategies (Self-Consistency, DeepConf)

Token tracking happens via `_record_generation()` per full trace. No step-level tracking.

---

## 6. Why PRM is ~10× More Expensive Than Uncertainty Scorers

Empirical observation (beam search on MATH benchmarks):

| Scorer | Total TFLOPs (3 seeds) |
|--------|------------------------|
| PRM | 642,145 |
| Entropy (MTE) | 60,593 |
| Perplexity | 61,203 |
| Sequence Prob | 72,223 |

### 6.1 Root Cause

Uncertainty scorers are **free** — they piggyback on the generator's logits computed during generation. No extra forward passes, no extra tokens. Their TFLOPs = generator TFLOPs only.

PRM runs a **separate 7B model** on every candidate trajectory at every step:

| | Generator | PRM |
|---|---|---|
| Forward passes per step | 3 (one per beam, vLLM `n=5` shares prefill) | **15** (one per candidate trajectory) |
| Tokens per forward pass | grows incrementally | up to **4000** (prm_max_tokens cap) |
| Total tokens per step | 3 × context + 15 × output | **15 × ~4000 = 60,000** |
| Model size | 8B | 7B |

### 6.2 Worked Example

Beam search with `beam_size=3`, `candidates_per_beam=5`, 30 steps:

**Generator per step** (average, mid-trajectory):
```
Input:  3 beams × 3,000 context tokens     =  9,000 tokens
Output: 3 beams × 5 candidates × 300 tokens =  4,500 tokens
Total:                                        13,500 tokens
```

**PRM per step:**
```
Input: 15 trajectories × 3,500 tokens each = 52,500 tokens
Output: 0 (reward model, no generation)
Total:                                       52,500 tokens
```

**Over 30 steps:**
```
Generator: 30 × 13,500 = 405,000 tokens → 2 × 8B × 405k / 10¹² = 6.5 TFLOPs
PRM:       30 × 52,500 = 1,575,000 tokens → 2 × 7B × 1.575M / 10¹² = 22.1 TFLOPs
Total:     6.5 + 22.1 = 28.6 TFLOPs
```

PRM contributes **~3.4× the generator cost per sample**. Across a full dataset the ratio compounds because PRM-scored beams may explore longer trajectories.

### 6.3 Why the Ratio Can Reach 10×

The above example assumes moderate trajectory lengths. The ratio increases when:

1. **Long trajectories**: PRM processes the full trajectory (up to 4000 tokens) for each candidate. Generator context tokens are counted once per beam (3×), while PRM tokens are counted for all candidates (15×).

2. **Many steps**: Non-thinking mode tends to produce shorter but more numerous steps. With 50+ steps, PRM's per-step cost of ~60k tokens adds up rapidly.

3. **Generator has efficient token accounting**: With `n=5`, the generator prefills once and decodes 5 sequences. The 5 decode sequences produce relatively few tokens each (~200-400). PRM has no such sharing — each candidate gets a full independent forward pass.

4. **PRM hits the 4000-token cap early**: Once trajectories exceed ~4000 tokens, every PRM call processes the maximum. Generator context continues growing but is shared across only 3 beams.

The net effect: PRM's token volume is 5-10× the generator's, and at 7B vs 8B parameters, the FLOP ratio is comparable to the token ratio.

---

## 7. Aggregation and Reporting

### 7.1 Per-Sample Metrics (logged to wandb)

```python
sample_metrics = {
    "tflops_this_sample": gen_tflops + prm_tflops,   # combined
    "prm_tflops_this_sample": prm_tflops,             # PRM only
    "prm_tokens_this_sample": prm_input_tokens,
    "total_tokens_this_sample": input + output,
    "running_total_tflops": cumulative,
}
```

### 7.2 Final Aggregate Metrics (saved to metrics.json)

```python
metrics = {
    "compute/total_tokens":              int,    # input + output across all samples
    "compute/total_input_tokens":        int,    # all context/prompt tokens
    "compute/total_output_tokens":       int,    # all generated tokens
    "compute/total_tflops":              float,  # gen + prm combined
    "compute/avg_tokens_per_sample":     float,
    "compute/avg_tflops_per_sample":     float,
    # PRM-specific (only present when PRM scorer is used)
    "compute/prm_input_tokens":          int,    # PRM prompt tokens only
    "compute/prm_tflops":               float,   # PRM FLOPs only
}
```

The `total_tflops` field is the primary metric for accuracy-vs-compute plots. It already includes PRM FLOPs when a PRM scorer is used.

---

## 8. vLLM Prefix Caching and FLOPs

### 8.1 What Prefix Caching Does

vLLM's prefix caching reuses KV cache blocks when multiple prompts share a common prefix. This reduces **actual GPU compute** but does **not** change the reported FLOP count.

- **Generator**: prefix caching is enabled by default (`enable_prefix_caching=True`)
- **PRM**: prefix caching is enabled (`enable_prefix_caching=True`)

### 8.2 Why FLOPs Don't Account for Caching

Token counts are measured **before** vLLM processes them:

```python
# Generator
per_prompt_context_tokens = [len(self.tokenizer.encode(p)) for p in prompts]

# PRM
num_prompt_tokens = len(self.prm_tokenizer.encode(prompt))
```

This counts **theoretical tokens** regardless of cache hits. This is by design:

1. **Reproducibility**: FLOPs are hardware-independent and don't depend on runtime caching behavior
2. **Comparability**: Papers report theoretical FLOPs as the standard compute metric
3. **Consistency**: The same configuration reports the same FLOPs regardless of GPU memory or batch ordering

### 8.3 Where Caching Helps at Runtime

| Scenario | Shared prefix | Benefit |
|----------|--------------|---------|
| Beam search: 3 beams, same question | system + question (~500 tokens) | 2/3 of prefix compute saved |
| PRM: 5 candidates from same beam | question + parent trajectory (~3000 tokens) | 4/5 of prefix compute saved |
| PRM: across steps | growing trajectory prefix | incremental prefill only |

This makes wall-clock time significantly lower than what FLOPs suggest, especially for PRM.

---

## 9. Known Limitations

### 9.1 Answer Step Tokens Not in Per-Sample Stats

`_generate_beam_answers()` calls `generate_answer_candidates_batch()` without `sample_ids`, so answer step tokens are recorded in aggregate stats but **not** in per-sample `_per_sample_stats`. This causes a minor undercount (~1-5%) in per-sample TFLOPs.

### 9.2 Prefix Caching Overcount

When multiple beams share a common prefix, the generator counts full context tokens for each beam. With prefix caching, the shared prefix is computed once. This causes a minor overcount (~2-3%) that grows with beam_size.

### 9.3 Simple vs Precise Method

The default `simple` method ignores attention costs that scale with sequence length. For very long sequences (>32k tokens), the precise method would be more accurate. Currently all experiments use the simple method for consistency.

---

## 10. Source Files

| File | Role |
|------|------|
| [`llm_tts/utils/flops.py`](../../llm_tts/utils/flops.py) | `FLOPCalculator` class, formulas, model architectures |
| [`llm_tts/generators/base.py`](../../llm_tts/generators/base.py) | `_per_sample_stats`, `record_sample_tokens()`, `get_sample_stats_for()` |
| [`llm_tts/generators/vllm.py`](../../llm_tts/generators/vllm.py) | Token counting in `_generate_step_candidates_impl()` |
| [`llm_tts/scorers/step_scorer_prm.py`](../../llm_tts/scorers/step_scorer_prm.py) | PRM token tracking, `_record_prm_tokens()`, `get_prm_stats_for()` |
| [`llm_tts/strategies/strategy_beam_search.py`](../../llm_tts/strategies/strategy_beam_search.py) | `_finalize_sample()`: merges gen + PRM stats |
| [`scripts/run_tts_eval.py`](../../scripts/run_tts_eval.py) | Final aggregation, metrics.json output |

---

## References

- [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) — Scaling Laws for Neural Language Models
- [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) — Training Compute-Optimal LLMs (Chinchilla)
- [Chowdhery et al. (2022)](https://arxiv.org/abs/2204.02311) — PaLM, Appendix B (FLOP counting)
- [Dao et al. (2022)](https://arxiv.org/abs/2205.14135) — FlashAttention
- [Kipply (2022)](https://kipp.ly/transformer-inference-arithmetic/) — Transformer Inference Arithmetic
