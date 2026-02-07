# Online Best-of-N Strategy

Online Best-of-N (BoN) is a test-time compute strategy that generates multiple candidate continuations at each step and selects the best one based on a scoring criterion (e.g., entropy, confidence).

## Algorithm Overview

```
1. Initialize empty trajectory
2. For each step (up to max_steps):
   a. Generate N candidates for next step
   b. Score each candidate using scorer
   c. Select best candidate
   d. Append to trajectory
   e. Check if trajectory is complete
3. Return final trajectory
```

## Token Usage Analysis

### Understanding Token Costs

Online BoN has **quadratic token growth** with respect to the number of steps. This is because:

1. Each step generates `N` candidates (typically 4)
2. Each candidate generation includes the full context (prompt + all previous steps)
3. As the trajectory grows, the context grows linearly
4. Total input tokens = `N × Σ(context_at_step_i)` for i=0 to num_steps

### Token Breakdown

For each sample, we track:

| Metric | Description |
|--------|-------------|
| `total_tokens_this_sample` | Sum of input + output tokens |
| `input_tokens` | Context tokens fed to model (prompt + trajectory) across all generations |
| `output_tokens` | Actually generated tokens (new content) |
| `generation_count` | Number of generation calls = `num_steps × candidates_per_step` |

### Example Analysis

From AIME 2025 experiments with `candidates_per_step=4`:

| Sample | Total Tokens | Input | Output | Generations | Input/Gen | Out/Gen |
|--------|--------------|-------|--------|-------------|-----------|---------|
| 0 | 3,080,764 | 2,967,824 (96%) | 112,940 (4%) | 251 | 11,824 | 450 |
| 1 | 134,933 | 116,727 (86%) | 18,206 (14%) | 55 | 2,122 | 331 |
| 6 | 28,260 | 19,996 (71%) | 8,264 (29%) | 21 | 952 | 394 |

**Key Observations:**

1. **Input tokens dominate** (70-96% of total) - most compute goes to processing context
2. **Samples with more steps are exponentially more expensive** - 250 generations costs ~100x more than 20 generations
3. **Output tokens are relatively constant** per generation (~300-500 tokens)
4. **Input tokens per generation grow linearly** with trajectory length

### Quadratic Growth Explanation

```
Step 0: 4 × context_0                    (initial prompt only)
Step 1: 4 × (context_0 + step_0)         (prompt + 1 step)
Step 2: 4 × (context_0 + step_0 + step_1) (prompt + 2 steps)
...
Step N: 4 × (context_0 + Σ steps)        (prompt + N steps)

Total input = 4 × Σ(i=0 to N) context_i
            = O(N²) where N = num_steps
```

For a trajectory with 63 steps (251 generations with 4 candidates):
- Average step length: ~450 tokens
- Context at step 63: ~28,000 tokens
- Total input: ~3M tokens

### Optimization Strategies

1. **Reduce `candidates_per_step`** (e.g., 4 → 2): Linear reduction in tokens
2. **Reduce `max_steps`**: Quadratic reduction in tokens
3. **Enable prefix caching** (`enable_prefix_caching: true`): Reduces actual compute by caching shared prefixes
4. **Use smaller models**: Faster inference, less memory pressure

## Configuration

```yaml
strategy:
  type: online_best_of_n
  candidates_per_step: 4    # N candidates per step
  max_steps: 250            # Maximum steps before forcing answer

  # Step boundary detection
  detector_type: thinking_marker
  min_step_tokens: 50
  max_step_tokens: 300

  # Marker categories for step detection
  use_sequence: true
  use_conclusion: true
  use_thinking: true
  use_verification: true
```

## Logged Metrics (WandB)

Per-sample metrics logged to WandB:

| Metric | Description |
|--------|-------------|
| `total_tokens_this_sample` | Total tokens (input + output) for this sample |
| `input_tokens_this_sample` | Context tokens processed |
| `output_tokens_this_sample` | Generated tokens |
| `generations_this_sample` | Number of generation calls |
| `reasoning_steps` | Number of reasoning steps |
| `tflops_this_sample` | Estimated TFLOPs for this sample |

Running totals:

| Metric | Description |
|--------|-------------|
| `running_total_tokens` | Cumulative tokens across all samples |
| `running_total_input_tokens` | Cumulative input tokens |
| `running_total_output_tokens` | Cumulative output tokens |
| `running_total_generations` | Cumulative generation calls |
| `running_total_tflops` | Cumulative TFLOPs |

## Console Logging Format

```
Sample token stats: total_tokens=3,080,764, input_tokens=2,967,824, output_tokens=112,940, generations=251, tflops=48491.762
```

## Comparison with Offline BoN

| Aspect | Online BoN | Offline BoN |
|--------|------------|-------------|
| Token growth | O(N²) per sample | O(N×K) per sample |
| Adaptivity | Adapts at each step | Fixed trajectories |
| Selection | Greedy (local) | Global best |
| Early stopping | Yes | No |

Where N = steps, K = trajectories in offline BoN.
