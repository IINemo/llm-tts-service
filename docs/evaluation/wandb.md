# WandB Logging Guide

This document describes conventions for logging experiment results to Weights & Biases (WandB).

## Online Logging (Recommended)

Enable live WandB logging by setting `report_to: wandb` in your experiment config:

```yaml
# In your experiment config (e.g., sc_vllm_qwen3_aime2025.yaml)
report_to: wandb
wandb_project: llm-tts-eval-aime2025
```

Then run:

```bash
python scripts/run_tts_eval.py \
    --config-name=experiments/self_consistency/sc_vllm_qwen3_aime2025 \
    --config-path=../config
```

This provides:
- Real-time accuracy tracking per sample
- Live metrics: `running_accuracy`, `running_correct`, `samples_completed`
- TFLOP usage tracking: `running_total_tflops`, `tflops_this_sample`
- Token statistics: `tokens_this_sample`, `avg_tokens_per_trace`
- Config and log files automatically saved as artifacts

See [metrics.md](metrics.md) for detailed metric definitions and formulas.

Results are also saved locally to `outputs/YYYY-MM-DD/run_name/`.

## Offline Logging (Alternative)

For environments without network access, run experiments locally then upload:

### Step 1: Run Experiments Locally

```bash
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_vllm_qwen3_aime2025
```

### Step 2: Upload to WandB

```bash
python scripts/log_results_to_wandb.py \
    outputs/2025-12-03/aime2025_deepconf_vllm_s0-29 \
    --project llm-tts-eval-aime2025 \
    --name "deepconf_qwen3-8b_vllm" \
    --tags deepconf qwen3-8b vllm
```

This uploads:
- `results.json` - Per-sample results
- `run_tts_eval.log` - Full experiment log
- `.hydra/config.yaml` - Experiment configuration
- All other files in the output directory

---

## Naming Conventions

### Project Names

Use the pattern: `llm-tts-eval-{dataset}`

| Dataset | Project Name |
|---------|-------------|
| AIME 2025 | `llm-tts-eval-aime2025` |
| GSM8K | `llm-tts-eval-gsm8k` |
| MATH-500 | `llm-tts-eval-math500` |

This keeps all experiments for a dataset in one project for easy comparison.

### Run Names

Use the pattern: `{strategy}_{model}_{backend}[_suffix]`

Examples:
- `deepconf_qwen3-8b_vllm`
- `deepconf_qwen3-32b_openai`
- `self-consistency_llama3-8b_hf`
- `baseline_gpt-4o_openai`

For partial runs (subset of samples):
- `deepconf_qwen3-8b_vllm_s0-12` (samples 0-12)
- `deepconf_qwen3-8b_vllm_s13-29` (samples 13-29)

### Tags

Use consistent tags for filtering runs:

| Category | Tags |
|----------|------|
| Strategy | `deepconf`, `self-consistency`, `baseline`, `mur`, `tot` |
| Model | `qwen3-8b`, `qwen3-32b`, `llama3-8b`, `gpt-4o` |
| Backend | `vllm`, `openai`, `hf` (huggingface) |
| Dataset | `aime2025`, `gsm8k`, `math500` |
| Config | `temp0.6`, `top_k16`, `threshold-10` |

---

## Example Commands

### Upload single experiment
```bash
python scripts/log_results_to_wandb.py \
    outputs/2025-12-03/aime2025_deepconf_vllm \
    --project llm-tts-eval-aime2025 \
    --name "deepconf_qwen3-8b_vllm" \
    --tags deepconf qwen3-8b vllm
```

### Upload multiple experiments (batch)
```bash
for dir in outputs/2025-12-03/*/; do
    python scripts/log_results_to_wandb.py "$dir" \
        --project llm-tts-eval-aime2025 \
        --tags deepconf qwen3-8b vllm
done
```

---

## Artifacts

Each upload creates a WandB artifact containing the entire output directory. Artifacts are versioned and can be downloaded later:

```python
import wandb
run = wandb.init()
artifact = run.use_artifact('nlpresearch.group/llm-tts-eval-aime2025/outputs-2025-12-03-run_name:latest')
artifact_dir = artifact.download()
```

---

## See Also

- [Results documentation](results/README.md) - How to document results
- [Metrics](metrics.md) - What metrics to track
- Upload script: [`scripts/log_results_to_wandb.py`](../../scripts/log_results_to_wandb.py)
