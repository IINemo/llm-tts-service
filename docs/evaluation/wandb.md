# WandB Logging Guide

This document describes conventions for logging experiment results to Weights & Biases (WandB).

## Recommended Workflow

We recommend **offline-first logging**: run experiments locally without real-time WandB tracking, then upload complete results afterwards. This approach:

- Avoids network issues interrupting experiments
- Allows reviewing results before uploading
- Reduces overhead during long-running experiments
- Enables batch uploading of multiple runs

### Step 1: Run Experiments Locally

Run experiments with default settings (no live WandB tracking):

```bash
python scripts/run_tts_eval.py \
    --config-name=experiments/deepconf/deepconf_vllm_qwen3_aime2025
```

Results are saved to `outputs/YYYY-MM-DD/run_name/`.

### Step 2: Upload to WandB

After experiments complete, upload entire output folders using:

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
