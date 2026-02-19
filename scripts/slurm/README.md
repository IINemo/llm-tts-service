# SLURM Experiment Submission Scripts

This directory contains scripts for submitting machine learning experiments to a SLURM cluster.

## Quick Start

The unified `submit.sh` script replaces all individual experiment scripts.

```bash
./submit.sh --strategy <strategy> --dataset <dataset> [OPTIONS]
```

### Basic Examples

```bash
# Single baseline run
./submit.sh --strategy baseline --dataset aime2024

# Baseline with multiple seeds (array job)
./submit.sh --strategy baseline --dataset aime2024 --seeds 4

# Offline BoN with specific scorer
./submit.sh --strategy offline_bon --dataset math500 --scorers entropy

# Offline BoN with all scorers (sequential to save job slots)
./submit.sh --strategy offline_bon --dataset math500 --scorers all --mode sequential

# Online BoN with PRM scorer
./submit.sh --strategy online_bon --dataset olympiadbench --scorers prm

# Preview without submitting
./submit.sh --strategy offline_bon --dataset math500 --scorers all --dry-run
```

## Available Options

### Required Arguments

| Argument | Description | Values |
|----------|-------------|--------|
| `--strategy` | Experiment strategy | `baseline`, `self_consistency`, `offline_bon`, `online_bon`, `beam_search`, `adaptive_scaling` |
| `--dataset` | Dataset to use | `aime2024`, `aime2025`, `math500`, `olympiadbench`, `gaokao2023en`, `minerva_math`, `gpqa_diamond` |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model variant | `qwen25_7b` |
| `--scorers` | Scorer(s) to use | `none` (for baseline) |
| `--seeds` | Number of seeds to run | `1` |
| `--seed` | Specific seed value | `42` |
| `--mode` | Execution mode | `parallel` |
| `--gpus` | Number of GPUs | `auto` |
| `--time` | Time limit | `auto` |
| `--dry-run` | Preview without submitting | - |

### Model Options

- `qwen25_7b` - Qwen 2.5 7B (default)
- `qwen3_8b_thinking` - Qwen 3 8B thinking mode
- `qwen3_8b` - Qwen 3 8B
- `qwen25_math_7b` - Qwen 2.5 Math 7B Instruct
- `qwen25_math_15b` - Qwen 2.5 Math 15B Instruct

### Scorer Options

- `all` - Run all scorers (entropy, perplexity, sequence_prob, prm)
- `prm` - Process Reward Model (requires 2 GPUs)
- `entropy` - Entropy-based scoring
- `perplexity` - Perplexity-based scoring
- `sequence_prob` - Sequence probability scoring
- Multiple scorers: `prm,entropy,perplexity`

### Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `parallel` | Array jobs run simultaneously | Faster execution, more job slots |
| `sequential` | For loop runs one after another | Save job slots, single GPU |

## Automatic Resource Allocation

### GPU Count
- **Non-PRM scorers**: 1 GPU
- **PRM scorer**: 2 GPUs
- **Sequential with mixed scorers**: Max GPU count (2 if PRM included)

### Time Limits
- **Baseline / Self-consistency**: 4 hours
- **Offline BoN / Online BoN / Beam search**: 24 hours
- **Adaptive scaling**: 72 hours
- **Sequential mode**: Time × number of experiments

## Examples by Strategy

### Baseline
```bash
# Single run
./submit.sh --strategy baseline --dataset aime2024 --model qwen3_8b_thinking

# Multiple seeds
./submit.sh --strategy baseline --dataset aime2024 --seeds 4
```

### Self-Consistency
```bash
./submit.sh --strategy self_consistency --dataset math500
```

### Offline Best-of-N
```bash
# Single scorer
./submit.sh --strategy offline_bon --dataset math500 --scorers entropy

# Multiple scorers (parallel)
./submit.sh --strategy offline_bon --dataset math500 --scorers prm,entropy

# All scorers (sequential)
./submit.sh --strategy offline_bon --dataset math500 --scorers all --mode sequential
```

### Online Best-of-N
```bash
# With PRM scorer
./submit.sh --strategy online_bon --dataset olympiadbench --scorers prm

# Multiple scorers with custom time
./submit.sh --strategy online_bon --dataset math500 --scorers all --time "48:00:00"
```

### Beam Search
```bash
./submit.sh --strategy beam_search --dataset gaokao2023en --scorers sequence_prob
```

### Adaptive Scaling
```bash
# All scorers sequentially (saves job slots)
./submit.sh --strategy adaptive_scaling --dataset math500 --scorers all --mode sequential
```

## Output Files

Jobs create output and error logs in the `logs/` directory:

```
logs/<strategy>_<scorer>_<dataset>_<seed>_<job_id>.out
logs/<strategy>_<scorer>_<dataset>_<seed>_<job_id>.err
```

For array jobs:
```
logs/<strategy>_<scorer>_<dataset>_<array_id>_<job_id>.out
logs/<strategy>_<scorer>_<dataset>_<array_id>_<job_id>.err
```

---

## Manual Sequential Job Submission

For running multiple seeds sequentially in a single SLURM job, use this template:

```bash
#!/bin/bash
#SBATCH --job-name=EXPERIMENT_NAME
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/EXPERIMENT_NAME_%j.log
#SBATCH --error=logs/EXPERIMENT_NAME_%j.err
#SBATCH --chdir=/home/artem.shelmanov/vlad/llm-tts-service

set -e

module load rocm 2>/dev/null || true
source ~/.bashrc
conda activate lm-polygraph-env

echo "============================================"
echo "EXPERIMENT_NAME - 3 Seeds Sequential"
echo "Start: $(date)"
echo "============================================"

for SEED in 42 43 44; do
    echo ""
    echo "============================================"
    echo "Seed $SEED starting at $(date)"
    echo "============================================"

    python scripts/run_tts_eval.py \
        --config-path=/home/artem.shelmanov/vlad/llm-tts-service/config \
        --config-name=CONFIG_NAME_HERE \
        system.seed=$SEED

    echo "Seed $SEED completed at $(date)"
done

echo ""
echo "============================================"
echo "All 3 seeds done at $(date)"
echo "============================================"
```

**Key parameters to modify:**
| Parameter | Description | Examples |
|-----------|-------------|----------|
| `--job-name` | Short identifier | `aime_entropy_min` |
| `--time` | Time limit | `24:00:00` (most), `72:00:00` (adaptive_scaling) |
| `--gpus-per-node` | GPU count | `1` (non-PRM), `2` (PRM) |
| `CONFIG_NAME_HERE` | Full config path | `experiments/beam_search/aime2024/...entropy_min` |
| `SEED in 42 43 44` | Seeds to run | Any space-separated list |

**Note:** `--chdir` is critical - it sets the working directory so Hydra finds the config correctly.

---

## Migration from Old Scripts

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for a complete mapping of old scripts to new `submit.sh` commands.

## Help

For complete usage information:
```bash
./submit.sh --help
```
