# SLURM Experiment Guide

This document provides a comprehensive reference for all experiments that can be run using the unified `submit.sh` script.

## Quick Reference

```bash
./submit.sh --strategy <strategy> --dataset <dataset> [OPTIONS]
```

### Required Arguments

| Flag | Description | Values |
|------|-------------|--------|
| `--strategy` | Experiment strategy | `baseline`, `self_consistency`, `offline_bon`, `online_bon`, `beam_search`, `adaptive_scaling` |
| `--dataset` | Dataset to use | `aime2024`, `aime2025`, `math500`, `olympiadbench`, `gaokao2023en`, `minerva_math`, `gpqa_diamond` |

### Optional Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model variant | `qwen25_7b` |
| `--scorers` | Scorer(s) to use (comma-separated) | `none` (for baseline) |
| `--seeds` | Number of seeds to run | `1` |
| `--seed` | Specific seed value | `42` |
| `--mode` | Execution mode | `parallel` |
| `--sequential` | Alias for `--mode sequential` | - |
| `--gpus` | Number of GPUs | `auto` (1 for non-PRM, 2 for PRM) |
| `--time` | Time limit (HH:MM:SS) | `auto` (4h/24h/72h based on strategy) |
| `--dry-run` | Preview without submitting | - |
| `--help` | Show help message | - |

### Model Options

| Value | Description |
|-------|-------------|
| `qwen25_7b` | Qwen 2.5 7B (default) |
| `qwen3_8b_thinking` | Qwen 3 8B thinking mode |
| `qwen3_8b` | Qwen 3 8B |
| `qwen25_math_7b` | Qwen 2.5 Math 7B Instruct |
| `qwen25_math_15b` | Qwen 2.5 Math 15B Instruct |

### Scorer Options

| Value | Description | GPUs |
|-------|-------------|------|
| `entropy` | Entropy-based scoring | 1 |
| `perplexity` | Perplexity-based scoring | 1 |
| `sequence_prob` | Sequence probability scoring | 1 |
| `prm` | Process Reward Model | 2 |
| `all` | Run all scorers | varies (max across scorers) |

### Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `parallel` | Array jobs run simultaneously | Faster execution, uses more job slots |
| `sequential` | For loop runs one after another | Slower, saves job slots |

---

## Baseline Experiments

### AIME 2024

| Strategy | Seeds | Command |
|----------|-------|---------|
| baseline | - | `./submit.sh --strategy baseline --dataset aime2024 --model qwen3_8b_thinking` |
| baseline | 4 | `./submit.sh --strategy baseline --dataset aime2024 --model qwen3_8b_thinking --seeds 4` |

### AIME 2025

| Strategy | Seeds | Command |
|----------|-------|---------|
| baseline | - | `./submit.sh --strategy baseline --dataset aime2025 --model qwen3_8b_thinking` |
| baseline | 6 | `./submit.sh --strategy baseline --dataset aime2025 --model qwen3_8b_thinking --seeds 6` |

### Other Datasets

| Dataset | Strategy | Command |
|---------|----------|---------|
| math500 | baseline | `./submit.sh --strategy baseline --dataset math500` |
| olympiadbench | baseline | `./submit.sh --strategy baseline --dataset olympiadbench --model qwen25_math_7b` |
| gpqa_diamond | baseline | `./submit.sh --strategy baseline --dataset gpqa_diamond --model qwen3_8b` |

---

## Self-Consistency Experiments

| Dataset | Strategy | Command |
|---------|----------|---------|
| aime2024 | self_consistency | `./submit.sh --strategy self_consistency --dataset aime2024` |
| aime2025 | self_consistency | `./submit.sh --strategy self_consistency --dataset aime2025` |
| math500 | self_consistency | `./submit.sh --strategy self_consistency --dataset math500` |
| olympiadbench | self_consistency | `./submit.sh --strategy self_consistency --dataset olympiadbench` |
| gpqa_diamond | self_consistency | `./submit.sh --strategy self_consistency --dataset gpqa_diamond` |

---

## Offline Best-of-N Experiments

### AIME 2024

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy offline_bon --dataset aime2024 --scorers entropy` |
| perplexity | `./submit.sh --strategy offline_bon --dataset aime2024 --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy offline_bon --dataset aime2024 --scorers sequence_prob` |
| all (sequential) | `./submit.sh --strategy offline_bon --dataset aime2024 --scorers all --mode sequential` |

### AIME 2025

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy offline_bon --dataset aime2025 --scorers entropy` |
| perplexity | `./submit.sh --strategy offline_bon --dataset aime2025 --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy offline_bon --dataset aime2025 --scorers sequence_prob` |
| all (sequential) | `./submit.sh --strategy offline_bon --dataset aime2025 --scorers all --mode sequential` |

### Math500

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy offline_bon --dataset math500 --scorers entropy` |
| perplexity | `./submit.sh --strategy offline_bon --dataset math500 --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy offline_bon --dataset math500 --scorers sequence_prob` |
| prm | `./submit.sh --strategy offline_bon --dataset math500 --scorers prm` |
| all (parallel) | `./submit.sh --strategy offline_bon --dataset math500 --scorers all --mode parallel` |
| all (sequential) | `./submit.sh --strategy offline_bon --dataset math500 --scorers all --mode sequential` |

### OlympiadBench

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy offline_bon --dataset olympiadbench --scorers entropy` |
| perplexity | `./submit.sh --strategy offline_bon --dataset olympiadbench --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy offline_bon --dataset olympiadbench --scorers sequence_prob` |
| all (sequential) | `./submit.sh --strategy offline_bon --dataset olympiadbench --scorers all --mode sequential` |

### GPQA Diamond

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy offline_bon --dataset gpqa_diamond --scorers entropy` |
| perplexity | `./submit.sh --strategy offline_bon --dataset gpqa_diamond --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy offline_bon --dataset gpqa_diamond --scorers sequence_prob` |

### Gaokao2023EN

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy offline_bon --dataset gaokao2023en --scorers entropy` |
| perplexity | `./submit.sh --strategy offline_bon --dataset gaokao2023en --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy offline_bon --dataset gaokao2023en --scorers sequence_prob` |
| all (sequential) | `./submit.sh --strategy offline_bon --dataset gaokao2023en --scorers all --mode sequential` |

### Minerva Math

| Scorer | Command |
|--------|---------|
| prm | `./submit.sh --strategy offline_bon --dataset minerva_math --scorers prm` |

---

## Online Best-of-N Experiments

### Math500

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy online_bon --dataset math500 --scorers entropy` |
| perplexity | `./submit.sh --strategy online_bon --dataset math500 --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy online_bon --dataset math500 --scorers sequence_prob` |
| prm | `./submit.sh --strategy online_bon --dataset math500 --scorers prm` |
| all (sequential) | `./submit.sh --strategy online_bon --dataset math500 --scorers all --mode sequential` |

### OlympiadBench

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy online_bon --dataset olympiadbench --scorers entropy` |
| perplexity | `./submit.sh --strategy online_bon --dataset olympiadbench --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy online_bon --dataset olympiadbench --scorers sequence_prob` |
| prm | `./submit.sh --strategy online_bon --dataset olympiadbench --scorers prm` |
| all (sequential) | `./submit.sh --strategy online_bon --dataset olympiadbench --scorers all --mode sequential` |

### Gaokao2023EN

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy online_bon --dataset gaokao2023en --scorers entropy` |
| perplexity | `./submit.sh --strategy online_bon --dataset gaokao2023en --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy online_bon --dataset gaokao2023en --scorers sequence_prob` |
| all (sequential) | `./submit.sh --strategy online_bon --dataset gaokao2023en --scorers all --mode sequential` |

---

## Beam Search Experiments

### Math500

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy beam_search --dataset math500 --scorers entropy` |
| perplexity | `./submit.sh --strategy beam_search --dataset math500 --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy beam_search --dataset math500 --scorers sequence_prob` |
| prm | `./submit.sh --strategy beam_search --dataset math500 --scorers prm` |
| all (sequential) | `./submit.sh --strategy beam_search --dataset math500 --scorers all --mode sequential` |

### OlympiadBench

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy beam_search --dataset olympiadbench --scorers entropy` |
| perplexity | `./submit.sh --strategy beam_search --dataset olympiadbench --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy beam_search --dataset olympiadbench --scorers sequence_prob` |
| all (sequential) | `./submit.sh --strategy beam_search --dataset olympiadbench --scorers all --mode sequential` |

### Gaokao2023EN

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy beam_search --dataset gaokao2023en --scorers entropy` |
| sequence_prob | `./submit.sh --strategy beam_search --dataset gaokao2023en --scorers sequence_prob` |
| all (sequential) | `./submit.sh --strategy beam_search --dataset gaokao2023en --scorers all --mode sequential` |

### Minerva Math

| Scorer | Command |
|--------|---------|
| prm | `./submit.sh --strategy beam_search --dataset minerva_math --scorers prm` |

---

## Adaptive Scaling Experiments

### Math500

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy adaptive_scaling --dataset math500 --scorers entropy` |
| perplexity | `./submit.sh --strategy adaptive_scaling --dataset math500 --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy adaptive_scaling --dataset math500 --scorers sequence_prob` |
| prm | `./submit.sh --strategy adaptive_scaling --dataset math500 --scorers prm` |
| all (sequential) | `./submit.sh --strategy adaptive_scaling --dataset math500 --scorers all --mode sequential` |

### OlympiadBench

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy adaptive_scaling --dataset olympiadbench --scorers entropy` |
| perplexity | `./submit.sh --strategy adaptive_scaling --dataset olympiadbench --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy adaptive_scaling --dataset olympiadbench --scorers sequence_prob` |
| prm | `./submit.sh --strategy adaptive_scaling --dataset olympiadbench --scorers prm` |
| all (sequential) | `./submit.sh --strategy adaptive_scaling --dataset olympiadbench --scorers all --mode sequential` |

### Gaokao2023EN

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy adaptive_scaling --dataset gaokao2023en --scorers entropy` |
| perplexity | `./submit.sh --strategy adaptive_scaling --dataset gaokao2023en --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy adaptive_scaling --dataset gaokao2023en --scorers sequence_prob` |
| all (sequential) | `./submit.sh --strategy adaptive_scaling --dataset gaokao2023en --scorers all --mode sequential` |

### Minerva Math

| Scorer | Command |
|--------|---------|
| entropy | `./submit.sh --strategy adaptive_scaling --dataset minerva_math --scorers entropy` |
| perplexity | `./submit.sh --strategy adaptive_scaling --dataset minerva_math --scorers perplexity` |
| sequence_prob | `./submit.sh --strategy adaptive_scaling --dataset minerva_math --scorers sequence_prob` |

---

## Resource Defaults

### GPU Allocation
- **Non-PRM scorers**: 1 GPU
- **PRM scorer**: 2 GPUs
- **Sequential with PRM**: 2 GPUs (max across scorers)

### Time Limits
- **Baseline / Self-consistency**: 4 hours
- **Offline BoN / Online BoN / Beam search**: 24 hours
- **Adaptive scaling**: 72 hours
- **Sequential mode**: Time per experiment × number of experiments

### Model Variants
- `qwen25_7b` (default)
- `qwen3_8b_thinking`
- `qwen3_8b`
- `qwen25_math_7b`
- `qwen25_math_15b`

---

## Common Patterns

### Running Multiple Scorers
```bash
# Parallel mode (faster, uses more job slots)
./submit.sh --strategy offline_bon --dataset math500 --scorers entropy,perplexity,prm

# Sequential mode (slower, saves job slots)
./submit.sh --strategy offline_bon --dataset math500 --scorers all --mode sequential
```

### Running with Multiple Seeds
```bash
# Array jobs (parallel execution)
./submit.sh --strategy baseline --dataset aime2024 --seeds 4

# Separate jobs (sequential execution)
./submit.sh --strategy baseline --dataset aime2024 --seeds 4 --mode sequential
```

### Custom Resources
```bash
# More GPUs
./submit.sh --strategy beam_search --dataset math500 --scorers prm --gpus 4

# More time
./submit.sh --strategy online_bon --dataset olympiadbench --scorers prm --time "48:00:00"
```

---

## Testing

Always use `--dry-run` to preview commands before submitting:

```bash
./submit.sh --strategy offline_bon --dataset math500 --scorers all --dry-run
```

This will show:
- Number of jobs to be submitted
- Config paths
- GPU allocation
- Time limits
- Full SLURM script contents
