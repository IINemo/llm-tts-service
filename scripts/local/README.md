# Local Job Scheduling with Task Spooler

A local alternative to SLURM for running experiments on shared GPU machines.

## Installation

```bash
apt update && apt install task-spooler
```

## Usage

Same interface as `scripts/slurm/submit.sh`:

```bash
./scripts/local/submit.sh --strategy <strategy> --dataset <dataset> [OPTIONS]
```

### Examples

```bash
# Baseline
./scripts/local/submit.sh --strategy baseline --dataset amc23 --model qwen25_math_7b

# Offline BoN with all scorers (queued, run one at a time)
./scripts/local/submit.sh --strategy offline_bon --dataset math500 --scorers all

# Specific GPU
./scripts/local/submit.sh --strategy beam_search --dataset olympiadbench --scorers prm --gpu 0,1

# Custom timeout (seconds)
./scripts/local/submit.sh --strategy adaptive_scaling --dataset gaokao2023en --scorers entropy --timeout 100000

# Preview without submitting
./scripts/local/submit.sh --strategy online_bon --dataset math500 --scorers entropy --dry-run
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--strategy` | Strategy name | (required) |
| `--dataset` | Dataset name | (required) |
| `--model` | Model variant | `qwen25_7b` |
| `--scorers` | Scorer(s) | none (baseline) |
| `--seed` | Random seed | `42` |
| `--gpu` | GPU id(s) | `0` (or `0,1` for PRM) |
| `--timeout` | Timeout in seconds | auto (4h/24h/72h) |
| `--label` | Custom job label | auto |
| `--dry-run` | Preview only | - |

## Managing Jobs

```bash
tsp              # Show job queue
tsp -c <id>      # Show stdout of job
tsp -i <id>      # Show job info
tsp -k <id>      # Kill running job
tsp -r <id>      # Remove queued job
tsp -S <n>       # Set max simultaneous jobs (default: 1)
```

## Key Differences from SLURM

- Jobs queue locally via `tsp` instead of `sbatch`
- Default: 1 job at a time (use `tsp -S 2` for parallel)
- No array jobs — multiple scorers are queued as separate jobs
- GPU assignment via `CUDA_VISIBLE_DEVICES` (not SLURM `--gres`)
- Timeout via `timeout` command (not SLURM `-t`)

## References

- [Task Spooler (shared)](https://github.com/bstee615/shared-task-spooler)
- [Task Spooler (GPU-aware)](https://github.com/justanhduc/task-spooler)
