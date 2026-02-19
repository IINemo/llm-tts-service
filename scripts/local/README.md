# Local Job Scheduling with Task Spooler

A local alternative to SLURM for running experiments on shared GPU machines.

## Installation

```bash
apt update && apt install task-spooler
```

Or build from source (GPU-aware version):
```bash
git clone https://github.com/justanhduc/task-spooler
cd task-spooler && make && cp ts ~/.local/bin/tsp
```

## Usage

Same interface as `scripts/slurm/submit.sh`:

```bash
./scripts/local/submit.sh --strategy <strategy> --dataset <dataset> [OPTIONS]
```

### Examples

```bash
# Baseline on AMC23
./scripts/local/submit.sh --strategy baseline --dataset amc23 --model qwen25_math_7b

# Beam search on OlympiadBench with all 4 scorers
./scripts/local/submit.sh --strategy beam_search --dataset olympiadbench --scorers all

# Offline BoN with specific scorers
./scripts/local/submit.sh --strategy offline_bon --dataset math500 --scorers entropy,perplexity

# Multiple seeds (queues 4 jobs with seeds 42,43,44,45)
./scripts/local/submit.sh --strategy baseline --dataset aime2024 --model qwen3_8b_thinking --seeds 4

# Force specific GPU
./scripts/local/submit.sh --strategy beam_search --dataset olympiadbench --scorers prm --gpu 0,1

# Custom timeout (seconds)
./scripts/local/submit.sh --strategy adaptive_scaling --dataset gaokao2023en --scorers entropy --timeout 100000

# Preview without submitting
./scripts/local/submit.sh --strategy online_bon --dataset math500 --scorers entropy --dry-run
```

### GPU-Aware Scheduling

The script uses tsp's native GPU management (`-G` flag). It auto-detects available GPUs, sets `TS_VISIBLE_DEVICES`, and declares how many GPUs each job needs. tsp checks actual GPU memory before assigning — a GPU is considered free when at least 90% of its memory is available.

On a 2-GPU machine with `--scorers all`:

```
$ ./scripts/local/submit.sh --strategy beam_search --dataset olympiadbench --scorers all

Queued job 0: beam_search_olympiadbench_entropy       (seed=42, gpus=1, timeout=86400s)
Queued job 1: beam_search_olympiadbench_perplexity    (seed=42, gpus=1, timeout=86400s)
Queued job 2: beam_search_olympiadbench_sequence_prob  (seed=42, gpus=1, timeout=86400s)
Queued job 3: beam_search_olympiadbench_prm           (seed=42, gpus=2, timeout=86400s)

Parallel slots: 2 (one per GPU). Run 'tsp' to view queue status.
```

- Jobs 0 and 1 run in parallel on separate GPUs (tsp assigns automatically)
- Job 2 starts when a GPU frees up
- Job 3 (PRM, 2 GPUs) waits until both GPUs are free

You can adjust the free memory threshold:
```bash
tsp --set_gpu_free_perc 80   # default is 90
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--strategy` | Strategy name | (required) |
| `--dataset` | Dataset name | (required) |
| `--model` | Model variant | `qwen25_7b` |
| `--scorers` | Scorer(s): `all`, or comma-separated | none (baseline) |
| `--seed` | Starting random seed | `42` |
| `--seeds` | Number of seeds to run (42, 43, ...) | `1` |
| `--gpu` | Override GPU id(s) (disables auto) | auto via tsp |
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
tsp -C           # Clear finished jobs
tsp -S <n>       # Set max simultaneous jobs
tsp -K           # Kill tsp server (clears all jobs)
```

---

## Advanced: Manual Sequential Chains

For precise control over job ordering (e.g., sequential seeds on specific GPUs), use tsp's dependency flag `-D`:

```bash
# Set parallel slots (1 per GPU)
tsp -S 2

# GPU 0: seed 42 -> 43 -> 44 (sequential)
J0=$(tsp -L label_s42 -g 0 timeout 86400 bash /tmp/wrapper_s42.sh)
J1=$(tsp -L label_s43 -g 0 -D $J0 timeout 86400 bash /tmp/wrapper_s43.sh)
J2=$(tsp -L label_s44 -g 0 -D $J1 timeout 86400 bash /tmp/wrapper_s44.sh)

# GPU 1: different experiment, seed 42 -> 43 -> 44 (sequential)
K0=$(tsp -L label2_s42 -g 1 timeout 86400 bash /tmp/wrapper2_s42.sh)
K1=$(tsp -L label2_s43 -g 1 -D $K0 timeout 86400 bash /tmp/wrapper2_s43.sh)
K2=$(tsp -L label2_s44 -g 1 -D $K1 timeout 86400 bash /tmp/wrapper2_s44.sh)
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `-g <gpu_id>` | Pin to specific GPU |
| `-D <job_id>` | Wait for that job to finish before starting |
| `-S <n>` | Max concurrent jobs (set to number of GPUs) |
| `-L <label>` | Human-readable label for the job |

**Or using submit.sh to generate wrappers, then requeue manually:**

```bash
# Step 1: dry-run to generate wrapper scripts
bash scripts/local/submit.sh \
    --strategy offline_bon \
    --dataset gaokao2023en \
    --model qwen25_math_7b \
    --scorers entropy \
    --seed 42 \
    --seeds 3 \
    --gpu 0 \
    --dry-run

# Step 2: submit manually with dependency chains
tsp -S 1
J0=$(tsp -g 0 CUDA_VISIBLE_DEVICES=0 python scripts/run_tts_eval.py \
    --config-path=../config \
    --config-name=<config> \
    system.seed=42)
J1=$(tsp -g 0 -D $J0 CUDA_VISIBLE_DEVICES=0 python scripts/run_tts_eval.py \
    --config-path=../config \
    --config-name=<config> \
    system.seed=43)
J2=$(tsp -g 0 -D $J1 CUDA_VISIBLE_DEVICES=0 python scripts/run_tts_eval.py \
    --config-path=../config \
    --config-name=<config> \
    system.seed=44)
```

---

## Key Differences from SLURM

- Jobs queue locally via `tsp` instead of `sbatch`
- GPU-aware: tsp tracks GPU memory and assigns free GPUs automatically
- No array jobs — multiple scorers/seeds are queued as separate jobs
- Timeout via `timeout` command (not SLURM `-t`)
- Support for job dependency chains via `-D` flag

## References

- [Task Spooler (GPU-aware)](https://github.com/justanhduc/task-spooler)
