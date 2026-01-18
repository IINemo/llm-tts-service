# Slurm Guide

A quick reference guide for submitting and managing jobs on the HPC cluster using Slurm workload manager.

> **Official Documentation**: For comprehensive Slurm documentation on the MBZUAI cluster, see the [HPC Wiki - Slurm Guide](https://hpc.mbzuai.ac.ae/wiki/camd/slurm.html#).

---

## Quick Reference

### Job Submission

| Command | Purpose | Example |
|---------|---------|---------|
| `sbatch script.sh` | Submit a batch job | `sbatch myjob.sh` |
| `srun command` | Run a command (interactive or parallel) | `srun hostname` |
| `salloc` | Allocate resources and get an interactive shell | `salloc -N 2 -n 4` |

### Job Monitoring

| Command | Purpose | Example |
|---------|---------|---------|
| `squeue` | List jobs in the queue | `squeue` |
| `squeue -u $USER` | Show only your jobs | `squeue -u $USER` |
| `sacct -j <jobid>` | Show job accounting info (after completion) | `sacct -j 12345` |

### Job Control

| Command | Purpose | Example |
|---------|---------|---------|
| `scancel <jobid>` | Cancel a job | `scancel 12345` |
| `scancel -u $USER` | Cancel all your jobs | `scancel -u $USER` |

### Resource Request Options

| Option | Meaning | Example |
|--------|---------|---------|
| `-N` | Number of nodes | `-N 2` |
| `-n` | Total tasks (MPI ranks) | `-n 8` |
| `--cpus-per-task` | CPU cores per task | `--cpus-per-task=4` |
| `--gres=gpu:count` | Request GPUs | `--gres=gpu:4` |
| `-t` | Time limit (HH:MM:SS) | `-t 01:30:00` |
| `-p` | Partition/queue | `-p long` |
| `-J` | Job name | `-J myjob` |

---

## Submitting Jobs

### Submit a batch job

```bash
sbatch -N 1 -n 4 -t 00:30:00 job.sh
```

### Interactive shell on a GPU node

```bash
salloc -N 1 --gres=gpu:1 -t 02:00:00
```

### Run MPI program on 4 nodes

```bash
srun -N 4 -n 64 ./my_mpi_app
```

---

## Example Batch Script

A basic Slurm batch script for running experiments on the cluster:

```bash
#!/bin/bash
#SBATCH -J my_gpu_job              # Job name
#SBATCH -N 1                       # Number of nodes
#SBATCH --ntasks-per-node=1        # Tasks per node
#SBATCH --gres=gpu:1               # GPUs per node
#SBATCH -p long                    # Partition/queue name
#SBATCH -t 04:00:00                # Time limit hh:mm:ss
#SBATCH -o logs/job_%j.out         # Standard output
#SBATCH -e logs/job_%j.err         # Standard error
#SBATCH --cpus-per-task=8          # CPU cores per task

# Exit on error
set -e

# Create logs directory
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate lm-polygraph-env

# Debug info
echo "============================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================"

# Show GPU info
nvidia-smi

# Run your experiment
python scripts/run_tts_eval.py \
    --config-path=../config \
    --config-name=experiments/your_experiment

echo "============================================"
echo "End time: $(date)"
echo "Job completed"
echo "============================================"
```

### SBATCH Directives Explained

| Directive | Description |
|-----------|-------------|
| `#SBATCH -N 1` | Request 1 node |
| `#SBATCH --ntasks-per-node=1` | One task per node |
| `#SBATCH --gres=gpu:1` | Request 1 GPU |
| `#SBATCH -p long` | Use the `long` partition |
| `#SBATCH -t 04:00:00` | 4 hour time limit |
| `#SBATCH -o logs/job_%j.out` | Output file (`%j` = job ID) |
| `#SBATCH -e logs/job_%j.err` | Error file |
| `#SBATCH --cpus-per-task=8` | 8 CPU cores per task |

---

## Running LLM-TTS Experiments

### Single GPU Experiment

```bash
# Submit baseline experiment
sbatch scripts/slurm/run_baseline_olympiadbench.sh

# Monitor job
squeue -u $USER

# Watch output
tail -f logs/baseline_olymp_<jobid>.out
```

### Check Available Partitions

```bash
sinfo -s
```

Common partitions:
- `long` - Default partition with GPU nodes
- `cscc-gpu-p` - Alternative GPU partition

---

## Helper Tools

Custom helper scripts are available at `/vast/users/guangyi.chen/slurm_tools/`:

### Submit Multi-GPU Job

Automatically splits GPU requests across multiple nodes (max 8 GPUs per job):

```bash
cd /vast/users/guangyi.chen/slurm_tools
./submit_job.sh <NAME> <TOTAL_GPUS>

# Example: Request 12 GPUs (creates 2 jobs: 8 GPUs + 4 GPUs)
./submit_job.sh myproj 12
```

### Attach to Running Job

Connect to a running job interactively:

```bash
cd /vast/users/guangyi.chen/slurm_tools
./attach_job.sh <JOBID>

# Example
./attach_job.sh 343
```

---

## Troubleshooting

### Common Issues

**Job stuck in PENDING state:**
```bash
squeue -j <jobid> -o "%R"  # Show reason
```

Common reasons:
- `Resources` - Waiting for resources
- `Priority` - Lower priority than other jobs
- `QOSMaxJobsPerUserLimit` - Too many jobs queued

**Job failed immediately:**
```bash
cat logs/job_<jobid>.err  # Check error log
```

**Check job history:**
```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
```

---

## Additional Resources

- [MBZUAI HPC Wiki - Slurm Guide](https://hpc.mbzuai.ac.ae/wiki/camd/slurm.html#)
- [Official Slurm Documentation](https://slurm.schedmd.com/documentation.html)
