#!/bin/bash
#SBATCH -J baseline_olymp              # Job name
#SBATCH -N 1                           # Number of nodes
#SBATCH --ntasks-per-node=1            # 1 task
#SBATCH --gres=gpu:1                   # 1 GPU (tensor_parallel_size=1)
#SBATCH -p long                        # Partition/queue name
#SBATCH -t 04:00:00                    # Time limit (4 hours for 675 samples)
#SBATCH -o logs/baseline_olymp_%j.out  # Standard output
#SBATCH -e logs/baseline_olymp_%j.err  # Standard error
#SBATCH --cpus-per-task=8              # CPU cores per task

# Exit on error
set -e

# Project directory
PROJECT_DIR="/home/artem.shelmanov/vlad/llm-tts-service"
cd "$PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust as needed for your cluster)
module load rocm 2>/dev/null || true

# Activate conda environment
source ~/.bashrc
conda activate lm-polygraph-env

# Debug info
echo "============================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo "============================================"

# Show GPU info
nvidia-smi || rocm-smi || echo "No GPU info available"

# Run the evaluation
python scripts/run_tts_eval.py \
    --config-path=../config \
    --config-name=experiments/baseline/olympiadbench/baseline_vllm_qwen25_math_7b_instruct_olympiadbench

echo "============================================"
echo "End time: $(date)"
echo "Job completed successfully"
echo "============================================"
