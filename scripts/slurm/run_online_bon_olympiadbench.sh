#!/bin/bash
#SBATCH -J bon_olymp                   # Job name
#SBATCH -N 1                           # Number of nodes
#SBATCH --ntasks-per-node=1            # 1 task
#SBATCH --gres=gpu:2                   # 2 GPUs (model + PRM scorer)
#SBATCH -p long                        # Partition/queue name
#SBATCH -t 12:00:00                    # Time limit (12 hours - BoN is slower)
#SBATCH -o logs/bon_olymp_%j.out       # Standard output
#SBATCH -e logs/bon_olymp_%j.err       # Standard error
#SBATCH --cpus-per-task=16             # CPU cores per task

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
    --config-name=experiments/online_best_of_n/olympiadbench/online_bon_vllm_nothink_qwen25_7b_prm_olympiadbench

echo "============================================"
echo "End time: $(date)"
echo "Job completed successfully"
echo "============================================"
