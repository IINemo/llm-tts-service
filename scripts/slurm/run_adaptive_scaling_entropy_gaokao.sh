#!/bin/bash
#SBATCH -J adapt_entropy_gaokao           # Job name
#SBATCH -N 1                              # Number of nodes
#SBATCH --ntasks-per-node=1               # 1 task
#SBATCH --gres=gpu:1                      # 1 GPU
#SBATCH -p long                           # Partition/queue name
#SBATCH -t 24:00:00                       # Time limit (24h)
#SBATCH -o logs/adapt_entropy_gaokao_%j.out
#SBATCH -e logs/adapt_entropy_gaokao_%j.err
#SBATCH --cpus-per-task=16

set -e

PROJECT_DIR="/home/artem.shelmanov/vlad/llm-tts-service"
cd "$PROJECT_DIR"
mkdir -p logs

module load rocm 2>/dev/null || true

source ~/.bashrc
conda activate lm-polygraph-env

echo "============================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================"

nvidia-smi || true

QWEN_EVAL_PYTHON="$HOME/miniconda3/envs/qwen-eval/bin/python"
if [ ! -f "$QWEN_EVAL_PYTHON" ]; then
    echo "Setting up qwen-eval environment for math evaluation..."
    conda create -n qwen-eval python=3.11 -y
    conda run -n qwen-eval pip install sympy==1.12 antlr4-python3-runtime==4.11.1 regex
    echo "qwen-eval environment created"
else
    echo "qwen-eval environment found at $QWEN_EVAL_PYTHON"
fi

echo "Starting experiment..."

python scripts/run_tts_eval.py \
    --config-path=../config \
    --config-name=experiments/adaptive_scaling/gaokao2023en/adaptive_scaling_vllm_qwen25_7.5b_gaokao2023_entropy

echo "============================================"
echo "End time: $(date)"
echo "Job completed"
echo "============================================"
