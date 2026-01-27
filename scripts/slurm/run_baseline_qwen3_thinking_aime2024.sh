#!/bin/bash
#SBATCH -J baseline_qwen3_thinking_aime2024   # Job name
#SBATCH -N 1                                   # Number of nodes
#SBATCH --ntasks-per-node=1                    # 1 task
#SBATCH --gres=gpu:1                           # 1 GPU (baseline only needs model)
#SBATCH -p long                                # Partition/queue name
#SBATCH -t 04:00:00                            # Time limit (4h for 30 problems)
#SBATCH -o logs/baseline_qwen3_thinking_aime2024_%j.out
#SBATCH -e logs/baseline_qwen3_thinking_aime2024_%j.err
#SBATCH --cpus-per-task=16

PROJECT_DIR="/home/artem.shelmanov/vlad/llm-tts-service"
cd "$PROJECT_DIR"
mkdir -p logs

module load rocm 2>/dev/null || true

echo "============================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================"

nvidia-smi || true

echo "Activating conda environment..."
source ~/.bashrc
conda activate lm-polygraph-env

echo "Python path: $(which python)"

# OpenAI API key for LLM-as-a-judge evaluation (set your own key)
# export OPENAI_API_KEY="your-api-key-here"

echo "Starting experiment..."

python scripts/run_tts_eval.py \
    --config-path=../config \
    --config-name=experiments/baseline/aime2024/baseline_vllm_qwen3_8b_thinking_aime2024

echo "============================================"
echo "End time: $(date)"
echo "Job completed"
echo "============================================"
