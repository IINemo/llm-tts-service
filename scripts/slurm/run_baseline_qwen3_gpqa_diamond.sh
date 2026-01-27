#!/bin/bash
#SBATCH -J baseline_qwen3_gpqa_diamond        # Job name
#SBATCH -N 1                                   # Number of nodes
#SBATCH --ntasks-per-node=1                    # 1 task
#SBATCH --gres=gpu:1                           # 1 GPU (baseline only needs model)
#SBATCH -p long                                # Partition/queue name
#SBATCH -t 04:00:00                            # Time limit (4h for 198 samples)
#SBATCH -o logs/baseline_qwen3_gpqa_diamond_%j.out
#SBATCH -e logs/baseline_qwen3_gpqa_diamond_%j.err
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

echo "Starting experiment..."

python scripts/run_tts_eval.py \
    --config-path=../config \
    --config-name=experiments/baseline/gpqa_diamond/baseline_vllm_qwen3_8b_gpqa_diamond

echo "============================================"
echo "End time: $(date)"
echo "Job completed"
echo "============================================"
