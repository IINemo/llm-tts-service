#!/bin/bash
#SBATCH -J sc_math500                       # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH --ntasks-per-node=1                 # 1 task
#SBATCH --gres=gpu:1                        # 1 GPU (entropy scorer, no PRM)
#SBATCH -p long                             # Partition/queue name
#SBATCH -t 12:00:00                         # Time limit (12h)
#SBATCH -o logs/self_consistency_math500_%j.out
#SBATCH -e logs/self_consistency_math500_%j.err
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
    --config-name=experiments/self_consistency/math500/self_consistency_vllm_nothink_qwen25_7b_entropy_math500

echo "============================================"
echo "End time: $(date)"
echo "Job completed"
echo "============================================"
