#!/bin/bash
#SBATCH --job-name=[MARS-3554]mur_he_plus_seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --partition=mts_ai

srun --container-image=artifactory.mts.ai/ml-docker/gpt_transformers_pytorch_24_12 \
     --container-name=mur_he_plus_seq \
     --no-container-entrypoint \
     --container-workdir=/home/s.senichev/llm-tts-service/ \
     bash -c "
       cd /home/s.senichev/llm-tts-service &&
       ./setup.sh &&
       pip install latex2sympy2 --no-deps &&
       pip install -e '.[dev,vllm]' &&

       echo '=== Running MUR (adaptive scaling): seq ===' &&
       python scripts/run_tts_eval.py \
         --config-path=../config \
         --config-name=experiments/adaptive_scaling/human_eval_plus/adaptive_scaling_vllm_qwen3_8b_human_eval_plus_sequence_prob \
         report_to=none generation.checkpoint_batch_size=32
     "
