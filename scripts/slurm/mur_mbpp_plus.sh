#!/bin/bash
#SBATCH --job-name=mur_mbpp_plus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1

srun --container-image=artifactory.mts.ai/ml-docker/gpt_transformers_pytorch_24_12 \
     --container-name=mur_mbpp_plus \
     --no-container-entrypoint \
     --container-workdir=/home/s.senichev/llm-tts-service/ \
     bash -c "
       cd /home/s.senichev/llm-tts-service &&
       ./setup.sh &&
       pip install -e '.[dev,vllm]' &&
       pip install numpy==1.26.4 &&
       pip install transformers==4.57.3 &&
       pip install flash_attn -U --force-reinstall &&

       echo '=== Running MUR (adaptive scaling): entropy ===' &&
       python scripts/run_tts_eval.py \
         --config-path=../config \
         --config-name=experiments/adaptive_scaling/mbpp_plus/adaptive_scaling_vllm_qwen3_8b_mbpp_plus_entropy &&

       echo '=== Running MUR (adaptive scaling): perplexity ===' &&
       python scripts/run_tts_eval.py \
         --config-path=../config \
         --config-name=experiments/adaptive_scaling/mbpp_plus/adaptive_scaling_vllm_qwen3_8b_mbpp_plus_perplexity &&

       echo '=== Running MUR (adaptive scaling): sequence_prob ===' &&
       python scripts/run_tts_eval.py \
         --config-path=../config \
         --config-name=experiments/adaptive_scaling/mbpp_plus/adaptive_scaling_vllm_qwen3_8b_mbpp_plus_sequence_prob
     "
