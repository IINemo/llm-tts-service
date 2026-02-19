#!/bin/bash
#SBATCH --job-name=ucot_he_plus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1

srun --container-image=artifactory.mts.ai/ml-docker/gpt_transformers_pytorch_24_12 \
     --container-name=ucot_he_plus \
     --no-container-entrypoint \
     --container-workdir=/home/s.senichev/llm-tts-service/ \
     bash -c "
       cd /home/s.senichev/llm-tts-service &&
       ./setup.sh &&
       pip install -e '.[dev,vllm]' &&
       pip install numpy==1.26.4 &&
       pip install transformers==4.57.3 &&
       pip install flash_attn -U --force-reinstall &&

       echo '=== Running uncert-cot: entropy sequence ===' &&
       python scripts/run_tts_eval.py \
         --config-path=../config \
         --config-name=experiments/uncertainty_cot/human_eval_plus/uncert_cot_vllm_qwen3_8b_human_eval_plus_entropy_sequence &&

       echo '=== Running uncert-cot: entropy token ===' &&
       python scripts/run_tts_eval.py \
         --config-path=../config \
         --config-name=experiments/uncertainty_cot/human_eval_plus/uncert_cot_vllm_qwen3_8b_human_eval_plus_entropy_token &&

       echo '=== Running uncert-cot: pd sequence ===' &&
       python scripts/run_tts_eval.py \
         --config-path=../config \
         --config-name=experiments/uncertainty_cot/human_eval_plus/uncert_cot_vllm_qwen3_8b_human_eval_plus_pd_sequence &&

       echo '=== Running uncert-cot: pd token ===' &&
       python scripts/run_tts_eval.py \
         --config-path=../config \
         --config-name=experiments/uncertainty_cot/human_eval_plus/uncert_cot_vllm_qwen3_8b_human_eval_plus_pd_token
     "
