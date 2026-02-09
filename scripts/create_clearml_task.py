#!/usr/bin/env python3
"""Create a ClearML task for running the TTS evaluation.

This script creates a ClearML task with dependency handling to avoid:
- OmegaConf/ANTLR runtime mismatch (pin antlr4-python3-runtime==4.9.3)
- vLLM needing newer transformers (pin transformers>=4.57.0,<5.0.0)
- lm-polygraph VCS install flakiness under ClearML 'packages='

IMPORTANT:
- Do NOT install lm-polygraph via BASE_PACKAGES. Install it at runtime inside
  scripts/run_tts_eval_clearml.py (post-venv) before running run_tts_eval.py.
"""

from clearml import Task

# Base packages - vllm docker image may already have some of these,
# but we pin explicitly for reproducibility.
BASE_PACKAGES = [
    # Pin conflicting versions FIRST to prevent upgrades
    "huggingface-hub>=0.34.0,<1.0",
    "tokenizers>=0.22.0,<0.24.0",
    "fsspec>=2023.1.0,<=2025.3.0",
    # Core ML packages with pinned versions for vllm 0.12.0
    "torch==2.9.0",
    "torchvision==0.24.0",
    "torchaudio==2.9.0",
    "triton>=3.3.1",
    "setuptools>=77.0.3",
    "transformers>=4.57.0,<5.0.0",
    # vLLM
    "vllm==0.12.0",
    # Utility packages
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "sympy>=1.12",
    "numpy<2.0.0",
    "pandas>=2.0",
    "tqdm>=4.60",
    "openai>=1.0",
    "httpx>=0.27",
    "pyyaml>=6.0",
    "parse",
    # HF eval/datasets stack
    "datasets>=2.18.0,<3.0.0",
    "evaluate>=0.4.0,<1.0.0",
    # Evaluation (ANTLR runtime pin avoids OmegaConf ATN mismatch)
    "antlr4-python3-runtime==4.9.3",
    "word2number>=1.1",
    "latex2sympy2>=1.9",
    # Common deps used by lm-polygraph / eval stack (lm-polygraph installed at runtime)
    "scikit-learn>=1.0",
    "sentence-transformers>=2.0",
    "spacy>=3.0",
    "sacrebleu>=2.0",
    "sentencepiece",
    "accelerate>=0.20",
]

# Docker bash setup script - runs BEFORE venv creation
# Only used for system-level setup (apt packages, disk cleanup)
DOCKER_BASH_SETUP = r"""
set -e


echo "=== Aggressive disk cleanup ==="
rm -rf /root/.clearml/venvs-cache/* 2>/dev/null || true
rm -rf /root/.clearml/venvs-builds/* 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
rm -rf /var/cache/apt/archives/* 2>/dev/null || true
rm -rf /root/.cache/pip/* 2>/dev/null || true
pip cache purge 2>/dev/null || true


# Show available disk space
df -h / || true


echo "=== Installing system packages ==="
rm -rf /var/lib/apt/lists/*
apt-get clean
apt-get update --allow-insecure-repositories || true
apt-get install -y --allow-unauthenticated ca-certificates gnupg
apt-get update
apt-get install -y git python3 python3-venv python3-pip


# Final cleanup
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "=== System setup complete ==="
df -h / || true
"""


def create_task(
    project_name: str = "llm-tts-service",
    task_name: str = "ToT Self-Verification MATH500",
    queue_name: str = "gpu-80gb",
    config_name: str = "online_bon_openrouter_entropy_math500",
):
    """Create and enqueue a ClearML task."""

    task = Task.create(
        project_name=project_name,
        task_name=task_name,
        repo="https://github.com/IINemo/llm-tts-service.git",
        branch="fix/clearml-hydra-wrapper",
        script="scripts/run_tts_eval_clearml.py",
        # Use minimal CUDA runtime image (much smaller than devel)
        docker="nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        docker_args="--entrypoint=",  # IMPORTANT: clear entrypoint
        docker_bash_setup_script=DOCKER_BASH_SETUP,
        packages=BASE_PACKAGES,
    )

    # Set Hydra arguments
    task.set_parameter(
        "HydraArgs/config_path", "../config/experiments/online_best_of_n/math500"
    )
    task.set_parameter("HydraArgs/config_name", config_name)
    task.set_parameter("HydraArgs/overrides", "[]")

    print(f"Created task: {task.id}")
    print(f"Task name: {task_name}")
    print(f"Project: {project_name}")

    # Enqueue to the specified queue
    Task.enqueue(task, queue_name=queue_name)
    print(f"Enqueued to: {queue_name}")

    return task


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create ClearML task for TTS evaluation"
    )
    parser.add_argument(
        "--project", default="llm-tts-service", help="ClearML project name"
    )
    parser.add_argument(
        "--name", default="ToT Self-Verification MATH500 v38", help="Task name"
    )
    parser.add_argument("--queue", default="high_q_80", help="Queue name")
    parser.add_argument(
        "--config",
        default="online_bon_openrouter_entropy_math500",
        help="Config name",
    )

    args = parser.parse_args()

    create_task(
        project_name=args.project,
        task_name=args.name,
        queue_name=args.queue,
        config_name=args.config,
    )
