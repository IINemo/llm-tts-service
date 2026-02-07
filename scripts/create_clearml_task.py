#!/usr/bin/env python3
"""Create a ClearML task for running the TTS evaluation.

This script creates a ClearML task with proper dependency handling to resolve
conflicts between lm-polygraph (requires transformers==4.51.3) and vllm 0.12.0
(requires transformers>=4.56.0).

Strategy:
1. Install all base packages including vllm with correct torch/transformers
2. In docker_bash_setup_script, install lm-polygraph (which downgrades transformers)
3. Then force-reinstall transformers>=4.57.0 to fix the downgrade
"""

from clearml import Task

# Base packages (everything except lm-polygraph)
BASE_PACKAGES = [
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
    # Evaluation
    "antlr4-python3-runtime==4.11.1",
    "word2number>=1.1",
    "latex2sympy2>=1.9",
    # For lm-polygraph deps (install before lm-polygraph to avoid conflicts)
    "scikit-learn>=1.0",
    "sentence-transformers>=2.0",
    "spacy>=3.0",
    "sacrebleu>=2.0",
    "sentencepiece",
    "accelerate>=0.20",
]

# Docker bash setup script to install lm-polygraph and fix transformers
DOCKER_BASH_SETUP = """
set -e

echo "=== Cleaning up disk space ==="
rm -rf /root/.clearml/venvs-cache/* 2>/dev/null || true
rm -rf /root/.clearml/venvs-builds/3.12 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
pip cache purge 2>/dev/null || true

echo "=== Installing lm-polygraph (will downgrade transformers) ==="
pip install 'lm-polygraph @ git+https://github.com/IINemo/lm-polygraph.git@dev'

echo "=== Force reinstalling transformers to fix vllm compatibility ==="
pip install 'transformers>=4.57.0,<5.0.0' --force-reinstall --no-deps

echo "=== Verifying installed versions ==="
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"

echo "=== Setup complete ==="
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
        docker="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
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
    parser.add_argument("--queue", default="gpu-80gb", help="Queue name")
    parser.add_argument(
        "--config", default="online_bon_openrouter_entropy_math500", help="Config name"
    )

    args = parser.parse_args()

    create_task(
        project_name=args.project,
        task_name=args.name,
        queue_name=args.queue,
        config_name=args.config,
    )
