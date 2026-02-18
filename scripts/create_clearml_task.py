#!/usr/bin/env python3
import json
import os

from clearml import Task

# vLLM v0.12.0 with CUDA 12.1 (compatible with older drivers)
DEFAULT_DOCKER_IMAGE = "vllm/vllm-openai:v0.12.0"
# Override entrypoint, skip lm_polygraph bootstrap
# Add OPENROUTER_API_KEY for llm_judge evaluation
DEFAULT_DOCKER_ARGS = (
    "--entrypoint= --network=host --shm-size=8g -e SKIP_LM_POLYGRAPH=1"
)

# Check GPU info, fix apt GPG signature issues
# NOTE: ClearML flattens this to one line with ";", so no curly-brace groups allowed
DOCKER_BASH_SETUP = r"""
echo "=== Disk before cleanup ==="
df -h /
rm -rf /root/.clearml/venvs-cache/* /root/.clearml/pip-download-cache/* /var/cache/apt/archives/*.deb
echo "=== Disk after cleanup ==="
df -h /
mkdir -p /tmp/apt-archives/partial
rm -rf /var/lib/apt/lists/*
apt-get -o dir::cache::archives=/tmp/apt-archives update --allow-insecure-repositories -qq
apt-get -o dir::cache::archives=/tmp/apt-archives install -y -qq --allow-unauthenticated --no-install-recommends git
git --version || echo "ERROR: git installation failed"
echo "=== GPU Info ==="
nvidia-smi
echo "=== CUDA Version ==="
nvcc --version 2>/dev/null || echo "nvcc not found"
echo "================"
"""


def create_task(
    project_name: str = "llm-tts-service",
    task_name: str = "BeamSearch SelfVerification MinervaMath",
    queue_name: str = "gpu-80gb",
    config_name: str = "experiments/beam_search/minerva_math/beam_search_vllm_self_verification_math500",
    config_path: str = "../config",
    use_docker: bool = True,
    overrides: list = None,
):
    if overrides is None:
        overrides = []

    # Build docker args, injecting API keys from local environment
    docker_args = DEFAULT_DOCKER_ARGS
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if openrouter_key:
        docker_args += f" -e OPENROUTER_API_KEY={openrouter_key}"
        print("OPENROUTER_API_KEY: injected from local environment")
    else:
        print("WARNING: OPENROUTER_API_KEY not set in local environment")

    if use_docker:
        task = Task.create(
            project_name=project_name,
            task_name=task_name,
            repo="https://github.com/IINemo/llm-tts-service.git",
            branch="fix/clearml-hydra-wrapper",
            script="scripts/run_tts_eval_clearml.py",
            docker=f"{DEFAULT_DOCKER_IMAGE} {docker_args}",
            docker_bash_setup_script=DOCKER_BASH_SETUP,
            packages=[],
        )
    else:
        task = Task.create(
            project_name=project_name,
            task_name=task_name,
            repo="https://github.com/IINemo/llm-tts-service.git",
            branch="fix/clearml-hydra-wrapper",
            script="scripts/run_tts_eval_clearml.py",
            packages=[],
        )

    # Hydra args (same format as working task)
    task.set_parameter("HydraArgs/config_path", config_path)
    task.set_parameter("HydraArgs/config_name", config_name)
    task.set_parameter("HydraArgs/overrides", json.dumps(overrides))

    print(f"Created task: {task.id}")
    Task.enqueue(task, queue_name=queue_name)
    print(f"Enqueued to: {queue_name}")
    return task


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create ClearML task for TTS evaluation"
    )
    parser.add_argument("--project", default="llm-tts-service")
    parser.add_argument("--name", default="BeamSearch SelfVerification MinervaMath")
    parser.add_argument("--queue", default="high_q_80")
    parser.add_argument(
        "--config",
        default="experiments/beam_search/minerva_math/beam_search_vllm_self_verification_minerva_math",
    )
    parser.add_argument("--config-path", dest="config_path", default="../config")
    parser.add_argument("--no-docker", action="store_true", help="Run without Docker")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra overrides (can be used multiple times)",
    )
    args = parser.parse_args()

    create_task(
        project_name=args.project,
        task_name=args.name,
        queue_name=args.queue,
        config_name=args.config,
        config_path=args.config_path,
        use_docker=not args.no_docker,
        overrides=args.override,
    )
