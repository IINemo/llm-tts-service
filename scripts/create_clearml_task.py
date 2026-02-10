#!/usr/bin/env python3
from clearml import Task

# PyTorch base image (has Python 3.11 with pre-built wheels for sentencepiece)
DEFAULT_DOCKER_IMAGE = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime"
DEFAULT_DOCKER_ARGS = "--shm-size=8g"


def create_task(
    project_name: str = "llm-tts-service",
    task_name: str = "BeamSearch SelfVerification MinervaMath",
    queue_name: str = "gpu-80gb",
    config_name: str = "experiments/beam_search/minerva_math/beam_search_vllm_self_verification_minerva_math",
    config_path: str = "../config",
    use_docker: bool = True,
    overrides: list = None,
):
    if overrides is None:
        overrides = []

    if use_docker:
        task = Task.create(
            project_name=project_name,
            task_name=task_name,
            repo="https://github.com/IINemo/llm-tts-service.git",
            branch="fix/clearml-hydra-wrapper",
            script="scripts/run_tts_eval_clearml.py",
            docker=f"{DEFAULT_DOCKER_IMAGE} {DEFAULT_DOCKER_ARGS}",
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
    task.set_parameter("HydraArgs/overrides", str(overrides))

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
