#!/usr/bin/env python3
"""
Create a ClearML task for the vLLM Scorer API endpoint.

Uses vLLM's built-in OpenAI-compatible server (python -m vllm.entrypoints.openai.api_server).

Usage:
    python scripts/create_clearml_scorer_task.py --name 'Scorer Qwen2.5-7B-Instruct' --queue high_q_80
    python scripts/create_clearml_scorer_task.py --name 'Scorer Qwen2.5-7B-Instruct' --queue high_q_80 --model Qwen/Qwen2.5-7B-Instruct --port 8000
"""

from clearml import Task

DEFAULT_DOCKER_IMAGE = "vllm/vllm-openai:v0.12.0"
DEFAULT_DOCKER_ARGS = "--entrypoint= --network=host --shm-size=4g --gpus all"

DOCKER_BASH_SETUP = r"""
echo "=== GPU Info ==="
nvidia-smi
nvidia-smi -L
echo "=== Installing dependencies ==="
pip install boto3
echo "=== Starting vLLM OpenAI server ==="
"""


def create_scorer_task(
    project_name: str = "llm-tts-service",
    task_name: str = "Scorer Qwen2.5-7B-Instruct",
    queue_name: str = "high_q_80",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    use_docker: bool = True,
):
    docker_args = DEFAULT_DOCKER_ARGS

    # vLLM server script uses argparse-style args
    argparse_args = [
        f"model={model_name}",
        f"port={port}",
        "host=0.0.0.0",
        f"max-model-len={max_model_len}",
        f"gpu-memory-utilization={gpu_memory_utilization}",
        "trust-remote-code=true",
    ]

    if use_docker:
        task = Task.create(
            project_name=project_name,
            task_name=task_name,
            repo="https://github.com/IINemo/llm-tts-service.git",
            branch="fix/clearml-hydra-wrapper",
            script="scripts/serve_scorer_vllm.py",
            argparse_args=argparse_args,
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
            script="scripts/serve_scorer_vllm.py",
            argparse_args=argparse_args,
            packages=[],
        )

    print(f"Created task: {task.id}")
    Task.enqueue(task, queue_name=queue_name)
    print(f"Enqueued to: {queue_name}")
    print("\nOnce running, check task logs for the scorer endpoint URL (base_url).")
    print("Then launch beam search with:")
    print("  --override scorer.model.type=openai_api")
    print(f"  --override scorer.model.base_url=http://<IP>:{port}/v1")
    print(f"  --override scorer.model.model_name={model_name}")
    print("  --override scorer.model.api_key=unused")
    return task


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create ClearML task for vLLM Scorer endpoint"
    )
    parser.add_argument("--project", default="llm-tts-service")
    parser.add_argument("--name", default="Scorer Qwen2.5-7B-Instruct")
    parser.add_argument("--queue", default="high_q_80")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--no-docker", action="store_true", help="Run without Docker")
    args = parser.parse_args()

    create_scorer_task(
        project_name=args.project,
        task_name=args.name,
        queue_name=args.queue,
        model_name=args.model,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_docker=not args.no_docker,
    )
