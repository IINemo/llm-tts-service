#!/usr/bin/env python3
"""
Create a ClearML task for the HuggingFace Scorer API endpoint.

Usage:
    python scripts/create_clearml_scorer_task.py --name 'Scorer Qwen2.5-7B-Instruct' --queue high_q
    python scripts/create_clearml_scorer_task.py --name 'Scorer Qwen2.5-7B-Instruct' --queue high_q --model Qwen/Qwen2.5-7B-Instruct --port 8000
"""

from clearml import Task

# HuggingFace transformers GPU image (lighter than vLLM)
DEFAULT_DOCKER_IMAGE = "huggingface/transformers-pytorch-gpu:latest"
DEFAULT_DOCKER_ARGS = "--entrypoint= --network=host --shm-size=4g"

DOCKER_BASH_SETUP = r"""
echo "=== GPU Info ==="
nvidia-smi
echo "=== Installing dependencies ==="
pip install fastapi uvicorn[standard] pydantic pydantic-settings
echo "================"
"""


def create_scorer_task(
    project_name: str = "llm-tts-service",
    task_name: str = "Scorer Qwen2.5-7B-Instruct",
    queue_name: str = "high_q",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    port: int = 8000,
    device: str = "cuda:0",
    use_docker: bool = True,
):
    docker_args = DEFAULT_DOCKER_ARGS

    if use_docker:
        task = Task.create(
            project_name=project_name,
            task_name=task_name,
            repo="https://github.com/IINemo/llm-tts-service.git",
            branch="fix/clearml-hydra-wrapper",
            script="scripts/serve_scorer.py",
            argparse_args=[
                "--model",
                model_name,
                "--port",
                str(port),
                "--device",
                device,
            ],
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
            script="scripts/serve_scorer.py",
            argparse_args=[
                "--model",
                model_name,
                "--port",
                str(port),
                "--device",
                device,
            ],
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
        description="Create ClearML task for HF Scorer endpoint"
    )
    parser.add_argument("--project", default="llm-tts-service")
    parser.add_argument("--name", default="Scorer Qwen2.5-7B-Instruct")
    parser.add_argument("--queue", default="high_q")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-docker", action="store_true", help="Run without Docker")
    args = parser.parse_args()

    create_scorer_task(
        project_name=args.project,
        task_name=args.name,
        queue_name=args.queue,
        model_name=args.model,
        port=args.port,
        device=args.device,
        use_docker=not args.no_docker,
    )
