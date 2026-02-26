#!/usr/bin/env python3
"""
Standalone vLLM OpenAI-compatible API server for LLM Critic scoring.

Thin wrapper around vLLM's built-in server that:
1. Integrates with ClearML (Task.init for remote execution)
2. Prints the local IP for easy endpoint discovery
3. Launches vllm.entrypoints.openai.api_server

Usage:
    python scripts/serve_scorer_vllm.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
    python scripts/serve_scorer_vllm.py --model Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 4096
"""

# ClearML auto-init (when run as ClearML task, agent injects Task.init)
try:
    from clearml import Task

    task = Task.init(
        project_name="llm-tts-service",
        task_name="scorer-vllm-endpoint",
        continue_last_task=True,
        output_uri=False,
    )
except Exception:
    pass

import argparse
import logging
import socket
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def get_local_ip():
    """Get the machine's local network IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Scorer API Server")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--max-model-len", type=int, default=4096, help="Max model context length"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--trust-remote-code",
        type=str,
        default="true",
        help="Trust remote code",
    )
    args = parser.parse_args()

    local_ip = get_local_ip()
    log.info("=" * 60)
    log.info("vLLM Scorer API server starting")
    log.info("Model: %s", args.model)
    log.info("Local endpoint: http://%s:%s/v1", local_ip, args.port)
    log.info(
        "Use this base_url for scorer config: http://%s:%s/v1", local_ip, args.port
    )
    log.info("=" * 60)

    # Launch vLLM's built-in OpenAI-compatible server
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--host",
        args.host,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]

    log.info("Running: %s", " ".join(cmd))
    sys.exit(subprocess.call(cmd))
