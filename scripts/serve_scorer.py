#!/usr/bin/env python3
"""
Standalone OpenAI-compatible API server for LLM Critic scoring.

Uses HuggingFace transformers (lighter than vLLM, works on smaller GPUs).
Implements /v1/chat/completions and /v1/models endpoints.

Usage:
    python scripts/serve_scorer.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
    python scripts/serve_scorer.py --model Qwen/Qwen2.5-7B-Instruct --port 8000 --device cuda:0
"""

import argparse
import logging
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Global model state
model = None
tokenizer = None
model_name = None
device = None
executor = ThreadPoolExecutor(max_workers=4)


# --- Request/Response schemas (OpenAI-compatible) ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    stop: Optional[list[str]] = None

    # Accept but ignore extra fields (e.g. reasoning_effort, extra_body)
    class Config:
        extra = "allow"


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = []
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = []


# --- Model loading ---


def load_model(model_path: str, device_str: str):
    global model, tokenizer, model_name, device
    model_name = model_path
    device = device_str

    log.info(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    log.info(f"Loading model: {model_path} on {device_str}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_str,
        trust_remote_code=True,
    )
    model.eval()
    log.info(f"Model loaded: {model_path} ({dtype}) on {device_str}")


# --- Inference ---


def generate_response(
    messages: list[ChatMessage],
    max_tokens: int,
    temperature: float,
    stop: list[str] | None,
) -> tuple[str, int, int]:
    """Generate a single response. Returns (text, prompt_tokens, completion_tokens)."""
    chat = [{"role": m.role, "content": m.content} for m in messages]

    input_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    new_tokens = output[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Apply stop sequences
    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]

    return text, prompt_len, len(new_tokens)


# --- FastAPI app ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    executor.shutdown(wait=False)


app = FastAPI(title="HF Scorer Server", lifespan=lifespan)


@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelInfo(id=model_name)])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    import asyncio

    loop = asyncio.get_event_loop()

    choices = []
    total_prompt = 0
    total_completion = 0

    for i in range(request.n):
        text, p_tokens, c_tokens = await loop.run_in_executor(
            executor,
            generate_response,
            request.messages,
            request.max_tokens,
            request.temperature,
            request.stop,
        )
        choices.append(
            ChatCompletionChoice(
                index=i,
                message=ChatMessage(role="assistant", content=text),
            )
        )
        total_prompt += p_tokens
        total_completion += c_tokens

    return ChatCompletionResponse(
        model=request.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        ),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name}


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
    parser = argparse.ArgumentParser(description="HuggingFace Scorer API Server")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cpu, etc.)")
    args = parser.parse_args()

    load_model(args.model, args.device)

    local_ip = get_local_ip()
    log.info("=" * 60)
    log.info("Scorer API server starting")
    log.info("Model: %s", args.model)
    log.info("Local endpoint: http://%s:%s/v1", local_ip, args.port)
    log.info(
        "Use this base_url for scorer config: http://%s:%s/v1", local_ip, args.port
    )
    log.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
