"""
OpenAI-compatible /v1/chat/completions endpoint.
"""

import logging
import time
import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException

from service.api.models.openai_compat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ErrorDetail,
    ErrorResponse,
    Usage,
)
from service.core.strategy_manager import strategy_manager

log = logging.getLogger(__name__)

router = APIRouter()


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


def extract_answer(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    import re

    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1)
    return ""


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion using TTS strategies.

    This endpoint is OpenAI-compatible, so you can use it as a drop-in
    replacement for OpenAI's API. Additional TTS-specific parameters can be
    passed to configure the test-time scaling strategy.

    Example with OpenAI Python SDK:
    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"  # Not used yet
    )

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Solve: 2+2=?"}
        ],
        # TTS-specific parameters
        extra_body={
            "tts_strategy": "deepconf",
            "tts_mode": "offline",
            "tts_budget": 8
        }
    )

    print(response.choices[0].message.content)
    ```
    """
    try:
        log.info(f"Received chat completion request for model: {request.model}")
        log.info(f"TTS strategy: {request.tts_strategy}, mode: {request.tts_mode}")

        # Validate streaming not yet supported
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": ErrorDetail(
                        message="Streaming is not yet supported",
                        type="invalid_request_error",
                        param="stream",
                    )
                },
            )

        # Convert messages to prompt format
        # For now, just concatenate user messages
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(msg.content)

        prompt = "\n\n".join(prompt_parts)
        log.debug(f"Prompt: {prompt[:100]}...")

        # Build strategy config
        strategy_config: Dict = {
            "provider": "openrouter",  # Default to OpenRouter
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens or 512,
        }

        if request.tts_strategy == "deepconf":
            strategy_config.update(
                {
                    "mode": request.tts_mode or "offline",
                    "budget": request.tts_budget or 8,
                    "filter_method": request.tts_filter_method or "top5",
                    "top_logprobs": 20,
                }
            )

        # Create strategy
        log.info(f"Creating strategy: {request.tts_strategy}")
        strategy = strategy_manager.create_strategy(
            strategy_type=request.tts_strategy or "deepconf",
            model_name=request.model,
            strategy_config=strategy_config,
        )

        # Generate trajectory
        log.info("Generating trajectory...")
        start_time = time.time()

        # Convert to message format expected by strategy
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        result = strategy.generate_trajectory(messages)
        elapsed_time = time.time() - start_time

        log.info(f"Trajectory generated in {elapsed_time:.2f}s")

        # Extract trajectory text
        trajectory = result.get("trajectory", "")
        if not trajectory:
            raise ValueError("Strategy returned empty trajectory")

        # Extract answer if available
        answer = extract_answer(trajectory)

        # Estimate token usage
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(trajectory)

        # Build TTS metadata
        tts_metadata = {
            "strategy": request.tts_strategy,
            "model": request.model,
            "elapsed_time": round(elapsed_time, 2),
            "completed": result.get("completed", True),
        }

        # Add strategy-specific metadata
        if "metadata" in result:
            metadata = result["metadata"]
            if request.tts_strategy == "deepconf":
                tts_metadata.update(
                    {
                        "mode": metadata.get("mode"),
                        "num_traces": metadata.get("num_traces"),
                        "filtered_traces": metadata.get("filtered_traces"),
                        "confidence": metadata.get("confidence"),
                        "agreement": metadata.get("agreement"),
                    }
                )

        if answer:
            tts_metadata["extracted_answer"] = answer

        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=trajectory),
                    finish_reason="stop",
                    tts_metadata=tts_metadata,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

        log.info("Chat completion successful")
        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": ErrorDetail(
                    message=str(e), type="internal_error", code="internal_server_error"
                )
            },
        )
