"""
OpenAI-compatible /v1/chat/completions endpoint.
"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from service_app.api.models.openai_compat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ErrorResponse,
    Usage,
)
from service_app.core.strategy_manager import strategy_manager

log = logging.getLogger(__name__)

router = APIRouter()


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


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
    Create a chat completion using self-consistency strategy.

    This endpoint is OpenAI-compatible. Use the OpenAI Python SDK:

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="your-openrouter-key"
    )

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reason step by step, put answer in \\\\boxed{}."},
            {"role": "user", "content": "What is 15 * 7?"}
        ],
        extra_body={
            "tts_strategy": "self_consistency",
            "num_paths": 5
        }
    )

    print(response.choices[0].message.content)
    print(response.choices[0].tts_metadata)  # consensus_score, answer_distribution
    ```
    """
    try:
        log.info(f"Received chat completion request for model: {request.model}")
        log.info(f"TTS strategy: {request.tts_strategy}")

        # Validate streaming not yet supported
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Streaming is not yet supported",
                        "type": "invalid_request_error",
                        "param": "stream",
                    }
                },
            )

        # Convert messages to dict format
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Determine strategy type and provider
        strategy_type = request.tts_strategy or "self_consistency"
        is_vllm_strategy = strategy_type in ("offline_bon", "online_bon", "beam_search")

        # Build strategy config
        strategy_config = {
            "provider": request.provider
            or ("vllm" if is_vllm_strategy else "openrouter"),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4096,
            "num_paths": request.num_paths or 5,
            # vLLM TTS params
            "scorer_type": request.tts_scorer or "entropy",
            "num_trajectories": request.tts_num_trajectories,
            "candidates_per_step": request.tts_candidates_per_step,
            "beam_size": request.tts_beam_size,
            "max_steps": request.tts_max_steps,
            "score_aggregation": request.tts_score_aggregation,
            "window_size": request.tts_window_size,
        }

        # Create strategy
        log.info(f"Creating strategy: {strategy_type}")
        strategy = strategy_manager.create_strategy(
            strategy_type=strategy_type,
            model_name=request.model,
            strategy_config=strategy_config,
        )

        # Generate trajectory
        log.info("Generating trajectory...")
        start_time = time.time()

        if is_vllm_strategy:
            results = strategy.generate_trajectories_batch([messages])
            result = results[0]
        else:
            result = strategy.generate_trajectory(messages)

        elapsed_time = time.time() - start_time

        log.info(f"Trajectory generated in {elapsed_time:.2f}s")

        # Extract trajectory text
        trajectory = result.get("trajectory", "")
        if not trajectory:
            raise ValueError("Strategy returned empty trajectory")

        # Get metadata
        metadata = result.get("metadata", {})
        metadata["elapsed_time"] = round(elapsed_time, 2)
        metadata["selected_answer"] = result.get("extracted_answer", "")
        if is_vllm_strategy:
            metadata["strategy"] = strategy_type
            metadata["reasoning_steps"] = result.get("reasoning_steps", 0)
            metadata["completed"] = result.get("completed", False)
            if "aggregated_score" in result:
                metadata["aggregated_score"] = result["aggregated_score"]
            if "validity_scores" in result:
                metadata["validity_scores"] = result["validity_scores"]

        # Estimate token usage
        prompt_text = " ".join(msg["content"] for msg in messages)
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = metadata.get("total_tokens", estimate_tokens(trajectory))

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
                    tts_metadata=metadata,
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
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_server_error",
                }
            },
        )
