"""
OpenAI-compatible /v1/models endpoint.
"""

import time

from fastapi import APIRouter

from service_app.api.models.openai_compat import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    List available models in OpenAI-compatible format.

    This endpoint returns models that support TTS strategies with logprobs.
    All models are accessed via OpenRouter by default.
    """
    # Available models with logprobs support via OpenRouter
    available_models = [
        {
            "id": "openai/gpt-4o-mini",
            "owned_by": "openai",
            "description": "Fast and cost-effective model with logprobs support",
        },
        {
            "id": "openai/gpt-4o",
            "owned_by": "openai",
            "description": "Most capable model with logprobs support",
        },
        {
            "id": "openai/gpt-3.5-turbo",
            "owned_by": "openai",
            "description": "Fast model with logprobs support",
        },
    ]

    timestamp = int(time.time())

    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id=model["id"],
                object="model",
                created=timestamp,
                owned_by=model["owned_by"],
            )
            for model in available_models
        ],
    )


@router.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get specific model information.

    Args:
        model_id: Model identifier (e.g., "openai/gpt-4o-mini")
    """
    # For simplicity, return generic info for any model
    return ModelInfo(
        id=model_id,
        object="model",
        created=int(time.time()),
        owned_by="openai" if "openai/" in model_id else "unknown",
    )
