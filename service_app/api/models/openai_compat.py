"""
OpenAI-compatible API models.
Implements the same interface as OpenAI's Chat Completions API.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Message role"
    )
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional message name")


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Supports self-consistency strategy via OpenAI/OpenRouter APIs.
    """

    # Standard OpenAI parameters
    model: str = Field(
        default="openai/gpt-4o-mini",
        description="Model to use (OpenRouter format: provider/model)",
    )
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=4096, ge=1)
    stream: bool = Field(
        default=False, description="Stream responses (not yet supported)"
    )

    # Strategy selection
    tts_strategy: Optional[str] = Field(
        default="self_consistency",
        description="TTS strategy: self_consistency, offline_bon, online_bon, beam_search",
    )

    # Self-consistency specific parameters
    num_paths: Optional[int] = Field(
        default=5, description="Number of reasoning paths to generate", ge=1, le=100
    )

    # Provider selection
    provider: Optional[str] = Field(
        default="openrouter",
        description="API provider: openrouter, openai, or vllm",
    )

    # vLLM TTS strategy parameters (passed via extra_body)
    tts_scorer: Optional[str] = Field(
        default="entropy",
        description="Scorer type: entropy, perplexity, sequence_prob, prm",
    )
    tts_num_trajectories: Optional[int] = Field(
        default=8, description="Number of trajectories for offline BoN", ge=1, le=256
    )
    tts_candidates_per_step: Optional[int] = Field(
        default=4,
        description="Candidates per step for online BoN / beam search",
        ge=1,
        le=64,
    )
    tts_beam_size: Optional[int] = Field(
        default=4, description="Beam size for beam search", ge=1, le=64
    )
    tts_max_steps: Optional[int] = Field(
        default=100, description="Max reasoning steps", ge=1, le=500
    )
    tts_score_aggregation: Optional[str] = Field(
        default="min",
        description="Score aggregation: min, mean, max, product, last",
    )
    tts_window_size: Optional[int] = Field(
        default=None, description="Window size for scoring (1-N steps)", ge=1, le=50
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                "temperature": 0.7,
                "tts_strategy": "self_consistency",
                "num_paths": 5,
            }
        }


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: str = Field(..., description="Why generation stopped")

    # TTS-specific metadata (optional)
    tts_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="TTS strategy metadata (consensus_score, answer_distribution, etc.)",
    )


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(..., description="Tokens in prompt")
    completion_tokens: int = Field(..., description="Tokens in completion")
    total_tokens: int = Field(..., description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(..., description="Unique completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[ChatCompletionChoice] = Field(..., description="Completion choices")
    usage: Usage = Field(..., description="Token usage")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Let me solve this step by step...\n\n\\boxed{4}",
                        },
                        "finish_reason": "stop",
                        "tts_metadata": {
                            "strategy": "self_consistency",
                            "num_paths": 5,
                            "consensus_score": 0.8,
                            "answer_distribution": {"4": 4, "5": 1},
                            "selected_answer": "4",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 200,
                    "total_tokens": 250,
                },
            }
        }


class ModelInfo(BaseModel):
    """Model information in OpenAI format."""

    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    owned_by: str = Field(..., description="Organization that owns the model")


class ModelsResponse(BaseModel):
    """List of available models in OpenAI format."""

    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of models")


class ErrorDetail(BaseModel):
    """Error detail."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: Optional[str] = Field(None, description="Parameter that caused error")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""

    error: ErrorDetail = Field(..., description="Error details")
