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

    Supports TTS-specific parameters via additional fields:
    - tts_strategy: Which TTS strategy to use (deepconf, online_best_of_n)
    - tts_config: Strategy-specific configuration
    """

    # Standard OpenAI parameters
    model: str = Field(..., description="Model to use (e.g., openai/gpt-4o-mini)")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)
    stream: bool = Field(
        default=False, description="Stream responses (not yet supported)"
    )

    # TTS-specific parameters (optional, for advanced usage)
    tts_strategy: Optional[str] = Field(
        default="deepconf", description="TTS strategy: deepconf, online_best_of_n, etc."
    )
    tts_mode: Optional[str] = Field(
        default="offline", description="DeepConf mode: offline or online"
    )
    tts_budget: Optional[int] = Field(
        default=8, description="Number of reasoning traces to generate", ge=1
    )
    tts_filter_method: Optional[str] = Field(
        default="top5", description="Trace filtering method: top5, top10, etc."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Solve: 2+2=?"}],
                "temperature": 0.7,
                "tts_strategy": "deepconf",
                "tts_mode": "offline",
                "tts_budget": 8,
            }
        }


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: str = Field(..., description="Why generation stopped")

    # TTS-specific metadata (optional)
    tts_metadata: Optional[Dict[str, Any]] = Field(
        None, description="TTS strategy metadata (confidence, num_traces, etc.)"
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
                            "content": "The answer is 4. Let me explain step by step...",
                        },
                        "finish_reason": "stop",
                        "tts_metadata": {
                            "strategy": "deepconf",
                            "num_traces": 8,
                            "confidence": 16.2,
                            "agreement": 1.0,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 50,
                    "total_tokens": 60,
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
