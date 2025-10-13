"""
API Request Models.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message format."""

    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class DeepConfConfig(BaseModel):
    """DeepConf strategy configuration."""

    mode: str = Field(
        default="offline", description="DeepConf mode: 'offline' or 'online'"
    )
    budget: int = Field(default=8, description="Number of traces to generate", ge=1)
    warmup_traces: int = Field(
        default=4, description="Number of warmup traces (online mode)", ge=1
    )
    total_budget: int = Field(
        default=10, description="Total budget for online mode", ge=1
    )
    confidence_percentile: int = Field(
        default=90, description="Confidence percentile threshold", ge=0, le=100
    )
    window_size: int = Field(
        default=2048, description="Sliding window size for confidence computation"
    )
    filter_method: str = Field(
        default="top5", description="Filtering method (top5, top10, etc.)"
    )
    temperature: float = Field(
        default=0.7, description="Sampling temperature", ge=0.0, le=2.0
    )
    top_p: float = Field(
        default=1.0, description="Nucleus sampling threshold", ge=0.0, le=1.0
    )
    max_tokens: int = Field(default=512, description="Maximum tokens to generate", ge=1)
    top_logprobs: int = Field(
        default=20, description="Number of top logprobs to request", ge=0, le=20
    )


class GenerateRequest(BaseModel):
    """Request model for /generate endpoint."""

    prompt: Union[str, List[Message]] = Field(
        ..., description="Input prompt (string or chat messages)"
    )
    strategy: str = Field(
        default="deepconf",
        description="TTS strategy to use (deepconf, online_best_of_n)",
    )
    model: str = Field(
        default="openai/gpt-4o-mini", description="Model name to use for generation"
    )
    provider: str = Field(
        default="openrouter", description="API provider (openrouter, openai)"
    )
    strategy_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Strategy-specific configuration"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Solve step by step: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "strategy": "deepconf",
                "model": "openai/gpt-4o-mini",
                "provider": "openrouter",
                "strategy_config": {
                    "mode": "offline",
                    "budget": 8,
                    "filter_method": "top5",
                },
            }
        }
