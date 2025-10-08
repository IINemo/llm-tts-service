"""
API Response Models.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepCandidate(BaseModel):
    """Single step candidate."""

    text: str = Field(..., description="Step text content")
    score: Optional[float] = Field(None, description="Validity score")


class TrajectoryMetadata(BaseModel):
    """Metadata about the trajectory generation."""

    strategy: str = Field(..., description="Strategy used")
    model: str = Field(..., description="Model used")
    num_steps: int = Field(..., description="Number of steps generated")
    avg_validity_score: Optional[float] = Field(
        None, description="Average validity score across steps"
    )
    completed: bool = Field(
        ..., description="Whether trajectory completed successfully"
    )
    extra: Optional[Dict[str, Any]] = Field(
        None, description="Strategy-specific metadata"
    )


class GenerateResponse(BaseModel):
    """Response model for /generate endpoint."""

    trajectory: str = Field(..., description="Generated reasoning trajectory")
    steps: List[StepCandidate] = Field(
        default_factory=list, description="Individual reasoning steps"
    )
    metadata: TrajectoryMetadata = Field(..., description="Generation metadata")
    answer: Optional[str] = Field(None, description="Extracted answer (if available)")

    class Config:
        json_schema_extra = {
            "example": {
                "trajectory": "To solve this problem, let's break it down step by step...",
                "steps": [
                    {"text": "Janet's ducks lay 16 eggs per day", "score": 16.5},
                    {"text": "She eats 3 for breakfast", "score": 15.8},
                ],
                "metadata": {
                    "strategy": "deepconf",
                    "model": "openai/gpt-4o-mini",
                    "num_steps": 1,
                    "avg_validity_score": 16.2,
                    "completed": True,
                    "extra": {
                        "mode": "offline",
                        "num_traces": 8,
                        "filtered_traces": 5,
                        "agreement": 1.0,
                    },
                },
                "answer": "18",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ModelInfo(BaseModel):
    """Model information."""

    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Provider")
    supports_logprobs: bool = Field(..., description="Whether model supports logprobs")


class ModelsResponse(BaseModel):
    """List of available models."""

    models: List[ModelInfo] = Field(..., description="Available models")


class StrategyInfo(BaseModel):
    """Strategy information."""

    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    config_schema: Optional[Dict[str, Any]] = Field(
        None, description="Configuration schema"
    )


class StrategiesResponse(BaseModel):
    """List of available strategies."""

    strategies: List[StrategyInfo] = Field(..., description="Available strategies")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
