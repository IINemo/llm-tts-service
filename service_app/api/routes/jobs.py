"""
Async Job Execution API - Allows polling for long-running ToT jobs.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from service_app.core.job_manager import job_manager
from service_app.core.strategy_manager import strategy_manager

log = logging.getLogger(__name__)

router = APIRouter()


class JobCreateRequest(BaseModel):
    """Request to create a new async job."""

    model: str = Field(..., description="Model to use (e.g., openai/gpt-4o-mini)")
    prompt: str = Field(..., description="The user's question/prompt")
    tts_strategy: Optional[str] = Field(
        default="tree_of_thoughts",
        description="TTS strategy: tree_of_thoughts, tot, deepconf, etc.",
    )

    # Strategy-specific parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)

    # Tree-of-Thoughts parameters
    tot_mode: Optional[str] = Field(default="generic")
    tot_method_generate: Optional[str] = Field(default="propose")
    tot_beam_width: Optional[int] = Field(default=3, ge=1)
    tot_n_generate_sample: Optional[int] = Field(default=5, ge=1)
    tot_steps: Optional[int] = Field(default=4, ge=1)
    tot_max_tokens_per_step: Optional[int] = Field(default=150, ge=1)


class JobCreateResponse(BaseModel):
    """Response when job is created."""

    job_id: str
    status: str
    message: str = (
        "Job created successfully. Use GET /v1/jobs/{job_id} to check progress."
    )


class ProgressInfo(BaseModel):
    """Progress information."""

    current_step: int
    total_steps: int
    nodes_explored: int
    api_calls: int


class IntermediateTree(BaseModel):
    """Intermediate reasoning tree state."""

    nodes: List[Dict]
    edges: List[Dict]
    question: str


class JobResult(BaseModel):
    """Job execution result."""

    trajectory: str
    reasoning_tree: Optional[Dict]
    metadata: Optional[Dict]


class JobStatusResponse(BaseModel):
    """Job status response."""

    job_id: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: ProgressInfo
    result: Optional[JobResult]
    error: Optional[str]
    intermediate_tree: Optional[IntermediateTree]


def progress_callback(job, update: Dict):
    """Called by strategy to report progress."""
    if "step" in update:
        job.current_step = update["step"]

    if "nodes_explored" in update:
        job.nodes_explored = update["nodes_explored"]

    if "api_calls" in update:
        job.api_calls = update["api_calls"]

    # Update intermediate tree
    if "node" in update:
        job.intermediate_nodes.append(update["node"])

    if "edge" in update:
        job.intermediate_edges.append(update["edge"])

    log.info(
        f"[Job {job.job_id}] Progress update: step={job.current_step}/{job.total_steps}, nodes={job.nodes_explored}, api_calls={job.api_calls}"
    )


@router.post("/v1/jobs", response_model=JobCreateResponse)
async def create_job(request: JobCreateRequest):
    """
    Create a new async ToT job.

    Returns job_id immediately. Use GET /v1/jobs/{job_id} to poll for progress.

    Example:
    ```
    POST /v1/jobs
    {
        "model": "openai/gpt-4o-mini",
        "prompt": "What is 2+2?",
        "tts_strategy": "tree_of_thoughts",
        "tot_beam_width": 3,
        "tot_steps": 3
    }
    ```
    """
    try:
        log.info(f"Creating job for model: {request.model}")

        # Build strategy config
        strategy_config: Dict = {
            "provider": "openrouter",
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens or 512,
        }

        if request.tts_strategy in ["tree_of_thoughts", "tot"]:
            strategy_config.update(
                {
                    "mode": request.tot_mode or "generic",
                    "method_generate": request.tot_method_generate or "propose",
                    "beam_width": request.tot_beam_width or 3,
                    "n_generate_sample": request.tot_n_generate_sample or 5,
                    "steps": request.tot_steps or 4,
                    "max_tokens_per_step": request.tot_max_tokens_per_step or 150,
                    "n_threads": 4,
                }
            )

        # Create job
        job = job_manager.create_job(
            prompt=request.prompt,
            strategy_type=request.tts_strategy or "tree_of_thoughts",
            model_name=request.model,
            strategy_config=strategy_config,
        )

        # Start job execution in background
        job_manager.start_job(
            job_id=job.job_id,
            strategy_factory=strategy_manager.create_strategy,
            progress_callback=progress_callback,
        )

        return JobCreateResponse(
            job_id=job.job_id,
            status=job.status.value,
        )

    except Exception as e:
        log.error(f"Error creating job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "job_creation_failed",
                }
            },
        )


@router.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get job status and progress.

    Poll this endpoint periodically to get real-time updates.

    Example:
    ```
    GET /v1/jobs/abc-123-def
    ```

    Response includes:
    - status: pending, running, completed, or failed
    - progress: current step, nodes explored, API calls
    - intermediate_tree: nodes and edges discovered so far (updated in real-time)
    - result: final trajectory and reasoning tree (when completed)
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Job {job_id} not found",
                    "type": "not_found",
                    "code": "job_not_found",
                }
            },
        )

    return JobStatusResponse(**job.to_dict())
