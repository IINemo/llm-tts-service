"""Routes for the Visual Debugger demo."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from service_app.core.visual_debugger_demo import (
    build_single_sample_payload,
    get_demo_scenario,
    list_demo_scenarios,
)

router = APIRouter()

_DEBUGGER_HTML_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "debugger" / "index.html"
)


class DebuggerSingleSampleRequest(BaseModel):
    """Run one custom sample through the debugger strategy/scorer matrix."""

    question: str = Field(..., min_length=1)
    gold_answer: str = Field(..., min_length=1)
    shared_prompt: str = Field(default="")
    budget: Optional[int] = Field(default=None, ge=1)
    provider: str = Field(default="openrouter")
    model_id: str = Field(default="openai/gpt-4o-mini")
    api_key: str = Field(default="")


@router.get("/debugger", include_in_schema=False)
@router.get("/debugger/", include_in_schema=False)
def visual_debugger_page() -> FileResponse:
    """Serve the Visual Debugger demo page."""
    if not _DEBUGGER_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Debugger UI is not available")
    return FileResponse(_DEBUGGER_HTML_PATH)


@router.get("/v1/debugger/demo/scenarios")
def list_visual_debugger_scenarios():
    """List available demo scenarios for the visual debugger."""
    scenarios = list_demo_scenarios()
    return {"scenarios": scenarios}


@router.get("/v1/debugger/demo/scenarios/{scenario_id}")
def get_visual_debugger_scenario(
    scenario_id: str,
    budget: Optional[int] = Query(default=None, ge=1),
):
    """Get one scenario payload with strategy runs resolved for a target budget."""
    try:
        payload = get_demo_scenario(scenario_id=scenario_id, budget=budget)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return payload


@router.post("/v1/debugger/demo/run-single")
def run_visual_debugger_single_sample(request: DebuggerSingleSampleRequest):
    """Build one strategy-by-scorer debugger payload for a custom sample."""
    return build_single_sample_payload(
        question=request.question,
        gold_answer=request.gold_answer,
        shared_prompt=request.shared_prompt,
        budget=request.budget,
        provider=request.provider,
        model_id=request.model_id,
        api_key=request.api_key,
        scenario_id="custom_1",
        scenario_title="Single Example",
        scenario_description=(
            "Custom single-sample run across baseline, beam search, "
            "online/offline best-of-n, and self-consistency with all scorers."
        ),
        input_source="custom_single",
    )
