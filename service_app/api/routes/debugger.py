"""Routes for the Visual Debugger demo."""

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from service_app.core.visual_debugger_demo import (
    build_single_sample_payload,
    get_advanced_config_template,
    get_demo_scenario,
    list_demo_scenarios,
    validate_model_capabilities,
)

router = APIRouter()

_DEBUGGER_HTML_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "debugger" / "index.html"
)


class DebuggerSingleSampleRequest(BaseModel):
    """Run one custom sample through one selected strategy and optional scorer."""

    question: str = Field(..., min_length=1)
    gold_answer: Optional[str] = Field(default=None)
    shared_prompt: str = Field(default="")
    budget: Optional[int] = Field(default=None, ge=1)
    provider: str = Field(default="openrouter")
    model_id: str = Field(default="openai/gpt-4o-mini")
    api_key: str = Field(default="")
    strategy_id: str = Field(..., min_length=1)
    scorer_id: Optional[str] = Field(default=None)
    advanced_config_yaml: Optional[str] = Field(default=None)


class DebuggerValidateModelRequest(BaseModel):
    """Validate model capability flags used to gate strategies/scorers."""

    provider: str = Field(default="openrouter")
    model_id: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)


@router.get("/debugger", include_in_schema=False)
@router.get("/debugger/", include_in_schema=False)
def visual_debugger_page() -> FileResponse:
    """Serve the Visual Debugger demo page."""
    if not _DEBUGGER_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Debugger UI is not available")
    return FileResponse(_DEBUGGER_HTML_PATH)


@router.get("/v1/debugger/demo/scenarios")
def list_visual_debugger_scenarios() -> Dict[str, Any]:
    """List available demo scenarios for the visual debugger."""
    scenarios = list_demo_scenarios()
    return {"scenarios": scenarios}


@router.get("/v1/debugger/demo/scenarios/{scenario_id}")
def get_visual_debugger_scenario(
    scenario_id: str,
    budget: Optional[int] = Query(default=None, ge=1),
) -> Dict[str, Any]:
    """Get one scenario payload with strategy runs resolved for a target budget."""
    try:
        payload = get_demo_scenario(scenario_id=scenario_id, budget=budget)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return payload


@router.post("/v1/debugger/demo/run-single")
def run_visual_debugger_single_sample(request: DebuggerSingleSampleRequest):
    """Build one debugger payload for the selected strategy and optional scorer."""
    try:
        return build_single_sample_payload(
            question=request.question,
            gold_answer=request.gold_answer,
            shared_prompt=request.shared_prompt,
            budget=request.budget,
            provider=request.provider,
            model_id=request.model_id,
            api_key=request.api_key,
            strategy_id=request.strategy_id,
            scorer_id=request.scorer_id,
            advanced_config_yaml=request.advanced_config_yaml,
            scenario_id="custom_1",
            scenario_title="Single Example",
            scenario_description=(
                "Custom single-sample run with selected strategy and optional scorer."
            ),
            input_source="custom_single",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/debugger/demo/validate-model")
def validate_visual_debugger_model(request: DebuggerValidateModelRequest):
    """Validate model capabilities and return available strategies/scorers."""
    try:
        return validate_model_capabilities(
            provider=request.provider,
            model_id=request.model_id,
            api_key=request.api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/v1/debugger/demo/advanced-config/template")
def get_visual_debugger_advanced_config_template(
    strategy_id: str = Query(..., min_length=1),
    scorer_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return YAML template for generation/strategy/scorer advanced config."""
    try:
        return get_advanced_config_template(
            strategy_id=strategy_id,
            scorer_id=scorer_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
