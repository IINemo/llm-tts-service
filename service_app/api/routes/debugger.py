"""Routes for the Visual Debugger demo."""

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from service_app.core.visual_debugger_demo import (
    get_advanced_config_template,
    get_debugger_runtime_health,
    get_demo_scenario,
    list_demo_scenarios,
    validate_model_capabilities,
)

router = APIRouter()

_DEBUGGER_HTML_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "debugger" / "index.html"
)


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
    try:
        scenarios = list_demo_scenarios()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"scenarios": scenarios}


@router.get("/v1/debugger/demo/scenarios/{scenario_id}")
def get_visual_debugger_scenario(
    scenario_id: str,
    budget: Optional[int] = Query(default=None, ge=1),
) -> Dict[str, Any]:
    """Get one scenario payload with strategy runs resolved for a target budget."""
    try:
        payload = get_demo_scenario(scenario_id=scenario_id, budget=budget)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return payload


@router.get("/v1/debugger/demo/health")
def get_visual_debugger_runtime_health() -> Dict[str, Any]:
    """Report runtime dependency health before running debugger strategies."""
    return get_debugger_runtime_health()


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
