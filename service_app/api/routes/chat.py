"""
OpenAI-compatible /v1/chat/completions endpoint.

Supports three URL patterns (all hit the same handler):
  POST /v1/chat/completions                          — strategy & scorer from body
  POST /v1/{strategy}/chat/completions               — strategy from URL
  POST /v1/{strategy}/{scorer}/chat/completions      — strategy & scorer from URL
"""

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from service_app.api.models.openai_compat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ErrorResponse,
    Usage,
)
from service_app.core.debugger_events import (
    StrategyProgressHandler,
    convert_strategy_result_to_debugger_run,
)
from llm_tts.strategies.strategy_base import StrategyCancelled
from service_app.core.strategy_manager import strategy_manager
from service_app.core.visual_debugger_demo import (
    SUPPORTED_SCORERS,
    SUPPORTED_STRATEGIES,
)

log = logging.getLogger(__name__)

router = APIRouter()

# Active streaming requests: request_id → cancel Event
_active_requests: Dict[str, threading.Event] = {}

_VALID_STRATEGIES = {"self_consistency", "offline_bon", "online_bon", "beam_search"}
_VALID_SCORERS = {"entropy", "perplexity", "sequence_prob", "prm"}


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


_completion_responses = {
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


def _find_strategy_meta(strategy_id: str) -> Dict[str, Any]:
    for s in SUPPORTED_STRATEGIES:
        if s["id"] == strategy_id:
            return s
    return {"id": strategy_id, "name": strategy_id, "family": "unknown"}


def _find_scorer_meta(scorer_id: str) -> Optional[Dict[str, Any]]:
    for s in SUPPORTED_SCORERS:
        if s["id"] == scorer_id:
            return s
    return None


# ------------------------------------------------------------------
# Map library strategy_type to the id used in SUPPORTED_STRATEGIES
# ------------------------------------------------------------------
_STRATEGY_TYPE_TO_ID = {
    "self_consistency": "self_consistency",
    "offline_bon": "offline_best_of_n",
    "online_bon": "online_best_of_n",
    "beam_search": "beam_search",
}


@router.post(
    "/v1/{url_strategy}/{url_scorer}/chat/completions",
    response_model=ChatCompletionResponse,
    responses=_completion_responses,
    include_in_schema=False,
)
@router.post(
    "/v1/{url_strategy}/chat/completions",
    response_model=ChatCompletionResponse,
    responses=_completion_responses,
    include_in_schema=False,
)
@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses=_completion_responses,
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    url_strategy: Optional[str] = None,
    url_scorer: Optional[str] = None,
):
    """
    Create a chat completion with TTS strategy.

    Strategy and scorer can be specified in **three ways** (highest priority first):

    1. **URL path** — `base_url="http://host:8001/v1/beam_search/prm"`
    2. **Request body** — `extra_body={"tts_strategy": "beam_search", "tts_scorer": "prm"}`
    3. **Defaults** — strategy=self_consistency, scorer=entropy

    URL path segments override body parameters when both are present.
    """
    try:
        # URL path segments override body params
        if url_strategy:
            if url_strategy not in _VALID_STRATEGIES:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": f"Unknown strategy in URL: '{url_strategy}'. "
                            f"Valid: {', '.join(sorted(_VALID_STRATEGIES))}",
                            "type": "invalid_request_error",
                            "param": "strategy",
                        }
                    },
                )
            request.tts_strategy = url_strategy

        if url_scorer:
            if url_scorer not in _VALID_SCORERS:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": f"Unknown scorer in URL: '{url_scorer}'. "
                            f"Valid: {', '.join(sorted(_VALID_SCORERS))}",
                            "type": "invalid_request_error",
                            "param": "scorer",
                        }
                    },
                )
            request.tts_scorer = url_scorer

        log.info(f"Received chat completion request for model: {request.model}")
        log.info(
            f"TTS strategy: {request.tts_strategy} (from_url={url_strategy is not None})"
        )

        # SSE streaming path
        if request.stream:
            return _handle_streaming(request)

        # Non-streaming path
        return _handle_sync(request)

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


# ------------------------------------------------------------------
# Cancel endpoint
# ------------------------------------------------------------------


@router.post("/v1/chat/cancel/{request_id}")
async def cancel_chat_completion(request_id: str):
    """Cancel an in-progress streaming chat completion."""
    event = _active_requests.get(request_id)
    if event is None:
        raise HTTPException(
            status_code=404, detail="Request not found or already finished"
        )
    event.set()
    log.info(f"Cancel requested for {request_id}")
    return {"status": "cancelled", "request_id": request_id}


# ------------------------------------------------------------------
# Synchronous (non-streaming) path
# ------------------------------------------------------------------


def _handle_sync(request: ChatCompletionRequest) -> ChatCompletionResponse:
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    strategy_type = request.tts_strategy or "self_consistency"
    is_vllm_strategy = strategy_type in ("offline_bon", "online_bon", "beam_search")

    strategy_config = _build_strategy_config(request)

    log.info(f"Creating strategy: {strategy_type}")
    strategy = strategy_manager.create_strategy(
        strategy_type=strategy_type,
        model_name=request.model,
        strategy_config=strategy_config,
    )

    log.info("Generating trajectory...")
    start_time = time.time()

    if is_vllm_strategy:
        results = strategy.generate_trajectories_batch([messages])
        result = results[0]
    else:
        result = strategy.generate_trajectory(
            input_chat=messages,
            sample_idx=0,
        )

    elapsed_time = time.time() - start_time
    log.info(f"Trajectory generated in {elapsed_time:.2f}s")

    return _build_response(
        request, result, elapsed_time, strategy_type, is_vllm_strategy
    )


# ------------------------------------------------------------------
# SSE streaming path
# ------------------------------------------------------------------


def _handle_streaming(request: ChatCompletionRequest) -> StreamingResponse:
    """Run strategy in a thread, stream progress events via SSE."""
    request_id = uuid.uuid4().hex[:16]
    cancel_event = threading.Event()
    _active_requests[request_id] = cancel_event

    result_q: queue.Queue = queue.Queue()
    progress_state: Dict[str, Any] = {"message": None}

    def _progress_callback(message: str) -> None:
        progress_state["message"] = message

    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    strategy_type = request.tts_strategy or "self_consistency"
    is_vllm_strategy = strategy_type in ("offline_bon", "online_bon", "beam_search")
    strategy_config = _build_strategy_config(request)

    def _run() -> None:
        # Attach progress handler to strategy + PRM loggers
        loggers_to_monitor = [
            logging.getLogger("llm_tts.strategies"),
            logging.getLogger("service_app.core.prm_scorer_factory"),
        ]
        handler = StrategyProgressHandler(_progress_callback)
        prev_levels = {}
        for lgr in loggers_to_monitor:
            prev_levels[lgr.name] = lgr.level
            lgr.setLevel(logging.DEBUG)
            lgr.addHandler(handler)
        try:
            strategy = strategy_manager.create_strategy(
                strategy_type=strategy_type,
                model_name=request.model,
                strategy_config=strategy_config,
                cancel_event=cancel_event,
            )

            start_time = time.time()
            if is_vllm_strategy:
                results = strategy.generate_trajectories_batch([messages])
                result = results[0]
            else:
                result = strategy.generate_trajectory(
                    input_chat=messages,
                    sample_idx=0,
                )

            elapsed_time = time.time() - start_time
            response = _build_response(
                request,
                result,
                elapsed_time,
                strategy_type,
                is_vllm_strategy,
            )
            result_q.put(
                {
                    "type": "complete",
                    "data": response.model_dump(mode="json"),
                }
            )
        except StrategyCancelled:
            log.info(f"Strategy cancelled for request {request_id}")
            result_q.put({"type": "cancelled"})
        except Exception as exc:
            log.error(f"Strategy streaming error: {exc}", exc_info=True)
            result_q.put({"type": "error", "message": str(exc)})
        finally:
            _active_requests.pop(request_id, None)
            for lgr in loggers_to_monitor:
                lgr.removeHandler(handler)
                lgr.setLevel(prev_levels[lgr.name])

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def _event_stream():
        # Emit request_id so the client can cancel this request
        yield f"data: {json.dumps({'type': 'started', 'request_id': request_id})}\n\n"

        last_sent: Optional[str] = None
        deadline = asyncio.get_event_loop().time() + 300  # 5 min max
        while asyncio.get_event_loop().time() < deadline:
            try:
                event = result_q.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
                return
            except queue.Empty:
                pass
            current = progress_state["message"]
            if current and current != last_sent:
                last_sent = current
                yield f"data: {json.dumps({'type': 'progress', 'message': current})}\n\n"
            await asyncio.sleep(0.25)
        cancel_event.set()  # signal background thread to stop and clean up
        yield f"data: {json.dumps({'type': 'error', 'message': 'Strategy execution timed out (5 min)'})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_strategy_config(request: ChatCompletionRequest) -> Dict[str, Any]:
    strategy_type = request.tts_strategy or "self_consistency"
    is_vllm_strategy = strategy_type in ("offline_bon", "online_bon", "beam_search")

    # When an API key is provided the request goes through the API backend,
    # not local vLLM, so default to openrouter (not vllm).
    if request.provider:
        provider = request.provider
    elif request.tts_api_key or request.model_base_url:
        provider = "openrouter"
    elif is_vllm_strategy:
        provider = "vllm"
    else:
        provider = "openrouter"

    return {
        "provider": provider,
        "model_base_url": request.model_base_url,
        "tts_api_key": request.tts_api_key,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens or 4096,
        "num_paths": request.num_paths or 5,
        "budget": request.num_paths or 5,
        # vLLM TTS params
        "scorer_type": request.tts_scorer or "entropy",
        "num_trajectories": request.tts_num_trajectories,
        "candidates_per_step": request.tts_candidates_per_step,
        "beam_size": request.tts_beam_size,
        "max_steps": request.tts_max_steps,
        "score_aggregation": request.tts_score_aggregation,
        "window_size": request.tts_window_size,
    }


def _build_response(
    request: ChatCompletionRequest,
    result: Dict[str, Any],
    elapsed_time: float,
    strategy_type: str,
    is_vllm_strategy: bool,
) -> ChatCompletionResponse:
    trajectory = result.get("trajectory", "")
    if not trajectory:
        raise ValueError("Strategy returned empty trajectory")

    metadata = result.get("metadata", {})
    metadata["elapsed_time"] = round(elapsed_time, 2)
    metadata["selected_answer"] = result.get("extracted_answer", "")
    if is_vllm_strategy:
        metadata["strategy"] = strategy_type
        metadata["completed"] = result.get("completed", False)
        metadata["completion_reason"] = result.get("completion_reason")

        if "validity_scores" in result:
            metadata["validity_scores"] = result["validity_scores"]
        if "aggregated_score" in result:
            metadata["aggregated_score"] = result["aggregated_score"]

        steps = result.get("steps", [])
        metadata["reasoning_steps"] = len(steps)
        metadata["steps"] = [
            {
                "text": getattr(s, "text", s) if not isinstance(s, str) else s,
                "score": (
                    result["validity_scores"][i]
                    if i < len(result.get("validity_scores", []))
                    else None
                ),
            }
            for i, s in enumerate(steps)
        ]

        token_stats = result.get("token_stats", {})
        if token_stats:
            metadata["token_stats"] = {
                "input_tokens": token_stats.get("input_tokens", 0),
                "output_tokens": token_stats.get("output_tokens", 0),
                "tflops": token_stats.get("tflops"),
            }

        if "all_trajectories" in result:
            metadata["all_trajectories"] = [
                {"text": t, "score": s}
                for t, s in zip(
                    result["all_trajectories"],
                    result.get("all_scores", []),
                )
            ]

    # Verbose / debugger mode: include events + run structure
    if request.tts_verbose:
        strategy_meta_id = _STRATEGY_TYPE_TO_ID.get(strategy_type, strategy_type)
        strategy_meta = _find_strategy_meta(strategy_meta_id)
        scorer_meta = (
            _find_scorer_meta(request.tts_scorer) if request.tts_scorer else None
        )

        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        prompt_text = " ".join(msg["content"] for msg in messages)

        run = convert_strategy_result_to_debugger_run(
            strategy=strategy_meta,
            scorer=scorer_meta,
            strategy_result=result,
            budget=request.num_paths or 8,
            latency_ms=int(elapsed_time * 1000),
            model_config={
                "provider": request.provider or "openrouter",
                "model_id": request.model,
            },
            generation_config={
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens or 4096,
            },
            strategy_config={},
            scorer_config={},
            has_gold_answer=False,
            gold_answer="",
        )
        metadata["debugger_run"] = run

    # Estimate token usage
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    prompt_text = " ".join(msg["content"] for msg in messages)
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = metadata.get("total_tokens", estimate_tokens(trajectory))

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
