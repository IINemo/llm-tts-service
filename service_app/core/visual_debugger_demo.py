"""Visual debugger payload helpers."""

from __future__ import annotations

import importlib
import json
import logging
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

DEFAULT_BUDGET = 8
DEBUGGER_BUDGETS = [4, 8, 12]

_CACHED_EXAMPLES_PATH = (
    Path(__file__).resolve().parents[1] / "static" / "debugger" / "cached_examples.json"
)
_DEBUGGER_CONFIG_ROOT = (
    Path(__file__).resolve().parents[1] / "static" / "debugger" / "config"
)
_DEFAULT_GENERATION_CONFIG_PATH = _DEBUGGER_CONFIG_ROOT / "generation" / "default.yaml"
_STRATEGY_CONFIG_PATHS = {
    "baseline": _DEBUGGER_CONFIG_ROOT / "strategy" / "baseline.yaml",
    "beam_search": _DEBUGGER_CONFIG_ROOT / "strategy" / "beam_search.yaml",
    "adaptive": _DEBUGGER_CONFIG_ROOT / "strategy" / "adaptive.yaml",
    "online_best_of_n": _DEBUGGER_CONFIG_ROOT / "strategy" / "online_best_of_n.yaml",
    "offline_best_of_n": _DEBUGGER_CONFIG_ROOT / "strategy" / "offline_best_of_n.yaml",
    "self_consistency": _DEBUGGER_CONFIG_ROOT / "strategy" / "self_consistency.yaml",
}
_SCORER_CONFIG_PATHS = {
    "prm": _DEBUGGER_CONFIG_ROOT / "scorer" / "prm.yaml",
    "sequence_prob": _DEBUGGER_CONFIG_ROOT / "scorer" / "sequence_prob.yaml",
    "perplexity": _DEBUGGER_CONFIG_ROOT / "scorer" / "perplexity.yaml",
    "entropy": _DEBUGGER_CONFIG_ROOT / "scorer" / "entropy.yaml",
}

SUPPORTED_STRATEGIES: List[Dict[str, Any]] = [
    {
        "id": "baseline",
        "name": "Baseline (Raw CoT)",
        "family": "single_pass",
        "summary": "Single-pass raw chain-of-thought without search or reranking.",
        "requires_scorer": False,
        "requires_logprobs": False,
        "requires_prefill": False,
    },
    {
        "id": "beam_search",
        "name": "Beam Search (ToT)",
        "family": "tree_search",
        "summary": "Tree-of-thought expansion with beam pruning.",
        "requires_scorer": True,
        "requires_logprobs": False,
        "requires_prefill": True,
    },
    # TODO: re-enable once adaptive visualization is fixed
    # {
    #     "id": "adaptive",
    #     "name": "Adaptive Best-of-N",
    #     "family": "reranking",
    #     "summary": "Online best-of-n with adaptive scaling across steps.",
    #     "requires_scorer": True,
    #     "requires_logprobs": False,
    #     "requires_prefill": True,
    # },
    {
        "id": "online_best_of_n",
        "name": "Online Best-of-N",
        "family": "reranking",
        "summary": "Iterative candidate generation with stepwise reranking.",
        "requires_scorer": True,
        "requires_logprobs": False,
        "requires_prefill": True,
    },
    {
        "id": "offline_best_of_n",
        "name": "Offline Best-of-N",
        "family": "reranking",
        "summary": "Generate full trajectories first, then rerank at the end.",
        "requires_scorer": True,
        "requires_logprobs": False,
        "requires_prefill": False,
    },
    {
        "id": "self_consistency",
        "name": "Self-Consistency",
        "family": "sample_and_vote",
        "summary": "Sample diverse trajectories and select by answer consensus.",
        "requires_scorer": False,
        "builtin_scorer": "Consensus score",
        "requires_logprobs": False,
        "requires_prefill": False,
    },
]

SUPPORTED_SCORERS: List[Dict[str, Any]] = [
    {
        "id": "prm",
        "name": "PRM",
        "direction": "higher_better",
        "summary": "Process Reward Model trajectory quality score.",
        "requires_logprobs": False,
    },
    {
        "id": "sequence_prob",
        "name": "Sequence Prob",
        "direction": "higher_better",
        "summary": "Cumulative sequence probability from token logprobs.",
        "requires_logprobs": True,
    },
    {
        "id": "perplexity",
        "name": "Perplexity",
        "direction": "lower_better",
        "summary": "Per-token perplexity estimated from generation logprobs.",
        "requires_logprobs": True,
    },
    {
        "id": "entropy",
        "name": "Entropy",
        "direction": "lower_better",
        "summary": "Mean token entropy of decoded reasoning steps.",
        "requires_logprobs": True,
    },
]

_PROVIDER_BASE_URLS = {
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
}


@dataclass
class ModelValidationResult:
    supports_logprobs: bool
    supports_prefill: bool
    supports_logprobs_reason: str = ""
    supports_prefill_reason: str = ""


def list_demo_scenarios() -> List[Dict[str, Any]]:
    """Return cached demo scenarios from JSON for the selector."""
    bundle = _load_cached_examples_bundle()
    return deepcopy(bundle["scenarios"])


def get_demo_scenario(
    scenario_id: str,
    budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Return one cached demo payload resolved for a target budget."""
    bundle = _load_cached_examples_bundle()
    scenario_payloads = bundle["payloads"].get(scenario_id)
    if not isinstance(scenario_payloads, dict):
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    available_budgets = _collect_available_budgets(scenario_payloads)
    if not available_budgets:
        raise KeyError(f"Scenario has no budgets: {scenario_id}")

    selected_budget = _pick_budget(budget, available_budgets)
    payload = scenario_payloads.get(str(selected_budget))
    if not isinstance(payload, dict):
        raise KeyError(
            f"Scenario payload missing for budget {selected_budget}: {scenario_id}"
        )

    payload_copy = deepcopy(payload)
    payload_copy["available_budgets"] = payload_copy.get(
        "available_budgets", available_budgets
    )
    payload_copy["selected_budget"] = selected_budget
    return payload_copy


def get_available_strategy_and_scorer_options(
    supports_logprobs: bool,
    supports_prefill: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    """Filter strategy/scorer options using model capability checks."""
    strategies = [
        item
        for item in SUPPORTED_STRATEGIES
        if (not item.get("requires_logprobs") or supports_logprobs)
        and (not item.get("requires_prefill") or supports_prefill)
    ]
    scorers = [
        item
        for item in SUPPORTED_SCORERS
        if not item.get("hidden")
        and (not item.get("requires_logprobs") or supports_logprobs)
    ]
    return {
        "strategies": deepcopy(strategies),
        "scorers": deepcopy(scorers),
    }


def validate_model_capabilities(
    provider: str,
    model_id: str,
    api_key: str,
) -> Dict[str, Any]:
    """Validate model capabilities used by the debugger setup UI."""
    provider_value = str(provider or "").strip().lower()
    model_id_value = str(model_id or "").strip()
    api_key_value = str(api_key or "").strip()

    # Allow OpenRouter-style "openai/gpt-4o-mini" with the openai provider
    if provider_value == "openai" and model_id_value.startswith("openai/"):
        model_id_value = model_id_value[len("openai/"):]

    if provider_value not in _PROVIDER_BASE_URLS:
        raise ValueError("Provider must be one of: openai, openrouter.")
    if not model_id_value:
        raise ValueError("Model ID is required.")
    if not api_key_value:
        raise ValueError("API key is required.")

    validation = _probe_model_capabilities(
        provider=provider_value,
        model_id=model_id_value,
        api_key=api_key_value,
    )
    available = get_available_strategy_and_scorer_options(
        supports_logprobs=validation.supports_logprobs,
        supports_prefill=validation.supports_prefill,
    )

    return {
        "provider": provider_value,
        "model_id": model_id_value,
        "api_key_masked": _mask_api_key(api_key_value),
        "supports_logprobs": validation.supports_logprobs,
        "supports_prefill": validation.supports_prefill,
        "supports_logprobs_reason": validation.supports_logprobs_reason,
        "supports_prefill_reason": validation.supports_prefill_reason,
        "strategies": available["strategies"],
        "scorers": available["scorers"],
    }


def get_advanced_config_template(
    strategy_id: str,
    scorer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return default advanced config template as parsed dict and YAML text."""
    strategy = _find_strategy(strategy_id)
    scorer: Optional[Dict[str, Any]] = None
    if scorer_id:
        scorer = _find_scorer(scorer_id)

    config = _build_advanced_config_template_dict(
        strategy_id=strategy["id"],
        scorer_id=scorer.get("id") if scorer else None,
    )
    return {
        "strategy_id": strategy["id"],
        "scorer_id": scorer.get("id") if scorer else None,
        "config": config,
        "config_yaml": _dump_yaml(config),
    }


def get_debugger_runtime_health() -> Dict[str, Any]:
    """Report runtime dependency health for real debugger execution."""
    checks = [
        _dependency_check(
            name="core_runtime",
            required=[
                "llm_tts.early_stopping:BoundaryEarlyStopping",
                "llm_tts.generators.api:StepCandidateGeneratorThroughAPI",
                "llm_tts.models.blackboxmodel_with_streaming:BlackboxModelWithStreaming",
                "llm_tts.scorers:ChainMajorityVotingScorer",
                "llm_tts.scorers:StepScorerConfidence",
                "llm_tts.step_boundary_detectors:ThinkingMarkerDetector",
                "llm_tts.strategies:StrategyBaseline",
                "llm_tts.strategies:StrategyBeamSearch",
                "llm_tts.strategies:AdaptiveScalingBestOfN",
                "llm_tts.strategies:StrategyOfflineBestOfN",
                "llm_tts.strategies:StrategyOnlineBestOfN",
                "llm_tts.strategies:StrategySelfConsistency",
                "lm_polygraph.utils.generation_parameters:GenerationParameters",
            ],
        ),
        _dependency_check(
            name="logprob_scorers",
            required=[
                "lm_polygraph.estimators:Perplexity",
                "lm_polygraph.estimators:MeanTokenEntropy",
                "lm_polygraph.estimators:MaximumSequenceProbability",
                "lm_polygraph.stat_calculators:EntropyCalculator",
                "lm_polygraph.stat_calculators:VLLMLogprobsCalculator",
                "lm_polygraph.utils:APIWithUncertainty",
            ],
        ),
        _dependency_check(
            name="prm_scorer",
            required=[
                "llm_tts.scorers:StepScorerPRM",
            ],
        ),
    ]

    missing_dependencies = sorted(
        {missing for check in checks for missing in check["missing_dependencies"]}
    )
    missing_dependency_details = [
        detail
        for check in checks
        for detail in check.get("missing_dependency_details", [])
    ]
    can_run = not missing_dependencies
    return {
        "status": "ok" if can_run else "degraded",
        "can_run": can_run,
        "missing_dependencies": missing_dependencies,
        "missing_dependency_details": missing_dependency_details,
        "checks": checks,
    }


def build_single_sample_payload(
    question: str,
    gold_answer: Optional[str] = None,
    shared_prompt: str = "",
    budget: Optional[int] = None,
    provider: str = "openrouter",
    model_id: str = "openai/gpt-4o-mini",
    api_key: str = "",
    strategy_id: str = "baseline",
    scorer_id: Optional[str] = None,
    advanced_config_yaml: Optional[str] = None,
    scenario_id: str = "custom_1",
    scenario_title: str = "Single Example",
    scenario_description: str = (
        "Custom single-sample run with selected strategy and optional scorer."
    ),
    input_source: str = "custom_single",
    api_key_masked_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one debugger payload for a single (question, gold answer) sample."""
    selected_budget = _pick_budget(budget, DEBUGGER_BUDGETS)
    normalized_gold_answer = str(gold_answer or "").strip()
    has_gold_answer = bool(normalized_gold_answer)
    masked_api_key = (
        api_key_masked_override
        if api_key_masked_override is not None
        else _mask_api_key(api_key)
    )

    model_config = {
        "provider": provider,
        "model_id": model_id,
        "api_key_masked": masked_api_key,
    }
    strategy = _find_strategy(strategy_id)
    scorer: Optional[Dict[str, Any]] = None
    if strategy.get("requires_scorer", True):
        if not scorer_id:
            raise ValueError(f"Strategy '{strategy_id}' requires scorer_id.")
        scorer = _find_scorer(scorer_id)

    resolved_advanced_config, resolved_advanced_config_yaml = _resolve_advanced_config(
        strategy_id=strategy["id"],
        scorer_id=scorer.get("id") if scorer else None,
        advanced_config_yaml=advanced_config_yaml,
    )
    resolved_shared_prompt = _extract_prompt_from_advanced_config(
        resolved_advanced_config,
        fallback_prompt=shared_prompt,
    )
    prompt = question.strip()
    if resolved_shared_prompt.strip():
        prompt = f"{resolved_shared_prompt.strip()}\n\nQuestion: {question.strip()}"

    strategy_entry = _run_real_strategy_entry(
        strategy=strategy,
        scorer=scorer,
        question=question,
        gold_answer=normalized_gold_answer,
        budget=selected_budget,
        shared_prompt=resolved_shared_prompt,
        model_config=model_config,
        api_key=api_key,
        has_gold_answer=has_gold_answer,
        advanced_config=resolved_advanced_config,
    )

    return {
        "scenario": {
            "id": scenario_id,
            "title": scenario_title,
            "description": scenario_description,
            "prompt": prompt,
            "question": question.strip(),
            "ground_truth": normalized_gold_answer if has_gold_answer else None,
            "shared_prompt": resolved_shared_prompt,
            "input_source": input_source,
            "model_config": model_config,
            "advanced_config": resolved_advanced_config,
            "advanced_config_yaml": resolved_advanced_config_yaml,
            "strategy_count": 1,
            "scorer_count": 1 if scorer else 0,
            "run_count": 1,
            "selected_strategy_id": strategy["id"],
            "selected_scorer_id": scorer["id"] if scorer else None,
            "has_gold_answer": has_gold_answer,
        },
        "available_budgets": DEBUGGER_BUDGETS,
        "selected_budget": selected_budget,
        "strategy_catalog": [deepcopy(strategy)],
        "scorer_catalog": [deepcopy(scorer)] if scorer else [],
        "strategies": [strategy_entry],
    }


def _dependency_check(name: str, required: List[str]) -> Dict[str, Any]:
    missing: List[str] = []
    missing_details: List[Dict[str, str]] = []
    for spec in required:
        module_name, attr_name = _split_dependency_spec(spec)
        try:
            module = importlib.import_module(module_name)
            if attr_name and not hasattr(module, attr_name):
                missing.append(spec)
                missing_details.append(
                    {
                        "dependency": spec,
                        "error": f"Attribute '{attr_name}' not found in module '{module_name}'.",
                    }
                )
        except Exception as exc:
            missing.append(spec)
            missing_details.append(
                {
                    "dependency": spec,
                    "error": _compact_error(exc),
                }
            )

    return {
        "name": name,
        "ok": not missing,
        "missing_dependencies": missing,
        "missing_dependency_details": missing_details,
    }


def _split_dependency_spec(spec: str) -> Tuple[str, Optional[str]]:
    if ":" not in spec:
        return spec, None
    module_name, attr_name = spec.split(":", 1)
    module_name = module_name.strip()
    attr_name = attr_name.strip() or None
    return module_name, attr_name


def _run_real_strategy_entry(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    question: str,
    gold_answer: str,
    budget: int,
    shared_prompt: str,
    model_config: Dict[str, str],
    api_key: str,
    has_gold_answer: bool,
    advanced_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generation_config = _safe_mapping((advanced_config or {}).get("generation"))
    strategy_config = _safe_mapping((advanced_config or {}).get("strategy"))
    scorer_config = _safe_mapping((advanced_config or {}).get("scorer"))
    model_overrides = _safe_mapping((advanced_config or {}).get("model"))

    request_messages = _build_request_messages(
        question=question,
        shared_prompt=shared_prompt,
    )

    runtime: Optional[Dict[str, Any]] = None
    start_time = time.perf_counter()

    try:
        runtime = _create_runtime_components(
            strategy=strategy,
            scorer=scorer,
            model_config=model_config,
            api_key=api_key,
            model_overrides=model_overrides,
            generation_config=generation_config,
            strategy_config=strategy_config,
            scorer_config=scorer_config,
            budget=budget,
        )
        strategy_instance = runtime["strategy_instance"]
        strategy_result = strategy_instance.generate_trajectory(
            input_chat=request_messages,
            sample_idx=0,
        )
    except ValueError:
        raise
    except ImportError as exc:
        raise RuntimeError(
            "Real debugger execution dependencies are missing. "
            "Install runtime packages for strategy execution (llm_tts + lm_polygraph)."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to execute strategy '{strategy['id']}'"
            + (f" with scorer '{scorer['id']}'" if scorer is not None else "")
            + f": {_compact_error(exc)}"
        ) from exc
    finally:
        if runtime is not None:
            _cleanup_runtime_components(runtime)

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    run = _convert_strategy_result_to_debugger_run(
        strategy=strategy,
        scorer=scorer,
        strategy_result=strategy_result,
        budget=budget,
        latency_ms=latency_ms,
        model_config=model_config,
        generation_config=generation_config,
        strategy_config=strategy_config,
        scorer_config=scorer_config if scorer else {},
        has_gold_answer=has_gold_answer,
        gold_answer=gold_answer,
    )

    run_id = strategy["id"] if scorer is None else f"{strategy['id']}__{scorer['id']}"
    run_name = (
        strategy["name"] if scorer is None else f"{strategy['name']} · {scorer['name']}"
    )
    summary = strategy.get("summary", "")
    if scorer is not None:
        summary = f"{summary} Evaluated with {scorer.get('name', scorer['id'])}."

    return {
        "id": run_id,
        "strategy_id": strategy["id"],
        "scorer_id": scorer.get("id") if scorer else None,
        "name": run_name,
        "family": strategy.get("family", "unknown"),
        "summary": summary,
        "run": run,
        "comparison_rank": 1,
    }


# ---------------------------------------------------------------------------
# Progress-reporting wrapper (used by the SSE streaming endpoint)
# ---------------------------------------------------------------------------

_STEP_PATTERNS = [
    # beam_search: "Beam Search Step 3: 4 active samples"
    (re.compile(r"Beam Search Step (\d+)"), lambda m: f"Step {m.group(1)}"),
    # online_best_of_n: "Online BoN Step 3: 1 active samples"
    (re.compile(r"Online BoN Step (\d+)"), lambda m: f"Step {m.group(1)}"),
    # adaptive: "=== Step 3 === (1/1 active samples)"
    (re.compile(r"=== Step (\d+) ==="), lambda m: f"Step {m.group(1)}"),
]


class StrategyProgressHandler(logging.Handler):
    """Logging handler that intercepts strategy log lines and fires a callback."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__(level=logging.INFO)
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        for pattern, formatter in _STEP_PATTERNS:
            m = pattern.search(msg)
            if m:
                self._callback(formatter(m))
                return


def run_real_strategy_entry_with_progress(
    progress_callback: Callable[[str], None],
    **kwargs,
) -> Dict[str, Any]:
    """Run ``_run_real_strategy_entry`` while forwarding progress to *callback*.

    A custom logging handler is temporarily attached to the
    ``llm_tts.strategies`` logger so that strategy log lines are intercepted
    without modifying any strategy code.
    """
    logger = logging.getLogger("llm_tts.strategies")
    handler = StrategyProgressHandler(progress_callback)
    logger.addHandler(handler)
    try:
        return _run_real_strategy_entry(**kwargs)
    finally:
        logger.removeHandler(handler)


def _build_request_messages(question: str, shared_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    prompt_text = str(shared_prompt or "").strip()
    if prompt_text:
        messages.append({"role": "system", "content": prompt_text})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def _create_runtime_components(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    model_config: Dict[str, str],
    api_key: str,
    model_overrides: Dict[str, Any],
    generation_config: Dict[str, Any],
    strategy_config: Dict[str, Any],
    scorer_config: Dict[str, Any],
    budget: int,
) -> Dict[str, Any]:
    provider = str(model_config.get("provider") or "").strip().lower()
    model_id = str(model_config.get("model_id") or "").strip()
    api_key_value = str(api_key or "").strip()

    # Allow OpenRouter-style "openai/gpt-4o-mini" with the openai provider
    if provider == "openai" and model_id.startswith("openai/"):
        model_id = model_id[len("openai/"):]

    if provider not in _PROVIDER_BASE_URLS:
        raise ValueError("Provider must be one of: openai, openrouter.")
    if not model_id:
        raise ValueError("Model ID is required to run strategy execution.")
    if not api_key_value:
        raise ValueError("API key is required to run strategy execution.")

    try:
        from lm_polygraph.utils.generation_parameters import GenerationParameters

        from llm_tts.early_stopping import BoundaryEarlyStopping
        from llm_tts.generators.api import StepCandidateGeneratorThroughAPI
        from llm_tts.models.blackboxmodel_with_streaming import (
            BlackboxModelWithStreaming,
        )
        from llm_tts.scorers import (
            ChainMajorityVotingScorer,
            StepScorerConfidence,
            StepScorerPRM,
        )
        from llm_tts.step_boundary_detectors import ThinkingMarkerDetector
        from llm_tts.strategies import (
            AdaptiveScalingBestOfN,
            StrategyBaseline,
            StrategyBeamSearch,
            StrategyOfflineBestOfN,
            StrategyOnlineBestOfN,
            StrategySelfConsistency,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import runtime dependencies for real strategy execution."
        ) from exc

    scorer_id = scorer.get("id") if scorer else None
    requires_logprobs = bool(scorer and scorer.get("requires_logprobs"))

    detector = ThinkingMarkerDetector(
        min_step_tokens=_coerce_int(
            strategy_config.get("min_step_tokens"),
            default=50,
            minimum=0,
            maximum=32768,
        ),
        max_step_tokens=_coerce_int(
            strategy_config.get("max_step_tokens"),
            default=1024,
            minimum=1,
            maximum=32768,
        ),
        use_sequence=_coerce_bool(strategy_config.get("use_sequence"), default=True),
        use_conclusion=_coerce_bool(
            strategy_config.get("use_conclusion"),
            default=True,
        ),
        use_thinking=_coerce_bool(strategy_config.get("use_thinking"), default=True),
        use_verification=_coerce_bool(
            strategy_config.get("use_verification"),
            default=True,
        ),
        use_structure=_coerce_bool(strategy_config.get("use_structure"), default=False),
        use_reasoning=_coerce_bool(strategy_config.get("use_reasoning"), default=True),
        use_sentence_start=_coerce_bool(
            strategy_config.get("use_sentence_start"),
            default=False,
        ),
        use_correction=_coerce_bool(
            strategy_config.get("use_correction"),
            default=False,
        ),
        custom_markers=_coerce_string_list(strategy_config.get("custom_markers")),
    )

    generation_parameters = GenerationParameters()
    generation_parameters.temperature = _coerce_float(
        generation_config.get("temperature"),
        default=0.7,
        minimum=0.0,
        maximum=2.0,
    )
    generation_parameters.max_new_tokens = _coerce_int(
        generation_config.get("max_new_tokens"),
        default=500,
        minimum=1,
        maximum=200000,
    )
    generation_parameters.top_p = _coerce_float(
        generation_config.get("top_p"),
        default=0.8,
        minimum=0.0,
        maximum=1.0,
    )
    generation_parameters.top_k = _coerce_int(
        generation_config.get("top_k"),
        default=20,
        minimum=0,
        maximum=1000,
    )

    base_model = BlackboxModelWithStreaming(
        openai_api_key=api_key_value,
        model_path=model_id,
        supports_logprobs=requires_logprobs,
        base_url=_PROVIDER_BASE_URLS.get(provider),
        early_stopping=BoundaryEarlyStopping(detector=detector),
        generation_parameters=generation_parameters,
    )

    model_for_generator: Any = base_model
    if scorer_id in {"perplexity", "entropy", "sequence_prob"}:
        try:
            from lm_polygraph.estimators import (
                MaximumSequenceProbability,
                MeanTokenEntropy,
                Perplexity,
            )
            from lm_polygraph.stat_calculators import (
                EntropyCalculator,
                VLLMLogprobsCalculator,
            )
            from lm_polygraph.utils import APIWithUncertainty
        except ImportError as exc:
            raise RuntimeError(
                "Scorer requires lm_polygraph uncertainty components, but they are unavailable."
            ) from exc

        if scorer_id == "perplexity":
            stat_calculators = [VLLMLogprobsCalculator()]
            estimator = Perplexity()
        elif scorer_id == "sequence_prob":
            stat_calculators = [VLLMLogprobsCalculator()]
            estimator = MaximumSequenceProbability()
        else:
            stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
            estimator = MeanTokenEntropy()

        model_for_generator = APIWithUncertainty(
            model=base_model,
            stat_calculators=stat_calculators,
            estimator=estimator,
        )

    disable_thinking_mode = model_overrides.get("disable_thinking_mode", None)
    if disable_thinking_mode not in (True, False, None):
        disable_thinking_mode = _coerce_bool(disable_thinking_mode, default=None)
    thinking_mode = disable_thinking_mode is False

    answer_patterns = _coerce_string_list(
        strategy_config.get("detector_answer_patterns"),
    )

    step_generator = StepCandidateGeneratorThroughAPI(
        model=model_for_generator,
        thinking_mode=thinking_mode,
        detector=detector,
        answer_patterns=answer_patterns,
        max_new_tokens=_coerce_int(
            generation_config.get("max_new_tokens"),
            default=500,
            minimum=1,
            maximum=200000,
        ),
        temperature=_coerce_float(
            generation_config.get("temperature"),
            default=0.7,
            minimum=0.0,
            maximum=2.0,
        ),
        top_p=_coerce_float(
            generation_config.get("top_p"),
            default=0.8,
            minimum=0.0,
            maximum=1.0,
        ),
        top_k=_coerce_int(
            generation_config.get("top_k"),
            default=20,
            minimum=0,
            maximum=1000,
        ),
        presence_penalty=_coerce_float(
            generation_config.get("presence_penalty"),
            default=0.0,
            minimum=-2.0,
            maximum=2.0,
        ),
        max_context_budget=_coerce_int(
            generation_config.get("max_length"),
            default=8192,
            minimum=1024,
            maximum=262144,
        ),
        prefill_mode=_coerce_bool(
            model_overrides.get("prefill_mode"),
            # Step-by-step strategies need prefill so the model continues
            # from the trajectory prefix rather than starting a new response.
            default=strategy["id"] in {"online_best_of_n", "beam_search", "adaptive"},
        ),
        disable_thinking_mode=disable_thinking_mode,
        supports_logprobs=requires_logprobs,
        max_concurrent_requests=_coerce_int(
            model_overrides.get("max_concurrent_requests"),
            default=128,
            minimum=1,
            maximum=1024,
        ),
    )

    scorer_instance: Optional[Any] = None
    if strategy["id"] not in {"baseline", "self_consistency"}:
        if scorer_id == "prm":
            scorer_instance = StepScorerPRM(
                prm_model_path=str(
                    scorer_config.get("model_path") or "Qwen/Qwen2.5-Math-PRM-7B"
                ),
                device=_resolve_prm_device(
                    str(scorer_config.get("device") or "auto")
                ),
                batch_size=_coerce_int(
                    scorer_config.get("batch_size"),
                    default=1,
                    minimum=1,
                    maximum=256,
                ),
                torch_dtype="bfloat16",
                use_vllm=_coerce_bool(scorer_config.get("use_vllm"), default=True),
                gpu_memory_utilization=_coerce_float(
                    scorer_config.get("gpu_memory_utilization"),
                    default=0.9,
                    minimum=0.1,
                    maximum=1.0,
                ),
                prm_max_tokens=_coerce_int(
                    scorer_config.get("prm_max_tokens"),
                    default=4000,
                    minimum=128,
                    maximum=65536,
                ),
            )
        elif scorer_id in {"perplexity", "entropy", "sequence_prob"}:
            scorer_instance = StepScorerConfidence()
        elif scorer_id is not None:
            raise ValueError(f"Unsupported scorer_id for real execution: {scorer_id}")

    strategy_id = strategy["id"]
    if strategy_id == "baseline":
        strategy_instance = StrategyBaseline(
            step_generator=step_generator,
            eos_patterns=_coerce_string_list(
                strategy_config.get("detector_eos_patterns")
            )
            or ["<end of response>"],
            stop_token_ids=_coerce_int_list(strategy_config.get("stop_token_ids")),
            batch_generation=_coerce_bool(
                strategy_config.get("batch_generation"),
                default=False,
            ),
        )
    elif strategy_id == "beam_search":
        if scorer_instance is None:
            raise ValueError("Beam search requires a scorer.")
        strategy_instance = StrategyBeamSearch(
            step_generator=step_generator,
            scorer=scorer_instance,
            beam_size=_coerce_int(
                strategy_config.get("beam_size"),
                default=min(max(2, budget // 2), 6),
                minimum=1,
                maximum=64,
            ),
            candidates_per_beam=_coerce_int(
                strategy_config.get("candidates_per_beam"),
                default=2,
                minimum=1,
                maximum=32,
            ),
            max_steps=_coerce_int(
                strategy_config.get("max_steps"),
                default=max(2, budget),
                minimum=1,
                maximum=128,
            ),
            aggregation=str(strategy_config.get("aggregation") or "mean"),
            batch_generation=_coerce_bool(
                strategy_config.get("batch_generation"),
                default=True,
            ),
            prompt_buffer=_coerce_int(
                strategy_config.get("prompt_buffer"),
                default=500,
                minimum=0,
                maximum=20000,
            ),
            scoring_window=_coerce_optional_int(strategy_config.get("scoring_window")),
        )
    elif strategy_id == "adaptive":
        if scorer_instance is None:
            raise ValueError("Adaptive best-of-n requires a scorer.")
        strategy_instance = AdaptiveScalingBestOfN(
            step_generator=step_generator,
            scorer=scorer_instance,
            candidates_per_step=_coerce_int(
                strategy_config.get("candidates_per_step"),
                default=max(2, budget),
                minimum=1,
                maximum=64,
            ),
            max_steps=_coerce_int(
                strategy_config.get("max_steps"),
                default=max(2, budget),
                minimum=1,
                maximum=128,
            ),
            scaling_rate=_coerce_float(
                strategy_config.get("scaling_rate"),
                default=0.9,
                minimum=0.0,
                maximum=1.0,
            ),
            momentum_rate=_coerce_float(
                strategy_config.get("momentum_rate"),
                default=0.9,
                minimum=0.0,
                maximum=1.0,
            ),
            adaptive_scaling_method=str(
                strategy_config.get("adaptive_scaling_method") or "momentum"
            ),
            batch_size=_coerce_int(
                strategy_config.get("batch_size"),
                default=1000,
                minimum=1,
                maximum=100000,
            ),
        )
    elif strategy_id == "online_best_of_n":
        if scorer_instance is None:
            raise ValueError("Online best-of-n requires a scorer.")
        strategy_instance = StrategyOnlineBestOfN(
            step_generator=step_generator,
            scorer=scorer_instance,
            candidates_per_step=_coerce_int(
                strategy_config.get("candidates_per_step"),
                default=max(2, budget),
                minimum=1,
                maximum=64,
            ),
            max_steps=_coerce_int(
                strategy_config.get("max_steps"),
                default=max(2, budget),
                minimum=1,
                maximum=128,
            ),
            batch_generation=_coerce_bool(
                strategy_config.get("batch_generation"),
                default=True,
            ),
            prompt_buffer=_coerce_int(
                strategy_config.get("prompt_buffer"),
                default=500,
                minimum=0,
                maximum=20000,
            ),
        )
    elif strategy_id == "offline_best_of_n":
        if scorer_instance is None:
            raise ValueError("Offline best-of-n requires a scorer.")
        strategy_instance = StrategyOfflineBestOfN(
            scorer=scorer_instance,
            num_trajectories=_coerce_int(
                strategy_config.get("num_trajectories"),
                default=max(4, budget),
                minimum=1,
                maximum=128,
            ),
            max_steps=_coerce_int(
                strategy_config.get("max_steps"),
                default=max(2, budget),
                minimum=1,
                maximum=128,
            ),
            step_generator=step_generator,
            score_aggregation=str(strategy_config.get("score_aggregation") or "mean"),
            batch_generation=_coerce_bool(
                strategy_config.get("batch_generation"),
                default=True,
            ),
            calculate_entropy_score=_coerce_bool(
                strategy_config.get("calculate_entropy_score"),
                default=False,
            ),
            calculate_perplexity_score=_coerce_bool(
                strategy_config.get("calculate_perplexity_score"),
                default=False,
            ),
            calculate_sequence_prob_score=_coerce_bool(
                strategy_config.get("calculate_sequence_prob_score"),
                default=False,
            ),
            calculate_pd_gap_score=_coerce_bool(
                strategy_config.get("calculate_pd_gap_score"),
                default=False,
            ),
            calculate_prm_score=_coerce_bool(
                strategy_config.get("calculate_prm_score"),
                default=False,
            ),
            scoring_window=_coerce_optional_int(strategy_config.get("scoring_window")),
        )
    elif strategy_id == "self_consistency":
        chain_patterns = _coerce_string_list(
            _safe_mapping(strategy_config.get("scorer")).get(
                "answer_extraction_patterns"
            )
        )
        chain_scorer = ChainMajorityVotingScorer(
            answer_extraction_patterns=chain_patterns or None,
        )
        strategy_instance = StrategySelfConsistency(
            step_generator=step_generator,
            num_paths=_coerce_int(
                strategy_config.get("num_paths"),
                default=max(4, budget),
                minimum=1,
                maximum=128,
            ),
            scorer=chain_scorer,
            batch_generation=_coerce_bool(
                strategy_config.get("batch_generation"),
                default=True,
            ),
        )
    else:
        raise ValueError(f"Unsupported strategy_id for real execution: {strategy_id}")

    return {
        "base_model": base_model,
        "strategy_instance": strategy_instance,
        "scorer_instance": scorer_instance,
        "step_generator": step_generator,
    }


def _cleanup_runtime_components(runtime: Dict[str, Any]) -> None:
    strategy_instance = runtime.get("strategy_instance")
    scorer_instance = runtime.get("scorer_instance")
    base_model = runtime.get("base_model")

    if strategy_instance is not None and hasattr(strategy_instance, "cleanup"):
        try:
            strategy_instance.cleanup()
        except Exception:
            pass

    if (
        scorer_instance is not None
        and hasattr(scorer_instance, "cleanup")
        and scorer_instance is not strategy_instance
    ):
        try:
            scorer_instance.cleanup()
        except Exception:
            pass

    if base_model is not None and hasattr(base_model, "shutdown"):
        try:
            base_model.shutdown()
        except Exception:
            pass


def _convert_strategy_result_to_debugger_run(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    strategy_result: Dict[str, Any],
    budget: int,
    latency_ms: int,
    model_config: Dict[str, str],
    generation_config: Dict[str, Any],
    strategy_config: Dict[str, Any],
    scorer_config: Dict[str, Any],
    has_gold_answer: bool,
    gold_answer: str,
) -> Dict[str, Any]:
    confidence = _estimate_result_confidence(
        strategy_result=strategy_result,
        scorer=scorer,
    )
    events = _build_events_from_strategy_result(
        strategy=strategy,
        scorer=scorer,
        strategy_result=strategy_result,
        confidence=confidence,
    )
    tokens_used = _estimate_result_tokens(strategy_result)

    score_label = (
        scorer["id"]
        if scorer
        else "consensus" if strategy["id"] == "self_consistency" else "confidence"
    )
    final: Dict[str, Any] = {
        "confidence": confidence,
        "score_label": score_label,
        "selected_trajectory": strategy_result.get("trajectory") or "",
        "selection_reason": (
            "Selected by majority voting across sampled trajectories."
            if strategy["id"] == "self_consistency"
            else (
                f"Selected by {strategy['name']}."
                if scorer is None
                else f"Selected by {strategy['name']} using {scorer['name']}."
            )
        ),
    }
    extracted_answer = str(strategy_result.get("extracted_answer") or "").strip()
    if has_gold_answer and extracted_answer:
        final["answer"] = extracted_answer
        final["is_correct"] = extracted_answer.strip() == gold_answer.strip()

    run: Dict[str, Any] = {
        "budget": budget,
        "budget_unit": _budget_unit_for_family(strategy.get("family", "single_pass")),
        "used_budget": max(1, len(events)) if events else 1,
        "tokens_used": tokens_used,
        "latency_ms": max(1, latency_ms),
        "provider": model_config.get("provider", "openrouter"),
        "model_id": model_config.get("model_id", ""),
        "strategy": {
            "id": strategy["id"],
            "name": strategy["name"],
            "family": strategy.get("family", "unknown"),
        },
        "scorer": (
            {
                "id": scorer["id"],
                "name": scorer["name"],
                "direction": scorer["direction"],
                "summary": scorer.get("summary", ""),
            }
            if scorer
            else None
        ),
        "final": final,
        "config": {
            "generation": deepcopy(generation_config),
            "strategy": deepcopy(strategy_config),
            "scorer": deepcopy(scorer_config) if scorer else None,
        },
        "events": events,
    }

    return run


def _estimate_result_confidence(
    strategy_result: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
) -> float:
    if not isinstance(strategy_result, dict):
        return 0.0

    best_idx = _coerce_int(strategy_result.get("best_idx"), default=0, minimum=0)
    all_scores = strategy_result.get("all_scores")
    if isinstance(all_scores, list) and all_scores:
        if best_idx >= len(all_scores):
            best_idx = len(all_scores) - 1
        return _confidence_from_score(all_scores[best_idx], scorer=scorer)

    aggregated_score = _to_float(strategy_result.get("aggregated_score"))
    if aggregated_score is not None:
        return _confidence_from_score(aggregated_score, scorer=scorer)

    validity_scores = [
        value
        for value in (
            _to_float(item) for item in strategy_result.get("validity_scores", [])
        )
        if value is not None and math.isfinite(value)
    ]
    if validity_scores:
        return _confidence_from_score(
            sum(validity_scores) / len(validity_scores),
            scorer=scorer,
        )

    metadata = strategy_result.get("metadata")
    consensus_score = _to_float(
        metadata.get("consensus_score") if isinstance(metadata, dict) else None
    )
    if consensus_score is not None:
        return _confidence_from_score(consensus_score, scorer=scorer)

    return 0.0


def _estimate_result_tokens(strategy_result: Dict[str, Any]) -> int:
    token_stats = strategy_result.get("token_stats")
    if isinstance(token_stats, dict):
        total = _coerce_int(
            token_stats.get("total_tokens_this_sample"),
            default=0,
            minimum=0,
        )
        if total > 0:
            return total

    total_tokens = _coerce_int(
        strategy_result.get("total_tokens"),
        default=0,
        minimum=0,
    )
    if total_tokens > 0:
        return total_tokens

    steps = _extract_step_entries(strategy_result.get("steps"))
    counted_tokens = sum(item["tokens"] for item in steps)
    if counted_tokens > 0:
        return counted_tokens

    trajectory_text = str(strategy_result.get("trajectory") or "")
    return max(1, len(trajectory_text.split()))


def _build_events_from_strategy_result(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    strategy_result: Dict[str, Any],
    confidence: float,
) -> List[Dict[str, Any]]:
    scorer_key = scorer["id"] if scorer else "confidence"
    scorer_direction = (
        scorer.get("direction", "higher_better") if scorer else "higher_better"
    )

    step_candidates = strategy_result.get("step_candidates")
    if isinstance(step_candidates, list) and step_candidates:
        return _build_events_from_step_candidates(
            strategy=strategy,
            scorer=scorer,
            step_candidates=step_candidates,
            fallback_confidence=confidence,
        )

    all_trajectories = strategy_result.get("all_trajectories")
    all_scores = strategy_result.get("all_scores")
    if isinstance(all_trajectories, list) and all_trajectories:
        best_idx_hint = _coerce_int(
            strategy_result.get("best_idx"), default=0, minimum=0
        )
        if best_idx_hint >= len(all_trajectories):
            best_idx_hint = 0
        all_trajectory_steps = strategy_result.get("all_trajectory_steps")
        expanded_trajectory_events = _build_events_from_trajectory_pool(
            strategy=strategy,
            scorer=scorer,
            all_trajectories=all_trajectories,
            all_trajectory_steps=(
                all_trajectory_steps if isinstance(all_trajectory_steps, list) else []
            ),
            all_scores=all_scores if isinstance(all_scores, list) else [],
            all_step_scores=(
                strategy_result.get("all_step_scores")
                if isinstance(strategy_result.get("all_step_scores"), list)
                else []
            ),
            fallback_confidence=confidence,
            preferred_best_idx=best_idx_hint,
        )
        if expanded_trajectory_events:
            return expanded_trajectory_events

        best_idx = min(best_idx_hint, len(all_trajectories) - 1)

        candidates: List[Dict[str, Any]] = []
        for index, trajectory_text in enumerate(all_trajectories):
            score_value = None
            if isinstance(all_scores, list) and index < len(all_scores):
                score_value = _to_float(all_scores[index])

            candidate_signals: Dict[str, float] = {
                "confidence": _confidence_from_score(
                    score_value,
                    scorer=scorer,
                    fallback=confidence,
                )
            }
            if scorer and score_value is not None:
                candidate_signals[scorer_key] = score_value

            candidates.append(
                {
                    "id": f"{strategy['id']}_{scorer_key}_traj_{index + 1}",
                    "label": f"Trajectory {index + 1}",
                    "text": str(trajectory_text or ""),
                    "status": "selected" if index == best_idx else "pruned",
                    "selected": index == best_idx,
                    "signals": candidate_signals,
                }
            )

        selected_score = (
            _to_float(all_scores[best_idx])
            if isinstance(all_scores, list) and best_idx < len(all_scores)
            else None
        )
        signals = [
            {
                "name": "confidence",
                "value": _confidence_from_score(
                    selected_score,
                    scorer=scorer,
                    fallback=confidence,
                ),
                "direction": "higher_better",
            }
        ]
        if scorer and selected_score is not None:
            signals.append(
                {
                    "name": scorer_key,
                    "value": selected_score,
                    "direction": scorer_direction,
                }
            )

        events = [
            {
                "step": 1,
                "title": "Trajectory reranking",
                "stage": "reranking",
                "decision": {
                    "action": "select",
                    "reason": "Selected the best complete trajectory score.",
                },
                "signals": signals,
                "candidates": candidates,
            }
        ]

        selected_steps = _extract_step_entries(strategy_result.get("steps"))
        selected_scores = strategy_result.get("validity_scores", [])
        events.extend(
            _build_stepwise_events(
                strategy=strategy,
                scorer=scorer,
                step_entries=selected_steps,
                step_scores=(
                    selected_scores if isinstance(selected_scores, list) else []
                ),
                confidence=confidence,
                start_step=2,
            )
        )
        return events

    all_traces = strategy_result.get("all_traces")
    if isinstance(all_traces, list) and all_traces:
        # Build independent branches from pre-split steps in each trace.
        # Each trace is an independent path; only common ancestor is root.
        trace_score_key = scorer_key if scorer else "consensus"
        best_idx = next((i for i, t in enumerate(all_traces) if t.get("selected")), 0)

        # Collect per-trace steps from the backend
        trace_step_lists: List[List[str]] = []
        for trace in all_traces:
            steps = trace.get("steps") if isinstance(trace, dict) else None
            if isinstance(steps, list) and steps:
                trace_step_lists.append([str(s or "") for s in steps])
            else:
                # Fallback: treat full text as a single step
                trace_step_lists.append([str((trace or {}).get("text") or "")])

        max_steps = max(len(sl) for sl in trace_step_lists)

        if max_steps > 1:
            events: List[Dict[str, Any]] = []
            for step_index in range(max_steps):
                event_candidates: List[Dict[str, Any]] = []
                selected_score: Optional[float] = None

                for traj_index, steps in enumerate(trace_step_lists):
                    if step_index >= len(steps):
                        continue
                    step_text = steps[step_index].strip()
                    if not step_text:
                        continue

                    trace = all_traces[traj_index]
                    score_value = _to_float(
                        trace.get("score") if isinstance(trace, dict) else None
                    )
                    conf = _confidence_from_score(
                        score_value, scorer=scorer, fallback=confidence
                    )
                    signal_map: Dict[str, float] = {}
                    if score_value is not None:
                        signal_map[trace_score_key] = score_value
                    else:
                        signal_map["confidence"] = conf

                    is_selected = traj_index == best_idx
                    if is_selected:
                        selected_score = score_value

                    candidate_entry: Dict[str, Any] = {
                        "id": f"{strategy['id']}_{trace_score_key}_trace_{traj_index + 1}_step_{step_index + 1}",
                        "label": f"Path {traj_index + 1}",
                        "text": step_text,
                        "status": "selected" if is_selected else "pruned",
                        "selected": is_selected,
                        "signals": signal_map,
                        "beam_uid": f"path_{traj_index}_step_{step_index}",
                    }
                    if step_index > 0:
                        candidate_entry["parent_beam_uid"] = (
                            f"path_{traj_index}_step_{step_index - 1}"
                        )
                    event_candidates.append(candidate_entry)

                if not event_candidates:
                    continue

                if selected_score is not None:
                    ev_signals = [
                        {
                            "name": trace_score_key,
                            "value": selected_score,
                            "direction": scorer_direction,
                        }
                    ]
                else:
                    ev_signals = [
                        {
                            "name": "confidence",
                            "value": _confidence_from_score(
                                selected_score, scorer=scorer, fallback=confidence
                            ),
                            "direction": "higher_better",
                        }
                    ]

                is_last = step_index == max_steps - 1
                events.append(
                    {
                        "step": step_index + 1,
                        "title": f"Reasoning step {step_index + 1}",
                        "stage": "reranking" if is_last else "candidate_generation",
                        "decision": {
                            "action": "select" if is_last else "inspect",
                            "reason": (
                                "Selected best path after self-consistency voting."
                                if is_last
                                else "Independent reasoning paths at this step."
                            ),
                        },
                        "signals": ev_signals,
                        "candidates": event_candidates,
                    }
                )

            if events:
                return events

        # Fallback: single-step traces → flat voting event
        candidates = []
        selected_trace_idx = 0
        for index, trace in enumerate(all_traces):
            trace_text = str((trace or {}).get("text") or "")
            trace_score = _to_float((trace or {}).get("score"))
            is_selected = bool((trace or {}).get("selected"))
            if is_selected:
                selected_trace_idx = index

            candidate_signals: Dict[str, float] = {}
            if trace_score is not None:
                candidate_signals[trace_score_key] = trace_score
            else:
                candidate_signals["confidence"] = _confidence_from_score(
                    trace_score,
                    scorer=scorer,
                    fallback=confidence,
                )

            candidates.append(
                {
                    "id": f"{strategy['id']}_{trace_score_key}_trace_{index + 1}",
                    "label": f"Path {index + 1}",
                    "text": trace_text,
                    "status": "selected" if is_selected else "pruned",
                    "selected": is_selected,
                    "signals": candidate_signals,
                }
            )

        selected_score = _to_float((all_traces[selected_trace_idx] or {}).get("score"))
        if selected_score is not None:
            signals = [
                {
                    "name": trace_score_key,
                    "value": selected_score,
                    "direction": scorer_direction,
                }
            ]
        else:
            signals = [
                {
                    "name": "confidence",
                    "value": _confidence_from_score(
                        selected_score,
                        scorer=scorer,
                        fallback=confidence,
                    ),
                    "direction": "higher_better",
                }
            ]

        return [
            {
                "step": 1,
                "title": "Self-consistency voting",
                "stage": "vote",
                "decision": {
                    "action": "select",
                    "reason": "Picked the path with the strongest answer consensus.",
                },
                "signals": signals,
                "candidates": candidates,
            }
        ]

    step_entries = _extract_step_entries(strategy_result.get("steps"))
    if not step_entries:
        trajectory_text = str(strategy_result.get("trajectory") or "").strip()
        if trajectory_text:
            step_entries = [{"text": trajectory_text, "tokens": 0}]
    step_scores = (
        strategy_result.get("validity_scores")
        if isinstance(strategy_result.get("validity_scores"), list)
        else []
    )
    return _build_stepwise_events(
        strategy=strategy,
        scorer=scorer,
        step_entries=step_entries,
        step_scores=step_scores,
        confidence=confidence,
        start_step=1,
    )


def _build_events_from_step_candidates(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    step_candidates: List[Dict[str, Any]],
    fallback_confidence: float,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    expanded_pools = _expand_step_candidate_pools(step_candidates)
    scorer_key = scorer["id"] if scorer else "confidence"
    scorer_direction = (
        scorer.get("direction", "higher_better") if scorer else "higher_better"
    )

    for index, pool in enumerate(expanded_pools):
        raw_candidates = pool.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            continue

        event_candidates: List[Dict[str, Any]] = []
        selected_score: Optional[float] = None

        for cand_idx, raw_candidate in enumerate(raw_candidates):
            if not isinstance(raw_candidate, dict):
                continue
            candidate_text = str(raw_candidate.get("text") or "").strip()
            if not candidate_text:
                continue

            score_value = _to_float(raw_candidate.get("score"))
            if score_value is None:
                score_value = _extract_first_numeric(raw_candidate.get("signals"))
            candidate_conf = _confidence_from_score(
                score_value,
                scorer=scorer,
                fallback=fallback_confidence,
            )

            signal_map: Dict[str, float] = {"confidence": candidate_conf}
            if score_value is not None:
                signal_map[scorer_key] = score_value

            status = str(raw_candidate.get("status") or "pruned")
            is_selected = bool(raw_candidate.get("selected")) or status == "selected"
            if is_selected:
                selected_score = score_value
                status = "selected"
            elif status not in {"selected", "kept", "pruned"}:
                status = "pruned"

            candidate_entry: Dict[str, Any] = {
                "id": str(
                    raw_candidate.get("id") or f"step_{index + 1}_cand_{cand_idx + 1}"
                ),
                "label": str(raw_candidate.get("label") or f"Candidate {cand_idx + 1}"),
                "text": candidate_text,
                "status": status,
                "selected": is_selected,
                "signals": signal_map,
            }
            # Propagate beam lineage for tree visualization
            if raw_candidate.get("beam_unique_id") is not None:
                candidate_entry["beam_uid"] = raw_candidate["beam_unique_id"]
            if raw_candidate.get("parent_beam_uid") is not None:
                candidate_entry["parent_beam_uid"] = raw_candidate["parent_beam_uid"]
            event_candidates.append(candidate_entry)

        if not event_candidates:
            continue

        signals = [
            {
                "name": "confidence",
                "value": _confidence_from_score(
                    selected_score,
                    scorer=scorer,
                    fallback=fallback_confidence,
                ),
                "direction": "higher_better",
            }
        ]
        if scorer is not None and selected_score is not None:
            signals.append(
                {
                    "name": scorer_key,
                    "value": selected_score,
                    "direction": scorer_direction,
                }
            )

        stage = str(pool.get("stage") or "").strip() or _event_stage_for_family(
            strategy.get("family", "single_pass"),
            index=index,
            total=len(expanded_pools),
        )
        selected_exists = any(
            candidate.get("selected") for candidate in event_candidates
        )
        decision = {
            "action": "select" if selected_exists else "inspect",
            "reason": (
                "Selected the top candidate from this generation step."
                if selected_exists
                else "Candidate scores are available for inspection."
            ),
        }

        event_entry: Dict[str, Any] = {
            "step": _coerce_int(
                pool.get("step"),
                default=index + 1,
                minimum=1,
            ),
            "title": str(pool.get("title") or f"Reasoning step {index + 1}"),
            "stage": stage,
            "decision": decision,
            "signals": signals,
            "candidates": event_candidates,
        }
        events.append(event_entry)

    return events


def _expand_step_candidate_pools(
    step_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [pool for pool in step_candidates if isinstance(pool, dict)]


def _build_events_from_trajectory_pool(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    all_trajectories: List[Any],
    all_trajectory_steps: List[Any],
    all_scores: List[Any],
    all_step_scores: List[Any],
    fallback_confidence: float,
    preferred_best_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # Use pre-split steps from the backend; fall back to full text as single step
    trajectory_steps: List[List[str]] = []
    for idx, trajectory in enumerate(all_trajectories):
        if (
            isinstance(all_trajectory_steps, list)
            and idx < len(all_trajectory_steps)
            and isinstance(all_trajectory_steps[idx], list)
            and all_trajectory_steps[idx]
        ):
            trajectory_steps.append([str(s or "") for s in all_trajectory_steps[idx]])
        else:
            trajectory_steps.append([str(trajectory or "")])

    scorer_direction = (
        scorer.get("direction", "higher_better") if scorer else "higher_better"
    )
    scorer_key = scorer["id"] if scorer else "confidence"

    best_idx = 0
    if preferred_best_idx is not None and 0 <= preferred_best_idx < len(
        trajectory_steps
    ):
        best_idx = preferred_best_idx
    else:
        best_score = None
        for index, score in enumerate(all_scores):
            if index >= len(trajectory_steps):
                break
            numeric_score = _to_float(score)
            if numeric_score is None:
                continue
            if best_score is None:
                best_score = numeric_score
                best_idx = index
                continue
            if scorer_direction == "lower_better":
                if numeric_score < best_score:
                    best_score = numeric_score
                    best_idx = index
            elif numeric_score > best_score:
                best_score = numeric_score
                best_idx = index

    if best_idx >= len(trajectory_steps):
        best_idx = 0
    best_steps = [
        str(step_text).strip()
        for step_text in (trajectory_steps[best_idx] if trajectory_steps else [])
        if str(step_text).strip()
    ]
    max_steps = len(best_steps)
    if max_steps <= 1:
        return []

    events: List[Dict[str, Any]] = []
    for step_index in range(max_steps):
        event_candidates: List[Dict[str, Any]] = []
        selected_score: Optional[float] = None

        for traj_index, parts in enumerate(trajectory_steps):
            if step_index >= len(parts):
                continue
            step_text = str(parts[step_index] or "").strip()
            if not step_text:
                continue

            score_value = None
            if traj_index < len(all_step_scores) and isinstance(
                all_step_scores[traj_index], list
            ):
                per_step_scores = all_step_scores[traj_index]
                if step_index < len(per_step_scores):
                    score_value = _to_float(per_step_scores[step_index])

            confidence_value = _confidence_from_score(
                score_value,
                scorer=scorer,
                fallback=fallback_confidence,
            )
            signal_map: Dict[str, float] = {"confidence": confidence_value}
            if score_value is not None:
                signal_map[scorer_key] = score_value

            is_selected = traj_index == best_idx
            if is_selected:
                selected_score = score_value

            candidate_entry: Dict[str, Any] = {
                "id": f"{strategy['id']}_{scorer_key}_traj_{traj_index + 1}_step_{step_index + 1}",
                "label": f"Trajectory {traj_index + 1}",
                "text": step_text,
                "status": "selected" if is_selected else "pruned",
                "selected": is_selected,
                "signals": signal_map,
                "beam_uid": f"traj_{traj_index}_step_{step_index}",
            }
            if step_index > 0:
                candidate_entry["parent_beam_uid"] = (
                    f"traj_{traj_index}_step_{step_index - 1}"
                )
            event_candidates.append(candidate_entry)

        if not event_candidates:
            continue

        signals = [
            {
                "name": "confidence",
                "value": _confidence_from_score(
                    selected_score,
                    scorer=scorer,
                    fallback=fallback_confidence,
                ),
                "direction": "higher_better",
            }
        ]
        if scorer is not None and selected_score is not None:
            signals.append(
                {
                    "name": scorer_key,
                    "value": selected_score,
                    "direction": scorer_direction,
                }
            )

        events.append(
            {
                "step": step_index + 1,
                "title": f"Reasoning step {step_index + 1}",
                "stage": (
                    "reranking"
                    if step_index == max_steps - 1
                    else "candidate_generation"
                ),
                "decision": {
                    "action": "select" if step_index == max_steps - 1 else "inspect",
                    "reason": (
                        "Selected best trajectory after comparing candidate traces."
                        if step_index == max_steps - 1
                        else "Inspect candidate trajectories for this reasoning step."
                    ),
                },
                "signals": signals,
                "candidates": event_candidates,
            }
        )

    return events


def _build_stepwise_events(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    step_entries: List[Dict[str, Any]],
    step_scores: List[Any],
    confidence: float,
    start_step: int,
) -> List[Dict[str, Any]]:
    if not step_entries:
        return []

    family = strategy.get("family", "single_pass")
    scorer_key = scorer["id"] if scorer else "confidence"
    scorer_direction = (
        scorer.get("direction", "higher_better") if scorer else "higher_better"
    )

    events: List[Dict[str, Any]] = []
    total_steps = len(step_entries)

    for index, step_entry in enumerate(step_entries):
        absolute_step = start_step + index
        raw_score = _to_float(step_scores[index]) if index < len(step_scores) else None
        score_for_step = raw_score if raw_score is not None else confidence
        confidence_for_step = _confidence_from_score(
            raw_score,
            scorer=scorer,
            fallback=confidence,
        )
        is_last_step = index == total_steps - 1

        stage = _event_stage_for_family(family, index=index, total=total_steps)
        decision = {
            "action": "stop" if is_last_step else "escalate",
            "reason": (
                "Reached final selected step."
                if is_last_step
                else "Continuing to next reasoning step."
            ),
        }

        signals = [
            {
                "name": "confidence",
                "value": confidence_for_step,
                "direction": "higher_better",
            }
        ]
        if scorer is not None:
            signals.append(
                {
                    "name": scorer_key,
                    "value": score_for_step,
                    "direction": scorer_direction,
                }
            )

        candidate_signals: Dict[str, float] = {"confidence": confidence_for_step}
        if scorer is not None:
            candidate_signals[scorer_key] = score_for_step

        candidate_text = str(step_entry.get("text") or "").strip()
        if not candidate_text:
            candidate_text = "(empty step)"

        events.append(
            {
                "step": absolute_step,
                "title": (
                    "Single-pass generation"
                    if family == "single_pass" and total_steps == 1
                    else f"Reasoning step {index + 1}"
                ),
                "stage": stage,
                "decision": decision,
                "signals": signals,
                "candidates": [
                    {
                        "id": (f"{strategy['id']}_{scorer_key}_s{absolute_step}_c1"),
                        "label": f"Step {index + 1}",
                        "text": candidate_text,
                        "status": "selected",
                        "selected": True,
                        "signals": candidate_signals,
                    }
                ],
            }
        )

    return events


def _extract_step_entries(raw_steps: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not isinstance(raw_steps, list):
        return entries

    for raw_step in raw_steps:
        step_text = ""
        token_count = 0

        if isinstance(raw_step, str):
            step_text = raw_step
        elif isinstance(raw_step, dict):
            step_text = str(raw_step.get("raw_text") or raw_step.get("text") or "")
            token_ids = raw_step.get("token_ids")
            if isinstance(token_ids, list):
                token_count = len(token_ids)
        else:
            step_text = str(
                getattr(raw_step, "raw_text", None)
                or getattr(raw_step, "text", None)
                or ""
            )
            token_ids = getattr(raw_step, "token_ids", None)
            if isinstance(token_ids, list):
                token_count = len(token_ids)

        if step_text.strip():
            entries.append({"text": step_text, "tokens": token_count})

    return entries


def _event_stage_for_family(family: str, index: int, total: int) -> str:
    if family == "single_pass":
        return "generation"
    if family == "tree_search":
        return "tree_select" if index == total - 1 else "tree_expand"
    if family == "reranking":
        return "selection" if index == total - 1 else "candidate_generation"
    if family == "sample_and_vote":
        return "selection" if index == total - 1 else "sampling"
    return "reasoning"


def _normalize_confidence(value: Any) -> float:
    numeric = _to_float(value)
    if numeric is None or not math.isfinite(numeric):
        return 0.0
    if 0.0 <= numeric <= 1.0:
        return float(numeric)
    # Convert unconstrained score into a bounded confidence value.
    stabilized = max(-60.0, min(60.0, numeric))
    return _clamp(1.0 / (1.0 + math.exp(-stabilized)))


def _confidence_from_score(
    value: Any,
    scorer: Optional[Dict[str, Any]],
    fallback: float = 0.0,
) -> float:
    numeric = _to_float(value)
    if numeric is None or not math.isfinite(numeric):
        return _normalize_confidence(fallback)

    confidence = _normalize_confidence(numeric)
    if scorer and scorer.get("direction") == "lower_better":
        return _clamp(1.0 - confidence)
    return confidence


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_first_numeric(value: Any) -> Optional[float]:
    if not isinstance(value, dict):
        return None
    for item in value.values():
        numeric = _to_float(item)
        if numeric is not None and math.isfinite(numeric):
            return numeric
    return None


def _coerce_bool(value: Any, default: Optional[bool]) -> Optional[bool]:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_prm_device(device: str) -> str:
    """Resolve PRM device string. 'auto' picks the last available GPU."""
    if device != "auto":
        return device
    try:
        import torch

        n = torch.cuda.device_count()
        if n > 1:
            return f"cuda:{n - 1}"
        return "cuda:0"
    except Exception:
        return "cuda:0"


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    return _coerce_int(value, default=0, minimum=0)


def _coerce_int_list(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list):
        return None
    parsed: List[int] = []
    for item in value:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed or None


def _coerce_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            output.append(text)
    return output


def _load_cached_examples_bundle() -> Dict[str, Any]:
    if not _CACHED_EXAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Cached debugger examples missing: {_CACHED_EXAMPLES_PATH}"
        )

    data = json.loads(_CACHED_EXAMPLES_PATH.read_text(encoding="utf-8"))
    examples = data.get("examples")

    if not isinstance(examples, list):
        # Backward compatibility while migrating to one-block examples schema.
        legacy_scenarios = data.get("scenarios")
        legacy_payloads = data.get("payloads")
        if isinstance(legacy_scenarios, list) and isinstance(legacy_payloads, dict):
            examples = []
            for scenario in legacy_scenarios:
                if not isinstance(scenario, dict):
                    continue
                scenario_id = str(scenario.get("id", "")).strip()
                if not scenario_id:
                    continue
                payloads = legacy_payloads.get(scenario_id)
                if not isinstance(payloads, dict):
                    continue
                examples.append(
                    {
                        "id": scenario_id,
                        "title": scenario.get("title"),
                        "description": scenario.get("description"),
                        "default_budget": scenario.get("default_budget"),
                        "payloads": payloads,
                    }
                )
        else:
            raise ValueError("cached_examples.json must contain top-level 'examples'.")

    normalized_scenarios: List[Dict[str, Any]] = []
    normalized_payloads: Dict[str, Dict[str, Any]] = {}
    for item in examples:
        if not isinstance(item, dict):
            continue

        scenario_id = str(item.get("id", "")).strip()
        scenario_payloads = item.get("payloads")
        if not scenario_id or not isinstance(scenario_payloads, dict):
            continue

        available_budgets = _collect_available_budgets(scenario_payloads)
        if not available_budgets:
            continue

        normalized_payloads[scenario_id] = scenario_payloads
        normalized_scenarios.append(
            {
                "id": scenario_id,
                "title": str(item.get("title") or scenario_id),
                "description": str(item.get("description") or ""),
                "available_budgets": available_budgets,
                "default_budget": _pick_budget(
                    item.get("default_budget"), available_budgets
                ),
            }
        )

    return {"scenarios": normalized_scenarios, "payloads": normalized_payloads}


def _collect_available_budgets(scenario_payloads: Dict[str, Any]) -> List[int]:
    budgets: List[int] = []
    for key in scenario_payloads.keys():
        try:
            value = int(key)
        except (TypeError, ValueError):
            continue
        budgets.append(value)
    return sorted(set(budgets))


def _find_strategy(strategy_id: str) -> Dict[str, Any]:
    for strategy in SUPPORTED_STRATEGIES:
        if strategy.get("id") == strategy_id:
            return deepcopy(strategy)
    raise ValueError(f"Unsupported strategy_id: {strategy_id}")


def _find_scorer(scorer_id: str) -> Dict[str, Any]:
    for scorer in SUPPORTED_SCORERS:
        if scorer.get("id") == scorer_id:
            return deepcopy(scorer)
    raise ValueError(f"Unsupported scorer_id: {scorer_id}")


def _probe_model_capabilities(
    provider: str,
    model_id: str,
    api_key: str,
) -> ModelValidationResult:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for model validation.") from exc

    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": 20.0,
        "max_retries": 0,
    }
    base_url = _PROVIDER_BASE_URLS.get(provider)
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    try:
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=4,
            temperature=0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Model validation request failed: {_compact_error(exc)}"
        ) from exc

    supports_logprobs, logprobs_reason = _probe_logprobs_support(client, model_id)
    supports_prefill, prefill_reason = _probe_prefill_support(client, model_id)

    return ModelValidationResult(
        supports_logprobs=supports_logprobs,
        supports_prefill=supports_prefill,
        supports_logprobs_reason=logprobs_reason,
        supports_prefill_reason=prefill_reason,
    )


def _probe_logprobs_support(client: Any, model_id: str) -> Tuple[bool, str]:
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Answer with one token: yes"}],
            max_tokens=2,
            temperature=0,
            logprobs=True,
            top_logprobs=2,
        )
    except Exception as exc:
        error_text = _compact_error(exc)
        if _is_capability_rejection(error_text, ("logprob", "top_logprobs")):
            return False, error_text
        return False, f"logprobs probe failed: {error_text}"

    choice = (response.choices or [None])[0]
    has_logprobs = bool(getattr(choice, "logprobs", None))
    if has_logprobs:
        return True, "logprobs probe succeeded."
    return False, "logprobs field missing in response."


def _probe_prefill_support(client: Any, model_id: str) -> Tuple[bool, str]:
    prefill = "A transformer model is a type of neural network that"
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Explain what a transformer model is in simple terms.",
                },
                {"role": "assistant", "content": prefill},
            ],
            max_tokens=60,
            temperature=0,
        )
    except Exception as exc:
        error_text = _compact_error(exc)
        if _is_capability_rejection(error_text, ("prefix", "prefill")):
            return False, error_text
        return False, f"prefill probe failed: {error_text}"

    text = (
        (response.choices[0].message.content or "").strip() if response.choices else ""
    )
    if not text:
        return False, "prefill probe: empty response"

    # Two valid prefill behaviors:
    # 1) Full echo: response includes prefix + continuation → starts_with
    # 2) Continuation only: response is just the new text that continues
    #    mid-sentence (e.g. "'s particularly good at..." continuing "that")
    if text.startswith(prefill):
        return True, "response starts with prefill text (full echo)"

    # Check continuation-only: the response should continue mid-sentence,
    # not start a fresh sentence.  A fresh response would start with a
    # capital letter or a complete new sentence.
    first_char = text[0] if text else ""
    starts_mid_sentence = first_char in ("'", ",", ".", ";", " ", "-") or (
        first_char.isalpha() and first_char.islower()
    )
    if starts_mid_sentence:
        return True, f"response continues mid-sentence: {text[:60]!r}"

    return False, f"response starts a new sentence (no continuation): {text[:60]!r}"


def _is_capability_rejection(error_text: str, tokens: Tuple[str, ...]) -> bool:
    lowered = error_text.lower()
    if not any(token in lowered for token in tokens):
        return False
    rejection_hints = (
        "unsupported",
        "not supported",
        "unrecognized",
        "unknown",
        "invalid",
        "not allowed",
        "does not support",
    )
    return any(hint in lowered for hint in rejection_hints)


def _compact_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)

    # OpenAI SDK APIStatusError stores parsed body in .body
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        # Standard: {"error": {"message": "..."}} or {"message": "..."}
        inner = body.get("error", body)
        if isinstance(inner, dict):
            msg = inner.get("message") or inner.get("msg")
            if msg:
                msg = str(msg).strip()
                return f"Error {status}: {msg}" if status else msg
        # Non-standard dict — try "detail" key (FastAPI-style)
        detail = body.get("detail")
        if detail:
            msg = str(detail).strip()
            return f"Error {status}: {msg}" if status else msg
    if isinstance(body, str) and body.strip():
        return f"Error {status}: {body.strip()}" if status else body.strip()

    # Fallback: full error string so users can debug unexpected providers
    text = " ".join(str(exc).split()).strip()
    return text if text else exc.__class__.__name__


def _budget_unit_for_family(family: str) -> str:
    if family == "tree_search":
        return "node_expansions"
    if family == "sample_and_vote":
        return "paths"
    if family == "reranking":
        return "candidate_rollouts"
    return "steps"


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return f"{api_key[:2]}***"
    return f"{api_key[:4]}...{api_key[-4:]}"


def _resolve_advanced_config(
    strategy_id: str,
    scorer_id: Optional[str],
    advanced_config_yaml: Optional[str],
) -> Tuple[Dict[str, Any], str]:
    base_config = _build_advanced_config_template_dict(strategy_id, scorer_id)
    if not advanced_config_yaml or not advanced_config_yaml.strip():
        return base_config, _dump_yaml(base_config)

    try:
        loaded = yaml.safe_load(advanced_config_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"Advanced config YAML is invalid: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("Advanced config YAML must be a mapping at top level.")

    merged = _merge_mappings(base_config, loaded)
    if not scorer_id:
        merged.pop("scorer", None)
    return merged, _dump_yaml(merged)


def _build_advanced_config_template_dict(
    strategy_id: str,
    scorer_id: Optional[str],
) -> Dict[str, Any]:
    strategy_path = _STRATEGY_CONFIG_PATHS.get(strategy_id)
    if strategy_path is None:
        raise ValueError(f"Unsupported strategy_id: {strategy_id}")

    config: Dict[str, Any] = {
        "prompt": """
You will be presented with a <Question>. Before providing the [Answer], you should first think step-by-step carefully.

Your response format:
<start of response>
Reasoning Steps:
- Step 1: [Your first reasoning step]
- Step 2: [Your second reasoning step]
- Step 3: [Next step, and so on...]
...
- Step N: [Final reasoning step]
<Answer>: [Your final answer]
<end of response>

Strict Requirements:
- DO NOT include any text outside the specified format.
- Each reasoning step MUST be written on a **single line only**: NO line breaks, bullet points, or substeps within a step.
- Each step should express one precise and **self-contained** logical operation, deduction, calculation, or fact application.
- Steps MUST provide explicit result of the step or concrete reasoning outcomes. Avoid vague explanations or meta-descriptions of the reasoning process.
    - For example:
        - Good: "- Step 1: Multiply 5 by 4, which equals 20."
        - Bad: "- Step 1: Multiply 5 by 4." (no result of the step or concrete reasoning outcome)
- Continue writing steps until the problem is solved.
- Violating ANY requirement above is NOT acceptable.

Now answer:
<Question>: """,
        "generation": _load_yaml_mapping(_DEFAULT_GENERATION_CONFIG_PATH),
        "strategy": _load_yaml_mapping(strategy_path),
    }

    if scorer_id:
        scorer_path = _SCORER_CONFIG_PATHS.get(scorer_id)
        if scorer_path is None:
            raise ValueError(f"Unsupported scorer_id: {scorer_id}")
        config["scorer"] = _load_yaml_mapping(scorer_path)

    return config


def _extract_prompt_from_advanced_config(
    advanced_config: Dict[str, Any],
    fallback_prompt: str = "",
) -> str:
    prompt_value = advanced_config.get("prompt")
    if isinstance(prompt_value, str):
        return prompt_value.strip()
    return str(fallback_prompt or "").strip()


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Advanced config source is missing: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return deepcopy(loaded)


def _merge_mappings(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )


def _safe_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def _coerce_int(
    value: Any,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _coerce_float(
    value: Any,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _pick_budget(requested: Optional[int], available: List[int]) -> int:
    if not available:
        return DEFAULT_BUDGET

    try:
        target = DEFAULT_BUDGET if requested is None else int(requested)
    except (TypeError, ValueError):
        target = DEFAULT_BUDGET

    return min(available, key=lambda value: (abs(value - target), value))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
