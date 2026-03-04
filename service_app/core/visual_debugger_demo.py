"""Visual debugger payload helpers."""

from __future__ import annotations

import importlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        model_id_value = model_id_value[len("openai/") :]

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

    if text.startswith(prefill):
        return True, "response starts with prefill text (full echo)"

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

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            msg = inner.get("message") or inner.get("msg")
            if msg:
                msg = str(msg).strip()
                return f"Error {status}: {msg}" if status else msg
        detail = body.get("detail")
        if detail:
            msg = str(detail).strip()
            return f"Error {status}: {msg}" if status else msg
    if isinstance(body, str) and body.strip():
        return f"Error {status}: {body.strip()}" if status else body.strip()

    text = " ".join(str(exc).split()).strip()
    return text if text else exc.__class__.__name__


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return f"{api_key[:2]}***"
    return f"{api_key[:4]}...{api_key[-4:]}"


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


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Advanced config source is missing: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return deepcopy(loaded)


def _dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )


def _pick_budget(requested: Optional[int], available: List[int]) -> int:
    if not available:
        return DEFAULT_BUDGET

    try:
        target = DEFAULT_BUDGET if requested is None else int(requested)
    except (TypeError, ValueError):
        target = DEFAULT_BUDGET

    return min(available, key=lambda value: (abs(value - target), value))
