"""Visual debugger payload helpers."""

from __future__ import annotations

import hashlib
import json
import random
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
        "requires_prefill": False,
    },
    {
        "id": "online_best_of_n",
        "name": "Online Best-of-N",
        "family": "reranking",
        "summary": "Iterative candidate generation with stepwise reranking.",
        "requires_scorer": True,
        "requires_logprobs": False,
        "requires_prefill": False,
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
        "requires_scorer": True,
        "requires_logprobs": False,
        "requires_prefill": False,
    },
]

SUPPORTED_SCORERS: List[Dict[str, Any]] = [
    {
        "id": "prm",
        "name": "PRM",
        "direction": "higher_better",
        "threshold": 0.72,
        "summary": "Process Reward Model trajectory quality score.",
        "requires_logprobs": False,
    },
    {
        "id": "sequence_prob",
        "name": "Sequence Prob",
        "direction": "higher_better",
        "threshold": 0.65,
        "summary": "Cumulative sequence probability from token logprobs.",
        "requires_logprobs": True,
    },
    {
        "id": "perplexity",
        "name": "Perplexity",
        "direction": "lower_better",
        "threshold": 0.36,
        "summary": "Per-token perplexity estimated from generation logprobs.",
        "requires_logprobs": True,
    },
    {
        "id": "entropy",
        "name": "Entropy",
        "direction": "lower_better",
        "threshold": 0.34,
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
        if not item.get("requires_logprobs") or supports_logprobs
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

    strategy_entry = _build_strategy_entry(
        strategy=strategy,
        scorer=scorer,
        question=question,
        gold_answer=normalized_gold_answer,
        budget=selected_budget,
        shared_prompt=resolved_shared_prompt,
        model_config=model_config,
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


def _build_strategy_entry(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    question: str,
    gold_answer: str,
    budget: int,
    shared_prompt: str,
    model_config: Dict[str, str],
    has_gold_answer: bool,
    advanced_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run = _build_strategy_scorer_run(
        strategy=strategy,
        scorer=scorer,
        question=question,
        gold_answer=gold_answer,
        budget=budget,
        shared_prompt=shared_prompt,
        model_config=model_config,
        has_gold_answer=has_gold_answer,
        advanced_config=advanced_config,
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
    try:
        client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Continue exactly from the assistant turn.",
                },
                {"role": "assistant", "content": "The answer is", "prefix": True},
            ],
            max_tokens=2,
            temperature=0,
        )
    except Exception as exc:
        error_text = _compact_error(exc)
        if _is_capability_rejection(error_text, ("prefix", "prefill")):
            return False, error_text
        return False, f"prefill probe failed: {error_text}"

    return True, "assistant prefix prefill probe succeeded."


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
    text = " ".join(str(exc).split())
    return text if text else exc.__class__.__name__


def _build_strategy_scorer_run(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    question: str,
    gold_answer: str,
    budget: int,
    shared_prompt: str,
    model_config: Dict[str, str],
    has_gold_answer: bool,
    advanced_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scorer_id = scorer["id"] if scorer else "none"
    generation_config = _safe_mapping(
        (advanced_config or {}).get("generation")
        if isinstance(advanced_config, dict)
        else None
    )
    strategy_config = _safe_mapping(
        (advanced_config or {}).get("strategy")
        if isinstance(advanced_config, dict)
        else None
    )
    scorer_config = (
        _safe_mapping((advanced_config or {}).get("scorer"))
        if scorer and isinstance(advanced_config, dict)
        else {}
    )

    max_new_tokens = _coerce_int(
        generation_config.get("max_new_tokens"),
        default=500,
        minimum=1,
        maximum=200000,
    )
    temperature = _coerce_float(
        generation_config.get("temperature"),
        default=0.7,
        minimum=0.0,
        maximum=2.0,
    )

    if scorer and "threshold" in scorer_config:
        scorer["threshold"] = _coerce_float(
            scorer_config.get("threshold"),
            default=float(scorer.get("threshold", 0.5)),
            minimum=0.0,
            maximum=1.0,
        )

    rng = _make_rng(
        strategy["id"],
        scorer_id,
        question,
        gold_answer or "",
        str(budget),
        shared_prompt,
        model_config.get("provider", ""),
        model_config.get("model_id", ""),
    )

    family = strategy.get("family", "single_pass")
    base_quality = _strategy_quality_base(strategy["id"], family)
    scorer_shift = _scorer_quality_shift(scorer_id if scorer else None)
    difficulty = _difficulty_from_question(question)
    budget_factor = budget / float(DEBUGGER_BUDGETS[-1])
    temperature_shift = (0.75 - temperature) * 0.04

    success_prob = _clamp(
        base_quality
        + scorer_shift
        + budget_factor * 0.14
        - difficulty
        + temperature_shift
    )
    is_correct: Optional[bool] = None
    if has_gold_answer:
        is_correct = rng.random() < success_prob
        answer = gold_answer if is_correct else _perturb_answer(gold_answer, rng)
    else:
        answer = _synthesize_answer_from_question(question, rng)

    quality_score = _clamp(
        base_quality
        + scorer_shift
        + (0.08 if is_correct is True else (-0.12 if is_correct is False else 0.0))
        + budget_factor * 0.10
        + temperature_shift
    )
    confidence = _clamp(
        quality_score
        + (0.07 if is_correct is True else (-0.08 if is_correct is False else 0.0))
        + (rng.random() - 0.5) * 0.06,
        0.05,
        0.98,
    )

    estimated_tokens = int(520 + budget * 92 + quality_score * 230 + rng.random() * 130)
    token_cap = max(256, int(max_new_tokens * 2.2))
    used_budget = _used_budget_for_family(family, budget, rng)
    strategy_budget_cap = _strategy_budget_cap(strategy_config)
    if strategy_budget_cap is not None:
        used_budget = min(used_budget, strategy_budget_cap)

    run: Dict[str, Any] = {
        "budget": budget,
        "budget_unit": _budget_unit_for_family(family),
        "used_budget": used_budget,
        "tokens_used": min(estimated_tokens, token_cap),
        "latency_ms": int(3200 + budget * 760 + rng.random() * 1700),
        "provider": model_config.get("provider", "openrouter"),
        "model_id": model_config.get("model_id", "openai/gpt-4o-mini"),
        "strategy": {
            "id": strategy["id"],
            "name": strategy["name"],
            "family": family,
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
        "final": {
            "answer": answer,
            "is_correct": is_correct,
            "selected_trajectory": f"{strategy['id']}_{scorer_id}_selected",
            "confidence": confidence,  # Required metric in summary UI.
            "selection_reason": (
                f"{strategy['name']} selected this trajectory."
                if scorer is None
                else (
                    f"{strategy['name']} selected this trajectory using "
                    f"{scorer['name']} scoring signals."
                )
            ),
        },
        "config": {
            "generation": deepcopy(generation_config),
            "strategy": deepcopy(strategy_config),
            "scorer": deepcopy(scorer_config) if scorer else None,
        },
    }

    run["events"] = _build_generic_events(
        strategy=strategy,
        scorer=scorer,
        answer=answer,
        confidence=confidence,
        budget=budget,
        rng=rng,
        gold_answer=gold_answer,
        has_gold_answer=has_gold_answer,
    )

    if family == "tree_search":
        run["tree"] = _build_tree_layout(
            run_key=f"{strategy['id']}_{scorer_id}",
            confidence=confidence,
            rng=rng,
            events=run["events"],
        )

    return run


def _build_generic_events(
    strategy: Dict[str, Any],
    scorer: Optional[Dict[str, Any]],
    answer: str,
    confidence: float,
    budget: int,
    rng: random.Random,
    gold_answer: str,
    has_gold_answer: bool,
) -> List[Dict[str, Any]]:
    family = strategy.get("family", "single_pass")
    scorer_signal = (
        _scorer_signal(scorer, _scorer_value_from_confidence(scorer, confidence))
        if scorer
        else None
    )
    scorer_key = scorer["id"] if scorer else "score"

    if family == "single_pass":
        signals: List[Dict[str, Any]] = [
            {
                "name": "confidence",
                "value": confidence,
                "direction": "higher_better",
                "threshold": 0.70,
            }
        ]
        if scorer_signal:
            signals.append(scorer_signal)

        candidate_signals: Dict[str, float] = {"trajectory_confidence": confidence}
        if scorer:
            candidate_signals[scorer_key] = _scorer_value_from_confidence(
                scorer, confidence
            )

        return [
            {
                "step": 1,
                "title": "Single-pass reasoning generation",
                "stage": "generation",
                "decision": {
                    "action": "stop",
                    "reason": "Single-pass baseline emits one chain.",
                },
                "signals": signals,
                "candidates": [
                    {
                        "id": f"{strategy['id']}_{scorer_key}_c1",
                        "label": "Generated chain",
                        "text": f"Predicted answer {answer}.",
                        "answer": answer,
                        "status": "selected",
                        "selected": True,
                        "signals": candidate_signals,
                    }
                ],
            }
        ]

    wrong_answer = (
        _perturb_answer(gold_answer, rng)
        if has_gold_answer
        else _synthesize_answer_from_question(f"{answer}_alt", rng)
    )
    warmup_signal = (
        _scorer_signal(
            scorer,
            _scorer_value_from_confidence(
                scorer, _clamp(confidence - 0.10 + rng.random() * 0.04)
            ),
        )
        if scorer
        else None
    )

    first_signals: List[Dict[str, Any]] = [
        {
            "name": "confidence",
            "value": _clamp(confidence - 0.12),
            "direction": "higher_better",
            "threshold": 0.70,
        }
    ]
    if warmup_signal:
        first_signals.append(warmup_signal)

    first_candidate_signals = {"score": _clamp(confidence - 0.05)}
    second_candidate_signals = {"score": _clamp(confidence - 0.18)}
    if scorer:
        first_candidate_signals[scorer_key] = _scorer_value_from_confidence(
            scorer, _clamp(confidence - 0.04)
        )
        second_candidate_signals[scorer_key] = _scorer_value_from_confidence(
            scorer, _clamp(confidence - 0.20)
        )

    first = {
        "step": 1,
        "title": "Warmup candidate generation",
        "stage": "sampling" if family == "sample_and_vote" else "candidate_generation",
        "decision": {
            "action": "escalate" if budget > 4 else "stop",
            "reason": "Need additional budget." if budget > 4 else "Budget exhausted.",
        },
        "signals": first_signals,
        "candidates": [
            {
                "id": f"{strategy['id']}_{scorer_key}_p1",
                "label": "Candidate 1",
                "text": f"Candidate predicts {answer}.",
                "answer": answer,
                "status": "kept" if budget > 4 else "selected",
                "selected": budget <= 4,
                "signals": first_candidate_signals,
            },
            {
                "id": f"{strategy['id']}_{scorer_key}_p2",
                "label": "Candidate 2",
                "text": f"Competing candidate predicts {wrong_answer}.",
                "answer": wrong_answer,
                "status": "pruned",
                "selected": False,
                "signals": second_candidate_signals,
            },
        ],
    }

    if budget <= 4:
        return [first]

    second_stage = "selection"
    if family == "reranking":
        second_stage = "reranking"
    elif family == "tree_search":
        second_stage = "tree_select"

    second_signals: List[Dict[str, Any]] = [
        {
            "name": "confidence",
            "value": confidence,
            "direction": "higher_better",
            "threshold": 0.70,
        }
    ]
    if scorer_signal:
        second_signals.append(scorer_signal)

    selected_candidate_signals = {"score": confidence}
    if scorer:
        selected_candidate_signals[scorer_key] = _scorer_value_from_confidence(
            scorer, confidence
        )

    return [
        first,
        {
            "step": 2,
            "title": "Escalated selection",
            "stage": second_stage,
            "decision": {
                "action": "stop",
                "reason": "Confidence stabilized after escalation.",
            },
            "signals": second_signals,
            "candidates": [
                {
                    "id": f"{strategy['id']}_{scorer_key}_p3",
                    "label": "Selected candidate",
                    "text": f"Selected answer {answer}.",
                    "answer": answer,
                    "status": "selected",
                    "selected": True,
                    "signals": selected_candidate_signals,
                }
            ],
        },
    ]


def _signal_list_to_score_map(
    signals: Optional[List[Dict[str, Any]]]
) -> Dict[str, float]:
    score_map: Dict[str, float] = {}
    for signal in signals or []:
        name = str(signal.get("name", "")).strip()
        value = signal.get("value")
        if not name or not isinstance(value, (int, float)):
            continue
        score_map[name] = float(value)
    return score_map


def _node_payload_from_candidate(
    step: int,
    candidate: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not candidate:
        return {"step": step}

    payload: Dict[str, Any] = {"step": step}
    candidate_id = candidate.get("id")
    if isinstance(candidate_id, str) and candidate_id:
        payload["candidate_id"] = candidate_id

    text = candidate.get("text")
    if isinstance(text, str) and text.strip():
        payload["text"] = text

    status = candidate.get("status")
    if isinstance(status, str) and status:
        payload["status"] = status

    signals = candidate.get("signals")
    if isinstance(signals, dict) and signals:
        payload["scores"] = deepcopy(signals)

    return payload


def _first_numeric_score(signals: Any) -> Optional[float]:
    if not isinstance(signals, dict):
        return None
    for value in signals.values():
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _build_tree_layout(
    run_key: str,
    confidence: float,
    rng: random.Random,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    event_list = events or []
    if not event_list:
        root = _clamp(0.50 + (rng.random() - 0.5) * 0.06)
        return {
            "nodes": [
                {
                    "id": f"{run_key}_root",
                    "label": "Root",
                    "value": root,
                    "depth": 0,
                    "x": 0.50,
                    "y": 0.14,
                },
                {
                    "id": f"{run_key}_selected",
                    "label": "Selected",
                    "value": confidence,
                    "depth": 1,
                    "x": 0.50,
                    "y": 0.55,
                },
            ],
            "edges": [
                {"source": f"{run_key}_root", "target": f"{run_key}_selected"},
            ],
            "selected_path": [f"{run_key}_root", f"{run_key}_selected"],
        }

    first_event = event_list[0]
    first_step = max(1, int(first_event.get("step", 1) or 1))
    first_candidates = list(first_event.get("candidates") or [])
    root_scores = _signal_list_to_score_map(first_event.get("signals"))
    root_value = (
        root_scores.get("confidence")
        if isinstance(root_scores.get("confidence"), (int, float))
        else _clamp(0.50 + (rng.random() - 0.5) * 0.06)
    )

    root_text_parts: List[str] = []
    first_title = first_event.get("title")
    if isinstance(first_title, str) and first_title.strip():
        root_text_parts.append(first_title.strip())
    first_reason = (first_event.get("decision") or {}).get("reason")
    if isinstance(first_reason, str) and first_reason.strip():
        root_text_parts.append(first_reason.strip())

    root_id = f"{run_key}_root"
    nodes: List[Dict[str, Any]] = [
        {
            "id": root_id,
            "label": "Root",
            "value": _clamp(float(root_value)),
            "depth": 0,
            "x": 0.50,
            "y": 0.14,
            "step": first_step,
            "text": " ".join(root_text_parts),
            "scores": root_scores,
        }
    ]
    edges: List[Dict[str, Any]] = []
    selected_path: List[str] = [root_id]
    used_node_ids = {root_id}

    selected_first_node_id = root_id
    selected_first_node_x = 0.50
    selected_first_candidate = next(
        (candidate for candidate in first_candidates if candidate.get("selected")),
        first_candidates[0] if first_candidates else None,
    )

    for index, candidate in enumerate(first_candidates):
        base_id = str(candidate.get("id") or f"{run_key}_s1_c{index + 1}")
        node_id = base_id
        if node_id in used_node_ids:
            node_id = f"{base_id}__{index + 1}"
        used_node_ids.add(node_id)

        total = len(first_candidates)
        x = 0.50 if total <= 1 else 0.15 + (0.70 * index) / max(total - 1, 1)
        candidate_signals = candidate.get("signals") or {}
        candidate_value = _first_numeric_score(candidate_signals)
        if candidate_value is None:
            candidate_value = _clamp(float(root_value))

        candidate_payload = _node_payload_from_candidate(first_step, candidate)
        nodes.append(
            {
                "id": node_id,
                "label": str(candidate.get("label") or f"Candidate {index + 1}"),
                "value": _clamp(float(candidate_value)),
                "depth": 1,
                "x": x,
                "y": 0.48,
                **candidate_payload,
            }
        )
        edges.append({"source": root_id, "target": node_id})

        if candidate is selected_first_candidate or (
            selected_first_candidate
            and candidate.get("id")
            and candidate.get("id") == selected_first_candidate.get("id")
        ):
            selected_first_node_id = node_id
            selected_first_node_x = x

    if selected_first_node_id != root_id:
        selected_path.append(selected_first_node_id)

    second_event = event_list[1] if len(event_list) > 1 else None
    if second_event:
        second_candidates = list(second_event.get("candidates") or [])
        second_selected = next(
            (candidate for candidate in second_candidates if candidate.get("selected")),
            second_candidates[0] if second_candidates else None,
        )
        if second_selected:
            second_step = max(
                1, int(second_event.get("step", first_step + 1) or (first_step + 1))
            )
            base_id = str(second_selected.get("id") or f"{run_key}_s2_selected")
            second_node_id = (
                f"{base_id}__s{second_step}" if base_id in used_node_ids else base_id
            )
            used_node_ids.add(second_node_id)

            second_signals = second_selected.get("signals") or {}
            second_value = _first_numeric_score(second_signals)
            if second_value is None:
                second_event_scores = _signal_list_to_score_map(
                    second_event.get("signals")
                )
                second_value = second_event_scores.get("confidence")
            if second_value is None:
                second_value = confidence

            second_payload = _node_payload_from_candidate(second_step, second_selected)
            nodes.append(
                {
                    "id": second_node_id,
                    "label": str(second_selected.get("label") or "Selected"),
                    "value": _clamp(float(second_value)),
                    "depth": 2,
                    "x": selected_first_node_x,
                    "y": 0.82,
                    **second_payload,
                }
            )
            edges.append({"source": selected_first_node_id, "target": second_node_id})
            selected_path.append(second_node_id)

    return {
        "nodes": nodes,
        "edges": edges,
        "selected_path": selected_path,
    }


def _scorer_signal(scorer: Dict[str, Any], value: float) -> Dict[str, Any]:
    return {
        "name": scorer["id"],
        "value": value,
        "direction": scorer["direction"],
        "threshold": scorer.get("threshold"),
    }


def _scorer_value_from_confidence(scorer: Dict[str, Any], confidence: float) -> float:
    if scorer["direction"] == "higher_better":
        return _clamp(confidence)
    return _clamp(1 - confidence)


def _difficulty_from_question(question: str) -> float:
    seed = _hash_seed(question)
    return 0.10 + (seed % 27) / 100.0


def _strategy_quality_base(strategy_id: str, family: str) -> float:
    by_strategy = {
        "baseline": 0.55,
        "beam_search": 0.67,
        "online_best_of_n": 0.66,
        "offline_best_of_n": 0.64,
        "self_consistency": 0.65,
    }
    by_family = {
        "single_pass": 0.56,
        "tree_search": 0.66,
        "reranking": 0.64,
        "sample_and_vote": 0.65,
    }
    return by_strategy.get(strategy_id, by_family.get(family, 0.60))


def _scorer_quality_shift(scorer_id: Optional[str]) -> float:
    if not scorer_id:
        return 0.0
    return {
        "prm": 0.05,
        "sequence_prob": 0.03,
        "perplexity": 0.01,
        "entropy": 0.00,
    }.get(scorer_id, 0.00)


def _budget_unit_for_family(family: str) -> str:
    if family == "tree_search":
        return "node_expansions"
    if family == "sample_and_vote":
        return "paths"
    if family == "reranking":
        return "candidate_rollouts"
    return "steps"


def _used_budget_for_family(family: str, budget: int, rng: random.Random) -> int:
    if family == "sample_and_vote":
        return budget
    if family == "single_pass":
        return max(1, min(2, budget))
    if family == "tree_search":
        return min(budget, max(3, int(budget * (0.7 + rng.random() * 0.2))))
    return min(budget, max(2, int(budget * (0.75 + rng.random() * 0.2))))


def _perturb_answer(gold_answer: Optional[str], rng: random.Random) -> str:
    text = str(gold_answer).strip()
    try:
        value = float(text)
        if value.is_integer():
            delta = max(1, int(rng.random() * 4) + 1)
            sign = -1 if rng.random() < 0.5 else 1
            return str(int(value + sign * delta))
        delta = (rng.random() - 0.5) * 0.8
        return f"{value + delta:.2f}"
    except ValueError:
        if not text:
            return "unknown"
        return f"{text}_alt"


def _synthesize_answer_from_question(question: str, rng: random.Random) -> str:
    """Create a deterministic synthetic answer when no gold answer is provided."""
    seed = _hash_seed(question)
    if seed % 2 == 0:
        return str((seed % 97) + 3)
    value = ((seed % 1300) / 100.0) + (rng.random() * 0.1)
    return f"{value:.2f}"


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
        "prompt": "Reason step-by-step carefully",
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


def _strategy_budget_cap(strategy_config: Dict[str, Any]) -> Optional[int]:
    raw_value = (
        strategy_config.get("max_steps")
        or strategy_config.get("num_paths")
        or strategy_config.get("num_trajectories")
    )
    if raw_value is None:
        return None
    return _coerce_int(raw_value, default=1, minimum=1)


def _pick_budget(requested: Optional[int], available: List[int]) -> int:
    if not available:
        return DEFAULT_BUDGET

    try:
        target = DEFAULT_BUDGET if requested is None else int(requested)
    except (TypeError, ValueError):
        target = DEFAULT_BUDGET

    return min(available, key=lambda value: (abs(value - target), value))


def _hash_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _make_rng(*parts: str) -> random.Random:
    return random.Random(_hash_seed(*parts))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _apply_comparison_rank(strategies: List[Dict[str, Any]]) -> None:
    ranked = sorted(
        strategies,
        key=lambda item: item.get("run", {}).get("final", {}).get("quality_score", 0.0),
        reverse=True,
    )
    rank_by_id = {item["id"]: index + 1 for index, item in enumerate(ranked)}
    for item in strategies:
        item["comparison_rank"] = rank_by_id.get(item["id"], len(strategies))
