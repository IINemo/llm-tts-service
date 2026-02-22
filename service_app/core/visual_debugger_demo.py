"""Visual debugger payload helpers.

- Demo scenario data is loaded from ``service_app/static/debugger/cached_examples.json``.
- Custom single-sample runs are synthesized server-side for prototype use.
"""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BUDGET = 8
DEBUGGER_BUDGETS = [4, 8, 12]

_CACHED_EXAMPLES_PATH = (
    Path(__file__).resolve().parents[1] / "static" / "debugger" / "cached_examples.json"
)

SUPPORTED_STRATEGIES: List[Dict[str, str]] = [
    {
        "id": "baseline",
        "name": "Baseline (Raw CoT)",
        "family": "single_pass",
        "summary": "Single-pass raw chain-of-thought without search or reranking.",
    },
    {
        "id": "beam_search",
        "name": "Beam Search (ToT)",
        "family": "tree_search",
        "summary": "Tree-of-thought expansion with beam pruning.",
    },
    {
        "id": "online_best_of_n",
        "name": "Online Best-of-N",
        "family": "reranking",
        "summary": "Iterative candidate generation with stepwise reranking.",
    },
    {
        "id": "offline_best_of_n",
        "name": "Offline Best-of-N",
        "family": "reranking",
        "summary": "Generate full trajectories first, then rerank at the end.",
    },
    {
        "id": "self_consistency",
        "name": "Self-Consistency",
        "family": "sample_and_vote",
        "summary": "Sample diverse trajectories and select by answer consensus.",
    },
]

SUPPORTED_SCORERS: List[Dict[str, Any]] = [
    {
        "id": "prm",
        "name": "PRM",
        "direction": "higher_better",
        "threshold": 0.72,
        "summary": "Process Reward Model trajectory quality score.",
    },
    {
        "id": "sequence_prob",
        "name": "Sequence Prob",
        "direction": "higher_better",
        "threshold": 0.65,
        "summary": "Cumulative sequence probability from token logprobs.",
    },
    {
        "id": "perplexity",
        "name": "Perplexity",
        "direction": "lower_better",
        "threshold": 0.36,
        "summary": "Per-token perplexity estimated from generation logprobs.",
    },
    {
        "id": "entropy",
        "name": "Entropy",
        "direction": "lower_better",
        "threshold": 0.34,
        "summary": "Mean token entropy of decoded reasoning steps.",
    },
]


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


def build_single_sample_payload(
    question: str,
    gold_answer: str,
    shared_prompt: str = "",
    budget: Optional[int] = None,
    provider: str = "openrouter",
    model_id: str = "openai/gpt-4o-mini",
    api_key: str = "",
    scenario_id: str = "custom_1",
    scenario_title: str = "Single Example",
    scenario_description: str = (
        "Custom single-sample run across all strategy/scorer combinations."
    ),
    input_source: str = "custom_single",
    api_key_masked_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one debugger payload for a single (question, gold answer) sample."""
    selected_budget = _pick_budget(budget, DEBUGGER_BUDGETS)
    masked_api_key = (
        api_key_masked_override
        if api_key_masked_override is not None
        else _mask_api_key(api_key)
    )

    prompt = question.strip()
    if shared_prompt.strip():
        prompt = f"{shared_prompt.strip()}\n\nQuestion: {question.strip()}"

    model_config = {
        "provider": provider,
        "model_id": model_id,
        "api_key_masked": masked_api_key,
    }

    strategies = _build_matrix_runs(
        question=question,
        gold_answer=gold_answer,
        budget=selected_budget,
        shared_prompt=shared_prompt,
        model_config=model_config,
    )
    _apply_comparison_rank(strategies)

    return {
        "scenario": {
            "id": scenario_id,
            "title": scenario_title,
            "description": scenario_description,
            "prompt": prompt,
            "ground_truth": gold_answer,
            "shared_prompt": shared_prompt,
            "input_source": input_source,
            "model_config": model_config,
            "strategy_count": len(SUPPORTED_STRATEGIES),
            "scorer_count": len(SUPPORTED_SCORERS),
            "run_count": len(strategies),
        },
        "available_budgets": DEBUGGER_BUDGETS,
        "selected_budget": selected_budget,
        "strategy_catalog": deepcopy(SUPPORTED_STRATEGIES),
        "scorer_catalog": deepcopy(SUPPORTED_SCORERS),
        "strategies": strategies,
    }


def _load_cached_examples_bundle() -> Dict[str, Any]:
    if not _CACHED_EXAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Cached debugger examples missing: {_CACHED_EXAMPLES_PATH}"
        )

    data = json.loads(_CACHED_EXAMPLES_PATH.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios")
    payloads = data.get("payloads")
    if not isinstance(scenarios, list) or not isinstance(payloads, dict):
        raise ValueError(
            "cached_examples.json must contain top-level 'scenarios' and 'payloads'."
        )

    normalized_scenarios: List[Dict[str, Any]] = []
    for item in scenarios:
        if not isinstance(item, dict):
            continue

        scenario_id = str(item.get("id", "")).strip()
        if not scenario_id:
            continue

        scenario_payloads = payloads.get(scenario_id)
        if not isinstance(scenario_payloads, dict):
            continue

        available_budgets = _collect_available_budgets(scenario_payloads)
        if not available_budgets:
            continue

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

    return {"scenarios": normalized_scenarios, "payloads": payloads}


def _collect_available_budgets(scenario_payloads: Dict[str, Any]) -> List[int]:
    budgets: List[int] = []
    for key in scenario_payloads.keys():
        try:
            value = int(key)
        except (TypeError, ValueError):
            continue
        budgets.append(value)
    return sorted(set(budgets))


def _build_matrix_runs(
    question: str,
    gold_answer: str,
    budget: int,
    shared_prompt: str,
    model_config: Dict[str, str],
) -> List[Dict[str, Any]]:
    runs = []
    for strategy in SUPPORTED_STRATEGIES:
        for scorer in SUPPORTED_SCORERS:
            run = _build_strategy_scorer_run(
                strategy=strategy,
                scorer=scorer,
                question=question,
                gold_answer=gold_answer,
                budget=budget,
                shared_prompt=shared_prompt,
                model_config=model_config,
            )
            run_id = f"{strategy['id']}__{scorer['id']}"
            runs.append(
                {
                    "id": run_id,
                    "strategy_id": strategy["id"],
                    "scorer_id": scorer["id"],
                    "name": f"{strategy['name']} · {scorer['name']}",
                    "family": strategy.get("family", "unknown"),
                    "summary": (
                        f"{strategy.get('summary', '')} Evaluated with "
                        f"{scorer.get('name', scorer['id'])}."
                    ),
                    "run": run,
                    "comparison_rank": 1,
                }
            )
    return runs


def _build_strategy_scorer_run(
    strategy: Dict[str, Any],
    scorer: Dict[str, Any],
    question: str,
    gold_answer: str,
    budget: int,
    shared_prompt: str,
    model_config: Dict[str, str],
) -> Dict[str, Any]:
    rng = _make_rng(
        strategy["id"],
        scorer["id"],
        question,
        gold_answer,
        str(budget),
        shared_prompt,
        model_config.get("provider", ""),
        model_config.get("model_id", ""),
    )

    family = strategy.get("family", "single_pass")
    base_quality = _strategy_quality_base(strategy["id"], family)
    scorer_shift = _scorer_quality_shift(scorer["id"])
    difficulty = _difficulty_from_question(question)
    budget_factor = budget / float(DEBUGGER_BUDGETS[-1])

    success_prob = _clamp(
        base_quality + scorer_shift + budget_factor * 0.14 - difficulty
    )
    is_correct = rng.random() < success_prob
    answer = gold_answer if is_correct else _perturb_answer(gold_answer, rng)

    quality_score = _clamp(
        base_quality
        + scorer_shift
        + (0.08 if is_correct else -0.12)
        + budget_factor * 0.10
    )
    confidence = _clamp(
        quality_score + (0.07 if is_correct else -0.08) + (rng.random() - 0.5) * 0.06,
        0.05,
        0.98,
    )

    run: Dict[str, Any] = {
        "budget": budget,
        "budget_unit": _budget_unit_for_family(family),
        "used_budget": _used_budget_for_family(family, budget, rng),
        "tokens_used": int(
            520 + budget * 92 + quality_score * 230 + rng.random() * 130
        ),
        "latency_ms": int(3200 + budget * 760 + rng.random() * 1700),
        "provider": model_config.get("provider", "openrouter"),
        "model_id": model_config.get("model_id", "openai/gpt-4o-mini"),
        "strategy": {
            "id": strategy["id"],
            "name": strategy["name"],
            "family": family,
        },
        "scorer": {
            "id": scorer["id"],
            "name": scorer["name"],
            "direction": scorer["direction"],
            "summary": scorer.get("summary", ""),
        },
        "final": {
            "answer": answer,
            "is_correct": is_correct,
            "selected_trajectory": f"{strategy['id']}_{scorer['id']}_selected",
            "confidence": confidence,
            "uncertainty": _clamp(1 - confidence),
            "quality_score": quality_score,
            "selection_reason": (
                f"{strategy['name']} selected this trajectory using "
                f"{scorer['name']} scoring signals."
            ),
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
    )

    if family == "tree_search":
        run["tree"] = _build_tree_layout(
            run_key=f"{strategy['id']}_{scorer['id']}",
            confidence=confidence,
            rng=rng,
        )

    return run


def _build_generic_events(
    strategy: Dict[str, Any],
    scorer: Dict[str, Any],
    answer: str,
    confidence: float,
    budget: int,
    rng: random.Random,
    gold_answer: str,
) -> List[Dict[str, Any]]:
    family = strategy.get("family", "single_pass")
    scorer_signal = _scorer_signal(
        scorer, _scorer_value_from_confidence(scorer, confidence)
    )

    if family == "single_pass":
        return [
            {
                "step": 1,
                "title": "Single-pass reasoning generation",
                "stage": "generation",
                "decision": {
                    "action": "stop",
                    "reason": "Single-pass baseline emits one chain.",
                },
                "signals": [
                    {
                        "name": "confidence",
                        "value": confidence,
                        "direction": "higher_better",
                        "threshold": 0.70,
                    },
                    scorer_signal,
                ],
                "candidates": [
                    {
                        "id": f"{strategy['id']}_{scorer['id']}_c1",
                        "label": "Generated chain",
                        "text": f"Predicted answer {answer}.",
                        "answer": answer,
                        "status": "selected",
                        "selected": True,
                        "signals": {
                            scorer["id"]: _scorer_value_from_confidence(
                                scorer, confidence
                            ),
                            "trajectory_confidence": confidence,
                        },
                    }
                ],
            }
        ]

    wrong_answer = _perturb_answer(gold_answer, rng)
    warmup_signal = _scorer_signal(
        scorer,
        _scorer_value_from_confidence(
            scorer, _clamp(confidence - 0.10 + rng.random() * 0.04)
        ),
    )

    first = {
        "step": 1,
        "title": "Warmup candidate generation",
        "stage": "sampling" if family == "sample_and_vote" else "candidate_generation",
        "decision": {
            "action": "escalate" if budget > 4 else "stop",
            "reason": "Need additional budget." if budget > 4 else "Budget exhausted.",
        },
        "signals": [
            {
                "name": "confidence",
                "value": _clamp(confidence - 0.12),
                "direction": "higher_better",
                "threshold": 0.70,
            },
            warmup_signal,
        ],
        "candidates": [
            {
                "id": f"{strategy['id']}_{scorer['id']}_p1",
                "label": "Candidate 1",
                "text": f"Candidate predicts {answer}.",
                "answer": answer,
                "status": "kept" if budget > 4 else "selected",
                "selected": budget <= 4,
                "signals": {
                    scorer["id"]: _scorer_value_from_confidence(
                        scorer, _clamp(confidence - 0.04)
                    ),
                    "score": _clamp(confidence - 0.05),
                },
            },
            {
                "id": f"{strategy['id']}_{scorer['id']}_p2",
                "label": "Candidate 2",
                "text": f"Competing candidate predicts {wrong_answer}.",
                "answer": wrong_answer,
                "status": "pruned",
                "selected": False,
                "signals": {
                    scorer["id"]: _scorer_value_from_confidence(
                        scorer, _clamp(confidence - 0.20)
                    ),
                    "score": _clamp(confidence - 0.18),
                },
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
            "signals": [
                {
                    "name": "confidence",
                    "value": confidence,
                    "direction": "higher_better",
                    "threshold": 0.70,
                },
                scorer_signal,
            ],
            "candidates": [
                {
                    "id": f"{strategy['id']}_{scorer['id']}_p3",
                    "label": "Selected candidate",
                    "text": f"Selected answer {answer}.",
                    "answer": answer,
                    "status": "selected",
                    "selected": True,
                    "signals": {
                        scorer["id"]: _scorer_value_from_confidence(scorer, confidence),
                        "score": confidence,
                    },
                }
            ],
        },
    ]


def _build_tree_layout(
    run_key: str,
    confidence: float,
    rng: random.Random,
) -> Dict[str, Any]:
    root = _clamp(0.50 + (rng.random() - 0.5) * 0.06)
    return {
        "nodes": [
            {
                "id": f"{run_key}_n1",
                "label": "Root",
                "value": root,
                "depth": 0,
                "x": 0.50,
                "y": 0.14,
            },
            {
                "id": f"{run_key}_n2",
                "label": "A",
                "value": _clamp(confidence - 0.17),
                "depth": 1,
                "x": 0.24,
                "y": 0.47,
            },
            {
                "id": f"{run_key}_n3",
                "label": "B",
                "value": _clamp(confidence - 0.08),
                "depth": 1,
                "x": 0.52,
                "y": 0.47,
            },
            {
                "id": f"{run_key}_n4",
                "label": "C",
                "value": _clamp(confidence - 0.24),
                "depth": 1,
                "x": 0.80,
                "y": 0.47,
            },
            {
                "id": f"{run_key}_n5",
                "label": "B2",
                "value": confidence,
                "depth": 2,
                "x": 0.52,
                "y": 0.82,
            },
        ],
        "edges": [
            {"source": f"{run_key}_n1", "target": f"{run_key}_n2"},
            {"source": f"{run_key}_n1", "target": f"{run_key}_n3"},
            {"source": f"{run_key}_n1", "target": f"{run_key}_n4"},
            {"source": f"{run_key}_n3", "target": f"{run_key}_n5"},
        ],
        "selected_path": [f"{run_key}_n1", f"{run_key}_n3", f"{run_key}_n5"],
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


def _scorer_quality_shift(scorer_id: str) -> float:
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


def _perturb_answer(gold_answer: str, rng: random.Random) -> str:
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


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return f"{api_key[:2]}***"
    return f"{api_key[:4]}...{api_key[-4:]}"


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
