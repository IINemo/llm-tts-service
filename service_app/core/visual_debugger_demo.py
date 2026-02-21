"""Synthetic payload generation for the Visual Debugger demo.

The debugger compares multiple strategy/scorer combinations for one prompt and
budget. Payloads are deterministic per (question, answer, budget, model config)
so the UI remains stable across refreshes.
"""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional

DEFAULT_BUDGET = 8
DEBUGGER_BUDGETS = [4, 8, 12]

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

_DEMO_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "prototype_local_demo",
        "title": "Prototype: Single-Sample Matrix",
        "description": (
            "Each strategy is evaluated with every supported scorer under the "
            "same question and compute budget."
        ),
        "question": (
            "A student buys 5 notebooks at $3 each and 2 pens at $2 each. "
            "What is the total cost?"
        ),
        "gold_answer": "19",
        "shared_prompt": "Reason step-by-step and place the final answer in \\boxed{}.",
        "model_config": {
            "provider": "openrouter",
            "model_id": "openai/gpt-4o-mini",
            "api_key_masked": "sk-or...demo",
        },
        "input_source": "prototype_dataset",
    }
]


def list_demo_scenarios() -> List[Dict[str, Any]]:
    """Return lightweight scenario summaries for the debugger selector."""
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "description": item.get("description", ""),
            "available_budgets": DEBUGGER_BUDGETS,
            "default_budget": _pick_budget(DEFAULT_BUDGET, DEBUGGER_BUDGETS),
            "strategy_count": len(SUPPORTED_STRATEGIES),
            "scorer_count": len(SUPPORTED_SCORERS),
            "run_count": len(SUPPORTED_STRATEGIES) * len(SUPPORTED_SCORERS),
        }
        for item in _DEMO_SCENARIOS
    ]


def get_demo_scenario(
    scenario_id: str,
    budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Return one scenario payload resolved for a target budget."""
    scenario = _find_demo_scenario(scenario_id)
    if scenario is None:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    return build_single_sample_payload(
        question=str(scenario["question"]),
        gold_answer=str(scenario["gold_answer"]),
        shared_prompt=str(scenario.get("shared_prompt", "")),
        budget=budget,
        provider=str(scenario.get("model_config", {}).get("provider", "openrouter")),
        model_id=str(
            scenario.get("model_config", {}).get("model_id", "openai/gpt-4o-mini")
        ),
        api_key="",
        scenario_id=str(scenario["id"]),
        scenario_title=str(scenario.get("title", scenario["id"])),
        scenario_description=str(scenario.get("description", "")),
        input_source=str(scenario.get("input_source", "demo")),
        api_key_masked_override=scenario.get("model_config", {}).get("api_key_masked"),
    )


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


def _find_demo_scenario(scenario_id: str) -> Optional[Dict[str, Any]]:
    for scenario in _DEMO_SCENARIOS:
        if scenario.get("id") == scenario_id:
            return scenario
    return None


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
    final_answer = gold_answer if is_correct else _perturb_answer(gold_answer, rng)

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
            "answer": final_answer,
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

    if family == "single_pass":
        run["events"] = _build_single_pass_events(
            strategy=strategy,
            scorer=scorer,
            question=question,
            answer=final_answer,
            confidence=confidence,
        )
    elif family == "sample_and_vote":
        run["events"] = _build_sample_vote_events(
            strategy=strategy,
            scorer=scorer,
            answer=final_answer,
            confidence=confidence,
            budget=budget,
            rng=rng,
            gold_answer=gold_answer,
        )
    elif family == "reranking":
        run["events"] = _build_rerank_events(
            strategy=strategy,
            scorer=scorer,
            answer=final_answer,
            confidence=confidence,
            budget=budget,
            rng=rng,
            gold_answer=gold_answer,
        )
    else:
        tree_payload = _build_tree_events(
            strategy=strategy,
            scorer=scorer,
            answer=final_answer,
            confidence=confidence,
            budget=budget,
            rng=rng,
            gold_answer=gold_answer,
        )
        run["events"] = tree_payload["events"]
        run["tree"] = tree_payload["tree"]

    return run


def _build_single_pass_events(
    strategy: Dict[str, Any],
    scorer: Dict[str, Any],
    question: str,
    answer: str,
    confidence: float,
) -> List[Dict[str, Any]]:
    scorer_value = _scorer_value_from_confidence(scorer, confidence)
    scorer_signal = _scorer_signal(scorer, scorer_value)
    return [
        {
            "step": 1,
            "title": "Single-pass reasoning generation",
            "stage": "generation",
            "decision": {
                "action": "stop",
                "reason": (
                    "Single-pass baseline emits one chain; scorer evaluates "
                    "the completed trajectory."
                ),
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
                    "text": f"{question} -> predicted answer {answer}.",
                    "answer": answer,
                    "status": "selected",
                    "selected": True,
                    "signals": {
                        scorer["id"]: scorer_value,
                        "trajectory_confidence": confidence,
                    },
                }
            ],
        }
    ]


def _build_sample_vote_events(
    strategy: Dict[str, Any],
    scorer: Dict[str, Any],
    answer: str,
    confidence: float,
    budget: int,
    rng: random.Random,
    gold_answer: str,
) -> List[Dict[str, Any]]:
    wrong_answer = _perturb_answer(gold_answer, rng)
    warmup_consensus = _clamp(0.40 + rng.random() * 0.20)
    scorer_value = _scorer_value_from_confidence(scorer, confidence)
    scorer_warmup = _scorer_signal(
        scorer,
        _scorer_value_from_confidence(
            scorer, _clamp(confidence - 0.12 + rng.random() * 0.06)
        ),
    )

    first = {
        "step": 1,
        "title": "Warmup trajectory sampling",
        "stage": "sampling",
        "decision": {
            "action": "escalate" if budget > 4 else "stop",
            "reason": (
                "Consensus below threshold after warmup."
                if budget > 4
                else "Sampling budget exhausted."
            ),
        },
        "signals": [
            {
                "name": "consensus",
                "value": warmup_consensus,
                "direction": "higher_better",
                "threshold": 0.65,
            },
            scorer_warmup,
        ],
        "candidates": [
            {
                "id": f"{strategy['id']}_{scorer['id']}_p1",
                "label": "Path 1",
                "text": f"Candidate path predicts {answer}.",
                "answer": answer,
                "status": "kept" if budget > 4 else "selected",
                "selected": budget <= 4,
                "signals": {
                    scorer["id"]: _scorer_value_from_confidence(
                        scorer, _clamp(confidence - 0.04)
                    ),
                    "path_score": _clamp(confidence - 0.05),
                },
            },
            {
                "id": f"{strategy['id']}_{scorer['id']}_p2",
                "label": "Path 2",
                "text": f"Competing path predicts {wrong_answer}.",
                "answer": wrong_answer,
                "status": "pruned",
                "selected": False,
                "signals": {
                    scorer["id"]: _scorer_value_from_confidence(
                        scorer, _clamp(confidence - 0.20)
                    ),
                    "path_score": _clamp(confidence - 0.18),
                },
            },
        ],
    }

    if budget <= 4:
        return [first]

    return [
        first,
        {
            "step": 2,
            "title": "Escalated sampling and final vote",
            "stage": "selection",
            "decision": {
                "action": "stop",
                "reason": "Consensus crossed threshold after additional samples.",
            },
            "signals": [
                {
                    "name": "consensus",
                    "value": _clamp(confidence - 0.03),
                    "direction": "higher_better",
                    "threshold": 0.65,
                },
                _scorer_signal(scorer, scorer_value),
            ],
            "candidates": [
                {
                    "id": f"{strategy['id']}_{scorer['id']}_p3",
                    "label": "Path 3",
                    "text": f"Escalated path verifies answer {answer}.",
                    "answer": answer,
                    "status": "selected",
                    "selected": True,
                    "signals": {
                        scorer["id"]: scorer_value,
                        "path_score": confidence,
                    },
                }
            ],
        },
    ]


def _build_rerank_events(
    strategy: Dict[str, Any],
    scorer: Dict[str, Any],
    answer: str,
    confidence: float,
    budget: int,
    rng: random.Random,
    gold_answer: str,
) -> List[Dict[str, Any]]:
    wrong_answer = _perturb_answer(gold_answer, rng)
    top_gap = _clamp(0.03 + rng.random() * 0.10, 0.0, 0.3)

    first = {
        "step": 1,
        "title": "Candidate generation and scoring",
        "stage": "candidate_generation",
        "decision": {
            "action": "escalate" if budget > 4 else "rerank",
            "reason": (
                "Top-2 score gap is narrow, request more candidates."
                if budget > 4
                else "Select best candidate from current pool."
            ),
        },
        "signals": [
            {
                "name": "top2_gap",
                "value": top_gap,
                "direction": "higher_better",
                "threshold": 0.08,
            },
            _scorer_signal(
                scorer,
                _scorer_value_from_confidence(
                    scorer, _clamp(confidence - 0.10 + rng.random() * 0.05)
                ),
            ),
        ],
        "candidates": [
            {
                "id": f"{strategy['id']}_{scorer['id']}_r1",
                "label": "Candidate 1",
                "text": f"High-scoring candidate predicts {answer}.",
                "answer": answer,
                "status": "kept" if budget > 4 else "selected",
                "selected": budget <= 4,
                "signals": {
                    scorer["id"]: _scorer_value_from_confidence(
                        scorer, _clamp(confidence - 0.05)
                    ),
                    "rerank_score": _clamp(confidence - 0.04),
                },
            },
            {
                "id": f"{strategy['id']}_{scorer['id']}_r2",
                "label": "Candidate 2",
                "text": f"Lower-scoring candidate predicts {wrong_answer}.",
                "answer": wrong_answer,
                "status": "pruned",
                "selected": False,
                "signals": {
                    scorer["id"]: _scorer_value_from_confidence(
                        scorer, _clamp(confidence - 0.22)
                    ),
                    "rerank_score": _clamp(confidence - 0.21),
                },
            },
        ],
    }

    if budget <= 4:
        return [first]

    return [
        first,
        {
            "step": 2,
            "title": "Escalated reranking",
            "stage": "reranking",
            "decision": {
                "action": "stop",
                "reason": "Rerank confidence stabilized after escalation.",
            },
            "signals": [
                {
                    "name": "top2_gap",
                    "value": _clamp(top_gap + 0.08),
                    "direction": "higher_better",
                    "threshold": 0.08,
                },
                _scorer_signal(
                    scorer, _scorer_value_from_confidence(scorer, confidence)
                ),
            ],
            "candidates": [
                {
                    "id": f"{strategy['id']}_{scorer['id']}_r3",
                    "label": "Candidate 3",
                    "text": f"Escalated candidate selected with answer {answer}.",
                    "answer": answer,
                    "status": "selected",
                    "selected": True,
                    "signals": {
                        scorer["id"]: _scorer_value_from_confidence(scorer, confidence),
                        "rerank_score": confidence,
                    },
                }
            ],
        },
    ]


def _build_tree_events(
    strategy: Dict[str, Any],
    scorer: Dict[str, Any],
    answer: str,
    confidence: float,
    budget: int,
    rng: random.Random,
    gold_answer: str,
) -> Dict[str, Any]:
    wrong_answer = _perturb_answer(gold_answer, rng)
    frontier = _clamp(0.58 + rng.random() * 0.12)

    events: List[Dict[str, Any]] = [
        {
            "step": 1,
            "title": "Depth-1 tree expansion",
            "stage": "tree_expand",
            "decision": {
                "action": "continue" if budget > 4 else "stop",
                "reason": (
                    "Need deeper expansion to disambiguate branches."
                    if budget > 4
                    else "Expansion budget exhausted."
                ),
            },
            "signals": [
                {
                    "name": "best_value",
                    "value": frontier,
                    "direction": "higher_better",
                    "threshold": 0.75,
                },
                _scorer_signal(
                    scorer,
                    _scorer_value_from_confidence(
                        scorer, _clamp(confidence - 0.12 + rng.random() * 0.06)
                    ),
                ),
            ],
            "candidates": [
                {
                    "id": f"{strategy['id']}_{scorer['id']}_t1",
                    "label": "Node A",
                    "text": f"Promising branch trending toward {answer}.",
                    "answer": "",
                    "status": "kept",
                    "selected": True,
                    "signals": {
                        scorer["id"]: _scorer_value_from_confidence(
                            scorer, _clamp(confidence - 0.10)
                        ),
                        "value": frontier,
                        "depth": 1,
                    },
                },
                {
                    "id": f"{strategy['id']}_{scorer['id']}_t2",
                    "label": "Node B",
                    "text": f"Weak branch drifting toward {wrong_answer}.",
                    "answer": "",
                    "status": "pruned",
                    "selected": False,
                    "signals": {
                        scorer["id"]: _scorer_value_from_confidence(
                            scorer, _clamp(confidence - 0.22)
                        ),
                        "value": _clamp(frontier - 0.18),
                        "depth": 1,
                    },
                },
            ],
        }
    ]

    if budget > 4:
        events.append(
            {
                "step": 2,
                "title": "Depth-2 selection",
                "stage": "selection",
                "decision": {
                    "action": "stop",
                    "reason": "Best branch reached the solve threshold.",
                },
                "signals": [
                    {
                        "name": "best_value",
                        "value": confidence,
                        "direction": "higher_better",
                        "threshold": 0.75,
                    },
                    _scorer_signal(
                        scorer, _scorer_value_from_confidence(scorer, confidence)
                    ),
                ],
                "candidates": [
                    {
                        "id": f"{strategy['id']}_{scorer['id']}_t3",
                        "label": "Node A2",
                        "text": f"Selected branch outputs answer {answer}.",
                        "answer": answer,
                        "status": "selected",
                        "selected": True,
                        "signals": {
                            scorer["id"]: _scorer_value_from_confidence(
                                scorer, confidence
                            ),
                            "value": confidence,
                            "depth": 2,
                        },
                    }
                ],
            }
        )

    tree = _build_tree_layout(
        strategy_id=f"{strategy['id']}_{scorer['id']}", confidence=confidence, rng=rng
    )

    return {"events": events, "tree": tree}


def _build_tree_layout(
    strategy_id: str, confidence: float, rng: random.Random
) -> Dict[str, Any]:
    root = _clamp(0.50 + (rng.random() - 0.5) * 0.06)
    return {
        "nodes": [
            {
                "id": f"{strategy_id}_n1",
                "label": "Root",
                "value": root,
                "depth": 0,
                "x": 0.50,
                "y": 0.14,
            },
            {
                "id": f"{strategy_id}_n2",
                "label": "A",
                "value": _clamp(confidence - 0.17),
                "depth": 1,
                "x": 0.24,
                "y": 0.47,
            },
            {
                "id": f"{strategy_id}_n3",
                "label": "B",
                "value": _clamp(confidence - 0.08),
                "depth": 1,
                "x": 0.52,
                "y": 0.47,
            },
            {
                "id": f"{strategy_id}_n4",
                "label": "C",
                "value": _clamp(confidence - 0.24),
                "depth": 1,
                "x": 0.80,
                "y": 0.47,
            },
            {
                "id": f"{strategy_id}_n5",
                "label": "B2",
                "value": confidence,
                "depth": 2,
                "x": 0.52,
                "y": 0.82,
            },
        ],
        "edges": [
            {"source": f"{strategy_id}_n1", "target": f"{strategy_id}_n2"},
            {"source": f"{strategy_id}_n1", "target": f"{strategy_id}_n3"},
            {"source": f"{strategy_id}_n1", "target": f"{strategy_id}_n4"},
            {"source": f"{strategy_id}_n3", "target": f"{strategy_id}_n5"},
        ],
        "selected_path": [
            f"{strategy_id}_n1",
            f"{strategy_id}_n3",
            f"{strategy_id}_n5",
        ],
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
    # Stable pseudo-difficulty in [0.10, 0.36]
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
    return by_strategy.get(strategy_id, by_family.get(family, 0.6))


def _scorer_quality_shift(scorer_id: str) -> float:
    shifts = {
        "prm": 0.05,
        "sequence_prob": 0.03,
        "perplexity": 0.01,
        "entropy": 0.0,
    }
    return shifts.get(scorer_id, 0.0)


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
        pass

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

    target = DEFAULT_BUDGET if requested is None else requested
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
