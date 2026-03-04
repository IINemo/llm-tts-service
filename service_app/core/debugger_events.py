"""Debugger event conversion — shared by the visual debugger and the REST API."""

from __future__ import annotations

import logging
import math
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Public converter
# ---------------------------------------------------------------------------


def convert_strategy_result_to_debugger_run(
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
    """Convert a raw strategy result dict into the debugger 'run' payload."""
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


# ---------------------------------------------------------------------------
# Confidence / token estimation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


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
        trace_score_key = scorer_key if scorer else "consensus"
        best_idx = next((i for i, t in enumerate(all_traces) if t.get("selected")), 0)

        trace_step_lists: List[List[str]] = []
        for trace in all_traces:
            steps = trace.get("steps") if isinstance(trace, dict) else None
            if isinstance(steps, list) and steps:
                trace_step_lists.append([str(s or "") for s in steps])
            else:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _budget_unit_for_family(family: str) -> str:
    if family == "tree_search":
        return "node_expansions"
    if family == "sample_and_vote":
        return "paths"
    if family == "reranking":
        return "candidate_rollouts"
    return "steps"


def _normalize_confidence(value: Any) -> float:
    numeric = _to_float(value)
    if numeric is None or not math.isfinite(numeric):
        return 0.0
    if 0.0 <= numeric <= 1.0:
        return float(numeric)
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


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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
