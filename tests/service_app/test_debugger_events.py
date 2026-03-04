"""Tests for debugger event conversion (service_app/core/debugger_events.py)."""

import logging

import pytest

from service_app.core.debugger_events import (
    StrategyProgressHandler,
    _budget_unit_for_family,
    _clamp,
    _coerce_int,
    _confidence_from_score,
    _estimate_result_confidence,
    _estimate_result_tokens,
    _event_stage_for_family,
    _extract_first_numeric,
    _extract_step_entries,
    _normalize_confidence,
    _to_float,
    convert_strategy_result_to_debugger_run,
)


# -------------------------------------------------------------------------
# convert_strategy_result_to_debugger_run
# -------------------------------------------------------------------------


class TestConvertStrategyResult:
    @pytest.fixture()
    def base_args(self):
        return dict(
            strategy={"id": "beam_search", "name": "Beam Search", "family": "tree_search"},
            scorer={"id": "entropy", "name": "Entropy", "direction": "lower_better", "summary": "token entropy"},
            strategy_result={
                "trajectory": "Step 1: Think.\nStep 2: Answer.\n\\boxed{42}",
                "extracted_answer": "42",
                "steps": ["Step 1: Think.", "Step 2: Answer."],
                "validity_scores": [0.3, 0.2],
                "aggregated_score": 0.25,
                "token_stats": {"total_tokens_this_sample": 200},
                "metadata": {},
            },
            budget=8,
            latency_ms=1500,
            model_config={"provider": "openrouter", "model_id": "openai/gpt-4o-mini"},
            generation_config={"temperature": 0.6, "max_new_tokens": 4096},
            strategy_config={"beam_size": 4},
            scorer_config={"direction": "lower_better"},
            has_gold_answer=True,
            gold_answer="42",
        )

    def test_top_level_keys(self, base_args):
        run = convert_strategy_result_to_debugger_run(**base_args)
        for key in ("budget", "budget_unit", "used_budget", "tokens_used",
                     "latency_ms", "provider", "model_id", "strategy",
                     "scorer", "final", "config", "events"):
            assert key in run

    def test_strategy_fields(self, base_args):
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert run["strategy"]["id"] == "beam_search"
        assert run["strategy"]["name"] == "Beam Search"

    def test_scorer_fields(self, base_args):
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert run["scorer"]["id"] == "entropy"
        assert run["scorer"]["direction"] == "lower_better"

    def test_scorer_none(self, base_args):
        base_args["scorer"] = None
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert run["scorer"] is None

    def test_confidence_in_range(self, base_args):
        run = convert_strategy_result_to_debugger_run(**base_args)
        conf = run["final"]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_gold_answer_correct(self, base_args):
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert run["final"]["answer"] == "42"
        assert run["final"]["is_correct"] is True

    def test_gold_answer_incorrect(self, base_args):
        base_args["gold_answer"] = "99"
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert run["final"]["is_correct"] is False

    def test_no_gold_answer(self, base_args):
        base_args["has_gold_answer"] = False
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert "is_correct" not in run["final"]

    def test_config_deep_copy(self, base_args):
        run = convert_strategy_result_to_debugger_run(**base_args)
        # Mutating the returned config should not affect the original
        run["config"]["generation"]["temperature"] = 999
        assert base_args["generation_config"]["temperature"] == 0.6

    def test_events_from_steps(self, base_args):
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert isinstance(run["events"], list)
        assert len(run["events"]) > 0

    def test_events_from_all_trajectories(self, base_args):
        base_args["strategy_result"]["all_trajectories"] = [
            "traj A step1\ntraj A step2",
            "traj B step1\ntraj B step2",
        ]
        base_args["strategy_result"]["all_scores"] = [0.9, 0.6]
        base_args["strategy_result"]["best_idx"] = 0
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert len(run["events"]) >= 1

    def test_events_from_step_candidates(self, base_args):
        base_args["strategy_result"]["step_candidates"] = [
            {
                "step": 1,
                "title": "Step 1",
                "candidates": [
                    {"text": "candidate A", "score": 0.9, "selected": True},
                    {"text": "candidate B", "score": 0.5, "selected": False},
                ],
            }
        ]
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert len(run["events"]) >= 1
        assert len(run["events"][0]["candidates"]) == 2

    def test_self_consistency_score_label(self, base_args):
        base_args["strategy"]["id"] = "self_consistency"
        base_args["scorer"] = None
        run = convert_strategy_result_to_debugger_run(**base_args)
        assert run["final"]["score_label"] == "consensus"


# -------------------------------------------------------------------------
# StrategyProgressHandler
# -------------------------------------------------------------------------


class TestStrategyProgressHandler:
    def _fire(self, message: str) -> str | None:
        result = []
        handler = StrategyProgressHandler(lambda msg: result.append(msg))
        record = logging.LogRecord(
            name="llm_tts.strategies",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        return result[0] if result else None

    def test_beam_search_step(self):
        assert self._fire("Beam Search Step 3: 4 active samples") == "Step 3"

    def test_online_bon_step(self):
        assert self._fire("Online BoN Step 7: 1 active samples") == "Step 7"

    def test_adaptive_step(self):
        assert self._fire("=== Step 12 === (1/1 active samples)") == "Step 12"

    def test_non_matching_no_callback(self):
        assert self._fire("Some random log message") is None

    def test_partial_match_no_callback(self):
        assert self._fire("Processing batch of 4 samples") is None


# -------------------------------------------------------------------------
# _estimate_result_confidence
# -------------------------------------------------------------------------


class TestEstimateResultConfidence:
    def test_from_all_scores(self):
        result = {"all_scores": [0.3, 0.8], "best_idx": 1}
        conf = _estimate_result_confidence(result, scorer=None)
        assert 0.0 <= conf <= 1.0

    def test_from_aggregated_score(self):
        result = {"aggregated_score": 0.7}
        conf = _estimate_result_confidence(result, scorer=None)
        assert conf > 0.0

    def test_from_validity_scores(self):
        result = {"validity_scores": [0.5, 0.6, 0.7]}
        conf = _estimate_result_confidence(result, scorer=None)
        assert conf > 0.0

    def test_from_metadata_consensus(self):
        result = {"metadata": {"consensus_score": 0.9}}
        conf = _estimate_result_confidence(result, scorer=None)
        assert conf > 0.0

    def test_fallback_zero(self):
        conf = _estimate_result_confidence({}, scorer=None)
        assert conf == 0.0

    def test_non_dict_returns_zero(self):
        conf = _estimate_result_confidence("not a dict", scorer=None)
        assert conf == 0.0


# -------------------------------------------------------------------------
# _estimate_result_tokens
# -------------------------------------------------------------------------


class TestEstimateResultTokens:
    def test_from_token_stats(self):
        result = {"token_stats": {"total_tokens_this_sample": 500}}
        assert _estimate_result_tokens(result) == 500

    def test_from_total_tokens(self):
        result = {"total_tokens": 300}
        assert _estimate_result_tokens(result) == 300

    def test_from_steps(self):
        result = {"steps": [
            {"text": "step one", "token_ids": [1, 2, 3]},
            {"text": "step two", "token_ids": [4, 5]},
        ]}
        assert _estimate_result_tokens(result) == 5

    def test_fallback_word_count(self):
        result = {"trajectory": "one two three four five"}
        tokens = _estimate_result_tokens(result)
        assert tokens >= 1


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------


class TestBudgetUnitForFamily:
    def test_tree_search(self):
        assert _budget_unit_for_family("tree_search") == "node_expansions"

    def test_sample_and_vote(self):
        assert _budget_unit_for_family("sample_and_vote") == "paths"

    def test_reranking(self):
        assert _budget_unit_for_family("reranking") == "candidate_rollouts"

    def test_default(self):
        assert _budget_unit_for_family("single_pass") == "steps"
        assert _budget_unit_for_family("unknown") == "steps"


class TestEventStageForFamily:
    def test_single_pass(self):
        assert _event_stage_for_family("single_pass", 0, 1) == "generation"

    def test_tree_search_last(self):
        assert _event_stage_for_family("tree_search", 2, 3) == "tree_select"

    def test_tree_search_not_last(self):
        assert _event_stage_for_family("tree_search", 0, 3) == "tree_expand"

    def test_reranking_last(self):
        assert _event_stage_for_family("reranking", 1, 2) == "selection"

    def test_sample_and_vote_not_last(self):
        assert _event_stage_for_family("sample_and_vote", 0, 3) == "sampling"


class TestHelperFunctions:
    def test_to_float_valid(self):
        assert _to_float(3.14) == 3.14
        assert _to_float("2.5") == 2.5
        assert _to_float(0) == 0.0

    def test_to_float_invalid(self):
        assert _to_float(None) is None
        assert _to_float("abc") is None

    def test_normalize_confidence_in_range(self):
        assert _normalize_confidence(0.5) == 0.5
        assert _normalize_confidence(0.0) == 0.0
        assert _normalize_confidence(1.0) == 1.0

    def test_normalize_confidence_sigmoid(self):
        # Large positive → close to 1
        assert _normalize_confidence(10.0) > 0.9
        # Large negative → close to 0
        assert _normalize_confidence(-10.0) < 0.1

    def test_normalize_confidence_none(self):
        assert _normalize_confidence(None) == 0.0

    def test_confidence_from_score_higher_better(self):
        scorer = {"id": "prm", "direction": "higher_better"}
        conf = _confidence_from_score(0.8, scorer=scorer)
        assert conf == pytest.approx(0.8)

    def test_confidence_from_score_lower_better(self):
        scorer = {"id": "entropy", "direction": "lower_better"}
        conf = _confidence_from_score(0.2, scorer=scorer)
        assert conf == pytest.approx(0.8)

    def test_confidence_from_score_none_uses_fallback(self):
        conf = _confidence_from_score(None, scorer=None, fallback=0.5)
        assert conf == pytest.approx(0.5)

    def test_clamp(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(-1.0) == 0.0
        assert _clamp(2.0) == 1.0

    def test_coerce_int(self):
        assert _coerce_int("5", default=0) == 5
        assert _coerce_int(None, default=3) == 3
        assert _coerce_int(-1, default=0, minimum=0) == 0
        assert _coerce_int(100, default=0, maximum=50) == 50

    def test_extract_first_numeric(self):
        assert _extract_first_numeric({"a": 0.5, "b": "text"}) == 0.5
        assert _extract_first_numeric({"a": "nope"}) is None
        assert _extract_first_numeric("not a dict") is None


class TestExtractStepEntries:
    def test_string_steps(self):
        entries = _extract_step_entries(["step one", "step two"])
        assert len(entries) == 2
        assert entries[0]["text"] == "step one"
        assert entries[0]["tokens"] == 0

    def test_dict_steps(self):
        entries = _extract_step_entries([
            {"text": "hello", "token_ids": [1, 2, 3]},
        ])
        assert entries[0]["text"] == "hello"
        assert entries[0]["tokens"] == 3

    def test_empty_steps_skipped(self):
        entries = _extract_step_entries(["", "  ", "valid"])
        assert len(entries) == 1
        assert entries[0]["text"] == "valid"

    def test_non_list_returns_empty(self):
        assert _extract_step_entries(None) == []
        assert _extract_step_entries("string") == []
