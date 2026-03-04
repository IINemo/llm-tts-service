"""Tests for POST /v1/chat/completions — non-streaming (service_app/api/routes/chat.py)."""

from unittest.mock import patch

import pytest


class TestResponseStructure:
    """Response format with mock strategy."""

    def test_returns_200(self, test_client, valid_chat_body, mock_create_strategy):
        resp = test_client.post("/v1/chat/completions", json=valid_chat_body)
        assert resp.status_code == 200

    def test_has_id_prefix(self, test_client, valid_chat_body, mock_create_strategy):
        data = test_client.post("/v1/chat/completions", json=valid_chat_body).json()
        assert data["id"].startswith("chatcmpl-")

    def test_object_field(self, test_client, valid_chat_body, mock_create_strategy):
        data = test_client.post("/v1/chat/completions", json=valid_chat_body).json()
        assert data["object"] == "chat.completion"

    def test_has_created(self, test_client, valid_chat_body, mock_create_strategy):
        data = test_client.post("/v1/chat/completions", json=valid_chat_body).json()
        assert isinstance(data["created"], int)

    def test_model_echoed(self, test_client, valid_chat_body, mock_create_strategy):
        data = test_client.post("/v1/chat/completions", json=valid_chat_body).json()
        assert data["model"] == "openai/gpt-4o-mini"

    def test_choices_message(self, test_client, valid_chat_body, mock_create_strategy):
        data = test_client.post("/v1/chat/completions", json=valid_chat_body).json()
        msg = data["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) > 0

    def test_usage_consistency(self, test_client, valid_chat_body, mock_create_strategy):
        data = test_client.post("/v1/chat/completions", json=valid_chat_body).json()
        usage = data["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_tts_metadata_present(self, test_client, valid_chat_body, mock_create_strategy):
        data = test_client.post("/v1/chat/completions", json=valid_chat_body).json()
        assert data["choices"][0]["tts_metadata"] is not None


class TestURLRouting:
    """Strategy/scorer selection via URL path segments."""

    def test_url_strategy_self_consistency(
        self, test_client, valid_chat_body, mock_create_strategy
    ):
        resp = test_client.post(
            "/v1/self_consistency/chat/completions", json=valid_chat_body
        )
        assert resp.status_code == 200
        mock_create_strategy.assert_called_once()
        call_kwargs = mock_create_strategy.call_args
        assert call_kwargs.kwargs["strategy_type"] == "self_consistency"

    def test_url_strategy_and_scorer(
        self, test_client, valid_chat_body, mock_create_strategy
    ):
        resp = test_client.post(
            "/v1/beam_search/prm/chat/completions", json=valid_chat_body
        )
        assert resp.status_code == 200

    def test_url_overrides_body_strategy(
        self, test_client, valid_chat_body, mock_create_strategy
    ):
        body = {**valid_chat_body, "tts_strategy": "self_consistency"}
        resp = test_client.post(
            "/v1/beam_search/chat/completions", json=body
        )
        assert resp.status_code == 200
        # The strategy_config should reflect beam_search from URL, not self_consistency
        call_kwargs = mock_create_strategy.call_args
        config = call_kwargs.kwargs.get("strategy_config", {})
        # Provider should be vllm for beam_search
        assert config.get("provider") == "vllm"

    def test_invalid_url_strategy_400(self, test_client, valid_chat_body):
        resp = test_client.post(
            "/v1/nonexistent_strategy/chat/completions", json=valid_chat_body
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data["detail"]
        assert "nonexistent_strategy" in data["detail"]["error"]["message"]

    def test_invalid_url_scorer_400(self, test_client, valid_chat_body):
        resp = test_client.post(
            "/v1/beam_search/bad_scorer/chat/completions", json=valid_chat_body
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "bad_scorer" in data["detail"]["error"]["message"]


class TestPydanticValidation:
    """Request validation — Pydantic rejects before reaching the handler."""

    def test_missing_messages(self, test_client):
        resp = test_client.post("/v1/chat/completions", json={"model": "x"})
        assert resp.status_code == 422

    def test_temperature_too_high(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "temperature": 2.5}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_temperature_negative(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "temperature": -0.1}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_top_p_too_high(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "top_p": 1.1}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_max_tokens_zero(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "max_tokens": 0}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_num_paths_zero(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "num_paths": 0}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_num_paths_over_100(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "num_paths": 101}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tts_num_trajectories_zero(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "tts_num_trajectories": 0}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tts_num_trajectories_over_256(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "tts_num_trajectories": 257}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tts_beam_size_zero(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "tts_beam_size": 0}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tts_beam_size_over_64(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "tts_beam_size": 65}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tts_max_steps_zero(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "tts_max_steps": 0}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tts_max_steps_over_500(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "tts_max_steps": 501}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tts_window_size_zero(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "tts_window_size": 0}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_tts_window_size_over_50(self, test_client, valid_chat_body):
        body = {**valid_chat_body, "tts_window_size": 51}
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422

    def test_invalid_message_role(self, test_client):
        body = {
            "messages": [{"role": "invalid_role", "content": "hi"}],
            "model": "openai/gpt-4o-mini",
        }
        assert test_client.post("/v1/chat/completions", json=body).status_code == 422


class TestStrategyDispatch:
    """Verify correct strategy type and method calls."""

    def test_self_consistency_calls_generate_trajectory(
        self, test_client, valid_chat_body, mock_create_strategy, mock_strategy
    ):
        test_client.post("/v1/chat/completions", json=valid_chat_body)
        mock_strategy.generate_trajectory.assert_called_once()
        kwargs = mock_strategy.generate_trajectory.call_args.kwargs
        assert "input_chat" in kwargs
        assert kwargs["sample_idx"] == 0

    def test_beam_search_calls_batch(
        self, test_client, valid_chat_body, mock_create_strategy, mock_strategy
    ):
        body = {**valid_chat_body, "tts_strategy": "beam_search"}
        test_client.post("/v1/chat/completions", json=body)
        mock_strategy.generate_trajectories_batch.assert_called_once()

    def test_offline_bon_calls_batch(
        self, test_client, valid_chat_body, mock_create_strategy, mock_strategy
    ):
        body = {**valid_chat_body, "tts_strategy": "offline_bon"}
        test_client.post("/v1/chat/completions", json=body)
        mock_strategy.generate_trajectories_batch.assert_called_once()

    def test_online_bon_calls_batch(
        self, test_client, valid_chat_body, mock_create_strategy, mock_strategy
    ):
        body = {**valid_chat_body, "tts_strategy": "online_bon"}
        test_client.post("/v1/chat/completions", json=body)
        mock_strategy.generate_trajectories_batch.assert_called_once()

    def test_create_strategy_receives_correct_type(
        self, test_client, valid_chat_body, mock_create_strategy
    ):
        body = {**valid_chat_body, "tts_strategy": "beam_search"}
        test_client.post("/v1/chat/completions", json=body)
        call_kwargs = mock_create_strategy.call_args.kwargs
        assert call_kwargs["strategy_type"] == "beam_search"

    def test_create_strategy_receives_config_values(
        self, test_client, valid_chat_body, mock_create_strategy
    ):
        body = {
            **valid_chat_body,
            "temperature": 0.5,
            "max_tokens": 2048,
            "num_paths": 10,
        }
        test_client.post("/v1/chat/completions", json=body)
        config = mock_create_strategy.call_args.kwargs["strategy_config"]
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 2048
        assert config["num_paths"] == 10

    def test_strategy_exception_returns_500(
        self, test_client, valid_chat_body, mock_create_strategy, mock_strategy
    ):
        mock_strategy.generate_trajectory.side_effect = RuntimeError("GPU OOM")
        resp = test_client.post("/v1/chat/completions", json=valid_chat_body)
        assert resp.status_code == 500

    def test_empty_trajectory_returns_500(
        self, test_client, valid_chat_body, mock_create_strategy, mock_strategy
    ):
        mock_strategy.generate_trajectory.return_value = {"trajectory": ""}
        resp = test_client.post("/v1/chat/completions", json=valid_chat_body)
        assert resp.status_code == 500


class TestVerboseMode:
    """tts_verbose=true includes debugger_run in metadata."""

    def test_verbose_includes_debugger_run(
        self, test_client, valid_chat_body, mock_create_strategy
    ):
        body = {**valid_chat_body, "tts_verbose": True}
        resp = test_client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        meta = resp.json()["choices"][0]["tts_metadata"]
        assert "debugger_run" in meta
        assert "events" in meta["debugger_run"]

    def test_non_verbose_no_debugger_run(
        self, test_client, valid_chat_body, mock_create_strategy
    ):
        resp = test_client.post("/v1/chat/completions", json=valid_chat_body)
        meta = resp.json()["choices"][0]["tts_metadata"]
        assert "debugger_run" not in meta
