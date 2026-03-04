"""Tests for POST /v1/chat/completions with stream=true (SSE)."""

import json
from unittest.mock import MagicMock, patch

import pytest


def parse_sse_events(text: str) -> list:
    """Parse SSE formatted text into a list of dicts."""
    events = []
    for line in text.strip().split("\n\n"):
        for part in line.strip().split("\n"):
            if part.startswith("data: "):
                events.append(json.loads(part[6:]))
    return events


class TestSSEStreaming:
    @pytest.fixture(autouse=True)
    def _setup(self, test_client, valid_chat_body, mock_strategy_result):
        self.client = test_client
        self.body = {**valid_chat_body, "stream": True}
        self.mock_result = mock_strategy_result

    def _post_stream(self, mock_result=None):
        result = mock_result or self.mock_result
        mock_strategy = MagicMock()
        mock_strategy.generate_trajectory.return_value = result

        with patch(
            "service_app.api.routes.chat.strategy_manager.create_strategy",
            return_value=mock_strategy,
        ):
            return self.client.post("/v1/chat/completions", json=self.body)

    def test_media_type_is_sse(self):
        resp = self._post_stream()
        assert "text/event-stream" in resp.headers["content-type"]

    def test_cache_control_header(self):
        resp = self._post_stream()
        assert resp.headers.get("cache-control") == "no-cache"

    def test_x_accel_buffering_header(self):
        resp = self._post_stream()
        assert resp.headers.get("x-accel-buffering") == "no"

    def test_final_event_is_complete(self):
        resp = self._post_stream()
        events = parse_sse_events(resp.text)
        complete_events = [e for e in events if e.get("type") == "complete"]
        assert len(complete_events) == 1
        data = complete_events[0]["data"]
        assert data["id"].startswith("chatcmpl-")
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_strategy_error_yields_error_event(self):
        mock_strategy = MagicMock()
        mock_strategy.generate_trajectory.side_effect = RuntimeError("boom")

        with patch(
            "service_app.api.routes.chat.strategy_manager.create_strategy",
            return_value=mock_strategy,
        ):
            resp = self.client.post("/v1/chat/completions", json=self.body)

        events = parse_sse_events(resp.text)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) >= 1
        assert "boom" in error_events[0]["message"]

    def test_sse_parse_helper_empty(self):
        assert parse_sse_events("") == []

    def test_sse_parse_helper_single_event(self):
        raw = 'data: {"type": "complete", "data": {}}\n\n'
        events = parse_sse_events(raw)
        assert len(events) == 1
        assert events[0]["type"] == "complete"
