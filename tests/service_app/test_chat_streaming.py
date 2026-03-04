"""Tests for POST /v1/chat/completions with stream=true (SSE)."""

import json
import threading
import time
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

    def test_started_event_includes_request_id(self):
        """First SSE event should be 'started' with a request_id."""
        resp = self._post_stream()
        events = parse_sse_events(resp.text)
        started = [e for e in events if e.get("type") == "started"]
        assert len(started) == 1
        assert "request_id" in started[0]
        assert len(started[0]["request_id"]) > 0

    def test_cancel_streaming_yields_cancelled_event(self):
        """Cancelling a running streaming request produces a 'cancelled' SSE event."""
        from llm_tts.strategies.strategy_base import StrategyBase

        captured_cancel_events = []

        class SlowStrategy(StrategyBase):
            """Strategy that blocks in a loop, checking cancel each iteration."""

            def generate_trajectories_batch(
                self, requests, sample_indices=None, save_callback=None,
            ):
                for _ in range(40):  # up to 20 seconds
                    self._check_cancelled()
                    time.sleep(0.5)
                return [{
                    "trajectory": "done",
                    "extracted_answer": "",
                    "completed": True,
                    "steps": [],
                    "validity_scores": [],
                    "token_stats": {},
                    "metadata": {},
                }]

        slow = SlowStrategy()

        def create_side_effect(*args, **kwargs):
            ce = kwargs.get("cancel_event")
            if ce:
                slow.set_cancel_event(ce)
                captured_cancel_events.append(ce)
            return slow

        def cancel_after_delay():
            # Wait for strategy to start, then cancel
            time.sleep(1)
            for ce in captured_cancel_events:
                ce.set()

        timer = threading.Thread(target=cancel_after_delay, daemon=True)
        timer.start()

        body = {**self.body, "tts_strategy": "online_bon"}
        with patch(
            "service_app.api.routes.chat.strategy_manager.create_strategy",
            side_effect=create_side_effect,
        ):
            resp = self.client.post("/v1/chat/completions", json=body)

        events = parse_sse_events(resp.text)
        started = [e for e in events if e.get("type") == "started"]
        cancelled = [e for e in events if e.get("type") == "cancelled"]
        assert len(started) == 1
        assert "request_id" in started[0]
        assert len(cancelled) == 1

    def test_cancel_endpoint_unknown_id_returns_404(self):
        """POST /v1/chat/cancel/{bad_id} returns 404."""
        resp = self.client.post("/v1/chat/cancel/nonexistent_id")
        assert resp.status_code == 404

    def test_cancel_cleans_up_active_requests(self):
        """After stream finishes, request_id should be removed from registry."""
        from service_app.api.routes.chat import _active_requests

        captured_request_ids = []

        class SlowStrategy:
            cancel_event = None

            def set_cancel_event(self, ev):
                self.cancel_event = ev

            def generate_trajectories_batch(self, requests, **kw):
                for _ in range(40):
                    if self.cancel_event and self.cancel_event.is_set():
                        from llm_tts.strategies.strategy_base import StrategyCancelled
                        raise StrategyCancelled("cancelled")
                    time.sleep(0.5)
                return []

        slow = SlowStrategy()

        def create_side_effect(*args, **kwargs):
            ce = kwargs.get("cancel_event")
            if ce:
                slow.set_cancel_event(ce)
            return slow

        def capture_and_cancel():
            # Poll for the active request, then cancel it
            for _ in range(40):
                time.sleep(0.25)
                if _active_requests:
                    rid = list(_active_requests.keys())[0]
                    captured_request_ids.append(rid)
                    _active_requests[rid].set()
                    return

        t = threading.Thread(target=capture_and_cancel, daemon=True)
        t.start()

        body = {**self.body, "tts_strategy": "online_bon"}
        with patch(
            "service_app.api.routes.chat.strategy_manager.create_strategy",
            side_effect=create_side_effect,
        ):
            self.client.post("/v1/chat/completions", json=body)

        t.join(timeout=5)
        # After stream ends, the request should be cleaned up
        assert len(captured_request_ids) == 1
        assert captured_request_ids[0] not in _active_requests
