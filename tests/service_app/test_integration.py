"""Integration tests that make real API calls to OpenRouter.

These tests are excluded from normal pytest runs via the `integration` marker
(addopts includes `-m "not integration"`).  Run them explicitly with:

    OPENROUTER_API_KEY=sk-... pytest tests/service_app/test_integration.py -m integration -v
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable (mirrors conftest.py).
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_api_key = os.environ.get("OPENROUTER_API_KEY", "")
_skip_reason = "OPENROUTER_API_KEY not set"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _api_key, reason=_skip_reason),
]

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
MODEL = "openai/gpt-4o-mini"
MATH_PROMPT = "What is 17 * 23? Give the final numerical answer."
MESSAGES = [{"role": "user", "content": MATH_PROMPT}]


def _chat_body(*, num_paths: int = 2, **overrides):
    body = {
        "messages": MESSAGES,
        "model": MODEL,
        "num_paths": num_paths,
        "temperature": 0.7,
        "tts_api_key": _api_key,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Fixture: real FastAPI TestClient (no mocks)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_client():
    import tempfile

    os.environ.setdefault("_SERVICE_LOG_DIR", tempfile.mkdtemp(prefix="integ_logs_"))

    from fastapi.testclient import TestClient

    from service_app.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSelfConsistencyE2E:
    """Full HTTP round-trip through /v1/chat/completions."""

    def test_self_consistency_e2e(self, real_client):
        resp = real_client.post("/v1/chat/completions", json=_chat_body())
        assert resp.status_code == 200

        data = resp.json()
        # Top-level OpenAI-compat fields
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert isinstance(data["created"], int)
        assert data["model"] == MODEL

        # Choices
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert len(choice["message"]["content"]) > 0

        # Usage
        usage = data["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

        # TTS metadata
        meta = choice["tts_metadata"]
        assert "elapsed_time" in meta
        assert "selected_answer" in meta


class TestURLStrategyRoutingE2E:
    """URL-based strategy routing with real backend."""

    def test_url_strategy_routing_e2e(self, real_client):
        resp = real_client.post(
            "/v1/self_consistency/chat/completions",
            json=_chat_body(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) >= 1


class TestStrategyManagerCreatesRealStrategy:
    """Direct StrategyManager integration (no HTTP layer)."""

    def test_strategy_manager_creates_real_strategy(self):
        from service_app.core.strategy_manager import strategy_manager

        strategy = strategy_manager.create_strategy(
            strategy_type="self_consistency",
            model_name=MODEL,
            strategy_config={
                "tts_api_key": _api_key,
                "temperature": 0.7,
                "max_tokens": 4096,
                "num_paths": 2,
            },
        )
        assert hasattr(strategy, "generate_trajectory")


class TestRealTrajectoryGeneration:
    """Call generate_trajectory() directly and verify the result dict."""

    def test_real_trajectory_generation(self):
        from service_app.core.strategy_manager import strategy_manager

        strategy = strategy_manager.create_strategy(
            strategy_type="self_consistency",
            model_name=MODEL,
            strategy_config={
                "tts_api_key": _api_key,
                "temperature": 0.7,
                "max_tokens": 4096,
                "num_paths": 2,
            },
        )
        # generate_trajectory expects a chat message list, not a plain string.
        result = strategy.generate_trajectory(MESSAGES)

        assert isinstance(result, dict)
        assert "trajectory" in result
        assert len(result["trajectory"]) > 0
        assert "extracted_answer" in result
        assert "all_traces" in result


class TestResponseContentIsMeaningful:
    """Verify the model actually answers the math question."""

    def test_response_content_is_meaningful(self, real_client):
        resp = real_client.post("/v1/chat/completions", json=_chat_body())
        assert resp.status_code == 200

        content = resp.json()["choices"][0]["message"]["content"]
        # The response should not be empty.
        assert len(content) > 0
        # 17 * 23 = 391 — the answer should appear somewhere.
        assert "391" in content


class TestInvalidAPIKeyReturnsError:
    """Supplying a bogus key should surface an upstream error."""

    def test_invalid_api_key_returns_error(self, real_client):
        body = _chat_body(tts_api_key="sk-invalid-key-for-testing")
        resp = real_client.post("/v1/chat/completions", json=body)
        assert resp.status_code >= 400
