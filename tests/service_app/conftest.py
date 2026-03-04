"""Shared fixtures for service_app tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the repo root is on sys.path so `service_app` is importable even
# when the package is not installed (only llm_tts is in setuptools.packages).
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Redirect service logs to a temp directory BEFORE importing the app.
_tmpdir = tempfile.mkdtemp(prefix="svc_test_logs_")
os.environ["_SERVICE_LOG_DIR"] = _tmpdir


from fastapi.testclient import TestClient  # noqa: E402

from service_app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def test_client():
    """FastAPI TestClient with logging pointed at a temp directory."""
    return TestClient(app)


@pytest.fixture()
def mock_strategy_result():
    """Realistic strategy result dict for mocking."""
    return {
        "trajectory": "Step 1: 2+2=4\nStep 2: The answer is 4.\n\\boxed{4}",
        "extracted_answer": "4",
        "completed": True,
        "completion_reason": "finished",
        "steps": ["Step 1: 2+2=4", "Step 2: The answer is 4."],
        "validity_scores": [0.85, 0.92],
        "aggregated_score": 0.88,
        "all_trajectories": [
            "Step 1: 2+2=4\nStep 2: The answer is 4.\n\\boxed{4}",
            "Step 1: 2+2 equals 4.\n\\boxed{4}",
        ],
        "all_scores": [0.88, 0.75],
        "best_idx": 0,
        "token_stats": {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens_this_sample": 150,
        },
        "metadata": {},
    }


@pytest.fixture()
def mock_strategy(mock_strategy_result):
    """Mock strategy with generate_trajectory and generate_trajectories_batch."""
    strategy = MagicMock()
    strategy.generate_trajectory.return_value = mock_strategy_result
    strategy.generate_trajectories_batch.return_value = [mock_strategy_result]
    return strategy


@pytest.fixture()
def mock_create_strategy(mock_strategy):
    """Patch strategy_manager.create_strategy to return mock_strategy."""
    with patch(
        "service_app.api.routes.chat.strategy_manager.create_strategy",
        return_value=mock_strategy,
    ) as mocked:
        yield mocked


@pytest.fixture()
def valid_chat_body():
    """Minimal valid ChatCompletionRequest body."""
    return {
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "model": "openai/gpt-4o-mini",
    }


@pytest.fixture()
def sample_messages():
    """Sample messages list."""
    return [{"role": "user", "content": "What is 2+2?"}]
