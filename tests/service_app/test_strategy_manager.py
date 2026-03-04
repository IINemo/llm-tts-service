"""Tests for StrategyManager (service_app/core/strategy_manager.py)."""

from unittest.mock import MagicMock, patch

import pytest

from service_app.core.strategy_manager import StrategyManager


class TestCreateStrategy:
    def setup_method(self):
        self.sm = StrategyManager()

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy type"):
            self.sm.create_strategy("nonexistent", "model-x")

    def test_api_key_selects_api_backend(self):
        with patch.object(self.sm, "_create_api_strategy") as mock_api:
            mock_api.return_value = MagicMock()
            self.sm.create_strategy(
                "self_consistency", "model-x",
                {"tts_api_key": "sk-test"},
            )
            mock_api.assert_called_once()

    def test_model_base_url_selects_api_backend(self):
        with patch.object(self.sm, "_create_api_strategy") as mock_api:
            mock_api.return_value = MagicMock()
            self.sm.create_strategy(
                "beam_search", "model-x",
                {"model_base_url": "http://localhost:8000/v1"},
            )
            mock_api.assert_called_once()

    def test_vllm_strategies_select_vllm_backend(self):
        with patch.object(self.sm, "_create_vllm_strategy") as mock_vllm:
            mock_vllm.return_value = MagicMock()
            for st in ("offline_bon", "online_bon", "beam_search"):
                self.sm.create_strategy(st, "model-x")
                mock_vllm.assert_called()
            assert mock_vllm.call_count == 3


class TestGetOrCreateClient:
    def setup_method(self):
        self.sm = StrategyManager()

    def test_creates_and_caches(self):
        with patch("service_app.core.strategy_manager.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-router"
            client1 = self.sm._get_or_create_client(provider="openrouter")
            client2 = self.sm._get_or_create_client(provider="openrouter")
            assert client1 is client2

    def test_per_request_key_ephemeral(self):
        client1 = self.sm._get_or_create_client(api_key="sk-a")
        client2 = self.sm._get_or_create_client(api_key="sk-b")
        assert client1 is not client2
        # Ephemeral clients should NOT be cached
        assert len(self.sm._client_cache) == 0

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            self.sm._get_or_create_client(provider="unknown")

    def test_missing_key_raises(self):
        with patch("service_app.core.strategy_manager.settings") as mock_settings:
            mock_settings.openrouter_api_key = None
            with pytest.raises(ValueError, match="API key not set"):
                self.sm._get_or_create_client(provider="openrouter")


class TestGetScorer:
    def setup_method(self):
        self.sm = StrategyManager()

    def test_unknown_scorer_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer type"):
            self.sm._get_scorer("invalid_scorer")

    def test_prm_calls_factory(self):
        with patch(
            "service_app.core.strategy_manager.prm_scorer_factory"
        ) as mock_factory:
            mock_factory.get_scorer.return_value = MagicMock()
            scorer = self.sm._get_scorer("prm")
            mock_factory.get_scorer.assert_called_once()

    def test_confidence_scorer_inits_vllm(self):
        mock_scorer = MagicMock()
        self.sm._confidence_scorer = mock_scorer
        result = self.sm._get_scorer("entropy")
        assert result is mock_scorer


class TestClearCache:
    def test_clears_all_state(self):
        sm = StrategyManager()
        sm._client_cache["key"] = MagicMock()
        sm._vllm_model = MagicMock()
        sm._step_generator = MagicMock()
        sm._confidence_scorer = MagicMock()

        with patch(
            "service_app.core.strategy_manager.prm_scorer_factory"
        ) as mock_factory:
            sm.clear_cache()
            mock_factory.cleanup.assert_called_once()

        assert len(sm._client_cache) == 0
        assert sm._vllm_model is None
        assert sm._step_generator is None
        assert sm._confidence_scorer is None
