"""Tests for PRM scorer factory (service_app/core/prm_scorer_factory.py)."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from service_app.core.prm_scorer_factory import PRMScorerFactory


class TestPRMScorerFactory:
    def test_no_model_path_raises(self):
        factory = PRMScorerFactory()
        with patch("service_app.core.prm_scorer_factory.settings") as mock_settings:
            mock_settings.prm_model_path = None
            with pytest.raises(ValueError, match="PRM model path not configured"):
                factory.get_scorer()

    def test_cached_instance_on_second_call(self):
        factory = PRMScorerFactory()
        mock_scorer = MagicMock()
        with patch("service_app.core.prm_scorer_factory.settings") as mock_settings, \
             patch("service_app.core.prm_scorer_factory.StepScorerPRM", return_value=mock_scorer):
            mock_settings.prm_model_path = "/path/to/model"
            mock_settings.prm_use_vllm = False
            mock_settings.vllm_model_path = None
            mock_settings.prm_device = "cpu"
            mock_settings.prm_batch_size = 8
            mock_settings.prm_torch_dtype = "bfloat16"
            mock_settings.prm_gpu_memory_utilization = 0.3

            scorer1 = factory.get_scorer()
            scorer2 = factory.get_scorer()
            assert scorer1 is scorer2

    def test_cleanup_clears_and_calls_cleanup(self):
        factory = PRMScorerFactory()
        mock_scorer = MagicMock()
        factory._scorer = mock_scorer

        factory.cleanup()
        mock_scorer.cleanup.assert_called_once()
        assert factory._scorer is None

    def test_cleanup_when_none_no_error(self):
        factory = PRMScorerFactory()
        factory.cleanup()  # Should not raise

    def test_gpu_memory_warning(self, caplog):
        factory = PRMScorerFactory()
        mock_scorer = MagicMock()

        with patch("service_app.core.prm_scorer_factory.settings") as mock_settings, \
             patch("service_app.core.prm_scorer_factory.StepScorerPRM", return_value=mock_scorer):
            mock_settings.prm_model_path = "/path/to/model"
            mock_settings.prm_use_vllm = True
            mock_settings.vllm_model_path = "/path/to/vllm"
            mock_settings.prm_device = "cuda:0"
            mock_settings.vllm_gpu_memory_utilization = 0.9
            mock_settings.prm_gpu_memory_utilization = 0.3
            mock_settings.prm_batch_size = 8
            mock_settings.prm_torch_dtype = "bfloat16"

            with caplog.at_level(logging.WARNING):
                factory.get_scorer()

            assert any("OOM" in rec.message for rec in caplog.records)


try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")
class TestPRMScorerIntegration:
    """Integration test — requires a real GPU and a PRM model path in settings."""

    def test_real_prm_init(self):
        from service_app.core.config import settings

        if not settings.prm_model_path:
            pytest.skip("PRM_MODEL_PATH not configured")

        factory = PRMScorerFactory()
        try:
            scorer = factory.get_scorer()
            assert scorer is not None
        finally:
            factory.cleanup()
