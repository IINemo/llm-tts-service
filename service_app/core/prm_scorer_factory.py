"""
PRM Scorer Factory - Creates and manages PRM scorer instances.

Provides factory for creating StepScorerPRM instances with service config.
"""

import logging
from typing import Optional

from llm_tts.scorers.step_scorer_prm import StepScorerPRM

from .config import settings

log = logging.getLogger(__name__)


class PRMScorerFactory:
    """
    Factory for creating PRM scorer instances.

    Lazily initializes and caches PRM scorer to avoid loading model until needed.
    """

    def __init__(self):
        self._scorer: Optional[StepScorerPRM] = None

    def get_scorer(self) -> StepScorerPRM:
        """
        Get or create PRM scorer instance.

        Returns:
            StepScorerPRM instance configured from settings

        Raises:
            ValueError: If PRM model path is not configured
        """
        if self._scorer is not None:
            return self._scorer

        if not settings.prm_model_path:
            raise ValueError(
                "PRM model path not configured. "
                "Set PRM_MODEL_PATH environment variable or prm_model_path in settings."
            )

        log.info(f"Initializing PRM scorer: {settings.prm_model_path}")

        self._scorer = StepScorerPRM(
            prm_model_path=settings.prm_model_path,
            device=settings.prm_device,
            batch_size=settings.prm_batch_size,
            torch_dtype=settings.prm_torch_dtype,
            use_vllm=settings.prm_use_vllm,
            gpu_memory_utilization=settings.prm_gpu_memory_utilization,
        )

        log.info("PRM scorer initialized successfully")
        return self._scorer

    def cleanup(self):
        """Clean up PRM scorer resources."""
        if self._scorer is not None:
            log.info("Cleaning up PRM scorer...")
            self._scorer.cleanup()
            self._scorer = None


# Global PRM scorer factory instance
prm_scorer_factory = PRMScorerFactory()
