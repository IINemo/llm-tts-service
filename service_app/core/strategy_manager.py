"""
Strategy Manager - Handles TTS strategy initialization and execution.
"""

import logging
from typing import Any, Dict, Optional

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.strategies.strategy_deepconf import StrategyDeepConf
from llm_tts.strategies.strategy_online_best_of_n import StrategyOnlineBestOfN

from .config import settings

log = logging.getLogger(__name__)


class StrategyManager:
    """Manages TTS strategy instances and model loading."""

    def __init__(self):
        self._model_cache: Dict[str, Any] = {}

    def _get_or_create_model(
        self,
        model_name: str,
        provider: str = "openrouter",
        supports_logprobs: bool = True,
    ) -> BlackboxModelWithStreaming:
        """Get cached model or create new one."""
        cache_key = f"{provider}:{model_name}"

        if cache_key in self._model_cache:
            log.info(f"Using cached model: {cache_key}")
            return self._model_cache[cache_key]

        log.info(f"Creating new model: {cache_key}")

        # Get API key based on provider
        if provider == "openrouter":
            api_key = settings.openrouter_api_key
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "openai":
            api_key = settings.openai_api_key
            base_url = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not api_key:
            raise ValueError(f"API key not set for provider: {provider}")

        model = BlackboxModelWithStreaming(
            openai_api_key=api_key,
            model_path=model_name,
            supports_logprobs=supports_logprobs,
            base_url=base_url,
        )

        self._model_cache[cache_key] = model
        return model

    def create_strategy(
        self,
        strategy_type: str,
        model_name: str,
        strategy_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a TTS strategy instance.

        Args:
            strategy_type: Type of strategy (e.g., "deepconf", "online_best_of_n")
            model_name: Name of the model to use
            strategy_config: Optional strategy-specific configuration

        Returns:
            Strategy instance ready for trajectory generation
        """
        strategy_config = strategy_config or {}

        if strategy_type == "deepconf":
            return self._create_deepconf_strategy(model_name, strategy_config)
        elif strategy_type == "online_best_of_n":
            return self._create_online_best_of_n_strategy(model_name, strategy_config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _create_deepconf_strategy(
        self, model_name: str, config: Dict[str, Any]
    ) -> StrategyDeepConf:
        """Create DeepConf strategy instance."""
        model = self._get_or_create_model(
            model_name=model_name,
            provider=config.get("provider", "openrouter"),
            supports_logprobs=True,
        )

        return StrategyDeepConf(
            model=model,
            budget=config.get("budget", settings.deepconf_budget),
            window_size=config.get("window_size", settings.deepconf_window_size),
            temperature=config.get("temperature", settings.deepconf_temperature),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 4096),
            top_logprobs=config.get("top_logprobs", 20),
            filter_method=config.get("filter_method", settings.deepconf_filter_method),
            confidence_threshold=config.get("confidence_threshold", None),
        )

    def _create_online_best_of_n_strategy(
        self, model_name: str, config: Dict[str, Any]
    ) -> StrategyOnlineBestOfN:
        """Create Online Best-of-N strategy instance."""
        # TODO: Implement when needed
        raise NotImplementedError("Online Best-of-N strategy not yet implemented")

    def clear_cache(self):
        """Clear model cache."""
        self._model_cache.clear()
        log.info("Model cache cleared")


# Global strategy manager instance
strategy_manager = StrategyManager()
