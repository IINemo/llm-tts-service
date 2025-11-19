"""
Strategy Manager - Handles TTS strategy initialization and execution.
"""

import logging
from typing import Any, Dict, Optional

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.scorers.tree_of_thoughts.value_scorer import TotValueScorer
from llm_tts.scorers.cot_uq_scorer import CotUqScorer
from llm_tts.strategies.tree_of_thoughts.strategy import StrategyTreeOfThoughts
from llm_tts.strategies.strategy_cot_uq import StrategyCoTUQ

from .config import settings

# Optional imports for other strategies (may not be available on all branches)
try:
    from llm_tts.strategies.strategy_deepconf import StrategyDeepConf

    HAS_DEEPCONF = True
except ImportError:
    HAS_DEEPCONF = False

try:
    from llm_tts.strategies.strategy_online_best_of_n import StrategyOnlineBestOfN

    HAS_ONLINE_BEST_OF_N = True
except ImportError:
    HAS_ONLINE_BEST_OF_N = False

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
            if not HAS_DEEPCONF:
                raise ValueError(
                    "DeepConf strategy is not available on this branch. Use 'tree_of_thoughts' or 'tot' instead."
                )
            return self._create_deepconf_strategy(model_name, strategy_config)
        elif strategy_type == "online_best_of_n":
            if not HAS_ONLINE_BEST_OF_N:
                raise ValueError(
                    "Online Best-of-N strategy is not available on this branch."
                )
            return self._create_online_best_of_n_strategy(model_name, strategy_config)
        elif strategy_type == "tree_of_thoughts" or strategy_type == "tot":
            return self._create_tree_of_thoughts_strategy(model_name, strategy_config)
        elif strategy_type == "cot_uq":
            return self._create_cot_uq_strategy(model_name, strategy_config)
        else:
            available = ["tree_of_thoughts", "tot"]
            if HAS_DEEPCONF:
                available.append("deepconf")
            if HAS_ONLINE_BEST_OF_N:
                available.append("online_best_of_n")
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. Available strategies: {', '.join(available)}"
            )

    def _create_deepconf_strategy(
        self, model_name: str, config: Dict[str, Any]
    ) -> "StrategyDeepConf":
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
    ) -> "StrategyOnlineBestOfN":
        """Create Online Best-of-N strategy instance."""
        # TODO: Implement when needed
        raise NotImplementedError("Online Best-of-N strategy not yet implemented")

    def _create_tree_of_thoughts_strategy(
        self, model_name: str, config: Dict[str, Any]
    ) -> StrategyTreeOfThoughts:
        """Create Tree-of-Thoughts strategy instance."""
        model = self._get_or_create_model(
            model_name=model_name,
            provider=config.get("provider", "openrouter"),
            supports_logprobs=False,  # ToT doesn't require logprobs
        )

        # Create scorer if not provided
        scorer_config = config.get("scorer", {})
        scorer = TotValueScorer(
            model=model,
            n_evaluate_sample=scorer_config.get("n_evaluate_sample", 3),
            temperature=scorer_config.get("temperature", 0.0),
            max_tokens=scorer_config.get("max_tokens", 50),
            timeout=scorer_config.get("timeout", 120),
            value_prompt_path=scorer_config.get(
                "value_prompt_path", "config/prompts/tree-of-thought/generic_value.txt"
            ),
        )

        return StrategyTreeOfThoughts(
            model=model,
            scorer=scorer,
            mode=config.get("mode", "generic"),
            method_generate=config.get("method_generate", "propose"),
            beam_width=config.get("beam_width", 3),
            n_generate_sample=config.get("n_generate_sample", 5),
            steps=config.get("steps", 4),
            temperature=config.get("temperature", 0.7),
            max_tokens_per_step=config.get("max_tokens_per_step", 150),
            n_threads=config.get("n_threads", 4),
            scorer_timeout=scorer_config.get("timeout", 120),
            propose_prompt_path=config.get(
                "propose_prompt_path",
                "config/prompts/tree-of-thought/generic_propose.txt",
            ),
        )

    def _create_cot_uq_strategy(self, model_name: str, config: Dict[str, Any]) -> StrategyCoTUQ:
        """Create CoT-UQ strategy instance."""
        model = self._get_or_create_model(
            model_name=model_name,
            provider=config.get("provider", "openrouter"),
            supports_logprobs=True,
        )

        # Create a CotUqScorer from config if requested, otherwise pass None
        scorer_config = config.get("scorer", {})
        use_scorer = scorer_config.get("enabled", True)
        scorer = None
        if use_scorer:
            scorer = CotUqScorer(
                model=model,
                alpha=scorer_config.get("alpha", config.get("alpha", 0.5)),
                top_logprobs=scorer_config.get(
                    "top_logprobs", config.get("top_logprobs", 10)
                ),
                step_patterns=config.get("detector_step_patterns"),
                answer_patterns=config.get("detector_answer_patterns"),
                max_steps=config.get("max_steps"),
                max_empty_steps=config.get("max_empty_steps"),
                max_keywords=config.get("max_keywords", 5),
            )

        return StrategyCoTUQ(
            model=model,
            budget=config.get("budget", 6),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 512),
            top_logprobs=config.get("top_logprobs", 10),
            alpha=config.get("alpha", 0.5),
            scorer=scorer,
            detector_step_patterns=config.get("detector_step_patterns"),
            detector_answer_patterns=config.get("detector_answer_patterns"),
            max_steps=config.get("max_steps"),
            max_empty_steps=config.get("max_empty_steps"),
            max_keywords=config.get("max_keywords", 5),
        )

    def clear_cache(self):
        """Clear model cache."""
        self._model_cache.clear()
        log.info("Model cache cleared")


# Global strategy manager instance
strategy_manager = StrategyManager()
