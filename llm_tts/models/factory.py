"""
Model factory for creating model instances from configuration.
"""

import logging
from typing import Any, Dict, Optional

from .base import BaseModel
from .openrouter import OpenRouterModel
from .together_ai import TogetherAIModel

log = logging.getLogger(__name__)


def create_model(
    provider: str, model_name: str, api_key: Optional[str] = None, **kwargs
) -> BaseModel:
    """
    Create a model instance from provider and configuration.

    Args:
        provider: Provider name ("together_ai", "openrouter", "local")
        model_name: Model identifier
        api_key: API key for the provider
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseModel instance

    Examples:
        >>> # Together AI
        >>> model = create_model("together_ai", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

        >>> # OpenRouter with logprobs support
        >>> model = create_model("openrouter", "openai/gpt-4o-mini", top_logprobs=10)
    """
    provider = provider.lower().replace("-", "_")

    if provider == "together_ai" or provider == "togetherai":
        return TogetherAIModel(model_name=model_name, api_key=api_key, **kwargs)

    elif provider == "openrouter":
        return OpenRouterModel(model_name=model_name, api_key=api_key, **kwargs)

    elif provider == "local":
        # TODO: Implement local model support
        raise NotImplementedError("Local model support not yet implemented")

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: together_ai, openrouter, local"
        )


def create_model_from_config(config: Dict[str, Any]) -> BaseModel:
    """
    Create a model from a configuration dictionary.

    Args:
        config: Configuration dictionary with keys:
            - provider: Provider name
            - model_name: Model identifier
            - api_key (optional): API key
            - Additional provider-specific parameters

    Returns:
        BaseModel instance

    Example:
        >>> config = {
        ...     "provider": "openrouter",
        ...     "model_name": "openai/gpt-4o-mini",
        ...     "api_key": "sk-...",
        ...     "top_logprobs": 10
        ... }
        >>> model = create_model_from_config(config)
    """
    if "provider" not in config:
        raise ValueError("Config must contain 'provider' key")

    if "model_name" not in config:
        raise ValueError("Config must contain 'model_name' key")

    provider = config.pop("provider")
    model_name = config.pop("model_name")
    api_key = config.pop("api_key", None)

    return create_model(
        provider=provider, model_name=model_name, api_key=api_key, **config
    )
