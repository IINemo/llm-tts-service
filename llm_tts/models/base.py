"""
Base model interface for all providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class BaseModel(ABC):
    """Abstract base class for all model providers"""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs
    ) -> List[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            List of generated texts
        """
        pass

    @abstractmethod
    def supports_logprobs(self) -> bool:
        """Check if this model supports token log probabilities"""
        pass

    def generate_with_confidence(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        n: int = 1,
        **kwargs,
    ) -> List[Tuple[str, Optional[List[Dict]]]]:
        """
        Generate text with token-level confidence data.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            n: Number of completions to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            List of (generated_text, token_confidence_data) tuples, one per completion.
            token_confidence_data is None if logprobs not supported
        """
        if not self.supports_logprobs():
            # Fallback: generate without confidence
            texts = self.generate(
                prompt, max_tokens, temperature, num_return_sequences=n, **kwargs
            )
            return [(text, None) for text in texts]

        # Default implementation - override in subclasses that support logprobs
        texts = self.generate(
            prompt, max_tokens, temperature, num_return_sequences=n, **kwargs
        )
        return [(text, None) for text in texts]
