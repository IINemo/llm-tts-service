"""
OpenRouter model adapter with token probability support.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from .base import BaseModel

log = logging.getLogger(__name__)


class OpenRouterModel(BaseModel):
    """
    OpenRouter API model adapter.
    Supports token log probabilities for compatible models (OpenAI models, some others).
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        top_logprobs: int = 10,
        max_retries: int = 10,
        retry_delay: float = 0.5,
    ):
        """
        Initialize OpenRouter model.

        Args:
            model_name: Model identifier on OpenRouter
            api_key: OpenRouter API key (if None, will look for OPENROUTER_API_KEY env var)
            base_url: API base URL
            top_logprobs: Number of top token alternatives to retrieve (max 20)
        """
        super().__init__(model_name, api_key)

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for OpenRouter support. "
                "Install it with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.top_logprobs = min(top_logprobs, 20)  # OpenRouter max is 20
        self.device = "api"
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = float(retry_delay)

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Either pass api_key parameter "
                "or set OPENROUTER_API_KEY environment variable."
            )

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        log.info(f"Initialized OpenRouter model: {self.model_name}")
        log.info(f"Requesting top-{self.top_logprobs} token probabilities")

    def supports_logprobs(self) -> bool:
        """
        Check if model likely supports logprobs.
        OpenAI models support it, most others don't.
        """
        # Models known to support logprobs via OpenRouter
        supported_prefixes = ["openai/", "gpt-", "anthropic/"]
        return any(self.model_name.startswith(prefix) for prefix in supported_prefixes)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> List[str]:
        """
        Generate text using OpenRouter API.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of completions (only 1 supported for chat)
            **kwargs: Additional API parameters

        Returns:
            List of generated text completions
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            generated_text = response.choices[0].message.content
            return [generated_text]

        except Exception as e:
            log.error(f"Generation failed: {e}")
            raise

    def generate_with_confidence(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        top_logprobs: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[str, Optional[List[Dict]]]]:
        """
        Generate text and extract token-level confidence scores.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of completions to generate
            stop: Stop sequences
            **kwargs: Additional API parameters

        Returns:
            List of (generated_text, token_confidence_data) tuples, one per completion.
            token_confidence_data contains logprobs for each token.

        Note: For backward compatibility, when called without reading the return as a list,
              it will still work for n=1 (returns list with one tuple).
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_return_sequences,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                    stop=stop,
                    **kwargs,
                )

                # Validate logprobs if requested
                if top_logprobs > 0:
                    missing = False
                    for choice in response.choices:
                        lp = getattr(choice, "logprobs", None)
                        if lp is None:
                            missing = True
                            break
                        content = getattr(lp, "content", None)
                        if not content:
                            missing = True
                            break
                    if missing:
                        log.warning(
                            "OpenRouter did not return logprobs despite request (attempt %d/%d)",
                            attempt + 1,
                            self.max_retries,
                        )
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        raise RuntimeError(
                            f"OpenRouter did not return logprobs for model '{self.model_name}'"
                        )

                results = []
                for choice in response.choices:
                    logprobs_obj = choice.logprobs
                    generated_text = choice.message.content or ""
                    token_confidence_data = []

                    if hasattr(logprobs_obj, "content") and logprobs_obj.content:
                        for token_info in logprobs_obj.content:
                            token_confidence_data.append(
                                {
                                    "token": token_info.token,
                                    "logprob": token_info.logprob,
                                    "top_logprobs": [
                                        {"token": t.token, "logprob": t.logprob}
                                        for t in token_info.top_logprobs
                                    ],
                                }
                            )

                    results.append(
                        (
                            generated_text,
                            token_confidence_data if token_confidence_data else None,
                        )
                    )

                return results

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                log.error(f"Generation with confidence failed: {e}")
                raise
