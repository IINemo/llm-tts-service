"""
OpenRouter model adapter with token probability support.
"""

import logging
import os
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
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs
    ) -> Tuple[str, Optional[List[Dict]]]:
        """
        Generate text and extract token-level confidence scores.

        Returns:
            Tuple of (generated_text, token_confidence_data)
            where token_confidence_data contains logprobs for each token
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=self.top_logprobs,
                **kwargs,
            )

            # Extract generated text
            generated_text = response.choices[0].message.content

            # Extract token confidence data
            token_confidence_data = []
            if (
                hasattr(response.choices[0], "logprobs")
                and response.choices[0].logprobs
            ):
                logprobs_obj = response.choices[0].logprobs

                # Check if it has 'content' attribute
                if hasattr(logprobs_obj, "content") and logprobs_obj.content:
                    for token_info in logprobs_obj.content:
                        token_data = {
                            "token": token_info.token,
                            "logprob": token_info.logprob,
                            "top_logprobs": [
                                {"token": t.token, "logprob": t.logprob}
                                for t in token_info.top_logprobs
                            ],
                        }
                        token_confidence_data.append(token_data)
                else:
                    log.warning(
                        "Logprobs object exists but has no 'content' attribute or it's empty"
                    )
            else:
                log.warning(
                    f"No logprobs in response! Model '{self.model_name}' "
                    "may not support logprobs via OpenRouter"
                )

            return generated_text, token_confidence_data

        except Exception as e:
            log.error(f"Generation with confidence failed: {e}")
            raise

    def stream_with_confidence(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        confidence_callback: Optional[callable] = None,
        **kwargs,
    ):
        """
        Stream text generation with token-level confidence scores.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            confidence_callback: Optional callback(token, logprob, top_logprobs) -> bool
                                 Return True to stop generation early
            **kwargs: Additional API parameters

        Yields:
            Dict with 'token', 'logprob', 'top_logprobs', and 'text' (accumulated)
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                logprobs=True,
                top_logprobs=self.top_logprobs,
                **kwargs,
            )

            accumulated_text = ""

            for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                # Check if we have delta content
                if hasattr(choice, "delta") and choice.delta:
                    delta = choice.delta

                    # Extract token text
                    token_text = getattr(delta, "content", None)
                    if token_text:
                        accumulated_text += token_text

                    # Extract logprobs if available
                    logprob_data = None
                    top_logprobs_data = []

                    if hasattr(choice, "logprobs") and choice.logprobs:
                        logprobs_obj = choice.logprobs

                        # Extract from content array
                        if hasattr(logprobs_obj, "content") and logprobs_obj.content:
                            for token_info in logprobs_obj.content:
                                logprob_data = token_info.logprob
                                top_logprobs_data = [
                                    {"token": t.token, "logprob": t.logprob}
                                    for t in token_info.top_logprobs
                                ]

                    # Yield current state
                    result = {
                        "token": token_text,
                        "logprob": logprob_data,
                        "top_logprobs": top_logprobs_data,
                        "text": accumulated_text,
                    }
                    yield result

                    # Check early stopping callback
                    if confidence_callback and token_text and logprob_data is not None:
                        should_stop = confidence_callback(
                            token_text, logprob_data, top_logprobs_data
                        )
                        if should_stop:
                            log.info("Early stopping triggered by confidence callback")
                            break

                # Check for finish reason
                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    log.debug(f"Stream finished: {choice.finish_reason}")
                    break

        except Exception as e:
            log.error(f"Streaming with confidence failed: {e}")
            raise
