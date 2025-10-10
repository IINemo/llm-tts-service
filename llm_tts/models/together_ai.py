"""
Together AI model adapter.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

try:
    from together import Together

    TOGETHER_SDK_AVAILABLE = True
except ImportError:
    TOGETHER_SDK_AVAILABLE = False
    Together = None

from .base import BaseModel

log = logging.getLogger(__name__)


class TogetherAIModel(BaseModel):
    """
    Together AI API model adapter.
    Supports token log probabilities via Together SDK.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: Optional[str] = None,
        base_url: str = "https://api.together.xyz/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(model_name, api_key)

        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = float(retry_delay)
        self.device = "api"

        if not self.api_key:
            raise ValueError(
                "Together AI API key not provided. Either pass api_key parameter "
                "or set TOGETHER_API_KEY environment variable."
            )

        # Use Together SDK if available for logprobs support
        if TOGETHER_SDK_AVAILABLE:
            self.client = Together(api_key=self.api_key)
            self._use_sdk = True
        else:
            raise RuntimeError(
                "Together SDK not available. Logprobs support disabled."
                "Install with: pip install together"
            )

        log.info(f"Initialized Together AI model: {self.model_name}")

    def supports_logprobs(self) -> bool:
        """Together AI supports token log probabilities when using SDK"""
        return self._use_sdk

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text using Together AI API.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            num_return_sequences: Number of completions to generate
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional API parameters

        Returns:
            List of generated text completions
        """
        completion_args = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "n": num_return_sequences,
            "max_tokens": max_tokens,
        }
        if stop_sequences:
            completion_args["stop"] = stop_sequences
        completion_args.update(kwargs)

        try:
            response = self.client.chat.completions.create(**completion_args)
            return [choice.text or "" for choice in response.choices]
        except Exception as e:
            raise RuntimeError(f"Together API request failed: {e}")

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
        Generate text with token-level confidence data.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of completions to generate
            top_logprobs: Number of top logprobs to retrieve
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            List of (generated_text, token_confidence_data) tuples
        """
        if (top_logprobs is None) or (top_logprobs == 0):
            log.warning(
                "You are requesting logprobs, but top_logprobs is set to 0. "
                "This will not return any logprobs."
            )

        completion_args = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "n": num_return_sequences,
            "max_tokens": max_tokens,
            "logprobs": top_logprobs,
        }
        if stop:
            completion_args["stop"] = stop
        completion_args.update(kwargs)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**completion_args)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise RuntimeError(f"Together.ai request failed: {e}") from e

            # Validate logprobs if requested
            if completion_args.get("logprobs"):
                missing = False
                for choice in response.choices:
                    lp = getattr(choice, "logprobs", None)
                    if lp is None:
                        missing = True
                        break
                    # Together SDK returns arrays for tokens & top_logprobs
                    tokens = getattr(lp, "tokens", None)
                    top_arr = getattr(lp, "top_logprobs", None)
                    if tokens is None or top_arr is None:
                        missing = True
                        break
                if missing:
                    if attempt < self.max_retries - 1:
                        log.warning(
                            "Together.ai did not return logprobs despite request (attempt %d/%d)",
                            attempt + 1,
                            self.max_retries,
                        )
                        time.sleep(self.retry_delay)
                        continue
                    raise RuntimeError(
                        f"Together.ai did not return logprobs for model '{self.model_name}'"
                    )

            results = []
            for choice in response.choices:
                logprobs_obj = choice.logprobs
                generated_text = choice.message.content or ""
                token_confidence_data = []

                for token_string, token_logprob, pos_top_logprobs in zip(
                    logprobs_obj.tokens,
                    logprobs_obj.token_logprobs,
                    logprobs_obj.top_logprobs,
                ):
                    token_confidence_data.append(
                        {
                            "token": token_string,
                            "logprob": token_logprob,
                            "top_logprobs": [
                                {"token": token, "logprob": logprob}
                                for token, logprob in pos_top_logprobs.items()
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
