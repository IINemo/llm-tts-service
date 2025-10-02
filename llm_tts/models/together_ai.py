"""
Together AI model adapter.
"""

import os
import logging
import time
import requests
from typing import List, Dict, Any, Optional, Tuple

from .base import BaseModel

log = logging.getLogger(__name__)


class TogetherAIModel(BaseModel):
    """
    Together AI API model adapter.
    Note: Together AI does not expose token probabilities.
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
        self.retry_delay = retry_delay
        self.device = "api"

        if not self.api_key:
            raise ValueError(
                "Together AI API key not provided. Either pass api_key parameter "
                "or set TOGETHER_API_KEY environment variable."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        log.info(f"Initialized Together AI model: {self.model_name}")

    def supports_logprobs(self) -> bool:
        """Together AI does not expose token log probabilities"""
        return False

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
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": num_return_sequences,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Add any additional kwargs
        payload.update(kwargs)

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()

                result = response.json()

                # Extract completions
                completions = []
                if "choices" in result:
                    for choice in result["choices"]:
                        text = choice.get("text", "")
                        completions.append(text)

                return completions if completions else [""]

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    log.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    log.error(
                        f"API request failed after {self.max_retries} retries: {e}"
                    )
                    raise RuntimeError(
                        f"API request failed after {self.max_retries} retries: {e}"
                    )
