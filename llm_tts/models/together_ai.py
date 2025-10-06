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
    Supports token log probabilities when using Together SDK.
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

        # Use Together SDK if available for logprobs support
        if TOGETHER_SDK_AVAILABLE:
            self.client = Together(api_key=self.api_key)
            self._use_sdk = True
        else:
            self.client = None
            self._use_sdk = False
            log.warning(
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
        if self._use_sdk:
            # Use SDK
            completion_args = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "n": num_return_sequences,
                "max_tokens": max_tokens,
            }
            if stop_sequences:
                completion_args["stop"] = stop_sequences
            completion_args.update(kwargs)

            try:
                response = self.client.completions.create(**completion_args)
                return [choice.text or "" for choice in response.choices]
            except Exception as e:
                log.error(f"Together API request failed: {e}")
                raise RuntimeError(f"Together API request failed: {e}")
        else:
            # Fallback to direct requests
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": num_return_sequences,
            }

            if stop_sequences:
                payload["stop"] = stop_sequences

            payload.update(kwargs)

            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/completions",
                        headers=headers,
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

    def generate_with_confidence(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        n: int = 1,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[str, Optional[List[Dict]]]]:
        """
        Generate text with token-level confidence data.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            n: Number of completions to generate
            top_k: Number of top logprobs to retrieve
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            List of (generated_text, token_confidence_data) tuples
        """
        if not self._use_sdk:
            log.warning("Together SDK not available, cannot provide logprobs")
            texts = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                num_return_sequences=n,
                stop_sequences=stop,
                **kwargs,
            )
            return [(text, None) for text in texts]

        k = min(max(top_k or 5, 0), 20)
        completion_args = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "n": n,
            "max_tokens": max_tokens,
            "logprobs": k,
            "top_logprobs": k,
        }
        if stop:
            completion_args["stop"] = stop
        completion_args.update(kwargs)

        try:
            resp = self.client.completions.create(**completion_args)
        except Exception as e:
            raise RuntimeError(f"Together.ai request failed: {e}") from e

        results = []
        for choice in resp.choices:
            text = choice.text or ""
            token_data = []

            lp = getattr(choice, "logprobs", None)
            if lp is not None:
                tokens = getattr(lp, "tokens", []) or []
                token_logprobs = getattr(lp, "token_logprobs", []) or []
                top_logprobs = getattr(lp, "top_logprobs", []) or []

                # Convert to standard format
                for i in range(len(tokens)):
                    token_info = {
                        "token": tokens[i],
                        "logprob": (
                            token_logprobs[i] if i < len(token_logprobs) else None
                        ),
                        "top_logprobs": [],
                    }

                    if i < len(top_logprobs):
                        top_lp = top_logprobs[i]
                        if isinstance(top_lp, dict):
                            token_info["top_logprobs"] = [
                                {"token": t, "logprob": lp} for t, lp in top_lp.items()
                            ]
                        elif isinstance(top_lp, list):
                            token_info["top_logprobs"] = [
                                {
                                    "token": getattr(
                                        e,
                                        "token",
                                        e.get("token") if isinstance(e, dict) else None,
                                    ),
                                    "logprob": getattr(
                                        e,
                                        "logprob",
                                        (
                                            e.get("logprob")
                                            if isinstance(e, dict)
                                            else None
                                        ),
                                    ),
                                }
                                for e in top_lp
                            ]

                    token_data.append(token_info)

            results.append((text, token_data if token_data else None))

        return results

    def generate_texts(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate text completions and return as list of strings (without logprobs).

        This is a convenience wrapper around generate_with_confidence.
        """
        results = self.generate_with_confidence(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=n,
            top_k=0,  # Don't need logprobs
            stop=stop,
        )
        return [text for text, _ in results]
