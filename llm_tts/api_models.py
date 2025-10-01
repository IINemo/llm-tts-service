"""
API-based model wrappers for Together AI and other providers
"""

import os
import logging
import time
import requests
from typing import List, Dict, Any, Optional
import json

# Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)


class TogetherAIModel:
    """
    Wrapper for Together AI API models.
    Provides a consistent interface similar to local models.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: Optional[str] = None,
        base_url: str = "https://api.together.xyz/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Together AI model wrapper.

        Args:
            model_name: Model identifier on Together AI
            api_key: Together AI API key (if None, will look for TOGETHER_API_KEY env var)
            base_url: API base URL
            max_retries: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.device = "api"  # Compatibility with local model interface

        if not self.api_key:
            raise ValueError(
                "Together AI API key not provided. Either pass api_key parameter "
                "or set TOGETHER_API_KEY environment variable."
            )

        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        log.info(f"Initialized Together AI model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text using Together AI API.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of sequences to generate
            stop_sequences: List of stop sequences
            **kwargs: Additional generation parameters

        Returns:
            List of generated text completions
        """
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "n": num_return_sequences,
            "stop": stop_sequences or [],
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stream": False
        }

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload and key not in ["repetition_penalty"]:
                payload[key] = value

        # Make API request with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    completions = [choice["text"] for choice in data["choices"]]
                    log.info(f"Generated {len(completions)} completions")
                    return completions

                elif response.status_code == 429:  # Rate limit
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (2 ** attempt)
                        log.warning(f"Rate limited. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"Rate limited after {self.max_retries} retries")

                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    log.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"API request failed after {self.max_retries} retries: {e}")

        raise Exception("All retry attempts exhausted")

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        batch_size: int = 5,
        **kwargs
    ) -> List[List[str]]:
        """
        Generate text for multiple prompts in batches.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            batch_size: Number of prompts to process at once
            **kwargs: Additional generation parameters

        Returns:
            List of lists - for each prompt, list of generated completions
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            log.info(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")

            batch_results = []
            for prompt in batch_prompts:
                completions = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    num_return_sequences=1,
                    **kwargs
                )
                batch_results.append(completions)

            results.extend(batch_results)

            # Add small delay between batches to be respectful
            if i + batch_size < len(prompts):
                time.sleep(0.5)

        return results

    def tokenize(self, texts: List[str]) -> Dict[str, List[List[int]]]:
        """
        Mock tokenization for API compatibility.
        Since we're using API, we don't have direct access to tokenizer.
        """
        # Return mock tokenization for compatibility
        return {
            'input_ids': [[1] * 10] * len(texts),  # Mock token IDs
            'attention_mask': [[1] * 10] * len(texts)
        }

    def cleanup(self):
        """Clean up resources (no-op for API)"""
        pass

    def __str__(self):
        return f"TogetherAI({self.model_name})"


class APIModelAdapter:
    """
    Adapter to make API models compatible with the existing evaluation framework.
    Provides the same interface as WhiteboxModel from lm-polygraph.
    """

    def __init__(self, api_model: TogetherAIModel):
        """
        Initialize adapter with an API model.

        Args:
            api_model: TogetherAIModel instance
        """
        self.api_model = api_model
        self.device = "api"

        # Mock tokenizer for compatibility
        self.tokenizer = MockTokenizer()

        # Mock model for compatibility with existing code
        self.model = MockModelForAPI(self.api_model)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate text using the API model"""
        return self.api_model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            **kwargs
        )

    def tokenize(self, texts: List[str]) -> Dict[str, Any]:
        """Mock tokenization for API compatibility"""
        return self.api_model.tokenize(texts)

    def cleanup(self):
        """Clean up resources"""
        self.api_model.cleanup()


class MockTokenizer:
    """Mock tokenizer for API model compatibility"""

    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0

    def decode(self, tokens, skip_special_tokens=True):
        """Mock decode - not used with API models"""
        return ""

    def encode(self, text):
        """Mock encode - not used with API models"""
        return [1] * 10  # Mock token IDs


class MockModelForAPI:
    """Mock model wrapper for API compatibility"""

    def __init__(self, api_model: TogetherAIModel):
        self.api_model = api_model

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        **kwargs
    ):
        """
        Mock generate method that uses the API.
        Since we're using API, we ignore input_ids and attention_mask.
        """
        # This will be called by self-consistency strategy
        # We need to return mock outputs that can be decoded
        # The actual generation happens in the strategy layer
        mock_outputs = []
        for _ in range(num_return_sequences):
            # Create mock output sequence
            if input_ids is not None:
                input_length = len(input_ids[0]) if hasattr(input_ids[0], '__len__') else 10
                mock_seq = list(range(input_length, input_length + max_new_tokens))
            else:
                mock_seq = list(range(max_new_tokens))
            mock_outputs.append(mock_seq)

        return mock_outputs


def create_together_ai_model(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_key: Optional[str] = None
) -> APIModelAdapter:
    """
    Factory function to create a Together AI model adapter.

    Args:
        model_name: Together AI model identifier
        api_key: API key (if None, uses TOGETHER_API_KEY env var)

    Returns:
        APIModelAdapter instance compatible with existing framework
    """
    api_model = TogetherAIModel(model_name=model_name, api_key=api_key)
    return APIModelAdapter(api_model)