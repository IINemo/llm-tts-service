"""
Strategy Manager - Handles TTS strategy initialization and execution.

Simplified for self-consistency strategy with external APIs (OpenAI/OpenRouter).
"""

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import settings

log = logging.getLogger(__name__)


class SelfConsistencyStrategy:
    """
    Self-consistency strategy using external APIs.

    Generates multiple reasoning paths and selects the most consistent answer
    via majority voting.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        num_paths: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.client = client
        self.model = model
        self.num_paths = num_paths
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _extract_answer(self, text: str) -> str:
        """Extract answer from \\boxed{} format."""
        # Try to find \boxed{...}
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

        # Fallback: look for "answer is X" pattern
        pattern = r'(?:answer|result)\s*(?:is|=|:)\s*([^\n.,]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return "no_answer"

    def _generate_single_path(
        self,
        messages: List[Dict[str, str]],
        path_idx: int,
    ) -> Dict[str, Any]:
        """Generate a single reasoning path."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content or ""
            answer = self._extract_answer(content)
            tokens = response.usage.completion_tokens if response.usage else 0

            log.info(f"  Path {path_idx + 1}: answer={answer}, tokens={tokens}")

            return {
                "text": content,
                "answer": answer,
                "tokens": tokens,
                "path_idx": path_idx,
            }
        except Exception as e:
            log.error(f"  Path {path_idx + 1}: Error - {e}")
            return {
                "text": "",
                "answer": "error",
                "tokens": 0,
                "path_idx": path_idx,
                "error": str(e),
            }

    def generate_trajectory(
        self,
        messages: List[Dict[str, str]],
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate multiple reasoning paths and select best via majority voting.
        """
        log.info(f"Generating {self.num_paths} reasoning paths...")

        # Generate paths in parallel
        paths = []
        with ThreadPoolExecutor(max_workers=min(self.num_paths, 10)) as executor:
            futures = {
                executor.submit(self._generate_single_path, messages, i): i
                for i in range(self.num_paths)
            }
            for future in as_completed(futures):
                paths.append(future.result())

        # Sort by path index
        paths.sort(key=lambda x: x["path_idx"])

        # Extract answers and do majority voting
        answers = [p["answer"] for p in paths]
        answer_counts = Counter(answers)

        # Find most common answer
        most_common = answer_counts.most_common(1)
        if most_common:
            best_answer, count = most_common[0]
            consensus_score = count / len(answers)
        else:
            best_answer = "no_answer"
            consensus_score = 0.0

        # Find the best path (first one with the winning answer)
        best_path = next(
            (p for p in paths if p["answer"] == best_answer),
            paths[0] if paths else {"text": "", "tokens": 0}
        )

        total_tokens = sum(p["tokens"] for p in paths)

        uncertainty_score = 1.0 - consensus_score

        log.info(f"Selected answer: {best_answer} (consensus: {consensus_score:.2f}, uncertainty: {uncertainty_score:.2f})")
        log.info(f"Answer distribution: {dict(answer_counts)}")
        log.info(f"Total tokens: {total_tokens}")

        return {
            "trajectory": best_path["text"],
            "extracted_answer": best_answer,
            "completed": True,
            "metadata": {
                "strategy": "self_consistency",
                "num_paths": self.num_paths,
                "consensus_score": consensus_score,
                "uncertainty_score": uncertainty_score,
                "answer_distribution": dict(answer_counts),
                "all_answers": answers,
                "total_tokens": total_tokens,
            },
        }


class StrategyManager:
    """Manages TTS strategy instances and model loading."""

    def __init__(self):
        self._client_cache: Dict[str, OpenAI] = {}

    def _get_or_create_client(self, provider: str = "openrouter") -> OpenAI:
        """Get cached OpenAI client or create new one."""
        if provider in self._client_cache:
            return self._client_cache[provider]

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

        client = OpenAI(api_key=api_key, base_url=base_url)
        self._client_cache[provider] = client

        log.info(f"Created OpenAI client for provider: {provider}")
        return client

    def create_strategy(
        self,
        strategy_type: str,
        model_name: str,
        strategy_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a TTS strategy instance.

        Args:
            strategy_type: Type of strategy (currently only "self_consistency")
            model_name: Model name (e.g., "openai/gpt-4o-mini")
            strategy_config: Optional strategy-specific configuration

        Returns:
            Strategy instance ready for trajectory generation
        """
        strategy_config = strategy_config or {}

        if strategy_type == "self_consistency":
            return self._create_self_consistency_strategy(model_name, strategy_config)
        else:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available strategies: self_consistency"
            )

    def _create_self_consistency_strategy(
        self, model_name: str, config: Dict[str, Any]
    ) -> SelfConsistencyStrategy:
        """Create self-consistency strategy instance."""
        provider = config.get("provider", "openrouter")
        client = self._get_or_create_client(provider)

        strategy = SelfConsistencyStrategy(
            client=client,
            model=model_name,
            num_paths=config.get("num_paths", 5),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4096),
        )

        log.info(
            f"Created self-consistency strategy: "
            f"model={model_name}, num_paths={config.get('num_paths', 5)}"
        )

        return strategy

    def clear_cache(self):
        """Clear client cache."""
        self._client_cache.clear()
        log.info("Client cache cleared")


# Global strategy manager instance
strategy_manager = StrategyManager()
