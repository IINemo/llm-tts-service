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
from .prm_scorer_factory import prm_scorer_factory

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
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

        # Fallback: look for "answer is X" pattern
        pattern = r"(?:answer|result)\s*(?:is|=|:)\s*([^\n.,]+)"
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
            paths[0] if paths else {"text": "", "tokens": 0},
        )

        total_tokens = sum(p["tokens"] for p in paths)

        uncertainty_score = 1.0 - consensus_score

        log.info(
            f"Selected answer: {best_answer} (consensus: {consensus_score:.2f}, uncertainty: {uncertainty_score:.2f})"
        )
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
        self._vllm_model = None
        self._step_generator = None
        self._confidence_scorer = None  # For entropy/perplexity/sequence_prob

    def _init_vllm_backend(self):
        """Load vLLM model, wrap with uncertainty, create step generator.
        Called lazily on first vLLM request, then cached."""
        from lm_polygraph.estimators import MeanTokenEntropy
        from lm_polygraph.stat_calculators import (
            EntropyCalculator,
            VLLMLogprobsCalculator,
        )
        from lm_polygraph.utils import VLLMWithUncertainty
        from vllm import LLM

        from llm_tts.generators.vllm import VLLMStepGenerator
        from llm_tts.scorers.step_scorer_confidence import StepScorerConfidence
        from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector

        log.info(f"Loading vLLM model: {settings.vllm_model_path}")

        llm = LLM(
            model=settings.vllm_model_path,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            tensor_parallel_size=settings.vllm_tensor_parallel_size,
            max_model_len=settings.vllm_max_model_len,
            trust_remote_code=True,
            seed=settings.vllm_seed,
        )

        stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
        estimator = MeanTokenEntropy()
        self._vllm_model = VLLMWithUncertainty(
            llm=llm, stat_calculators=stat_calculators, estimator=estimator
        )

        detector = ThinkingMarkerDetector(
            min_step_tokens=10,
            max_step_tokens=2048,
            use_sequence=True,
            use_conclusion=True,
            use_thinking=True,
            use_verification=True,
            use_reasoning=True,
        )

        self._step_generator = VLLMStepGenerator(
            model=self._vllm_model,
            thinking_mode=settings.default_thinking_mode,
            detector=detector,
            max_new_tokens=settings.default_max_tokens,
            temperature=settings.default_temperature,
            max_context_budget=settings.vllm_max_model_len,
            disable_thinking_mode=None if settings.default_thinking_mode else True,
        )

        self._confidence_scorer = StepScorerConfidence()

        log.info("vLLM backend initialized successfully")

    _VALID_SCORER_TYPES = {"entropy", "perplexity", "sequence_prob", "prm"}

    def _get_scorer(self, scorer_type: str):
        """
        Get scorer instance based on scorer type.

        Args:
            scorer_type: One of 'entropy', 'perplexity', 'sequence_prob', 'prm'

        Returns:
            Scorer instance (StepScorerConfidence or StepScorerPRM)

        Raises:
            ValueError: If scorer_type is not recognised
        """
        if scorer_type not in self._VALID_SCORER_TYPES:
            raise ValueError(
                f"Unknown scorer type: {scorer_type!r}. "
                f"Available types: {', '.join(sorted(self._VALID_SCORER_TYPES))}"
            )
        if scorer_type == "prm":
            return prm_scorer_factory.get_scorer()
        else:
            # For entropy, perplexity, sequence_prob - use confidence scorer
            if self._confidence_scorer is None:
                self._init_vllm_backend()
            return self._confidence_scorer

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
            strategy_type: Type of strategy
            model_name: Model name
            strategy_config: Optional strategy-specific configuration

        Returns:
            Strategy instance ready for trajectory generation
        """
        strategy_config = strategy_config or {}

        if strategy_type == "self_consistency":
            return self._create_self_consistency_strategy(model_name, strategy_config)
        elif strategy_type in ("offline_bon", "online_bon", "beam_search"):
            return self._create_vllm_strategy(strategy_type, strategy_config)
        else:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available strategies: self_consistency, offline_bon, online_bon, beam_search"
            )

    def _create_vllm_strategy(self, strategy_type: str, config: Dict[str, Any]):
        """Create a vLLM-backed TTS strategy instance."""
        if self._step_generator is None:
            self._init_vllm_backend()

        scorer_type = config.get("scorer_type", "entropy")
        scorer = self._get_scorer(scorer_type)

        if strategy_type == "offline_bon":
            from llm_tts.strategies.strategy_offline_best_of_n import (
                StrategyOfflineBestOfN,
            )

            strategy = StrategyOfflineBestOfN(
                scorer=scorer,
                num_trajectories=config.get("num_trajectories", 8),
                max_steps=config.get("max_steps", 100),
                step_generator=self._step_generator,
                score_aggregation=config.get("score_aggregation", "min"),
                batch_generation=True,
            )
        elif strategy_type == "online_bon":
            from llm_tts.strategies.strategy_online_best_of_n import (
                StrategyOnlineBestOfN,
            )

            strategy = StrategyOnlineBestOfN(
                scorer=scorer,
                candidates_per_step=config.get("candidates_per_step", 4),
                max_steps=config.get("max_steps", 100),
                step_generator=self._step_generator,
                batch_generation=True,
            )
        elif strategy_type == "beam_search":
            from llm_tts.strategies.strategy_beam_search import StrategyBeamSearch

            strategy = StrategyBeamSearch(
                step_generator=self._step_generator,
                scorer=scorer,
                beam_size=config.get("beam_size", 4),
                candidates_per_beam=config.get("candidates_per_step", 4),
                max_steps=config.get("max_steps", 100),
                aggregation=config.get("score_aggregation", "mean"),
                scoring_window=config.get("window_size", None),
            )

        log.info(f"Created vLLM strategy: {strategy_type} with scorer: {scorer_type}")
        return strategy

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
        """Clear client cache and vLLM resources."""
        self._client_cache.clear()
        self._vllm_model = None
        self._step_generator = None
        self._confidence_scorer = None
        prm_scorer_factory.cleanup()
        log.info("Client cache cleared")


# Global strategy manager instance
strategy_manager = StrategyManager()
