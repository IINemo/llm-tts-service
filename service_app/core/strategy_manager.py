"""
Strategy Manager - Handles TTS strategy initialization and execution.

Supports two backends:
  1. API backend  — uses BlackboxModelWithStreaming + StepCandidateGeneratorThroughAPI
                    (for any OpenAI-compatible endpoint, including the debugger).
  2. vLLM backend — uses a locally loaded vLLM model (for offline_bon/online_bon/beam_search).
"""

import logging
import threading
from typing import Any, Dict, Optional

from openai import OpenAI

from .config import settings
from .prm_scorer_factory import prm_scorer_factory

log = logging.getLogger(__name__)

_PROVIDER_BASE_URLS = {
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
}

# Models known to return logprobs via API (from scripts/probe_model_capabilities.py).
_LOGPROB_MODEL_PATTERNS = ("gpt-4o", "gpt-4.1", "qwen3.5-27b")


def _model_supports_logprobs(model_name: str) -> bool:
    """Check if a model is known to support logprobs via API."""
    name = model_name.lower()
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    return any(p in name for p in _LOGPROB_MODEL_PATTERNS)


class StrategyManager:
    """Manages TTS strategy instances and model loading."""

    def __init__(self):
        self._client_cache: Dict[str, OpenAI] = {}
        self._vllm_model = None
        self._step_generator = None
        self._confidence_scorer = None  # For entropy/perplexity/sequence_prob

    # ------------------------------------------------------------------
    # vLLM backend
    # ------------------------------------------------------------------

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
        """Get scorer for vLLM-backed strategies (needs logprobs for non-PRM)."""
        if scorer_type not in self._VALID_SCORER_TYPES:
            raise ValueError(
                f"Unknown scorer type: {scorer_type!r}. "
                f"Available types: {', '.join(sorted(self._VALID_SCORER_TYPES))}"
            )
        if scorer_type == "prm":
            return prm_scorer_factory.get_scorer()
        else:
            if self._confidence_scorer is None:
                self._init_vllm_backend()
            return self._confidence_scorer

    def _get_api_scorer(self, scorer_type: str, supports_logprobs: bool = False):
        """Get scorer for API-backed strategies.

        PRM works standalone (separate model). Other scorers (entropy etc.)
        use StepScorerConfidence — which reads validity_score computed from
        logprobs by the generator. When the model doesn't support logprobs
        scores fall back to a neutral 0.5.
        """
        if scorer_type == "prm":
            return prm_scorer_factory.get_scorer()

        from llm_tts.scorers.step_scorer_confidence import StepScorerConfidence

        if not supports_logprobs:
            log.warning(
                f"Scorer '{scorer_type}' requires logprobs but the current model "
                f"does not support them — scores will be neutral (0.5)."
            )
        return StepScorerConfidence()

    # ------------------------------------------------------------------
    # Uncertainty wrapper (mirrors run_tts_eval.py logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_with_uncertainty(base_model, scorer_type: str):
        """Wrap a BlackboxModelWithStreaming with APIWithUncertainty.

        This gives the model an ``estimator`` attribute so that strategies
        like offline_bon compute per-step uncertainty during generation
        instead of calling ``score_trajectory`` afterwards.
        """
        try:
            from lm_polygraph.estimators import (
                MaximumSequenceProbability,
                MeanTokenEntropy,
                Perplexity,
            )
            from lm_polygraph.stat_calculators import (
                EntropyCalculator,
                VLLMLogprobsCalculator,
            )
            from lm_polygraph.utils import APIWithUncertainty
        except ImportError:
            log.warning(
                "lm-polygraph uncertainty components not available, "
                "skipping APIWithUncertainty wrapper."
            )
            return base_model

        if scorer_type == "perplexity":
            stat_calculators = [VLLMLogprobsCalculator()]
            estimator = Perplexity()
        elif scorer_type == "sequence_prob":
            stat_calculators = [VLLMLogprobsCalculator()]
            estimator = MaximumSequenceProbability()
        elif scorer_type in ("entropy", "uncertainty"):
            stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
            estimator = MeanTokenEntropy()
        else:
            return base_model

        wrapped = APIWithUncertainty(
            model=base_model,
            stat_calculators=stat_calculators,
            estimator=estimator,
        )
        log.info(f"Wrapped model with APIWithUncertainty({type(estimator).__name__})")
        return wrapped

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------

    def _get_or_create_client(
        self,
        provider: str = "openrouter",
        model_base_url: str = None,
        api_key: str = None,
    ) -> OpenAI:
        """Get cached OpenAI client or create new one.

        Args:
            provider: Named provider (openrouter, openai) or ignored when
                      model_base_url is set.
            model_base_url: Custom base URL for any OpenAI-compatible endpoint.
            api_key: Per-request API key. When provided the client is NOT cached.
        """
        # Per-request key → ephemeral client, no caching
        if api_key:
            base_url = model_base_url or _PROVIDER_BASE_URLS.get(provider)
            client = OpenAI(api_key=api_key, base_url=base_url)
            log.info(f"Created ephemeral OpenAI client: base_url={base_url}")
            return client

        cache_key = model_base_url or provider
        if cache_key in self._client_cache:
            return self._client_cache[cache_key]

        if model_base_url:
            resolved_key = (
                settings.openai_api_key
                if provider == "openai"
                else settings.openrouter_api_key
            )
            base_url = model_base_url
        elif provider == "openrouter":
            resolved_key = settings.openrouter_api_key
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "openai":
            resolved_key = settings.openai_api_key
            base_url = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not resolved_key:
            raise ValueError(f"API key not set for provider: {provider}")

        client = OpenAI(api_key=resolved_key, base_url=base_url)
        self._client_cache[cache_key] = client

        log.info(f"Created OpenAI client: provider={provider}, base_url={base_url}")
        return client

    # ------------------------------------------------------------------
    # Strategy factory
    # ------------------------------------------------------------------

    def create_strategy(
        self,
        strategy_type: str,
        model_name: str,
        strategy_config: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """
        Create a TTS strategy instance.

        When ``tts_api_key`` or ``model_base_url`` is present in
        *strategy_config* the API backend is used (all strategies).
        Otherwise, self_consistency uses the simple OpenAI client and
        everything else uses the vLLM backend.
        """
        strategy_config = strategy_config or {}

        use_api_backend = bool(
            strategy_config.get("tts_api_key") or strategy_config.get("model_base_url")
        )

        if use_api_backend:
            strategy = self._create_api_strategy(
                strategy_type, model_name, strategy_config
            )
        elif strategy_type == "self_consistency":
            strategy = self._create_self_consistency_simple(model_name, strategy_config)
        elif strategy_type in ("offline_bon", "online_bon", "beam_search"):
            strategy = self._create_vllm_strategy(strategy_type, strategy_config)
        else:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available strategies: self_consistency, offline_bon, "
                f"online_bon, beam_search"
            )

        if cancel_event is not None:
            strategy.set_cancel_event(cancel_event)

        return strategy

    # ------------------------------------------------------------------
    # API-based strategies (library classes via BlackboxModelWithStreaming)
    # ------------------------------------------------------------------

    def _create_api_strategy(
        self,
        strategy_type: str,
        model_name: str,
        config: Dict[str, Any],
    ):
        """Create a strategy backed by an OpenAI-compatible HTTP API.

        Uses llm_tts library classes: BlackboxModelWithStreaming,
        StepCandidateGeneratorThroughAPI, and the matching strategy class.
        """
        from lm_polygraph.utils.generation_parameters import GenerationParameters

        from llm_tts.early_stopping import BoundaryEarlyStopping
        from llm_tts.generators.api import StepCandidateGeneratorThroughAPI
        from llm_tts.models.blackboxmodel_with_streaming import (
            BlackboxModelWithStreaming,
        )
        from llm_tts.step_boundary_detectors import ThinkingMarkerDetector

        provider = config.get("provider", "openrouter")
        model_base_url = config.get("model_base_url")
        api_key = config.get("tts_api_key") or ""
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 4096)
        budget = config.get("budget", 8)

        # Resolve API key: per-request → server-side setting
        if not api_key:
            if provider == "openai":
                api_key = settings.openai_api_key or ""
            else:
                api_key = settings.openrouter_api_key or ""
        if not api_key:
            raise ValueError(
                "API key required: set tts_api_key in request or configure "
                "server-side OPENROUTER_API_KEY / OPENAI_API_KEY."
            )

        base_url = model_base_url or _PROVIDER_BASE_URLS.get(provider)

        detector = ThinkingMarkerDetector(
            min_step_tokens=50,
            max_step_tokens=1024,
            use_sequence=True,
            use_conclusion=True,
            use_thinking=True,
            use_verification=True,
            use_reasoning=True,
        )

        generation_parameters = GenerationParameters()
        generation_parameters.temperature = temperature
        generation_parameters.max_new_tokens = max_tokens
        generation_parameters.top_p = 0.8
        generation_parameters.top_k = 20

        logprobs_supported = _model_supports_logprobs(model_name)

        base_model = BlackboxModelWithStreaming(
            openai_api_key=api_key,
            model_path=model_name,
            supports_logprobs=logprobs_supported,
            base_url=base_url,
            early_stopping=BoundaryEarlyStopping(detector=detector),
            generation_parameters=generation_parameters,
        )

        # Wrap with APIWithUncertainty when logprobs are available so that
        # uncertainty scores are computed during generation.  This mirrors
        # what run_tts_eval.py does (lines 786-820) and is required for
        # offline_bon which checks `hasattr(model, "estimator")`.
        scorer_type = config.get("scorer_type", "entropy")
        if logprobs_supported and scorer_type != "prm":
            model_for_gen = self._wrap_with_uncertainty(base_model, scorer_type)
        else:
            model_for_gen = base_model

        needs_prefill = strategy_type in (
            "online_bon",
            "beam_search",
            "adaptive",
        )
        step_generator = StepCandidateGeneratorThroughAPI(
            model=model_for_gen,
            thinking_mode=False,
            detector=detector,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            top_k=20,
            max_context_budget=8192,
            prefill_mode=needs_prefill,
            supports_logprobs=logprobs_supported,
            max_concurrent_requests=128,
        )

        # Resolve scorer for API-backed strategies
        scorer = self._get_api_scorer(scorer_type, logprobs_supported)

        if strategy_type == "self_consistency":
            strategy = self._build_self_consistency(
                step_generator,
                config,
                budget,
            )
        elif strategy_type == "offline_bon":
            strategy = self._build_offline_bon(
                step_generator,
                config,
                budget,
                scorer=scorer,
            )
        elif strategy_type == "online_bon":
            strategy = self._build_online_bon(
                step_generator,
                config,
                budget,
                scorer=scorer,
            )
        elif strategy_type == "beam_search":
            strategy = self._build_beam_search(
                step_generator,
                config,
                budget,
                scorer=scorer,
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Attach base_model so callers can clean up
        strategy._api_base_model = base_model
        strategy._api_step_generator = step_generator

        log.info(
            f"Created API strategy: {strategy_type}, "
            f"model={model_name}, base_url={base_url}"
        )
        return strategy

    # ------------------------------------------------------------------
    # Strategy builders (shared between API and future uses)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_self_consistency(step_generator, config, budget):
        from llm_tts.scorers import ChainMajorityVotingScorer
        from llm_tts.strategies import StrategySelfConsistency

        chain_scorer = ChainMajorityVotingScorer()
        return StrategySelfConsistency(
            step_generator=step_generator,
            num_paths=config.get("num_paths", max(4, budget)),
            scorer=chain_scorer,
            batch_generation=True,
        )

    @staticmethod
    def _build_offline_bon(step_generator, config, budget, scorer=None):
        from llm_tts.strategies import StrategyOfflineBestOfN

        if scorer is None:
            from llm_tts.scorers import StepScorerConfidence

            scorer = StepScorerConfidence()
        return StrategyOfflineBestOfN(
            scorer=scorer,
            num_trajectories=config.get("num_trajectories", max(4, budget)),
            max_steps=config.get("max_steps", max(2, budget)),
            step_generator=step_generator,
            score_aggregation=config.get("score_aggregation", "mean"),
            batch_generation=True,
        )

    @staticmethod
    def _build_online_bon(step_generator, config, budget, scorer=None):
        from llm_tts.strategies import StrategyOnlineBestOfN

        if scorer is None:
            from llm_tts.scorers import StepScorerConfidence

            scorer = StepScorerConfidence()
        return StrategyOnlineBestOfN(
            step_generator=step_generator,
            scorer=scorer,
            candidates_per_step=config.get("candidates_per_step", max(2, budget)),
            max_steps=config.get("max_steps", max(2, budget)),
            batch_generation=True,
        )

    @staticmethod
    def _build_beam_search(step_generator, config, budget, scorer=None):
        from llm_tts.strategies import StrategyBeamSearch

        if scorer is None:
            from llm_tts.scorers import StepScorerConfidence

            scorer = StepScorerConfidence()
        return StrategyBeamSearch(
            step_generator=step_generator,
            scorer=scorer,
            beam_size=config.get("beam_size", min(max(2, budget // 2), 6)),
            candidates_per_beam=config.get("candidates_per_step", 2),
            max_steps=config.get("max_steps", max(2, budget)),
            aggregation=config.get("score_aggregation", "mean"),
            batch_generation=True,
        )

    # ------------------------------------------------------------------
    # Simple self-consistency (no library deps, plain OpenAI client)
    # ------------------------------------------------------------------

    def _create_self_consistency_simple(
        self,
        model_name: str,
        config: Dict[str, Any],
    ):
        """Fallback: use library StrategySelfConsistency when llm_tts is
        available, otherwise raise."""
        try:
            return self._create_api_strategy(
                "self_consistency",
                model_name,
                config,
            )
        except ImportError as exc:
            raise ValueError(
                "llm_tts library is required for self-consistency but could not be imported. "
                "Install with: pip install -e '.[service]'"
            ) from exc

    # ------------------------------------------------------------------
    # vLLM-backed strategies
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

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
