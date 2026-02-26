#!/usr/bin/env python3
# flake8: noqa: E402
# E402: Module level import not at top of file
# This is intentional - we must set multiprocessing method before CUDA imports

# IMPORTANT: Set multiprocessing method BEFORE any CUDA imports
# This is required for vLLM which uses multiprocessing internally
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import multiprocessing

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")

import json
import logging
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import openai
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from lm_polygraph import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from omegaconf import OmegaConf


def _make_output_name(run_name: str, strategy_type: str, data_name: str) -> str:
    """Strip strategy prefix and dataset name from run_name for shorter output dirs."""
    name = run_name
    # Strip known strategy prefixes (longest first to avoid partial matches)
    for prefix in [
        "adaptive_scaling_",
        "self_consistency_",
        "offline_bon_",
        "online_bon_",
        "beam_search_",
        "uncert_cot_",
        "deepconf_",
        "baseline_",
        "sc_",
        "test_",
        "run_",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    # Strip dataset name (try exact data_name and common abbreviations)
    variants = [data_name]
    abbrevs = {
        "aime2024": ["aime24", "aime_2024"],
        "aime2025": ["aime25", "aime_2025"],
        "gaokao2023en": ["gaokao2023en", "gaokao2023"],
        "minerva_math": ["minerva"],
        "gpqa_diamond": ["gpqadiamon"],
    }
    variants.extend(abbrevs.get(data_name, []))
    for v in variants:
        if f"_{v}_" in name:
            name = name.replace(f"_{v}_", "_", 1)
            break
        elif name.endswith(f"_{v}"):
            name = name[: -len(f"_{v}")]
            break
        elif name.startswith(f"{v}_"):
            name = name[len(f"{v}_") :]
            break
    # Clean up double underscores and edge underscores
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


OmegaConf.register_new_resolver("output_name", _make_output_name)
from transformers import AutoModelForCausalLM, AutoTokenizer

# vLLM imports (optional, only if vLLM is installed)
try:
    from lm_polygraph.model_adapters import WhiteboxModelvLLM
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# lm-polygraph uncertainty wrapper (for vLLM uncertainty scoring)
try:
    from lm_polygraph.estimators import (
        MaximumSequenceProbability,
        MeanTokenEntropy,
        Perplexity,
    )
    from lm_polygraph.stat_calculators import EntropyCalculator, VLLMLogprobsCalculator
    from lm_polygraph.utils import VLLMWithUncertainty

    POLYGRAPH_UNCERTAINTY_AVAILABLE = True
except ImportError:
    POLYGRAPH_UNCERTAINTY_AVAILABLE = False
    VLLMWithUncertainty = None
from utils.results import load_results_json, parse_resume_arguments, save_results_json

from llm_tts.evaluation import (
    EvaluatorAlignScore,
    EvaluatorExactMatch,
    EvaluatorHumanEvalPlus,
    EvaluatorLLMAsAJudge,
    EvaluatorMBPPPlus,
)
from llm_tts.evaluation.grader import get_timeout_count
from llm_tts.generators import (
    StepCandidateGeneratorThroughAPI,
    StepCandidateGeneratorThroughHuggingface,
)
from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.scorers import (
    StepScorerConfidence,
    StepScorerLLMCritic,
    StepScorerPRM,
    StepScorerUncertainty,
)
from llm_tts.step_boundary_detectors import ThinkingMarkerDetector

# vLLM step generator (optional)
try:
    from llm_tts.generators.vllm import VLLMStepGenerator

    VLLM_GENERATOR_AVAILABLE = True
except ImportError:
    VLLM_GENERATOR_AVAILABLE = False
from llm_tts.strategies import (
    AdaptiveScalingBestOfN,
    PhiDecoding,
    StrategyBaseline,
    StrategyBeamSearch,
    StrategyDeepConf,
    StrategyExtendedThinking,
    StrategyOfflineBestOfN,
    StrategyOnlineBestOfN,
    StrategySelfConsistency,
    StrategyUncertaintyCoT,
)
from llm_tts.utils import get_torch_dtype
from llm_tts.utils.flops import FLOPCalculator

# Load environment variables from .env file
load_dotenv()

log = logging.getLogger(__name__)

_tflops_warned = set()


def _validate_api_keys(config):
    """Validate that required API keys are set and working before starting experiments.

    Checks all evaluators (both per-sample and batch) that need API access.
    Fails fast with a clear error message instead of failing hours later.
    """
    evaluator_names = list(config.evaluation.get("evaluators", []))
    evaluator_names += list(config.evaluation.get("batch_evaluators", []))

    if "llm_judge" not in evaluator_names:
        return

    llm_cfg = config.evaluation.get("llm_judge", {})
    provider = llm_cfg.get("provider", "openai")
    base_url = llm_cfg.get("base_url", None)
    model = llm_cfg.get("model", "unknown")

    # Determine which key is needed
    if provider == "openrouter":
        key_name = "OPENROUTER_API_KEY"
    elif provider == "deepseek":
        key_name = "DEEPSEEK_API_KEY"
    else:
        key_name = "OPENAI_API_KEY"

    api_key = os.environ.get(key_name)
    if not api_key:
        raise ValueError(
            f"LLM judge requires {key_name} but it is not set. "
            f"Set it in your .env file or environment. "
            f"(provider={provider}, model={model})"
        )

    # Ping the API with a minimal request to verify the key works
    log.info(f"Validating {key_name} for LLM judge ({provider}/{model})...")
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_completion_tokens=16,
        )
        log.info(f"API key validated successfully: {key_name}")
    except Exception as e:
        raise ValueError(
            f"LLM judge API key validation failed for {key_name}: {e}. "
            f"Check that your API key is valid and the provider is accessible. "
            f"(provider={provider}, model={model}, base_url={base_url})"
        ) from e


def _safe_tflops(stats: dict, key: str = "tflops") -> float:
    """Extract tflops value from stats dict, warning once if missing."""
    val = stats.get(key)
    if val is None:
        if key not in _tflops_warned:
            log.warning(
                f"Missing '{key}' in token_stats — compute tracking may be broken"
            )
            _tflops_warned.add(key)
        return 0.0
    return val


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.chat_template = None
    # tokenizer.padding_side = "left"  # Fix padding side for decoder-only models
    return tokenizer


def load_model(
    model_path: str,
    device_map: str,
    torch_dtype: str,
    gpu_memory_utilization: float = None,
):
    dtype = get_torch_dtype(torch_dtype)

    # Limit GPU memory if gpu_memory_utilization is specified
    if gpu_memory_utilization is not None and gpu_memory_utilization < 1.0:
        import torch

        # Set memory fraction for all visible GPUs
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(gpu_memory_utilization, i)
        log.info(f"Set GPU memory fraction to {gpu_memory_utilization}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    log.info(f"Loaded model with {torch_dtype}")
    return model


def load_prompt_template(prompt_file: str) -> str:
    """Load prompt template from file"""
    with open(prompt_file, "r") as f:
        return f.read().strip()


def build_evaluators(config):
    """
    Create evaluators from config.
    The list of evaluator names in config.evaluation.evaluators

    Args:
        config: Hydra config
    """
    evaluators = {}

    for evaluator_name in config.evaluation.evaluators:
        if evaluator_name == "llm_judge":
            llm_cfg = OmegaConf.to_container(config.evaluation.llm_judge, resolve=True)
            prompt_template = (
                load_prompt_template(llm_cfg.get("prompt_file"))
                if llm_cfg.get("prompt_file")
                else ""
            )
            if "{question}" in prompt_template:
                prompt_template = prompt_template.replace("{question}", "{q}")

            # Set API key in environment based on provider
            provider = llm_cfg.get("provider")
            if provider == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY")

            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

            # Remove config-only params not needed by evaluator
            llm_cfg.pop("prompt_file", None)
            llm_cfg.pop("provider", None)
            llm_cfg["prompt"] = prompt_template

            # Include model name in evaluator key to support multiple LLM judge models
            model_name = llm_cfg.get("model", "unknown")
            # Sanitize model name (remove slashes, colons, etc.)
            sanitized_model = model_name.replace("/", "_").replace(":", "_")
            eval_key = f"llm_judge_{sanitized_model}"
            evaluators[eval_key] = EvaluatorLLMAsAJudge(**llm_cfg)

        elif evaluator_name == "exact_match":
            # Get data_name for official extraction (from dataset or strategy config)
            data_name = config.dataset.get("data_name", None) or config.strategy.get(
                "data_name", None
            )
            if not data_name:
                raise ValueError(
                    "data_name must be set in config.dataset or config.strategy"
                )
            evaluators["exact_match"] = EvaluatorExactMatch(
                config.dataset.answer_format,
                data_name=data_name,
            )

        elif evaluator_name == "alignscore":
            align_cfg = OmegaConf.to_container(
                config.evaluation.alignscore, resolve=True
            )
            evaluators["alignscore"] = EvaluatorAlignScore(**align_cfg)

        elif evaluator_name == "mbpp_plus":
            # MBPP+ evaluator for code generation (uses EvalPlus)
            mbpp_cfg = config.evaluation.get("mbpp_plus", {})
            if mbpp_cfg:
                mbpp_cfg = OmegaConf.to_container(mbpp_cfg, resolve=True)
            else:
                mbpp_cfg = {}
            evaluators["mbpp_plus"] = EvaluatorMBPPPlus(
                mode=mbpp_cfg.get("mode", "full"),
                timeout=mbpp_cfg.get("timeout", 10),
            )

        elif evaluator_name == "human_eval_plus":
            # HumanEval+ evaluator for code generation (uses EvalPlus)
            he_cfg = config.evaluation.get("human_eval_plus", {})
            if he_cfg:
                he_cfg = OmegaConf.to_container(he_cfg, resolve=True)
            else:
                he_cfg = {}
            evaluators["human_eval_plus"] = EvaluatorHumanEvalPlus(
                mode=he_cfg.get("mode", "full"),
                timeout=he_cfg.get("timeout", 10),
            )

        else:
            log.warning(f"Unknown evaluator type '{evaluator_name}', skipping")

    return evaluators


def wandb_save_directory(directory_path):
    import wandb

    for file_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, file_name)
        if os.path.isfile(full_path):  # Make sure it's a file, not a directory
            wandb.save(full_path)


def set_random_seeds(seed):
    log.info(f"Set random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_scorer(config):
    # DeepConf and self_consistency don't use a scorer
    if config.strategy.type in ("deepconf", "self_consistency"):
        return None
    if config.scorer is None:
        return None
    if config.scorer.type == "prm":
        scorer = StepScorerPRM(
            prm_model_path=config.scorer.model_path,
            device=config.scorer.device,
            batch_size=config.scorer.batch_size,
            torch_dtype=config.system.torch_dtype,
            use_vllm=getattr(config.scorer, "use_vllm", True),
            gpu_memory_utilization=getattr(
                config.scorer, "gpu_memory_utilization", 0.9
            ),
        )
        try:
            scorer.init_flop_calculator(config.scorer.model_path)
        except Exception as e:
            log.warning(f"Could not init PRM FLOP calculator: {e}")
    elif config.scorer.type == "llm_critic":
        # LLM Critic Scorer (Tree of Thoughts paper)
        # Model will be set later in create_tts_strategy() after model is initialized
        scorer = StepScorerLLMCritic(
            method=config.scorer.method,
            n_evaluate_sample=config.scorer.n_evaluate_sample,
            temperature=config.scorer.temperature,
            max_tokens=config.scorer.max_tokens,
            timeout=config.scorer.timeout,
            value_prompt_file=config.scorer.value_prompt_file,
            vote_prompt_file=config.scorer.vote_prompt_file,
            score_aggregation=getattr(config.scorer, "score_aggregation", "min"),
            context_window=getattr(config.scorer, "context_window", 0),
        )
    elif config.scorer.type == "uncertainty":
        scorer = StepScorerUncertainty()
    elif config.scorer.type in (
        "perplexity",
        "entropy",
        "uncertainty_pd",
        "sequence_prob",
    ):
        scorer = StepScorerConfidence()
    else:
        raise ValueError(f"Scorer type {config.scorer.type} not supported")

    return scorer


def create_model(config):
    if config.model.type == "vllm":
        # vLLM backend - fast inference with PagedAttention
        # Uncertainty scoring is done locally using lm-polygraph estimators
        # (Perplexity, MeanTokenEntropy) computed from vLLM logprobs
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        # Initialize vLLM engine with seed for reproducibility
        llm = LLM(
            model=config.model.model_path,
            gpu_memory_utilization=config.model.get("gpu_memory_utilization", 0.9),
            tensor_parallel_size=config.model.get("tensor_parallel_size", 1),
            enable_prefix_caching=config.model.get("enable_prefix_caching", True),
            trust_remote_code=config.model.get("trust_remote_code", True),
            max_model_len=config.model.get(
                "max_context_budget", config.model.get("max_model_len", 32768)
            ),
            seed=config.system.seed,  # Reproducibility
        )

        # Create sampling params (will be updated by strategy)
        sampling_params = SamplingParams(
            max_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            logprobs=config.strategy.get("top_logprobs", 20),
            seed=config.system.seed,  # Reproducibility
        )

        # Wrap with lm-polygraph adapter for compatibility with strategies
        model = WhiteboxModelvLLM(
            model=llm,
            sampling_params=sampling_params,
            device=config.model.get("device", "cuda"),
        )

        # Mark as vLLM model for strategy detection
        model.is_vllm = True
        model.vllm_engine = llm

        model_gpu_util = config.model.get("gpu_memory_utilization", 0.9)
        log.info(
            f"vLLM model loaded successfully "
            f"(gpu_memory_utilization={model_gpu_util})"
        )

        # Create step generator for strategies that need it
        # DeepConf has its own generation logic
        step_generator = None
        if config.strategy.type not in ("deepconf",):
            if not VLLM_GENERATOR_AVAILABLE:
                raise ImportError(
                    "vLLM step generator not available. "
                    "Ensure llm_tts.step_candidate_generator_through_vllm is installed."
                )

            # Self-consistency, baseline, extended_thinking, and llm_critic don't need uncertainty wrapper
            scorer_type = config.scorer.type if config.scorer else "entropy"
            if (
                config.strategy.type
                in ("self_consistency", "baseline", "extended_thinking")
                or scorer_type == "llm_critic"
            ):
                vllm_model = llm
                log.info(
                    f"Strategy={config.strategy.type}, scorer={scorer_type}: "
                    f"using raw vLLM (no uncertainty wrapper)"
                )
            else:
                if not POLYGRAPH_UNCERTAINTY_AVAILABLE:
                    raise ImportError(
                        "lm-polygraph uncertainty components not available. "
                        "Ensure lm_polygraph_updates package is installed."
                    )

                # Select estimator based on scorer config
                scorer_type = config.scorer.type if config.scorer else "entropy"
                if scorer_type == "perplexity":
                    stat_calculators = [VLLMLogprobsCalculator()]
                    estimator = Perplexity()
                elif scorer_type == "sequence_prob":
                    # Sequence probability scoring (sum of log-probs, not normalized)
                    stat_calculators = [VLLMLogprobsCalculator()]
                    estimator = MaximumSequenceProbability()
                elif scorer_type == "uncertainty_pd":
                    # PD-Gap scoring using top-k logprobs matrix
                    from llm_tts.scorers.estimator_uncertainty_pd import PDGap

                    stat_calculators = [VLLMLogprobsCalculator(output_matrix=True)]
                    estimator = PDGap()
                elif scorer_type == "entropy":
                    # Entropy-based scoring
                    stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
                    estimator = MeanTokenEntropy()
                elif scorer_type == "prm":
                    # PRM scorer uses its own model for scoring
                    # Use entropy wrapper for generation (scores not used for selection)
                    stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
                    estimator = MeanTokenEntropy()
                else:
                    raise ValueError(
                        f"Unsupported scorer type for vLLM: {scorer_type}. "
                        f"Supported types: perplexity, sequence_prob, uncertainty_pd, entropy, prm"
                    )

                vllm_model = VLLMWithUncertainty(
                    llm=llm,
                    stat_calculators=stat_calculators,
                    estimator=estimator,
                )
                log.info(
                    f"Created VLLMWithUncertainty wrapper with {type(estimator).__name__}"
                )

            # Always use ThinkingMarkerDetector for step boundary detection
            # Stop tokens are derived from detector's semantic markers
            # thinking_mode controls two-phase generation (<think>...</think>)
            # Logic for disable_thinking_mode:
            #   None  = model doesn't support thinking (e.g., Qwen2.5-Math) -> thinking_mode=False
            #   False = model supports thinking, enabled (e.g., Qwen3) -> thinking_mode=True
            #   True  = model supports thinking, disabled -> thinking_mode=False
            disable_thinking_mode = config.model.get("disable_thinking_mode", None)
            thinking_mode = disable_thinking_mode is False
            log.info(
                f"Creating VLLMStepGenerator with ThinkingMarkerDetector "
                f"(thinking_mode={thinking_mode})"
            )

            if "min_step_tokens" not in config.strategy:
                log.warning("strategy.min_step_tokens not set, defaulting to 50")
            if "max_step_tokens" not in config.strategy:
                log.warning("strategy.max_step_tokens not set, defaulting to 300")
            detector = ThinkingMarkerDetector(
                min_step_tokens=config.strategy.get("min_step_tokens", 50),
                max_step_tokens=config.strategy.get("max_step_tokens", 300),
                use_sequence=config.strategy.get("use_sequence", True),
                use_conclusion=config.strategy.get("use_conclusion", True),
                use_thinking=config.strategy.get("use_thinking", True),
                use_verification=config.strategy.get("use_verification", True),
                use_reasoning=config.strategy.get("use_reasoning", False),
                use_correction=config.strategy.get("use_correction", False),
                use_structure=config.strategy.get("use_structure", False),
                custom_markers=config.strategy.get("custom_markers", None),
            )

            # Stop token IDs (e.g., [151645, 151643] for Qwen EOS)
            # Stop tokens are derived from detector's use_* flags automatically
            stop_token_ids = config.strategy.get("stop_token_ids", None)
            if stop_token_ids is not None:
                stop_token_ids = list(stop_token_ids)

            step_generator = VLLMStepGenerator(
                model=vllm_model,
                thinking_mode=thinking_mode,
                detector=detector,
                stop_token_ids=stop_token_ids,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                top_k=config.generation.get("top_k", 20),
                presence_penalty=config.generation.get("presence_penalty", 0.0),
                answer_patterns=config.strategy.get(
                    "detector_answer_patterns",
                    [],  # Empty by default - rely on EOS token IDs
                ),
                max_context_budget=config.model.get(
                    "max_context_budget", config.model.get("max_model_len", 32768)
                ),
                disable_thinking_mode=config.model.get("disable_thinking_mode", None),
            )

            log.info(f"Created vLLM step generator: {type(step_generator).__name__}")

        return model, step_generator

    elif config.model.type == "local":
        scorer_type = config.scorer.type if config.scorer else None
        if scorer_type in ["uncertainty", "uncertainty_pd", "entropy", "perplexity"]:
            log.info(
                f"Loading uncertainty model: {config.scorer.uncertainty_model_creator}"
            )

            import importlib

            # Add working directory to path for config module imports
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)

            mod = importlib.import_module(config.scorer.uncertainty_model_creator)
            model = mod.create_uncertainty_model(config)
            model.generation_parameters = GenerationParameters()
            model.generation_parameters.temperature = config.generation.temperature
            model.generation_parameters.max_new_tokens = (
                config.generation.max_new_tokens
            )
            model.generation_parameters.top_p = config.generation.top_p
            model.generation_parameters.top_k = config.generation.top_k

        else:
            log.info(f"Loading model: {config.model.model_path}")
            tokenizer = load_tokenizer(config.model.model_path)
            base_model = load_model(
                config.model.model_path,
                config.system.device,
                config.system.torch_dtype,
                gpu_memory_utilization=config.model.get("gpu_memory_utilization"),
            )
            base_model.eval()
            model = WhiteboxModel(base_model, tokenizer)

        # Always use ThinkingMarkerDetector for step boundary detection
        log.info("Using ThinkingMarkerDetector for local model")
        if "min_step_tokens" not in config.strategy:
            log.warning("strategy.min_step_tokens not set, defaulting to 50")
        if "max_step_tokens" not in config.strategy:
            log.warning("strategy.max_step_tokens not set, defaulting to 300")
        detector = ThinkingMarkerDetector(
            min_step_tokens=config.strategy.get("min_step_tokens", 50),
            max_step_tokens=config.strategy.get("max_step_tokens", 300),
            use_sequence=config.strategy.get("use_sequence", True),
            use_conclusion=config.strategy.get("use_conclusion", True),
            use_thinking=config.strategy.get("use_thinking", True),
            use_verification=config.strategy.get("use_verification", True),
            use_structure=config.strategy.get("use_structure", False),
            use_reasoning=config.strategy.get("use_reasoning", False),
            use_sentence_start=config.strategy.get("use_sentence_start", False),
            use_correction=config.strategy.get("use_correction", False),
            custom_markers=config.strategy.get("custom_markers"),
        )
        # Set answer patterns if provided
        if config.strategy.get("detector_answer_patterns"):
            detector.answer_patterns = config.strategy.get("detector_answer_patterns")
        step_generator = StepCandidateGeneratorThroughHuggingface(
            model=model,
            detector=detector,
            temperature=config.generation.temperature,
            max_new_tokens=config.generation.max_new_tokens,
            max_length=config.generation.max_length,
            top_p=config.generation.top_p,
            top_k=config.generation.top_k,
            disable_thinking_mode=config.model.disable_thinking_mode,
            generation_batch_size=config.generation.batch_size,
        )

    elif config.model.type == "openai_api":
        # Use model_name if available, otherwise fall back to model_path
        model_path = config.model.get("model_name") or config.model.get("model_path")
        log.info(f"Using OpenAI API model: {model_path}")

        # Check provider for API key and base URL (applies to all strategies)
        if config.model.get("provider") == "openrouter":
            api_key = config.model.get("api_key") or os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
        else:
            api_key = config.model.get("api_key") or os.getenv("OPENAI_API_KEY")
            base_url = config.model.get("base_url", None)

        # Check if DeepConf strategy
        if config.strategy.type == "deepconf":
            # DeepConf uses streaming with logprobs but no boundary detector
            model = BlackboxModelWithStreaming(
                openai_api_key=api_key,
                model_path=model_path,
                supports_logprobs=True,
                base_url=base_url,
            )
            step_generator = None  # DeepConf doesn't use step generator
        else:
            # Other strategies use boundary detection via early stopping
            from lm_polygraph.utils import APIWithUncertainty

            from llm_tts.early_stopping import BoundaryEarlyStopping

            # Determine thinking mode (same logic as vLLM)
            disable_thinking_mode = config.model.get("disable_thinking_mode", None)
            thinking_mode = disable_thinking_mode is False

            # Always use ThinkingMarkerDetector for step boundary detection
            if "min_step_tokens" not in config.strategy:
                log.warning("strategy.min_step_tokens not set, defaulting to 50")
            if "max_step_tokens" not in config.strategy:
                log.warning("strategy.max_step_tokens not set, defaulting to 300")
            detector = ThinkingMarkerDetector(
                min_step_tokens=config.strategy.get("min_step_tokens", 50),
                max_step_tokens=config.strategy.get("max_step_tokens", 300),
                use_sequence=config.strategy.get("use_sequence", True),
                use_conclusion=config.strategy.get("use_conclusion", True),
                use_thinking=config.strategy.get("use_thinking", True),
                use_verification=config.strategy.get("use_verification", True),
                use_structure=config.strategy.get("use_structure", False),
                use_reasoning=config.strategy.get("use_reasoning", False),
                use_correction=config.strategy.get("use_correction", False),
                custom_markers=config.strategy.get("custom_markers"),
            )

            generation_parameters = GenerationParameters()
            generation_parameters.temperature = config.generation.temperature
            generation_parameters.max_new_tokens = config.generation.max_new_tokens
            generation_parameters.top_p = config.generation.top_p
            generation_parameters.top_k = config.generation.top_k

            # Create boundary-based early stopping
            early_stopping = BoundaryEarlyStopping(detector=detector)

            supports_logprobs = config.model.get("supports_logprobs", True)

            model = BlackboxModelWithStreaming(
                openai_api_key=api_key,
                model_path=model_path,
                supports_logprobs=supports_logprobs,
                early_stopping=early_stopping,
                generation_parameters=generation_parameters,
                base_url=base_url,
            )

            # Set up uncertainty scorer if logprobs are supported and scorer is configured
            scorer_type = config.scorer.type if config.scorer else None
            if supports_logprobs and scorer_type and POLYGRAPH_UNCERTAINTY_AVAILABLE:
                if scorer_type == "perplexity":
                    stat_calculators = [VLLMLogprobsCalculator()]
                    estimator = Perplexity()
                elif scorer_type == "sequence_prob":
                    stat_calculators = [VLLMLogprobsCalculator()]
                    estimator = MaximumSequenceProbability()
                elif scorer_type == "uncertainty_pd":
                    from llm_tts.scorers.estimator_uncertainty_pd import PDGap

                    stat_calculators = [VLLMLogprobsCalculator(output_matrix=True)]
                    estimator = PDGap()
                elif scorer_type == "entropy":
                    stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
                    estimator = MeanTokenEntropy()
                elif scorer_type == "prm":
                    # PRM uses its own model; use entropy for generation scoring
                    stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
                    estimator = MeanTokenEntropy()
                else:
                    stat_calculators = None
                    estimator = None

                if stat_calculators and estimator:
                    # Wrap model with uncertainty scorer (same pattern as VLLMWithUncertainty)
                    model = APIWithUncertainty(
                        model=model,
                        stat_calculators=stat_calculators,
                        estimator=estimator,
                    )
                    log.info(
                        f"Wrapped model with APIWithUncertainty({type(estimator).__name__})"
                    )

            step_generator = StepCandidateGeneratorThroughAPI(
                model=model,
                thinking_mode=thinking_mode,
                detector=detector,
                answer_patterns=config.strategy.get(
                    "detector_answer_patterns",
                    [],
                ),
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                top_k=config.generation.get("top_k", 20),
                presence_penalty=config.generation.get("presence_penalty", 0.0),
                max_context_budget=config.model.get(
                    "max_context_budget", config.model.get("max_model_len", 32768)
                ),
                prefill_mode=config.model.get("prefill_mode", False),
                disable_thinking_mode=disable_thinking_mode,
                supports_logprobs=supports_logprobs,
                max_concurrent_requests=config.model.get(
                    "max_concurrent_requests", 256
                ),
            )

            log.info(
                f"Created API step generator: thinking_mode={thinking_mode}, "
                f"uncertainty={'APIWithUncertainty' if hasattr(model, 'estimator') else 'no'}"
            )
    else:
        raise ValueError(f"Model type {config.model.type} not supported")

    return model, step_generator


def _create_api_model_for_scorer(model_cfg):
    """Create an API-backed model instance for scoring only."""
    model_path = model_cfg.get("model_name") or model_cfg.get("model_path")
    log.info(f"LLM critic scorer API model: {model_path}")

    provider = model_cfg.get("provider")
    base_url = model_cfg.get("base_url")
    api_key_env = model_cfg.get("api_key_env")

    if api_key_env:
        api_key = model_cfg.get("api_key") or os.getenv(api_key_env)
    elif provider == "openrouter":
        api_key = model_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if base_url is None:
            base_url = "https://openrouter.ai/api/v1"
    else:
        api_key = model_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")

    return BlackboxModelWithStreaming(
        openai_api_key=api_key,
        model_path=model_path,
        supports_logprobs=model_cfg.get("supports_logprobs", False),
        base_url=base_url,
    )


def create_tts_strategy(
    config, model, step_generator, scorer, output_dir=None, flop_calculator=None
):
    # Set model on scorer if it supports it (e.g., StepScorerLLMCritic)
    if scorer is not None and hasattr(scorer, "set_model"):
        scorer_model_cfg = getattr(config.scorer, "model", None)
        scorer_model_type = (
            scorer_model_cfg.get("type", "openai_api") if scorer_model_cfg else None
        )

        if scorer_model_type == "vllm_shared":
            # Reuse the same vLLM engine that is used for generation
            vllm_engine = getattr(model, "vllm_engine", None)
            if vllm_engine is None:
                raise ValueError(
                    "scorer.model.type=vllm_shared requires a vLLM generation model"
                )
            scorer.set_model(vllm_engine, use_vllm=True)
            log.info("Scorer: reusing generation vLLM engine for scoring")
        elif scorer_model_type == "vllm":
            # Launch a vLLM OpenAI-compatible server as a subprocess on a
            # dedicated GPU, then connect to it via the API scorer path.
            import atexit
            import subprocess
            import time

            scorer_gpu = str(scorer_model_cfg.get("gpu", "1"))
            scorer_model_path = scorer_model_cfg.get(
                "model_path"
            ) or scorer_model_cfg.get("model_name")
            scorer_port = int(scorer_model_cfg.get("port", 8711))
            scorer_gpu_util = scorer_model_cfg.get("gpu_memory_utilization", 0.9)
            scorer_max_model_len = scorer_model_cfg.get("max_model_len", 4096)
            scorer_max_num_seqs = scorer_model_cfg.get("max_num_seqs", 8)
            scorer_max_num_batched_tokens = scorer_model_cfg.get(
                "max_num_batched_tokens", None
            )

            log.info(
                f"Scorer: launching vLLM server for {scorer_model_path} "
                f"on GPU {scorer_gpu}, port {scorer_port}, "
                f"gpu_memory_utilization={scorer_gpu_util}, "
                f"max_num_seqs={scorer_max_num_seqs}"
            )

            vllm_cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                scorer_model_path,
                "--port",
                str(scorer_port),
                "--gpu-memory-utilization",
                str(scorer_gpu_util),
                "--max-model-len",
                str(scorer_max_model_len),
                "--tensor-parallel-size",
                str(scorer_model_cfg.get("tensor_parallel_size", 1)),
                "--max-num-seqs",
                str(scorer_max_num_seqs),
                "--trust-remote-code",
                "--seed",
                str(config.system.seed),
            ]
            if scorer_max_num_batched_tokens is not None:
                vllm_cmd.extend(
                    [
                        "--max-num-batched-tokens",
                        str(scorer_max_num_batched_tokens),
                    ]
                )

            scorer_env = os.environ.copy()
            scorer_env["CUDA_VISIBLE_DEVICES"] = scorer_gpu

            # Redirect stdout/stderr to file to avoid pipe buffer deadlock.
            scorer_log_dir = output_dir or "."
            scorer_stdout_path = os.path.join(scorer_log_dir, "scorer_vllm_server.log")
            scorer_stdout_f = open(scorer_stdout_path, "w")
            scorer_proc = subprocess.Popen(
                vllm_cmd,
                env=scorer_env,
                stdout=scorer_stdout_f,
                stderr=subprocess.STDOUT,
            )

            def _kill_scorer_server():
                if scorer_proc.poll() is None:
                    log.info("Shutting down scorer vLLM server...")
                    scorer_proc.terminate()
                    try:
                        scorer_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        scorer_proc.kill()
                scorer_stdout_f.close()

            atexit.register(_kill_scorer_server)

            # Wait for the server to be ready
            import httpx

            base_url = f"http://localhost:{scorer_port}/v1"
            max_wait = 300  # 5 minutes
            start_time = time.time()
            while time.time() - start_time < max_wait:
                if scorer_proc.poll() is not None:
                    scorer_stdout_f.flush()
                    try:
                        with open(scorer_stdout_path, "r") as f:
                            log_tail = f.read()[-2000:]
                    except Exception:
                        log_tail = "(could not read log)"
                    raise RuntimeError(
                        f"Scorer vLLM server exited with code "
                        f"{scorer_proc.returncode}:\n{log_tail}"
                    )
                try:
                    resp = httpx.get(f"{base_url}/models", timeout=5)
                    if resp.status_code == 200:
                        log.info(
                            f"Scorer vLLM server ready at {base_url} "
                            f"(waited {time.time() - start_time:.1f}s)"
                        )
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                _kill_scorer_server()
                raise RuntimeError(
                    f"Scorer vLLM server did not start within {max_wait}s"
                )

            scorer_model = _create_api_model_for_scorer(
                {
                    "type": "openai_api",
                    "model_name": scorer_model_path,
                    "base_url": base_url,
                    "api_key": "unused",
                }
            )
            scorer.set_model(scorer_model, use_vllm=False)
            log.info(
                f"Scorer: using vLLM server on GPU {scorer_gpu} "
                f"via API at {base_url}"
            )
        elif scorer_model_cfg is not None and scorer_model_type == "openai_api":
            scorer_model = _create_api_model_for_scorer(scorer_model_cfg)
            scorer.set_model(scorer_model, use_vllm=False)
            log.info("Scorer: using API model")
        else:
            raise ValueError(
                "scorer.model must be specified for llm_critic scorer. "
                "Supported types: openai_api, vllm, vllm_shared"
            )

        # Initialize FLOP calculator for LLM critic token/compute tracking
        if hasattr(scorer, "init_flop_calculator"):
            try:
                scorer_model_name = (
                    scorer_model_cfg.get("model_name")
                    or scorer_model_cfg.get("model_path")
                    if scorer_model_cfg
                    else getattr(config.model, "model_path", None)
                )
                if scorer_model_name:
                    scorer.init_flop_calculator(scorer_model_name)
            except Exception as e:
                log.warning(f"Could not init LLM critic FLOP calculator: {e}")

    if config.strategy.type == "baseline":
        # Get eos_patterns from config, default to ["<end of response>"]
        eos_patterns = getattr(config.strategy, "detector_eos_patterns", None)
        if eos_patterns:
            eos_patterns = list(eos_patterns)
        # Get stop_token_ids from config (e.g., [151645, 151643] for Qwen2)
        stop_token_ids = getattr(config.strategy, "stop_token_ids", None)
        if stop_token_ids:
            stop_token_ids = list(stop_token_ids)
        # Get batch_generation flag (default True for backwards compatibility)
        # Set to False to enable uncertainty scoring via VLLMWithUncertainty wrapper
        batch_generation = config.strategy.get("batch_generation", True)
        strategy = StrategyBaseline(
            step_generator=step_generator,
            output_dir=output_dir,
            eos_patterns=eos_patterns,
            stop_token_ids=stop_token_ids,
            batch_generation=batch_generation,
        )
    elif config.strategy.type == "online_best_of_n":
        batch_generation = config.strategy.get("batch_generation", True)
        strategy = StrategyOnlineBestOfN(
            step_generator=step_generator,
            scorer=scorer,
            candidates_per_step=config.strategy.candidates_per_step,
            max_steps=config.strategy.max_steps,
            output_dir=output_dir,
            batch_generation=batch_generation,
        )
    elif config.strategy.type == "offline_best_of_n":
        # Offline Best-of-N generates N trajectories, scores with PRM, selects best
        # With batch_generation=True, all M×N trajectories generated in single vLLM call
        batch_generation = config.strategy.get("batch_generation", True)

        # Multi-scoring: per-scorer flags for computing extra metrics
        multi_score_flags = {}
        for flag in [
            "calculate_entropy_score",
            "calculate_perplexity_score",
            "calculate_sequence_prob_score",
            "calculate_pd_gap_score",
            "calculate_prm_score",
        ]:
            multi_score_flags[flag] = getattr(config.strategy, flag, False)

        # Create separate PRM scorer if calculate_prm_score=True
        # and primary scorer is not already PRM
        extra_prm_scorer = None
        if multi_score_flags.get("calculate_prm_score"):
            primary_is_prm = (
                hasattr(scorer, "prm_model") and scorer.prm_model is not None
            )
            if not primary_is_prm:
                prm_model_path = getattr(config.strategy, "prm_model_path", None)
                if prm_model_path:
                    log.info(
                        f"Creating extra PRM scorer for multi-scoring: {prm_model_path}"
                    )
                    extra_prm_scorer = StepScorerPRM(
                        prm_model_path=prm_model_path,
                        device=getattr(config.strategy, "prm_device", "cuda:1"),
                        batch_size=getattr(config.strategy, "prm_batch_size", 4),
                        torch_dtype=getattr(
                            config.strategy, "prm_torch_dtype", "bfloat16"
                        ),
                        use_vllm=getattr(config.strategy, "prm_use_vllm", True),
                    )
                else:
                    log.warning(
                        "calculate_prm_score=True but no prm_model_path specified"
                    )

        strategy = StrategyOfflineBestOfN(
            scorer=scorer,
            num_trajectories=config.strategy.get("num_trajectories", 4),
            max_steps=config.strategy.max_steps,
            step_generator=step_generator,
            score_aggregation=config.strategy.get("score_aggregation", "mean"),
            output_dir=output_dir,
            batch_generation=batch_generation,
            scoring_window=config.strategy.get("scoring_window", None),
            prm_scorer=extra_prm_scorer,
            **multi_score_flags,
        )
    elif config.strategy.type == "adaptive":
        strategy = AdaptiveScalingBestOfN(
            step_generator=step_generator,
            scorer=scorer,
            candidates_per_step=config.strategy.candidates_per_step,
            max_steps=config.strategy.max_steps,
            adaptive_scaling_method=config.strategy.adaptive_scaling_method,
            scaling_rate=config.strategy.scaling_rate,
            momentum_rate=config.strategy.momentum_rate,
            batch_size=config.strategy.get("batch_size", 1000),
        )
    elif config.strategy.type == "deepconf":
        # DeepConf supports both API models (with logprobs) and local HuggingFace models
        # Validation is done inside StrategyDeepConf.__init__
        strategy = StrategyDeepConf(
            model=model,
            mode=config.strategy.mode,
            budget=config.strategy.get("budget", 8),
            warmup_traces=config.strategy.get("warmup_traces", 4),
            total_budget=config.strategy.get("total_budget", 10),
            confidence_percentile=config.strategy.get("confidence_percentile", 90),
            window_size=config.strategy.get("window_size", 2048),
            filter_method=config.strategy.get("filter_method", "top10"),
            temperature=config.strategy.get("temperature", 0.7),
            top_p=config.strategy.get("top_p", 1.0),
            max_tokens=config.strategy.get("max_tokens", 512),
            top_logprobs=config.strategy.get("top_logprobs", 20),
            n_threads=config.strategy.get("n_threads", 8),
            disable_thinking_mode=config.model.get("disable_thinking_mode", True),
            seed=config.system.seed,
        )

    elif config.strategy.type == "beam_search":
        strategy = StrategyBeamSearch(
            step_generator=step_generator,
            scorer=scorer,
            beam_size=config.strategy.beam_size,
            candidates_per_beam=config.strategy.candidates_per_beam,
            max_steps=config.strategy.max_steps,
            aggregation=getattr(config.strategy, "aggregation", "mean"),
            batch_generation=config.strategy.get("batch_generation", True),
            scoring_window=config.strategy.get("scoring_window", None),
        )
    elif config.strategy.type == "phi_decoding":
        strategy = PhiDecoding(
            step_generator=step_generator,
            scorer=scorer,
            candidates_per_step=config.strategy.candidates_per_step,
            max_steps=config.strategy.max_steps,
            cluster_num=config.strategy.cluster_num,
        )
    elif config.strategy.type == "self_consistency":
        if step_generator is None:
            raise ValueError(
                "Self-consistency strategy requires step_generator. "
                "Ensure model.type is 'vllm' and step generator is created."
            )
        # Get batch_generation flag (default True for fully batched mode)
        batch_generation = config.strategy.get("batch_generation", True)
        # Get data_name for official answer extraction (ensures consistency with final evaluation)
        data_name = config.strategy.get("data_name", None)
        strategy = StrategySelfConsistency(
            step_generator=step_generator,
            num_paths=config.strategy.get("num_paths", 10),
            scorer=scorer,
            batch_generation=batch_generation,
            data_name=data_name,
        )
    elif config.strategy.type == "uncertainty_cot":
        strategy = StrategyUncertaintyCoT(
            step_generator=step_generator,
            candidates_per_step=config.strategy.candidates_per_step,
            max_steps=config.strategy.max_steps,
            max_empty_steps=config.strategy.max_empty_steps,
            uncertainty_threshold=config.strategy.uncertainty_threshold,
            uncertainty_sampling=config.strategy.uncertainty_sampling,
        )
    elif config.strategy.type == "extended_thinking":
        eos_patterns = getattr(config.strategy, "detector_eos_patterns", None)
        if eos_patterns:
            eos_patterns = list(eos_patterns)
        stop_token_ids = getattr(config.strategy, "stop_token_ids", None)
        if stop_token_ids:
            stop_token_ids = list(stop_token_ids)
        strategy = StrategyExtendedThinking(
            step_generator=step_generator,
            max_continuations=config.strategy.get("max_continuations", 3),
            continuation_token=config.strategy.get("continuation_token", "\nWait, "),
            max_steps=config.strategy.get("max_steps", 50),
            output_dir=output_dir,
            eos_patterns=eos_patterns,
            stop_token_ids=stop_token_ids,
        )
    else:
        raise ValueError(f"Strategy type {config.strategy.type} not supported")

    return strategy


def _generate_trajectories_batch(
    results,
    save_path,
    strategy,  # StrategyBaseline or StrategySelfConsistency
    dataset: Dataset,
    processed_indices: set,
    prompt_template: str,
    system_prompt: str,
    question_field: str,
    answer_field: str,
    phase1_evaluators: dict,  # Dict of evaluator_name -> evaluator
    save_path_file: Path,
    sample_metrics_path: Path,
    checkpoint_batch_size: int = 32,  # Save intermediate results every N samples
):
    """
    Batch generation for strategies that support it (baseline, self_consistency).

    Generates all samples in a single vLLM call, which is significantly faster
    than sequential generation because vLLM can process all prompts together
    with continuous batching.

    With checkpointing: Large batches are split into smaller chunks to save
    intermediate results after each chunk, preventing data loss for long-running jobs.
    """
    strategy_name = getattr(strategy, "__class__", type(strategy)).__name__
    log.info(f"Using batch generation mode for {strategy_name}")

    subset_size = len(dataset)

    # Collect all requests that need to be processed
    requests_to_process = []
    indices_to_process = []
    instances_to_process = []
    gold_answers = []

    for i in range(subset_size):
        if i in processed_indices:
            log.info(f"Skipping sample {i} (already processed)")
            continue

        instance = dataset[i]
        question = instance[question_field]

        # Handle answer with fallback for Game of 24
        if answer_field and answer_field in instance and instance[answer_field]:
            if "####" in instance[answer_field]:
                from llm_tts.datasets.gsm8k import extract_answer_from_gsm8k

                gold_answer_num = extract_answer_from_gsm8k(instance[answer_field])
            else:
                gold_answer_num = instance[answer_field]
        else:
            gold_answer_num = "24"

        # Build request
        request = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    prompt_template.format(question=question)
                    if prompt_template and "{question}" in prompt_template
                    else question
                ),
            },
        ]

        requests_to_process.append(request)
        indices_to_process.append(i)
        instances_to_process.append(instance)
        gold_answers.append(gold_answer_num)

    if not requests_to_process:
        log.info("No new samples to process")
        return results

    total_to_process = len(requests_to_process)
    log.info(
        f"Batch generating {total_to_process} samples with checkpoint_batch_size={checkpoint_batch_size}..."
    )

    # Split into chunks for intermediate checkpointing
    num_chunks = (total_to_process + checkpoint_batch_size - 1) // checkpoint_batch_size

    # Collect candidates data for multi-scoring analysis (saved to candidates.json)
    all_candidates_data = []

    for chunk_idx in range(num_chunks):
        batch_start = chunk_idx * checkpoint_batch_size
        batch_end = min(batch_start + checkpoint_batch_size, total_to_process)

        # Extract this chunk
        chunk_requests = requests_to_process[batch_start:batch_end]
        chunk_indices = indices_to_process[batch_start:batch_end]
        chunk_instances = instances_to_process[batch_start:batch_end]
        chunk_gold_answers = gold_answers[batch_start:batch_end]

        log.info(
            f"Processing chunk {chunk_idx + 1}/{num_chunks}: "
            f"samples {batch_start}-{batch_end - 1} ({len(chunk_requests)} samples)"
        )

        # Define progressive save callback for this chunk
        def _save_callback(strategy_results, phase="post_generation"):
            temp_results = list(results)  # copy previous chunks
            for i_idx, inst, gold, strat_res in zip(
                chunk_indices, chunk_instances, chunk_gold_answers, strategy_results
            ):
                temp_results.append(
                    {
                        "index": i_idx,
                        "question": inst[question_field],
                        "gold_answer": gold,
                        "generated_trajectory": strat_res.get("trajectory", ""),
                        "extracted_answer": strat_res.get("extracted_answer", ""),
                        "answer_step": strat_res.get("answer_step"),
                        "steps": [
                            s.text if hasattr(s, "text") else str(s)
                            for s in strat_res.get("steps", [])
                        ],
                        "reasoning_steps": strat_res.get("reasoning_steps", 0),
                        "validity_scores": strat_res.get("validity_scores", []),
                        "aggregated_score": strat_res.get("aggregated_score", 0.0),
                        "all_scores": strat_res.get("all_scores", []),
                        "all_step_scores": strat_res.get("all_step_scores", []),
                        "best_idx": strat_res.get("best_idx"),
                        "completed": strat_res.get("completed", False),
                        "is_correct": None,
                        "eval": {},
                        "scoring_phase": phase,
                    }
                )
            save_results_json(temp_results, save_path_file)
            log.info(
                f"Progressive save ({phase}): {len(temp_results)} results to {save_path_file}"
            )

        # Generate this chunk
        try:
            chunk_results = strategy.generate_trajectories_batch(
                chunk_requests, chunk_indices, save_callback=_save_callback
            )
        except Exception as e:
            import traceback

            log.error(f"Chunk {chunk_idx + 1} generation failed: {e}")
            log.error(f"Traceback:\n{traceback.format_exc()}")
            log.error("Saving partial results collected so far and exiting")
            save_results_json(results, save_path_file)
            log.info(f"Checkpoint saved: {len(results)}/{subset_size} samples complete")
            return results

        if len(chunk_results) != len(chunk_indices):
            log.error(
                f"Chunk generation returned {len(chunk_results)} results "
                f"but expected {len(chunk_indices)}. Truncating to shorter list."
            )
            # Truncate to match
            chunk_indices = chunk_indices[: len(chunk_results)]
            chunk_instances = chunk_instances[: len(chunk_results)]
            chunk_gold_answers = chunk_gold_answers[: len(chunk_results)]

        # Save batch results immediately to avoid data loss
        batch_results_path = save_path_file.parent / "batch_results.jsonl"

        # Append mode for chunks after the first one
        mode = "a" if (batch_start > 0 or batch_results_path.exists()) else "w"
        with open(batch_results_path, mode) as f:
            for idx, (i, instance, gold_answer, result) in enumerate(
                zip(chunk_indices, chunk_instances, chunk_gold_answers, chunk_results)
            ):
                record = {
                    "index": i,
                    "question": instance[question_field],
                    "gold_answer": gold_answer,
                    "trajectory": result.get("trajectory", ""),
                    "extracted_answer": result.get("extracted_answer", ""),
                    "steps": [
                        s.text if hasattr(s, "text") else str(s)
                        for s in result.get("steps", [])
                    ],
                    "answer_step": result.get(
                        "answer_step"
                    ),  # Final answer text (thinking mode)
                    "validity_scores": result.get("validity_scores", []),
                    "all_step_scores": result.get("all_step_scores", []),
                    "all_scores": result.get("all_scores", []),
                    "best_idx": result.get("best_idx"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info(f"Saved {len(chunk_results)} batch results to {batch_results_path}")

        # Process results from this chunk
        for idx, (i, instance, gold_answer_num, result) in enumerate(
            zip(chunk_indices, chunk_instances, chunk_gold_answers, chunk_results)
        ):
            question = instance[question_field]

            # Extract generated answer
            if "extracted_answer" in result and result["extracted_answer"]:
                generated_text = result["extracted_answer"]
            else:
                log.warning(
                    f"Sample {i}: no extracted_answer found, answer extraction may have failed"
                )
                generated_text = ""

            # Log result
            log.info("\n" + "=" * 60)
            log.info(f"Sample {i + 1}/{subset_size}")
            log.info(f"Question: {question[:200]}...")
            log.info(f"Gold answer: {gold_answer_num}")

            log.info("\n" + "-" * 60)
            log.info("GENERATED STEPS:")
            log.info("-" * 60)

            if result["steps"] and isinstance(result["steps"], list):
                # Skip last step if it duplicates the answer_step content
                # (online BoN / self-consistency append answer to steps AND store in answer_step)
                answer_step_text = result.get("answer_step") or ""
                last_step_text = (
                    result["steps"][-1].text
                    if hasattr(result["steps"][-1], "text")
                    else str(result["steps"][-1])
                )
                skip_last = (
                    bool(answer_step_text)
                    and len(result["steps"]) > 1
                    and last_step_text.strip()[:200] == answer_step_text.strip()[:200]
                )
                steps_to_log = result["steps"][:-1] if skip_last else result["steps"]
                for step_idx, step in enumerate(steps_to_log):
                    validity = (
                        result.get("validity_scores", [])[step_idx]
                        if "validity_scores" in result
                        and step_idx < len(result["validity_scores"])
                        else "N/A"
                    )
                    confidence_str = (
                        f"{validity:.3f}"
                        if isinstance(validity, (int, float))
                        else validity
                    )
                    log.info(f"\nStep {step_idx + 1} (confidence: {confidence_str}):")
                    step_text = step.text if hasattr(step, "text") else str(step)
                    log.info(step_text)

            # Log answer step separately for thinking mode, or full trajectory for non-thinking
            if result.get("answer_step"):
                log.info("\nGenerated Answer (confidence: N/A):")
                log.info(result["answer_step"])
            else:
                log.info(f"\nFull trajectory:\n{result['trajectory']}")

            # Check correctness with all evaluators
            eval_results = {}
            for eval_name, evaluator in phase1_evaluators.items():
                try:
                    if isinstance(evaluator, EvaluatorExactMatch):
                        # EvaluatorExactMatch._score_single takes 3-tuple, returns float
                        # For self-consistency, use extracted_answer if available (trajectory is aggregated)
                        solution = result.get("extracted_answer")
                        # Convert to string for comparison
                        if isinstance(solution, (int, float)):
                            solution = str(solution)
                        score = evaluator._score_single(
                            (question, solution, str(gold_answer_num)),
                            pre_extracted=True,
                        )
                        is_correct_eval = bool(score)
                    elif isinstance(evaluator, EvaluatorLLMAsAJudge):
                        # LLM judges: __call__ takes lists, returns (labels, responses, consensus_scores)
                        # For answer_only mode, use extracted answer
                        if (
                            hasattr(evaluator, "mode")
                            and evaluator.mode == "answer_only"
                        ):
                            # Get proposed answer - check if actually provided
                            proposed_answer = result.get("extracted_answer")
                            if not proposed_answer or (
                                isinstance(proposed_answer, str)
                                and proposed_answer.strip() == ""
                            ):
                                # No answer produced - mark as incorrect, don't waste API call on empty trajectory
                                is_correct_eval = False
                                eval_results[eval_name] = {
                                    "is_correct": is_correct_eval,
                                    "consensus": 0.0,
                                    "response": "No answer generated",
                                }
                                continue
                            solution = proposed_answer
                        else:
                            solution = result["trajectory"]
                        labels, responses, consensus_scores = evaluator(
                            [question], [solution], [str(gold_answer_num)]
                        )
                        is_correct_eval = labels[0] == 1 if labels else False
                        eval_results[eval_name] = {
                            "is_correct": is_correct_eval,
                            "consensus": (
                                consensus_scores[0] if consensus_scores else 0.0
                            ),
                            "response": responses[0] if responses else "",
                        }
                        continue
                    elif isinstance(
                        evaluator, (EvaluatorMBPPPlus, EvaluatorHumanEvalPlus)
                    ):
                        # Skip EvalPlus in phase 1 - will run batch evaluation once in phase 2
                        # (Running EvalPlus per-sample is inefficient)
                        log.debug(
                            f"Skipping EvalPlus evaluation for sample {i} in phase 1"
                        )
                        continue
                    else:
                        # Fallback: try __call__ with lists
                        result_output = evaluator(
                            [question], [result["trajectory"]], [str(gold_answer_num)]
                        )
                        if isinstance(result_output, tuple) and len(result_output) == 2:
                            labels, responses = result_output
                            is_correct_eval = labels[0] == 1 if labels else False
                        elif isinstance(result_output, list):
                            is_correct_eval = (
                                bool(result_output[0]) if result_output else False
                            )
                        else:
                            is_correct_eval = False
                    eval_results[eval_name] = {"is_correct": is_correct_eval}
                except Exception as e:
                    log.error(f"Evaluator {eval_name} failed: {e}")
                    traceback.print_exc()
                    # Don't store failed evaluation - let it retry in batch phase
                    log.warning(
                        f"Skipping {eval_name} for sample {i} in phase 1, "
                        "will retry in batch evaluation phase"
                    )

            # Use exact_match as primary for logging (if available)
            is_correct = eval_results.get("exact_match", {}).get("is_correct", False)

            log.info("\n" + "=" * 60)
            log.info(f"FINAL ANSWER: {generated_text}")
            log.info(f"Gold answer:  {gold_answer_num}")
            for eval_name, eval_result in eval_results.items():
                status = "✓ YES" if eval_result.get("is_correct") else "✗ NO"
                log.info(f"[{eval_name}]: {status}")
            log.info("-" * 60)
            log.info(f"Num steps: {len(result['steps'])}")
            if "validity_scores" in result and result["validity_scores"]:
                scores = [s for s in result["validity_scores"] if s is not None]
                if scores:
                    log.info(
                        f"Confidence:  avg={np.mean(scores):.3f}, min={np.min(scores):.3f}, max={np.max(scores):.3f}"
                    )
            log.info("=" * 60)

            # Store result with per-evaluator results
            result_dict = {
                "index": i,
                "question": question,
                "gold_answer": gold_answer_num,
                "generated_trajectory": result["trajectory"],
                "extracted_answer": generated_text,
                "answer_step": result.get("answer_step"),
                "steps": result["steps"],
                "reasoning_steps": result.get("reasoning_steps", len(result["steps"])),
                "validity_scores": result.get("validity_scores", []),
                "completed": result["completed"],
                "is_correct": bool(is_correct),  # Primary (exact_match)
                "eval": eval_results,  # Per-evaluator results
                "instance_data": dict(
                    instance
                ),  # Store full instance for MBPP+ evaluation
            }

            if "token_stats" in result:
                result_dict["token_stats"] = result["token_stats"]

            # Propagate beam search diagnostics
            for key in (
                "trajectory_tokens",
                "answer_tokens",
                "context_limit_hit",
                "max_steps_hit",
            ):
                if key in result:
                    result_dict[key] = result[key]

            # Store answer_step for thinking mode strategies
            if "answer_step" in result:
                result_dict["answer_step"] = result["answer_step"]

            # Store dataset-specific fields for evaluators (e.g., MBPP+ test_list, task_id)
            if "task_id" in instance:
                result_dict["task_id"] = instance["task_id"]
            if "test_list" in instance:
                result_dict["test_list"] = instance["test_list"]
            if "entry_point" in instance:
                result_dict["entry_point"] = instance["entry_point"]

            results.append(result_dict)

            # Collect candidates_data for multi-scoring analysis
            if "candidates_data" in result:
                all_candidates_data.append(
                    {
                        "index": i,
                        "question": question,
                        "gold_answer": gold_answer_num,
                        "num_candidates": len(result["candidates_data"]),
                        "candidates": result["candidates_data"],
                    }
                )

            # Compute running metrics
            token_stats = result.get("token_stats")
            if token_stats is None:
                log.warning(f"Sample {i}: missing 'token_stats' in result")
                token_stats = {}
            all_token_stats = []
            for r in results:
                ts = r.get("token_stats")
                if ts is None:
                    ts = {}
                all_token_stats.append(ts)

            running_total_tokens = sum(
                ts.get("total_tokens_this_sample", 0) for ts in all_token_stats
            )
            running_total_tflops = sum(
                _safe_tflops(ts, "tflops") for ts in all_token_stats
            )

            # Compute running accuracy per evaluator
            running_stats = {}
            for eval_name in phase1_evaluators.keys():
                correct_count = sum(
                    1
                    for r in results
                    if r.get("eval", {}).get(eval_name, {}).get("is_correct", False)
                )
                accuracy = (correct_count / len(results)) if results else 0.0
                running_stats[eval_name] = {
                    "correct": correct_count,
                    "accuracy": accuracy,
                }
                log.info(
                    f"Running accuracy [{eval_name}]: {correct_count}/{len(results)} = {accuracy:.3f}"
                )

            sample_metrics = {
                "sample_index": i,
                "is_correct": bool(is_correct),
                "reasoning_steps": result.get("reasoning_steps", len(result["steps"])),
                "samples_completed": len(results),
                "total_tokens_this_sample": token_stats.get(
                    "total_tokens_this_sample", 0
                ),
                "input_tokens_this_sample": token_stats.get("input_tokens", 0),
                "output_tokens_this_sample": token_stats.get("output_tokens", 0),
                "generations_this_sample": token_stats.get("generation_count", 0),
                "tflops_this_sample": _safe_tflops(token_stats, "tflops"),
                "prm_tokens_this_sample": token_stats.get("prm_input_tokens", 0),
                "prm_tflops_this_sample": _safe_tflops(token_stats, "prm_tflops"),
                "running_avg_tokens_per_sample": (
                    (running_total_tokens / len(results)) if results else 0.0
                ),
                "running_total_tokens": running_total_tokens,
                "running_total_tflops": running_total_tflops,
            }
            # Add per-evaluator running accuracy to metrics
            for eval_name, stats in running_stats.items():
                safe_name = eval_name.replace("-", "_").replace(".", "_")
                sample_metrics[f"running_correct_{safe_name}"] = stats["correct"]
                sample_metrics[f"running_accuracy_{safe_name}"] = stats["accuracy"]

            # Beam search per-sample metrics: token breakdown and context limit
            if "trajectory_tokens" in result:
                sample_metrics["trajectory_tokens"] = result["trajectory_tokens"]
            if "answer_tokens" in result:
                sample_metrics["answer_tokens"] = result["answer_tokens"]
            if "context_limit_hit" in result:
                sample_metrics["context_limit_hit"] = int(result["context_limit_hit"])
            if "max_steps_hit" in result:
                sample_metrics["max_steps_hit"] = int(result["max_steps_hit"])
            if "completion_reason" in result and result["completion_reason"]:
                sample_metrics["completion_reason"] = result["completion_reason"]

            if "validity_scores" in result and result["validity_scores"]:
                valid_scores = [s for s in result["validity_scores"] if s is not None]
                if valid_scores:
                    sample_metrics["confidence"] = float(np.mean(valid_scores))

            # Append metrics line
            try:
                with open(sample_metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample_metrics, ensure_ascii=False) + "\n")
            except Exception as e:
                log.warning(f"Failed to append sample metrics: {e}")

            # Log to wandb
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(sample_metrics)
            except Exception:
                pass

        # Checkpoint: save results after each chunk completes
        save_results_json(results, save_path_file)
        log.info(
            f"Chunk {chunk_idx + 1}/{num_chunks} complete: "
            f"{len(results)}/{subset_size} samples processed, checkpoint saved"
        )

    # Final save after all chunks complete
    save_results_json(results, save_path_file)
    log.info(f"Final save: {len(results)} results to {save_path_file}")

    # Save candidates.json for multi-scoring post-hoc analysis
    if all_candidates_data:
        candidates_path = save_path_file.parent / "candidates.json"
        save_results_json(all_candidates_data, candidates_path)
        log.info(
            f"Saved {len(all_candidates_data)} samples with candidates data "
            f"to {candidates_path}"
        )

    return results


def generate_trajectories(
    results,
    save_path,
    strategy,
    dataset: Dataset,
    processed_indices: set,
    prompt_template: str,
    system_prompt: str = "",
    question_field: str = "question",
    answer_field: str = "answer",
    exact_match_dataset_answer_format: str = "numeric",
    data_name: str = None,  # Required - must be passed explicitly
    config=None,  # Optional - needed for multi-evaluator support
):
    if not data_name:
        raise ValueError("data_name is required for generate_trajectories()")

    # Phase 1: Generate trajectories (without checking correctness)
    log.info("\n" + "=" * 60)
    log.info("Phase 1: Generating trajectories")
    log.info("=" * 60)

    save_path_file = Path(save_path) / "results.json"
    sample_metrics_path = Path(save_path) / "sample_metrics.jsonl"

    # Build all evaluators if config is provided, otherwise use just exact_match
    if config is not None:
        phase1_evaluators = build_evaluators(config)
        log.info(f"Phase 1 evaluators: {list(phase1_evaluators.keys())}")
    else:
        exact_match_evaluator = EvaluatorExactMatch(
            dataset_answer_format=exact_match_dataset_answer_format,
            data_name=data_name,
        )
        phase1_evaluators = {"exact_match": exact_match_evaluator}
        log.info(f"Phase 1 evaluator: data_name={exact_match_evaluator.data_name}")

    # Get checkpoint batch size: default to dataset subset size (process all at once)
    checkpoint_batch_size = len(dataset)
    if config is not None and hasattr(config, "generation"):
        checkpoint_batch_size = config.generation.get(
            "checkpoint_batch_size", checkpoint_batch_size
        )

    return _generate_trajectories_batch(
        results=results,
        save_path=save_path,
        strategy=strategy,
        dataset=dataset,
        processed_indices=processed_indices,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        question_field=question_field,
        answer_field=answer_field,
        phase1_evaluators=phase1_evaluators,
        save_path_file=save_path_file,
        sample_metrics_path=sample_metrics_path,
        checkpoint_batch_size=checkpoint_batch_size,
    )


def log_evaluation_inconsistencies(results, save_path: str):
    """Log samples where exact_match and llm_judge disagree."""
    if not results:
        log.info("No results to check for inconsistencies")
        return

    # Find llm_judge evaluator name
    llm_judge_name = None
    for key in results[0].get("eval", {}):
        if key.startswith("llm_judge"):
            llm_judge_name = key
            break

    if not llm_judge_name:
        log.info("No LLM judge evaluator found, skipping inconsistency logging")
        return

    if "exact_match" not in results[0].get("eval", {}):
        log.info("No exact_match evaluator found, skipping inconsistency logging")
        return

    inconsistencies = []
    for result in results:
        eval_data = result.get("eval", {})
        if "exact_match" not in eval_data or llm_judge_name not in eval_data:
            continue

        em_result = eval_data["exact_match"].get("is_correct", False)
        llm_result = eval_data[llm_judge_name].get("is_correct", False)

        if em_result != llm_result:
            # Create inconsistency record with required keys
            inconsistency = {
                "index": result.get("index"),
                "question": result.get("question"),
                "gold_answer": result.get("gold_answer"),
                "generated_trajectory": result.get("generated_trajectory", ""),
                "extracted_answer": result.get("extracted_answer", ""),
                "answer_step": result.get("answer_step", ""),
                "eval": eval_data,
                "instance_data": result.get("instance_data", {}),
            }
            inconsistencies.append(inconsistency)

    if inconsistencies:
        inconsistencies_path = Path(save_path) / "eval_inconsistencies.json"
        try:
            with open(inconsistencies_path, "w", encoding="utf-8") as f:
                json.dump(inconsistencies, f, indent=2, ensure_ascii=False)
            log.info(
                f"Logged {len(inconsistencies)} inconsistencies to {inconsistencies_path}"
            )

            # Log summary of inconsistencies
            em_correct_llm_incorrect = sum(
                1 for i in inconsistencies if i["eval"]["exact_match"]["is_correct"]
            )
            llm_correct_em_incorrect = sum(
                1 for i in inconsistencies if i["eval"][llm_judge_name]["is_correct"]
            )
            log.info(f"  EM correct, LLM incorrect: {em_correct_llm_incorrect}")
            log.info(f"  LLM correct, EM incorrect: {llm_correct_em_incorrect}")
        except Exception as e:
            log.warning(f"Failed to save inconsistencies: {e}")
    else:
        log.info(
            "No evaluation inconsistencies found between exact_match and llm_judge"
        )


def evaluate_results(
    config,
    results,
    save_path: str,
):
    # Phase 2: Check correctness for all results
    log.info("\n" + "=" * 60)
    log.info("Phase 2: Checking correctness")
    log.info("=" * 60)

    # Build evaluators dynamically (regular evaluators from config.evaluation.evaluators)
    evaluators = build_evaluators(config)
    log.info(f"Using evaluators: {list(evaluators.keys())}")

    # Move EvalPlus evaluators to batch evaluation (running per-sample is inefficient)
    mbpp_evaluator = evaluators.pop("mbpp_plus", None)
    human_eval_plus_evaluator = evaluators.pop("human_eval_plus", None)

    # Build batch evaluators (from config.evaluation.batch_evaluators)
    batch_evaluator_names = config.evaluation.get("batch_evaluators", [])
    batch_evaluators = {}
    for eval_name in batch_evaluator_names:
        if eval_name == "llm_judge":
            # API-based LLM judge for batch evaluation
            llm_cfg = OmegaConf.to_container(config.evaluation.llm_judge, resolve=True)
            prompt_template = (
                load_prompt_template(llm_cfg.get("prompt_file"))
                if llm_cfg.get("prompt_file")
                else ""
            )
            if "{question}" in prompt_template:
                prompt_template = prompt_template.replace("{question}", "{q}")

            # Set API key in environment based on provider
            provider = llm_cfg.get("provider")
            if provider == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY")

            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

            # Remove config-only params not needed by evaluator
            llm_cfg.pop("prompt_file", None)
            llm_cfg.pop("provider", None)
            llm_cfg["prompt"] = prompt_template

            # Include model name in evaluator key
            model_name = llm_cfg.get("model", "unknown")
            sanitized_model = model_name.replace("/", "_").replace(":", "_")
            eval_key = f"llm_judge_{sanitized_model}"
            batch_evaluators[eval_key] = EvaluatorLLMAsAJudge(**llm_cfg)

    # Add EvalPlus evaluators to batch (single EvalPlus run is much more efficient)
    if mbpp_evaluator is not None:
        batch_evaluators["mbpp_plus"] = mbpp_evaluator
        log.info("MBPP+ moved to batch evaluation (single EvalPlus run)")
    if human_eval_plus_evaluator is not None:
        batch_evaluators["human_eval_plus"] = human_eval_plus_evaluator
        log.info("HumanEval+ moved to batch evaluation (single EvalPlus run)")

    if batch_evaluators:
        log.info(f"Batch evaluators: {list(batch_evaluators.keys())}")

    save_path_file = Path(save_path) / "results.json"

    # Process each evaluator separately (allows resuming per-evaluator)
    for eval_name, evaluator_fn in evaluators.items():
        log.info(f"\n--- Evaluator: {eval_name} ---")
        if hasattr(evaluator_fn, "data_name"):
            log.info(f"  data_name: {evaluator_fn.data_name}")

        # Evaluate samples one at a time and save after each
        samples_evaluated = 0
        for i, result in enumerate(results):
            if "error" in result:
                continue

            # Check if this evaluator has already processed this sample
            if "eval" in result and eval_name in result["eval"]:
                log.info(
                    f"Skipping sample {result['index']} (already evaluated by {eval_name})"
                )
                continue

            log.info(f"Evaluating sample {result['index']} with {eval_name}...")

            try:
                # Evaluate single sample - use SAME code path as Phase 1
                solution = result.get(
                    "generated_trajectory", result.get("trajectory", "")
                )

                # For exact_match: call _score_single exactly like Phase 1
                if eval_name == "exact_match" and hasattr(
                    evaluator_fn, "_score_single"
                ):
                    gold_str = str(result["gold_answer"])

                    # SAME call as Phase 1 line 869-871
                    score = evaluator_fn._score_single(
                        (result["question"], solution, gold_str)
                    )
                    is_correct = score == 1.0
                    annotation = 1.0 if is_correct else 0.0

                    # Debug: compare with Phase 1 result
                    phase1_correct = result.get("is_correct", None)
                    if phase1_correct is not None and phase1_correct != is_correct:
                        log.warning(
                            f"MISMATCH sample {result['index']}: phase1={phase1_correct}, phase2={is_correct}"
                        )
                        log.warning(
                            f"  solution_len={len(solution)}, gold={repr(gold_str[:50])}"
                        )
                elif isinstance(
                    evaluator_fn, (EvaluatorMBPPPlus, EvaluatorHumanEvalPlus)
                ):
                    # Skip EvalPlus in per-sample loop - will run batch evaluation once
                    # (Running EvalPlus per-sample is inefficient)
                    continue
                else:
                    # For LLM judges in answer_only mode, check for empty answer
                    if (
                        hasattr(evaluator_fn, "mode")
                        and evaluator_fn.mode == "answer_only"
                    ):
                        extracted_answer = result.get("extracted_answer", "")
                        if not extracted_answer or (
                            isinstance(extracted_answer, str)
                            and extracted_answer.strip() == ""
                        ):
                            # Empty answer - mark as incorrect without calling evaluator
                            annotation = 0
                            is_correct = False
                            eval_data = {
                                "label": int(annotation),
                                "is_correct": bool(is_correct),
                                "response": "No answer generated",
                            }
                            results[i].setdefault("eval", {})[eval_name] = eval_data
                            log.info(
                                f"Sample {result['index']} [{eval_name}]: "
                                f"0 (Incorrect) - No answer generated"
                            )
                            save_results_json(results, save_path_file)
                            samples_evaluated += 1
                            continue
                    else:
                        extracted_answer = result.get(
                            "generated_trajectory", result.get("trajectory", "")
                        )

                    eval_result = evaluator_fn(
                        [result["question"]],
                        [extracted_answer],
                        [result["gold_answer"]],
                    )
                    if isinstance(eval_result, tuple):
                        annotations, responses = eval_result
                    else:
                        annotations = eval_result
                    annotation = annotations[0]
                    if np.isnan(annotation):
                        log.warning(
                            f"{eval_name} returned unclear result for sample "
                            f"{result['index']}, marking as incorrect"
                        )
                        is_correct = False
                    else:
                        is_correct = annotation == 1

                eval_data = {
                    "label": int(annotation),
                    "is_correct": bool(is_correct),
                }

                results[i].setdefault("eval", {})[eval_name] = eval_data

                log.info(
                    f"Sample {result['index']} [{eval_name}]: "
                    f"{annotation} ({'Correct' if is_correct else 'Incorrect'})"
                )

                # Save after each sample evaluation
                save_results_json(results, save_path_file)
                samples_evaluated += 1

            except Exception as e:
                traceback.print_exc()
                log.error(
                    f"Error during {eval_name} verification for sample {result['index']}: {e}"
                )
                results[i].setdefault("eval", {})[eval_name] = {
                    "label": None,
                    "is_correct": False,
                }
                # Save even after errors
                save_results_json(results, save_path_file)

        if samples_evaluated == 0:
            log.info(f"All samples already evaluated by {eval_name}, skipping")
        else:
            log.info(
                f"Completed evaluation with {eval_name} ({samples_evaluated} samples evaluated)"
            )

    # Batch evaluation for batch_evaluators (more efficient)
    for eval_name, evaluator_fn in batch_evaluators.items():
        log.info(f"\n--- Batch Evaluator: {eval_name} ---")

        # Collect samples that need evaluation
        samples_to_eval = []
        indices_to_eval = []
        for i, result in enumerate(results):
            if "error" in result:
                continue
            if "eval" in result and eval_name in result["eval"]:
                log.info(
                    f"Skipping sample {result['index']} (already evaluated by {eval_name})"
                )
                continue
            samples_to_eval.append(result)
            indices_to_eval.append(i)

        if not samples_to_eval:
            log.info(f"All samples already evaluated by {eval_name}, skipping")
            continue

        log.info(f"Batch evaluating {len(samples_to_eval)} samples with {eval_name}...")

        # Prepare batch inputs
        problems = [r["question"] for r in samples_to_eval]
        gold_answers = [str(r["gold_answer"]) for r in samples_to_eval]

        # For answer_only mode, use extracted answer; otherwise use full solution
        if hasattr(evaluator_fn, "mode") and evaluator_fn.mode == "answer_only":
            # For LLM judges in answer_only mode: use extracted_answer, or mark as incorrect if empty
            # Do NOT fall back to trajectory - empty answers should be marked incorrect
            solutions = []
            for r in samples_to_eval:
                extracted = r.get("extracted_answer", "")
                if extracted and (not isinstance(extracted, str) or extracted.strip()):
                    solutions.append(extracted)
                else:
                    # Empty answer - will be marked as incorrect below
                    solutions.append("")
        else:
            solutions = [
                r.get("generated_trajectory", r.get("trajectory", ""))
                for r in samples_to_eval
            ]

        # Prepare optional parameters for evaluators that need them (e.g., EvalPlus)
        task_ids = None
        instance_data_list = None
        if isinstance(evaluator_fn, (EvaluatorMBPPPlus, EvaluatorHumanEvalPlus)):
            # Get task_ids from instance_data or direct field
            task_ids = []
            for r in samples_to_eval:
                inst_data = r.get("instance_data", {})
                task_id = inst_data.get("task_id") or r.get("task_id", r["index"])
                task_ids.append(task_id)
            # Get instance_data for each sample
            instance_data_list = [r.get("instance_data", {}) for r in samples_to_eval]

        try:
            # For answer_only mode: handle empty answers without calling evaluator
            if hasattr(evaluator_fn, "mode") and evaluator_fn.mode == "answer_only":
                # Check for empty answers and mark them as incorrect upfront
                empty_answer_indices = [
                    idx for idx, sol in enumerate(solutions) if not sol
                ]
                valid_indices = [idx for idx, sol in enumerate(solutions) if sol]

                if empty_answer_indices:
                    # Mark empty answers as incorrect
                    for idx in empty_answer_indices:
                        i = indices_to_eval[idx]
                        results[i].setdefault("eval", {})[eval_name] = {
                            "label": 0,
                            "is_correct": False,
                            "response": "No answer generated",
                        }

                if not valid_indices:
                    # All answers are empty, skip evaluator call
                    log.info(
                        f"All samples have empty answers, skipping {eval_name} evaluation"
                    )
                    save_results_json(results, save_path_file)
                    continue

                # Filter to only evaluate samples with valid answers
                indices_to_eval = [indices_to_eval[idx] for idx in valid_indices]
                problems = [problems[idx] for idx in valid_indices]
                solutions = [solutions[idx] for idx in valid_indices]
                gold_answers = [gold_answers[idx] for idx in valid_indices]
                # Also filter task_ids and instance_data_list if they exist
                if task_ids is not None:
                    task_ids = [task_ids[idx] for idx in valid_indices]
                if instance_data_list is not None:
                    instance_data_list = [
                        instance_data_list[idx] for idx in valid_indices
                    ]

            # Batch evaluate - pass additional params if evaluator needs them
            if isinstance(evaluator_fn, (EvaluatorMBPPPlus, EvaluatorHumanEvalPlus)):
                eval_result = evaluator_fn(
                    problems, solutions, gold_answers, task_ids, instance_data_list
                )
            else:
                eval_result = evaluator_fn(problems, solutions, gold_answers)
            if isinstance(eval_result, tuple) and len(eval_result) == 3:
                annotations, responses, consensus_scores = eval_result
            elif isinstance(eval_result, tuple) and len(eval_result) == 2:
                annotations, responses = eval_result
                consensus_scores = [None] * len(annotations)
            else:
                annotations = eval_result
                responses = [None] * len(annotations)
                consensus_scores = [None] * len(annotations)

            # Store results
            for idx, (i, annotation, response, consensus) in enumerate(
                zip(indices_to_eval, annotations, responses, consensus_scores)
            ):
                is_correct = annotation == 1
                eval_data = {
                    "label": int(annotation) if not np.isnan(annotation) else None,
                    "is_correct": bool(is_correct),
                }
                if consensus is not None:
                    eval_data["consensus"] = consensus
                if response:
                    eval_data["response"] = response
                results[i].setdefault("eval", {})[eval_name] = eval_data

            # Save after batch
            save_results_json(results, save_path_file)
            log.info(
                f"Completed batch evaluation with {eval_name} ({len(samples_to_eval)} samples)"
            )

        except Exception as e:
            traceback.print_exc()
            log.error(f"Error during batch {eval_name} evaluation: {e}")
            # Mark all as failed
            for i in indices_to_eval:
                results[i].setdefault("eval", {})[eval_name] = {
                    "label": None,
                    "is_correct": False,
                    "error": str(e),
                }
            save_results_json(results, save_path_file)

    # Collect statistics from LLM judge evaluators
    llm_judge_stats = {}
    for eval_name, evaluator_fn in batch_evaluators.items():
        if isinstance(evaluator_fn, EvaluatorLLMAsAJudge):
            stats = evaluator_fn.get_stats()
            # Prefix with evaluator name for disambiguation
            for key, value in stats.items():
                llm_judge_stats[f"{eval_name}/{key}"] = value

    # Combine all evaluator names for summary
    all_evaluator_names = list(evaluators.keys()) + list(batch_evaluators.keys())

    completed = sum(r.get("completed", False) for r in results)
    errors = sum("error" in r for r in results)

    # Compute per-evaluator correctness
    summary_correct = {name: 0 for name in all_evaluator_names}
    summary_incorrect = {name: 0 for name in all_evaluator_names}

    for r in results:
        for name in all_evaluator_names:
            # Check if this result has been evaluated by this evaluator
            if "eval" in r and name in r["eval"]:
                if r["eval"][name].get("is_correct"):
                    summary_correct[name] += 1
                else:
                    summary_incorrect[name] += 1

    log.info("Summary:")
    log.info(f"Total samples: {len(results)}")
    if results:
        log.info(f"Completed: {completed} ({completed/len(results):.1%})")
        log.info(f"Errors: {errors} ({errors/len(results):.1%})")
        for name in sorted(all_evaluator_names):
            correct = summary_correct[name]
            incorrect = summary_incorrect[name]
            log.info(f"[{name}]")
            log.info(f"Correct: {correct} ({correct/len(results):.1%})")
            log.info(f"Incorrect: {incorrect} ({incorrect/len(results):.1%})")
    else:
        log.info("Completed: 0 (0.0%)")
        log.info("Errors: 0 (0.0%)")
        for name in sorted(all_evaluator_names):
            correct = summary_correct[name]
            incorrect = summary_incorrect[name]
            log.info(f"[{name}]")
            log.info(f"Correct: {correct} (0.0%)")
            log.info(f"Incorrect: {incorrect} (0.0%)")

    # Average statistics
    all_validities = []
    all_reasoning_steps = []
    for r in results:
        if "validity_scores" in r and r["validity_scores"]:
            valid = [s for s in r["validity_scores"] if s is not None]
            if valid:
                all_validities.extend(valid)
                all_reasoning_steps.append(r.get("reasoning_steps", len(r["steps"])))

    # Token / FLOPs aggregates
    missing_stats_count = sum(1 for r in results if r.get("token_stats") is None)
    if missing_stats_count > 0:
        log.warning(
            f"{missing_stats_count}/{len(results)} results missing 'token_stats'"
        )
    all_token_stats = [r.get("token_stats") or {} for r in results]
    total_tokens = sum(ts.get("total_tokens_this_sample", 0) for ts in all_token_stats)
    total_input_tokens = sum(ts.get("input_tokens", 0) for ts in all_token_stats)
    total_output_tokens = sum(ts.get("output_tokens", 0) for ts in all_token_stats)
    total_generations = sum(ts.get("generation_count", 0) for ts in all_token_stats)
    total_tflops = sum(_safe_tflops(ts, "tflops") for ts in all_token_stats)

    # PRM token/FLOP aggregates
    total_prm_tokens = sum(ts.get("prm_input_tokens", 0) for ts in all_token_stats)
    total_prm_tflops = sum(_safe_tflops(ts, "prm_tflops") for ts in all_token_stats)

    log.info("Compute:")
    log.info(f"Total tokens: {total_tokens:,}")
    log.info(f"Total input tokens: {total_input_tokens:,}")
    log.info(f"Total output tokens: {total_output_tokens:,}")
    log.info(f"Total TFLOPs: {total_tflops:.2f}")
    if total_prm_tokens > 0:
        log.info(f"Total PRM input tokens: {total_prm_tokens:,}")
        log.info(f"Total PRM TFLOPs: {total_prm_tflops:.2f}")
    if results:
        log.info(f"Avg tokens per sample: {total_tokens / len(results):,.0f}")
        log.info(
            f"Avg output tokens per sample: {total_output_tokens / len(results):,.0f}"
        )
        log.info(f"Avg TFLOPs per sample: {total_tflops / len(results):.4f}")
    log.info("Step Statistics:")
    if all_reasoning_steps:
        log.info(
            f"Avg reasoning steps per trajectory: {np.mean(all_reasoning_steps):.1f}"
        )
    if all_validities:
        log.info(f"Avg validity score: {np.mean(all_validities):.3f}")

    # Build final metrics (also saved locally)
    metrics = {
        "total_samples": len(results),
        "completed": completed,
        "completed_pct": completed / len(results) if results else 0.0,
        "errors": errors,
        "errors_pct": errors / len(results) if results else 0.0,
    }

    # Add per-evaluator metrics
    for name in all_evaluator_names:
        correct = summary_correct[name]
        incorrect = summary_incorrect[name]
        metrics[f"{name}/correct"] = correct
        metrics[f"{name}/correct_pct"] = correct / len(results) if results else 0.0
        metrics[f"{name}/incorrect"] = incorrect
        metrics[f"{name}/incorrect_pct"] = incorrect / len(results) if results else 0.0
        metrics[f"{name}/accuracy"] = correct / len(results) if results else 0.0

    # Add step statistics
    if all_reasoning_steps:
        metrics["avg_reasoning_steps_per_trajectory"] = float(
            np.mean(all_reasoning_steps)
        )
    if all_validities:
        metrics["avg_validity_score"] = float(np.mean(all_validities))

    # Add token / FLOPs aggregates (computed above)
    metrics["compute/total_tokens"] = int(total_tokens)
    metrics["compute/total_input_tokens"] = int(total_input_tokens)
    metrics["compute/total_output_tokens"] = int(total_output_tokens)
    metrics["compute/total_generations"] = int(total_generations)
    metrics["compute/total_tflops"] = float(total_tflops)
    metrics["compute/avg_tokens_per_sample"] = (
        float(total_tokens / len(results)) if results else 0.0
    )
    metrics["compute/avg_output_tokens_per_sample"] = (
        float(total_output_tokens / len(results)) if results else 0.0
    )
    metrics["compute/avg_tflops_per_sample"] = (
        float(total_tflops / len(results)) if results else 0.0
    )
    if total_prm_tokens > 0:
        metrics["compute/prm_input_tokens"] = int(total_prm_tokens)
        metrics["compute/prm_tflops"] = float(total_prm_tflops)

    # Beam search token breakdown and termination diagnostics
    all_trajectory_tokens = [
        r["trajectory_tokens"] for r in results if "trajectory_tokens" in r
    ]
    all_answer_tokens = [r["answer_tokens"] for r in results if "answer_tokens" in r]
    context_limit_hits = sum(1 for r in results if r.get("context_limit_hit", False))
    max_steps_hits = sum(1 for r in results if r.get("max_steps_hit", False))
    no_answer_count = sum(1 for r in results if not r.get("extracted_answer"))
    if all_trajectory_tokens:
        metrics["compute/avg_trajectory_tokens"] = float(np.mean(all_trajectory_tokens))
        metrics["compute/avg_answer_tokens"] = float(np.mean(all_answer_tokens))
        metrics["context_limit_hit_count"] = context_limit_hits
        metrics["context_limit_hit_rate"] = (
            context_limit_hits / len(results) if results else 0.0
        )
        metrics["max_steps_hit_count"] = max_steps_hits
        metrics["max_steps_hit_rate"] = (
            max_steps_hits / len(results) if results else 0.0
        )
        log.info(
            f"Token breakdown: avg trajectory={np.mean(all_trajectory_tokens):.0f}, "
            f"avg answer={np.mean(all_answer_tokens):.0f}"
        )
        log.info(
            f"Context limit hits: {context_limit_hits}/{len(results)} "
            f"({context_limit_hits / len(results):.1%})"
        )
        log.info(
            f"Max steps hits: {max_steps_hits}/{len(results)} "
            f"({max_steps_hits / len(results):.1%})"
        )
    # No-answer rate (applies to all strategies)
    if results:
        metrics["no_answer_count"] = no_answer_count
        metrics["no_answer_rate"] = no_answer_count / len(results)
        if no_answer_count > 0:
            log.warning(
                f"No answer extracted: {no_answer_count}/{len(results)} "
                f"({no_answer_count / len(results):.1%})"
            )

    # Add LLM judge statistics (if available)
    if llm_judge_stats:
        metrics.update(llm_judge_stats)
        # Log LLM judge stats to console
        log.info("LLM Judge Statistics:")
        for key, value in llm_judge_stats.items():
            if isinstance(value, float):
                log.info(f"  {key}: {value:.4f}")
            else:
                log.info(f"  {key}: {value}")

    # Compute additional LLM judge stats from stored results
    for eval_name in batch_evaluators.keys():
        if isinstance(batch_evaluators[eval_name], EvaluatorLLMAsAJudge):
            # Count unclear samples (where all votes failed to parse)
            unclear_count = 0
            consensus_scores = []
            for r in results:
                if "eval" in r and eval_name in r["eval"]:
                    eval_data = r["eval"][eval_name]
                    if eval_data.get("label") is None:  # NaN/None = unclear
                        unclear_count += 1
                    if "consensus" in eval_data and eval_data["consensus"] is not None:
                        consensus_scores.append(eval_data["consensus"])

            if unclear_count > 0:
                metrics[f"{eval_name}/unclear_count"] = unclear_count
                metrics[f"{eval_name}/unclear_rate"] = (
                    unclear_count / len(results) if results else 0.0
                )

            if consensus_scores:
                metrics[f"{eval_name}/avg_consensus"] = float(np.mean(consensus_scores))
                metrics[f"{eval_name}/min_consensus"] = float(np.min(consensus_scores))

    # Record symbolic comparison timeouts
    symbolic_timeouts = get_timeout_count()
    metrics["eval/symbolic_equal_timeouts"] = symbolic_timeouts
    if symbolic_timeouts > 0:
        log.warning(
            "Total symbolic_equal timeouts during this run: %d", symbolic_timeouts
        )

    # Save metrics locally (so FLOPs metrics aren't only in W&B)
    metrics_path = Path(save_path) / "metrics.json"
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, sort_keys=True)
        log.info(f"Saved metrics to {metrics_path}")
    except Exception as e:
        log.warning(f"Failed to save metrics to {metrics_path}: {e}")

    # Log evaluation inconsistencies between exact_match and llm_judge
    log_evaluation_inconsistencies(results, save_path)

    # Log key metrics to wandb if enabled
    wandb_url = None
    wandb_group_url = None
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics)
            log.info("Logged metrics to wandb")
            wandb_url = wandb.run.get_url()
            group = wandb.run.group
            if group:
                entity = wandb.run.entity
                project_name = wandb.run.project
                wandb_group_url = (
                    f"https://wandb.ai/{entity}/{project_name}/groups/{group}/workspace"
                )

            # Log all output files as artifacts
            save_path_obj = Path(save_path)
            if save_path_obj.exists():
                # Log log files
                for log_file in ["run_tts_eval.log", "stderr.log"]:
                    log_path = save_path_obj / log_file
                    if log_path.exists():
                        wandb.save(str(log_path), base_path=str(save_path_obj))
                        log.info(f"Logged {log_file} to wandb")

                # Log all JSON files
                for json_file in save_path_obj.glob("*.json"):
                    wandb.save(str(json_file), base_path=str(save_path_obj))
                    log.info(f"Logged {json_file.name} to wandb")

                # Log all JSONL files
                for jsonl_file in save_path_obj.glob("*.jsonl"):
                    wandb.save(str(jsonl_file), base_path=str(save_path_obj))
                    log.info(f"Logged {jsonl_file.name} to wandb")
    except ImportError:
        pass  # wandb not installed
    except Exception as e:
        log.warning(f"Failed to log metrics to wandb: {e}")

    # Print tab-separated summary for spreadsheet copy-paste
    em_acc = None
    llm_judge_acc = None
    for name in all_evaluator_names:
        acc = metrics.get(f"{name}/accuracy")
        if name == "exact_match":
            em_acc = acc
        elif name.startswith("llm_judge_"):
            llm_judge_acc = acc

    spreadsheet_parts = [
        wandb_url or "",
        wandb_group_url or "",
        f"{em_acc:.3f}" if em_acc is not None else "",
        f"{llm_judge_acc:.3f}" if llm_judge_acc is not None else "",
        f"{total_tflops:.0f}",
    ]
    log.info("=" * 60)
    log.info(
        "SPREADSHEET (wandb_url | wandb_group_url | exact_match | llm_judge | tflops):"
    )
    log.info("\t".join(spreadsheet_parts))
    log.info("=" * 60)


@hydra.main(
    version_base=None,
    config_path=None,
    config_name=None,
)
def main(config):
    """Main evaluation function"""
    stderr_file = None  # Initialize for cleanup

    import socket

    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    machine_name = os.environ.get("MACHINE_NAME", hostname)
    log.info(f"Host: {machine_name} ({ip_addr})")

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    log.info(f"Command: CUDA_VISIBLE_DEVICES={cuda_devices} {' '.join(sys.argv)}")
    config_dir = [
        path["path"]
        for path in HydraConfig.get().runtime.config_sources
        if path["schema"] == "file"
    ][0]
    config_file = Path(config_dir) / f"{HydraConfig.get().job.config_name}.yaml"
    log.info(f"Config: {config_file}")
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")

    # Redirect stderr to file in output directory (captures tqdm progress bars)
    stderr_log_path = Path(output_dir) / "stderr.log"
    stderr_file = open(stderr_log_path, "w", buffering=1)  # Line buffered
    sys.stderr = stderr_file
    log.info(f"Stderr redirected to: {stderr_log_path}")

    # Setup wandb if configured
    if getattr(config, "report_to", None) == "wandb":
        import wandb

        wandb_cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        config_path_hydra = [
            path["path"]
            for path in HydraConfig.get().runtime.config_sources
            if path["schema"] == "file"
        ][0]
        wandb_cfg["HYDRA_CONFIG"] = (
            Path(config_path_hydra) / HydraConfig.get().job.config_name
        )
        os.environ["WANDB_DIR"] = str(Path(output_dir))
        # Project name: config > env var > default
        project = getattr(config, "wandb_project", None) or os.environ.get(
            "WANDB_PROJECT", "llm-tts-eval"
        )
        run_name = config.get("run_name", None)
        wandb_group = getattr(config, "wandb_group", None)

        # Prepend date to wandb run name to match directory structure
        if run_name:
            date_str = datetime.now().strftime("%Y-%m-%d")
            wandb_run_name = f"{date_str}_{run_name}"
        else:
            wandb_run_name = None

        wandb.init(
            project=project,
            name=wandb_run_name,
            group=wandb_group,
            dir=output_dir,
            config=wandb_cfg,
        )
        log.info(f"WandB run URL: {wandb.run.get_url()}")
        if wandb_group:
            entity = wandb.run.entity
            group_url = (
                f"https://wandb.ai/{entity}/{project}/groups/{wandb_group}/workspace"
            )
            log.info(f"WandB group URL: {group_url}")
        wandb_save_directory(Path(output_dir) / ".hydra")

    # Validate API keys early (before spending hours on model loading / generation)
    _validate_api_keys(config)

    # Set random seeds
    set_random_seeds(config.system.seed)

    # Load dataset
    log.info(
        f"Loading dataset: {config.dataset.dataset_path} ({config.dataset.dataset_split})"
    )
    # Special handling for EvalPlus datasets to use EvalPlus API (provides correct prompt format)
    data_name = config.dataset.get("data_name", "")
    if data_name == "mbpp_plus" or "mbppplus" in config.dataset.dataset_path.lower():
        from llm_tts.datasets.mbpp_plus import load_mbpp_plus

        log.info(
            "Using EvalPlus API for MBPP+ (provides correct prompt format with function name)"
        )
        mbpp_data = load_mbpp_plus(subset_size=None)  # Load all, subset later
        # Convert to HuggingFace Dataset format
        # Remove fields that can't be serialized by Arrow (nested tuples)
        serializable_data = []
        for item in mbpp_data:
            serializable_item = {
                "question": item["question"],
                "answer": item["answer"],
                "task_id": item["task_id"],
                "entry_point": item["entry_point"],
                "test_list": item["test_list"],
                "assertion": item.get("assertion", ""),
            }
            serializable_data.append(serializable_item)
        dataset = Dataset.from_list(serializable_data)
    elif (
        data_name == "human_eval_plus"
        or "humanevalplus" in config.dataset.dataset_path.lower()
    ):
        from llm_tts.datasets.human_eval_plus import load_human_eval_plus

        log.info(
            "Using EvalPlus API for HumanEval+ (provides correct prompt format with function signature)"
        )
        he_data = load_human_eval_plus(subset_size=None)  # Load all, subset later
        serializable_data = []
        for item in he_data:
            serializable_item = {
                "question": item["question"],
                "answer": item["answer"],
                "task_id": item["task_id"],
                "entry_point": item["entry_point"],
            }
            serializable_data.append(serializable_item)
        dataset = Dataset.from_list(serializable_data)
    # Support loading local JSON/JSONL files via data_files parameter
    elif config.dataset.get("data_files", None):
        data_files = config.dataset.get("data_files", None)
        log.info(f"Loading from local file: {data_files}")
        dataset = load_dataset(
            config.dataset.dataset_path,
            data_files=data_files,
            split=config.dataset.dataset_split,
            cache_dir=config.system.hf_cache,
        )
    else:
        dataset = load_dataset(
            config.dataset.dataset_path,
            config.dataset.get("dataset_config", None),
            split=config.dataset.dataset_split,
            cache_dir=config.system.hf_cache,
        )
    # Apply offset and subset
    offset = config.dataset.get("offset", 0)
    subset = config.dataset.get("subset", None)
    if offset > 0 or subset:
        start_idx = offset
        end_idx = len(dataset)
        if subset:
            end_idx = min(start_idx + subset, len(dataset))
        dataset = dataset.select(range(start_idx, end_idx))
        log.info(
            f"Dataset: using samples {start_idx} to {end_idx-1} ({len(dataset)} samples)"
        )

    prompt_template = (
        load_prompt_template(config.dataset.prompt_file)
        if config.dataset.prompt_file
        else ""
    )

    # Load system prompt if configured
    system_prompt = getattr(config.dataset, "system_prompt", "") or ""

    # Load model
    model_name = config.model.get("model_name") or config.model.get("model_path")
    log.info(f"Loading model: {model_name}")
    try:
        model, step_generator = create_model(config)
        log.info("Model loaded successfully")
    except Exception as e:
        log.exception(f"Model loading failed: {e}")
        raise

    # Create FLOP calculator for compute cost estimation
    flop_calculator = None
    if model_name:
        try:
            flop_calculator = FLOPCalculator(model_name, method="simple")
            log.info(
                f"FLOP calculator initialized: {flop_calculator.tflops_per_1k_tokens:.3f} TFLOPs/1k tokens"
            )
        except Exception as e:
            log.warning(f"Failed to initialize FLOP calculator: {e}")

    # Set FLOP calculator on step generator for token/FLOP tracking
    if step_generator is not None and flop_calculator is not None:
        step_generator.flop_calculator = flop_calculator
        log.info("FLOP calculator attached to step generator for token tracking")

    # Create scorer (skip for DeepConf)
    scorer = create_scorer(config)

    # Create tts strategy
    generator = create_tts_strategy(
        config=config,
        model=model,
        step_generator=step_generator,
        scorer=scorer,
        output_dir=output_dir,
        flop_calculator=flop_calculator,
    )

    # Load existing results if available (for resuming interrupted runs)
    results_path = Path(output_dir) / "results.json"
    results, processed_indices = load_results_json(results_path)

    # NOTE: Don't shuffle - keep original dataset order for reproducibility
    # dataset = dataset.shuffle(seed=config.system.seed)

    # Generate trajectories
    # Get data_name for official extraction (from dataset or strategy config)
    data_name = config.dataset.get("data_name", None) or config.strategy.get(
        "data_name", None
    )
    if not data_name:
        raise ValueError("data_name must be set in config.dataset or config.strategy")

    # Generate trajectories with error logging
    log.info("Starting trajectory generation...")
    try:
        results = generate_trajectories(
            results=results,
            save_path=output_dir,
            strategy=generator,
            dataset=dataset,
            processed_indices=processed_indices,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            question_field=config.dataset.get("question_field", "question"),
            answer_field=config.dataset.get("answer_field", "answer"),
            exact_match_dataset_answer_format=config.dataset.answer_format,
            data_name=data_name,
            config=config,  # Pass config for multi-evaluator support
        )
        log.info("Trajectory generation completed successfully")

    except Exception as e:
        log.exception(f"Trajectory generation failed: {e}")

        # Save partial results before crashing
        try:
            partial_results_path = Path(output_dir) / "results_partial.json"
            with open(partial_results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            log.info(f"Saved partial results to {partial_results_path}")
        except Exception as save_err:
            log.error(f"Failed to save partial results: {save_err}")
        raise

    # Free GPU memory before evaluation (model not needed for LLM judge API calls)
    log.info("Freeing GPU memory before evaluation phase...")
    try:
        if hasattr(model, "shutdown"):
            model.shutdown()
        if hasattr(generator, "cleanup"):
            generator.cleanup()
        # Delete vLLM engine and model to release GPU memory
        if hasattr(model, "vllm_engine"):
            del model.vllm_engine
        del model
        del step_generator
        del generator
        if scorer is not None:
            del scorer
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("GPU memory freed successfully")
    except Exception as e:
        log.warning(f"Failed to free GPU memory: {e}")

    # Evaluate results
    log.info("Starting evaluation phase...")
    try:
        evaluate_results(
            config=config,
            results=results,
            save_path=output_dir,
        )
        log.info("Evaluation completed successfully")
    except Exception as e:
        log.exception(f"Evaluation failed: {e}")
        raise

    # Save log files and finish wandb session
    if getattr(config, "report_to", None) == "wandb":
        try:
            import wandb

            if wandb.run is not None:
                log.info("Saving output files to wandb...")
                # Save all JSON, JSONL, and log files
                for ext in ["*.json", "*.jsonl", "*.log"]:
                    for output_file in Path(output_dir).glob(ext):
                        try:
                            wandb.save(str(output_file))
                            log.debug(f"Saved {output_file.name} to wandb")
                        except Exception as save_err:
                            log.warning(
                                f"Failed to save {output_file.name}: {save_err}"
                            )
            wandb.finish()
            log.info("Finished wandb session")
        except Exception as e:
            log.warning(f"Failed to finish wandb session: {e}")

    # Close stderr redirect file
    if stderr_file:
        sys.stderr = sys.__stderr__  # Restore original stderr
        stderr_file.close()


if __name__ == "__main__":
    # Parse custom resume arguments before Hydra processes sys.argv
    parse_resume_arguments()

    # Store variables for cleanup on error
    output_dir = None
    stderr_file = None
    config = None

    try:
        main()
    except (KeyboardInterrupt, MemoryError, Exception) as e:
        import logging

        error_log = logging.getLogger(__name__)

        # Try to get output_dir from HydraConfig if main() didn't set it
        if output_dir is None:
            try:
                from hydra.core.hydra_config import HydraConfig

                output_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                error_log.warning("Could not get output_dir from HydraConfig")

        # Log the error
        if isinstance(e, KeyboardInterrupt):
            error_log.info("Job interrupted by user (KeyboardInterrupt)")
        elif isinstance(e, MemoryError):
            error_log.error(f"Job failed due to MemoryError (GPU/CPU OOM): {e}")
        else:
            error_log.exception(f"Job failed with exception: {e}")

        # Save logs to wandb even on error
        if output_dir:
            try:
                from pathlib import Path

                import wandb

                # Sync all output files to wandb before crashing
                if wandb.run is not None:
                    error_log.info("Saving log files to wandb after error...")

                    # Save all JSON, JSONL, and log files
                    for ext in ["*.json", "*.jsonl", "*.log"]:
                        for log_file in Path(output_dir).glob(ext):
                            try:
                                wandb.save(str(log_file))
                                error_log.info(f"Saved {log_file.name} to wandb")
                            except Exception as save_err:
                                error_log.warning(
                                    f"Failed to save {log_file.name}: {save_err}"
                                )

                    # Finish wandb run with error status
                    wandb.finish(exit_code=1)
                    error_log.info("Wandb session finished (with error)")
            except Exception as wandb_err:
                error_log.warning(f"Failed to save logs to wandb on error: {wandb_err}")

        # Close stderr redirect file
        if stderr_file:
            sys.stderr = sys.__stderr__
            stderr_file.close()

        raise
