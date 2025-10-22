#!/usr/bin/env python3

import json
import logging
import os
import random
import traceback
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from lm_polygraph import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_tts.evaluation import (
    EvaluatorAlignScore,
    EvaluatorExactMatch,
    EvaluatorLLMAsAJudge,
)
from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.scorers import StepScorerPRM, StepScorerUncertainty
from llm_tts.step_boundary_detector import StepBoundaryDetector
from llm_tts.step_candidate_generator_base import StepCandidate
from llm_tts.step_candidate_generator_through_api import (
    StepCandidateGeneratorThroughAPI,
)
from llm_tts.step_candidate_generator_through_huggingface import (
    StepCandidateGeneratorThroughHuggingface,
)
from llm_tts.strategies import (
    StrategyBeamSearch,
    StrategyDeepConf,
    StrategyOnlineBestOfN,
    StrategyUncertaintyCoT,
)

# Load environment variables from .env file
load_dotenv()

log = logging.getLogger(__name__)


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.chat_template = None
    # tokenizer.padding_side = "left"  # Fix padding side for decoder-only models
    return tokenizer


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, trust_remote_code=True
    )
    return model


def load_prompt_template(prompt_file: str) -> str:
    """Load prompt template from file"""
    with open(prompt_file, "r") as f:
        return f.read().strip()


def build_evaluators(config):
    """
    Create evaluators from config.
    The list of evaluator names in config.evaluation.evaluators
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

            evaluators["llm_judge"] = EvaluatorLLMAsAJudge(**llm_cfg)

        elif evaluator_name == "exact_match":
            evaluators["exact_match"] = EvaluatorExactMatch()

        elif evaluator_name == "alignscore":
            align_cfg = OmegaConf.to_container(
                config.evaluation.alignscore, resolve=True
            )
            evaluators["alignscore"] = EvaluatorAlignScore(**align_cfg)

        else:
            log.warning(f"Unknown evaluator type '{evaluator_name}', skipping")

    return evaluators


def _safe_serialize(obj):
    """Convert arbitrary Python objects (including tensors, numpy) to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, StepCandidate):
        return {
            "text": obj.text,
            "token_ids": list(obj.token_ids) if obj.token_ids is not None else None,
            "is_complete": obj.is_complete,
            "is_trajectory_complete": obj.is_trajectory_complete,
            "generation_scores": _safe_serialize(obj.generation_scores),
            "raw_text": obj.raw_text,
            "other_data": _safe_serialize(obj.other_data) if obj.other_data else None,
        }
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return _safe_serialize(vars(obj))

    # Fallback to string representation
    return str(obj)


def _save_results_json(results, json_path: Path):
    json_path = Path(json_path)
    json_data = [_safe_serialize(r) for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


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
    # DeepConf doesn't use a scorer

    if config.strategy.type == "deepconf":
        return None
    if config.scorer.type == "uncertainty_pd":
        return None
    if config.scorer is None:
        return None

    if config.scorer.type == "prm":
        scorer = StepScorerPRM(
            prm_model_path=config.scorer.model_path,
            device=config.scorer.device,
            batch_size=config.scorer.batch_size,
        )

    elif config.scorer.type == "uncertainty":
        scorer = StepScorerUncertainty()

    else:
        raise ValueError(f"Scorer type {config.scorer.type} not supported")

    return scorer


def create_model(config):
    if config.model.type == "local":
        if (
            config.scorer.type == "uncertainty"
            or config.scorer.type == "uncertainty_pd"
        ):
            log.info(
                f"Loading uncertainty model: {config.scorer.uncertainty_model_creator}"
            )

            import importlib

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
            base_model = load_model(config.model.model_path, config.system.device)
            base_model.eval()
            model = WhiteboxModel(base_model, tokenizer)

        detector = StepBoundaryDetector(
            step_patterns=config.strategy.get(
                "detector_step_patterns", ["- Step", "<Answer>:", "\n<Answer>:"]
            ),
            answer_patterns=config.strategy.get(
                "detector_answer_patterns", ["<Answer>:", "\n<Answer>:"]
            ),
            max_tokens_per_step=config.generation.max_new_tokens,
        )
        step_generator = StepCandidateGeneratorThroughHuggingface(
            model=model,
            detector=detector,
            temperature=config.generation.temperature,
            max_new_tokens=config.generation.max_new_tokens,
            top_p=config.generation.top_p,
            top_k=config.generation.top_k,
            disable_thinking_mode=config.model.disable_thinking_mode,
            generation_batch_size=config.generation.batch_size,
        )

    elif config.model.type == "openai_api":
        # Use model_name if available, otherwise fall back to model_path
        model_path = config.model.get("model_name") or config.model.get("model_path")
        log.info(f"Using OpenAI API model: {model_path}")

        # Check if DeepConf strategy
        if config.strategy.type == "deepconf":
            # DeepConf uses streaming with logprobs but no boundary detector
            import os

            # Check provider for API key and base URL
            if config.model.get("provider") == "openrouter":
                api_key = config.model.get("api_key") or os.getenv("OPENROUTER_API_KEY")
                base_url = "https://openrouter.ai/api/v1"
            else:
                api_key = config.model.get("api_key") or os.getenv("OPENAI_API_KEY")
                base_url = None

            model = BlackboxModelWithStreaming(
                openai_api_key=api_key,
                model_path=model_path,
                supports_logprobs=True,
                base_url=base_url,
            )
            step_generator = None  # DeepConf doesn't use step generator
        else:
            # Other strategies use boundary detection via early stopping
            from llm_tts.early_stopping import BoundaryEarlyStopping

            detector = StepBoundaryDetector(
                step_patterns=["- Step", "<Answer>:", "\n<Answer>:"],
                answer_patterns=["<Answer>:", "\n<Answer>:"],
                max_tokens_per_step=config.generation.max_new_tokens,
            )

            generation_parameters = GenerationParameters()
            generation_parameters.temperature = config.generation.temperature
            generation_parameters.max_new_tokens = config.generation.max_new_tokens
            generation_parameters.top_p = config.generation.top_p
            generation_parameters.top_k = config.generation.top_k

            # Create boundary-based early stopping
            early_stopping = BoundaryEarlyStopping(detector=detector)

            model = BlackboxModelWithStreaming(
                openai_api_key=config.model.api_key,
                model_path=model_path,
                supports_logprobs=config.model.supports_logprobs,
                early_stopping=early_stopping,
                generation_parameters=generation_parameters,
            )

            step_generator = StepCandidateGeneratorThroughAPI(
                model=model,
                detector=detector,
                prefill_mode=config.model.prefill_mode,
            )
    else:
        raise ValueError(f"Model type {config.model.type} not supported")

    return model, step_generator


def create_tts_strategy(config, model, step_generator, scorer):
    if config.strategy.type == "online_best_of_n":
        strategy = StrategyOnlineBestOfN(
            step_generator=step_generator,
            scorer=scorer,
            candidates_per_step=config.strategy.candidates_per_step,
            max_steps=config.strategy.max_steps,
        )
    elif config.strategy.type == "deepconf":
        # DeepConf requires BlackboxModel with logprobs support
        if not isinstance(model, BlackboxModelWithStreaming):
            raise ValueError(
                f"DeepConf requires BlackboxModelWithStreaming, got {type(model).__name__}"
            )

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
        )

    elif config.strategy.type == "beam_search":
        strategy = StrategyBeamSearch(
            step_generator=step_generator,
            scorer=scorer,
            beam_size=config.strategy.beam_size,
            candidates_per_beam=config.strategy.candidates_per_beam,
            max_steps=config.strategy.max_steps,
            aggregation=getattr(config.strategy, "aggregation", "mean"),
        )

    elif config.strategy.type == "uncertainty_cot":
        strategy = StrategyUncertaintyCoT(
            config=config,
            step_generator=step_generator,
            candidates_per_step=config.strategy.candidates_per_step,
            max_steps=config.strategy.max_steps,
            max_empty_steps=config.strategy.max_empty_steps,
            uncertainty_threshold=config.strategy.uncertainty_threshold,
        )
    else:
        raise ValueError(f"Strategy type {config.strategy.type} not supported")

    return strategy


def generate_trajectories(
    results,
    save_path,
    strategy: StrategyOnlineBestOfN,
    dataset: Dataset,
    processed_indices: set,
    prompt_template: str,
):
    # Phase 1: Generate trajectories (without checking correctness)
    log.info("\n" + "=" * 60)
    log.info("Phase 1: Generating trajectories")
    log.info("=" * 60)

    save_path_file = Path(save_path) / "results.json"

    subset_size = len(dataset)
    for i in tqdm(range(subset_size), desc="Generating trajectories"):
        # Skip if already processed
        if i in processed_indices:
            log.info(f"Skipping sample {i} (already processed)")
            continue

        instance = dataset[i]

        log.info("\n" + "=" * 60)
        log.info(f"Sample {i + 1}/{subset_size}")
        log.info(f"Question: {instance['question'][:200]}...")

        # Extract and log gold answer
        from llm_tts.datasets.gsm8k import extract_answer_from_gsm8k

        gold_answer_num = extract_answer_from_gsm8k(instance["answer"])
        log.info(f"Gold answer: {gold_answer_num}")

        # Generate trajectory
        request = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": (
                    prompt_template.format(question=instance["question"])
                    if prompt_template
                    else instance["question"]
                ),
            },
        ]

        result = strategy.generate_trajectory(request)

        # Extract generated answer (but don't check correctness yet)
        generated_text = result["trajectory"]
        if instance["question"] in generated_text:
            generated_text = generated_text.replace(instance["question"], "").strip()

        # Log detailed traces
        log.info("\n" + "-" * 60)
        log.info("GENERATED TRACES:")
        log.info("-" * 60)

        # For DeepConf, steps contain the individual traces
        if result["steps"] and isinstance(result["steps"], list):
            for step_idx, step in enumerate(result["steps"]):
                validity = (
                    result["validity_scores"][step_idx]
                    if step_idx < len(result["validity_scores"])
                    else "N/A"
                )
                # Format confidence score (handle both numeric and "N/A")
                confidence_str = (
                    f"{validity:.3f}"
                    if isinstance(validity, (int, float))
                    else validity
                )
                log.info(f"\nTrace {step_idx + 1} (confidence: {confidence_str}):")
                log.info(step)
        else:
            # Fallback: show full trajectory
            log.info(f"\nFull trajectory:\n{result['trajectory']}")

        log.info("\n" + "-" * 60)
        log.info("FINAL ANSWER:")
        log.info("-" * 60)
        log.info(f"Generated: {generated_text}")
        log.info(f"Num traces: {len(result['steps'])}")
        log.info(f"Avg confidence: {np.mean(result['validity_scores']):.3f}")
        log.info("-" * 60)

        # Store result WITHOUT correctness check
        result_dict = {
            "index": i,
            "question": instance["question"],
            "gold_answer": instance["answer"],
            "generated_trajectory": result["trajectory"],
            "generated_answer": generated_text,
            "steps": result["steps"],
            "validity_scores": result["validity_scores"],
            "completed": result["completed"],
        }

        # Include metadata if present (contains trace summaries and details)
        if "metadata" in result:
            result_dict["metadata"] = result["metadata"]

        results.append(result_dict)

        # Save periodically
        if len(results) % 10 == 0:
            _save_results_json(results, save_path_file)
            log.info(f"Saved {len(results)} results to {save_path_file}")

    # Final save after generation
    _save_results_json(results, save_path_file)
    log.info(f"Final save after generation: {len(results)} results to {save_path_file}")

    return results


def evaluate_results(
    config,
    results,
    save_path: str,
):
    # Phase 2: Check correctness for all results
    log.info("\n" + "=" * 60)
    log.info("Phase 2: Checking correctness")
    log.info("=" * 60)

    # Build evaluators dynamically
    evaluators = build_evaluators(config)
    log.info(f"Using evaluators: {list(evaluators.keys())}")

    # Prepare data for batch processing
    problems = []
    solutions = []
    gold_answers = []
    result_indices = []

    for i, result in enumerate(results):
        if "error" not in result:
            problems.append(result["question"])
            solutions.append(result["generated_answer"])
            gold_answers.append(result["gold_answer"])
            result_indices.append(i)

    if problems:
        log.info(
            f"Verifying {len(problems)} solutions with {list(evaluators.keys())}..."
        )

        for eval_name, evaluator_fn in evaluators.items():
            try:
                annotations = evaluator_fn(problems, solutions, gold_answers)
                for idx, annotation in zip(result_indices, annotations):
                    if np.isnan(annotation):
                        log.warning(
                            f"{eval_name} returned unclear result for sample "
                            f"{results[idx]['index']}, marking as incorrect"
                        )
                        is_correct = False
                    else:
                        is_correct = annotation == 1  # 1 = correct, 0 = incorrect

                    results[idx].setdefault("eval", {})[eval_name] = {
                        "label": None if np.isnan(annotation) else int(annotation),
                        "is_correct": bool(is_correct),
                    }

                    if (idx - result_indices[0]) % 10 == 0:
                        log.info(f"Sample {results[idx]['index']} [{eval_name}]:")
                        log.info(f"Annotation: {annotation}")
                        log.info(f"Correct: {is_correct}")

            except Exception as e:
                traceback.print_exc()
                log.error(f"Error during {eval_name} verification: {e}")
                for idx in result_indices:
                    results[idx].setdefault("eval", {})[eval_name] = {
                        "label": None,
                        "is_correct": False,
                    }

    # Final save with correctness results
    save_path_file = Path(save_path) / "results.json"
    _save_results_json(results, save_path_file)
    log.info(f"Final save with correctness: {len(results)} results to {save_path_file}")

    completed = sum(r.get("completed", False) for r in results)
    errors = sum("error" in r for r in results)

    # Compute per-evaluator correctness
    summary_correct = {name: 0 for name in evaluators.keys()}
    summary_incorrect = {name: 0 for name in evaluators.keys()}

    for r in results:
        for name in evaluators.keys():
            if r.get("eval").get(name).get("is_correct"):
                summary_correct[name] += 1
            elif not r.get("eval").get(name).get("is_correct"):
                summary_incorrect[name] += 1

    log.info("Summary:")
    log.info(f"Total samples: {len(results)}")
    log.info(f"Completed: {completed} ({completed/len(results):.1%})")
    log.info(f"Errors: {errors} ({errors/len(results):.1%})")
    for name in sorted(list(evaluators.keys())):
        correct = summary_correct[name]
        incorrect = summary_incorrect[name]
        log.info(f"[{name}]")
        log.info(f"Correct: {correct} ({correct/len(results):.1%})")
        log.info(f"Incorrect: {incorrect} ({incorrect/len(results):.1%})")

    # Average statistics
    all_validities = []
    all_steps = []
    for r in results:
        if "validity_scores" in r and r["validity_scores"]:
            all_validities.extend(r["validity_scores"])
            all_steps.append(len(r["steps"]))

    log.info("Step Statistics:")
    log.info(f"Avg steps per trajectory: {np.mean(all_steps):.1f}")
    log.info(f"Avg validity score: {np.mean(all_validities):.3f}")

    # Log key metrics to wandb if enabled
    try:
        import wandb

        if wandb.run is not None:
            metrics = {
                "total_samples": len(results),
                "completed": completed,
                "completed_pct": completed / len(results),
                "errors": errors,
                "errors_pct": errors / len(results),
            }

            # Add per-evaluator metrics
            for name in evaluators.keys():
                correct = summary_correct[name]
                incorrect = summary_incorrect[name]
                metrics[f"{name}/correct"] = correct
                metrics[f"{name}/correct_pct"] = correct / len(results)
                metrics[f"{name}/incorrect"] = incorrect
                metrics[f"{name}/incorrect_pct"] = incorrect / len(results)
                metrics[f"{name}/accuracy"] = correct / len(results)

            # Add step statistics
            if all_steps:
                metrics["avg_steps_per_trajectory"] = np.mean(all_steps)
            if all_validities:
                metrics["avg_validity_score"] = np.mean(all_validities)

            wandb.log(metrics)
            log.info("Logged metrics to wandb")
    except ImportError:
        pass  # wandb not installed
    except Exception as e:
        log.warning(f"Failed to log metrics to wandb: {e}")


@hydra.main(
    version_base=None,
    config_path=None,
    config_name=None,
)
def main(config):
    """Main evaluation function"""

    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")

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
        project = os.environ.get("WANDB_PROJECT", "llm-tts-eval")
        run_name = config.get("run_name", None)

        # Prepend date to wandb run name to match directory structure
        if run_name:
            date_str = datetime.now().strftime("%Y-%m-%d")
            wandb_run_name = f"{date_str}_{run_name}"
        else:
            wandb_run_name = None

        wandb.init(
            project=project, name=wandb_run_name, dir=output_dir, config=wandb_cfg
        )
        wandb_save_directory(Path(output_dir) / ".hydra")

    # Set random seeds
    set_random_seeds(config.system.seed)

    # Load dataset
    log.info(
        f"Loading dataset: {config.dataset.dataset_path} ({config.dataset.dataset_split})"
    )
    dataset = load_dataset(
        config.dataset.dataset_path,
        config.dataset.get("dataset_config", None),
        split=config.dataset.dataset_split,
        cache_dir=config.system.hf_cache,
    )
    if config.dataset.subset:
        dataset = dataset.select(range(min(config.dataset.subset, len(dataset))))

    prompt_template = (
        load_prompt_template(config.dataset.prompt_file)
        if config.dataset.prompt_file
        else ""
    )

    # Load model
    model_name = config.model.get("model_name") or config.model.get("model_path")
    log.info(f"Loading model: {model_name}")
    model, step_generator = create_model(config)

    # Create scorer (skip for DeepConf)
    scorer = create_scorer(config)

    # Create tts strategy
    generator = create_tts_strategy(
        config=config, model=model, step_generator=step_generator, scorer=scorer
    )

    processed_indices = set()
    results = []  # TODO: add logic for resuming from existing results
    # Generate trajectories
    results = generate_trajectories(
        results=results,
        save_path=output_dir,
        strategy=generator,
        dataset=dataset,
        processed_indices=processed_indices,
        prompt_template=prompt_template,
    )

    # Evaluate results
    evaluate_results(
        config=config,
        results=results,
        save_path=output_dir,
    )


if __name__ == "__main__":
    main()
