#!/usr/bin/env python3

import json
import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
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
from llm_tts.strategies import StrategyOnlineBestOfN

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

            llm_cfg.pop("prompt_file", None)
            llm_cfg["prompt"] = prompt_template

            evaluators["llm_judge"] = EvaluatorLLMAsAJudge(**llm_cfg)

        elif evaluator_name == "exact_match":
            evaluators["exact_match"] = EvaluatorExactMatch(
                dataset_name=config.dataset.get("dataset_path")
            )

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


def load_existing_results(save_path: str, dataset):
    log.info(f"Loading existing results from {save_path}")

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_indices = {r["index"] for r in results}
        log.info(f"Loaded {len(results)} existing results")
        log.info(f"Already processed indices: {sorted(processed_indices)}")

        # Validate all existing results match current dataset
        log.info("Validating existing results against current dataset...")
        for result in results:
            idx = result["index"]
            if idx < len(dataset):
                sample = dataset[idx]
                if (
                    result["question"] != sample["question"]
                    or result["gold_answer"] != sample["answer"]
                ):
                    raise ValueError(
                        f"Sample mismatch at index {idx}!\n"
                        f"Existing question: {result['question']}...\n"
                        f"Current question: {sample['question']}...\n"
                        f"Existing answer: {result['gold_answer']}\n"
                        f"Current answer: {sample['answer']}\n"
                        f"The saved results appear to be from a different dataset!"
                    )
        log.info("Validation passed - all existing results match current dataset")

    except Exception as e:
        if "Sample mismatch" in str(e):
            raise  # Re-raise validation errors

        log.warning(f"Failed to load existing results: {e}")
        results = []
        processed_indices = set()

    return results, processed_indices


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


def create_scorer(config, model):
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
        if config.scorer.type == "uncertainty":
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
            step_patterns=["- Step", "<Answer>:", "\n<Answer>:"],
            answer_patterns=["<Answer>:", "\n<Answer>:"],
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
        log.info(f"Using OpenAI API model: {config.model.model_path}")

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

        model = BlackboxModelWithStreaming(
            openai_api_key=config.model.api_key,
            model_path=config.model.model_path,
            supports_logprobs=config.model.supports_logprobs,
            boundary_detector=detector,
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


def create_tts_strategy(config, step_generator, scorer):
    if config.strategy.type == "online_best_of_n":
        strategy = StrategyOnlineBestOfN(
            step_generator=step_generator,
            scorer=scorer,
            candidates_per_step=config.strategy.candidates_per_step,
            max_steps=config.strategy.max_steps,
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

        # Store result WITHOUT correctness check
        results.append(
            {
                "index": i,
                "question": instance["question"],
                "gold_answer": instance["answer"],
                "generated_trajectory": result["trajectory"],
                "generated_answer": generated_text,
                "steps": result["steps"],
                "validity_scores": result["validity_scores"],
                "completed": result["completed"],
            }
        )

        log.info(f"Generated: {generated_text}")
        log.info(f"Num steps: {len(result['steps'])}")
        log.info(f"Avg validity: {np.mean(result['validity_scores']):.3f}")

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

    # always process all results, since we have deepseek cache.
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
        wandb.init(project=project, dir=output_dir, config=wandb_cfg)
        wandb_save_directory(Path(output_dir) / ".hydra")

    # Set random seeds
    set_random_seeds(config.system.seed)

    # Load dataset
    log.info(
        f"Loading dataset: {config.dataset.dataset_path} ({config.dataset.dataset_split})"
    )
    # MATH dataset
    if config.dataset.dataset_path == "EleutherAI/hendrycks_math":
        configs = get_dataset_config_names(config.dataset.dataset_path)
        datasets_by_subset = {
            cfg: load_dataset(
                config.dataset.dataset_path, cfg, split=config.dataset.dataset_split
            )
            for cfg in configs
        }
        dataset = concatenate_datasets(list(datasets_by_subset.values()))
        dataset = dataset.rename_columns({"problem": "question", "solution": "answer"})
    # proofNet dataset
    elif config.dataset.dataset_path == "hoskinson-center/proofnet":
        dataset = load_dataset(
            config.dataset.dataset_path,
            split=config.dataset.dataset_split,
            cache_dir=config.system.hf_cache,
        )
        dataset = dataset.rename_columns(
            {"nl_statement": "question", "nl_proof": "answer"}
        )
    else:
        dataset = load_dataset(
            config.dataset.dataset_path,
            config.dataset.dataset_config,
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
    log.info(f"Loading model: {config.model.model_path}")
    model, step_generator = create_model(config)

    # Create scorer
    scorer = create_scorer(config, model)

    # Create tts strategy
    generator = create_tts_strategy(
        config=config, step_generator=step_generator, scorer=scorer
    )

    # Load existing results if resuming
    if config.output.resume:
        results, processed_indices = load_existing_results(
            Path(output_dir) / "results.json", dataset
        )
    else:
        results = []
        processed_indices = set()

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
