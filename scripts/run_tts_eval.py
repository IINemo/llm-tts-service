#!/usr/bin/env python3

import os
import logging
import random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
import traceback
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

from lm_polygraph import WhiteboxModel

from llm_tts.strategies import (
    DirectOnlineBestOfNReasonEvalSeparate, 
    run_separate_evaluations
)
from llm_tts import Annotator, _is_correct_answer

import logging

log = logging.getLogger(__name__)


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = None
    tokenizer.padding_side = 'left'  # Fix padding side for decoder-only models
    return tokenizer


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, trust_remote_code=True)
    return model


def load_prompt_template(prompt_file: str) -> str:
    """Load prompt template from file"""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    else:
        # Default prompt template for ReasonEval
        return "Question: {question}\n\nLet's solve this step by step.\n\n"


def prepare_dataset_with_prompts(dataset, prompt_template: str):
    """Add prompts to dataset questions"""
    
    def add_prompt(example):
        # Format prompt with question
        if "{question}" in prompt_template:
            example["question"] = prompt_template.format(question=example["question"])
        else:
            example["question"] = prompt_template + example["question"]
        return example
    
    return dataset.map(add_prompt)


def run_single_criterion(
    generator: DirectOnlineBestOfNReasonEvalSeparate,
    dataset,
    save_path: str,
    subset_size: int,
    verbose: bool,
    resume: bool = False,
    correctness_mode: str = "exact_match",
    n_threads: int = 1,
    annotation_prompt_type: str = "non_unique",
    prompt_file: str = None
):
    """Run evaluation for a single criterion"""
    
    log.info(f"\n{'='*60}")
    log.info(f"Running evaluation")
    log.info(f"{'='*60}")
    
    # Load existing results if resuming
    results = []
    processed_indices = set()
    
    if resume and os.path.exists(save_path):
        log.info(f"Loading existing results from {save_path}")
        try:
            results = torch.load(save_path)
            processed_indices = {r["index"] for r in results}
            log.info(f"Loaded {len(results)} existing results")
            log.info(f"Already processed indices: {sorted(processed_indices)}")
            
            # Validate all existing results match current dataset
            log.info("Validating existing results against current dataset...")
            for result in results:
                idx = result["index"]
                if idx < len(dataset):
                    sample = dataset[idx]
                    if (result["question"] != sample["question"] or 
                        result["gold_answer"] != sample["answer"]):
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
    
    # Phase 1: Generate trajectories (without checking correctness)
    log.info(f"\n{'='*60}")
    log.info(f"Phase 1: Generating trajectories")
    log.info(f"{'='*60}")
    
    for i in tqdm(range(subset_size), desc=f"Generating trajectories"):
        # Skip if already processed
        if i in processed_indices:
            if verbose:
                log.info(f"Skipping sample {i} (already processed)")
            continue
            
        sample = dataset[i]
        
        if verbose:
            log.info(f"\n{'='*60}")
            log.info(f"Sample {i+1}/{subset_size}")
            log.info(f"Question: {sample['question'][:200]}...")
        
        try:
            # Generate trajectory
            result = generator.generate_trajectory(sample["question"])
            
            # Extract generated answer (but don't check correctness yet)
            generated_text = result["trajectory"]
            if sample["question"] in generated_text:
                generated_text = generated_text.replace(sample["question"], "").strip()
            
            # Store result WITHOUT correctness check
            results.append({
                "index": i,
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "generated_trajectory": result["trajectory"],
                "generated_answer": generated_text,
                "steps": result["steps"],
                "validity_scores": result["validity_scores"],
                "redundancy_scores": result["redundancy_scores"],
                "criterion_used": result["criterion_used"],
                "completed": result["completed"]
            })
            
            if verbose:
                log.info(f"Generated: {generated_text}")
                log.info(f"Num steps: {len(result['steps'])}")
                if result['validity_scores']:
                    log.info(f"Avg validity: {np.mean(result['validity_scores']):.3f}")
                if result['redundancy_scores']:
                    log.info(f"Avg redundancy: {np.mean(result['redundancy_scores']):.3f}")
            
        except Exception as e:
            log.error(f"Error processing sample {i}: {e}")
            traceback.print_exc()
            
            results.append({
                "index": i,
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "error": str(e),
                "criterion_used": "validity",
                "completed": False
            })
        
        # Save periodically
        if len(results) % 10 == 0:
            torch.save(results, save_path)
            log.info(f"Saved {len(results)} results to {save_path}")
    
    # Final save after generation
    torch.save(results, save_path)
    log.info(f"Final save after generation: {len(results)} results to {save_path}")
    
    # Phase 2: Check correctness for all results
    log.info(f"\n{'='*60}")
    log.info(f"Phase 2: Checking correctness")
    log.info(f"{'='*60}")
    
    if correctness_mode == "exact_match":
        # Use exact match checking
        for i, result in enumerate(tqdm(results, desc=f"Checking correctness (exact match, {criterion})")):
            # Skip if this result has an error or already has correctness checked
            if "error" in result or "is_correct_exact_match" in result:
                continue
                
            try:
                is_correct = _is_correct_answer(result["generated_answer"], result["gold_answer"])
                result["is_correct_exact_match"] = is_correct
                
                if verbose and i % 10 == 0:  # Log every 10th for less clutter
                    log.info(f"\nSample {result['index']}:")
                    log.info(f"Generated answer: {result['generated_answer'][:100]}...")
                    log.info(f"Gold answer: {result['gold_answer'][:100]}...")
                    log.info(f"Correct: {is_correct}")
            except Exception as e:
                log.error(f"Error checking correctness for sample {result['index']}: {e}")
                result["is_correct_exact_match"] = False
                
    elif correctness_mode == "deepseek":
        # Use DeepSeek verification
        log.info(f"Using DeepSeek verification with {n_threads} threads")
        
        # Load prompt template and ensure compatibility
        prompt_template = load_prompt_template(prompt_file) if prompt_file else "{q}"
        if "{question}" in prompt_template:
            prompt_template = prompt_template.replace("{question}", "{q}")
        
        # print(f'Using prompt template:\n{prompt_template}')
        # import pdb; pdb.set_trace()
        # Create annotator
        annotator = Annotator(
            prompt=prompt_template,
            n_threads=n_threads,
            cache_path="~/.cache",
            annotation_prompt_type=annotation_prompt_type
        )
        
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
            log.info(f"Verifying {len(problems)} solutions with DeepSeek ({annotation_prompt_type} prompt)...")
            
            # Get annotations from DeepSeek
            try:
                annotations = annotator(problems, solutions, gold_answers)
                
                # Update results with correctness
                for idx, annotation in zip(result_indices, annotations):
                    if np.isnan(annotation):
                        log.warning(f"DeepSeek returned unclear result for sample {results[idx]['index']}, marking as incorrect")
                        results[idx]["is_correct_deepseek"] = False
                    else:
                        results[idx]["is_correct_deepseek"] = (annotation == 0)  # 0 = correct, 1 = incorrect
                    
                    if verbose and (idx - result_indices[0]) % 10 == 0:
                        log.info(f"\nSample {results[idx]['index']}:")
                        log.info(f"DeepSeek annotation: {annotation}")
                        log.info(f"Correct: {results[idx]['is_correct_deepseek']}")
                        
            except Exception as e:
                log.error(f"Error during DeepSeek verification: {e}")
                # Fall back to marking all as incorrect
                for idx in result_indices:
                    results[idx]["is_correct_deepseek"] = False
    
    # Final save with correctness results
    torch.save(results, save_path)
    log.info(f"Final save with correctness: {len(results)} results to {save_path}")
    
    # Print summary
    # Use the appropriate correctness key based on the mode
    correctness_key = f"is_correct_{correctness_mode}"
    correct = sum(r.get(correctness_key, False) for r in results)
    completed = sum(r.get("completed", False) for r in results)
    errors = sum("error" in r for r in results)
    
    log.info(f"\nSummary:")
    log.info(f"  - Correctness mode: {correctness_mode}")
    log.info(f"  - Total samples: {len(results)}")
    log.info(f"  - Completed: {completed} ({completed/len(results):.1%})")
    log.info(f"  - Correct ({correctness_mode}): {correct} ({correct/len(results):.1%})")
    log.info(f"  - Errors: {errors}")
    
    if completed > 0:
        log.info(f"  - Accuracy (of completed): {correct/completed:.1%}")
    
    # Average statistics
    all_validities = []
    all_redundancies = []
    all_steps = []
    
    for r in results:
        if "validity_scores" in r and r["validity_scores"]:
            all_validities.extend(r["validity_scores"])
            all_redundancies.extend(r["redundancy_scores"])
            all_steps.append(len(r["steps"]))
    
    if all_validities:
        log.info(f"\nStep Statistics:")
        log.info(f"  - Avg steps per trajectory: {np.mean(all_steps):.1f}")
        log.info(f"  - Avg validity score: {np.mean(all_validities):.3f}")
        log.info(f"  - Avg redundancy score: {np.mean(all_redundancies):.3f}")
    
    return results


def wandb_save_directory(directory_path):
    import wandb

    for file_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, file_name)
        if os.path.isfile(full_path):  # Make sure it's a file, not a directory
            wandb.save(full_path)



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
    if getattr(config, 'report_to', None) == "wandb":
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
    random.seed(config.system.seed)
    np.random.seed(config.system.seed)
    torch.manual_seed(config.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.system.seed)
        torch.cuda.manual_seed_all(config.system.seed)
    log.info(f"Set random seed to {config.system.seed}")
    
    # Extract dataset name and ReasonEval name for directory structure
    dataset_name = config.dataset.dataset_path.split('/')[-1] if '/' in config.dataset.dataset_path else config.dataset.dataset_path
    reasoneval_name = config.model.reasoneval_path.split('/')[-1] if '/' in config.model.reasoneval_path else config.model.reasoneval_path
    
    # Create save paths with directory structure
    save_dir = os.path.join(config.output.save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path_validity = os.path.join(save_dir, f"{reasoneval_name}_validity.pt")
    
    # Load dataset
    log.info(f"Loading dataset: {config.dataset.dataset_path} ({config.dataset.dataset_split})")
    dataset = load_dataset(
        config.dataset.dataset_path, 
        split=config.dataset.dataset_split,
        cache_dir=config.system.hf_cache
    )
    
    # Load model
    log.info(f"Loading model: {config.model.model_path}")
    tokenizer = load_tokenizer(config.model.model_path)
    base_model = load_model(config.model.model_path, config.system.device)
    base_model.eval()
    model = WhiteboxModel(base_model, tokenizer)
    
    # Create generator
    log.info(f"Using device {config.system.device} for base model, {config.system.reasoneval_device} for ReasonEval")
    
    # Determine batch size for generation
    batch_size = config.generation.batch_size if config.generation.batch_size else config.generation.n
    if config.generation.sequential_generation:
        batch_size = 1
        log.info("Using sequential generation (batch_size=1) to save memory")

    elif batch_size < config.generation.n:
        log.info(f"Using batch generation with batch_size={batch_size}")
    
    generator = DirectOnlineBestOfNReasonEvalSeparate(
        model=model,
        reasoneval_model_path=config.model.reasoneval_path,
        candidates_per_step=config.generation.n,
        max_steps=config.generation.max_steps,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        device=config.system.device,
        reasoneval_device=config.system.reasoneval_device,
        verbose=config.verbose,
        generation_batch_size=batch_size
    )
    
    # Process dataset
    subset_size = min(config.dataset.subset, len(dataset)) if config.dataset.subset else len(dataset)
    
    run_single_criterion(
        generator, dataset, "validity", 
        save_path_validity, subset_size, config.verbose,
        resume=config.output.resume,
        correctness_mode=config.output.correctness_mode,
        n_threads=config.output.n_threads,
        annotation_prompt_type=config.output.annotation_prompt_type,
        prompt_file=config.dataset.prompt_file
    )

if __name__ == "__main__":
    main()