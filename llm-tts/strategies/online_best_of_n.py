"""
Direct online best-of-n implementation using separate validity and redundancy evaluations
"""

import torch
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

from datasets import Dataset
from lm_polygraph import WhiteboxModel

from online_bestofn.step_detection import StepBoundaryDetector
from online_bestofn.step_generation import StepCandidateGenerator
from online_bestofn.scorers.reasoneval_direct import DirectReasonEvalScorerSeparate
from online_bestofn.direct_online_bestofn import _is_correct_answer

log = logging.getLogger(__name__)


class DirectOnlineBestOfNReasonEvalSeparate:
    """
    Online best-of-n using ReasonEval with separate validity and redundancy evaluations.
    
    This runs the evaluation twice:
    1. Once selecting based on validity (higher validity = better)
    2. Once selecting based on redundancy (lower redundancy = better)
    """
    
    def __init__(
        self,
        model: WhiteboxModel,
        reasoneval_model_path: str = "GAIR/ReasonEval-7B",
        candidates_per_step: int = 10,
        max_steps: int = 20,
        max_new_tokens: int = 350,
        temperature: float = 0.7,
        device: str = "cuda",
        reasoneval_device: str = None,
        verbose: bool = True,
        generation_batch_size: int = None
    ):
        self.model = model
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.verbose = verbose
        self.generation_batch_size = generation_batch_size or candidates_per_step
        
        # Initialize components
        self.detector = StepBoundaryDetector(
            step_patterns=["- Step", "<Answer>:", "\n<Answer>:"],
            answer_patterns=["<Answer>:", "\n<Answer>:"],
            max_tokens_per_step=max_new_tokens
        )
        
        self.step_generator = StepCandidateGenerator(
            model=model,
            detector=self.detector,
            candidates_per_step=candidates_per_step,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device
        )
        
        self.scorer = DirectReasonEvalScorerSeparate(
            model=model,
            reasoneval_model_path=reasoneval_model_path,
            device=reasoneval_device if reasoneval_device else device,
            batch_size=candidates_per_step
        )
    
    def generate_trajectory(self, prompt: str, criterion: str = "validity") -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using specified criterion.
        
        Args:
            prompt: Initial prompt/question
            criterion: "validity" or "redundancy"
            
        Returns:
            Dictionary with:
                - trajectory: Final generated trajectory
                - steps: List of selected steps
                - validity_scores: Validity scores for each selected step
                - redundancy_scores: Redundancy scores for each selected step
                - criterion_used: Which criterion was used for selection
                - completed: Whether trajectory reached completion
        """
        
        if criterion not in ["validity", "redundancy", "both"]:
            raise ValueError(f"Invalid criterion: {criterion}. Must be 'validity', 'redundancy', or 'both'")
        
        trajectory = prompt
        selected_steps = []
        validity_scores = []
        redundancy_scores = []
        
        for step_num in range(self.max_steps):
            if self.verbose:
                log.info(f"\n=== Step {step_num} (selecting by {criterion}) ===")
            
            # Generate candidates
            if self.verbose:
                log.info(f"Generating candidates with temperature={self.temperature}")
            
            # Generate candidates in batches if needed
            if self.generation_batch_size < self.candidates_per_step:
                candidates = self._generate_candidates_in_batches(trajectory)
            else:
                candidates = self.step_generator.generate_candidates(
                    trajectory, 
                    verbose=self.verbose
                )
            
            if not candidates:
                if self.verbose:
                    log.info("No candidates generated, stopping")
                break
            
            # Score candidates with ReasonEval
            all_validities, all_redundancies = self.scorer.compute_validity_redundancy_scores(
                trajectory,
                [c.text for c in candidates]
            )
            
            # Aggregate scores for each candidate
            candidate_validity_scores = []
            candidate_redundancy_scores = []
            
            for validities, redundancies in zip(all_validities, all_redundancies):
                # Compute mean validity
                if validities and len(validities) > 0:
                    if hasattr(validities, 'mean'):
                        validity_score = float(validities.mean())
                    else:
                        validity_score = float(sum(validities) / len(validities))
                else:
                    validity_score = 0.5
                    
                # Compute mean redundancy
                if redundancies and len(redundancies) > 0:
                    if hasattr(redundancies, 'mean'):
                        redundancy_score = float(redundancies.mean())
                    else:
                        redundancy_score = float(sum(redundancies) / len(redundancies))
                else:
                    redundancy_score = 0.5
                    
                candidate_validity_scores.append(validity_score)
                candidate_redundancy_scores.append(redundancy_score)
            
            # Log all candidates
            if self.verbose:
                log.info(f"Generated {len(candidates)} candidates:")
                for i, (candidate, val_score, red_score) in enumerate(
                    zip(candidates, candidate_validity_scores, candidate_redundancy_scores)
                ):
                    log.info(f"  [{i}] Validity: {val_score:.3f}, Redundancy: {red_score:.3f} | Text: '{candidate.text}'")
            
            # Select best candidate based on criterion
            if criterion == "validity":
                # Higher validity is better
                best_idx = max(range(len(candidate_validity_scores)), 
                              key=lambda i: candidate_validity_scores[i])
            elif criterion == "redundancy":
                # Lower redundancy is better
                best_idx = min(range(len(candidate_redundancy_scores)), 
                              key=lambda i: candidate_redundancy_scores[i])
            else:  # both
                # Combine scores: higher validity + lower redundancy is better
                # Score = validity - redundancy (higher is better)
                combined_scores = [v - r for v, r in zip(candidate_validity_scores, candidate_redundancy_scores)]
                best_idx = max(range(len(combined_scores)), 
                              key=lambda i: combined_scores[i])
            
            selected_candidate = candidates[best_idx]
            selected_validity = candidate_validity_scores[best_idx]
            selected_redundancy = candidate_redundancy_scores[best_idx]
            
            if self.verbose:
                log.info(f"Selected candidate {best_idx} based on {criterion}")
                log.info(f"Validity: {selected_validity:.3f}, Redundancy: {selected_redundancy:.3f}")
                if criterion == "both":
                    log.info(f"Combined score: {selected_validity - selected_redundancy:.3f}")
                log.info(f"Text: {selected_candidate.text}")
            
            # Update trajectory
            trajectory += selected_candidate.text
            selected_steps.append(selected_candidate.text)
            validity_scores.append(selected_validity)
            redundancy_scores.append(selected_redundancy)
            
            # Check if trajectory is complete
            if selected_candidate.is_trajectory_complete:
                if self.verbose:
                    log.info("Answer pattern detected - generating final answer")
                
                # Generate final answer
                final_answer, final_validity, final_redundancy = self._generate_final_answer(
                    trajectory, criterion
                )
                trajectory += final_answer
                selected_steps.append(final_answer)
                validity_scores.append(final_validity)
                redundancy_scores.append(final_redundancy)
                break
        
        return {
            "trajectory": trajectory,
            "steps": selected_steps,
            "validity_scores": validity_scores,
            "redundancy_scores": redundancy_scores,
            "criterion_used": criterion,
            "completed": len(selected_steps) > 0
        }
    
    def _generate_candidates_in_batches(self, trajectory: str) -> List:
        """Generate candidates in smaller batches to avoid OOM"""
        all_candidates = []
        
        # Calculate number of batches needed
        num_batches = (self.candidates_per_step + self.generation_batch_size - 1) // self.generation_batch_size
        
        # Temporarily store original setting
        original_candidates = self.step_generator.candidates_per_step
        
        try:
            for batch_idx in range(num_batches):
                # Calculate batch size for this iteration
                start_idx = batch_idx * self.generation_batch_size
                end_idx = min((batch_idx + 1) * self.generation_batch_size, self.candidates_per_step)
                batch_size = end_idx - start_idx
                
                if self.verbose:
                    log.info(f"Generating batch {batch_idx+1}/{num_batches} ({batch_size} candidates)")
                
                # Set batch size for this generation
                self.step_generator.candidates_per_step = batch_size
                
                # Generate batch
                batch_candidates = self.step_generator.generate_candidates(
                    trajectory, 
                    verbose=False  # Avoid too much logging
                )
                
                if batch_candidates:
                    all_candidates.extend(batch_candidates)
                    
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
                
        finally:
            # Always restore original setting
            self.step_generator.candidates_per_step = original_candidates
        
        return all_candidates
    
    def _generate_final_answer(self, trajectory: str, criterion: str) -> tuple:
        """Generate and select best final answer based on criterion"""
        
        # Generate answer candidates (without step detection)
        inputs = self.model.tokenize([trajectory])
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generate answer candidates in batches if needed
        if self.generation_batch_size < self.candidates_per_step:
            outputs = []
            num_batches = (self.candidates_per_step + self.generation_batch_size - 1) // self.generation_batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.generation_batch_size
                end_idx = min((batch_idx + 1) * self.generation_batch_size, self.candidates_per_step)
                batch_size = end_idx - start_idx
                
                with torch.no_grad():
                    batch_outputs = self.model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=self.temperature,
                        num_return_sequences=batch_size,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id
                    )
                    outputs.extend(batch_outputs)
                    
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                outputs = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=self.temperature,
                    num_return_sequences=self.candidates_per_step,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    eos_token_id=self.model.tokenizer.eos_token_id
                )
        
        # Extract answer candidates
        answer_candidates = []
        for seq in outputs:
            new_tokens = seq[input_ids.shape[1]:]
            answer_text = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer_candidates.append(answer_text)
        
        # Score answer candidates
        all_validities, all_redundancies = self.scorer.compute_validity_redundancy_scores(
            trajectory,
            answer_candidates
        )
        
        # Aggregate scores
        answer_validity_scores = []
        answer_redundancy_scores = []
        
        for validities, redundancies in zip(all_validities, all_redundancies):
            if validities and len(validities) > 0:
                validity_score = float(sum(validities) / len(validities))
            else:
                validity_score = 0.5
                
            if redundancies and len(redundancies) > 0:
                redundancy_score = float(sum(redundancies) / len(redundancies))
            else:
                redundancy_score = 0.5
                
            answer_validity_scores.append(validity_score)
            answer_redundancy_scores.append(redundancy_score)
        
        # Select best answer based on criterion
        best_idx = max(range(len(answer_validity_scores)), 
                        key=lambda i: answer_validity_scores[i])
        
        if self.verbose:
            log.info(f"Generated {len(answer_candidates)} answer candidates")
            log.info(f"Selected answer {best_idx} based on {criterion}")
            log.info(f"Validity: {answer_validity_scores[best_idx]:.3f}, " +
                    f"Redundancy: {answer_redundancy_scores[best_idx]:.3f}")
        
        return (answer_candidates[best_idx], 
                answer_validity_scores[best_idx],
                answer_redundancy_scores[best_idx])
    
    def cleanup(self):
        """Clean up resources"""
        self.scorer.cleanup()


def run_separate_evaluations(
    dataset: Dataset,
    model: WhiteboxModel,
    reasoneval_model_path: str,
    save_path: str,
    n: int,
    max_new_tokens: int,
    subset: Optional[int] = None,
    verbose: bool = True
):
    """
    Run separate evaluations for validity and redundancy criteria.
    
    Args:
        dataset: Evaluation dataset
        model: Language model
        reasoneval_model_path: Path to ReasonEval model
        save_path_validity: Path to save validity-based results
        save_path_redundancy: Path to save redundancy-based results
        n: Number of candidates per step
        max_new_tokens: Max tokens per step
        subset: Evaluate only first N samples
        verbose: Enable verbose logging
    """
    
    if subset:
        dataset = dataset.select(range(min(subset, len(dataset))))
        log.info(f"Using subset of {len(dataset)} samples")
    
    # Initialize generator
    generator = DirectOnlineBestOfNReasonEvalSeparate(
        model=model,
        reasoneval_model_path=reasoneval_model_path,
        candidates_per_step=n,
        temperature=0.7,
        device=str(model.device()),
        verbose=verbose
    )
    
        
    results = []
    
    try:
        for i, sample in enumerate(tqdm(dataset, desc=f"Processing samples")):
            if verbose:
                log.info(f"\n{'='*60}")
                log.info(f"Sample {i+1}/{len(dataset)}")
                log.info(f"Question: {sample['question']}")
                log.info(f"Gold Answer: {sample['answer']}")
            
            # Generate trajectory
            result = generator.generate_trajectory(sample["question"], criterion=criterion)
            
            # Extract answer from trajectory
            generated_text = result["trajectory"]
            if sample["question"] in generated_text:
                generated_text = generated_text.replace(sample["question"], "").strip()
            
            # Check correctness
            is_correct = _is_correct_answer(generated_text, sample["answer"])
            
            # Store result
            results.append({
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "generated_trajectory": result["trajectory"],
                "generated_answer": generated_text,
                "steps": result["steps"],
                "validity_scores": result["validity_scores"],
                "redundancy_scores": result["redundancy_scores"],
                "criterion_used": result["criterion_used"],
                "is_correct": is_correct,
                "completed": result["completed"]
            })
            
            if verbose:
                log.info(f"Generated: {generated_text[:200]}...")
                log.info(f"Correct: {is_correct}")
            
            # Save periodically
            if (i + 1) % 10 == 0:
                torch.save(results, save_path)
                log.info(f"Saved {len(results)} results to {save_path}")
    
    finally:
        # Final save
        torch.save(results, save_path)
        log.info(f"Final save: {len(results)} results to {save_path}")
    
    # Cleanup
    generator.cleanup()
    
    return results