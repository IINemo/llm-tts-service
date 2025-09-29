import torch
from typing import List, Dict

from llm_tts.step_generation import StepCandidateGenerator
from llm_tts.step_detection import StepBoundaryDetector
from .strategy_base import StrategyBase

import logging

log = logging.getLogger(__name__)


class StrategyOnlineBestOfN(StrategyBase):
    """
    Greedy online best-of-n strategy.
    """
    
    def __init__(
        self, 
        model,
        scorer,
        candidates_per_step: int,
        max_steps: int,
        max_new_tokens: int,
        temperature: float,
        generation_batch_size: int
    ):
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size or candidates_per_step
        self.model = model
        self.scorer = scorer
        
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
            max_new_tokens=max_new_tokens
        )
    
    def generate_trajectory(self, prompt: str) -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using specified criterion.
        
        Args:
            prompt: Initial prompt/question
            
        Returns:
            Dictionary with:
                - trajectory: Final generated trajectory
                - steps: List of selected steps
                - completed: Whether trajectory reached completion
        """

        trajectory = prompt
        selected_steps = []
        validity_scores = []
        
        for step_num in range(self.max_steps):
            log.info(f"\n=== Step {step_num} ===")

            # Generate candidates in batches if needed
            if self.generation_batch_size < self.candidates_per_step:
                candidates = self._generate_candidates_in_batches(trajectory)
            else:
                candidates = self.step_generator.generate_candidates(
                    trajectory
                )
            
            if not candidates:
                log.info("No candidates generated, stopping")
                break
            
            # Score candidates 
            all_validities = self.scorer.score_candidates(
                trajectory,
                [c.text for c in candidates]
            )
            
            # Aggregate scores for each candidate
            candidate_validity_scores = self._aggregate_scores(all_validities)
            
            # Log all candidates
            log.info(f"Generated {len(candidates)} candidates:")
            for i, (candidate, val_score) in enumerate(
                zip(candidates, candidate_validity_scores)
            ):
                log.info(f"  [{i}] Validity: {val_score:.3f} | Text: '{candidate.text}'")
            
            # Select best candidate 
            best_idx, selected_candidate = self._select_best_candidate(candidates, candidate_validity_scores)
            log.info(f"Selected candidate {best_idx}")
            log.info(f"Text: {selected_candidate.text}")
            
            # Update trajectory
            trajectory += selected_candidate.text
            selected_steps.append(selected_candidate.text)
            
            # Check if trajectory is complete
            if selected_candidate.is_trajectory_complete:
                log.info("Answer pattern detected - generating final answer")
                
                # Generate final answer
                final_answer, final_validity = self._generate_final_answer(
                    trajectory
                )
                trajectory += final_answer
                selected_steps.append(final_answer)
                validity_scores.append(final_validity)
                break
        
        return {
            "trajectory": trajectory,
            "steps": selected_steps,
            "validity_scores": validity_scores,
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
                
                log.info(f"Generating batch {batch_idx+1}/{num_batches} ({batch_size} candidates)")
                
                # Set batch size for this generation
                self.step_generator.candidates_per_step = batch_size
                
                # Generate batch
                batch_candidates = self.step_generator.generate_candidates(
                    trajectory
                )
                if batch_candidates:
                    all_candidates.extend(batch_candidates)
                    
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
                
        finally:
            # Always restore original setting
            self.step_generator.candidates_per_step = original_candidates
        
        return all_candidates
    
    def _aggregate_scores(self, all_validities) -> List[float]:
        """Aggregate validity scores from multiple evaluations"""
        scores = []
        for validities in all_validities:
            # Compute mean validity
            if validities and len(validities) > 0:
                if hasattr(validities, 'mean'):
                    validity_score = float(validities.mean())
                else:
                    validity_score = float(sum(validities) / len(validities))
            else:
                validity_score = 0.5
                
            scores.append(validity_score)
        return scores
    
    def _select_best_candidate(self, candidates: List, scores: List[float]) -> tuple:
        """Select the best candidate based on scores"""
        # Higher validity is better
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, candidates[best_idx]
    
    def _generate_answer_outputs(self, input_ids, attention_mask):
        """Generate answer outputs in batches if needed"""
        if self.generation_batch_size < self.candidates_per_step:
            outputs = []
            num_batches = (self.candidates_per_step + self.generation_batch_size - 1) // self.generation_batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.generation_batch_size
                end_idx = min((batch_idx + 1) * self.generation_batch_size, self.candidates_per_step)
                batch_size = end_idx - start_idx
                
                with torch.no_grad():
                    batch_outputs = self.step_generator.model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        num_return_sequences=batch_size,
                        pad_token_id=self.step_generator.tokenizer.eos_token_id,
                        eos_token_id=self.step_generator.tokenizer.eos_token_id
                    )
                    outputs.extend(batch_outputs)
                    
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                outputs = self.step_generator.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    num_return_sequences=self.candidates_per_step,
                    pad_token_id=self.step_generator.tokenizer.eos_token_id,
                    eos_token_id=self.step_generator.tokenizer.eos_token_id
                )
        return outputs
    
    def _extract_answer_candidates(self, outputs, input_ids):
        """Extract answer candidates from model outputs"""
        answer_candidates = []
        for seq in outputs:
            new_tokens = seq[input_ids.shape[1]:]
            answer_text = self.step_generator.tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer_candidates.append(answer_text)
            
        return answer_candidates
    
    def _generate_final_answer(self, trajectory: str) -> tuple:
        """Generate and select best final answer based on criterion"""
        
        # Generate answer candidates (without step detection)
        inputs = self.model.tokenize([trajectory])
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        
        # Generate answer candidates in batches if needed
        outputs = self._generate_answer_outputs(input_ids, attention_mask)
        
        # Extract answer candidates
        answer_candidates = self._extract_answer_candidates(outputs, input_ids)
        
        # Score answer candidates
        all_validities = self.scorer.compute_scores(
            trajectory,
            answer_candidates
        )
        
        # Aggregate scores
        answer_validity_scores = self._aggregate_scores(all_validities)
        
        # Select best answer based on criterion
        best_idx, _ = self._select_best_candidate(answer_candidates, answer_validity_scores)
        
        log.info(f"Generated {len(answer_candidates)} answer candidates")
        log.info(f"Selected answer {best_idx}")
        log.info(f"Validity: {answer_validity_scores[best_idx]:.3f}")
        
        return answer_candidates[best_idx]
    
    def cleanup(self):
        """Clean up resources"""

        self.scorer.cleanup()
