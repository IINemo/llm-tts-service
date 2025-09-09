"""
Candidate step generation system for online best-of-n
"""

import torch
from typing import List, Dict, Tuple, Optional
from transformers import StoppingCriteriaList
import logging
import time

from lm_polygraph import WhiteboxModel
from .step_detection import (
    StepBoundaryDetector, 
    BatchStepStoppingCriteria, 
    StepExtractionResult
)

log = logging.getLogger(__name__)


class StepCandidate:
    """Represents a candidate next step in trajectory"""
    
    def __init__(
        self,
        text: str,
        token_ids: List[int],
        is_complete: bool,
        is_trajectory_complete: bool,
        generation_scores: Optional[torch.Tensor] = None,
        raw_text: str = None
    ):
        self.text = text
        self.token_ids = token_ids
        self.is_complete = is_complete
        self.is_trajectory_complete = is_trajectory_complete
        self.generation_scores = generation_scores
        self.raw_text = raw_text or text
        
    def __str__(self):
        return f"StepCandidate(text='{self.text[:50]}...', complete={self.is_complete})"


class StepCandidateGenerator:
    """Generates N candidate next steps for online best-of-n"""
    
    def __init__(
        self,
        model: WhiteboxModel,
        detector: StepBoundaryDetector = None,
        candidates_per_step: int = 5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 250,
        device: str = "cuda"
    ):
        self.model = model
        self.detector = detector or StepBoundaryDetector()
        self.candidates_per_step = candidates_per_step
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device
        
    def generate_candidates(
        self, 
        trajectory: str,
        verbose: bool = False
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory"""
        
        if verbose:
            log.info(f"Generating {self.candidates_per_step} candidates from trajectory")
            
        # Tokenize current trajectory
        inputs = self.model.tokenize([trajectory])
        input_length = inputs['input_ids'].shape[1]
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Create stopping criteria for batch generation
        stopping_criteria = BatchStepStoppingCriteria(
            tokenizer=self.model.tokenizer,
            start_length=input_length,
            detector=self.detector,
            batch_size=self.candidates_per_step
        )
        
        try:
            start_time = time.time()
            
            gen_params = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_return_sequences": self.candidates_per_step,
                "output_scores": True,
                "return_dict_in_generate": True,
                "stopping_criteria": StoppingCriteriaList([stopping_criteria]),
                "pad_token_id": self.model.tokenizer.eos_token_id,
                "eos_token_id": self.model.tokenizer.eos_token_id
            }
            
            if verbose:
                log.info(f"Generation params: do_sample={gen_params['do_sample']}, temp={gen_params['temperature']}, top_p={gen_params['top_p']}, num_return_sequences={gen_params['num_return_sequences']}")
                log.info(f"Model generation_parameters.do_sample: {self.model.generation_parameters.do_sample}")
                log.info(f"Model generation_parameters.temperature: {self.model.generation_parameters.temperature}")
            
            # Override model's default generation parameters to ensure sampling
            old_do_sample = self.model.generation_parameters.do_sample
            old_temperature = self.model.generation_parameters.temperature
            old_top_p = self.model.generation_parameters.top_p
            old_top_k = self.model.generation_parameters.top_k
            
            self.model.generation_parameters.do_sample = True
            self.model.generation_parameters.temperature = self.temperature
            self.model.generation_parameters.top_p = self.top_p
            self.model.generation_parameters.top_k = self.top_k
            
            if verbose:
                log.info(f"After override - do_sample: {self.model.generation_parameters.do_sample}, temp: {self.model.generation_parameters.temperature}")
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_params)
            
            # Restore original parameters
            self.model.generation_parameters.do_sample = old_do_sample
            self.model.generation_parameters.temperature = old_temperature
            self.model.generation_parameters.top_p = old_top_p
            self.model.generation_parameters.top_k = old_top_k
                
            generation_time = time.time() - start_time
            if verbose:
                log.info(f"Generated candidates in {generation_time:.2f}s")
                
        except torch.OutOfMemoryError as e:
            log.error(f"CUDA OOM during candidate generation: {e}")
            # torch.cuda.empty_cache()
            raise e
            # Fallback to single candidate
            # return self._generate_single_candidate(trajectory, verbose)
            
        # Extract step candidates
        candidates = []
        for i, sequence in enumerate(outputs.sequences):
            # Get newly generated tokens
            new_tokens = sequence[input_length:]
            raw_generated_text = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Extract step using detector
            step_text = self.detector.extract_step_text(raw_generated_text)
            is_complete = self.detector.is_step_complete(raw_generated_text)
            is_trajectory_complete = self.detector.is_trajectory_complete(raw_generated_text)
            
            # Get generation scores if available
            gen_scores = None
            if hasattr(outputs, 'scores') and outputs.scores:
                gen_scores = torch.stack(outputs.scores, dim=1)[i] if i < len(outputs.scores) else None
                
            candidate = StepCandidate(
                text=step_text,
                token_ids=new_tokens.tolist(),
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                generation_scores=gen_scores,
                raw_text=raw_generated_text
            )
            candidates.append(candidate)
            
        # if verbose:
        #     log.info(f"Generated {len(candidates)} candidates:")
        #     for i, candidate in enumerate(candidates):
        #         log.info(f"  {i}: '{candidate.text}'")
                
        return candidates
        
    def _generate_single_candidate(
        self, 
        trajectory: str, 
        verbose: bool = False
    ) -> List[StepCandidate]:
        """Fallback to single candidate generation on OOM"""
        if verbose:
            log.warning("Falling back to single candidate generation due to memory constraints")
            
        inputs = self.model.tokenize([trajectory])
        input_length = inputs['input_ids'].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                num_return_sequences=1,
                pad_token_id=self.model.tokenizer.eos_token_id,
                eos_token_id=self.model.tokenizer.eos_token_id
            )
            
        new_tokens = outputs[0][input_length:]
        # Decode with special tokens to preserve EOS detection
        raw_text_with_special = self.model.tokenizer.decode(new_tokens, skip_special_tokens=False)
        raw_text = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        step_text = self.detector.extract_step_text(raw_text)
        
        # Check if EOS token was reached
        # import pdb; pdb.set_trace()
        reached_eos = (new_tokens[-1].item() == self.model.tokenizer.eos_token_id) if len(new_tokens) > 0 else False
        
        candidate = StepCandidate(
            text=step_text,
            token_ids=new_tokens.tolist(),
            is_complete=self.detector.is_step_complete(raw_text),
            is_trajectory_complete=self.detector.is_trajectory_complete(raw_text, reached_eos=reached_eos),
            raw_text=raw_text
        )
        # import pdb; pdb.set_trace()
        return [candidate]
        
    def filter_valid_candidates(self, candidates: List[StepCandidate]) -> List[StepCandidate]:
        """Filter out invalid or empty candidates"""
        valid_candidates = []
        
        for candidate in candidates:
            # Skip empty or very short candidates
            if len(candidate.text.strip()) < 3:
                continue
                
            # Skip candidates that are just punctuation or whitespace
            if not any(c.isalnum() for c in candidate.text):
                continue
                
            valid_candidates.append(candidate)
            
        # If no valid candidates, return at least one
        print(f"valid_candidates: {valid_candidates}")
        if not valid_candidates and candidates:
            valid_candidates = [candidates[0]]
            
        return valid_candidates