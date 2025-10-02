"""
Candidate step generation system for online best-of-n
"""

import torch
from typing import List
from transformers import StoppingCriteriaList
import time

from llm_tts.step_candidate_generator_base import StepCandidate, StepCandidateGeneratorBase

from lm_polygraph import WhiteboxModel

from .step_detection import (
    StepBoundaryDetector,
    BatchStepStoppingCriteria,
)

import logging

log = logging.getLogger(__name__)


class StepCandidateGeneratorThroughHuggingface(StepCandidateGeneratorBase):
    """Generates N candidate next steps for online best-of-n"""

    def __init__(
        self,
        model: WhiteboxModel,
        detector: StepBoundaryDetector,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
    ):
        self.model = model
        self.detector = detector or StepBoundaryDetector()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = model.device()

    def generate_candidates(self, request, candidates_per_step: int) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory"""

        log.info(f"Generating {candidates_per_step} candidates from trajectory")

        # Tokenize current trajectory
        inputs = self.model.tokenizer.apply_chat_template([request], tokenize=False, add_generation_prompt=True)
        inputs = self.model.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_new_tokens,
        )
        input_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Create stopping criteria for batch generation
        stopping_criteria = BatchStepStoppingCriteria(
            tokenizer=self.model.tokenizer,
            start_length=input_length,
            detector=self.detector,
            batch_size=candidates_per_step,
        )

        try:
            start_time = time.time()

            gen_params = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_return_sequences": candidates_per_step,
                "output_scores": True,
                "return_dict_in_generate": True,
                "stopping_criteria": StoppingCriteriaList([stopping_criteria]),
                "pad_token_id": self.model.tokenizer.eos_token_id,
                "eos_token_id": self.model.tokenizer.eos_token_id,
            }

            log.info(
                f"Generation params: do_sample={gen_params['do_sample']}, temp={gen_params['temperature']}, top_p={gen_params['top_p']}, num_return_sequences={gen_params['num_return_sequences']}"
            )
            log.info(
                f"Model generation_parameters.do_sample: {self.model.generation_parameters.do_sample}"
            )
            log.info(
                f"Model generation_parameters.temperature: {self.model.generation_parameters.temperature}"
            )

            # Override model's default generation parameters to ensure sampling
            old_do_sample = self.model.generation_parameters.do_sample
            old_temperature = self.model.generation_parameters.temperature
            old_top_p = self.model.generation_parameters.top_p
            old_top_k = self.model.generation_parameters.top_k

            self.model.generation_parameters.do_sample = True
            self.model.generation_parameters.temperature = self.temperature
            self.model.generation_parameters.top_p = self.top_p
            self.model.generation_parameters.top_k = self.top_k

            log.info(
                f"After override - do_sample: {self.model.generation_parameters.do_sample}, temp: {self.model.generation_parameters.temperature}"
            )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_params)

            # Restore original parameters
            self.model.generation_parameters.do_sample = old_do_sample
            self.model.generation_parameters.temperature = old_temperature
            self.model.generation_parameters.top_p = old_top_p
            self.model.generation_parameters.top_k = old_top_k

            generation_time = time.time() - start_time

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
            raw_generated_text = self.model.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            # Extract step using detector
            step_text = self.detector.extract_step_text(raw_generated_text)
            is_complete = self.detector.is_step_complete(raw_generated_text)
            is_trajectory_complete = self.detector.is_trajectory_complete(
                raw_generated_text
            )

            # Get generation scores if available
            gen_scores = None
            if hasattr(outputs, "scores") and outputs.scores:
                gen_scores = (
                    torch.stack(outputs.scores, dim=1)[i]
                    if i < len(outputs.scores)
                    else None
                )

            candidate = StepCandidate(
                text=step_text,
                token_ids=new_tokens.tolist(),
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                generation_scores=gen_scores,
                raw_text=raw_generated_text,
            )
            candidates.append(candidate)

        return candidates

    def generate_answer(self, request, candidates_per_step: int) -> str:
        """Generate and select best final answer based on criterion"""

        return self.generate_candidates(request, candidates_per_step)
