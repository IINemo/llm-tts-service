"""
Candidate step generation system for online best-of-n
"""

import inspect
import logging
import time
from typing import Dict, List

import torch
from lm_polygraph import WhiteboxModel
from transformers import StoppingCriteria, StoppingCriteriaList

from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.step_boundary_detectors import (
    StructuredStepDetector,
    ThinkingMarkerDetector,
)

log = logging.getLogger(__name__)


class BatchStepStoppingCriteria(StoppingCriteria):
    """Stopping criteria for batch step generation"""

    def __init__(
        self,
        tokenizer,
        start_length: int,
        detector: StructuredStepDetector,
        batch_size: int,
    ):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.detector = detector
        self.batch_size = batch_size
        self.finished = [False] * batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        """Check stopping criteria for entire batch"""

        # Check each sequence in batch
        for i in range(min(input_ids.shape[0], self.batch_size)):
            if not self.finished[i]:
                generated_ids = input_ids[i][self.start_length :]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                if self.detector.is_step_complete(
                    generated_text, token_count=len(generated_ids)
                ):
                    self.finished[i] = True

        # Stop when all sequences are finished
        return all(self.finished)


class ThinkingStepStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria for thinking mode step generation.

    Uses ThinkingMarkerDetector to detect semantic step boundaries during generation.
    Stops when a new step boundary is detected (step count increases).

    Args:
        tokenizer: Tokenizer for decoding generated ids
        start_length: Length of input (to extract only generated tokens)
        detector: ThinkingMarkerDetector instance for step detection
        batch_size: Number of sequences in batch
        min_chars_for_step: Minimum characters before checking for steps
    """

    def __init__(
        self,
        tokenizer,
        start_length: int,
        detector: ThinkingMarkerDetector,
        batch_size: int,
        min_chars_for_step: int = 100,
    ):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.detector = detector
        self.batch_size = batch_size
        self.min_chars_for_step = min_chars_for_step

        # Track state per sequence
        self.finished = [False] * batch_size
        self.step_counts = [0] * batch_size  # Track detected steps per sequence
        self.detected_steps = [
            [] for _ in range(batch_size)
        ]  # Store steps per sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        """Check if new step boundary detected for each sequence in batch."""

        for i in range(min(input_ids.shape[0], self.batch_size)):
            if not self.finished[i]:
                generated_ids = input_ids[i][self.start_length :]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                # Skip if text too short
                if len(generated_text) < self.min_chars_for_step:
                    continue

                # Detect steps in generated text
                current_steps = self.detector.detect_steps(generated_text)

                # New step boundary detected?
                if len(current_steps) > self.step_counts[i]:
                    self.step_counts[i] = len(current_steps)
                    self.detected_steps[i] = current_steps
                    self.finished[i] = True

        # Stop when all sequences have hit a step boundary
        return all(self.finished)

    def get_detected_steps(self, sequence_idx: int = 0) -> list:
        """Get detected steps for a specific sequence."""
        return self.detected_steps[sequence_idx]

    def reset(self):
        """Reset state for new generation."""
        self.finished = [False] * self.batch_size
        self.step_counts = [0] * self.batch_size
        self.detected_steps = [[] for _ in range(self.batch_size)]


class StepCandidateGeneratorThroughHuggingface(StepCandidateGeneratorBase):
    """Generates N candidate next steps for online best-of-n"""

    def __init__(
        self,
        model: WhiteboxModel,
        detector: StructuredStepDetector,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        max_length: int,
        disable_thinking_mode: bool,
        generation_batch_size: int,
        return_generation_scores: bool = False,
    ):
        super().__init__(generation_batch_size)

        self.model = model
        self.detector = detector or StructuredStepDetector()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.device = model.device()
        self.disable_thinking_mode = disable_thinking_mode
        self.return_generation_scores = return_generation_scores

    def generate_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory"""

        log.info(f"Generating {candidates_per_step} candidates from trajectory")

        # Tokenize current trajectory
        tokenizer_signature = inspect.signature(
            self.model.tokenizer.apply_chat_template
        )
        has_enable_thinking = "enable_thinking" in tokenizer_signature.parameters

        # Call tokenizer depending on whether it supports `enable_thinking`
        if has_enable_thinking:
            inputs = self.model.tokenizer.apply_chat_template(
                [request],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(not self.disable_thinking_mode),
            )
        else:
            inputs = self.model.tokenizer.apply_chat_template(
                [request], tokenize=False, add_generation_prompt=True
            )

        # Ensure inputs is a list (some tokenizers return str, some return list)
        if isinstance(inputs, str):
            inputs = [inputs]

        if self.disable_thinking_mode and not has_enable_thinking:
            inputs[0] += "\n<think>\n\n</think>\n\n"

        inputs[0] = inputs[0] + convert_trajectory_to_string(trajectory)

        inputs = self.model.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_length,
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
            f"Generation params: do_sample={gen_params['do_sample']}, "
            f"temp={gen_params['temperature']}, "
            f"top_p={gen_params['top_p']}, "
            f"num_return_sequences={gen_params['num_return_sequences']}"
        )
        log.info(
            f"Model generation_parameters.do_sample: "
            f"{self.model.generation_parameters.do_sample}"
        )
        log.info(
            f"Model generation_parameters.temperature: "
            f"{self.model.generation_parameters.temperature}"
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
            f"After override - do_sample: "
            f"{self.model.generation_parameters.do_sample}, "
            f"temp: {self.model.generation_parameters.temperature}"
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

        # Extract step candidates
        candidates = []
        for i, sequence in enumerate(outputs.sequences):
            # Get newly generated tokens
            new_tokens = sequence[input_length:]
            raw_generated_text = self.model.tokenizer.decode(
                new_tokens, skip_special_tokens=False
            )

            # Extract step using detector
            step_text = self.detector.extract_step_text(raw_generated_text)
            is_complete = self.detector.is_step_complete(raw_generated_text)
            is_trajectory_complete = self.detector.is_trajectory_complete(
                # TODO: does not work even if it generates <end of response>
                raw_generated_text
            )

            # Get generation scores if available
            gen_scores = None
            if (
                self.return_generation_scores
                and hasattr(outputs, "scores")
                and outputs.scores
            ):
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
                # Convert entropy to validity: lower entropy = higher validity
                other_data=(
                    {"uncertainty_score": 1.0 / (1.0 + outputs.uncertainty_score[i])}
                    if hasattr(outputs, "uncertainty_score")
                    else None
                ),
            )
            candidates.append(candidate)

        return candidates

    def generate_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate final answer without step boundary stopping.

        Unlike step generation, answer generation continues until:
        - </think> tag (thinking mode)
        - <end of response> marker
        - \\boxed{} pattern
        - EOS token or max tokens
        """
        # Build input same as generate_candidates but without step stopping
        tokenizer_signature = inspect.signature(
            self.model.tokenizer.apply_chat_template
        )
        has_enable_thinking = "enable_thinking" in tokenizer_signature.parameters

        if has_enable_thinking:
            inputs = self.model.tokenizer.apply_chat_template(
                [request],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(not self.disable_thinking_mode),
            )
        else:
            inputs = self.model.tokenizer.apply_chat_template(
                [request], tokenize=False, add_generation_prompt=True
            )

        if isinstance(inputs, str):
            inputs = [inputs]

        if self.disable_thinking_mode and not has_enable_thinking:
            inputs[0] += "\n<think>\n\n</think>\n\n"

        inputs[0] = inputs[0] + convert_trajectory_to_string(trajectory)

        inputs = self.model.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_length,
        )
        input_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start_time = time.time()

        # Generate WITHOUT step boundary stopping criteria
        # Let model generate until EOS or max tokens
        gen_params = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_return_sequences": candidates_per_step,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.model.tokenizer.eos_token_id,
            "eos_token_id": self.model.tokenizer.eos_token_id,
        }

        log.info(
            f"Generating {candidates_per_step} answer candidates (no step stopping)"
        )

        # Override model's generation parameters
        old_do_sample = self.model.generation_parameters.do_sample
        old_temperature = self.model.generation_parameters.temperature
        old_top_p = self.model.generation_parameters.top_p
        old_top_k = self.model.generation_parameters.top_k

        self.model.generation_parameters.do_sample = True
        self.model.generation_parameters.temperature = self.temperature
        self.model.generation_parameters.top_p = self.top_p
        self.model.generation_parameters.top_k = self.top_k

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        # Restore original parameters
        self.model.generation_parameters.do_sample = old_do_sample
        self.model.generation_parameters.temperature = old_temperature
        self.model.generation_parameters.top_p = old_top_p
        self.model.generation_parameters.top_k = old_top_k

        generation_time = time.time() - start_time
        log.info(f"Generated answer candidates in {generation_time:.2f}s")

        # Extract answer candidates
        candidates = []
        for i, sequence in enumerate(outputs.sequences):
            new_tokens = sequence[input_length:]
            raw_generated_text = self.model.tokenizer.decode(
                new_tokens, skip_special_tokens=False
            )

            # Get generation scores if available
            gen_scores = None
            if (
                self.return_generation_scores
                and hasattr(outputs, "scores")
                and outputs.scores
            ):
                gen_scores = (
                    torch.stack(outputs.scores, dim=1)[i]
                    if i < len(outputs.scores)
                    else None
                )

            candidate = StepCandidate(
                text=raw_generated_text.strip(),
                token_ids=new_tokens.tolist(),
                is_complete=True,
                is_trajectory_complete=True,
                generation_scores=gen_scores,
                raw_text=raw_generated_text,
                # Convert entropy to validity: lower entropy = higher validity
                other_data=(
                    {"uncertainty_score": 1.0 / (1.0 + outputs.uncertainty_score[i])}
                    if hasattr(outputs, "uncertainty_score")
                    else None
                ),
            )
            candidates.append(candidate)

        return candidates
