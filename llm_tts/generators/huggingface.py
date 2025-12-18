"""
HuggingFace-based candidate step generation for online best-of-n.

This module provides step generation using HuggingFace transformers, with two modes:

1. **Thinking Mode**: For models with <think>...</think> reasoning blocks.
   Uses semantic step boundary detection via ThinkingMarkerDetector.
   Matches vLLM generator behavior for consistency.

2. **Structured Mode**: For models with explicit <step N> markers.
   Uses StructuredStepDetector for boundary detection.

Key classes:
- ThinkingStepStoppingCriteria: HuggingFace stopping criteria matching vLLM stop_tokens
- StepCandidateGeneratorThroughHuggingface: Main generator class
"""

import inspect
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from llm_tts.utils.flops import FLOPCalculator
from lm_polygraph import WhiteboxModel
from transformers import StoppingCriteria, StoppingCriteriaList

from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.step_boundary_detectors import (
    StepBoundaryDetectorBase,
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
    Stopping criteria for thinking mode that matches vLLM's stop_tokens behavior.

    This criteria implements the same logic as vLLM's SamplingParams with stop_tokens:
    - Won't stop until at least `min_tokens` are generated
    - After min_tokens, stops when any stop_string is NEWLY generated
    - Stop strings that appeared before min_tokens are ignored

    Example:
        If text is "<think>\\nOkay, let's solve this. So, first..." and min_tokens=50:
        - "\\nOkay," at position 7 is ignored (generated before min_tokens)
        - "\\nSo," at position 35 would trigger stop (generated after min_tokens)

    Args:
        tokenizer: HuggingFace tokenizer for decoding generated ids
        start_length: Input sequence length (to extract only new tokens)
        stop_strings: List of strings that trigger stopping (from detector.get_vllm_stop_tokens())
        min_tokens: Minimum tokens before stopping is allowed (matches vLLM min_tokens)
    """

    def __init__(
        self,
        tokenizer,
        start_length: int,
        stop_strings: List[str],
        min_tokens: int = 0,
    ):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.stop_strings = stop_strings
        self.min_tokens = min_tokens

        # State tracking
        self.finished = False
        self.matched_stop_string: Optional[str] = None
        self.stop_position = -1

        # Track text length at min_tokens threshold
        # Only stop strings appearing AFTER this position will trigger stopping
        self.min_tokens_text_len = -1

        # Cache max stop string length to avoid recomputing on every call
        self._max_stop_len = max(len(s) for s in stop_strings) if stop_strings else 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        """Check if generation should stop (called after each token)."""
        if self.finished:
            return True

        generated_ids = input_ids[0][self.start_length :]
        num_tokens = len(generated_ids)

        # Phase 1: Wait for min_tokens
        if num_tokens < self.min_tokens:
            return False

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Phase 2: Record baseline text length when min_tokens first reached
        if self.min_tokens_text_len < 0:
            self.min_tokens_text_len = len(generated_text)
            return False  # Don't stop on the exact token that reaches min_tokens

        # Phase 3: Search for stop strings only in new content
        # Start search slightly before min_tokens_text_len to catch stop strings
        # that span the boundary (e.g., "\nSo," where "\n" was before min_tokens)
        search_start = max(0, self.min_tokens_text_len - self._max_stop_len)

        # Find earliest matching stop string
        earliest_pos = -1
        earliest_stop = None
        for stop_str in self.stop_strings:
            pos = generated_text.find(stop_str, search_start)
            if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                earliest_pos = pos
                earliest_stop = stop_str

        if earliest_pos != -1:
            self.finished = True
            self.matched_stop_string = earliest_stop
            self.stop_position = earliest_pos
            return True

        return False

    def get_truncated_text(self, full_text: str) -> str:
        """
        Truncate text at stop string position (excluding the stop string).

        This matches vLLM behavior where output.text excludes the matched stop string.
        """
        if self.stop_position >= 0:
            return full_text[: self.stop_position]
        return full_text


class StepCandidateGeneratorThroughHuggingface(StepCandidateGeneratorBase):
    """Generates N candidate next steps for online best-of-n.

    Supports two modes:
    - Thinking mode (disable_thinking_mode=False): Uses ThinkingMarkerDetector
      to detect semantic step boundaries based on thinking patterns.
    - Structured mode (disable_thinking_mode=True): Uses StructuredStepDetector
      to detect explicit step markers like <step N>.
    """

    def __init__(
        self,
        model: WhiteboxModel,
        detector: StepBoundaryDetectorBase,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        max_length: int,
        disable_thinking_mode: bool,
        generation_batch_size: int,
        return_generation_scores: bool = False,
        flop_calculator: Optional["FLOPCalculator"] = None,
    ):
        # Check thinking mode first to determine batch size
        use_thinking_mode = not disable_thinking_mode and isinstance(
            detector, ThinkingMarkerDetector
        )
        # Thinking mode generates sequentially, use large batch size to avoid
        # going through _generate_candidates_in_batches (matches vLLM behavior)
        effective_batch_size = 1024 if use_thinking_mode else generation_batch_size
        super().__init__(effective_batch_size, flop_calculator=flop_calculator)

        self.model = model
        self.detector = detector
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.device = model.device()
        self.disable_thinking_mode = disable_thinking_mode
        self.return_generation_scores = return_generation_scores
        self.use_thinking_mode = use_thinking_mode

        # Thinking mode specific settings
        if self.use_thinking_mode:
            self.min_step_tokens = getattr(detector, "min_step_tokens", 50)
            self.max_step_tokens = getattr(detector, "max_step_tokens", 300)
            self.answer_patterns = getattr(
                detector, "answer_patterns", ["</think>", "<Answer>:"]
            )
            # Get stop tokens from detector (same as vLLM)
            self.thinking_stop_tokens = detector.get_vllm_stop_tokens()
            if "</think>" not in self.thinking_stop_tokens:
                self.thinking_stop_tokens.append("</think>")

    # =========================================================================
    # Common utilities
    # =========================================================================

    def _tokenize(
        self,
        prompt: str,
        trajectory: Optional[List[StepCandidate]] = None,
    ) -> tuple[Dict[str, torch.Tensor], int]:
        """
        Tokenize prompt (with optional trajectory) and prepare for generation.

        Args:
            prompt: Prompt string to tokenize
            trajectory: Optional trajectory to append before tokenizing

        Returns:
            Tuple of (inputs dict, input_length)
        """
        if trajectory:
            prompt = prompt + convert_trajectory_to_string(trajectory)

        inputs = self.model.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_length,
        )
        input_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs, input_length

    def _get_base_gen_params(
        self,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        output_scores: Optional[bool] = None,
    ) -> Dict:
        """
        Get base generation parameters used by all generation methods.

        Args:
            max_new_tokens: Maximum tokens to generate in this call
            num_return_sequences: Number of sequences to generate (default: 1)
            output_scores: Whether to output scores. If None, uses self.return_generation_scores
        """
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_return_sequences": num_return_sequences,
            "output_scores": (
                output_scores
                if output_scores is not None
                else self.return_generation_scores
            ),
            "return_dict_in_generate": True,
            "pad_token_id": self.model.tokenizer.eos_token_id,
            "eos_token_id": self.model.tokenizer.eos_token_id,
        }

    def _apply_chat_template(
        self,
        request: List[Dict[str, str]],
        enable_thinking: bool = True,
    ) -> str:
        """
        Apply chat template to request with enable_thinking support.

        Args:
            request: Chat messages in OpenAI format
            enable_thinking: Whether to enable thinking mode (if tokenizer supports it)

        Returns:
            Formatted prompt string
        """
        tokenizer_signature = inspect.signature(
            self.model.tokenizer.apply_chat_template
        )
        has_enable_thinking = "enable_thinking" in tokenizer_signature.parameters

        if has_enable_thinking:
            result = self.model.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            result = self.model.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Some tokenizers return a list, ensure we return a string
        if isinstance(result, list):
            result = result[0]

        # Force-close thinking when disabled but tokenizer doesn't support enable_thinking
        if self.disable_thinking_mode and not has_enable_thinking:
            result += "<think>\n\n</think>\n\n"

        return result

    def _get_generation_scores(self, outputs, index: int) -> Optional[torch.Tensor]:
        """Extract generation scores for a sequence from outputs."""
        if (
            self.return_generation_scores
            and hasattr(outputs, "scores")
            and outputs.scores
        ):
            return (
                torch.stack(outputs.scores, dim=1)[index]
                if index < len(outputs.scores)
                else None
            )
        return None

    def _get_uncertainty_data(self, outputs, index: int) -> Optional[Dict]:
        """Extract uncertainty data for a sequence from outputs."""
        if hasattr(outputs, "uncertainty_score"):
            return {
                "validity_score": 1.0 / (1.0 + outputs.uncertainty_score[index]),
                "uncertainty_score": outputs.uncertainty_score[index],
            }
        return None

    # =========================================================================
    # Thinking mode helper methods
    # =========================================================================

    def _build_prompt(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> str:
        """Build prompt from request and trajectory for thinking mode."""
        base_prompt = self._apply_chat_template(request, enable_thinking=True)

        if trajectory:
            return base_prompt + convert_trajectory_to_string(trajectory)

        return base_prompt

    def _is_thinking_complete(self, text: str) -> bool:
        """Check if thinking phase is complete (contains </think>)."""
        return "</think>" in text

    def _is_valid_step_boundary(self, text: str, num_tokens: int) -> bool:
        """Check if text represents a valid step boundary."""
        if num_tokens < self.min_step_tokens:
            return False
        text = text.strip()
        return bool(text) and text[-1] in ".!?\n"

    def _generate_single_thinking_step(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> StepCandidate:
        """
        Generate a single thinking step.

        Generates text until a stop_string is encountered (after min_step_tokens).
        Returns StepCandidate with truncated text (stop string excluded).

        NOTE: Currently we don't validate that the step ends with proper punctuation
        (see _is_valid_step_boundary). If needed, a continuation loop could be added
        to retry generation when boundary is invalid, but this requires deciding how
        to aggregate uncertainty scores across multiple generation attempts (e.g.,
        average, last, or max uncertainty).
        """
        prompt = self._build_prompt(request, trajectory)
        inputs, input_length = self._tokenize(prompt)

        # Create stopping criteria
        stopping_criteria = ThinkingStepStoppingCriteria(
            tokenizer=self.model.tokenizer,
            start_length=input_length,
            stop_strings=self.thinking_stop_tokens,
            min_tokens=self.min_step_tokens,
        )

        # Build generation params - limit to max_step_tokens for thinking steps
        gen_params = self._get_base_gen_params(self.max_step_tokens)
        gen_params["stopping_criteria"] = StoppingCriteriaList([stopping_criteria])

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        # Extract generated text
        new_tokens = outputs.sequences[0][input_length:]
        generated_text = self.model.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        # Truncate at stop string if matched (like vLLM - excludes stop string)
        if stopping_criteria.matched_stop_string:
            truncated_part = generated_text[stopping_criteria.stop_position :]
            log.debug(
                f"Step boundary: matched '{stopping_criteria.matched_stop_string}' "
                f"at pos {stopping_criteria.stop_position}, truncated: '{truncated_part}'"
            )
            generated_text = stopping_criteria.get_truncated_text(generated_text)
            # Re-tokenize truncated text to get correct token count
            token_ids = self.model.tokenizer.encode(
                generated_text, add_special_tokens=False
            )
        else:
            token_ids = new_tokens.tolist()

        # Check if thinking is complete (</think> found)
        thinking_complete = self._is_thinking_complete(generated_text)
        if thinking_complete:
            think_pos = generated_text.find("</think>")
            generated_text = generated_text[: think_pos + len("</think>")]
            # Re-tokenize to get correct token_ids after truncation
            token_ids = self.model.tokenizer.encode(
                generated_text, add_special_tokens=False
            )

        # Get uncertainty from model output
        other_data = self._get_uncertainty_data(outputs, 0)

        return StepCandidate(
            text=generated_text.strip(),
            token_ids=token_ids,
            is_complete=True,
            is_trajectory_complete=thinking_complete,
            generation_scores=None,
            raw_text=generated_text,
            other_data=other_data,
        )

    def _generate_thinking_response(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
    ) -> StepCandidate:
        """
        Generate response after thinking phase (after </think>).

        This method generates the final answer after the model has completed
        its thinking process. It continues generation until EOS or max tokens.
        """
        prompt = self._build_prompt(request, trajectory)
        inputs, input_length = self._tokenize(prompt)

        gen_params = self._get_base_gen_params(self.max_new_tokens)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        # Extract generated text
        new_tokens = outputs.sequences[0][input_length:]
        generated_text = self.model.tokenizer.decode(
            new_tokens, skip_special_tokens=False
        )

        # Ensure answer pattern is present
        has_answer_pattern = any(p in generated_text for p in self.answer_patterns)
        if not has_answer_pattern:
            generated_text = generated_text + self.answer_patterns[0]

        # Get uncertainty from model output
        other_data = self._get_uncertainty_data(outputs, 0)

        return StepCandidate(
            text=generated_text.strip(),
            token_ids=new_tokens.tolist(),
            is_complete=True,
            is_trajectory_complete=True,
            generation_scores=None,
            raw_text=generated_text,
            other_data=other_data,
        )

    # =========================================================================
    # Main generation methods
    # =========================================================================

    def generate_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory."""

        # Thinking mode: sequential generation like vLLM
        if self.use_thinking_mode:
            return self._generate_thinking_candidates(
                request, trajectory, candidates_per_step
            )
        else:
            # Structured mode: batch generation
            return self._generate_structured_candidates(
                request, trajectory, candidates_per_step
            )

    def _generate_thinking_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate candidates in thinking mode (sequential, like vLLM)."""
        trajectory = trajectory or []
        full_trajectory = convert_trajectory_to_string(trajectory)
        thinking_complete = "</think>" in full_trajectory

        candidates = []

        if thinking_complete:
            log.info("Thinking complete, generating response")
            for _ in range(candidates_per_step):
                candidate = self._generate_thinking_response(request, trajectory)
                candidates.append(candidate)
        else:
            for _ in range(candidates_per_step):
                candidate = self._generate_single_thinking_step(request, trajectory)
                candidates.append(candidate)

        return candidates

    def _generate_structured_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate candidates in structured mode (batch generation)."""
        # Build prompt and tokenize
        base_prompt = self._apply_chat_template(request, enable_thinking=False)
        inputs, input_length = self._tokenize(base_prompt, trajectory)

        # Create stopping criteria for structured mode
        stopping_criteria = BatchStepStoppingCriteria(
            tokenizer=self.model.tokenizer,
            start_length=input_length,
            detector=self.detector,
            batch_size=candidates_per_step,
        )

        # Build generation parameters (structured mode always needs scores)
        gen_params = self._get_base_gen_params(
            self.max_new_tokens, candidates_per_step, output_scores=True
        )
        gen_params["stopping_criteria"] = StoppingCriteriaList([stopping_criteria])

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        # Extract step candidates
        candidates = []
        for i, sequence in enumerate(outputs.sequences):
            new_tokens = sequence[input_length:]
            raw_generated_text = self.model.tokenizer.decode(
                new_tokens, skip_special_tokens=False
            )

            step_text = self.detector.extract_step_text(raw_generated_text)

            # Ensure step text starts with "- Step" marker (remove any continuation text before it)
            step_marker_pos = step_text.find("- Step")
            if step_marker_pos > 0:
                step_text = step_text[step_marker_pos:]

            # TODO: Recalculate uncertainty score based on truncated tokens only.
            # Currently uncertainty is computed on ALL generated tokens (including next step marker
            # tokens that get truncated). This affects both:
            # - Structured mode: tokens before "- Step" marker are truncated here
            # - Thinking mode: tokens after step boundary marker are truncated in detector.extract_step_text()
            # The uncertainty data from lm-polygraph (greedy_log_probs, greedy_logits, hidden_states)
            # is available in other_data and could be used to recalculate scores for truncated tokens.
            candidate = StepCandidate(
                text=step_text,
                token_ids=new_tokens.tolist(),
                is_complete=self.detector.is_step_complete(raw_generated_text),
                is_trajectory_complete=self.detector.is_trajectory_complete(
                    raw_generated_text
                ),
                generation_scores=self._get_generation_scores(outputs, i),
                raw_text=raw_generated_text,
                other_data=self._get_uncertainty_data(outputs, i),
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
        # Thinking mode: sequential generation like vLLM
        if self.use_thinking_mode:
            return self._generate_thinking_answer_candidates(
                request, trajectory, candidates_per_step
            )
        else:
            # Structured mode: batch generation
            return self._generate_structured_answer_candidates(
                request, trajectory, candidates_per_step
            )

    def _generate_thinking_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate answer candidates in thinking mode (sequential, like vLLM)."""
        # Ensure thinking is complete before generating response
        full_trajectory = convert_trajectory_to_string(trajectory)
        if "</think>" not in full_trajectory:
            log.warning(
                "generate_answer_candidates called without </think> in trajectory. "
                "Adding </think> to close thinking phase."
            )
            close_thinking_step = StepCandidate(
                text="</think>\n\n",
                token_ids=[],
                is_complete=True,
                is_trajectory_complete=True,
            )
            trajectory = list(trajectory) + [close_thinking_step]

        candidates = []
        for _ in range(candidates_per_step):
            candidate = self._generate_thinking_response(request, trajectory)
            candidates.append(candidate)
        return candidates

    def _generate_structured_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate answer candidates in structured mode (batch generation)."""
        base_prompt = self._apply_chat_template(request, enable_thinking=False)
        inputs, input_length = self._tokenize(base_prompt, trajectory)

        # Generate WITHOUT step boundary stopping criteria (answer continues to EOS)
        gen_params = self._get_base_gen_params(
            self.max_new_tokens, candidates_per_step, output_scores=True
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        # Extract answer candidates
        candidates = []
        for i, sequence in enumerate(outputs.sequences):
            new_tokens = sequence[input_length:]
            raw_generated_text = self.model.tokenizer.decode(
                new_tokens, skip_special_tokens=False
            )
            # Clean text: skip special tokens for display
            clean_text = self.model.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            candidate = StepCandidate(
                text=clean_text.strip(),
                token_ids=new_tokens.tolist(),
                is_complete=True,
                is_trajectory_complete=True,
                generation_scores=self._get_generation_scores(outputs, i),
                raw_text=raw_generated_text,
                other_data=self._get_uncertainty_data(outputs, i),
            )
            candidates.append(candidate)

        return candidates
