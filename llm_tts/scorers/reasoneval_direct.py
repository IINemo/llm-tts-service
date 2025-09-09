"""
Direct ReasonEval scorer that bypasses the stat calculator pipeline for efficient stepwise scoring
"""

import torch
import torch.nn as nn
from typing import List, Any, Tuple
import logging
from transformers import AutoTokenizer

from .base import UncertaintyBasedScorer

from typing import Dict, List, Tuple
from parse import parse
import numpy as np

from lm_polygraph import WhiteboxModel
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model, Claim
from transformers import (
    MistralModel, MistralPreTrainedModel,
    LlamaModel, LlamaPreTrainedModel,
    AutoTokenizer
)
from transformers.configuration_utils import PretrainedConfig

import logging

log = logging.getLogger(__name__)


class StepsExtractor(StatCalculator):
    def __init__(
            self,
            sent_separators: str = "\n",
            skip_starts: list[str] = ['Reasoning Steps:', 'SOLUTION:', '<start of response>','<end of response>'],
            progress_bar: bool = True,
    ):
        super().__init__()
        self.sent_separators = sent_separators
        self.skip_starts = skip_starts
        self.progress_bar = progress_bar

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return (
            [
                "claims",
                "claim_texts_concatenated",
                "claim_input_texts_concatenated",
            ],
            [
                "greedy_texts",
                "greedy_tokens",
            ],
        )

    def __call__(
            self,
            dependencies: Dict[str, object],
            texts: List[str],
            model: WhiteboxModel,
            *args,
            **kwargs,
    ) -> Dict[str, List]:
        claims: list[list[Claim]] = []
        claim_texts_concatenated: list[str] = []
        claim_input_texts_concatenated: list[str] = []

        data = zip(
            texts,
            dependencies["greedy_texts"],
            dependencies["greedy_tokens"],
        )
        if self.progress_bar:
            data = tqdm(data, total=len(texts), desc='Extracting steps')
        for input_text, greedy_text, greedy_tokens in data:
            steps: list[Claim] = self.split_to_steps(greedy_text, greedy_tokens, model.tokenizer)
            claims.append(steps)
            claim_texts_concatenated += [c.claim_text for c in steps]
            claim_input_texts_concatenated += [input_text for c in steps]

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def filter_claim_texts(self, claim_text: str) -> bool:
        claim_text = claim_text.strip()
        return len(claim_text) > 0 and not any(claim_text.lower().startswith(b.lower()) for b in self.skip_starts)

    def split_to_steps(
            self,
            text: str,
            tokens: list[int],
            tokenizer,
    ) -> list[Claim]:
        if not tokenizer.decode(tokens).startswith(text):
            return []
        prev_token_i, token_i = 0, 0
        prev_text_i = 0
        claims: list[Claim] = []
        for text_i in range(len(text)):
            if text[text_i] in self.sent_separators and self.filter_claim_texts(text[prev_text_i:text_i + 1]):
                claims.append(Claim(
                    claim_text=text[prev_text_i:text_i + 1].strip(),
                    sentence=text[prev_text_i:text_i + 1],
                    aligned_token_ids=list(range(prev_token_i, min(token_i + 1, len(tokens) - 1)))
                ))
            while token_i < len(tokens) and tokenizer.decode(tokens[:token_i + 1]) in text[:text_i + 1]:
                token_i += 1
            if text[text_i] in self.sent_separators:
                prev_text_i = text_i + 1
                prev_token_i = token_i
        if self.filter_claim_texts(text[prev_text_i:]):
            claims.append(Claim(
                claim_text=text[prev_text_i:].strip(),
                sentence=text[prev_text_i:],
                aligned_token_ids=list(range(prev_token_i, min(token_i + 1, len(tokens) - 1)))
            ))
        return claims


class ReasonEval_7B(MistralPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = MistralModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dimension, bias=config.use_bias)
        self.post_init()

    def forward(self, input_ids, attention_mask, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        scores = self.score_head(hidden_states)
        return scores


class ReasonEval_34B(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)
        self.post_init()

    def forward(self, input_ids, attention_mask, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        scores = self.score_head(hidden_states)
        return scores


class ReasonEvalStatCalculator(StatCalculator):
    def __init__(
            self,
            prompt_path: str | None = None,
            reasoneval_model_path: str = "GAIR/ReasonEval-7B",
            device: str = "auto",
            offload_to_cpu_between_calls: bool = False,
    ):
        super().__init__()
        self.reasoneval_model_path = reasoneval_model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.prompt = open(prompt_path, 'r').read() if prompt_path else "{q}"
        self.offload_to_cpu_between_calls = offload_to_cpu_between_calls
        if offload_to_cpu_between_calls:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["reasoneval_scores"], ["claims"]

    def init(self):
        if self.model is not None:
            return
        device = "cpu" if self.offload_to_cpu_between_calls else self.device
        log.info(f"Initializing {self.reasoneval_model_path} model on device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.reasoneval_model_path, device_map=device)
        if self.reasoneval_model_path.endswith('7B'):
            self.model = ReasonEval_7B.from_pretrained(self.reasoneval_model_path, device_map=device)
            self.model_size = '7B'
        elif self.reasoneval_model_path.endswith('34B'):
            self.model = ReasonEval_34B.from_pretrained(self.reasoneval_model_path, device_map=device)
            self.model_size = '34B'
        else:
            raise ValueError(f"Could not determine model size from path: {self.reasoneval_model_path}")

    def get_step_level_scores(self, question, reasoning_steps) -> list[dict[str, float]]:
        self.init()
        PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        step_separator = f"{self.tokenizer.pad_token}"
        combined_steps = "".join(step.claim_text + step_separator for step in reasoning_steps)
        prompt = PROMPT_FORMAT.format(input=question)
        tokenized_result = self.tokenizer(prompt + step_separator + combined_steps)['input_ids']
        separator_token_id = self.tokenizer(step_separator)['input_ids'][-1]

        labeled_token_indices = []
        adjusted_token_ids = []
        separator_count = 0
        for idx, token_id in enumerate(tokenized_result):
            if token_id == separator_token_id:
                labeled_token_indices.append(idx - 1 - separator_count)
                separator_count += 1
            else:
                adjusted_token_ids.append(token_id)

        if self.model_size == '7B':
            adjusted_token_ids = [1] + adjusted_token_ids
            adjusted_token_ids = torch.tensor([adjusted_token_ids])
            labeled_token_indices = labeled_token_indices[2:]
        elif self.model_size == '34B':
            adjusted_token_ids = torch.tensor([adjusted_token_ids])
            labeled_token_indices = labeled_token_indices[1:]
        else:
            raise ValueError(f"Invalid model size: {self.model_size}")

        attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
        adjusted_token_ids = adjusted_token_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            reasoning_scores = self.model(adjusted_token_ids, attention_mask)[0, labeled_token_indices, :]
            scores = torch.softmax(reasoning_scores, dim=-1).tolist()

        step_level_validity_scores = [(score[1] + score[2]) for score in scores]
        step_level_redundancy_scores = [score[1] for score in scores]
        return [{'validity': v, 'redundancy': r} for v, r in
                zip(step_level_validity_scores, step_level_redundancy_scores)]

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model, max_new_tokens: int = 100,
                 **kwargs) -> Dict[str, np.ndarray]:
        self.init()
        if self.offload_to_cpu_between_calls:
            log.info(f"Uploading ReasonEval model to {self.device}...")
            self.model = self.model.to(self.device)
            log.info(f"Done.")
        scores: list[list[dict]] = []
        for input_text, claims in zip(texts, dependencies["claims"]):
            question = parse(self.prompt, input_text).named['q']
            r = self.get_step_level_scores(question, claims)
            assert len(r) == len(claims)
            scores.append(r)
        if self.offload_to_cpu_between_calls:
            log.info(f"Offloading ReasonEval model to cpu...")
            self.model = self.model.cpu()
            log.info(f"Done.")
        return {"reasoneval_scores": scores}


class DirectReasonEvalScorer(UncertaintyBasedScorer):
    """
    Direct ReasonEval scorer that applies ReasonEval without stat calculator pipeline.
    
    This implementation:
    1. Extracts claims/steps from candidates
    2. Formats them for ReasonEval evaluation
    3. Computes validity and redundancy scores directly
    4. Returns uncertainty scores based on validity/redundancy
    
    Much cleaner and more efficient than going through the full pipeline.
    """
    
    def __init__(
        self,
        model: WhiteboxModel,
        reasoneval_model_path: str = "GAIR/ReasonEval-7B",
        device: str = "cuda",
        batch_size: int = 8,
        aggregation: str = "default",  # "default", "validity", "redundancy"
        prompt_template: str = None
    ):
        super().__init__("DirectReasonEval")
        self.model = model
        self.reasoneval_model_path = reasoneval_model_path
        self.device = device
        self.batch_size = batch_size
        self.aggregation = aggregation
        self.prompt_template = prompt_template or "Question: {q}\\nLet's solve this step by step.\\n\\n"
        self.reasoneval_model = None
        self.reasoneval_tokenizer = None
        self.model_size = None
        self.steps_extractor = StepsExtractor(progress_bar=False)
        
    def prepare_model(self):
        """Load ReasonEval model and tokenizer"""
        if self.reasoneval_model is None:
            log.info(f"Loading ReasonEval model from {self.reasoneval_model_path}")
            
            # Load tokenizer
            self.reasoneval_tokenizer = AutoTokenizer.from_pretrained(
                self.reasoneval_model_path
            )
            
            # Load appropriate model based on size
            if self.reasoneval_model_path.endswith('7B'):
                self.reasoneval_model = ReasonEval_7B.from_pretrained(
                    self.reasoneval_model_path, 
                    device_map=self.device,
                    trust_remote_code=True
                ).eval()
                self.model_size = '7B'
            elif self.reasoneval_model_path.endswith('34B'):
                self.reasoneval_model = ReasonEval_34B.from_pretrained(
                    self.reasoneval_model_path, 
                    device_map=self.device,
                    trust_remote_code=True
                ).eval()
                self.model_size = '34B'
            else:
                raise ValueError(f"Could not determine model size from path: {self.reasoneval_model_path}")
            
    def cleanup(self):
        """Free ReasonEval model memory"""
        if self.reasoneval_model is not None:
            del self.reasoneval_model
            self.reasoneval_model = None
            del self.reasoneval_tokenizer
            self.reasoneval_tokenizer = None
            torch.cuda.empty_cache()
    
    def compute_claim_uncertainties(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Compute uncertainty scores for claims in each candidate.
        
        Args:
            trajectory: Current trajectory text
            candidates: List of candidate next steps
            
        Returns:
            List of claim uncertainty lists (one per candidate)
        """
        self.prepare_model()
        
        if not candidates:
            return []
        
        # Extract question from trajectory
        question = self._extract_question(trajectory)
        
        # Score all candidates
        all_uncertainties = []
        
        for candidate in candidates:
            try:
                uncertainties = self._score_single_candidate(question, trajectory, candidate)
                all_uncertainties.append(uncertainties)
            except Exception as e:
                log.warning(f"Failed to score candidate: {e}")
                all_uncertainties.append([0.5])  # Neutral uncertainty
            
            # Clean up memory after each candidate
            torch.cuda.empty_cache()
        
        return all_uncertainties
    
    def _extract_question(self, trajectory: str) -> str:
        """Extract the original question from the trajectory"""
        # Look for common patterns that indicate end of question
        end_patterns = [
            "Reasoning Steps:",
            "Solution:",
            "Answer:",
            "- Step"
        ]
        
        question = trajectory
        for pattern in end_patterns:
            if pattern in trajectory:
                parts = trajectory.split(pattern)
                if parts[0].strip():
                    question = parts[0].strip()
                    break
        
        # Remove any system prompts if present
        if "<|im_start|>" in question:
            # Extract content between user tags
            start = question.find("<|im_start|>user")
            end = question.find("<|im_end|>", start)
            if start != -1 and end != -1:
                question = question[start+len("<|im_start|>user"):end].strip()
        
        return question
    
    def _score_single_candidate(
        self, 
        question: str, 
        trajectory: str,
        candidate: str
    ) -> List[float]:
        """Score a single candidate using ReasonEval"""
        
        # Extract claims from candidate
        try:
            candidate_tokens = self.model.tokenize([candidate])
            if candidate_tokens is None or 'input_ids' not in candidate_tokens:
                log.warning(f"Failed to tokenize candidate: {candidate[:50]}...")
                return [0.5]
                
            claims = self.steps_extractor.split_to_steps(
                candidate,
                candidate_tokens['input_ids'][0],
                self.model.tokenizer
            )
            
            if not claims:
                log.debug(f"No claims extracted from candidate: {candidate[:50]}...")
                return [0.5]
                
        except Exception as e:
            log.warning(f"Error extracting claims: {e}")
            return [0.5]
        
        # Get ReasonEval scores
        try:
            scores = self._compute_reasoneval_scores(question, claims)
            return scores if scores else [0.5]
        except Exception as e:
            log.warning(f"Error computing ReasonEval scores: {e}")
            return [0.5]
    
    def _compute_reasoneval_scores(self, question: str, claims: List[Any]) -> List[float]:
        """Compute ReasonEval uncertainty scores for claims"""
        
        if not claims:
            return []
        
        # Format input for ReasonEval
        PROMPT_FORMAT = "Question:\\n{input}\\nAnswer:\\nLet's think step by step.\\n"
        step_separator = f"{self.reasoneval_tokenizer.pad_token}"
        
        # Combine all claims with separator
        combined_steps = "".join(claim.claim_text + step_separator for claim in claims)
        prompt = PROMPT_FORMAT.format(input=question)
        
        # Tokenize with separators
        tokenized_result = self.reasoneval_tokenizer(prompt + step_separator + combined_steps)['input_ids']
        separator_token_id = self.reasoneval_tokenizer(step_separator)['input_ids'][-1]
        
        # Find labeled token indices (tokens right before separators)
        labeled_token_indices = []
        adjusted_token_ids = []
        separator_count = 0
        
        for idx, token_id in enumerate(tokenized_result):
            if token_id == separator_token_id:
                labeled_token_indices.append(idx - 1 - separator_count)
                separator_count += 1
            else:
                adjusted_token_ids.append(token_id)
        
        # Adjust for model-specific formatting
        if self.model_size == '7B':
            adjusted_token_ids = [1] + adjusted_token_ids
            adjusted_token_ids = torch.tensor([adjusted_token_ids])
            labeled_token_indices = labeled_token_indices[2:]  # Skip first two
        elif self.model_size == '34B':
            adjusted_token_ids = torch.tensor([adjusted_token_ids])
            labeled_token_indices = labeled_token_indices[1:]  # Skip first one
        else:
            raise ValueError(f"Invalid model size: {self.model_size}")
        
        # Create attention mask and move to device
        attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
        adjusted_token_ids = adjusted_token_ids.to(self.reasoneval_model.device)
        attention_mask = attention_mask.to(self.reasoneval_model.device)
        
        # Get model outputs
        with torch.no_grad():
            reasoning_scores = self.reasoneval_model(
                adjusted_token_ids, 
                attention_mask
            )[0, labeled_token_indices, :]
            scores = torch.softmax(reasoning_scores, dim=-1).tolist()
        
        # Convert to uncertainties based on aggregation strategy
        uncertainties = []
        for score in scores:
            validity = score[1] + score[2]  # Classes 1 and 2
            redundancy = score[1]  # Class 1 only
            
            uncertainty = self._aggregate_to_uncertainty(validity, redundancy)
            uncertainties.append(uncertainty)
        
        return uncertainties
    
    def _aggregate_to_uncertainty(self, validity: float, redundancy: float) -> float:
        """Convert validity and redundancy to uncertainty score"""
        
        if self.aggregation == 'validity':
            # Lower validity = higher uncertainty
            return 1.0 - validity
        elif self.aggregation == 'redundancy':
            # Higher redundancy = higher uncertainty
            return redundancy
        else:
            # Default: combine both (high redundancy + low validity = high uncertainty)
            # This follows the pattern from the original implementation
            return redundancy - validity + 0.5  # Shift to roughly [0, 1] range


class DirectReasonEvalScorerSeparate(DirectReasonEvalScorer):
    """
    ReasonEval scorer that returns validity and redundancy scores separately.
    
    This allows running separate evaluations for validity-based and redundancy-based selection.
    """
    
    def compute_validity_redundancy_scores(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Compute validity and redundancy scores separately for claims in each candidate.
        
        Args:
            trajectory: Current trajectory text
            candidates: List of candidate next steps
            
        Returns:
            Tuple of (validity_scores, redundancy_scores), each a list of claim score lists
        """
        self.prepare_model()
        
        if not candidates:
            return [], []
        
        # Extract question from trajectory
        question = self._extract_question(trajectory)
        
        # Score all candidates
        all_validities = []
        all_redundancies = []
        
        for candidate in candidates:
            try:
                validities, redundancies = self._score_single_candidate_separate(
                    question, trajectory, candidate
                )
                all_validities.append(validities)
                all_redundancies.append(redundancies)
            except Exception as e:
                log.warning(f"Failed to score candidate: {e}")
                all_validities.append([0.5])  # Neutral scores
                all_redundancies.append([0.5])
            
            # Clean up memory after each candidate
            torch.cuda.empty_cache()
        
        return all_validities, all_redundancies
    
    def _score_single_candidate_separate(
        self, 
        question: str, 
        trajectory: str,
        candidate: str
    ) -> Tuple[List[float], List[float]]:
        """Score a single candidate and return validity/redundancy separately"""
        
        # Extract claims from candidate
        try:
            candidate_tokens = self.model.tokenize([candidate])
            if candidate_tokens is None or 'input_ids' not in candidate_tokens:
                log.warning(f"Failed to tokenize candidate: {candidate[:50]}...")
                return [0.5], [0.5]
                
            claims = self.steps_extractor.split_to_steps(
                candidate,
                candidate_tokens['input_ids'][0],
                self.model.tokenizer
            )
            
            if not claims:
                log.debug(f"No claims extracted from candidate: {candidate[:50]}...")
                return [0.5], [0.5]
                
        except Exception as e:
            log.warning(f"Error extracting claims: {e}")
            return [0.5], [0.5]
        
        # Get ReasonEval scores
        try:
            validities, redundancies = self._compute_reasoneval_scores_separate(question, claims)
            return (validities if validities else [0.5], 
                    redundancies if redundancies else [0.5])
        except Exception as e:
            log.warning(f"Error computing ReasonEval scores: {e}")
            return [0.5], [0.5]
    
    def _compute_reasoneval_scores_separate(
        self, 
        question: str, 
        claims: List[Any]
    ) -> Tuple[List[float], List[float]]:
        """Compute ReasonEval scores and return validity/redundancy separately"""
        
        if not claims:
            return [], []
        
        # Format input for ReasonEval
        PROMPT_FORMAT = "Question:\\n{input}\\nAnswer:\\nLet's think step by step.\\n"
        step_separator = f"{self.reasoneval_tokenizer.pad_token}"
        
        # Combine all claims with separator
        combined_steps = "".join(claim.claim_text + step_separator for claim in claims)
        prompt = PROMPT_FORMAT.format(input=question)
        
        # Tokenize with separators
        tokenized_result = self.reasoneval_tokenizer(prompt + step_separator + combined_steps)['input_ids']
        separator_token_id = self.reasoneval_tokenizer(step_separator)['input_ids'][-1]
        
        # Find labeled token indices (tokens right before separators)
        labeled_token_indices = []
        adjusted_token_ids = []
        separator_count = 0
        
        for idx, token_id in enumerate(tokenized_result):
            if token_id == separator_token_id:
                labeled_token_indices.append(idx - 1 - separator_count)
                separator_count += 1
            else:
                adjusted_token_ids.append(token_id)
        
        # Adjust for model-specific formatting
        if self.model_size == '7B':
            adjusted_token_ids = [1] + adjusted_token_ids
            adjusted_token_ids = torch.tensor([adjusted_token_ids])
            labeled_token_indices = labeled_token_indices[2:]  # Skip first two
        elif self.model_size == '34B':
            adjusted_token_ids = torch.tensor([adjusted_token_ids])
            labeled_token_indices = labeled_token_indices[1:]  # Skip first one
        else:
            raise ValueError(f"Invalid model size: {self.model_size}")
        
        # Create attention mask and move to device
        attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
        adjusted_token_ids = adjusted_token_ids.to(self.reasoneval_model.device)
        attention_mask = attention_mask.to(self.reasoneval_model.device)
        
        # Get model outputs
        with torch.no_grad():
            reasoning_scores = self.reasoneval_model(
                adjusted_token_ids, 
                attention_mask
            )[0, labeled_token_indices, :]
            scores = torch.softmax(reasoning_scores, dim=-1).tolist()
        
        # Extract validity and redundancy scores
        validities = []
        redundancies = []
        for score in scores:
            validity = score[1] + score[2]  # Classes 1 and 2
            redundancy = score[1]  # Class 1 only
            validities.append(validity)
            redundancies.append(redundancy)
        
        return validities, redundancies
