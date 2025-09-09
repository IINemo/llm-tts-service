"""
Direct ReasonEval scorer that bypasses the stat calculator pipeline for efficient stepwise scoring
"""

import torch
from typing import List, Any, Tuple
import logging
from transformers import AutoTokenizer

from lm_polygraph import WhiteboxModel
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor
from baselines.reasoneval import ReasonEval_7B, ReasonEval_34B
from .base import UncertaintyBasedScorer

log = logging.getLogger(__name__)


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


class DirectReasonEvalScorerOptimized(DirectReasonEvalScorer):
    """
    Optimized version with better batching for multiple candidates.
    
    Additional optimizations:
    1. Batch multiple candidates together when possible
    2. Cache question extraction
    3. Reuse tokenization results
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question_cache = {}
        self.max_cache_size = 100
        
    def compute_claim_uncertainties(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Compute uncertainties with optimized batching"""
        
        self.prepare_model()
        
        if not candidates:
            return []
        
        # Get question (with caching)
        trajectory_hash = hash(trajectory[:200])  # Hash prefix for stability
        if trajectory_hash in self.question_cache:
            question = self.question_cache[trajectory_hash]
        else:
            question = self._extract_question(trajectory)
            # Cache with size limit
            if len(self.question_cache) >= self.max_cache_size:
                self.question_cache.pop(next(iter(self.question_cache)))
            self.question_cache[trajectory_hash] = question
        
        # Process in batches for efficiency
        all_uncertainties = []
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i + self.batch_size]
            batch_uncertainties = self._score_batch(question, trajectory, batch_candidates)
            all_uncertainties.extend(batch_uncertainties)
        
        return all_uncertainties
    
    def _score_batch(
        self,
        question: str,
        trajectory: str,
        candidates: List[str]
    ) -> List[List[float]]:
        """Score a batch of candidates"""
        # For now, fall back to individual scoring
        # (ReasonEval batching would require careful handling of different claim counts)
        uncertainties = []
        for candidate in candidates:
            uncertainties.append(self._score_single_candidate(question, trajectory, candidate))
        return uncertainties
    
    def cleanup(self):
        """Clean up resources including cache"""
        self.question_cache.clear()
        super().cleanup()