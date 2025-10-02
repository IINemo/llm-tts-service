"""
Self-consistency strategy for LLM reasoning.

Based on "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
by Wang et al. (2022). Generates multiple diverse reasoning paths and selects
the most consistent answer via majority voting.
"""

import torch
from typing import List, Dict, Any, Optional
import logging
import numpy as np

from llm_tts.scorers.majority_voting import ChainMajorityVotingScorer
from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategySelfConsistency(StrategyBase):
    """
    Self-consistency strategy that generates multiple reasoning paths
    and selects the most consistent answer via majority voting.
    """

    def __init__(
        self,
        model,
        num_paths: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        generation_batch_size: int = None,
        scorer: Optional[Any] = None
    ):
        """
        Initialize self-consistency strategy.

        Args:
            model: Language model for generation
            num_paths: Number of reasoning paths to generate
            max_new_tokens: Maximum tokens per reasoning path
            temperature: Sampling temperature (> 0 for diversity)
            generation_batch_size: Batch size for generation (None = all at once)
            scorer: Custom scorer for answer selection (defaults to majority voting)
        """
        self.model = model
        self.num_paths = num_paths
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size or num_paths

        # Use majority voting scorer by default
        self.scorer = scorer or ChainMajorityVotingScorer()
        if hasattr(self.scorer, 'prepare_model'):
            self.scorer.prepare_model()

    def generate_reasoning_paths(self, prompt: str) -> List[str]:
        """
        Generate multiple diverse reasoning paths for the given prompt.

        Args:
            prompt: Input prompt/question

        Returns:
            List of complete reasoning paths (prompt + generated reasoning)
        """
        log.info(f"Generating {self.num_paths} reasoning paths with temperature {self.temperature}")

        all_paths = []

        # Check if this is an API model (has generate method directly)
        if hasattr(self.model, 'api_model') or self.model.device == "api":
            log.info("Using API model for generation")
            # Use API model directly
            for i in range(self.num_paths):
                log.info(f"Generating path {i+1}/{self.num_paths}")

                try:
                    if hasattr(self.model, 'generate'):
                        # Direct API model
                        completions = self.model.generate(
                            prompt=prompt,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            num_return_sequences=1
                        )
                        generated_text = completions[0] if completions else ""
                    else:
                        # API model wrapped in adapter
                        completions = self.model.api_model.generate(
                            prompt=prompt,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            num_return_sequences=1
                        )
                        generated_text = completions[0] if completions else ""

                    # Combine with original prompt
                    full_path = prompt + generated_text
                    all_paths.append(full_path)

                except Exception as e:
                    log.warning(f"Failed to generate path {i+1}: {e}")
                    # Add fallback path
                    all_paths.append(prompt + f" [Generation failed: {str(e)}]")

        else:
            # Use local model with batched generation
            log.info("Using local model for generation")
            # Generate in batches if needed to manage memory
            num_batches = (self.num_paths + self.generation_batch_size - 1) // self.generation_batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.generation_batch_size
                end_idx = min((batch_idx + 1) * self.generation_batch_size, self.num_paths)
                batch_size = end_idx - start_idx

                log.info(f"Generating batch {batch_idx + 1}/{num_batches} ({batch_size} paths)")

                # Tokenize prompt
                inputs = self.model.tokenize([prompt] * batch_size)
                input_ids = inputs['input_ids'].to(self.model.device)
                attention_mask = inputs['attention_mask'].to(self.model.device)

                # Generate reasoning paths
                with torch.no_grad():
                    outputs = self.model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        num_return_sequences=1,  # Generate 1 per input in batch
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Reduce repetition
                        length_penalty=1.0
                    )

                # Decode generated paths
                batch_paths = []
                for i in range(batch_size):
                    output_seq = outputs[i]
                    # Extract only the newly generated tokens
                    new_tokens = output_seq[input_ids.shape[1]:]
                    generated_text = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)

                    # Combine with original prompt to get full reasoning path
                    full_path = prompt + generated_text
                    batch_paths.append(full_path)

                all_paths.extend(batch_paths)

                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        log.info(f"Generated {len(all_paths)} reasoning paths")
        return all_paths

    def select_best_answer(self, reasoning_paths: List[str]) -> Dict[str, Any]:
        """
        Select the best answer using majority voting across reasoning paths.

        Args:
            reasoning_paths: List of complete reasoning paths

        Returns:
            Dictionary containing:
                - best_path: The reasoning path with the most consistent answer
                - best_answer: The extracted answer
                - consensus_score: Confidence based on answer frequency
                - all_answers: All extracted answers for debugging
                - answer_distribution: Answer frequency distribution
        """
        if not reasoning_paths:
            return {
                "best_path": "",
                "best_answer": "no_answer",
                "consensus_score": 0.0,
                "all_answers": [],
                "answer_distribution": {}
            }

        # Use the scorer to get consensus scores
        scores = self.scorer.score_complete_chains(reasoning_paths)

        # Find the path with highest consensus
        best_idx = np.argmax(scores)
        best_path = reasoning_paths[best_idx]
        best_score = scores[best_idx]

        # Extract the best answer
        best_answer = self.scorer.extract_answer(best_path)

        # Get all answers for analysis
        all_answers = [self.scorer.extract_answer(path) for path in reasoning_paths]

        # Calculate answer distribution
        from collections import Counter
        answer_counts = Counter(all_answers)

        log.info(f"Selected reasoning path {best_idx} with consensus score {best_score:.3f}")
        log.info(f"Best answer: {best_answer}")
        log.info(f"Answer distribution: {dict(answer_counts)}")

        return {
            "best_path": best_path,
            "best_answer": best_answer,
            "consensus_score": best_score,
            "all_answers": all_answers,
            "answer_distribution": dict(answer_counts),
            "all_paths": reasoning_paths,
            "all_scores": scores
        }

    def generate_trajectory(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry point for self-consistency reasoning.

        Args:
            prompt: Input prompt/question

        Returns:
            Dictionary with trajectory information compatible with evaluation framework
        """
        
        log.info(f"Starting self-consistency reasoning for prompt: {prompt[:100]}...")

        # Generate multiple reasoning paths
        reasoning_paths = self.generate_reasoning_paths(prompt)

        # Select best answer via majority voting
        result = self.select_best_answer(reasoning_paths)

        # Format output to match expected interface
        return {
            "trajectory": result["best_path"],
            "steps": [result["best_path"]],  # Single step containing full reasoning
            "completed": True,
            "strategy": "self_consistency",
            "metadata": {
                "num_paths": len(reasoning_paths),
                "consensus_score": result["consensus_score"],
                "answer_distribution": result["answer_distribution"],
                "all_answers": result["all_answers"],
                "selected_answer": result["best_answer"]
            }
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.scorer, 'cleanup'):
            self.scorer.cleanup()
