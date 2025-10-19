"""
Self-consistency strategy for LLM reasoning.

Based on "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
by Wang et al. (2022). Generates multiple diverse reasoning paths and selects
the most consistent answer via majority voting.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
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
        scorer: Optional[Any] = None,
        n_threads: int = 8,
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
            n_threads: Number of parallel threads for API calls (default: 8)
        """
        self.model = model
        self.num_paths = num_paths
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size or num_paths
        self.n_threads = n_threads

        # Use majority voting scorer by default
        self.scorer = scorer or ChainMajorityVotingScorer()
        if hasattr(self.scorer, "prepare_model"):
            self.scorer.prepare_model()

    def _generate_single_path(self, args) -> Optional[str]:
        """
        Generate a single reasoning path (for multithreading).

        Args:
            args: Tuple of (prompt, path_index, total_paths)

        Returns:
            Full reasoning path (prompt + generated text), or None if error
        """
        prompt, i, total = args

        try:
            # Check if this is an API-based model
            if isinstance(self.model, BlackboxModelWithStreaming):
                # Generate single completion via API
                completions = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    num_return_sequences=1,
                )

                if completions and completions[0]:
                    full_path = prompt + completions[0]
                    log.info(f"  ✓ Generated path {i+1}/{total}")
                    return full_path
                else:
                    log.warning(f"  ⚠ Empty generation for path {i+1}/{total}")
                    return None

            else:
                # Local model generation
                inputs = self.model.tokenize([prompt])
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        num_return_sequences=1,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        length_penalty=1.0,
                    )

                # Decode generated path
                output_seq = outputs[0]
                new_tokens = output_seq[input_ids.shape[1] :]
                generated_text = self.model.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                )
                full_path = prompt + generated_text

                log.info(f"  ✓ Generated path {i+1}/{total}")

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return full_path

        except Exception as e:
            log.error(f"  ❌ Error generating path {i+1}/{total}: {e}")
            return None

    def generate_reasoning_paths(self, prompt: str) -> List[str]:
        """
        Generate multiple diverse reasoning paths for the given prompt.

        Args:
            prompt: Input prompt/question

        Returns:
            List of complete reasoning paths (prompt + generated reasoning)
        """
        log.info(
            f"Generating {self.num_paths} reasoning paths with temperature {self.temperature}"
        )

        # Prepare arguments for each path: (prompt, index, total)
        path_args = [(prompt, i, self.num_paths) for i in range(self.num_paths)]

        # Use base class parallel generation with our path-specific worker
        paths = self._parallel_generate(
            worker_func=self._generate_single_path,
            task_args=path_args,
            n_threads=self.n_threads,
            desc=f"Generating {self.num_paths} reasoning paths",
        )

        return paths

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
                "answer_distribution": {},
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

        log.info(
            f"Selected reasoning path {best_idx} with consensus score {best_score:.3f}"
        )
        log.info(f"Best answer: {best_answer}")
        log.info(f"Answer distribution: {dict(answer_counts)}")

        return {
            "best_path": best_path,
            "best_answer": best_answer,
            "consensus_score": best_score,
            "all_answers": all_answers,
            "answer_distribution": dict(answer_counts),
            "all_paths": reasoning_paths,
            "all_scores": scores,
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
                "selected_answer": result["best_answer"],
            },
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
