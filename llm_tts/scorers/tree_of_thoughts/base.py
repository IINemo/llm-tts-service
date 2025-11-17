"""
Base class for Tree-of-Thoughts state evaluation scorers.

ToT scorers evaluate intermediate reasoning states (not just next-step candidates)
for their quality and likelihood of reaching the correct solution.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

log = logging.getLogger(__name__)


class TotStateScorerBase(ABC):
    """
    Abstract base class for scoring intermediate states in Tree-of-Thoughts search.

    Unlike step scorers (which score next-step candidates), state scorers evaluate
    the quality of partial solutions and their potential to reach correct answers.
    """

    def __init__(self, model, name: str = "tot_state_scorer"):
        """
        Initialize ToT state scorer.

        Args:
            model: Language model for evaluation
            name: Scorer name for logging
        """
        self.model = model
        self.name = name
        self.total_evaluations = 0
        self.cache = {}

    @abstractmethod
    def score_states(
        self, problem: str, states: List[str], cache_results: bool = True, **kwargs
    ) -> List[float]:
        """
        Score intermediate reasoning states.

        Args:
            problem: Original problem statement
            states: List of partial solutions to evaluate
            cache_results: Whether to cache evaluation results
            **kwargs: Additional evaluation parameters

        Returns:
            List of scores (higher = better quality state)
        """
        pass

    @abstractmethod
    def build_evaluation_prompt(self, problem: str, state: str) -> str:
        """
        Build prompt for state evaluation.

        Args:
            problem: Original problem
            state: Current partial solution

        Returns:
            Evaluation prompt
        """
        pass

    @abstractmethod
    def parse_evaluation_output(self, output: str) -> float:
        """
        Parse model output into numerical score.

        Args:
            output: Model's evaluation output

        Returns:
            Numerical score
        """
        pass

    def is_final_state(self, state: str) -> bool:
        """
        Check if state represents a final answer.

        Args:
            state: Current state

        Returns:
            True if this is a final answer
        """
        state_lower = state.lower()
        return any(
            [
                "answer:" in state_lower,
                "\\boxed{" in state,
                state_lower.strip().endswith("="),
            ]
        )

    def cleanup(self):
        """Clean up resources (e.g., clear cache)"""
        self.cache.clear()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
