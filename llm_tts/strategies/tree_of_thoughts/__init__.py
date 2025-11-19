"""
Tree-of-Thoughts strategy for deliberate problem solving.

Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
by Yao et al. (2023).

Paper: https://arxiv.org/abs/2305.10601
Reference: https://github.com/princeton-nlp/tree-of-thought-llm
"""

from .strategy import StrategyTreeOfThoughts

__all__ = ["StrategyTreeOfThoughts"]
