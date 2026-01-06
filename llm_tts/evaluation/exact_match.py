"""
Exact match evaluator aligned with official Qwen2.5-Math evaluation.

This evaluator uses the official extract_answer, strip_string normalization
and math_equal comparison from Qwen2.5-Math to ensure benchmark compatibility.

Sources:
- https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/math_eval.py
- https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/grader.py
- https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/parser.py
"""

import logging
import re

from tqdm import tqdm

from .grader import math_equal
from .parser import extract_answer, strip_string

log = logging.getLogger()


def _extract_boolean_answer(text: str) -> str | None:
    """Extract True/False boolean answers from text."""
    if not text:
        return None

    # Normalize text for case-insensitive matching
    text_lower = text.lower().strip()

    # Look for explicit True/False patterns
    true_patterns = [
        r"\btrue\b",
    ]
    false_patterns = [
        r"\bfalse\b",
    ]

    for pattern in true_patterns:
        if re.search(pattern, text_lower):
            return "True"
    for pattern in false_patterns:
        if re.search(pattern, text_lower):
            return "False"

    return None


def _extract_single_letter_answer(text: str) -> str | None:
    """Extract single alphabetical character answers (A, B, C, D, etc.) from text."""
    if not text:
        return None

    # Specific patterns for single letter answers only
    # Pattern 1: Look for "Answer: A" patterns first
    answer_patterns = [
        r"(?:<answer>)\s*:?\s*([A-Z])",
        r"([A-Z])\s*(?:is\s*)?(?:the\s*)?(?:correct\s*)?(?:<answer>)",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).upper()

    # Pattern 2: Single letter at the end of text (standalone) - only if it's truly alone
    match = re.search(r"^([A-Z])\s*$", text.strip())
    if match:
        return match.group(1).upper()

    # Pattern 3: Single letter followed by period, comma, or end - only at the very end
    match = re.search(r"\b([A-Z])[.,]?\s*$", text.strip())
    if match:
        return match.group(1).upper()

    return None


class EvaluatorExactMatch:
    def __init__(self, dataset_answer_format: str = "numeric"):
        """
        Initialize the exact match evaluator.

        Args:
            dataset_answer_format: Type of answers to look for in answer. Set in `config.dataset.answer_format`.
            Options:
                - "numeric": Math/numeric answers (default)
                - "boolean": True/False answers (for StrategyQA)
                - "char": Single letter answers (for CSQA)
                - "string": Direct string comparison (any text)
        """
        self.dataset_answer_format = dataset_answer_format.lower()
        if self.dataset_answer_format not in ["numeric", "boolean", "char", "string"]:
            raise ValueError(
                f"dataset_answer_format must be 'numeric', 'boolean', 'char', or 'string', got '{dataset_answer_format}'"
            )

    def _score_single(self, inp: tuple[str, str, str]) -> float:
        q, solution, gold_answer = inp

        # Handle numeric format using official Qwen2.5-Math extraction
        if self.dataset_answer_format == "numeric":
            try:
                # Extract answer from model solution (finds \boxed{} or last number)
                # Then apply strip_string normalization (matching official run_execute)
                pred = extract_answer(solution, data_name="math500")
                pred = strip_string(pred)

                # Gold answer: apply strip_string normalization
                # (matching official parse_ground_truth for math500)
                gold = strip_string(gold_answer)

                # Use official math_equal with timeout=False (matching official eval)
                if math_equal(pred, gold, timeout=False):
                    return 1.0

            except Exception as e:
                log.warning(f"Math grading failed: {e}")

            return 0.0

        # Handle boolean format
        if self.dataset_answer_format == "boolean":
            pred_bool = _extract_boolean_answer(solution)
            gold_bool = _extract_boolean_answer(gold_answer)

            if pred_bool and gold_bool:
                return 1.0 if pred_bool.lower() == gold_bool.lower() else 0.0
            return 0.0

        # Handle char format (multiple choice)
        if self.dataset_answer_format == "char":
            pred_char = _extract_single_letter_answer(solution)
            gold_char = _extract_single_letter_answer(gold_answer)

            if pred_char and gold_char:
                return 1.0 if pred_char.upper() == gold_char.upper() else 0.0
            return 0.0

        # Handle string format (direct comparison)
        if self.dataset_answer_format == "string":
            pred_str = strip_string(solution) if solution else ""
            gold_str = strip_string(gold_answer) if gold_answer else ""
            return 1.0 if pred_str.lower() == gold_str.lower() else 0.0

        return 0.0

    def __call__(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> list[float]:
        all_inputs = zip(problems, solutions, gold_answers)
        labels = []
        for item in tqdm(all_inputs, desc="Verifying solutions"):
            labels.append(self._score_single(item))
        return labels
