import logging
import re

import numpy as np
from tqdm import tqdm

from .grader import grade_answer

log = logging.getLogger()


def _extract_answer_block(text: str) -> str | None:
    """Extract text following '<Answer>:' marker up to an end marker or end of text."""
    if not text:
        return None
    marker = "<Answer>:"
    idx = text.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    # Known terminators that indicate end of model response
    terminators = ["<end of response>", "<|im_end|>", "</s>"]
    end = len(text)
    for t in terminators:
        t_idx = text.find(t, start)
        if t_idx != -1:
            end = min(end, t_idx)
    return text[start:end].strip()


def _extract_last_boxed(text: str) -> str | None:
    """Extract the content of the last \boxed{...} block with balanced braces."""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def _extract_last_inline_math(text: str) -> str | None:
    """Extract the last $...$ or \\(...\\) or \\[...\\] math block, non-greedy."""
    # $...$
    dollar_matches = list(re.finditer(r"\$(.+?)\$", text, flags=re.DOTALL))
    if dollar_matches:
        return dollar_matches[-1].group(1).strip()

    # \(...\)
    paren_matches = list(re.finditer(r"\\\((.+?)\\\)", text, flags=re.DOTALL))
    if paren_matches:
        return paren_matches[-1].group(1).strip()

    # \[...\]
    bracket_matches = list(re.finditer(r"\\\[(.+?)\\\]", text, flags=re.DOTALL))
    if bracket_matches:
        return bracket_matches[-1].group(1).strip()

    return None


def _extract_numeric(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    return matches[-1]


def _extract_mathish_answer(text: str) -> str | None:
    """Heuristic: prefer boxed, then inline math, then numeric."""
    return (
        _extract_last_boxed(text)
        or _extract_last_inline_math(text)
        or _extract_numeric(text)
    )


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


def _extract_answer_by_format(text: str, dataset_format: str) -> str | None:
    """Extract answer based on the specified dataset format."""
    if not text:
        return None

    # Strategy 1: Try structured format first (always check this)
    structured_answer = _extract_answer_block(text)
    if structured_answer:
        return structured_answer

    # Strategy 2: Extract based on dataset format
    if dataset_format == "boolean":
        # For boolean datasets (StrategyQA)
        boolean_answer = _extract_boolean_answer(text)
        if boolean_answer:
            return boolean_answer
    elif dataset_format == "char":
        # For char datasets (CSQA) - single letter answers
        letter_answer = _extract_single_letter_answer(text)
        if letter_answer:
            return letter_answer
    elif dataset_format == "string":
        # For string datasets - direct string comparison (no extraction needed)
        return text.strip()
    elif dataset_format == "numeric":
        # For numeric datasets (math problems)
        math_answer = _extract_mathish_answer(text)
        if math_answer:
            return math_answer

    # Strategy 3: Fallback to original text
    return text.strip()


class EvaluatorExactMatch:
    def __init__(self, dataset_answer_format: str = "numeric"):
        """
        Initialize the exact match evaluator.

        Args:
            dataset_answer_format: Type of answers to look for in answer. Set in `config.dataset.answer_format`.
            Options:
                - "numeric": Math/numeric answers (default, original behavior)
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
        _, solution, gold_answer = inp

        # Step 1: Extract the answers using format-specific approach
        candidate = _extract_answer_by_format(solution, self.dataset_answer_format)
        gold_candidate = _extract_answer_by_format(
            gold_answer, self.dataset_answer_format
        )

        # Step 2: Direct string comparison for exact matches
        if candidate and gold_candidate:
            # Normalize both answers for comparison
            candidate_norm = candidate.strip().lower()
            gold_norm = gold_candidate.strip().lower()

            # Direct match
            if candidate_norm == gold_norm:
                return 1.0

        # Step 3: Try mathematical comparison (for numeric datasets)
        if self.dataset_answer_format == "numeric":
            try:
                if grade_answer(candidate, gold_candidate):
                    return 1.0
                # Fallback: if structured extraction failed, try raw solution
                if candidate is not solution and grade_answer(solution, gold_candidate):
                    return 1.0
            except Exception as e:
                log.warning(f"Robust grading failed: {e}")

            # Step 4: Fallback to simple numeric comparison
            sol_num_str = _extract_numeric(solution)
            gold_num_str = _extract_numeric(gold_answer)

            if not sol_num_str or not gold_num_str:
                return 0.0

            # Convert to numeric values with proper error handling
            try:
                sol_val = float(sol_num_str.replace(",", ""))
                gold_val = float(gold_num_str.replace(",", ""))

                # Convert to integers if they are whole numbers
                if sol_val.is_integer():
                    sol_val = int(sol_val)
                if gold_val.is_integer():
                    gold_val = int(gold_val)

            except (ValueError, TypeError) as e:
                log.warning(f"Could not convert numeric values: {e}")
                return np.nan

            return 1.0 if sol_val == gold_val else 0.0

        # For boolean and char datasets, if we get here, it's a mismatch
        return 0.0

    def _is_boolean_answer(self, answer: str) -> bool:
        """Check if the answer is a boolean (True/False)."""
        if not answer:
            return False
        answer_lower = answer.strip().lower()
        return answer_lower in ["true", "false"]

    def _is_single_letter_answer(self, answer: str) -> bool:
        """Check if the answer is a single alphabetical character."""
        if not answer:
            return False
        answer_clean = answer.strip()
        return len(answer_clean) == 1 and answer_clean.isalpha()

    def __call__(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> list[float]:
        all_inputs = zip(problems, solutions, gold_answers)
        labels = []
        for item in tqdm(all_inputs, desc="Verifying solutions"):
            labels.append(self._score_single(item))
        return labels
