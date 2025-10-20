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


class EvaluatorExactMatch:
    def __init__(self):
        pass

    def _score_single(self, inp: tuple[str, str, str]) -> float:
        _, solution, gold_answer = inp

        # Step 1: Extract candidate answer using hierarchical approach
        # Priority: structured format -> math format -> raw text
        candidate = (
            _extract_answer_block(solution)  # Try <Answer>: format first
            or _extract_mathish_answer(solution)  # Then try boxed/math/numeric
            or solution  # Finally use raw text
        )

        # Step 2: Clean gold answer using same approach
        gold_candidate = _extract_mathish_answer(gold_answer) or gold_answer

        # Step 3: Try robust mathematical comparison first
        # This handles complex expressions, LaTeX, fractions, etc.
        try:
            if grade_answer(candidate, gold_candidate):
                return 1.0
            # Fallback: if structured extraction failed, try raw solution
            # This handles cases where extraction methods miss the answer
            if candidate is not solution and grade_answer(solution, gold_candidate):
                return 1.0
        except Exception as e:
            log.warning(f"Robust grading failed: {e}")

        # Step 4: Fallback to simple numeric comparison
        # Extract last numeric values from both solution and gold answer
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

    def __call__(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> list[float]:
        all_inputs = zip(problems, solutions, gold_answers)
        labels = []
        for item in tqdm(all_inputs, desc="Verifying solutions"):
            labels.append(self._score_single(item))
        return labels
