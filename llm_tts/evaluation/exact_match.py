"""
Exact match evaluator - EXACT copy of official Qwen2.5-Math evaluation logic.

Uses the same functions and flow as:
- https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/evaluate.py
- https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/parser.py
"""

import logging
import re

from tqdm import tqdm

from .grader import math_equal
from .parser import STRIP_EXCEPTIONS, extract_answer, strip_string

log = logging.getLogger()


def _normalize_gold_answer(gold_answer: str, data_name: str) -> str:
    """
    Normalize gold answer EXACTLY as in official parse_ground_truth post-processing.

    From Qwen2.5-Math/evaluation/parser.py lines 641-650:
    - If data_name NOT in STRIP_EXCEPTIONS: apply strip_string
    - If data_name IN STRIP_EXCEPTIONS: only basic replacements
    """
    if not gold_answer:
        return ""

    if data_name not in STRIP_EXCEPTIONS:
        return strip_string(gold_answer, skip_unit=data_name == "carp_en")
    else:
        # For minerva_math, carp_en: only basic replacements, NO strip_string
        return (
            gold_answer.replace("\\neq", "\\ne")
            .replace("\\leq", "\\le")
            .replace("\\geq", "\\ge")
        )


def _extract_boolean_answer(text: str) -> str | None:
    """Extract True/False boolean answers from text."""
    if not text:
        return None
    text = _strip_thinking_tags(text)
    text_lower = text.lower().strip()
    if re.search(r"\btrue\b", text_lower):
        return "True"
    if re.search(r"\bfalse\b", text_lower):
        return "False"
    return None


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> tags, keeping only content after the last closing tag.

    If there is content after </think>, return that. Otherwise return the
    content inside the last <think>...</think> block (for models that put
    everything inside thinking tags).
    """
    # If there is content after the last </think>, use that
    parts = text.rsplit("</think>", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    # Otherwise strip all thinking tags and return the inner content
    return re.sub(r"</?think>", "", text).strip()


def _extract_single_letter_answer(text: str) -> str | None:
    """Extract single alphabetical character answers (A, B, C, D, etc.) from text."""
    if not text:
        return None
    text = _strip_thinking_tags(text)

    # Try to extract from \boxed{X} format (last occurrence)
    boxed_matches = re.findall(r"\\boxed\{([A-Za-z])\}", text)
    if boxed_matches:
        return boxed_matches[-1].upper()

    # Single letter at end
    match = re.search(r"\b([A-Z])[.,]?\s*$", text)
    if match:
        return match.group(1).upper()

    return None


class EvaluatorExactMatch:
    def __init__(self, dataset_answer_format: str = "numeric", data_name: str = None):
        """
        Initialize the exact match evaluator.

        Args:
            dataset_answer_format: "numeric", "boolean", "char", or "string"
            data_name: Dataset name (e.g., "minerva_math", "math500", "gsm8k") - REQUIRED
        """
        if not data_name:
            raise ValueError("data_name is required for EvaluatorExactMatch")
        self.dataset_answer_format = dataset_answer_format.lower()
        self.data_name = data_name
        if self.dataset_answer_format not in ["numeric", "boolean", "char", "string"]:
            raise ValueError(
                f"dataset_answer_format must be 'numeric', 'boolean', 'char', or 'string', got '{dataset_answer_format}'"
            )

    def _score_single(self, inp: tuple[str, str, str]) -> float:
        """
        Score a single sample - used for running accuracy during generation.
        Uses EXACT same logic as batch evaluation but for one sample.
        """
        _, solution, gold_answer = inp

        if not gold_answer or gold_answer.strip() == "":
            return 0.0

        if self.dataset_answer_format != "numeric":
            # Non-numeric formats
            if self.dataset_answer_format == "boolean":
                pred_bool = _extract_boolean_answer(solution)
                gold_bool = _extract_boolean_answer(gold_answer)
                if pred_bool and gold_bool:
                    return 1.0 if pred_bool.lower() == gold_bool.lower() else 0.0
                return 0.0
            elif self.dataset_answer_format == "char":
                pred_char = _extract_single_letter_answer(solution)
                gold_char = _extract_single_letter_answer(gold_answer)
                if pred_char and gold_char:
                    return 1.0 if pred_char.upper() == gold_char.upper() else 0.0
                return 0.0
            elif self.dataset_answer_format == "string":
                pred_str = strip_string(solution) if solution else ""
                gold_str = strip_string(gold_answer) if gold_answer else ""
                return 1.0 if pred_str.lower() == gold_str.lower() else 0.0
            return 0.0

        # Numeric: call math_equal directly (no ProcessPool for single samples - too slow)
        try:
            pred = extract_answer(solution, data_name=self.data_name)
            gold = _normalize_gold_answer(gold_answer, self.data_name)
            log.info(
                f"_score_single BEFORE math_equal: pred={repr(pred)}, gold={repr(gold)}"
            )
            result = math_equal(pred, gold)
            log.info(f"_score_single AFTER math_equal: result={result}")
            return 1.0 if result else 0.0
        except Exception as e:
            import traceback

            log.error(f"Math grading error: {e}\n{traceback.format_exc()}")
            return 0.0

    def __call__(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> list[float]:
        """
        Evaluate solutions against gold answers.

        Uses same logic as official Qwen2.5-Math but with direct math_equal calls
        (no ProcessPool overhead for small batches).
        """
        if self.dataset_answer_format != "numeric":
            # For non-numeric, use simple comparison
            return self._evaluate_non_numeric(problems, solutions, gold_answers)

        # Numeric evaluation - direct math_equal calls (same logic, no ProcessPool overhead)
        scores = []
        for idx, (solution, gold) in enumerate(
            tqdm(
                zip(solutions, gold_answers),
                total=len(solutions),
                desc="Verifying solutions",
            )
        ):
            try:
                pred = extract_answer(solution, data_name=self.data_name)
                gold_normalized = _normalize_gold_answer(gold, self.data_name)
                log.info(
                    f"__call__ idx={idx} BEFORE math_equal: pred={repr(pred)}, gold={repr(gold_normalized)}"
                )
                result = math_equal(pred, gold_normalized)
                log.info(f"__call__ idx={idx} AFTER math_equal: result={result}")
                scores.append(1.0 if result else 0.0)
            except Exception as e:
                import traceback

                log.error(
                    f"__call__ idx={idx} Math grading error: {e}\n{traceback.format_exc()}"
                )
                scores.append(0.0)

        return scores

    def _evaluate_non_numeric(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> list[float]:
        """Handle boolean, char, and string formats."""
        scores = []
        for solution, gold in tqdm(
            zip(solutions, gold_answers), desc="Verifying solutions"
        ):
            if not gold or gold.strip() == "":
                scores.append(0.0)
                continue

            if self.dataset_answer_format == "boolean":
                pred_bool = _extract_boolean_answer(solution)
                gold_bool = _extract_boolean_answer(gold)
                if pred_bool and gold_bool:
                    scores.append(
                        1.0 if pred_bool.lower() == gold_bool.lower() else 0.0
                    )
                else:
                    scores.append(0.0)

            elif self.dataset_answer_format == "char":
                pred_char = _extract_single_letter_answer(solution)
                gold_char = _extract_single_letter_answer(gold)
                if pred_char and gold_char:
                    scores.append(
                        1.0 if pred_char.upper() == gold_char.upper() else 0.0
                    )
                else:
                    scores.append(0.0)

            elif self.dataset_answer_format == "string":
                pred_str = strip_string(solution) if solution else ""
                gold_str = strip_string(gold) if gold else ""
                scores.append(1.0 if pred_str.lower() == gold_str.lower() else 0.0)
            else:
                scores.append(0.0)

        return scores
