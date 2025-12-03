"""
Answer extraction utilities for LLM outputs.

Supports multiple answer formats:
- Default format: <Answer>: ... <end of response>
- Boxed format: \\boxed{...}
- Auto: tries default first, then boxed
"""

import logging
import re

log = logging.getLogger(__name__)


def extract_answer(text: str, answer_format: str = "auto") -> str:
    """
    Extract answer from generated text.

    Supports multiple formats:
    - Default format: <Answer>: ... <end of response>
    - Boxed format: \\boxed{...}
    - Auto: tries default first, then boxed

    Args:
        text: Generated text
        answer_format: "default", "boxed", or "auto"

    Returns:
        Extracted answer (cleaned of \\boxed{} wrapper) or empty string
    """
    if answer_format == "auto":
        # Try default format first
        default_answer = _extract_default_answer(text)
        if default_answer:
            return default_answer
        # Fall back to boxed format
        answer = _extract_boxed_answer(text)
    elif answer_format == "default":
        answer = _extract_default_answer(text)
    else:
        answer = _extract_boxed_answer(text)

    # Final cleanup: ensure no \boxed{} wrapper remains
    return _clean_boxed_from_answer(answer) if answer else ""


def _extract_default_answer(text: str) -> str:
    """Extract answer from <Answer>: ... format."""
    # Pattern to match <Answer>: followed by the answer (up to end of response or end of text)
    pattern = re.compile(r"<Answer>:\s*(.+?)(?:\s*<end of response>|$)", re.DOTALL)
    match = pattern.search(text)

    if match:
        answer = match.group(1).strip()
        # Clean up the answer - take first line if multiline
        answer = answer.split("\n")[0].strip()
        # Clean up \boxed{} if present in the answer
        answer = _clean_boxed_from_answer(answer)
        return answer

    return ""


def _clean_boxed_from_answer(answer: str) -> str:
    """Remove \\boxed{} wrapper from answer if present, extracting just the content."""
    # Check if answer is wrapped in \boxed{}
    boxed_pattern = re.compile(r"\\boxed\{(.+)\}$")
    match = boxed_pattern.search(answer)
    if match:
        return match.group(1).strip()

    return answer


def _extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{answer} format with nested braces support.

    Finds the last COMPLETE \\boxed{...} pattern (with balanced braces).
    This handles cases where early stopping may leave an incomplete boxed at the end.
    """
    if "boxed" not in text:
        return ""

    # Find all complete boxed patterns with balanced braces
    complete_answers = []

    # Split on "boxed" and process each occurrence
    parts = text.split("boxed")
    for part in parts[1:]:  # Skip first part (before any "boxed")
        if len(part) == 0 or part[0] != "{":
            continue

        # Try to extract complete boxed content with balanced braces
        stack = 1
        answer = ""
        complete = False
        for c in part[1:]:
            if c == "{":
                stack += 1
                answer += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    complete = True
                    break
                answer += c
            else:
                answer += c

        if complete:
            complete_answers.append(answer.strip())

    # Return the last complete answer (most likely the final answer)
    if complete_answers:
        return complete_answers[-1]

    # Fallback: try the old method for edge cases
    ans = text.split("boxed")[-1]
    if len(ans) == 0:
        return ""

    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a.strip()

    a = ans.split("$")[0].strip()
    return a.strip()
