"""
Parsing utilities for Tree-of-Thoughts responses.
"""

import re
from typing import List


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from reasoning text.

    For Game of 24:
    - Looks for "Answer: expression" format
    - Looks for final expression in parentheses
    - Validates using sympy if possible

    Other formats:
    - \\boxed{answer}
    - answer = value
    - Last number in text

    Args:
        text: Reasoning text

    Returns:
        Extracted answer string
    """
    text = text.strip()

    # Try "Answer:" format first (for Game of 24)
    answer_match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if answer_match:
        answer_expr = answer_match.group(1).strip()
        # Remove trailing text like "= 24"
        answer_expr = re.sub(r"\s*=\s*24\s*$", "", answer_expr)
        return answer_expr

    # Try \\boxed{} format
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try final line with (left: 24) pattern
    if "(left: 24)" in text:
        # Found solution! Extract the final expression
        lines = text.strip().split("\n")
        for line in reversed(lines):
            if "answer:" in line.lower():
                expr = line.split(":", 1)[-1].strip()
                expr = re.sub(r"\s*=\s*24\s*$", "", expr)
                return expr

    # Try "= value" at end of line
    equals_match = re.search(r"=\s*([0-9,.]+)(?:\s|$)", text)
    if equals_match:
        return equals_match.group(1).strip()

    # Fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else text[:50]


def extract_new_step(candidate: str, parent_states: List[str]) -> str:
    """
    Extract just the new reasoning step added to a candidate.

    Args:
        candidate: Full candidate state
        parent_states: List of parent states

    Returns:
        The new step that was added
    """
    # Find the parent that this candidate extends
    best_parent = ""
    for parent in parent_states:
        if candidate.startswith(parent):
            if len(parent) > len(best_parent):
                best_parent = parent

    if best_parent:
        new_step = candidate[len(best_parent) :].strip()
        # Extract first line as the new step
        lines = new_step.split("\n")
        return lines[0] if lines else new_step[:100]

    # Fallback: return last line of candidate
    lines = candidate.strip().split("\n")
    return lines[-1][:100] if lines else candidate[:100]


def parse_proposals(text: str, current_state: str, mode: str) -> List[str]:
    """Parse model response into list of proposals (mode-agnostic entry point)."""
    if mode == "game24":
        return parse_game24_format(text, current_state)
    else:
        return parse_generic_format(text, current_state)


def parse_game24_format(text: str, current_state: str) -> List[str]:
    """
    Parse Game of 24 format proposals.

    Expected format:
    4 + 8 = 12 (left: 6 12)
    Explanation: ...
    """
    proposals = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("explanation"):
            continue

        # Look for lines with arithmetic and "left:"
        if "left:" in line.lower():
            # Extract just the step part
            if "(" in line:
                step = line.split("(")[0].strip()
                remaining = line.split("(")[-1].strip()
            else:
                step = line
                remaining = ""

            # Combine with current state
            if current_state:
                new_state = f"{current_state}\n{step} {remaining}"
            else:
                new_state = f"{step} {remaining}"

            proposals.append(new_state)

    return proposals if proposals else [text]


def parse_generic_format(text: str, current_state: str) -> List[str]:
    """
    Parse generic reasoning format proposals.

    Expected format:
    **Step X**: Reasoning step here
    """
    proposals = []

    # Try to extract structured steps
    step_pattern = r"\*\*(?:Step|Final)(?:\s+\d+)?\*\*:?\s*(.+?)(?=\n\*\*|$)"
    matches = re.findall(step_pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        for match in matches:
            step_text = match.strip()
            # Remove any trailing markdown or formatting
            step_text = re.sub(r"\*\*.*$", "", step_text).strip()

            # Combine with current state
            if current_state:
                new_state = f"{current_state}\n**Step**: {step_text}"
            else:
                new_state = f"**Step**: {step_text}"

            proposals.append(new_state)
    else:
        # Fallback: treat whole response as one proposal
        if current_state:
            new_state = f"{current_state}\n{text.strip()}"
        else:
            new_state = text.strip()
        proposals.append(new_state)

    return proposals
