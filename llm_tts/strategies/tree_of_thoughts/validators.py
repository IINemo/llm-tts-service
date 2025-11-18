"""
Validation utilities for Tree-of-Thoughts states and answers.
"""

import logging
import re
from typing import List

try:
    import sympy
except ImportError:
    sympy = None

log = logging.getLogger(__name__)


def get_current_numbers(state: str, problem: str) -> str:
    """
    Extract current remaining numbers from Game of 24 state.

    Args:
        state: Current state string
        problem: Original problem (fallback)

    Returns:
        String of remaining numbers (e.g., "4 6 8")
    """
    if not state:
        return problem

    # Look for "left: [numbers]" in last line
    lines = state.strip().split("\n")
    for line in reversed(lines):
        if "left:" in line.lower():
            # Extract numbers after "left:"
            numbers_part = line.split("left:")[-1].strip()
            # Remove parentheses if present
            numbers_part = numbers_part.strip("()")
            # Extract just the numbers
            numbers = re.findall(r"\d+", numbers_part)
            return " ".join(numbers)

    return problem


def validate_game24_answer(expression: str, input_numbers: str) -> bool:
    """
    Validate that an expression:
    1. Uses exactly the input numbers
    2. Evaluates to 24

    Args:
        expression: Mathematical expression (e.g., "(4 + 8) * (6 - 2)")
        input_numbers: Original input numbers (e.g., "4 4 6 8")

    Returns:
        True if answer is correct, False otherwise
    """
    if not sympy:
        log.warning("sympy not available, cannot validate Game of 24 answers")
        return False

    try:
        # Extract numbers from expression
        used_numbers = re.findall(r"\d+", expression)
        problem_numbers = re.findall(r"\d+", input_numbers)

        # Check if numbers match (same count and values)
        if sorted(used_numbers) != sorted(problem_numbers):
            log.debug(
                f"Number mismatch: used {used_numbers}, expected {problem_numbers}"
            )
            return False

        # Evaluate expression using sympy
        result = sympy.simplify(expression)
        is_correct = result == 24

        log.debug(f"Expression '{expression}' = {result}, correct={is_correct}")
        return is_correct

    except Exception as e:
        log.debug(f"Validation error for '{expression}': {e}")
        return False


def is_final_answer(state: str, mode: str) -> bool:
    """Check if state represents a final answer (mode-agnostic entry point)."""
    if mode == "game24":
        return is_game24_terminal(state)
    else:
        return is_generic_terminal(state)


def is_game24_terminal(state: str) -> bool:
    """
    Check if a Game of 24 state has reached a final answer.

    Terminal conditions:
    - Contains "(left: 24)" indicating success
    - Contains "Answer:" with expression
    """
    state_lower = state.lower()

    # Success: only 24 left
    if "(left: 24)" in state_lower or "left: 24" in state_lower:
        return True

    # Check for explicit answer
    if "answer:" in state_lower:
        return True

    return False


def is_generic_terminal(state: str) -> bool:
    """
    Check if a generic reasoning state has reached a final answer.

    Terminal conditions:
    - Contains **Final** or **Answer** markers
    - Contains conclusion indicators like "therefore", "in conclusion"
    - Appears complete based on structure (at least 2 reasoning steps)

    Args:
        state: The reasoning state to check

    Returns:
        True if state represents a final answer
    """
    state_lower = state.lower()

    # Check for explicit final markers
    if "**final" in state_lower or "**answer" in state_lower:
        return True

    # Check for conclusion words
    conclusion_words = [
        "therefore",
        "in conclusion",
        "finally",
        "the answer is",
        "thus",
    ]
    if any(word in state_lower for word in conclusion_words):
        # Count steps to see if we have enough reasoning
        step_count = state.count("**Step")
        if step_count >= 2:  # At least 2 steps of reasoning
            return True

    return False


def filter_valid_game24_answers(candidates: List[str], problem: str) -> List[str]:
    """
    Filter Game of 24 candidates to only valid solutions.

    Args:
        candidates: List of candidate states
        problem: Original problem numbers

    Returns:
        List of valid candidates (may be empty)
    """
    valid = []

    for candidate in candidates:
        # Check if it's a terminal state
        if not is_game24_terminal(candidate):
            continue

        # Try to extract and validate the answer
        lines = candidate.strip().split("\n")

        # Look for the final expression
        for line in reversed(lines):
            if "left: 24" in line.lower():
                # Extract the expression before "(left: 24)"
                expr_part = line.split("(left")[0].strip()

                # Validate it
                if validate_game24_answer(expr_part, problem):
                    valid.append(candidate)
                    break

    return valid
