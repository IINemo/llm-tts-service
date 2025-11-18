"""
Prompt templates for Tree-of-Thoughts strategy.

Loads prompts from config files and fills in variables.
"""

import os
from pathlib import Path
from typing import Optional


def _load_prompt_template(prompt_path: str) -> str:
    """
    Load a prompt template from a file.

    Args:
        prompt_path: Path to prompt template file

    Returns:
        Prompt template string
    """
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    with open(prompt_path, "r") as f:
        return f.read()


def build_propose_prompt(
    problem: str,
    state: str,
    mode: str,
    propose_prompt_path: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Build prompt for propose method.

    Args:
        problem: The problem/question to solve
        state: Current reasoning state
        mode: Strategy mode ('game24' or 'generic')
        propose_prompt_path: Path to prompt template file

    Returns:
        Formatted prompt string
    """
    if propose_prompt_path and os.path.exists(propose_prompt_path):
        # Load from provided path
        template = _load_prompt_template(propose_prompt_path)
    else:
        # Use default template based on mode
        if mode == "game24":
            # For game24, use default hardcoded template (fallback)
            template = _get_default_game24_propose_template()
        else:
            # For generic mode, try to load default
            default_path = (
                Path(__file__).parent.parent.parent.parent
                / "config"
                / "prompts"
                / "tree-of-thought"
                / "generic_propose.txt"
            )
            if default_path.exists():
                template = _load_prompt_template(str(default_path))
            else:
                template = _get_default_generic_propose_template()

    # Fill in template variables
    return template.format(problem=problem, state=state if state else "(starting)")


def build_cot_prompt(
    problem: str,
    state: str,
    cot_prompt_path: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Build chain-of-thought prompt for sample method.

    Args:
        problem: The problem/question to solve
        state: Current reasoning state
        cot_prompt_path: Path to prompt template file

    Returns:
        Formatted prompt string
    """
    if cot_prompt_path and os.path.exists(cot_prompt_path):
        # Load from provided path
        template = _load_prompt_template(cot_prompt_path)
        return template.format(problem=problem, state=state if state else "(starting)")

    # Fallback to simple hardcoded template
    if state == "":
        prompt = (
            f"Solve this problem using step-by-step reasoning. "
            f"Show your complete thought process.\n\n"
            f"Question: {problem}\n\n"
            f"Think step by step and provide a complete solution:"
        )
    else:
        prompt = (
            f"Continue solving this problem step by step.\n\n"
            f"Question: {problem}\n\n"
            f"Reasoning so far:\n{state}\n\n"
            f"Continue your reasoning to reach a final answer:"
        )

    return prompt


# Fallback templates if files not found


def _get_default_game24_propose_template() -> str:
    """Get default Game of 24 propose template."""
    return """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you can choose two of the remaining numbers to combine. You need to reason step by step to find a solution.

Input: {problem}

Steps so far:
{state}

Given the steps above, what are the next possible steps? Propose exactly 1 next step.
Format your response as:
[numbers used] (left: [remaining numbers])
Explanation: ...

Your response:"""


def _get_default_generic_propose_template() -> str:
    """Get default generic propose template."""
    return """Solve the following problem step by step. Think carefully about each step.

Question: {problem}

Reasoning so far:
{state}

Propose 1 possible next reasoning step. Format as:
**Step**: [Your reasoning step here]

Your response:"""
