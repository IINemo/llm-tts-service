"""
MBPP+ dataset loader and utilities.

MBPP+ (Mostly Basic Python Problems Plus) is an enhanced version of MBPP
with 35x more test cases for rigorous evaluation of code generation.

Dataset: https://huggingface.co/datasets/evalplus/mbppplus
EvalPlus: https://github.com/evalplus/evalplus
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from evalplus.data import get_mbpp_plus, write_jsonl

log = logging.getLogger(__name__)


def load_mbpp_plus(
    subset_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load MBPP+ dataset using evalplus API.

    Args:
        subset_size: If provided, only load first N examples

    Returns:
        List of dicts with formatted data for the evaluation pipeline
    """
    return _load_from_evalplus(subset_size)


def _load_from_evalplus(subset_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MBPP+ using evalplus API.

    Formats prompts to match EvalPlus official methodology:
    - instruction_prefix + code block with docstring
    """
    log.info("Loading MBPP+ using evalplus API...")

    # EvalPlus instruction prefix for chat/instruction models
    INSTRUCTION_PREFIX = (
        "Please provide a self-contained Python script that solves the "
        "following problem in a markdown code block:"
    )

    problems = get_mbpp_plus()
    formatted_data = []

    for task_id, problem in problems.items():
        # Extract test_list from assertion field (newline-separated assert statements)
        assertion_text = problem.get("assertion", "")
        test_list = [
            line.strip()
            for line in assertion_text.split("\n")
            if line.strip() and line.strip().startswith("assert ")
        ]

        # Format prompt exactly like EvalPlus does for chat models:
        # instruction_prefix + "\n```python\n" + prompt + "\n```"
        raw_prompt = problem["prompt"].strip()
        formatted_prompt = f"{INSTRUCTION_PREFIX}\n```python\n{raw_prompt}\n```"

        formatted = {
            # Standard fields for the evaluation pipeline
            "question": formatted_prompt,
            "answer": problem["canonical_solution"],
            # MBPP+ specific fields
            "task_id": task_id,
            "entry_point": problem.get(
                "entry_point", _extract_function_name(problem["prompt"])
            ),
            "test_list": test_list,  # Extracted from assertion field
            "assertion": assertion_text,  # Keep original assertion text
            "base_input": problem.get("base_input", []),
            "plus_input": problem.get("plus_input", []),
            "atol": problem.get("atol", 0),
            "contract": problem.get("contract", ""),
        }
        formatted_data.append(formatted)

        if subset_size and len(formatted_data) >= subset_size:
            break

    log.info(f"Loaded {len(formatted_data)} MBPP+ problems via evalplus API")
    return formatted_data


def _extract_function_name(prompt: str) -> str:
    """Extract function name from MBPP prompt."""
    # MBPP prompts often mention "Write a function" or similar
    # Try to find function name in various patterns

    # Pattern: "Write a function X to..."
    match = re.search(r"[Ww]rite a (?:python )?function (\w+)", prompt)
    if match:
        return match.group(1)

    # Pattern: "def function_name"
    match = re.search(r"def (\w+)\s*\(", prompt)
    if match:
        return match.group(1)

    # Pattern: function name mentioned with backticks
    match = re.search(r"`(\w+)`", prompt)
    if match:
        return match.group(1)

    # Default fallback
    return "solution"


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from model response.

    Handles various formats:
    - Code blocks with ```python or ``` markers
    - Raw code
    - Code with explanation

    Args:
        response: Model's response text

    Returns:
        Extracted code string
    """
    # Try to extract from code blocks first
    # Match ```python ... ``` or ``` ... ``` blocks
    code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
    code_blocks = re.findall(code_block_pattern, response, re.DOTALL)

    if code_blocks:
        # Return the last code block (usually the final solution)
        return code_blocks[-1].strip()

    # Try to find function definition
    # Look for def ... up to the next blank line or end
    func_pattern = r"(def \w+\s*\([^)]*\):.*?)(?:\n\n|\Z)"
    func_matches = re.findall(func_pattern, response, re.DOTALL)

    if func_matches:
        return func_matches[-1].strip()

    # Return the response as-is (might be raw code)
    return response.strip()


def format_mbpp_prompt(
    problem: Dict[str, Any],
    prompt_template: Optional[str] = None,
) -> str:
    """
    Format an MBPP+ problem into a prompt for the model.

    Args:
        problem: A formatted MBPP+ problem dict
        prompt_template: Optional template with {prompt}, {entry_point} placeholders

    Returns:
        Formatted prompt string
    """
    if prompt_template:
        return prompt_template.format(
            prompt=problem["question"],
            entry_point=problem.get("entry_point", "solution"),
            task_id=problem.get("task_id", ""),
        )

    # Default formatting - just return the prompt
    return problem["question"]


def create_evalplus_samples(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Create a samples file in EvalPlus format for evaluation.

    The format expected by evalplus is JSONL with:
    - task_id: str
    - solution: str (the complete solution code)

    Args:
        results: List of result dicts with 'task_id' and generated code
        output_path: Path to save the samples file
    """
    log.info(f"Saving {len(results)} samples to {output_path}")

    samples = []
    for result in results:
        task_id = result.get("task_id", "")
        # Get the generated code from various possible fields
        solution = (
            result.get("generated_code")
            or result.get("extracted_answer")
            or result.get("generated_answer")
            or result.get("generated_trajectory", "")
        )

        # Extract code if it contains markdown
        if "```" in solution:
            solution = extract_code_from_response(solution)

        samples.append(
            {
                "task_id": task_id,
                "solution": solution,
            }
        )

    write_jsonl(output_path, samples)

    log.info(f"Samples saved to {output_path}")


def load_evalplus_samples(path: str) -> List[Dict[str, Any]]:
    """
    Load samples from an EvalPlus format file.

    Args:
        path: Path to the samples JSONL file

    Returns:
        List of sample dictionaries
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing MBPP+ loader ===\n")

    # Load small subset
    data = load_mbpp_plus(subset_size=5)

    print(f"Loaded {len(data)} problems\n")

    for i, item in enumerate(data[:3]):
        print(f"Problem {i + 1}:")
        print(f"  Task ID: {item['task_id']}")
        print(f"  Entry point: {item['entry_point']}")
        print(f"  Prompt: {item['question'][:100]}...")
        print(f"  Solution preview: {item['answer'][:80]}...")
        print()

    # Test code extraction
    print("\n=== Testing code extraction ===\n")

    test_response = """
Here's the solution:

```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
```

This function finds common elements between two tuples.
"""

    extracted = extract_code_from_response(test_response)
    print(f"Extracted code:\n{extracted}")
