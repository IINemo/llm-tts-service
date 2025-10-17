"""
GSM8K dataset loader and preprocessing for DeepConf evaluation.

GSM8K (Grade School Math 8K) is a dataset of 8.5K grade school math word problems.
Each problem requires multi-step reasoning to solve.
"""

import logging
from typing import Dict, List, Optional

from datasets import load_dataset

log = logging.getLogger(__name__)


def extract_answer_from_gsm8k(solution: str) -> str:
    """
    Extract the final numerical answer from GSM8K solution format.

    GSM8K solutions end with "#### {answer}" format.

    Args:
        solution: The solution string from GSM8K

    Returns:
        The numerical answer as a string
    """
    if "####" in solution:
        answer = solution.split("####")[-1].strip()
        # Remove commas from numbers (e.g., "1,000" -> "1000")
        answer = answer.replace(",", "")
        return answer
    return ""


def format_gsm8k_for_deepconf(question: str, answer: str) -> Dict[str, str]:
    """
    Format GSM8K data for DeepConf evaluation.

    Converts GSM8K format to the format expected by DeepConf:
    - Question stays as is
    - Answer is extracted from "#### X" format
    - Expected output format is \\boxed{X}

    Args:
        question: The question text
        answer: The solution text (includes #### {answer})

    Returns:
        Dict with 'question' and 'answer' keys
    """
    extracted_answer = extract_answer_from_gsm8k(answer)

    return {
        "question": question.strip(),
        "answer": extracted_answer,
        "original_solution": answer,  # Keep for reference
    }


def load_gsm8k(
    split: str = "test",
    subset_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset and format for DeepConf.

    Args:
        split: Dataset split ('train' or 'test')
        subset_size: If provided, only load first N examples
        cache_dir: Cache directory for HuggingFace datasets

    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    log.info(f"Loading GSM8K dataset (split={split})...")

    # Load from HuggingFace
    dataset = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)

    # Take subset if requested
    if subset_size is not None:
        dataset = dataset.select(range(min(subset_size, len(dataset))))
        log.info(f"Using subset of {len(dataset)} examples")

    # Format for DeepConf
    formatted_data = []
    for item in dataset:
        formatted = format_gsm8k_for_deepconf(
            question=item["question"], answer=item["answer"]
        )
        formatted_data.append(formatted)

    log.info(f"Loaded {len(formatted_data)} GSM8K examples")

    return formatted_data


def evaluate_gsm8k_answer(predicted: str, ground_truth: str) -> bool:
    """
    Evaluate if predicted answer matches ground truth for GSM8K.

    Handles:
    - Numeric comparison (with tolerance for floats)
    - String normalization (strip whitespace, lowercase)
    - Comma removal from numbers

    Args:
        predicted: Predicted answer (extracted from \\boxed{})
        ground_truth: Ground truth answer

    Returns:
        True if answers match, False otherwise
    """
    # Normalize both answers
    pred_clean = predicted.strip().replace(",", "").lower()
    gt_clean = ground_truth.strip().replace(",", "").lower()

    # Direct string match
    if pred_clean == gt_clean:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)

        # Use relative tolerance for floating point comparison
        return abs(pred_num - gt_num) < 1e-6 * max(abs(pred_num), abs(gt_num), 1)
    except (ValueError, TypeError):
        pass

    return False


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing GSM8K loader ===\n")

    # Load small subset
    data = load_gsm8k(split="test", subset_size=5)

    print(f"Loaded {len(data)} examples\n")

    for i, item in enumerate(data[:3]):
        print(f"Example {i+1}:")
        print(f"  Question: {item['question'][:100]}...")
        print(f"  Answer: {item['answer']}")
        print(f"  Original: {item['original_solution'][:80]}...")
        print()

    # Test answer evaluation
    print("\n=== Testing answer evaluation ===\n")
    test_cases = [
        ("70", "70", True),
        ("70", "70.0", True),
        ("1000", "1,000", True),
        ("70", "71", False),
        ("abc", "ABC", True),
    ]

    for pred, gt, expected in test_cases:
        result = evaluate_gsm8k_answer(pred, gt)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{pred}' vs '{gt}': {result} (expected {expected})")
