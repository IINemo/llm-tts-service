#!/usr/bin/env python3
"""
Convert various reasoning datasets to unified format for test-time scaling.

Unified Format:
{
    "question": str,  # The problem statement
    "answer": str,  # The gold standard answer (cleaned, just the final answer)
    "metadata": {  # Optional metadata
        "dataset": str,  # e.g., "gsm8k", "aime_2025"
        "problem_idx": int/str,  # Optional problem index
        "problem_type": str,  # Optional problem type
        "difficulty": str,  # Optional difficulty level
        "original_answer": str,  # Original answer field (may include reasoning)
    }
}

Usage:
    python scripts/convert_datasets.py --dataset gsm8k --output_dir data/unified
    python scripts/convert_datasets.py --dataset aime_2025 --output_dir data/unified
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm


def extract_answer_gsm8k(answer_text: str) -> str:
    """
    Extract final answer from GSM8K format.
    GSM8K answers end with #### <answer>
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip()
    # Fallback: return last number found
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return numbers[-1] if numbers else answer_text.strip()


def extract_answer_boxed(solution_text: str) -> str:
    """
    Extract final answer from \\boxed{} format.
    Used by Minerva Math and other datasets.
    """
    # Find \boxed{...} pattern, handling nested braces
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution_text)
    if match:
        return match.group(1).strip()
    # Fallback: return empty string
    return ""


def convert_gsm8k(split: str) -> List[Dict]:
    """Convert GSM8K dataset to unified format."""
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("test-time-compute/test_gsm8k", "default", split=split)

    unified_data = []
    for idx, example in enumerate(tqdm(dataset, desc="Converting GSM8K")):
        unified_example = {
            "question": example["question"],
            "answer": extract_answer_gsm8k(example["answer"]),
            "metadata": {
                "dataset": "gsm8k",
                "problem_idx": idx,
                "difficulty": "grade_school",
                "original_answer": example["answer"],
            },
        }
        unified_data.append(unified_example)

    print(f"Converted {len(unified_data)} GSM8K examples")
    return unified_data


def convert_aime_2025(split: str) -> List[Dict]:
    """Convert AIME 2025 dataset to unified format."""
    print(f"Loading AIME 2025 {split} split...")
    dataset = load_dataset("MathArena/aime_2025", split=split)

    unified_data = []
    for example in tqdm(dataset, desc="Converting AIME 2025"):
        unified_example = {
            "question": example["problem"],
            "answer": str(example["answer"]),
            "metadata": {
                "dataset": "aime_2025",
                "problem_idx": example["problem_idx"],
                "problem_type": example.get("problem_type", "unknown"),
                "difficulty": "competition",
            },
        }
        unified_data.append(unified_example)

    print(f"Converted {len(unified_data)} AIME 2025 examples")
    return unified_data


def convert_olympiadbench(split: str, data_path: str = None) -> List[Dict]:
    """
    Convert OlympiadBench dataset to unified format.

    OlympiadBench contains 675 olympiad-level math problems from IMO, USAMO, etc.
    Source: https://github.com/OpenBMB/OlympiadBench

    Args:
        split: Dataset split (only 'test' available)
        data_path: Path to local test.json file (optional, for local conversion)
    """
    print(f"Loading OlympiadBench {split} split...")

    if data_path:
        # Load from local file
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Try to load from HuggingFace or raise error
        raise ValueError(
            "OlympiadBench requires local data_path. "
            "Use --data_path to specify path to test.json"
        )

    unified_data = []
    for idx, example in enumerate(tqdm(data, desc="Converting OlympiadBench")):
        # Extract answer - final_answer is a list, take first element
        answer = example["final_answer"][0] if example["final_answer"] else ""
        # Clean answer: remove $ signs
        answer = answer.strip("$").strip()

        unified_example = {
            "question": example["question"],
            "answer": answer,
            "metadata": {
                "dataset": "olympiadbench",
                "problem_idx": example.get("id", idx),
                "problem_type": example.get("subfield", "unknown"),
                "answer_type": example.get("answer_type", "unknown"),
                "difficulty": "olympiad",
                "unit": example.get("unit"),
                "is_multiple_answer": example.get("is_multiple_answer", False),
            },
        }
        unified_data.append(unified_example)

    print(f"Converted {len(unified_data)} OlympiadBench examples")
    return unified_data


def convert_minerva_math(split: str, data_path: str = None) -> List[Dict]:
    """
    Convert Minerva Math dataset to unified format.

    Minerva Math contains 272 STEM problems from MIT OpenCourseWare.
    Source: "Solving Quantitative Reasoning Problems with Language Models"

    Args:
        split: Dataset split (only 'test' available)
        data_path: Path to local test.jsonl file
    """
    print(f"Loading Minerva Math {split} split...")

    if not data_path:
        raise ValueError(
            "Minerva Math requires local data_path. "
            "Use --data_path to specify path to test.jsonl"
        )

    # Load from local JSONL file
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    unified_data = []
    for idx, example in enumerate(tqdm(data, desc="Converting Minerva Math")):
        # Extract answer from \boxed{} in solution
        answer = extract_answer_boxed(example["solution"])

        unified_example = {
            "question": example["problem"],
            "answer": answer,
            "metadata": {
                "dataset": "minerva_math",
                "problem_idx": example.get("idx", idx),
                "problem_type": example.get("type", "unknown"),
                "difficulty": "college",
                "original_solution": example["solution"],
            },
        }
        unified_data.append(unified_example)

    print(f"Converted {len(unified_data)} Minerva Math examples")
    return unified_data


DATASET_CONVERTERS = {
    "gsm8k": convert_gsm8k,
    "aime_2025": convert_aime_2025,
    "olympiadbench": convert_olympiadbench,
    "minerva_math": convert_minerva_math,
}


def save_unified_dataset(
    data: List[Dict], dataset_name: str, split: str, output_dir: Path
):
    """Save unified dataset in both JSON and JSONL formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    jsonl_path = output_dir / f"{dataset_name}_{split}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Saved JSONL: {jsonl_path}")

    # Save as JSON
    json_path = output_dir / f"{dataset_name}_{split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "split": split,
        "num_examples": len(data),
        "format": "unified_tts_v1",
        "fields": ["question", "answer", "metadata"],
    }
    metadata_path = output_dir / f"{dataset_name}_{split}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert reasoning datasets to unified format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONVERTERS.keys()),
        help="Dataset to convert",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: auto-detect common splits)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/unified"),
        help="Output directory for converted datasets",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to local data file (required for some datasets like olympiadbench)",
    )

    args = parser.parse_args()

    # Auto-detect splits if not specified
    splits = []
    if args.split:
        splits = [args.split]
    else:
        # Default splits for each dataset
        if args.dataset == "gsm8k":
            splits = ["test"]
        elif args.dataset == "aime_2025":
            splits = ["train"]
        elif args.dataset == "olympiadbench":
            splits = ["test"]
        elif args.dataset == "minerva_math":
            splits = ["test"]

    # Convert each split
    converter_fn = DATASET_CONVERTERS[args.dataset]
    for split in splits:
        print(f"\n{'=' * 80}")
        print(f"Converting {args.dataset} ({split} split)")
        print(f"{'=' * 80}\n")

        try:
            # Pass data_path for datasets that need it
            if args.dataset in ["olympiadbench", "minerva_math"]:
                unified_data = converter_fn(split, data_path=args.data_path)
            else:
                unified_data = converter_fn(split)
            save_unified_dataset(unified_data, args.dataset, split, args.output_dir)
            print(f"\n✓ Successfully converted {args.dataset} ({split})")
        except Exception as e:
            print(f"\n✗ Error converting {args.dataset} ({split}): {e}")
            raise

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
