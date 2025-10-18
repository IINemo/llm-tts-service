#!/usr/bin/env python3
"""
Upload converted datasets to HuggingFace Hub.

Usage:
    # Login first (one-time)
    huggingface-cli login

    # Upload AIME 2025
    python scripts/upload_to_hf.py --dataset aime_2025 --org test-time-compute

    # Dry run (preview without uploading)
    python scripts/upload_to_hf.py --dataset aime_2025 --org test-time-compute --dry-run
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def load_converted_dataset(dataset_name: str, data_dir: Path) -> Dataset:
    """Load converted dataset from JSONL file."""
    jsonl_path = data_dir / f"{dataset_name}_train.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {jsonl_path}\n"
            f"Please run: python scripts/convert_datasets.py --dataset {dataset_name}"
        )

    # Load from JSONL
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Loaded {len(data)} examples from {jsonl_path}")

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(data)

    return dataset


def create_dataset_card(dataset_name: str, num_examples: int) -> str:
    """Create a dataset card (README.md) for the HuggingFace Hub."""

    if dataset_name == "aime_2025":
        card = f"""---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- math
- reasoning
- test-time-scaling
- aime
size_categories:
- n<1K
---

# AIME 2025 - Unified Test-Time Scaling Format

This is the AIME (American Invitational Mathematics Examination) 2025 dataset in a unified format for test-time scaling experiments.

## Dataset Description

**Source**: MathArena/aime_2025
**Size**: {num_examples} competition-level mathematics problems
**Format**: Unified TTS format (question, answer, metadata)

## Dataset Structure

### Fields

- `question` (string): The mathematical problem statement
- `answer` (string): The numerical answer (integer from 0-999)
- `metadata` (dict): Additional information
  - `dataset`: "aime_2025"
  - `problem_idx`: Problem number (1-30)
  - `problem_type`: Type of problem (e.g., "Number Theory", "Geometry")
  - `difficulty`: "competition"

### Example

```json
{{
  "question": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$",
  "answer": "70",
  "metadata": {{
    "dataset": "aime_2025",
    "problem_idx": 1,
    "problem_type": ["Number Theory"],
    "difficulty": "competition"
  }}
}}
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("test-time-compute/aime_2025")

# Access examples
for example in dataset["train"]:
    question = example["question"]
    answer = example["answer"]
    print(f"Q: {{question}}")
    print(f"A: {{answer}}")
```

## Test-Time Scaling

This dataset is formatted for test-time scaling experiments with LLMs. The unified format enables:
- Consistent evaluation across different datasets
- Easy integration with TTS strategies (DeepConf, Best-of-N, etc.)
- Standardized metadata tracking

## Citation

```bibtex
@misc{{aime2025,
  title={{AIME 2025 - Unified Test-Time Scaling Format}},
  author={{Test-Time Compute Organization}},
  year={{2025}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/test-time-compute/aime_2025}}}}
}}
```

## License

MIT License
"""
    else:
        card = f"""---
license: mit
task_categories:
- question-answering
language:
- en
tags:
- math
- reasoning
- test-time-scaling
---

# {dataset_name.upper()} - Unified Test-Time Scaling Format

Dataset with {num_examples} examples in unified format for test-time scaling.
"""

    return card


def upload_dataset(
    dataset_name: str,
    org_name: str,
    data_dir: Path,
    dry_run: bool = False,
):
    """Upload dataset to HuggingFace Hub."""

    print(f"\n{'=' * 80}")
    print(f"Uploading {dataset_name} to {org_name}/{dataset_name}")
    print(f"{'=' * 80}\n")

    # Load dataset
    dataset = load_converted_dataset(dataset_name, data_dir)

    # Create DatasetDict with train split
    dataset_dict = DatasetDict({"train": dataset})

    # Preview
    print("\nDataset Preview:")
    print(f"  Splits: {list(dataset_dict.keys())}")
    print(f"  Train examples: {len(dataset_dict['train'])}")
    print(f"  Features: {dataset_dict['train'].features}")
    print("\nFirst example:")
    print(f"  Question: {dataset_dict['train'][0]['question'][:100]}...")
    print(f"  Answer: {dataset_dict['train'][0]['answer']}")

    if dry_run:
        print("\n[DRY RUN] Would upload dataset to HuggingFace Hub")
        print(f"  Repository: {org_name}/{dataset_name}")
        print(f"  Examples: {len(dataset_dict['train'])}")
        return

    # Upload to Hub
    repo_id = f"{org_name}/{dataset_name}"
    print(f"\nUploading to {repo_id}...")

    try:
        # Upload dataset
        dataset_dict.push_to_hub(
            repo_id,
            private=False,
        )

        # Upload README.md separately
        card_content = create_dataset_card(dataset_name, len(dataset_dict["train"]))
        api = HfApi()

        # Write card to temporary file and upload
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(card_content)
            temp_path = f.name

        try:
            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )
        finally:
            import os

            os.unlink(temp_path)

        print(f"\n✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print(f"2. Check you have access to the '{org_name}' organization")
        print("3. Verify the dataset name doesn't already exist")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload converted datasets to HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'aime_2025')",
    )
    parser.add_argument(
        "--org",
        type=str,
        default="test-time-compute",
        help="HuggingFace organization name (default: test-time-compute)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/unified"),
        help="Directory containing converted datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without uploading",
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("\nRun conversion first:")
        print(f"  python scripts/convert_datasets.py --dataset {args.dataset}")
        return 1

    # Upload
    upload_dataset(
        dataset_name=args.dataset,
        org_name=args.org,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    exit(main())
