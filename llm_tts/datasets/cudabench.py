"""
CUDABench dataset loader for CUDA kernel optimization evaluation.

CUDABench (from robust-kbench) is a benchmark for evaluating LLM-generated CUDA kernels.
Each task contains a PyTorch reference implementation that should be translated to an
optimized CUDA kernel.

Source: https://github.com/SakanaAI/robust-kbench
Paper: https://arxiv.org/abs/2509.14279
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

log = logging.getLogger(__name__)

# Default paths
DEFAULT_LOCAL_PATH = Path(__file__).parent.parent.parent / "data" / "cudabench" / "cudabench.jsonl"
HF_DATASET_ID = "test-time-compute/cudabench"


def load_cudabench_local(
    data_path: Path,
    category: Optional[str] = None,
    pass_type: Optional[Literal["forward", "backward"]] = None,
    subset_size: Optional[int] = None,
) -> List[Dict]:
    """
    Load CUDABench dataset from local JSONL file.

    Args:
        data_path: Path to cudabench.jsonl file
        category: Filter by category (robust_kbench, kernelbench_level1, kernelbench_level2)
        pass_type: Filter by pass type (forward, backward)
        subset_size: Limit to first N examples

    Returns:
        List of task dictionaries
    """
    if not data_path.exists():
        raise FileNotFoundError(f"CUDABench data not found at {data_path}. Run convert_cudabench_to_hf.py first.")

    tasks = []
    with open(data_path) as f:
        for line in f:
            task = json.loads(line)

            # Apply filters
            if category and task["category"] != category:
                continue
            if pass_type and task["pass_type"] != pass_type:
                continue

            # Parse config back to dict
            task["config"] = json.loads(task["config"])
            tasks.append(task)

            if subset_size and len(tasks) >= subset_size:
                break

    return tasks


def load_cudabench_hf(
    repo_id: str = HF_DATASET_ID,
    split: str = "full",
    category: Optional[str] = None,
    pass_type: Optional[Literal["forward", "backward"]] = None,
    subset_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Load CUDABench dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace dataset repository ID
        split: Dataset split (full, robust_kbench, kernelbench_level1, kernelbench_level2)
        category: Additional category filter
        pass_type: Filter by pass type (forward, backward)
        subset_size: Limit to first N examples
        cache_dir: Cache directory for HuggingFace datasets

    Returns:
        List of task dictionaries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install: pip install datasets")

    log.info(f"Loading CUDABench from HuggingFace: {repo_id} (split={split})")

    dataset = load_dataset(repo_id, split=split, cache_dir=cache_dir)

    tasks = []
    for item in dataset:
        task = dict(item)

        # Apply filters
        if category and task["category"] != category:
            continue
        if pass_type and task["pass_type"] != pass_type:
            continue

        # Parse config
        if isinstance(task["config"], str):
            task["config"] = json.loads(task["config"])

        tasks.append(task)

        if subset_size and len(tasks) >= subset_size:
            break

    log.info(f"Loaded {len(tasks)} CUDABench tasks")
    return tasks


def load_cudabench(
    source: Literal["local", "hf"] = "local",
    category: Optional[str] = None,
    pass_type: Optional[Literal["forward", "backward"]] = None,
    subset_size: Optional[int] = None,
    data_path: Optional[Path] = None,
    repo_id: str = HF_DATASET_ID,
    split: str = "full",
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Load CUDABench dataset.

    Args:
        source: Load from "local" file or "hf" (HuggingFace Hub)
        category: Filter by category:
            - robust_kbench: Custom neural network operations (19 tasks)
            - kernelbench_level1: Basic operations (100 tasks)
            - kernelbench_level2: Complex operations (100 tasks)
        pass_type: Filter by pass type (forward, backward)
        subset_size: Limit to first N examples
        data_path: Path to local JSONL file (for source="local")
        repo_id: HuggingFace repo ID (for source="hf")
        split: Dataset split (for source="hf")
        cache_dir: Cache directory for HuggingFace datasets

    Returns:
        List of task dictionaries with keys:
            - task_id: Unique identifier
            - task_name: Human-readable name
            - category: Task category
            - level: Difficulty level (for kernelbench)
            - pass_type: "forward" or "backward"
            - pytorch_code: PyTorch reference implementation
            - config: Evaluation configuration
            - description: Task description
            - use_case: Example use case
    """
    if source == "local":
        path = data_path or DEFAULT_LOCAL_PATH
        return load_cudabench_local(path, category, pass_type, subset_size)
    else:
        return load_cudabench_hf(repo_id, split, category, pass_type, subset_size, cache_dir)


def format_cudabench_prompt(
    task: Dict,
    prompt_template: Optional[str] = None,
) -> str:
    """
    Format a CUDABench task as a prompt for LLM.

    Args:
        task: Task dictionary from load_cudabench
        prompt_template: Custom prompt template. Use {pytorch_code}, {task_name}, {description}

    Returns:
        Formatted prompt string
    """
    if prompt_template is None:
        prompt_template = """You are an expert CUDA programmer. Translate the following PyTorch code to an optimized CUDA kernel as a PyTorch C++ extension.

Task: {task_name}
Description: {description}

PyTorch Reference Implementation:
```python
{pytorch_code}
```

Write a complete PyTorch CUDA extension that:
1. Produces numerically correct results (matches PyTorch within rtol=1e-5, atol=1e-5)
2. Is faster than the PyTorch implementation
3. Handles the input shapes specified in get_inputs()

Required structure:
- Include torch/extension.h and cuda_runtime.h
- CUDA kernel with __global__ function
- A `forward` function that takes the SAME arguments as forward_fn and RETURNS torch::Tensor
- PYBIND11_MODULE to export forward

IMPORTANT: The forward function must RETURN a tensor, not take output as parameter.

Put your implementation in a ```cuda code block."""

    return prompt_template.format(
        task_name=task["task_name"],
        description=task.get("description", ""),
        pytorch_code=task["pytorch_code"],
        use_case=task.get("use_case", ""),
        pass_type=task["pass_type"],
    )


def get_task_by_id(tasks: List[Dict], task_id: str) -> Optional[Dict]:
    """Get a specific task by its ID."""
    for task in tasks:
        if task["task_id"] == task_id:
            return task
    return None


def list_categories(tasks: List[Dict]) -> Dict[str, int]:
    """List available categories and their task counts."""
    categories = {}
    for task in tasks:
        cat = task["category"]
        categories[cat] = categories.get(cat, 0) + 1
    return categories


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing CUDABench loader ===\n")

    # Load from local
    try:
        tasks = load_cudabench(source="local", subset_size=5)
        print(f"Loaded {len(tasks)} tasks from local\n")

        for task in tasks[:3]:
            print(f"Task: {task['task_id']}")
            print(f"  Name: {task['task_name']}")
            print(f"  Category: {task['category']}")
            print(f"  Pass: {task['pass_type']}")
            print(f"  Description: {task.get('description', '')[:80]}...")
            print()

        # Test prompt formatting
        print("\n=== Sample Prompt ===\n")
        prompt = format_cudabench_prompt(tasks[0])
        print(prompt[:500] + "...")

        # List categories
        all_tasks = load_cudabench(source="local")
        print("\n=== Categories ===")
        for cat, count in list_categories(all_tasks).items():
            print(f"  {cat}: {count} tasks")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python scripts/convert_cudabench_to_hf.py first")
