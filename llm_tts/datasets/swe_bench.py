"""
SWE-bench Lite dataset loader and utilities.

SWE-bench is a benchmark for evaluating language models on real-world software
engineering tasks. Given a codebase and an issue description, the model must
generate a patch that resolves the described problem.

Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
Paper: https://arxiv.org/abs/2310.06770
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

log = logging.getLogger(__name__)


@dataclass
class SWEBenchInstance:
    """A single SWE-bench instance with all relevant fields."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    patch: str  # Gold patch
    test_patch: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    environment_setup_commit: str
    version: str
    created_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SWEBenchInstance":
        """Create instance from HuggingFace dataset row."""
        # Parse FAIL_TO_PASS and PASS_TO_PASS from JSON strings
        fail_to_pass = data.get("FAIL_TO_PASS", "[]")
        pass_to_pass = data.get("PASS_TO_PASS", "[]")

        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = []

        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = []

        return cls(
            instance_id=data["instance_id"],
            repo=data["repo"],
            base_commit=data["base_commit"],
            problem_statement=data["problem_statement"],
            hints_text=data.get("hints_text", ""),
            patch=data["patch"],
            test_patch=data["test_patch"],
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            environment_setup_commit=data["environment_setup_commit"],
            version=data.get("version", ""),
            created_at=data.get("created_at", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
            "environment_setup_commit": self.environment_setup_commit,
            "version": self.version,
            "created_at": self.created_at,
        }


def load_swe_bench_lite(
    split: str = "test",
    subset_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    include_hints: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load SWE-bench Lite dataset and format for evaluation.

    Args:
        split: Dataset split ('test' or 'dev')
        subset_size: If provided, only load first N examples
        cache_dir: Cache directory for HuggingFace datasets
        include_hints: Whether to include hints in the problem statement

    Returns:
        List of dicts with formatted data for the evaluation pipeline
    """
    log.info(f"Loading SWE-bench Lite dataset (split={split})...")

    dataset = load_dataset(
        "princeton-nlp/SWE-bench_Lite",
        split=split,
        cache_dir=cache_dir,
    )

    if subset_size is not None:
        dataset = dataset.select(range(min(subset_size, len(dataset))))
        log.info(f"Using subset of {len(dataset)} examples")

    formatted_data = []
    for item in dataset:
        instance = SWEBenchInstance.from_dict(item)

        # Build the question/problem description
        problem_text = instance.problem_statement

        if include_hints and instance.hints_text:
            problem_text += f"\n\nHints:\n{instance.hints_text}"

        formatted = {
            # Standard fields for the evaluation pipeline
            "question": problem_text,
            "answer": instance.patch,  # Gold patch is the "answer"
            # SWE-bench specific fields
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "test_patch": instance.test_patch,
            "fail_to_pass": instance.fail_to_pass,
            "pass_to_pass": instance.pass_to_pass,
            "environment_setup_commit": instance.environment_setup_commit,
            "version": instance.version,
            "hints_text": instance.hints_text,
        }
        formatted_data.append(formatted)

    log.info(f"Loaded {len(formatted_data)} SWE-bench Lite examples")

    # Log repository distribution
    repo_counts = {}
    for item in formatted_data:
        repo = item["repo"]
        repo_counts[repo] = repo_counts.get(repo, 0) + 1

    log.info("Repository distribution:")
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {repo}: {count}")

    return formatted_data


def format_swe_bench_prompt(
    instance: Dict[str, Any],
    prompt_template: Optional[str] = None,
    include_repo_info: bool = True,
) -> str:
    """
    Format a SWE-bench instance into a prompt for the model.

    Args:
        instance: A formatted SWE-bench instance dict
        prompt_template: Optional template with {problem}, {repo}, {hints} placeholders
        include_repo_info: Whether to include repository information

    Returns:
        Formatted prompt string
    """
    if prompt_template:
        return prompt_template.format(
            problem=instance["question"],
            repo=instance.get("repo", ""),
            hints=instance.get("hints_text", ""),
            instance_id=instance.get("instance_id", ""),
        )

    # Default formatting
    parts = []

    if include_repo_info:
        parts.append(f"Repository: {instance['repo']}")
        parts.append("")

    parts.append("Issue Description:")
    parts.append(instance["question"])
    parts.append("")
    parts.append(
        "Please provide a patch in unified diff format that resolves this issue."
    )

    return "\n".join(parts)


def extract_patch_from_response(response: str) -> str:
    """
    Extract a unified diff patch from model response.

    Handles various formats:
    - Code blocks with ```diff or ``` markers
    - Raw diff content
    - Multiple patches (returns concatenated)

    Args:
        response: Model's response text

    Returns:
        Extracted patch string (may be empty if no patch found)
    """
    import re

    # Try to extract from code blocks first
    # Match ```diff ... ``` or ``` ... ``` blocks
    code_block_pattern = r"```(?:diff)?\s*\n(.*?)```"
    code_blocks = re.findall(code_block_pattern, response, re.DOTALL)

    if code_blocks:
        # Filter to only blocks that look like diffs
        diff_blocks = []
        for block in code_blocks:
            block = block.strip()
            if _looks_like_diff(block):
                diff_blocks.append(block)

        if diff_blocks:
            return "\n".join(diff_blocks)

    # Try to find raw diff content
    lines = response.split("\n")
    diff_lines = []
    in_diff = False

    for line in lines:
        # Start of a diff
        if line.startswith("diff --git") or line.startswith("--- "):
            in_diff = True

        if in_diff:
            diff_lines.append(line)

            # End of diff hunk (empty line after +/- lines)
            if (
                line.strip() == ""
                and diff_lines
                and not diff_lines[-2].startswith(("+", "-", " ", "@"))
            ):
                in_diff = False

    if diff_lines:
        return "\n".join(diff_lines)

    return ""


def _looks_like_diff(text: str) -> bool:
    """Check if text looks like a unified diff."""
    indicators = [
        "diff --git",
        "--- a/",
        "+++ b/",
        "@@",
        "--- ",
        "+++ ",
    ]
    return any(indicator in text for indicator in indicators)


def create_prediction_file(
    predictions: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Create a predictions file in SWE-bench format.

    The format expected by swebench.harness.run_evaluation is JSONL with:
    - instance_id: str
    - model_patch: str (the predicted patch)
    - model_name_or_path: str (optional)

    Args:
        predictions: List of dicts with 'instance_id' and 'model_patch' keys
        output_path: Path to save the predictions file
    """
    log.info(f"Saving {len(predictions)} predictions to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            record = {
                "instance_id": pred["instance_id"],
                "model_patch": pred.get(
                    "model_patch", pred.get("generated_answer", "")
                ),
                "model_name_or_path": pred.get("model_name_or_path", "llm-tts"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"Predictions saved to {output_path}")


def load_predictions_file(path: str) -> List[Dict[str, Any]]:
    """
    Load predictions from a SWE-bench format predictions file.

    Args:
        path: Path to the predictions JSONL file

    Returns:
        List of prediction dictionaries
    """
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing SWE-bench Lite loader ===\n")

    # Load small subset
    data = load_swe_bench_lite(split="test", subset_size=5)

    print(f"Loaded {len(data)} examples\n")

    for i, item in enumerate(data[:2]):
        print(f"Example {i + 1}:")
        print(f"  Instance ID: {item['instance_id']}")
        print(f"  Repo: {item['repo']}")
        print(f"  Problem: {item['question'][:200]}...")
        print(f"  Patch length: {len(item['answer'])} chars")
        print(f"  FAIL_TO_PASS tests: {len(item['fail_to_pass'])}")
        print(f"  PASS_TO_PASS tests: {len(item['pass_to_pass'])}")
        print()

    # Test patch extraction
    print("\n=== Testing patch extraction ===\n")

    test_response = """
Here's the fix for the issue:

```diff
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@
 def foo():
-    return None
+    return 42
```

This should resolve the problem.
"""

    extracted = extract_patch_from_response(test_response)
    print(f"Extracted patch:\n{extracted}")
