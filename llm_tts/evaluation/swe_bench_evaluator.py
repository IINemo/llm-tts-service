"""
SWE-bench evaluator for code patch generation.

This evaluator supports two modes:
1. Full evaluation: Uses the official swebench harness with Docker containers
2. Patch validation: Quick check that validates patch syntax and structure

For full evaluation, you need:
- Docker installed and running
- swebench package: pip install swebench
- Sufficient disk space (~120GB recommended)

For patch validation, no additional dependencies are needed.
"""

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

log = logging.getLogger(__name__)


class EvaluatorSWEBench:
    """
    Evaluator for SWE-bench code patch generation.

    Supports both full Docker-based evaluation and quick patch validation.
    """

    def __init__(
        self,
        mode: str = "patch_validation",
        max_workers: int = 4,
        timeout: int = 1800,
        run_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_modal: bool = False,
    ):
        """
        Initialize the SWE-bench evaluator.

        Args:
            mode: Evaluation mode - "full" for Docker evaluation,
                  "patch_validation" for quick syntax check
            max_workers: Number of parallel workers for full evaluation
            timeout: Timeout per instance in seconds (full mode)
            run_id: Unique identifier for this evaluation run
            cache_dir: Directory for caching evaluation artifacts
            use_modal: If True, run evaluation on Modal cloud (full mode)
        """
        self.mode = mode
        self.max_workers = max_workers
        self.timeout = timeout
        self.run_id = run_id or "llm_tts_eval"
        self.cache_dir = cache_dir
        self.use_modal = use_modal

        if mode == "full":
            self._check_swebench_available()

    def _check_swebench_available(self) -> None:
        """Check if swebench package is available."""
        try:
            import swebench  # noqa: F401

            log.info("swebench package found")
        except ImportError:
            log.warning(
                "swebench package not installed. Install with: pip install swebench\n"
                "Full evaluation requires swebench. Falling back to patch_validation mode."
            )
            self.mode = "patch_validation"

    def _score_single(self, inp: Tuple[str, str, str]) -> float:
        """
        Score a single prediction.

        For patch_validation mode: Returns 1.0 if patch is valid, 0.0 otherwise.
        For full mode: This method is not used; use __call__ for batch evaluation.

        Args:
            inp: Tuple of (question, solution, gold_answer)

        Returns:
            Score (0.0 or 1.0)
        """
        _, solution, gold_answer = inp

        if self.mode == "patch_validation":
            patch = self._extract_patch(solution)
            if self._is_valid_patch(patch):
                return 1.0
            return 0.0

        # Full mode doesn't support single scoring
        log.warning("Full evaluation mode doesn't support single scoring")
        return 0.0

    def __call__(
        self,
        problems: List[str],
        solutions: List[str],
        gold_answers: List[str],
        instance_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """
        Evaluate solutions against gold patches.

        Args:
            problems: List of problem statements
            solutions: List of model-generated solutions (containing patches)
            gold_answers: List of gold patches
            instance_data: Optional list of full instance data (for full evaluation)

        Returns:
            List of scores (0.0 or 1.0 for each instance)
        """
        if self.mode == "patch_validation":
            return self._evaluate_patch_validation(problems, solutions, gold_answers)
        elif self.mode == "full":
            if instance_data is None:
                log.warning(
                    "Full evaluation requires instance_data. "
                    "Falling back to patch_validation."
                )
                return self._evaluate_patch_validation(
                    problems, solutions, gold_answers
                )
            return self._evaluate_full(solutions, instance_data)
        else:
            raise ValueError(f"Unknown evaluation mode: {self.mode}")

    def _evaluate_patch_validation(
        self,
        problems: List[str],
        solutions: List[str],
        gold_answers: List[str],
    ) -> List[float]:
        """
        Quick validation of generated patches.

        Checks:
        - Patch syntax is valid unified diff format
        - Patch is non-empty
        - Patch modifies at least one file

        Args:
            problems: List of problem statements
            solutions: List of model-generated solutions
            gold_answers: List of gold patches (used for comparison metrics)

        Returns:
            List of scores (1.0 if valid patch, 0.0 otherwise)
        """
        scores = []
        valid_count = 0
        has_content_count = 0

        for idx, (solution, gold) in enumerate(
            tqdm(
                zip(solutions, gold_answers),
                total=len(solutions),
                desc="Validating patches",
            )
        ):
            patch = self._extract_patch(solution)

            if not patch:
                scores.append(0.0)
                continue

            if self._is_valid_patch(patch):
                valid_count += 1
                scores.append(1.0)

                # Check if patch has meaningful content
                if self._has_meaningful_changes(patch):
                    has_content_count += 1
            else:
                scores.append(0.0)

        log.info(
            f"Patch validation results: {valid_count}/{len(solutions)} valid patches, "
            f"{has_content_count}/{len(solutions)} with meaningful changes"
        )

        return scores

    def _evaluate_full(
        self,
        solutions: List[str],
        instance_data: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Full evaluation using swebench harness with Docker.

        This runs the actual tests in containerized environments.

        Args:
            solutions: List of model-generated solutions
            instance_data: List of full instance data dicts

        Returns:
            List of scores (1.0 if resolved, 0.0 otherwise)
        """
        try:
            from swebench.harness.run_evaluation import main as run_evaluation
        except ImportError:
            log.error("swebench not installed. Cannot run full evaluation.")
            return [0.0] * len(solutions)

        # Create temporary directory for predictions
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_path = Path(tmpdir) / "predictions.jsonl"
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir(exist_ok=True)

            # Create predictions file
            predictions = []
            for solution, instance in zip(solutions, instance_data):
                patch = self._extract_patch(solution)
                predictions.append(
                    {
                        "instance_id": instance["instance_id"],
                        "model_patch": patch,
                        "model_name_or_path": "llm-tts",
                    }
                )

            with open(predictions_path, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")

            log.info(f"Running swebench evaluation on {len(predictions)} instances...")

            # Run evaluation
            try:
                # Build command line arguments
                args = [
                    "--predictions_path",
                    str(predictions_path),
                    "--max_workers",
                    str(self.max_workers),
                    "--run_id",
                    self.run_id,
                    "--timeout",
                    str(self.timeout),
                ]

                if self.cache_dir:
                    args.extend(["--cache_level", "instance"])

                if self.use_modal:
                    args.extend(["--modal", "true"])

                # Run evaluation
                run_evaluation(args)

                # Parse results
                return self._parse_evaluation_results(
                    results_dir, [inst["instance_id"] for inst in instance_data]
                )

            except Exception as e:
                log.error(f"Evaluation failed: {e}")
                return [0.0] * len(solutions)

    def _parse_evaluation_results(
        self,
        results_dir: Path,
        instance_ids: List[str],
    ) -> List[float]:
        """Parse evaluation results from swebench output."""
        scores = []
        results_file = results_dir / f"{self.run_id}.json"

        if not results_file.exists():
            log.warning(f"Results file not found: {results_file}")
            return [0.0] * len(instance_ids)

        with open(results_file) as f:
            results = json.load(f)

        resolved_ids = set(results.get("resolved", []))

        for instance_id in instance_ids:
            if instance_id in resolved_ids:
                scores.append(1.0)
            else:
                scores.append(0.0)

        resolved_count = sum(scores)
        log.info(
            f"Full evaluation results: {resolved_count}/{len(instance_ids)} resolved"
        )

        return scores

    def _extract_patch(self, solution: str) -> str:
        """
        Extract unified diff patch from model solution.

        Args:
            solution: Model's full response

        Returns:
            Extracted patch string
        """
        from llm_tts.datasets.swe_bench import extract_patch_from_response

        return extract_patch_from_response(solution)

    def _is_valid_patch(self, patch: str) -> bool:
        """
        Check if patch is valid unified diff format.

        Args:
            patch: Patch string to validate

        Returns:
            True if valid, False otherwise
        """
        if not patch or not patch.strip():
            return False

        # Must have diff header or file indicators
        has_diff_header = "diff --git" in patch
        has_file_headers = "---" in patch and "+++" in patch
        has_hunks = "@@" in patch

        # Basic structure check
        if not (has_diff_header or has_file_headers):
            return False

        if not has_hunks:
            return False

        # Check for actual changes (+ or - lines)
        lines = patch.split("\n")
        has_additions = any(
            line.startswith("+") and not line.startswith("+++") for line in lines
        )
        has_deletions = any(
            line.startswith("-") and not line.startswith("---") for line in lines
        )

        return has_additions or has_deletions

    def _has_meaningful_changes(self, patch: str) -> bool:
        """
        Check if patch has meaningful (non-whitespace) changes.

        Args:
            patch: Patch string

        Returns:
            True if has meaningful changes
        """
        lines = patch.split("\n")
        meaningful_changes = 0

        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                content = line[1:].strip()
                if content and not content.isspace():
                    meaningful_changes += 1
            elif line.startswith("-") and not line.startswith("---"):
                content = line[1:].strip()
                if content and not content.isspace():
                    meaningful_changes += 1

        return meaningful_changes > 0

    def get_detailed_results(
        self,
        solutions: List[str],
        instance_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Get detailed evaluation results for analysis.

        Args:
            solutions: List of model-generated solutions
            instance_data: List of full instance data

        Returns:
            List of detailed result dicts with analysis
        """
        results = []

        for solution, instance in zip(solutions, instance_data):
            patch = self._extract_patch(solution)
            is_valid = self._is_valid_patch(patch)
            has_meaningful = self._has_meaningful_changes(patch) if is_valid else False

            # Analyze patch structure
            files_modified = self._count_files_modified(patch)
            hunks_count = (
                patch.count("@@") // 2 if patch else 0
            )  # @@ appears twice per hunk

            result = {
                "instance_id": instance["instance_id"],
                "repo": instance.get("repo", ""),
                "extracted_patch": patch,
                "patch_length": len(patch),
                "is_valid_patch": is_valid,
                "has_meaningful_changes": has_meaningful,
                "files_modified": files_modified,
                "hunks_count": hunks_count,
                "gold_patch_length": len(instance.get("answer", "")),
                "fail_to_pass_count": len(instance.get("fail_to_pass", [])),
                "pass_to_pass_count": len(instance.get("pass_to_pass", [])),
            }

            results.append(result)

        return results

    def _count_files_modified(self, patch: str) -> int:
        """Count number of files modified in a patch."""
        if not patch:
            return 0

        # Count "diff --git" headers or "---" file headers
        git_headers = patch.count("diff --git")
        if git_headers > 0:
            return git_headers

        # Fallback: count --- headers
        return len(re.findall(r"^--- ", patch, re.MULTILINE))


def run_swebench_evaluation_cli(
    predictions_path: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    max_workers: int = 4,
    run_id: str = "llm_tts_eval",
    timeout: int = 1800,
    use_modal: bool = False,
) -> Dict[str, Any]:
    """
    Run SWE-bench evaluation using the CLI interface.

    This is an alternative to the Python API that uses subprocess.

    Args:
        predictions_path: Path to predictions JSONL file
        dataset_name: HuggingFace dataset name
        max_workers: Number of parallel workers
        run_id: Unique run identifier
        timeout: Timeout per instance
        use_modal: Whether to use Modal cloud

    Returns:
        Dict with evaluation results
    """
    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        predictions_path,
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--timeout",
        str(timeout),
    ]

    if use_modal:
        cmd.extend(["--modal", "true"])

    log.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * max_workers + 3600,  # Overall timeout
        )

        if result.returncode != 0:
            log.error(f"Evaluation failed: {result.stderr}")
            return {"error": result.stderr, "resolved": []}

        # Parse output for results
        return {"stdout": result.stdout, "stderr": result.stderr}

    except subprocess.TimeoutExpired:
        log.error("Evaluation timed out")
        return {"error": "timeout", "resolved": []}
    except FileNotFoundError:
        log.error("swebench not found. Install with: pip install swebench")
        return {"error": "swebench not installed", "resolved": []}


if __name__ == "__main__":
    # Test the evaluator
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing SWE-bench Evaluator ===\n")

    evaluator = EvaluatorSWEBench(mode="patch_validation")

    # Test cases
    test_solutions = [
        # Valid patch
        """
Here's the fix:

```diff
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@
 def foo():
-    return None
+    return 42
```
""",
        # Invalid - no patch
        "I think you should fix the bug by changing the return value.",
        # Invalid - malformed
        "```\n--- a/file.py\n+++ b/file.py\nsome random text\n```",
        # Valid patch without code block
        """
diff --git a/lib/utils.py b/lib/utils.py
--- a/lib/utils.py
+++ b/lib/utils.py
@@ -5,6 +5,7 @@
 def process():
     data = load()
+    validate(data)
     return data
""",
    ]

    problems = ["Fix bug"] * len(test_solutions)
    gold_answers = ["patch"] * len(test_solutions)

    scores = evaluator(problems, test_solutions, gold_answers)

    for i, (solution, score) in enumerate(zip(test_solutions, scores)):
        status = "VALID" if score == 1.0 else "INVALID"
        print(f"Test {i + 1}: {status}")
        print(f"  Solution preview: {solution[:80].replace(chr(10), ' ')}...")
        print()

    print(f"Total valid: {sum(scores)}/{len(scores)}")
