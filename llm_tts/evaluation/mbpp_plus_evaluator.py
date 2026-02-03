"""
MBPP+ evaluator for code generation.

This evaluator supports two modes:
1. Syntax validation: Quick check that code is valid Python
2. Full evaluation: Run code against EvalPlus test suite

For full evaluation, you need:
- evalplus package: pip install evalplus
"""

import ast
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

log = logging.getLogger(__name__)

# Try to import evalplus
try:
    from evalplus.data import get_mbpp_plus

    EVALPLUS_AVAILABLE = True
except ImportError:
    EVALPLUS_AVAILABLE = False


class EvaluatorMBPPPlus:
    """
    Evaluator for MBPP+ code generation.

    Supports syntax validation and full test execution via EvalPlus.
    """

    def __init__(
        self,
        mode: str = "syntax",
        timeout: int = 10,
        min_time_limit: float = 1.0,
        gt_time_limit_factor: float = 4.0,
    ):
        """
        Initialize the MBPP+ evaluator.

        Args:
            mode: Evaluation mode
                - "syntax": Check if generated code is valid Python
                - "full": Run against EvalPlus test suite
            timeout: Timeout per test case in seconds (full mode)
            min_time_limit: Minimum time limit for test execution
            gt_time_limit_factor: Factor to multiply ground truth time
        """
        self.mode = mode
        self.timeout = timeout
        self.min_time_limit = min_time_limit
        self.gt_time_limit_factor = gt_time_limit_factor

        if mode == "full" and not EVALPLUS_AVAILABLE:
            log.warning(
                "evalplus not installed. Install with: pip install evalplus\n"
                "Falling back to syntax validation mode."
            )
            self.mode = "syntax"

        # Load problems for full evaluation
        if self.mode == "full":
            self._problems = get_mbpp_plus()
        else:
            self._problems = None

    def _score_single(self, inp: Tuple[str, str, str]) -> float:
        """
        Score a single prediction.

        Args:
            inp: Tuple of (question, solution, gold_answer)

        Returns:
            Score (0.0 or 1.0)
        """
        _, solution, _ = inp

        if self.mode == "syntax":
            code = self._extract_code(solution)
            if self._is_valid_python(code):
                return 1.0
            return 0.0

        # Full mode doesn't support single scoring efficiently
        log.warning("Full evaluation mode doesn't support single scoring")
        return 0.0

    def __call__(
        self,
        problems: List[str],
        solutions: List[str],
        gold_answers: List[str],
        task_ids: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Evaluate solutions.

        Args:
            problems: List of problem prompts
            solutions: List of model-generated solutions
            gold_answers: List of gold/reference solutions
            task_ids: Optional list of task IDs (required for full evaluation)

        Returns:
            List of scores (0.0 or 1.0 for each instance)
        """
        if self.mode == "syntax":
            return self._evaluate_syntax(solutions)
        elif self.mode == "full":
            if task_ids is None:
                log.warning(
                    "Full evaluation requires task_ids. "
                    "Falling back to syntax validation."
                )
                return self._evaluate_syntax(solutions)
            return self._evaluate_full(solutions, task_ids)
        else:
            raise ValueError(f"Unknown evaluation mode: {self.mode}")

    def _evaluate_syntax(self, solutions: List[str]) -> List[float]:
        """
        Validate that solutions are valid Python code.

        Args:
            solutions: List of model-generated solutions

        Returns:
            List of scores (1.0 if valid Python, 0.0 otherwise)
        """
        scores = []
        valid_count = 0
        has_function_count = 0

        for solution in tqdm(solutions, desc="Validating syntax"):
            code = self._extract_code(solution)

            if self._is_valid_python(code):
                valid_count += 1
                score = 1.0

                # Bonus check: has a function definition
                if self._has_function_definition(code):
                    has_function_count += 1
            else:
                score = 0.0

            scores.append(score)

        log.info(
            f"Syntax validation: {valid_count}/{len(solutions)} valid Python, "
            f"{has_function_count}/{len(solutions)} with function definitions"
        )

        return scores

    def _evaluate_full(
        self,
        solutions: List[str],
        task_ids: List[str],
    ) -> List[float]:
        """
        Full evaluation using EvalPlus test suite.

        Args:
            solutions: List of model-generated solutions
            task_ids: List of task IDs

        Returns:
            List of scores (1.0 if all tests pass, 0.0 otherwise)
        """
        # Create temporary directory for samples
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples.jsonl"
            results_path = Path(tmpdir) / "results.json"  # noqa

            # Create samples file
            samples = []
            for task_id, solution in zip(task_ids, solutions):
                code = self._extract_code(solution)
                samples.append(
                    {
                        "task_id": task_id,
                        "solution": code,
                    }
                )

            with open(samples_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            log.info(f"Running EvalPlus evaluation on {len(samples)} samples...")

            # Run evalplus evaluation
            try:
                cmd = [
                    "evalplus.evaluate",
                    "--dataset",
                    "mbpp",
                    "--samples",
                    str(samples_path),
                    "--i-just-wanna-run",  # Skip safety prompts
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout * len(solutions) + 300,
                )

                if result.returncode != 0:
                    log.error(f"EvalPlus evaluation failed: {result.stderr}")
                    return [0.0] * len(solutions)

                # Parse results
                return self._parse_evalplus_results(result.stdout, task_ids)

            except subprocess.TimeoutExpired:
                log.error("EvalPlus evaluation timed out")
                return [0.0] * len(solutions)
            except FileNotFoundError:
                log.error(
                    "evalplus command not found. Install with: pip install evalplus"
                )
                return [0.0] * len(solutions)

    def _parse_evalplus_results(
        self,
        output: str,
        task_ids: List[str],
    ) -> List[float]:
        """Parse EvalPlus output to extract pass/fail for each task."""
        scores = []

        # EvalPlus outputs results in various formats
        # Try to find pass rate or individual results
        for task_id in task_ids:
            # Default to 0 if we can't determine
            scores.append(0.0)

        # Try to parse pass@1 from output
        import re

        match = re.search(r"pass@1:\s*([\d.]+)", output)
        if match:
            pass_rate = float(match.group(1))
            log.info(f"EvalPlus pass@1: {pass_rate}")

        return scores

    def _extract_code(self, solution: str) -> str:
        """Extract Python code from model solution."""
        from llm_tts.datasets.mbpp_plus import extract_code_from_response

        return extract_code_from_response(solution)

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        if not code or not code.strip():
            return False

        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _has_function_definition(self, code: str) -> bool:
        """Check if code contains a function definition."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return True
            return False
        except SyntaxError:
            return False

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
            List of detailed result dicts
        """
        results = []

        for solution, instance in zip(solutions, instance_data):
            code = self._extract_code(solution)
            is_valid = self._is_valid_python(code)
            has_function = self._has_function_definition(code) if is_valid else False

            # Count lines of code
            code_lines = (
                len([l for l in code.split("\n") if l.strip()]) if code else 0  # noqa
            )

            result = {
                "task_id": instance.get("task_id", ""),
                "entry_point": instance.get("entry_point", ""),
                "extracted_code": code,
                "code_length": len(code),
                "code_lines": code_lines,
                "is_valid_python": is_valid,
                "has_function_definition": has_function,
                "prompt_length": len(instance.get("question", "")),
                "gold_solution_length": len(instance.get("answer", "")),
            }

            results.append(result)

        return results


def run_evalplus_cli(
    samples_path: str,
    dataset: str = "mbpp",
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Run EvalPlus evaluation using the CLI.

    Args:
        samples_path: Path to samples JSONL file
        dataset: Dataset name ("mbpp" or "humaneval")
        timeout: Timeout per test case

    Returns:
        Dict with evaluation results
    """
    cmd = [
        "evalplus.evaluate",
        "--dataset",
        dataset,
        "--samples",
        samples_path,
        "--i-just-wanna-run",
    ]

    log.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * 1000 + 600,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except FileNotFoundError:
        return {"error": "evalplus not found"}


if __name__ == "__main__":
    # Test the evaluator
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing MBPP+ Evaluator ===\n")

    evaluator = EvaluatorMBPPPlus(mode="syntax")

    # Test cases
    test_solutions = [
        # Valid Python with function
        """
```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
```
""",
        # Valid Python without code block
        """
def add_numbers(a, b):
    return a + b
""",
        # Invalid Python
        """
def broken_function(
    return 42
""",
        # Empty response
        "",
        # Just explanation, no code
        "You should implement a function that adds two numbers.",
    ]

    problems = ["prompt"] * len(test_solutions)
    gold_answers = ["answer"] * len(test_solutions)

    scores = evaluator(problems, test_solutions, gold_answers)

    for i, (solution, score) in enumerate(zip(test_solutions, scores)):
        status = "VALID" if score == 1.0 else "INVALID"
        preview = solution[:50].replace("\n", " ").strip()
        print(f"Test {i + 1}: {status}")
        print(f"  Preview: {preview}...")
        print()

    print(f"Total valid: {sum(scores)}/{len(scores)}")
