"""
MBPP+ evaluator for code generation using EvalPlus methodology.

Two evaluation modes:
1. test: Run code against basic test assertions (test_list field, 3 tests per problem)
2. full: Run EvalPlus evaluation with full test suite (base + plus inputs, ~100+ tests per problem)

Reference: https://github.com/evalplus/evalplus
"""

import ast
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

log = logging.getLogger(__name__)


class EvaluatorMBPPPlus:
    """
    Evaluator for MBPP+ code generation following EvalPlus methodology.

    Modes:
    - test: Run against basic test assertions (test_list, quick evaluation)
    - full: Run EvalPlus evaluation with complete test suite (thorough evaluation)
    """

    def __init__(
        self,
        mode: str = "test",
        timeout: int = 10,
    ):
        """
        Initialize the MBPP+ evaluator.

        Args:
            mode: "test" for basic assertions, "full" for EvalPlus evaluation
            timeout: Timeout per test case in seconds
        """
        if mode not in ("test", "full"):
            raise ValueError(f"Unknown mode: {mode}. Use 'test' or 'full'.")

        self.mode = mode
        self.timeout = timeout
        log.info(f"MBPP+ Evaluator initialized with mode='{mode}'")

    def __call__(
        self,
        problems: List[str],
        solutions: List[str],
        gold_answers: List[str],
        task_ids: Optional[List[str]] = None,
        instance_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """
        Evaluate solutions.

        Args:
            problems: List of problem prompts
            solutions: List of model-generated solutions
            gold_answers: List of gold/reference solutions
            task_ids: List of task IDs (required for full mode)
            instance_data: List of instance data (contains test_list for test mode)

        Returns:
            List of scores (0.0 or 1.0 for each instance)
        """
        if self.mode == "test":
            return self._evaluate_test(solutions, instance_data)
        elif self.mode == "full":
            if task_ids is None:
                raise ValueError("Full evaluation mode requires task_ids parameter.")
            return self._evaluate_full(solutions, task_ids)

    def _evaluate_test(
        self,
        solutions: List[str],
        instance_data: Optional[List[Dict[str, Any]]],
    ) -> List[float]:
        """
        Run code against basic test assertions (test_list).

        This executes the generated code and runs the assertions from test_list.
        """
        if not instance_data:
            raise ValueError("Test mode requires instance_data with test_list field.")

        scores = []
        passed_count = 0
        total_tests = 0

        for idx, solution in enumerate(tqdm(solutions, desc="Running tests")):
            code = self._extract_code(solution)

            # Check syntax first
            if not self._is_valid_python(code):
                log.debug(f"Sample {idx}: Invalid Python syntax")
                scores.append(0.0)
                continue

            # Get test assertions
            test_list = instance_data[idx].get("test_list", [])
            if not test_list:
                raise ValueError(
                    f"No test_list found for sample {idx}. "
                    f"Ensure instance_data contains test_list field."
                )

            # Run tests
            passed, error = self._run_assertions(code, test_list)
            total_tests += len(test_list)

            if passed:
                scores.append(1.0)
                passed_count += 1
            else:
                scores.append(0.0)
                log.debug(f"Sample {idx} failed: {error}")

        log.info(
            f"Test evaluation: {passed_count}/{len(solutions)} passed "
            f"({total_tests} total assertions)"
        )
        return scores

    def _run_assertions(
        self, code: str, test_list: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Run test assertions against generated code.

        Returns:
            Tuple of (passed: bool, error_message: Optional[str])
        """
        # Build test script: code + assertions
        test_code = code + "\n\n"
        for test in test_list:
            test_code += test + "\n"

        try:
            exec_globals: Dict[str, Any] = {}
            exec(compile(test_code, "<test>", "exec"), exec_globals)
            return True, None
        except AssertionError as e:
            return False, f"AssertionError: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _evaluate_full(
        self,
        solutions: List[str],
        task_ids: List[str],
    ) -> List[float]:
        """
        Full evaluation using EvalPlus.

        Creates a samples file and runs evalplus.evaluate command.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples.jsonl"

            # Create samples file in EvalPlus format
            samples = []
            evaluated_task_ids = set()

            for task_id, solution in zip(task_ids, solutions):
                code = self._extract_code(solution)
                # Ensure task_id format is "Mbpp/X"
                task_id_str = self._normalize_task_id(task_id)
                samples.append({"task_id": task_id_str, "solution": code})
                evaluated_task_ids.add(task_id_str)

            # EvalPlus requires all problems to be in the samples file
            # Add dummy entries for problems not being evaluated
            try:
                from evalplus.data import get_mbpp_plus

                all_problems = get_mbpp_plus()

                for full_task_id in all_problems.keys():
                    if full_task_id not in evaluated_task_ids:
                        # Add dummy entry with empty solution
                        samples.append({"task_id": full_task_id, "solution": ""})

                log.info(
                    f"Created samples file with {len(evaluated_task_ids)} evaluated "
                    f"+ {len(samples) - len(evaluated_task_ids)} dummy entries "
                    f"= {len(samples)} total"
                )
            except ImportError:
                log.warning(
                    "Could not import evalplus.data.get_mbpp_plus, "
                    "proceeding with partial samples (may fail)"
                )

            with open(samples_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            log.info(f"Running EvalPlus on {len(samples)} samples...")
            log.info(f"Samples file: {samples_path}")

            # Run evalplus evaluation
            try:
                cmd = [
                    "evalplus.evaluate",
                    "--dataset",
                    "mbpp",
                    "--samples",
                    str(samples_path),
                    "--i-just-wanna-run",
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout * len(solutions) + 300,
                )

                log.info(f"EvalPlus stdout:\n{result.stdout}")
                if result.stderr:
                    log.warning(f"EvalPlus stderr:\n{result.stderr}")

                if result.returncode != 0:
                    log.error(f"EvalPlus failed with return code {result.returncode}")
                    return [0.0] * len(solutions)

                # Parse results from EvalPlus output
                return self._parse_evalplus_output(result.stdout, task_ids, tmpdir)

            except subprocess.TimeoutExpired:
                log.error("EvalPlus evaluation timed out")
                return [0.0] * len(solutions)
            except FileNotFoundError:
                log.error(
                    "evalplus command not found. Install with: pip install evalplus"
                )
                return [0.0] * len(solutions)

    def _parse_evalplus_output(
        self,
        stdout: str,
        task_ids: List[str],
        tmpdir: str,
    ) -> List[float]:
        """Parse EvalPlus output to get per-task results."""
        # Try to find the results JSON file
        results_pattern = Path(tmpdir).glob("*_eval_results.json")
        for results_file in results_pattern:
            try:
                with open(results_file) as f:
                    results = json.load(f)

                scores = []
                for task_id in task_ids:
                    task_id_str = self._normalize_task_id(task_id)
                    task_results = results.get("eval", {}).get(task_id_str, [])
                    # Check if any solution passed (for greedy, there's only one)
                    if task_results and task_results[0].get("base_status") == "pass":
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                return scores
            except Exception as e:
                log.warning(f"Failed to parse results file: {e}")

        # Fallback: parse pass@1 from stdout
        match = re.search(r"pass@1:\s*([\d.]+)", stdout)
        if match:
            pass_rate = float(match.group(1))
            log.info(f"EvalPlus pass@1: {pass_rate}")
            # Can't get per-task scores, return overall rate for all
            return [pass_rate] * len(task_ids)

        log.warning("Could not parse EvalPlus results, defaulting to 0.0")
        return [0.0] * len(task_ids)

    def _normalize_task_id(self, task_id: Any) -> str:
        """Ensure task_id is in 'Mbpp/X' format."""
        if isinstance(task_id, int):
            return f"Mbpp/{task_id}"
        task_id_str = str(task_id)
        if not task_id_str.startswith("Mbpp/"):
            return f"Mbpp/{task_id_str}"
        return task_id_str

    def _extract_code(self, solution: str) -> str:
        """Extract Python code from model response."""
        if not solution:
            return ""

        # Try to extract from code blocks
        code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
        code_blocks = re.findall(code_block_pattern, solution, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()

        # Try to find function definition
        func_pattern = r"(def \w+\s*\([^)]*\):.*?)(?:\n\n|\Z)"
        func_matches = re.findall(func_pattern, solution, re.DOTALL)
        if func_matches:
            return func_matches[-1].strip()

        # Return as-is
        return solution.strip()

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        if not code or not code.strip():
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _score_single(self, inp: Tuple[str, str, str]) -> float:
        """Score single prediction (for compatibility)."""
        raise NotImplementedError(
            "Use __call__ method for batch evaluation instead of _score_single."
        )
