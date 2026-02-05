"""
MBPP+ evaluator for code generation using EvalPlus methodology.

Uses EvalPlus evaluation with full test suite (base + plus inputs, ~100+ tests per problem).

Reference: https://github.com/evalplus/evalplus
"""

import ast
import json
import logging
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class EvaluatorMBPPPlus:
    """
    Evaluator for MBPP+ code generation following EvalPlus methodology.

    Uses EvalPlus evaluation with complete test suite for thorough evaluation.
    """

    def __init__(
        self,
        mode: str = "full",
        timeout: int = 10,
    ):
        """
        Initialize the MBPP+ evaluator.

        Args:
            mode: Evaluation mode. Only "full" is supported. Other modes are deprecated.
            timeout: Timeout per test case in seconds
        """
        # Handle deprecated modes
        if mode in ("test", "syntax"):
            warnings.warn(
                f"Mode '{mode}' is deprecated and will be removed in a future version. "
                f"Using 'full' mode instead (EvalPlus evaluation).",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "full"

        if mode != "full":
            raise ValueError(
                f"Unknown mode: {mode}. Only 'full' mode is supported. "
                f"Modes 'test' and 'syntax' are deprecated."
            )

        self.mode = mode
        self.timeout = timeout
        log.info(f"MBPP+ Evaluator initialized with mode='{mode}' (EvalPlus)")

    def __call__(
        self,
        problems: List[str],
        solutions: List[str],
        gold_answers: List[str],
        task_ids: Optional[List[str]] = None,
        instance_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """
        Evaluate solutions using EvalPlus.

        Args:
            problems: List of problem prompts
            solutions: List of model-generated solutions
            gold_answers: List of gold/reference solutions (unused, kept for API compatibility)
            task_ids: List of task IDs (required)
            instance_data: List of instance data (unused, kept for API compatibility)

        Returns:
            List of scores (0.0 or 1.0 for each instance)
        """
        if task_ids is None:
            raise ValueError("EvalPlus evaluation requires task_ids parameter.")
        return self._evaluate_full(solutions, task_ids)

    # def _evaluate_full(
    #     self,
    #     solutions: List[str],
    #     task_ids: List[str],
    # ) -> List[float]:
    #     """
    #     Full evaluation using EvalPlus.

    #     Creates a samples file and runs evalplus.evaluate command.
    #     """
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         samples_path = Path(tmpdir) / "samples.jsonl"

    #         # Create samples file in EvalPlus format
    #         samples = []
    #         evaluated_task_ids = set()

    #         for task_id, solution in zip(task_ids, solutions):
    #             code = self._extract_code(solution)
    #             # Ensure task_id format is "Mbpp/X"
    #             task_id_str = self._normalize_task_id(task_id)
    #             samples.append({"task_id": task_id_str, "solution": code})
    #             evaluated_task_ids.add(task_id_str)

    #         # EvalPlus requires all problems to be in the samples file
    #         # Add dummy entries for problems not being evaluated
    #         try:
    #             from evalplus.data import get_mbpp_plus

    #             all_problems = get_mbpp_plus()

    #             for full_task_id in all_problems.keys():
    #                 if full_task_id not in evaluated_task_ids:
    #                     # Add dummy entry with empty solution
    #                     samples.append({"task_id": full_task_id, "solution": ""})

    #             log.info(
    #                 f"Created samples file with {len(evaluated_task_ids)} evaluated "
    #                 f"+ {len(samples) - len(evaluated_task_ids)} dummy entries "
    #                 f"= {len(samples)} total"
    #             )
    #         except ImportError:
    #             log.warning(
    #                 "Could not import evalplus.data.get_mbpp_plus, "
    #                 "proceeding with partial samples (may fail)"
    #             )

    #         with open(samples_path, "w") as f:
    #             for sample in samples:
    #                 f.write(json.dumps(sample) + "\n")

    #         log.info(f"Running EvalPlus on {len(samples)} samples...")
    #         log.info(f"Samples file: {samples_path}")

    #         # Run evalplus evaluation
    #         try:
    #             cmd = [
    #                 "evalplus.evaluate",
    #                 "--dataset",
    #                 "mbpp",
    #                 "--samples",
    #                 str(samples_path),
    #                 "--i-just-wanna-run",
    #             ]

    #             result = subprocess.run(
    #                 cmd,
    #                 capture_output=True,
    #                 text=True,
    #                 timeout=self.timeout * len(solutions) + 300,
    #             )

    #             log.info(f"EvalPlus stdout:\n{result.stdout}")
    #             if result.stderr:
    #                 log.warning(f"EvalPlus stderr:\n{result.stderr}")

    #             if result.returncode != 0:
    #                 log.error(f"EvalPlus failed with return code {result.returncode}")
    #                 return [0.0] * len(solutions)

    #             # Parse results from EvalPlus output
    #             return self._parse_evalplus_output(result.stdout, task_ids, tmpdir)

    #         except subprocess.TimeoutExpired:
    #             log.error("EvalPlus evaluation timed out")
    #             return [0.0] * len(solutions)
    #         except FileNotFoundError:
    #             log.error(
    #                 "evalplus command not found. Install with: pip install evalplus"
    #             )
    #             return [0.0] * len(solutions)

    def _evaluate_full(
        self,
        solutions: List[str],
        task_ids: List[str],
    ) -> List[float]:
        """
        Full evaluation using EvalPlus.

        Creates a samples file with dummy entries for all MBPP+ problems.
        EvalPlus requires all 378 problems to be present, so we include empty
        solutions for problems we're not evaluating.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples.jsonl"

            # Load the full MBPP+ dataset
            from evalplus.data import get_mbpp_plus

            all_problems = get_mbpp_plus()

            # Create samples file with actual solutions + dummy entries
            samples = []
            evaluated_task_ids = set()

            # Add actual solutions for tasks being evaluated
            for task_id, solution in zip(task_ids, solutions):
                code = self._extract_code(solution)
                task_id_str = self._normalize_task_id(task_id)
                samples.append({"task_id": task_id_str, "solution": code})
                evaluated_task_ids.add(task_id_str)

            # Add dummy entries for all other problems (required by EvalPlus)
            for task_id_str in all_problems.keys():
                if task_id_str not in evaluated_task_ids:
                    samples.append({"task_id": task_id_str, "solution": ""})

            with open(samples_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            log.info(
                f"Created samples file: {len(evaluated_task_ids)} evaluated + "
                f"{len(samples) - len(evaluated_task_ids)} dummy = {len(samples)} total"
            )
            log.info(
                f"Running EvalPlus on {len(evaluated_task_ids)} actual problems..."
            )

            # Run evalplus evaluation via subprocess (for process isolation)
            cmd = [
                "evalplus.evaluate",
                "--dataset",
                "mbpp",
                "--samples",
                str(samples_path),
                "--i-just-wanna-run",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout * len(solutions) + 300,
                    cwd=None,  # Don't inherit parent process state
                )

                log.info(f"EvalPlus stdout:\n{result.stdout}")
                if result.stderr:
                    log.warning(f"EvalPlus stderr:\n{result.stderr}")

                if result.returncode != 0:
                    raise RuntimeError(
                        f"EvalPlus failed with return code {result.returncode}"
                    )

                # Parse results from EvalPlus JSON output file
                return self._parse_evalplus_output(task_ids, tmpdir)

            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"EvalPlus evaluation timed out after "
                    f"{self.timeout * len(solutions) + 300} seconds"
                )

    def _parse_evalplus_output(
        self,
        task_ids: List[str],
        tmpdir: str,
    ) -> List[float]:
        """Parse EvalPlus output to get per-task results from JSON file."""
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

                        # Log detailed failure information
                        if task_results:
                            result = task_results[0]
                            base_status = result.get("base_status", "unknown")
                            plus_status = result.get("plus_status", "unknown")

                            log.info(
                                f"Task {task_id_str} failed: "
                                f"base={base_status}, plus={plus_status}"
                            )

                            # Log which specific tests failed
                            base_tests = result.get("base", [])
                            for i, test in enumerate(base_tests):
                                if test.get("status") == "fail":
                                    details = test.get("details", "No details")
                                    log.info(
                                        f"  Base test {i+1} failed: {details[:200]}"
                                    )

                            plus_tests = result.get("plus", [])
                            for i, test in enumerate(plus_tests):
                                if test.get("status") == "fail":
                                    details = test.get("details", "No details")
                                    log.info(
                                        f"  Plus test {i+1} failed: {details[:200]}"
                                    )

                return scores
            except Exception as e:
                # Re-raise to fail fast
                raise RuntimeError(f"Failed to parse EvalPlus results file: {e}") from e

        # No results file found - this is an error condition
        raise FileNotFoundError(
            f"EvalPlus results file not found in {tmpdir}. "
            f"Check EvalPlus output for errors."
        )

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
