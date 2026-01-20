"""
Math evaluation via subprocess using qwen-eval conda environment.

The qwen-eval environment has correct library versions:
- sympy==1.12
- antlr4-python3-runtime==4.11.1
- latex2sympy2 (Qwen's version)

This avoids ANTLR version conflicts with hydra/omegaconf in main environment.
"""

import json
import logging
import os
import select
import subprocess
from pathlib import Path
from typing import Union

_EVAL_TIMEOUT = 5  # seconds per evaluation

# Subprocess-based math evaluation using qwen-eval environment
_SUBPROCESS_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "math_eval_subprocess.py"
_CONDA_ENV = "qwen-eval"

_subprocess_proc = None
_log = logging.getLogger(__name__)


def _get_subprocess():
    """Get or create persistent subprocess for math evaluation."""
    global _subprocess_proc
    if _subprocess_proc is None or _subprocess_proc.poll() is not None:
        conda_base = os.path.expanduser("~/miniconda3")
        python_path = f"{conda_base}/envs/{_CONDA_ENV}/bin/python"
        if not os.path.exists(python_path):
            raise RuntimeError(
                f"qwen-eval conda environment not found at {python_path}. "
                f"Create it with: conda create -n qwen-eval python=3.11 && "
                f"conda activate qwen-eval && pip install sympy==1.12 antlr4-python3-runtime==4.11.1"
            )
        _subprocess_proc = subprocess.Popen(
            [python_path, str(_SUBPROCESS_SCRIPT), "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    return _subprocess_proc


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Check math equality using subprocess with correct library versions.
    """
    if prediction is None or reference is None:
        return False

    pred_str = str(prediction).strip()
    gold_str = str(reference).strip()

    # Quick check for exact string match
    if pred_str.lower() == gold_str.lower():
        return True

    # Use subprocess for all evaluation
    global _subprocess_proc
    proc = _get_subprocess()
    try:
        request = json.dumps({"pred": pred_str, "gold": gold_str})
        proc.stdin.write(request + "\n")
        proc.stdin.flush()

        # Wait for response with timeout
        ready, _, _ = select.select([proc.stdout], [], [], _EVAL_TIMEOUT)
        if ready:
            response = proc.stdout.readline().strip()
            if response:
                result = json.loads(response)
                return result.get("result", False)
        else:
            # Timeout - kill and restart subprocess
            _log.warning(f"Subprocess timeout for pred={pred_str[:50]}, gold={gold_str[:50]}")
            proc.kill()
            _subprocess_proc = None
            return False
    except Exception as e:
        _log.error(f"Subprocess eval failed: {e}")
        # Restart subprocess on next call
        proc.kill()
        _subprocess_proc = None
        return False

    return False


# Legacy compatibility wrapper
def grade_answer(given_answer: str, ground_truth: str, timeout: bool = False) -> bool:
    """Legacy grader wrapper that uses math_equal."""
    if given_answer is None:
        return False
    try:
        return math_equal(str(given_answer), str(ground_truth), timeout=timeout)
    except Exception:
        return False
