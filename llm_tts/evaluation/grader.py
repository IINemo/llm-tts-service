"""
Math evaluation logic - exact copy of Qwen2.5-Math/evaluation/grader.py.

Uses direct imports (no subprocess). ANTLR version mismatch warnings may appear
but functionality is correct.
"""

import logging
import re
import signal
from math import isclose
from typing import Union

import regex
from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

log = logging.getLogger(__name__)

# Timeout for symbolic comparison (seconds)
SYMBOLIC_EQUAL_TIMEOUT = 30

# Counter for timed-out comparisons
_timeout_count = 0


def get_timeout_count() -> int:
    """Return the number of symbolic comparisons that timed out."""
    return _timeout_count


def reset_timeout_count():
    """Reset the timeout counter."""
    global _timeout_count
    _timeout_count = 0


class _SymbolicTimeoutError(Exception):
    pass


def _symbolic_timeout_handler(signum, frame):
    raise _SymbolicTimeoutError()


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    pred = pred.rstrip(".").rstrip("/")
    return pred


def normalize_scientific_notation(s):
    """Normalize scientific notation: 3.83e35 <-> 3.83\\times10^{35}"""
    s = str(s).strip()
    s = re.sub(r"\\times\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", r"e\1", s)
    s = re.sub(r"×\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", r"e\1", s)
    s = re.sub(r"(\d)\s*\\cdot\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", r"\1e\2", s)
    return s


def normalize_python_notation(s):
    """Normalize Python/numpy notation to standard math."""
    s = str(s).strip()
    s = re.sub(r"np\.(\w+)", r"\1", s)
    s = re.sub(r"math\.(\w+)", r"\1", s)
    return s


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    num = normalize_scientific_notation(num)
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def is_digit(num):
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)
    return ", ".join(pmatrix_list)


def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)


def _symbolic_equal_impl(a, b):
    """Inner implementation of symbolic equality (no timeout)."""

    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal(a, b):
    """Symbolic equality with timeout protection."""
    global _timeout_count
    old_handler = signal.signal(signal.SIGALRM, _symbolic_timeout_handler)
    signal.alarm(SYMBOLIC_EQUAL_TIMEOUT)
    try:
        result = _symbolic_equal_impl(a, b)
        signal.alarm(0)
        return result
    except _SymbolicTimeoutError:
        _timeout_count += 1
        log.warning(
            "symbolic_equal timed out after %ds comparing %r vs %r "
            "(total timeouts: %d)",
            SYMBOLIC_EQUAL_TIMEOUT,
            str(a)[:100],
            str(b)[:100],
            _timeout_count,
        )
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if prediction is None or reference is None:
        return False

    pred_str = str(prediction).strip()
    ref_str = str(reference).strip()

    if pred_str.lower() == ref_str.lower():
        return True

    # Normalize scientific notation and python notation
    pred_norm = normalize_python_notation(normalize_scientific_notation(pred_str))
    ref_norm = normalize_python_notation(normalize_scientific_notation(ref_str))

    if pred_norm.lower() == ref_norm.lower():
        return True

    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    # Try numeric comparison with normalized strings
    try:
        if is_digit(pred_norm) and is_digit(ref_norm):
            prediction = parse_digits(pred_norm)
            reference = parse_digits(ref_norm)
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # pmatrix (amps)
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    # deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    # [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    if symbolic_equal(prediction, reference):
        return True

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
