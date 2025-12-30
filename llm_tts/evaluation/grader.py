"""
Answer checker API that uses sympy to simplify expressions and check for equality.

This is based on the Hendrycks' MATH release (math_equivalence), and incorporates
logic from various sources:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
- https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/grader.py

Call grade_answer(given_answer: str, ground_truth: str) for legacy grading.
Call math_equal(prediction, reference) for comprehensive math grading.
"""

import multiprocessing
import re
from math import isclose
from typing import Union

import regex
import sympy
from pylatexenc import latex2text
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex

from . import math_normalize

# Try to import latex2sympy2 for better LaTeX parsing
try:
    from latex2sympy2 import latex2sympy

    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False
    latex2sympy = None


# ============================================================================
# Math Equal - comprehensive math answer comparison
# ============================================================================


def parse_digits(num):
    """Parse a number string, handling commas and percentages."""
    num = regex.sub(",", "", str(num))
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
    """Check if a string can be parsed as a digit."""
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    """Convert {a,b} style vectors to LaTeX pmatrix format."""
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def choice_answer_clean(pred: str):
    """Clean up multiple choice answers."""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    pred = pred.rstrip(".").rstrip("/")
    return pred


def numeric_equal(prediction: float, reference: float) -> bool:
    """Check if two numbers are equal within relative tolerance."""
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b) -> bool:
    """Check if two expressions are symbolically equal using sympy."""

    def _parse(s):
        """Try multiple parsers to convert string to sympy expression."""
        parsers = [parse_latex, sympy_parser.parse_expr]
        if LATEX2SYMPY_AVAILABLE:
            parsers.append(latex2sympy)

        for f in parsers:
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

    # Direct string/value equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # Simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # Equation equal (for expressions like "x = 5")
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    # Numeric evaluation
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # Matrix comparison
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    """Wrapper for multiprocessing timeout."""
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=5, **kwargs):
    """Call a function with a timeout using multiprocessing.

    Returns the function result, or None if timeout/error occurred.
    Note: Returns None (not False) on failure to distinguish from actual False results.
    """
    try:
        output_queue = multiprocessing.Queue()
        process_args = args + (output_queue,)
        process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return None  # Timeout - return None to indicate failure

        try:
            return output_queue.get_nowait()
        except Exception:
            return None  # Queue error - return None to indicate failure
    except Exception:
        # Multiprocessing can fail after tokenizer fork in CI - return None to trigger fallback
        return None


def _normalize_spaces(s: str) -> str:
    """Remove all spaces for comparison."""
    return re.sub(r"\s+", "", s)


def _normalize_text_command(s: str) -> str:
    """Normalize \\text{} commands by stripping internal spaces."""
    # \text{ Navin} -> \text{Navin}
    return re.sub(r"\\text\{\s*([^}]*?)\s*\}", r"\\text{\1}", s)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Comprehensive math answer comparison.

    Exact match if and only if:
    1. String equal (case-insensitive)
    2. Numerical equal: both can convert to float and are equal
    3. Symbolic equal: both can convert to sympy expression and are equal

    Args:
        prediction: The predicted answer
        reference: The ground truth answer
        include_percentage: Whether to check percentage equivalents (e.g., 0.5 == 50%)
        is_close: Whether to use relative tolerance for numeric comparison
        timeout: Whether to use timeout for symbolic comparison

    Returns:
        True if answers are mathematically equal
    """
    if prediction is None or reference is None:
        return False

    pred_str = str(prediction).strip()
    ref_str = str(reference).strip()

    # 0. Direct string match (case-insensitive)
    if pred_str.lower() == ref_str.lower():
        return True

    # 0a. Match after removing all spaces (handles "x^3 + 3x - 6" vs "x^3+3x-6")
    if _normalize_spaces(pred_str).lower() == _normalize_spaces(ref_str).lower():
        return True

    # 0b. Match after normalizing \text{} commands
    pred_text_norm = _normalize_text_command(pred_str)
    ref_text_norm = _normalize_text_command(ref_str)
    if pred_text_norm.lower() == ref_text_norm.lower():
        return True

    # Multiple choice answer handling
    if reference in ["A", "B", "C", "D", "E"]:
        if choice_answer_clean(str(prediction)) == reference:
            return True

    # 1. Numerical equal
    try:
        if is_digit(prediction) and is_digit(reference):
            pred_val = parse_digits(prediction)
            ref_val = parse_digits(reference)

            if include_percentage:
                gt_variants = [ref_val / 100, ref_val, ref_val * 100]
            else:
                gt_variants = [ref_val]

            for item in gt_variants:
                try:
                    if is_close:
                        if numeric_equal(pred_val, item):
                            return True
                    else:
                        if item == pred_val:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. Symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # Handle pmatrix conversion
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    # Handle bracket normalization [], (), {}
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

    # Element-wise comparison for tuples/intervals [a, b] vs [c, d]
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(
                    pred_parts[i].strip(),
                    ref_parts[i].strip(),
                    include_percentage,
                    is_close,
                )
                for i in range(len(pred_parts))
            ):
                return True

    # Matrix comparison (pmatrix/bmatrix)
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
                        math_equal(
                            pred_parts[i].strip(),
                            ref_parts[i].strip(),
                            include_percentage,
                            is_close,
                        )
                        for i in range(len(pred_parts))
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

    # Equation handling (x = 5 style)
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

    # Symbolic equal with optional timeout
    if timeout:
        result = call_with_timeout(symbolic_equal_process, prediction, reference)
        if result is True:
            return True
        elif result is None:
            # Multiprocessing failed (timeout/fork issue) - fall back to direct call
            if symbolic_equal(prediction, reference):
                return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


# Alias for backward compatibility
def grade_answer_qwen(
    given_answer: str, ground_truth: str, timeout: bool = True
) -> bool:
    """
    Grade answer using comprehensive math_equal function.

    This provides compatibility with Qwen2.5-Math benchmark grading.
    """
    if given_answer is None:
        return False

    try:
        return math_equal(str(given_answer), str(ground_truth), timeout=timeout)
    except Exception:
        # Fall back to simpler grader on any error
        return grade_answer(given_answer, ground_truth)


# ============================================================================
# Legacy grader (kept for backward compatibility)
# ============================================================================

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex_legacy(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex_legacy(expr)
        except Exception:
            pass

    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def _numeric_equal_with_percentage(
    given: str, ground_truth: str, rel_tol: float = 1e-4
) -> bool:
    """Check numeric equality including percentage equivalence."""
    try:
        given_val = float(given.replace(",", "").replace("%", "").strip())
        gt_val = float(ground_truth.replace(",", "").replace("%", "").strip())

        gt_variants = [gt_val, gt_val / 100, gt_val * 100]

        for variant in gt_variants:
            if abs(variant) < 1e-9:
                if abs(given_val) < 1e-9:
                    return True
            elif abs(given_val - variant) / abs(variant) <= rel_tol:
                return True

        return False
    except (ValueError, TypeError):
        return False


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False

    if _numeric_equal_with_percentage(given_normalized, ground_truth_normalized):
        return True

    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """Split elements in a tuple/interval, handling well-formatted commas."""
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    Legacy grader. The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct
