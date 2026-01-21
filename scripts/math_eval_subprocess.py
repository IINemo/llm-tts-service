#!/usr/bin/env python
"""
Standalone math evaluation script - exact copy of Qwen2.5-Math/evaluation/grader.py logic.
Uses qwen-eval conda environment with correct library versions.

Usage:
    python math_eval_subprocess.py --batch < input.jsonl > output.jsonl
"""

import json
import re
import sys
from math import isclose
from typing import Union

import regex
from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


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
    """Normalize scientific notation: 3.83e35 <-> 3.83\times10^{35}"""
    s = str(s).strip()
    # Convert LaTeX \times10^{n} to e notation
    s = re.sub(r"\\times\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", r"e\1", s)
    # Convert ×10^n to e notation
    s = re.sub(r"×\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", r"e\1", s)
    # Convert 10^{n} alone to e notation when preceded by number
    s = re.sub(r"(\d)\s*\\cdot\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", r"\1e\2", s)
    return s


def normalize_python_notation(s):
    """Normalize Python/numpy notation to standard math."""
    s = str(s).strip()
    # np.func -> func
    s = re.sub(r"np\.(\w+)", r"\1", s)
    # math.func -> func
    s = re.sub(r"math\.(\w+)", r"\1", s)
    return s


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    # Try to parse scientific notation first
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


def symbolic_equal(a, b):
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


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    if prediction is None or reference is None:
        return False

    # Normalize both for comparison
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


def main():
    # Test mode - verify environment is correctly set up
    if "--test" in sys.argv:
        print("Testing math evaluation environment...")
        test_cases = [
            ("2", "2", True),
            ("\\frac{1}{2}", "0.5", True),
            ("x^2 + 2x + 1", "(x+1)^2", True),
            ("\\boxed{42}", "42", True),
        ]
        all_passed = True
        for pred, gold, expected in test_cases:
            result = math_equal(pred, gold)
            status = "✓" if result == expected else "✗"
            if result != expected:
                all_passed = False
            print(f"  {status} math_equal('{pred}', '{gold}') = {result} (expected {expected})")
        if all_passed:
            print("\n✓ All tests passed! Math evaluation environment is working correctly.")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)

    batch_mode = "--batch" in sys.argv

    if batch_mode:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pred = data.get("pred", "")
                gold = data.get("gold", "")
                result = math_equal(pred, gold)
                print(json.dumps({"result": result}))
            except Exception as e:
                print(json.dumps({"result": False, "error": str(e)}))
            sys.stdout.flush()
    else:
        try:
            data = json.load(sys.stdin)
            pred = data.get("pred", "")
            gold = data.get("gold", "")
            result = math_equal(pred, gold)
            print(json.dumps({"result": result}))
        except Exception as e:
            print(json.dumps({"result": False, "error": str(e)}))


if __name__ == "__main__":
    main()
