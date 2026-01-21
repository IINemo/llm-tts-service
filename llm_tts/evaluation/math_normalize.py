"""
Answer normalization logic from Qwen2.5-Math evaluation.

This module provides `strip_string` which is the official normalization function
used by Qwen2.5-Math for benchmark evaluation. Using this ensures our results
are comparable to official benchmarks.

Sources:
- https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/parser.py
- https://github.com/hendrycks/math (original MATH dataset)
"""

import re
from typing import Optional

try:
    from word2number import w2n

    WORD2NUMBER_AVAILABLE = True
except ImportError:
    WORD2NUMBER_AVAILABLE = False


# Unit texts to remove (from official Qwen2.5-Math parser)
unit_texts = [
    "east",
    "move",
    "west",
    "south",
    "north",
    "square",
    "ways",
    "games",
    "booklet",
    "booklets",
    "pieces",
    "units",
    " of ",
    "gallons",
    "googol",
    "degree",
    "balls",
    "degree",
    "degrees",
    "mph",
    "kilometers",
    "meters",
    " meter",
    " inch",
    "centimeters",
    " feet",
    " foot",
    "inches",
    " mile",
    "miles",
    " cm",
    " mm",
    " m ",
    "hour",
    "hours",
    "minute",
    "minutes",
    "second",
    "seconds",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "digit",
    "digits",
]


def convert_word_number(text: str) -> str:
    """Convert word numbers to digits using word2number."""
    if not WORD2NUMBER_AVAILABLE:
        return text
    try:
        return str(w2n.word_to_num(text))
    except Exception:
        return text


def _fix_fracs(string):
    """Fix fraction formatting: \frac1b -> \frac{1}{b}"""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    """Convert a/b to \frac{a}{b} for simple integer fractions."""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _fix_sqrt(string):
    r"""Fix sqrt formatting: \sqrt3 -> \sqrt{3}"""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def find_box(pred_str: str) -> str:
    """Extract content from \\boxed{} with proper brace matching."""
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def extract_answer(
    pred_str: str, data_name: str = None, use_last_number: bool = True
) -> str:
    """
    Official Qwen2.5-Math answer extraction function.

    This is the authoritative extraction function from the official
    Qwen2.5-Math evaluation code. Using this ensures benchmark compatibility.

    Args:
        pred_str: The model output string to extract answer from
        data_name: Dataset name for format-specific handling - REQUIRED
        use_last_number: Whether to fall back to last number extraction

    Returns:
        Extracted and normalized answer string
    """
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math format
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:
        # use the last number
        if use_last_number:
            pattern = r"-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # multiple line cleanup
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def strip_string(string, skip_unit=False):
    """
    Official Qwen2.5-Math string normalization for answer comparison.

    This is the authoritative normalization function from the official
    Qwen2.5-Math evaluation code. Using this ensures benchmark compatibility.

    Args:
        string: The answer string to normalize
        skip_unit: Whether to skip unit removal (for certain datasets)

    Returns:
        Normalized string for comparison
    """
    string = str(string).strip()

    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # matrix normalization
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # Handle bracketed alphanumeric strings
    if (
        string.startswith("{")
        and string.endswith("}")
        and string[1:-1].isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string[1:-1].isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string[1:-1].isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string = string.replace("'", "")
    string = string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # Fix fractions
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset
    string = _fix_a_slash_b(string)

    return string


# Legacy function for backward compatibility
def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Legacy normalization function. Use strip_string for official compatibility."""
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return strip_string(answer)
    except Exception:
        return answer
