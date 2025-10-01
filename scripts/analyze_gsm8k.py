#!/usr/bin/env python3
"""
Analyze GSM8K-style results by Exact Match (EM), with optional symbolic verification.

Supports the provided results structure (objects containing `base` and/or `uncert`)
and is resilient to files being either a JSON array or JSONL (one JSON object per line).

Optional: if a gold file is provided, EM is computed by comparing the normalized
predicted answers with the normalized gold answers. Without a gold file, EM is
computed using the precomputed `correct` boolean in the results (if present).

Optional: symbolic verification (`--symbolic-verify`) uses sympy (if installed)
to compare expressions numerically/symbolically when a gold answer is provided.

Usage examples:
  python3 scripts/analyze_gsm8k.py \
    --input /home/s.senichev/reasonscale/llm-tts-service/test_together_result.jsonl

  python scripts/analyze_gsm8k.py \
    --input /home/s.senichev/reasonscale/llm-tts-service/test_together_result.jsonl \
    --gold /path/to/gsm8k_gold.jsonl --symbolic-verify

The gold file is expected to be JSONL (or JSON array) with at least the fields:
  - question: str
  - answer: str

If your gold format differs (e.g., uses `label` or `final_answer`), the script
attempts to autodetect common field names.
"""

import argparse
import json
import math
import os
import re
import sys
from decimal import Decimal, InvalidOperation, getcontext
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        # Try JSON array first
        if content[0] == "[":
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        # Fallback to JSONL
        records: List[Dict[str, Any]] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                # Ignore malformed lines silently
                continue
        return records


def infer_gold_answer(obj: Dict[str, Any]) -> Optional[str]:
    # Try common field names
    for key in ("answer", "label", "final_answer", "gold", "gold_answer"):
        if key in obj and obj[key] is not None:
            val = obj[key]
            if isinstance(val, (str, int, float)):
                return str(val)
    return None


NUM_RE = re.compile(r"[-+]?\d+(?:[\,\d]*\d)?(?:\.\d+)?")
FRAC_RE = re.compile(r"^\s*([-+]?\d+)\s*/\s*(\d+)\s*$")


def normalize_answer_str(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    text = str(ans).strip()
    # Remove enclosing formatting, currency, commas
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("\\$", "")
    text = text.strip()
    if not text:
        return None
    return text


def extract_number_like(text: str) -> Optional[str]:
    """Attempt to extract the final numeric-looking token from a string.
    Returns a string that looks like an int/float or a fraction.
    """
    if not text:
        return None
    # Direct fraction like "a/b"
    m = FRAC_RE.match(text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # Find all numeric tokens; prefer the last occurrence
    candidates = NUM_RE.findall(text)
    if candidates:
        return candidates[-1]
    return None


def to_number(value: str) -> Optional[Tuple[str, Any]]:
    """Convert a normalized string to a numeric representation.
    Returns (kind, obj) where kind in {"int", "frac", "decimal"} and obj is parsed value.
    """
    val = normalize_answer_str(value)
    if val is None:
        return None
    # Try fraction
    m = FRAC_RE.match(val)
    if m:
        try:
            num = int(m.group(1))
            den = int(m.group(2))
            if den != 0:
                return ("frac", Fraction(num, den))
        except Exception:
            pass
    # Try int
    try:
        iv = int(val)
        return ("int", iv)
    except Exception:
        pass
    # Try decimal
    try:
        getcontext().prec = 50
        dv = Decimal(val)
        return ("decimal", dv)
    except InvalidOperation:
        # Try extracting a number-like token from the string
        token = extract_number_like(val)
        if token is None:
            return None
        return to_number(token)


def numbers_equal(a: Tuple[str, Any], b: Tuple[str, Any]) -> bool:
    ak, av = a
    bk, bv = b
    # Convert both to Fraction where possible for exact comparison
    try:
        if ak == "int":
            af = Fraction(av, 1)
        elif ak == "frac":
            af = av
        elif ak == "decimal":
            # Convert decimal to Fraction exactly
            af = Fraction(av)
        else:
            return False
        if bk == "int":
            bf = Fraction(bv, 1)
        elif bk == "frac":
            bf = bv
        elif bk == "decimal":
            bf = Fraction(bv)
        else:
            return False
        return af == bf
    except Exception:
        # Fallback to float-ish comparison with tolerance
        try:
            af = float(av)
            bf = float(bv)
            return math.isclose(af, bf, rel_tol=1e-9, abs_tol=1e-9)
        except Exception:
            return False


def sympy_equal(a: str, b: str) -> Optional[bool]:
    try:
        import sympy as sp  # noqa: WPS433
    except Exception:
        return None
    try:
        # Parse as sympy expressions; if parsing fails, try numeric tokens
        ea = sp.sympify(normalize_answer_str(a))
        eb = sp.sympify(normalize_answer_str(b))
        return sp.simplify(ea - eb) == 0
    except Exception:
        # Fallback to numeric token extraction
        na = extract_number_like(a)
        nb = extract_number_like(b)
        if na is None or nb is None:
            return None
        try:
            ea = sp.nsimplify(na)
            eb = sp.nsimplify(nb)
            return sp.simplify(ea - eb) == 0
        except Exception:
            return None


def pick_predicted_answer(track_obj: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit `answer` field if present and non-empty
    ans = track_obj.get("answer")
    if isinstance(ans, (str, int, float)):
        ans_norm = normalize_answer_str(str(ans))
        if ans_norm:
            return ans_norm
    # Try to extract from completion/trajectory by searching for the last number
    for key in ("completion", "trajectory"):
        val = track_obj.get(key)
        if isinstance(val, str):
            token = extract_number_like(val)
            if token:
                return token
    # As a last resort, attempt steps (list of strings)
    steps = track_obj.get("steps")
    if isinstance(steps, list):
        for item in reversed(steps):
            if isinstance(item, str):
                token = extract_number_like(item)
                if token:
                    return token
    return None


def analyze(records: List[Dict[str, Any]],
            tracks: List[str],
            gold: Optional[Dict[str, str]] = None,
            enable_symbolic: bool = False) -> None:
    total = len(records)
    print(f"Total examples: {total}")

    # If tracks not specified, infer from first record
    if not tracks:
        inferred = set()
        for r in records:
            for k in ("base", "uncert"):
                if isinstance(r.get(k), dict):
                    inferred.add(k)
        tracks = sorted(inferred)
    if not tracks:
        print("No tracks found (e.g., 'base', 'uncert'). Nothing to analyze.")
        return

    for track in tracks:
        n_have_pred = 0
        em_correct = 0
        em_total = 0
        judge_accept = 0
        judge_total = 0
        judge_vs_gold_correct = 0
        judge_vs_gold_total = 0
        sym_correct = 0
        sym_total = 0

        for r in records:
            t = r.get(track)
            if not isinstance(t, dict):
                continue

            pred_str = pick_predicted_answer(t)
            has_pred = pred_str is not None
            if has_pred:
                n_have_pred += 1

            # LLM judge acceptance rate: fraction marked correct among judged
            judge_field = t.get("is_correct_verifier")
            if isinstance(judge_field, bool):
                judge_total += 1
                if judge_field:
                    judge_accept += 1

            if gold is None:
                # Use precomputed correctness if available
                if "correct" in t and isinstance(t["correct"], bool):
                    em_total += 1
                    if t["correct"]:
                        em_correct += 1
                # Judge accuracy vs precomputed correctness when both exist
                if isinstance(judge_field, bool) and "correct" in t and isinstance(t["correct"], bool):
                    judge_vs_gold_total += 1
                    if bool(judge_field) == bool(t["correct"]):
                        judge_vs_gold_correct += 1
                else:
                    # Without gold or precomputed correctness, we cannot score EM
                    continue
            else:
                # Need to align this record to its gold answer
                q = r.get("question")
                gold_ans = None
                if isinstance(q, str) and q in gold:
                    gold_ans = gold[q]
                # Fallback: try by index if provided and present in gold with same key
                if gold_ans is None and "index" in r:
                    idx_key = f"index::{r['index']}"
                    gold_ans = gold.get(idx_key)
                if gold_ans is None:
                    # Can't evaluate this record without a gold
                    continue

                if has_pred:
                    em_total += 1
                    pa = to_number(pred_str)
                    ga = to_number(gold_ans)
                    if pa is not None and ga is not None and numbers_equal(pa, ga):
                        em_correct += 1
                    # Judge accuracy vs gold when both exist
                    if isinstance(judge_field, bool):
                        judge_vs_gold_total += 1
                        gold_ok = (pa is not None and ga is not None and numbers_equal(pa, ga))
                        if bool(judge_field) == gold_ok:
                            judge_vs_gold_correct += 1
                    # Symbolic check
                    if enable_symbolic:
                        eq = sympy_equal(pred_str, gold_ans)
                        if eq is not None:
                            sym_total += 1
                            if eq:
                                sym_correct += 1

        cov = (n_have_pred / total * 100.0) if total else 0.0
        em_acc = (em_correct / em_total * 100.0) if em_total else 0.0
        print(f"\nTrack: {track}")
        print(f"  Coverage (has prediction): {n_have_pred}/{total} = {cov:.2f}%")
        if gold is None:
            print(f"  EM (from precomputed 'correct'): {em_correct}/{em_total} = {em_acc:.2f}%")
            if judge_total:
                judge_accept_rate = (judge_accept / judge_total * 100.0)
                judge_acc = (judge_vs_gold_correct / judge_vs_gold_total * 100.0) if judge_vs_gold_total else 0.0
                print(f"  LLM judge acceptance rate: {judge_accept}/{judge_total} = {judge_accept_rate:.2f}%")
                print(f"  LLM judge accuracy vs precomputed: {judge_vs_gold_correct}/{judge_vs_gold_total} = {judge_acc:.2f}%")
        else:
            print(f"  EM (vs gold): {em_correct}/{em_total} = {em_acc:.2f}%")
            if judge_total:
                judge_accept_rate = (judge_accept / judge_total * 100.0)
                judge_acc = (judge_vs_gold_correct / judge_vs_gold_total * 100.0) if judge_vs_gold_total else 0.0
                print(f"  LLM judge acceptance rate: {judge_accept}/{judge_total} = {judge_accept_rate:.2f}%")
                print(f"  LLM judge accuracy vs gold: {judge_vs_gold_correct}/{judge_vs_gold_total} = {judge_acc:.2f}%")
        if enable_symbolic and sym_total:
            sym_acc = sym_correct / sym_total * 100.0
            print(f"  Symbolic-EM (sympy): {sym_correct}/{sym_total} = {sym_acc:.2f}%")
        elif enable_symbolic:
            print("  Symbolic-EM (sympy): not enough comparable items")


def load_gold(path: str) -> Dict[str, str]:
    """Return a mapping from question text to gold answer.
    Also stores by synthetic key 'index::<index>' if present in the entry.
    """
    items = read_json_or_jsonl(path)
    out: Dict[str, str] = {}
    for obj in items:
        q = obj.get("question") if isinstance(obj, dict) else None
        a = infer_gold_answer(obj) if isinstance(obj, dict) else None
        if isinstance(q, str) and isinstance(a, str):
            out[q] = a
        if isinstance(obj, dict) and "index" in obj and isinstance(a, str):
            out[f"index::{obj['index']}"] = a
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze GSM8K results by EM with optional symbolic verification.")
    p.add_argument("--input", required=True, help="Path to results JSON/JSONL file (e.g., test_together_result.jsonl)")
    p.add_argument("--tracks", nargs="*", default=None, help="Tracks to analyze (e.g., base uncert). Defaults to autodetect.")
    p.add_argument("--gold", default=None, help="Optional path to gold JSON/JSONL file with 'question'+'answer'.")
    p.add_argument("--symbolic-verify", action="store_true", help="Enable sympy-based symbolic verification metric (requires --gold).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    records = read_json_or_jsonl(args.input)
    if not records:
        print("No records found in input.")
        return 0

    gold_map: Optional[Dict[str, str]] = None
    if args.gold:
        if not os.path.isfile(args.gold):
            print(f"Gold file not found: {args.gold}", file=sys.stderr)
            return 2
        gold_map = load_gold(args.gold)
        if not gold_map:
            print("Gold file parsed, but no (question -> answer) pairs found.")

    tracks = args.tracks or []
    analyze(records, tracks, gold_map, enable_symbolic=bool(args.symbolic_verify and gold_map))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


