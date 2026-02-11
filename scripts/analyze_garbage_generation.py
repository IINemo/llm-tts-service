#!/usr/bin/env python3
"""Analyze wandb runs or local output directories for degenerate (garbage) token generation.

Usage:
    # Analyze wandb runs
    python scripts/analyze_garbage_generation.py URL1 URL2 ...
    python scripts/analyze_garbage_generation.py --runs-file runs.txt

    # Analyze local output directories
    python scripts/analyze_garbage_generation.py --results-dir outputs/2026-02-03/run_name/
    python scripts/analyze_garbage_generation.py --results-dir outputs/2026-02-03/run1/ outputs/2026-02-03/run2/

    # With LLM diagnosis
    python scripts/analyze_garbage_generation.py --results-dir outputs/... --llm-diagnose --llm-api-key sk-...

    # JSON output
    python scripts/analyze_garbage_generation.py --results-dir outputs/... --json-output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as stats_module
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GARBAGE_TOKENS = [
    "\U0001f308",  # 🌈
    "\u8e69",  # 蹩
    "ebx",
    "Leone",
    "SEEK",
    "cdr",
    "legate",
    "witty",
    "m\u0119",  # mę
    "afi",
    "uellen",
    "ARRANT",
    "ponsored",
    "isor",
]

# Default threshold for hardcoded-token detector
GARBAGE_THRESHOLD = 3

SAMPLE_MARKER_RE = re.compile(r"Sample (\d+)/(\d+)")

# Reused from strategy_beam_search.py / strategy_online_best_of_n.py
_GARBAGE_UNICODE_PATTERN = re.compile(
    r"[\U0001F300-\U0001F9FF]"  # Emojis
    r"|[\u4E00-\u9FFF]"  # CJK Unified Ideographs
    r"|[\u3040-\u309F\u30A0-\u30FF]"  # Japanese Hiragana/Katakana
    r"|[\uFF01-\uFF60]"  # Fullwidth punctuation
    r"|[\u0100-\u024F]{2,}"  # Extended Latin with diacritics (2+ consecutive)
)

# Characters expected in English math reasoning text
_EXPECTED_CHARS_RE = re.compile(
    r"[a-zA-Z0-9\s"
    r"\.\,\;\:\!\?\'\"\-\+\=\*\/\\\(\)\[\]\{\}\<\>\|\_\^\~\@\#\$\%\&"
    r"\u00b0-\u00bf"  # Common Latin-1 supplement (degree, fractions, etc.)
    r"]"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Result from a single detection method for a single sample."""

    method: str  # e.g. "hardcoded_tokens", "unicode_anomaly"
    is_garbage: bool
    severity: float  # 0.0-1.0 normalized severity
    onset_position: float | None  # 0.0-1.0, where in text garbage first appears
    details: dict = field(default_factory=dict)


@dataclass
class SampleAnalysis:
    """Complete analysis of a single sample across all detection methods."""

    sample_index: int
    sample_text_preview: str  # Truncated for report
    total_text_length: int
    detection_results: list[DetectionResult]
    is_garbage: bool  # Any method flagged it
    methods_flagged: list[str]  # Which methods flagged it
    onset_position: float | None  # Earliest onset across methods
    # From results.json if available:
    is_correct: bool | None = None
    validity_scores: list[float] | None = None
    mean_validity: float | None = None
    output_tokens: int | None = None
    step_count: int | None = None
    garbage_step_index: int | None = None  # First step with garbage


@dataclass
class RunData:
    """Unified representation of a run's data."""

    run_id: str
    url: str | None
    run_name: str
    temperature: float | None
    top_p: float | None
    model_path: str
    strategy_type: str
    dataset: str
    num_paths: int | None
    log_content: str | None = None
    results: list[dict] | None = None  # Parsed results.json


@dataclass
class RunStatistics:
    """All statistics for a single run."""

    total_samples: int
    affected_samples: int
    affected_indices: list[int]

    # Per-token frequency
    token_frequency: dict[str, int] = field(default_factory=dict)

    # Per-method breakdown
    method_counts: dict[str, int] = field(default_factory=dict)

    # Garbage onset distribution
    onset_quartiles: dict[str, int] = field(default_factory=dict)

    # Step-level degeneration
    step_degeneration: dict[int, int] | None = None
    avg_garbage_step: float | None = None

    # Correctness correlation
    garbage_correct: int | None = None
    garbage_incorrect: int | None = None
    clean_correct: int | None = None
    clean_incorrect: int | None = None

    # Validity score comparison
    garbage_mean_validity: float | None = None
    clean_mean_validity: float | None = None

    # Token count comparison
    garbage_mean_tokens: float | None = None
    clean_mean_tokens: float | None = None

    # Repetition stats
    most_repeated_ngrams: list[tuple[str, int]] = field(default_factory=list)

    # Total garbage occurrences (compat with old report)
    total_garbage_occurrences: int = 0
    garbage_lines_count: int = 0
    total_log_lines: int = 0


# ---------------------------------------------------------------------------
# Detection Methods
# ---------------------------------------------------------------------------


def detect_hardcoded_tokens(
    text: str, tokens: list[str] | None = None, threshold: int = GARBAGE_THRESHOLD
) -> DetectionResult:
    """Detect known garbage tokens by exact string match."""
    if tokens is None:
        tokens = GARBAGE_TOKENS

    found = {}
    total_count = 0
    earliest_pos = None

    for tok in tokens:
        count = text.count(tok)
        if count > 0:
            found[tok] = count
            total_count += count
            pos = text.find(tok)
            rel_pos = pos / len(text) if len(text) > 0 else 0.0
            if earliest_pos is None or rel_pos < earliest_pos:
                earliest_pos = rel_pos

    distinct = len(found)
    is_garbage = distinct >= threshold
    severity = min(distinct / max(threshold * 2, 1), 1.0)

    return DetectionResult(
        method="hardcoded_tokens",
        is_garbage=is_garbage,
        severity=severity,
        onset_position=earliest_pos if is_garbage else None,
        details={
            "distinct_tokens": distinct,
            "total_occurrences": total_count,
            "token_counts": found,
        },
    )


def detect_unicode_anomaly(text: str, threshold: int = 2) -> DetectionResult:
    """Detect emoji, CJK, Japanese, fullwidth, extended-Latin anomalies."""
    matches = _GARBAGE_UNICODE_PATTERN.findall(text)
    if not matches:
        return DetectionResult(
            method="unicode_anomaly",
            is_garbage=False,
            severity=0.0,
            onset_position=None,
        )

    match_count = len(matches)
    is_garbage = match_count >= threshold

    # Find onset
    earliest_pos = None
    if is_garbage:
        m = _GARBAGE_UNICODE_PATTERN.search(text)
        if m:
            earliest_pos = m.start() / len(text) if len(text) > 0 else 0.0

    # Categorize matches
    categories = Counter()
    for m in matches:
        if re.match(r"[\U0001F300-\U0001F9FF]", m):
            categories["emoji"] += 1
        elif re.match(r"[\u4E00-\u9FFF]", m):
            categories["cjk"] += 1
        elif re.match(r"[\u3040-\u30FF]", m):
            categories["japanese"] += 1
        elif re.match(r"[\uFF01-\uFF60]", m):
            categories["fullwidth"] += 1
        else:
            categories["extended_latin"] += 1

    severity = min(match_count / 10.0, 1.0)

    return DetectionResult(
        method="unicode_anomaly",
        is_garbage=is_garbage,
        severity=severity,
        onset_position=earliest_pos,
        details={
            "match_count": match_count,
            "categories": dict(categories),
        },
    )


def detect_ngram_repetition(
    text: str,
    n_values: tuple[int, ...] = (3, 4),
    max_ratio: float = 0.3,
    min_count: int = 5,
) -> DetectionResult:
    """Detect repeated word n-grams indicating degeneration loops.

    Flags text where the ratio of unique n-grams to total n-grams is below
    ``max_ratio`` for any of the given n values, and the most-repeated n-gram
    appears at least ``min_count`` times.
    """
    words = text.split()
    if len(words) < 10:
        return DetectionResult(
            method="ngram_repetition",
            is_garbage=False,
            severity=0.0,
            onset_position=None,
        )

    is_garbage = False
    worst_ratio = 1.0
    top_ngrams: list[tuple[str, int]] = []
    earliest_pos = None

    for n in n_values:
        if len(words) < n:
            continue
        ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        total = len(ngrams)
        counts = Counter(ngrams)
        unique = len(counts)
        ratio = unique / total if total > 0 else 1.0

        most_common = counts.most_common(1)[0] if counts else ("", 0)

        if ratio < max_ratio and most_common[1] >= min_count:
            is_garbage = True
            if ratio < worst_ratio:
                worst_ratio = ratio

            # Find onset: first occurrence of most-repeated n-gram's second appearance
            ngram_text = most_common[0]
            first = text.find(ngram_text)
            second = (
                text.find(ngram_text, first + len(ngram_text)) if first >= 0 else -1
            )
            if second >= 0 and len(text) > 0:
                pos = second / len(text)
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos

        top_ngrams.extend(counts.most_common(3))

    # Deduplicate and sort top ngrams
    ngram_map: dict[str, int] = {}
    for ng, cnt in top_ngrams:
        ngram_map[ng] = max(ngram_map.get(ng, 0), cnt)
    top_ngrams = sorted(ngram_map.items(), key=lambda x: -x[1])[:5]

    severity = max(0.0, 1.0 - worst_ratio) if is_garbage else 0.0

    return DetectionResult(
        method="ngram_repetition",
        is_garbage=is_garbage,
        severity=severity,
        onset_position=earliest_pos if is_garbage else None,
        details={
            "worst_unique_ratio": round(worst_ratio, 3),
            "top_ngrams": top_ngrams,
        },
    )


def detect_line_repetition(
    text: str, min_lines: int = 4, max_unique_ratio: float = 0.3
) -> DetectionResult:
    """Detect text with many repeated lines (ported from vllm.py _detect_line_repetitions)."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < min_lines:
        return DetectionResult(
            method="line_repetition",
            is_garbage=False,
            severity=0.0,
            onset_position=None,
        )

    unique = len(set(lines))
    ratio = unique / len(lines)
    is_garbage = ratio <= max_unique_ratio

    # Find onset: first line that repeats
    earliest_pos = None
    if is_garbage:
        seen: dict[str, int] = {}
        char_offset = 0
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped in seen:
                earliest_pos = char_offset / len(text) if len(text) > 0 else 0.0
                break
            seen[stripped] = 1
            char_offset += len(line) + 1

    severity = max(0.0, 1.0 - ratio) if is_garbage else 0.0

    return DetectionResult(
        method="line_repetition",
        is_garbage=is_garbage,
        severity=severity,
        onset_position=earliest_pos,
        details={
            "total_lines": len(lines),
            "unique_lines": unique,
            "unique_ratio": round(ratio, 3),
        },
    )


def detect_char_class_shift(
    text: str, window_size: int = 200, stride: int = 100, threshold: float = 0.10
) -> DetectionResult:
    """Sliding window detection of unexpected character classes.

    For math reasoning in English, most characters should be ASCII letters, digits,
    math symbols, and LaTeX. A sudden spike in other characters suggests garbage.
    """
    if len(text) < window_size:
        # Check the whole text as one window
        windows = [(0, text)]
    else:
        windows = []
        for start in range(0, len(text) - window_size + 1, stride):
            windows.append((start, text[start : start + window_size]))

    worst_ratio = 0.0
    worst_pos = None

    for start, window in windows:
        expected_count = len(_EXPECTED_CHARS_RE.findall(window))
        unexpected_ratio = (
            1.0 - (expected_count / len(window)) if len(window) > 0 else 0.0
        )
        if unexpected_ratio > worst_ratio:
            worst_ratio = unexpected_ratio
            worst_pos = start / len(text) if len(text) > 0 else 0.0

    is_garbage = worst_ratio >= threshold
    severity = min(worst_ratio / 0.5, 1.0) if is_garbage else 0.0

    return DetectionResult(
        method="char_class_shift",
        is_garbage=is_garbage,
        severity=severity,
        onset_position=worst_pos if is_garbage else None,
        details={
            "worst_unexpected_ratio": round(worst_ratio, 3),
        },
    )


def analyze_sample(
    text: str,
    sample_index: int,
    garbage_tokens: list[str] | None = None,
    garbage_threshold: int = GARBAGE_THRESHOLD,
    disabled_detectors: set[str] | None = None,
    results_entry: dict | None = None,
) -> SampleAnalysis:
    """Run all enabled detectors on a single sample."""
    if garbage_tokens is None:
        garbage_tokens = GARBAGE_TOKENS
    if disabled_detectors is None:
        disabled_detectors = set()

    detections: list[DetectionResult] = []

    if "hardcoded" not in disabled_detectors:
        detections.append(
            detect_hardcoded_tokens(text, garbage_tokens, garbage_threshold)
        )
    if "unicode" not in disabled_detectors:
        detections.append(detect_unicode_anomaly(text))
    if "ngram" not in disabled_detectors:
        detections.append(detect_ngram_repetition(text))
    if "line_repetition" not in disabled_detectors:
        detections.append(detect_line_repetition(text))
    if "char_shift" not in disabled_detectors:
        detections.append(detect_char_class_shift(text))

    flagged = [d for d in detections if d.is_garbage]
    methods_flagged = [d.method for d in flagged]

    onset_positions = [
        d.onset_position for d in flagged if d.onset_position is not None
    ]
    earliest_onset = min(onset_positions) if onset_positions else None

    # Extract data from results.json entry if available
    is_correct = None
    validity_scores = None
    mean_validity = None
    output_tokens = None
    step_count = None
    garbage_step_idx = None

    if results_entry:
        is_correct = results_entry.get("is_correct")
        if is_correct is None:
            eval_data = results_entry.get("eval", {})
            for ev in eval_data.values():
                if isinstance(ev, dict) and "is_correct" in ev:
                    is_correct = ev["is_correct"]
                    break

        validity_scores = results_entry.get("validity_scores")
        if validity_scores:
            mean_validity = sum(validity_scores) / len(validity_scores)

        token_stats = results_entry.get("token_stats", {})
        output_tokens = token_stats.get("output_tokens")

        steps = results_entry.get("steps", [])
        step_count = len(steps)

        # Find first step with garbage
        if flagged and steps:
            for i, step in enumerate(steps):
                step_text = (
                    step.get("text", "") if isinstance(step, dict) else str(step)
                )
                for det_func in [detect_unicode_anomaly, detect_line_repetition]:
                    r = det_func(step_text)
                    if r.is_garbage:
                        garbage_step_idx = i
                        break
                if garbage_step_idx is not None:
                    break
                # Also check hardcoded tokens
                r = detect_hardcoded_tokens(
                    step_text, garbage_tokens, garbage_threshold
                )
                if r.is_garbage:
                    garbage_step_idx = i
                    break

    preview = text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")

    return SampleAnalysis(
        sample_index=sample_index,
        sample_text_preview=preview,
        total_text_length=len(text),
        detection_results=detections,
        is_garbage=bool(flagged),
        methods_flagged=methods_flagged,
        onset_position=earliest_onset,
        is_correct=is_correct,
        validity_scores=validity_scores,
        mean_validity=mean_validity,
        output_tokens=output_tokens,
        step_count=step_count,
        garbage_step_index=garbage_step_idx,
    )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def parse_wandb_url(url: str):
    """Extract (entity, project, run_id) from a wandb URL."""
    url = url.strip().rstrip("/")
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?\s]+)", url)
    if not m:
        raise ValueError(f"Cannot parse wandb URL: {url}")
    return m.group(1), m.group(2), m.group(3)


def fetch_run_data(api, entity, project, run_id) -> RunData:
    """Fetch config, log file, and results.json from a wandb run."""
    run = api.run(f"{entity}/{project}/{run_id}")
    cfg = run.config

    run_data = RunData(
        run_id=run_id,
        url=f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
        run_name=run.name,
        temperature=cfg.get("generation", {}).get("temperature"),
        top_p=cfg.get("generation", {}).get("top_p"),
        model_path=cfg.get("model", {}).get("model_path", ""),
        strategy_type=cfg.get("strategy", {}).get("type", ""),
        dataset=cfg.get("dataset", {}).get(
            "data_name", cfg.get("strategy", {}).get("data_name", "")
        ),
        num_paths=cfg.get("strategy", {}).get("num_paths"),
    )

    tmpdir = tempfile.mkdtemp()

    # Download log file
    for f in run.files():
        if f.name.endswith("run_tts_eval.log"):
            f.download(root=tmpdir, replace=True)
            path = os.path.join(tmpdir, f.name)
            with open(path) as fh:
                run_data.log_content = fh.read()
            break

    if run_data.log_content is None:
        print(f"  WARNING: No run_tts_eval.log found for {run_id}", file=sys.stderr)

    # Try to download results.json
    for f in run.files():
        if f.name.endswith("results.json"):
            f.download(root=tmpdir, replace=True)
            path = os.path.join(tmpdir, f.name)
            with open(path) as fh:
                run_data.results = json.load(fh)
            break

    return run_data


def load_from_local_dir(run_dir: Path) -> RunData:
    """Load run data from a local output directory.

    Reads .hydra/config.yaml, results.json, and run_tts_eval.log.
    """
    run_id = run_dir.name
    run_name = run_dir.name

    # Parse hydra config
    temperature = None
    top_p = None
    model_path = ""
    strategy_type = ""
    dataset = ""
    num_paths = None

    config_path = run_dir / ".hydra" / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            gen = cfg.get("generation", {})
            temperature = gen.get("temperature")
            top_p = gen.get("top_p")

            model = cfg.get("model", {})
            model_path = model.get("model_name") or model.get("model_path", "")

            strategy = cfg.get("strategy", {})
            strategy_type = strategy.get("type", "")
            num_paths = (
                strategy.get("num_paths")
                or strategy.get("candidates_per_beam")
                or strategy.get("candidates_per_step")
            )

            ds = cfg.get("dataset", {})
            dataset = ds.get("data_name", "")
            if not dataset:
                # Infer from dataset_path (e.g. "test-time-compute/test_MATH" -> "math")
                ds_path = ds.get("dataset_path", "")
                if ds_path:
                    name = ds_path.split("/")[-1].lower()
                    name = name.removeprefix("test_")
                    dataset = name
        except Exception as e:
            print(f"  WARNING: Failed to parse {config_path}: {e}", file=sys.stderr)

    # Load results.json
    results = None
    results_path = run_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    # Load log
    log_content = None
    log_path = run_dir / "run_tts_eval.log"
    if log_path.exists():
        with open(log_path) as f:
            log_content = f.read()

    if results is None and log_content is None:
        print(f"  WARNING: No results.json or log found in {run_dir}", file=sys.stderr)

    return RunData(
        run_id=run_id,
        url=None,
        run_name=run_name,
        temperature=temperature,
        top_p=top_p,
        model_path=model_path,
        strategy_type=strategy_type,
        dataset=dataset,
        num_paths=num_paths,
        log_content=log_content,
        results=results,
    )


# ---------------------------------------------------------------------------
# Sample Text Extraction
# ---------------------------------------------------------------------------


def extract_samples_from_results(results: list[dict]) -> list[tuple[int, str, dict]]:
    """Extract (index, text, entry) tuples from results.json data."""
    samples = []
    for entry in results:
        idx = entry.get("index", len(samples))
        text = entry.get("generated_trajectory", "")
        samples.append((idx, text, entry))
    return samples


def extract_samples_from_log(log_content: str) -> list[tuple[int, str, dict]]:
    """Extract (index, text, empty_dict) tuples from log content.

    Parses the log by finding Sample N/M markers and extracting text between them.
    """
    lines = log_content.split("\n")
    sample_boundaries = []
    for i, line in enumerate(lines):
        m = SAMPLE_MARKER_RE.search(line)
        if m:
            sample_num = int(m.group(1))
            sample_boundaries.append((i, sample_num))

    if not sample_boundaries:
        return []

    samples = []
    for idx, (start_line, sample_num) in enumerate(sample_boundaries):
        if idx + 1 < len(sample_boundaries):
            end_line = sample_boundaries[idx + 1][0]
        else:
            end_line = len(lines)
        text = "\n".join(lines[start_line:end_line])
        samples.append((sample_num, text, {}))

    return samples


# ---------------------------------------------------------------------------
# Statistics Computation
# ---------------------------------------------------------------------------


def compute_run_statistics(
    analyses: list[SampleAnalysis],
) -> RunStatistics:
    """Compute all statistics from per-sample analyses."""
    total = len(analyses)
    garbage_analyses = [a for a in analyses if a.is_garbage]
    clean_analyses = [a for a in analyses if not a.is_garbage]
    affected = len(garbage_analyses)
    affected_indices = [a.sample_index for a in garbage_analyses]

    # Per-token frequency (from hardcoded_tokens detector)
    token_freq: Counter = Counter()
    total_garbage_occ = 0
    for a in analyses:
        for d in a.detection_results:
            if d.method == "hardcoded_tokens" and d.details.get("token_counts"):
                for tok, cnt in d.details["token_counts"].items():
                    token_freq[tok] += cnt
                    total_garbage_occ += cnt

    # Per-method breakdown
    method_counts: Counter = Counter()
    for a in garbage_analyses:
        for m in a.methods_flagged:
            method_counts[m] += 1

    # Onset quartiles
    onset_q: dict[str, int] = {"first_25%": 0, "middle_50%": 0, "last_25%": 0}
    for a in garbage_analyses:
        if a.onset_position is not None:
            if a.onset_position < 0.25:
                onset_q["first_25%"] += 1
            elif a.onset_position < 0.75:
                onset_q["middle_50%"] += 1
            else:
                onset_q["last_25%"] += 1

    # Step-level degeneration
    step_degen: dict[int, int] = defaultdict(int)
    garbage_steps = []
    for a in garbage_analyses:
        if a.garbage_step_index is not None:
            step_degen[a.garbage_step_index] += 1
            garbage_steps.append(a.garbage_step_index)
    avg_garbage_step = stats_module.mean(garbage_steps) if garbage_steps else None

    # Correctness correlation
    has_correctness = any(a.is_correct is not None for a in analyses)
    gc = gi = cc = ci = None
    if has_correctness:
        gc = sum(1 for a in garbage_analyses if a.is_correct is True)
        gi = sum(1 for a in garbage_analyses if a.is_correct is False)
        cc = sum(1 for a in clean_analyses if a.is_correct is True)
        ci = sum(1 for a in clean_analyses if a.is_correct is False)

    # Validity score comparison
    garbage_validities = [
        a.mean_validity for a in garbage_analyses if a.mean_validity is not None
    ]
    clean_validities = [
        a.mean_validity for a in clean_analyses if a.mean_validity is not None
    ]
    g_mean_v = stats_module.mean(garbage_validities) if garbage_validities else None
    c_mean_v = stats_module.mean(clean_validities) if clean_validities else None

    # Token count comparison
    garbage_tokens_list = [
        a.output_tokens
        for a in garbage_analyses
        if a.output_tokens is not None and a.output_tokens > 0
    ]
    clean_tokens_list = [
        a.output_tokens
        for a in clean_analyses
        if a.output_tokens is not None and a.output_tokens > 0
    ]
    g_mean_t = stats_module.mean(garbage_tokens_list) if garbage_tokens_list else None
    c_mean_t = stats_module.mean(clean_tokens_list) if clean_tokens_list else None

    # Top repeated n-grams across all garbage samples
    all_ngrams: Counter = Counter()
    for a in garbage_analyses:
        for d in a.detection_results:
            if d.method == "ngram_repetition" and d.details.get("top_ngrams"):
                for ng, cnt in d.details["top_ngrams"]:
                    all_ngrams[ng] = max(all_ngrams[ng], cnt)
    top_ngrams = all_ngrams.most_common(10)

    return RunStatistics(
        total_samples=total,
        affected_samples=affected,
        affected_indices=affected_indices,
        token_frequency=dict(token_freq.most_common()),
        method_counts=dict(method_counts),
        onset_quartiles=onset_q,
        step_degeneration=dict(step_degen) if step_degen else None,
        avg_garbage_step=avg_garbage_step,
        garbage_correct=gc,
        garbage_incorrect=gi,
        clean_correct=cc,
        clean_incorrect=ci,
        garbage_mean_validity=g_mean_v,
        clean_mean_validity=c_mean_v,
        garbage_mean_tokens=g_mean_t,
        clean_mean_tokens=c_mean_t,
        most_repeated_ngrams=top_ngrams,
        total_garbage_occurrences=total_garbage_occ,
    )


def compute_correlation_tables(
    all_results: list[tuple[RunData, RunStatistics]],
) -> dict[str, list[tuple[str, int, int, float]]]:
    """Cross-run correlation: garbage rate grouped by temperature, model, strategy.

    Returns {group_name: [(value, affected, total, pct), ...]}.
    """
    tables: dict[str, list[tuple[str, int, int, float]]] = {}

    groupings = {
        "By Temperature": lambda rd: str(rd.temperature),
        "By Model": lambda rd: model_short_name(rd.model_path),
        "By Strategy": lambda rd: rd.strategy_type,
        "By top_p": lambda rd: str(rd.top_p),
    }

    for group_name, key_fn in groupings.items():
        buckets: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for rd, rs in all_results:
            k = key_fn(rd)
            aff, tot = buckets[k]
            buckets[k] = (aff + rs.affected_samples, tot + rs.total_samples)

        rows = []
        for k, (aff, tot) in sorted(buckets.items()):
            pct = (aff / tot * 100) if tot > 0 else 0.0
            rows.append((k, aff, tot, pct))
        tables[group_name] = rows

    return tables


# ---------------------------------------------------------------------------
# LLM Diagnosis
# ---------------------------------------------------------------------------

DIAGNOSIS_PROMPT = """You are an expert in debugging LLM text generation failures.

I will show you {n} samples from a text-to-solution LLM system that exhibit degenerate output \
(garbage generation). These samples come from a **{strategy}** strategy using **{model}** at \
temperature={temperature}, top_p={top_p}.

Detection methods that flagged these samples:
{methods_summary}

Here are the garbage samples (showing first 500 chars of each):

{samples_text}

Please analyze these samples and provide:

1. **Degeneration Type**: What type(s) of degeneration are these? (e.g., token repetition loop, \
unicode/CJK character injection, random token sampling, attention collapse, etc.)

2. **Likely Root Causes**: What are the most probable causes? Consider:
   - EOS token configuration issues
   - Temperature/sampling parameter issues
   - Context window overflow
   - Tokenizer vocabulary issues (merged tokens from multilingual training)
   - Missing or incorrect stop sequences
   - Model-specific known issues

3. **Suggested Fixes**: Concrete, actionable fixes ranked by likelihood of success. Include:
   - Specific parameter changes (temperature, top_p, presence_penalty, etc.)
   - Stop token / EOS configuration changes
   - Post-processing strategies (output filtering, truncation)

Format your response as markdown with ## headers for each section."""


def select_representative_samples(
    analyses: list[SampleAnalysis], max_samples: int = 5
) -> list[SampleAnalysis]:
    """Select diverse garbage samples covering different detection methods."""
    garbage = [a for a in analyses if a.is_garbage]
    if not garbage:
        return []

    # Pick one sample per unique set of methods, then fill with highest severity
    selected: list[SampleAnalysis] = []
    seen_methods: set[frozenset[str]] = set()

    for a in garbage:
        key = frozenset(a.methods_flagged)
        if key not in seen_methods and len(selected) < max_samples:
            selected.append(a)
            seen_methods.add(key)

    # Fill remaining slots with highest severity
    if len(selected) < max_samples:
        remaining = [a for a in garbage if a not in selected]
        remaining.sort(
            key=lambda a: max((d.severity for d in a.detection_results), default=0),
            reverse=True,
        )
        for a in remaining:
            if len(selected) >= max_samples:
                break
            selected.append(a)

    return selected


def diagnose_with_llm(
    representative: list[SampleAnalysis],
    run_data: RunData,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> str:
    """Send representative garbage samples to an LLM for diagnosis."""
    try:
        import openai
    except ImportError:
        return "*LLM diagnosis skipped: `openai` package not installed.*"

    # Build samples text
    samples_parts = []
    all_methods: set[str] = set()
    for i, a in enumerate(representative, 1):
        samples_parts.append(
            f"### Sample {i} (index={a.sample_index}, methods={a.methods_flagged})\n"
            f"```\n{a.sample_text_preview[:500]}\n```"
        )
        all_methods.update(a.methods_flagged)

    methods_summary = ", ".join(sorted(all_methods))

    prompt = DIAGNOSIS_PROMPT.format(
        n=len(representative),
        strategy=run_data.strategy_type,
        model=model_short_name(run_data.model_path),
        temperature=run_data.temperature,
        top_p=run_data.top_p,
        methods_summary=methods_summary,
        samples_text="\n\n".join(samples_parts),
    )

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"*LLM diagnosis failed: {e}*"


# ---------------------------------------------------------------------------
# Rule-Based Recommendations
# ---------------------------------------------------------------------------


def generate_recommendations(
    all_run_stats: list[tuple[RunData, RunStatistics, list[SampleAnalysis]]],
) -> list[str]:
    """Generate rule-based recommendations from analysis results."""
    recs: list[str] = []

    # Aggregate across runs
    total_onset = {"first_25%": 0, "middle_50%": 0, "last_25%": 0}
    total_method: Counter = Counter()
    total_affected = 0
    total_samples = 0
    temps_with_high_garbage = []

    for rd, rs, _ in all_run_stats:
        total_affected += rs.affected_samples
        total_samples += rs.total_samples
        for k, v in rs.onset_quartiles.items():
            total_onset[k] += v
        for m, c in rs.method_counts.items():
            total_method[m] += c

        if rs.total_samples > 0:
            rate = rs.affected_samples / rs.total_samples
            if rate > 0.20 and rd.temperature is not None and rd.temperature > 0.8:
                temps_with_high_garbage.append(rd.temperature)

    onset_total = sum(total_onset.values())

    if onset_total > 0:
        late_frac = total_onset["last_25%"] / onset_total
        early_frac = total_onset["first_25%"] / onset_total

        if late_frac > 0.5:
            recs.append(
                "**EOS/stop token issue likely**: >50% of garbage starts in the last 25% of text. "
                "Check that all required EOS token IDs are configured (e.g., `<|im_end|>`, "
                "`<|endoftext|>`). Verify the stop token list in your generation config."
            )
        if early_frac > 0.5:
            recs.append(
                "**Early degeneration detected**: >50% of garbage starts in the first 25% of text. "
                "Check prompt formatting and tokenizer compatibility. The model may not understand "
                "the input format."
            )

    if total_method.get("ngram_repetition", 0) > total_affected * 0.3:
        recs.append(
            "**Repetition loops are common**: Consider adding `presence_penalty` (e.g., 1.0-1.5) "
            "or `frequency_penalty` to discourage repeated token sequences."
        )

    if total_method.get("unicode_anomaly", 0) > total_affected * 0.3:
        recs.append(
            "**Unicode anomalies frequent**: The model outputs CJK/emoji/unusual characters. "
            "This is common with multilingual models (Qwen, etc.) when sampling is too random. "
            "Consider adding a post-processing filter for non-Latin characters, or lowering temperature."
        )

    if temps_with_high_garbage:
        recs.append(
            f"**High temperature + high garbage rate**: Runs with temperature "
            f">{min(temps_with_high_garbage):.1f} have >20% garbage rate. "
            f"Consider reducing temperature to 0.6-0.7."
        )

    # Check if garbage samples are always incorrect
    all_gc = sum(rs.garbage_correct or 0 for _, rs, _ in all_run_stats)
    all_gi = sum(rs.garbage_incorrect or 0 for _, rs, _ in all_run_stats)
    if all_gi > 0 and all_gc == 0:
        recs.append(
            "**All garbage samples are incorrect**: Garbage generation always leads to wrong answers. "
            "Early detection and re-generation could improve accuracy."
        )

    if not recs:
        recs.append(
            "No specific recommendations. Garbage rate appears low across all runs."
        )

    return recs


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def model_short_name(model_path: str) -> str:
    """Shorten model path for display."""
    if "/" in model_path:
        return model_path.split("/")[-1]
    return model_path


def _bar(value: float, max_value: float, width: int = 20) -> str:
    """Simple text bar chart."""
    if max_value <= 0:
        return ""
    filled = int(value / max_value * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def generate_report(
    all_run_results: list[tuple[RunData, RunStatistics, list[SampleAnalysis]]],
    correlation_tables: dict[str, list[tuple[str, int, int, float]]],
    diagnosis_text: str | None = None,
    recommendations: list[str] | None = None,
) -> str:
    """Generate comprehensive markdown report."""
    lines: list[str] = []
    lines.append("# Garbage Generation Analysis Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"\nGarbage token set ({len(GARBAGE_TOKENS)} tokens): "
        f"`{'`, `'.join(GARBAGE_TOKENS)}`"
    )
    lines.append(
        "\nDetection methods: hardcoded_tokens, unicode_anomaly, ngram_repetition, "
        "line_repetition, char_class_shift"
    )

    # ------------------------------------------------------------------
    # 1. Summary Table
    # ------------------------------------------------------------------
    lines.append("\n## Summary\n")
    lines.append(
        "| Dataset | Model | Strategy | Temp | top_p | Paths | Total | Affected | % |"
    )
    lines.append(
        "|---------|-------|----------|------|-------|-------|-------|----------|---|"
    )

    for rd, rs, _ in sorted(
        all_run_results, key=lambda x: (x[0].dataset, x[0].temperature or 0)
    ):
        pct = (
            (rs.affected_samples / rs.total_samples * 100)
            if rs.total_samples > 0
            else 0
        )
        lines.append(
            f"| {rd.dataset} "
            f"| {model_short_name(rd.model_path)} "
            f"| {rd.strategy_type} "
            f"| {rd.temperature} "
            f"| {rd.top_p} "
            f"| {rd.num_paths} "
            f"| {rs.total_samples} "
            f"| {rs.affected_samples} "
            f"| {pct:.1f}% |"
        )

    # ------------------------------------------------------------------
    # 2. Detection Methods Overview
    # ------------------------------------------------------------------
    lines.append("\n## Detection Methods Overview\n")
    all_methods: Counter = Counter()
    for _, rs, _ in all_run_results:
        for m, c in rs.method_counts.items():
            all_methods[m] += c

    if all_methods:
        total_flagged = sum(rs.affected_samples for _, rs, _ in all_run_results)
        lines.append("| Method | Samples Flagged | % of Garbage |")
        lines.append("|--------|-----------------|--------------|")
        for method, count in all_methods.most_common():
            pct = (count / total_flagged * 100) if total_flagged > 0 else 0
            lines.append(f"| {method} | {count} | {pct:.1f}% |")
    else:
        lines.append("No garbage detected by any method.")

    # ------------------------------------------------------------------
    # 3. Per-Token Frequency Table
    # ------------------------------------------------------------------
    merged_freq: Counter = Counter()
    for _, rs, _ in all_run_results:
        for tok, cnt in rs.token_frequency.items():
            merged_freq[tok] += cnt

    if merged_freq:
        lines.append("\n## Garbage Token Frequency\n")
        lines.append("| Token | Total Occurrences |")
        lines.append("|-------|-------------------|")
        for tok, cnt in merged_freq.most_common():
            display = repr(tok) if not tok.isprintable() or len(tok) == 1 else tok
            lines.append(f"| {display} | {cnt} |")

    # ------------------------------------------------------------------
    # 4. Garbage Onset Analysis
    # ------------------------------------------------------------------
    merged_onset: dict[str, int] = {"first_25%": 0, "middle_50%": 0, "last_25%": 0}
    for _, rs, _ in all_run_results:
        for k, v in rs.onset_quartiles.items():
            merged_onset[k] += v

    onset_total = sum(merged_onset.values())
    if onset_total > 0:
        lines.append("\n## Garbage Onset Position\n")
        lines.append("Where in the generated text does garbage first appear?\n")
        interpretations = {
            "first_25%": "Early (prompt/tokenizer issue)",
            "middle_50%": "Mid-generation (context overflow, attention drift)",
            "last_25%": "Late (EOS/stop token issue)",
        }
        lines.append("| Position | Count | % | Interpretation |")
        lines.append("|----------|-------|---|----------------|")
        for pos, label in interpretations.items():
            count = merged_onset[pos]
            pct = (count / onset_total * 100) if onset_total > 0 else 0
            bar = _bar(count, onset_total, 15)
            lines.append(f"| {pos} | {count} | {pct:.0f}% {bar} | {label} |")

    # ------------------------------------------------------------------
    # 5. Step-Level Analysis
    # ------------------------------------------------------------------
    has_step_data = any(rs.step_degeneration for _, rs, _ in all_run_results)
    if has_step_data:
        lines.append("\n## Step-Level Degeneration\n")
        lines.append("Which reasoning step tends to degenerate first?\n")

        merged_steps: Counter = Counter()
        all_avg_steps = []
        for _, rs, _ in all_run_results:
            if rs.step_degeneration:
                for step, cnt in rs.step_degeneration.items():
                    merged_steps[step] += cnt
            if rs.avg_garbage_step is not None:
                all_avg_steps.append(rs.avg_garbage_step)

        if merged_steps:
            lines.append("| Step # | Garbage Count |")
            lines.append("|--------|---------------|")
            for step in sorted(merged_steps.keys()):
                lines.append(f"| {step} | {merged_steps[step]} |")

        if all_avg_steps:
            lines.append(
                f"\nMean garbage step index: **{stats_module.mean(all_avg_steps):.1f}**"
            )

    # ------------------------------------------------------------------
    # 6. Correctness & Validity Correlation
    # ------------------------------------------------------------------
    has_correctness = any(
        rs.garbage_correct is not None for _, rs, _ in all_run_results
    )
    if has_correctness:
        lines.append("\n## Correctness Correlation\n")
        gc = sum(rs.garbage_correct or 0 for _, rs, _ in all_run_results)
        gi = sum(rs.garbage_incorrect or 0 for _, rs, _ in all_run_results)
        cc = sum(rs.clean_correct or 0 for _, rs, _ in all_run_results)
        ci = sum(rs.clean_incorrect or 0 for _, rs, _ in all_run_results)

        lines.append("|  | Correct | Incorrect | Total | Accuracy |")
        lines.append("|--|---------|-----------|-------|----------|")
        g_total = gc + gi
        c_total = cc + ci
        g_acc = (gc / g_total * 100) if g_total > 0 else 0
        c_acc = (cc / c_total * 100) if c_total > 0 else 0
        lines.append(f"| Garbage | {gc} | {gi} | {g_total} | {g_acc:.1f}% |")
        lines.append(f"| Clean | {cc} | {ci} | {c_total} | {c_acc:.1f}% |")

    # Validity comparison
    has_validity = any(
        rs.garbage_mean_validity is not None or rs.clean_mean_validity is not None
        for _, rs, _ in all_run_results
    )
    if has_validity:
        lines.append("\n## Validity Score Comparison\n")
        g_vals = [
            rs.garbage_mean_validity
            for _, rs, _ in all_run_results
            if rs.garbage_mean_validity is not None
        ]
        c_vals = [
            rs.clean_mean_validity
            for _, rs, _ in all_run_results
            if rs.clean_mean_validity is not None
        ]
        lines.append("| Group | Mean Validity Score |")
        lines.append("|-------|---------------------|")
        if g_vals:
            lines.append(f"| Garbage samples | {stats_module.mean(g_vals):.3f} |")
        if c_vals:
            lines.append(f"| Clean samples | {stats_module.mean(c_vals):.3f} |")

    # Token count comparison
    has_tokens = any(
        rs.garbage_mean_tokens is not None or rs.clean_mean_tokens is not None
        for _, rs, _ in all_run_results
    )
    if has_tokens:
        lines.append("\n## Token Count Comparison\n")
        lines.append("| Group | Mean Output Tokens |")
        lines.append("|-------|---------------------|")
        g_toks = [
            rs.garbage_mean_tokens
            for _, rs, _ in all_run_results
            if rs.garbage_mean_tokens is not None
        ]
        c_toks = [
            rs.clean_mean_tokens
            for _, rs, _ in all_run_results
            if rs.clean_mean_tokens is not None
        ]
        if g_toks:
            lines.append(f"| Garbage samples | {stats_module.mean(g_toks):.0f} |")
        if c_toks:
            lines.append(f"| Clean samples | {stats_module.mean(c_toks):.0f} |")

    # ------------------------------------------------------------------
    # 7. N-gram Repetition Patterns
    # ------------------------------------------------------------------
    all_ngrams: Counter = Counter()
    for _, rs, _ in all_run_results:
        for ng, cnt in rs.most_repeated_ngrams:
            all_ngrams[ng] = max(all_ngrams[ng], cnt)

    if all_ngrams:
        lines.append("\n## Most Repeated N-grams (in garbage samples)\n")
        lines.append("| N-gram | Max Repetitions |")
        lines.append("|--------|-----------------|")
        for ng, cnt in all_ngrams.most_common(10):
            lines.append(f"| {ng} | {cnt} |")

    # ------------------------------------------------------------------
    # 8. Cross-Run Correlation
    # ------------------------------------------------------------------
    if len(all_run_results) > 1 and correlation_tables:
        lines.append("\n## Cross-Run Correlation\n")
        for group_name, rows in correlation_tables.items():
            if not rows or all(r[2] == 0 for r in rows):
                continue
            lines.append(f"\n### {group_name}\n")
            lines.append("| Value | Affected | Total | Rate | |")
            lines.append("|-------|----------|-------|------|-|")
            max_pct = max((r[3] for r in rows), default=0)
            for val, aff, tot, pct in rows:
                bar = _bar(pct, max(max_pct, 1), 15)
                lines.append(f"| {val} | {aff} | {tot} | {pct:.1f}% | {bar} |")

    # ------------------------------------------------------------------
    # 9. Example Garbage Snippets
    # ------------------------------------------------------------------
    all_garbage: list[SampleAnalysis] = []
    for _, _, analyses in all_run_results:
        all_garbage.extend(a for a in analyses if a.is_garbage)

    if all_garbage:
        lines.append("\n## Example Garbage Snippets\n")
        shown = all_garbage[:5]
        for a in shown:
            methods_str = ", ".join(a.methods_flagged)
            onset_str = (
                f"{a.onset_position:.2f}" if a.onset_position is not None else "N/A"
            )
            lines.append(
                f"### Sample {a.sample_index} "
                f"(methods: {methods_str}, onset: {onset_str})\n"
            )
            lines.append(f"```\n{a.sample_text_preview[:300]}\n```\n")

    # ------------------------------------------------------------------
    # 10. LLM Diagnosis
    # ------------------------------------------------------------------
    if diagnosis_text:
        lines.append("\n## LLM Diagnosis\n")
        lines.append(diagnosis_text)
        lines.append("")

    # ------------------------------------------------------------------
    # 11. Recommendations
    # ------------------------------------------------------------------
    if recommendations:
        lines.append("\n## Recommendations\n")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # ------------------------------------------------------------------
    # 12. Per-Dataset / Per-Run Detail
    # ------------------------------------------------------------------
    by_dataset: dict[str, list[tuple[RunData, RunStatistics, list[SampleAnalysis]]]] = (
        defaultdict(list)
    )
    for entry in all_run_results:
        by_dataset[entry[0].dataset].append(entry)

    for dataset, entries in sorted(by_dataset.items()):
        lines.append(f"\n## Dataset: {dataset}\n")
        for rd, rs, _ in sorted(entries, key=lambda x: x[0].temperature or 0):
            pct = (
                (rs.affected_samples / rs.total_samples * 100)
                if rs.total_samples > 0
                else 0
            )
            lines.append(
                f"### {model_short_name(rd.model_path)} | "
                f"temp={rd.temperature} | top_p={rd.top_p}"
            )
            if rd.url:
                lines.append(f"\n- **Run**: [{rd.run_id}]({rd.url})")
            else:
                lines.append(f"\n- **Run**: {rd.run_id}")
            lines.append(f"- **Strategy**: {rd.strategy_type} ({rd.num_paths} paths)")
            lines.append(
                f"- **Samples**: {rs.total_samples} total, "
                f"**{rs.affected_samples}** affected ({pct:.1f}%)"
            )
            lines.append(
                f"- **Total garbage token occurrences**: {rs.total_garbage_occurrences}"
            )
            if rs.method_counts:
                mc_str = ", ".join(
                    f"{m}: {c}" for m, c in sorted(rs.method_counts.items())
                )
                lines.append(f"- **Methods triggered**: {mc_str}")
            if rs.affected_indices:
                indices_str = ", ".join(str(i) for i in rs.affected_indices[:50])
                if len(rs.affected_indices) > 50:
                    indices_str += f" ... ({len(rs.affected_indices)} total)"
                lines.append(f"- **Affected sample indices**: {indices_str}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------


def export_json(
    all_run_results: list[tuple[RunData, RunStatistics, list[SampleAnalysis]]],
    correlation_tables: dict,
    diagnosis_text: str | None,
    recommendations: list[str] | None,
    output_path: Path,
):
    """Export structured analysis results to JSON."""
    data = {
        "generated": datetime.now().isoformat(),
        "runs": [],
        "correlation_tables": correlation_tables,
        "diagnosis": diagnosis_text,
        "recommendations": recommendations,
    }

    for rd, rs, analyses in all_run_results:
        run_entry = {
            "run_data": asdict(rd),
            "statistics": asdict(rs),
            "garbage_samples": [
                {
                    "sample_index": a.sample_index,
                    "methods_flagged": a.methods_flagged,
                    "onset_position": a.onset_position,
                    "is_correct": a.is_correct,
                    "mean_validity": a.mean_validity,
                    "output_tokens": a.output_tokens,
                    "step_count": a.step_count,
                    "garbage_step_index": a.garbage_step_index,
                    "text_preview": a.sample_text_preview[:500],
                }
                for a in analyses
                if a.is_garbage
            ],
        }
        # Remove large fields from JSON
        run_entry["run_data"].pop("log_content", None)
        run_entry["run_data"].pop("results", None)
        data["runs"].append(run_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze wandb runs or local directories for garbage token generation"
    )
    # Input sources
    parser.add_argument("urls", nargs="*", help="wandb run URLs")
    parser.add_argument(
        "--runs-file", type=str, help="File with one wandb URL per line"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        nargs="+",
        help="Local output directories to analyze",
    )

    # Detection tuning
    parser.add_argument(
        "--garbage-threshold",
        type=int,
        default=GARBAGE_THRESHOLD,
        help=f"Min distinct hardcoded tokens for 'affected' (default: {GARBAGE_THRESHOLD})",
    )
    parser.add_argument(
        "--disable-detectors",
        type=str,
        nargs="*",
        default=[],
        choices=["hardcoded", "unicode", "ngram", "line_repetition", "char_shift"],
        help="Disable specific detection methods",
    )

    # LLM diagnosis
    parser.add_argument(
        "--llm-diagnose", action="store_true", help="Enable LLM-based diagnosis"
    )
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument(
        "--llm-base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key (or set OPENAI_API_KEY / OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--llm-max-samples",
        type=int,
        default=5,
        help="Max garbage samples to send for diagnosis",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Output markdown file path (default: reports/garbage_analysis_<timestamp>.md)",
    )
    parser.add_argument(
        "--json-output", type=str, help="Also save structured JSON results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include more garbage snippets in report",
    )

    args = parser.parse_args()

    # Collect URLs
    urls = list(args.urls) if args.urls else []
    if args.runs_file:
        with open(args.runs_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

    local_dirs = [Path(d) for d in (args.results_dir or [])]

    if not urls and not local_dirs:
        parser.error(
            "No input provided. Pass wandb URLs, --runs-file, or --results-dir."
        )

    disabled = set(args.disable_detectors) if args.disable_detectors else set()

    # ---- Load data ----
    all_run_data: list[RunData] = []

    if urls:
        import wandb

        api = wandb.Api()
        for url in urls:
            try:
                entity, project, run_id = parse_wandb_url(url)
            except ValueError as e:
                print(f"Skipping invalid URL: {e}", file=sys.stderr)
                continue
            print(f"Fetching {entity}/{project}/{run_id}...", file=sys.stderr)
            rd = fetch_run_data(api, entity, project, run_id)
            all_run_data.append(rd)

    for d in local_dirs:
        if not d.exists():
            print(f"Skipping non-existent directory: {d}", file=sys.stderr)
            continue
        print(f"Loading {d}...", file=sys.stderr)
        rd = load_from_local_dir(d)
        all_run_data.append(rd)

    if not all_run_data:
        print("No valid runs to analyze.", file=sys.stderr)
        sys.exit(1)

    # ---- Analyze ----
    all_run_results: list[tuple[RunData, RunStatistics, list[SampleAnalysis]]] = []

    for rd in all_run_data:
        print(f"  Analyzing {rd.run_id}...", file=sys.stderr)

        # Extract samples: prefer results.json, fall back to log
        if rd.results:
            samples = extract_samples_from_results(rd.results)
        elif rd.log_content:
            samples = extract_samples_from_log(rd.log_content)
        else:
            print(f"  WARNING: No data to analyze for {rd.run_id}", file=sys.stderr)
            continue

        analyses = []
        for idx, text, entry in samples:
            a = analyze_sample(
                text=text,
                sample_index=idx,
                garbage_threshold=args.garbage_threshold,
                disabled_detectors=disabled,
                results_entry=entry if entry else None,
            )
            analyses.append(a)

        run_stats = compute_run_statistics(analyses)
        all_run_results.append((rd, run_stats, analyses))

        pct = (
            (run_stats.affected_samples / run_stats.total_samples * 100)
            if run_stats.total_samples > 0
            else 0
        )
        print(
            f"  -> {run_stats.affected_samples}/{run_stats.total_samples} affected ({pct:.1f}%)",
            file=sys.stderr,
        )

    if not all_run_results:
        print("No valid runs analyzed.", file=sys.stderr)
        sys.exit(1)

    # ---- Correlation tables ----
    corr_tables = compute_correlation_tables(
        [(rd, rs) for rd, rs, _ in all_run_results]
    )

    # ---- LLM diagnosis ----
    diagnosis_text = None
    if args.llm_diagnose:
        api_key = args.llm_api_key or os.environ.get(
            "OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY", "")
        )
        if not api_key:
            print(
                "WARNING: --llm-diagnose requires an API key. "
                "Set --llm-api-key or OPENAI_API_KEY env var.",
                file=sys.stderr,
            )
        else:
            # Collect garbage samples across all runs
            all_garbage = []
            first_rd = all_run_results[0][0]
            for _, _, analyses in all_run_results:
                all_garbage.extend(a for a in analyses if a.is_garbage)

            if all_garbage:
                representative = select_representative_samples(
                    all_garbage, args.llm_max_samples
                )
                print(
                    f"  Sending {len(representative)} samples to LLM for diagnosis...",
                    file=sys.stderr,
                )
                diagnosis_text = diagnose_with_llm(
                    representative,
                    first_rd,
                    api_key=api_key,
                    base_url=args.llm_base_url,
                    model=args.llm_model,
                )
            else:
                print("  No garbage samples to diagnose.", file=sys.stderr)

    # ---- Recommendations ----
    recommendations = generate_recommendations(all_run_results)

    # ---- Generate report ----
    report = generate_report(
        all_run_results, corr_tables, diagnosis_text, recommendations
    )

    # Save markdown report
    if args.output:
        output_path = Path(args.output)
    else:
        reports_dir = Path(__file__).resolve().parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = reports_dir / f"garbage_analysis_{timestamp}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}", file=sys.stderr)

    # Save JSON
    if args.json_output:
        json_path = Path(args.json_output)
        export_json(
            all_run_results, corr_tables, diagnosis_text, recommendations, json_path
        )
        print(f"JSON saved to: {json_path}", file=sys.stderr)

    # Print to stdout
    print(report)


if __name__ == "__main__":
    main()
