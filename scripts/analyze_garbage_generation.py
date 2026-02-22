#!/usr/bin/env python3
"""Analyze wandb runs for degenerate (garbage) token generation.

Usage:
    python scripts/analyze_garbage_generation.py URL1 URL2 ...
    python scripts/analyze_garbage_generation.py --runs-file runs.txt
"""

import argparse
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import wandb

GARBAGE_TOKENS = [
    "🌈",
    "蹩",
    "ebx",
    "Leone",
    "SEEK",
    "cdr",
    "legate",
    "witty",
    "mę",
    "afi",
    "uellen",
    "ARRANT",
    "ponsored",
    "isor",
]

# A sample is garbage-affected if it contains >= this many distinct garbage tokens
GARBAGE_THRESHOLD = 3

SAMPLE_MARKER_RE = re.compile(r"Sample (\d+)/(\d+)")


def parse_wandb_url(url: str):
    """Extract (entity, project, run_id) from a wandb URL."""
    url = url.strip().rstrip("/")
    # https://wandb.ai/entity/project/runs/run_id
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?\s]+)", url)
    if not m:
        raise ValueError(f"Cannot parse wandb URL: {url}")
    return m.group(1), m.group(2), m.group(3)


def fetch_run_data(api, entity, project, run_id):
    """Fetch config and log file content from a wandb run."""
    run = api.run(f"{entity}/{project}/{run_id}")
    cfg = run.config

    info = {
        "url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
        "run_id": run_id,
        "run_name": run.name,
        "temperature": cfg.get("generation", {}).get("temperature"),
        "top_p": cfg.get("generation", {}).get("top_p"),
        "model_path": cfg.get("model", {}).get("model_path", ""),
        "strategy_type": cfg.get("strategy", {}).get("type", ""),
        "dataset": cfg.get("dataset", {}).get(
            "data_name", cfg.get("strategy", {}).get("data_name", "")
        ),
        "num_paths": cfg.get("strategy", {}).get("num_paths"),
    }

    # Download log file
    log_content = None
    for f in run.files():
        if f.name.endswith("run_tts_eval.log"):
            tmpdir = tempfile.mkdtemp()
            f.download(root=tmpdir, replace=True)
            path = os.path.join(tmpdir, f.name)
            with open(path) as fh:
                log_content = fh.read()
            break

    if log_content is None:
        print(f"  WARNING: No run_tts_eval.log found for {run_id}", file=sys.stderr)

    return info, log_content


def analyze_log(log_content: str):
    """Parse log into samples and detect garbage generation.

    Returns dict with:
        total_samples, affected_samples, affected_indices,
        total_garbage_occurrences, garbage_lines_count, total_log_lines
    """
    if not log_content:
        return {
            "total_samples": 0,
            "affected_samples": 0,
            "affected_indices": [],
            "total_garbage_occurrences": 0,
            "garbage_lines_count": 0,
            "total_log_lines": 0,
        }

    lines = log_content.split("\n")

    # Find sample boundary line indices
    sample_boundaries = []
    for i, line in enumerate(lines):
        m = SAMPLE_MARKER_RE.search(line)
        if m:
            sample_num = int(m.group(1))
            total = int(m.group(2))
            sample_boundaries.append((i, sample_num, total))

    if not sample_boundaries:
        return {
            "total_samples": 0,
            "affected_samples": 0,
            "affected_indices": [],
            "total_garbage_occurrences": 0,
            "garbage_lines_count": 0,
            "total_log_lines": len(lines),
        }

    total_samples = sample_boundaries[-1][2]
    affected_indices = []
    total_garbage_occurrences = 0
    garbage_lines_count = 0

    for idx, (start_line, sample_num, _) in enumerate(sample_boundaries):
        # Sample text goes from this marker to the next marker (or end of file)
        if idx + 1 < len(sample_boundaries):
            end_line = sample_boundaries[idx + 1][0]
        else:
            end_line = len(lines)

        # Count distinct garbage tokens in this sample
        distinct_found = set()
        sample_occurrences = 0
        sample_garbage_lines = 0

        for line in lines[start_line:end_line]:
            line_has_garbage = False
            for tok in GARBAGE_TOKENS:
                count = line.count(tok)
                if count > 0:
                    distinct_found.add(tok)
                    sample_occurrences += count
                    line_has_garbage = True
            if line_has_garbage:
                sample_garbage_lines += 1

        total_garbage_occurrences += sample_occurrences
        garbage_lines_count += sample_garbage_lines

        if len(distinct_found) >= GARBAGE_THRESHOLD:
            affected_indices.append(sample_num)

    # Total lines in sample regions (from first marker to end)
    total_sample_lines = len(lines) - sample_boundaries[0][0]

    return {
        "total_samples": total_samples,
        "affected_samples": len(affected_indices),
        "affected_indices": affected_indices,
        "total_garbage_occurrences": total_garbage_occurrences,
        "garbage_lines_count": garbage_lines_count,
        "total_log_lines": total_sample_lines,
    }


def model_short_name(model_path: str) -> str:
    """Shorten model path for display."""
    if "/" in model_path:
        return model_path.split("/")[-1]
    return model_path


def generate_report(results: list[dict]) -> str:
    """Generate markdown report from analysis results."""
    lines = []
    lines.append("# Garbage Generation Analysis Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"\nGarbage detection threshold: ≥{GARBAGE_THRESHOLD} distinct garbage tokens per sample"
    )
    lines.append(f"\nGarbage token set: `{'`, `'.join(GARBAGE_TOKENS)}`")

    # Overall summary table
    lines.append("\n## Summary\n")
    lines.append(
        "| Dataset | Model | Strategy | Temp | top_p | Paths | Total | Affected | % | Garbage Lines | Lines % |"
    )
    lines.append(
        "|---------|-------|----------|------|-------|-------|-------|----------|---|---------------|---------|"
    )

    for r in sorted(results, key=lambda x: (x["dataset"], x["temperature"] or 0)):
        pct = (
            (r["affected_samples"] / r["total_samples"] * 100)
            if r["total_samples"] > 0
            else 0
        )
        lines_pct = (
            (r["garbage_lines_count"] / r["total_log_lines"] * 100)
            if r["total_log_lines"] > 0
            else 0
        )
        lines.append(
            f"| {r['dataset']} "
            f"| {model_short_name(r['model_path'])} "
            f"| {r['strategy_type']} "
            f"| {r['temperature']} "
            f"| {r['top_p']} "
            f"| {r['num_paths']} "
            f"| {r['total_samples']} "
            f"| {r['affected_samples']} "
            f"| {pct:.1f}% "
            f"| {r['garbage_lines_count']} "
            f"| {lines_pct:.1f}% |"
        )

    # Per-dataset grouping
    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r["dataset"]].append(r)

    for dataset, runs in sorted(by_dataset.items()):
        lines.append(f"\n## Dataset: {dataset}\n")
        for r in sorted(runs, key=lambda x: x["temperature"] or 0):
            pct = (
                (r["affected_samples"] / r["total_samples"] * 100)
                if r["total_samples"] > 0
                else 0
            )
            lines.append(
                f"### {model_short_name(r['model_path'])} | temp={r['temperature']} | top_p={r['top_p']}"
            )
            lines.append(f"\n- **Run**: [{r['run_id']}]({r['url']})")
            lines.append(
                f"- **Strategy**: {r['strategy_type']} ({r['num_paths']} paths)"
            )
            lines.append(
                f"- **Samples**: {r['total_samples']} total, **{r['affected_samples']}** affected ({pct:.1f}%)"
            )
            lines.append(f"- **Garbage occurrences**: {r['total_garbage_occurrences']}")
            lines_pct = (
                (r["garbage_lines_count"] / r["total_log_lines"] * 100)
                if r["total_log_lines"] > 0
                else 0
            )
            lines.append(
                f"- **Garbage lines**: {r['garbage_lines_count']}/{r['total_log_lines']} ({lines_pct:.1f}% of log)"
            )
            if r["affected_indices"]:
                indices_str = ", ".join(str(i) for i in r["affected_indices"])
                lines.append(f"- **Affected sample indices**: {indices_str}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze wandb runs for garbage token generation"
    )
    parser.add_argument("urls", nargs="*", help="wandb run URLs")
    parser.add_argument(
        "--runs-file", type=str, help="File with one wandb URL per line"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: reports/garbage_analysis_<timestamp>.md)",
    )
    args = parser.parse_args()

    urls = list(args.urls) if args.urls else []
    if args.runs_file:
        with open(args.runs_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

    if not urls:
        parser.error("No URLs provided. Pass URLs as arguments or use --runs-file.")

    api = wandb.Api()
    results = []

    for url in urls:
        try:
            entity, project, run_id = parse_wandb_url(url)
        except ValueError as e:
            print(f"Skipping invalid URL: {e}", file=sys.stderr)
            continue

        print(f"Fetching {entity}/{project}/{run_id}...", file=sys.stderr)
        info, log_content = fetch_run_data(api, entity, project, run_id)

        print("  Analyzing log...", file=sys.stderr)
        analysis = analyze_log(log_content)

        result = {**info, **analysis}
        results.append(result)

        pct = (
            (analysis["affected_samples"] / analysis["total_samples"] * 100)
            if analysis["total_samples"] > 0
            else 0
        )
        print(
            f"  → {analysis['affected_samples']}/{analysis['total_samples']} affected ({pct:.1f}%), "
            f"{analysis['total_garbage_occurrences']} garbage occurrences",
            file=sys.stderr,
        )

    if not results:
        print("No valid runs to analyze.", file=sys.stderr)
        sys.exit(1)

    report = generate_report(results)

    # Save report
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

    # Also print to stdout
    print(report)


if __name__ == "__main__":
    main()
