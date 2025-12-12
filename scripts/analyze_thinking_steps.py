#!/usr/bin/env python3
"""
Analyze thinking trajectories using different step boundary detectors.

This script loads results from thinking mode experiments and applies
various step boundary detectors to extract and compare step segmentations.

Measures:
- Processing time per sample and average
- Number of steps detected
- Step length statistics (min, max, avg, total chars)
"""

import argparse
import json
import logging
import re
import sys
import time
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_tts.step_boundary_detectors import (
    ThinkingAdaptiveDetector,
    ThinkingHybridDetector,
    ThinkingLLMDetector,
    ThinkingLLMDetectorVLLM,
    ThinkingMarkerDetector,
    ThinkingSentenceDetector,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class StepStats:
    """Statistics for steps detected in a single sample."""
    num_steps: int = 0
    total_chars: int = 0
    min_chars: int = 0
    max_chars: int = 0
    avg_chars: float = 0.0
    processing_time_ms: float = 0.0


def compute_step_stats(steps: List[str], processing_time_ms: float) -> StepStats:
    """Compute statistics for a list of steps."""
    if not steps:
        return StepStats(processing_time_ms=processing_time_ms)

    step_lengths = [len(s) for s in steps]
    return StepStats(
        num_steps=len(steps),
        total_chars=sum(step_lengths),
        min_chars=min(step_lengths),
        max_chars=max(step_lengths),
        avg_chars=statistics.mean(step_lengths),
        processing_time_ms=processing_time_ms,
    )


def extract_thinking_content(text: str) -> str:
    """Extract content from <think> tags if present."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return text.strip()


def analyze_trajectory(
    thinking_content: str,
    detectors: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply all detectors to a single thinking trajectory with timing.

    Args:
        thinking_content: The thinking content (without <think> tags)
        detectors: Dictionary of detector name -> detector instance

    Returns:
        Dictionary with detection results for each detector including timing
    """
    results = {}

    for name, detector in detectors.items():
        try:
            # Measure processing time
            start_time = time.perf_counter()
            steps = detector.detect_steps(thinking_content)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Compute stats
            stats = compute_step_stats(steps, elapsed_ms)

            results[name] = {
                "num_steps": stats.num_steps,
                "steps": steps,
                "step_lengths": [len(s) for s in steps],
                "total_chars": stats.total_chars,
                "min_chars": stats.min_chars,
                "max_chars": stats.max_chars,
                "avg_step_length": stats.avg_chars,
                "processing_time_ms": stats.processing_time_ms,
            }

            # For marker detector, also get marker stats
            if hasattr(detector, "get_marker_stats"):
                results[name]["marker_stats"] = detector.get_marker_stats(thinking_content)

        except Exception as e:
            log.warning(f"Detector {name} failed: {e}")
            results[name] = {"error": str(e), "num_steps": 0, "steps": [], "processing_time_ms": 0}

    return results


def analyze_results_file(
    input_path: str,
    output_path: str,
    detectors: Optional[Dict[str, Any]] = None,
    max_samples: Optional[int] = None,
    use_gpt4: bool = False,
    use_qwen3: bool = False,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load results file, apply detectors to all trajectories, and save analysis.

    Args:
        input_path: Path to input results.json
        output_path: Path to output analysis JSON
        detectors: Optional custom detectors dict

    Returns:
        Summary statistics
    """
    log.info(f"Loading results from: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    # Handle both formats: list of results or dict with "results" key
    if isinstance(data, dict):
        results = data.get("results", data.get("samples", []))
    else:
        results = data

    # Limit samples if specified
    if max_samples is not None and max_samples > 0:
        results = results[:max_samples]

    log.info(f"Loaded {len(results)} samples")

    # Initialize detectors if not provided
    if detectors is None:
        detectors = {
            "sentence_paragraph": ThinkingSentenceDetector(
                split_mode="paragraph",
                min_step_chars=50,
                max_step_chars=1000,
            ),
            "sentence_both": ThinkingSentenceDetector(
                split_mode="both",
                min_step_chars=50,
                max_step_chars=800,
            ),
            "marker_all": ThinkingMarkerDetector(
                use_sequence=True,
                use_conclusion=True,
                use_thinking=True,
                use_verification=True,
                use_structure=True,
                min_step_chars=50,
                max_step_chars=800,
            ),
            "marker_semantic": ThinkingMarkerDetector(
                use_sequence=True,
                use_conclusion=True,
                use_thinking=True,
                use_verification=True,
                use_structure=False,  # No paragraph/structure markers
                min_step_chars=100,
                max_step_chars=600,
            ),
            "hybrid": ThinkingHybridDetector(
                min_steps=3,
                max_steps=30,
                min_step_chars=50,
                max_step_chars=800,
            ),
            "adaptive": ThinkingAdaptiveDetector(
                min_step_chars=50,
                max_step_chars=800,
            ),
        }

        # Add LLM-based detectors if requested
        if use_gpt4:
            api_key = openai_api_key
            if not api_key:
                import os
                api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                detectors["llm_gpt4"] = ThinkingLLMDetector(
                    llm_client=client,
                    model_name="gpt-4.1-mini",
                    temperature=0.0,
                    max_tokens=4096,
                    cache_results=True,
                )
                log.info("Added GPT-4.1-mini detector")
            else:
                log.warning("No OpenAI API key provided, skipping GPT-4.1-mini")

        if use_qwen3:
            try:
                from vllm import LLM
                log.info("Loading Qwen3-8B via vLLM...")
                vllm_engine = LLM(
                    model="Qwen/Qwen3-8B",
                    trust_remote_code=True,
                    gpu_memory_utilization=0.85,
                    max_model_len=32768,
                    enforce_eager=True,
                )
                detectors["llm_qwen3_8b"] = ThinkingLLMDetectorVLLM(
                    vllm_engine=vllm_engine,
                    temperature=0.0,
                    max_tokens=4096,
                    cache_results=True,
                )
                log.info("Added Qwen3-8B detector via vLLM")
            except Exception as e:
                log.warning(f"Could not load Qwen3-8B: {e}")

    # Process each sample
    analyzed_results = []
    summary_stats = {
        detector_name: {
            "total_steps": 0,
            "samples": 0,
            "total_time_ms": 0.0,
            "all_step_lengths": [],
        }
        for detector_name in detectors
    }

    for i, sample in enumerate(results):
        log.info(f"Processing sample {i + 1}/{len(results)}: index={sample.get('index')}")

        # Get all traces
        all_traces = sample.get("all_traces", [])
        if not all_traces:
            log.warning(f"No traces found for sample {sample.get('index')}")
            continue

        # Analyze each trace
        analyzed_traces = []
        for trace_idx, trace in enumerate(all_traces):
            text = trace.get("text", "")
            thinking_content = extract_thinking_content(text)

            if not thinking_content:
                log.warning(f"No thinking content in trace {trace_idx}")
                continue

            # Apply detectors
            detection_results = analyze_trajectory(thinking_content, detectors)

            analyzed_trace = {
                "trace_idx": trace_idx,
                "original_answer": trace.get("answer"),
                "original_score": trace.get("score"),
                "thinking_length": len(thinking_content),
                "detections": detection_results,
            }
            analyzed_traces.append(analyzed_trace)

            # Update summary stats
            for detector_name, det_result in detection_results.items():
                summary_stats[detector_name]["total_steps"] += det_result.get("num_steps", 0)
                summary_stats[detector_name]["samples"] += 1
                summary_stats[detector_name]["total_time_ms"] += det_result.get("processing_time_ms", 0)
                summary_stats[detector_name]["all_step_lengths"].extend(det_result.get("step_lengths", []))

        # Build analyzed sample
        analyzed_sample = {
            "index": sample.get("index"),
            "question": sample.get("question"),
            "gold_answer": sample.get("gold_answer"),
            "generated_answer": sample.get("generated_answer"),
            "correct": str(sample.get("generated_answer")) == str(sample.get("gold_answer")),
            "analyzed_traces": analyzed_traces,
        }
        analyzed_results.append(analyzed_sample)

    # Calculate summary statistics
    for detector_name in detectors:
        stats = summary_stats[detector_name]
        if stats["samples"] > 0:
            stats["avg_steps_per_trace"] = stats["total_steps"] / stats["samples"]
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["samples"]
        else:
            stats["avg_steps_per_trace"] = 0
            stats["avg_time_ms"] = 0

        # Calculate step length statistics
        step_lengths = stats["all_step_lengths"]
        if step_lengths:
            stats["min_step_chars"] = min(step_lengths)
            stats["max_step_chars"] = max(step_lengths)
            stats["avg_step_chars"] = statistics.mean(step_lengths)
            stats["median_step_chars"] = statistics.median(step_lengths)
        else:
            stats["min_step_chars"] = 0
            stats["max_step_chars"] = 0
            stats["avg_step_chars"] = 0
            stats["median_step_chars"] = 0

        # Remove raw step lengths from summary (too large)
        del stats["all_step_lengths"]

    # Build output
    def serialize_params(obj):
        """Convert detector params to JSON-serializable format."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [serialize_params(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: serialize_params(v) for k, v in obj.items()}
        else:
            return str(type(obj).__name__)

    output_data = {
        "source_file": str(input_path),
        "num_samples": len(analyzed_results),
        "detector_configs": {
            name: {
                "class": det.__class__.__name__,
                "params": {
                    k: serialize_params(v)
                    for k, v in det.__dict__.items()
                    if not k.startswith("_")
                    and not callable(v)
                    and k not in ("pattern", "marker_detector", "sentence_detector", "hybrid_detector", "llm_detector")
                },
            }
            for name, det in detectors.items()
        },
        "summary_stats": summary_stats,
        "samples": analyzed_results,
    }

    # Save results
    log.info(f"Saving analysis to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary table
    log.info("\n" + "=" * 120)
    log.info("SUMMARY STATISTICS")
    log.info("=" * 120)
    log.info(f"{'Detector':<20} {'Steps':>7} {'Avg/Tr':>8} {'AvgChar':>8} {'MinChar':>8} {'MaxChar':>8} {'MedChar':>8} {'AvgMs':>10} {'TotalMs':>12}")
    log.info("-" * 120)
    for detector_name, stats in summary_stats.items():
        log.info(
            f"{detector_name:<20} {stats['total_steps']:>7} "
            f"{stats['avg_steps_per_trace']:>8.1f} "
            f"{stats['avg_step_chars']:>8.1f} "
            f"{stats['min_step_chars']:>8} "
            f"{stats['max_step_chars']:>8} "
            f"{stats['median_step_chars']:>8.1f} "
            f"{stats['avg_time_ms']:>10.2f} "
            f"{stats['total_time_ms']:>12.2f}"
        )
    log.info("=" * 120)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Analyze thinking trajectories with step boundary detectors"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input results.json file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output analysis JSON (default: input_dir/step_boundary_analysis.json)",
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)",
    )
    parser.add_argument(
        "--use-gpt4",
        action="store_true",
        help="Include GPT-4.1-mini detector (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--use-qwen3",
        action="store_true",
        help="Include Qwen3-8B detector via vLLM",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"_n{args.max_samples}" if args.max_samples else ""
        output_path = input_path.parent / f"step_boundary_timing{suffix}.json"

    analyze_results_file(
        str(input_path),
        str(output_path),
        max_samples=args.max_samples,
        use_gpt4=args.use_gpt4,
        use_qwen3=args.use_qwen3,
        openai_api_key=args.openai_api_key,
    )


if __name__ == "__main__":
    main()
