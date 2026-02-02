"""Utilities for managing evaluation results (reading, writing, resuming)."""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from llm_tts.generators import StepCandidate

log = logging.getLogger(__name__)


def safe_serialize(obj):
    """Convert arbitrary Python objects (including tensors, numpy) to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_serialize(v) for v in obj]
    if isinstance(obj, StepCandidate):
        return {
            "text": obj.text,
            "token_ids": list(obj.token_ids) if obj.token_ids is not None else None,
            "is_complete": obj.is_complete,
            "is_trajectory_complete": obj.is_trajectory_complete,
            "generation_scores": safe_serialize(obj.generation_scores),
            "raw_text": obj.raw_text,
            "other_data": safe_serialize({k: v for k, v in obj.other_data.items() if k != "raw_logprobs"}) if obj.other_data else None,
        }
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return safe_serialize(vars(obj))

    # Fallback to string representation
    return str(obj)


def save_results_json(results, json_path: Path):
    """
    Save results to a JSON file with safe serialization.

    Args:
        results: List of result dictionaries
        json_path: Path to save the JSON file
    """
    json_path = Path(json_path)
    json_data = [safe_serialize(r) for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def load_results_json(json_path: Path):
    """
    Load existing results from a JSON file for resuming evaluation.

    Args:
        json_path: Path to the results.json file

    Returns:
        Tuple of (results, processed_indices)
    """
    json_path = Path(json_path)

    if not json_path.exists():
        return [], set()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Extract processed indices
        processed_indices = {r["index"] for r in results if "index" in r}

        log.info(f"Loaded {len(results)} existing results from {json_path}")
        log.info(f"Resuming from {len(processed_indices)} processed samples")

        return results, processed_indices

    except Exception as e:
        log.warning(f"Failed to load existing results from {json_path}: {e}")
        return [], set()


def find_latest_output_dir():
    """Find the most recent output directory."""
    outputs_root = Path("outputs")
    if not outputs_root.exists():
        return None

    # Find all timestamped directories (YYYY-MM-DD/HH-MM-SS)
    all_dirs = []
    for date_dir in outputs_root.iterdir():
        if date_dir.is_dir() and date_dir.name.count("-") == 2:  # YYYY-MM-DD format
            for time_dir in date_dir.iterdir():
                if (
                    time_dir.is_dir() and time_dir.name.count("-") == 2
                ):  # HH-MM-SS format
                    all_dirs.append(time_dir)

    if not all_dirs:
        return None

    # Sort by modification time, return most recent
    latest_dir = max(all_dirs, key=lambda p: p.stat().st_mtime)
    return latest_dir


def parse_resume_arguments():
    """
    Parse custom --resume and --resume-from arguments.

    Returns the resume directory path if found, otherwise None.
    Modifies sys.argv to remove custom arguments and inject hydra.run.dir override.
    """
    resume_dir = None
    args_to_remove = []

    for i, arg in enumerate(sys.argv):
        if arg == "--resume":
            # Find latest directory
            resume_dir = find_latest_output_dir()
            if not resume_dir:
                print("ERROR: No previous output directories found to resume from.")
                sys.exit(1)
            args_to_remove.append(i)
            print(f"Resuming from latest directory: {resume_dir}")

        elif arg == "--resume-from":
            # Get the next argument as the directory path
            if i + 1 >= len(sys.argv):
                print("ERROR: --resume-from requires a directory path argument")
                sys.exit(1)
            resume_dir = Path(sys.argv[i + 1])
            if not resume_dir.exists():
                print(f"ERROR: Resume directory does not exist: {resume_dir}")
                sys.exit(1)
            args_to_remove.extend([i, i + 1])
            print(f"Resuming from specified directory: {resume_dir}")

    # Remove custom arguments from sys.argv
    for idx in sorted(args_to_remove, reverse=True):
        sys.argv.pop(idx)

    # If resuming, inject hydra.run.dir override
    if resume_dir:
        hydra_override = f"hydra.run.dir={resume_dir}"
        sys.argv.append(hydra_override)

    return resume_dir
