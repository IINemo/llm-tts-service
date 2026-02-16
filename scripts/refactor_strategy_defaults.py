#!/usr/bin/env python3
"""Refactor experiment configs to use Hydra strategy defaults.

This script:
1. Inserts /strategy/<name> into each experiment's Hydra defaults list
2. Removes redundant strategy fields from experiment configs whose values
   match the strategy default config
"""

import re
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
EXPERIMENTS = CONFIG / "experiments"

# Map experiment directory name -> strategy default config name
DIR_TO_STRATEGY = {
    "offline_best_of_n": "offline_bon",
    "adaptive_scaling": "adaptive",
    "beam_search": "beam_search",
    "online_best_of_n": "online_bon",
    "baseline": "baseline",
}

# Fields that are uniform across ALL experiments of each strategy type.
# These will be inherited from the strategy default config and removed
# from individual experiment configs when values match.
STRATEGY_DEFAULTS = {
    "offline_bon": {
        "type": "offline_best_of_n",
        "num_trajectories": 8,
        "batch_generation": True,
        "score_aggregation": "min",
        "min_step_tokens": 50,
    },
    "adaptive": {
        "type": "adaptive",
        "candidates_per_step": 8,
        "scaling_rate": 0.9,
        "momentum_rate": 0.9,
        "adaptive_scaling_method": "momentum",
        "detector_type": "structured",
        "min_step_tokens": 50,
    },
    "beam_search": {
        "type": "beam_search",
        "aggregation": "mean",
        "batch_generation": True,
        "prompt_buffer": 500,
        "min_step_tokens": 50,
    },
    "online_bon": {
        "type": "online_best_of_n",
        "candidates_per_step": 8,
        "min_step_tokens": 50,
    },
    "baseline": {
        "type": "baseline",
        "min_step_tokens": 0,
        "max_step_tokens": 32768,
        "use_sequence": False,
        "use_conclusion": False,
        "use_thinking": False,
        "use_verification": False,
        "use_structure": False,
        "use_reasoning": False,
        "use_correction": False,
    },
}


def parse_yaml_value(value_str):
    """Parse a YAML scalar value from a raw string."""
    if not value_str:
        return None
    try:
        return yaml.safe_load(value_str)
    except yaml.YAMLError:
        return value_str


def insert_strategy_default(lines, strategy_name):
    """Insert /strategy/<name> into the defaults list before _self_.

    Inserts before the /scorer/ line if present, otherwise before _self_.
    Returns (new_lines, inserted_bool).
    """
    # Find the defaults: block
    defaults_start = None
    for i, line in enumerate(lines):
        if line.strip() == "defaults:":
            defaults_start = i
            break

    if defaults_start is None:
        return lines, False

    # Check if already present
    for line in lines:
        if f"/strategy/{strategy_name}" in line:
            return lines, False

    # Scan defaults entries to find insertion point
    insert_idx = None
    defaults_indent = None

    for i in range(defaults_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- "):
            if defaults_indent is None:
                defaults_indent = len(line) - len(line.lstrip())

            item = stripped[2:].strip()

            # Insert before /scorer/ line
            if "/scorer/" in item and insert_idx is None:
                insert_idx = i

            # Fall through to _self_ if no scorer found
            if "_self_" in item:
                if insert_idx is None:
                    insert_idx = i
                break
        elif not stripped.startswith("-"):
            # End of defaults block
            break

    if insert_idx is None:
        return lines, False

    if defaults_indent is None:
        defaults_indent = 2

    indent = " " * defaults_indent
    new_line = f"{indent}- /strategy/{strategy_name}\n"

    new_lines = lines[:insert_idx] + [new_line] + lines[insert_idx:]
    return new_lines, True


def remove_matching_strategy_fields(lines, defaults):
    """Remove strategy fields whose values match the defaults.

    Only removes scalar key: value lines at the first level inside
    the strategy: block. Multi-line values (lists, dicts) are never removed.
    Returns (new_lines, removed_field_names).
    """
    # Find strategy: block
    strategy_start = None
    strategy_indent = None

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("strategy:") and not line.lstrip().startswith("#"):
            strategy_start = i
            strategy_indent = len(line) - len(stripped)
            break

    if strategy_start is None:
        return lines, []

    field_indent = strategy_indent + 2

    # Find end of strategy block
    strategy_end = len(lines)
    for i in range(strategy_start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        current_indent = len(lines[i]) - len(lines[i].lstrip())
        if current_indent <= strategy_indent:
            strategy_end = i
            break

    # Identify lines to remove
    lines_to_remove = set()
    removed_fields = []

    i = strategy_start + 1
    while i < strategy_end:
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        current_indent = len(line) - len(line.lstrip())

        if current_indent == field_indent:
            match = re.match(r"([\w][\w-]*)\s*:\s*(.*)", stripped)
            if match:
                key = match.group(1)
                value_and_comment = match.group(2).strip()

                # Multi-line value (list or dict) — skip entirely
                if not value_and_comment or value_and_comment.startswith("#"):
                    j = i + 1
                    while j < strategy_end:
                        if not lines[j].strip():
                            j += 1
                            continue
                        if lines[j].strip().startswith("#"):
                            j += 1
                            continue
                        sub_indent = len(lines[j]) - len(lines[j].lstrip())
                        if sub_indent > field_indent:
                            j += 1
                        else:
                            break
                    i = j
                    continue

                # Scalar value — extract and compare
                value_str = value_and_comment
                if "#" in value_str:
                    value_str = value_str.split("#")[0].strip()

                parsed_value = parse_yaml_value(value_str)

                if key in defaults and parsed_value == defaults[key]:
                    lines_to_remove.add(i)
                    removed_fields.append(key)

        i += 1

    if not lines_to_remove:
        return lines, []

    new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]

    # Clean up consecutive blank lines in strategy area
    cleaned = []
    prev_blank = False
    for line in new_lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return cleaned, removed_fields


def process_experiment_config(filepath, strategy_name):
    """Process a single experiment config file.

    Returns (inserted_bool, list_of_removed_field_names).
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Insert strategy default into defaults list
    lines, inserted = insert_strategy_default(lines, strategy_name)

    # Remove matching fields from strategy block
    defaults = STRATEGY_DEFAULTS[strategy_name]
    lines, removed = remove_matching_strategy_fields(lines, defaults)

    if inserted or removed:
        with open(filepath, "w") as f:
            f.writelines(lines)

    return inserted, removed


def main():
    print("=" * 60)
    print("Refactoring experiment configs to use strategy defaults")
    print("=" * 60)

    summary = defaultdict(
        lambda: {"count": 0, "inserted": 0, "fields_removed": defaultdict(int)}
    )

    for exp_dir_name, strategy_name in DIR_TO_STRATEGY.items():
        exp_dir = EXPERIMENTS / exp_dir_name
        if not exp_dir.exists():
            print(f"\n  WARNING: Directory not found: {exp_dir}")
            continue

        yaml_files = sorted(exp_dir.rglob("*.yaml"))
        print(f"\n  {exp_dir_name} -> /strategy/{strategy_name} ({len(yaml_files)} files)")

        for filepath in yaml_files:
            inserted, removed = process_experiment_config(filepath, strategy_name)
            rel_path = filepath.relative_to(ROOT)

            summary[strategy_name]["count"] += 1
            if inserted:
                summary[strategy_name]["inserted"] += 1
            for field in removed:
                summary[strategy_name]["fields_removed"][field] += 1

            status = []
            if inserted:
                status.append("default added")
            if removed:
                status.append(f"removed: {', '.join(removed)}")
            if status:
                print(f"    {rel_path}: {'; '.join(status)}")
            else:
                print(f"    {rel_path}: no changes")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_files = 0
    total_inserted = 0
    total_removals = 0

    for strategy_name, info in summary.items():
        total_files += info["count"]
        total_inserted += info["inserted"]
        removals = sum(info["fields_removed"].values())
        total_removals += removals
        print(f"\n  {strategy_name}:")
        print(f"    Files processed: {info['count']}")
        print(f"    Defaults inserted: {info['inserted']}")
        if info["fields_removed"]:
            print(f"    Fields removed:")
            for field, count in sorted(info["fields_removed"].items()):
                print(f"      {field}: {count} times")

    print(f"\n  TOTAL: {total_files} files, {total_inserted} defaults inserted, {total_removals} field removals")


if __name__ == "__main__":
    main()
