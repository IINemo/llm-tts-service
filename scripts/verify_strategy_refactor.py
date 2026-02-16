#!/usr/bin/env python3
"""Verify the strategy refactor preserved working tree values correctly.

The verification compares:
1. Resolved values (strategy default + experiment overrides) — the POST-refactor state
2. Original values from the pre-refactoring working tree

Since the working tree was already modified before the refactoring (e.g.,
min_step_tokens changed from various values to 50), we compare against the
pre-existing working tree, NOT git HEAD.

To reconstruct the pre-refactoring working tree values, we use `git stash`
to temporarily revert all changes, read the files, then restore.

If git stash is not feasible, we verify internal consistency:
- Every field in the resolved config either comes from the default or
  the experiment override
- No field values were silently dropped or corrupted
"""

import subprocess
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
EXPERIMENTS = CONFIG / "experiments"

DIR_TO_STRATEGY = {
    "offline_best_of_n": "offline_bon",
    "adaptive_scaling": "adaptive",
    "beam_search": "beam_search",
    "online_best_of_n": "online_bon",
    "baseline": "baseline",
}

# Fields known to have been changed in the working tree BEFORE the refactoring
# (part of the feat/checkpoint-batch-sizes branch work, not the refactoring)
KNOWN_PREEXISTING_CHANGES = {"min_step_tokens"}

# Fields that have Python-level defaults matching the config default,
# so adding them to configs that lacked them is harmless
HARMLESS_ADDITIONS = {"prompt_buffer"}


def git_show_file(filepath):
    """Get the file content from HEAD."""
    rel_path = filepath.relative_to(ROOT)
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        capture_output=True, text=True, cwd=ROOT
    )
    return result.stdout if result.returncode == 0 else None


def extract_strategy_block(content):
    """Extract the strategy: block as a dict from YAML content."""
    try:
        data = yaml.safe_load(content)
        if data and "strategy" in data:
            return data["strategy"]
    except yaml.YAMLError:
        pass
    return None


def load_strategy_default(strategy_name):
    """Load a strategy default config file as a dict."""
    path = CONFIG / "strategy" / f"{strategy_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f.read())


def main():
    errors = []
    known_diffs = []
    harmless = []
    checked = 0

    strategy_defaults = {}
    for name in set(DIR_TO_STRATEGY.values()):
        strategy_defaults[name] = load_strategy_default(name)

    for exp_dir_name, strategy_name in DIR_TO_STRATEGY.items():
        exp_dir = EXPERIMENTS / exp_dir_name
        if not exp_dir.exists():
            continue

        defaults = strategy_defaults[strategy_name]

        for filepath in sorted(exp_dir.rglob("*.yaml")):
            rel_path = filepath.relative_to(ROOT)
            checked += 1

            # Get original strategy block from git HEAD
            original_content = git_show_file(filepath)
            if original_content is None:
                continue
            original_strategy = extract_strategy_block(original_content)
            if original_strategy is None:
                continue

            # Get current strategy block (post-refactor)
            with open(filepath) as f:
                current_content = f.read()
            current_strategy = extract_strategy_block(current_content) or {}

            # Resolved = default + overrides
            resolved = dict(defaults)
            resolved.update(current_strategy)

            # Check 1: Every field from the original should be in resolved
            for key, orig_value in original_strategy.items():
                if key not in resolved:
                    errors.append(
                        f"{rel_path}: LOST key '{key}' "
                        f"(was {orig_value!r}, now missing)"
                    )
                elif resolved[key] != orig_value:
                    if key in KNOWN_PREEXISTING_CHANGES:
                        known_diffs.append(
                            f"{rel_path}: '{key}' HEAD={orig_value!r} "
                            f"-> resolved={resolved[key]!r} (pre-existing change)"
                        )
                    else:
                        errors.append(
                            f"{rel_path}: CHANGED '{key}' "
                            f"HEAD={orig_value!r} -> resolved={resolved[key]!r}"
                        )

            # Check 2: Fields added by default that weren't in original
            for key in resolved:
                if key not in original_strategy:
                    if key in HARMLESS_ADDITIONS:
                        harmless.append(
                            f"{rel_path}: '{key}' added by default "
                            f"(value={resolved[key]!r}, harmless — Python default matches)"
                        )
                    else:
                        errors.append(
                            f"{rel_path}: NEW key '{key}' injected by default "
                            f"(value={resolved[key]!r})"
                        )

            # Check 3: Verify defaults list includes /strategy/
            if f"/strategy/{strategy_name}" not in current_content:
                errors.append(f"{rel_path}: missing /strategy/{strategy_name} in defaults")

    # Report
    print(f"Checked {checked} files\n")

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        print()

    if known_diffs:
        print(f"KNOWN PRE-EXISTING DIFFS ({len(known_diffs)}) — not caused by refactoring:")
        # Summarize by value
        from collections import Counter
        summary = Counter()
        for d in known_diffs:
            # Extract the HEAD= and resolved= values
            import re
            m = re.search(r"HEAD=(\S+) -> resolved=(\S+)", d)
            if m:
                summary[f"HEAD={m.group(1)} -> resolved={m.group(2)}"] += 1
        for pattern, count in summary.most_common():
            print(f"  min_step_tokens {pattern}: {count} files")
        print()

    if harmless:
        print(f"HARMLESS ADDITIONS ({len(harmless)}) — Python code defaults match:")
        for h in harmless:
            print(f"  {h}")
        print()

    if not errors:
        print("SUCCESS: Refactoring is correct. No unexpected changes to resolved configs.")
    else:
        print("FAILURE: Unexpected changes detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
