"""Test that experiment config filenames follow the naming convention.

Convention:
    config/experiments/{strategy_dir}/{dataset_dir}/{strategy_prefix}_{backend}_{model}_{dataset}_{scorer}.yaml

Rules:
    1. Filename must start with the correct strategy prefix
    2. Dataset directory name must appear in the filename (no abbreviations)
    3. For scored strategies, dataset must come BEFORE scorer
"""

import re
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "config" / "experiments"

# Strategy directory name -> expected filename prefix
STRATEGY_PREFIXES = {
    "offline_best_of_n": "offline_bon",
    "online_best_of_n": "online_bon",
    "beam_search": "beam_search",
    "adaptive_scaling": "adaptive_scaling",
    "baseline": "baseline",
    "self_consistency": "self_consistency",
}

# Strategies that require a scorer in the config name
SCORED_STRATEGIES = {
    "offline_best_of_n",
    "online_best_of_n",
    "beam_search",
    "adaptive_scaling",
}

KNOWN_SCORERS = {"entropy", "perplexity", "sequence_prob", "prm", "pd"}

KNOWN_DATASETS = {
    "amc23",
    "aime2024",
    "aime2025",
    "math500",
    "olympiadbench",
    "gaokao2023en",
    "minerva_math",
    "gpqa_diamond",
    "gsm8k",
    "game24",
    "mbpp_plus",
}


def _collect_configs(strategy_filter=None):
    """Collect config files, optionally filtered by strategy set."""
    configs = []
    for strategy_dir in CONFIGS_DIR.iterdir():
        if not strategy_dir.is_dir():
            continue
        if strategy_dir.name not in STRATEGY_PREFIXES:
            continue
        if strategy_filter and strategy_dir.name not in strategy_filter:
            continue
        for yaml_file in strategy_dir.rglob("*.yaml"):
            configs.append((strategy_dir.name, yaml_file))
    return configs


def test_filename_starts_with_strategy_prefix():
    """Verify that config filenames start with the correct strategy prefix."""
    configs = _collect_configs()
    assert len(configs) > 0, f"No configs found under {CONFIGS_DIR}"

    violations = []
    for strategy, config_path in sorted(configs, key=lambda x: x[1]):
        stem = config_path.stem
        expected_prefix = STRATEGY_PREFIXES[strategy]
        if not stem.startswith(expected_prefix + "_"):
            rel = config_path.relative_to(CONFIGS_DIR.parent.parent)
            violations.append(f"  {rel}: expected prefix '{expected_prefix}_'")

    assert not violations, (
        "Config files with wrong strategy prefix:\n" + "\n".join(violations)
    )


def test_filename_contains_dataset_dir_name():
    """Verify that the dataset directory name appears in the filename (no abbreviations)."""
    configs = _collect_configs()
    violations = []
    for strategy, config_path in sorted(configs, key=lambda x: x[1]):
        stem = config_path.stem
        dataset_dir = config_path.parent.name
        if dataset_dir not in stem:
            rel = config_path.relative_to(CONFIGS_DIR.parent.parent)
            violations.append(
                f"  {rel}: dataset dir '{dataset_dir}' not found in filename"
            )

    assert not violations, (
        "Config files where dataset directory name doesn't appear in filename "
        "(no abbreviations allowed):\n" + "\n".join(violations)
    )


def _find_position(stem, name):
    """Find position of _name_ or _name$ in stem."""
    for pattern in [f"_{name}_", f"_{name}$"]:
        m = re.search(pattern, stem)
        if m:
            return m.start()
    return None


def test_scored_configs_dataset_before_scorer():
    """Verify that in scored strategy configs, dataset appears before scorer in filename."""
    configs = _collect_configs(strategy_filter=SCORED_STRATEGIES)
    assert len(configs) > 0, f"No scored configs found under {CONFIGS_DIR}"

    violations = []
    for strategy, config_path in sorted(configs, key=lambda x: x[1]):
        stem = config_path.stem
        dataset_dir = config_path.parent.name

        dataset_pos = _find_position(stem, dataset_dir)
        if dataset_pos is None:
            continue  # already caught by test_filename_contains_dataset_dir_name

        # Find any scorer in the filename
        for scorer in sorted(KNOWN_SCORERS, key=len, reverse=True):
            scorer_pos = _find_position(stem, scorer)
            if scorer_pos is not None and scorer_pos < dataset_pos:
                rel = config_path.relative_to(CONFIGS_DIR.parent.parent)
                violations.append(
                    f"  {rel}: scorer '{scorer}' before dataset '{dataset_dir}'"
                )
                break

    assert not violations, (
        "Config files with scorer BEFORE dataset in filename "
        "(convention: _dataset_scorer):\n" + "\n".join(violations)
    )
