"""
Strategy Registry and Test Validation

This module maintains a registry of all TTS strategies and validates
that each strategy has corresponding tests.

Usage:
    python tests/strategy_registry.py --validate
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class StrategyInfo:
    """Information about a registered strategy"""

    name: str  # Strategy name (e.g., "deepconf")
    class_name: str  # Class name in code (e.g., "StrategyDeepConf")
    module_path: str  # Path to strategy module
    test_dir: str  # Path to test directory
    required_tests: List[str]  # Required test files
    description: str  # Brief description


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================
# ADD YOUR STRATEGY HERE when implementing a new strategy
# This ensures tests are required before merging

REGISTERED_STRATEGIES = [
    StrategyInfo(
        name="deepconf",
        class_name="StrategyDeepConf",
        module_path="llm_tts/strategies/deepconf/strategy.py",
        test_dir="tests/deepconf",
        required_tests=[
            "test_deepconf_accurate.py",  # Unit tests (pytest)
            "test_online_mode.py",  # Online mode tests (pytest)
            # Note: test_deepconf_math.py is a standalone script, not pytest
        ],
        description="Confidence-based test-time scaling with trace filtering",
    ),
    StrategyInfo(
        name="online_best_of_n",
        class_name="StrategyOnlineBestOfN",
        module_path="llm_tts/strategies/strategy_online_best_of_n.py",
        test_dir="tests/online_best_of_n",
        required_tests=[
            "test_online_best_of_n.py",  # Strategy logic tests
        ],
        description="Step-by-step generation with PRM scoring",
    ),
    StrategyInfo(
        name="uncert_cot",
        class_name="StrategyUncertaintyCoT",
        module_path="llm_tts/strategies/strategy_uncertainty_cot.py",
        test_dir="tests/uncert_cot",
        required_tests=[
            "test_uncert_cot.py",
        ],
        description="Uncertainty-guided chain-of-thought with PD",
        name="tree_of_thoughts",
        class_name="StrategyTreeOfThoughts",
        module_path="llm_tts/strategies/tree_of_thoughts/strategy.py",
        test_dir="tests/tree_of_thoughts",
        required_tests=[
            "test_tree_of_thoughts.py",  # Strategy logic and integration tests
        ],
        description="Beam search with LLM-based state evaluation",
    ),
    # TODO: Add tests for self_consistency strategy
    # StrategyInfo(
    #     name="self_consistency",
    #     class_name="StrategySelfConsistency",
    #     module_path="llm_tts/strategies/strategy_self_consistency.py",
    #     test_dir="tests/self_consistency",
    #     required_tests=[
    #         "test_self_consistency.py",
    #     ],
    #     description="Majority voting across multiple reasoning paths",
    # ),
    # TODO: Add tests for chain_of_thought strategy
    # StrategyInfo(
    #     name="chain_of_thought",
    #     class_name="StrategyChainOfThought",
    #     module_path="llm_tts/strategies/strategy_chain_of_thought.py",
    #     test_dir="tests/chain_of_thought",
    #     required_tests=[
    #         "test_chain_of_thought.py",
    #     ],
    #     description="Single-pass step-by-step reasoning",
    # ),
]


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_strategy(strategy: StrategyInfo, repo_root: Path) -> List[str]:
    """
    Validate that a strategy has all required components.

    Args:
        strategy: Strategy information
        repo_root: Path to repository root

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check strategy module exists
    strategy_path = repo_root / strategy.module_path
    if not strategy_path.exists():
        errors.append(
            f"❌ Strategy module not found: {strategy.module_path}\n"
            f"   Expected at: {strategy_path}"
        )

    # Check test directory exists
    test_dir = repo_root / strategy.test_dir
    if not test_dir.exists():
        errors.append(
            f"❌ Test directory not found: {strategy.test_dir}\n"
            f"   Expected at: {test_dir}\n"
            f"   Create with: mkdir -p {strategy.test_dir}"
        )
        return errors  # No point checking test files if dir doesn't exist

    # Check each required test file exists
    for test_file in strategy.required_tests:
        test_path = test_dir / test_file
        if not test_path.exists():
            errors.append(
                f"❌ Required test file missing: {strategy.test_dir}/{test_file}\n"
                f"   Expected at: {test_path}\n"
                f"   Create with: touch {test_path}"
            )

    # Check test files have actual test functions
    if test_dir.exists():
        for test_file in strategy.required_tests:
            test_path = test_dir / test_file
            if test_path.exists():
                content = test_path.read_text()
                if "def test_" not in content:
                    errors.append(
                        f"⚠️  Test file has no test functions: {strategy.test_dir}/{test_file}\n"
                        f"   Add at least one function starting with 'def test_'"
                    )

    return errors


def validate_all_strategies(repo_root: Optional[Path] = None) -> bool:
    """
    Validate all registered strategies have required tests.

    Args:
        repo_root: Path to repository root (auto-detected if None)

    Returns:
        True if all strategies are valid, False otherwise
    """
    if repo_root is None:
        # Auto-detect repo root (assume script is in tests/)
        repo_root = Path(__file__).parent.parent

    print("=" * 80)
    print("STRATEGY REGISTRY VALIDATION")
    print("=" * 80)
    print(f"\nRepository root: {repo_root}")
    print(f"Registered strategies: {len(REGISTERED_STRATEGIES)}\n")

    all_valid = True
    all_errors = []

    for strategy in REGISTERED_STRATEGIES:
        print(f"\n📋 Validating: {strategy.name}")
        print(f"   Class: {strategy.class_name}")
        print(f"   Module: {strategy.module_path}")
        print(f"   Tests: {strategy.test_dir}")

        errors = validate_strategy(strategy, repo_root)

        if errors:
            all_valid = False
            print(f"\n   ❌ FAILED - {len(errors)} error(s) found:")
            for error in errors:
                print(f"      {error}")
                all_errors.append(f"{strategy.name}: {error}")
        else:
            print("   ✅ PASSED - All required tests present")

    print("\n" + "=" * 80)
    if all_valid:
        print("✅ ALL STRATEGIES VALID")
        print("=" * 80)
        return True
    else:
        print("❌ VALIDATION FAILED")
        print("=" * 80)
        print(f"\nTotal errors: {len(all_errors)}")
        print("\nFix these issues before merging:\n")
        for error in all_errors:
            print(f"  • {error}")
        print("\nSee tests/README.md for testing guidelines")
        print("=" * 80)
        return False


def list_strategies():
    """List all registered strategies with details"""
    print("=" * 80)
    print("REGISTERED TTS STRATEGIES")
    print("=" * 80)
    print(f"\nTotal strategies: {len(REGISTERED_STRATEGIES)}\n")

    for i, strategy in enumerate(REGISTERED_STRATEGIES, 1):
        print(f"{i}. {strategy.name}")
        print(f"   Class: {strategy.class_name}")
        print(f"   Description: {strategy.description}")
        print(f"   Module: {strategy.module_path}")
        print(f"   Test dir: {strategy.test_dir}")
        print("   Required tests:")
        for test in strategy.required_tests:
            print(f"     - {test}")
        print()


def get_strategy_template(name: str) -> str:
    """
    Get a template for adding a new strategy to the registry.

    Args:
        name: Strategy name (e.g., "my_strategy")

    Returns:
        Python code to add to REGISTERED_STRATEGIES
    """
    class_name = "".join(word.capitalize() for word in name.split("_"))
    class_name = f"Strategy{class_name}"

    template = f"""
StrategyInfo(
    name="{name}",
    class_name="{class_name}",
    module_path="llm_tts/strategies/strategy_{name}.py",
    test_dir="tests/{name}",
    required_tests=[
        "test_{name}_logic.py",      # Unit tests (no API)
        "test_{name}_integration.py", # Integration tests (with API)
        "test_{name}_math.py",        # Math validation tests
    ],
    description="TODO: Brief description of {name} strategy",
),
"""
    return template


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Strategy Registry and Test Validation"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all strategies have required tests",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all registered strategies"
    )
    parser.add_argument(
        "--template",
        metavar="NAME",
        help="Generate template for new strategy (e.g., my_strategy)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Path to repository root (auto-detected if not provided)",
    )

    args = parser.parse_args()

    if args.list:
        list_strategies()
    elif args.template:
        print("Add this to REGISTERED_STRATEGIES in tests/strategy_registry.py:")
        print(get_strategy_template(args.template))
    elif args.validate:
        success = validate_all_strategies(args.repo_root)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python tests/strategy_registry.py --validate")
        print("  python tests/strategy_registry.py --list")
        print("  python tests/strategy_registry.py --template my_strategy")
