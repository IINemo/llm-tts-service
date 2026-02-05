#!/usr/bin/env python3
"""
Integration tests for HumanEval+ support.

This test validates the full pipeline:
1. Dataset loading
2. Code extraction
3. Evaluator scoring
4. Config parsing

Run: python tests/test_human_eval_plus_integration.py
"""

import sys
import tempfile

# Suppress ANTLR warnings
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="ANTLR runtime")


def test_dataset_loading():
    """Test loading HumanEval+ dataset."""
    print("=" * 60)
    print("Test 1: Dataset Loading")
    print("=" * 60)

    from llm_tts.datasets import load_human_eval_plus

    # Load small subset
    data = load_human_eval_plus(subset_size=5)

    assert len(data) == 5, f"Expected 5 samples, got {len(data)}"
    assert "question" in data[0], "Missing 'question' field"
    assert "answer" in data[0], "Missing 'answer' field"
    assert "task_id" in data[0], "Missing 'task_id' field"
    assert "entry_point" in data[0], "Missing 'entry_point' field"

    print(f"  Loaded {len(data)} problems")
    print(f"  First task: {data[0]['task_id']}")
    print(f"  Entry point: {data[0]['entry_point']}")
    print(f"  Prompt preview: {data[0]['question'][:80]}...")
    print("  PASSED")
    print()

    return data


def test_code_extraction():
    """Test code extraction from model responses."""
    print("=" * 60)
    print("Test 2: Code Extraction")
    print("=" * 60)

    from llm_tts.datasets.human_eval_plus import extract_code_from_response

    # Test case 1: Code in Python code block
    response1 = """
Here's the solution:

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

This function checks if any two elements are closer than the threshold.
"""
    code1 = extract_code_from_response(response1)
    assert "def has_close_elements" in code1, "Failed to extract from code block"
    assert "return True" in code1, "Missing function body"
    print("  Test case 1 (python code block): PASSED")

    # Test case 2: Code in generic code block
    response2 = """
```
def add(a, b):
    return a + b
```
"""
    code2 = extract_code_from_response(response2)
    assert "def add" in code2, "Failed to extract from generic code block"
    print("  Test case 2 (generic code block): PASSED")

    # Test case 3: Raw code without block
    response3 = """
def multiply(x, y):
    return x * y
"""
    code3 = extract_code_from_response(response3)
    assert "def multiply" in code3, "Failed to extract raw code"
    print("  Test case 3 (raw code): PASSED")

    # Test case 4: Multiple code blocks (should get last one)
    response4 = """
First attempt:
```python
def wrong():
    pass
```

Better solution:
```python
def correct(n):
    return n * 2
```
"""
    code4 = extract_code_from_response(response4)
    assert "def correct" in code4, "Should extract last code block"
    print("  Test case 4 (multiple blocks): PASSED")

    print()
    return True


def test_evalplus_samples():
    """Test EvalPlus samples file creation and loading."""
    print("=" * 60)
    print("Test 3: EvalPlus Samples I/O")
    print("=" * 60)

    from llm_tts.datasets.human_eval_plus import (
        create_evalplus_samples,
        load_evalplus_samples,
    )

    results = [
        {
            "task_id": "HumanEval/0",
            "generated_code": "def has_close_elements(numbers, threshold): return False",
        },
        {
            "task_id": "HumanEval/1",
            "extracted_answer": "def separate_paren_groups(s): return []",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "samples.jsonl"
        create_evalplus_samples(results, str(output_path))

        assert output_path.exists(), "Samples file not created"
        print("  File creation: PASSED")

        loaded = load_evalplus_samples(str(output_path))
        assert len(loaded) == 2, f"Expected 2 samples, got {len(loaded)}"
        assert loaded[0]["task_id"] == "HumanEval/0"
        assert "solution" in loaded[0]
        print("  File loading: PASSED")

    print()
    return True


def test_config_integration():
    """Test that HumanEval+ config integrates with the evaluation framework."""
    print("=" * 60)
    print("Test 4: Config Integration")
    print("=" * 60)

    import sys

    sys.path.insert(0, "scripts")

    from omegaconf import OmegaConf

    # Import build_evaluators from run_tts_eval
    # This may fail in some environments due to missing dependencies
    try:
        from run_tts_eval import build_evaluators
    except ImportError as e:
        print(f"  Skipping config integration test due to import error: {e}")
        print("  SKIPPED (environment missing dependencies)")
        print()
        return True

    # Create mock config
    config = OmegaConf.create(
        {
            "dataset": {
                "data_name": "human_eval_plus",
                "answer_format": "code",
            },
            "strategy": {},
            "evaluation": {
                "evaluators": ["human_eval_plus"],
                "human_eval_plus": {
                    "mode": "full",
                    "timeout": 10,
                },
            },
        }
    )

    evaluators = build_evaluators(config)
    assert "human_eval_plus" in evaluators, "HumanEval+ evaluator not built"
    assert evaluators["human_eval_plus"].mode == "full"
    print("  Evaluator built from config: PASSED")

    print()
    return True


def test_evaluator_initialization():
    """Test HumanEval+ evaluator initialization."""
    print("=" * 60)
    print("Test 5: Evaluator Initialization")
    print("=" * 60)

    from llm_tts.evaluation import EvaluatorHumanEvalPlus

    # Test default initialization
    evaluator = EvaluatorHumanEvalPlus()
    assert evaluator.mode == "full", f"Expected mode='full', got '{evaluator.mode}'"
    assert evaluator.timeout == 10, f"Expected timeout=10, got {evaluator.timeout}"
    print("  Default initialization: PASSED")

    # Test custom timeout
    evaluator2 = EvaluatorHumanEvalPlus(timeout=20)
    assert evaluator2.timeout == 20
    print("  Custom timeout: PASSED")

    # Test deprecated mode warning
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        evaluator3 = EvaluatorHumanEvalPlus(mode="syntax")
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()
        assert evaluator3.mode == "full"  # Should fallback to full
        print("  Deprecated mode warning: PASSED")

    print()
    return True


def test_task_id_normalization():
    """Test task ID normalization."""
    print("=" * 60)
    print("Test 6: Task ID Normalization")
    print("=" * 60)

    from llm_tts.evaluation import EvaluatorHumanEvalPlus

    evaluator = EvaluatorHumanEvalPlus()

    # Test various input formats
    assert evaluator._normalize_task_id(0) == "HumanEval/0"
    assert evaluator._normalize_task_id("0") == "HumanEval/0"
    assert evaluator._normalize_task_id("HumanEval/0") == "HumanEval/0"
    assert evaluator._normalize_task_id(42) == "HumanEval/42"
    print("  Task ID normalization: PASSED")

    print()
    return True


def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("HumanEval+ Integration Tests")
    print("=" * 60)
    print()

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Code Extraction", test_code_extraction),
        ("EvalPlus Samples I/O", test_evalplus_samples),
        ("Config Integration", test_config_integration),
        ("Evaluator Initialization", test_evaluator_initialization),
        ("Task ID Normalization", test_task_id_normalization),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
