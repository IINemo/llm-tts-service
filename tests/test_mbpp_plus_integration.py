#!/usr/bin/env python3
"""
Integration tests for MBPP+ support.

This test validates the full pipeline:
1. Dataset loading
2. Code extraction
3. Evaluator scoring
4. Config parsing

Run: python tests/test_mbpp_plus_integration.py
"""

import sys
import tempfile

# Suppress ANTLR warnings
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="ANTLR runtime")


def test_dataset_loading():
    """Test loading MBPP+ dataset."""
    print("=" * 60)
    print("Test 1: Dataset Loading")
    print("=" * 60)

    from llm_tts.datasets import load_mbpp_plus

    # Load small subset
    data = load_mbpp_plus(subset_size=5)

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

    from llm_tts.datasets import extract_code_from_response

    # Test case 1: Code in Python code block
    response1 = """
Here's the solution:

```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
```

This function finds common elements.
"""
    code1 = extract_code_from_response(response1)
    assert "def similar_elements" in code1, "Failed to extract from code block"
    assert "return tuple" in code1, "Missing function body"
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


def test_evaluator():
    """Test MBPP+ evaluator."""
    print("=" * 60)
    print("Test 3: Evaluator")
    print("=" * 60)

    from llm_tts.evaluation import EvaluatorMBPPPlus

    evaluator = EvaluatorMBPPPlus(mode="syntax")

    # Test valid Python
    valid_code = """
```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
```
"""
    scores = evaluator(["prompt"], [valid_code], ["answer"])
    assert scores[0] == 1.0, f"Expected 1.0 for valid Python, got {scores[0]}"
    print("  Valid Python score: PASSED")

    # Test invalid Python (syntax error)
    invalid_code = """
def broken_function(
    return 42
"""
    scores = evaluator(["prompt"], [invalid_code], ["answer"])
    assert scores[0] == 0.0, f"Expected 0.0 for invalid Python, got {scores[0]}"
    print("  Invalid Python score: PASSED")

    # Test empty response
    scores = evaluator(["prompt"], [""], ["answer"])
    assert scores[0] == 0.0, f"Expected 0.0 for empty response, got {scores[0]}"
    print("  Empty response score: PASSED")

    # Test batch evaluation
    responses = [valid_code, invalid_code, valid_code]
    scores = evaluator(["p"] * 3, responses, ["a"] * 3)
    assert scores == [1.0, 0.0, 1.0], f"Unexpected batch scores: {scores}"
    print("  Batch evaluation: PASSED")

    print()
    return True


def test_evalplus_samples():
    """Test EvalPlus samples file creation and loading."""
    print("=" * 60)
    print("Test 4: EvalPlus Samples I/O")
    print("=" * 60)

    from llm_tts.datasets import create_evalplus_samples, load_evalplus_samples

    results = [
        {
            "task_id": "Mbpp/2",
            "generated_code": "def similar_elements(a, b): return tuple(set(a) & set(b))",
        },
        {
            "task_id": "Mbpp/3",
            "extracted_answer": "def is_palindrome(s): return s == s[::-1]",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "samples.jsonl"
        create_evalplus_samples(results, str(output_path))

        assert output_path.exists(), "Samples file not created"
        print("  File creation: PASSED")

        loaded = load_evalplus_samples(str(output_path))
        assert len(loaded) == 2, f"Expected 2 samples, got {len(loaded)}"
        assert loaded[0]["task_id"] == "Mbpp/2"
        assert "solution" in loaded[0]
        print("  File loading: PASSED")

    print()
    return True


def test_config_integration():
    """Test that MBPP+ config integrates with the evaluation framework."""
    print("=" * 60)
    print("Test 5: Config Integration")
    print("=" * 60)

    import sys

    sys.path.insert(0, "scripts")

    from omegaconf import OmegaConf

    # Import build_evaluators from run_tts_eval
    from run_tts_eval import build_evaluators

    # Create mock config
    config = OmegaConf.create(
        {
            "dataset": {
                "data_name": "mbpp_plus",
                "answer_format": "code",
            },
            "strategy": {},
            "evaluation": {
                "evaluators": ["mbpp_plus"],
                "mbpp_plus": {
                    "mode": "syntax",
                    "timeout": 10,
                },
            },
        }
    )

    evaluators = build_evaluators(config)
    assert "mbpp_plus" in evaluators, "MBPP+ evaluator not built"
    assert evaluators["mbpp_plus"].mode == "syntax"
    print("  Evaluator built from config: PASSED")

    print()
    return True


def test_detailed_results():
    """Test detailed results generation."""
    print("=" * 60)
    print("Test 6: Detailed Results")
    print("=" * 60)

    from llm_tts.evaluation import EvaluatorMBPPPlus

    evaluator = EvaluatorMBPPPlus(mode="syntax")

    solutions = [
        """
```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
```
""",
        "Not valid Python code here {{{",
    ]

    instance_data = [
        {
            "task_id": "Mbpp/2",
            "entry_point": "similar_elements",
            "question": "Write a function to find similar elements.",
            "answer": "def similar_elements(a, b): return tuple(set(a) & set(b))",
        },
        {
            "task_id": "Mbpp/3",
            "entry_point": "is_not_prime",
            "question": "Write a function.",
            "answer": "def is_not_prime(n): return n < 2",
        },
    ]

    results = evaluator.get_detailed_results(solutions, instance_data)

    assert len(results) == 2
    assert results[0]["is_valid_python"] is True
    assert results[0]["has_function_definition"] is True
    assert results[1]["is_valid_python"] is False
    print("  Detailed results generation: PASSED")

    print()
    return True


def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("MBPP+ Integration Tests")
    print("=" * 60)
    print()

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Code Extraction", test_code_extraction),
        ("Evaluator", test_evaluator),
        ("EvalPlus Samples I/O", test_evalplus_samples),
        ("Config Integration", test_config_integration),
        ("Detailed Results", test_detailed_results),
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
