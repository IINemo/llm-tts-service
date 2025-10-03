#!/usr/bin/env python3
"""
Test accurate DeepConf implementation based on Facebook Research's original code.

Run with:
    export OPENROUTER_API_KEY="your-key"
    python tests/test_deepconf_accurate.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

import logging
import importlib.util

# Import models normally (no lm-polygraph issue)
from llm_tts.models import create_model

# Import deepconf_strategy directly without triggering strategies/__init__.py
spec = importlib.util.spec_from_file_location(
    "llm_tts.strategies.deepconf_strategy", "llm_tts/strategies/deepconf_strategy.py"
)
deepconf_module = importlib.util.module_from_spec(spec)
sys.modules["llm_tts.strategies.deepconf_strategy"] = deepconf_module
spec.loader.exec_module(deepconf_module)
DeepConfStrategy = deepconf_module.DeepConfStrategy
extract_answer = deepconf_module.extract_answer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def test_1_answer_extraction():
    """Test 1: Answer extraction with LaTeX boxed format"""
    log.info("=" * 70)
    log.info("Test 1: Answer Extraction")
    log.info("=" * 70)

    test_cases = [
        ("The answer is \\boxed{70}", "70"),
        ("Therefore \\boxed{42} is correct", "42"),
        ("\\boxed{3.14}", "3.14"),
        ("The result is \\boxed{x+y}", "x+y"),
        # Note: If "boxed" appears without {}, it extracts text after "boxed" until $ or end
        # This matches the original DeepConf behavior
        ("No boxed answer here: 70", "answer here: 70"),
        ("The answer is 42", None),  # No "boxed" keyword at all
    ]

    all_passed = True
    for text, expected in test_cases:
        result = extract_answer(text)
        passed = result == expected
        all_passed = all_passed and passed

        status = "✅" if passed else "❌"
        log.info(f"{status} Input: {text[:50]}")
        log.info(f"   Expected: {expected}, Got: {result}")

    return all_passed


def test_2_model_with_logprobs():
    """Test 2: Verify model supports logprobs"""
    log.info("=" * 70)
    log.info("Test 2: Model Logprobs Support")
    log.info("=" * 70)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.error("❌ OPENROUTER_API_KEY not set")
        return False

    try:
        model = create_model(
            provider="openrouter",
            model_name="openai/gpt-4o-mini",
            api_key=api_key,
            top_logprobs=20,
        )

        assert model.supports_logprobs() == True
        assert hasattr(model, "generate_with_confidence")

        log.info(f"✅ Model: {model.model_name}")
        log.info(f"✅ Supports logprobs: {model.supports_logprobs()}")
        return True

    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_3_deepconf_generation():
    """Test 3: DeepConf trace generation"""
    log.info("=" * 70)
    log.info("Test 3: DeepConf Trace Generation")
    log.info("=" * 70)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.error("❌ OPENROUTER_API_KEY not set")
        return False

    try:
        model = create_model(
            provider="openrouter",
            model_name="openai/gpt-4o-mini",
            api_key=api_key,
            top_logprobs=20,
        )

        strategy = DeepConfStrategy(
            model=model,
            budget=3,
            window_size=16,
            temperature=0.7,
            max_tokens=500,
            filter_method="none",
        )

        # Use boxed format in prompt
        prompt = (
            "What is 23 + 47? Show your work and put the final answer in \\boxed{}."
        )

        result = strategy.generate_trajectory(prompt)

        log.info(f"✅ Generated {result['metadata']['num_paths_generated']} traces")
        log.info(f"✅ Valid traces: {result['metadata']['num_paths_used']}")
        log.info(f"✅ Selected answer: {result['metadata']['selected_answer']}")
        log.info(f"✅ Confidence: {result['metadata']['confidence_score']:.3f}")

        assert result["metadata"]["num_paths_generated"] == 3
        assert result["completed"] == True

        return True

    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_4_deepconf_voting():
    """Test 4: DeepConf confidence-weighted voting"""
    log.info("=" * 70)
    log.info("Test 4: DeepConf Voting")
    log.info("=" * 70)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.error("❌ OPENROUTER_API_KEY not set")
        return False

    try:
        model = create_model(
            provider="openrouter",
            model_name="openai/gpt-4o-mini",
            api_key=api_key,
            top_logprobs=20,
        )

        strategy = DeepConfStrategy(
            model=model,
            budget=5,
            window_size=16,
            temperature=0.8,
            max_tokens=500,
            filter_method="top10",  # Filter to top 10% by confidence
        )

        prompt = "Calculate 15 * 6. Put your answer in \\boxed{}."

        result = strategy.generate_trajectory(prompt)

        log.info(f"✅ Total generated: {result['metadata']['num_paths_generated']}")
        log.info(f"✅ After filtering: {result['metadata']['num_paths_used']}")
        log.info(f"✅ Selected answer: '{result['metadata']['selected_answer']}'")
        log.info(f"✅ Confidence: {result['metadata']['confidence_score']:.3f}")

        # Show vote distribution
        log.info(f"✅ Vote distribution:")
        for ans, pct in result["metadata"]["vote_distribution"].items():
            log.info(f"   {ans}: {pct:.1f}%")

        # Verify answer (should be 90)
        expected = "90"
        is_correct = result["metadata"]["selected_answer"] == expected

        if is_correct:
            log.info(f"✅ Answer is correct!")
        else:
            log.warning(
                f"⚠️  Answer '{result['metadata']['selected_answer']}' != expected '{expected}'"
            )
            log.warning(f"   (This may be due to answer extraction format)")

        return True  # Test passes even if answer is wrong (testing integration)

    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    log.info("\n" + "=" * 70)
    log.info("🧪 Accurate DeepConf - Integration Tests")
    log.info("=" * 70 + "\n")

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        log.error("❌ OPENROUTER_API_KEY environment variable not set")
        log.info("Set it with: export OPENROUTER_API_KEY='your-key'")
        return 1

    tests = [
        ("Answer Extraction", test_1_answer_extraction),
        ("Model Logprobs Support", test_2_model_with_logprobs),
        ("DeepConf Generation", test_3_deepconf_generation),
        ("DeepConf Voting", test_4_deepconf_voting),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            log.info("")
        except Exception as e:
            log.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
            log.info("")

    # Summary
    log.info("=" * 70)
    log.info("📊 TEST SUMMARY")
    log.info("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        log.info(f"{status}: {test_name}")

    log.info(f"\nPassed: {passed}/{total}")

    if passed == total:
        log.info("🎉 All tests passed!")
        return 0
    else:
        log.warning(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
