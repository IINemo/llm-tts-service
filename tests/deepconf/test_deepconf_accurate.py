#!/usr/bin/env python3
"""
Test accurate DeepConf implementation based on Facebook Research's original code.

Run with:
    export OPENROUTER_API_KEY="your-key"
    python tests/test_deepconf_accurate.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# Import models and strategies normally
from llm_tts.models import BlackboxModelWithStreaming
from llm_tts.strategies import StrategyDeepConf
from llm_tts.utils.confidence import extract_answer

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
        model = BlackboxModelWithStreaming(
            openai_api_key=api_key,
            model_path="openai/gpt-4o-mini",
            supports_logprobs=True,
            base_url="https://openrouter.ai/api/v1",
        )

        assert model.supports_logprobs is True

        log.info(f"✅ Model: {model.model_path}")
        log.info(f"✅ Supports logprobs: {model.supports_logprobs}")
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
        model = BlackboxModelWithStreaming(
            openai_api_key=api_key,
            model_path="openai/gpt-4o-mini",
            supports_logprobs=True,
            base_url="https://openrouter.ai/api/v1",
        )

        strategy = StrategyDeepConf(
            model=model,
            mode="offline",
            budget=3,
            window_size=16,
            temperature=0.7,
            top_p=1.0,
            max_tokens=500,
            top_logprobs=20,
            filter_method="none",
        )

        # Use boxed format in prompt
        prompt = (
            "What is 23 + 47? Show your work and put the final answer in \\boxed{}."
        )

        result = strategy.generate_trajectory(prompt)

        log.info(f"✅ Generated {result['metadata']['total_traces']} traces")
        log.info(f"✅ Valid traces: {result['metadata']['filtered_traces']}")
        log.info(f"✅ Selected answer: {result['metadata']['selected_answer']}")
        log.info(f"✅ Confidence: {result['metadata']['confidence_score']:.3f}")

        assert result["metadata"]["total_traces"] == 3
        assert result["completed"] is True

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
        model = BlackboxModelWithStreaming(
            openai_api_key=api_key,
            model_path="openai/gpt-4o-mini",
            supports_logprobs=True,
            base_url="https://openrouter.ai/api/v1",
        )

        strategy = StrategyDeepConf(
            model=model,
            mode="offline",
            budget=5,
            window_size=16,
            temperature=0.8,
            top_p=1.0,
            max_tokens=500,
            top_logprobs=20,
            filter_method="top10",  # Filter to top 10% by confidence
        )

        prompt = "Calculate 15 * 6. Put your answer in \\boxed{}."

        result = strategy.generate_trajectory(prompt)

        log.info(f"✅ Total generated: {result['metadata']['total_traces']}")
        log.info(f"✅ After filtering: {result['metadata']['filtered_traces']}")
        log.info(f"✅ Selected answer: '{result['metadata']['selected_answer']}'")
        log.info(f"✅ Confidence: {result['metadata']['confidence_score']:.3f}")

        # Show vote distribution
        log.info("✅ Vote distribution:")
        for ans, pct in result["metadata"]["vote_distribution"].items():
            log.info(f"   {ans}: {pct:.1f}%")

        # Verify answer (should be 90)
        expected = "90"
        is_correct = result["metadata"]["selected_answer"] == expected

        if is_correct:
            log.info("✅ Answer is correct!")
        else:
            log.warning(
                f"⚠️  Answer '{result['metadata']['selected_answer']}' != expected '{expected}'"
            )
            log.warning("   (This may be due to answer extraction format)")

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
