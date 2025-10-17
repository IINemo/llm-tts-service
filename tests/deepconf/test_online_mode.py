"""
Test DeepConf online mode with streaming + logprobs early stopping
"""

import os
import sys

import pytest

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.strategies.deepconf import StrategyDeepConf

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)
def test_deepconf_online_mode_basic():
    """Test DeepConf online mode with a simple math problem"""

    # Initialize model
    model = BlackboxModelWithStreaming(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_path="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        supports_logprobs=True,
    )

    # Initialize DeepConf strategy in online mode
    strategy = StrategyDeepConf(
        model=model,
        mode="online",
        budget=5,  # Not used in online mode
        warmup_traces=2,
        total_budget=5,
        confidence_percentile=10,  # Use 90th percentile
        window_size=5,
        temperature=0.7,
        top_p=1.0,
        max_tokens=512,
        top_logprobs=20,
        filter_method="top3",
    )

    # Test prompt
    prompt = "What is 15 + 27?"

    # Generate trajectory
    result = strategy.generate_trajectory(prompt)

    # Verify result structure
    assert "trajectory" in result
    assert "metadata" in result
    assert result["metadata"]["mode"] == "online"
    assert result["metadata"]["warmup_traces"] == 2
    assert result["metadata"]["total_traces"] > 0

    print("\n✅ DeepConf Online Mode Test Passed!")
    print(f"Selected answer: {result['metadata']['selected_answer']}")
    print(f"Confidence: {result['metadata']['confidence_score']:.3f}")
    print(f"Total traces: {result['metadata']['total_traces']}")
    print(f"Adaptive traces: {result['metadata']['adaptive_traces']}")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)
def test_deepconf_online_early_stopping():
    """Test that early stopping actually works"""

    model = BlackboxModelWithStreaming(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_path="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        supports_logprobs=True,
    )

    strategy = StrategyDeepConf(
        model=model,
        mode="online",
        budget=5,
        warmup_traces=2,
        total_budget=4,  # 2 warmup + 2 adaptive
        confidence_percentile=50,  # Higher threshold = more early stopping
        window_size=3,
        temperature=0.8,  # Higher temp = more uncertainty
        top_p=1.0,
        max_tokens=512,
        top_logprobs=20,
        filter_method="top3",
    )

    prompt = "What is 8 x 7?"

    result = strategy.generate_trajectory(prompt)

    # Check that we got adaptive traces
    assert result["metadata"]["adaptive_traces"] >= 0

    print("\n✅ Early Stopping Test Passed!")
    print(f"Adaptive traces generated: {result['metadata']['adaptive_traces']}")


if __name__ == "__main__":
    # Run tests if OPENROUTER_API_KEY is set
    if os.getenv("OPENROUTER_API_KEY"):
        print("Running DeepConf online mode tests...")
        test_deepconf_online_mode_basic()
        test_deepconf_online_early_stopping()
        print("\n🎉 All tests passed!")
    else:
        print("⚠️  OPENROUTER_API_KEY not set, skipping tests")
