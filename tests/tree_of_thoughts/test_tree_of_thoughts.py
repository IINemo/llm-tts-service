"""
Tests for Tree-of-Thoughts strategy.

These tests verify the ToT implementation works correctly with various configurations.
"""

import os
import sys

import pytest

sys.path.insert(0, ".")  # noqa: E402

from llm_tts.models.blackboxmodel_with_streaming import (  # noqa: E402
    BlackboxModelWithStreaming,
)
from llm_tts.strategies import StrategyTreeOfThoughts  # noqa: E402


@pytest.fixture
def api_key():
    """Get API key from environment."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture
def test_model(api_key):
    """Create a test model with API access."""
    model = BlackboxModelWithStreaming(
        openai_api_key=api_key,
        model_path="openai/gpt-4o-mini",
        supports_logprobs=False,  # Not needed for ToT
        base_url="https://openrouter.ai/api/v1",
    )
    return model


def test_tot_import():
    """Test that ToT strategy can be imported."""
    assert StrategyTreeOfThoughts is not None


def test_tot_initialization(test_model):
    """Test ToT strategy initialization with various configurations."""
    # Default configuration
    strategy = StrategyTreeOfThoughts(
        model=test_model,
        beam_width=3,
        steps=2,
    )
    assert strategy.beam_width == 3
    assert strategy.steps == 2
    assert strategy.method_generate == "propose"
    assert strategy.scorer is not None  # Default scorer created

    # Custom configuration with explicit scorer
    from llm_tts.scorers.tree_of_thoughts import TotVoteScorer

    vote_scorer = TotVoteScorer(
        model=test_model,
        n_evaluate_sample=5,
        temperature=0.5,
    )

    strategy = StrategyTreeOfThoughts(
        model=test_model,
        scorer=vote_scorer,
        method_generate="sample",
        beam_width=5,
        n_generate_sample=7,
        steps=4,
        temperature=0.8,
    )
    assert strategy.method_generate == "sample"
    assert strategy.beam_width == 5
    assert strategy.n_generate_sample == 7
    assert strategy.temperature == 0.8
    assert strategy.scorer == vote_scorer


def test_tot_answer_extraction(test_model):
    """Test answer extraction from various formats."""
    strategy = StrategyTreeOfThoughts(model=test_model, beam_width=2, steps=1)

    # Test \\boxed{} format
    assert strategy._extract_answer("The answer is \\boxed{42}") == "42"

    # Test Answer: format
    assert strategy._extract_answer("Answer: 15") == "15"

    # Test = format
    assert strategy._extract_answer("x = 27") == "27"

    # Test last number fallback
    assert strategy._extract_answer("After calculation, we get 99") == "99"

    # Test no answer
    assert strategy._extract_answer("No numbers here") == "no_answer"


def test_tot_final_answer_detection(test_model):
    """Test detection of final answers."""
    # Test generic mode (default)
    strategy = StrategyTreeOfThoughts(model=test_model, beam_width=2, steps=1)

    # Should detect these as final answers in generic mode
    assert strategy._is_final_answer("final answer: 42")
    assert strategy._is_final_answer("Therefore, the result is 15")
    assert strategy._is_final_answer("In conclusion, x = 27")
    assert strategy._is_final_answer("The answer is: 42")

    # Should not detect these as final answers
    assert not strategy._is_final_answer("Step 1: Add the numbers")
    assert not strategy._is_final_answer("Let's think about this")

    # Test game24 mode
    strategy_game24 = StrategyTreeOfThoughts(
        model=test_model, beam_width=2, steps=1, mode="game24"
    )
    assert strategy_game24._is_final_answer("Answer: (4 + 8) * (6 - 4) = 24")
    assert strategy_game24._is_final_answer("Steps:\n4 + 8 = 12\n(left: 24)")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping API test",
)
def test_tot_simple_math_problem(test_model):
    """Test ToT on a simple math problem with API calls."""
    from llm_tts.scorers.tree_of_thoughts import TotValueScorer

    # Create scorer with specific configuration
    value_scorer = TotValueScorer(
        model=test_model,
        n_evaluate_sample=2,
        temperature=0.0,
    )

    strategy = StrategyTreeOfThoughts(
        model=test_model,
        scorer=value_scorer,
        method_generate="propose",
        beam_width=3,
        n_generate_sample=3,
        steps=3,
        temperature=0.7,
    )

    question = "If Tom has 5 apples and buys 3 more, how many apples does he have?"
    prompt = f"Solve this math problem step by step:\n\n{question}"

    result = strategy.generate_trajectory(prompt)

    # Verify result structure
    assert "trajectory" in result
    assert "steps" in result
    assert "validity_scores" in result
    assert "completed" in result
    assert "metadata" in result
    assert "strategy" in result

    assert result["strategy"] == "tree_of_thoughts"
    assert isinstance(result["steps"], list)
    assert isinstance(result["validity_scores"], list)
    assert isinstance(result["completed"], bool)

    # Verify metadata
    metadata = result["metadata"]
    assert "strategy" in metadata
    assert "config" in metadata
    assert "results" in metadata
    assert metadata["strategy"] == "tree_of_thoughts"

    # Check API call tracking
    assert strategy.total_api_calls > 0
    print(f"\nAPI calls made: {strategy.total_api_calls}")
    print(f"Scorer evaluations: {strategy.scorer.total_evaluations}")
    print(f"Extracted answer: {metadata['results'].get('selected_answer', 'N/A')}")
    print(f"Best score: {metadata['results'].get('best_score', 'N/A')}")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping API test",
)
def test_tot_with_message_format(test_model):
    """Test ToT with chat message format."""
    strategy = StrategyTreeOfThoughts(
        model=test_model,
        beam_width=2,
        steps=2,
        n_generate_sample=2,
    )

    messages = [{"role": "user", "content": "What is 7 + 8?"}]

    result = strategy.generate_trajectory(messages)

    assert result["strategy"] == "tree_of_thoughts"
    assert "trajectory" in result
    print(f"\nFinal trajectory: {result['trajectory'][:200]}...")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping API test",
)
def test_tot_sample_method(test_model):
    """Test ToT with sample generation method."""
    strategy = StrategyTreeOfThoughts(
        model=test_model,
        method_generate="sample",  # Use sampling instead of proposals
        beam_width=2,
        steps=2,
        n_generate_sample=2,
    )

    question = "Calculate 12 * 3"
    result = strategy.generate_trajectory(question)

    assert result["strategy"] == "tree_of_thoughts"
    assert result["completed"] is not None
    print(
        f"\nSample method result: {result['metadata']['results'].get('selected_answer')}"
    )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
