import sys
import unittest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any, Tuple

import numpy as np

sys.path.insert(0, ".")

from llm_tts.strategies import StrategyCoTUQ
from llm_tts.step_candidate_generator_base import StepCandidate


class MockBlackboxModelWithStreaming:
    """Mock model that supports logprobs for testing CoT-UQ strategy."""
    
    def __init__(self, responses: List[Tuple[str, List[Tuple[str, float]]]]):
        self.supports_logprobs = True
        self.responses = responses
        self.call_count = 0
    
    def generate_with_logprobs(self, request, temperature=0.7, top_p=0.95, max_tokens=512, top_logprobs=10):
        """Mock generation with logprobs."""
        if not self.responses:
            response = Mock()
            response.text = ""
            response.token_probs = []
            return response
        if self.call_count >= len(self.responses):
            # Cycle through responses if we run out
            self.call_count = 0
        
        text, token_probs = self.responses[self.call_count]
        self.call_count += 1
        
        # Create a mock response object
        response = Mock()
        response.text = text
        response.token_probs = token_probs
        return response


class DummyStepGenerator:
    """Minimal step generator that appends a fixed answer."""

    def __init__(self, answer_text: str):
        self.answer_text = answer_text
        self.called = 0

    def generate_answer_candidates(
        self,
        request,
        trajectory,
        candidates_per_step,
    ):
        self.called += 1
        return [
            StepCandidate(
                text=f"\n<Answer>: {self.answer_text}",
                token_ids=[],
                is_complete=True,
                is_trajectory_complete=True,
                generation_scores=None,
                raw_text=f"\n<Answer>: {self.answer_text}",
            )
        ]


def create_request(question: str) -> List[Dict[str, str]]:
    """Create a test request with the standard prompt template."""
    prompt_template = """You will be presented with a <Question>. Before providing the [Answer], you should first think step-by-step carefully.

Your response format:
<start of response>
Reasoning Steps:
- Step 1: Your reasoning step 1
- Step 2: Your reasoning step 2
- Step 3: Your reasoning step 3
...
- Step N: Your reasoning step N
<Answer>: Your final answer
<end of response>

Follow the above output format STRICTLY! Do not add any other additional texts outside the template.
Keep each reasoning step concise (single steps should not be too long).
Each reasoning step must be on a single line (no line breaks within a step).

Now answer:
<Question>: {question}"""

    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt_template.format(question=question)},
    ]


class TestStrategyCoTUQ(unittest.TestCase):
    """Test cases for StrategyCoTUQ."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.question = "Tom had 8 apples. He gave 3 to his friend and bought 5 more. How many apples does Tom have now?"
        self.request = create_request(self.question)
        
        # Mock responses with different quality levels
        self.mock_responses = [
            # High quality response
            (
                "Reasoning Steps:\n- Step 1: Tom starts with 8 apples\n- Step 2: He gives away 3 apples, so he has 8 - 3 = 5 apples\n- Step 3: He buys 5 more apples, so he has 5 + 5 = 10 apples\n<Answer>: 10",
                [("Tom", 0.9), ("starts", 0.8), ("with", 0.9), ("8", 0.95), ("apples", 0.9), ("10", 0.95)]
            ),
            # Medium quality response
            (
                "Reasoning Steps:\n- Step 1: Tom has 8 apples initially\n- Step 2: After giving 3 away, he has 5 apples\n- Step 3: Adding 5 more gives him 10 apples total\n<Answer>: 10",
                [("Tom", 0.7), ("has", 0.6), ("8", 0.8), ("apples", 0.7), ("10", 0.85)]
            ),
            # Lower quality response
            (
                "Reasoning Steps:\n- Step 1: 8 - 3 = 5\n- Step 2: 5 + 5 = 10\n<Answer>: 10",
                [("8", 0.6), ("-", 0.5), ("3", 0.6), ("=", 0.5), ("5", 0.7), ("10", 0.8)]
            )
        ]
    
    def test_strategy_initialization(self):
        """Test that StrategyCoTUQ initializes correctly."""
        model = MockBlackboxModelWithStreaming(self.mock_responses)
        
        strategy = StrategyCoTUQ(
            model=model,
            budget=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200,
            top_logprobs=10,
            alpha=0.5
        )
        
        self.assertEqual(strategy.budget, 3)
        self.assertEqual(strategy.temperature, 0.7)
        self.assertEqual(strategy.top_p, 0.95)
        self.assertEqual(strategy.max_tokens, 200)
        self.assertEqual(strategy.top_logprobs, 10)
        self.assertEqual(strategy.alpha, 0.5)
        self.assertEqual(strategy.model, model)
    
    def test_strategy_initialization_without_logprobs_support(self):
        """Test that StrategyCoTUQ raises error for models without logprobs support."""
        model = Mock()
        model.supports_logprobs = False
        
        with self.assertRaises(ValueError) as context:
            StrategyCoTUQ(
                model=model,
                budget=3,
                temperature=0.7,
                top_p=0.95,
                max_tokens=200
            )
        
        self.assertIn("requires a model with logprobs support", str(context.exception))
    
    def test_extract_answer_span(self):
        """Test answer span extraction."""
        # Test with explicit answer marker
        text1 = "Reasoning Steps:\n- Step 1: Some reasoning\n<Answer>: 10"
        start, end = StrategyCoTUQ._extract_answer_span(text1)
        self.assertEqual(text1[start:end].strip(), "10")
        
        # Test fallback to last line
        text2 = "Some reasoning without answer marker\nFinal answer: 10"
        start, end = StrategyCoTUQ._extract_answer_span(text2)
        self.assertEqual(text2[start:end].strip(), "10")
    
    def test_aggregate_answer_prob(self):
        """Test probability aggregation over answer tokens."""
        token_probs = [("token1", 0.8), ("token2", 0.9), ("token3", 0.7), ("token4", 0.85)]
        answer_text = "token3 token4"
        
        prob = StrategyCoTUQ._aggregate_answer_prob(token_probs, answer_text)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        
        # Test with empty token_probs
        prob_empty = StrategyCoTUQ._aggregate_answer_prob([], "answer")
        self.assertEqual(prob_empty, 0.5)
    
    def test_compute_reasoning_importance(self):
        """Test reasoning importance computation."""
        # High importance text (lots of numbers and operations)
        high_importance = "Calculate 8 - 3 + 5 = 10 with 2 operations"
        score1 = StrategyCoTUQ._compute_reasoning_importance(high_importance)
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)
        
        # Low importance text (no numbers or operations)
        low_importance = "This is just some text without numbers or operations"
        score2 = StrategyCoTUQ._compute_reasoning_importance(low_importance)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 1.0)
        
        # High importance should be greater than low importance
        self.assertGreater(score1, score2)
    
    def test_score_trace(self):
        """Test trace scoring functionality."""
        model = MockBlackboxModelWithStreaming(self.mock_responses)
        strategy = StrategyCoTUQ(
            model=model,
            budget=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200,
            alpha=0.5
        )
        
        text = "Reasoning Steps:\n- Step 1: 8 - 3 = 5\n- Step 2: 5 + 5 = 10\n<Answer>: 10"
        token_probs = [("8", 0.8), ("-", 0.7), ("3", 0.8), ("=", 0.6), ("5", 0.9), ("10", 0.95)]
        
        score, answer, evidence = strategy._score_trace(text, token_probs)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(answer.strip(), "10")
        self.assertIsInstance(evidence, dict)
        self.assertIn("answer", evidence)
        self.assertIn("steps", evidence)
    
    def test_generate_trajectory_success(self):
        """Test successful trajectory generation."""
        model = MockBlackboxModelWithStreaming(self.mock_responses)
        strategy = StrategyCoTUQ(
            model=model,
            budget=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200,
            alpha=0.5
        )
        
        result = strategy.generate_trajectory(self.request)
        
        # Verify result structure
        self.assertIn("trajectory", result)
        self.assertIn("steps", result)
        self.assertIn("validity_scores", result)
        self.assertIn("completed", result)
        
        # Verify result content
        self.assertTrue(result["completed"])
        self.assertEqual(len(result["validity_scores"]), 3)
        self.assertEqual(len(result["steps"]), 1)
        self.assertIsInstance(result["steps"][0], StepCandidate)
        self.assertIn("metadata", result)
        self.assertIn("cot_uq", result["metadata"])
        self.assertIn("best_evidence", result["metadata"]["cot_uq"])

        # Verify step candidate properties
        step = result["steps"][0]
        self.assertTrue(step.is_complete)
        self.assertTrue(step.is_trajectory_complete)
        self.assertIn("answer", step.other_data)
        self.assertIn("cot_uq_score", step.other_data)
        self.assertIn("cot_uq_evidence", step.other_data)

    def test_force_answer_with_step_generator(self):
        """Ensure missing answers trigger fallback generation."""
        incomplete_trace = [
            (
                "Reasoning Steps:\n- Step 1: Compute 2 + 2 = 4\n- Step 2: We need to report the answer.\n<Answer>:",
                [("Reasoning", 0.8), ("Steps", 0.7)],
            )
        ]
        model = MockBlackboxModelWithStreaming(incomplete_trace)
        step_generator = DummyStepGenerator("4")
        strategy = StrategyCoTUQ(
            model=model,
            budget=1,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200,
            step_generator=step_generator,
        )

        result = strategy.generate_trajectory(self.request)

        self.assertTrue(result["completed"])
        self.assertGreater(step_generator.called, 0)
        step = result["steps"][0]
        self.assertIn("4", step.other_data["answer"])
        self.assertIn("cot_uq_evidence", step.other_data)
        answer_text = step.other_data["cot_uq_evidence"]["answer"]["clean_text"]
        self.assertEqual(answer_text.strip(), "4")

    def test_generate_trajectory_empty_traces(self):
        """Test trajectory generation with empty traces."""
        model = MockBlackboxModelWithStreaming([])  # No responses
        strategy = StrategyCoTUQ(
            model=model,
            budget=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200
        )
        
        result = strategy.generate_trajectory(self.request)
        
        # Should return empty result
        self.assertFalse(result["completed"])
        self.assertEqual(result["trajectory"], "")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["validity_scores"], [])
    
    def test_generate_trajectory_single_trace(self):
        """Test trajectory generation with single trace."""
        model = MockBlackboxModelWithStreaming([self.mock_responses[0]])  # Single response
        strategy = StrategyCoTUQ(
            model=model,
            budget=1,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200
        )
        
        result = strategy.generate_trajectory(self.request)
        
        self.assertTrue(result["completed"])
        self.assertEqual(len(result["validity_scores"]), 1)
        self.assertEqual(len(result["steps"]), 1)
    
    def test_cleanup(self):
        """Test cleanup method."""
        model = MockBlackboxModelWithStreaming(self.mock_responses)
        strategy = StrategyCoTUQ(
            model=model,
            budget=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200
        )
        
        # Should not raise any exceptions
        strategy.cleanup()
    
    def test_alpha_parameter_effect(self):
        """Test that alpha parameter affects scoring."""
        model = MockBlackboxModelWithStreaming(self.mock_responses)
        
        # Test with alpha=0 (only CoT importance)
        strategy_cot_only = StrategyCoTUQ(
            model=model,
            budget=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200,
            alpha=0.0
        )
        
        # Test with alpha=1 (only probability confidence)
        strategy_prob_only = StrategyCoTUQ(
            model=model,
            budget=3,
            temperature=0.7,
            top_p=0.95,
            max_tokens=200,
            alpha=1.0
        )
        
        # Both should work without errors
        result1 = strategy_cot_only.generate_trajectory(self.request)
        result2 = strategy_prob_only.generate_trajectory(self.request)
        
        self.assertTrue(result1["completed"])
        self.assertTrue(result2["completed"])


def test_cot_uq_integration():
    """Integration test for CoT-UQ strategy (requires actual model)."""
    # This test would require a real model with logprobs support
    # For now, we'll just verify the test structure
    print("CoT-UQ integration test would require a real model with logprobs support")
    print("Mock-based tests are sufficient for unit testing")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)
