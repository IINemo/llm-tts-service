from llm_tts.evaluation.exact_match import EvaluatorExactMatch


class TestEvaluatorExactMatch:
    """Test suite for exact match evaluator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = EvaluatorExactMatch()

    def test_exact_numeric_match(self):
        problems = ["What is 2+2?"]
        solutions = ["4"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_numeric_mismatch(self):
        problems = ["What is 2+2?"]
        solutions = ["5"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 0.0

    def test_answer_block_extraction(self):
        """Test extraction from <Answer>: blocks."""
        problems = ["What is 2+2?"]
        solutions = ["Let me think... <Answer>: 4"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_boxed_extraction(self):
        """Test extraction from \\boxed{} blocks."""
        problems = ["What is 2+2?"]
        solutions = ["The answer is \\boxed{4}"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_inline_math_extraction(self):
        """Test extraction from inline math."""
        problems = ["What is 2+2?"]
        solutions = ["The answer is $4$"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_comma_separated_numbers(self):
        """Test numbers with commas."""
        problems = ["What is 1,000 + 1,000?"]
        solutions = ["2,000"]
        gold_answers = ["2000"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_decimal_equivalence(self):
        """Test decimal equivalence."""
        problems = ["What is 0.5 + 0.5?"]
        solutions = ["1.0"]
        gold_answers = ["1"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_latex_math(self):
        """Test LaTeX mathematical expressions."""
        problems = ["What is 2+2?"]
        solutions = ["The answer is \\[4\\]"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_multiple_candidates(self):
        """Test batch evaluation with multiple problems."""
        problems = ["What is 2+2?", "What is 3+3?"]
        solutions = ["4", "6"]
        gold_answers = ["4", "6"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert all(score == 1.0 for score in scores)

    def test_mixed_correctness(self):
        """Test batch with mixed correct/incorrect answers."""
        problems = ["What is 2+2?", "What is 3+3?"]
        solutions = ["4", "7"]
        gold_answers = ["4", "6"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0
        assert scores[1] == 0.0

    def test_empty_solution(self):
        problems = ["What is 2+2?"]
        solutions = [""]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 0.0

    def test_no_numeric_answer(self):
        """Test solution with no numeric answer."""
        problems = ["What is 2+2?"]
        solutions = ["I don't know"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 0.0

    def test_terminator_handling(self):
        """Test handling of response terminators."""
        problems = ["What is 2+2?"]
        solutions = ["<Answer>: 4 <end of response>"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_nested_boxed(self):
        """Test nested \\boxed{} expressions."""
        problems = ["What is 2+2?"]
        solutions = ["The answer is \\boxed{\\boxed{4}}"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_negative_numbers(self):
        """Test negative number handling."""
        problems = ["What is 2-4?"]
        solutions = ["-2"]
        gold_answers = ["-2"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_scientific_notation(self):
        """Test scientific notation handling."""
        problems = ["What is 1e3?"]
        solutions = ["1000"]
        gold_answers = ["1000"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_robust_grading_fallback(self):
        problems = ["What is 1/2 + 1/4?"]
        solutions = ["3/4"]
        gold_answers = ["0.75"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_fraction_equivalence(self):
        problems = ["What is 1/2 + 1/4?"]
        solutions = ["frac{14}{3}"]
        gold_answers = ["14/3"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_invalid_numeric_conversion(self):
        """Test handling of invalid numeric conversions."""
        problems = ["What is 2+2?"]
        solutions = ["abc"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 0.0

    def test_gold_answer_extraction(self):
        """Test that gold answer extraction works."""
        problems = ["What is 2+2?"]
        solutions = ["4"]
        gold_answers = ["\\boxed{4}"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 1.0

    def test_single_evaluation(self):
        """Test single evaluation method."""
        problem = "What is 2+2?"
        solution = "4"
        gold_answer = "4"

        score = self.evaluator._score_single((problem, solution, gold_answer))
        assert score == 1.0

    def test_nan_handling(self):
        """Test NaN handling for invalid inputs."""
        problems = ["What is 2+2?"]
        solutions = ["invalid"]
        gold_answers = ["4"]

        scores = self.evaluator(problems, solutions, gold_answers)
        assert scores[0] == 0.0
