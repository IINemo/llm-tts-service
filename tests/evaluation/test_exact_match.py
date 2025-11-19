from llm_tts.evaluation.exact_match import EvaluatorExactMatch


class TestEvaluatorExactMatch:
    """Test suite for exact match evaluator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = EvaluatorExactMatch(dataset_answer_format="numeric")

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

    def test_boolean_answers(self):
        problem = "Did the Hopi Indians use a symbol that was similar to the swastika?"
        solution = "true"
        gold_answer = ["True", "true"]

        score1 = self.evaluator._score_single((problem, solution, gold_answer[0]))
        score2 = self.evaluator._score_single((problem, solution, gold_answer[1]))
        assert score1 == 1.0
        assert score2 == 1.0

    def test_boolean_answers_advanced(self):
        problem = "Did the Hopi Indians use a symbol that was similar to the swastika?"
        solution_true = ["<Answer>: True", "<Answer>: true"]
        solution_false = ["<Answer>: False", "<Answer>: false"]

        for st in solution_true:
            score = self.evaluator._score_single((problem, st, "true"))
            assert score == 1.0
        for sf in solution_false:
            score = self.evaluator._score_single((problem, sf, "false"))
            assert score == 1.0
        for st in solution_true:
            score = self.evaluator._score_single((problem, st, "false"))
            assert score == 0.0
        for sf in solution_false:
            score = self.evaluator._score_single((problem, sf, "true"))
            assert score == 0.0

    def test_boolean_answers_parsed(self):
        problem = [
            "Did the Hopi Indians use a symbol that was similar to the swastika?"
        ]
        solution_true = ["The final answer is True", "The final answer is true"]
        solution_false = ["The final answer is False", "The final answer is false"]

        boolean_evaluator = EvaluatorExactMatch(dataset_answer_format="boolean")

        for st in solution_true:
            score = boolean_evaluator._score_single((problem, st, "true"))
            assert score == 1.0
        for sf in solution_false:
            score = boolean_evaluator._score_single((problem, sf, "false"))
            assert score == 1.0
        for st in solution_true:
            score = boolean_evaluator._score_single((problem, st, "false"))
            assert score == 0.0
        for sf in solution_false:
            score = boolean_evaluator._score_single((problem, sf, "true"))
            assert score == 0.0

    def test_char_answers(self):
        problem = "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Options: \nA: ignore\nB: enforce\nC: authoritarian\nD: yell at\nE: avoid"
        solution = "A"
        gold_answer = "A"

        char_evaluator = EvaluatorExactMatch(dataset_answer_format="char")

        score = char_evaluator._score_single((problem, solution, gold_answer))
        assert score == 1.0

    def test_char_answers_advanced(self):
        problem = [
            "Did the Hopi Indians use a symbol that was similar to the swastika?"
        ]
        solution_true = ["<Answer>: A", "<Answer>: A"]
        solution_false = ["<Answer>: C", "<Answer>: C"]

        char_evaluator = EvaluatorExactMatch(dataset_answer_format="char")

        for st in solution_true:
            score = char_evaluator._score_single((problem, st, "A"))
            assert score == 1.0
        for sf in solution_false:
            score = char_evaluator._score_single((problem, sf, "C"))
            assert score == 1.0
        for st in solution_true:
            score = char_evaluator._score_single((problem, st, "C"))
            assert score == 0.0
        for sf in solution_false:
            score = char_evaluator._score_single((problem, sf, "A"))
            assert score == 0.0

    def test_string_answers(self):
        """Test direct string comparison format."""
        problem = "What is the capital of France?"
        solution = "<Answer>: Paris"
        gold_answer = "Paris"

        # Create evaluator with string format
        string_evaluator = EvaluatorExactMatch(dataset_answer_format="string")

        score = string_evaluator._score_single((problem, solution, gold_answer))
        assert score == 1.0

        # Test case sensitivity
        solution_upper = "PARIS"
        score = string_evaluator._score_single((problem, solution_upper, gold_answer))
        assert score == 1.0  # Should match due to normalization

        # Test mismatch
        solution_wrong = "London"
        score = string_evaluator._score_single((problem, solution_wrong, gold_answer))
        assert score == 0.0

    def test_not_parsed_in_numeric(self):
        problem = "What is the capital of France?"
        solution = "<start of response>\nReasoning Steps:\n- Step 1: Recognize that the given double sum can be rewritten by changing variables to simplify the expression.\n- Step 2: Let $ n = j + k $, then for each $ n $, count the number of pairs $ (j, k) $ such that $ j + k = n $, which is $ n - 1 $ for $ n \\geq 2 $.\n- Step 3: Rewrite the double sum as a single sum over $ n $, resulting in $ \\sum_{n=2}^\\infty \\frac{n - 1}{n^3} $.\n- Step 4: Split the fraction to separate the terms, resulting in $ \\sum_{n=2}^\\infty \\left( \\frac{1}{n^2} - \\frac{1}{n^3} \\right) $.\n- Step 5: Express the sum in terms of $ p $ and $ q $ by subtracting the first term of each series, yielding $ p - 1 - (q - 1) $.\n- Step 6: Simplify the expression to get $ p - q $.\n<Answer>:\n<Answer>:\n$ p - q $"
        gold_answer = "p - q"

        score = self.evaluator._score_single((problem, solution, gold_answer))
        assert score == 1.0

    def test_parsed_in_numeric(self):
        problem = "What is the capital of France?"
        solution = "<Answer>:\n$ p - q $"
        gold_answer = "p - q"

        score = self.evaluator._score_single((problem, solution, gold_answer))
        assert score == 1.0

    def test_string_not_parsed_in_numeric(self):
        problem = "What is the capital of France?"
        solution = "<start of response>\nReasoning Steps:\n- Step 1: The graph shows distance on the y-axis and time on the x-axis.\n- Step 2: Average speed is calculated as distance divided by time.\n- Step 3: To find the average speed, we need to determine the distance and time for each student.\n- Step 4: For Evelyn, the distance is 4.5 units and the time is 1.25 units.\n- Step 5: For Briana, the distance is 2.2 units and the time is 2.5 units.\n- Step 6: For Carla, the distance is 5.2 units and the time is 4.25 units.\n- Step 7: For Debra, the distance is 2.8 units and the time is 5.6 units.\n- Step 8: For Angela, the distance is 1.4 units and the time is 6.8 units.\n- Step 9: Calculate average speed for each student: Evelyn = 4.5 / 1.25 = 3.6, Briana = 2.2 / 2.5 = 0.88, Carla = 5.2 / 4.25 ≈ 1.22, Debra = 2.8 / 5.6 = 0.5, Angela = 1.4 / 6.8 ≈ 0.21.\n- Step 10: The student with the greatest average speed is Evelyn.\n<Answer>:\n<Answer>:\nEvelyn"
        gold_answer = "\\text{Evelyn}"

        score = self.evaluator._score_single((problem, solution, gold_answer))
        assert score == 1.0

    def test_string_parsed_in_numeric(self):
        problem = "What is the capital of France?"
        solution = "<Answer>:\nEvelyn"
        gold_answer = "\\text{Evelyn}"

        score = self.evaluator._score_single((problem, solution, gold_answer))
        assert score == 1.0

    def test_poly(self):
        problem = ""
        solution = "\\boxed{x^3 + 3x - 6}"
        gold_answer = "x^3+3x-6"

        score = self.evaluator._score_single((problem, solution, gold_answer))
        assert score == 1.0
