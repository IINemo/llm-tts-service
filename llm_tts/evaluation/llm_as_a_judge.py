import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import openai
from parse import parse
from tqdm import tqdm

log = logging.getLogger()


PROMPT_FULL_SOLUTION = r"""
You will be given a <Problem>, its proposed <Solution>, and the <Gold Answer>.
Your task is to assess whether the solution arrives at the **correct final answer**.

**IMPORTANT RULES**:
1. The solution is ONLY correct if the final answer matches the Gold Answer.
2. Good reasoning with a wrong final answer is INCORRECT.
3. Answers in DIFFERENT FORMS that are MATHEMATICALLY EQUIVALENT should be considered CORRECT. Examples:
   - "sinh(at)/a" equals "(exp(at) - exp(-at))/(2a)"
   - "1/(2√X)" equals "(1/2)X^(-1/2)"
   - "-8 - 8√3 i" equals "-8-8√3i" (spacing differences don't matter)
   - "0.5" equals "1/2"
4. Answers must be EXACTLY equal numerically. There is NO tolerance for approximate values. Examples of INCORRECT answers:
   - "698" does NOT equal "700"
   - "9.7" does NOT equal "9.6"
   - "0.147" does NOT equal "0.15"
   - "0.822" does NOT equal "8.233e-19" (different numbers, even if same physical quantity in different units)

Respond using the **exact format** below, do not include any text outside this template.
Output format:
<start of response>
Solution comments:
... brief comments on the solution ...
Final answer in solution: [extract the final answer from the solution]
Gold answer: [the provided gold answer]
Mathematically equivalent: (Yes|No)
<Grade>: (Correct|Incorrect)
<end of response>

<Problem>: {problem}

<Solution>: {solution}

<Gold Answer>: {gold_answer}
"""

PROMPT_ANSWER_ONLY = r"""
You will be given a <Problem>, a <Proposed Answer>, and a <Gold Answer>.
Your task is to assess whether the proposed answer is correct.

**IMPORTANT RULES**:
1. Answers in DIFFERENT FORMS that are MATHEMATICALLY EQUIVALENT should be considered CORRECT. Examples:
   - "sinh(at)/a" equals "(exp(at) - exp(-at))/(2a)"
   - "1/(2√X)" equals "(1/2)X^(-1/2)"
   - "-8 - 8√3 i" equals "-8-8√3i" (spacing differences don't matter)
   - "0.5" equals "1/2"
   - "x = 2" equals "2"
   - "\\frac{{1}}{{2}}" equals "0.5"
2. Answers must be EXACTLY equal numerically. There is NO tolerance for approximate values. Examples of INCORRECT answers:
   - "698" does NOT equal "700"
   - "9.7" does NOT equal "9.6"
   - "0.147" does NOT equal "0.15"
   - "0.822" does NOT equal "8.233e-19" (different numbers, even if same physical quantity in different units)
3. Consider the problem context when comparing answers (e.g., units, variables).

Respond using the **exact format** below, do not include any text outside this template.
Output format:
<start of response>
Proposed answer: [restate the proposed answer]
Gold answer: [restate the gold answer]
Mathematically equivalent: (Yes|No)
<Grade>: (Correct|Incorrect)
<end of response>

<Problem>: {problem}

<Proposed Answer>: {answer}

<Gold Answer>: {gold_answer}
"""

# Default prompt for backward compatibility
ANNOTATION_PROMPT = PROMPT_FULL_SOLUTION


class EvaluatorLLMAsAJudge:
    def __init__(
        self,
        prompt: str,
        cache_path: str,
        base_url: str,
        model: str,
        n_threads: int,
        budget: int = 3,
        mode: str = "answer_only",
        api_key: str | None = None,
    ):
        """
        Args:
            mode: "full_solution" - pass entire reasoning to judge
                  "answer_only" - compare just extracted answer vs gold (default)
            api_key: OpenAI-compatible API key. Falls back to OPENAI_API_KEY env var.
        """
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.prompt = prompt
        self.n_threads = n_threads
        self.budget = budget  # Number of evaluations for majority voting
        self.mode = mode

    def _parse_reply(self, reply: str) -> tuple[int, str]:
        """Parse a single reply and return (label, result_str)."""
        if "<Grade>: Correct" in reply:
            return 1, "Correct"
        elif "<Grade>: Incorrect" in reply:
            return 0, "Incorrect"
        else:
            return -1, "Unclear"  # Use -1 for unclear to distinguish from 0

    def _score_single(self, inp: tuple[str, str, str, int]) -> tuple[float, str, float]:
        """Score a single solution with majority voting and return (label, raw_response, consensus)."""
        problem, solution, gold_answer, idx = inp
        parsed = parse(self.prompt, problem)
        if parsed is not None:
            problem = parsed.named["q"]

        log.info(
            f"Evaluating sample {idx + 1} (budget={self.budget}, mode={self.mode})..."
        )
        if self.mode == "answer_only":
            prompt = PROMPT_ANSWER_ONLY.format(
                problem=problem, answer=solution, gold_answer=gold_answer
            )
        else:
            prompt = PROMPT_FULL_SOLUTION.format(
                problem=problem, solution=solution, gold_answer=gold_answer
            )

        # Collect multiple evaluations for majority voting
        votes = []
        replies = []
        for i in range(self.budget):
            # Use unique cache key per vote to avoid caching identical responses
            if self.budget > 1:
                vote_prompt = f"{prompt}\n<!-- vote {i+1}/{self.budget} -->"
            else:
                vote_prompt = prompt

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an intelligent assistant.",
                        },
                        {"role": "user", "content": vote_prompt},
                    ],
                    temperature=1,
                    max_completion_tokens=1024,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                log.warning(f"API call failed for sample {idx + 1}: {e}")
                reply = f"Error: {e}"

            label, _ = self._parse_reply(reply)
            votes.append(label)
            replies.append(reply)

        # Majority voting (exclude unclear votes)
        valid_votes = [v for v in votes if v >= 0]
        if valid_votes:
            # Count votes: 1 = correct, 0 = incorrect
            correct_votes = sum(1 for v in valid_votes if v == 1)
            incorrect_votes = sum(1 for v in valid_votes if v == 0)
            final_label = 1 if correct_votes > incorrect_votes else 0
            consensus = max(correct_votes, incorrect_votes) / len(valid_votes)
            result_str = (
                f"Correct ({correct_votes}/{len(valid_votes)})"
                if final_label == 1
                else f"Incorrect ({incorrect_votes}/{len(valid_votes)})"
            )
        else:
            final_label = np.nan
            consensus = 0.0
            result_str = "Unclear"

        # Combine all replies for transparency
        if self.budget > 1:
            combined_reply = f"=== Majority Vote: {result_str} ===\n"
            for i, (reply, vote) in enumerate(zip(replies, votes)):
                vote_str = {1: "Correct", 0: "Incorrect", -1: "Unclear"}[vote]
                combined_reply += f"\n--- Vote {i+1}: {vote_str} ---\n{reply}\n"
        else:
            combined_reply = replies[0]

        log.info(f"Sample {idx + 1}: {result_str}")
        return final_label, combined_reply, consensus

    def __call__(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> tuple[list[float], list[str], list[float]]:
        """Evaluate solutions and return (labels, raw_responses, consensus_scores)."""
        log.info(f"Starting evaluation of {len(problems)} samples...")
        all_inputs = [
            (problem, solution, gold_answer, idx)
            for idx, (problem, solution, gold_answer) in enumerate(
                zip(problems, solutions, gold_answers)
            )
        ]

        # Use dict to collect results by index (as_completed returns out of order)
        results_by_idx = {}

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            future_to_idx = {
                executor.submit(self._score_single, item): item[3]
                for item in all_inputs
            }

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(all_inputs),
                desc="Verifying solutions",
            ):
                idx = future_to_idx[future]
                label, response, consensus = future.result()
                results_by_idx[idx] = (label, response, consensus)

        # Reconstruct ordered lists
        labels = []
        responses = []
        consensus_scores = []
        for idx in range(len(all_inputs)):
            label, response, consensus = results_by_idx[idx]
            labels.append(label)
            responses.append(response)
            consensus_scores.append(consensus)

        # Summary statistics
        correct_count = sum(1 for label in labels if label == 1)
        incorrect_count = sum(1 for label in labels if label == 0)
        unclear_count = sum(1 for label in labels if np.isnan(label))
        accuracy = (correct_count / len(labels) * 100) if labels else 0

        log.info(
            f"Evaluation complete: {correct_count}/{len(labels)} correct ({accuracy:.1f}%)"
        )
        if incorrect_count > 0:
            log.info(f"  Incorrect: {incorrect_count}")
        if unclear_count > 0:
            log.info(f"  Unclear: {unclear_count}")

        return labels, responses, consensus_scores
