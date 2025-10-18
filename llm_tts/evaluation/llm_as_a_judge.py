import logging
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from lm_polygraph.utils.openai_chat import OpenAIChat
from parse import parse
from tqdm import tqdm

log = logging.getLogger()


ANNOTATION_PROMPT = r"""
You will be given a <Problem> and its proposed <Solution>.
Your task is to assess whether the solution is **correct** or **incorrect**.

Respond using the **exact format** below,
do not include any text outside this template.
Output format:
<start of response>
Solution comments:
... your comments on the solution, explaining reasoning,
    pointing out any errors or confirming correctness ...
<Grade>: (Correct|Incorrect)
<end of response>

<Problem>: {problem}

<Solution>: {solution}

<Gold Answer>: {gold_answer}
"""


class EvaluatorLLMAsAJudge:
    def __init__(
        self, prompt: str, cache_path: str, base_url: str, model: str, n_threads: int
    ):
        self.chat = OpenAIChat(
            cache_path=cache_path,
            openai_model=model,
            base_url=base_url,
        )
        self.prompt = prompt
        self.n_threads = n_threads

    def _score_single(self, inp: tuple[str, str, str, int]) -> tuple[float, str]:
        """Score a single solution and return (label, raw_response)."""
        problem, solution, gold_answer, idx = inp
        # This line extracts the question from the problem string using
        # the prompt template. It uses the parse library to match the
        # prompt template against the problem string and extracts the
        # named field "q" (which corresponds to {q} in the template).
        # If parsing fails, parse() returns None and accessing
        # .named["q"] will raise an AttributeError
        parsed = parse(self.prompt, problem)
        if parsed is not None:
            problem = parsed.named["q"]
            # If parsing fails, keep the original problem string unchanged

        log.info(f"Evaluating sample {idx + 1}...")
        prompt = ANNOTATION_PROMPT.format(
            problem=problem, solution=solution, gold_answer=gold_answer
        )
        reply = self.chat.ask(prompt)

        # Determine the numeric label
        if "<Grade>: Correct" in reply:
            label = 1
            result_str = "Correct"
        elif "<Grade>: Incorrect" in reply:
            label = 0
            result_str = "Incorrect"
        else:
            label = np.nan
            result_str = "Unclear"

        log.info(f"Sample {idx + 1}: {result_str}")
        return label, reply

    def __call__(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> tuple[list[float], list[str]]:
        """Evaluate solutions and return (labels, raw_responses)."""
        log.info(f"Starting evaluation of {len(problems)} samples...")
        all_inputs = [
            (problem, solution, gold_answer, idx)
            for idx, (problem, solution, gold_answer) in enumerate(
                zip(problems, solutions, gold_answers)
            )
        ]
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(self._score_single, item) for item in all_inputs]
            labels = []
            responses = []
            for future in tqdm(futures, desc="Verifying solutions"):
                label, response = future.result()
                labels.append(label)
                responses.append(response)

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

            return labels, responses
