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


class EvaluatorGoldStandard:
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

    def _score_single(self, inp: tuple[str, str, str]) -> float:
        problem, solution, gold_answer = inp
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

        prompt = ANNOTATION_PROMPT.format(
            problem=problem, solution=solution, gold_answer=gold_answer
        )
        reply = self.chat.ask(prompt)
        if "<Grade>: Correct" in reply:
            return 0
        elif "<Grade>: Incorrect" in reply:
            return 1
        else:
            return np.nan

    def __call__(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> list[float]:  # TODO: gold_answers are not used
        all_inputs = zip(problems, solutions, gold_answers)
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(self._score_single, item) for item in all_inputs]
            labels = []
            for future in tqdm(futures, desc="Verifying solutions"):
                labels.append(future.result())
            return labels
