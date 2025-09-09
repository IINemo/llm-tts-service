from parse import parse
from concurrent.futures.thread import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

from .deepseek_chat import DeepSeekChat

import logging

log = logging.getLogger()


ANNOTATION_PROMPT = r'''
You will be given a <Problem> and its proposed <Solution>. Your task is to assess whether the solution is **correct** or **incorrect**.

Respond using the **exact format** below, do not include any text outside this template.
Output format:
<start of response>
Solution comments:
... your comments on the solution, explaining reasoning, pointing out any errors or confirming correctness ...
<Grade>: (Correct|Incorrect)
<end of response>

<Problem>: {problem}

<Solution>: {solution}
'''


class DeepSeekAnnotator:
    def __init__(
            self,
            prompt: str,
            cache_path: str = "~/.cache",
            model: str = 'deepseek-reasoner',
            api_key: str | None = None,
            n_threads: int = 1,
            wait_times: tuple = (5, 10, 30, 60, 120),
    ):
        self.chat = DeepSeekChat(cache_path, model=model, api_key=api_key, wait_times=wait_times)
        self.prompt = prompt
        self.n_threads = n_threads

    def _score_single(self, inp: tuple[str, str]) -> float:
        problem, solution = inp
        problem = parse(self.prompt, problem).named['q']
        prompt = ANNOTATION_PROMPT.format(problem=problem, solution=solution)
        reply = self.chat.ask(prompt)
        if '<Grade>: Correct' in reply:
            return 0
        elif '<Grade>: Incorrect' in reply:
            return 1
        else:
            return np.nan

    def __call__(self, problems: list[str], solutions: list[str]) -> list[float]:
        all_inputs = zip(problems, solutions)
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(self._score_single, item) for item in all_inputs]
            labels = []
            for future in tqdm(futures, desc="Verifying solutions"):
                labels.append(future.result())
            return labels
