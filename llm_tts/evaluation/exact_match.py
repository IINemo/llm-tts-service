import logging
import re

import numpy as np
from tqdm import tqdm

log = logging.getLogger()


class EvaluatorExactMatch:
    def __init__(self):
        pass

    def _score_single(self, inp: tuple[str, str, str]) -> float:
        _, solution, gold_answer = inp
        matches = re.findall(r"[-+]?\d[\d,]*\.?\d*", solution)

        if not matches:
            log.warning(f"Could not find a number in {solution}")
            return np.nan

        last = matches[-1]
        cleaned = last.replace(",", "")

        try:
            answer_number = float(cleaned)
            if answer_number.is_integer():
                answer_number = int(answer_number)
        except ValueError:
            log.warning(f"Could not convert {cleaned} to a number")
            return np.nan

        if answer_number == gold_answer:
            return 0
        elif answer_number != gold_answer:
            return 1
        else:
            return np.nan

    def __call__(
        self, problems: list[str], solutions: list[str], gold_answers: list[str]
    ) -> list[float]:
        all_inputs = zip(problems, solutions, gold_answers)
        labels = []
        for item in tqdm(all_inputs, desc="Verifying solutions"):
            labels.append(self._score_single(item))
        return labels
