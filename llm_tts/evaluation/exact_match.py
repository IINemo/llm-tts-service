import logging
import re

import numpy as np
from tqdm import tqdm

log = logging.getLogger()


class EvaluatorExactMatch:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def _score_single(self, inp: tuple[str, str, str]) -> float:
        _, solution, gold_answer = inp
        if "gsm8k" in self.dataset_name.lower():
            gold_answer = self.parse_gsm8k_gold_answer(gold_answer)
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
            return 1
        elif answer_number != gold_answer:
            return 0
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

    def parse_gsm8k_gold_answer(self, answer: str):
        if answer is None:
            return None
        tail = str(answer).split("####")[-1].strip()

        m = re.search(r"[-+]?\d[\d,]*\.?\d*", tail)
        if not m:
            return tail if tail else None
        num = m.group(0).replace(",", "")
        try:
            val = float(num)
            if val.is_integer():
                return int(val)
            return val
        except Exception:
            return tail if tail else None
