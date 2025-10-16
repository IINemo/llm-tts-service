import logging
from typing import Dict, List

import numpy as np
import torch
from lm_polygraph.generation_metrics.alignscore_utils import AlignScorer
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric

log = logging.getLogger()


class EvaluatorAlignScore(GenerationMetric):
    """
    Calculates AlignScore metric (https://aclanthology.org/2023.acl-long.634/)
    between model-generated texts and ground truth texts.
    """

    def __init__(
        self,
        ckpt_path: str = "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",
        batch_size: int = 16,
        target_is_claims: bool = True,
        threshold: float = 0.5,
    ):
        super().__init__(["greedy_texts", "input_texts"], "sequence")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_is_claims = target_is_claims
        self.batch_size = batch_size
        self.threshold = float(threshold)
        self.scorer = AlignScorer(
            model="roberta-large",
            batch_size=batch_size,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
        )

    def __str__(self):
        return "AlignScore"

    def _normalize_empty(self, text: str) -> str:
        return text if text and text.strip() else "(empty)"

    def compute_scores(
        self,
        stats: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute raw AlignScore values between generated texts and targets.
        """
        filtered_outputs = [self._normalize_empty(x) for x in stats["solutions"]]

        if self.target_is_claims:
            claims = stats["gold_answers"]
            contexts = filtered_outputs
        else:
            claims = filtered_outputs
            contexts = stats["gold_answers"]

        scores = np.array(
            self.scorer.score(
                claims=claims,
                contexts=contexts,
            )
        )

        return scores

    def __call__(
        self, problems: List[str], solutions: List[str], gold_answers: List[str]
    ) -> List[float]:
        try:
            stats = {
                "solutions": np.array([s if s is not None else "" for s in solutions])
            }
            stats["gold_answers"] = np.array(
                [g if g is not None else "" for g in gold_answers]
            )

            scores = self.compute_scores(stats)
            labels = [1.0 if s >= self.threshold else 0.0 for s in scores]
            return labels

        except Exception as e:
            log.error(f"Error calculating AlignScore: {e}")
            return [np.nan] * len(solutions)
