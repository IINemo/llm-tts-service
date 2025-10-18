from typing import Dict

import numpy as np
from lm_polygraph.estimators.estimator import Estimator


class PDGap(Estimator):
    """Probability Differential-based uncertainty estimator

    Computes sequence-level uncertainty as the mean of token-level PD-Gap scores:
        1 - (p1 - p2)
    where p1 and p2 are the top-2 token probabilities at each position.

    Input:
        stats["greedy_log_probs"]: List[np.ndarray] — each array of shape [T, V].

    Output:
        np.ndarray — per-sequence mean uncertainties.
    """

    def __init__(self):
        super().__init__(["uncertainty_pd"], "sequence")

    def __str__(self):
        return "PDGap"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs_list = stats.get("greedy_log_probs", [])
        if not log_probs_list:
            return np.array([])

        uncertainties = []
        for lp in log_probs_list:
            if lp is None or lp.size == 0:
                uncertainties.append(np.nan)
                continue

            try:
                probs = np.exp(lp)
                # extract top-2 probabilities per token
                top2 = np.partition(probs, -2, axis=1)[:, -2:]
                p1, p2 = np.sort(top2, axis=1)[:, ::-1].T  # p1=max, p2=second
                # Calculate mean PD-Gap across all tokens for this sequence
                pd_gap_tokens = 1.0 - (p1 - p2)
                uncertainties.append(np.mean(pd_gap_tokens))
            except Exception:
                uncertainties.append(np.nan)

        return np.array(uncertainties)
