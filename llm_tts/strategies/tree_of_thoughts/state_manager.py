"""
State management utilities for Tree-of-Thoughts beam search.
"""

from typing import List, Tuple

import numpy as np


def select_top_states(
    candidates: List[str], scores: List[float], beam_width: int
) -> Tuple[List[str], List[float]]:
    """
    Select top-k candidates based on scores.

    Args:
        candidates: List of candidate states
        scores: List of scores (higher is better)
        beam_width: Number of states to keep

    Returns:
        Tuple of (selected_states, selected_scores)
    """
    if len(candidates) <= beam_width:
        return candidates, scores

    # Get indices of top-k scores
    top_indices = np.argsort(scores)[-beam_width:][::-1]

    selected_states = [candidates[i] for i in top_indices]
    selected_scores = [scores[i] for i in top_indices]

    return selected_states, selected_scores
