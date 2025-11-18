import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llm_tts.step_candidate_generator_base import (
    StepCandidate,
    covert_trajectory_to_string,
)
from llm_tts.step_boundary_detector import StepBoundaryDetector
from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming

from .strategy_base import StrategyBase


log = logging.getLogger(__name__)


class StrategyCoTUQ(StrategyBase):
    """
    CoT-UQ: Response-wise uncertainty with Chain-of-Thought cues.

    High-level algorithm (simplified):
      1) Sample multiple full traces with token logprobs enabled.
      2) For each trace, extract final answer span and the reasoning span.
      3) Compute aggregated probability metrics over the answer tokens (e.g., mean prob).
      4) Compute token-importance weights from reasoning (keyword emphasis heuristic).
      5) Combine into a response-wise uncertainty score; pick the best answer.
    """

    def __init__(
        self,
        model: BlackboxModelWithStreaming,
        budget: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        top_logprobs: int = 10,
        alpha: float = 0.5,
    ):
        """
        Args:
            model: Blackbox API model supporting logprobs.
            budget: Number of full traces to sample.
            temperature, top_p, max_tokens: generation parameters.
            top_logprobs: request top-k logprobs for token confidences.
            alpha: weighting between prob-based confidence and CoT importance.
        """
        if not getattr(model, "supports_logprobs", False):
            raise ValueError("CoT-UQ requires a model with logprobs support")

        self.model = model
        self.budget = budget
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.alpha = alpha

        # Reuse boundary patterns from elsewhere
        self.detector = StepBoundaryDetector(
            step_patterns=["- Step", "<Answer>:", "\n<Answer>:"],
            answer_patterns=["<Answer>:", "\n<Answer>:"],
            max_tokens_per_step=max_tokens,
        )

    def _sample_full_trace(self, request: List[Dict[str, str]]) -> Tuple[str, List[Tuple[str, float]]]:
        """Generate a full trace with token-level probs.

        Returns:
            (generated_text, list of (token_text, prob))
        """
        response = self.model.generate_with_logprobs(
            request,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs,
        )

        generated_text = response.text
        token_probs = response.token_probs  # List[Tuple[token, prob]] if available
        return generated_text, token_probs or []

    @staticmethod
    def _extract_answer_span(text: str) -> Tuple[int, int]:
        """Find answer segment indices in the text; fallback to last line."""
        marker = "<Answer>:"
        idx = text.rfind(marker)
        if idx >= 0:
            start = idx + len(marker)
            return start, len(text)
        # fallback: last line
        last_nl = text.rfind("\n")
        start = last_nl + 1 if last_nl >= 0 else 0
        return start, len(text)

    @staticmethod
    def _aggregate_answer_prob(token_probs: List[Tuple[str, float]], answer_text: str) -> float:
        """Aggregate probabilities over answer tokens (mean prob)."""
        if not token_probs:
            return 0.5
        # Simple heuristic: use the tail tokens with total length ~ answer length
        answer_len = max(1, len(answer_text))
        flat_tokens = [t for t, p in token_probs]
        flat_probs = [p for t, p in token_probs]
        # Approximate: use last K tokens
        k = min(len(flat_probs), max(1, int(0.5 * len(flat_probs))))
        if k <= 0:
            return 0.5
        tail_probs = flat_probs[-k:]
        return float(np.mean(tail_probs))

    @staticmethod
    def _compute_reasoning_importance(text: str) -> float:
        """Compute a lightweight importance from CoT: presence of numbers and ops as proxy."""
        # Count digits and math ops as a crude proxy for salient reasoning
        digits = sum(ch.isdigit() for ch in text)
        ops = sum(ch in "+-*/" for ch in text)
        tokens = max(1, len(text.split()))
        score = (digits + ops) / tokens
        # Clamp to [0,1]
        return float(max(0.0, min(1.0, score)))

    def _score_trace(self, text: str, token_probs: List[Tuple[str, float]]) -> Tuple[float, str]:
        """Return (score, answer_text) for one trace."""
        a_start, a_end = self._extract_answer_span(text)
        answer_text = text[a_start:a_end].strip()
        prob_conf = self._aggregate_answer_prob(token_probs, answer_text)
        cot_importance = self._compute_reasoning_importance(text[:a_start])
        # Combine (higher is better)
        score = self.alpha * prob_conf + (1 - self.alpha) * cot_importance
        return score, answer_text

    def generate_trajectory(self, request: List[Dict[str, str]]) -> Dict[str, Any]:
        traces: List[str] = []
        answers: List[str] = []
        scores: List[float] = []

        for i in range(self.budget):
            log.info(f"CoT-UQ sampling trace {i+1}/{self.budget}")
            text, token_probs = self._sample_full_trace(request)
            score, answer = self._score_trace(text, token_probs)
            traces.append(text)
            answers.append(answer)
            scores.append(score)

        if not traces:
            return {"trajectory": "", "steps": [], "validity_scores": [], "completed": False}

        best_idx = int(np.argmax(scores))
        best_text = traces[best_idx]

        # Represent the best trace as a single step candidate containing the whole content
        best_step = StepCandidate(
            text=best_text,
            token_ids=[],
            is_complete=True,
            is_trajectory_complete=True,
            generation_scores=None,
            raw_text=best_text,
            other_data={"answer": answers[best_idx], "cot_uq_score": scores[best_idx]},
        )

        log.info(
            f"CoT-UQ selected trace {best_idx} with score {scores[best_idx]:.3f} and answer: {answers[best_idx]}"
        )

        return {
            "trajectory": covert_trajectory_to_string([best_step]),
            "steps": [best_step],
            "validity_scores": scores,
            "completed": True,
        }

    def cleanup(self):
        pass


