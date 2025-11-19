"""
CoT-UQ Scorer

Provides a reusable scorer implementation encapsulating the scoring
logic originally contained in `StrategyCoTUQ`. Implements the common
APIs used by strategies:
 - `score_candidates(chat, candidates, **kwargs)`
 - `score_complete_chains(chains, **kwargs)`

The scorer can optionally hold a `model` (a `BlackboxModelWithStreaming`) to
recompute token-level probabilities. If `token_probs` are provided by the
caller (strategy that sampled traces), those are used directly.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.scorers.step_scorer_base import CandidateScore, StepScorerBase

log = logging.getLogger(__name__)


class CotUqScorer(StepScorerBase):
    """CoT-UQ scorer implementing step-style and chain-style scoring APIs.

    Args:
        model: Optional model used to request token logprobs if needed.
        alpha: blend weight between probability-based confidence and CoT importance.
        top_logprobs: top-k to request when calling the model for token probs.
    """

    def __init__(
        self,
        model: Any = None,
        alpha: float = 0.5,
        top_logprobs: int = 10,
        name: str = "cot_uq_scorer",
    ):
        super().__init__(name)
        self.model = model if isinstance(model, BlackboxModelWithStreaming) else None
        self.alpha = alpha
        self.top_logprobs = top_logprobs

    # -- Internal helpers (ported/adapted from strategy_cot_uq) --
    @staticmethod
    def _extract_answer_span(text: str) -> Tuple[int, int]:
        marker = "<Answer>:"
        idx = text.rfind(marker)
        if idx >= 0:
            start = idx + len(marker)
            tail_markers = ["<end of response>", "<end response>", "<end>"]
            end = len(text)
            for m in tail_markers:
                m_idx = text.find(m, start)
                if m_idx != -1:
                    end = min(end, m_idx)
            return start, end
        last_nl = text.rfind("\n")
        start = last_nl + 1 if last_nl >= 0 else 0
        tail_markers = ["<end of response>", "<end response>", "<end>"]
        end = len(text)
        for m in tail_markers:
            m_idx = text.find(m, start)
            if m_idx != -1:
                end = min(end, m_idx)
        return start, end

    @staticmethod
    def _clean_answer_text(answer: str) -> str:
        import re

        if not answer:
            return ""

        s = answer.strip()
        tail_markers = [r"<end of response>", r"<end response>", r"<end>"]
        for m in tail_markers:
            s = re.sub(re.escape(m) + r"\s*$", "", s, flags=re.IGNORECASE)

        s = s.strip()

        m = re.match(r"^\\\\\((.*)\\\\\)$", s, flags=re.DOTALL)
        if m:
            s = m.group(1).strip()

        m = re.match(r"^\${1,2}(.*)\${1,2}$", s, flags=re.DOTALL)
        if m:
            s = m.group(1).strip()

        s = re.sub(r"\\n$", "", s)
        s = s.strip()
        return s

    @staticmethod
    def _compute_reasoning_importance(text: str) -> float:
        digits = sum(ch.isdigit() for ch in text)
        ops = sum(ch in "+-*/" for ch in text)
        tokens = max(1, len(text.split()))
        score = (digits + ops) / tokens
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _aggregate_answer_prob(
        token_probs: List[Tuple[str, float]], answer_text: str
    ) -> float:
        if not token_probs:
            return 0.5

        import re

        sig_pattern = re.compile(r"\\pi|\\frac|\d+|\(|\)|/")
        answer_sig = bool(sig_pattern.search(answer_text))

        flat_probs = [p for t, p in token_probs if p is not None]
        if answer_sig:
            relevant_probs = []
            for tok, prob in token_probs:
                if prob is None:
                    continue
                if sig_pattern.search(tok):
                    relevant_probs.append(prob)
            if relevant_probs:
                return float(np.mean(relevant_probs))

        if not flat_probs:
            return 0.5

        k = min(len(flat_probs), max(1, int(0.5 * len(flat_probs))))
        tail_probs = flat_probs[-k:]
        return float(np.mean(tail_probs))

    def _score_trace(
        self, text: str, token_probs: Optional[List[Tuple[str, float]]] = None
    ) -> Tuple[float, str]:
        a_start, a_end = self._extract_answer_span(text)
        answer_text = text[a_start:a_end]
        answer_text = self._clean_answer_text(answer_text)
        prob_conf = self._aggregate_answer_prob(token_probs or [], answer_text)
        cot_importance = self._compute_reasoning_importance(text[:a_start])
        score = self.alpha * prob_conf + (1 - self.alpha) * cot_importance
        return score, answer_text

    # -- Public scorer APIs --
    def score_complete_chains(
        self,
        chains: List[str],
        token_probs: Optional[List[List[Tuple[str, float]]]] = None,
    ) -> List[float]:
        """Score complete chains (full trajectories)."""
        scores: List[float] = []
        token_probs = token_probs or [None] * len(chains)
        for txt, tp in zip(chains, token_probs):
            try:
                s, _ = self._score_trace(txt, tp)
                scores.append(float(s))
            except Exception as e:
                log.error(f"Error scoring chain: {e}")
                scores.append(0.5)
        return scores

    def score_candidates_detailed(
        self,
        chat: List[Dict[str, str]],
        candidates: List[str],
        token_probs: Optional[List[List[Tuple[str, float]]]] = None,
        **kwargs,
    ) -> List[CandidateScore]:
        """Implements StepScorerBase API and returns detailed CandidateScore entries."""
        scores: List[CandidateScore] = []
        token_probs = token_probs or [None] * len(candidates)

        for idx, cand in enumerate(candidates):
            tp = token_probs[idx] if idx < len(token_probs) else None
            try:
                s, answer = self._score_trace(cand, tp)
                # Put the combined score into candidate claim_scores for compatibility
                cs = CandidateScore(
                    candidate_text=cand, claim_scores=[float(s)], aggregate_scores={}
                )
                cs.metadata = {"scorer_type": "cot_uq", "answer": answer}
                scores.append(cs)
            except Exception as e:
                log.error(f"Error scoring candidate: {e}")
                cs = CandidateScore(
                    candidate_text=cand, claim_scores=[0.5], aggregate_scores={}
                )
                cs.metadata = {"scorer_type": "cot_uq", "error": str(e)}
                scores.append(cs)

        return scores

    # Provide a lightweight cleanup hook for parity with other scorers
    def cleanup(self):
        # No persistent resources for now
        return
