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
from llm_tts.strategies.cot_uq_evidence import (
    DEFAULT_ANSWER_PATTERNS,
    DEFAULT_STEP_PATTERNS,
    CotUqEvidenceExtractor,
    clean_answer_text,
    compute_reasoning_importance,
    extract_answer_span,
)

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
        step_patterns: Optional[List[str]] = None,
        answer_patterns: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        max_empty_steps: Optional[int] = None,
        max_keywords: int = 5,
        name: str = "cot_uq_scorer",
    ):
        super().__init__(name)
        self.model = model if isinstance(model, BlackboxModelWithStreaming) else None
        self.alpha = alpha
        self.top_logprobs = top_logprobs
        self.extractor = CotUqEvidenceExtractor(
            step_patterns=step_patterns or DEFAULT_STEP_PATTERNS,
            answer_patterns=answer_patterns or DEFAULT_ANSWER_PATTERNS,
            max_steps=max_steps,
            max_empty_steps=max_empty_steps,
            max_keywords=max_keywords,
        )

    # -- Internal helpers (ported/adapted from strategy_cot_uq) --
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
    ) -> Tuple[float, Dict[str, Any]]:
        token_probs = token_probs or []
        try:
            evidence = self.extractor.extract(text, token_probs)
            answer_info = evidence.get("answer", {})
            prob_conf = float(answer_info.get("mean_probability", 0.5))
            cot_importance = float(evidence.get("keyword_confidence", 0.5))
        except Exception:
            log.exception(
                "CotUqScorer failed to extract evidence; using fallback heuristics"
            )
            a_start, a_end = extract_answer_span(text, DEFAULT_ANSWER_PATTERNS)
            answer_text = clean_answer_text(text[a_start:a_end])
            prob_conf = self._aggregate_answer_prob(token_probs, answer_text)
            cot_importance = compute_reasoning_importance(text[:a_start])
            evidence = {
                "answer": {
                    "text": text[a_start:a_end],
                    "clean_text": answer_text,
                    "span": [a_start, a_end],
                    "token_details": [],
                    "probabilities": [],
                    "mean_probability": prob_conf,
                },
                "reasoning_text": text[:a_start],
                "steps": [],
                "keyword_confidence": cot_importance,
            }
        score = self.alpha * prob_conf + (1 - self.alpha) * cot_importance
        return score, evidence

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
                s, evidence = self._score_trace(cand, tp)
                # Put the combined score into candidate claim_scores for compatibility
                cs = CandidateScore(
                    candidate_text=cand, claim_scores=[float(s)], aggregate_scores={}
                )
                cs.metadata = {
                    "scorer_type": "cot_uq",
                    "answer": evidence.get("answer", {}).get("clean_text"),
                    "cot_uq_evidence": evidence,
                }
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
