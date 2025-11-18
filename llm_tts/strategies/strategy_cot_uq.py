import logging
from typing import Any, Dict, List, Optional, Tuple
import re

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
        scorer: Any = None,
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
        # Optional external scorer implementing common scorer APIs.
        # Expected methods: score_complete_chains(chains, **kwargs) and/or
        # score_candidates(chat, candidates, **kwargs)
        self.scorer = scorer

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
            # Trim common trailing markers that may appear after the answer
            # (e.g., "<end of response>"). Search for such markers after
            # the start and cut the end before them if present.
            tail_markers = ["<end of response>", "<end response>", "<end>"]
            end = len(text)
            for m in tail_markers:
                m_idx = text.find(m, start)
                if m_idx != -1:
                    end = min(end, m_idx)
            return start, end
        # fallback: last line
        last_nl = text.rfind("\n")
        start = last_nl + 1 if last_nl >= 0 else 0
        # Also trim tail markers in the fallback region
        tail_markers = ["<end of response>", "<end response>", "<end>"]
        end = len(text)
        for m in tail_markers:
            m_idx = text.find(m, start)
            if m_idx != -1:
                end = min(end, m_idx)
        return start, end

    @staticmethod
    def _aggregate_answer_prob(token_probs: List[Tuple[str, float]], answer_text: str) -> float:
        """Aggregate probabilities over answer tokens (mean prob)."""
        if not token_probs:
            return 0.5

        # Try to detect math-like answers (numbers, fractions, or \pi) and
        # aggregate probabilities over tokens that look relevant to the answer.
        import re

        sig_pattern = re.compile(r"\\pi|\\frac|\d+|\(|\)|/")
        answer_sig = bool(sig_pattern.search(answer_text))

        flat_tokens = [t for t, p in token_probs]
        flat_probs = [p for t, p in token_probs]

        if answer_sig:
            # Collect tokens that contain digits or common math markers
            relevant_probs = []
            for tok, prob in token_probs:
                try:
                    if prob is None:
                        continue
                except Exception:
                    continue
                if sig_pattern.search(tok):
                    relevant_probs.append(prob)

            if relevant_probs:
                return float(np.mean(relevant_probs))

        # Fallback: use the tail tokens heuristic (last ~50% of tokens)
        flat_probs = [p for p in flat_probs if p is not None]
        if not flat_probs:
            return 0.5
        k = min(len(flat_probs), max(1, int(0.5 * len(flat_probs))))
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
        answer_text = text[a_start:a_end]
        answer_text = self._clean_answer_text(answer_text)
        prob_conf = self._aggregate_answer_prob(token_probs, answer_text)
        cot_importance = self._compute_reasoning_importance(text[:a_start])
        # Combine (higher is better)
        score = self.alpha * prob_conf + (1 - self.alpha) * cot_importance
        return score, answer_text

    @staticmethod
    def _clean_answer_text(answer: str) -> str:
        """Clean extracted answer text by removing trailing markers, LaTeX wrappers,
        and stray whitespace/newlines.

        Examples removed: `\n<end of response>`, surrounding `\\( ... \\)`, `$...$`.
        """
        import re

        if not answer:
            return ""

        s = answer.strip()

        # Remove common tail markers if they somehow remained
        tail_markers = [r"<end of response>", r"<end response>", r"<end>"]
        for m in tail_markers:
            s = re.sub(re.escape(m) + r"\s*$", "", s, flags=re.IGNORECASE)

        s = s.strip()

        # Remove surrounding LaTeX inline delimiters: \( ... \), $...$, $$...$$
        # Use non-greedy match to capture inner content
        # \( ... \)
        m = re.match(r"^\\\\\((.*)\\\\\)$", s, flags=re.DOTALL)
        if m:
            s = m.group(1).strip()

        # $...$ or $$...$$
        m = re.match(r"^\${1,2}(.*)\${1,2}$", s, flags=re.DOTALL)
        if m:
            s = m.group(1).strip()

        # Also strip leading/trailing parentheses produced in some outputs
        s = s.strip()

        # If answer ends with a stray newline or literal escaped newline, remove it
        s = re.sub(r"\\n$", "", s)
        s = s.strip()

        return s

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think> tags/blocks and normalise blank lines."""
        if not text:
            return text
        s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        s = re.sub(r"</?think>", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _format_trace(self, raw_text: str, answer_text: str) -> str:
        """Enforce template-ish structure: Reasoning Steps + <Answer>."""
        cleaned = self._strip_thinking(raw_text or "")

        # Ensure Reasoning Steps header exists
        if "Reasoning Steps:" not in cleaned:
            cleaned = f"Reasoning Steps:\n- Step 1: {cleaned}"

        # Attach <Answer>: block if missing
        if "<Answer>:" not in cleaned:
            cleaned = f"{cleaned}\n<Answer>: {answer_text or ''}"

        return cleaned.strip()

    def generate_trajectory(self, request: List[Dict[str, str]]) -> Dict[str, Any]:
        traces: List[str] = []
        answers: List[str] = []
        scores: List[float] = []
        token_probs_list: List[List[Tuple[str, float]]] = []

        for i in range(self.budget):
            log.info(f"CoT-UQ sampling trace {i+1}/{self.budget}")
            text, token_probs = self._sample_full_trace(request)
            # Keep raw traces and token_probs; scoring may be delegated to external scorer
            traces.append(text)
            token_probs_list.append(token_probs)
            # Keep internal answer extraction as fallback/metadata
            _, answer = self._score_trace(text, token_probs)
            answers.append(answer)

        # If an external scorer is provided, prefer it for ranking
        if self.scorer is not None:
            try:
                # Build full chains by concatenating prompt and generated trace
                try:
                    prompt = "".join([m.get("content", "") for m in request])
                except Exception:
                    prompt = ""

                chains = [prompt + t for t in traces]

                if hasattr(self.scorer, "score_complete_chains"):
                    trajectory_scores = self.scorer.score_complete_chains(chains, token_probs=token_probs_list)
                else:
                    # Fall back to candidate-style scoring
                    trajectory_scores = self.scorer.score_candidates(request, traces, token_probs=token_probs_list)

                # Ensure we have a list of floats matching traces
                if not trajectory_scores or len(trajectory_scores) != len(traces):
                    raise ValueError("Scorer returned invalid scores")

                scores = [float(s) for s in trajectory_scores]
            except Exception as e:
                log.error(f"External scorer failed: {e}. Falling back to internal CoT-UQ scoring")
                # Fall back to internal scoring
                scores = []
                for t, tp in zip(traces, token_probs_list):
                    s, _ = self._score_trace(t, tp)
                    scores.append(s)
        else:
            # No external scorer provided: use internal CoT-UQ scoring
            for t, tp in zip(traces, token_probs_list):
                s, _ = self._score_trace(t, tp)
                scores.append(s)

        if not traces:
            return {"trajectory": "", "steps": [], "validity_scores": [], "completed": False}

        best_idx = int(np.argmax(scores))
        best_text = traces[best_idx]
        best_answer = answers[best_idx]

        formatted_trace = self._format_trace(best_text, best_answer)

        # Represent the best trace as a single step candidate containing the whole content
        best_step = StepCandidate(
            text=formatted_trace,
            token_ids=[],
            is_complete=True,
            is_trajectory_complete=True,
            generation_scores=None,
            raw_text=formatted_trace,
            other_data={
                "answer": best_answer,
                "cot_uq_score": scores[best_idx],
                "raw_original_trace": best_text,
            },
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
        # Allow external scorer to clean up resources
        if hasattr(self, "scorer") and self.scorer is not None:
            if hasattr(self.scorer, "cleanup"):
                try:
                    self.scorer.cleanup()
                except Exception:
                    log.exception("Error during scorer cleanup")

