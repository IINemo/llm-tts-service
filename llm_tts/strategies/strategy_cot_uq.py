import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.step_boundary_detector import StepBoundaryDetector
from llm_tts.step_candidate_generator_base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    covert_trajectory_to_string,
)

from .cot_uq_evidence import (
    CotUqEvidenceExtractor,
    DEFAULT_ANSWER_PATTERNS,
    DEFAULT_STEP_PATTERNS,
    clean_answer_text,
    compute_reasoning_importance,
    extract_answer_span,
)
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
        max_tokens: int = 1024,
        top_logprobs: int = 10,
        alpha: float = 0.5,
        scorer: Any = None,
        detector_step_patterns: Optional[List[str]] = None,
        detector_answer_patterns: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        max_empty_steps: Optional[int] = None,
        max_keywords: int = 5,
        step_generator: Optional[StepCandidateGeneratorBase] = None,
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
        self.max_steps = max_steps
        self.max_empty_steps = max_empty_steps
        self.step_patterns = detector_step_patterns or DEFAULT_STEP_PATTERNS
        self.answer_patterns = detector_answer_patterns or DEFAULT_ANSWER_PATTERNS
        self.step_generator = step_generator
        # Optional external scorer implementing common scorer APIs.
        # Expected methods: score_complete_chains(chains, **kwargs) and/or
        # score_candidates(chat, candidates, **kwargs)
        self.scorer = scorer

        # Reuse boundary patterns from elsewhere
        self.detector = StepBoundaryDetector(
            step_patterns=self.step_patterns,
            answer_patterns=self.answer_patterns,
            max_tokens_per_step=max_tokens,
        )
        self.evidence_extractor = CotUqEvidenceExtractor(
            step_patterns=self.step_patterns,
            answer_patterns=self.answer_patterns,
            max_steps=max_steps,
            max_empty_steps=max_empty_steps,
            max_keywords=max_keywords,
        )

    def _sample_full_trace(
        self, request: List[Dict[str, str]]
    ) -> Tuple[str, List[Tuple[str, float]]]:
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
        return extract_answer_span(text, DEFAULT_ANSWER_PATTERNS)

    @staticmethod
    def _aggregate_answer_prob(
        token_probs: List[Tuple[str, float]], answer_text: str
    ) -> float:
        """Aggregate probabilities over answer tokens (mean prob)."""
        if not token_probs:
            return 0.5

        # Try to detect math-like answers (numbers, fractions, or \pi) and
        # aggregate probabilities over tokens that look relevant to the answer.
        import re

        sig_pattern = re.compile(r"\\pi|\\frac|\d+|\(|\)|/")
        answer_sig = bool(sig_pattern.search(answer_text))

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
        return compute_reasoning_importance(text)

    def _score_trace(
        self, text: str, token_probs: List[Tuple[str, float]]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Return (score, answer_text, evidence) for one trace."""
        try:
            evidence = self.evidence_extractor.extract(text, token_probs)
            answer_info = evidence.get("answer", {})
            answer_text = answer_info.get("clean_text", "") or ""
            prob_conf = float(answer_info.get("mean_probability", 0.5))
            cot_importance = float(evidence.get("keyword_confidence", 0.5))
        except Exception:
            log.exception("Failed to extract CoT-UQ evidence; falling back to heuristics")
            a_start, a_end = self._extract_answer_span(text)
            answer_text = self._clean_answer_text(text[a_start:a_end])
            prob_conf = self._aggregate_answer_prob(token_probs, answer_text)
            cot_importance = self._compute_reasoning_importance(text[:a_start])
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
        return score, answer_text, evidence

    @staticmethod
    def _clean_answer_text(answer: str) -> str:
        """Clean extracted answer text by removing trailing markers, LaTeX wrappers,
        and stray whitespace/newlines.

        Examples removed: `\n<end of response>`, surrounding `\\( ... \\)`, `$...$`.
        """
        return clean_answer_text(answer)

    def _force_answer_with_step_generator(
        self,
        request: List[Dict[str, str]],
        partial_text: str,
    ) -> Optional[str]:
        if not self.step_generator:
            return None

        try:
            trajectory: List[StepCandidate] = []
            if partial_text:
                trajectory.append(
                    StepCandidate(
                        text=partial_text,
                        token_ids=[],
                        is_complete=True,
                        is_trajectory_complete=False,
                        generation_scores=None,
                        raw_text=partial_text,
                    )
                )

            candidates = self.step_generator.generate_answer_candidates(
                request, trajectory, candidates_per_step=1
            )
        except Exception:
            log.exception("Step generator failed to produce fallback answer")
            return None

        if not candidates:
            return None

        addition = candidates[0].text or ""
        if not addition.strip():
            return None

        combined = partial_text or ""
        combined += addition
        return combined

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
        token_probs_list: List[List[Tuple[str, float]]] = []
        internal_scores: List[float] = []
        evidences: List[Dict[str, Any]] = []
        scores: List[float] = []

        for i in range(self.budget):
            log.info(f"CoT-UQ sampling trace {i+1}/{self.budget}")
            text, token_probs = self._sample_full_trace(request)
            if not text:
                log.warning("Received empty trace; skipping")
                continue
            score, answer, evidence = self._score_trace(text, token_probs)
            if not answer.strip():
                forced_text = self._force_answer_with_step_generator(request, text)
                if forced_text and forced_text != text:
                    log.info("CoT-UQ forcing answer completion via step generator")
                    text = forced_text
                    token_probs = []
                    score, answer, evidence = self._score_trace(text, token_probs)

            # Keep raw traces and token_probs; scoring may be delegated to external scorer
            traces.append(text)
            token_probs_list.append(token_probs)
            answers.append(answer)
            internal_scores.append(score)
            evidences.append(evidence)

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
                    trajectory_scores = self.scorer.score_complete_chains(
                        chains, token_probs=token_probs_list
                    )
                else:
                    # Fall back to candidate-style scoring
                    trajectory_scores = self.scorer.score_candidates(
                        request, traces, token_probs=token_probs_list
                    )

                # Ensure we have a list of floats matching traces
                if not trajectory_scores or len(trajectory_scores) != len(traces):
                    raise ValueError("Scorer returned invalid scores")

                scores = [float(s) for s in trajectory_scores]
            except Exception as e:
                log.error(
                    f"External scorer failed: {e}. Falling back to internal CoT-UQ scoring"
                )
                scores = list(internal_scores)
        else:
            scores = list(internal_scores)

        if not traces:
            return {
                "trajectory": "",
                "steps": [],
                "validity_scores": [],
                "completed": False,
            }

        best_idx = int(np.argmax(scores))
        best_text = traces[best_idx]
        best_answer = answers[best_idx]
        best_evidence = evidences[best_idx] if evidences else {}

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
                "cot_uq_evidence": best_evidence,
                "raw_original_trace": best_text,
            },
        )

        log.info(
            f"CoT-UQ selected trace {best_idx} with score {scores[best_idx]:.3f} and answer: {answers[best_idx]}"
        )

        metadata = {
            "cot_uq": {
                "selected_index": best_idx,
                "scores": scores,
                "best_evidence": best_evidence,
                "all_evidence": evidences,
            }
        }

        return {
            "trajectory": covert_trajectory_to_string([best_step]),
            "steps": [best_step],
            "validity_scores": scores,
            "completed": True,
            "metadata": metadata,
        }

    def cleanup(self):
        # Allow external scorer to clean up resources
        if hasattr(self, "scorer") and self.scorer is not None:
            if hasattr(self.scorer, "cleanup"):
                try:
                    self.scorer.cleanup()
                except Exception:
                    log.exception("Error during scorer cleanup")
