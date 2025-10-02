"""
DeepConf strategy - accurate implementation based on Facebook Research's deepconf
https://github.com/facebookresearch/deepconf

Adapted for OpenRouter API instead of vLLM.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


def extract_answer(text: str) -> Optional[str]:
    """
    Extract boxed answer from text (from original DeepConf).

    Looks for LaTeX \boxed{answer} format commonly used in math problems.
    """
    if "boxed" not in text.lower():
        return None

    # Find the last occurrence of "boxed" (case-insensitive)
    text_lower = text.lower()
    last_boxed_idx = text_lower.rfind("boxed")
    if last_boxed_idx == -1:
        return None

    # Extract from original text (preserving case)
    ans = text[last_boxed_idx + 5 :]  # Skip "boxed"

    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        # Parse balanced braces
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a.strip()
    else:
        # No braces, take until $ or whitespace
        a = ans.split("$")[0].strip()
        return a


def compute_token_confidence_from_logprobs(
    top_logprobs: List[Dict], topk: int = 20
) -> float:
    """
    Compute token confidence from top-k logprobs (DeepConf formula).

    Formula: C_i = -mean(log P_i(top-k))

    Args:
        top_logprobs: List of dicts with 'logprob' values
        topk: Number of top logprobs to average

    Returns:
        Confidence score (higher = more confident)
    """
    if not top_logprobs:
        return 0.0

    k = min(topk, len(top_logprobs))
    if k == 0:
        return 0.0

    # Take top-k logprobs and compute negative mean
    mean_logprob = np.mean([lp["logprob"] for lp in top_logprobs[:k]])
    confidence = -mean_logprob

    return round(float(confidence), 3)


def compute_sliding_window_confidence(
    confs: List[float], window_size: int
) -> List[float]:
    """
    Compute sliding window mean confidence (from DeepConf).

    Args:
        confs: Per-token confidence scores
        window_size: Size of sliding window

    Returns:
        List of window mean confidences
    """
    if len(confs) < window_size:
        return [sum(confs) / len(confs)] if confs else [0.0]

    sliding_means = []
    for i in range(len(confs) - window_size + 1):
        window = confs[i : i + window_size]
        sliding_means.append(round(sum(window) / len(window), 3))

    return sliding_means


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """
    Perform weighted majority voting (from DeepConf).

    Args:
        answers: List of answer strings
        weights: Confidence weights for each answer

    Returns:
        Most voted answer (weighted by confidence)
    """
    if not answers:
        return None

    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(
                weight
            )

    if not answer_weights:
        return None

    return max(answer_weights.keys(), key=lambda x: answer_weights[x])


class StrategyDeepConf(StrategyBase):
    """
    DeepConf strategy - based on Facebook Research's implementation.

    Offline mode: Generate multiple traces, filter by confidence, vote.
    """

    def __init__(
        self,
        model,
        budget: int = 8,
        window_size: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        top_logprobs: int = 20,
        filter_method: str = "none",  # "none", "top10", "threshold"
        confidence_threshold: Optional[float] = None,
    ):
        """
        Initialize DeepConf strategy.

        Args:
            model: Model with generate_with_confidence() method
            budget: Number of reasoning traces to generate
            window_size: Sliding window size for group confidence
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Max tokens per generation
            top_logprobs: Number of top logprobs to request
            filter_method: How to filter traces ("none", "top10", "threshold")
            confidence_threshold: Threshold for filtering (if using "threshold")
        """
        self.model = model
        self.budget = budget
        self.window_size = window_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.filter_method = filter_method
        self.confidence_threshold = confidence_threshold

        # Check model supports logprobs
        if not hasattr(model, "supports_logprobs") or not model.supports_logprobs():
            log.warning(
                "⚠️  Model does not support logprobs - DeepConf will not work correctly"
            )

        if not hasattr(model, "generate_with_confidence"):
            raise ValueError("Model must have generate_with_confidence() method")

        log.info(
            f"✅ DeepConf initialized: budget={budget}, window={window_size}, filter={filter_method}"
        )

    def generate_trajectory(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry point - generate multiple traces and select best answer.

        Args:
            prompt: Input question/prompt

        Returns:
            Dict with trajectory and metadata
        """
        log.info(f"🚀 DeepConf: Generating {self.budget} reasoning traces")
        log.info(f"   Prompt: {prompt[:100]}...")

        # Generate all traces
        traces = self._generate_traces(prompt)

        # Extract answers and confidences
        valid_traces = [t for t in traces if t.get("extracted_answer")]
        log.info(f"📊 Valid traces with answers: {len(valid_traces)}/{len(traces)}")

        if not valid_traces:
            log.warning("⚠️  No valid answers extracted!")
            return {
                "trajectory": "",
                "steps": [],
                "completed": False,
                "strategy": "deepconf",
                "metadata": {
                    "num_paths_generated": len(traces),
                    "num_paths_used": 0,
                    "selected_answer": None,
                    "confidence_score": 0.0,
                    "vote_distribution": {},
                    "all_traces": traces,
                },
            }

        # Filter traces by confidence
        filtered_traces = self._filter_traces(valid_traces)
        log.info(f"🔍 Filtered traces: {len(filtered_traces)}/{len(valid_traces)}")

        # Perform weighted voting
        answers = [t["extracted_answer"] for t in filtered_traces]
        weights = [t["min_conf"] for t in filtered_traces]

        selected_answer = weighted_majority_vote(answers, weights)

        # Calculate vote distribution
        vote_dist = {}
        for ans, weight in zip(answers, weights):
            vote_dist[ans] = vote_dist.get(ans, 0.0) + weight

        total_weight = sum(vote_dist.values())
        vote_percentages = {
            ans: (w / total_weight) * 100 for ans, w in vote_dist.items()
        }

        # Get confidence (proportion of vote)
        confidence_score = (
            vote_dist.get(selected_answer, 0.0) / total_weight
            if total_weight > 0
            else 0.0
        )

        # Find selected trace
        selected_trace = None
        for trace in filtered_traces:
            if trace["extracted_answer"] == selected_answer:
                selected_trace = trace["text"]
                break

        log.info(f"🏆 Selected answer: '{selected_answer}'")
        log.info(f"   Confidence: {confidence_score:.3f}")
        log.info("   Vote distribution:")
        for ans, pct in sorted(
            vote_percentages.items(), key=lambda x: x[1], reverse=True
        ):
            log.info(f"     {ans}: {pct:.1f}%")

        return {
            "trajectory": selected_trace or "",
            "steps": [selected_trace] if selected_trace else [],
            "completed": True,
            "strategy": "deepconf",
            "metadata": {
                "num_paths_generated": len(traces),
                "num_paths_used": len(filtered_traces),
                "selected_answer": selected_answer,
                "confidence_score": confidence_score,
                "vote_distribution": vote_percentages,
                "filter_method": self.filter_method,
                "all_traces": traces,
                "filtered_traces": filtered_traces,
            },
        }

    def _generate_traces(self, prompt: str) -> List[Dict[str, Any]]:
        """Generate multiple reasoning traces with confidence scores."""
        traces = []

        for i in range(self.budget):
            log.info(f"🧠 Generating trace {i+1}/{self.budget}")

            try:
                # Generate with confidence
                text, token_data = self.model.generate_with_confidence(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

                # Compute per-token confidences
                token_confs = []
                for token_info in token_data:
                    conf = compute_token_confidence_from_logprobs(
                        token_info.get("top_logprobs", []), topk=self.top_logprobs
                    )
                    token_confs.append(conf)

                # Compute sliding window confidences
                window_confs = compute_sliding_window_confidence(
                    token_confs, self.window_size
                )
                min_conf = min(window_confs) if window_confs else 0.0

                # Extract answer
                extracted_answer = extract_answer(text)

                trace = {
                    "text": text,
                    "num_tokens": len(token_data),
                    "confs": token_confs,
                    "window_confs": window_confs,
                    "min_conf": min_conf,
                    "extracted_answer": extracted_answer,
                }

                traces.append(trace)

                log.info(
                    f"   Tokens: {len(token_data)}, Min conf: {min_conf:.3f}, "
                    f"Answer: {extracted_answer}"
                )

            except Exception as e:
                log.error(f"   ❌ Error generating trace {i+1}: {e}")
                continue

        return traces

    def _filter_traces(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter traces based on confidence."""
        if self.filter_method == "none":
            return traces

        if self.filter_method == "top10":
            # Keep top 10% by min_conf
            confs = [t["min_conf"] for t in traces]
            threshold = np.percentile(confs, 90)
            filtered = [t for t in traces if t["min_conf"] >= threshold]
            return filtered if filtered else traces  # Return all if none pass

        if self.filter_method == "threshold" and self.confidence_threshold is not None:
            # Keep traces above threshold
            filtered = [t for t in traces if t["min_conf"] >= self.confidence_threshold]
            return filtered if filtered else traces  # Return all if none pass

        return traces
