"""
DeepConf strategy - Confidence-based test-time scaling

Based on Facebook Research's DeepConf:
https://github.com/facebookresearch/deepconf

Supports both offline and online modes:
- Offline: Generate N traces, filter by confidence, majority vote
- Online: Adaptive generation with confidence-based early stopping
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from lm_polygraph import BlackboxModel

from llm_tts.utils.confidence import (
    compute_sliding_window_confidence,
    compute_token_confidence_from_logprobs,
    extract_answer,
)

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyDeepConf(StrategyBase):
    """
    DeepConf strategy with offline and online modes.

    Offline mode:
        1. Generate N complete traces with logprobs
        2. Compute confidence scores for each trace
        3. Filter traces by confidence threshold
        4. Majority vote on answers

    Online mode:
        1. Warmup: Generate K traces to estimate confidence threshold
        2. Adaptive: Generate remaining traces with early stopping
        3. Filter and majority vote
    """

    def __init__(
        self,
        model: BlackboxModel,
        mode: str,
        budget: int,
        window_size: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        top_logprobs: int,
        filter_method: str,
        warmup_traces: Optional[int] = None,
        total_budget: Optional[int] = None,
        confidence_percentile: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ):
        """
        Initialize DeepConf strategy.

        Args:
            model: BlackboxModel with logprobs support
            mode: "offline" or "online"
            budget: Number of traces for offline mode
            warmup_traces: Warmup traces for online mode
            total_budget: Total budget for online mode
            confidence_percentile: Confidence threshold percentile for online
            window_size: Window size for confidence computation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens per trace
            top_logprobs: Number of top logprobs to request
            filter_method: Filtering method ("topK", "threshold", or specific like "top10")
            confidence_threshold: Manual confidence threshold (optional)
        """
        self.model = model
        self.mode = mode.lower()
        self.budget = budget
        self.warmup_traces = warmup_traces
        self.total_budget = total_budget
        self.confidence_percentile = confidence_percentile
        self.window_size = window_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.filter_method = filter_method
        self.confidence_threshold = confidence_threshold

        # Validate model supports logprobs
        if not hasattr(model, "supports_logprobs") or not model.supports_logprobs:
            raise ValueError("Model must support logprobs for DeepConf")

        log.info(
            f"✅ DeepConf initialized: mode={mode}, "
            f"budget={budget if mode == 'offline' else total_budget}, "
            f"window={window_size}, filter={filter_method}"
        )

    def generate_trajectory(self, prompt) -> Dict[str, Any]:
        """
        Main entry point - generate traces and select best answer.

        Args:
            prompt: Input prompt/question (str or list of messages)

        Returns:
            Dictionary with trajectory, steps, and metadata
        """
        # Handle both string and message list formats
        if isinstance(prompt, list):
            # Extract user message from message list
            user_msg = next((m["content"] for m in prompt if m["role"] == "user"), None)
            if user_msg is None:
                raise ValueError("No user message found in prompt list")
            prompt = user_msg

        if self.mode == "offline":
            return self._generate_offline(prompt)
        elif self.mode == "online":
            return self._generate_online(prompt)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _generate_offline(self, prompt: str) -> Dict[str, Any]:
        """
        Offline mode: Generate all traces, then filter and vote.
        """
        log.info(f"🎯 DeepConf Offline: Generating {self.budget} traces...")

        traces = self._generate_traces_batch(prompt, self.budget)

        log.info(f"📊 Generated {len(traces)} traces")
        for i, trace in enumerate(traces):
            log.info(
                f"  Trace {i+1}: min_conf={trace['min_conf']:.3f}, "
                f"answer={trace['extracted_answer']}"
            )

        # Filter and vote
        result = self._filter_and_vote(traces)

        return {
            "trajectory": result["selected_text"],
            "steps": [result["selected_text"]],
            "validity_scores": [result["confidence_score"]],
            "completed": True,
            "metadata": {
                "mode": "offline",
                "total_traces": len(traces),
                "filtered_traces": len(result["filtered_traces"]),
                "confidence_score": result["confidence_score"],
                "selected_answer": result["selected_answer"],
                "vote_distribution": result["vote_distribution"],
            },
        }

    def _generate_online(self, prompt: str) -> Dict[str, Any]:
        """
        Online mode: Warmup to get threshold, then adaptive generation.
        """
        log.info(
            f"🎯 DeepConf Online: Warmup {self.warmup_traces}, "
            f"Total {self.total_budget} traces..."
        )

        # Phase 1: Warmup
        log.info("Phase 1: Warmup...")
        warmup_traces = self._generate_traces_batch(prompt, self.warmup_traces)

        # Compute confidence threshold
        warmup_min_confs = [t["min_conf"] for t in warmup_traces]
        conf_threshold = float(
            np.percentile(warmup_min_confs, 100 - self.confidence_percentile)
        )

        log.info(f"📊 Warmup complete: conf_threshold={conf_threshold:.3f}")
        log.info(f"   Min confs: {[f'{c:.3f}' for c in warmup_min_confs]}")

        # Phase 2: Adaptive generation with early stopping
        log.info("Phase 2: Adaptive generation...")
        remaining = self.total_budget - self.warmup_traces
        adaptive_traces = self._generate_traces_adaptive(
            prompt, remaining, conf_threshold
        )

        log.info(f"📊 Adaptive complete: generated {len(adaptive_traces)} traces")
        for i, trace in enumerate(adaptive_traces):
            log.info(
                f"  Trace {i+1}: min_conf={trace['min_conf']:.3f}, "
                f"stopped_early={trace.get('stopped_early', False)}, "
                f"answer={trace['extracted_answer']}"
            )

        # Combine all traces
        all_traces = warmup_traces + adaptive_traces

        # Filter and vote
        result = self._filter_and_vote(all_traces, conf_threshold)

        return {
            "trajectory": result["selected_text"],
            "steps": [result["selected_text"]],
            "validity_scores": [result["confidence_score"]],
            "completed": True,
            "metadata": {
                "mode": "online",
                "warmup_traces": len(warmup_traces),
                "adaptive_traces": len(adaptive_traces),
                "total_traces": len(all_traces),
                "filtered_traces": len(result["filtered_traces"]),
                "conf_threshold": conf_threshold,
                "confidence_score": result["confidence_score"],
                "selected_answer": result["selected_answer"],
                "vote_distribution": result["vote_distribution"],
            },
        }

    def _generate_traces_batch(self, prompt: str, n: int) -> List[Dict[str, Any]]:
        """
        Generate n traces using standard BlackboxModel generation.

        Returns:
            List of trace dictionaries with text, confidence, and answer
        """
        traces = []

        for i in range(n):
            try:
                # Prepare chat messages
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a math problem solver. Always put your "
                            "final numerical answer in \\boxed{answer} format."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]

                # Generate with logprobs
                texts = self.model.generate_texts(
                    [messages],
                    max_new_tokens=self.max_tokens,  # Use max_new_tokens, not max_tokens
                    temperature=self.temperature,
                    top_p=self.top_p,
                    output_scores=True,
                    top_logprobs=self.top_logprobs,
                )

                text = texts[0]

                # Extract logprobs
                token_confs, token_data = self._extract_logprobs_from_model()

                # Compute sliding window confidences
                window_confs = compute_sliding_window_confidence(
                    token_confs, self.window_size
                )
                min_conf = min(window_confs) if window_confs else 0.0

                # Extract answer
                extracted_answer = extract_answer(text)

                trace = {
                    "text": text,
                    "min_conf": min_conf,
                    "extracted_answer": extracted_answer,
                    "token_confs": token_confs,
                    "window_confs": window_confs,
                    "token_data": token_data,
                }

                traces.append(trace)

                log.info(
                    f"  Trace {i+1}/{n}: tokens={len(token_data)}, "
                    f"min_conf={min_conf:.3f}, answer={extracted_answer}"
                )

                # Debug: log text if no answer extracted
                if not extracted_answer:
                    log.warning(f"    No answer extracted. Text length: {len(text)}")
                    log.warning(f"    Text ends with: ...{text[-150:]}")

            except Exception as e:
                log.error(f"  ❌ Error generating trace {i+1}: {e}")
                continue

        return traces

    def _generate_traces_adaptive(
        self, prompt: str, n: int, conf_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Generate n traces with adaptive early stopping based on confidence.

        TODO: This requires streaming with logprobs - currently not implemented.
        For now, falls back to batch generation.
        """
        # TODO: Implement streaming with confidence-based early stopping
        # For now, use batch generation
        log.warning(
            "⚠️  Adaptive early stopping not yet implemented, " "using batch generation"
        )
        return self._generate_traces_batch(prompt, n)

    def _extract_logprobs_from_model(self) -> tuple:
        """
        Extract logprobs from model's stored data.

        Returns:
            (token_confidences, token_data)
        """
        token_confs = []
        token_data = []

        if hasattr(self.model, "logprobs") and len(self.model.logprobs) > 0:
            logprobs_obj = self.model.logprobs[0]

            if hasattr(logprobs_obj, "content") and logprobs_obj.content:
                for token_info in logprobs_obj.content:
                    # Build token data
                    token_entry = {
                        "token": token_info.token,
                        "logprob": token_info.logprob,
                        "top_logprobs": [
                            {"token": t.token, "logprob": t.logprob}
                            for t in token_info.top_logprobs
                        ],
                    }
                    token_data.append(token_entry)

                    # Compute confidence
                    conf = compute_token_confidence_from_logprobs(
                        token_entry["top_logprobs"], topk=self.top_logprobs
                    )
                    token_confs.append(conf)

        return token_confs, token_data

    def _filter_and_vote(
        self, traces: List[Dict[str, Any]], conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Filter traces by confidence and perform majority voting.

        Args:
            traces: List of trace dictionaries
            conf_threshold: Confidence threshold (optional)

        Returns:
            Dictionary with selected answer and metadata
        """
        if not traces:
            return {
                "selected_answer": "",
                "selected_text": "",
                "confidence_score": 0.0,
                "filtered_traces": [],
                "vote_distribution": {},
            }

        # Apply filtering
        if self.filter_method.startswith("top"):
            # TopK filtering
            k = int(self.filter_method[3:])  # e.g., "top10" -> 10
            filtered = sorted(traces, key=lambda t: t["min_conf"], reverse=True)[:k]
            log.info(f"🔍 Top-{k} filtering: {len(filtered)}/{len(traces)} traces")

        elif conf_threshold is not None or self.confidence_threshold is not None:
            # Threshold filtering
            threshold = conf_threshold or self.confidence_threshold
            filtered = [t for t in traces if t["min_conf"] >= threshold]
            log.info(
                f"🔍 Threshold filtering (≥{threshold:.3f}): "
                f"{len(filtered)}/{len(traces)} traces"
            )

        else:
            # No filtering
            filtered = traces
            log.info(f"🔍 No filtering: {len(filtered)} traces")

        if not filtered:
            log.warning("⚠️  No traces passed filter, using all traces")
            filtered = traces

        # Weighted majority voting (each trace votes with weight = min_conf)
        answers = [t["extracted_answer"] for t in filtered if t["extracted_answer"]]
        weights = [t["min_conf"] for t in filtered if t["extracted_answer"]]

        if not answers:
            log.warning("⚠️  No valid answers extracted")
            return {
                "selected_answer": "",
                "selected_text": filtered[0]["text"] if filtered else "",
                "confidence_score": 0.0,
                "filtered_traces": filtered,
                "vote_distribution": {},
            }

        # Sum weights per answer
        answer_weights = {}
        for answer, weight in zip(answers, weights):
            answer_weights[answer] = answer_weights.get(answer, 0.0) + weight

        # Select answer with highest total weight
        selected_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
        total_weight = sum(answer_weights.values())

        # Confidence = mean of all trace confidences (not voting-based)
        # Based on deepconf/utils.py:219
        confidence_score = sum(weights) / len(weights) if weights else 0.0

        # Get trace with selected answer
        selected_trace = None
        for trace in filtered:
            if trace["extracted_answer"] == selected_answer:
                selected_trace = trace
                break

        # Vote distribution (normalized weights)
        vote_dist = {
            ans: weight / total_weight for ans, weight in answer_weights.items()
        }

        log.info(f"🏆 Selected answer: '{selected_answer}'")
        log.info(f"   Confidence: {confidence_score:.3f}")
        log.info("   Vote distribution (weighted by min_conf):")
        for ans, pct in sorted(vote_dist.items(), key=lambda x: x[1], reverse=True):
            raw_weight = answer_weights[ans]
            log.info(f"     {ans}: {pct:.1%} (weight={raw_weight:.3f})")

        return {
            "selected_answer": selected_answer,
            "selected_text": selected_trace["text"] if selected_trace else "",
            "confidence_score": confidence_score,
            "filtered_traces": filtered,
            "vote_distribution": vote_dist,
        }
