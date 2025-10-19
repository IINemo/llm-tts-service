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

from llm_tts.early_stopping import ConfidenceEarlyStopping

from ..metadata_builder import StrategyMetadataBuilder
from ..strategy_base import StrategyBase
from .utils import (
    compute_sliding_window_confidence,
    compute_token_confidence_from_logprobs,
    extract_answer,
)

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
        n_threads: int = 8,
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
            n_threads: Number of threads for parallel API requests (default: 8)
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
        self.n_threads = n_threads

        # Validate model supports logprobs
        if not hasattr(model, "supports_logprobs") or not model.supports_logprobs:
            raise ValueError("Model must support logprobs for DeepConf")

        log.info(
            f"DeepConf initialized: mode={mode}, "
            f"budget={budget if mode == 'offline' else total_budget}, "
            f"window={window_size}, filter={filter_method}, threads={n_threads}"
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

        # Build enhanced metadata
        filtered_set = set(id(t) for t in result["filtered_traces"])

        # Summary stats for all traces
        trace_summaries = []
        for i, trace in enumerate(traces):
            trace_summaries.append(
                {
                    "trace_id": i,
                    "min_conf": trace["min_conf"],
                    "mean_conf": (
                        sum(trace["token_confs"]) / len(trace["token_confs"])
                        if trace["token_confs"]
                        else 0.0
                    ),
                    "answer": trace["extracted_answer"],
                    "num_tokens": len(trace.get("token_data", [])),
                    "selected": id(trace) in filtered_set,
                }
            )

        # Full details for filtered traces only
        filtered_trace_details = []
        for trace in result["filtered_traces"]:
            trace_idx = traces.index(trace)
            filtered_trace_details.append(
                {
                    "trace_id": trace_idx,
                    "text": trace["text"],
                    "min_conf": trace["min_conf"],
                    "answer": trace["extracted_answer"],
                    "num_tokens": len(trace.get("token_data", [])),
                }
            )

        # Build metadata using StrategyMetadataBuilder
        builder = StrategyMetadataBuilder("deepconf")

        # Add configuration
        builder.add_config(
            mode="offline",
            budget=self.budget,
            window_size=self.window_size,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs,
            filter_method=self.filter_method,
            n_threads=self.n_threads,
        )

        # Add results
        builder.add_results(
            selected_answer=result["selected_answer"],
            confidence_score=result["confidence_score"],
            vote_distribution=result["vote_distribution"],
            total_traces=len(traces),
            filtered_traces=len(result["filtered_traces"]),
        )

        # Add generation details
        builder.add_generation_details(
            trace_summaries=trace_summaries,
            filtered_trace_details=filtered_trace_details,
        )

        # Log summary to console
        builder.log_summary(log)

        return {
            "trajectory": result["selected_text"],
            "steps": [result["selected_text"]],
            "validity_scores": [result["confidence_score"]],
            "completed": True,
            "metadata": builder.build(),
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

        # Clear any existing early stopping for warmup phase
        self.model.early_stopping = None

        warmup_traces = self._generate_traces_batch(prompt, self.warmup_traces)

        # Compute confidence threshold
        warmup_min_confs = [t["min_conf"] for t in warmup_traces]
        conf_threshold = float(
            np.percentile(warmup_min_confs, 100 - self.confidence_percentile)
        )

        # Log warmup statistics
        log.info(f"📊 Warmup complete: conf_threshold={conf_threshold:.3f}")
        log.info(f"   Min confs: {[f'{c:.3f}' for c in warmup_min_confs]}")
        log.info(
            f"   Mean: {np.mean(warmup_min_confs):.3f}, "
            f"Median: {np.median(warmup_min_confs):.3f}, "
            f"Std: {np.std(warmup_min_confs):.3f}"
        )
        log.info(
            f"   Percentile: {100 - self.confidence_percentile}th = {conf_threshold:.3f}"
        )

        # Phase 2: Adaptive generation with early stopping
        log.info("Phase 2: Adaptive generation...")
        remaining = self.total_budget - self.warmup_traces
        adaptive_traces = self._generate_traces_adaptive(
            prompt, remaining, conf_threshold
        )

        # Calculate adaptive phase statistics
        num_stopped_early = sum(
            1 for t in adaptive_traces if t.get("stopped_early", False)
        )
        num_completed = len(adaptive_traces) - num_stopped_early
        adaptive_tokens = sum(len(t.get("token_data", [])) for t in adaptive_traces)
        warmup_tokens = sum(len(t.get("token_data", [])) for t in warmup_traces)

        log.info(f"📊 Adaptive complete: generated {len(adaptive_traces)} traces")
        log.info(f"   Stopped early: {num_stopped_early}/{len(adaptive_traces)}")
        log.info(f"   Completed: {num_completed}/{len(adaptive_traces)}")
        log.info(f"   Tokens: warmup={warmup_tokens}, adaptive={adaptive_tokens}")

        if warmup_traces and adaptive_traces:
            avg_warmup_tokens = warmup_tokens / len(warmup_traces)
            avg_adaptive_tokens = adaptive_tokens / len(adaptive_traces)
            if num_stopped_early > 0:
                early_stopped = [
                    t for t in adaptive_traces if t.get("stopped_early", False)
                ]
                avg_early_stop_tokens = (
                    sum(len(t.get("token_data", [])) for t in early_stopped)
                    / num_stopped_early
                )
                potential_tokens = num_stopped_early * avg_warmup_tokens
                saved_tokens = potential_tokens - sum(
                    len(t.get("token_data", [])) for t in early_stopped
                )
                savings_pct = (
                    (saved_tokens / potential_tokens * 100)
                    if potential_tokens > 0
                    else 0
                )
                log.info(
                    f"   Avg tokens: warmup={avg_warmup_tokens:.1f}, \
                        adaptive={avg_adaptive_tokens:.1f}"
                )
                log.info(f"   Early stop avg: {avg_early_stop_tokens:.1f} tokens")
                log.info(f"   Token savings: {saved_tokens:.0f} ({savings_pct:.1f}%)")

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

        # Build enhanced metadata (consistent with offline mode)
        filtered_set = set(id(t) for t in result["filtered_traces"])

        # Summary stats for all traces
        trace_summaries = []
        for i, trace in enumerate(all_traces):
            trace_summaries.append(
                {
                    "trace_id": i,
                    "min_conf": trace["min_conf"],
                    "mean_conf": (
                        sum(trace["token_confs"]) / len(trace["token_confs"])
                        if trace["token_confs"]
                        else 0.0
                    ),
                    "answer": trace["extracted_answer"],
                    "num_tokens": len(trace.get("token_data", [])),
                    "selected": id(trace) in filtered_set,
                    "phase": "warmup" if i < len(warmup_traces) else "adaptive",
                    "stopped_early": trace.get("stopped_early", False),
                }
            )

        # Full details for filtered traces only
        filtered_trace_details = []
        for trace in result["filtered_traces"]:
            trace_idx = all_traces.index(trace)
            filtered_trace_details.append(
                {
                    "trace_id": trace_idx,
                    "text": trace["text"],
                    "min_conf": trace["min_conf"],
                    "answer": trace["extracted_answer"],
                    "num_tokens": len(trace.get("token_data", [])),
                    "phase": "warmup" if trace_idx < len(warmup_traces) else "adaptive",
                    "stopped_early": trace.get("stopped_early", False),
                }
            )

        # Build metadata using StrategyMetadataBuilder
        builder = StrategyMetadataBuilder("deepconf")

        # Add configuration
        builder.add_config(
            mode="online",
            window_size=self.window_size,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs,
            filter_method=self.filter_method,
            n_threads=self.n_threads,
        )

        # Add results
        builder.add_results(
            selected_answer=result["selected_answer"],
            confidence_score=result["confidence_score"],
            vote_distribution=result["vote_distribution"],
            total_traces=len(all_traces),
            filtered_traces=len(result["filtered_traces"]),
        )

        # Add generation details
        builder.add_generation_details(
            trace_summaries=trace_summaries,
            filtered_trace_details=filtered_trace_details,
        )

        # Add online-specific metadata
        builder.add_strategy_specific(
            warmup_traces=len(warmup_traces),
            adaptive_traces=len(adaptive_traces),
            conf_threshold=conf_threshold,
            confidence_percentile=self.confidence_percentile,
        )

        # Log summary to console
        builder.log_summary(log)

        return {
            "trajectory": result["selected_text"],
            "steps": [result["selected_text"]],
            "validity_scores": [result["confidence_score"]],
            "completed": True,
            "metadata": builder.build(),
        }

    def _generate_single_trace(self, args: tuple) -> Optional[Dict[str, Any]]:
        """
        Generate a single trace with logprobs (for multithreading).

        Args:
            args: Tuple of (prompt, trace_index, total_traces)

        Returns:
            Trace dictionary with text, confidence, and answer, or None if error
        """
        prompt, i, n = args
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
            # Disable early stopping for batch generation
            self.model.early_stopping = None

            results = self.model.generate_texts(
                chats=[messages],
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                output_scores=True,
                top_logprobs=self.top_logprobs,
            )

            # Extract text and logprobs from result
            text = results[0]["text"]
            logprobs_data = results[0].get("logprobs", [])

            # Extract confidences from logprobs
            token_confs = []
            for token_info in logprobs_data:
                # Use mean of top-k logprobs (same as online mode)
                top_logprobs_list = token_info["top_logprobs"][: self.top_logprobs]
                mean_logprob = sum(t["logprob"] for t in top_logprobs_list) / len(
                    top_logprobs_list
                )
                token_confs.append(-mean_logprob)

            token_data = logprobs_data

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

            log.info(
                f"  Trace {i+1}/{n}: tokens={len(token_data)}, "
                f"min_conf={min_conf:.3f}, answer={extracted_answer}"
            )

            # Debug: log text if no answer extracted
            if not extracted_answer:
                log.warning(f"    No answer extracted. Text length: {len(text)}")
                log.warning(f"    Text ends with: ...{text[-150:]}")

            return trace

        except Exception as e:
            log.error(f"  Error generating trace {i+1}: {e}")
            return None

    def _generate_traces_batch(self, prompt: str, n: int) -> List[Dict[str, Any]]:
        """
        Generate n traces using parallel API requests.

        Returns:
            List of trace dictionaries with text, confidence, and answer
        """
        # Prepare arguments for each trace: (prompt, index, total)
        trace_args = [(prompt, i, n) for i in range(n)]

        # Use base class parallel generation with our trace-specific worker
        return self._parallel_generate(
            worker_func=self._generate_single_trace,
            task_args=trace_args,
            n_threads=self.n_threads,
            desc=f"Generating {n} traces",
        )

    def _generate_traces_adaptive(
        self, prompt: str, n: int, conf_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Generate n traces with adaptive early stopping based on confidence.

        Uses streaming + logprobs to stop generation when confidence drops.
        """
        log.info(
            f"🔄 Generating {n} adaptive traces with confidence "
            f"threshold={conf_threshold:.3f}"
        )

        traces = []

        for i in range(n):
            try:
                # Prepare messages
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

                # Configure model with confidence-based early stopping
                self.model.early_stopping = ConfidenceEarlyStopping(
                    threshold=conf_threshold,
                    window_size=self.window_size,
                    top_k=self.top_logprobs,
                    method="mean_logprob",  # Same formula as warmup phase
                )

                # Generate with early stopping configured in model
                results = self.model.generate_texts(
                    [messages],
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                result = results[0]
                text = result["text"]
                logprobs_data = result["logprobs"]
                stopped_early = result.get("stopped_early", False)

                # Compute token confidences from logprobs
                token_confs = []
                for token_info in logprobs_data:
                    conf = compute_token_confidence_from_logprobs(
                        token_info["top_logprobs"], topk=self.top_logprobs
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
                    "min_conf": min_conf,
                    "extracted_answer": extracted_answer,
                    "token_confs": token_confs,
                    "window_confs": window_confs,
                    "token_data": logprobs_data,
                    "stopped_early": stopped_early,
                }

                traces.append(trace)

                early_str = " (stopped early)" if stopped_early else ""
                log.info(
                    f"  Trace {i+1}/{n}: tokens={len(logprobs_data)}, "
                    f"min_conf={min_conf:.3f}, answer={extracted_answer}{early_str}"
                )

                # Debug: log text if no answer extracted
                if not extracted_answer:
                    log.warning(f"    No answer extracted. Text length: {len(text)}")
                    log.warning(f"    Text ends with: ...{text[-150:]}")

            except Exception as e:
                log.error(f"  ❌ Error generating adaptive trace {i+1}: {e}")
                import traceback

                log.error(traceback.format_exc())
                continue

        return traces

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

        # Calculate answer statistics
        num_unique_answers = len(answer_weights)
        total_answers = len(answers)
        agreement_rate = (
            max(answers.count(ans) for ans in answer_weights.keys()) / total_answers
            if total_answers > 0
            else 0
        )

        log.info(f"🏆 Selected answer: '{selected_answer}'")
        log.info(f"   Confidence: {confidence_score:.3f}")
        log.info(f"   Answers: {total_answers} total, {num_unique_answers} unique")
        log.info(
            f"   Agreement: {agreement_rate:.1%} \
                ({max(answers.count(ans) for ans in answer_weights.keys())}/{total_answers} traces)"
        )
        log.info("   Vote distribution (weighted by min_conf):")
        for ans, pct in sorted(vote_dist.items(), key=lambda x: x[1], reverse=True):
            raw_weight = answer_weights[ans]
            count = answers.count(ans)
            log.info(f"     {ans}: {pct:.1%} (weight={raw_weight:.3f}, count={count})")

        return {
            "selected_answer": selected_answer,
            "selected_text": selected_trace["text"] if selected_trace else "",
            "confidence_score": confidence_score,
            "filtered_traces": filtered,
            "vote_distribution": vote_dist,
        }
