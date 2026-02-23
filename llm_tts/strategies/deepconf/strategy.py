"""
DeepConf strategy - Confidence-based test-time scaling

Based on Facebook Research's DeepConf:
https://github.com/facebookresearch/deepconf

Uses framework's step_generator for generation and extracts logprobs from
StepCandidate.other_data["raw_logprobs"] for confidence computation.

Offline mode:
  1. Generate N complete traces via step_generator (single batched call)
  2. Compute per-token confidence from raw_logprobs
  3. Apply sliding window to get min_conf per trace
  4. Filter traces by confidence, majority vote on answers

Online mode:
  1. Generate traces in small batches (online_batch_size per round)
  2. After each batch, check agreement among filtered traces
  3. Stop early when agreement >= min_agreement or budget exhausted
  4. Different samples converge independently
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from llm_tts.generators.base import convert_trajectory_to_string
from llm_tts.utils import extract_answer

from ..metadata_builder import StrategyMetadataBuilder
from ..strategy_base import StrategyBase
from .utils import compute_sliding_window_confidence

if TYPE_CHECKING:
    from llm_tts.generators import StepCandidateGeneratorBase

log = logging.getLogger(__name__)


class StrategyDeepConf(StrategyBase):
    """
    DeepConf strategy using framework's step_generator.

    Generates N traces via generate_step_candidates_batch(), computes
    per-token confidence from raw vLLM logprobs, filters by confidence,
    and performs weighted majority voting.

    Supports two modes:
    - offline: Generate all traces in a single batched call
    - online: Generate traces in small batches with early stopping
    """

    def __init__(
        self,
        step_generator: "StepCandidateGeneratorBase",
        budget: int = 8,
        window_size: int = 2048,
        filter_method: str = "top10",
        confidence_threshold: float = None,
        data_name: str = None,
        mode: str = "offline",
        online_batch_size: int = 4,
        min_agreement: float = 0.8,
    ):
        """
        Initialize DeepConf strategy.

        Args:
            step_generator: Step generator for batched generation.
            budget: Number of traces to generate per sample.
            window_size: Sliding window size for confidence computation.
            filter_method: Filtering method ("topK", "threshold", or "top10").
            confidence_threshold: Manual confidence threshold (for threshold filtering).
            data_name: Dataset name for answer extraction.
            mode: "offline" (single batch) or "online" (incremental with early stopping).
            online_batch_size: Traces per batch in online mode.
            min_agreement: Agreement threshold for early stopping in online mode.
        """
        self.step_generator = step_generator
        self.budget = budget
        self.window_size = window_size
        self.filter_method = filter_method
        self.confidence_threshold = confidence_threshold
        self.data_name = data_name
        self.mode = mode
        self.online_batch_size = online_batch_size
        self.min_agreement = min_agreement

        log.info(
            f"DeepConf initialized: budget={budget}, window={window_size}, "
            f"filter={filter_method}, mode={mode}"
        )
        if mode == "online":
            log.info(
                f"  Online params: batch_size={online_batch_size}, "
                f"min_agreement={min_agreement}"
            )

    def _complete_thinking_paths(
        self,
        request: List[Dict[str, str]],
        candidates: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Complete thinking-mode candidates by generating answer phases.

        For each candidate that stopped at </think>, generates the answer phase
        via generate_answer_candidates_batch, producing a proper two-step trajectory.

        Args:
            request: Chat messages for the request
            candidates: List of StepCandidate objects from generation

        Returns:
            List of path dictionaries with text, num_tokens, and steps info
        """
        # Identify which candidates need answer generation
        thinking_indices = []
        for i, candidate in enumerate(candidates):
            if (
                getattr(self.step_generator, "thinking_mode", False)
                and candidate.is_thinking_complete
                and not candidate.is_trajectory_complete
            ):
                thinking_indices.append(i)

        # Batch generate all answer phases in one call
        answer_map = {}
        if thinking_indices:
            log.info(
                f"Generating {len(thinking_indices)} answer phases in batched call"
            )
            batch_requests = [request] * len(thinking_indices)
            batch_trajectories = [[candidates[i]] for i in thinking_indices]
            answer_results = self.step_generator.generate_answer_candidates_batch(
                batch_requests,
                batch_trajectories,
                candidates_per_step=1,
            )
            for batch_idx, cand_idx in enumerate(thinking_indices):
                if answer_results[batch_idx]:
                    answer_map[cand_idx] = answer_results[batch_idx][0]

        # Build paths from candidates + answers
        paths = []
        for i, candidate in enumerate(candidates):
            text = candidate.raw_text or candidate.text
            num_tokens = (candidate.other_data or {}).get(
                "original_token_count", len(candidate.token_ids)
            )

            if i in answer_map:
                answer_step = answer_map[i]
                answer_step.is_trajectory_complete = True
                trajectory = [candidate, answer_step]
                full_text = convert_trajectory_to_string(trajectory)
                answer_tokens = (
                    len(answer_step.token_ids) if answer_step.token_ids else 0
                )
                num_tokens += answer_tokens
                paths.append(
                    {
                        "text": full_text,
                        "num_tokens": num_tokens,
                        "steps": [candidate.text, answer_step.text],
                        "is_complete": True,
                    }
                )
                continue

            # Non-thinking or no </think>: single-step path
            paths.append(
                {
                    "text": text,
                    "num_tokens": num_tokens,
                    "steps": [text],
                    "is_complete": candidate.is_trajectory_complete,
                }
            )

        return paths

    def _compute_trace_confidence(self, raw_logprobs, window_size, top_k=20):
        """Compute sliding-window min confidence from vLLM raw_logprobs.

        Args:
            raw_logprobs: List[Dict[token_id -> Logprob]] from vLLM.
                Each dict maps token IDs to Logprob objects with .logprob attribute.
            window_size: Sliding window size.
            top_k: Number of top logprobs to use for confidence.

        Returns:
            (min_conf, token_confs, window_confs) tuple.
        """
        token_confs = []
        for token_logprobs_dict in raw_logprobs:
            if token_logprobs_dict is None:
                token_confs.append(float("inf"))
                continue
            # token_logprobs_dict: Dict[token_id -> Logprob]
            logprob_values = sorted(
                [lp.logprob for lp in token_logprobs_dict.values()],
                reverse=True,
            )[:top_k]
            valid = [lp for lp in logprob_values if lp > float("-inf")]
            if valid:
                token_confs.append(-sum(valid) / len(valid))
            else:
                token_confs.append(float("inf"))

        window_confs = compute_sliding_window_confidence(token_confs, window_size)
        min_conf = min(window_confs) if window_confs else 0.0
        return min_conf, token_confs, window_confs

    def _filter_and_vote(
        self, traces: List[Dict[str, Any]], conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Filter traces by confidence and perform majority voting.

        Args:
            traces: List of trace dictionaries with min_conf, extracted_answer, text.
            conf_threshold: Confidence threshold (optional).

        Returns:
            Dictionary with selected answer and metadata.
        """
        if not traces:
            return {
                "selected_answer": "",
                "selected_text": "",
                "confidence_score": 0.0,
                "filtered_traces": [],
                "vote_distribution": {},
                "agreement_rate": 0.0,
            }

        # Apply filtering
        if self.filter_method.startswith("top"):
            k = int(self.filter_method[3:])
            filtered = sorted(traces, key=lambda t: t["min_conf"], reverse=True)[:k]
            log.info(f"Top-{k} filtering: {len(filtered)}/{len(traces)} traces")

        elif conf_threshold is not None or self.confidence_threshold is not None:
            threshold = conf_threshold or self.confidence_threshold
            filtered = [t for t in traces if t["min_conf"] >= threshold]
            log.info(
                f"Threshold filtering (>={threshold:.3f}): "
                f"{len(filtered)}/{len(traces)} traces"
            )

        else:
            filtered = traces
            log.info(f"No filtering: {len(filtered)} traces")

        if not filtered:
            log.warning("No traces passed filter, using all traces")
            filtered = traces

        # Weighted majority voting (each trace votes with weight = min_conf)
        answers = [t["extracted_answer"] for t in filtered if t["extracted_answer"]]
        weights = [t["min_conf"] for t in filtered if t["extracted_answer"]]

        if not answers:
            log.warning("No valid answers extracted")
            return {
                "selected_answer": "",
                "selected_text": filtered[0]["text"] if filtered else "",
                "confidence_score": 0.0,
                "filtered_traces": filtered,
                "vote_distribution": {},
                "agreement_rate": 0.0,
            }

        # Sum weights per answer
        answer_weights = {}
        for answer, weight in zip(answers, weights):
            answer_weights[answer] = answer_weights.get(answer, 0.0) + weight

        # Select answer with highest total weight
        selected_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
        total_weight = sum(answer_weights.values())

        # Confidence = mean of all trace confidences
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

        # Agreement rate
        total_answers = len(answers)
        agreement_rate = (
            max(answers.count(ans) for ans in answer_weights.keys()) / total_answers
            if total_answers > 0
            else 0
        )

        log.info(f"Selected answer: '{selected_answer}'")
        log.info(f"   Confidence: {confidence_score:.3f}")
        log.info(f"   Agreement: {agreement_rate:.1%}")
        for ans, pct in sorted(vote_dist.items(), key=lambda x: x[1], reverse=True):
            count = answers.count(ans)
            log.info(f"     {ans}: {pct:.1%} (count={count})")

        return {
            "selected_answer": selected_answer,
            "selected_text": selected_trace["text"] if selected_trace else "",
            "confidence_score": confidence_score,
            "agreement_rate": agreement_rate,
            "filtered_traces": filtered,
            "vote_distribution": vote_dist,
        }

    def _process_candidates(
        self,
        request: List[Dict[str, str]],
        candidates: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Process candidates into trace dicts with confidence and answers.

        Handles thinking path completion, confidence computation, and answer
        extraction. Used by both offline and online modes.

        Args:
            request: Chat messages for the request.
            candidates: List of StepCandidate objects.

        Returns:
            List of trace dicts with text, min_conf, extracted_answer,
            token_confs, window_confs, num_tokens.
        """
        paths = self._complete_thinking_paths(request, candidates)

        traces = []
        for candidate, path in zip(candidates, paths):
            raw_logprobs = (candidate.other_data or {}).get("raw_logprobs", [])
            min_conf, token_confs, window_confs = self._compute_trace_confidence(
                raw_logprobs, self.window_size
            )

            extracted_answer = extract_answer(path["text"], answer_format="auto")

            traces.append(
                {
                    "text": path["text"],
                    "min_conf": min_conf,
                    "extracted_answer": extracted_answer,
                    "token_confs": token_confs,
                    "window_confs": window_confs,
                    "num_tokens": path["num_tokens"],
                }
            )

        return traces

    def _build_result(
        self,
        traces: List[Dict[str, Any]],
        sample_idx: int,
        token_stats: Dict[str, Any],
        mode: str = "offline",
        early_stopped: bool = False,
    ) -> Dict[str, Any]:
        """
        Build result dict from processed traces.

        Logs per-trace info, runs filter/vote, and assembles the final result
        dictionary. Used by both offline and online modes.

        Args:
            traces: List of trace dicts from _process_candidates.
            sample_idx: Sample index for logging.
            token_stats: Token statistics dict.
            mode: "offline" or "online".
            early_stopped: Whether the sample stopped early (online mode).

        Returns:
            Result dictionary with trajectory, metadata, etc.
        """
        if not traces:
            return self._empty_result()

        # Log trace info
        for i, trace in enumerate(traces):
            log.info(
                f"  Sample {sample_idx}, Trace {i + 1}/{len(traces)}: "
                f"tokens={trace['num_tokens']}, min_conf={trace['min_conf']:.3f}, "
                f"answer={trace['extracted_answer']}"
            )

        # Filter and vote
        vote_result = self._filter_and_vote(traces)

        token_stats["generation_count"] = len(traces)

        # Build metadata
        filtered_set = set(id(t) for t in vote_result["filtered_traces"])

        all_traces_details = []
        for i, trace in enumerate(traces):
            all_traces_details.append(
                {
                    "trace_id": i,
                    "text": trace["text"],
                    "min_conf": trace["min_conf"],
                    "mean_conf": (
                        sum(trace["token_confs"]) / len(trace["token_confs"])
                        if trace["token_confs"]
                        else 0.0
                    ),
                    "answer": trace["extracted_answer"],
                    "num_tokens": trace["num_tokens"],
                    "selected": id(trace) in filtered_set,
                    "window_confs": trace.get("window_confs", []),
                    "token_confs": trace.get("token_confs", []),
                }
            )

        builder = StrategyMetadataBuilder("deepconf")
        config_kwargs = dict(
            budget=self.budget,
            window_size=self.window_size,
            filter_method=self.filter_method,
            mode=mode,
        )
        if mode == "online":
            config_kwargs["online_batch_size"] = self.online_batch_size
            config_kwargs["min_agreement"] = self.min_agreement
        builder.add_config(**config_kwargs)
        builder.add_results(
            selected_answer=vote_result["selected_answer"],
            confidence_score=vote_result["confidence_score"],
            vote_distribution=vote_result["vote_distribution"],
            total_traces=len(traces),
            filtered_traces=len(vote_result["filtered_traces"]),
        )
        builder.add_generation_details(
            all_traces=all_traces_details,
        )

        result = {
            "trajectory": vote_result["selected_text"],
            "steps": [t["text"] for t in all_traces_details],
            "validity_scores": [t["min_conf"] for t in all_traces_details],
            "completed": bool(traces),
            "strategy": "deepconf",
            "extracted_answer": vote_result["selected_answer"],
            "consensus_score": vote_result["agreement_rate"],
            "vote_distribution": vote_result["vote_distribution"],
            "metadata": builder.build(),
            "all_traces": all_traces_details,
            "total_tokens": sum(t["num_tokens"] for t in traces),
            "token_stats": token_stats,
        }

        if mode == "online":
            result["early_stopped"] = early_stopped

        return result

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int] = None,
        save_callback: Callable = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate N traces for each of M samples using step_generator.

        Dispatches to offline or online mode based on self.mode.

        Args:
            requests: List of M chat message lists (one per sample).
            sample_indices: Optional list of sample indices for logging.
            save_callback: Optional callback for progressive saves.

        Returns:
            List of M result dictionaries (one per sample).
        """
        if sample_indices is None:
            sample_indices = list(range(len(requests)))

        if self.mode == "online":
            return self._generate_online(requests, sample_indices, save_callback)
        else:
            return self._generate_offline(requests, sample_indices, save_callback)

    def _build_stop_tokens(self):
        """Build stop tokens list based on generator config."""
        stop_tokens = ["<end of response>"]
        if (
            getattr(self.step_generator, "thinking_mode", False)
            and "</think>" not in stop_tokens
        ):
            stop_tokens.append("</think>")
        return stop_tokens

    def _generate_offline(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int],
        save_callback: Callable = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate traces in offline mode (single batched call).

        All M x N trajectories are generated in one call, then confidence is
        computed from raw_logprobs and filtering + voting is applied per sample.
        """
        M = len(requests)
        N = self.budget

        log.info(
            f"DeepConf offline: generating {M} samples x {N} traces = {M * N} "
            f"trajectories via generate_step_candidates_batch"
        )

        stop_tokens = self._build_stop_tokens()

        # Single batched call: M samples x N traces
        self.step_generator.reset_per_sample_stats()
        batch_results = self.step_generator.generate_step_candidates_batch(
            requests=requests,
            trajectories=[[]] * M,
            candidates_per_step=N,
            stop_tokens_override=stop_tokens,
            max_tokens=self.step_generator.generation_limit,
            compute_uncertainty=False,
            sample_ids=list(range(M)),
        )

        results = []
        for idx, (candidates, sample_idx) in enumerate(
            zip(batch_results, sample_indices)
        ):
            if not candidates:
                log.error(f"No output generated for sample {sample_idx}")
                results.append(self._empty_result())
                continue

            traces = self._process_candidates(requests[idx], candidates)
            token_stats = self.step_generator.get_sample_stats_for(idx)
            result = self._build_result(traces, sample_idx, token_stats)
            results.append(result)

        log.info(
            f"DeepConf offline: completed {len(results)} samples, "
            f"total {M * N} trajectories generated"
        )
        return results

    def _generate_online(
        self,
        requests: List[List[Dict[str, str]]],
        sample_indices: List[int],
        save_callback: Callable = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate traces in online mode with early stopping.

        Generates traces in small batches per round. After each round,
        checks agreement among filtered traces for each sample. Samples
        that reach sufficient agreement or exhaust their budget stop
        participating in future rounds.
        """
        M = len(requests)

        log.info(
            f"DeepConf online: {M} samples, batch_size={self.online_batch_size}, "
            f"budget={self.budget}, min_agreement={self.min_agreement}"
        )

        stop_tokens = self._build_stop_tokens()

        # Per-sample state
        all_traces = [[] for _ in range(M)]
        completed = [False] * M
        early_stopped = [False] * M

        self.step_generator.reset_per_sample_stats()
        round_num = 0

        while not all(completed):
            round_num += 1
            active_indices = [i for i in range(M) if not completed[i]]

            log.info(
                f"Online round {round_num}: generating for {len(active_indices)} "
                f"active samples"
            )

            # Generate batch for active samples only
            active_requests = [requests[i] for i in active_indices]

            batch_results = self.step_generator.generate_step_candidates_batch(
                requests=active_requests,
                trajectories=[[]] * len(active_indices),
                candidates_per_step=self.online_batch_size,
                stop_tokens_override=stop_tokens,
                max_tokens=self.step_generator.generation_limit,
                compute_uncertainty=False,
                sample_ids=active_indices,
            )

            for batch_idx, sample_idx in enumerate(active_indices):
                candidates = batch_results[batch_idx]
                if not candidates:
                    log.warning(
                        f"No candidates for sample {sample_indices[sample_idx]} "
                        f"in round {round_num}"
                    )
                    continue

                # Process new candidates and add to pool
                new_traces = self._process_candidates(
                    requests[sample_idx], candidates
                )
                all_traces[sample_idx].extend(new_traces)
                current_count = len(all_traces[sample_idx])

                # Check stopping conditions
                if current_count >= self.budget:
                    # Trim to budget if overshot
                    all_traces[sample_idx] = all_traces[sample_idx][: self.budget]
                    completed[sample_idx] = True
                    log.info(
                        f"Sample {sample_indices[sample_idx]}: budget reached "
                        f"({self.budget} traces)"
                    )
                else:
                    # Check agreement among filtered traces
                    vote_result = self._filter_and_vote(all_traces[sample_idx])
                    agreement = vote_result["agreement_rate"]
                    if agreement >= self.min_agreement:
                        completed[sample_idx] = True
                        early_stopped[sample_idx] = True
                        log.info(
                            f"Sample {sample_indices[sample_idx]}: early stop at "
                            f"{current_count} traces, agreement={agreement:.0%}"
                        )

        # Build final results
        results = []
        for i in range(M):
            token_stats = self.step_generator.get_sample_stats_for(i)
            result = self._build_result(
                all_traces[i],
                sample_indices[i],
                token_stats,
                mode="online",
                early_stopped=early_stopped[i],
            )
            results.append(result)

        total_traces = sum(len(t) for t in all_traces)
        early_count = sum(early_stopped)
        log.info(
            f"DeepConf online: completed {M} samples, {total_traces} total traces "
            f"({early_count} early stopped)"
        )
        return results

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for failed generation."""
        return {
            "trajectory": "",
            "steps": [],
            "validity_scores": [],
            "completed": False,
            "strategy": "deepconf",
            "extracted_answer": "",
            "consensus_score": 0.0,
            "vote_distribution": {},
            "metadata": {},
            "all_traces": [],
            "total_tokens": 0,
            "token_stats": {},
        }
