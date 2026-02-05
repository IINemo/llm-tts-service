"""
ST-BoN (Self-Truncation Best-of-N) strategy for efficient self-consistency.

Based on "Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N
Sampling in Early Decoding" (Wang et al., NeurIPS 2025).

ST-BoN improves upon standard self-consistency by:
1. Using early decoding consistency to identify promising samples
2. Truncating suboptimal samples before full generation
3. Reducing memory and latency by 50-80% while maintaining accuracy

Algorithm (3 steps):
1. Generate N samples until earliest estimation time c (all pairwise inconsistent)
2. Continue for buffer window τ = m*c, performing self-estimation at each step
3. Truncate N-1 suboptimal samples, complete only the best estimated sample

Key features:
- For HuggingFace: Uses Chain-of-Embedding (CoE) for internal consistency
- For vLLM: Uses semantic or string similarity as fallback
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from llm_tts.generators.base import StepCandidate, convert_trajectory_to_string
from llm_tts.scorers.coe_scorer import (
    CoEScorer,
    SemanticSimilarityScorer,
    StringSimilarityScorer,
)
from llm_tts.scorers.majority_voting import ChainMajorityVotingScorer
from llm_tts.utils import extract_answer

from .metadata_builder import StrategyMetadataBuilder
from .strategy_base import StrategyBase, count_thinking_and_response_steps

if TYPE_CHECKING:
    from llm_tts.generators import StepCandidateGeneratorBase

log = logging.getLogger(__name__)


class StrategySTBoN(StrategyBase):
    """
    Self-Truncation Best-of-N (ST-BoN) strategy.

    Efficiently generates multiple reasoning paths by truncating suboptimal
    samples early based on internal consistency signals.

    Supports two modes:
    - HuggingFace mode: Uses hidden states for CoE-based consistency
    - vLLM mode: Uses semantic/string similarity (no hidden states available)
    """

    def __init__(
        self,
        step_generator: "StepCandidateGeneratorBase",
        num_paths: int = 10,
        max_steps: int = 250,
        buffer_multiplier: float = 1.0,
        similarity_mode: str = "auto",
        scorer: Optional[Any] = None,
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize ST-BoN strategy.

        Args:
            step_generator: Generator for step-by-step generation
            num_paths: Number of reasoning paths to generate (N)
            max_steps: Maximum steps per path before forced completion
            buffer_multiplier: m value in τ = m*c (default 1.0)
            similarity_mode: How to measure consistency:
                - "coe": Chain-of-Embedding using hidden states (HuggingFace)
                - "semantic": Sentence embeddings (vLLM/API)
                - "string": Rouge-L string similarity (lightweight)
                - "auto": Auto-detect based on generator capabilities
            scorer: Custom scorer for final answer selection (default: majority voting)
            semantic_model: Model for semantic similarity mode
        """
        self.step_generator = step_generator
        self.num_paths = num_paths
        self.max_steps = max_steps
        self.buffer_multiplier = buffer_multiplier
        self.similarity_mode = similarity_mode
        self.semantic_model = semantic_model

        self._init_similarity_scorer()

        self.scorer = scorer or ChainMajorityVotingScorer()
        if hasattr(self.scorer, "prepare_model"):
            self.scorer.prepare_model()

    def _init_similarity_scorer(self):
        """Initialize the appropriate similarity scorer based on mode."""
        mode = self.similarity_mode

        if mode == "auto":
            has_hidden_states = (
                hasattr(self.step_generator, "capture_hidden_states")
                and self.step_generator.capture_hidden_states
            )
            if has_hidden_states:
                mode = "coe"
                log.info(
                    "ST-BoN: Auto-detected hidden states support, using CoE scorer"
                )
            else:
                mode = "semantic"
                log.info("ST-BoN: No hidden states, using semantic similarity scorer")

        if mode == "coe":
            self.consistency_scorer = CoEScorer()
            self._use_hidden_states = True
        elif mode == "semantic":
            self.consistency_scorer = SemanticSimilarityScorer(
                model_name=self.semantic_model
            )
            self._use_hidden_states = False
        elif mode == "string":
            self.consistency_scorer = StringSimilarityScorer()
            self._use_hidden_states = False
        else:
            raise ValueError(f"Unknown similarity_mode: {mode}")

        self._actual_mode = mode

    def _are_all_pairwise_inconsistent(self, texts: List[str]) -> bool:
        """
        Check if all N samples are pairwise inconsistent (no two are identical).

        This determines the earliest estimation time c from Eq. 3 in the paper:
        Σ_{i,j,i≠j} I(Y_i = Y_j) = 0

        Args:
            texts: List of current sample texts

        Returns:
            True if all samples are pairwise different
        """
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                if texts[i] == texts[j]:
                    return False
        return True

    def _compute_consistency_scores(
        self,
        trajectories: List[List[StepCandidate]],
    ) -> List[float]:
        """
        Compute consistency scores for all samples at current step.

        For CoE mode: Uses hidden states from trajectories
        For semantic/string mode: Uses text content

        Args:
            trajectories: List of N trajectories (each is a list of StepCandidates)

        Returns:
            List of consistency scores (lower is more consistent)
        """
        if self._use_hidden_states:
            samples_hidden_states = []
            for trajectory in trajectories:
                all_layer_states = []
                for step in trajectory:
                    if (
                        step.other_data
                        and "hidden_state_mean_per_layer" in step.other_data
                    ):
                        layer_states = step.other_data["hidden_state_mean_per_layer"]
                        all_layer_states.append(layer_states)

                if all_layer_states:
                    num_layers = len(all_layer_states[0])
                    avg_per_layer = []
                    for layer in range(num_layers):
                        layer_values = [
                            np.array(step_states[layer])
                            for step_states in all_layer_states
                        ]
                        avg_per_layer.append(np.mean(layer_values, axis=0))
                    samples_hidden_states.append(avg_per_layer)
                else:
                    log.warning("No hidden states found for trajectory, using zeros")
                    samples_hidden_states.append([np.zeros(1)])

            coe_features = self.consistency_scorer.compute_features_from_hidden_states(
                samples_hidden_states
            )
            return self.consistency_scorer.compute_sample_distances(coe_features)

        else:
            texts = [convert_trajectory_to_string(t) for t in trajectories]
            return self.consistency_scorer.compute_consistency_scores(texts)

    def _select_best_at_step(
        self,
        trajectories: List[List[StepCandidate]],
    ) -> int:
        """
        Select the best sample at current step based on consistency.

        Args:
            trajectories: List of N trajectories

        Returns:
            Index of the best sample (most consistent with others)
        """
        scores = self._compute_consistency_scores(trajectories)
        return int(np.argmin(scores))

    def _generate_with_early_truncation(
        self, request: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Main ST-BoN generation loop with early truncation.

        Implements the 3-step algorithm from the paper:
        1. Generate until earliest estimation time c
        2. Continue for buffer window τ, collecting estimations
        3. Truncate and complete the best sample

        Args:
            request: Chat messages in OpenAI format

        Returns:
            Dictionary with generation results
        """
        log.info(f"ST-BoN: Starting generation with N={self.num_paths} paths")

        trajectories = [[] for _ in range(self.num_paths)]
        path_tokens = [0 for _ in range(self.num_paths)]
        active_paths = set(range(self.num_paths))

        earliest_estimation_time = None
        buffer_window_size = None
        estimation_votes = []
        completed_early = set()

        for step_num in range(self.max_steps):
            if not active_paths:
                break

            active_indices = sorted(active_paths)
            active_trajectories = [trajectories[i] for i in active_indices]

            # Handle different generator APIs:
            # - vLLM: accepts List[List[StepCandidate]], returns List[List[StepCandidate]]
            # - HuggingFace: accepts List[StepCandidate], returns List[StepCandidate]
            generator_name = type(self.step_generator).__name__.lower()
            if "vllm" in generator_name:
                # vLLM batch API
                step_candidates_nested = self.step_generator.generate_step_candidates(
                    request, active_trajectories, candidates_per_step=1
                )
                step_candidates = [cands[0] for cands in step_candidates_nested]
            else:
                # HuggingFace single-trajectory API - loop over trajectories
                step_candidates = []
                for traj in active_trajectories:
                    cands = self.step_generator.generate_step_candidates(
                        request, traj, candidates_per_step=1
                    )
                    step_candidates.append(cands[0])

            newly_completed = []
            for idx, path_idx in enumerate(active_indices):
                candidate = step_candidates[idx]
                if isinstance(candidate, list):
                    candidate = candidate[0] if candidate else None
                if candidate is None:
                    log.warning(
                        "ST-BoN: Empty candidate list for path %s at step %s",
                        path_idx,
                        step_num,
                    )
                    active_paths.discard(path_idx)
                    continue
                trajectories[path_idx].append(candidate)

                step_tokens = len(candidate.token_ids) if candidate.token_ids else 0
                path_tokens[path_idx] += step_tokens

                if candidate.is_trajectory_complete:
                    newly_completed.append(path_idx)
                    completed_early.add(path_idx)

            for path_idx in newly_completed:
                active_paths.discard(path_idx)

            if earliest_estimation_time is None and len(active_paths) >= 2:
                texts = [
                    convert_trajectory_to_string(trajectories[i]) for i in active_paths
                ]
                if self._are_all_pairwise_inconsistent(texts):
                    earliest_estimation_time = step_num
                    buffer_window_size = int(step_num * self.buffer_multiplier)
                    log.info(
                        f"ST-BoN: Earliest estimation time c={earliest_estimation_time}, "
                        f"buffer τ={buffer_window_size}"
                    )

            if (
                earliest_estimation_time is not None
                and step_num >= earliest_estimation_time
                and step_num <= earliest_estimation_time + buffer_window_size
                and len(active_paths) >= 2
            ):
                active_indices_for_vote = sorted(active_paths)
                active_trajs = [trajectories[i] for i in active_indices_for_vote]
                scores = self._compute_consistency_scores(active_trajs)
                best_local_idx = int(np.argmin(scores))
                best_path_idx = active_indices_for_vote[best_local_idx]
                estimation_votes.append(best_path_idx)

                log.debug(
                    f"ST-BoN: Step {step_num}, best estimation: path {best_path_idx}"
                )

            if (
                earliest_estimation_time is not None
                and step_num == earliest_estimation_time + buffer_window_size
                and len(active_paths) > 1
            ):
                if estimation_votes:
                    vote_counts = Counter(estimation_votes)
                    best_path = vote_counts.most_common(1)[0][0]
                    log.info(
                        f"ST-BoN: Truncating at step {step_num}, "
                        f"keeping path {best_path} (votes: {dict(vote_counts)})"
                    )

                    paths_to_truncate = active_paths - {best_path}
                    for path_idx in paths_to_truncate:
                        active_paths.discard(path_idx)

            if (step_num + 1) % 10 == 0:
                log.info(
                    f"ST-BoN: Step {step_num + 1}, {len(active_paths)} active paths"
                )

        for path_idx in list(active_paths):
            trajectory = trajectories[path_idx]
            if trajectory and not trajectory[-1].is_trajectory_complete:
                trace_text = convert_trajectory_to_string(trajectory)
                if "<Answer>:" not in trace_text:
                    answer_candidates = self.step_generator.generate_answer_candidates(
                        request, trajectory, candidates_per_step=1
                    )
                    if answer_candidates:
                        trajectories[path_idx].append(answer_candidates[0])
                        path_tokens[path_idx] += len(
                            answer_candidates[0].token_ids or []
                        )

        paths = []
        for i in range(self.num_paths):
            trajectory = trajectories[i]
            trace_text = convert_trajectory_to_string(trajectory)
            thinking_steps, response_steps = count_thinking_and_response_steps(
                trajectory
            )

            answer = extract_answer(trace_text, answer_format="auto") or "no_answer"

            paths.append(
                {
                    "text": trace_text,
                    "num_tokens": path_tokens[i],
                    "steps": [s.text for s in trajectory],
                    "thinking_steps": thinking_steps,
                    "response_steps": response_steps,
                    "answer": answer,
                    "was_truncated": i not in active_paths and i not in completed_early,
                    "completed_early": i in completed_early,
                }
            )

        return {
            "paths": paths,
            "earliest_estimation_time": earliest_estimation_time,
            "buffer_window_size": buffer_window_size,
            "estimation_votes": estimation_votes,
            "total_tokens": sum(path_tokens),
        }

    def select_best_answer(self, generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best answer using majority voting on completed paths.

        Args:
            generation_result: Result from _generate_with_early_truncation

        Returns:
            Dictionary with selection results
        """
        paths = generation_result["paths"]

        completed_paths = [p for p in paths if not p.get("was_truncated", False)]

        if not completed_paths:
            log.warning("ST-BoN: No completed paths, using all paths")
            completed_paths = paths

        path_texts = [p["text"] for p in completed_paths]
        scores = self.scorer.score_complete_chains(path_texts)

        best_idx = int(np.argmax(scores))
        best_path = completed_paths[best_idx]
        best_answer = self.scorer.extract_answer(best_path["text"])

        all_answers = [self.scorer.extract_answer(p["text"]) for p in completed_paths]
        answer_counts = Counter(all_answers)

        log.info(f"ST-BoN: Selected path with answer: {best_answer}")
        log.info(f"ST-BoN: Answer distribution: {dict(answer_counts)}")

        return {
            "best_path": best_path,
            "best_answer": best_answer,
            "consensus_score": float(scores[best_idx]),
            "all_answers": all_answers,
            "answer_distribution": dict(answer_counts),
            "completed_paths": completed_paths,
        }

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Main entry point for ST-BoN reasoning.

        Args:
            request: Chat messages in OpenAI format
            sample_idx: Index of current sample (for logging)

        Returns:
            Dictionary with trajectory information
        """
        prompt_preview = request[0]["content"][:100] if request else ""
        log.info(f"ST-BoN: Starting reasoning for: {prompt_preview}...")

        self.step_generator.reset_sample_stats()

        generation_result = self._generate_with_early_truncation(request)

        selection_result = self.select_best_answer(generation_result)

        self.step_generator.finalize_sample_stats()
        token_stats = self.step_generator.get_sample_stats()

        builder = StrategyMetadataBuilder("st_bon")

        builder.add_config(
            num_paths=self.num_paths,
            max_steps=self.max_steps,
            buffer_multiplier=self.buffer_multiplier,
            similarity_mode=self._actual_mode,
        )

        builder.add_results(
            selected_answer=selection_result["best_answer"],
            consensus_score=selection_result["consensus_score"],
            answer_distribution=selection_result["answer_distribution"],
        )

        # Add ST-BoN specific metrics
        builder.add_strategy_specific(
            earliest_estimation_time=generation_result["earliest_estimation_time"],
            buffer_window_size=generation_result["buffer_window_size"],
            estimation_votes=generation_result["estimation_votes"],
            truncated_paths=sum(
                1 for p in generation_result["paths"] if p.get("was_truncated", False)
            ),
            completed_early_paths=sum(
                1 for p in generation_result["paths"] if p.get("completed_early", False)
            ),
        )

        builder.log_summary(log)

        completed_paths = selection_result["completed_paths"]
        avg_thinking_steps = sum(
            p.get("thinking_steps", 0) for p in completed_paths
        ) / max(len(completed_paths), 1)
        avg_response_steps = sum(
            p.get("response_steps", 0) for p in completed_paths
        ) / max(len(completed_paths), 1)

        all_traces = []
        for i, path in enumerate(generation_result["paths"]):
            all_traces.append(
                {
                    "text": path["text"],
                    "num_tokens": path["num_tokens"],
                    "num_steps": len(path.get("steps", [])),
                    "thinking_steps": path.get("thinking_steps", 0),
                    "response_steps": path.get("response_steps", 0),
                    "answer": path.get("answer", "no_answer"),
                    "was_truncated": path.get("was_truncated", False),
                    "selected": path == selection_result["best_path"],
                }
            )

        return {
            "trajectory": selection_result["best_path"]["text"],
            "steps": [selection_result["best_path"]["text"]],
            "validity_scores": [selection_result["consensus_score"]],
            "completed": bool(completed_paths),
            "strategy": "st_bon",
            "extracted_answer": selection_result["best_answer"],
            "metadata": builder.build(),
            "all_traces": all_traces,
            "total_tokens": generation_result["total_tokens"],
            "token_stats": token_stats,
            "thinking_num_steps": avg_thinking_steps,
            "response_num_steps": avg_response_steps,
        }

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
        if hasattr(self.consistency_scorer, "cleanup"):
            self.consistency_scorer.cleanup()
