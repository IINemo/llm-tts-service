"""
LLM Critic Scorer based on Tree of Thoughts paper.

Implements the State Evaluator from "Tree of Thoughts: Deliberate Problem Solving
with Large Language Models" (Yao et al., 2023).

Two evaluation strategies:
1. Value: Evaluates each step independently, returns sure/likely/unlikely/impossible
2. Vote: Presents multiple candidates and asks which is most promising

Paper: https://arxiv.org/abs/2305.10601

Supports both:
- vLLM backend (for local GPU inference)
- OpenAI-compatible API (OpenRouter, OpenAI, etc.)
"""

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_tts.utils.flops import FLOPCalculator

from .step_scorer_base import CandidateScore, StepScorerBase

log = logging.getLogger(__name__)


class _ScorerCircuitBreaker:
    """Circuit breaker for scorer API calls.

    Tracks consecutive failures across all scorer threads.  When enough
    failures accumulate, imposes an exponential cooldown so that the scorer
    does not waste minutes retrying a dead API.

    Thread-safe: uses a lock around all mutable state.
    """

    def __init__(
        self,
        max_consecutive_failures: int = 15,
        max_total_downtime: float = 300.0,
        initial_cooldown: float = 5.0,
        max_cooldown: float = 60.0,
    ):
        self._lock = threading.Lock()
        self.max_consecutive_failures = max_consecutive_failures
        self.max_total_downtime = max_total_downtime
        self.initial_cooldown = initial_cooldown
        self.max_cooldown = max_cooldown

        self._consecutive_failures = 0
        self._total_downtime = 0.0
        self._tripped = False

    def record_success(self):
        with self._lock:
            self._consecutive_failures = 0
            self._total_downtime = 0.0
            self._tripped = False

    def record_failure(self) -> bool:
        """Record a failure. Returns True if we should keep retrying, False to abort."""
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.max_consecutive_failures:
                self._tripped = True
                log.error(
                    f"[SCORER CB] Circuit breaker tripped after "
                    f"{self._consecutive_failures} consecutive failures"
                )
                return False
            if self._total_downtime >= self.max_total_downtime:
                self._tripped = True
                log.error(
                    f"[SCORER CB] Circuit breaker tripped: "
                    f"total downtime {self._total_downtime:.0f}s >= {self.max_total_downtime}s"
                )
                return False
            return True

    def should_attempt(self) -> bool:
        """Check if we should attempt an API call."""
        with self._lock:
            return not self._tripped

    def wait_cooldown(self, attempt: int):
        """Wait with exponential backoff and track total downtime."""
        delay = min(self.initial_cooldown * (2**attempt), self.max_cooldown)
        with self._lock:
            self._total_downtime += delay
        log.info(
            f"[SCORER CB] Cooling down {delay:.1f}s (total downtime: {self._total_downtime:.0f}s)"
        )
        time.sleep(delay)


# Default prompt file paths (relative to project root config/prompts/)
DEFAULT_VALUE_PROMPT_FILE = "tree-of-thought/llm_critic/value_prompt_math_strict.txt"
DEFAULT_VOTE_PROMPT_FILE = "tree-of-thought/llm_critic/vote_prompt.txt"


def _load_prompt_from_file(prompt_path: str) -> Optional[str]:
    """Load prompt from file. Returns None if file not found."""
    if os.path.isabs(prompt_path) and os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            return f.read()

    project_root = Path(__file__).parent.parent.parent
    config_prompts_path = project_root / "config" / "prompts" / prompt_path
    if config_prompts_path.exists():
        with open(config_prompts_path, "r") as f:
            return f.read()

    cwd_path = Path.cwd() / "config" / "prompts" / prompt_path
    if cwd_path.exists():
        with open(cwd_path, "r") as f:
            return f.read()

    return None


class StepScorerLLMCritic(StepScorerBase):
    """
    LLM Critic Scorer from Tree of Thoughts paper.

    Uses the LLM itself to evaluate reasoning steps, implementing
    the "State Evaluator V(pθ, S)" from the ToT framework.

    Args:
        model: Language model for evaluation. Can be:
            - vLLM LLM instance (for local GPU inference)
            - BlackboxModelWithStreaming (for API-based inference)
            - None (will be set later via set_model())
        method: Evaluation method - "value" or "vote"
        n_evaluate_sample: Number of evaluation samples per step (for aggregation)
        temperature: Sampling temperature for evaluation
        max_tokens: Maximum tokens for evaluation response
        timeout: Timeout in seconds for API calls
        value_prompt: Custom value evaluation prompt template
        vote_prompt: Custom vote evaluation prompt template
        use_vllm: If True, use vLLM backend; if False, use API
    """

    def __init__(
        self,
        model=None,
        method: str = "value",
        n_evaluate_sample: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 100,
        timeout: int = 60,
        value_prompt: str = None,
        value_prompt_file: str = None,
        vote_prompt: str = None,
        vote_prompt_file: str = None,
        use_vllm: bool = False,
        score_aggregation: str = "min",
        trajectory_context_steps: float = 0,
        name: str = "llm_critic",
    ):
        super().__init__(name=name)

        self.model = model
        self.method = method
        self.n_evaluate_sample = n_evaluate_sample
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_vllm = use_vllm
        self.use_local = False
        self.score_aggregation = score_aggregation
        self.trajectory_context_steps = trajectory_context_steps

        # Load prompts: priority is direct string > custom file > default file
        self.value_prompt = self._load_prompt(
            value_prompt, value_prompt_file, DEFAULT_VALUE_PROMPT_FILE, "value"
        )
        self.vote_prompt = self._load_prompt(
            vote_prompt, vote_prompt_file, DEFAULT_VOTE_PROMPT_FILE, "vote"
        )

        # Value mapping aligned to ToT labels: sure/likely/impossible.
        self.value_map = {
            "sure": 3.0,
            "likely": 1.0,
            "impossible": 0.1,
        }

        # Extended synonyms that map to the base labels above
        # Note: paper says "sure/maybe/impossible", code uses "sure/likely/impossible"
        self.value_synonyms = {
            "maybe": 1.0,  # paper uses "maybe" as equivalent to "likely"
            "correct": 3.0,
            "definitely": 3.0,
            "unlikely": 0.1,
            "incorrect": 0.1,
            "wrong": 0.1,
        }

        # Circuit breaker shared across all scorer API calls
        # Note: with 8 concurrent threads, a single batch timeout counts as
        # 8 failures, so threshold must be high enough to survive a few bad batches.
        self._circuit_breaker = _ScorerCircuitBreaker(
            max_consecutive_failures=50,
            max_total_downtime=600.0,
            initial_cooldown=5.0,
            max_cooldown=60.0,
        )

        # Statistics
        self.total_evaluations = 0
        self.cache: Dict[str, float] = {}

        # FLOP/token tracking for LLM critic evaluations
        self.flop_calculator: Optional[FLOPCalculator] = None
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._per_sample_input_tokens: Dict[Any, int] = {}
        self._per_sample_output_tokens: Dict[Any, int] = {}
        self._current_sample_id: Any = None
        self._tokens_lock = threading.Lock()

        log.info(
            f"StepScorerLLMCritic initialized: method={method}, "
            f"n_evaluate_sample={n_evaluate_sample}, use_vllm={use_vllm}, "
            f"score_aggregation={score_aggregation}"
        )

    def _aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate multiple evaluation scores using the configured method."""
        if not scores:
            return 0.0
        if self.score_aggregation == "min":
            return float(min(scores))
        elif self.score_aggregation == "mean":
            return float(sum(scores) / len(scores))
        elif self.score_aggregation == "max":
            return float(max(scores))
        else:  # "sum" (default, as in original ToT paper)
            return float(sum(scores))

    def _load_prompt(
        self,
        prompt_str: Optional[str],
        prompt_file: Optional[str],
        default_file: str,
        prompt_name: str,
    ) -> str:
        """Load prompt from string, file, or default file."""
        # Direct string has highest priority
        if prompt_str:
            return prompt_str

        # Custom file path
        if prompt_file:
            loaded = _load_prompt_from_file(prompt_file)
            if loaded:
                log.debug(f"Loaded {prompt_name} prompt from: {prompt_file}")
                return loaded
            log.warning(f"Could not load {prompt_name} prompt from: {prompt_file}")

        # Default file
        loaded = _load_prompt_from_file(default_file)
        if loaded:
            log.debug(f"Loaded {prompt_name} prompt from default: {default_file}")
            return loaded

        raise FileNotFoundError(
            f"Could not load {prompt_name} prompt. "
            f"Expected at: config/prompts/{default_file}"
        )

    def set_model(self, model, use_vllm: bool = None, use_local: bool = False):
        """Set or update the model for evaluation."""
        self.model = model
        if use_vllm is not None:
            self.use_vllm = use_vllm
        self.use_local = use_local

    # -------------------------------------------------------------------------
    # FLOP/Token Tracking Methods
    # -------------------------------------------------------------------------

    def init_flop_calculator(self, model_name: str):
        """Initialize FLOP calculator for LLM critic token/compute tracking."""
        self.flop_calculator = FLOPCalculator(model_name, method="simple")
        log.info(
            f"Self-verification FLOP calculator initialized: "
            f"{self.flop_calculator.tflops_per_1k_tokens:.3f} TFLOPs/1k tokens"
        )

    def _record_tokens(
        self, input_tokens: int, output_tokens: int, sample_id: Any = None
    ):
        """Record input and output tokens for tracking (thread-safe)."""
        with self._tokens_lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            sid = sample_id if sample_id is not None else self._current_sample_id
            if sid is not None:
                self._per_sample_input_tokens[sid] = (
                    self._per_sample_input_tokens.get(sid, 0) + input_tokens
                )
                self._per_sample_output_tokens[sid] = (
                    self._per_sample_output_tokens.get(sid, 0) + output_tokens
                )

    def set_current_sample_id(self, sample_id: Any):
        """Set the current sample ID for token tracking."""
        self._current_sample_id = sample_id

    def reset_stats(self):
        """Clear per-sample stats (call before each batch)."""
        self._per_sample_input_tokens.clear()
        self._per_sample_output_tokens.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._current_sample_id = None

    def get_stats_for(self, sample_id: Any) -> Dict[str, Any]:
        """Get LLM critic stats for a specific sample."""
        input_tokens = self._per_sample_input_tokens.get(sample_id, 0)
        output_tokens = self._per_sample_output_tokens.get(sample_id, 0)
        total_tokens = input_tokens + output_tokens

        tflops = (
            self.flop_calculator.compute_tflops(total_tokens)
            if self.flop_calculator
            else None
        )
        return {
            "llm_critic_input_tokens": input_tokens,
            "llm_critic_output_tokens": output_tokens,
            "llm_critic_total_tokens": total_tokens,
            "llm_critic_tflops": tflops,
        }

    def get_total_stats(self) -> Dict[str, Any]:
        """Get aggregate LLM critic stats across all samples."""
        total_tokens = self._total_input_tokens + self._total_output_tokens
        tflops = (
            self.flop_calculator.compute_tflops(total_tokens)
            if self.flop_calculator
            else None
        )
        return {
            "llm_critic_input_tokens": self._total_input_tokens,
            "llm_critic_output_tokens": self._total_output_tokens,
            "llm_critic_total_tokens": total_tokens,
            "llm_critic_tflops": tflops,
        }

    def score_candidates_detailed(
        self,
        chat: List[Dict[str, str]],
        candidates: List[Any],
        trajectory: List[Any] = None,
        **kwargs,
    ) -> List[CandidateScore]:
        """
        Score candidates with detailed information.

        Args:
            chat: Current chat (contains the problem/question)
            candidates: List of candidate steps to evaluate
            trajectory: Previous steps in the reasoning chain

        Returns:
            List of CandidateScore objects with detailed scoring info
        """
        if not candidates:
            return []

        # Extract problem from chat
        problem = self._extract_problem(chat)

        # Convert trajectory to text
        trajectory_text = self._trajectory_to_text(trajectory)

        if self.method == "value":
            return self._score_value_method(problem, trajectory_text, candidates)
        elif self.method == "vote":
            return self._score_vote_method(problem, trajectory_text, candidates)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def score_candidates_batch(
        self,
        chats: List[List[Dict[str, str]]],
        candidates_list: List[List[Any]],
        trajectories: Optional[List[List[Any]]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Batch score candidates across multiple prompts (vLLM only).

        Returns aggregated scores aligned with candidates_list.
        """
        if trajectories is None:
            trajectories = [None] * len(candidates_list)

        if len(chats) != len(candidates_list):
            raise ValueError("chats and candidates_list must have same length")

        if self.method == "vote":
            return self._score_vote_method_batch(chats, candidates_list, trajectories)

        # Value method (default)
        return self._score_value_method_batch(chats, candidates_list, trajectories)

    def _score_value_method_batch(
        self,
        chats: List[List[Dict[str, str]]],
        candidates_list: List[List[Any]],
        trajectories: List[List[Any]],
    ) -> List[List[float]]:
        """Batch score value method across multiple prompt groups."""
        results: List[List[float]] = [[] for _ in candidates_list]
        pending: List[Dict[str, Any]] = []
        score_map: Dict[tuple, float] = {}

        for group_idx, (chat, candidates, trajectory) in enumerate(
            zip(chats, candidates_list, trajectories)
        ):
            if not candidates:
                continue

            problem = self._extract_problem(chat)
            trajectory_text = self._trajectory_to_text(trajectory)
            local_seen_texts: Dict[str, bool] = {}

            for cand_idx, candidate in enumerate(candidates):
                step_text = self._get_step_text(candidate)
                cache_key = f"{problem}|||{trajectory_text}|||{step_text}"

                if step_text in local_seen_texts:
                    score_map[(group_idx, cand_idx)] = 0.0
                elif cache_key in self.cache:
                    score_map[(group_idx, cand_idx)] = self.cache[cache_key]
                else:
                    prompt = self.value_prompt.format(
                        problem=problem,
                        trajectory=trajectory_text if trajectory_text else "(empty)",
                        step=step_text,
                    )
                    pending.append(
                        {
                            "group_idx": group_idx,
                            "cand_idx": cand_idx,
                            "prompt": prompt,
                            "cache_key": cache_key,
                        }
                    )

                local_seen_texts[step_text] = True

        if pending and self.use_vllm:
            eval_scores: Dict[tuple, List[float]] = {
                (p["group_idx"], p["cand_idx"]): [] for p in pending
            }

            for i in range(self.n_evaluate_sample):
                prompts = [item["prompt"] for item in pending]
                outputs: List[str] = []
                try:
                    outputs = self._call_vllm_batch(prompts)
                except Exception as e:
                    log.warning(f"Batch evaluation {i+1} failed: {e}")

                for item_idx, item in enumerate(pending):
                    output = outputs[item_idx] if item_idx < len(outputs) else ""
                    log.info(
                        f"Scorer output [vLLM] eval={i+1} "
                        f"group={item['group_idx']} cand={item['cand_idx']}: '{output[:200]}'"
                    )
                    try:
                        score = self._parse_value_output(output)
                    except ValueError as e:
                        log.warning(
                            f"Parse failed for vLLM candidate "
                            f"{item['group_idx']}/{item['cand_idx']}: {e}"
                        )
                        try:
                            retry_output = self._call_vllm(item["prompt"])
                            score = self._parse_value_output(retry_output)
                        except (ValueError, Exception):
                            log.warning("Retry also failed, skipping evaluation")
                            continue
                    except Exception as e:
                        log.warning(
                            f"Evaluation {i+1} failed for candidate "
                            f"{item['group_idx']}/{item['cand_idx']}: {e}"
                        )
                        continue
                    eval_scores[(item["group_idx"], item["cand_idx"])].append(score)

            for item in pending:
                key = (item["group_idx"], item["cand_idx"])
                if not eval_scores[key]:
                    log.warning(
                        f"All evaluations unparseable for vLLM candidate "
                        f"{key}, using 0.001 (impossible)"
                    )
                    score_map[key] = 0.001
                else:
                    score_map[key] = self._aggregate_scores(eval_scores[key])
                self.cache[item["cache_key"]] = score_map[key]
            self.total_evaluations += len(pending)
        elif pending and not self.use_local:
            eval_scores: Dict[tuple, List[float]] = {
                (p["group_idx"], p["cand_idx"]): [] for p in pending
            }

            for i in range(self.n_evaluate_sample):
                prompts = [item["prompt"] for item in pending]
                outputs: List[str] = []
                try:
                    outputs = self._call_api_batch(prompts)
                except Exception as e:
                    log.warning(f"Batch evaluation {i+1} failed: {e}")

                for item_idx, item in enumerate(pending):
                    output = outputs[item_idx] if item_idx < len(outputs) else ""
                    log.info(
                        f"Scorer output [API] eval={i+1} "
                        f"group={item['group_idx']} cand={item['cand_idx']}: '{output[:200]}'"
                    )
                    try:
                        score = self._parse_value_output(output)
                    except ValueError as e:
                        log.warning(
                            f"Parse failed for API candidate "
                            f"{item['group_idx']}/{item['cand_idx']}: {e}"
                        )
                        try:
                            retry_output = self._call_api_single(item["prompt"])
                            score = self._parse_value_output(retry_output)
                        except (ValueError, Exception):
                            log.warning("Retry also failed, skipping evaluation")
                            continue
                    except Exception as e:
                        log.warning(
                            f"Evaluation {i+1} failed for candidate "
                            f"{item['group_idx']}/{item['cand_idx']}: {e}"
                        )
                        continue
                    eval_scores[(item["group_idx"], item["cand_idx"])].append(score)

            for item in pending:
                key = (item["group_idx"], item["cand_idx"])
                if not eval_scores[key]:
                    log.warning(
                        f"All evaluations unparseable for API candidate "
                        f"{key}, using 0.001 (impossible)"
                    )
                    score_map[key] = 0.001
                else:
                    score_map[key] = self._aggregate_scores(eval_scores[key])
                self.cache[item["cache_key"]] = score_map[key]
            self.total_evaluations += len(pending)
        elif pending:
            for item in pending:
                score = self._evaluate_single_step(
                    self._extract_problem(chats[item["group_idx"]]),
                    self._trajectory_to_text(trajectories[item["group_idx"]]),
                    self._get_step_text(
                        candidates_list[item["group_idx"]][item["cand_idx"]]
                    ),
                )
                score_map[(item["group_idx"], item["cand_idx"])] = score
                self.cache[item["cache_key"]] = score
                self.total_evaluations += 1

        for group_idx, candidates in enumerate(candidates_list):
            group_scores: List[float] = []
            for cand_idx in range(len(candidates)):
                group_scores.append(score_map.get((group_idx, cand_idx), 0.0))
            results[group_idx] = group_scores

        return results

    def _score_vote_method_batch(
        self,
        chats: List[List[Dict[str, str]]],
        candidates_list: List[List[Any]],
        trajectories: List[List[Any]],
    ) -> List[List[float]]:
        """Batch score vote method across multiple prompt groups."""
        results: List[List[float]] = [[] for _ in candidates_list]

        prompts = []
        group_sizes = []
        for chat, candidates, trajectory in zip(chats, candidates_list, trajectories):
            if not candidates:
                prompts.append("")
                group_sizes.append(0)
                continue
            problem = self._extract_problem(chat)
            trajectory_text = self._trajectory_to_text(trajectory)
            candidate_texts = [self._get_step_text(c) for c in candidates]
            candidates_str = "\n".join(
                f"{i+1}. {text}" for i, text in enumerate(candidate_texts)
            )
            prompt = self.vote_prompt.format(
                problem=problem,
                trajectory=trajectory_text if trajectory_text else "(empty)",
                candidates=candidates_str,
                n_candidates=len(candidate_texts),
            )
            prompts.append(prompt)
            group_sizes.append(len(candidate_texts))

        votes_by_group = [[0.0] * n for n in group_sizes]

        if self.use_vllm:
            for i in range(self.n_evaluate_sample):
                batch_prompts = [p for p in prompts if p]
                outputs: List[str] = []
                try:
                    outputs = self._call_vllm_batch(batch_prompts)
                except Exception as e:
                    log.warning(f"Vote batch {i+1} failed: {e}")
                    outputs = []

                out_idx = 0
                for group_idx, prompt in enumerate(prompts):
                    if not prompt:
                        continue
                    output = outputs[out_idx] if out_idx < len(outputs) else ""
                    out_idx += 1
                    vote_idx = self._parse_vote_output(output, group_sizes[group_idx])
                    if vote_idx is not None:
                        votes_by_group[group_idx][vote_idx] += 1.0
        else:
            if self.use_local:
                for group_idx, prompt in enumerate(prompts):
                    if not prompt:
                        continue
                    for i in range(self.n_evaluate_sample):
                        try:
                            output = self._call_model(prompt)
                            vote_idx = self._parse_vote_output(
                                output, group_sizes[group_idx]
                            )
                            if vote_idx is not None:
                                votes_by_group[group_idx][vote_idx] += 1.0
                        except Exception as e:
                            log.warning(f"Vote {i+1} failed: {e}")
            else:
                for i in range(self.n_evaluate_sample):
                    batch_prompts = [p for p in prompts if p]
                    outputs: List[str] = []
                    try:
                        outputs = self._call_api_batch(batch_prompts)
                    except Exception as e:
                        log.warning(f"Vote batch {i+1} failed: {e}")
                        outputs = []

                    out_idx = 0
                    for group_idx, prompt in enumerate(prompts):
                        if not prompt:
                            continue
                        output = outputs[out_idx] if out_idx < len(outputs) else ""
                        out_idx += 1
                        vote_idx = self._parse_vote_output(
                            output, group_sizes[group_idx]
                        )
                        if vote_idx is not None:
                            votes_by_group[group_idx][vote_idx] += 1.0

        for group_idx, votes in enumerate(votes_by_group):
            results[group_idx] = votes

        self.total_evaluations += len([p for p in prompts if p])
        return results

    def _score_value_method(
        self,
        problem: str,
        trajectory_text: str,
        candidates: List[Any],
    ) -> List[CandidateScore]:
        """
        Score each candidate independently using value method.

        From ToT paper: "V(pθ, S)(s) ∼ p_value_θ(v|s)"
        Prompts LLM to classify each state as sure/likely/unlikely/impossible.
        """
        results = []
        # Track duplicates within this batch (as in original ToT)
        local_seen_texts: Dict[str, bool] = {}

        pending: List[Dict[str, Any]] = []
        scores: Dict[int, float] = {}

        for cand_idx, candidate in enumerate(candidates):
            step_text = self._get_step_text(candidate)
            cache_key = f"{problem}|||{trajectory_text}|||{step_text}"

            # Handle duplicates within same batch (return 0 as in original ToT)
            if step_text in local_seen_texts:
                scores[cand_idx] = 0.0
                log.debug(
                    f"Duplicate candidate {cand_idx}: score=0 (duplicate in batch)"
                )
            # Check cache
            elif cache_key in self.cache:
                scores[cand_idx] = self.cache[cache_key]
                log.debug(
                    f"Cache hit for candidate {cand_idx}: score={scores[cand_idx]:.3f}"
                )
            else:
                pending.append(
                    {
                        "idx": cand_idx,
                        "step_text": step_text,
                        "cache_key": cache_key,
                    }
                )

            local_seen_texts[step_text] = True

        # Batch-evaluate pending candidates for vLLM
        if pending and self.use_vllm:
            eval_scores: Dict[int, List[float]] = {item["idx"]: [] for item in pending}

            for i in range(self.n_evaluate_sample):
                prompts = [
                    self.value_prompt.format(
                        problem=problem,
                        trajectory=trajectory_text if trajectory_text else "(empty)",
                        step=item["step_text"],
                    )
                    for item in pending
                ]
                outputs: List[str] = []
                try:
                    outputs = self._call_vllm_batch(prompts)
                except Exception as e:
                    log.warning(f"Batch evaluation {i+1} failed: {e}")

                for item_idx, item in enumerate(pending):
                    output = outputs[item_idx] if item_idx < len(outputs) else ""
                    log.info(
                        f"Scorer output [vLLM] eval={i+1} cand={item['idx']}: '{output[:200]}'"
                    )
                    try:
                        score = self._parse_value_output(output)
                    except Exception as e:
                        log.warning(
                            f"Evaluation {i+1} failed for candidate {item['idx']}: {e}"
                        )
                        score = 0.0
                    eval_scores[item["idx"]].append(score)

            for item in pending:
                score = self._aggregate_scores(eval_scores[item["idx"]])
                scores[item["idx"]] = score
                self.cache[item["cache_key"]] = score
                self.total_evaluations += 1

        # Sequential fallback for non-vLLM or if no pending batch
        for item in pending:
            if item["idx"] in scores:
                continue
            score = self._evaluate_single_step(
                problem, trajectory_text, item["step_text"]
            )
            scores[item["idx"]] = score
            self.cache[item["cache_key"]] = score
            self.total_evaluations += 1

        for cand_idx, candidate in enumerate(candidates):
            step_text = self._get_step_text(candidate)
            score = scores.get(cand_idx, 0.0)
            results.append(
                CandidateScore(
                    candidate_text=step_text,
                    claim_scores=[score],
                    aggregate_scores={"value": score},
                    metadata={
                        "scorer_type": "llm_critic",
                        "method": "value",
                    },
                )
            )

            log.info(f"Candidate {cand_idx}: value_score={score:.3f}")

        return results

    def _score_vote_method(
        self,
        problem: str,
        trajectory_text: str,
        candidates: List[Any],
    ) -> List[CandidateScore]:
        """
        Score candidates using voting method.

        From ToT paper: "V(pθ, S)(s) = 1[s = s*], where s* ∼ p_vote_θ(s*|S)"
        Presents all candidates and asks which is most promising.
        """
        if len(candidates) == 1:
            # Single candidate (no voting then)
            step_text = self._get_step_text(candidates[0])
            return [
                CandidateScore(
                    candidate_text=step_text,
                    claim_scores=[1.0],
                    aggregate_scores={"votes": 1.0},
                    metadata={
                        "scorer_type": "llm_critic",
                        "method": "vote",
                    },
                )
            ]

        # Get candidate texts
        candidate_texts = [self._get_step_text(c) for c in candidates]

        # Run voting
        votes = self._run_voting(problem, trajectory_text, candidate_texts)

        # Create results
        results = []
        for cand_idx, (candidate, vote_count) in enumerate(zip(candidates, votes)):
            step_text = self._get_step_text(candidate)
            results.append(
                CandidateScore(
                    candidate_text=step_text,
                    claim_scores=[vote_count],
                    aggregate_scores={"votes": vote_count},
                    metadata={
                        "scorer_type": "llm_critic",
                        "method": "vote",
                        "total_votes": sum(votes),
                    },
                )
            )
            log.info(f"Candidate {cand_idx}: votes={vote_count}")

        self.total_evaluations += 1
        return results

    def _evaluate_single_step(
        self,
        problem: str,
        trajectory_text: str,
        step_text: str,
    ) -> float:
        """
        Evaluate a single step using value method.

        Calls the LLM n_evaluate_sample times and aggregates scores.
        """
        # Build prompt
        prompt = self.value_prompt.format(
            problem=problem,
            trajectory=trajectory_text if trajectory_text else "(empty)",
            step=step_text,
        )

        # Get evaluations
        scores = []
        for i in range(self.n_evaluate_sample):
            try:
                output = self._call_model(prompt)
                score = self._parse_value_output(output)
                scores.append(score)
                log.info(
                    f"Evaluation {i+1}/{self.n_evaluate_sample}: score={score:.3f}, output='{output[:200]}'"
                )
            except ValueError as e:
                log.warning(f"Parse failed on eval {i+1}: {e}")
                try:
                    retry_output = self._call_model(prompt)
                    score = self._parse_value_output(retry_output)
                    scores.append(score)
                except (ValueError, Exception):
                    log.warning("Retry also failed, skipping evaluation")
            except Exception as e:
                log.warning(f"Evaluation {i+1} failed: {e}")

        if not scores:
            log.warning("All evaluations unparseable in single step, using 0.001")
            return 0.001
        return self._aggregate_scores(scores)

    def _run_voting(
        self,
        problem: str,
        trajectory_text: str,
        candidate_texts: List[str],
    ) -> List[float]:
        """
        Run voting evaluation for candidates.

        Returns vote counts for each candidate.
        """
        # Format candidates for prompt
        candidates_str = "\n".join(
            f"{i+1}. {text}" for i, text in enumerate(candidate_texts)
        )

        prompt = self.vote_prompt.format(
            problem=problem,
            trajectory=trajectory_text if trajectory_text else "(empty)",
            candidates=candidates_str,
            n_candidates=len(candidate_texts),
        )

        # Collect votes
        votes = [0.0] * len(candidate_texts)

        if self.use_vllm:
            prompts = [prompt] * self.n_evaluate_sample
            try:
                outputs = self._call_vllm_batch(prompts)
            except Exception as e:
                log.warning(f"Vote batch failed: {e}")
                outputs = []

            for i, output in enumerate(outputs):
                vote_idx = self._parse_vote_output(output, len(candidate_texts))
                if vote_idx is not None:
                    votes[vote_idx] += 1.0
                log.debug(
                    f"Vote {i+1}/{self.n_evaluate_sample}: output='{output[:50]}...', vote_idx={vote_idx}"
                )
        else:
            for i in range(self.n_evaluate_sample):
                try:
                    output = self._call_model(prompt)
                    vote_idx = self._parse_vote_output(output, len(candidate_texts))
                    if vote_idx is not None:
                        votes[vote_idx] += 1.0
                    log.debug(
                        f"Vote {i+1}/{self.n_evaluate_sample}: output='{output[:50]}...', vote_idx={vote_idx}"
                    )
                except Exception as e:
                    log.warning(f"Vote {i+1} failed: {e}")

        return votes

    def _call_model(self, prompt: str) -> str:
        """
        Call the model with the given prompt.

        Supports both vLLM and API backends.
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        if self.use_vllm:
            return self._call_vllm(prompt)
        elif self.use_local:
            return self._call_local(prompt)
        else:
            return self._call_api(prompt)

    def _format_prompt_for_vllm(self, prompt: str) -> str:
        """Wrap prompt in chat template if the vLLM model has a tokenizer."""
        try:
            tokenizer = self.model.get_tokenizer()
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return formatted
        except Exception:
            # Fallback to raw prompt if tokenizer/chat template not available
            return prompt

    def _call_vllm(self, prompt: str) -> str:
        """Call vLLM model for evaluation."""
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        formatted_prompt = self._format_prompt_for_vllm(prompt)
        outputs = self.model.generate([formatted_prompt], sampling_params)

        if outputs and outputs[0].outputs:
            output = outputs[0]
            # Track tokens: prompt tokens + generated tokens
            input_tokens = (
                len(output.prompt_token_ids) if output.prompt_token_ids else 0
            )
            output_tokens = (
                len(output.outputs[0].token_ids) if output.outputs[0].token_ids else 0
            )
            self._record_tokens(input_tokens, output_tokens)
            return output.outputs[0].text
        return ""

    def _call_vllm_batch(self, prompts: List[str]) -> List[str]:
        """Call vLLM model for evaluation with batched prompts."""
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        formatted_prompts = [self._format_prompt_for_vllm(p) for p in prompts]
        outputs = self.model.generate(formatted_prompts, sampling_params)
        texts: List[str] = []

        for output in outputs or []:
            if output and output.outputs:
                input_tokens = (
                    len(output.prompt_token_ids) if output.prompt_token_ids else 0
                )
                output_tokens = (
                    len(output.outputs[0].token_ids)
                    if output.outputs[0].token_ids
                    else 0
                )
                self._record_tokens(input_tokens, output_tokens)
                texts.append(output.outputs[0].text)
            else:
                texts.append("")

        return texts

    def _call_local(self, prompt: str) -> str:
        """Call local WhiteboxModel (lm_polygraph) for evaluation."""
        messages = [{"role": "user", "content": prompt}]
        formatted = self.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Count input tokens before generation
        input_tokens = len(self.model.tokenizer.encode(formatted))

        results = self.model.generate_texts(
            input_texts=[formatted],
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if results and results[0]:
            # Count output tokens
            output_tokens = len(self.model.tokenizer.encode(results[0]))
            self._record_tokens(input_tokens, output_tokens)
            return results[0]
        return ""

    def _call_api(self, prompt: str, sample_id: Any = None) -> str:
        """Call API-based model for evaluation with retry logic."""
        return self._call_api_single(prompt, sample_id=sample_id)

    def _call_api_single(self, prompt: str, sample_id: Any = None) -> str:
        """Call API for a single prompt with retry logic and circuit breaker.

        Uses the model OpenAI client directly for better concurrent throughput.
        The circuit breaker prevents wasting minutes on a dead API.
        """
        import openai

        messages = [{"role": "user", "content": prompt}]
        max_retries = 3  # Reduced from 5 — circuit breaker handles prolonged outages
        base_delay = 3.0

        # Check circuit breaker before even trying
        if not self._circuit_breaker.should_attempt():
            log.warning("[SCORER] Circuit breaker is open, skipping API call")
            return ""

        # Use model's openai client directly for thread-safe concurrent calls
        client = getattr(self.model, "client", None) or getattr(
            self.model, "openai_api", None
        )
        model_name = getattr(self.model, "model_path", None) or getattr(
            self.model, "model_name", "unknown"
        )

        if client is None:
            log.error("No OpenAI client found on model")
            return ""

        for attempt in range(max_retries):
            # Re-check circuit breaker on each retry
            if not self._circuit_breaker.should_attempt():
                log.warning("[SCORER] Circuit breaker tripped during retries")
                return ""

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    n=1,
                    timeout=self.timeout,
                    extra_body={"reasoning_effort": "low"},
                )

                if response.choices:
                    msg = response.choices[0].message
                    # Reasoning models (e.g. gpt-oss-120b) put CoT in
                    # reasoning_content and the final answer in content.
                    # Fall back to reasoning_content if content is empty.
                    text = msg.content or getattr(msg, "reasoning_content", None) or ""
                else:
                    text = ""
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = (
                    response.usage.completion_tokens if response.usage else 0
                )
                if input_tokens == 0 and output_tokens == 0:
                    input_tokens = len(prompt) // 4
                    output_tokens = len(text) // 4
                self._record_tokens(input_tokens, output_tokens, sample_id=sample_id)
                self._circuit_breaker.record_success()
                return text

            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                can_continue = self._circuit_breaker.record_failure()
                if not can_continue:
                    log.error(f"[SCORER] Circuit breaker tripped on {type(e).__name__}")
                    return ""
                if attempt < max_retries - 1:
                    self._circuit_breaker.wait_cooldown(attempt)
                else:
                    log.error(
                        f"[SCORER] API call failed after {max_retries} attempts: {e}"
                    )
                    return ""

            except openai.RateLimitError as e:
                can_continue = self._circuit_breaker.record_failure()
                if not can_continue:
                    log.error("[SCORER] Circuit breaker tripped on RateLimitError")
                    return ""
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) * 3
                    log.warning(
                        f"Rate limit on attempt {attempt + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    log.error(f"Rate limit persists after {max_retries} attempts: {e}")
                    return ""

            except Exception as e:
                log.error(f"[SCORER] Unexpected error: {type(e).__name__}: {e}")
                self._circuit_breaker.record_failure()
                return ""

        return ""

    def _call_api_batch(
        self, prompts: List[str], sample_ids: Optional[List[Any]] = None
    ) -> List[str]:
        """Call API for each prompt concurrently with per-prompt retry.

        Each prompt runs independently via _call_api (which has its own retry
        and circuit breaker check). If the circuit breaker is open, all
        prompts return empty immediately.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Fast-path: if circuit breaker is already open, skip everything
        if not self._circuit_breaker.should_attempt():
            log.warning("[SCORER] Circuit breaker open, skipping entire batch")
            return [""] * len(prompts)

        results: List[Optional[str]] = [None] * len(prompts)

        def _call_single(idx: int, prompt: str) -> tuple:
            sid = sample_ids[idx] if sample_ids and idx < len(sample_ids) else None
            try:
                return idx, self._call_api(prompt, sample_id=sid)
            except Exception as e:
                log.warning(f"API call failed for prompt {idx}: {e}")
                return idx, ""

        with ThreadPoolExecutor(max_workers=min(len(prompts), 4)) as pool:
            futures = [
                pool.submit(_call_single, idx, prompt)
                for idx, prompt in enumerate(prompts)
            ]
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text

        return [t if t is not None else "" for t in results]

    def _parse_value_output(self, output: str) -> float:
        """
        Parse value evaluation output into numerical score.

        Matching priority (following original ToT paper):
        1. Exact match on last line (primary ToT labels)
        2. Token search on last line (primary + synonym labels)
        3. Prefix match (keyword at start of token, e.g. "likelyMK" -> "likely")
        4. Regex word boundary search on full output
        5. Loose keyword search on full normalized output
        6. Default to 0.0 (unmatched = no contribution, as in original ToT)
        """
        output_lower = output.lower().strip()
        all_keywords = {**self.value_map, **self.value_synonyms}

        # 1. Prefer exact label matching on the last non-empty line (ToT behavior).
        lines = [line.strip() for line in output_lower.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            normalized = re.sub(r"[^a-z]+", " ", last_line).strip()
            if normalized in self.value_map:
                return self.value_map[normalized]
            # 2. Token search on last line (primary + synonyms)
            for token in normalized.split():
                if token in all_keywords:
                    return all_keywords[token]

        # 3. Prefix match: check if output starts with a keyword
        #    (handles garbage glued to keyword, e.g. "likelyMK", "surelyABC")
        # Sort by length descending so "impossible" matches before "imp..."
        for keyword in sorted(all_keywords.keys(), key=len, reverse=True):
            if output_lower.startswith(keyword):
                return all_keywords[keyword]

        # 4. Regex search on full output for keyword as a word or at start of token
        for keyword in sorted(all_keywords.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(keyword) + r"\b", output_lower):
                return all_keywords[keyword]

        # 5. Search full output for any keyword (loose fallback)
        full_normalized = re.sub(r"[^a-z\s]+", " ", output_lower)
        for token in full_normalized.split():
            if token in all_keywords:
                return all_keywords[token]

        # 6. Default to 0.0 — unmatched outputs contribute nothing
        #    (matches original ToT where unrecognized labels are not counted)
        log.debug(f"No rating found in output: '{output[:100]}', defaulting to 0.0")
        return 0.0

    def _parse_vote_output(self, output: str, n_candidates: int) -> Optional[int]:
        """
        Parse vote output to get selected candidate index (0-based).

        Priority:
        1. "The best choice is X" pattern (original ToT paper format)
        2. First valid number in output
        """
        # 1. Try "best choice is X" pattern (as in original ToT paper)
        match = re.search(r"best choice is\s*(\d+)", output, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            if 1 <= num <= n_candidates:
                return num - 1

        # 2. Fallback: find any valid candidate number in output
        numbers = re.findall(r"\d+", output)
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= n_candidates:
                return num - 1

        log.debug(f"No valid vote found in output: '{output[:100]}'")
        return None

    def _extract_problem(self, chat: List[Dict[str, str]]) -> str:
        """Extract problem/question from chat messages."""
        for msg in chat:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return chat[-1].get("content", "") if chat else ""

    def _trajectory_to_text(self, trajectory: Optional[List[Any]]) -> str:
        """Convert trajectory to text representation.

        If trajectory_context_steps is an int > 0, only the last N steps are included.
        If trajectory_context_steps is a float between 0 and 1, it's treated as a
        fraction of total steps (e.g., 0.2 = last 20%), with a minimum of 1 step.
        """
        if not trajectory:
            return ""

        steps = []
        for step in trajectory:
            if hasattr(step, "text"):
                steps.append(step.text)
            else:
                steps.append(str(step))

        if self.trajectory_context_steps > 0:
            if 0 < self.trajectory_context_steps < 1:
                # Fraction: take last X% of steps
                import math

                n_keep = max(1, math.ceil(len(steps) * self.trajectory_context_steps))
            else:
                n_keep = int(self.trajectory_context_steps)
            if len(steps) > n_keep:
                steps = steps[-n_keep:]

        return "\n".join(steps)

    def _get_step_text(self, candidate: Any) -> str:
        """Extract text from a candidate step."""
        if hasattr(candidate, "text"):
            return candidate.text
        return str(candidate)

    def cleanup(self):
        """Clean up resources."""
        # Log final stats before cleanup
        total_stats = self.get_total_stats()
        log.info(
            f"SelfVerification scorer cleanup: "
            f"total_tokens={total_stats['llm_critic_total_tokens']}, "
            f"tflops={total_stats['llm_critic_tflops']}"
        )

        cache_size = len(self.cache)
        self.cache.clear()
        self.reset_stats()
        log.info(f"SelfVerification scorer: cleared {cache_size} cached entries")

    def __str__(self):
        return f"StepScorerLLMCritic(method={self.method}, n_samples={self.n_evaluate_sample})"
