"""
Self-Verification Scorer based on Tree of Thoughts paper.

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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .step_scorer_base import CandidateScore, StepScorerBase

log = logging.getLogger(__name__)

# Default prompt file paths (relative to project root config/prompts/)
DEFAULT_VALUE_PROMPT_FILE = "tree-of-thought/self_verification/value_prompt.txt"
DEFAULT_VOTE_PROMPT_FILE = "tree-of-thought/self_verification/vote_prompt.txt"


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


class StepScorerSelfVerification(StepScorerBase):
    """
    Self-Verification Scorer from Tree of Thoughts paper.

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
        name: str = "self_verification",
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

        # Load prompts: priority is direct string > custom file > default file
        self.value_prompt = self._load_prompt(
            value_prompt, value_prompt_file, DEFAULT_VALUE_PROMPT_FILE, "value"
        )
        self.vote_prompt = self._load_prompt(
            vote_prompt, vote_prompt_file, DEFAULT_VOTE_PROMPT_FILE, "vote"
        )

        # Value mapping from ToT paper (Table in Section 4.1)
        # sure=20, likely=1, unlikely=0.001, impossible=0.001
        self.value_map = {
            "sure": 20.0,
            "certain": 20.0,
            "correct": 20.0,
            "likely": 1.0,
            "probable": 1.0,
            "maybe": 1.0,
            "unlikely": 0.001,
            "uncertain": 0.001,
            "impossible": 0.001,
            "wrong": 0.001,
            "incorrect": 0.001,
        }

        # Statistics
        self.total_evaluations = 0
        self.cache: Dict[str, float] = {}

        log.info(
            f"StepScorerSelfVerification initialized: method={method}, "
            f"n_evaluate_sample={n_evaluate_sample}, use_vllm={use_vllm}"
        )

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

        Note: Uses SUM aggregation as in original ToT paper (not mean).
        Duplicate candidates in the same batch get score 0 to avoid redundancy.
        """
        results = []
        # Track duplicates within this batch (as in original ToT implementation)
        local_seen_texts: Dict[str, bool] = {}

        for cand_idx, candidate in enumerate(candidates):
            step_text = self._get_step_text(candidate)

            # Handle duplicates within same batch (return 0 as in original ToT)
            if step_text in local_seen_texts:
                score = 0.0
                log.debug(
                    f"Duplicate candidate {cand_idx}: score=0 (duplicate in batch)"
                )
            else:
                local_seen_texts[step_text] = True

                # Check global cache
                cache_key = f"{problem}|||{trajectory_text}|||{step_text}"
                if cache_key in self.cache:
                    score = self.cache[cache_key]
                    log.debug(f"Cache hit for candidate {cand_idx}: score={score:.3f}")
                else:
                    # Evaluate step
                    score = self._evaluate_single_step(
                        problem, trajectory_text, step_text
                    )
                    self.cache[cache_key] = score
                    self.total_evaluations += 1

            local_seen_texts[step_text] = True

            results.append(
                CandidateScore(
                    candidate_text=step_text,
                    claim_scores=[score],
                    aggregate_scores={"value": score},
                    metadata={
                        "scorer_type": "self_verification",
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
                        "scorer_type": "self_verification",
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
                        "scorer_type": "self_verification",
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

        Calls the LLM n_evaluate_sample times and aggregates scores using SUM.

        From original ToT paper (game24.py:86-92):
            value = sum(value * value_names.count(name) for name, value in value_map.items())

        This means with n_evaluate_sample=3 and all "sure" responses:
            score = 20 + 20 + 20 = 60 (not 20 as with mean)
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
                log.debug(
                    f"Evaluation {i+1}/{self.n_evaluate_sample}: output='{output[:50]}...', score={score:.3f}"
                )
            except Exception as e:
                log.warning(f"Evaluation {i+1} failed: {e}")
                scores.append(1.0)  # Default to "likely" on failure

        # Aggregate scores using SUM (as in original ToT paper, not mean)
        return float(sum(scores)) if scores else float(self.n_evaluate_sample)

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

        outputs = self.model.generate([prompt], sampling_params)

        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return ""

    def _call_local(self, prompt: str) -> str:
        """Call local WhiteboxModel (lm_polygraph) for evaluation."""
        messages = [{"role": "user", "content": prompt}]
        formatted = self.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        results = self.model.generate_texts(
            input_texts=[formatted],
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if results and results[0]:
            return results[0]
        return ""

    def _call_api(self, prompt: str) -> str:
        """Call API-based model for evaluation with retry logic."""
        import openai

        messages = [{"role": "user", "content": prompt}]

        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                results = self.model.generate_texts(
                    chats=[messages],
                    n=1,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )

                if results and results[0]:
                    return results[0].get("text", "")
                return ""

            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    log.warning(
                        f"API error on attempt {attempt + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    log.error(f"API call failed after {max_retries} attempts: {e}")
                    raise

            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) * 3
                    log.warning(
                        f"Rate limit on attempt {attempt + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    log.error(f"Rate limit persists after {max_retries} attempts: {e}")
                    raise

        return ""

    def _parse_value_output(self, output: str) -> float:
        """
        Parse value evaluation output into numerical score.

        Looks for keywords: sure, likely, unlikely, impossible
        """
        output_lower = output.lower().strip()

        # Check for each rating in order of priority
        for rating, score in self.value_map.items():
            if rating in output_lower:
                return score

        # Default to "likely" if no match
        log.debug(f"No rating found in output: '{output[:100]}', defaulting to 1.0")
        return 1.0

    def _parse_vote_output(self, output: str, n_candidates: int) -> Optional[int]:
        """
        Parse vote output to get selected candidate index (0-based).
        """
        # Find numbers in output
        numbers = re.findall(r"\d+", output)

        for num_str in numbers:
            num = int(num_str)
            # Check if valid candidate number (1-indexed in prompt)
            if 1 <= num <= n_candidates:
                return num - 1  # Convert to 0-indexed

        log.debug(f"No valid vote found in output: '{output[:100]}'")
        return None

    def _extract_problem(self, chat: List[Dict[str, str]]) -> str:
        """Extract problem/question from chat messages."""
        for msg in chat:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return chat[-1].get("content", "") if chat else ""

    def _trajectory_to_text(self, trajectory: Optional[List[Any]]) -> str:
        """Convert trajectory to text representation."""
        if not trajectory:
            return ""

        steps = []
        for step in trajectory:
            if hasattr(step, "text"):
                steps.append(step.text)
            else:
                steps.append(str(step))

        return "\n".join(steps)

    def _get_step_text(self, candidate: Any) -> str:
        """Extract text from a candidate step."""
        if hasattr(candidate, "text"):
            return candidate.text
        return str(candidate)

    def cleanup(self):
        """Clean up resources."""
        self.cache.clear()
        log.info(
            f"SelfVerification scorer cleanup: cleared {len(self.cache)} cached entries"
        )

    def __str__(self):
        return f"StepScorerSelfVerification(method={self.method}, n_samples={self.n_evaluate_sample})"
