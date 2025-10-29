"""
ToT Value Scorer - Individual state quality evaluation.

Evaluates each state independently by asking the LLM:
"Can this partial solution reach the correct answer?"

Ratings:
- correct/sure: 10.0 - Solution is correct or approach will definitely work
- likely: 5.0 - Approach looks promising
- unlikely: 1.0 - Approach has issues
- incorrect/impossible: 0.1 - Approach cannot work
"""

import logging
from typing import List

import numpy as np

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming

from .base import TotStateScorerBase

log = logging.getLogger(__name__)


class TotValueScorer(TotStateScorerBase):
    """
    Value-based state scorer for Tree-of-Thoughts.

    Evaluates each state individually for its quality and likelihood
    of reaching the correct solution.
    """

    def __init__(
        self,
        model,
        n_evaluate_sample: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 50,
        timeout: int = 120,
        name: str = "tot_value_scorer",
    ):
        """
        Initialize value scorer.

        Args:
            model: Language model for evaluation
            n_evaluate_sample: Number of evaluation samples per state
            temperature: Sampling temperature for evaluation (0.0 = deterministic)
            max_tokens: Maximum tokens per evaluation
            timeout: Timeout in seconds for each evaluation call (default: 120s)
            name: Scorer name
        """
        super().__init__(model, name)
        self.n_evaluate_sample = n_evaluate_sample
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Value mapping from text ratings to scores
        # Based on original ToT paper (Yao et al., 2023)
        # https://arxiv.org/abs/2305.10601
        self.value_map = {
            "correct": 20.0,  # For final answer correctness
            "sure": 20.0,  # Paper value: high confidence
            "likely": 1.0,  # Paper value: moderate confidence
            "unlikely": 0.001,  # Low confidence
            "incorrect": 0.001,  # For final answer incorrectness
            "impossible": 0.001,  # Paper value: cannot reach solution
        }

    def build_evaluation_prompt(self, problem: str, state: str) -> str:
        """Build prompt for state value evaluation."""
        # Check if this is a final answer
        if self.is_final_state(state):
            return f"""Evaluate if this solution correctly solves the problem:

Problem: {problem}

Solution:
{state}

Is this solution correct? Rate as one of:
- correct: The solution is mathematically correct and answers the problem
- incorrect: The solution has errors or doesn't answer the problem
- incomplete: The solution is on track but not finished

Rating:"""
        else:
            return f"""Evaluate this partial solution for a math problem:

Problem: {problem}

Partial solution so far:
{state}

IMPORTANT: Only rate as "sure" or "likely" if the solution contains CONCRETE STEPS with actual numbers and calculations. Vague statements like "we can approach this" or "here are possible steps" should be rated as "unlikely".

Can this approach reach the correct answer? Rate as one of:
- sure: This has concrete calculations and will definitely work
- likely: This has specific steps and looks promising
- unlikely: This is vague, incomplete, or has issues
- impossible: This approach cannot work

Rating:"""

    def parse_evaluation_output(self, output: str) -> float:
        """
        Parse evaluation output into numerical score.

        Args:
            output: Model's text output

        Returns:
            Numerical score
        """
        output_lower = output.lower().strip()

        # Find matching rating
        for rating, score in self.value_map.items():
            if rating in output_lower:
                return score

        # Default score if no match
        return 1.0

    def score_states(
        self, problem: str, states: List[str], cache_results: bool = True, **kwargs
    ) -> List[float]:
        """
        Score multiple states using value function.

        Args:
            problem: Original problem statement
            states: List of partial solutions to evaluate
            cache_results: Whether to cache evaluation results

        Returns:
            List of value scores for each state
        """
        try:
            log.info(f"[SCORER START] Evaluating {len(states)} states")
            scores = []

            # Deduplicate states to avoid redundant evaluations
            local_cache = {}

            for idx, state in enumerate(states):
                log.info(f"[SCORER] State {idx+1}/{len(states)}: {state[:100]}...")

                # Check if we've already evaluated this state
                if state in local_cache:
                    log.info(f"[SCORER] State {idx+1}: Using local cache")
                    scores.append(local_cache[state])
                    continue

                # Check global cache
                cache_key = f"{problem}|||{state}"
                if cache_results and cache_key in self.cache:
                    log.info(f"[SCORER] State {idx+1}: Using global cache")
                    score = self.cache[cache_key]
                    scores.append(score)
                    local_cache[state] = score
                    continue

                # Evaluate state
                log.info(f"[SCORER] State {idx+1}: Calling _evaluate_single_state...")
                score = self._evaluate_single_state(problem, state)
                log.info(f"[SCORER] State {idx+1}: Got score {score}")
                scores.append(score)
                local_cache[state] = score

                # Cache result
                if cache_results:
                    self.cache[cache_key] = score

                self.total_evaluations += 1

            log.info(f"[SCORER DONE] Completed {len(scores)} evaluations")
            return scores

        except Exception as e:
            log.error(f"Error in score_states: {e}. Returning default scores.")
            # Return default scores for all states
            return [1.0] * len(states)

    def _evaluate_single_state(self, problem: str, state: str) -> float:
        """
        Evaluate a single state with retry logic.

        Args:
            problem: Original problem
            state: Current partial solution

        Returns:
            Value score
        """
        import time

        import openai

        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            try:

                # Build evaluation prompt
                log.info(
                    f"[EVAL] Attempt {attempt + 1}/{max_retries}: Building prompt for state evaluation"
                )
                prompt = self.build_evaluation_prompt(problem, state)
                log.info(f"[EVAL] Prompt built, length={len(prompt)}")

                # Call model for evaluation
                if isinstance(self.model, BlackboxModelWithStreaming):
                    from llm_tts.utils.parallel import parallel_execute

                    messages = [{"role": "user", "content": prompt}]

                    # Make multiple evaluation calls in PARALLEL using shared utility
                    # This is much faster than sequential calls
                    if self.n_evaluate_sample > 1:
                        log.info(
                            f"[EVAL] Making {self.n_evaluate_sample} parallel evaluation calls, timeout={self.timeout}s each"
                        )

                        # Worker function for parallel execution
                        def eval_worker(index):
                            """Generate single evaluation response"""
                            result = self.model.generate_texts(
                                chats=[messages],
                                n=1,  # Single response per call
                                max_new_tokens=self.max_tokens,
                                temperature=self.temperature,
                                timeout=self.timeout,
                            )
                            return result[0] if result else None

                        # Execute using shared parallel utility (same as strategy_base)
                        # Pass model for automatic client recreation on failures
                        outputs = parallel_execute(
                            worker_func=eval_worker,
                            task_args=list(range(self.n_evaluate_sample)),
                            n_workers=self.n_evaluate_sample,
                            desc=f"[EVAL] Evaluating state ({self.n_evaluate_sample} samples)",
                            model=self.model,  # Enable automatic client recreation
                        )
                    else:
                        # Single evaluation
                        log.info(
                            f"[EVAL] Making single evaluation call, timeout={self.timeout}s"
                        )
                        outputs = self.model.generate_texts(
                            chats=[messages],
                            n=1,
                            max_new_tokens=self.max_tokens,
                            temperature=self.temperature,
                            timeout=self.timeout,
                        )
                        log.info(f"[EVAL] Call completed, got {len(outputs)} result")

                    # Extract text outputs
                    output_texts = [
                        result.get("text", "") for result in outputs if result
                    ]
                    log.info(f"[EVAL] Extracted {len(output_texts)} text outputs")

                    # Client recreation now handled automatically by parallel_execute()
                    # If all calls failed, raise error to trigger full retry logic
                    if len(output_texts) == 0:
                        log.warning(
                            f"[EVAL] All {self.n_evaluate_sample} parallel calls failed/timed out"
                        )
                        raise openai.APITimeoutError(
                            "All parallel evaluation calls failed"
                        )
                else:
                    # Local model - not yet implemented
                    raise NotImplementedError("Local models not yet supported")

                # Parse outputs and average scores
                scores_list = [
                    self.parse_evaluation_output(output) for output in output_texts
                ]
                avg_score = np.mean(scores_list) if scores_list else 1.0

                log.info(
                    f"[EVAL] State value: {avg_score:.2f} (from {len(scores_list)} samples)"
                )

                return float(avg_score)

            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                if attempt < max_retries - 1:
                    # Client already recreated when timeout detected, just delay before retry
                    delay = 10 + (base_delay * (2**attempt))  # 12s, 14s, 18s
                    log.warning(
                        f"[EVAL] API timeout/connection error on attempt {attempt + 1}/{max_retries}. "
                        f"Waiting {delay:.1f}s before retry... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        f"[EVAL] API call failed after {max_retries} attempts: {e}. Returning default score."
                    )
                    return 1.0  # Return default score instead of crashing

            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) * 3  # 6s, 12s, 24s
                    log.warning(
                        f"[EVAL] Rate limit hit on attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {delay:.1f}s... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        f"[EVAL] Rate limit persists after {max_retries} attempts: {e}. Returning default score."
                    )
                    return 1.0  # Return default score instead of crashing

            except Exception as e:
                # For other errors, return default score
                log.error(
                    f"[EVAL ERROR] Error evaluating state: {type(e).__name__}: {e}. Returning default score 1.0"
                )
                return 1.0  # Return default "unlikely" score on error

        # Should never reach here
        return 1.0
