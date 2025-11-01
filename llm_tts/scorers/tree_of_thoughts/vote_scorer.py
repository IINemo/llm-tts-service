"""
ToT Vote Scorer - Consensus-based state evaluation.

Instead of scoring states individually, presents all states to the LLM
and asks which is most promising via voting.
"""

import logging
import re
from typing import List

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming

from .base import TotStateScorerBase

log = logging.getLogger(__name__)


class TotVoteScorer(TotStateScorerBase):
    """
    Vote-based state scorer for Tree-of-Thoughts.

    Evaluates states collectively by asking the LLM to vote on
    which state is most promising.
    """

    def __init__(
        self,
        model,
        n_evaluate_sample: int = 3,
        temperature: float = 0.5,
        max_tokens: int = 100,
        timeout: int = 120,
        name: str = "tot_vote_scorer",
    ):
        """
        Initialize vote scorer.

        Args:
            model: Language model for evaluation
            n_evaluate_sample: Number of voting samples
            temperature: Sampling temperature for voting
            max_tokens: Maximum tokens per vote
            timeout: Timeout in seconds for each voting call (default: 120s)
            name: Scorer name
        """
        super().__init__(model, name)
        self.n_evaluate_sample = n_evaluate_sample
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def build_evaluation_prompt(self, problem: str, state: str) -> str:
        """Not used in vote scorer (uses build_vote_prompt instead)."""
        raise NotImplementedError("Vote scorer uses build_vote_prompt()")

    def build_vote_prompt(self, problem: str, states: List[str]) -> str:
        """
        Build prompt for voting evaluation.

        Args:
            problem: Original problem
            states: List of states to vote on

        Returns:
            Voting prompt
        """
        states_text = "\n\n".join(
            [f"Option {i+1}:\n{state}" for i, state in enumerate(states)]
        )

        return f"""Given a math problem and multiple solution attempts, select which is most promising:

Problem: {problem}

{states_text}

Which option is most likely to lead to the correct answer? Respond with just the option number (1-{len(states)}):

Best option:"""

    def parse_evaluation_output(self, output: str) -> float:
        """Not used in vote scorer (uses parse_vote_outputs instead)."""
        raise NotImplementedError("Vote scorer uses parse_vote_outputs()")

    def parse_vote_outputs(self, outputs: List[str], n_states: int) -> List[float]:
        """
        Parse vote outputs and return vote counts.

        Args:
            outputs: List of model outputs
            n_states: Number of states being voted on

        Returns:
            List of vote counts (one per state)
        """
        votes = [0.0] * n_states

        for output in outputs:
            # Extract number from output
            numbers = re.findall(r"\d+", output)
            if numbers:
                vote_idx = int(numbers[0]) - 1  # Convert to 0-indexed
                if 0 <= vote_idx < n_states:
                    votes[vote_idx] += 1.0

        return votes

    def score_states(
        self,
        problem: str,
        states: List[str],
        cache_results: bool = False,  # Voting not cacheable (depends on full set)
        **kwargs,
    ) -> List[float]:
        """
        Score multiple states using voting.

        Args:
            problem: Original problem statement
            states: List of partial solutions to evaluate
            cache_results: Ignored for voting (not cacheable)

        Returns:
            List of vote counts for each state
        """
        if not states:
            return []

        if len(states) == 1:
            # Single state - no voting needed
            return [1.0]

        # Retry logic for voting (similar to value_scorer)
        import time

        import openai

        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                # Build voting prompt
                prompt = self.build_vote_prompt(problem, states)

                # Call model for votes using shared parallel utility
                if isinstance(self.model, BlackboxModelWithStreaming):
                    from llm_tts.utils.parallel import parallel_execute

                    messages = [{"role": "user", "content": prompt}]

                    if self.n_evaluate_sample > 1:
                        log.info(
                            f"[VOTE] Making {self.n_evaluate_sample} parallel voting calls, timeout={self.timeout}s each"
                        )

                        # Worker function for parallel execution
                        def vote_worker(index):
                            """Generate single vote response"""
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
                            worker_func=vote_worker,
                            task_args=list(range(self.n_evaluate_sample)),
                            n_workers=self.n_evaluate_sample,
                            desc=f"[VOTE] Voting on states ({self.n_evaluate_sample} samples)",
                            model=self.model,  # Enable automatic client recreation
                        )
                    else:
                        # Single vote
                        log.info(
                            f"[VOTE] Making single voting call, timeout={self.timeout}s"
                        )
                        outputs = self.model.generate_texts(
                            chats=[messages],
                            n=1,
                            max_new_tokens=self.max_tokens,
                            temperature=self.temperature,
                            timeout=self.timeout,
                        )

                    # Extract text outputs
                    output_texts = [
                        result.get("text", "") for result in outputs if result
                    ]
                    log.info(f"[VOTE] Extracted {len(output_texts)} text outputs")

                    # Client recreation now handled automatically by parallel_execute()
                    # If all calls failed, raise error to trigger full retry logic
                    if len(output_texts) == 0:
                        log.warning(
                            f"[VOTE] All {self.n_evaluate_sample} parallel calls failed/timed out"
                        )
                        raise openai.APITimeoutError("All parallel voting calls failed")
                else:
                    # Local model - not yet implemented
                    raise NotImplementedError("Local models not yet supported")

                # Parse votes
                votes = self.parse_vote_outputs(output_texts, len(states))

                self.total_evaluations += 1

                log.debug(f"Vote distribution: {votes}")

                return votes

            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                if attempt < max_retries - 1:
                    # Client already recreated when timeout detected, just delay before retry
                    delay = 10 + (base_delay * (2**attempt))  # 12s, 14s, 18s
                    log.warning(
                        f"[VOTE] API timeout/connection error on attempt {attempt + 1}/{max_retries}. "
                        f"Waiting {delay:.1f}s before retry... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        f"[VOTE] API call failed after {max_retries} attempts: {e}. Returning uniform votes."
                    )
                    # Return uniform votes (no preference)
                    return [1.0] * len(states)

            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) * 3  # 6s, 12s, 24s
                    log.warning(
                        f"[VOTE] Rate limit hit on attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {delay:.1f}s... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        f"[VOTE] Rate limit persists after {max_retries} attempts: {e}. Returning uniform votes."
                    )
                    return [1.0] * len(states)

            except Exception as e:
                # For other errors, return uniform votes
                log.error(
                    f"[VOTE ERROR] Error in voting: {type(e).__name__}: {e}. Returning uniform votes."
                )
                return [1.0] * len(states)

        # Should never reach here
        return [1.0] * len(states)
