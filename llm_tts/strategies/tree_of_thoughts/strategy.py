"""
Tree-of-Thoughts (ToT) strategy for LLM reasoning.

Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
by Yao et al. (2023). Explores multiple reasoning paths via beam search with
intermediate state evaluation using dedicated scorer classes.

Paper: https://arxiv.org/abs/2305.10601
Reference: https://github.com/princeton-nlp/tree-of-thought-llm
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.scorers.tree_of_thoughts import TotValueScorer

from ..metadata_builder import StrategyMetadataBuilder
from ..strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyTreeOfThoughts(StrategyBase):
    """
    Tree-of-Thoughts strategy using beam search with state evaluation scorers.

    Key components:
    1. Generate: Propose multiple next steps from each current state
    2. Evaluate: Score each candidate state using ToT scorers
    3. Select: Prune to top-k states (beam width) for next iteration
    4. Repeat: Continue until max steps or solution found
    """

    def __init__(
        self,
        model,
        scorer=None,
        method_generate: str = "propose",
        beam_width: int = 5,
        n_generate_sample: int = 5,
        steps: int = 4,
        temperature: float = 0.7,
        max_tokens_per_step: int = 100,
        n_threads: int = 8,
        scorer_timeout: int = 120,
        propose_prompt_path: str = None,
        cot_prompt_path: str = None,
        value_prompt_path: str = None,
        value_last_step_prompt_path: str = None,
    ):
        """
        Initialize Tree-of-Thoughts strategy.

        Args:
            model: Language model for generation
            scorer: ToT state scorer (TotValueScorer or TotVoteScorer).
                   If None, defaults to TotValueScorer.
            method_generate: Generation method ("propose" or "sample")
                - propose: Sequential next-step proposal given current state
                - sample: Independent step generation with CoT
            beam_width: Number of states to keep at each step
            n_generate_sample: Number of candidates to generate per state
            steps: Maximum number of reasoning steps
            temperature: Sampling temperature for generation
            max_tokens_per_step: Maximum tokens per reasoning step
            n_threads: Number of parallel threads for API calls
            scorer_timeout: Timeout in seconds for scorer evaluation calls (default: 120s)
            propose_prompt_path: Path to propose prompt template
            cot_prompt_path: Path to CoT prompt template (for final answers)
            value_prompt_path: Path to value scorer prompt template
            value_last_step_prompt_path: Path to final answer validation prompt
        """
        self.model = model
        self.method_generate = method_generate
        self.beam_width = beam_width
        self.n_generate_sample = n_generate_sample
        self.steps = steps
        self.temperature = temperature
        self.max_tokens_per_step = max_tokens_per_step
        self.n_threads = n_threads

        # Store prompt paths
        self.propose_prompt_path = propose_prompt_path
        self.cot_prompt_path = cot_prompt_path
        self.value_prompt_path = value_prompt_path
        self.value_last_step_prompt_path = value_last_step_prompt_path

        # Initialize scorer
        if scorer is None:
            # Default to value scorer matching original ToT paper parameters
            # (Yao et al., 2023: https://arxiv.org/abs/2305.10601)
            self.scorer = TotValueScorer(
                model=model,
                n_evaluate_sample=3,  # Paper: 3 samples, aggregate scores
                temperature=0.7,  # Paper: 0.7 temperature
                timeout=scorer_timeout,
                value_prompt_path=value_prompt_path,
                value_last_step_prompt_path=value_last_step_prompt_path,
            )
        else:
            self.scorer = scorer

        # Statistics tracking
        self.total_api_calls = 0

    def get_current_numbers(self, state: str, problem: str) -> str:
        """
        Extract remaining numbers from state for Game of 24.

        Looks for (left: X Y Z) pattern. If not found, returns original problem numbers.

        Args:
            state: Current partial solution
            problem: Original problem

        Returns:
            String of remaining numbers (e.g., "3 8 10")
        """
        if not state.strip():
            return problem.strip()

        # Get last line of state
        last_line = state.strip().split("\n")[-1]

        # Check for (left: ...) pattern
        if "(left: " in last_line:
            # Extract numbers after "left: " and before ")"
            remaining = last_line.split("left: ")[-1].split(")")[0]
            return remaining.strip()

        # No pattern found, return original problem
        return problem.strip()

    def validate_game24_answer(self, expression: str, input_numbers: str) -> bool:
        """
        Validate a Game of 24 answer using sympy.

        Checks:
        1. Expression uses exactly the same numbers as input (no more, no less)
        2. Expression evaluates to 24

        Args:
            expression: Mathematical expression (e.g., "(4 + 8) * (6 - 4)")
            input_numbers: Original input numbers (e.g., "4 4 6 8")

        Returns:
            True if answer is correct, False otherwise
        """
        try:
            import sympy

            # Extract numbers from expression
            used_numbers = re.findall(r"\d+", expression)
            problem_numbers = re.findall(r"\d+", input_numbers)

            # Check if numbers match (same count and values)
            if sorted(used_numbers) != sorted(problem_numbers):
                log.debug(
                    f"Number mismatch: used {used_numbers}, expected {problem_numbers}"
                )
                return False

            # Evaluate expression using sympy
            result = sympy.simplify(expression)
            is_correct = result == 24

            log.debug(f"Expression '{expression}' = {result}, correct={is_correct}")
            return is_correct

        except Exception as e:
            log.debug(f"Validation error for '{expression}': {e}")
            return False

    def _extract_answer(self, text: str) -> str:
        """
        Extract numerical answer from reasoning text.

        For Game of 24:
        - Looks for "Answer: expression" format
        - Looks for final expression in parentheses
        - Validates using sympy if possible

        Other formats:
        - \\boxed{answer}
        - answer = value
        - Last number in text

        Args:
            text: Reasoning text

        Returns:
            Extracted answer string
        """
        text = text.strip()

        # Try "Answer:" format first (for Game of 24)
        answer_match = re.search(r"Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if answer_match:
            answer_expr = answer_match.group(1).strip()
            # Remove trailing text like "= 24"
            answer_expr = re.sub(r"\s*=\s*24\s*$", "", answer_expr)
            return answer_expr

        # Try \\boxed{} format
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try final line with (left: 24) pattern
        if "(left: 24)" in text:
            # Found solution! Extract the final expression
            lines = text.strip().split("\n")
            for line in reversed(lines):
                if "answer:" in line.lower():
                    expr = line.split(":", 1)[-1].strip()
                    expr = re.sub(r"\s*=\s*24\s*$", "", expr)
                    return expr

        # Try "= value" at end of line
        equals_match = re.search(r"=\s*([0-9,.]+)(?:\s|$)", text)
        if equals_match:
            return equals_match.group(1).strip()

        # Fallback: last number
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return numbers[-1]

        return "no_answer"

    def _call_model(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Call model with given prompt and return generated texts.

        Includes retry logic with exponential backoff for timeouts and rate limits.

        Args:
            prompt: Input prompt
            n: Number of completions to generate
            temperature: Sampling temperature (None = use default)
            max_tokens: Max tokens per completion (None = use default)

        Returns:
            List of generated texts
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens_per_step

        self.total_api_calls += 1

        if isinstance(self.model, BlackboxModelWithStreaming):
            # Convert to chat format
            messages = [{"role": "user", "content": prompt}]

            # Retry logic with exponential backoff
            max_retries = 3
            base_delay = 2.0
            timeout = 60  # 60s timeout per attempt

            for attempt in range(max_retries):
                try:
                    log.info(
                        f"[STRATEGY] Attempt {attempt + 1}/{max_retries}, timeout={timeout}s"
                    )

                    # Generate texts with timeout enforcement
                    results = self.model.generate_texts(
                        chats=[messages] * n,
                        max_new_tokens=max_tok,
                        temperature=temp,
                        timeout=timeout,
                    )

                    # Extract texts
                    texts = []
                    for result in results:
                        if result and result.get("text"):
                            texts.append(result["text"])

                    return texts

                except (openai.APITimeoutError, openai.APIConnectionError) as e:
                    if attempt < max_retries - 1:
                        # Longer delay to let previous background thread finish
                        delay = 10 + (base_delay * (2**attempt))  # 12s, 14s, 18s
                        log.warning(
                            f"API timeout/connection error on attempt {attempt + 1}/{max_retries}. "
                            f"Waiting {delay:.1f}s for background request to finish before retry... Error: {e}"
                        )
                        time.sleep(delay)
                    else:
                        log.error(f"API call failed after {max_retries} attempts: {e}")
                        raise

                except openai.RateLimitError as e:
                    if attempt < max_retries - 1:
                        delay = (
                            base_delay * (2**attempt) * 3
                        )  # Longer delay for rate limits: 6s, 12s, 24s
                        log.warning(
                            f"Rate limit hit on attempt {attempt + 1}/{max_retries}. "
                            f"Retrying in {delay:.1f}s... Error: {e}"
                        )
                        time.sleep(delay)
                    else:
                        log.error(
                            f"Rate limit persists after {max_retries} attempts: {e}"
                        )
                        raise

                except Exception as e:
                    # For other errors, don't retry
                    log.error(f"API call failed with unexpected error: {e}")
                    raise

            # Should never reach here, but return empty list as fallback
            return []
        else:
            # Local model - not yet implemented
            raise NotImplementedError("Local models not yet supported for ToT")

    def _generate_proposals(
        self,
        problem: str,
        current_state: str,
    ) -> List[str]:
        """
        Generate next-step proposals given current state.

        Uses the "propose" method: given problem and partial solution,
        what are the possible next steps?

        Args:
            problem: Original problem statement
            current_state: Current partial solution

        Returns:
            List of proposed next steps
        """
        # Build proposal prompt
        if current_state:
            prompt = self._build_propose_prompt(problem, current_state)
        else:
            # Initial state - generate first steps
            prompt = self._build_propose_prompt(problem, "")

        # Generate proposals with error handling
        try:
            outputs = self._call_model(
                prompt,
                n=1,  # Single call, extract multiple proposals from output
                temperature=self.temperature,
                max_tokens=self.max_tokens_per_step
                * 3,  # More tokens for multiple proposals
            )

            if not outputs:
                log.warning(
                    "No outputs from proposal generation, skipping this candidate"
                )
                return []

            # Parse proposals from output
            proposals_text = outputs[0]
            proposals = self._parse_proposals(proposals_text, current_state)

            return proposals[: self.n_generate_sample]

        except openai.APITimeoutError as e:
            log.error(
                f"Proposal generation timed out after all retries: {e}. Skipping this state."
            )
            return []  # Return empty list to skip this candidate
        except Exception as e:
            log.error(f"Proposal generation failed: {e}. Skipping this state.")
            return []

    def _generate_samples(
        self,
        problem: str,
        current_state: str,
    ) -> List[str]:
        """
        Generate independent reasoning steps via sampling.

        Uses the "sample" method: generate next step independently
        using chain-of-thought prompting.

        Args:
            problem: Original problem statement
            current_state: Current partial solution

        Returns:
            List of sampled next steps
        """
        # Build sampling prompt with CoT
        prompt = self._build_cot_prompt(problem, current_state)

        # Generate samples with error handling
        try:
            outputs = self._call_model(
                prompt,
                n=self.n_generate_sample,
                temperature=self.temperature,
            )

            if not outputs:
                log.warning(
                    "No outputs from sample generation, skipping this candidate"
                )
                return []

            # Append to current state
            samples = [current_state + output for output in outputs]

        except openai.APITimeoutError as e:
            log.error(
                f"Sample generation timed out after all retries: {e}. Skipping this state."
            )
            return []  # Return empty list to skip this candidate
        except Exception as e:
            log.error(f"Sample generation failed: {e}. Skipping this state.")
            return []

        return samples

    def _build_propose_prompt(self, problem: str, state: str) -> str:
        """
        Build prompt for next-step proposal.

        For Game of 24, uses the original ToT prompt format that generates
        multiple possible next steps in the format: "X op Y = Z (left: remaining)"
        """
        # Check if this is a Game of 24 problem
        # Look for "obtain 24" phrase (specific to Game24) or check first line format
        is_game24 = "obtain 24" in problem.lower()
        if not is_game24:
            # Alternative check: first line has format "Input: X Y Z W" with 2-4 numbers
            first_line = problem.strip().split("\n")[0]
            if first_line.startswith("Input:"):
                input_numbers = re.findall(r"\d+", first_line.replace("Input:", ""))
                is_game24 = len(input_numbers) in [2, 3, 4]

        if is_game24:
            # Get current numbers (either from state or original problem)
            current_numbers = self.get_current_numbers(state, problem)

            # CRITICAL: When we reach (left: 24), switch to CoT prompt
            # to generate the final "Answer: expression = 24" line
            if current_numbers == "24":
                # Load CoT prompt with full examples (if configured)
                if self.cot_prompt_path:
                    try:
                        with open(self.cot_prompt_path, "r") as f:
                            template = f.read()
                        # Append "Steps:" + current state to prompt model for final answer
                        return template.format(input=problem) + "Steps:\n" + state
                    except FileNotFoundError:
                        pass
                # Fallback
                return f"""Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: {problem}
Steps:
{state}
"""
            else:
                # Generate next step proposals
                if self.propose_prompt_path:
                    try:
                        with open(self.propose_prompt_path, "r") as f:
                            template = f.read()
                        return template.format(input=current_numbers)
                    except FileNotFoundError:
                        pass

                # Fallback to inline prompt
                return f"""Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {current_numbers}
Possible next steps:
"""

        # Non-Game24 problems use original verbose prompt
        if not state:
            return f"""Solve the following math problem step-by-step. Provide 3-5 CONCRETE first steps with specific calculations or actions:

Problem: {problem}

Provide specific next steps (not just plans, but actual reasoning with numbers):
- Step 1: [concrete action with calculations]
- Step 2: [concrete action with calculations]
- Step 3: [concrete action with calculations]

Example format:
- Step 1: Identify that we start with 16 eggs total
- Step 2: Calculate eggs remaining after breakfast: 16 - 3 = 13 eggs
"""
        else:
            # Extract intermediate state
            return f"""Continue solving this math problem with CONCRETE steps. Show specific calculations, not just plans.

Problem: {problem}

Current progress:
{state}

Provide 3-5 specific next steps with calculations:
- Next step 1: [perform specific calculation]
- Next step 2: [perform specific calculation]
- Next step 3: [perform specific calculation]

Show your work with actual numbers. DO NOT just describe what you would do - actually do it.
"""

    def _build_cot_prompt(self, problem: str, state: str) -> str:
        """Build prompt for CoT sampling."""
        if not state:
            prompt_prefix = f"""Solve this math problem step-by-step with concrete calculations:

Problem: {problem}

Solution (show all work with numbers and calculations):
"""
        else:
            prompt_prefix = f"""Continue solving this math problem with specific calculations:

Problem: {problem}

Current progress:
{state}

Continue with concrete steps and numbers (show your work):
"""
        return prompt_prefix

    def _parse_proposals(self, text: str, current_state: str) -> List[str]:
        """
        Parse proposals from model output.

        For Game of 24, expects format: "X op Y = Z (left: remaining)"
        For other problems, extracts bullet points or numbered items.
        """
        proposals = []

        # Check if this is Game of 24 format (contains "(left: " or "Answer:" pattern)
        lines = text.split("\n")
        has_game24_format = any(
            "(left: " in line or "answer:" in line.lower() for line in lines
        )

        if has_game24_format:
            # Parse Game of 24 proposals
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Accept lines with "(left: " OR "Answer:" (for final step)
                if "(left: " in line or "answer:" in line.lower():
                    # This line is a complete next step
                    # Append to current state
                    full_step = current_state + ("\n" if current_state else "") + line
                    proposals.append(full_step)

        else:
            # Parse regular proposals (bullet points, numbered items)
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Remove bullet markers
                line = re.sub(r"^[-*•]\s*", "", line)
                line = re.sub(
                    r"^(?:Step|Next step)\s*\d+:\s*", "", line, flags=re.IGNORECASE
                )

                if line and len(line) > 10:  # Minimum length check
                    # Append to current state
                    full_step = current_state + ("\n" if current_state else "") + line
                    proposals.append(full_step)

        # Fallback: treat whole output as single proposal
        if not proposals and text.strip():
            proposals.append(
                current_state + ("\n" if current_state else "") + text.strip()
            )

        return proposals

    def _is_final_answer(self, state: str) -> bool:
        """
        Check if state contains a final answer.

        For Game of 24, checks for (left: 24) pattern.
        For other problems, checks for various answer indicators.
        """
        state_lower = state.lower()

        # Game of 24: solution reached when (left: 24)
        if "(left: 24)" in state_lower:
            return True

        # Other answer formats
        return any(
            [
                "answer:" in state_lower,
                "\\boxed{" in state,
                state.strip().endswith(" = 24"),  # Game 24 final expression
            ]
        )

    def _generate_candidates(
        self,
        problem: str,
        states: List[str],
    ) -> List[str]:
        """
        Generate candidate next states from current states.

        Args:
            problem: Original problem
            states: Current states in beam

        Returns:
            List of candidate states
        """
        all_candidates = []

        # Generate from each state
        for state in states:
            if self.method_generate == "propose":
                candidates = self._generate_proposals(problem, state)
            else:  # sample
                candidates = self._generate_samples(problem, state)

            all_candidates.extend(candidates)

        return all_candidates

    def _evaluate_candidates(
        self,
        problem: str,
        candidates: List[str],
    ) -> List[float]:
        """
        Evaluate candidate states using the scorer.

        Args:
            problem: Original problem
            candidates: Candidate states to evaluate

        Returns:
            List of scores (one per candidate)
        """
        # Use the scorer to evaluate states
        scores = self.scorer.score_states(problem, candidates)
        return scores

    def _select_top_states(
        self,
        candidates: List[str],
        scores: List[float],
    ) -> Tuple[List[str], List[float]]:
        """
        Select top-k candidates by score.

        Args:
            candidates: Candidate states
            scores: Scores for each candidate

        Returns:
            Tuple of (selected_states, selected_scores)
        """
        # Deduplicate candidates
        unique_map = {}
        for cand, score in zip(candidates, scores):
            if cand not in unique_map or score > unique_map[cand]:
                unique_map[cand] = score

        # Sort by score
        sorted_items = sorted(unique_map.items(), key=lambda x: x[1], reverse=True)

        # Take top-k
        top_items = sorted_items[: self.beam_width]

        top_states = [item[0] for item in top_items]
        top_scores = [item[1] for item in top_items]

        return top_states, top_scores

    def generate_trajectory(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry point for Tree-of-Thoughts reasoning.

        Args:
            prompt: Input prompt/question (string or message list)

        Returns:
            Dictionary with trajectory information compatible with evaluation framework:
            {
                'trajectory': str,  # Best final state
                'steps': List[str],  # Reasoning steps
                'validity_scores': List[float],  # Scores per step
                'completed': bool,  # Success flag
                'metadata': Dict,  # Detailed execution info
            }
        """
        # Handle message format
        if isinstance(prompt, list):
            # Extract user message
            for msg in prompt:
                if msg.get("role") == "user":
                    problem = msg.get("content", "")
                    break
            else:
                problem = ""
        else:
            problem = prompt

        log.info("\n" + "=" * 80)
        log.info("TREE-OF-THOUGHTS SEARCH")
        log.info("=" * 80)
        log.info(f"\nProblem:\n{problem}")
        log.info("\nConfiguration:")
        log.info(f"  - Beam width: {self.beam_width}")
        log.info(f"  - Steps: {self.steps}")
        log.info(f"  - Generation method: {self.method_generate}")
        log.info(f"  - Scorer: {self.scorer.name}")
        log.info(f"  - Temperature: {self.temperature}")

        # Initialize beam
        states = [""]  # Start with empty state
        all_steps = []

        # Reset statistics
        self.total_api_calls = 0

        # Beam search
        for step_idx in range(self.steps):
            log.info(f"\n{'='*80}")
            log.info(f"Step {step_idx + 1}/{self.steps}: {len(states)} states in beam")
            log.info(f"{'='*80}")

            # GENERATE: Expand beam with candidates
            candidates = self._generate_candidates(problem, states)
            log.info(f"\n[GENERATE] Generated {len(candidates)} candidates:")
            for i, candidate in enumerate(candidates):
                log.info(f"  Candidate {i+1}:")
                log.info(f"    {candidate}")

            if not candidates:
                log.warning(f"  No candidates generated at step {step_idx}")
                break

            # EVALUATE: Score all candidates using scorer
            scores = self._evaluate_candidates(problem, candidates)
            log.info(
                f"\n[EVALUATE] Scored {len(candidates)} candidates (mean: {np.mean(scores):.2f}, std: {np.std(scores):.2f}):"
            )
            for i, (candidate, score) in enumerate(zip(candidates, scores)):
                log.info(f"  Candidate {i+1} (score={score:.2f}):")
                log.info(f"    {candidate}")

            # SELECT: Prune to top-k
            states, state_scores = self._select_top_states(candidates, scores)
            log.info(f"\n[SELECT] Selected top {len(states)} states:")
            for i, (state, score) in enumerate(zip(states, state_scores)):
                log.info(f"  State {i+1} (score={score:.2f}):")
                log.info(f"    {state}")

            # Record step
            all_steps.append(
                {
                    "step_idx": step_idx,
                    "candidates": candidates,
                    "scores": scores,
                    "selected_states": states,
                    "selected_scores": state_scores,
                }
            )

            # Check if any state has reached final answer
            final_states = [s for s in states if self._is_final_answer(s)]
            if final_states:
                log.info(f"\n[FINAL] Found {len(final_states)} final answer(s):")
                for i, final_state in enumerate(final_states):
                    log.info(f"  Final {i+1}: {final_state}")
                # Could early stop here, but continue to explore

        # Select best final state
        log.info("\n" + "=" * 80)
        log.info("FINAL SELECTION")
        log.info("=" * 80)

        if states:
            final_scores = self._evaluate_candidates(problem, states)
            log.info(f"\nEvaluating {len(states)} final states:")
            for i, (state, score) in enumerate(zip(states, final_scores)):
                log.info(f"  Final state {i+1} (score={score:.2f}):")
                log.info(f"    {state}")

            # IMPROVED SELECTION LOGIC:
            # 1. First, identify states that have reached a valid final answer
            final_answer_indices = [
                i for i, state in enumerate(states) if self._is_final_answer(state)
            ]

            if final_answer_indices:
                # If we have states with final answers, select the highest scoring one among them
                final_answer_scores = [final_scores[i] for i in final_answer_indices]
                best_among_finals = np.argmax(final_answer_scores)
                best_idx = final_answer_indices[best_among_finals]
                log.info(
                    f"\n[SELECT] Found {len(final_answer_indices)} states with final answers"
                )
                log.info(
                    f"[SELECT] Selecting best among them (index {best_idx+1}, score {final_scores[best_idx]:.2f})"
                )
            else:
                # No final answers found, select by highest score (fallback)
                best_idx = np.argmax(final_scores)
                log.info(
                    "\n[SELECT] No final answers found, selecting highest scoring state"
                )

            best_state = states[best_idx]
            best_score = final_scores[best_idx]
            best_answer = self._extract_answer(best_state)

            log.info(
                f"\n[BEST] Selected state {best_idx+1} with score {best_score:.2f}:"
            )
            log.info(f"  Full state:\n{best_state}")
            log.info(f"  Extracted answer: {best_answer}")
        else:
            best_state = ""
            best_score = 0.0
            best_answer = "no_answer"
            log.warning("No final states available!")

        log.info("\n" + "=" * 80)
        log.info("SEARCH SUMMARY")
        log.info("=" * 80)
        log.info(f"Best score: {best_score:.2f}")
        log.info(f"Extracted answer: {best_answer}")
        log.info(f"API calls: {self.total_api_calls}")
        log.info(f"Scorer evaluations: {self.scorer.total_evaluations}")

        # Build metadata
        builder = StrategyMetadataBuilder("tree_of_thoughts")

        # Add configuration
        builder.add_config(
            method_generate=self.method_generate,
            scorer=str(self.scorer),
            beam_width=self.beam_width,
            n_generate_sample=self.n_generate_sample,
            steps=self.steps,
            temperature=self.temperature,
        )

        # Add results
        builder.add_results(
            selected_answer=best_answer,
            best_score=best_score,
            final_states=states,
            final_scores=final_scores if states else [],
        )

        # Add search details
        builder.add_generation_details(
            all_steps=all_steps,
            total_api_calls=self.total_api_calls,
            scorer_evaluations=self.scorer.total_evaluations,
            total_candidates_evaluated=sum(
                len(step["candidates"]) for step in all_steps
            ),
        )

        # Log summary
        builder.log_summary(log)

        # Format output to match expected interface
        return {
            "trajectory": best_state,
            "steps": [step["selected_states"] for step in all_steps],
            "validity_scores": (
                [np.mean(step["selected_scores"]) for step in all_steps]
                if all_steps
                else [0.0]
            ),
            "completed": bool(states and best_answer != "no_answer"),
            "strategy": "tree_of_thoughts",
            "metadata": builder.build(),
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
