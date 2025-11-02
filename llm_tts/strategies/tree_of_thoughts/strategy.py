"""
Tree-of-Thoughts (ToT) strategy for LLM reasoning.

Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
by Yao et al. (2023). Explores multiple reasoning paths via beam search with
intermediate state evaluation using dedicated scorer classes.

Paper: https://arxiv.org/abs/2305.10601
Reference: https://github.com/princeton-nlp/tree-of-thought-llm
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import openai

from llm_tts.components import ReasoningGraph, ReasoningNode
from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.scorers.tree_of_thoughts import TotValueScorer

from ..metadata_builder import StrategyMetadataBuilder
from ..strategy_base import StrategyBase
from .parsers import extract_answer, extract_new_step, parse_proposals
from .prompts import build_cot_prompt, build_propose_prompt
from .validators import (
    filter_valid_game24_answers,
    get_current_numbers,
    validate_game24_answer,
)

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
        mode: str = "generic",
        method_generate: str = "propose",
        beam_width: int = 5,
        n_generate_sample: int = 5,
        steps: int = 4,
        temperature: float = 0.7,
        max_tokens_per_step: int = 1000,
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
            mode: Strategy mode - "generic" (default, any prompt) or "game24" (validation benchmark)
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
        self.mode = mode
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

        # Progress callback for real-time updates
        self.progress_callback = None

        # Configure mode-specific settings
        if mode == "game24":
            self._setup_game24_mode()
        elif mode == "generic":
            self._setup_generic_mode()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'generic' or 'game24'.")

    def set_progress_callback(self, callback):
        """
        Set progress callback for real-time updates.

        Args:
            callback: Function to call with progress updates.
                     Receives dict with keys: step, nodes_explored, api_calls, node, edge
        """
        self.progress_callback = callback

    def _report_progress(self, update: Dict[str, Any]):
        """Report progress if callback is set."""
        if self.progress_callback:
            try:
                self.progress_callback(update)
            except Exception as e:
                log.warning(f"Progress callback error: {e}")

    def _setup_generic_mode(self):
        """Configure for generic reasoning (any user prompt)."""
        self.validation_enabled = False
        log.info("[ToT] Mode: GENERIC - works for any user prompt")

    def _setup_game24_mode(self):
        """Configure for Game24 (paper-exact validation benchmark)."""
        self.validation_enabled = True
        self.target_value = 24
        log.info("[ToT] Mode: GAME24 - paper-exact implementation for validation")

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
        return get_current_numbers(state, problem)

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
        return validate_game24_answer(expression, input_numbers)

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
        return extract_answer(text)

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
                            f"API timeout/connection error on attempt "
                            f"{attempt + 1}/{max_retries}. "
                            f"Waiting {delay:.1f}s for background request "
                            f"to finish before retry... Error: {e}"
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
                n=self.n_generate_sample,  # Generate multiple proposals
                temperature=self.temperature,
                max_tokens=self.max_tokens_per_step,
            )

            if not outputs:
                log.warning(
                    "No outputs from proposal generation, skipping this candidate"
                )
                return []

            # Each output is a proposal - append to current state
            proposals = [
                (
                    current_state + "\n" + output.strip()
                    if current_state
                    else output.strip()
                )
                for output in outputs
            ]

            # Validate Game of 24 final answers (only in game24 mode)
            if self.validation_enabled:
                proposals = self._filter_valid_game24_answers(proposals, problem)

            return proposals

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

            # Validate Game of 24 final answers (only in game24 mode)
            if self.validation_enabled:
                samples = self._filter_valid_game24_answers(samples, problem)

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

        Dispatches to mode-specific implementations.
        """
        return build_propose_prompt(
            problem,
            state,
            self.mode,
            propose_prompt_path=self.propose_prompt_path,
        )

    def _filter_valid_game24_answers(
        self, proposals: List[str], problem: str
    ) -> List[str]:
        """
        Filter Game of 24 proposals to keep only valid final answers.

        For states with Answer lines, validates that the expression:
        1. Uses exactly the same numbers as the input
        2. Evaluates to 24 using sympy

        Non-answer proposals (with "(left: )" pattern) are kept as-is.

        Args:
            proposals: List of candidate states
            problem: Original problem statement

        Returns:
            Filtered list with only valid answers
        """
        return filter_valid_game24_answers(proposals, problem)

    def _build_cot_prompt(self, problem: str, state: str) -> str:
        """Build prompt for CoT sampling."""
        return build_cot_prompt(problem, state, cot_prompt_path=self.cot_prompt_path)

    def _parse_proposals(self, text: str, current_state: str) -> List[str]:
        """
        Parse proposals from model output (mode dispatcher).

        Delegates to mode-specific parsing methods:
        - game24: Parses "X op Y = Z (left: ...)" format
        - generic: Parses bullets, numbers, or paragraphs

        Args:
            text: Model output text
            current_state: Current reasoning state

        Returns:
            List of proposal states (each = current_state + new_line)
        """
        return parse_proposals(text, current_state, self.mode)

    def _generate_candidates(
        self,
        problem: str,
        parent_nodes: List[ReasoningNode],
        step_idx: int,
        timestamp: int,
    ) -> List[ReasoningNode]:
        """
        Generate candidate next states from current states.

        Args:
            problem: Original problem
            parent_nodes: Current nodes in beam
            step_idx: Current step index
            timestamp: Timestamp for new nodes

        Returns:
            List of candidate nodes
        """
        all_candidate_nodes = []

        # Generate from each parent node
        for parent_node in parent_nodes:
            # Generate candidate state strings
            if self.method_generate == "propose":
                candidate_states = self._generate_proposals(problem, parent_node.state)
            else:  # sample
                candidate_states = self._generate_samples(problem, parent_node.state)

            # Create child nodes for each candidate
            for candidate_state in candidate_states:
                child_node = self.graph.create_child(
                    parent=parent_node,
                    state=candidate_state,
                    score=0.0,  # Will be set by evaluation
                    timestamp=timestamp,
                    is_selected=False,  # Will be updated after selection
                    is_final=False,  # Will be updated if needed
                )
                all_candidate_nodes.append(child_node)

        return all_candidate_nodes

    def _evaluate_candidates(
        self,
        problem: str,
        candidate_nodes: List[ReasoningNode],
    ) -> List[float]:
        """
        Evaluate candidate states using the scorer.

        Args:
            problem: Original problem
            candidate_nodes: Candidate nodes to evaluate

        Returns:
            List of scores (one per candidate)
        """
        # Extract state strings from nodes
        states = [node.state for node in candidate_nodes]

        # Use the scorer to evaluate states
        scores = self.scorer.score_states(problem, states)

        # Update node scores
        for node, score in zip(candidate_nodes, scores):
            node.score = score

        return scores

    def _select_top_states(
        self,
        candidate_nodes: List[ReasoningNode],
    ) -> List[ReasoningNode]:
        """
        Select top-k candidates by score using stable sort.

        Uses stable sorting (matches original ToT) to preserve order for ties.

        Args:
            candidate_nodes: Candidate nodes with scores

        Returns:
            List of selected nodes
        """
        # Extract scores from nodes
        scores = [node.score for node in candidate_nodes]

        # Get top-k indices using stable sort (matches original ToT)
        if len(scores) <= self.beam_width:
            top_indices = list(range(len(scores)))
        else:
            # Stable sort: for ties, preserves original order
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: self.beam_width]

        # Select top nodes and mark them as selected
        selected_nodes = [candidate_nodes[i] for i in top_indices]
        for node in selected_nodes:
            node.is_selected = True

        return selected_nodes

    def _extract_new_step(self, candidate: str, parent_states: List[str]) -> str:
        """
        Extract the new step added to create this candidate.

        Args:
            candidate: Full candidate state
            parent_states: List of parent states from previous step

        Returns:
            The new step (last line added)
        """
        return extract_new_step(candidate, parent_states)

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

        # Initialize reasoning graph
        self.graph = ReasoningGraph(question=problem)

        # Create root node
        root = self.graph.create_root(state="", timestamp=0)

        # Initialize beam with root node
        current_beam = [root]

        # Track all steps for metadata (backward compatibility)
        all_steps = []

        # Reset statistics
        self.total_api_calls = 0

        # Report initial progress
        self._report_progress(
            {
                "step": 0,
                "nodes_explored": 0,
                "api_calls": 0,
            }
        )

        # Report root node
        self._report_progress({"node": root.to_dict()})

        # Beam search
        for step_idx in range(self.steps):
            log.info(f"\n{'='*80}")
            log.info(
                f"Step {step_idx + 1}/{self.steps}: {len(current_beam)} nodes in beam"
            )
            log.info(f"{'='*80}")

            # Report progress: starting new step
            self._report_progress(
                {
                    "step": step_idx + 1,
                    "nodes_explored": (
                        len(all_steps) * self.n_generate_sample if all_steps else 0
                    ),
                    "api_calls": self.total_api_calls,
                }
            )

            # GENERATE: Expand beam with candidate nodes
            candidate_nodes = self._generate_candidates(
                problem, current_beam, step_idx + 1, step_idx + 1
            )
            log.info(f"\n[GENERATE] Generated {len(candidate_nodes)} candidates:")

            # Show tree structure with parent relationships
            for i, candidate_node in enumerate(candidate_nodes):
                # Get parent node for display
                parent_idx = (
                    current_beam.index(candidate_node.parent)
                    if candidate_node.parent in current_beam
                    else -1
                )

                # Extract the new step (last line added)
                parent_states = [node.state for node in current_beam]
                new_step = self._extract_new_step(candidate_node.state, parent_states)

                # Show tree structure
                indent = "  " * (step_idx + 1)
                tree_marker = "└─" if i == len(candidate_nodes) - 1 else "├─"
                log.info(
                    f"{tree_marker} [Level {step_idx + 1}] "
                    f"Candidate {i+1} (from node {parent_idx + 1}):"
                )
                log.info(f"{indent}New step: {new_step}")

            if not candidate_nodes:
                log.warning(f"  No candidates generated at step {step_idx}")
                break

            # EVALUATE: Score all candidates using scorer
            scores = self._evaluate_candidates(problem, candidate_nodes)
            log.info(
                f"\n[EVALUATE] Scored {len(candidate_nodes)} candidates "
                f"(mean: {np.mean(scores):.2f}, std: {np.std(scores):.2f}):"
            )
            for i, (candidate_node, score) in enumerate(zip(candidate_nodes, scores)):
                log.info(f"  Candidate {i+1} (score={score:.2f}):")
                log.info(f"    {candidate_node.state}")

            # Report progress: candidates scored
            self._report_progress(
                {
                    "step": step_idx + 1,
                    "nodes_explored": (step_idx + 1) * self.n_generate_sample,
                    "api_calls": self.total_api_calls,
                }
            )

            # Report intermediate tree updates: new nodes and edges
            for candidate_node in candidate_nodes:
                self._report_progress({"node": candidate_node.to_dict()})
                if candidate_node.parent:
                    self._report_progress(
                        {
                            "edge": {
                                "from": candidate_node.parent.id,
                                "to": candidate_node.id,
                            }
                        }
                    )

            # SELECT: Prune to top-k
            current_beam = self._select_top_states(candidate_nodes)
            log.info(
                f"\n[SELECT] Selected top {len(current_beam)} nodes for next step:"
            )
            for i, node in enumerate(current_beam):
                # Extract last step for compact display
                last_step = (
                    node.state.strip().split("\n")[-1] if node.state else "(empty)"
                )
                log.info(f"  ✓ Node {i+1} (score={node.score:.2f}): {last_step}")

            # Record step (for backward compatibility with metadata)
            all_steps.append(
                {
                    "step_idx": step_idx,
                    "candidates": [node.state for node in candidate_nodes],
                    "scores": scores,
                    "selected_states": [node.state for node in current_beam],
                    "selected_scores": [node.score for node in current_beam],
                }
            )

        # Select best final state
        log.info("\n" + "=" * 80)
        log.info("FINAL SELECTION")
        log.info("=" * 80)

        if current_beam:
            # Re-evaluate final nodes
            final_scores = self._evaluate_candidates(problem, current_beam)
            log.info(f"\nEvaluating {len(current_beam)} final nodes:")
            for i, (node, score) in enumerate(zip(current_beam, final_scores)):
                log.info(f"  Final node {i+1} (score={score:.2f}):")
                log.info(f"    {node.state}")

            # Select highest scoring node using stable sort (matches original ToT)
            # Stable sort preserves original order for ties
            best_idx = sorted(
                range(len(final_scores)), key=lambda i: final_scores[i], reverse=True
            )[0]
            log.info(
                f"\n[SELECT] Selecting highest scoring node "
                f"(index {best_idx+1}, score {final_scores[best_idx]:.2f})"
            )

            best_node = current_beam[best_idx]
            best_node.is_final = True
            best_score = best_node.score
            best_answer = self._extract_answer(best_node.state)

            # Mark the selected path
            self.graph.mark_selected_path(best_node)

            # SYNTHESIS: Generate final answer from best reasoning path
            log.info("\n" + "=" * 80)
            log.info("FINAL ANSWER SYNTHESIS")
            log.info("=" * 80)

            synthesis_prompt = (
                "Based on the following reasoning steps, provide a complete, "
                "detailed answer to the original question.\n\n"
                f"Original Question:\n{problem}\n\n"
                f"Reasoning Steps Taken:\n{best_node.state}\n\n"
                "Now provide the actual answer with full details, not just "
                "the steps. Write a comprehensive response that addresses "
                "the question."
            )

            log.info("[SYNTHESIS] Generating final answer from reasoning path...")

            try:
                final_answer_list = self._call_model(
                    synthesis_prompt,
                    n=1,  # Generate just 1 answer
                    temperature=0.3,  # Lower temperature for final answer
                    max_tokens=2000,  # Allow longer answer
                )

                # Extract first (and only) answer from list
                if final_answer_list and len(final_answer_list) > 0:
                    final_answer_text = final_answer_list[0]
                    log.info(
                        f"[SYNTHESIS] Generated final answer "
                        f"({len(final_answer_text)} chars)"
                    )
                    # Use the synthesized answer as the trajectory
                    trajectory = final_answer_text
                else:
                    log.warning(
                        "[SYNTHESIS] No answer generated, using reasoning steps"
                    )
                    # Keep best_node.state as is (reasoning steps)
                    trajectory = best_node.state

                # Report final progress
                self._report_progress(
                    {
                        "step": self.steps,
                        "nodes_explored": sum(len(s["candidates"]) for s in all_steps),
                        "api_calls": self.total_api_calls,
                    }
                )
            except Exception as e:
                log.warning(f"[SYNTHESIS] Failed to generate final answer: {e}")
                log.info("[SYNTHESIS] Falling back to reasoning steps as answer")
                trajectory = best_node.state

            log.info(
                f"\n[BEST] Selected node {best_idx+1} with score {best_score:.2f}:"
            )
            log.info(f"  Full state:\n{best_node.state[:200]}...")
            log.info(f"  Extracted answer: {best_answer}")
        else:
            trajectory = ""
            best_score = 0.0
            best_answer = "no_answer"
            log.warning("No final nodes available!")

        log.info("\n" + "=" * 80)
        log.info("SEARCH SUMMARY")
        log.info("=" * 80)
        log.info(f"Best score: {best_score:.2f}")
        log.info(f"Extracted answer: {best_answer}")
        log.info(f"API calls: {self.total_api_calls}")
        log.info(f"Scorer evaluations: {self.scorer.total_evaluations}")

        # Get graph stats
        graph_stats = self.graph.get_stats()
        log.info(f"Graph stats: {graph_stats}")

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
            final_states=[node.state for node in current_beam] if current_beam else [],
            final_scores=[node.score for node in current_beam] if current_beam else [],
        )

        # Add search details
        builder.add_generation_details(
            all_steps=all_steps,
            total_api_calls=self.total_api_calls,
            scorer_evaluations=self.scorer.total_evaluations,
            total_candidates_evaluated=sum(
                len(step["candidates"]) for step in all_steps
            ),
            graph_stats=graph_stats,
        )

        # Log summary
        builder.log_summary(log)

        # Format output to match expected interface
        # Log graph structure to file for debugging
        graph_dict = self.graph.to_dict()
        self._save_graph_to_file(graph_dict, problem)

        return {
            "trajectory": trajectory,
            "steps": [step["selected_states"] for step in all_steps],
            "validity_scores": (
                [np.mean(step["selected_scores"]) for step in all_steps]
                if all_steps
                else [0.0]
            ),
            "completed": bool(current_beam and best_answer != "no_answer"),
            "strategy": "tree_of_thoughts",
            "metadata": builder.build(),
            "reasoning_tree": graph_dict,
        }

    def _save_graph_to_file(self, graph_dict: Dict[str, Any], problem: str) -> None:
        """
        Save reasoning graph to JSON file for debugging.

        Args:
            graph_dict: The graph dictionary from self.graph.to_dict()
            problem: The original problem/question
        """
        try:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.getcwd(), "logs", "reasoning_graphs")
            os.makedirs(logs_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize problem text for filename (first 50 chars)
            problem_slug = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in problem[:50]
            )
            filename = f"tot_graph_{timestamp}_{problem_slug}.json"
            filepath = os.path.join(logs_dir, filename)

            # Prepare data to save
            data = {
                "timestamp": timestamp,
                "problem": problem,
                "graph": graph_dict,
                "stats": self.graph.get_stats(),
            }

            # Save to file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            log.info(f"[GRAPH] Saved reasoning graph to: {filepath}")
        except Exception as e:
            log.warning(f"[GRAPH] Failed to save reasoning graph: {e}")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
