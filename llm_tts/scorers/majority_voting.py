"""
Majority voting scorer for self-consistency reasoning
"""

import logging
import re
from collections import Counter
from typing import List, Optional

from llm_tts.evaluation.parser import extract_answer as extract_answer_official
from llm_tts.utils import extract_answer as extract_answer_util

from .step_scorer_base import CandidateScore, StepScorerBase

log = logging.getLogger(__name__)


class MajorityVotingScorer(StepScorerBase):
    """
    Scorer that implements majority voting for self-consistency.
    Scores candidates based on how frequently their final answers appear
    across multiple reasoning paths.
    """

    def __init__(self, answer_extraction_patterns: List[str] = None):
        super().__init__("majority_voting")
        # Patterns to extract final answers from reasoning chains
        self.answer_patterns = answer_extraction_patterns or [
            r"\\boxed\{([^}]+)\}",  # LaTeX boxed answer (most specific)
            r"<Answer>:\s*(.+?)(?:\n|$)",
            r"The answer is\s*(.+?)(?:\n|\.|\$)",
            r"Therefore,?\s*(.+?)(?:\n|\.|\$)",
            r"Final answer:\s*(.+?)(?:\n|\.|\$)",
            r"Answer:\s*(.+?)(?:\n|\.|\$)",
        ]

    def prepare_model(self):
        """No model preparation needed for majority voting"""
        pass

    def extract_answer(self, text: str) -> str:
        """Extract the final answer from a reasoning chain"""
        text = text.strip()

        # Try each pattern
        for pattern in self.answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                answer = matches[-1].strip()  # Take the last match
                # Clean up the answer
                answer = re.sub(r"[^\w\s\-\+\*\/\(\)\.\,]", "", answer)
                return answer.lower()

        # If no pattern matches, try to extract number from the end
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return numbers[-1].lower()

        # Fallback: return last meaningful word/phrase
        words = text.split()
        if words:
            return words[-1].lower()

        return "unknown"

    def score_candidates_detailed(
        self, trajectory: str, candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        """
        Score candidates based on majority voting.

        For self-consistency, we need to see the full reasoning chains,
        not just individual steps. This scorer works best when candidates
        represent complete reasoning paths to final answers.
        """
        if not candidates:
            return []

        # Extract answers from all candidates
        answers = []
        for candidate in candidates:
            # Combine trajectory with candidate to get full reasoning chain
            full_chain = trajectory + candidate
            answer = self.extract_answer(full_chain)
            answers.append(answer)

        # Count answer frequencies
        answer_counts = Counter(answers)
        total_candidates = len(candidates)

        log.info(f"Answer frequency distribution: {dict(answer_counts)}")

        # Score each candidate based on how common its answer is
        detailed_scores = []
        for i, candidate in enumerate(candidates):
            answer = answers[i]
            frequency = answer_counts[answer]
            # Score is the proportion of candidates that gave this answer
            confidence_score = frequency / total_candidates

            candidate_score = CandidateScore(
                candidate_text=candidate,
                claim_scores=[confidence_score],  # Single score per candidate
                aggregate_scores={},
                metadata={
                    "scorer_type": "majority_voting",
                    "extracted_answer": answer,
                    "answer_frequency": frequency,
                    "total_candidates": total_candidates,
                    "all_answers": answers,
                },
            )
            detailed_scores.append(candidate_score)

        return detailed_scores


class ChainMajorityVotingScorer(StepScorerBase):
    """
    Alternative majority voting scorer that works on complete reasoning chains
    rather than individual steps. Better suited for self-consistency evaluation.
    """

    def __init__(
        self,
        answer_extraction_patterns: List[str] = None,
        data_name: Optional[str] = None,
    ):
        super().__init__("chain_majority_voting")
        self.data_name = data_name  # Dataset name for official extraction (e.g., "minerva_math", "math500")
        self.answer_patterns = answer_extraction_patterns or [
            r"\\boxed\{([^}]+)\}",  # LaTeX boxed answer (most specific)
            r"<Answer>:\s*(.+?)(?:\n|$)",
            r"The answer is\s*(.+?)(?:\n|\.|\$)",
            r"Therefore,?\s*(.+?)(?:\n|\.|\$)",
        ]

    def prepare_model(self):
        """No model preparation needed"""
        pass

    def extract_answer(self, text: str) -> str:
        """
        Extract final answer using official Qwen2.5-Math extraction.

        When data_name is provided, uses llm_tts.evaluation.parser.extract_answer
        which is the official extraction method matching the final evaluation.
        This ensures consistency between running accuracy and final evaluation.

        Supports:
        - \\boxed{} format with proper brace counting
        - Dataset-specific handling (e.g., skip_unit for minerva_math)
        - Proper normalization via strip_string()
        """
        # Use official extraction when data_name is available (preferred)
        if self.data_name:
            answer = extract_answer_official(text, self.data_name, use_last_number=True)
            if answer:
                return answer.strip()
            return "no_answer"

        # Fallback: Use utility function when data_name not specified
        answer = extract_answer_util(text, answer_format="auto")
        if answer:
            return answer.strip()

        # Fallback to legacy pattern matching
        text = text.strip()
        for pattern in self.answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].strip()

        # Fallback to number extraction
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]

        return "no_answer"

    def score_complete_chains(self, chains: List[str]) -> List[float]:
        """
        Score complete reasoning chains using majority voting.

        Args:
            chains: List of complete reasoning chains (prompt + reasoning + answer)

        Returns:
            List of scores (higher = more consensus)
        """
        if not chains:
            return []

        # Extract answers from all chains (original case preserved)
        answers = [self.extract_answer(chain) for chain in chains]

        # Count frequencies using lowercased answers for comparison
        # but keep original answers for display
        normalized_answers = [a.lower() for a in answers]
        answer_counts = Counter(normalized_answers)
        total_chains = len(chains)

        # Score each chain based on normalized answer frequency
        scores = []
        for norm_answer in normalized_answers:
            frequency = answer_counts[norm_answer]
            score = frequency / total_chains
            scores.append(score)

        log.info(f"Chain consensus scores: {scores}")
        log.info(f"Answer distribution: {dict(Counter(answers))}")

        return scores

    def score_candidates_detailed(
        self, trajectory: str, candidates: List[str], **kwargs
    ) -> List[CandidateScore]:
        """Score candidates - but this scorer is meant for complete chains"""
        # For individual steps, we can't do proper majority voting
        # Return uniform scores as fallback
        uniform_score = 1.0 / len(candidates) if candidates else 0.0

        detailed_scores = []
        for candidate in candidates:
            candidate_score = CandidateScore(
                candidate_text=candidate,
                claim_scores=[uniform_score],
                aggregate_scores={},
                metadata={
                    "scorer_type": "chain_majority_voting",
                    "note": "uniform_scores_for_individual_steps",
                },
            )
            detailed_scores.append(candidate_score)

        return detailed_scores
