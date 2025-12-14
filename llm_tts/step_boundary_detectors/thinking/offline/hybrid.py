"""
Hybrid step boundary detector for thinking mode.

Combines marker-based detection with fallback strategies.
"""

import logging
import re
from typing import Any, List, Optional

from ...base import StepBoundaryDetectorBase
from ..marker import ThinkingMarkerDetector
from .sentence import ThinkingSentenceDetector

log = logging.getLogger(__name__)


class ThinkingHybridDetector(StepBoundaryDetectorBase):
    """
    Hybrid detector that combines multiple strategies.

    Strategy:
    1. Try marker-based detection first
    2. If too few steps found, fall back to sentence/paragraph splitting
    3. If steps are too long, chunk them
    4. Optionally use LLM for refinement

    Best for: Robust step detection across different thinking styles.
    """

    def __init__(
        self,
        min_steps: int = 2,
        max_steps: int = 20,
        min_step_chars: int = 50,
        max_step_chars: int = 800,
        marker_detector: Optional[ThinkingMarkerDetector] = None,
        sentence_detector: Optional[ThinkingSentenceDetector] = None,
        llm_detector: Optional[Any] = None,
        use_llm_refinement: bool = False,
    ):
        """
        Args:
            min_steps: Minimum expected steps (triggers fallback if fewer)
            max_steps: Maximum steps to return
            min_step_chars: Minimum characters per step
            max_step_chars: Maximum characters per step
            marker_detector: Custom marker detector instance
            sentence_detector: Custom sentence detector instance
            llm_detector: Optional LLM detector for refinement
            use_llm_refinement: Whether to use LLM for final refinement
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.min_step_chars = min_step_chars
        self.max_step_chars = max_step_chars
        self.use_llm_refinement = use_llm_refinement

        # Initialize sub-detectors
        self.marker_detector = marker_detector or ThinkingMarkerDetector(
            min_step_chars=min_step_chars,
            max_step_chars=max_step_chars,
        )
        self.sentence_detector = sentence_detector or ThinkingSentenceDetector(
            split_mode="both",
            min_step_chars=min_step_chars,
            max_step_chars=max_step_chars,
        )
        self.llm_detector = llm_detector

    def detect_steps(self, text: str, **kwargs) -> List[str]:
        """
        Detect steps using hybrid approach.

        Args:
            text: Thinking content (inside <think> tags)

        Returns:
            List of step strings
        """
        text = self._extract_thinking_content(text)

        if not text.strip():
            return []

        # Strategy 1: Try marker-based detection
        steps = self.marker_detector.detect_steps(text)
        log.debug(f"Marker detection found {len(steps)} steps")

        # Check if we need fallback
        if len(steps) < self.min_steps:
            log.debug(
                f"Too few steps ({len(steps)} < {self.min_steps}), "
                "falling back to sentence detection"
            )
            steps = self.sentence_detector.detect_steps(text)
            log.debug(f"Sentence detection found {len(steps)} steps")

        # Ensure steps are within size bounds
        steps = self._normalize_step_sizes(steps)

        # Limit total steps
        if len(steps) > self.max_steps:
            log.debug(f"Limiting from {len(steps)} to {self.max_steps} steps")
            steps = self._merge_to_limit(steps, self.max_steps)

        # Optional LLM refinement
        if self.use_llm_refinement and self.llm_detector:
            try:
                refined_steps = self.llm_detector.detect_steps(text)
                if self.min_steps <= len(refined_steps) <= self.max_steps:
                    steps = refined_steps
                    log.debug(f"LLM refinement: {len(steps)} steps")
            except Exception as e:
                log.warning(f"LLM refinement failed, using heuristic result: {e}")

        return steps

    def _extract_thinking_content(self, text: str) -> str:
        """Extract content from <think> tags if present."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return text.strip()

    def _normalize_step_sizes(self, steps: List[str]) -> List[str]:
        """Ensure all steps are within size bounds."""
        normalized = []

        for step in steps:
            if len(step) > self.max_step_chars:
                # Split long steps
                chunks = self._chunk_text(step, self.max_step_chars)
                normalized.extend(chunks)
            elif len(step) < self.min_step_chars and normalized:
                # Merge short step with previous
                normalized[-1] = normalized[-1] + "\n" + step
            else:
                normalized.append(step)

        return normalized

    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks of approximately max_chars."""
        chunks = []
        current_chunk = ""

        # Try to split at sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _merge_to_limit(self, steps: List[str], limit: int) -> List[str]:
        """Merge steps to fit within limit."""
        if len(steps) <= limit:
            return steps

        # Calculate how many steps to merge
        merge_factor = len(steps) / limit

        merged = []
        current_group = []

        for i, step in enumerate(steps):
            current_group.append(step)

            # Check if we should merge this group
            if len(current_group) >= merge_factor or i == len(steps) - 1:
                merged_step = "\n".join(current_group)
                merged.append(merged_step)
                current_group = []

                if len(merged) >= limit:
                    # Add any remaining steps to last group
                    if i < len(steps) - 1:
                        remaining = "\n".join(steps[i + 1 :])
                        merged[-1] = merged[-1] + "\n" + remaining
                    break

        return merged


class ThinkingAdaptiveDetector(StepBoundaryDetectorBase):
    """
    Adaptive detector that selects strategy based on content analysis.

    Analyzes the thinking content first, then chooses the best detection
    strategy based on characteristics like:
    - Presence of markers
    - Text structure (paragraphs, lists)
    - Content length
    """

    def __init__(
        self,
        marker_detector: Optional[ThinkingMarkerDetector] = None,
        sentence_detector: Optional[ThinkingSentenceDetector] = None,
        hybrid_detector: Optional[ThinkingHybridDetector] = None,
        min_step_chars: int = 50,
        max_step_chars: int = 800,
    ):
        """
        Args:
            marker_detector: Marker-based detector instance
            sentence_detector: Sentence-based detector instance
            hybrid_detector: Hybrid detector instance
            min_step_chars: Minimum characters per step
            max_step_chars: Maximum characters per step
        """
        self.min_step_chars = min_step_chars
        self.max_step_chars = max_step_chars

        self.marker_detector = marker_detector or ThinkingMarkerDetector(
            min_step_chars=min_step_chars,
            max_step_chars=max_step_chars,
        )
        self.sentence_detector = sentence_detector or ThinkingSentenceDetector(
            split_mode="both",
            min_step_chars=min_step_chars,
            max_step_chars=max_step_chars,
        )
        self.hybrid_detector = hybrid_detector or ThinkingHybridDetector(
            min_step_chars=min_step_chars,
            max_step_chars=max_step_chars,
        )

    def detect_steps(self, text: str, **kwargs) -> List[str]:
        """
        Detect steps using adaptively selected strategy.

        Args:
            text: Thinking content (inside <think> tags)

        Returns:
            List of step strings
        """
        text = self._extract_thinking_content(text)

        if not text.strip():
            return []

        # Analyze content characteristics
        analysis = self._analyze_content(text)
        log.debug(f"Content analysis: {analysis}")

        # Select strategy based on analysis
        strategy = self._select_strategy(analysis)
        log.debug(f"Selected strategy: {strategy}")

        # Apply selected strategy
        if strategy == "marker":
            return self.marker_detector.detect_steps(text)
        elif strategy == "sentence":
            return self.sentence_detector.detect_steps(text)
        else:  # "hybrid"
            return self.hybrid_detector.detect_steps(text)

    def _extract_thinking_content(self, text: str) -> str:
        """Extract content from <think> tags if present."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return text.strip()

    def _analyze_content(self, text: str) -> dict:
        """Analyze content characteristics."""
        return {
            "length": len(text),
            "num_paragraphs": len(re.split(r"\n\s*\n", text)),
            "num_sentences": len(re.split(r"(?<=[.!?])\s+", text)),
            "has_markers": bool(
                re.search(
                    r"\b(first|then|next|so|therefore|let me|wait)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "has_structure": bool(re.search(r"\n[-*\d]\s", text)),
            "has_lists": bool(re.search(r"\n\d+\.\s|\n-\s|\n\*\s", text)),
        }

    def _select_strategy(self, analysis: dict) -> str:
        """Select the best strategy based on content analysis."""
        # If text has clear markers, use marker-based
        if analysis["has_markers"] and analysis["num_paragraphs"] >= 2:
            return "marker"

        # If text has list structure, use marker-based
        if analysis["has_lists"]:
            return "marker"

        # If text is short with no structure, use sentence-based
        if analysis["length"] < 500 and not analysis["has_structure"]:
            return "sentence"

        # Default to hybrid for complex cases
        return "hybrid"
