"""
Sentence/paragraph-based step boundary detector for thinking mode.

Splits thinking content by sentences or paragraphs.
"""

import re
from typing import List

from ..base import StepBoundaryDetectorBase


class ThinkingSentenceDetector(StepBoundaryDetectorBase):
    """
    Detects step boundaries in <think> content using sentence/paragraph splitting.

    Simple and fast approach that splits text by:
    - Paragraph breaks (double newlines)
    - Sentence endings (periods, question marks, exclamation marks)

    Best for: Quick processing without semantic understanding.
    """

    def __init__(
        self,
        split_mode: str = "paragraph",
        min_step_chars: int = 50,
        max_step_chars: int = 1000,
        merge_short: bool = True,
    ):
        """
        Args:
            split_mode: How to split text - "paragraph", "sentence", or "both"
            min_step_chars: Minimum characters per step (merge if smaller)
            max_step_chars: Maximum characters per step (split if larger)
            merge_short: Whether to merge steps shorter than min_step_chars
        """
        self.split_mode = split_mode
        self.min_step_chars = min_step_chars
        self.max_step_chars = max_step_chars
        self.merge_short = merge_short

    def detect_steps(self, text: str, **kwargs) -> List[str]:
        """
        Detect steps by splitting on sentence/paragraph boundaries.

        Args:
            text: Thinking content (inside <think> tags)

        Returns:
            List of step strings
        """
        text = self._extract_thinking_content(text)

        if not text.strip():
            return []

        # Initial split based on mode
        if self.split_mode == "paragraph":
            parts = self._split_paragraphs(text)
        elif self.split_mode == "sentence":
            parts = self._split_sentences(text)
        else:  # "both" - paragraphs first, then sentences for long paragraphs
            parts = self._split_both(text)

        # Normalize step sizes
        steps = self._normalize_steps(parts)

        return steps

    def _extract_thinking_content(self, text: str) -> str:
        """Extract content from <think> tags if present."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return text.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph breaks (double newlines)."""
        # Split on 2+ newlines
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text by sentence boundaries."""
        # Split on sentence-ending punctuation followed by space or newline
        # Handles common abbreviations and decimal numbers
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n"
        parts = re.split(sentence_pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def _split_both(self, text: str) -> List[str]:
        """Split by paragraphs first, then by sentences if paragraphs are too long."""
        paragraphs = self._split_paragraphs(text)
        parts = []

        for para in paragraphs:
            if len(para) > self.max_step_chars:
                # Split long paragraphs into sentences
                sentences = self._split_sentences(para)
                parts.extend(sentences)
            else:
                parts.append(para)

        return parts

    def _normalize_steps(self, parts: List[str]) -> List[str]:
        """Merge short parts and split long parts to normalize step sizes."""
        if not parts:
            return []

        steps = []
        current_step = ""

        for part in parts:
            # If current accumulator + new part is still reasonable, merge
            if self.merge_short and len(current_step) + len(part) < self.min_step_chars:
                current_step = (current_step + "\n" + part).strip()
            elif current_step:
                # Save current and start new
                if len(current_step) >= self.min_step_chars or not self.merge_short:
                    steps.append(current_step)
                    current_step = part
                else:
                    current_step = (current_step + "\n" + part).strip()
            else:
                current_step = part

            # If current step is too long, save it and reset
            if len(current_step) >= self.max_step_chars:
                steps.append(current_step)
                current_step = ""

        # Don't forget the last accumulated step
        if current_step.strip():
            steps.append(current_step)

        return steps
