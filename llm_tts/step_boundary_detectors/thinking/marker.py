"""
Marker-based step boundary detector for thinking mode.

Detects reasoning transitions using linguistic markers like
"first", "then", "so", "therefore", "let me", etc.
"""

import re
from typing import List, Optional, Set

from ..base import StepBoundaryDetectorBase


class ThinkingMarkerDetector(StepBoundaryDetectorBase):
    """
    Detects step boundaries in <think> content using linguistic markers.

    Looks for natural reasoning transition markers:
    - Sequence markers: "first", "second", "then", "next", "finally"
    - Conclusion markers: "so", "therefore", "thus", "hence", "the answer"
    - Thinking markers: "let me", "I need to", "wait", "hmm", "okay", "wait but"
    - Verification markers: "let's check", "to verify", "this means"
    - Reasoning markers: "alternatively", "for example", "consider", "suppose"
    - Sentence-start markers: "but", "however", "since" (only after .!?\\n)
    - Correction markers: "mistake", "error", "wrong"
    - Structure markers: paragraph breaks, bullet points, numbered lists

    Best for: Semantic understanding of reasoning flow.
    """

    # Default marker categories
    SEQUENCE_MARKERS = [
        r"\bfirst\b",
        r"\bsecond\b",
        r"\bthird\b",
        r"\bnext\b",
        r"\bthen\b",
        r"\bfinally\b",
        r"\blastly\b",
        r"\bafter that\b",
    ]

    CONCLUSION_MARKERS = [
        r"\bso\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bconsequently\b",
        r"\bas a result\b",
        r"\bthis means\b",
        r"\bwhich means\b",
        r"\bwhich gives\b",
        r"\bthis suggests\b",
        r"\bthe answer\b",
    ]

    THINKING_MARKERS = [
        r"\blet me\b",
        r"\blet's\b",
        r"\bi need to\b",
        r"\bi should\b",
        r"\bi can\b",
        r"\bi'll\b",
        r"\bwait\b",
        r"\bhmm\b",
        r"\bokay\b",
        r"\boh\b",
        r"\bactually\b",
        # Extended thinking patterns
        r"\blet me think\b",
        r"\blet me consider\b",
        r"\blet me compute\b",
        r"\blet me try\b",
        r"\blet me denote\b",
        r"\bwait but\b",
        r"\bwait no\b",
        r"\bwait maybe\b",
        r"\bwait perhaps\b",
        r"\bso maybe\b",
        r"\bso perhaps\b",
    ]

    VERIFICATION_MARKERS = [
        r"\bto verify\b",
        r"\bto check\b",
        r"\blet's check\b",
        r"\blet's verify\b",
        r"\bsubstituting\b",
        r"\bplugging in\b",
        r"\bif we\b",
        r"\bwhen we\b",
    ]

    # Markers safe to use anywhere (multi-word phrases unlikely mid-sentence)
    REASONING_MARKERS = [
        r"\balternatively\b",
        r"\bfor example\b",
        r"\bsimilarly\b",
        r"\bnote that\b",
        r"\brecall that\b",
        r"\bgiven that\b",
        r"\bconsider\b",
        r"\bassume\b",
        r"\bsuppose\b",
        r"\bwe have\b",
        r"\bwe can\b",
        r"\bwe need\b",
    ]

    # Markers that should only match at sentence start (after . ! ? or newline)
    # These are common words that would cause over-splitting if matched mid-sentence
    SENTENCE_START_MARKERS = [
        r"(?<=[.!?\n])\s*\bbut\b",
        r"(?<=[.!?\n])\s*\bhowever\b",
        r"(?<=[.!?\n])\s*\bsince\b",
        r"(?<=[.!?\n])\s*\bbecause\b",
        r"(?<=[.!?\n])\s*\bno\b",
        r"(?<=[.!?\n])\s*\byes\b",
        r"(?<=[.!?\n])\s*\bright\b",
        r"(?<=[.!?\n])\s*\bcorrect\b",
    ]

    # Self-correction markers (important for reasoning traces)
    CORRECTION_MARKERS = [
        r"\bmistake\b",
        r"\berror\b",
        r"\bwrong\b",
    ]

    STRUCTURE_MARKERS = [
        r"\n\n",  # Paragraph breaks
        r"\n-\s",  # Bullet points
        r"\n\d+\.\s",  # Numbered lists
        r"\n\*\s",  # Asterisk bullets
    ]

    def __init__(
        self,
        use_sequence: bool = True,
        use_conclusion: bool = True,
        use_thinking: bool = True,
        use_verification: bool = True,
        use_structure: bool = True,
        use_reasoning: bool = True,
        use_sentence_start: bool = True,
        use_correction: bool = True,
        custom_markers: Optional[List[str]] = None,
        min_step_tokens: int = 50,
        max_step_tokens: int = 300,
        case_sensitive: bool = False,
    ):
        """
        Args:
            use_sequence: Include sequence markers (first, then, next...)
            use_conclusion: Include conclusion markers (so, therefore...)
            use_thinking: Include thinking markers (let me, wait...)
            use_verification: Include verification markers (to check, verify...)
            use_structure: Include structure markers (paragraphs, bullets...)
            use_reasoning: Include reasoning markers (alternatively, for example, consider...)
            use_sentence_start: Include sentence-start markers (but, however, since... only after .!?\\n)
            use_correction: Include self-correction markers (mistake, error, wrong)
            custom_markers: Additional custom marker patterns
            min_step_tokens: Minimum tokens per step
            max_step_tokens: Maximum tokens per step
            case_sensitive: Whether marker matching is case sensitive
        """
        self.min_step_tokens = min_step_tokens
        self.max_step_tokens = max_step_tokens
        # Approximate char limits for text-based detection (~4 chars per token)
        self.min_step_chars = min_step_tokens * 4
        self.max_step_chars = max_step_tokens * 4
        self.case_sensitive = case_sensitive

        # Store flags for later use (e.g., deriving vLLM stop tokens)
        self.use_sequence = use_sequence
        self.use_conclusion = use_conclusion
        self.use_thinking = use_thinking
        self.use_verification = use_verification
        self.use_structure = use_structure
        self.use_reasoning = use_reasoning
        self.use_sentence_start = use_sentence_start
        self.use_correction = use_correction
        self.custom_markers = custom_markers

        # Build marker list
        self.markers = []
        if use_sequence:
            self.markers.extend(self.SEQUENCE_MARKERS)
        if use_conclusion:
            self.markers.extend(self.CONCLUSION_MARKERS)
        if use_thinking:
            self.markers.extend(self.THINKING_MARKERS)
        if use_verification:
            self.markers.extend(self.VERIFICATION_MARKERS)
        if use_structure:
            self.markers.extend(self.STRUCTURE_MARKERS)
        if use_reasoning:
            self.markers.extend(self.REASONING_MARKERS)
        if use_sentence_start:
            self.markers.extend(self.SENTENCE_START_MARKERS)
        if use_correction:
            self.markers.extend(self.CORRECTION_MARKERS)
        if custom_markers:
            self.markers.extend(custom_markers)

        # Compile combined pattern
        self._compile_pattern()

    def _compile_pattern(self):
        """Compile the combined regex pattern for all markers."""
        if not self.markers:
            self.pattern = None
            return

        # Join all markers with | for alternation
        combined = "|".join(f"({m})" for m in self.markers)
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self.pattern = re.compile(combined, flags)

    def detect_steps(self, text: str, **kwargs) -> List[str]:
        """
        Detect steps by finding linguistic marker boundaries.

        Args:
            text: Thinking content (inside <think> tags)

        Returns:
            List of step strings
        """
        text = self._extract_thinking_content(text)

        if not text.strip():
            return []

        if not self.pattern:
            return [text]

        # Find all marker positions
        marker_positions = self._find_marker_positions(text)

        # Split at marker positions
        parts = self._split_at_positions(text, marker_positions)

        # Normalize step sizes
        steps = self._normalize_steps(parts)

        return steps

    def _extract_thinking_content(self, text: str) -> str:
        """Extract content from <think> tags if present."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return text.strip()

    def _find_marker_positions(self, text: str) -> List[int]:
        """Find all positions where markers occur."""
        positions: Set[int] = set()

        for match in self.pattern.finditer(text):
            pos = match.start()
            # For markers at line start, include the newline
            if pos > 0 and text[pos - 1] == "\n":
                pos -= 1
            positions.add(pos)

        return sorted(positions)

    def _split_at_positions(self, text: str, positions: List[int]) -> List[str]:
        """Split text at the given positions, only at sentence boundaries."""
        if not positions:
            return [text] if text.strip() else []

        parts = []
        prev_pos = 0
        sentence_endings = ".!?\n"

        for pos in positions:
            if pos > prev_pos:
                part = text[prev_pos:pos].strip()
                if part:
                    # Only split if part ends with sentence punctuation
                    # This avoids mid-sentence splits
                    if part[-1] in sentence_endings:
                        parts.append(part)
                        prev_pos = pos
                    # Otherwise, don't split here - continue accumulating

        # Add remaining text
        if prev_pos < len(text):
            part = text[prev_pos:].strip()
            if part:
                parts.append(part)

        return parts

    def _normalize_steps(self, parts: List[str]) -> List[str]:
        """Merge short parts and split long parts."""
        if not parts:
            return []

        steps = []
        current_step = ""

        for part in parts:
            combined_len = len(current_step) + len(part) + 1  # +1 for newline

            if combined_len < self.min_step_chars:
                # Merge with current
                current_step = (current_step + "\n" + part).strip()
            elif len(current_step) >= self.min_step_chars:
                # Save current and start new
                steps.append(current_step)
                current_step = part
            else:
                # Current is too short but combined would be OK
                current_step = (current_step + "\n" + part).strip()

            # If current step exceeds max, save it
            if len(current_step) >= self.max_step_chars:
                steps.append(current_step)
                current_step = ""

        # Save final step
        if current_step.strip():
            steps.append(current_step)

        return steps

    # =========================================================================
    # Methods for online generation (StoppingCriteria compatibility)
    # =========================================================================

    def __init_online_state__(self):
        """Initialize state for online generation tracking."""
        if not hasattr(self, "_last_step_count"):
            self._last_step_count = 0
            self._trajectory_complete = False
            self._last_marker_pos = 0
            self._last_step_end_pos = 0
            self._step_boundary_pos = None  # Position where current step ends

    def reset_online_state(self):
        """Reset state for a new generation session."""
        self._last_step_count = 0
        self._trajectory_complete = False
        self._last_marker_pos = 0
        self._last_step_end_pos = 0
        self._step_boundary_pos = None

    def mark_step_complete(self, generated_text: str):
        """
        Mark current step as complete and update state.

        Call this after is_step_complete() returns True to advance
        the internal position tracker for the next step.

        Args:
            generated_text: The full generated text at step completion
        """
        self.__init_online_state__()
        text = self._extract_thinking_content(generated_text)

        # Advance to boundary position if set, otherwise to end of text
        if self._step_boundary_pos is not None:
            self._last_step_end_pos = self._step_boundary_pos
        else:
            self._last_step_end_pos = len(text)

        self._last_step_count += 1
        self._step_boundary_pos = None  # Reset for next step

    def is_step_complete(self, generated_text: str, token_count: int = None) -> bool:
        """
        Check if current generation represents a complete step.

        Strategy: Match offline detect_steps() behavior.
        - Find markers in new content since last step
        - Complete step when we have a marker AND enough content (min_step_chars)
        - Or when content exceeds max_step_chars (force split)

        Args:
            generated_text: Text generated so far (cumulative)
            token_count: Number of tokens generated (optional)

        Returns:
            True if step boundary detected
        """
        self.__init_online_state__()

        # Check for trajectory completion first
        if self.is_trajectory_complete(generated_text):
            self._step_boundary_pos = None  # No truncation, use full text
            return True

        text = self._extract_thinking_content(generated_text)

        # Content since last step completed
        new_content = text[self._last_step_end_pos :]
        new_content_len = len(new_content.strip())

        # Not enough content yet
        if new_content_len < self.min_step_chars:
            return False

        # Find marker positions in new content
        marker_positions = self._find_marker_positions(new_content)

        # If we have enough content AND found a marker after min_step_chars,
        # that marker indicates start of next step -> current step is complete
        if marker_positions:
            sentence_endings = ".!?\n"
            for pos in marker_positions:
                content_before_marker = new_content[:pos].strip()
                # Must have >= min_step_chars before marker
                if len(content_before_marker) >= self.min_step_chars:
                    # Only split at sentence boundaries to avoid mid-sentence cuts
                    if (
                        content_before_marker
                        and content_before_marker[-1] in sentence_endings
                    ):
                        self._step_boundary_pos = self._last_step_end_pos + pos
                        return True

        # If over max and no valid split found, keep accumulating until marker found
        # This matches offline behavior where max is a soft limit checked after merging
        # Don't force split - wait for next marker at sentence boundary

        # Hard limit: force split if way over max (4x) to prevent infinite accumulation
        # Using 4x allows offline-like behavior where steps can grow significantly
        # before finding a valid marker/sentence boundary
        if new_content_len >= self.max_step_chars * 4:
            self._step_boundary_pos = None
            return True

        return False

    def is_trajectory_complete(
        self, generated_text: str, reached_eos: bool = False
    ) -> bool:
        """
        Check if trajectory is complete (answer found or end of thinking).

        Args:
            generated_text: Full generated text
            reached_eos: Whether EOS token was reached

        Returns:
            True if trajectory is complete
        """
        self.__init_online_state__()

        # Check for </think> tag (end of thinking)
        if "</think>" in generated_text:
            self._trajectory_complete = True
            return True

        # Check for answer patterns
        for pattern in self.answer_patterns:
            if pattern in generated_text:
                self._trajectory_complete = True
                return True

        # Check EOS
        if reached_eos:
            self._trajectory_complete = True
            return True

        return False

    def contains_answer_pattern(self, generated_text: str) -> bool:
        """Check if text contains any answer pattern."""
        for pattern in self.answer_patterns:
            if pattern in generated_text:
                return True
        return False

    def extract_step_text(self, generated_text: str) -> str:
        """
        Extract the step text from generated content.

        For online generation, returns text from last step end to boundary position.
        This ensures steps don't include content past the marker.

        Call this AFTER is_step_complete() returns True but BEFORE mark_step_complete().
        """
        self.__init_online_state__()
        text = self._extract_thinking_content(generated_text)

        # Extract text from last step end to boundary (or end of text)
        start = self._last_step_end_pos
        end = (
            self._step_boundary_pos
            if self._step_boundary_pos is not None
            else len(text)
        )

        return text[start:end].strip()

    # Default answer patterns for trajectory completion
    @property
    def answer_patterns(self) -> List[str]:
        """Answer patterns that indicate trajectory completion."""
        if not hasattr(self, "_answer_patterns"):
            self._answer_patterns = [
                "</think>",
                "<Answer>:",
                "\n<Answer>:",
            ]
        return self._answer_patterns

    @answer_patterns.setter
    def answer_patterns(self, patterns: List[str]):
        """Set custom answer patterns."""
        self._answer_patterns = patterns

    # Step patterns property for vLLM stop tokens
    @property
    def step_patterns(self) -> List[str]:
        """Step patterns for vLLM stop tokens (not used in marker detection)."""
        # For thinking mode, we don't use explicit step patterns like "- Step"
        # Instead, return answer patterns as stop tokens
        return self.answer_patterns

    def get_marker_stats(self, text: str) -> dict:
        """
        Get statistics about markers found in text.

        Useful for debugging and understanding thinking patterns.
        """
        text = self._extract_thinking_content(text)

        stats = {
            "total_markers": 0,
            "marker_counts": {},
            "marker_positions": [],
        }

        if not self.pattern:
            return stats

        for match in self.pattern.finditer(text):
            marker_text = match.group().lower()
            stats["total_markers"] += 1
            stats["marker_counts"][marker_text] = (
                stats["marker_counts"].get(marker_text, 0) + 1
            )
            stats["marker_positions"].append(
                {"marker": marker_text, "position": match.start()}
            )

        return stats

    def get_vllm_stop_tokens(self, include_answer_tokens: bool = False) -> List[str]:
        """
        Get vLLM stop tokens derived from this detector's configuration.

        Uses sentence-start matching by default to avoid mid-sentence breaks.
        Only matches words at the START of a new sentence (after . ? ! or newline).

        Args:
            include_answer_tokens: If True, include answer patterns like </think>.
                                  Default False - generator adds </think> separately.

        Returns:
            List of stop token strings for vLLM SamplingParams.stop
        """
        from llm_tts.step_boundary_detectors.thinking.vllm import (
            get_stop_tokens_sentence_start,
        )

        return get_stop_tokens_sentence_start(
            use_sequence=self.use_sequence,
            use_conclusion=self.use_conclusion,
            use_thinking=self.use_thinking,
            use_verification=self.use_verification,
            use_reasoning=self.use_reasoning,
            use_correction=self.use_correction,
            use_structure=self.use_structure,
            custom_words=self.custom_markers,
        )
