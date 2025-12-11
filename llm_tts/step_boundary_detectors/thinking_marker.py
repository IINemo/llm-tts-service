"""
Marker-based step boundary detector for thinking mode.

Detects reasoning transitions using linguistic markers like
"first", "then", "so", "therefore", "let me", etc.
"""

import re
from typing import List, Optional, Set

from .base import StepBoundaryDetectorBase


class ThinkingMarkerDetector(StepBoundaryDetectorBase):
    """
    Detects step boundaries in <think> content using linguistic markers.

    Looks for natural reasoning transition markers:
    - Sequence markers: "first", "second", "then", "next", "finally"
    - Conclusion markers: "so", "therefore", "thus", "hence"
    - Thinking markers: "let me", "I need to", "wait", "hmm", "okay"
    - Verification markers: "let's check", "to verify", "this means"

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
        custom_markers: Optional[List[str]] = None,
        min_step_chars: int = 50,
        max_step_chars: int = 800,
        case_sensitive: bool = False,
    ):
        """
        Args:
            use_sequence: Include sequence markers (first, then, next...)
            use_conclusion: Include conclusion markers (so, therefore...)
            use_thinking: Include thinking markers (let me, wait...)
            use_verification: Include verification markers (to check, verify...)
            use_structure: Include structure markers (paragraphs, bullets...)
            custom_markers: Additional custom marker patterns
            min_step_chars: Minimum characters per step
            max_step_chars: Maximum characters per step
            case_sensitive: Whether marker matching is case sensitive
        """
        self.min_step_chars = min_step_chars
        self.max_step_chars = max_step_chars
        self.case_sensitive = case_sensitive

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
        """Split text at the given positions."""
        if not positions:
            return [text] if text.strip() else []

        parts = []
        prev_pos = 0

        for pos in positions:
            if pos > prev_pos:
                part = text[prev_pos:pos].strip()
                if part:
                    parts.append(part)
            prev_pos = pos

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
