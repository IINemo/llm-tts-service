"""
Step boundary detector for structured responses (non-thinking mode).

This module handles responses with explicit step markers like "- Step 1:", "- Step 2:".
For native thinking mode (<think> tags), see thinking_*.py modules.
"""

from typing import List, Optional

from ..base import StepBoundaryDetectorBase


class StructuredStepDetector(StepBoundaryDetectorBase):
    """
    Detects step boundaries in structured responses (non-thinking mode).

    This detector is designed for responses that follow explicit step formatting
    like "- Step 1:", "- Step 2:", etc.
    """

    def __init__(
        self,
        step_patterns: Optional[List[str]] = None,
        answer_patterns: Optional[List[str]] = None,
        max_tokens_per_step: int = 512,
        eos_patterns: Optional[List[str]] = None,
    ):
        """
        Args:
            step_patterns: Patterns that indicate step boundaries
                (e.g., ["\n- Step", "\nStep"])
            answer_patterns: Patterns that indicate final answer
                (e.g., ["<Answer>:", "\n\nAnswer:"])
            max_tokens_per_step: Maximum tokens allowed per step
            eos_patterns: Patterns indicating end of sequence
        """
        self.step_patterns = step_patterns or [
            "\n- Step",
            "- Step",
            "\nStep",
            "\n\n",
            "\n**Step",
            "\n## Step",
            "<Answer>:",
            "\n<Answer>:",
            "\n\nAnswer:",
            "\nFinal Answer:",
            "\n\nThe answer is",
        ]
        self.answer_patterns = answer_patterns or [
            "<Answer>:",
            "\n<Answer>:",
            "\n\nAnswer:",
            "\nFinal Answer:",
            "\n\nThe answer is",
        ]
        self.eos_patterns = eos_patterns or [
            "<end of response>",
            "<|im_end|>",
        ]
        self.max_tokens_per_step = max_tokens_per_step

    def detect_steps(self, text: str, **kwargs) -> List[str]:
        """
        Detect steps in structured response text.

        Args:
            text: Response text with explicit step markers

        Returns:
            List of step strings
        """
        steps = []

        # Find all "- Step" occurrences
        step_marker = "- Step"
        positions = []
        pos = text.find(step_marker)
        while pos != -1:
            positions.append(pos)
            pos = text.find(step_marker, pos + 1)

        # Extract steps between markers
        for i, start_pos in enumerate(positions):
            if i + 1 < len(positions):
                end_pos = positions[i + 1]
            else:
                # Last step - find answer pattern or end
                end_pos = len(text)
                for pattern in self.answer_patterns:
                    ans_pos = text.find(pattern, start_pos)
                    if ans_pos != -1 and ans_pos < end_pos:
                        end_pos = ans_pos

            step_text = text[start_pos:end_pos].strip()
            if step_text:
                steps.append(step_text)

        return steps

    def is_step_complete(self, generated_text: str, token_count: int = None) -> bool:
        """Check if current generation represents a complete step."""

        # Immediate completion if we hit an answer pattern
        for pattern in self.answer_patterns:
            if pattern in generated_text and not generated_text.startswith(pattern):
                return True

        # Count occurrences of "- Step" pattern
        step_count = generated_text.count("- Step")

        # Stop when we see 2 or more "- Step" markers
        if step_count >= 2:
            return True

        # Check token limit
        if token_count and token_count >= self.max_tokens_per_step:
            return True

        return False

    def is_trajectory_complete(
        self, generated_text: str, reached_eos: bool = False
    ) -> bool:
        """Check if trajectory is complete.

        Trajectory is complete when we see <end of response> or other eos_patterns,
        NOT just <Answer>:. The answer marker indicates the answer is coming,
        but <end of response> marks the actual end.
        """
        if reached_eos:
            return True

        for pattern in self.eos_patterns:
            if pattern in generated_text:
                return True

        return False

    def contains_answer_pattern(self, generated_text: str) -> bool:
        """Check if text contains any answer pattern."""
        for pattern in self.answer_patterns:
            if pattern in generated_text:
                return True
        return False

    def extract_step_text(self, generated_text: str) -> str:
        """Extract the step text, removing boundary markers at the END only."""
        step_text = generated_text.strip()

        # Handle multiple "- Step" occurrences
        step_count = step_text.count("- Step")
        if step_count >= 2:
            first_pos = step_text.find("- Step")
            second_pos = step_text.find("- Step", first_pos + 1)
            if second_pos != -1:
                step_text = step_text[:second_pos].strip()

        # Remove answer patterns from end
        for pattern in self.answer_patterns:
            if pattern in step_text and not step_text.startswith(pattern):
                pos = step_text.find(pattern)
                if pos != -1:
                    step_text = step_text[:pos].strip()
                    break

        return step_text
