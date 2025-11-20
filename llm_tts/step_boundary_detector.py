"""
Real-time step boundary detection during generation
"""

from typing import List


class StepBoundaryDetector:
    """Detects when a reasoning step is complete during generation"""

    def __init__(
        self,
        step_patterns: List[str],
        answer_patterns: List[str],
        max_tokens_per_step: int,
        eos_patterns: List[str] = None,
    ):
        """
        Args:
            step_patterns: Patterns that indicate step boundaries
                (e.g., ["\n- Step", "\nStep"])
            answer_patterns: Patterns that indicate final answer
                (e.g., ["<Answer>:", "\n\nAnswer:"])
            max_tokens_per_step: Maximum tokens allowed per step
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

    def is_step_complete(self, generated_text: str, token_count: int = None) -> bool:
        """Check if current generation represents a complete step"""

        # Immediate completion if we hit an answer pattern - triggers answer phase
        for pattern in self.answer_patterns:
            if pattern in generated_text and not generated_text.startswith(pattern):
                return True

        # Count occurrences of "- Step" pattern specifically
        # We need to see it twice: once at the beginning of current step,
        # once at the beginning of next step
        step_count = generated_text.count("- Step")

        # Stop when we see 2 or more "- Step" markers (current step + next step beginning)
        if step_count >= 2:
            return True

        # Check token limit
        if token_count and token_count >= self.max_tokens_per_step:
            return True

        return False

    def is_trajectory_complete(
        self, generated_text: str, reached_eos: bool = False
    ) -> bool:
        """Check if trajectory is complete (second step marker is answer tag)"""
        first_answer_pos = None
        active_answer_pattern = None
        for pattern in self.answer_patterns:
            pos = generated_text.find(pattern)
            if pos != -1:
                if first_answer_pos is None or pos < first_answer_pos:
                    first_answer_pos = pos
                    active_answer_pattern = pattern  # noqa: F841

        if first_answer_pos is not None:
            # If the answer marker occurs at the beginning of the current chunk it means
            # we have just started the answer step (typically "\n<Answer>:" with leading newline).
            # We treat this as a completed trajectory.
            return True

        if reached_eos:
            return True

        for pattern in self.eos_patterns:
            if pattern in generated_text:
                return True

        return False

    def contains_answer_pattern(self, generated_text: str) -> bool:
        """Check if text contains any answer pattern"""
        for pattern in self.answer_patterns:
            if pattern in generated_text:
                return True
        return False

    def extract_step_text(self, generated_text: str) -> str:
        """Extract the step text, removing boundary markers at the END only"""

        step_text = generated_text.strip()

        # Special handling for "- Step" pattern
        # If we have 2+ occurrences, remove everything from the second occurrence onwards
        step_count = step_text.count("- Step")
        if step_count >= 2:
            # Find the position of the second "- Step"
            first_pos = step_text.find("- Step")
            second_pos = step_text.find("- Step", first_pos + 1)
            if second_pos != -1:
                step_text = step_text[:second_pos].strip()

        # For answer patterns, remove from the first occurrence
        for pattern in self.answer_patterns:
            if pattern in step_text and not step_text.startswith(pattern):
                pos = step_text.find(pattern)
                if pos != -1:
                    step_text = step_text[:pos].strip()
                    break

        return step_text
