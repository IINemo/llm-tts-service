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
        self.max_tokens_per_step = max_tokens_per_step

    def is_step_complete(self, generated_text: str, token_count: int = None) -> bool:
        """Check if current generation represents a complete step"""

        # Immediate completion if we hit an answer pattern - triggers answer phase
        for pattern in self.answer_patterns:
            if pattern in generated_text and not generated_text.startswith(pattern):
                return True

        # Count occurrences of "- Step" pattern specifically
        # We need to see it twice:
        # once at the beginning of current step, once at the beginning of next step
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
        # Find all step marker positions
        marker_positions = []
        for pattern in self.step_patterns:
            pos = 0
            while True:
                pos = generated_text.find(pattern, pos)
                if pos == -1:
                    break
                marker_positions.append((pos, pattern))
                pos += len(pattern)

        # Sort by position
        marker_positions.sort()

        # If we have 2+ markers, check if the second one is an answer pattern
        if len(marker_positions) >= 2:
            second_marker_pattern = marker_positions[1][1]
            # Check if second marker is an answer pattern
            if second_marker_pattern in self.answer_patterns:
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


class UncertStepBoundaryDetector:
    """Detector tuned for uncert-cot: no '- Step' stopping; modern answer matches."""

    def __init__(
        self,
        step_patterns: List[str] = None,
        answer_patterns: List[str] = None,
        max_tokens_per_step: int = 250,
    ):
        self.step_patterns = step_patterns or [
            "\n**Step",
            "\n## Step",
        ]

        self.answer_patterns = answer_patterns or [
            "<answer>:",
            "\n<answer>:",
            "final answer",
            "answer:",
            "answer is",
        ]
        self.max_tokens_per_step = max_tokens_per_step

    def is_step_complete(self, generated_text: str, token_count: int = None) -> bool:
        for pattern in self.answer_patterns:
            if pattern in generated_text.lower():
                return True
        if token_count and token_count >= self.max_tokens_per_step:
            return True
        return False

    def is_trajectory_complete(
        self, generated_text: str, reached_eos: bool = False
    ) -> bool:
        return self.contains_answer_pattern(generated_text)

    def contains_answer_pattern(self, generated_text: str) -> bool:
        text = generated_text.lower()
        for pattern in self.answer_patterns:
            if pattern in text:
                return True
        return False

    def extract_step_text(self, generated_text: str) -> str:
        step_text = generated_text.strip()
        for pattern in self.answer_patterns:
            pos = step_text.lower().find(pattern)
            if pos != -1:
                step_text = step_text[:pos].strip()
                break
        return step_text


def uncert_detector(
    step_patterns: List[str] = None,
    answer_patterns: List[str] = None,
    max_tokens_per_step: int = 250,
):
    """New factory for uncert-cot behavior."""
    return UncertStepBoundaryDetector(
        step_patterns=step_patterns,
        answer_patterns=answer_patterns,
        max_tokens_per_step=max_tokens_per_step,
    )
