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
            "\n**Step",
            "\n## Step",
            "<Answer>:",
            "\n<Answer>:",
            "\n\nAnswer:",
            "\nFinal Answer:",
            "\n\nThe answer is",
        ]
        
        self.answer_patterns = answer_patterns or [
            "<answer>:",
            "\n<answer>:",
            "the final answer is: ",
        ]
        self.max_tokens_per_step = max_tokens_per_step

    def is_step_complete(self, generated_text: str, token_count: int = None) -> bool:
        """Check if current generation represents a complete step"""

        # Immediate completion if we hit an answer pattern - triggers answer phase
        for pattern in self.answer_patterns:
            if pattern in generated_text and not generated_text.startswith(pattern):
                return True

        # Do not prematurely stop on "- Step" occurrences; allow planning to continue

        # Check token limit
        if token_count and token_count >= self.max_tokens_per_step:
            return True


        return False
        
    def is_trajectory_complete(self, generated_text: str, reached_eos: bool = False) -> bool:
        """Check if trajectory is complete (contains explicit answer pattern)"""
        return self.contains_answer_pattern(generated_text)
        
    def contains_answer_pattern(self, generated_text: str) -> bool:
        """Check if text contains any answer pattern"""
        text = generated_text.lower()
        for pattern in self.answer_patterns:
            if pattern in text:
                return True
        return False

    def extract_step_text(self, generated_text: str) -> str:
        step_text = generated_text.strip()
        # For answer patterns, remove from the first occurrence
        for pattern in self.answer_patterns:
            if pattern in step_text and not step_text.startswith(pattern):
                pos = step_text.find(pattern)
                if pos != -1:
                    step_text = step_text[:pos].strip()
                    break
        return step_text
