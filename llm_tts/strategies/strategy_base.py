import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from llm_tts.utils.parallel import parallel_execute

if TYPE_CHECKING:
    from llm_tts.generators import StepCandidate

log = logging.getLogger(__name__)


def count_response_steps(text: str) -> int:
    """Count structured steps in response text (e.g., '- Step 1:', '- Step 2:')."""
    pattern = r"- Step \d+:"
    matches = re.findall(pattern, text)
    return len(matches)


def count_thinking_and_response_steps(steps: list) -> Tuple[int, int]:
    """
    Count thinking steps and response steps separately.

    Args:
        steps: List of step candidates/dicts from trajectory

    Returns:
        (thinking_num_steps, response_num_steps)
    """
    thinking_steps = 0
    response_steps = 0

    for step in steps:
        # Get text from step (could be StepCandidate or dict)
        if hasattr(step, "text"):
            text = step.text
        elif isinstance(step, dict):
            text = step.get("text", "")
        else:
            text = str(step)

        # Check if this is a response step (contains <start of response> or <end of response>)
        if "<start of response>" in text or "<end of response>" in text:
            # Count structured steps within response
            response_steps += count_response_steps(text)
            if response_steps == 0:
                response_steps = 1  # At least 1 response step
        else:
            # It's a thinking step
            thinking_steps += 1

    return thinking_steps, response_steps


class StrategyBase(ABC):
    """Abstract base class for TTS strategies with parallel generation support"""

    @abstractmethod
    def generate_trajectory(self, input_chat: List[Dict[str, str]]) -> Dict[str, any]:
        pass

    def _parallel_generate(
        self,
        worker_func: Callable[[Any], Any],
        task_args: List[Any],
        n_threads: int = 8,
        desc: str = "Generating",
        model: Any = None,
    ) -> List[Any]:
        """
        Execute tasks in parallel using shared parallel execution utility.

        This is a convenience wrapper around llm_tts.utils.parallel.parallel_execute
        that maintains backward compatibility with existing strategy code.

        Args:
            worker_func: Function to execute for each task (must accept one argument)
            task_args: List of arguments to pass to worker_func
            n_threads: Number of parallel threads (default: 8)
            desc: Description for logging (default: "Generating")
            model: Optional model instance for automatic client recreation on failures

        Returns:
            List of results (None results are filtered out)

        Example:
            >>> def worker(args):
            >>>     prompt, index, total = args
            >>>     # Do work...
            >>>     return result
            >>> args_list = [(prompt, i, n) for i in range(n)]
            >>> results = self._parallel_generate(worker, args_list, n_threads=8, model=self.model)
        """
        return parallel_execute(
            worker_func=worker_func,
            task_args=task_args,
            n_workers=n_threads,
            desc=desc,
            model=model,
        )

    def _has_answer_content(
        self, candidate: "StepCandidate", min_answer_chars: int = 1
    ) -> bool:
        """
        Check if candidate has actual answer content after the answer pattern.

        When using HuggingFace/vLLM with stopping criteria, generation may stop
        right at "<Answer>:" without generating the actual answer content.
        This checks if there's meaningful content after the answer pattern.

        Uses answer_patterns from step_generator.detector if available,
        otherwise falls back to DEFAULT_ANSWER_PATTERNS.

        Args:
            candidate: The step candidate to check
            min_answer_chars: Minimum characters expected after answer pattern

        Returns:
            True if answer content is present, False otherwise
        """
        # Get answer_patterns from detector
        if not hasattr(self, "step_generator") or not hasattr(
            self.step_generator, "detector"
        ):
            return False

        detector = self.step_generator.detector
        if not hasattr(detector, "answer_patterns"):
            return False

        text = candidate.raw_text if candidate.raw_text else candidate.text

        for pattern in detector.answer_patterns:
            pos = text.find(pattern)
            if pos != -1:
                content_after = text[pos + len(pattern) :].strip()
                if len(content_after) >= min_answer_chars:
                    return True
                log.debug(
                    f"Answer pattern found but content too short: "
                    f"'{content_after}' ({len(content_after)} chars)"
                )
                return False

        # No answer pattern found
        return False
