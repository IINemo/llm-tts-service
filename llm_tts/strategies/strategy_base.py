import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from llm_tts.utils.parallel import parallel_execute

if TYPE_CHECKING:
    from llm_tts.generators import StepCandidate

log = logging.getLogger(__name__)


def count_reasoning_steps(steps: list, thinking_mode: bool) -> int:
    """Count reasoning steps in a trajectory.

    Args:
        steps: List of step candidates/dicts from trajectory
        thinking_mode: Whether the model uses thinking mode

    Returns:
        Number of reasoning steps:
        - Non-thinking mode: total number of generated steps (len(steps))
        - Thinking mode: number of thinking steps only (len(steps) - 1, excluding the answer step)
    """
    if not steps:
        return 0
    if thinking_mode:
        return max(len(steps) - 1, 0)  # exclude answer step
    return len(steps)


class StrategyBase(ABC):
    """Abstract base class for TTS strategies with batch generation support.

    Strategies must implement generate_trajectories_batch() for efficient
    batch processing. Single-sample calls via generate_trajectory() are
    automatically wrapped to use the batch method.
    """

    @abstractmethod
    def generate_trajectories_batch(
        self, requests: List[List[Dict[str, str]]], sample_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """Generate trajectories for multiple samples in batch.

        Args:
            requests: List of input chats (one per sample)
            sample_indices: List of sample indices (for logging/tracking)

        Returns:
            List of result dictionaries (one per sample)
        """
        pass

    def generate_trajectory(
        self, input_chat: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, Any]:
        """Generate trajectory for a single sample.

        Default implementation wraps generate_trajectories_batch for convenience.
        Subclasses can override if they need specialized single-sample behavior.

        Args:
            input_chat: Input chat messages
            sample_idx: Sample index for logging (default: 0)

        Returns:
            Result dictionary for the single sample
        """
        results = self.generate_trajectories_batch([input_chat], [sample_idx])
        if not results:
            raise ValueError("generate_trajectories_batch returned empty results")
        return results[0]

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
