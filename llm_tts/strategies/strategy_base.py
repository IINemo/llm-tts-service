import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from llm_tts.utils.parallel import parallel_execute

log = logging.getLogger(__name__)


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
