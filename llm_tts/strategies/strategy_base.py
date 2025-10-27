import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

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
    ) -> List[Any]:
        """
        Execute tasks in parallel using ThreadPoolExecutor with futures.

        This is a template method that handles the threading infrastructure,
        while allowing subclasses to define their own worker functions.

        Args:
            worker_func: Function to execute for each task (must accept one argument)
            task_args: List of arguments to pass to worker_func
            n_threads: Number of parallel threads (default: 8)
            desc: Description for logging (default: "Generating")

        Returns:
            List of results (None results are filtered out)

        Example:
            >>> def worker(args):
            >>>     prompt, index, total = args
            >>>     # Do work...
            >>>     return result
            >>> args_list = [(prompt, i, n) for i in range(n)]
            >>> results = self._parallel_generate(worker, args_list, n_threads=8)
        """
        log.info(f"{desc} with {n_threads} parallel threads")

        results = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit all tasks and create future-to-arg mapping
            future_to_arg = {
                executor.submit(worker_func, arg): arg for arg in task_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_arg):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log.error(f"Task failed with exception: {e}")
                    results.append(None)

        # Filter out None results (failed tasks)
        valid_results = [r for r in results if r is not None]

        log.info(f"Completed {len(valid_results)}/{len(task_args)} tasks successfully")

        return valid_results
