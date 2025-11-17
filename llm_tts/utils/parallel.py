"""Parallel execution utilities for strategies and scorers."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

log = logging.getLogger(__name__)


def parallel_execute(
    worker_func: Callable[[Any], Any],
    task_args: List[Any],
    n_workers: int = 8,
    desc: str = "Executing tasks",
    model: Optional[Any] = None,
) -> List[Any]:
    """
    Execute tasks in parallel using ThreadPoolExecutor.

    This utility function handles threading infrastructure, error handling,
    and automatic client recreation on failures (timeouts).

    Args:
        worker_func: Function to execute for each task (must accept one argument)
        task_args: List of arguments to pass to worker_func
        n_workers: Number of parallel workers (default: 8)
        desc: Description for logging (default: "Executing tasks")
        model: Optional model instance for client recreation on failures

    Returns:
        List of results (None results from failed tasks are filtered out)

    Note:
        If ANY tasks fail (e.g., due to timeouts) and model is provided,
        the model's client will be recreated to clear stuck connections.
        This happens automatically for all strategies and scorers.

    Example:
        >>> def worker(args):
        >>>     prompt, index, total = args
        >>>     # Do work...
        >>>     return result
        >>> args_list = [(prompt, i, n) for i in range(n)]
        >>> results = parallel_execute(worker, args_list, n_workers=8, model=my_model)
    """
    log.info(f"{desc} with {n_workers} parallel workers")

    results = []
    num_failures = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks and create future-to-arg mapping
        future_to_arg = {executor.submit(worker_func, arg): arg for arg in task_args}

        # Collect results as they complete
        for future in as_completed(future_to_arg):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                log.error(f"Task failed with exception: {e}")
                results.append(None)
                num_failures += 1

    # Filter out None results (failed tasks)
    valid_results = [r for r in results if r is not None]

    log.info(f"Completed {len(valid_results)}/{len(task_args)} tasks successfully")

    # If ANY tasks failed and model provided, recreate client to clear stuck connections
    if num_failures > 0 and model is not None:
        from llm_tts.models.blackboxmodel_with_streaming import (
            BlackboxModelWithStreaming,
        )

        if isinstance(model, BlackboxModelWithStreaming):
            log.warning(
                f"{num_failures}/{len(task_args)} parallel tasks failed - recreating client"
            )
            model.recreate_client()

    return valid_results
