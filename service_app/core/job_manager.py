"""
Job Manager - Handles asynchronous ToT job execution and progress tracking.
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from threading import Thread
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """Represents an asynchronous ToT execution job."""

    def __init__(
        self,
        job_id: str,
        prompt: str,
        strategy_type: str,
        model_name: str,
        strategy_config: Dict[str, Any],
    ):
        self.job_id = job_id
        self.prompt = prompt
        self.strategy_type = strategy_type
        self.model_name = model_name
        self.strategy_config = strategy_config

        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Progress tracking
        self.current_step = 0
        self.total_steps = strategy_config.get("steps", 4)
        self.nodes_explored = 0
        self.api_calls = 0

        # Results
        self.reasoning_tree: Optional[Dict[str, Any]] = None
        self.trajectory: Optional[str] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

        # Intermediate state
        self.intermediate_nodes: list = []
        self.intermediate_edges: list = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API response."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "progress": {
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "nodes_explored": self.nodes_explored,
                "api_calls": self.api_calls,
            },
            "result": (
                {
                    "trajectory": self.trajectory,
                    "reasoning_tree": self.reasoning_tree,
                    "metadata": self.metadata,
                }
                if self.status == JobStatus.COMPLETED
                else None
            ),
            "error": self.error if self.status == JobStatus.FAILED else None,
            # Include intermediate state for real-time updates
            "intermediate_tree": (
                {
                    "nodes": self.intermediate_nodes,
                    "edges": self.intermediate_edges,
                    "question": self.prompt,
                }
                if self.intermediate_nodes
                else None
            ),
        }


class JobManager:
    """Manages asynchronous job execution and tracking."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._max_jobs = 100  # Prevent memory issues

    def create_job(
        self,
        prompt: str,
        strategy_type: str,
        model_name: str,
        strategy_config: Dict[str, Any],
    ) -> Job:
        """Create a new job."""
        job_id = str(uuid.uuid4())
        job = Job(job_id, prompt, strategy_type, model_name, strategy_config)

        # Clean up old jobs if needed
        if len(self._jobs) >= self._max_jobs:
            self._cleanup_old_jobs()

        self._jobs[job_id] = job
        log.info(f"Created job {job_id} for strategy {strategy_type}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def start_job(
        self,
        job_id: str,
        strategy_factory: Callable,
        progress_callback: Optional[Callable] = None,
    ):
        """Start job execution in background thread."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job {job_id} is not pending")

        # Start execution in background thread
        thread = Thread(
            target=self._execute_job,
            args=(job, strategy_factory, progress_callback),
            daemon=True,
        )
        thread.start()
        log.info(f"Started job {job_id} in background thread")

    def _execute_job(
        self,
        job: Job,
        strategy_factory: Callable,
        progress_callback: Optional[Callable] = None,
    ):
        """Execute job in background thread."""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

            log.info(f"[Job {job.job_id}] Executing strategy {job.strategy_type}")

            # Create strategy
            strategy = strategy_factory(
                strategy_type=job.strategy_type,
                model_name=job.model_name,
                strategy_config=job.strategy_config,
            )

            # Set progress callback if strategy supports it
            if hasattr(strategy, "set_progress_callback") and progress_callback:
                strategy.set_progress_callback(
                    lambda update: progress_callback(job, update)
                )

            # Convert prompt to messages format
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": job.prompt},
            ]

            # Execute strategy
            result = strategy.generate_trajectory(messages)

            # Calculate elapsed time
            elapsed = (datetime.now() - job.started_at).total_seconds()

            # Store results
            trajectory = result.get("trajectory", "")

            # Ensure trajectory is a string (handle case where it might be a list)
            if isinstance(trajectory, list):
                job.trajectory = trajectory[0] if trajectory else ""
                log.warning(
                    f"[Job {job.job_id}] Trajectory was a list, extracted first element"
                )
            else:
                job.trajectory = trajectory

            job.metadata = result.get("metadata", {})

            # Add job-level metadata
            if job.metadata:
                job.metadata["elapsed_time"] = round(elapsed, 2)

                # Extract commonly used fields to top level for easier access
                gen_details = job.metadata.get("generation_details", {})
                if gen_details:
                    job.metadata["total_api_calls"] = gen_details.get("total_api_calls")
                    job.metadata["scorer_evaluations"] = gen_details.get(
                        "scorer_evaluations"
                    )

                results = job.metadata.get("results", {})
                if results:
                    job.metadata["best_score"] = results.get("best_score")

            # Get reasoning tree from strategy result (if available)
            job.reasoning_tree = result.get("reasoning_tree")

            # Legacy fallback: Build reasoning tree from metadata (for old strategies)
            if not job.reasoning_tree and job.metadata:
                generation = job.metadata.get("generation_details", {})
                all_steps = generation.get("all_steps", [])

                reasoning_tree = {"nodes": [], "edges": [], "question": job.prompt}

                node_id = 0

                # Root node
                reasoning_tree["nodes"].append(
                    {
                        "id": node_id,
                        "step": 0,
                        "state": "",
                        "score": 0.0,
                        "is_root": True,
                        "is_selected": True,
                        "is_final": False,
                        "timestamp": 0,
                    }
                )
                node_id += 1

                # Track mapping from selected states to node IDs for edge building
                # Key: state content, Value: node ID
                state_to_node_id = {"": 0}  # Root state maps to node 0

                # Track selected states from previous step for parent lookup
                prev_selected_states = [""]  # Step 0 only has root

                # Process each step
                for step_data in all_steps:
                    step_idx = step_data["step_idx"]
                    candidates = step_data["candidates"]
                    scores = step_data["scores"]
                    selected_states = step_data["selected_states"]
                    parent_indices = step_data.get("parent_indices", [])

                    # Create nodes for all candidates in this step
                    step_node_ids = []
                    for i, (candidate, score) in enumerate(zip(candidates, scores)):
                        is_selected = candidate in selected_states

                        reasoning_tree["nodes"].append(
                            {
                                "id": node_id,
                                "step": step_idx + 1,
                                "state": candidate,
                                "score": float(score),
                                "is_root": False,
                                "is_selected": is_selected,
                                "is_final": step_idx == len(all_steps) - 1,
                                "timestamp": step_idx + 1,
                            }
                        )

                        # Store mapping for selected states
                        if is_selected:
                            state_to_node_id[candidate] = node_id

                        step_node_ids.append(node_id)

                        # Create edge from parent to this child
                        if i < len(parent_indices):
                            parent_idx = parent_indices[i]
                            if parent_idx < len(prev_selected_states):
                                parent_state = prev_selected_states[parent_idx]
                                parent_node_id = state_to_node_id.get(parent_state, 0)

                                reasoning_tree["edges"].append(
                                    {
                                        "from": parent_node_id,
                                        "to": node_id,
                                    }
                                )

                        node_id += 1

                    # Update for next iteration
                    prev_selected_states = selected_states

                job.reasoning_tree = reasoning_tree

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()

            elapsed = (job.completed_at - job.started_at).total_seconds()
            log.info(f"[Job {job.job_id}] Completed in {elapsed:.2f}s")

        except Exception as e:
            log.error(f"[Job {job.job_id}] Failed: {e}", exc_info=True)
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()

    def _cleanup_old_jobs(self):
        """Remove oldest completed/failed jobs."""
        # Sort by creation time
        sorted_jobs = sorted(
            self._jobs.values(), key=lambda j: j.created_at, reverse=True
        )

        # Keep only the most recent jobs
        keep_ids = {job.job_id for job in sorted_jobs[: self._max_jobs - 1]}

        # Remove old jobs
        job_ids = list(self._jobs.keys())
        for job_id in job_ids:
            if job_id not in keep_ids:
                del self._jobs[job_id]

        log.info(f"Cleaned up old jobs, {len(self._jobs)} remaining")


# Global job manager instance
job_manager = JobManager()
