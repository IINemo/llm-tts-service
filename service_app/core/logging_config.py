"""
Logging configuration for the service.

Each run gets its own log directory: logs/<date>/<time>/service.log
Console output stays human-readable; file output uses JSON for easy parsing.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from pythonjsonlogger import jsonlogger

from .config import settings

CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_run_dir: Path | None = None


def setup_logging() -> Path:
    """
    Configure root logger with console + per-run JSON file handler.

    Uses an environment variable to share the log directory path across
    the reloader and worker processes spawned by uvicorn, so both write
    to the same ``logs/<date>/<time>/service.log``.

    Safe to call multiple times within the same process — only the first
    call attaches handlers.

    Returns:
        Path to the run's log directory (e.g. logs/2026-02-19/15-30-42/)
    """
    global _run_dir
    if _run_dir is not None:
        return _run_dir

    # First process sets the env var; child processes reuse it.
    env_key = "_SERVICE_LOG_DIR"
    run_dir_str = os.environ.get(env_key)
    if run_dir_str:
        run_dir = Path(run_dir_str)
    else:
        now = datetime.now()
        run_dir = (
            Path(settings.log_dir) / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )
        os.environ[env_key] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    # Root must be at DEBUG so the file handler can capture everything;
    # the console handler filters to the configured level.
    root.setLevel(logging.DEBUG)

    # Console handler — human-readable
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)

    # File handler — JSON, always DEBUG for full detail
    log_file = run_dir / "service.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )
    root.addHandler(file_handler)

    # Suppress watchfiles debug noise (uvicorn's file watcher); the log file
    # changes trigger watch events, which get logged, which change the file…
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    _run_dir = run_dir
    return run_dir
