"""Container-local logging setup (orchestrator-independent)."""

import logging
import os

LOG_LEVEL_ENV = "MVEXP_LOG_LEVEL"
"""Set by the orchestrator (and forwardable by hand) to control verbosity.

Kept in sync with ``multiverse.logging_utils.LOG_LEVEL_ENV`` but duplicated
here so the worker SDK stays dependency-free of the orchestrator package.
"""


def resolve_log_level(default: int = logging.INFO) -> int:
    """Map ``MVEXP_LOG_LEVEL`` to a numeric logging level.

    Accepts a level name (e.g. ``DEBUG``) or an integer string; returns
    ``default`` when the variable is unset.
    """
    raw = os.environ.get(LOG_LEVEL_ENV)
    if not raw:
        return default
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    resolved = logging.getLevelName(raw.upper())
    return resolved if isinstance(resolved, int) else default


def setup_logging(log_dir: str, log_level=None):
    """Configure the root logger to append to ``<log_dir>/run.log``.

    Replaces existing root handlers so repeated setup in one process does
    not duplicate log lines.
    """
    if log_level is None:
        log_level = resolve_log_level()
    log_file = os.path.join(log_dir, "run.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a standard library logger for the given module name."""
    return logging.getLogger(name)
