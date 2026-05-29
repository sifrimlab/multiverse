import logging
import os

LOG_LEVEL_ENV = "MVEXP_LOG_LEVEL"
"""Set by the orchestrator (and forwardable by hand) to control verbosity.

Kept in sync with ``multiverse.logging_utils.LOG_LEVEL_ENV`` but duplicated
here so the worker SDK stays dependency-free of the orchestrator package.
"""


def resolve_log_level(default: int = logging.INFO) -> int:
    raw = os.environ.get(LOG_LEVEL_ENV)
    if not raw:
        return default
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    resolved = logging.getLevelName(raw.upper())
    return resolved if isinstance(resolved, int) else default


def setup_logging(log_dir: str, log_level=None):
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
    return logging.getLogger(name)
