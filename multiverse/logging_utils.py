import logging
import os

LOG_LEVEL_ENV = "MVEXP_LOG_LEVEL"
"""Environment variable that overrides the default log level everywhere.

Accepts either a level name (``DEBUG``, ``INFO``, ``WARNING``, ...) or a
numeric level. Honoured by :func:`setup_logging`, the per-run orchestrator
log, and forwarded into model containers so ``mvr_worker`` matches the host.
"""


def resolve_log_level(default: int = logging.INFO) -> int:
    """Resolve the effective log level from ``$MVEXP_LOG_LEVEL``.

    Falls back to ``default`` when the variable is unset or unparseable, so
    a typo never silences logging entirely.
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
    """Configures the root logger to write to a file.

    Args:
        log_dir (str): The directory where the log file will be saved.
        log_level (int | None): The logging level to set (e.g., logging.INFO,
            logging.DEBUG). When ``None`` (the default) the level is resolved
            from ``$MVEXP_LOG_LEVEL``, defaulting to ``logging.INFO``.
    """
    if log_level is None:
        log_level = resolve_log_level()
    log_file = os.path.join(log_dir, "multiverse.log")

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="a")  # Append mode
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str):
    """Returns a logger instance with the given name.

    Args:
        name (str): The name for the logger.

    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(name)
