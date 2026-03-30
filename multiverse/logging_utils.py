import logging
import os

def setup_logging(log_dir: str, log_level=logging.INFO):
    """
    Configures the root logger to write to a file.
    """
    log_file = os.path.join(log_dir, "multiverse.log")

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

def get_logger(name: str):
    """
    Returns a logger instance.
    """
    return logging.getLogger(name)
