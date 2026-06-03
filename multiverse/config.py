import json
from typing import Union

from .logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: Union[str, dict] = "./config.json"):
    """Load the configuration from a JSON file or return an in-memory dict unchanged.

    Args:
        config_path: Path to the JSON configuration file, or a configuration dict.

    Returns:
        dict: Dictionary of hyperparameters and settings.

    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.
        json.JSONDecodeError: If the configuration file contains invalid JSON.
        Exception: For any other unexpected errors during file loading.
    """
    if isinstance(config_path, dict):
        return config_path

    try:
        logger.info("Loading .json file")
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        logger.info("Information from json file loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading the configuration file: {e}"
        )
        raise
    return config
