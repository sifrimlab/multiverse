import json
from .logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path="./config.json"):
    """Load the configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file. Defaults to "./config.json".

    Returns:
        dict: Dictionary of hyperparameters and settings.

    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.
        json.JSONDecodeError: If the configuration file contains invalid JSON.
        Exception: For any other unexpected errors during file loading.
    """
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
        logger.error(f"An unexpected error occurred while loading the configuration file: {e}")
        raise
    return config