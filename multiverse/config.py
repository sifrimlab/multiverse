import json
from .logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path="./config.json"):
    """
    Load the configuration from a JSON file
    Parameters:
    - config_path (str): Path to the JSON configuration file.

    Returns:
    - dict: Dictionary of hyperparameters and settings.
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