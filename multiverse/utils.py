import torch

from .logging_utils import get_logger

logger = get_logger(__name__)


def get_device(device_str: str):
    """Creates a torch.device object based on the provided string.

    Args:
        device_str (str): The device identifier string (e.g., "cpu", "cuda:0").

    Returns:
        torch.device: The corresponding torch device object.
    """
    if device_str != "cpu":
        if torch.cuda.is_available():
            logger.info(f"GPU available. Using device: {device_str}")
            return torch.device(device_str)
        else:
            logger.warning("GPU not available, defaulting to CPU.")
            return torch.device("cpu")
    else:
        logger.info("Using CPU device.")
        return torch.device("cpu")
