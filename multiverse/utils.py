import torch
from .logging_utils import get_logger

logger = get_logger(__name__)

def get_device(device_str: str):
    """
    Gets a torch.device object.

    Args:
        device_str: "cpu" or a cuda device string like "cuda:0".

    Returns:
        A torch.device object.
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
