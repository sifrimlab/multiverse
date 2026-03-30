import docker
import os
from ..logging_utils import get_logger

logger = get_logger(__name__)


def run_model_container(
    model_name, input_dir, output_dir, extra_args=None, use_gpu=True
):
    """
    Run a model container with optional GPU support.

    Parameters
    ----------
    model_name : str
        Name of the model to run. Must be one of: {"pca", "mofa", "multivi", "mowgli", "cobolt"}.
    input_dir : str
        Path to the input directory.
    output_dir : str
        Path to the output directory.
    extra_args : list, optional
        Extra command arguments for the container.
    use_gpu : bool, default=True
        Whether to attempt running the container with GPU support if available.
    """
    client = docker.from_env()

    image_map = {
        "pca": "multiverse-pca",
        "mofa": "multiverse-mofa",
        "multivi": "multiverse-multivi",
        "mowgli": "multiverse-mowgli",
        "cobolt": "multiverse-cobolt",
        "totalvi": "multiverse-totalvi",
    }
    if model_name not in image_map:
        raise ValueError(f"Unknown model name: {model_name}")
    image = image_map[model_name]

    run_kwargs = {
        "image": image,
        "command": extra_args or [],
        "volumes": {
            os.path.abspath(input_dir): {"bind": "/data/input", "mode": "ro"},
            os.path.abspath(output_dir): {"bind": "/data/outputs", "mode": "rw"},
        },
        "detach": True,
        "remove": True,
    }

    # Add GPU support if requested and available
    if use_gpu:
        try:
            # Docker SDK only allows `device_requests` for GPU scheduling
            from docker.types import DeviceRequest

            run_kwargs["device_requests"] = [
                DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        except Exception as e:
            logger.warning(f"GPU support not available, running on CPU. ({e})")

    container = client.containers.run(**run_kwargs)

    for log in container.logs(stream=True):
        logger.info(log.decode().strip())


def run_evaluation_container(input_dir, output_dir, extra_args=None, use_gpu=True):
    """Run the evaluation container with optional GPU support."""
    client = docker.from_env()

    image = "multiverse-evaluate"

    run_kwargs = {
        "image": image,
        "command": extra_args or [],
        "volumes": {
            os.path.abspath(input_dir): {"bind": "/data/input", "mode": "ro"},
            os.path.abspath(output_dir): {"bind": "/data/outputs", "mode": "rw"},
        },
        "detach": True,
        "remove": True,
    }

    # Add GPU support if requested and available
    if use_gpu:
        try:
            # Docker SDK only allows `device_requests` for GPU scheduling
            from docker.types import DeviceRequest

            run_kwargs["device_requests"] = [
                DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        except Exception as e:
            logger.warning(f"GPU support not available, running on CPU. ({e})")

    container = client.containers.run(**run_kwargs)

    for log in container.logs(stream=True):
        logger.info(log.decode().strip())
