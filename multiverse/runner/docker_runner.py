import docker
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ..logging_utils import get_logger

logger = get_logger(__name__)


async def build_images_concurrently(image_tags):
    """
    Ensure all required Docker images for eligible models are built/pulled concurrently.

    Args:
        image_tags (list): List of Docker image tags to pull/build.
    """
    client = docker.from_env()
    loop = asyncio.get_running_loop()

    def pull_image(tag):
        try:
            logger.info(f"Pulling/Building image: {tag}")
            # In a real scenario, this might be client.images.pull(tag)
            # or client.images.build(path=..., tag=tag)
            # For this implementation, we'll use pull as it's more common for "getting" images.
            client.images.pull(tag)
            logger.info(f"Successfully pulled image: {tag}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull image {tag}: {e}")
            raise

    tasks = [loop.run_in_executor(None, pull_image, tag) for tag in image_tags]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [res for res in results if isinstance(res, Exception)]
    if failures:
        logger.error(f"Failed to prepare {len(failures)} images.")
        raise RuntimeError(f"Failed to prepare some Docker images: {failures}")


async def run_models_concurrently(models_info, data_path, seed, output_dir):
    """
    Execute eligible models in parallel via Docker.

    Args:
        models_info (list of dict): List containing model name and image tag.
        data_path (str): Path to the input dataset (will be mounted RO).
        seed (int): Random seed to inject as environment variable.
        output_dir (str): Base output directory.

    Returns:
        dict: Summary of model runs mapping model name to success status.
    """
    client = docker.from_env()
    loop = asyncio.get_running_loop()

    async def run_single_model(model_name, image_tag):
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        run_kwargs = {
            "image": image_tag,
            "environment": {"RANDOM_SEED": str(seed)},
            "volumes": {
                os.path.abspath(data_path): {"bind": "/data/input", "mode": "ro"},
                os.path.abspath(model_output_dir): {"bind": "/data/outputs", "mode": "rw"},
            },
            "detach": True,
            "remove": False, # We want to check exit code before removal
        }

        try:
            logger.info(f"Starting container for model: {model_name} using image: {image_tag}")
            container = await loop.run_in_executor(None, lambda: client.containers.run(**run_kwargs))

            # Wait for container to finish
            result = await loop.run_in_executor(None, container.wait)
            exit_code = result.get("StatusCode", 1)

            logs = await loop.run_in_executor(None, container.logs)
            if exit_code == 0:
                logger.info(f"Model {model_name} completed successfully.")
            else:
                logger.error(f"Model {model_name} failed with exit code {exit_code}. Logs: {logs.decode('utf-8')[-500:]}")

            # Clean up
            await loop.run_in_executor(None, container.remove)

            return model_name, exit_code == 0
        except Exception as e:
            logger.error(f"Error running model {model_name}: {e}")
            return model_name, False

    tasks = [run_single_model(m["name"], m["image"]) for m in models_info]
    results = await asyncio.gather(*tasks)

    summary = {name: "success" if success else "failed" for name, success in results}
    return summary


def run_model_container(
    model_name, input_dir, output_dir, extra_args=None, use_gpu=True
):
    """
    Run a model container with optional GPU support (Synchronous version).
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
    """Run the evaluation container with optional GPU support (Synchronous version)."""
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
