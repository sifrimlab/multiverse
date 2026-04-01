import docker
import os
import asyncio
from ..logging_utils import get_logger

logger = get_logger(__name__)


async def build_images_concurrently(
    image_tags: list, status_callback: callable = None
):
    """Ensures all required Docker images for models are prepared concurrently.

    Args:
        image_tags (list): A list of Docker image tags to pull or build.
        status_callback (callable, optional): A function called with (image_tag, status)
            to update the progress in real-time.

    Raises:
        RuntimeError: If one or more images fail to pull or build.
    """
    client = docker.from_env()
    loop = asyncio.get_running_loop()

    def pull_image(tag):
        try:
            if status_callback:
                status_callback(tag, "Building/Pulling")
            logger.info(f"Pulling/Building image: {tag}")
            client.images.pull(tag)
            logger.info(f"Successfully pulled image: {tag}")
            if status_callback:
                status_callback(tag, "Ready")
            return True
        except Exception as e:
            logger.error(f"Failed to pull image {tag}: {e}")
            if status_callback:
                status_callback(tag, "Failed")
            raise

    tasks = [loop.run_in_executor(None, pull_image, tag) for tag in image_tags]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [res for res in results if isinstance(res, Exception)]
    if failures:
        logger.error(f"Failed to prepare {len(failures)} images.")
        raise RuntimeError(f"Failed to prepare some Docker images: {failures}")


async def run_models_concurrently(
    models_info: list,
    data_path: str,
    seed: int,
    output_dir: str,
    status_callback: callable = None,
):
    """Executes eligible models in parallel using isolated Docker containers.

    Each model is run in its own container with the input data mounted as read-only.
    The process captures exit codes and handles failures without stopping other models.

    Args:
        models_info (list of dict): A list of dictionaries containing 'name' and 'image' keys.
        data_path (str): The host path to the input dataset.
        seed (int): The random seed to inject as an environment variable.
        output_dir (str): The host path where results will be stored.
        status_callback (callable, optional): A function called with (model_name, status)
            to update the execution progress.

    Returns:
        dict: A dictionary mapping model names to their final status ("success" or "failed").
    """
    client = docker.from_env()
    loop = asyncio.get_running_loop()

    async def run_single_model(model_name, image_tag):
        if status_callback:
            status_callback(model_name, "Starting")
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
            "remove": False,
        }

        try:
            logger.info(f"Starting container for model: {model_name} using image: {image_tag}")
            container = await loop.run_in_executor(None, lambda: client.containers.run(**run_kwargs))

            if status_callback:
                status_callback(model_name, "Running")

            # Wait for container to finish
            result = await loop.run_in_executor(None, container.wait)
            exit_code = result.get("StatusCode", 1)

            logs = await loop.run_in_executor(None, container.logs)
            if exit_code == 0:
                logger.info(f"Model {model_name} completed successfully.")
                if status_callback:
                    status_callback(model_name, "Success")
            else:
                logger.error(f"Model {model_name} failed with exit code {exit_code}. Logs: {logs.decode('utf-8')[-500:]}")
                if status_callback:
                    status_callback(model_name, f"Failed ({exit_code})")

            # Clean up
            await loop.run_in_executor(None, container.remove)

            return model_name, exit_code == 0
        except Exception as e:
            logger.error(f"Error running model {model_name}: {e}")
            if status_callback:
                status_callback(model_name, "Error")
            return model_name, False

    tasks = [run_single_model(m["name"], m["image"]) for m in models_info]
    results = await asyncio.gather(*tasks)

    summary = {name: "success" if success else "failed" for name, success in results}
    return summary


def run_model_container(
    model_name: str,
    input_dir: str,
    output_dir: str,
    extra_args: list = None,
    use_gpu: bool = True,
):
    """Runs a single model container synchronously.

    Args:
        model_name (str): The name of the model to run.
        input_dir (str): The host path for input data.
        output_dir (str): The host path for output results.
        extra_args (list, optional): Additional command-line arguments for the container.
        use_gpu (bool): Whether to enable GPU support. Defaults to True.

    Raises:
        ValueError: If the model name is not recognized.
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


def run_evaluation_container(
    input_dir: str, output_dir: str, extra_args: list = None, use_gpu: bool = True
):
    """Runs the evaluation metrics container synchronously.

    Args:
        input_dir (str): The host path for input data.
        output_dir (str): The host path for output results.
        extra_args (list, optional): Additional command-line arguments for the container.
        use_gpu (bool): Whether to enable GPU support. Defaults to True.
    """
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
