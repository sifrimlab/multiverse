"""multiverse_config.py — read/write multiverse.config.yaml at project root."""

import os
import yaml

from .registry_db import BASE_DIR

CONFIG_PATH = os.path.join(BASE_DIR, "multiverse.config.yaml")
DEFAULT_DOCKER_DATA_ROOT = os.path.join(BASE_DIR, ".docker-data")


def get_config() -> dict:
    """Return the current config, falling back to defaults if the file is absent."""
    if not os.path.exists(CONFIG_PATH):
        return {"docker_data_root": DEFAULT_DOCKER_DATA_ROOT}
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("docker_data_root", DEFAULT_DOCKER_DATA_ROOT)
    return data


def save_config(config: dict) -> None:
    """Persist *config* to multiverse.config.yaml."""
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)


def get_docker_data_root() -> str:
    """Return the configured Docker data root path."""
    return get_config()["docker_data_root"]
