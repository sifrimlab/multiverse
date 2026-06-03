"""multiverse_config.py — read/write the per-user multiverse config file.

Before M1 this lived at ``<package install dir>/multiverse.config.yaml``,
which collides on shared installs. The file now lives under the per-user
config directory (XDG-aware), and ``docker_data_root`` defaults to a
subdirectory of the resolved state root rather than the package dir.

Legacy installs are honored read-only: if a config file exists at the
old location and no per-user one does yet, we read it but never write
back there. ``mvexp migrate-state-dir`` is the one-shot relocation.
"""

import os
from pathlib import Path
from typing import Optional

import yaml

from .state_paths import CONFIG_FILENAME
from .state_paths import REPO_ROOT_GUESS as _REPO_ROOT_GUESS
from .state_paths import find_config_file, resolve_state_root


def _default_user_config_path() -> Path:
    """Where ``save_config`` writes when no file exists yet."""
    env = os.environ
    xdg = env.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "mvexp" / CONFIG_FILENAME
    home = env.get("HOME")
    if home:
        return Path(home) / ".config" / "mvexp" / CONFIG_FILENAME
    return Path.cwd() / CONFIG_FILENAME


def _legacy_config_path() -> Path:
    return _REPO_ROOT_GUESS / CONFIG_FILENAME


def _resolve_config_path_for_read() -> Optional[Path]:
    """First per-user location with a file; fall back to legacy if any."""
    found = find_config_file()
    if found is not None:
        return found
    legacy = _legacy_config_path()
    return legacy if legacy.is_file() else None


def _default_docker_data_root() -> str:
    return str(resolve_state_root() / ".docker-data")


# Kept as a module-level path for callers that import it directly. It now
# points at the user-writable location chosen by the resolver.
CONFIG_PATH = str(_default_user_config_path())
DEFAULT_DOCKER_DATA_ROOT = _default_docker_data_root()


def get_config() -> dict:
    """Return the current config, falling back to defaults if absent."""
    path = _resolve_config_path_for_read()
    if path is None:
        return {"docker_data_root": _default_docker_data_root()}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return {"docker_data_root": _default_docker_data_root()}
    data.setdefault("docker_data_root", _default_docker_data_root())
    return data


def save_config(config: dict) -> None:
    """Persist *config* to the per-user config file.

    Writes never target the legacy package-directory location even if a
    legacy file exists; reads honor it but writes migrate to user space.
    """
    target = _default_user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def get_docker_data_root() -> str:
    """Return the configured Docker data root path."""
    return get_config()["docker_data_root"]
