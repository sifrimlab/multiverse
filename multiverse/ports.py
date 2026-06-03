"""Default host ports for observability services on shared machines."""

from __future__ import annotations

import os

DEFAULT_MLFLOW_PORT = 25_000
DEFAULT_OPTUNA_PORT = 28_080
DEFAULT_STREAMLIT_PORT = 28_501


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    return int(raw)


def mlflow_port() -> int:
    return _int_env("MLFLOW_PORT", DEFAULT_MLFLOW_PORT)


def optuna_port() -> int:
    return _int_env("OPTUNA_PORT", DEFAULT_OPTUNA_PORT)


def streamlit_port() -> int:
    return _int_env("STREAMLIT_PORT", DEFAULT_STREAMLIT_PORT)


def default_mlflow_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"http://localhost:{mlflow_port()}"
