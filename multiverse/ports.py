"""Default host ports for observability services on shared machines."""

from __future__ import annotations

import os

DEFAULT_MLFLOW_PORT = 25_000
DEFAULT_OPTUNA_PORT = 28_080
DEFAULT_STREAMLIT_PORT = 28_501


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default.

    Args:
        name: Environment variable name to read.
        default: Value returned when the variable is unset or empty.

    Returns:
        The parsed integer, or ``default`` when the variable is absent.

    Raises:
        ValueError: If the variable is set but not a valid integer.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    return int(raw)


def mlflow_port() -> int:
    """Return the MLflow host port (``MLFLOW_PORT`` override or the default)."""
    return _int_env("MLFLOW_PORT", DEFAULT_MLFLOW_PORT)


def optuna_port() -> int:
    """Return the Optuna dashboard host port (``OPTUNA_PORT`` override or the default)."""
    return _int_env("OPTUNA_PORT", DEFAULT_OPTUNA_PORT)


def streamlit_port() -> int:
    """Return the Streamlit host port (``STREAMLIT_PORT`` override or the default)."""
    return _int_env("STREAMLIT_PORT", DEFAULT_STREAMLIT_PORT)


def default_mlflow_tracking_uri() -> str:
    """Return the MLflow tracking URI.

    Honors ``MLFLOW_TRACKING_URI`` when set; otherwise points at the local
    MLflow projection on :func:`mlflow_port`.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or f"http://localhost:{mlflow_port()}"
