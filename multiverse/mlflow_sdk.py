"""Thread-safe access to the MLflow Python SDK.

MLflow 3.x performs a large import graph on first load. Concurrent imports
from Streamlit worker threads (fragments, ``@st.cache_data``) can leave the
package partially initialized and raise circular-import errors. All host-side
code should obtain the module or client through this module.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Optional

_lock = threading.Lock()
_mlflow: Optional[Any] = None


def import_mlflow() -> Any:
    """Return the fully initialized ``mlflow`` module."""
    global _mlflow
    if _mlflow is not None:
        return _mlflow
    with _lock:
        if _mlflow is None:
            _mlflow = importlib.import_module("mlflow")
        return _mlflow


def get_mlflow_client(*, tracking_uri: str | None = None) -> Any:
    """Return an ``MlflowClient``, optionally bound to *tracking_uri*."""
    mlflow = import_mlflow()
    if tracking_uri is None:
        return mlflow.tracking.MlflowClient()
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
