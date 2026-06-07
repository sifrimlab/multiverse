"""Dependency-clean JobSpec model and writer.

No h5py, numpy, mudata, anndata, Docker, MLflow, Optuna, or Streamlit here.
This module is safe to import in the thin host environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

from .paths import CONTAINER_INPUT_DATA_PATH


class JobSpec(BaseModel):
    """Schema for the job_spec.json written by the orchestrator before container launch.

    Fields are kept additive-only.  Do not remove or rename fields while
    containers built against old images are still in use.
    """

    model_name: str
    model_version: str
    dataset_slug: str
    dataset_path_in_container: str = CONTAINER_INPUT_DATA_PATH
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None
    batch_key: Optional[str] = None
    cell_type_key: Optional[str] = None
    preprocessing: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


def job_spec_payload(
    *,
    model_name: str,
    model_version: str,
    dataset_slug: str,
    hyperparameters: Dict[str, Any],
    seed: Optional[int] = None,
    batch_key: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    dataset_path_in_container: str = CONTAINER_INPUT_DATA_PATH,
) -> Dict[str, Any]:
    """Build a canonical job-spec payload dict from executor-level inputs.

    Returns a plain dict (not a ``JobSpec``) so callers can pass it directly
    to ``write_job_spec`` or compare payloads across backends without
    deserialising.
    """
    payload: Dict[str, Any] = {
        "dataset_path_in_container": dataset_path_in_container,
        "dataset_slug": dataset_slug,
        "hyperparameters": dict(hyperparameters),
        "model_name": model_name,
        "model_version": model_version,
    }
    if seed is not None:
        payload["seed"] = seed
    if batch_key is not None:
        payload["batch_key"] = batch_key
    if cell_type_key is not None:
        payload["cell_type_key"] = cell_type_key
    if preprocessing is not None:
        payload["preprocessing"] = dict(preprocessing)
    return payload


def write_job_spec(
    path: Union[str, Path],
    payload: Union[Dict[str, Any], JobSpec],
) -> None:
    """Write ``payload`` to ``path`` as stable, sorted, UTF-8 JSON.

    ``payload`` may be a plain dict (from :func:`job_spec_payload`) or a
    :class:`JobSpec` instance.  The output is always ``sort_keys=True``
    with ``indent=2`` so byte-stability holds for the same logical job.
    """
    if isinstance(payload, JobSpec):
        data = payload.model_dump(exclude_none=True)
    else:
        data = dict(payload)

    Path(path).write_text(
        json.dumps(data, sort_keys=True, indent=2),
        encoding="utf-8",
    )
