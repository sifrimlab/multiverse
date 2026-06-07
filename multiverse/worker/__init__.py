"""Worker SDK for model containers (I/O, preprocessing, metrics, device).

Containers import this package instead of the orchestrator. Environment
variables ``MVR_*`` locate inputs and outputs; see :mod:`multiverse.worker.io`.

Install with: ``pip install multiverse[worker]``
"""

from .base import ModelFactory
from .device import get_device, resolve_device
from .epoch_logger import (
    EpochLogger,
    replay_history,
    sanitize_nan_inf,
    scvi_history_to_dict,
    series_to_float_list,
)
from .io import (
    INPUT_DATA_PATH,
    JOB_SPEC_PATH,
    OUTPUT_DIR,
    anndata_concatenate,
    build_model_config,
    load_config,
    load_input_mudata,
    load_job_spec,
    preprocess_mudata,
    resolve_labels_key_params,
    resolve_preprocess_params,
    save_embeddings,
    save_umap,
    setup_container_logging,
)
from .logging import get_logger, setup_logging

__all__ = [
    "INPUT_DATA_PATH",
    "JOB_SPEC_PATH",
    "OUTPUT_DIR",
    "EpochLogger",
    "ModelFactory",
    "anndata_concatenate",
    "build_model_config",
    "get_device",
    "get_logger",
    "load_config",
    "load_input_mudata",
    "load_job_spec",
    "preprocess_mudata",
    "replay_history",
    "resolve_device",
    "resolve_labels_key_params",
    "resolve_preprocess_params",
    "sanitize_nan_inf",
    "save_embeddings",
    "save_umap",
    "scvi_history_to_dict",
    "series_to_float_list",
    "setup_container_logging",
    "setup_logging",
]
