"""Worker SDK for model containers (I/O, preprocessing, metrics, device).

Containers import this package instead of the orchestrator. Environment
variables ``MVR_*`` locate inputs and outputs; see :mod:`mvr_worker.io`.
"""

from .base import ModelFactory
from .device import get_device, resolve_device
from .epoch_logger import (
    EpochLogger,
    replay_history,
    scvi_history_to_dict,
    sanitize_nan_inf,
)
from .io import (
    INPUT_DATA_PATH,
    JOB_SPEC_PATH,
    OUTPUT_DIR,
    anndata_concatenate,
    build_model_config,
    load_input_mudata,
    load_job_spec,
    preprocess_mudata,
    resolve_preprocess_params,
    save_embeddings,
    save_umap,
    setup_container_logging,
    load_config,
)
from .logging import get_logger, setup_logging

__all__ = [
    "INPUT_DATA_PATH",
    "JOB_SPEC_PATH",
    "OUTPUT_DIR",
    "EpochLogger",
    "anndata_concatenate",
    "replay_history",
    "build_model_config",
    "get_logger",
    "load_input_mudata",
    "preprocess_mudata",
    "load_job_spec",
    "resolve_preprocess_params",
    "resolve_device",
    "save_embeddings",
    "save_umap",
    "setup_container_logging",
    "setup_logging",
    "ModelFactory",
    "get_device",
    "scvi_history_to_dict",
    "sanitize_nan_inf",
    "load_config",
]
