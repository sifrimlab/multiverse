from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .logging_utils import get_logger

logger = get_logger(__name__)


def _to_flat_float_metrics(metrics_obj: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested metric payloads into scalar floats for MLflow."""
    flat: Dict[str, float] = {}
    for key, value in metrics_obj.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_to_flat_float_metrics(value, prefix=metric_key))
            continue
        if isinstance(value, bool):
            flat[metric_key] = float(int(value))
            continue
        if isinstance(value, (int, float)):
            flat[metric_key] = float(value)
    return flat


def _to_flat_str_params(params_obj: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Flatten nested params into string values accepted by MLflow."""
    flat: Dict[str, str] = {}
    for key, value in params_obj.items():
        param_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_to_flat_str_params(value, prefix=param_key))
            continue
        flat[param_key] = str(value)
    return flat


def log_successful_run_to_mlflow(
    *,
    job_context: Dict[str, Any],
    job_spec: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts_dir: str,
) -> None:
    """Best-effort MLflow proxy logger for successful runs."""
    try:
        mlflow = importlib.import_module("mlflow")
    except Exception as exc:  # pragma: no cover - depends on runtime env
        logger.warning(f"MLflow unavailable; skipping tracking. ({exc})")
        return

    run_settings = job_spec.get("run_settings", {}) if isinstance(job_spec, dict) else {}
    tracking_uri = (
        run_settings.get("mlflow_tracking_uri")
        or job_context.get("mlflow_tracking_uri")
        or os.getenv("MLFLOW_TRACKING_URI")
        or "file:./mlruns"
    )
    experiment_name = (
        run_settings.get("mlflow_experiment_name")
        or job_context.get("experiment_name")
        or "default_experiment"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params = _to_flat_str_params(job_spec.get("hyperparameters", {}))
    metrics_flat = _to_flat_float_metrics(metrics)
    run_name = (
        f"{job_context.get('dataset_name', 'dataset')}"
        f"-{job_spec.get('model_name', job_context.get('model_name_orig', 'model'))}"
        f"-{Path(artifacts_dir).name}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow_tags = {}
        if isinstance(run_settings.get("mlflow_tags"), dict):
            mlflow_tags.update({str(k): str(v) for k, v in run_settings["mlflow_tags"].items()})
        if "optuna_trial_id" in run_settings:
            mlflow_tags["optuna_trial_id"] = str(run_settings["optuna_trial_id"])
        if mlflow_tags:
            mlflow.set_tags(mlflow_tags)
        if params:
            mlflow.log_params(params)
        if metrics_flat:
            mlflow.log_metrics(metrics_flat)
        mlflow.log_artifacts(artifacts_dir)

