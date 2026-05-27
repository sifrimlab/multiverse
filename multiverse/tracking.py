from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logging_utils import get_logger

logger = get_logger(__name__)


def sanitize_nan_inf(obj: Any) -> Any:
    """Recursively replace NaN and +/-Inf float values with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: sanitize_nan_inf(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan_inf(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_nan_inf(value) for value in obj)
    return obj


def _to_flat_float_metrics(metrics_obj: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested metric payloads into scalar floats for MLflow."""
    flat: Dict[str, float] = {}
    for key, value in metrics_obj.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if key == "history":
            continue
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


def _coerce_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    out: List[float] = []
    for item in values:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def split_metrics_payload(metrics: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Split metrics.json into final scalars and per-epoch time series."""
    scalars: Dict[str, float] = {}
    histories: Dict[str, List[float]] = {}

    history_block = metrics.get("history")
    if isinstance(history_block, dict):
        for key, values in history_block.items():
            series = _coerce_float_list(values)
            if series:
                histories[str(key)] = series

    for key, value in metrics.items():
        if key == "history":
            continue
        if isinstance(value, list):
            series = _coerce_float_list(value)
            if series:
                histories[str(key)] = series
                scalars[str(key)] = series[-1]
            continue
        if isinstance(value, (int, float)):
            scalars[str(key)] = float(value)
        elif isinstance(value, dict):
            scalars.update(_to_flat_float_metrics({key: value}))

    for key, series in histories.items():
        scalars.setdefault(key, series[-1])

    return scalars, histories


def _log_metric_histories(mlflow: Any, histories: Dict[str, List[float]]) -> None:
    for name, series in histories.items():
        for step, value in enumerate(series):
            mlflow.log_metric(name, value, step=step)


def _log_final_scalars(
    mlflow: Any,
    scalars: Dict[str, float],
    histories: Dict[str, List[float]],
) -> None:
    standalone = {k: v for k, v in scalars.items() if k not in histories}
    if not standalone:
        return
    if histories:
        final_step = max(len(series) for series in histories.values()) - 1
        mlflow.log_metrics(standalone, step=final_step)
    else:
        mlflow.log_metrics(standalone)


def _log_final_scalars_client(
    client: Any,
    run_id: str,
    scalars: Dict[str, float],
    histories: Dict[str, List[float]],
) -> None:
    standalone = {k: v for k, v in scalars.items() if k not in histories}
    if not standalone:
        return
    step = max(len(series) for series in histories.values()) - 1 if histories else None
    for key, value in standalone.items():
        if step is None:
            client.log_metric(run_id, key, value)
        else:
            client.log_metric(run_id, key, value, step=step)


def find_metrics_json(workspace_dir: str) -> Optional[str]:
    """Return the path to metrics.json under a run workspace, if present."""
    root = os.path.join(workspace_dir, "metrics.json")
    if os.path.isfile(root):
        return root
    for dirpath, _, filenames in os.walk(workspace_dir):
        if "metrics.json" in filenames:
            return os.path.join(dirpath, "metrics.json")
    return None


def load_run_metrics(workspace_dir: str) -> Dict[str, Any]:
    """Load metrics.json from workspace root or nested model output."""
    path = find_metrics_json(workspace_dir)
    if not path:
        return {}
    try:
        import json

        with open(path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        return sanitize_nan_inf(loaded) if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.warning("Failed reading metrics.json from %s: %s", path, exc)
        return {}


def _resolve_mlflow_settings(
    job_context: Dict[str, Any],
    job_spec: Dict[str, Any],
) -> Tuple[Optional[Any], str, Dict[str, Any]]:
    """Return (mlflow module, experiment_name, run_settings) or (None, "", {}) on failure."""
    try:
        mlflow = importlib.import_module("mlflow")
    except Exception as exc:
        logger.warning(f"MLflow unavailable; skipping tracking. ({exc})")
        return None, "", {}

    run_settings = job_spec.get("run_settings", {}) if isinstance(job_spec, dict) else {}
    tracking_uri = (
        run_settings.get("mlflow_tracking_uri")
        or job_context.get("mlflow_tracking_uri")
        or os.getenv("MLFLOW_TRACKING_URI")
        or "http://localhost:5000"
    )
    experiment_name = (
        run_settings.get("mlflow_experiment_name")
        or job_context.get("experiment_name")
        or "default_experiment"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow, experiment_name, run_settings


def start_parent_mlflow_run(
    *,
    job_context: Dict[str, Any],
    job_spec: Dict[str, Any],
    run_name: str,
) -> Optional[str]:
    """Create an MLflow parent run without relying on fluent active-run state.

    The returned run_id is forwarded to the container via ``MLFLOW_RUN_ID`` so
    EpochLogger can attach to the same run. Host-side finalization uses
    MlflowClient with explicit run IDs to avoid cross-job active-run conflicts.
    """
    mlflow, experiment_name, run_settings = _resolve_mlflow_settings(job_context, job_spec)
    if mlflow is None:
        return None

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(experiment_name)
        )
        tags: Dict[str, str] = {"mlflow.runName": run_name}
        if isinstance(run_settings.get("mlflow_tags"), dict):
            tags.update({str(k): str(v) for k, v in run_settings["mlflow_tags"].items()})
        if "optuna_trial_id" in run_settings:
            tags["optuna_trial_id"] = str(run_settings["optuna_trial_id"])

        run = client.create_run(str(experiment_id), tags=tags)
        run_id = run.info.run_id
        for key, value in _to_flat_str_params(job_spec.get("hyperparameters", {})).items():
            client.log_param(run_id, key, value)
        return run_id
    except Exception as exc:
        logger.warning("MLflow start_parent failed: %s", exc)
        return None


def finalize_parent_mlflow_run(
    *,
    run_id: str,
    job_context: Dict[str, Any],
    job_spec: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts_dir: str,
    status: str = "FINISHED",
) -> None:
    """Attach to a parent MLflow run, append final scalars + artifacts, end it.

    Per-epoch history is *not* re-logged here — the container's
    :class:`EpochLogger` already streamed it into the same run. This finalizer
    only adds whatever the container could not produce on its own: the final
    artifact bundle and any scalar metrics missing from the live stream.
    """
    mlflow, _experiment, _settings = _resolve_mlflow_settings(job_context, job_spec)
    if mlflow is None:
        return

    try:
        client = mlflow.tracking.MlflowClient()
    except Exception as exc:
        logger.warning("MLflow finalize client init failed: %s", exc)
        return

    try:
        scalars, histories = split_metrics_payload(metrics)
        try:
            _log_final_scalars_client(client, run_id, scalars, histories)
        except Exception as exc:
            logger.warning("MLflow final scalar logging failed: %s", exc)
        _log_artifacts_filtered_client(client, run_id, artifacts_dir)
    except Exception as exc:
        logger.warning("MLflow finalize body raised: %s", exc)
    finally:
        try:
            client.set_terminated(run_id, status=status)
        except Exception as exc:
            logger.warning("MLflow set_terminated failed: %s", exc)


def log_successful_run_to_mlflow(
    *,
    job_context: Dict[str, Any],
    job_spec: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts_dir: str,
) -> None:
    """Legacy single-shot logger used when no parent run is open.

    Kept for non-Docker callers; new container path uses
    :func:`start_parent_mlflow_run` + :func:`finalize_parent_mlflow_run`.
    """
    mlflow, _experiment, run_settings = _resolve_mlflow_settings(job_context, job_spec)
    if mlflow is None:
        return

    params = _to_flat_str_params(job_spec.get("hyperparameters", {}))
    scalars, histories = split_metrics_payload(metrics)
    dataset_name = (
        job_context.get("dataset_name")
        or job_context.get("dataset_slug")
        or "dataset"
    )
    run_name = (
        f"{dataset_name}"
        f"-{job_spec.get('model_name', job_context.get('model_name_orig', 'model'))}"
        f"-{Path(artifacts_dir).name}"
    )

    start_kwargs: Dict[str, Any] = {"run_name": run_name}
    try:
        import inspect
        if "log_system_metrics" in inspect.signature(mlflow.start_run).parameters:
            start_kwargs["log_system_metrics"] = True
    except Exception:
        pass

    with mlflow.start_run(**start_kwargs):
        try:
            mlflow_tags: Dict[str, str] = {}
            if isinstance(run_settings.get("mlflow_tags"), dict):
                mlflow_tags.update({str(k): str(v) for k, v in run_settings["mlflow_tags"].items()})
            if "optuna_trial_id" in run_settings:
                mlflow_tags["optuna_trial_id"] = str(run_settings["optuna_trial_id"])
            if mlflow_tags:
                try:
                    mlflow.set_tags(mlflow_tags)
                except Exception as exc:
                    logger.warning("MLflow set_tags failed: %s", exc)
            if params:
                try:
                    mlflow.log_params(params)
                except Exception as exc:
                    logger.warning("MLflow log_params failed: %s", exc)
            if histories:
                try:
                    _log_metric_histories(mlflow, histories)
                except Exception as exc:
                    logger.warning("MLflow metric history logging failed: %s", exc)
            try:
                _log_final_scalars(mlflow, scalars, histories)
            except Exception as exc:
                logger.warning("MLflow final scalar logging failed: %s", exc)
            _log_artifacts_filtered(mlflow, artifacts_dir)
        except Exception as exc:
            logger.warning("MLflow run body raised; marking FINISHED with partial data: %s", exc)


_ARTIFACT_SKIP_NAMES = {".promotion_complete"}


def _log_artifacts_filtered(mlflow: Any, artifacts_dir: str) -> None:
    """Log artifacts file-by-file, skipping markers and tolerating per-file errors."""
    if not os.path.isdir(artifacts_dir):
        return
    for dirpath, _, filenames in os.walk(artifacts_dir):
        rel_dir = os.path.relpath(dirpath, artifacts_dir)
        artifact_path = None if rel_dir in (".", "") else rel_dir
        for filename in filenames:
            if filename in _ARTIFACT_SKIP_NAMES:
                continue
            full = os.path.join(dirpath, filename)
            try:
                mlflow.log_artifact(full, artifact_path=artifact_path)
            except Exception as exc:
                logger.warning("MLflow log_artifact failed for %s: %s", full, exc)


def _log_artifacts_filtered_client(client: Any, run_id: str, artifacts_dir: str) -> None:
    """Log artifacts with explicit run_id, skipping markers and tolerating per-file errors."""
    if not os.path.isdir(artifacts_dir):
        return
    for dirpath, _, filenames in os.walk(artifacts_dir):
        rel_dir = os.path.relpath(dirpath, artifacts_dir)
        artifact_path = None if rel_dir in (".", "") else rel_dir
        for filename in filenames:
            if filename in _ARTIFACT_SKIP_NAMES:
                continue
            full = os.path.join(dirpath, filename)
            try:
                client.log_artifact(run_id, full, artifact_path=artifact_path)
            except Exception as exc:
                logger.warning("MLflow log_artifact failed for %s: %s", full, exc)
