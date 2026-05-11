from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict

from ..logging_utils import get_logger
from .docker_runner import run_job_container_sync

logger = get_logger(__name__)


def _sample_param(trial: Any, name: str, spec: Dict[str, Any]) -> Any:
    dist_type = spec.get("type")
    if dist_type == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if dist_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if dist_type == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    raise ValueError(f"Unsupported distribution type for '{name}': {dist_type}")


def sample_hyperparameters(trial: Any, distributions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}
    for param_name, spec in distributions.items():
        sampled[param_name] = _sample_param(trial, param_name, spec)
    return sampled


def _extract_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    if metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
        return float(metrics[metric_name])
    # fallback: dotted path traversal for nested metrics
    cur: Any = metrics
    for key in metric_name.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Metric '{metric_name}' not found in metrics.json")
        cur = cur[key]
    if not isinstance(cur, (int, float)):
        raise TypeError(f"Metric '{metric_name}' is not numeric")
    return float(cur)


def objective(trial: Any, job_manifest: Dict[str, Any]) -> float:
    optuna = __import__("optuna")

    params = sample_hyperparameters(trial, job_manifest["search_space"])
    run_settings = dict(job_manifest.get("run_settings", {}))
    run_settings["optuna_trial_id"] = trial.number
    mlflow_tags = dict(run_settings.get("mlflow_tags", {}))
    mlflow_tags["optuna_trial_id"] = str(trial.number)
    run_settings["mlflow_tags"] = mlflow_tags

    trial_job = dict(job_manifest)
    trial_job["hyperparameters"] = params
    trial_job["run_settings"] = run_settings
    trial_job["name"] = f"{job_manifest.get('name', 'sweep')}_trial_{trial.number}"

    result = run_job_container_sync(
        trial_job,
        seed=int(job_manifest.get("seed", 42)),
        use_gpu=bool(job_manifest.get("use_gpu", True)),
        mem_limit=str(job_manifest.get("mem_limit", "16g")),
    )
    if result["exit_code"] != 0:
        logger.warning(f"Trial {trial.number} failed container execution; pruning.")
        raise optuna.exceptions.TrialPruned()

    metrics_path = os.path.join(result["output_path"], "metrics.json")
    if not os.path.exists(metrics_path):
        raise RuntimeError(f"metrics.json missing for successful trial at {result['output_path']}")
    with open(metrics_path, "r", encoding="utf-8") as fp:
        metrics_payload = json.load(fp)

    return _extract_metric(metrics_payload, str(job_manifest["optimize_metric"]))


def _apply_wal_mode_to_optuna_db(storage_uri: str) -> None:
    """Apply WAL journal mode to the Optuna SQLite DB before the study is created.

    WAL mode is a persistent DB property, so optuna-dashboard (and any other
    reader) will benefit from concurrent reads without acquiring a write lock.
    This is a no-op for non-SQLite backends.
    """
    prefix = "sqlite:///"
    if not storage_uri.startswith(prefix):
        return
    db_path = storage_uri[len(prefix):]
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"Could not apply WAL mode to Optuna DB '{db_path}': {exc}")


def run_sweep(job: Dict[str, Any]) -> Dict[str, Any]:
    optuna = __import__("optuna")

    study_name = str(job.get("study_name", f"sweep_{job.get('dataset_name', 'dataset')}_{job.get('model_name_orig', 'model')}"))
    storage_uri = str(job.get("study_storage", "sqlite:///store/optuna.db"))
    direction = str(job.get("direction", "maximize"))
    n_trials = int(job.get("n_trials", 10))

    _apply_wal_mode_to_optuna_db(storage_uri)

    logger.info(
        f"Starting Optuna sweep study='{study_name}' storage='{storage_uri}' "
        f"trials={n_trials} direction={direction}"
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction=direction,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, job), n_trials=n_trials)

    return {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
    }
