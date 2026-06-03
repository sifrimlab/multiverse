from __future__ import annotations

import sqlite3
from typing import Any, Dict

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _sample_param(trial: Any, name: str, spec: Dict[str, Any]) -> Any:
    dist_type = spec.get("type")
    if dist_type == "int":
        return trial.suggest_int(
            name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
        )
    if dist_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if dist_type == "loguniform":
        return trial.suggest_float(
            name, float(spec["low"]), float(spec["high"]), log=True
        )
    raise ValueError(f"Unsupported distribution type for '{name}': {dist_type}")


def sample_hyperparameters(
    trial: Any, distributions: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}
    for param_name, spec in distributions.items():
        sampled[param_name] = _sample_param(trial, param_name, spec)
    return sampled


def _extract_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    optuna = __import__("optuna")
    if metric_name == "history":
        raise optuna.TrialPruned("Metric 'history' is not a scalar optimize target")
    if metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
        return float(metrics[metric_name])
    cur: Any = metrics
    for key in metric_name.split("."):
        if not isinstance(cur, dict):
            raise optuna.TrialPruned(
                f"Metric '{metric_name}' not found in metrics.json"
            )
        cur = cur.get(key)
        if cur is None:
            raise optuna.TrialPruned(
                f"Metric '{metric_name}' not found in metrics.json"
            )
    if not isinstance(cur, (int, float)):
        raise optuna.TrialPruned(f"Metric '{metric_name}' is not numeric")
    return float(cur)


def objective(trial: Any, job_manifest: Dict[str, Any]) -> float:
    raise NotImplementedError(
        "The legacy docker_runner sweep path was removed in G6. "
        "Wire MvdDockerExecutor / MvdSlurmExecutor into your Optuna objective instead."
    )


def _apply_wal_mode_to_optuna_db(storage_uri: str) -> None:
    """Apply WAL journal mode to the Optuna SQLite DB before the study is created.

    WAL mode is a persistent DB property, so optuna-dashboard (and any other
    reader) will benefit from concurrent reads without acquiring a write lock.
    This is a no-op for non-SQLite backends.
    """
    prefix = "sqlite:///"
    if not storage_uri.startswith(prefix):
        return
    db_path = storage_uri[len(prefix) :]
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

    study_name = str(
        job.get(
            "study_name",
            f"sweep_{job.get('dataset_name', 'dataset')}_{job.get('model_name_orig', 'model')}",
        )
    )
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
