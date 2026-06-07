"""Optuna hyperparameter tuning, driven against the mvd executors.

Optuna stays the sequential controller (a Bayesian sampler/pruner cannot be
flattened into a static DAG): :func:`run_sweep` creates the study and
:func:`objective` evaluates one trial by sampling a hyperparameter vector,
running it as a single container attempt through the mvd kernel, then reading
that attempt's ``metrics.json``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Snapshot from the kernel for a successfully promoted attempt. Kept as a bare
# string so this module does not import the kernel state machine eagerly.
_ARTIFACT_SUCCESS = "ARTIFACT_SUCCESS"

# A trial runner takes (base job manifest, sampled params, trial number) and
# returns the kernel snapshot for the resulting attempt. Injected by
# :func:`run_sweep` so :func:`objective` is testable without Docker/Slurm.
TrialRunner = Callable[[Dict[str, Any], Dict[str, Any], int], Dict[str, Any]]


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


def objective(
    trial: Any,
    job_manifest: Dict[str, Any],
    *,
    run_trial: Optional[TrialRunner] = None,
) -> float:
    """Evaluate one Optuna trial as a single mvd run.

    1. Sample a hyperparameter vector from ``job_manifest['search_space']``.
    2. Run it as one container attempt through the kernel (``run_trial``).
    3. Read the attempt's ``metrics.json`` and return ``optimize_metric``.

    A trial whose attempt does not reach ``ARTIFACT_SUCCESS`` is pruned (so the
    study keeps going) rather than failing the whole sweep. ``run_trial`` is
    injected by :func:`run_sweep`; it defaults to the real kernel-backed runner.
    """
    optuna = __import__("optuna")

    search_space = job_manifest.get("search_space") or {}
    if not search_space:
        raise ValueError(
            "sweep job has an empty 'search_space'; nothing to tune"
        )
    metric_name = job_manifest.get("optimize_metric")
    if not metric_name:
        raise ValueError("sweep job is missing 'optimize_metric'")

    sampled = sample_hyperparameters(trial, search_space)

    runner = run_trial if run_trial is not None else _default_trial_runner(job_manifest)
    snapshot = runner(job_manifest, sampled, trial.number)

    primary_state = snapshot.get("primary_state")
    if primary_state != _ARTIFACT_SUCCESS:
        raise optuna.TrialPruned(
            f"trial {trial.number} attempt did not succeed "
            f"(state={primary_state}, reason={snapshot.get('failure_reason')})"
        )

    artifact_dir = snapshot.get("artifact_dir")
    trial.set_user_attr("artifact_dir", artifact_dir)
    trial.set_user_attr("physical_attempt_id", snapshot.get("physical_attempt_id"))
    trial.set_user_attr("trial_number", trial.number)
    if job_manifest.get("study_name"):
        trial.set_user_attr("study_name", job_manifest["study_name"])

    from ..tracking import load_run_metrics

    metrics = load_run_metrics(str(artifact_dir)) if artifact_dir else {}
    return _extract_metric(metrics, metric_name)


def _default_trial_runner(job_manifest: Dict[str, Any]) -> TrialRunner:
    """Build the production trial runner from a sweep job's execution context.

    ``run_sweep`` stamps the kernel-execution knobs (state/artifact roots,
    backend, seed, …) onto a private ``_exec`` block of the job dict before the
    study starts. Each trial reuses them, overriding only the model params and
    the per-trial artifact directory so trials never clobber one another.
    """
    from .mvd_entrypoint import run_trial_attempt

    exec_ctx = dict(job_manifest.get("_exec") or {})

    def _run(base_job: Dict[str, Any], sampled: Dict[str, Any], trial_number: int):
        trial_job = {k: v for k, v in base_job.items() if k != "_exec"}
        merged_params = dict(base_job.get("model_params") or {})
        merged_params.update(sampled)
        trial_job["model_params"] = merged_params
        # Each trial is a plain single run with its own artifact directory.
        trial_job["mode"] = "run"
        base_name = (
            base_job.get("artifact_dir_name")
            or base_job.get("name")
            or f"{base_job.get('dataset_name', 'dataset')}_"
            f"{base_job.get('model_slug') or base_job.get('model_name', 'model')}"
        )
        trial_job["artifact_dir_name"] = f"{base_name}_trial{trial_number}"
        if base_job.get("output_path"):
            trial_job["output_path"] = (
                f"{base_job['output_path']}_trial{trial_number}"
            )
        # Stamp trial identity into container env so the workspace and
        # optuna-dashboard can correlate container runs to study trials.
        env_extra: Dict[str, str] = dict(trial_job.get("container_env_extra") or {})
        env_extra["MULTIVERSE_TRIAL_NUMBER"] = str(trial_number)
        if exec_ctx.get("study_name"):
            env_extra["MULTIVERSE_STUDY_NAME"] = str(exec_ctx["study_name"])
        trial_job["container_env_extra"] = env_extra
        return run_trial_attempt(
            job=trial_job,
            state_root=exec_ctx["state_root"],
            artifact_root=exec_ctx.get("artifact_root"),
            manifest_hash=exec_ctx.get("manifest_hash", ""),
            seed=exec_ctx.get("seed"),
            backend=exec_ctx.get("backend", "docker"),
            accept_degraded=exec_ctx.get("accept_degraded", False),
        )

    return _run


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


def run_sweep(
    job: Dict[str, Any],
    *,
    state_root: "str | Path | None" = None,
    artifact_root: "str | Path | None" = None,
    manifest_hash: str = "",
    seed: Optional[int] = None,
    backend: str = "docker",
    accept_degraded: bool = False,
    run_trial: Optional[TrialRunner] = None,
) -> Dict[str, Any]:
    """Drive one Optuna study, evaluating each trial through the mvd kernel.

    ``state_root`` (and the other execution knobs) are threaded in by the mvd
    entrypoint and stamped onto the job's private ``_exec`` block so each trial
    can build a kernel attempt. ``run_trial`` overrides the per-trial runner
    (used in tests to avoid launching containers); when omitted the default
    kernel-backed runner is used, which requires ``state_root``.
    """
    optuna = __import__("optuna")

    study_name = str(
        job.get(
            "study_name",
            f"sweep_{job.get('dataset_name', 'dataset')}_"
            f"{job.get('model_slug') or job.get('model_name_orig') or job.get('model_name', 'model')}",
        )
    )
    storage_uri = str(job.get("study_storage", "sqlite:///store/optuna.db"))
    direction = str(job.get("direction", "maximize"))
    n_trials = int(job.get("n_trials", 10))

    if run_trial is None and state_root is None:
        raise ValueError(
            "run_sweep needs 'state_root' to build kernel attempts "
            "(or an explicit 'run_trial' for tests)"
        )

    # Stamp the kernel-execution context onto the job so the default trial
    # runner can rebuild it per trial. Kept under a private key so it never
    # leaks into the executor options.
    job = dict(job)
    job["study_name"] = study_name
    job["_exec"] = {
        "state_root": str(state_root) if state_root is not None else None,
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "manifest_hash": manifest_hash,
        "seed": seed,
        "backend": backend,
        "accept_degraded": accept_degraded,
        "study_name": study_name,
    }

    _apply_wal_mode_to_optuna_db(storage_uri)

    logger.info(
        f"Starting Optuna sweep study='{study_name}' storage='{storage_uri}' "
        f"trials={n_trials} direction={direction} backend={backend}"
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction=direction,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, job, run_trial=run_trial),
        n_trials=n_trials,
    )

    return {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
    }
