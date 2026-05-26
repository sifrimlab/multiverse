"""Local model runner — executes model container scripts as host subprocesses.

This is the Docker-free alternative for ``multiverse-cli run``. It uses the
same YAML run manifest, SQLite registry, and job-spec contract as the Docker
runner, but invokes ``store/models/<slug>/container/run.py`` directly via
``asyncio.create_subprocess_exec`` with ``MVR_*`` env vars pointing at a
local workspace.

Requirements on the host:
- ``mvr-worker`` SDK installed: ``pip install -e sdk/mvr-worker``
- Model-specific Python dependencies installed for each model being run.
- No Docker required.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..logging_utils import get_logger
from ..registry_db import ARTIFACTS_DIR, MODELS_DIR, WORKSPACES_DIR, get_db_connection
from ..tracking import load_run_metrics

logger = get_logger(__name__)


def _flatten_metric_rows(metrics: Dict[str, Any], prefix: str = "") -> list[tuple[str, float | None, str]]:
    rows: list[tuple[str, float | None, str]] = []
    for key, value in metrics.items():
        metric_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_metric_rows(value, metric_name))
        elif isinstance(value, list):
            numeric = []
            for item in value:
                if isinstance(item, bool):
                    numeric.append(float(int(item)))
                elif isinstance(item, (int, float)):
                    numeric.append(float(item))
            if numeric:
                rows.append((metric_name, numeric[-1], "history_summary"))
        elif isinstance(value, bool):
            rows.append((metric_name, float(int(value)), "scalar"))
        elif isinstance(value, (int, float)):
            rows.append((metric_name, float(value), "scalar"))
        elif value is None:
            rows.append((metric_name, None, "scalar"))
    return rows


def _model_entrypoint(model_slug: str) -> Path:
    """Resolve store/models/<slug>/container/run.py for a registered model slug."""
    run_py = Path(MODELS_DIR) / model_slug / "container" / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(
            f"Local entrypoint not found for model '{model_slug}': {run_py}\n"
            "Ensure store/models/<slug>/container/run.py exists and the model slug matches."
        )
    return run_py


def _insert_run_row(job: Dict[str, Any], workspace_dir: Path) -> int | None:
    dataset_id = job.get("dataset_id")
    if dataset_id is None:
        return None
    conn = get_db_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO runs
            (dataset_id, model_slug, model_version, model_name, status, output_path,
             failure_reason, manifest_run_id, params_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                job.get("model_slug") or job.get("model_name_orig"),
                job.get("model_version", "0.0.0"),
                job.get("model_name_orig") or job.get("model_slug"),
                "RUNNING",
                str(workspace_dir),
                None,
                job.get("manifest_run_id"),
                job.get("params_hash"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _update_run_row(
    run_id: int | None,
    *,
    status: str,
    output_path: Path,
    failure_reason: str | None = None,
) -> None:
    if run_id is None:
        return
    conn = get_db_connection()
    try:
        conn.execute(
            """
            UPDATE runs
            SET status = ?, output_path = ?, failure_reason = ?
            WHERE run_id = ?
            """,
            (status, str(output_path), failure_reason, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _persist_run_metrics(run_id: int | None, output_dir: Path) -> None:
    if run_id is None:
        return
    metrics = load_run_metrics(str(output_dir))
    rows = _flatten_metric_rows(metrics)
    conn = get_db_connection()
    try:
        for metric_name, metric_value, metric_kind in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_metrics
                (run_id, metric_name, metric_value, metric_kind)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, metric_name, metric_value, metric_kind),
            )
        conn.commit()
    finally:
        conn.close()


def _write_job_spec(output_dir: Path, job: Dict[str, Any], seed: int) -> Path:
    spec = {
        "seed": seed,
        "hyperparameters": job.get("model_params") or {},
        "dataset_id": job.get("dataset_id"),
        "dataset_name": job.get("dataset_name"),
        "model_name": job.get("model_name_orig") or job.get("model_slug"),
        "run_id": job.get("_local_run_id"),
        "metrics": job.get("metrics", {}),
    }
    path = output_dir / "job_spec.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


async def run_model_local(
    job: Dict[str, Any],
    seed: int,
    status_callback: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Run one model job locally without Docker. Returns ``'success'`` or ``'failed'``."""
    name = job.get("name") or f"{job.get('dataset_name')}_{job.get('model_slug')}"
    model_slug = job.get("model_slug") or job.get("model_name_orig", "")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    job["_local_run_id"] = run_id
    ws = Path(WORKSPACES_DIR) / run_id
    input_dir = ws / "input"
    output_dir = ws / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_db_id = _insert_run_row(job, ws)

    dataset_src = Path(job["dataset_path"]).resolve()
    local_input = input_dir / "data.h5mu"
    if local_input.exists() or local_input.is_symlink():
        local_input.unlink()
    local_input.symlink_to(dataset_src)

    _write_job_spec(output_dir, job, seed)

    try:
        run_py = _model_entrypoint(model_slug)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        _update_run_row(run_db_id, status="FAILED", output_path=ws, failure_reason="LOCAL_ENTRYPOINT_MISSING")
        if status_callback:
            status_callback(name, "Failed: entrypoint not found")
        return "failed"

    env = os.environ.copy()
    env["MVR_INPUT_DATA_PATH"] = str(local_input)
    env["MVR_OUTPUT_DIR"] = str(output_dir)
    env["MVR_JOB_SPEC_PATH"] = str(output_dir / "job_spec.json")

    if status_callback:
        status_callback(name, "Running (local)")
    logger.info("[local] %s -> %s", name, run_py)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(run_py),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    logs_text = ""
    if stdout:
        logs_text += stdout.decode(errors="replace")
    if stderr:
        logs_text += stderr.decode(errors="replace")
    (output_dir / "container.log").write_text(logs_text, encoding="utf-8")

    if proc.returncode != 0:
        logger.error(
            "[local] %s failed (exit %d):\n%s",
            name,
            proc.returncode,
            stderr.decode(errors="replace"),
        )
        _update_run_row(
            run_db_id,
            status="FAILED",
            output_path=ws,
            failure_reason=f"LOCAL_EXIT:{proc.returncode}",
        )
        if status_callback:
            status_callback(name, f"Failed (exit {proc.returncode})")
        return "failed"

    try:
        from ..evaluate import evaluate_single_run

        evaluate_single_run(
            output_dir=str(output_dir),
            dataset_path=str(dataset_src),
            batch_key=job.get("batch_key"),
            label_key=job.get("cell_type_key"),
        )
    except Exception as exc:
        logger.warning("[local] per-job evaluation failed for %s: %s", name, exc)

    declared_output = job.get("output_path")
    dest = Path(declared_output) if declared_output else Path(ARTIFACTS_DIR) / name / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for item in output_dir.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    _update_run_row(run_db_id, status="SUCCESS", output_path=dest)
    _persist_run_metrics(run_db_id, dest)

    if status_callback:
        status_callback(name, "Success")
    logger.info("[local] %s completed -> %s", name, dest)
    return "success"


async def run_jobs_locally(
    jobs: List[Dict[str, Any]],
    seed: int,
    status_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, str]:
    """Run all jobs concurrently on the local host.

    Returns a mapping of job name -> ``'success'`` | ``'failed'``.
    """
    tasks = [run_model_local(job, seed, status_callback) for job in jobs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: Dict[str, str] = {}
    for job, result in zip(jobs, results):
        name = job.get("name") or f"{job.get('dataset_name')}_{job.get('model_slug')}"
        if isinstance(result, Exception):
            logger.error("[local] Unhandled error for %s: %s", name, result)
            out[name] = "failed"
        else:
            out[name] = str(result)
    return out
