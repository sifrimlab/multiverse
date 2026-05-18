import docker
import os
import asyncio
import json
import re
import shutil
import uuid
import psutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import subprocess
import time
from ..logging_utils import get_logger
from ..registry_db import ARTIFACTS_DIR, WORKSPACES_DIR, get_db_connection
from ..tracking import load_run_metrics, log_successful_run_to_mlflow
from ..multiverse_config import get_docker_data_root

logger = get_logger(__name__)



def _emit_runner_event(event: str, **payload: Any) -> None:
    try:
        print(json.dumps({"event": event, **payload}, sort_keys=True), file=__import__("sys").stderr, flush=True)
    except Exception:
        pass


def _classify_docker_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if isinstance(exc, PermissionError) or "permission" in text or "denied" in text:
        return "daemon_permission_denied"
    if "pull" in text or "not found" in text:
        return "image_pull_failed"
    if "connection" in name or "connection" in text or "socket" in text or "daemon" in text:
        return "daemon_offline"
    return "docker_error"


def get_docker_client():
    try:
        return docker.from_env()
    except Exception as exc:
        kind = _classify_docker_exception(exc)
        _emit_runner_event("error", kind=kind, message=str(exc))
        raise


def flatten_metric_rows(metrics: dict[str, Any], prefix: str = "") -> list[tuple[str, float | None, str]]:
    rows: list[tuple[str, float | None, str]] = []
    for key, value in metrics.items():
        metric_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_metric_rows(value, metric_name))
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


def _docker_gpu_available(client) -> bool:
    """Return True if the Docker daemon has an NVIDIA runtime registered.

    Importing docker.types.DeviceRequest succeeds even without the NVIDIA
    Container Toolkit, so the import-level try/except used elsewhere is not
    sufficient — the actual failure surfaces only when the container starts.
    Checking the daemon's runtime list catches this before any container launch.
    """
    try:
        return "nvidia" in client.info().get("Runtimes", {})
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Docker data-root configuration
# ---------------------------------------------------------------------------

def ensure_docker_data_root() -> None:
    """Sync ~/.config/docker/daemon.json with the configured docker_data_root.

    If the stored data-root differs from what is currently in daemon.json (or
    is absent), the file is updated and the Docker systemd user service is
    restarted.  Waits up to 10 s for the daemon to become responsive again.
    """
    desired = get_docker_data_root()
    daemon_json_path = Path.home() / ".config" / "docker" / "daemon.json"

    # Read existing daemon.json (or start with an empty dict).
    daemon_json_path.parent.mkdir(parents=True, exist_ok=True)
    if daemon_json_path.exists():
        with open(daemon_json_path) as fh:
            try:
                daemon_cfg = json.load(fh)
            except json.JSONDecodeError:
                daemon_cfg = {}
    else:
        daemon_cfg = {}

    current = daemon_cfg.get("data-root", "")
    if current == desired:
        logger.debug("docker data-root already set to %s — no restart needed", desired)
        return

    logger.info("Updating docker data-root: %r -> %r", current or "<unset>", desired)
    daemon_cfg["data-root"] = desired
    with open(daemon_json_path, "w") as fh:
        json.dump(daemon_cfg, fh, indent=2)

    # Restart the user Docker daemon.
    subprocess.run(
        ["systemctl", "--user", "restart", "docker"],
        check=True,
    )
    logger.info("docker systemd user service restarted; waiting for daemon …")

    # Poll docker info for up to 10 s.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            logger.info("Docker daemon is responsive with data-root=%s", desired)
            return
        time.sleep(0.5)

    logger.warning("Docker daemon did not become responsive within 10 s after restart")

# ---------------------------------------------------------------------------
# Single-writer DB actor
#
# All SQLite writes from inside parallel worker coroutines flow through this
# asyncio.Queue.  A single background task (db_writer_task) owns the only
# write connection, so concurrent workers never contend for the DB lock.
#
# Read-path callers (planner, GUI) open their own short-lived connections;
# WAL mode in get_db_connection() keeps those reads non-blocking.
# ---------------------------------------------------------------------------

@dataclass
class _DbWriteOp:
    sql: str
    params: tuple
    # Resolved with cursor.lastrowid on success; set_exception on DB error.
    result_future: Optional[asyncio.Future] = field(default=None, compare=False)


_db_write_queue: Optional[asyncio.Queue] = None
_db_writer_task_handle: Optional[asyncio.Task] = None


async def db_writer_task(queue: asyncio.Queue) -> None:
    """Background coroutine: sole owner of the SQLite write connection."""
    conn = get_db_connection()
    consecutive_failures = 0
    logger.info("db_writer_task started — single-writer SQLite actor is live")
    try:
        while True:
            op = await queue.get()
            if op is None:
                queue.task_done()
                break
            try:
                cursor = conn.execute(op.sql, op.params)
                conn.commit()
                consecutive_failures = 0
                if op.result_future is not None and not op.result_future.done():
                    op.result_future.set_result(cursor.lastrowid)
            except Exception as exc:
                consecutive_failures += 1
                logger.error(
                    "db_writer_task error (%s consecutive): %s | sql=%s | params=%s",
                    consecutive_failures,
                    exc,
                    op.sql,
                    op.params,
                )
                if op.result_future is not None and not op.result_future.done():
                    op.result_future.set_exception(exc)
                if consecutive_failures >= 3:
                    _emit_runner_event(
                        "error",
                        kind="db_writer_failed",
                        message="DB writer hit 3 consecutive write failures; restarting",
                    )
                    raise RuntimeError("DB writer hit 3 consecutive write failures") from exc
            finally:
                queue.task_done()
    finally:
        conn.close()
        logger.info("db_writer_task stopped — write connection closed")


async def db_writer_supervisor(queue: asyncio.Queue) -> None:
    """Restart the DB writer after crashes, escalating after repeated restart failures."""
    restart_failures = 0
    while True:
        try:
            await db_writer_task(queue)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            restart_failures += 1
            logger.error("db_writer_task crashed; restart attempt %s/3: %s", restart_failures, exc)
            if restart_failures >= 3:
                _emit_runner_event(
                    "error",
                    kind="db_writer_supervisor_failed",
                    message=str(exc),
                )
                raise
            await asyncio.sleep(0.2)


def mark_active_runs_failed_direct(reason: str = "CANCELLED") -> int:
    """One-shot fallback write for shutdown when the async writer may be unavailable."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            UPDATE runs
            SET status = 'FAILED', failure_reason = ?
            WHERE status IN ('RUNNING', 'PROMOTING')
            """,
            (reason,),
        )
        conn.commit()
        return int(cursor.rowcount or 0)
    finally:
        conn.close()


def start_db_writer() -> asyncio.Task:
    """Create the write queue and launch the db_writer_task.

    Must be called from within a running event loop (e.g. inside
    ``run_workflow_async``).  Returns the Task handle.
    """
    global _db_write_queue, _db_writer_task_handle
    _db_write_queue = asyncio.Queue(maxsize=256)
    _db_writer_task_handle = asyncio.create_task(
        db_writer_supervisor(_db_write_queue), name="db_writer"
    )
    return _db_writer_task_handle


async def stop_db_writer() -> None:
    """Drain the queue, send the shutdown sentinel, await task completion."""
    global _db_write_queue, _db_writer_task_handle
    if _db_write_queue is not None:
        await _db_write_queue.put(None)   # sentinel triggers exit
        await _db_write_queue.join()      # block until all items processed
    if _db_writer_task_handle is not None:
        try:
            await _db_writer_task_handle
        except Exception:
            pass
    _db_write_queue = None
    _db_writer_task_handle = None


async def _db_write(sql: str, params: tuple) -> int:
    """Enqueue one write and await its lastrowid result.

    This is the *only* function worker coroutines call for DB writes.
    It never opens a SQLite connection directly.

    Raises RuntimeError if start_db_writer() has not been called.
    """
    if _db_write_queue is None:
        raise RuntimeError(
            "_db_write() called before start_db_writer(). "
            "Call start_db_writer() at the start of run_workflow_async()."
        )
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    await _db_write_queue.put(_DbWriteOp(sql=sql, params=params, result_future=fut))
    return await fut

_MEM_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([gGmMkK]?)b?$")


def _parse_mem_gb(mem_str: str) -> float:
    """Convert Docker mem_limit strings (e.g. '16g', '8192m') to GiB float."""
    if isinstance(mem_str, (int, float)):
        return float(mem_str)
    m = _MEM_PATTERN.match(str(mem_str).strip())
    if not m:
        raise ValueError(f"Cannot parse mem_limit: {mem_str!r}")
    value, unit = float(m.group(1)), m.group(2).lower()
    if unit in ("", "g"):
        return value
    if unit == "m":
        return value / 1024
    if unit == "k":
        return value / (1024 * 1024)
    raise ValueError(f"Unrecognised mem unit in {mem_str!r}")


class ResourcePool:
    """Committed-memory admission ledger backed by asyncio.Condition.

    Suspends tasks that would exceed available RAM; releases capacity in
    finally blocks so waiting tasks are always notified.
    """

    def __init__(self, total_gb: float) -> None:
        self.total_gb = total_gb
        self._available_gb = total_gb
        self._condition = asyncio.Condition()

    async def acquire(self, gb: float) -> None:
        if gb > self.total_gb:
            logger.critical(
                "Job requests %.1f GiB but host capacity is only %.1f GiB — "
                "marking job FAILED: INSUFFICIENT_RESOURCES",
                gb,
                self.total_gb,
            )
            raise InsufficientResourcesError(
                f"Job requires {gb:.1f} GiB; host total is {self.total_gb:.1f} GiB"
            )
        async with self._condition:
            while self._available_gb < gb:
                logger.info(
                    "Waiting for %.1f GiB (%.1f GiB available of %.1f GiB total)",
                    gb,
                    self._available_gb,
                    self.total_gb,
                )
                await self._condition.wait()
            self._available_gb -= gb
            logger.info(
                "Admitted %.1f GiB — %.1f GiB remaining",
                gb,
                self._available_gb,
            )

    def release(self, gb: float) -> None:
        async def _notify():
            async with self._condition:
                self._available_gb += gb
                logger.info(
                    "Released %.1f GiB — %.1f GiB now available",
                    gb,
                    self._available_gb,
                )
                self._condition.notify_all()

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_notify())
        except RuntimeError:
            pass


class InsufficientResourcesError(RuntimeError):
    """Raised when a single job exceeds total host RAM capacity."""


CONTAINER_INPUT_DATA_PATH = "/input/data.h5mu"
CONTAINER_OUTPUT_DIR = "/output"

DEFAULT_MODEL_IMAGES = {
    "pca": "multiverse-pca",
    "mofa": "multiverse-mofa",
    "multivi": "multiverse-multivi",
    "mowgli": "multiverse-mowgli",
    "cobolt": "multiverse-cobolt",
    "totalvi": "multiverse-totalvi",
}


def ensure_image_prepared(tag: str, status_callback: callable = None) -> bool:
    """Ensure an image is available locally: local -> build from manifest -> pull."""
    client = get_docker_client()
    try:
        if status_callback:
            status_callback(tag, "Building/Pulling")

        # 1) Already local
        try:
            client.images.get(tag)
            logger.info(f"Using local image: {tag}")
            if status_callback:
                status_callback(tag, "Ready")
            return True
        except Exception:
            pass

        # 2) Build locally from registered manifest
        built_locally = False
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT manifest_path
                FROM models
                WHERE docker_image = ? AND status = 'ACTIVE'
                ORDER BY version DESC
                LIMIT 1
                """,
                (tag,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                manifest_path = row[0]
                logger.info(f"Building local image {tag} from manifest {manifest_path}")
                from ..models_ingest import load_model_manifest
                from ..builder import build_local_model

                manifest = load_model_manifest(manifest_path)
                build_local_model(manifest)
                built_locally = True
        finally:
            if conn is not None:
                conn.close()

        if not built_locally:
            # 3) Fallback pull
            logger.info(f"Pulling remote image: {tag}")
            try:
                client.images.pull(tag)
            except Exception as exc:
                _emit_runner_event("error", kind="image_pull_failed", image=tag, message=str(exc))
                raise
            logger.info(f"Successfully pulled image: {tag}")

        if status_callback:
            status_callback(tag, "Ready")
        return True
    except Exception as exc:
        logger.error(f"Failed to prepare image {tag}: {exc}")
        if status_callback:
            status_callback(tag, "Failed")
        if _classify_docker_exception(exc) != "image_pull_failed":
            _emit_runner_event("error", kind=_classify_docker_exception(exc), image=tag, message=str(exc))
        raise


def _extract_hyperparameters(job: dict) -> dict:
    for key in ("hyperparameters", "params", "model_params"):
        value = job.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _write_job_spec(output_dir: str, job: dict, seed: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    job_spec = {
        "seed": seed,
        "dataset_id": job.get("dataset_id"),
        "dataset_name": job.get("dataset_name") or job.get("dataset_slug"),
        "model_name": job.get("model_name_orig") or job.get("model_name") or job.get("name"),
        "hyperparameters": _extract_hyperparameters(job),
        "run_settings": job.get("run_settings", {}),
        "metrics": job.get("metrics", {}),
    }
    job_spec_path = os.path.join(output_dir, "job_spec.json")
    with open(job_spec_path, "w", encoding="utf-8") as fp:
        json.dump(job_spec, fp, indent=2, sort_keys=True)
    return job_spec_path


def _standard_volumes(dataset_path: str, output_dir: str) -> dict:
    return {
        os.path.abspath(dataset_path): {"bind": CONTAINER_INPUT_DATA_PATH, "mode": "ro"},
        os.path.abspath(output_dir): {"bind": CONTAINER_OUTPUT_DIR, "mode": "rw"},
    }


def _safe_name(value: str, fallback: str) -> str:
    if not value:
        return fallback
    allowed = [c if c.isalnum() or c in ("-", "_") else "_" for c in str(value)]
    cleaned = "".join(allowed).strip("_")
    return cleaned or fallback


def _build_artifact_destination(job: dict, run_id: str) -> str:
    experiment_name = _safe_name(job.get("experiment_name", "default_experiment"), "default_experiment")
    dataset_name = _safe_name(job.get("dataset_name") or job.get("dataset_slug") or "dataset", "dataset")
    model_name = _safe_name(job.get("model_name_orig") or job.get("model_name") or job.get("name"), "model")
    return os.path.join(ARTIFACTS_DIR, experiment_name, dataset_name, model_name, run_id)


def _persist_run_status(
    dataset_id: int,
    model_slug: str,
    model_version: str,
    model_name: str,
    status: str,
    output_path: str,
    failure_reason: str | None = None,
) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO runs
        (dataset_id, model_slug, model_version, model_name, status, output_path, failure_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (dataset_id, model_slug, model_version, model_name, status, output_path, failure_reason),
    )
    conn.commit()
    conn.close()


def _run_and_promote_sync(
    client,
    run_kwargs: dict,
    workspace_dir: str,
    final_artifact_dir: str,
    job_context: dict | None = None,
):
    """Synchronous promotion used only by legacy single-container callers.

    The parallel orchestration path uses the async ``run_and_promote`` below,
    which routes all DB writes through the single-writer queue.  This sync
    variant is kept exclusively for ``run_model_container`` and
    ``run_job_container_sync`` — legacy code paths that do not participate in
    the async event loop.
    """
    job_context = job_context or {}
    os.makedirs(os.path.dirname(final_artifact_dir), exist_ok=True)
    container = client.containers.run(**run_kwargs)
    wait_result = container.wait()
    exit_code = wait_result.get("StatusCode", 1)

    logs_raw = container.logs(stdout=True, stderr=True)
    logs_text = logs_raw.decode("utf-8", errors="replace")
    Path(os.path.join(workspace_dir, "container.log")).write_text(logs_text, encoding="utf-8")

    status = "FAILED"
    final_output_path = workspace_dir
    if exit_code == 0:
        job_spec_payload: dict = {}
        job_spec_path = os.path.join(workspace_dir, "job_spec.json")
        if os.path.exists(job_spec_path):
            try:
                with open(job_spec_path, "r", encoding="utf-8") as fp:
                    job_spec_payload = json.load(fp)
            except Exception as exc:
                logger.warning("Failed reading job_spec.json for MLflow: %s", exc)
        metrics_payload = load_run_metrics(workspace_dir)

        shutil.move(workspace_dir, final_artifact_dir)
        status = "SUCCESS"
        final_output_path = final_artifact_dir
        try:
            log_successful_run_to_mlflow(
                job_context=job_context,
                job_spec=job_spec_payload,
                metrics=metrics_payload,
                artifacts_dir=final_artifact_dir,
            )
        except Exception as exc:
            logger.warning("MLflow tracking failed; continuing. (%s)", exc)
    else:
        logger.warning(
            "Container failed (exit code %s). Workspace preserved at %s",
            exit_code,
            workspace_dir,
        )

    container.remove()
    return exit_code, status, final_output_path, logs_text




async def _persist_run_metrics(run_id: int, metrics_payload: dict[str, Any]) -> None:
    rows = flatten_metric_rows(metrics_payload)
    for metric_name, metric_value, metric_kind in rows:
        await _db_write(
            """
            INSERT OR REPLACE INTO run_metrics
            (run_id, metric_name, metric_value, metric_kind)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, metric_name, metric_value, metric_kind),
        )

async def _supervise_container(
    container,
    run_id: int,
    workspace_dir: str,
    final_artifact_dir: str,
    job_context: dict | None = None,
) -> tuple:
    """Wait for a launched container and finalize the persisted run row."""
    loop = asyncio.get_running_loop()
    job_context = job_context or {}

    try:
        wait_result = await loop.run_in_executor(None, container.wait)
        exit_code = wait_result.get("StatusCode", 1)
        logs_raw = await loop.run_in_executor(
            None, lambda: container.logs(stdout=True, stderr=True)
        )
        logs_text = logs_raw.decode("utf-8", errors="replace")
        await loop.run_in_executor(
            None,
            lambda: Path(os.path.join(workspace_dir, "container.log")).write_text(
                logs_text, encoding="utf-8"
            ),
        )

        if exit_code != 0:
            logger.warning(
                "Container failed (exit %s). Workspace preserved at %s", exit_code, workspace_dir
            )
            await _db_write(
                """
                UPDATE runs
                SET status = ?, output_path = ?, failure_reason = ?
                WHERE run_id = ?
                """,
                ("FAILED", workspace_dir, f"CONTAINER_EXIT:{exit_code}", run_id),
            )
            await asyncio.shield(loop.run_in_executor(None, container.remove))
            return exit_code, "FAILED", workspace_dir, logs_text

        await _db_write(
            "UPDATE runs SET status = ?, output_path = ?, failure_reason = NULL WHERE run_id = ?",
            ("PROMOTING", final_artifact_dir, run_id),
        )

        try:
            if os.path.exists(final_artifact_dir):
                await loop.run_in_executor(None, lambda: shutil.rmtree(final_artifact_dir))
            await loop.run_in_executor(None, lambda: shutil.move(workspace_dir, final_artifact_dir))
            await loop.run_in_executor(
                None,
                lambda: Path(final_artifact_dir, ".promotion_complete").touch(),
            )
        except Exception as exc:
            logger.error("Promotion move failed: %s", exc)
            await asyncio.shield(loop.run_in_executor(None, container.remove))
            await _db_write(
                """
                UPDATE runs
                SET status = ?, output_path = ?, failure_reason = ?
                WHERE run_id = ?
                """,
                ("FAILED", workspace_dir, "INCOMPLETE_PROMOTION", run_id),
            )
            return exit_code, "FAILED", workspace_dir, logs_text

        await _db_write(
            "UPDATE runs SET status = ?, output_path = ?, failure_reason = NULL WHERE run_id = ?",
            ("SUCCESS", final_artifact_dir, run_id),
        )

        await asyncio.shield(loop.run_in_executor(None, container.remove))

        try:
            dataset_path = job_context.get("dataset_path")
            batch_key = job_context.get("batch_key") or "batch"
            label_key = job_context.get("cell_type_key")
            if dataset_path and os.path.exists(dataset_path):
                from ..evaluate import evaluate_single_run
                await loop.run_in_executor(
                    None,
                    lambda: evaluate_single_run(
                        output_dir=final_artifact_dir,
                        dataset_path=dataset_path,
                        batch_key=batch_key,
                        label_key=label_key,
                    ),
                )
        except Exception as exc:
            logger.warning("Per-job evaluation failed for run %s: %s", run_id, exc)

        metrics_payload = load_run_metrics(final_artifact_dir)
        try:
            await _persist_run_metrics(run_id, metrics_payload)
        except Exception as exc:
            logger.warning("run_metrics persistence failed for run %s: %s", run_id, exc)

        try:
            job_spec_payload: dict = {}
            job_spec_path = os.path.join(final_artifact_dir, "job_spec.json")
            if os.path.exists(job_spec_path):
                with open(job_spec_path, "r", encoding="utf-8") as fp:
                    job_spec_payload = json.load(fp)
            await loop.run_in_executor(
                None,
                lambda: log_successful_run_to_mlflow(
                    job_context=job_context,
                    job_spec=job_spec_payload,
                    metrics=metrics_payload,
                    artifacts_dir=final_artifact_dir,
                ),
            )
        except Exception as exc:
            logger.warning("MLflow tracking failed; continuing pipeline promotion. (%s)", exc)

        return exit_code, "SUCCESS", final_artifact_dir, logs_text
    except asyncio.CancelledError:
        await asyncio.shield(loop.run_in_executor(None, container.stop))
        await asyncio.shield(loop.run_in_executor(None, container.remove))
        await _db_write(
            """
            UPDATE runs
            SET status = ?, output_path = ?, failure_reason = ?
            WHERE run_id = ?
            """,
            ("FAILED", workspace_dir, "CANCELLED", run_id),
        )
        raise
    except Exception as exc:
        logger.error("Container supervision failed: %s", exc)
        await _db_write(
            """
            UPDATE runs
            SET status = ?, output_path = ?, failure_reason = ?
            WHERE run_id = ?
            """,
            ("FAILED", workspace_dir, f"SUPERVISION_ERROR:{type(exc).__name__}", run_id),
        )
        raise


async def run_and_promote(
    client,
    run_kwargs: dict,
    workspace_dir: str,
    final_artifact_dir: str,
    job_context: dict | None = None,
) -> tuple:
    """Launch a container and drive the persisted run FSM.

    The row is created before Docker launch as RUNNING/container_id=NULL, then
    updated with the Docker container ID immediately after launch. Successful
    containers transition through PROMOTING and write .promotion_complete in the
    final artifact directory before the terminal SUCCESS update.
    """
    loop = asyncio.get_running_loop()
    job_context = job_context or {}
    dataset_id = job_context.get("dataset_id")
    model_slug = job_context.get("model_slug") or job_context.get("model_name_orig", "")
    model_version = job_context.get("model_version", "0.0.0")
    model_name = job_context.get("model_name_orig", model_slug)

    os.makedirs(os.path.dirname(final_artifact_dir), exist_ok=True)

    db_run_id: Optional[int] = None
    if dataset_id is not None:
        db_run_id = await _db_write(
            """
            INSERT INTO runs
            (dataset_id, model_slug, model_version, model_name, status, output_path, container_id, failure_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, model_slug, model_version, model_name, "RUNNING", workspace_dir, None, None),
        )

    labels = dict(run_kwargs.get("labels") or {})
    labels.update({
        "multiverse.run.workspace_dir": workspace_dir,
        "multiverse.run.final_artifact_dir": final_artifact_dir,
    })
    run_kwargs = {**run_kwargs, "labels": labels}

    try:
        container = await loop.run_in_executor(
            None, lambda: client.containers.run(**run_kwargs)
        )
    except Exception as exc:
        if db_run_id is not None:
            await _db_write(
                "UPDATE runs SET status = ?, failure_reason = ? WHERE run_id = ?",
                ("FAILED", f"LAUNCH_ERROR:{type(exc).__name__}", db_run_id),
            )
        raise

    if db_run_id is None:
        # Legacy callers without registry context still get the same promotion semantics,
        # backed by a temporary row so crash recovery can reason about the container.
        db_run_id = await _db_write(
            """
            INSERT INTO runs
            (dataset_id, model_slug, model_version, model_name, status, output_path, container_id, failure_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (None, model_slug, model_version, model_name, "RUNNING", workspace_dir, container.id, None),
        )
    else:
        await _db_write(
            "UPDATE runs SET container_id = ? WHERE run_id = ?",
            (container.id, db_run_id),
        )

    return await _supervise_container(
        container,
        db_run_id,
        workspace_dir,
        final_artifact_dir,
        job_context,
    )


async def build_images_concurrently(
    image_tags: list, status_callback: callable = None, max_concurrent: int = 3
):
    """Ensures all required Docker images for models are prepared concurrently.

    Args:
        image_tags (list): A list of Docker image tags to pull or build.
        status_callback (callable, optional): A function called with (image_tag, status)
            to update the progress in real-time.
        max_concurrent (int): Maximum number of images to build at the same time.
            Conda/pip builds can consume several GB each; limiting concurrency
            prevents "no space left on device" errors on shared hosts.

    Raises:
        RuntimeError: If one or more images fail to pull or build.
    """
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def prepare_image(tag):
        async with semaphore:
            return await loop.run_in_executor(
                None, ensure_image_prepared, tag, status_callback
            )

    tasks = [prepare_image(tag) for tag in image_tags]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [res for res in results if isinstance(res, Exception)]
    if failures:
        logger.error(f"Failed to prepare {len(failures)} images.")
        raise RuntimeError(f"Failed to prepare some Docker images: {failures}")


async def run_models_concurrently(
    models_info: list,
    data_path: str,
    seed: int,
    output_dir: str,
    status_callback: callable = None,
    mem_limit: str = "16g",
):
    """Executes eligible models in parallel using isolated Docker containers.

    Each model is run in its own container with the input data mounted as read-only.
    The process captures exit codes and handles failures without stopping other models.

    Args:
        models_info (list of dict): A list of dictionaries containing 'name' and 'image' keys.
        data_path (str): The host path to the input dataset.
        seed (int): The random seed to inject as an environment variable.
        output_dir (str): The host path where results will be stored.
        status_callback (callable, optional): A function called with (model_name, status)
            to update the execution progress.

    Returns:
        dict: A dictionary mapping model names to their final status ("success" or "failed").
    """
    client = get_docker_client()
    loop = asyncio.get_running_loop()

    async def run_single_model(model_name, image_tag):
        if status_callback:
            status_callback(model_name, "Starting")
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        model_output_dir = os.path.join(WORKSPACES_DIR, run_id)
        final_artifact_dir = os.path.join(os.path.abspath(output_dir), model_name, run_id)
        _write_job_spec(
            output_dir=model_output_dir,
            job={
                "model_name": model_name,
                "dataset_name": os.path.splitext(os.path.basename(data_path))[0],
            },
            seed=seed,
        )

        run_kwargs = {
            "image": image_tag,
            "volumes": _standard_volumes(data_path, model_output_dir),
            "detach": True,
            "remove": False,
            "mem_limit": mem_limit,
        }

        try:
            logger.info(f"Starting container for model: {model_name} using image: {image_tag}")
            exit_code, status, promoted_path, logs_text = await run_and_promote(
                client,
                run_kwargs,
                model_output_dir,
                final_artifact_dir,
                {
                    "dataset_name": os.path.splitext(os.path.basename(data_path))[0],
                    "model_name_orig": model_name,
                    "experiment_name": "default_experiment",
                },
            )

            if status_callback:
                status_callback(model_name, "Running")
            if exit_code == 0:
                logger.info(f"Model {model_name} completed successfully. Promoted to {promoted_path}")
                if status_callback:
                    status_callback(model_name, "Success")
            else:
                logger.error(f"Model {model_name} failed with exit code {exit_code}. Logs: {logs_text[-500:]}")
                if status_callback:
                    status_callback(model_name, f"Failed ({exit_code})")

            return model_name, exit_code == 0
        except Exception as e:
            logger.error(f"Error running model {model_name}: {e}")
            if status_callback:
                status_callback(model_name, "Error")
            return model_name, False

    tasks = [run_single_model(m["name"], m["image"]) for m in models_info]
    results = await asyncio.gather(*tasks)

    summary = {name: "success" if success else "failed" for name, success in results}
    return summary


async def run_jobs_concurrently(
    jobs_info: list,
    seed: int,
    status_callback: callable = None,
    mem_limit: str = "16g",
    use_gpu: bool = True,
    host_ram_gb: float | None = None,
):
    """Executes jobs in parallel with a committed-memory admission ledger.

    Before each container starts, the scheduler acquires the job's mem_limit
    from a ResourcePool so that total committed memory never exceeds host RAM.
    Capacity is released in a finally block regardless of container outcome.

    Args:
        jobs_info (list of dict): keys 'name', 'image', 'dataset_path', 'output_path',
            'dataset_id', 'model_name_orig'. Each dict may include 'mem_limit' to
            override the function-level default.
        host_ram_gb: Override total host RAM (GiB) used by the ResourcePool.
            Defaults to psutil.virtual_memory().total.
    """
    if host_ram_gb is None:
        host_ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    pool = ResourcePool(total_gb=host_ram_gb)
    logger.info("ResourcePool initialised: %.1f GiB total host RAM", host_ram_gb)

    client = get_docker_client()
    loop = asyncio.get_running_loop()

    if use_gpu and not _docker_gpu_available(client):
        logger.warning(
            "GPU requested but Docker NVIDIA runtime is not available "
            "(nvidia-container-toolkit may not be configured). "
            "All jobs will run on CPU."
        )
        use_gpu = False

    async def run_single_job(job):
        model_display_name = job["name"]
        job_mem_limit = job.get("mem_limit", mem_limit)
        job_gb = _parse_mem_gb(job_mem_limit)

        # --- Admission gate ---
        try:
            await pool.acquire(job_gb)
        except InsufficientResourcesError as exc:
            logger.critical("Job %s cannot be admitted: %s", model_display_name, exc)
            if status_callback:
                status_callback(model_display_name, "Failed (INSUFFICIENT_RESOURCES)")
            _persist_run_status(
                job["dataset_id"],
                job.get("model_slug", job["model_name_orig"]),
                job.get("model_version", "0.0.0"),
                job["model_name_orig"],
                "FAILED",
                "",
                "INSUFFICIENT_RESOURCES",
            )
            return model_display_name, False

        if status_callback:
            status_callback(model_display_name, "Starting")
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        workspace_dir = os.path.join(WORKSPACES_DIR, run_id)
        final_artifact_dir = _build_artifact_destination(job, run_id)
        _write_job_spec(output_dir=workspace_dir, job=job, seed=seed)

        run_kwargs = {
            "image": job["image"],
            "volumes": _standard_volumes(job["dataset_path"], workspace_dir),
            "detach": True,
            "remove": False,
            "mem_limit": job_mem_limit,  # kernel hard-enforcer via cgroups
        }

        if use_gpu:
            try:
                from docker.types import DeviceRequest
                run_kwargs["device_requests"] = [
                    DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]
            except Exception as e:
                logger.warning(f"GPU support not available for {model_display_name}, running on CPU. ({e})")

        try:
            logger.info(f"Starting container for {model_display_name} using image: {job['image']}")
            if status_callback:
                status_callback(model_display_name, "Running")

            # run_and_promote is async and owns all DB writes via the queue.
            exit_code, status, final_output_path, _ = await run_and_promote(
                client, run_kwargs, workspace_dir, final_artifact_dir, job
            )

            if status == "SUCCESS":
                logger.info(f"Job {model_display_name} completed successfully.")
                if status_callback:
                    status_callback(model_display_name, "Success")
            else:
                logger.error(f"Job {model_display_name} failed with exit code {exit_code}.")
                if status_callback:
                    status_callback(model_display_name, f"Failed ({exit_code})")

            return model_display_name, status == "SUCCESS"
        except Exception as e:
            logger.error(f"Error running job {model_display_name}: {e}")
            if status_callback:
                status_callback(model_display_name, "Error")
            # Fallback: queue a FAILED write directly so the planner can skip
            # this job on the next run even if run_and_promote raised before
            # writing anything.
            if _db_write_queue is not None:
                try:
                    await _db_write(
                        "INSERT INTO runs "
                        "(dataset_id, model_slug, model_version, model_name, status, output_path) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            job["dataset_id"],
                            job.get("model_slug", job["model_name_orig"]),
                            job.get("model_version", "0.0.0"),
                            job["model_name_orig"],
                            "FAILED",
                            workspace_dir,
                        ),
                    )
                except Exception:
                    pass
            return model_display_name, False
        finally:
            pool.release(job_gb)

    tasks = [run_single_job(job) for job in jobs_info]
    results = await asyncio.gather(*tasks)

    summary = {name: "success" if success else "failed" for name, success in results}
    return summary


def run_model_container(
    model_name: str,
    input_dir: str,
    output_dir: str,
    seed: int = 42,
    extra_args: list = None,
    use_gpu: bool = True,
    mem_limit: str = "16g",
):
    """Runs a single model container synchronously.

    Args:
        model_name (str): The name of the model to run.
        input_dir (str): The host path for input data.
        output_dir (str): The host path for output results.
        extra_args (list, optional): Additional command-line arguments for the container.
        use_gpu (bool): Whether to enable GPU support. Defaults to True.

    Raises:
        ValueError: If the model name is not recognized.
    """
    client = get_docker_client()
    if use_gpu and not _docker_gpu_available(client):
        logger.warning("GPU requested but Docker NVIDIA runtime is not available — running on CPU.")
        use_gpu = False

    image = None
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT docker_image FROM models WHERE slug = ? AND status = 'ACTIVE' ORDER BY version DESC LIMIT 1",
            (model_name,),
        )
        row = cursor.fetchone()
        if row:
            image = row[0]
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()
    if image is None:
        image = DEFAULT_MODEL_IMAGES.get(model_name)
    if image is None:
        raise ValueError(f"Unknown model name: {model_name}")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    workspace_dir = os.path.join(WORKSPACES_DIR, run_id)
    final_artifact_dir = os.path.join(os.path.abspath(output_dir), model_name, run_id)

    run_kwargs = {
        "image": image,
        "command": extra_args or [],
        "volumes": _standard_volumes(input_dir, workspace_dir),
        "detach": True,
        "remove": False,
        "mem_limit": mem_limit,
    }
    _write_job_spec(
        output_dir=workspace_dir,
        job={"model_name": model_name, "dataset_name": os.path.splitext(os.path.basename(input_dir))[0]},
        seed=seed,
    )

    # Add GPU support if requested and available
    if use_gpu:
        try:
            # Docker SDK only allows `device_requests` for GPU scheduling
            from docker.types import DeviceRequest

            run_kwargs["device_requests"] = [
                DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        except Exception as e:
            logger.warning(f"GPU support not available, running on CPU. ({e})")

    exit_code, _, final_output_path, logs_text = _run_and_promote_sync(
        client,
        run_kwargs,
        workspace_dir,
        final_artifact_dir,
        {
            "dataset_name": os.path.splitext(os.path.basename(input_dir))[0],
            "model_name_orig": model_name,
            "experiment_name": os.path.basename(os.path.normpath(output_dir)) if output_dir else "default_experiment",
        },
    )
    for log_line in logs_text.splitlines():
        logger.info(log_line)
    if exit_code != 0:
        raise RuntimeError(
            f"Model container {model_name} failed with exit code {exit_code}. "
            f"Workspace retained at {final_output_path}"
        )
    return final_output_path


def run_job_container_sync(
    job: dict,
    seed: int,
    use_gpu: bool = True,
    mem_limit: str = "16g",
):
    """Run one registry/planned job synchronously with workspace promotion."""
    client = get_docker_client()
    if use_gpu and not _docker_gpu_available(client):
        logger.warning("GPU requested but Docker NVIDIA runtime is not available — running on CPU.")
        use_gpu = False
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    workspace_dir = os.path.join(WORKSPACES_DIR, run_id)
    final_artifact_dir = _build_artifact_destination(job, run_id)
    _write_job_spec(output_dir=workspace_dir, job=job, seed=seed)

    run_kwargs = {
        "image": job["image"],
        "volumes": _standard_volumes(job["dataset_path"], workspace_dir),
        "detach": True,
        "remove": False,
        "mem_limit": mem_limit,
    }
    if use_gpu:
        try:
            from docker.types import DeviceRequest

            run_kwargs["device_requests"] = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
        except Exception as e:
            logger.warning(f"GPU support not available for {job.get('name', 'job')}, running on CPU. ({e})")

    exit_code, status, final_output_path, logs_text = _run_and_promote_sync(
        client,
        run_kwargs,
        workspace_dir,
        final_artifact_dir,
        job,
    )
    return {
        "exit_code": exit_code,
        "status": status,
        "output_path": final_output_path,
        "logs": logs_text,
    }


def run_evaluation_container(
    input_dir: str, output_dir: str, extra_args: list = None, use_gpu: bool = True
):
    """Runs the evaluation metrics container synchronously.

    Args:
        input_dir (str): The host path for input data.
        output_dir (str): The host path for output results.
        extra_args (list, optional): Additional command-line arguments for the container.
        use_gpu (bool): Whether to enable GPU support. Defaults to True.
    """
    client = get_docker_client()
    if use_gpu and not _docker_gpu_available(client):
        logger.warning("GPU requested but Docker NVIDIA runtime is not available — running on CPU.")
        use_gpu = False

    image = "multiverse-evaluate"

    run_kwargs = {
        "image": image,
        "command": extra_args or [],
        "volumes": {
            os.path.abspath(input_dir): {"bind": "/data/input", "mode": "ro"},
            os.path.abspath(output_dir): {"bind": "/data/outputs", "mode": "rw"},
        },
        "detach": True,
        "remove": True,
    }

    # Add GPU support if requested and available
    if use_gpu:
        try:
            # Docker SDK only allows `device_requests` for GPU scheduling
            from docker.types import DeviceRequest

            run_kwargs["device_requests"] = [
                DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        except Exception as e:
            logger.warning(f"GPU support not available, running on CPU. ({e})")

    container = client.containers.run(**run_kwargs)

    for log in container.logs(stream=True):
        logger.info(log.decode().strip())
