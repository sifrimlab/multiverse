"""Process-local mvd controller for GUI/client cutover.

The Streamlit GUI cannot ``await`` kernel calls directly and should not own
long-running run tasks in session state. This module hosts one kernel per
state root on a background asyncio loop and exposes a small synchronous facade
that the GUI can call on each rerun.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from ..artifact import BootContext, compute_manifest_hash
from ..broker import ResourceBroker
from ..docker_supervisor import DockerSupervisor
from ..index.sqlite_index import INDEX_FILENAME, open_index
from ..journal import JournalKind, JournalLayout, JournalReader, JournalWriter
from ..logging_utils import get_logger
from ..mvd import Kernel, KernelConfig, MvdDockerExecutor, PrimaryState
from .mvd_entrypoint import (_build_engine, _job_name, _observer,
                             _options_for_job, _store_for_output)

logger = get_logger(__name__)


TERMINAL_STATES = {
    PrimaryState.ARTIFACT_SUCCESS.value,
    PrimaryState.CANCELLED.value,
    PrimaryState.FAILED.value,
    PrimaryState.RECOVERY_PENDING.value,
}


@dataclass(frozen=True)
class SubmittedRun:
    """Summary returned to the GUI after submitting one manifest job."""

    attempt_id: str
    job_name: str
    dataset: str
    model: str
    logical_run_id: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "attempt_id": self.attempt_id,
            "job_name": self.job_name,
            "dataset": self.dataset,
            "model": self.model,
            "logical_run_id": self.logical_run_id,
        }


class InProcessMvdController:
    """Thread-safe synchronous facade around one in-process kernel."""

    def __init__(self, *, state_root: Path, artifact_root: Path | None = None) -> None:
        self.state_root = state_root.expanduser().resolve()
        self.artifact_root = (
            artifact_root.expanduser().resolve() if artifact_root is not None else None
        )
        if self.artifact_root is not None:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.state_root / INDEX_FILENAME
        self._loop_ready = threading.Event()
        self._closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._kernel: Optional[Kernel] = None
        self._mlflow_run_id_by_attempt: Dict[str, str] = {}
        self._mlflow_sync_in_flight: Set[str] = set()
        self._mlflow_sync_done: Set[str] = set()
        self._mlflow_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"mvd-gui-{self.state_root.name}",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait(timeout=10)
        if self._loop is None:
            raise RuntimeError("mvd controller event loop did not start")
        self._call(self._init_kernel())

    def submit_manifest(
        self,
        *,
        manifest_path: Path,
        pending_jobs: List[Dict[str, Any]],
        manifest_text: str,
        seed: Optional[int],
    ) -> List[SubmittedRun]:
        """Submit pending jobs from a manifest; may start MLflow sync in background."""
        return self._call(
            self._submit_manifest(
                manifest_path=manifest_path,
                pending_jobs=pending_jobs,
                manifest_text=manifest_text,
                seed=seed,
            )
        )

    def query_many(self, attempt_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Blocking query of multiple runs on the background event loop."""
        return self._call(self._query_many(list(attempt_ids)))

    def list_runs(self, *, state: Optional[str] = None) -> List[Dict[str, Any]]:
        """List runs, optionally filtered by primary state string."""
        return self._call(self._list_runs(state=state))

    def cancel_many(self, attempt_ids: Iterable[str]) -> None:
        """Request cancellation for each attempt id."""
        self._call(self._cancel_many(list(attempt_ids)))

    def health(self) -> Dict[str, Any]:
        """Return kernel health snapshot (runs active, journal seq, etc.)."""
        return self._call(self._require_kernel().health())

    def shutdown(self) -> None:
        """Stop the background loop and close the kernel."""
        if self._closed:
            return
        self._closed = True
        try:
            self._call(self._require_kernel().shutdown())
        finally:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        loop.run_forever()
        loop.close()

    def _call(self, coro):
        if self._loop is None:
            raise RuntimeError("mvd controller loop is not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    async def _init_kernel(self) -> None:
        if self._kernel is not None:
            return
        boot = BootContext.new(mvd_version="0.1.0-mvd")
        # Locally-built Docker images are the normal case in the GUI; default open.
        config = KernelConfig(state_root=self.state_root, accept_degraded=True)
        layout = JournalLayout.at(self.state_root / "journal").ensure()
        journal = JournalWriter(layout, boot_id=boot.boot_id, user_id=config.user_id)
        store = _store_for_output(
            state_root=self.state_root, artifact_root=self.artifact_root
        )
        supervisor = DockerSupervisor(
            engine=_build_engine(),
            journal=journal,
            mvd_version="0.1.0-mvd",
        )
        broker = ResourceBroker(observer=_observer(), journal=journal)
        executor = MvdDockerExecutor(
            journal=journal,
            boot=boot,
            store=store,
            supervisor=supervisor,
            broker=broker,
            state_root=self.state_root,
            accept_degraded=config.accept_degraded,
            user_id=config.user_id,
        )
        kernel = Kernel(
            config,
            executor=executor,
            journal=journal,
            boot=boot,
            broker=broker,
        )
        kernel.replay_from_journal()
        self._kernel = kernel
        self._project_snapshots(await kernel.list_runs())

    async def _submit_manifest(
        self,
        *,
        manifest_path: Path,
        pending_jobs: List[Dict[str, Any]],
        manifest_text: str,
        seed: Optional[int],
    ) -> List[SubmittedRun]:
        kernel = self._require_kernel()
        manifest_hash = compute_manifest_hash(manifest_text or "")
        experiment_name = _experiment_name_from_manifest(manifest_text)
        tracking_uri = _mlflow_tracking_uri()
        submitted: List[SubmittedRun] = []
        projected_attempts: List[str] = []
        for job in pending_jobs:
            if job.get("_skipped"):
                continue
            options = _options_for_job(job, manifest_hash=manifest_hash, seed=seed)
            if experiment_name:
                options["experiment_name"] = experiment_name
            parent_run_id = _maybe_start_parent_mlflow_run(
                job=job,
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
            )
            env_extra: Dict[str, str] = {}
            if tracking_uri:
                env_extra["MLFLOW_TRACKING_URI"] = _rewrite_tracking_uri_for_docker(
                    tracking_uri
                )
            if experiment_name:
                env_extra["MLFLOW_EXPERIMENT_NAME"] = experiment_name
            if parent_run_id:
                env_extra["MLFLOW_RUN_ID"] = parent_run_id
            if env_extra:
                options["container_env_extra"] = env_extra
            attempt_id = await kernel.submit_run(
                manifest_path=str(manifest_path),
                options=options,
            )
            if parent_run_id:
                with self._mlflow_lock:
                    self._mlflow_run_id_by_attempt[attempt_id] = parent_run_id
            projected_attempts.append(attempt_id)
            submitted.append(
                SubmittedRun(
                    attempt_id=attempt_id,
                    job_name=_job_name(job),
                    dataset=str(
                        job.get("dataset_name") or job.get("dataset_slug") or "?"
                    ),
                    model=str(job.get("model_slug") or job.get("model_name") or "?"),
                    logical_run_id=str(job.get("_logical_run_id") or ""),
                )
            )
        if projected_attempts:
            self._project_snapshots(
                [
                    await kernel.query_run(physical_attempt_id=attempt_id)
                    for attempt_id in projected_attempts
                ]
            )
        return submitted

    async def _query_many(self, attempt_ids: List[str]) -> List[Dict[str, Any]]:
        kernel = self._require_kernel()
        out: List[Dict[str, Any]] = []
        for attempt_id in attempt_ids:
            out.append(await kernel.query_run(physical_attempt_id=attempt_id))
        self._project_snapshots(out)
        return out

    async def _list_runs(self, *, state: Optional[str] = None) -> List[Dict[str, Any]]:
        snapshots = await self._require_kernel().list_runs(state=state)
        self._project_snapshots(snapshots)
        return snapshots

    async def _cancel_many(self, attempt_ids: List[str]) -> None:
        kernel = self._require_kernel()
        projected: List[Dict[str, Any]] = []
        for attempt_id in attempt_ids:
            await kernel.cancel_run(physical_attempt_id=attempt_id)
            projected.append(await kernel.query_run(physical_attempt_id=attempt_id))
        self._project_snapshots(projected)

    def _project_snapshots(self, snapshots: Iterable[Dict[str, Any]]) -> None:
        snapshots = list(snapshots)
        if not snapshots:
            return
        try:
            with open_index(self._index_path) as index:
                for snap in snapshots:
                    index.upsert_run(snap)
                    for plugin, status in (snap.get("projections") or {}).items():
                        index.set_projection(
                            physical_attempt_id=str(snap["physical_attempt_id"]),
                            plugin=str(plugin),
                            status=str(status),
                        )
        except Exception:
            # The SQLite index is a rebuildable GUI projection. Do not let a
            # read-only/corrupt projection database block kernel queries or run
            # cancellation; the journal remains authoritative.
            pass
        self._maybe_schedule_mlflow_syncs(snapshots)

    def _maybe_schedule_mlflow_syncs(self, snapshots: Iterable[Dict[str, Any]]) -> None:
        """Kick off MLflow sync for each newly-successful run with a
        pending tracking projection. Idempotent: each attempt_id is synced
        at most once per controller lifetime."""
        loop = getattr(self, "_loop", None)
        if loop is None:
            return
        if not hasattr(self, "_mlflow_lock"):
            return
        for snap in snapshots:
            if snap.get("primary_state") != PrimaryState.ARTIFACT_SUCCESS.value:
                continue
            projections = snap.get("projections") or {}
            if projections.get("mlflow") != "TRACKING_PENDING":
                continue
            attempt_id = str(snap["physical_attempt_id"])
            bundle_dir = snap.get("artifact_dir")
            if not bundle_dir:
                continue
            with self._mlflow_lock:
                if attempt_id in self._mlflow_sync_done:
                    continue
                if attempt_id in self._mlflow_sync_in_flight:
                    continue
                self._mlflow_sync_in_flight.add(attempt_id)
                parent_run_id = self._mlflow_run_id_by_attempt.get(attempt_id)
            options = snap.get("options") or {}
            experiment_name = str(options.get("experiment_name") or "multiverse")
            asyncio.run_coroutine_threadsafe(
                self._run_mlflow_sync(
                    attempt_id=attempt_id,
                    bundle_dir=Path(bundle_dir),
                    experiment_name=experiment_name,
                    existing_run_id=parent_run_id,
                ),
                loop,
            )

    async def _run_mlflow_sync(
        self,
        *,
        attempt_id: str,
        bundle_dir: Path,
        experiment_name: str,
        existing_run_id: Optional[str],
    ) -> None:
        """Run the MLflow sync off the loop and report the outcome to the
        kernel. Exceptions are swallowed so a misconfigured MLflow doesn't
        knock out the controller."""
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                _do_mlflow_sync,
                bundle_dir,
                experiment_name,
                existing_run_id,
            )
            kernel = self._kernel
            if kernel is not None and result is not None:
                try:
                    await kernel.report_projection_status(
                        plugin="mlflow",
                        physical_attempt_id=attempt_id,
                        status=result.outcome.value,
                        details=(
                            {"failure_reason": result.failure_reason}
                            if result.failure_reason
                            else {"target_run_id": result.target_run_id or ""}
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        "report_projection_status for %s failed: %s", attempt_id, exc
                    )
        finally:
            with self._mlflow_lock:
                self._mlflow_sync_in_flight.discard(attempt_id)
                self._mlflow_sync_done.add(attempt_id)

    def _require_kernel(self) -> Kernel:
        if self._kernel is None:
            raise RuntimeError("mvd kernel has not been initialised")
        return self._kernel


def snapshots_from_journal(
    *,
    state_root: Path,
    attempt_ids: Iterable[str] | None = None,
    state: str | None = None,
) -> List[Dict[str, Any]]:
    """Reconstruct run snapshots without acquiring the journal writer lock."""
    wanted = set(attempt_ids) if attempt_ids is not None else None
    records: Dict[str, Dict[str, Any]] = {}
    submitted_order: Dict[str, int] = {}
    reader = JournalReader(JournalLayout.at(state_root / "journal"))
    for record in reader.replay().records:
        attempt = record.physical_attempt_id
        if not attempt or (wanted is not None and attempt not in wanted):
            continue
        if record.kind is JournalKind.JOB_INTENT:
            records[attempt] = {
                "physical_attempt_id": attempt,
                "logical_run_id": record.logical_run_id,
                "primary_state": "PENDING",
                "cancel_requested": False,
                "failure_reason": None,
                "artifact_dir": None,
                "workspace_dir": None,
                "manifest_path": record.payload.get("manifest_path"),
                "submitted_wall_iso": record.wall_iso,
                "projections": {
                    "mlflow": "TRACKING_NOT_CONFIGURED",
                    "optuna": "TRACKING_NOT_APPLICABLE",
                },
                "options": dict(record.payload.get("options") or {}),
            }
            submitted_order[attempt] = record.monotonic_ns
            continue
        if attempt not in records:
            continue
        snap = records[attempt]
        if record.logical_run_id and not snap.get("logical_run_id"):
            snap["logical_run_id"] = record.logical_run_id
        if record.kind is JournalKind.STATE_TRANSITION:
            next_state = record.payload.get("to_state")
            if next_state:
                snap["primary_state"] = str(next_state)
            reason = record.payload.get("reason")
            if reason:
                snap["failure_reason"] = str(reason)
        elif record.kind is JournalKind.CONTAINER_LAUNCH:
            labels = record.payload.get("labels") or {}
            workspace = labels.get("multiverse.workspace")
            if workspace:
                snap["workspace_dir"] = str(workspace)
        elif record.kind is JournalKind.PROMOTE_PREPARE:
            snap["workspace_dir"] = record.payload.get("workspace_dir")
            snap["artifact_dir"] = record.payload.get("final_artifact_dir")
        elif record.kind is JournalKind.PROMOTE_COMMIT_MANIFEST:
            snap["artifact_dir"] = record.payload.get(
                "artifact_dir", snap.get("artifact_dir")
            )
        elif record.kind is JournalKind.PROMOTION_QUARANTINE:
            source = record.payload.get("source")
            if source and not snap.get("failure_reason"):
                snap["failure_reason"] = f"promotion quarantined: {source}"
        elif record.kind is JournalKind.CANCEL_REQUESTED:
            snap["cancel_requested"] = True
        elif record.kind is JournalKind.CANCELLED:
            snap["primary_state"] = "CANCELLED"
            snap["cancel_requested"] = True
        elif record.kind is JournalKind.PROJECTION_STATUS:
            plugin = record.payload.get("plugin")
            projection_state = record.payload.get("status")
            if plugin and projection_state:
                snap.setdefault("projections", {})[str(plugin)] = str(projection_state)
    out = list(records.values())
    if state is not None:
        out = [snap for snap in out if snap.get("primary_state") == state]
    out.sort(key=lambda snap: submitted_order.get(str(snap["physical_attempt_id"]), 0))
    return out


def _do_mlflow_sync(
    bundle_dir: Path,
    experiment_name: str,
    existing_run_id: Optional[str],
):
    """Thread-pool worker that runs the MLflow sync against a real target.

    Returns a ``SyncResult`` or ``None`` on import / target construction
    failures (treated as "tracking not configured" by the caller).
    """
    try:
        from ..projection.mlflow_sync import sync_artifact_bundle
    except Exception as exc:
        logger.warning("projection.mlflow_sync unavailable: %s", exc)
        return None
    target = _build_real_mlflow_target()
    if target is None:
        return None
    try:
        return sync_artifact_bundle(
            bundle_dir=bundle_dir,
            target=target,
            experiment_name=experiment_name,
            existing_run_id=existing_run_id,
        )
    except Exception as exc:
        logger.warning("MLflow sync_artifact_bundle raised: %s", exc)
        return None


def _build_real_mlflow_target():
    """Construct a real MLflow target using the local ``mlflow`` SDK.

    Mirrors ``cli_entrypoints._build_mlflow_target`` so the GUI controller
    and the standalone ``multiverse mlflow-sync`` CLI hit the same server
    with the same auth flow.
    """
    try:
        from multiverse.mlflow_sdk import import_mlflow

        mlflow = import_mlflow()
    except Exception as exc:
        logger.warning("mlflow SDK unavailable; skipping sync. (%s)", exc)
        return None

    from multiverse.ports import default_mlflow_tracking_uri

    uri = _mlflow_tracking_uri() or default_mlflow_tracking_uri()

    class _RealAdapter:
        name = "mlflow"

        def __init__(self) -> None:
            mlflow.set_tracking_uri(uri)

        def create_run(self, *, experiment_name, run_name, tags):
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name, tags=dict(tags)) as run:
                return run.info.run_id

        def log_params(self, *, run_id, params):
            with mlflow.start_run(run_id=run_id):
                mlflow.log_params(dict(params))

        def log_metrics(self, *, run_id, metrics):
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(dict(metrics))

        def log_artifact(self, *, run_id, path):
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(path)

        def set_terminal_status(self, *, run_id, status):
            client = mlflow.tracking.MlflowClient()
            client.set_terminated(run_id, status=status)

    try:
        return _RealAdapter()
    except Exception as exc:
        logger.warning("could not construct MLflow target: %s", exc)
        return None


def _mlflow_tracking_uri() -> Optional[str]:
    """Return the configured MLflow tracking URI, or None to skip MLflow."""
    from multiverse.ports import default_mlflow_tracking_uri

    return os.environ.get("MLFLOW_TRACKING_URI") or default_mlflow_tracking_uri()


def _rewrite_tracking_uri_for_docker(uri: str) -> str:
    """Replace localhost in a tracking URI with host.docker.internal so
    a container can reach an MLflow server bound to the host."""
    return uri.replace("//localhost", "//host.docker.internal").replace(
        "//127.0.0.1", "//host.docker.internal"
    )


def _maybe_start_parent_mlflow_run(
    *,
    job: Dict[str, Any],
    experiment_name: Optional[str],
    tracking_uri: Optional[str],
) -> Optional[str]:
    """Create an MLflow parent run for ``job`` and return its run_id.

    Returns ``None`` if MLflow is unavailable, the tracking URI is unset,
    or run creation fails. The container's EpochLogger attaches to the
    parent run via ``MLFLOW_RUN_ID``; the controller's post-success sync
    appends final scalars + artifacts to the same run.
    """
    if not tracking_uri:
        return None
    try:
        from ..tracking import start_parent_mlflow_run
    except Exception as exc:
        logger.warning("MLflow tracking helpers unavailable: %s", exc)
        return None

    job_context: Dict[str, Any] = {
        "experiment_name": experiment_name or "multiverse",
        "dataset_name": job.get("dataset_name") or job.get("dataset_slug") or "dataset",
        "dataset_slug": job.get("dataset_slug") or job.get("dataset_name") or "dataset",
        "mlflow_tracking_uri": tracking_uri,
    }
    job_spec: Dict[str, Any] = {
        "model_name": job.get("model_name") or job.get("model_slug") or "model",
        "hyperparameters": {
            (job.get("model_slug") or job.get("model_name") or "model"): dict(
                job.get("model_params") or {}
            ),
        },
        "run_settings": {
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment_name": experiment_name or "multiverse",
        },
    }
    run_name = f"{job_context['dataset_name']}-{job_spec['model_name']}"
    try:
        return start_parent_mlflow_run(
            job_context=job_context,
            job_spec=job_spec,
            run_name=run_name,
        )
    except Exception as exc:
        logger.warning("start_parent_mlflow_run failed: %s", exc)
        return None


def _experiment_name_from_manifest(manifest_text: str) -> Optional[str]:
    """Pull ``globals.experiment_name`` out of a manifest's YAML text.

    Returns ``None`` if PyYAML is unavailable, the text is unparseable, or
    no experiment name is set. Tolerated as best-effort: the sync falls
    back to a default experiment when this returns ``None``.
    """
    if not manifest_text:
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    try:
        data = yaml.safe_load(manifest_text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    globals_block = data.get("globals") if isinstance(data.get("globals"), dict) else {}
    name = globals_block.get("experiment_name") or data.get("experiment_name")
    return str(name) if name else None


_CONTROLLERS: Dict[tuple[Path, Path | None], InProcessMvdController] = {}
_CONTROLLERS_LOCK = threading.Lock()


def get_controller(
    *, state_root: Path, artifact_root: Path | None = None
) -> InProcessMvdController:
    root = state_root.expanduser().resolve()
    artifacts = (
        artifact_root.expanduser().resolve() if artifact_root is not None else None
    )
    key = (root, artifacts)
    with _CONTROLLERS_LOCK:
        controller = _CONTROLLERS.get(key)
        if controller is None:
            controller = InProcessMvdController(
                state_root=root, artifact_root=artifacts
            )
            _CONTROLLERS[key] = controller
        return controller
