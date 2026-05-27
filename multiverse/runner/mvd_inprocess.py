"""Process-local mvd controller for GUI/client cutover.

The Streamlit GUI cannot ``await`` kernel calls directly and should not own
long-running run tasks in session state. This module hosts one kernel per
state root on a background asyncio loop and exposes a small synchronous facade
that the GUI can call on each rerun.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..artifact import BootContext, compute_manifest_hash
from ..broker import ResourceBroker
from ..docker_supervisor import DockerSupervisor
from ..index.sqlite_index import INDEX_FILENAME, open_index
from ..journal import JournalLayout, JournalWriter
from ..mvd import Kernel, KernelConfig, MvdDockerExecutor, PrimaryState
from ..promotion import StoreLayout
from .mvd_entrypoint import _build_engine, _job_name, _observer, _options_for_job


TERMINAL_STATES = {
    PrimaryState.ARTIFACT_SUCCESS.value,
    PrimaryState.CANCELLED.value,
    PrimaryState.FAILED.value,
    PrimaryState.RECOVERY_PENDING.value,
}


@dataclass(frozen=True)
class SubmittedRun:
    attempt_id: str
    job_name: str
    dataset: str
    model: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "attempt_id": self.attempt_id,
            "job_name": self.job_name,
            "dataset": self.dataset,
            "model": self.model,
        }


class InProcessMvdController:
    """Thread-safe synchronous facade around one in-process kernel."""

    def __init__(self, *, state_root: Path) -> None:
        self.state_root = state_root.expanduser().resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.state_root / INDEX_FILENAME
        self._loop_ready = threading.Event()
        self._closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._kernel: Optional[Kernel] = None
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
        return self._call(
            self._submit_manifest(
                manifest_path=manifest_path,
                pending_jobs=pending_jobs,
                manifest_text=manifest_text,
                seed=seed,
            )
        )

    def query_many(self, attempt_ids: Iterable[str]) -> List[Dict[str, Any]]:
        return self._call(self._query_many(list(attempt_ids)))

    def list_runs(self, *, state: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._call(self._list_runs(state=state))

    def cancel_many(self, attempt_ids: Iterable[str]) -> None:
        self._call(self._cancel_many(list(attempt_ids)))

    def health(self) -> Dict[str, Any]:
        return self._call(self._require_kernel().health())

    def shutdown(self) -> None:
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
        layout = JournalLayout.at(self.state_root / "journal").ensure()
        journal = JournalWriter(layout, boot_id=boot.boot_id)
        store = StoreLayout(root=self.state_root / "store").ensure()
        supervisor = DockerSupervisor(
            engine=_build_engine(),
            journal=journal,
            mvd_version="0.1.0-mvd",
        )
        executor = MvdDockerExecutor(
            journal=journal,
            boot=boot,
            store=store,
            supervisor=supervisor,
            broker=ResourceBroker(observer=_observer()),
            state_root=self.state_root,
        )
        kernel = Kernel(
            KernelConfig(state_root=self.state_root),
            executor=executor,
            journal=journal,
            boot=boot,
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
        submitted: List[SubmittedRun] = []
        projected_attempts: List[str] = []
        for job in pending_jobs:
            if job.get("_skipped"):
                continue
            options = _options_for_job(job, manifest_hash=manifest_hash, seed=seed)
            attempt_id = await kernel.submit_run(
                manifest_path=str(manifest_path),
                options=options,
            )
            projected_attempts.append(attempt_id)
            submitted.append(
                SubmittedRun(
                    attempt_id=attempt_id,
                    job_name=_job_name(job),
                    dataset=str(job.get("dataset_name") or job.get("dataset_slug") or "?"),
                    model=str(job.get("model_slug") or job.get("model_name") or "?"),
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
            return

    def _require_kernel(self) -> Kernel:
        if self._kernel is None:
            raise RuntimeError("mvd kernel has not been initialised")
        return self._kernel


_CONTROLLERS: Dict[Path, InProcessMvdController] = {}
_CONTROLLERS_LOCK = threading.Lock()


def get_controller(*, state_root: Path) -> InProcessMvdController:
    root = state_root.expanduser().resolve()
    with _CONTROLLERS_LOCK:
        controller = _CONTROLLERS.get(root)
        if controller is None:
            controller = InProcessMvdController(state_root=root)
            _CONTROLLERS[root] = controller
        return controller
