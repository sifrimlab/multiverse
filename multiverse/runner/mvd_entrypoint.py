"""CLI ↔ mvd-kernel bridge (STRATEGY v2 §6 cutover).

Default ``multiverse run`` now routes here. The bridge:

1. Resolves the run manifest exactly as the legacy CLI did (so existing
   ``run_manifest.yaml`` files keep working).
2. Constructs a :class:`MvdDockerExecutor` wired to a real Docker engine
   (lazy-imported), a :class:`ResourceBroker` observing host metrics, and
   a :class:`DockerSupervisor` for labels + leases.
3. Submits each job through ``Kernel.submit_run`` and waits for terminal
   states.
4. Reports a per-job summary and exits non-zero if any job did not reach
   ``ARTIFACT_SUCCESS``.

The bridge intentionally does NOT import ``docker_runner.py`` from the
legacy path. The grep gate in
``tests/unit/test_cli_cutover.py::test_default_cli_does_not_pull_in_legacy_runner``
locks this invariant.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..artifact import BootContext, compute_manifest_hash
from ..broker import HostMetrics, InMemoryHostObserver, ResourceBroker
from ..docker_supervisor import DockerSupervisor
from ..index.sqlite_index import INDEX_FILENAME, open_index
from ..journal import JournalLayout, JournalWriter
from ..mvd import (
    Kernel,
    KernelConfig,
    MvdDockerExecutor,
    PrimaryState,
    build_executor_options,
)
from ..promotion import StoreLayout


def run_via_mvd(args: argparse.Namespace) -> int:
    """Drive the CLI's planned jobs through the mvd kernel.

    Returns a Unix exit code: 0 if every job reached
    ``ARTIFACT_SUCCESS``, otherwise 1.
    """
    output = getattr(args, "output", None)
    if not output:
        print("--output is required for mvd-backed runs", file=sys.stderr)
        return 2
    state_root = Path(output).expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)

    # Plan resolution: re-use the legacy parser so existing manifests
    # keep working, but never call into ``docker_runner`` afterwards.
    from .cli import (
        ManifestValidationError,
        generate_execution_plan,
        require_parsed_manifest,
    )
    from ..registry_db import get_db_connection

    conn = get_db_connection()
    try:
        if getattr(args, "manifest", None):
            try:
                parsed = require_parsed_manifest(args.manifest, conn)
            except ManifestValidationError as exc:
                print("Manifest validation failed:", file=sys.stderr)
                for err in exc.parsed.errors:
                    print(
                        f"  - {err['field']}: {err['message']} ({err['code']})",
                        file=sys.stderr,
                    )
                return 2
            manifest_text = Path(args.manifest).read_text(encoding="utf-8")
            pending_jobs = parsed.plan
        else:
            manifest_text = ""
            pending_jobs = generate_execution_plan(conn)
    finally:
        conn.close()

    if not pending_jobs:
        print("No pending jobs to execute.", file=sys.stderr)
        return 0

    return asyncio.run(
        _drive_jobs(
            state_root=state_root,
            pending_jobs=pending_jobs,
            manifest_text=manifest_text,
            seed=getattr(args, "seed", None),
        )
    )


async def _drive_jobs(
    *,
    state_root: Path,
    pending_jobs: List[Dict[str, Any]],
    manifest_text: str,
    seed: int | None,
) -> int:
    boot = BootContext.new(mvd_version="0.1.0-mvd")
    layout = JournalLayout.at(state_root / "journal").ensure()
    journal = JournalWriter(layout, boot_id=boot.boot_id)
    store = StoreLayout(root=state_root / "store").ensure()

    # Lazy-import the real Docker engine adapter only when we actually
    # need it. ``mvd_entrypoint`` itself stays import-clean.
    engine = _build_engine()
    supervisor = DockerSupervisor(
        engine=engine,
        journal=journal,
        mvd_version="0.1.0-mvd",
    )
    broker = ResourceBroker(observer=_observer())
    executor = MvdDockerExecutor(
        journal=journal,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=broker,
        state_root=state_root,
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root),
        executor=executor,
        journal=journal,
        boot=boot,
    )

    manifest_hash = compute_manifest_hash(manifest_text or "")
    attempt_ids: List[str] = []
    job_names: List[str] = []
    for job in pending_jobs:
        if job.get("_skipped"):
            continue
        options = _options_for_job(job, manifest_hash=manifest_hash, seed=seed)
        attempt = await kernel.submit_run(
            manifest_path=str(getattr(job, "manifest_path", "") or ""),
            options=options,
        )
        attempt_ids.append(attempt)
        job_names.append(_job_name(job))

    # Wait for each execution task — they are kernel-owned.
    n_success = 0
    n_fail = 0
    for attempt, name in zip(attempt_ids, job_names):
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task is not None:
            try:
                await task
            except Exception as exc:
                print(f"[!!] {name}: executor raised {exc}", file=sys.stderr)
        snap = await kernel.query_run(physical_attempt_id=attempt)
        _project_snapshot_to_index(state_root, snap)
        if snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value:
            n_success += 1
            print(f"[ok] {name}: ARTIFACT_SUCCESS at {snap['artifact_dir']}")
        else:
            n_fail += 1
            print(
                f"[!!] {name}: {snap['primary_state']} — "
                f"{snap.get('failure_reason')}",
                file=sys.stderr,
            )

    await kernel.shutdown()
    print(f"mvd run complete: {n_success} succeeded, {n_fail} failed", file=sys.stderr)
    return 0 if n_fail == 0 else 1




def _project_snapshot_to_index(state_root: Path, snap: Dict[str, Any]) -> None:
    try:
        with open_index(state_root / INDEX_FILENAME) as index:
            index.upsert_run(snap)
            for plugin, status in (snap.get("projections") or {}).items():
                index.set_projection(
                    physical_attempt_id=str(snap["physical_attempt_id"]),
                    plugin=str(plugin),
                    status=str(status),
                )
    except Exception:
        # Rebuildable GUI projection only. The journal/artifact store remain
        # authoritative, and `multiverse rebuild-index` can recreate this DB.
        return

def _options_for_job(job: Dict[str, Any], *, manifest_hash: str, seed: int | None = None) -> Dict[str, Any]:
    """Build ``Kernel.submit_run`` options from a legacy job dict."""
    return build_executor_options(
        model_slug=str(
            job.get("model_slug")
            or job.get("model_name")
            or "model"
        ),
        model_image=str(job.get("model_image") or ""),
        image_digest=job.get("image_digest"),
        dataset_slug=str(job.get("dataset_slug") or job.get("dataset_name") or "dataset"),
        dataset_path=str(job.get("dataset_path") or ""),
        dataset_n_obs=int(job.get("dataset_n_obs") or job.get("n_obs") or 0),
        dataset_n_vars=job.get("dataset_n_vars") or job.get("n_vars"),
        params=dict(job.get("model_params") or {}),
        model_version=str(job.get("model_version") or "0.0.0"),
        manifest_text="",
        validators=str(job.get("validators") or "basic"),
        artifact_dir_name=os.path.basename(str(job.get("output_path") or "")) or None,
        mem_limit=job.get("mem_limit"),
        seed=seed,
    ) | {"manifest_hash": manifest_hash}


def _job_name(job: Dict[str, Any]) -> str:
    return (
        job.get("name")
        or f"{job.get('dataset_name', '?')}_{job.get('model_slug') or job.get('model_name', '?')}"
    )


# ---------------------------------------------------------------------------
# Engine + observer wiring (production)
# ---------------------------------------------------------------------------


def _build_engine():
    """Construct the production Docker engine adapter.

    The adapter imports the Docker SDK lazily, so this module remains clean
    at import time. Runtime failures are intentional and fail closed: a
    production run must not silently downgrade to the in-memory test engine.
    """
    try:
        from ..docker_supervisor.client import RealDockerEngine
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"could not load the Docker engine adapter: {exc}"
        ) from exc
    return RealDockerEngine()


def _observer():
    """Production observer. Uses psutil if available; otherwise reports
    generous synthetic free memory so the broker admits in test
    environments."""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return InMemoryHostObserver(
            HostMetrics(
                ram_free_bytes=int(vm.available),
                ram_total_bytes=int(vm.total),
            )
        )
    except Exception:
        return InMemoryHostObserver(
            HostMetrics(ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3)
        )
