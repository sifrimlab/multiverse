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
from ..mvd import (Kernel, KernelConfig, MvdDockerExecutor, PrimaryState,
                   build_executor_options)
from ..promotion import StoreLayout


def _state_root_for_output(output_root: Path) -> Path:
    return output_root / ".mvd-state"


def _store_for_output(
    *, state_root: Path, artifact_root: Path | None = None
) -> StoreLayout:
    kwargs: Dict[str, Path] = {}
    if artifact_root is not None:
        kwargs["artifacts_root"] = artifact_root
    return StoreLayout(root=state_root / "store", **kwargs).ensure()


def run_via_mvd(args: argparse.Namespace) -> int:
    """Drive the CLI's planned jobs through the mvd kernel.

    Returns a Unix exit code: 0 if every job reached
    ``ARTIFACT_SUCCESS``, otherwise 1.
    """
    output = getattr(args, "output", None)
    if not output:
        print("--output is required for mvd-backed runs", file=sys.stderr)
        return 2
    artifact_root = Path(output).expanduser().resolve()
    state_root = _state_root_for_output(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    # Session-level CLI log. Per-run logs live next to each run's artifacts
    # (run.log / container.log / orchestrator.log); this top-level file
    # captures CLI/plan-resolution events for the whole invocation.
    from ..logging_utils import setup_logging

    setup_logging(str(state_root))

    # Plan resolution: re-use the legacy parser so existing manifests
    # keep working, but never call into ``docker_runner`` afterwards.
    from ..registry_db import get_db_connection
    from .cli import (ManifestValidationError, build_missing_images,
                      generate_execution_plan, require_parsed_manifest)

    # Auto-build missing images by default; --no-build opts out. When building,
    # skip the manifest's image-availability probe so a missing image does not
    # fail validation before we get a chance to build it.
    autobuild = not getattr(args, "no_build", False)

    conn = get_db_connection()
    try:
        if getattr(args, "manifest", None):
            try:
                parsed = require_parsed_manifest(
                    args.manifest,
                    conn,
                    backend_override="docker",
                    check_images=not autobuild,
                )
            except ManifestValidationError as exc:
                print("Manifest validation failed:", file=sys.stderr)
                for err in exc.parsed.errors:
                    print(
                        f"  - {err['field']}: {err['message']} ({err['code']})",
                        file=sys.stderr,
                    )
                return 2
            manifest_text = Path(args.manifest).read_text(encoding="utf-8")
            manifest_data = parsed.data
            seed = resolve_effective_seed(getattr(args, "seed", None), manifest_data)
            pending_jobs = parsed.plan
            pending_jobs = _maybe_apply_resume(
                pending_jobs,
                args=args,
                manifest_data=manifest_data,
                manifest_text=manifest_text,
                state_root=state_root,
                backend="docker",
                seed=seed,
            )
        else:
            manifest_text = ""
            manifest_data = None
            seed = resolve_effective_seed(getattr(args, "seed", None), None)
            pending_jobs = generate_execution_plan(conn)

        _print_preflight(
            manifest_path=getattr(args, "manifest", None),
            manifest_text=manifest_text,
            pending_jobs=pending_jobs,
            manifest_data=manifest_data,
        )

        if pending_jobs and autobuild:
            failures = build_missing_images(pending_jobs, conn)
            if failures:
                print("Auto-build of model images failed:", file=sys.stderr)
                for slug, msg in failures:
                    print(f"  - {slug}: {msg}", file=sys.stderr)
                return 2
    finally:
        conn.close()

    if not pending_jobs:
        print("No pending jobs to execute.", file=sys.stderr)
        return 0

    # Default: locally-built Docker images are the normal case for researchers.
    # --strict opts into publication mode (requires a registry digest).
    accept_degraded = not getattr(args, "strict", False)
    return asyncio.run(
        _drive_jobs(
            state_root=state_root,
            artifact_root=artifact_root,
            pending_jobs=pending_jobs,
            manifest_text=manifest_text,
            seed=seed,
            accept_degraded=accept_degraded,
        )
    )


async def _drive_jobs(
    *,
    state_root: Path,
    pending_jobs: List[Dict[str, Any]],
    manifest_text: str,
    seed: int | None,
    artifact_root: Path | None = None,
    accept_degraded: bool = False,
) -> int:
    boot = BootContext.new(mvd_version="0.1.0-mvd")
    config = KernelConfig(state_root=state_root, accept_degraded=accept_degraded)
    layout = JournalLayout.at(state_root / "journal").ensure()
    journal = JournalWriter(layout, boot_id=boot.boot_id, user_id=config.user_id)
    store = _store_for_output(state_root=state_root, artifact_root=artifact_root)

    # Lazy-import the real Docker engine adapter only when we actually
    # need it. ``mvd_entrypoint`` itself stays import-clean.
    engine = _build_engine()
    supervisor = DockerSupervisor(
        engine=engine,
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
        state_root=state_root,
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


def run_via_slurm(args: argparse.Namespace) -> int:
    """Drive the CLI's planned jobs through the mvd Slurm kernel."""
    output = getattr(args, "output", None)
    if not output:
        print("--output is required for Slurm-backed runs", file=sys.stderr)
        return 2
    artifact_root = Path(output).expanduser().resolve()
    state_root = _state_root_for_output(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    from ..logging_utils import setup_logging

    setup_logging(str(state_root))

    from ..registry_db import get_db_connection
    from .cli import (ManifestValidationError, generate_execution_plan,
                      require_parsed_manifest)

    conn = get_db_connection()
    try:
        if getattr(args, "manifest", None):
            try:
                # Force the Slurm backend so manifest validation resolves SIFs
                # and skips Docker image probing even when the manifest omits
                # globals.backend.
                parsed = require_parsed_manifest(
                    args.manifest, conn, backend_override="slurm"
                )
            except ManifestValidationError as exc:
                print("Manifest validation failed:", file=sys.stderr)
                for err in exc.parsed.errors:
                    print(
                        f"  - {err['field']}: {err['message']} ({err['code']})",
                        file=sys.stderr,
                    )
                return 2
            manifest_text = Path(args.manifest).read_text(encoding="utf-8")
            manifest_data = parsed.data
            seed = resolve_effective_seed(getattr(args, "seed", None), manifest_data)
            pending_jobs = parsed.plan
            # Apply resume after Slurm SIF/GPU validation (done inside
            # parse_manifest), because image identity resolution feeds the
            # logical run identity used for matching completed work.
            pending_jobs = _maybe_apply_resume(
                pending_jobs,
                args=args,
                manifest_data=manifest_data,
                manifest_text=manifest_text,
                state_root=state_root,
                backend="slurm",
                seed=seed,
            )
        else:
            manifest_text = ""
            manifest_data = None
            seed = resolve_effective_seed(getattr(args, "seed", None), None)
            pending_jobs = generate_execution_plan(conn)
    finally:
        conn.close()

    _print_preflight(
        manifest_path=getattr(args, "manifest", None),
        manifest_text=manifest_text,
        pending_jobs=pending_jobs,
        manifest_data=manifest_data,
    )

    if not pending_jobs:
        print("No pending jobs to execute.", file=sys.stderr)
        return 0

    accept_degraded = getattr(args, "accept_degraded", False)
    return asyncio.run(
        _drive_jobs_slurm(
            state_root=state_root,
            artifact_root=artifact_root,
            pending_jobs=pending_jobs,
            manifest_text=manifest_text,
            seed=seed,
            accept_degraded=accept_degraded,
        )
    )


async def _drive_jobs_slurm(
    *,
    state_root: Path,
    pending_jobs: List[Dict[str, Any]],
    manifest_text: str,
    seed: int | None,
    artifact_root: Path | None = None,
    accept_degraded: bool = False,
) -> int:
    from ..artifact import BootContext, compute_manifest_hash
    from ..broker import HostMetrics, InMemoryHostObserver, ResourceBroker
    from ..journal import JournalLayout, JournalWriter
    from ..mvd import Kernel, KernelConfig, PrimaryState
    from ..mvd.slurm_executor import MvdSlurmExecutor
    from ..slurm.engine import RealSlurmEngine

    boot = BootContext.new(mvd_version="0.1.0-mvd")
    config = KernelConfig(state_root=state_root, accept_degraded=accept_degraded)
    layout = JournalLayout.at(state_root / "journal").ensure()
    journal = JournalWriter(layout, boot_id=boot.boot_id, user_id=config.user_id)
    store = _store_for_output(state_root=state_root, artifact_root=artifact_root)

    engine = RealSlurmEngine()
    broker = ResourceBroker(observer=_observer(), journal=journal)
    executor = MvdSlurmExecutor(
        journal=journal,
        boot=boot,
        store=store,
        engine=engine,
        broker=broker,
        state_root=state_root,
        accept_degraded=accept_degraded,
        user_id=config.user_id,
    )
    kernel = Kernel(
        config,
        executor=executor,
        journal=journal,
        boot=boot,
        broker=broker,
    )

    manifest_hash = compute_manifest_hash(manifest_text or "")
    attempt_ids: List[str] = []
    job_names: List[str] = []
    for job in pending_jobs:
        if job.get("_skipped"):
            continue
        options = _options_for_slurm_job(job, manifest_hash=manifest_hash, seed=seed)
        attempt = await kernel.submit_run(
            manifest_path=str(getattr(job, "manifest_path", "") or ""),
            options=options,
        )
        attempt_ids.append(attempt)
        job_names.append(_job_name(job))

    n_success = 0
    n_fail = 0
    for attempt, name in zip(attempt_ids, job_names):
        task = kernel._execution_tasks.get(attempt)
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
                f"[!!] {name}: {snap['primary_state']} — {snap.get('failure_reason')}",
                file=sys.stderr,
            )

    await kernel.shutdown()
    print(
        f"slurm run complete: {n_success} succeeded, {n_fail} failed", file=sys.stderr
    )
    return 0 if n_fail == 0 else 1


def _maybe_apply_resume(
    pending_jobs: List[Dict[str, Any]],
    *,
    args: argparse.Namespace,
    manifest_data: Dict[str, Any],
    manifest_text: str,
    state_root: Path,
    backend: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """Decorate the plan with mvd-backed resume when skip-completed is enabled.

    Resume policy precedence (CLI flag > manifest globals > default off) is
    resolved in :func:`multiverse.runner.resume.resolve_skip_completed`. The
    decoration marks completed jobs ``_skipped`` against durable mvd
    ``ARTIFACT_SUCCESS`` state — never the legacy ``runs`` table.

    ``seed`` is the already-resolved effective seed (see
    :func:`resolve_effective_seed`) so the resume identity matches what the
    executor will actually run with.
    """
    from ..artifact import compute_manifest_hash
    from .resume import decorate_plan_with_resume, resolve_skip_completed

    skip_completed = resolve_skip_completed(
        cli_flag=getattr(args, "skip_completed", None),
        manifest_data=manifest_data,
    )
    if not skip_completed:
        return pending_jobs
    return decorate_plan_with_resume(
        pending_jobs,
        state_root=state_root,
        manifest_hash=compute_manifest_hash(manifest_text or ""),
        seed=seed,
        backend=backend,
    )


def resolve_effective_seed(
    cli_seed: "int | None", manifest_data: "Dict[str, Any] | None"
) -> int:
    """Resolve the execution seed with documented precedence (Gap 4).

    1. an explicit ``--seed`` value (``cli_seed`` not ``None``);
    2. ``globals.random_seed`` from the manifest;
    3. fallback ``42``.

    The same resolved seed is used for executor submission *and* the resume
    logical-run identity, so a manifest's declared ``random_seed`` actually runs
    (and is reflected in resume matching) without requiring ``--seed``.
    """
    if cli_seed is not None:
        return int(cli_seed)
    if manifest_data:
        globals_block = manifest_data.get("globals") or {}
        if (
            isinstance(globals_block, dict)
            and globals_block.get("random_seed") is not None
        ):
            return int(globals_block["random_seed"])
    return 42


def _print_preflight(
    *,
    manifest_path: str | None,
    manifest_text: str,
    pending_jobs: List[Dict[str, Any]],
    manifest_data: "Dict[str, Any] | None" = None,
) -> None:
    """Print a dry-run/preflight summary so launches are auditable.

    Shows the manifest path and hash that was launched, distinct job counts, and
    per-job model, dataset, params hash, and skip reason. The counts are kept
    separate (Gap 5) because one YAML job can expand into several model jobs:

    * ``yaml jobs`` — entries in the manifest ``jobs:`` list;
    * ``planned jobs`` — the expanded plan (one per model × dataset);
    * ``runnable jobs`` — planned jobs that will be submitted;
    * ``skipped completed jobs`` — planned jobs resumed from mvd state.
    """
    from ..artifact import compute_manifest_hash

    manifest_hash = compute_manifest_hash(manifest_text or "")
    runnable = [j for j in pending_jobs if not j.get("_skipped")]
    skipped = [j for j in pending_jobs if j.get("_skipped")]
    yaml_jobs = None
    if isinstance(manifest_data, dict) and isinstance(manifest_data.get("jobs"), list):
        yaml_jobs = len(manifest_data["jobs"])
    print("mvd preflight:", file=sys.stderr)
    print(f"  manifest: {manifest_path or '(none)'}", file=sys.stderr)
    print(f"  manifest_hash: {manifest_hash}", file=sys.stderr)
    if yaml_jobs is not None:
        print(f"  yaml jobs: {yaml_jobs}", file=sys.stderr)
    print(f"  planned jobs: {len(pending_jobs)}", file=sys.stderr)
    print(f"  runnable jobs: {len(runnable)}", file=sys.stderr)
    print(f"  skipped completed jobs: {len(skipped)}", file=sys.stderr)
    for job in pending_jobs:
        model = job.get("model_slug") or job.get("model_name") or "?"
        dataset = job.get("dataset_name") or job.get("dataset_slug") or "?"
        params_hash = job.get("params_hash") or "-"
        if job.get("_skipped"):
            reason = job.get("_skip_reason", "skipped")
            attempt = job.get("_completed_attempt_id")
            prov = f" [attempt={attempt}]" if attempt else ""
            print(
                f"  - SKIP {dataset}/{model} params={params_hash}: {reason}{prov}",
                file=sys.stderr,
            )
        else:
            print(
                f"  - RUN  {dataset}/{model} params={params_hash}",
                file=sys.stderr,
            )


def _options_for_slurm_job(
    job: Dict[str, Any], *, manifest_hash: str, seed: int | None = None
) -> Dict[str, Any]:
    """Build ``Kernel.submit_run`` options for a Slurm job dict.

    The Slurm executor reads its knobs from the nested ``options["slurm"]``
    block, so we use :func:`build_slurm_executor_options` (which produces that
    shape) rather than the flat Docker option builder.
    """
    from ..mvd.slurm_executor import build_slurm_executor_options

    slurm_cfg = dict(job.get("_slurm", {}) or {})

    def _opt_int(key: str):
        val = slurm_cfg.get(key)
        return int(val) if val is not None else None

    options = build_slurm_executor_options(
        model_slug=str(job.get("model_slug") or job.get("model_name") or "model"),
        image_sif=str(job.get("image_sif") or ""),
        dataset_slug=str(
            job.get("dataset_slug") or job.get("dataset_name") or "dataset"
        ),
        dataset_path=str(job.get("dataset_path") or ""),
        dataset_n_obs=int(job.get("dataset_n_obs") or job.get("n_obs") or 0),
        params=dict(job.get("model_params") or {}),
        image_digest=job.get("image_digest"),
        model_version=str(job.get("model_version") or "0.0.0"),
        dataset_n_vars=job.get("dataset_n_vars") or job.get("n_vars"),
        validators=str(job.get("validators") or "basic"),
        artifact_dir_name=(
            str(job.get("artifact_dir_name"))
            if job.get("artifact_dir_name")
            else os.path.basename(str(job.get("output_path") or "")) or None
        ),
        seed=seed,
        partition=slurm_cfg.get("partition"),
        account=slurm_cfg.get("account"),
        qos=slurm_cfg.get("qos"),
        time_minutes=_opt_int("time_minutes"),
        mem_gb=_opt_int("mem_gb"),
        cpus_per_task=int(slurm_cfg.get("cpus_per_task", 1)),
        gpus=_opt_int("gpus"),
        extra_directives=slurm_cfg.get("extra_directives"),
    )
    options["manifest_hash"] = manifest_hash
    # $SLURM_TMPDIR staging flags live in the nested slurm block too.
    options["slurm"]["use_tmpdir"] = bool(slurm_cfg.get("use_tmpdir", False))
    options["slurm"]["use_tmpdir_sif"] = bool(slurm_cfg.get("use_tmpdir_sif", False))
    return options


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


def _options_for_job(
    job: Dict[str, Any], *, manifest_hash: str, seed: int | None = None
) -> Dict[str, Any]:
    """Build ``Kernel.submit_run`` options from a legacy job dict."""
    return build_executor_options(
        model_slug=str(job.get("model_slug") or job.get("model_name") or "model"),
        model_image=str(job.get("model_image") or ""),
        image_digest=job.get("image_digest"),
        dataset_slug=str(
            job.get("dataset_slug") or job.get("dataset_name") or "dataset"
        ),
        dataset_path=str(job.get("dataset_path") or ""),
        dataset_n_obs=int(job.get("dataset_n_obs") or job.get("n_obs") or 0),
        dataset_n_vars=job.get("dataset_n_vars") or job.get("n_vars"),
        params=dict(job.get("model_params") or {}),
        model_version=str(job.get("model_version") or "0.0.0"),
        manifest_text="",
        validators=str(job.get("validators") or "basic"),
        artifact_dir_name=(
            str(job.get("artifact_dir_name"))
            if job.get("artifact_dir_name")
            else os.path.basename(str(job.get("output_path") or "")) or None
        ),
        mem_limit=job.get("mem_limit"),
        gpu_requested=bool(job.get("gpu", False)),
        preprocessing=job.get("preprocessing"),
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
        raise RuntimeError(f"could not load the Docker engine adapter: {exc}") from exc
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
