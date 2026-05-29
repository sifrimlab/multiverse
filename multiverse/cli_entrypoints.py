"""First-class maintenance CLI commands (STRATEGY v2 §7).

* ``multiverse doctor`` — read-only diagnostics. ``--repair-health-probes``
  invokes the sweeper.
* ``multiverse rebuild-index`` — paused-kernel rebuild from journal +
  artifacts.
* ``multiverse gc`` — dry-run report by default; ``--apply`` actually
  deletes (with all three Tier-2 gates).
* ``multiverse mlflow-sync`` — push artifact manifests to MLflow; outage
  is reported, never crashes.

Each command is lazy-import-friendly: ``--help`` works without
``mlflow``/``optuna``/``scanpy``/etc. installed. The grep gate test in
``tests/unit/test_first_class_commands.py`` enforces that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .state_paths import resolve_state_root


def _default_state_root() -> Path:
    """argparse default that follows the M1 resolver precedence chain."""
    return resolve_state_root()


def _default_store_root() -> Path:
    return _default_state_root() / "store"


# ---------------------------------------------------------------------------
# multiverse doctor
# ---------------------------------------------------------------------------


def doctor_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="multiverse doctor")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Store root to probe (default: <state-root>/store).",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="State root for the state-paths probe "
        "(default: $MVEXP_STATE_DIR / config / $XDG_STATE_HOME/mvexp / $HOME/.mvexp).",
    )
    parser.add_argument(
        "--repair-health-probes",
        action="store_true",
        help="Sweep expired entries from hidden health-probe namespaces.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report on stdout.",
    )
    parser.add_argument(
        "--deep-slurm",
        action="store_true",
        help=(
            "Probe Slurm beyond binary presence: enumerate partitions via "
            "sinfo and submit a one-second `--wrap=true` smoke job. "
            "Allocates a node briefly; off by default."
        ),
    )
    parser.add_argument(
        "--slurm-smoke-partition",
        type=str,
        default=None,
        help="Partition for the --deep-slurm smoke job (default: first sinfo entry).",
    )
    args = parser.parse_args(argv)

    from .doctor import run_storage_probes, sweep_expired_health_probes
    from .doctor.engines_probe import probe_container_engines
    from .doctor.slurm_probe import probe_slurm_deep
    from .doctor.health_probes import probe_workspace_directory
    from .doctor.projection_probe import probe_projection_consistency
    from .doctor.report import DoctorReport, DoctorSection, SectionStatus
    from .doctor.reservation_probe import probe_reservation_ledger
    from .doctor.state_paths_probe import probe_state_root

    explicit = args.state_root is not None or args.root is not None
    if args.state_root is not None:
        state_root = args.state_root
    elif args.root is not None:
        state_root = args.root.parent
    else:
        state_root = _default_state_root()
    store_root = args.root or (state_root / "store")

    state_paths_report = probe_state_root(state_root, explicit=explicit)
    state_paths_section = DoctorSection(
        name="state_paths",
        status=(
            SectionStatus.BLOCKED
            if state_paths_report.probe.value == "fail"
            else SectionStatus.OK
        ),
        rows=[state_paths_report.to_dict()],
    )

    engines_report = probe_container_engines(deep=False)
    engines_rows = [engines_report.to_dict()]
    engines_status = SectionStatus.OK  # diagnostic-only; never blocks
    if args.deep_slurm:
        slurm_deep = probe_slurm_deep(
            smoke_test=True,
            smoke_partition=args.slurm_smoke_partition,
        )
        engines_rows.append(slurm_deep.to_dict())
        if slurm_deep.probe.value == "fail":
            engines_status = SectionStatus.WARNING
    engines_section = DoctorSection(
        name="engines",
        status=engines_status,
        rows=engines_rows,
    )

    storage = run_storage_probes(store_root)
    storage_section = DoctorSection(
        name="storage",
        status=_storage_status(storage),
        rows=[r.to_dict() for r in storage.results],
        summary=f"worst-level={storage.worst_level.value}",
    )
    # Workspace health probe.
    workspaces_root = store_root / "workspaces"
    workspaces_root.mkdir(parents=True, exist_ok=True)
    health = probe_workspace_directory(workspaces_root)
    health_section = DoctorSection(
        name="health_probes",
        status=(
            SectionStatus.WARNING
            if health.leak_count > 0 or health.cleanup.value != "clean"
            else SectionStatus.OK
        ),
        rows=[health.to_dict()],
    )

    reservation_report = probe_reservation_ledger(state_root)
    reservation_section = DoctorSection(
        name="reservations",
        status=(
            SectionStatus.WARNING
            if reservation_report.leak_count > 0
            or reservation_report.probe.value == "fail"
            else SectionStatus.OK
        ),
        rows=[reservation_report.to_dict()],
    )

    projection_report = probe_projection_consistency(state_root)
    projection_section = DoctorSection(
        name="projection",
        status=(
            SectionStatus.WARNING
            if projection_report.probe.value == "fail"
            else SectionStatus.OK
        ),
        rows=[projection_report.to_dict()],
    )

    sweep_report: dict = {}
    if args.repair_health_probes:
        sweep_report = sweep_expired_health_probes(workspaces_root)

    report = DoctorReport(
        sections=[
            state_paths_section,
            engines_section,
            storage_section,
            health_section,
            reservation_section,
            projection_section,
        ],
        accept_degraded=False,
    )
    payload = report.to_dict()
    payload["sweep"] = sweep_report
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_human_doctor(report, sweep_report)

    if report.overall_status is SectionStatus.BLOCKED:
        return 2
    if report.overall_status is SectionStatus.WARNING:
        return 1
    return 0


def _storage_status(report) -> "SectionStatus":
    from .doctor import StorageLevel
    from .doctor.report import SectionStatus

    worst = report.worst_level
    if worst is StorageLevel.BLOCKED:
        return SectionStatus.BLOCKED
    if worst in (StorageLevel.DEGRADED, StorageLevel.DANGEROUS):
        return SectionStatus.WARNING
    return SectionStatus.OK


def _print_human_doctor(report, sweep_report) -> None:
    print(f"=== multiverse doctor ({report.overall_status.value}) ===")
    for section in report.sections:
        print(f"-- {section.name}: {section.status.value} --")
        for row in section.rows:
            print(f"  {row}")
    if sweep_report:
        print(f"sweep: {sweep_report}")


# ---------------------------------------------------------------------------
# multiverse rebuild-index
# ---------------------------------------------------------------------------


def rebuild_index_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="multiverse rebuild-index")
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="State root containing journal/ "
        "(default: $MVEXP_STATE_DIR / config / $XDG_STATE_HOME/mvexp / $HOME/.mvexp).",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=None,
        help="Store root containing artifacts/ (default: <state-root>/store).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite index file (default <state-root>/mvexp_state.db).",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Append-style rebuild — do not truncate the index first.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Read-only mode: report drift between the journal and the "
            "projection without writing. Exit code 0 if in sync, 1 "
            "otherwise. STRATEGY M5 — the projection is rebuildable; "
            "this flag lets operators check before doing it."
        ),
    )
    args = parser.parse_args(argv)

    from .index import INDEX_FILENAME, open_index, rebuild_index
    from .index_projection import verify_projection_against_journal
    from .promotion import StoreLayout

    state_root = args.state_root or _default_state_root()
    store_root = args.store_root or (state_root / "store")
    db_path = args.db or (state_root / INDEX_FILENAME)
    state_root.mkdir(parents=True, exist_ok=True)

    if args.verify:
        report = verify_projection_against_journal(state_root)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.in_sync else 1

    store = StoreLayout(root=store_root).ensure()
    with open_index(db_path) as idx:
        result = rebuild_index(
            index=idx,
            state_root=state_root,
            store=store,
            truncate=not args.no_truncate,
        )
    print(json.dumps(result.summary_dict(), indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# multiverse gc
# ---------------------------------------------------------------------------


def gc_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="multiverse gc")
    parser.add_argument(
        "--store-root",
        type=Path,
        default=None,
        help="Store root (default: <state-root>/store).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (default is dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run (default; useful for clarity).",
    )
    parser.add_argument(
        "--retention-failed",
        type=int,
        default=None,
        help="Retention threshold (seconds) for failed workspaces.",
    )
    parser.add_argument(
        "--retention-cancelled",
        type=int,
        default=None,
        help="Retention threshold (seconds) for cancelled workspaces.",
    )
    parser.add_argument(
        "--retention-quarantine",
        type=int,
        default=None,
        help="Retention threshold (seconds) for quarantine entries.",
    )
    parser.add_argument(
        "--no-export-required",
        action="store_true",
        help="Do not require an EXPORTED marker before deletion.",
    )
    parser.add_argument(
        "--apply-to-promoted",
        action="store_true",
        help="Consider promoted artifacts as candidates (still requires "
        "--apply; default policy refuses).",
    )
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        print("--apply and --dry-run are mutually exclusive", file=sys.stderr)
        return 2

    from .gc import (
        RetentionPolicy,
        apply_plan,
        build_plan,
        enumerate_candidates,
    )
    from .promotion import StoreLayout

    store_root = args.store_root or _default_store_root()
    store = StoreLayout(root=store_root).ensure()
    policy = RetentionPolicy(
        failed_workspaces_seconds=args.retention_failed,
        cancelled_workspaces_seconds=args.retention_cancelled,
        quarantine_seconds=args.retention_quarantine,
    )
    candidates = enumerate_candidates(store)
    plan = build_plan(
        candidates,
        policy=policy,
        require_export=not args.no_export_required,
        apply_to_promoted=args.apply_to_promoted,
    )
    result = apply_plan(
        plan, store_root=store.root, apply=bool(args.apply)
    )
    print(
        f"gc {'apply' if args.apply else 'dry-run'}: "
        f"would-delete={len(plan.to_delete)} kept={len(plan.to_keep)}; "
        f"report={result.report_path}"
    )
    if args.apply:
        print(
            f"deleted={len(result.deleted_paths)} refused={len(result.refused_paths)}"
        )
    return 0


# ---------------------------------------------------------------------------
# multiverse mlflow-sync
# ---------------------------------------------------------------------------


def mlflow_sync_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="multiverse mlflow-sync")
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Artifact-bundle directory (contains artifact_manifest.json).",
    )
    parser.add_argument(
        "--experiment",
        default="multiverse",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI env var).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON sync result on stdout.",
    )
    args = parser.parse_args(argv)

    from .projection import MLflowSyncPlugin, SyncOutcome

    try:
        target = _build_mlflow_target(args.tracking_uri)
    except Exception as exc:
        print(f"mlflow-sync: could not construct MLflow target: {exc}", file=sys.stderr)
        return 2

    plugin = MLflowSyncPlugin(target=target, experiment_name=args.experiment)
    result = plugin.sync_bundle(args.bundle)
    if args.json:
        json.dump(
            {
                "outcome": result.outcome.value,
                "physical_attempt_id": result.physical_attempt_id,
                "target_run_id": result.target_run_id,
                "failure_reason": result.failure_reason,
                "metrics_logged": result.metrics_logged,
                "artifacts_logged": result.artifacts_logged,
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        print(
            f"mlflow-sync: {args.bundle} -> {result.outcome.value}"
            + (f" (run_id={result.target_run_id})" if result.target_run_id else "")
        )
    return 0 if result.outcome is SyncOutcome.SYNCED else 1


def _build_mlflow_target(tracking_uri: Optional[str]):
    """Lazy-construct a real MLflow target. Tests substitute via
    ``--bundle`` against an unreachable URI to exercise the failure
    path; the function itself raises ImportError if MLflow is absent so
    the caller can surface a clean error."""
    import os

    try:
        import mlflow  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the mlflow package is required for mlflow-sync; install the "
            "ml-legacy extra"
        ) from exc

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Defer to a thin adapter living alongside the projection module.
    from .projection.base import MLflowTarget  # for typing only

    class _RealAdapter:
        name = "mlflow"

        def __init__(self) -> None:
            mlflow.set_tracking_uri(uri)

        def create_run(self, *, experiment_name, run_name, tags):
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name, tags=dict(tags)) as run:
                self._current = run.info.run_id
            return self._current

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
            client = mlflow.MlflowClient()
            client.set_terminated(run_id, status=status)

    return _RealAdapter()


# ---------------------------------------------------------------------------
# multiverse migrate-state-dir
# ---------------------------------------------------------------------------


def migrate_state_dir_main(argv: Optional[List[str]] = None) -> int:
    """Relocate a pre-M1 state directory (DB + store/) to the resolver's
    chosen location.

    The default behavior is a dry-run: report what *would* move. ``--apply``
    actually performs the move (using ``os.replace`` when the source and
    target live on the same filesystem; falling back to copy+remove
    otherwise). The DB is moved last so a failed copy mid-flight leaves
    the legacy install intact.
    """
    parser = argparse.ArgumentParser(prog="multiverse migrate-state-dir")
    parser.add_argument(
        "--from",
        dest="src",
        type=Path,
        default=None,
        help="Source legacy state root (default: auto-detected package directory).",
    )
    parser.add_argument(
        "--to",
        dest="dst",
        type=Path,
        default=None,
        help="Destination state root (default: M1 resolver output).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files (default is dry-run).",
    )
    args = parser.parse_args(argv)

    import shutil
    from .state_paths import REPO_ROOT_GUESS, find_legacy_db

    legacy_db = find_legacy_db()
    if args.src is None:
        if legacy_db is None:
            print(
                "migrate-state-dir: no legacy state detected. "
                "Specify --from explicitly if you need to migrate from elsewhere.",
                file=sys.stderr,
            )
            return 0
        src = legacy_db.parent
    else:
        src = args.src.expanduser().resolve()
    dst = (args.dst or _default_state_root()).expanduser().resolve()

    if src == dst:
        print(f"migrate-state-dir: source and destination are the same ({src!r}); nothing to do.")
        return 0

    src_db = src / "mvexp_state.db"
    src_store = src / "store"
    src_journal = src / "journal"

    plan: list[tuple[Path, Path, str]] = []
    if src_journal.is_dir():
        plan.append((src_journal, dst / "journal", "journal/"))
    if src_store.is_dir():
        plan.append((src_store, dst / "store", "store/"))
    if src_db.is_file():
        plan.append((src_db, dst / "mvexp_state.db", "mvexp_state.db"))

    if not plan:
        print(f"migrate-state-dir: nothing to migrate at {str(src)!r}.")
        return 0

    print(f"migrate-state-dir: {'APPLY' if args.apply else 'dry-run'}")
    print(f"  from: {str(src)!r}")
    print(f"  to:   {str(dst)!r}")
    for source, target, label in plan:
        existing = " (target exists)" if target.exists() else ""
        print(f"  - {label}: {str(source)!r} -> {str(target)!r}{existing}")

    if not args.apply:
        print("\nRe-run with --apply to perform the move.")
        return 0

    # Refuse to clobber a non-empty destination — the user must resolve.
    conflicts = [t for _, t, _ in plan if t.exists()]
    if conflicts:
        print(
            "migrate-state-dir: destination already has entries that would be "
            "overwritten:",
            file=sys.stderr,
        )
        for c in conflicts:
            print(f"  {str(c)!r}", file=sys.stderr)
        print(
            "Refusing to proceed. Remove or rename them, or pick a different --to.",
            file=sys.stderr,
        )
        return 2

    dst.mkdir(parents=True, exist_ok=True)
    # Move the DB last; a failure mid-flight in store/ leaves the legacy
    # install fully bootable.
    plan_ordered = sorted(plan, key=lambda item: item[2] == "mvexp_state.db")
    for source, target, label in plan_ordered:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.rename(target)
        except OSError:
            # Cross-filesystem; fall back to copy + remove.
            if source.is_dir():
                shutil.copytree(source, target)
                shutil.rmtree(source)
            else:
                shutil.copy2(source, target)
                source.unlink()
        print(f"  moved {label}")

    # Drop a one-line breadcrumb at the old location so a confused user
    # can find their data.
    breadcrumb = src / "MIGRATED_TO_STATE_ROOT.txt"
    if not breadcrumb.exists() and src != REPO_ROOT_GUESS.parent:
        try:
            breadcrumb.write_text(
                f"multiverse state migrated to {str(dst)!r} by "
                "`mvexp migrate-state-dir --apply`.\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    print(f"\nmigrate-state-dir: done. New state root: {str(dst)!r}")
    return 0


# ---------------------------------------------------------------------------
# Dispatch table for ``multiverse <cmd>``
# ---------------------------------------------------------------------------


def slurm_submit_main(argv: Optional[List[str]] = None) -> int:
    """``multiverse slurm-submit`` (STRATEGY M4).

    Submit one job through :class:`~multiverse.mvd.MvdSlurmExecutor`
    against the real Slurm engine on a login node. The shape is
    intentionally narrow — one job per invocation — so a user can wire
    this up in a shell loop without us reinventing the manifest parser
    for HPC.

    The broker is configured with ``max_inflight_dispatches`` so the
    M3 reservation ledger still records the dispatch.
    """
    parser = argparse.ArgumentParser(prog="multiverse slurm-submit")
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="State root (default: $MVEXP_STATE_DIR / config / $XDG_STATE_HOME/mvexp / $HOME/.mvexp).",
    )
    parser.add_argument("--model-slug", required=True)
    parser.add_argument(
        "--image-sif",
        required=True,
        type=Path,
        help="Path to a pre-built SIF; M2 dual-digest invariant applies.",
    )
    parser.add_argument(
        "--image-digest",
        default=None,
        help="OCI digest the SIF was built from (sha256:...).",
    )
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--dataset-n-obs", required=True, type=int)
    parser.add_argument("--dataset-n-vars", type=int, default=None)
    parser.add_argument(
        "--params-json",
        default="{}",
        help="JSON object of hyperparameters; defaults to '{}'.",
    )
    parser.add_argument(
        "--validators",
        choices=["basic", "strict", "developer"],
        default="basic",
    )
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--time-minutes", type=int, default=None)
    parser.add_argument("--mem-gb", type=int, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=8,
        help="Broker dispatch budget; M4 §2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--accept-degraded",
        action="store_true",
        default=False,
        help=(
            "Allow launching with a non-strict-acceptable image identity "
            "(e.g. unverified_local). Off by default — the M2 fail-closed default."
        ),
    )
    args = parser.parse_args(argv)

    import asyncio as _asyncio

    from .artifact import BootContext
    from .broker import HostMetrics, InMemoryHostObserver, ResourceBroker
    from .journal import JournalLayout, JournalWriter
    from .mvd import (
        Kernel,
        KernelConfig,
        MvdSlurmExecutor,
        PrimaryState,
        build_slurm_executor_options,
    )
    from .promotion import StoreLayout
    from .slurm import RealSlurmEngine

    try:
        params = json.loads(args.params_json)
    except json.JSONDecodeError as exc:
        print(f"--params-json is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(params, dict):
        print("--params-json must be a JSON object", file=sys.stderr)
        return 2

    state_root = args.state_root or _default_state_root()
    state_root.mkdir(parents=True, exist_ok=True)
    config = KernelConfig(
        state_root=state_root,
        accept_degraded=args.accept_degraded,
    )
    layout = JournalLayout.at(state_root / "journal").ensure()
    boot = BootContext.new(mvd_version="0.1.0-mvd")
    journal = JournalWriter(layout, boot_id=boot.boot_id, user_id=config.user_id)
    store = StoreLayout(root=state_root / "store").ensure()
    broker = ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=1, ram_total_bytes=1024)
        ),
        max_inflight_dispatches=args.max_inflight,
        journal=journal,
    )
    executor = MvdSlurmExecutor(
        journal=journal,
        boot=boot,
        store=store,
        engine=RealSlurmEngine(),
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

    options = build_slurm_executor_options(
        model_slug=args.model_slug,
        image_sif=str(args.image_sif),
        image_digest=args.image_digest,
        dataset_slug=args.dataset_slug,
        dataset_path=str(args.dataset_path),
        dataset_n_obs=args.dataset_n_obs,
        dataset_n_vars=args.dataset_n_vars,
        params=params,
        validators=args.validators,
        seed=args.seed,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        time_minutes=args.time_minutes,
        mem_gb=args.mem_gb,
        cpus_per_task=args.cpus_per_task,
        gpus=args.gpus,
    )

    async def _drive() -> int:
        attempt = await kernel.submit_run(
            manifest_path="(slurm-submit cli)", options=options
        )
        task = kernel._execution_tasks.get(attempt)  # type: ignore[attr-defined]
        if task is not None:
            try:
                await task
            except Exception as exc:  # noqa: BLE001 — surface, don't swallow
                print(
                    f"executor crashed for {attempt}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        snap = await kernel.query_run(physical_attempt_id=attempt)
        print(json.dumps(snap, indent=2, sort_keys=True, default=str))
        await kernel.shutdown()
        return 0 if snap.get("primary_state") == PrimaryState.ARTIFACT_SUCCESS.value else 1

    return _asyncio.run(_drive())


COMMANDS = {
    "doctor": doctor_main,
    "rebuild-index": rebuild_index_main,
    "gc": gc_main,
    "mlflow-sync": mlflow_sync_main,
    "migrate-state-dir": migrate_state_dir_main,
    "slurm-submit": slurm_submit_main,
}

RUNNER_COMMANDS = {
    "run",
    "register-dataset",
    "preprocess-dataset",
    "register-model",
    "init-db",
    "models",
}


def _runner_cli_main(cmd: str, argv: List[str]) -> int:
    """Delegate legacy runner/registration commands through the canonical
    top-level command without importing the runner at module load time.
    """
    from .runner import cli as runner_cli

    old_argv = sys.argv
    sys.argv = ["multiverse", cmd, *argv]
    try:
        try:
            result = runner_cli.main()
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            return int(code) if isinstance(code, int) else 1
        return int(result or 0)
    finally:
        sys.argv = old_argv


def main(argv: Optional[List[str]] = None) -> int:
    """Top-level entry point for the canonical ``multiverse`` command."""
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        _print_usage()
        return 0 if argv else 2
    cmd = argv.pop(0)
    handler = COMMANDS.get(cmd)
    if handler is not None:
        return handler(argv)
    if cmd in RUNNER_COMMANDS:
        return _runner_cli_main(cmd, argv)
    print(f"unknown command: {cmd}", file=sys.stderr)
    _print_usage()
    return 2


def _print_usage() -> None:
    print(
        "usage: multiverse <command> [options]\n"
        "\n"
        "commands:\n"
        "  run             execute a benchmark through the mvd-backed runner\n"
        "  register-dataset register a dataset manifest\n"
        "  register-model  register a model manifest\n"
        "  models          model registry/build commands\n"
        "  init-db         initialize local registry/index state\n"
        "  doctor          run diagnostics; --repair-health-probes invokes sweeper\n"
        "  rebuild-index   rebuild the SQLite index from journal + artifacts\n"
        "  gc              dry-run by default; --apply to delete (Tier-2 gates)\n"
        "  mlflow-sync     push an artifact bundle into MLflow\n"
        "  migrate-state-dir  move a pre-M1 state directory to the resolver location\n"
        "  slurm-submit    submit one job through MvdSlurmExecutor (M4)",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
