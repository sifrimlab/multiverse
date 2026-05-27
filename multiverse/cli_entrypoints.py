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


# ---------------------------------------------------------------------------
# multiverse doctor
# ---------------------------------------------------------------------------


def doctor_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="multiverse doctor")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./store"),
        help="Store root to probe (default ./store).",
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
    args = parser.parse_args(argv)

    from .doctor import run_storage_probes, sweep_expired_health_probes
    from .doctor.health_probes import probe_workspace_directory
    from .doctor.report import DoctorReport, DoctorSection, SectionStatus

    storage = run_storage_probes(args.root)
    storage_section = DoctorSection(
        name="storage",
        status=_storage_status(storage),
        rows=[r.to_dict() for r in storage.results],
        summary=f"worst-level={storage.worst_level.value}",
    )
    # Workspace health probe.
    workspaces_root = args.root / "workspaces"
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

    sweep_report: dict = {}
    if args.repair_health_probes:
        sweep_report = sweep_expired_health_probes(workspaces_root)

    report = DoctorReport(
        sections=[storage_section, health_section],
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
        default=Path("./state"),
        help="State root containing journal/ (default ./state).",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path("./store"),
        help="Store root containing artifacts/ (default ./store).",
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
    args = parser.parse_args(argv)

    from .index import INDEX_FILENAME, open_index, rebuild_index
    from .promotion import StoreLayout

    db_path = args.db or (args.state_root / INDEX_FILENAME)
    args.state_root.mkdir(parents=True, exist_ok=True)
    store = StoreLayout(root=args.store_root).ensure()

    with open_index(db_path) as idx:
        result = rebuild_index(
            index=idx,
            state_root=args.state_root,
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
        "--store-root", type=Path, default=Path("./store"), help="Store root."
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

    store = StoreLayout(root=args.store_root).ensure()
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
# Dispatch table for ``multiverse <cmd>``
# ---------------------------------------------------------------------------


COMMANDS = {
    "doctor": doctor_main,
    "rebuild-index": rebuild_index_main,
    "gc": gc_main,
    "mlflow-sync": mlflow_sync_main,
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
        "  mlflow-sync     push an artifact bundle into MLflow",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
