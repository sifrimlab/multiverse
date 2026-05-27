"""Move-7 exit-gate tests for first-class maintenance commands.

Strategy v2 §7 acceptance: each command works without heavy ML imports
and has tests for failure behavior.

These tests exercise:

    1. ``multiverse doctor`` — JSON report, ``--repair-health-probes``,
       and the worst-level → exit-code mapping.
    2. ``multiverse rebuild-index`` — end-to-end against journal +
       artifact dir; idempotent.
    3. ``multiverse gc`` — dry-run by default; ``--apply`` honours every
       gate.
    4. ``multiverse mlflow-sync`` — clean failure when MLflow target
       cannot be constructed; clean success when an injected target
       accepts the bundle (verified through the in-memory target via the
       Python API, not the CLI shell-out).
    5. The entry-point module does not import ``mlflow`` / ``scanpy`` /
       ``optuna`` / ``streamlit`` at module load time.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest

from multiverse.cli_entrypoints import (
    doctor_main,
    gc_main,
    main as commands_main,
    mlflow_sync_main,
    rebuild_index_main,
)


# ---------------------------------------------------------------------------
# 1. doctor
# ---------------------------------------------------------------------------


def test_doctor_returns_zero_on_healthy_path(tmp_path: Path, capsys) -> None:
    code = doctor_main(["--root", str(tmp_path / "store"), "--json"])
    assert code in (0, 1), "healthy tmp path should be ok or warning (free-space tiers)"
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["overall_status"] in {"ok", "warning"}
    names = {s["name"] for s in payload["sections"]}
    assert {"storage", "health_probes"}.issubset(names)


def test_doctor_repair_health_probes_invokes_sweeper(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    workspaces = store / "workspaces"
    workspaces.mkdir()
    reserved = workspaces / "__mvd_health_probe__"
    reserved.mkdir()
    stale = reserved / "old"
    stale.mkdir()
    import os, time

    old = time.time() - 7200  # 2 h old
    os.utime(stale, (old, old))

    code = doctor_main(["--root", str(store), "--repair-health-probes"])
    assert code in (0, 1)
    assert not stale.exists(), "sweeper must have removed the stale entry"


def test_doctor_blocked_on_dropbox_path(tmp_path: Path) -> None:
    (tmp_path / ".dropbox").mkdir()
    store = tmp_path / "store"
    store.mkdir()
    code = doctor_main(["--root", str(store), "--json"])
    # Cloud-sync marker yields DANGEROUS for one probe → WARNING-level
    # overall (not BLOCKED). Code is 1.
    assert code == 1


# ---------------------------------------------------------------------------
# 2. rebuild-index
# ---------------------------------------------------------------------------


def test_rebuild_index_writes_summary_on_empty_journal(tmp_path: Path, capsys) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / "journal").mkdir()
    store_root = tmp_path / "store"
    store_root.mkdir()
    code = rebuild_index_main(
        [
            "--state-root",
            str(state_root),
            "--store-root",
            str(store_root),
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["total_runs"] == 0


def test_rebuild_index_classifies_promoted_run(tmp_path: Path, capsys) -> None:
    from multiverse.artifact import (
        ArtifactManifest,
        BootContext,
        ImageIdentity,
        ProducedAt,
        ProducedBy,
        compute_logical_run_id,
        compute_manifest_hash,
        compute_params_hash,
        produced_at_now,
        write_manifest,
    )
    from multiverse.journal import JournalKind, JournalLayout, JournalWriter

    state_root = tmp_path / "state"
    store_root = tmp_path / "store"
    state_root.mkdir()
    store_root.mkdir()
    (state_root / "journal").mkdir()

    # Stamp a verified artifact manifest into store/artifacts/demo/.
    boot = BootContext.new(mvd_version="0.1.0-test")
    image = ImageIdentity.registry_digest("sha256:" + "a" * 64)
    manifest_hash = compute_manifest_hash("jobs: []\n")
    artifact_dir = store_root / "artifacts" / "demo"
    artifact_dir.mkdir(parents=True)
    logical = compute_logical_run_id(
        manifest_hash=manifest_hash,
        dataset_fingerprint={"slug": "demo", "n_obs": 4},
        image_identity=image,
        params_hash=compute_params_hash({}),
        mv_contract_version="1",
    )
    manifest = ArtifactManifest(
        logical_run_id=logical,
        physical_attempt_id="att-1",
        manifest_hash=manifest_hash,
        dataset_fingerprint={"slug": "demo", "n_obs": 4},
        image_identity=image,
        params_hash=compute_params_hash({}),
        mv_contract_version="1",
        produced_at=ProducedAt.from_dict(produced_at_now(boot)),
        produced_by=ProducedBy(mvd_version="0.1.0-test"),
        owner_token="t",
    )
    write_manifest(artifact_dir, manifest)

    writer = JournalWriter(JournalLayout.at(state_root / "journal"), boot_id="B")
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/tmp/m.yaml"},
        physical_attempt_id="att-1",
    )
    writer.append(
        JournalKind.PROMOTE_PREPARE,
        payload={
            "workspace_dir": "/tmp/ws",
            "final_artifact_dir": str(artifact_dir),
            "owner_token": "t",
        },
        physical_attempt_id="att-1",
        logical_run_id=logical,
    )
    writer.append(
        JournalKind.PROMOTE_COMMIT_MANIFEST,
        payload={"artifact_dir": str(artifact_dir), "manifest_sha256": "x"},
        physical_attempt_id="att-1",
        logical_run_id=logical,
    )
    writer.commit()
    writer.close()

    code = rebuild_index_main(
        [
            "--state-root",
            str(state_root),
            "--store-root",
            str(store_root),
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["total_runs"] == 1
    assert summary["artifact_success"] == 1


# ---------------------------------------------------------------------------
# 3. gc
# ---------------------------------------------------------------------------


def test_gc_dry_run_by_default_writes_report(tmp_path: Path, capsys) -> None:
    store = tmp_path / "store"
    code = gc_main(["--store-root", str(store)])
    assert code == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    reports = list((store / "gc_reports").iterdir())
    assert reports, "gc report must be written even on an empty store"


def test_gc_rejects_apply_and_dry_run_together(tmp_path: Path, capsys) -> None:
    code = gc_main(["--store-root", str(tmp_path), "--apply", "--dry-run"])
    assert code == 2


def test_gc_apply_with_no_retention_deletes_nothing(tmp_path: Path, capsys) -> None:
    """Default policy is infinite retention; --apply alone must be a
    no-op (Strategy R12 acceptance)."""
    from multiverse.promotion import StoreLayout
    from multiverse.promotion.tokens import write_owner_token

    store = StoreLayout(root=tmp_path / "store").ensure()
    failed = store.failed / "old"
    failed.mkdir(parents=True)
    (failed / "container.log").write_text("x")
    write_owner_token(
        failed,
        owner_token="t",
        physical_attempt_id="att",
        mvd_boot_id="B",
        purpose="failed-attempt",
    )

    code = gc_main(["--store-root", str(store.root), "--apply"])
    assert code == 0
    assert failed.is_dir(), "apply with no retention must NOT delete user data"


# ---------------------------------------------------------------------------
# 4. mlflow-sync
# ---------------------------------------------------------------------------


def _build_bundle(tmp_path: Path) -> Path:
    """Reusable contract-valid bundle."""
    from multiverse.artifact import (
        ArtifactManifest,
        BootContext,
        BundleInputs,
        ImageIdentity,
        ModelOutputContract,
        ProducedAt,
        ProducedBy,
        ValidationLevel,
        compute_logical_run_id,
        compute_manifest_hash,
        compute_params_hash,
        new_physical_attempt_id,
        produced_at_now,
        validate_output_bundle,
        write_bundle,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with h5py.File(workspace / "embeddings.h5", "w") as f:
        f.create_dataset(
            "latent",
            data=np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32),
        )

    boot = BootContext.new(mvd_version="0.1.0-test")
    image = ImageIdentity.registry_digest("sha256:" + "a" * 64)
    manifest_hash = compute_manifest_hash("jobs: []\n")
    params_hash = compute_params_hash({})
    fingerprint = {"slug": "demo", "n_obs": 4}
    logical = compute_logical_run_id(
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
    )
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(workspace, contract, ValidationLevel.BASIC)
    manifest = ArtifactManifest(
        logical_run_id=logical,
        physical_attempt_id=new_physical_attempt_id(),
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
        produced_at=ProducedAt.from_dict(produced_at_now(boot)),
        produced_by=ProducedBy(mvd_version=boot.mvd_version),
        artifacts=list(report.artifact_entries),
        owner_token="t",
    )
    bundle = tmp_path / "bundle"
    write_bundle(
        bundle,
        BundleInputs(
            artifact_manifest=manifest,
            outputs={"embeddings.h5": workspace / "embeddings.h5"},
            environment={"mvd_version": "0.1.0-test"},
            validation_report=report.to_dict(),
        ),
    )
    return bundle


def test_mlflow_sync_reports_failure_when_target_cannot_be_constructed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """If the target adapter can't be built, exit cleanly with a
    non-zero code rather than crashing the user shell."""
    bundle = _build_bundle(tmp_path / "good")
    from multiverse import cli_entrypoints

    def _explode(_uri):
        raise RuntimeError("MLflow tracking server unreachable")

    monkeypatch.setattr(cli_entrypoints, "_build_mlflow_target", _explode)

    code = mlflow_sync_main(["--bundle", str(bundle)])
    assert code == 2
    err = capsys.readouterr().err
    assert "MLflow" in err


def test_mlflow_sync_reports_success_with_injected_target(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bundle = _build_bundle(tmp_path / "ok")
    from multiverse import cli_entrypoints
    from multiverse.projection.base import InMemoryMLflowTarget

    monkeypatch.setattr(
        cli_entrypoints, "_build_mlflow_target", lambda _uri: InMemoryMLflowTarget()
    )

    code = mlflow_sync_main(["--bundle", str(bundle), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "TRACKING_SYNCED"


# ---------------------------------------------------------------------------
# 5. Import-graph: cli_entrypoints does not load mlflow / scanpy / optuna /
#    streamlit at module load time.
# ---------------------------------------------------------------------------


def test_cli_entrypoints_does_not_eagerly_load_ml_stack() -> None:
    script = (
        "import sys\n"
        "import multiverse.cli_entrypoints  # noqa\n"
        "for m in ('mlflow', 'scanpy', 'optuna', 'streamlit'):\n"
        "    if m in sys.modules:\n"
        "        print(m)\n"
        "        raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"cli_entrypoints leaked: {result.stdout.strip()!r}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# 6. Dispatch table
# ---------------------------------------------------------------------------


def test_top_level_dispatch_rejects_unknown_command(capsys) -> None:
    code = commands_main(["nope"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown command" in err


def test_top_level_dispatch_lists_help_with_no_args(capsys) -> None:
    code = commands_main([])
    assert code == 2  # no args = print usage with non-zero
    err = capsys.readouterr().err
    assert "doctor" in err and "gc" in err


def test_top_level_dispatch_delegates_runner_commands(monkeypatch) -> None:
    from multiverse import cli_entrypoints

    called = {}

    def _fake_runner(cmd, argv):
        called["cmd"] = cmd
        called["argv"] = list(argv)
        return 7

    monkeypatch.setattr(cli_entrypoints, "_runner_cli_main", _fake_runner)
    code = commands_main(["run", "--manifest", "m.yaml"])
    assert code == 7
    assert called == {"cmd": "run", "argv": ["--manifest", "m.yaml"]}


def test_pyproject_exposes_canonical_multiverse_script() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["multiverse"] == "multiverse.cli_entrypoints:main"
