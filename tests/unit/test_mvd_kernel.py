"""Milestone-7 exit-gate tests for the mvd kernel.

Coverage:
    1. **Seven verbs** — the public surface matches R2 exactly. Adding a
       verb or removing one fails this test.
    2. Import-graph cleanliness — the kernel package does not pull in
       MLflow, Optuna, the GC plugin, the exporter, or Streamlit.
    3. submit/query/list end-to-end through ``SyntheticRunExecutor``.
    4. submit_run is idempotent under the same idempotency_key (R12).
    5. cancel_run flips the cancel flag and the executor honours it.
    6. State machine refuses illegal transitions.
    7. report_projection_status validates plugin + status against the
       allowed value sets and updates ``runs.projections`` without
       changing ``primary_state`` (R6 acceptance).
    8. health() reports the kernel snapshot without running probes.
    9. Paused kernel refuses submit_run (R1 maintenance lock).
   10. Replay reconstructs the registry from the journal.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import pytest

from multiverse.mvd import (
    KERNEL_VERBS,
    Kernel,
    KernelConfig,
    NullRunExecutor,
    PrimaryState,
    RunRegistry,
    SyntheticRunExecutor,
    assert_valid_transition,
)
from multiverse.mvd.api import KernelAPI
from multiverse.journal import JournalKind, JournalLayout, JournalWriter
from multiverse.mvd.runs import assert_projection_status_valid


# ---------------------------------------------------------------------------
# 1. Seven verbs
# ---------------------------------------------------------------------------


def test_kernel_exposes_exactly_seven_verbs() -> None:
    """R2 acceptance — adding a verb requires a strategy edit."""
    assert set(KERNEL_VERBS) == {
        "submit_run",
        "cancel_run",
        "query_run",
        "list_runs",
        "stream_events",
        "health",
        "report_projection_status",
    }
    # The Kernel class implements exactly these public, non-dunder methods
    # that are part of the protocol.
    api_methods = {
        name
        for name in dir(KernelAPI)
        if not name.startswith("_")
    }
    assert api_methods == set(KERNEL_VERBS)


def test_kernel_class_implements_all_seven_verbs() -> None:
    for verb in KERNEL_VERBS:
        assert hasattr(Kernel, verb), f"Kernel missing verb {verb}"
        attr = getattr(Kernel, verb)
        assert callable(attr), f"{verb} is not callable"


# ---------------------------------------------------------------------------
# 2. Import-graph cleanliness (R1 acceptance)
# ---------------------------------------------------------------------------


_FORBIDDEN_IN_KERNEL = ("mlflow", "optuna", "streamlit")


def test_kernel_does_not_import_forbidden_packages() -> None:
    """The kernel hot path imports neither MLflow, Optuna, nor Streamlit.

    Run in a subprocess so this test's sys.modules surgery does not
    invalidate enum identities used by sibling tests in the same module.
    """
    import subprocess

    script = (
        "import sys\n"
        "import multiverse.mvd\n"
        "forbidden = ('mlflow', 'optuna', 'streamlit')\n"
        "leaked = [m for m in forbidden if m in sys.modules]\n"
        "if leaked:\n"
        "    print(','.join(leaked))\n"
        "    raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"mvd kernel leaked imports: {result.stdout.strip()!r}\n"
        f"stderr: {result.stderr}"
    )


def test_kernel_source_has_no_forbidden_top_level_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    kernel_dir = root / "multiverse" / "mvd"
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(mlflow|optuna|streamlit)\b", re.MULTILINE
    )
    for source in kernel_dir.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        m = pattern.search(text)
        assert m is None, f"forbidden import {m.group(1)} in {source.name}"


# ---------------------------------------------------------------------------
# 3-6. End-to-end run lifecycle with SyntheticRunExecutor
# ---------------------------------------------------------------------------


@pytest.fixture
def kernel_success(tmp_path: Path) -> Kernel:
    return Kernel(
        KernelConfig(state_root=tmp_path / "state"),
        executor=SyntheticRunExecutor("success"),
    )


@pytest.fixture
def kernel_eval_fail(tmp_path: Path) -> Kernel:
    return Kernel(
        KernelConfig(state_root=tmp_path / "state"),
        executor=SyntheticRunExecutor("eval_fail"),
    )


async def _run_to_terminal(kernel: Kernel, attempt_id: str) -> dict:
    # Drive the kernel's scheduled execution task to completion.
    task = kernel._execution_tasks.get(attempt_id)  # type: ignore[attr-defined]
    if task is not None:
        await task
    return await kernel.query_run(physical_attempt_id=attempt_id)


def test_submit_drives_run_to_artifact_success(kernel_success: Kernel) -> None:
    async def _scenario() -> None:
        attempt = await kernel_success.submit_run(manifest_path="/tmp/m.yaml")
        snapshot = await _run_to_terminal(kernel_success, attempt)
        assert snapshot["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        listing = await kernel_success.list_runs(state="ARTIFACT_SUCCESS")
        assert any(r["physical_attempt_id"] == attempt for r in listing)
        await kernel_success.shutdown()

    asyncio.run(_scenario())


def test_eval_fail_leaves_run_in_recovery_pending(kernel_eval_fail: Kernel) -> None:
    async def _scenario() -> None:
        attempt = await kernel_eval_fail.submit_run(manifest_path="/tmp/m.yaml")
        snapshot = await _run_to_terminal(kernel_eval_fail, attempt)
        # The synthetic eval_fail outcome lands in RECOVERY_PENDING (S5).
        assert snapshot["primary_state"] == PrimaryState.RECOVERY_PENDING.value
        await kernel_eval_fail.shutdown()

    asyncio.run(_scenario())


def test_submit_is_idempotent_under_explicit_key(tmp_path: Path) -> None:
    kernel = Kernel(
        KernelConfig(state_root=tmp_path / "state"),
        executor=SyntheticRunExecutor("success"),
    )

    async def _scenario() -> None:
        a = await kernel.submit_run(
            manifest_path="/tmp/m.yaml",
            options={"idempotency_key": "stable-1"},
        )
        b = await kernel.submit_run(
            manifest_path="/tmp/m.yaml",
            options={"idempotency_key": "stable-1"},
        )
        assert a == b
        listing = await kernel.list_runs()
        assert len(listing) == 1
        await _run_to_terminal(kernel, a)
        await kernel.shutdown()

    asyncio.run(_scenario())


def test_cancel_run_honored_by_executor(tmp_path: Path) -> None:
    # Use an executor that pauses between steps so we have a chance to
    # request cancel before it finishes.
    class _PausingExecutor:
        name = "pausing"

        def __init__(self) -> None:
            self.cancel_seen = False

        async def execute(self, *, record, kernel) -> None:
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.ADMITTED
            )
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.RUNNING
            )
            # Yield so the test loop can request cancel.
            for _ in range(5):
                await asyncio.sleep(0)
                if record.cancel_requested:
                    self.cancel_seen = True
                    # RUNNING → CANCEL_REQUESTED → CANCELLED (state machine
                    # forbids direct RUNNING → CANCELLED jumps; see
                    # multiverse/mvd/state.py STATE_TRANSITIONS).
                    await kernel.transition(
                        record.physical_attempt_id,
                        to_state=PrimaryState.CANCEL_REQUESTED,
                    )
                    await kernel.transition(
                        record.physical_attempt_id,
                        to_state=PrimaryState.CANCELLED,
                    )
                    return
            # Should not reach here under normal test execution.
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.FAILED
            )

    executor = _PausingExecutor()
    kernel = Kernel(
        KernelConfig(state_root=tmp_path / "state"), executor=executor
    )

    async def _scenario() -> None:
        attempt = await kernel.submit_run(manifest_path="/tmp/m.yaml")
        await asyncio.sleep(0)
        await kernel.cancel_run(physical_attempt_id=attempt)
        snapshot = await _run_to_terminal(kernel, attempt)
        assert snapshot["primary_state"] == PrimaryState.CANCELLED.value
        assert executor.cancel_seen
        await kernel.shutdown()

    asyncio.run(_scenario())


def test_state_machine_refuses_illegal_transition() -> None:
    with pytest.raises(ValueError):
        assert_valid_transition(
            PrimaryState.PENDING, PrimaryState.ARTIFACT_SUCCESS
        )
    # Legal transitions don't raise.
    assert_valid_transition(PrimaryState.PENDING, PrimaryState.ADMITTED)


# ---------------------------------------------------------------------------
# 7. Projection status
# ---------------------------------------------------------------------------


def test_report_projection_status_updates_record_without_changing_primary_state(
    tmp_path: Path,
) -> None:
    kernel = Kernel(
        KernelConfig(state_root=tmp_path / "state"),
        executor=SyntheticRunExecutor("success"),
    )

    async def _scenario() -> None:
        attempt = await kernel.submit_run(manifest_path="/tmp/m.yaml")
        snapshot = await _run_to_terminal(kernel, attempt)
        before_state = snapshot["primary_state"]
        await kernel.report_projection_status(
            plugin="mlflow",
            physical_attempt_id=attempt,
            status="TRACKING_SYNC_FAILED",
            details={"error": "connection refused"},
        )
        after = await kernel.query_run(physical_attempt_id=attempt)
        assert after["primary_state"] == before_state, (
            "report_projection_status must not change primary_state (R6)"
        )
        assert after["projections"]["mlflow"] == "TRACKING_SYNC_FAILED"
        await kernel.shutdown()

    asyncio.run(_scenario())


def test_report_projection_status_validates_plugin_and_status() -> None:
    with pytest.raises(ValueError):
        assert_projection_status_valid("nonexistent_plugin", "TRACKING_SYNCED")
    with pytest.raises(ValueError):
        assert_projection_status_valid("mlflow", "WAT")
    # Allowed values do not raise.
    assert_projection_status_valid("mlflow", "TRACKING_SYNCED")


# ---------------------------------------------------------------------------
# 8. Health
# ---------------------------------------------------------------------------


def test_health_returns_kernel_snapshot(tmp_path: Path) -> None:
    kernel = Kernel(
        KernelConfig(state_root=tmp_path / "state", mvd_version="42.0.0"),
        executor=NullRunExecutor(),
    )

    async def _scenario() -> None:
        snapshot = await kernel.health()
        assert snapshot["ok"] is True
        assert snapshot["mvd_version"] == "42.0.0"
        assert snapshot["paused"] is False
        assert snapshot["executor"] == "null"
        await kernel.shutdown()

    asyncio.run(_scenario())


# ---------------------------------------------------------------------------
# 9. Paused-kernel maintenance lock
# ---------------------------------------------------------------------------


def test_paused_kernel_refuses_submit(tmp_path: Path) -> None:
    kernel = Kernel(
        KernelConfig(state_root=tmp_path / "state", paused=True),
        executor=SyntheticRunExecutor("success"),
    )

    async def _scenario() -> None:
        with pytest.raises(RuntimeError):
            await kernel.submit_run(manifest_path="/tmp/m.yaml")
        await kernel.shutdown()

    asyncio.run(_scenario())


# ---------------------------------------------------------------------------
# 10. Replay reconstructs registry
# ---------------------------------------------------------------------------


def test_replay_reconstructs_registry_from_journal(tmp_path: Path) -> None:
    state_root = tmp_path / "state"

    # First boot: submit some runs, drive them, then shut down.
    async def _first_boot() -> tuple[str, str]:
        k1 = Kernel(
            KernelConfig(state_root=state_root),
            executor=SyntheticRunExecutor("success"),
        )
        a = await k1.submit_run(manifest_path="/tmp/a.yaml")
        b = await k1.submit_run(manifest_path="/tmp/b.yaml")
        await _run_to_terminal(k1, a)
        await _run_to_terminal(k1, b)
        await k1.shutdown()
        return a, b

    a_id, b_id = asyncio.run(_first_boot())

    # Second boot: do NOT execute, just replay and inspect.
    async def _second_boot() -> None:
        k2 = Kernel(
            KernelConfig(state_root=state_root),
            executor=NullRunExecutor(),
        )
        k2.replay_from_journal()
        assert (await k2.query_run(physical_attempt_id=a_id))[
            "primary_state"
        ] == PrimaryState.ARTIFACT_SUCCESS.value
        assert (await k2.query_run(physical_attempt_id=b_id))[
            "primary_state"
        ] == PrimaryState.ARTIFACT_SUCCESS.value
        await k2.shutdown()

    asyncio.run(_second_boot())


def test_replay_restores_persisted_run_metadata(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    layout = JournalLayout.at(state_root / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-test", fsync_enabled=False)
    attempt = "attempt-1"
    writer.append(
        JournalKind.JOB_INTENT,
        physical_attempt_id=attempt,
        logical_run_id="logical-1",
        payload={
            "manifest_path": "/tmp/manifest.yaml",
            "options": {
                "idempotency_key": "idem-1",
                "dataset_slug": "cells",
                "model_slug": "vae",
            },
        },
        next_state=PrimaryState.PENDING.value,
    )
    writer.append(
        JournalKind.CONTAINER_LAUNCH,
        physical_attempt_id=attempt,
        logical_run_id="logical-1",
        payload={
            "labels": {"multiverse.workspace": str(state_root / "workspaces" / attempt)}
        },
    )
    writer.append(
        JournalKind.STATE_TRANSITION,
        physical_attempt_id=attempt,
        logical_run_id="logical-1",
        payload={
            "from_state": PrimaryState.RUNNING.value,
            "to_state": PrimaryState.EVALUATION_FAILED.value,
            "reason": "metrics.json missing",
        },
    )
    writer.append(
        JournalKind.PROMOTE_PREPARE,
        physical_attempt_id=attempt,
        logical_run_id="logical-1",
        payload={
            "workspace_dir": str(state_root / "workspace-final"),
            "final_artifact_dir": str(state_root / "store" / "artifacts" / attempt),
        },
    )
    writer.append(
        JournalKind.PROMOTE_COMMIT_MANIFEST,
        physical_attempt_id=attempt,
        logical_run_id="logical-1",
        payload={"artifact_dir": str(state_root / "store" / "artifacts" / attempt)},
    )
    writer.append(
        JournalKind.PROJECTION_STATUS,
        physical_attempt_id=attempt,
        logical_run_id="logical-1",
        payload={"plugin": "mlflow", "status": "TRACKING_SYNCED"},
    )
    writer.commit()
    writer.close()

    kernel = Kernel(KernelConfig(state_root=state_root), executor=NullRunExecutor())

    async def _scenario() -> None:
        kernel.replay_from_journal()
        snapshot = await kernel.query_run(physical_attempt_id=attempt)
        assert snapshot["logical_run_id"] == "logical-1"
        assert snapshot["manifest_path"] == "/tmp/manifest.yaml"
        assert snapshot["primary_state"] == PrimaryState.EVALUATION_FAILED.value
        assert snapshot["failure_reason"] == "metrics.json missing"
        assert snapshot["workspace_dir"].endswith("workspace-final")
        assert snapshot["artifact_dir"].endswith(f"store/artifacts/{attempt}")
        assert snapshot["options"]["dataset_slug"] == "cells"
        assert snapshot["projections"]["mlflow"] == "TRACKING_SYNCED"
        await kernel.shutdown()

    asyncio.run(_scenario())


# ---------------------------------------------------------------------------
# 11. Run registry helpers
# ---------------------------------------------------------------------------


def test_run_registry_list_filters() -> None:
    from multiverse.mvd.runs import new_run_record

    registry = RunRegistry()
    pending = new_run_record(physical_attempt_id="p", manifest_path="/p.yaml")
    success = new_run_record(physical_attempt_id="s", manifest_path="/s.yaml")
    success.primary_state = PrimaryState.ARTIFACT_SUCCESS
    registry.add(pending)
    registry.add(success)

    assert [r.physical_attempt_id for r in registry.list()] == ["p", "s"]
    assert [r.physical_attempt_id for r in registry.list(state=PrimaryState.PENDING)] == ["p"]
    assert [
        r.physical_attempt_id for r in registry.list(state=PrimaryState.ARTIFACT_SUCCESS)
    ] == ["s"]
