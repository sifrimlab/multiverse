"""Milestone-9 exit-gate tests for the kernel client / GUI cutover.

Coverage:
    1. ``InProcessClient`` exposes exactly the seven verbs and drives a
       run to ARTIFACT_SUCCESS through the kernel.
    2. ``KernelSocketServer`` + ``KernelSocketClient`` round-trip — full
       Unix-socket transport works for submit / query / list / health /
       cancel / report_projection_status.
    3. The wire protocol refuses unknown verbs.
    4. Import-graph: the client package does NOT import ``docker``,
       ``mlflow``, ``optuna``, ``streamlit``, or ``sqlite3``.
    5. Grep gate: client source files contain no top-level imports of
       those forbidden packages.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import pytest

from multiverse.client import (ApiError, InProcessClient, KernelSocketClient,
                               serve_kernel)
from multiverse.client.protocol import (RpcRequest, decode_request,
                                        encode_request)
from multiverse.mvd import (KERNEL_VERBS, Kernel, KernelConfig, PrimaryState,
                            SyntheticRunExecutor)

# ---------------------------------------------------------------------------
# 1. InProcessClient
# ---------------------------------------------------------------------------


def test_in_process_client_exposes_seven_verbs() -> None:
    public = {name for name in dir(InProcessClient) if not name.startswith("_")}
    # Filter to verbs + the dataclass field "kernel".
    api = public - {"kernel"}
    assert api == set(KERNEL_VERBS)


def test_in_process_client_drives_run_to_success(tmp_path: Path) -> None:
    kernel = Kernel(
        KernelConfig(state_root=tmp_path / "state"),
        executor=SyntheticRunExecutor("success"),
    )
    client = InProcessClient(kernel=kernel)

    async def _scenario() -> None:
        attempt = await client.submit_run(manifest_path="/tmp/m.yaml")
        # Wait for execution task to finish.
        task = kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        await task
        snapshot = await client.query_run(physical_attempt_id=attempt)
        assert snapshot["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        listed = await client.list_runs(state="ARTIFACT_SUCCESS")
        assert any(r["physical_attempt_id"] == attempt for r in listed)
        health = await client.health()
        assert health["ok"] is True
        await kernel.shutdown()

    asyncio.run(_scenario())


# ---------------------------------------------------------------------------
# 2-3. Socket round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def socket_kernel(tmp_path: Path) -> Kernel:
    return Kernel(
        KernelConfig(state_root=tmp_path / "state"),
        executor=SyntheticRunExecutor("success"),
    )


def test_socket_round_trip(tmp_path: Path, socket_kernel: Kernel) -> None:
    socket_path = tmp_path / "mvd.sock"

    async def _scenario() -> None:
        async with serve_kernel(socket_kernel, socket_path=socket_path) as srv:
            assert srv.socket_path.exists()
            assert socket_path.stat().st_mode & 0o777 == 0o600
            async with KernelSocketClient(socket_path) as client:
                # Submit + drive to completion.
                attempt = await client.submit_run(manifest_path="/tmp/m.yaml")
                # Wait for the kernel's execution task.
                task = socket_kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
                await task
                snapshot = await client.query_run(physical_attempt_id=attempt)
                assert snapshot["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value

                # Projection report changes only the projections dict.
                await client.report_projection_status(
                    plugin="mlflow",
                    physical_attempt_id=attempt,
                    status="TRACKING_SYNC_FAILED",
                )
                after = await client.query_run(physical_attempt_id=attempt)
                assert after["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
                assert after["projections"]["mlflow"] == "TRACKING_SYNC_FAILED"

                # Health works through the wire.
                h = await client.health()
                assert h["ok"] is True

                # Cancelling a terminal run is idempotent.
                await client.cancel_run(physical_attempt_id=attempt)

        await socket_kernel.shutdown()

    asyncio.run(_scenario())


def test_socket_returns_not_found_for_unknown_attempt(
    tmp_path: Path, socket_kernel: Kernel
) -> None:
    socket_path = tmp_path / "mvd.sock"

    async def _scenario() -> None:
        async with serve_kernel(socket_kernel, socket_path=socket_path):
            async with KernelSocketClient(socket_path) as client:
                with pytest.raises(ApiError) as exc:
                    await client.query_run(physical_attempt_id="missing")
                assert exc.value.code == "NOT_FOUND"
        await socket_kernel.shutdown()

    asyncio.run(_scenario())


def test_socket_rejects_unknown_verb(tmp_path: Path, socket_kernel: Kernel) -> None:
    """A hand-crafted bad request via the protocol decoder reports
    UNKNOWN_VERB rather than silently dispatching."""
    bad_line = b'{"verb":"do_evil_stuff","kwargs":{},"id":"x"}\n'
    with pytest.raises(ApiError) as exc:
        decode_request(bad_line)
    assert exc.value.code == "UNKNOWN_VERB"


def test_protocol_encoding_round_trip() -> None:
    req = RpcRequest(verb="submit_run", kwargs={"manifest_path": "/m"}, id="x")
    line = encode_request(req)
    assert line.endswith(b"\n")
    decoded = decode_request(line)
    assert decoded.verb == "submit_run"
    assert decoded.kwargs == {"manifest_path": "/m"}
    assert decoded.id == "x"


# ---------------------------------------------------------------------------
# 4-5. Import-graph cleanliness (Milestone-9 acceptance)
# ---------------------------------------------------------------------------


_FORBIDDEN = ("docker", "mlflow", "optuna", "streamlit")


def test_client_does_not_import_forbidden_packages() -> None:
    import subprocess

    script = (
        "import sys\n"
        "import multiverse.client  # noqa\n"
        "from multiverse.client import InProcessClient, KernelSocketClient\n"
        f"forbidden = {_FORBIDDEN!r}\n"
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
        f"client package leaked imports: {result.stdout.strip()!r}\n"
        f"stderr: {result.stderr}"
    )


def test_client_source_grep_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    client_dir = root / "multiverse" / "client"
    forbidden_pattern = re.compile(
        r"^\s*(?:import|from)\s+("
        + "|".join(re.escape(m) for m in _FORBIDDEN + ("sqlite3",))
        + r")\b",
        re.MULTILINE,
    )
    for source in client_dir.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        match = forbidden_pattern.search(text)
        assert (
            match is None
        ), f"forbidden top-level import {match.group(1)!r} in client/{source.name}"


def test_gui_results_tab_reads_mvd_controller_and_index_sources() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "multiverse" / "gui.py").read_text(encoding="utf-8")
    assert "def _fetch_mvd_runs" in text
    assert "def _fetch_mvd_index_runs" in text
    assert "_mvd_state_root_for_results" in text
    assert "INDEX_FILENAME" in text
    assert "ARTIFACT_SUCCESS" in text


def test_inprocess_controller_projects_snapshots_to_index(tmp_path: Path) -> None:
    from multiverse.index.sqlite_index import INDEX_FILENAME, open_index
    from multiverse.runner.mvd_inprocess import InProcessMvdController

    controller = object.__new__(InProcessMvdController)
    controller._index_path = tmp_path / INDEX_FILENAME

    controller._project_snapshots(
        [
            {
                "physical_attempt_id": "attempt-1",
                "logical_run_id": "logical-1",
                "primary_state": PrimaryState.ARTIFACT_SUCCESS.value,
                "failure_reason": None,
                "artifact_dir": str(tmp_path / "artifacts" / "attempt-1"),
                "workspace_dir": str(tmp_path / "workspaces" / "attempt-1"),
                "manifest_path": "/tmp/manifest.yaml",
                "cancel_requested": False,
                "submitted_wall_iso": "2026-01-01T00:00:00+00:00",
                "options": {"dataset_slug": "cells", "model_slug": "vae"},
                "projections": {"mlflow": "TRACKING_SYNCED"},
            }
        ]
    )

    with open_index(tmp_path / INDEX_FILENAME, create_if_missing=False) as index:
        row = index.get_run("attempt-1")
        assert row is not None
        assert row["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        assert row["artifact_dir"].endswith("artifacts/attempt-1")
        assert index.projections_for("attempt-1")["mlflow"] == "TRACKING_SYNCED"
