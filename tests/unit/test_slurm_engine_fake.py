"""InMemorySlurmEngine lifecycle tests (STRATEGY M4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse.slurm import (
    InMemorySlurmEngine,
    SlurmEngine,
    SlurmJobSpec,
    SlurmJobState,
)


pytestmark = pytest.mark.control_plane


def _spec(workspace: Path) -> SlurmJobSpec:
    return SlurmJobSpec(
        job_name="mvd-test",
        image_sif=Path("/sif/model.sif"),
        workspace=workspace,
        dataset_path=workspace / "data.h5mu",
        command=["python", "-m", "model.run"],
        env={},
        cpus_per_task=1,
    )


def test_in_memory_engine_satisfies_protocol() -> None:
    assert isinstance(InMemorySlurmEngine(), SlurmEngine)


def test_submit_assigns_sequential_ids_and_writes_script(tmp_path: Path) -> None:
    engine = InMemorySlurmEngine()
    a = engine.submit(_spec(tmp_path / "a"), script_dir=tmp_path / "scripts")
    b = engine.submit(_spec(tmp_path / "b"), script_dir=tmp_path / "scripts")
    assert a.job_id != b.job_id
    assert int(b.job_id) == int(a.job_id) + 1
    assert a.script_path.is_file()
    assert "--job-name=mvd-test" in a.script_path.read_text()


def test_query_pending_until_simulated(tmp_path: Path) -> None:
    engine = InMemorySlurmEngine()
    sub = engine.submit(_spec(tmp_path / "ws"), script_dir=tmp_path / "scripts")
    assert engine.query(sub.job_id).state is SlurmJobState.PENDING
    engine.simulate_running(sub.job_id)
    assert engine.query(sub.job_id).state is SlurmJobState.RUNNING
    engine.simulate_completed(sub.job_id)
    info = engine.query(sub.job_id)
    assert info.state is SlurmJobState.COMPLETED
    assert info.exit_code == 0


def test_cancel_terminates_nonterminal_jobs(tmp_path: Path) -> None:
    engine = InMemorySlurmEngine()
    sub = engine.submit(_spec(tmp_path / "ws"), script_dir=tmp_path / "scripts")
    engine.simulate_running(sub.job_id)
    engine.cancel(sub.job_id)
    info = engine.query(sub.job_id)
    assert info.state is SlurmJobState.CANCELLED


def test_cancel_does_not_overwrite_terminal_state(tmp_path: Path) -> None:
    engine = InMemorySlurmEngine()
    sub = engine.submit(_spec(tmp_path / "ws"), script_dir=tmp_path / "scripts")
    engine.simulate_completed(sub.job_id)
    engine.cancel(sub.job_id)
    info = engine.query(sub.job_id)
    assert info.state is SlurmJobState.COMPLETED


def test_simulate_oom_classifies_as_out_of_memory(tmp_path: Path) -> None:
    engine = InMemorySlurmEngine()
    sub = engine.submit(_spec(tmp_path / "ws"), script_dir=tmp_path / "scripts")
    engine.simulate_oom(sub.job_id)
    info = engine.query(sub.job_id)
    assert info.state is SlurmJobState.OUT_OF_MEMORY
    assert info.oom_killed
    assert info.is_terminal


def test_query_unknown_job_returns_pending(tmp_path: Path) -> None:
    engine = InMemorySlurmEngine()
    info = engine.query("999999")
    assert info.state is SlurmJobState.PENDING


def test_submit_count_tracks_dispatches(tmp_path: Path) -> None:
    engine = InMemorySlurmEngine()
    for i in range(5):
        engine.submit(_spec(tmp_path / f"ws{i}"), script_dir=tmp_path / "scripts")
    assert engine.submit_count == 5
