"""Tests for the Slurm state model + sacct state parsing (STRATEGY M4)."""

from __future__ import annotations

import pytest

from multiverse.slurm import SlurmJobState, from_sacct_state

pytestmark = pytest.mark.control_plane


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("COMPLETED", SlurmJobState.COMPLETED),
        ("RUNNING", SlurmJobState.RUNNING),
        ("PENDING", SlurmJobState.PENDING),
        ("FAILED", SlurmJobState.FAILED),
        ("CANCELLED by 1001", SlurmJobState.CANCELLED),
        ("TIMEOUT", SlurmJobState.TIMEOUT),
        ("OUT_OF_MEMORY", SlurmJobState.OUT_OF_MEMORY),
        ("NODE_FAIL", SlurmJobState.NODE_FAIL),
        ("PREEMPTED", SlurmJobState.PREEMPTED),
        ("RUNNING+", SlurmJobState.RUNNING),  # array-suffix tolerated
        ("", SlurmJobState.UNKNOWN),
        ("BOGUS", SlurmJobState.UNKNOWN),
    ],
)
def test_from_sacct_state(raw: str, expected: SlurmJobState) -> None:
    assert from_sacct_state(raw) is expected


def test_terminality_partitions_correctly() -> None:
    non_terminal = {SlurmJobState.PENDING, SlurmJobState.RUNNING}
    for state in SlurmJobState:
        assert state.is_terminal != (state in non_terminal)


def test_oom_is_failure_and_terminal() -> None:
    assert SlurmJobState.OUT_OF_MEMORY.is_terminal
    assert SlurmJobState.OUT_OF_MEMORY.is_failure


def test_completed_is_terminal_not_failure() -> None:
    assert SlurmJobState.COMPLETED.is_terminal
    assert not SlurmJobState.COMPLETED.is_failure
