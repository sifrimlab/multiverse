"""Tests for the doctor's state-paths probe (STRATEGY M1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse import state_paths
from multiverse.doctor.health_probes import ProbeOutcome
from multiverse.doctor.state_paths_probe import probe_state_root

pytestmark = pytest.mark.control_plane


def test_probe_passes_for_user_writable_path(tmp_path, monkeypatch):
    # Suppress legacy detection so this test is hermetic.
    monkeypatch.setattr(state_paths, "find_legacy_db", lambda: None)
    from multiverse.doctor import state_paths_probe as probe_module

    monkeypatch.setattr(probe_module, "find_legacy_db", lambda: None)

    report = probe_state_root(tmp_path / "my-state")
    assert report.probe is ProbeOutcome.PASS


def test_probe_fails_when_state_root_inside_package_dir(monkeypatch):
    from multiverse.doctor import state_paths_probe as probe_module

    monkeypatch.setattr(probe_module, "find_legacy_db", lambda: None)

    bad = state_paths.PACKAGE_DIR / "state"
    report = probe_state_root(bad)
    assert report.probe is ProbeOutcome.FAIL
    assert "inside the package directory" in (report.detail or "")


def test_probe_fails_when_legacy_db_would_be_orphaned(tmp_path, monkeypatch):
    legacy = tmp_path / "old-install" / "multiverse_state.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"")
    from multiverse.doctor import state_paths_probe as probe_module

    monkeypatch.setattr(probe_module, "find_legacy_db", lambda: legacy)

    report = probe_state_root(tmp_path / "new-state")
    assert report.probe is ProbeOutcome.FAIL
    assert "legacy database" in (report.detail or "")
    assert str(legacy) in (report.detail or "")


def test_probe_passes_when_legacy_db_matches_chosen_state_root(tmp_path, monkeypatch):
    """If the user explicitly chose state_root == legacy parent, that's
    a deliberate keep-using-the-old-place choice, not an orphan."""
    legacy_parent = tmp_path / "old"
    legacy_parent.mkdir()
    legacy = legacy_parent / "multiverse_state.db"
    legacy.write_bytes(b"")
    from multiverse.doctor import state_paths_probe as probe_module

    monkeypatch.setattr(probe_module, "find_legacy_db", lambda: legacy)

    report = probe_state_root(legacy_parent)
    assert report.probe is ProbeOutcome.PASS
