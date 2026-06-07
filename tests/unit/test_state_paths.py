"""Tests for the M1 state-path resolver and the legacy-DB refusal."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from multiverse import state_paths

pytestmark = pytest.mark.control_plane


@pytest.fixture
def no_repo_config(monkeypatch, tmp_path):
    monkeypatch.setattr(state_paths, "REPO_ROOT_GUESS", tmp_path / "no-config")


# ---------------------------------------------------------------------------
# resolve_state_root precedence
# ---------------------------------------------------------------------------


def test_multiverse_state_dir_wins(tmp_path):
    env = {"MULTIVERSE_STATE_DIR": str(tmp_path / "explicit"), "HOME": str(tmp_path)}
    assert state_paths.resolve_state_root(env) == (tmp_path / "explicit").resolve()


def test_config_file_state_root_used_when_no_env(tmp_path, monkeypatch):
    cfg_root = tmp_path / "cfg"
    cfg_dir = cfg_root / "multiverse"
    cfg_dir.mkdir(parents=True)
    cfg_dir.joinpath("multiverse.config.yaml").write_text(
        f"state_root: {tmp_path / 'from-config'}\n", encoding="utf-8"
    )
    env = {"XDG_CONFIG_HOME": str(cfg_root), "HOME": str(tmp_path)}
    assert state_paths.resolve_state_root(env) == (tmp_path / "from-config").resolve()


def test_xdg_state_home_used_when_no_env_no_config(tmp_path, no_repo_config):
    env = {"XDG_STATE_HOME": str(tmp_path / "xdg"), "HOME": str(tmp_path)}
    assert state_paths.resolve_state_root(env) == (tmp_path / "xdg" / "multiverse").resolve()


def test_home_default(tmp_path, no_repo_config):
    env = {"HOME": str(tmp_path)}
    assert state_paths.resolve_state_root(env) == (tmp_path / ".multiverse").resolve()


def test_resolve_state_root_returns_absolute(tmp_path):
    env = {"HOME": str(tmp_path)}
    assert state_paths.resolve_state_root(env).is_absolute()


# ---------------------------------------------------------------------------
# resolve_user_id
# ---------------------------------------------------------------------------


def test_user_id_override():
    env = {"MULTIVERSE_USER_ID": "tenant-7"}
    assert state_paths.resolve_user_id(env) == "tenant-7"


def test_user_id_defaults_to_getuser():
    import getpass

    assert state_paths.resolve_user_id({}) == getpass.getuser()


# ---------------------------------------------------------------------------
# is_inside_package_dir
# ---------------------------------------------------------------------------


def test_package_dir_is_inside_package_dir():
    assert state_paths.is_inside_package_dir(state_paths.PACKAGE_DIR)


def test_repo_root_guess_is_inside_package_dir():
    # The repo root (one level up from the package) is also rejected —
    # it is where the legacy install lived.
    assert state_paths.is_inside_package_dir(state_paths.REPO_ROOT_GUESS)


def test_home_is_not_inside_package_dir(tmp_path):
    assert not state_paths.is_inside_package_dir(tmp_path)


# ---------------------------------------------------------------------------
# StatePaths bundle
# ---------------------------------------------------------------------------


def test_state_paths_derived(tmp_path):
    env = {"MULTIVERSE_STATE_DIR": str(tmp_path), "HOME": str(tmp_path)}
    paths = state_paths.resolve_paths(env)
    assert paths.state_root == tmp_path.resolve()
    assert paths.db_path == tmp_path.resolve() / "multiverse_state.db"
    assert paths.store_root == tmp_path.resolve() / "store"
    assert paths.artifacts_dir == tmp_path.resolve() / "store" / "artifacts"
    assert paths.workspaces_dir == tmp_path.resolve() / "store" / "workspaces"
    assert paths.journal_root == tmp_path.resolve() / "journal"


# ---------------------------------------------------------------------------
# registry_db legacy refusal
# ---------------------------------------------------------------------------


def test_legacy_db_refusal_skipped_when_db_name_monkeypatched(monkeypatch, tmp_path):
    """The check must NOT fire when a caller (test or user with
    MULTIVERSE_STATE_DIR) has explicitly set DB_NAME away from the default."""
    from multiverse import registry_db

    monkeypatch.setattr(registry_db, "DB_NAME", str(tmp_path / "state.db"))
    # Should not raise even though a legacy file may exist in the repo.
    registry_db._check_legacy_db_refusal()


def test_legacy_db_refusal_fires_when_default_and_legacy_exists(
    monkeypatch, tmp_path, no_repo_config
):
    """Simulate the upgrade scenario: resolver default is $HOME/.multiverse,
    a legacy DB exists at the repo root."""
    from multiverse import registry_db

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("MULTIVERSE_STATE_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("MULTIVERSE_ALLOW_LEGACY_DB", raising=False)

    # Plant a fake legacy DB and point the search at it.
    legacy_dir = tmp_path / "fake-package"
    legacy_dir.mkdir()
    fake_legacy = legacy_dir / "multiverse_state.db"
    fake_legacy.write_bytes(b"")
    monkeypatch.setattr(registry_db, "_find_legacy_db", lambda: fake_legacy)
    # DB_NAME equals the new resolver default (the trigger condition).
    expected_default = str(fake_home / ".multiverse" / "multiverse_state.db")
    monkeypatch.setattr(registry_db, "DB_NAME", expected_default)

    with pytest.raises(registry_db.LegacyStateDirError) as excinfo:
        registry_db._check_legacy_db_refusal()
    msg = str(excinfo.value)
    assert "migrate-state-dir" in msg
    assert str(fake_legacy) in msg


def test_legacy_db_refusal_bypassed_by_env(monkeypatch, tmp_path):
    from multiverse import registry_db

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("MULTIVERSE_ALLOW_LEGACY_DB", "1")

    legacy_dir = tmp_path / "fake-package"
    legacy_dir.mkdir()
    fake_legacy = legacy_dir / "multiverse_state.db"
    fake_legacy.write_bytes(b"")
    monkeypatch.setattr(registry_db, "_find_legacy_db", lambda: fake_legacy)
    monkeypatch.setattr(
        registry_db,
        "DB_NAME",
        str(fake_home / ".multiverse" / "multiverse_state.db"),
    )

    # Must not raise.
    registry_db._check_legacy_db_refusal()
