"""Tests for `multiverse migrate-state-dir` (STRATEGY M1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from multiverse import cli_entrypoints

pytestmark = pytest.mark.control_plane


def _seed_legacy(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "store").mkdir()
    (root / "store" / "artifacts").mkdir()
    (root / "store" / "artifacts" / "marker.txt").write_text("hi", encoding="utf-8")
    (root / "journal").mkdir()
    (root / "journal" / "seg-0001.ndjson").write_text(
        '{"kind":"BOOT"}\n', encoding="utf-8"
    )
    conn = sqlite3.connect(str(root / "multiverse_state.db"))
    conn.execute("CREATE TABLE t (k TEXT)")
    conn.execute("INSERT INTO t VALUES ('x')")
    conn.commit()
    conn.close()


def test_dry_run_reports_plan_without_moving(tmp_path, capsys):
    src = tmp_path / "legacy"
    dst = tmp_path / "new"
    _seed_legacy(src)

    rc = cli_entrypoints.migrate_state_dir_main(["--from", str(src), "--to", str(dst)])
    out = capsys.readouterr().out
    assert rc == 0
    assert (src / "multiverse_state.db").is_file()
    assert (src / "store" / "artifacts" / "marker.txt").is_file()
    assert not dst.exists() or not (dst / "multiverse_state.db").exists()
    assert "dry-run" in out
    assert "store/" in out
    assert "journal/" in out


def test_apply_moves_db_store_and_journal(tmp_path):
    src = tmp_path / "legacy"
    dst = tmp_path / "new"
    _seed_legacy(src)

    rc = cli_entrypoints.migrate_state_dir_main(
        ["--from", str(src), "--to", str(dst), "--apply"]
    )
    assert rc == 0
    assert (dst / "multiverse_state.db").is_file()
    assert (dst / "store" / "artifacts" / "marker.txt").read_text() == "hi"
    assert (dst / "journal" / "seg-0001.ndjson").is_file()
    # Originals are gone.
    assert not (src / "multiverse_state.db").exists()
    assert not (src / "store").exists()
    assert not (src / "journal").exists()


def test_apply_refuses_to_clobber_existing_destination(tmp_path, capsys):
    src = tmp_path / "legacy"
    dst = tmp_path / "new"
    _seed_legacy(src)
    (dst / "store").mkdir(parents=True)
    (dst / "store" / "pre-existing.txt").write_text("keep me", encoding="utf-8")

    rc = cli_entrypoints.migrate_state_dir_main(
        ["--from", str(src), "--to", str(dst), "--apply"]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "destination already has entries" in err
    # Originals are still in place.
    assert (src / "multiverse_state.db").is_file()
    assert (dst / "store" / "pre-existing.txt").read_text() == "keep me"


def test_noop_when_src_equals_dst(tmp_path, capsys):
    src = tmp_path / "same"
    _seed_legacy(src)
    rc = cli_entrypoints.migrate_state_dir_main(
        ["--from", str(src), "--to", str(src), "--apply"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to do" in out
    assert (src / "multiverse_state.db").is_file()
