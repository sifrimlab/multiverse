"""Milestone-14 exit-gate tests for registration hardening.

Coverage:
    1. ``safe_under_root`` rejects ``..`` escapes (S19 acceptance):
       ``raw_files: rna: ../../../etc/passwd`` is refused at parse time.
    2. Absolute paths outside the store root are refused.
    3. Symlinks that resolve outside the store root are refused.
    4. Symlinks within the store root are accepted (canonicalisation
       only checks the destination).
    5. Privilege audit flags ``privileged``, ``network: host``,
       ``pid: host``, ``cap_add: SYS_ADMIN``, and unauthorised volume
       targets.
    6. ``validate_model_manifest`` raises ``PrivilegedRegistrationError``
       on elevated flags unless ``allow_elevated=True``.
    7. Trust classification distinguishes BUILTIN (under
       ``store/models/``) from IMPORTED.
    8. Reports are JSON-serialisable for the GUI banner.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from multiverse.registration import (PathEscapeError,
                                     PrivilegedRegistrationError, TrustLevel,
                                     audit_docker_flags, classify_trust,
                                     safe_under_root, validate_model_manifest,
                                     validate_paths_in_mapping)

# ---------------------------------------------------------------------------
# 1-2. Path escape detection
# ---------------------------------------------------------------------------


def test_safe_under_root_accepts_in_tree_path(tmp_path: Path) -> None:
    inside = safe_under_root("subdir/file", root=tmp_path)
    assert inside.is_relative_to(tmp_path.resolve())


def test_safe_under_root_rejects_dot_dot_escape(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        safe_under_root("../../../etc/passwd", root=tmp_path)


def test_safe_under_root_rejects_absolute_outside(tmp_path: Path) -> None:
    with pytest.raises(PathEscapeError):
        safe_under_root("/etc/passwd", root=tmp_path)


def test_validate_paths_in_mapping_finds_nested_raw_files(tmp_path: Path) -> None:
    manifest = {
        "name": "demo",
        "raw_files": {"rna": "rna/data.h5", "atac": "atac/data.h5"},
        "model": {"image": "x:1"},
    }
    resolved = validate_paths_in_mapping(manifest, root=tmp_path)
    assert set(resolved.keys()) == {"raw_files.rna", "raw_files.atac"}


def test_validate_paths_in_mapping_rejects_nested_escape(tmp_path: Path) -> None:
    manifest = {"raw_files": {"rna": "../../../etc/passwd"}}
    with pytest.raises(PathEscapeError):
        validate_paths_in_mapping(manifest, root=tmp_path)


# ---------------------------------------------------------------------------
# 3-4. Symlink behaviour
# ---------------------------------------------------------------------------


def test_symlink_resolving_outside_is_rejected(tmp_path: Path) -> None:
    """A symlink under the store that points outside is treated as path
    escape because we canonicalise with realpath before checking."""
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "outside_file"
    outside.write_bytes(b"x")
    (store / "evil").symlink_to(outside)
    with pytest.raises(PathEscapeError):
        safe_under_root("evil", root=store)


def test_symlink_resolving_inside_is_accepted(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    real = store / "real"
    real.write_bytes(b"x")
    (store / "alias").symlink_to(real)
    resolved = safe_under_root("alias", root=store)
    assert resolved.is_relative_to(store.resolve())


# ---------------------------------------------------------------------------
# 5. Privilege audit
# ---------------------------------------------------------------------------


def test_privilege_audit_flags_privileged_true() -> None:
    audit = audit_docker_flags({"docker": {"privileged": True}})
    assert audit.elevated is True
    assert any("privileged" in r for r in audit.reasons)


def test_privilege_audit_flags_network_host() -> None:
    audit = audit_docker_flags({"docker": {"network": "host"}})
    assert audit.elevated is True


def test_privilege_audit_flags_pid_host() -> None:
    audit = audit_docker_flags({"docker": {"pid": "host"}})
    assert audit.elevated is True


def test_privilege_audit_flags_sys_admin_cap() -> None:
    audit = audit_docker_flags({"docker": {"cap_add": ["NET_ADMIN", "SYS_ADMIN"]}})
    assert audit.elevated is True


def test_privilege_audit_flags_unauthorised_volume() -> None:
    audit = audit_docker_flags({"docker": {"volumes": ["/etc:/etc:ro"]}})
    assert audit.elevated is True


def test_privilege_audit_passes_clean_manifest() -> None:
    audit = audit_docker_flags(
        {"docker": {"volumes": ["/host/in:/input:ro", "/host/out:/output"]}}
    )
    assert audit.elevated is False
    assert audit.reasons == []


# ---------------------------------------------------------------------------
# 6. validate_model_manifest refuses elevated without opt-in
# ---------------------------------------------------------------------------


def test_validate_model_manifest_refuses_elevated_by_default(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    manifest_path = store / "model.yaml"
    manifest_path.write_text(
        "name: evil\n"
        "docker:\n"
        "  privileged: true\n"
        "raw_files:\n"
        "  src: src/file\n",
        encoding="utf-8",
    )
    with pytest.raises(PrivilegedRegistrationError):
        validate_model_manifest(manifest_path, store_root=store)


def test_validate_model_manifest_accepts_with_allow_elevated(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    manifest_path = store / "model.yaml"
    manifest_path.write_text(
        "name: special\n" "docker:\n" "  privileged: true\n",
        encoding="utf-8",
    )
    report = validate_model_manifest(
        manifest_path, store_root=store, allow_elevated=True
    )
    assert report.privilege_audit.elevated is True


def test_validate_model_manifest_rejects_path_escape(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    manifest_path = store / "model.yaml"
    manifest_path.write_text(
        "name: bad\nraw_files:\n  rna: ../../../etc/passwd\n",
        encoding="utf-8",
    )
    with pytest.raises(PathEscapeError):
        validate_model_manifest(manifest_path, store_root=store)


# ---------------------------------------------------------------------------
# 7. Trust classification
# ---------------------------------------------------------------------------


def test_classify_trust_builtin_under_store_models(tmp_path: Path) -> None:
    builtin_root = tmp_path / "store" / "models"
    builtin_root.mkdir(parents=True)
    manifest = builtin_root / "pca" / "model.yaml"
    manifest.parent.mkdir()
    manifest.write_text("")
    assert classify_trust(manifest, builtin_root=builtin_root) is TrustLevel.BUILTIN


def test_classify_trust_imported_outside_builtin(tmp_path: Path) -> None:
    elsewhere = tmp_path / "user_models" / "custom" / "model.yaml"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("")
    builtin_root = tmp_path / "store" / "models"
    builtin_root.mkdir(parents=True)
    assert classify_trust(elsewhere, builtin_root=builtin_root) is TrustLevel.IMPORTED


def test_classify_trust_imported_when_no_builtin_root(tmp_path: Path) -> None:
    p = tmp_path / "any" / "model.yaml"
    p.parent.mkdir()
    p.write_text("")
    assert classify_trust(p) is TrustLevel.IMPORTED


# ---------------------------------------------------------------------------
# 8. Reports are JSON-serialisable
# ---------------------------------------------------------------------------


def test_model_registration_report_is_json_serialisable(tmp_path: Path) -> None:
    import json

    store = tmp_path / "store"
    store.mkdir()
    manifest_path = store / "model.yaml"
    manifest_path.write_text(
        "name: ok\nraw_files:\n  src: src/file\n", encoding="utf-8"
    )
    report = validate_model_manifest(manifest_path, store_root=store)
    blob = json.dumps(report.to_dict(), sort_keys=True)
    parsed = json.loads(blob)
    assert parsed["trust"] in {"builtin", "imported"}
    assert parsed["privilege_audit"]["elevated"] is False
