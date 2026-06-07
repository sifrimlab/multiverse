"""Regression tests for issue #29 — re-registering a deleted model.

After a model is soft-deleted and then re-registered (with an *unchanged*
manifest, i.e. the same manifest hash), it must reappear as ACTIVE in both
the legacy registry and the canonical asset registry. The previous code
short-circuited the legacy write on an unchanged hash, leaving a deleted row
stuck at INACTIVE even though re-registration reported success.
"""

from __future__ import annotations

from pathlib import Path


def _isolate_registries(monkeypatch, tmp_path: Path):
    from multiverse import asset_registry, registry_db

    for attr, path in {
        "DB_NAME": tmp_path / "state.db",
        "STORE_DIR": tmp_path / "store",
        "DATASETS_DIR": tmp_path / "store" / "datasets",
        "RAW_DATASETS_DIR": tmp_path / "store" / "datasets" / "raw",
        "MODELS_DIR": tmp_path / "store" / "models",
        "ARTIFACTS_DIR": tmp_path / "store" / "artifacts",
        "WORKSPACES_DIR": tmp_path / "store" / "workspaces",
    }.items():
        monkeypatch.setattr(registry_db, attr, str(path))
    # Point the canonical asset registry at the same tmp state root so the
    # default (state_root=None) resolution used by registry_db.mark_model_inactive
    # lands inside the test sandbox.
    monkeypatch.setattr(asset_registry, "_DEFAULT_STATE_ROOT", tmp_path)
    return registry_db, asset_registry


def _write_model_manifest(tmp_path: Path) -> Path:
    model_dir = tmp_path / "store" / "models" / "demo"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = model_dir / "model.yaml"
    manifest.write_text(
        "name: demo\n"
        'version: "1.0.0"\n'
        'supported_omics: ["rna"]\n'
        "runtime:\n"
        '  image: "multiverse-demo:1.0.0"\n',
        encoding="utf-8",
    )
    return manifest


def _legacy_status(registry_db, slug: str) -> str | None:
    conn = registry_db.get_db_connection()
    try:
        row = conn.execute(
            "SELECT status FROM models WHERE slug = ?", (slug,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _asset_status(asset_registry, tmp_path: Path, slug: str) -> str | None:
    conn = asset_registry.get_asset_registry_connection(tmp_path)
    try:
        row = conn.execute(
            "SELECT status FROM models WHERE slug = ?", (slug,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def test_reregister_deleted_model_reactivates_in_both_registries(
    monkeypatch, tmp_path: Path
) -> None:
    from multiverse.models_ingest import register_model_from_manifest

    registry_db, asset_registry = _isolate_registries(monkeypatch, tmp_path)
    manifest = _write_model_manifest(tmp_path)

    # 1. Register.
    register_model_from_manifest(str(manifest), state_root=tmp_path)
    assert _legacy_status(registry_db, "demo") == "ACTIVE"
    assert _asset_status(asset_registry, tmp_path, "demo") == "ACTIVE"

    # 2. Delete (GUI path goes through registry_db.mark_model_inactive).
    assert registry_db.mark_model_inactive("demo", "1.0.0")
    assert _legacy_status(registry_db, "demo") == "INACTIVE"
    # Deletion must be symmetric across both registries (issue #29).
    assert _asset_status(asset_registry, tmp_path, "demo") == "INACTIVE"

    # 3. Re-register the *unchanged* manifest. The model must come back ACTIVE.
    register_model_from_manifest(str(manifest), state_root=tmp_path)
    assert _legacy_status(registry_db, "demo") == "ACTIVE"
    assert _asset_status(asset_registry, tmp_path, "demo") == "ACTIVE"
    # And it must be visible to the GUI's read path.
    assert any(m["slug"] == "demo" for m in registry_db.get_all_models())
