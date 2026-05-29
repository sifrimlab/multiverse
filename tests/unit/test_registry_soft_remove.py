from __future__ import annotations


def _patch_registry_paths(monkeypatch, tmp_path):
    from multiverse import registry_db

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
    return registry_db


def test_mark_dataset_removed_preserves_runs(monkeypatch, tmp_path):
    registry_db = _patch_registry_paths(monkeypatch, tmp_path)
    registry_db.init_db()
    conn = registry_db.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO datasets (id, slug, name, path, omics_available, status) VALUES (1, 'ds', 'Dataset One', '/tmp/ds', '[\"rna\"]', 'READY')"
        )
        conn.execute(
            "INSERT INTO runs (dataset_id, model_slug, model_version, model_name, status, output_path) VALUES (1, 'pca', '1.0.0', 'PCA', 'SUCCESS', '/tmp/out')"
        )
        conn.commit()
    finally:
        conn.close()

    assert registry_db.mark_dataset_removed('ds')

    conn = registry_db.get_db_connection()
    try:
        status = conn.execute("SELECT status FROM datasets WHERE slug = 'ds'").fetchone()[0]
        run_count = conn.execute("SELECT COUNT(*) FROM runs WHERE dataset_id = 1").fetchone()[0]
    finally:
        conn.close()
    assert status == 'REMOVED'
    assert run_count == 1


def test_mark_model_inactive_hides_active_model(monkeypatch, tmp_path):
    registry_db = _patch_registry_paths(monkeypatch, tmp_path)
    registry_db.init_db()
    conn = registry_db.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO models (slug, version, name, docker_image, supported_omics, manifest_path, manifest_hash, status) "
            "VALUES ('pca', '1.0.0', 'PCA', 'pca:1.0.0', '[\"rna\"]', '/tmp/model.yaml', 'hash', 'ACTIVE')"
        )
        conn.commit()
    finally:
        conn.close()

    assert registry_db.mark_model_inactive('pca', '1.0.0')
    assert registry_db.get_all_models() == []
