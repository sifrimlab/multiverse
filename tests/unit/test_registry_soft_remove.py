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

    assert registry_db.mark_dataset_removed("ds")

    conn = registry_db.get_db_connection()
    try:
        status = conn.execute(
            "SELECT status FROM datasets WHERE slug = 'ds'"
        ).fetchone()[0]
        run_count = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE dataset_id = 1"
        ).fetchone()[0]
    finally:
        conn.close()
    assert status == "REMOVED"
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

    assert registry_db.mark_model_inactive("pca", "1.0.0")
    assert registry_db.get_all_models() == []


def test_delete_run_by_id_removes_run_and_metrics(monkeypatch, tmp_path):
    registry_db = _patch_registry_paths(monkeypatch, tmp_path)
    registry_db.init_db()
    conn = registry_db.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO datasets (id, slug, name, path, omics_available, status) "
            "VALUES (1, 'ds', 'Dataset', '/tmp/ds', '[\"rna\"]', 'READY')"
        )
        conn.execute(
            "INSERT INTO runs (run_id, dataset_id, model_slug, model_version, model_name, "
            "status, output_path, params_hash) VALUES (42, 1, 'pca', '1.0.0', 'PCA', 'SUCCESS', '/tmp/out', 'abc123')"
        )
        conn.execute(
            "INSERT INTO run_metrics (run_id, metric_name, metric_value) VALUES (42, 'total_variance', 0.9)"
        )
        conn.commit()
    finally:
        conn.close()

    assert registry_db.delete_run_by_id(42)

    conn = registry_db.get_db_connection()
    try:
        run_row = conn.execute("SELECT 1 FROM runs WHERE run_id = 42").fetchone()
        metric_row = conn.execute(
            "SELECT 1 FROM run_metrics WHERE run_id = 42"
        ).fetchone()
    finally:
        conn.close()
    assert run_row is None, "run row should be gone"
    assert metric_row is None, "metric row should be cascade-deleted"


def test_delete_run_by_id_returns_false_for_missing_id(monkeypatch, tmp_path):
    registry_db = _patch_registry_paths(monkeypatch, tmp_path)
    registry_db.init_db()
    assert not registry_db.delete_run_by_id(9999)


def test_legacy_success_row_does_not_suppress_manifest_job(monkeypatch, tmp_path):
    """A legacy ``runs.SUCCESS`` row must NOT drop an explicit manifest job.

    Per STRATEGY (MVD Manifest Resume and Dedupe) planning is a pure manifest →
    plan expansion; the legacy ``runs`` table is never consulted. Resume against
    durable mvd ``ARTIFACT_SUCCESS`` state is the only opt-in skip path.
    """
    import hashlib
    import json

    from multiverse.runner.cli import generate_execution_plan_from_manifest

    registry_db = _patch_registry_paths(monkeypatch, tmp_path)
    registry_db.init_db()
    conn = registry_db.get_db_connection()

    params = {"n_components": 5}
    ph = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]

    try:
        conn.execute(
            "INSERT INTO datasets (id, slug, name, path, omics_available, batch_key, cell_type_key, status) "
            "VALUES (1, 'ds', 'ds', '/tmp/ds', '[\"rna\"]', 'batch', 'cell_type', 'READY')"
        )
        conn.execute(
            "INSERT INTO models (slug, docker_image, supported_omics, version, manifest_path, manifest_hash, status) "
            "VALUES ('pca', 'pca:1.0', '[\"rna\"]', '1.0', '/tmp/model.yaml', 'hash', 'ACTIVE')"
        )
        conn.execute(
            "INSERT INTO runs (run_id, dataset_id, model_slug, model_version, model_name, "
            "status, output_path, params_hash) VALUES (1, 1, 'pca', '1.0', 'pca', 'SUCCESS', '/tmp/out', ?)",
            (ph,),
        )
        conn.commit()
    finally:
        conn.close()

    manifest = {
        "jobs": [{"dataset_id": "ds", "models": ["pca"], "model_params": params}]
    }

    # Even with a matching legacy SUCCESS row, the job is planned and runnable.
    plan = generate_execution_plan_from_manifest(
        registry_db.get_db_connection(), manifest
    )
    assert len(plan) == 1
    assert not plan[0].get("_skipped")

    # Deleting the legacy row changes nothing — the planner never read it.
    registry_db.delete_run_by_id(1)
    plan_after = generate_execution_plan_from_manifest(
        registry_db.get_db_connection(), manifest
    )
    assert len(plan_after) == 1
