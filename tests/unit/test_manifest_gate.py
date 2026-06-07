import hashlib
import json
import sqlite3

from multiverse.runner.cli import parse_manifest


def _make_conn():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE datasets ("
        "id INTEGER PRIMARY KEY, name TEXT, slug TEXT, path TEXT, "
        "omics_available TEXT, batch_key TEXT, cell_type_key TEXT, status TEXT)"
    )
    cursor.execute(
        "CREATE TABLE models ("
        "slug TEXT, docker_image TEXT, supported_omics TEXT, "
        "version TEXT, status TEXT)"
    )
    cursor.execute(
        "CREATE TABLE runs ("
        "run_id INTEGER PRIMARY KEY, dataset_id INTEGER, model_slug TEXT, "
        "model_version TEXT, status TEXT, output_path TEXT, params_hash TEXT)"
    )
    cursor.execute(
        "INSERT INTO datasets (id, name, slug, path, omics_available, batch_key, cell_type_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "dataset1",
            "dataset1",
            "/data/d1.h5mu",
            json.dumps(["rna"]),
            "batch",
            "cell_type",
            "READY",
        ),
    )
    cursor.execute(
        "INSERT INTO models (slug, docker_image, supported_omics, version, status) VALUES (?, ?, ?, ?, ?)",
        ("pca", "multiverse-pca", json.dumps(["rna"]), "1.0.0", "ACTIVE"),
    )
    conn.commit()
    return conn


def test_yaml_syntax_error_returns_structured_error(tmp_path):
    manifest = tmp_path / "bad.yaml"
    manifest.write_text("jobs: [\n", encoding="utf-8")
    parsed = parse_manifest(str(manifest), _make_conn())
    assert not parsed.ok
    assert parsed.errors[0]["code"] == "yaml_error"


def test_stale_dataset_slug_blocks_launch(tmp_path):
    manifest = tmp_path / "stale.yaml"
    manifest.write_text(
        "jobs:\n  - dataset_slug: missing\n    models: [pca]\n",
        encoding="utf-8",
    )
    parsed = parse_manifest(str(manifest), _make_conn())
    assert not parsed.ok
    assert any(err["code"] == "stale_dataset_slug" for err in parsed.errors)


def test_legacy_success_row_does_not_empty_plan(tmp_path):
    # STRATEGY (MVD Manifest Resume and Dedupe): planning is a pure manifest →
    # plan expansion and never consults the legacy ``runs`` table. A prior
    # legacy SUCCESS row — even with the matching params_hash — must NOT empty
    # the plan; the job stays runnable. (check_images=False keeps the test off
    # the Docker daemon; the point here is plan membership, not image probing.)
    conn = _make_conn()
    empty_params_hash = hashlib.sha256(
        json.dumps({}, sort_keys=True).encode()
    ).hexdigest()[:12]
    conn.execute(
        "INSERT INTO runs (dataset_id, model_slug, model_version, status, output_path, params_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (1, "pca", "1.0.0", "SUCCESS", "/artifacts/dataset1/pca", empty_params_hash),
    )
    conn.commit()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "jobs:\n  - dataset_slug: dataset1\n    models: [pca]\n",
        encoding="utf-8",
    )
    parsed = parse_manifest(str(manifest), conn, check_images=False)
    assert parsed.ok, parsed.errors
    assert len(parsed.plan) == 1
    assert not any(err["code"] == "empty_plan" for err in parsed.errors)
    assert not parsed.plan[0].get("_skipped")
