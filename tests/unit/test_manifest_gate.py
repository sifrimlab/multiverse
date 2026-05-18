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
        "model_version TEXT, status TEXT, output_path TEXT)"
    )
    cursor.execute(
        "INSERT INTO datasets (id, name, slug, path, omics_available, batch_key, cell_type_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "dataset1", "dataset1", "/data/d1.h5mu", json.dumps(["rna"]), "batch", "cell_type", "READY"),
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


def test_empty_plan_blocks_launch(tmp_path):
    conn = _make_conn()
    conn.execute(
        "INSERT INTO runs (dataset_id, model_slug, model_version, status, output_path) VALUES (?, ?, ?, ?, ?)",
        (1, "pca", "1.0.0", "SUCCESS", "/artifacts/dataset1/pca"),
    )
    conn.commit()
    manifest = tmp_path / "empty.yaml"
    manifest.write_text(
        "jobs:\n  - dataset_slug: dataset1\n    models: [pca]\n",
        encoding="utf-8",
    )
    parsed = parse_manifest(str(manifest), conn)
    assert not parsed.ok
    assert parsed.errors == [
        {"field": "jobs", "message": "manifest dry-run produced no runnable jobs", "code": "empty_plan"}
    ]
