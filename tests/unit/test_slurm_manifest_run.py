"""Tests for H2: manifest-driven Slurm execution."""
from __future__ import annotations
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


def _make_db(tmpdir):
    """Create a minimal in-memory-like DB for testing parse_manifest."""
    db_path = Path(tmpdir) / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE datasets (
        id INTEGER PRIMARY KEY, slug TEXT, name TEXT, path TEXT,
        omics_available TEXT, status TEXT, batch_key TEXT, cell_type_key TEXT,
        manifest_path TEXT, manifest_hash TEXT)""")
    conn.execute("""CREATE TABLE models (
        slug TEXT, version TEXT, name TEXT, docker_image TEXT,
        image_digest TEXT, supported_omics TEXT, manifest_path TEXT,
        manifest_hash TEXT, hyperparameters_schema TEXT, status TEXT,
        sif_path TEXT, gpu_required INTEGER DEFAULT 0,
        PRIMARY KEY (slug, version))""")
    conn.execute("""CREATE TABLE runs (
        id INTEGER PRIMARY KEY, dataset_id INTEGER, model_slug TEXT,
        model_version TEXT, status TEXT, params_hash TEXT)""")
    conn.execute("INSERT INTO datasets VALUES (1,'ds1','Dataset 1','/data/ds.h5mu','[\"rna\"]','READY',NULL,NULL,'/ds/manifest.yaml','abc')")
    conn.execute("INSERT INTO models VALUES ('pca','1.0.0','PCA','pca:1.0.0',NULL,'[\"rna\"]','/pca/model.yaml','def',NULL,'ACTIVE','/hpc/pca.sif',0)")
    conn.execute("INSERT INTO models VALUES ('totalvi','1.0.0','TotalVI','totalvi:1.0.0',NULL,'[\"rna\"]','/totalvi/model.yaml','ghi',NULL,'ACTIVE','/hpc/totalvi.sif',1)")
    conn.commit()
    return conn


SLURM_MANIFEST = """
globals:
  backend: slurm
  slurm:
    partition: gpu
    account: mylab
    time_minutes: 120
    mem_gb: 32
    cpus_per_task: 8
    gpus: 0

jobs:
  - dataset_slug: ds1
    models:
      - pca
    model_version: "1.0.0"
"""


def test_parse_manifest_slurm_extracts_globals(tmp_path):
    from multiverse.runner.cli import parse_manifest
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(SLURM_MANIFEST)
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = _make_db(tmpdir)
        parsed = parse_manifest(str(manifest_file), conn)
        conn.close()
    assert parsed.ok, parsed.errors
    assert parsed.data.get("_backend") == "slurm"
    slurm_globals = parsed.data.get("_slurm_globals", {})
    assert slurm_globals.get("partition") == "gpu"
    assert slurm_globals.get("time_minutes") == 120


def test_parse_manifest_slurm_sif_from_registry(tmp_path):
    from multiverse.runner.cli import parse_manifest
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(SLURM_MANIFEST)
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = _make_db(tmpdir)
        parsed = parse_manifest(str(manifest_file), conn)
        conn.close()
    assert parsed.ok, parsed.errors
    # SIF should have been resolved from registry
    plan_jobs = [j for j in parsed.plan if not j.get("_skipped")]
    if plan_jobs:
        assert plan_jobs[0].get("image_sif") == "/hpc/pca.sif"


MISSING_SIF_MANIFEST = """
globals:
  backend: slurm
  slurm:
    partition: cpu

jobs:
  - dataset_slug: ds1
    models:
      - pca
    model_version: "1.0.0"
"""


def test_parse_manifest_missing_sif_raises(tmp_path):
    from multiverse.runner.cli import parse_manifest
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(MISSING_SIF_MANIFEST)
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = _make_db(tmpdir)
        # Remove sif_path from pca
        conn.execute("UPDATE models SET sif_path = NULL WHERE slug = 'pca'")
        conn.commit()
        parsed = parse_manifest(str(manifest_file), conn)
        conn.close()
    assert not parsed.ok
    assert any("SIF" in e["message"] or "sif" in e["message"].lower() for e in parsed.errors)


GPU_CONFLICT_MANIFEST = """
globals:
  backend: slurm
  slurm:
    partition: gpu
    gpus: 1

jobs:
  - dataset_slug: ds1
    models:
      - pca
    model_version: "1.0.0"
"""


def test_parse_manifest_gpu_conflict_raises(tmp_path):
    from multiverse.runner.cli import parse_manifest
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(GPU_CONFLICT_MANIFEST)
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = _make_db(tmpdir)
        parsed = parse_manifest(str(manifest_file), conn)
        conn.close()
    assert not parsed.ok
    assert any("gpu_required" in e["message"] or "gpu" in e["message"].lower() for e in parsed.errors)


SLURM_JOB_OVERRIDE_MANIFEST = """
globals:
  backend: slurm
  slurm:
    partition: cpu
    time_minutes: 60

jobs:
  - dataset_slug: ds1
    models:
      - pca
    model_version: "1.0.0"
    slurm:
      partition: gpu
      time_minutes: 240
"""


def test_parse_manifest_job_slurm_override(tmp_path):
    from multiverse.runner.cli import parse_manifest
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(SLURM_JOB_OVERRIDE_MANIFEST)
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = _make_db(tmpdir)
        parsed = parse_manifest(str(manifest_file), conn)
        conn.close()
    assert parsed.ok, parsed.errors
    plan_jobs = [j for j in parsed.plan if not j.get("_skipped")]
    if plan_jobs:
        slurm_cfg = plan_jobs[0].get("_slurm", {})
        assert slurm_cfg.get("partition") == "gpu"
        assert slurm_cfg.get("time_minutes") == 240
