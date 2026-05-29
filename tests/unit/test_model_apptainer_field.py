"""Tests for H1: ApptainerSpec in ModelManifest and asset_registry query surface."""
from __future__ import annotations
import sqlite3
import tempfile
from pathlib import Path

import pytest
import yaml

from multiverse.models_ingest import ModelManifest, ApptainerSpec, load_model_manifest
from multiverse.asset_registry import (
    init_asset_registry,
    get_asset_registry_connection,
    get_model_sif_path,
    get_model_gpu_flag,
    set_model_sif_path,
)


def _insert_model(conn, slug, version, sif_path=None, gpu_required=0, docker_image="img:1.0"):
    conn.execute(
        """INSERT OR REPLACE INTO models
        (slug, version, name, docker_image, supported_omics, manifest_path, manifest_hash, status, sif_path, gpu_required)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)""",
        (slug, version, slug, docker_image, '["any"]', f"/{slug}/model.yaml", "hash", sif_path, gpu_required),
    )
    conn.commit()


def test_apptainer_spec_roundtrip():
    spec = ApptainerSpec(sif_path="/data/pca.sif", gpu_required=False)
    assert spec.sif_path == "/data/pca.sif"
    assert spec.gpu_required is False


def test_model_manifest_with_apptainer_and_docker():
    m = ModelManifest(
        name="test",
        version="1.0.0",
        supported_omics=["rna"],
        runtime={"image": "myrepo/test:1.0.0"},
        apptainer={"sif_path": "/hpc/test.sif", "gpu_required": False},
    )
    assert m.apptainer.sif_path == "/hpc/test.sif"
    assert m.runtime.image == "myrepo/test:1.0.0"


def test_model_manifest_apptainer_only():
    m = ModelManifest(
        name="test",
        version="1.0.0",
        supported_omics=["rna"],
        apptainer={"sif_path": "/hpc/test.sif", "gpu_required": True},
    )
    assert m.runtime is None
    assert m.apptainer.gpu_required is True


def test_model_manifest_neither_raises():
    with pytest.raises(Exception, match="at least one"):
        ModelManifest(
            name="test",
            version="1.0.0",
            supported_omics=["rna"],
        )


def test_get_model_sif_path_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_root = Path(tmpdir)
        init_asset_registry(state_root)
        conn = get_asset_registry_connection(state_root)
        _insert_model(conn, "pca", "1.0.0", sif_path="/hpc/pca.sif")
        result = get_model_sif_path(conn, "pca", "1.0.0")
        conn.close()
        assert result == "/hpc/pca.sif"


def test_get_model_sif_path_docker_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_root = Path(tmpdir)
        init_asset_registry(state_root)
        conn = get_asset_registry_connection(state_root)
        _insert_model(conn, "pca", "1.0.0", sif_path=None)
        result = get_model_sif_path(conn, "pca", "1.0.0")
        conn.close()
        assert result is None


def test_get_model_gpu_flag_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_root = Path(tmpdir)
        init_asset_registry(state_root)
        conn = get_asset_registry_connection(state_root)
        _insert_model(conn, "totalvi", "1.0.0", gpu_required=1)
        result = get_model_gpu_flag(conn, "totalvi", "1.0.0")
        conn.close()
        assert result is True


def test_get_model_gpu_flag_default_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_root = Path(tmpdir)
        init_asset_registry(state_root)
        conn = get_asset_registry_connection(state_root)
        _insert_model(conn, "pca", "1.0.0", gpu_required=0)
        result = get_model_gpu_flag(conn, "pca", "1.0.0")
        conn.close()
        assert result is False


def test_set_model_sif_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_root = Path(tmpdir)
        init_asset_registry(state_root)
        conn = get_asset_registry_connection(state_root)
        _insert_model(conn, "pca", "1.0.0", sif_path=None)
        conn.close()
        updated = set_model_sif_path("pca", "1.0.0", "/hpc/pca-1.0.0.sif", state_root=state_root)
        assert updated is True
        conn2 = get_asset_registry_connection(state_root)
        assert get_model_sif_path(conn2, "pca", "1.0.0") == "/hpc/pca-1.0.0.sif"
        conn2.close()
