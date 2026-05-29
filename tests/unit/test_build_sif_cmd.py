"""Tests for H3: multiverse build-sif command."""
from __future__ import annotations
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import subprocess

import pytest
import yaml


def _write_manifest(path, with_docker=True, with_apptainer=False, with_def_file=False):
    """Write a minimal model.yaml to path."""
    data = {
        "name": "PCA",
        "version": "1.0.0",
        "supported_omics": ["rna"],
        "contract_version": "1.0.0",
    }
    if with_docker:
        data["runtime"] = {"image": "pca:1.0.0"}
        data["build"] = {"context": ".", "dockerfile": "Dockerfile"}
    if with_apptainer or with_def_file:
        data["apptainer"] = {
            "sif_path": None,
            "gpu_required": False,
        }
        if with_def_file:
            data["apptainer"]["def_file"] = "Singularity.def"
            data["apptainer"]["build_from"] = "def_file"
    Path(path).write_text(yaml.dump(data), encoding="utf-8")


def test_preflight_no_apptainer(tmp_path):
    manifest = tmp_path / "model.yaml"
    _write_manifest(manifest)
    with patch("shutil.which", return_value=None):
        from multiverse.cli_entrypoints import build_sif_main
        rc = build_sif_main(["--slug", "pca", "--manifest", str(manifest)])
    assert rc != 0


def test_preflight_docker_image_missing(tmp_path):
    manifest = tmp_path / "model.yaml"
    _write_manifest(manifest, with_docker=True)
    with (
        patch("shutil.which", return_value="/usr/bin/apptainer"),
        patch("subprocess.run", return_value=MagicMock(returncode=1)) as mock_run,
        patch("subprocess.Popen") as mock_popen,
    ):
        from multiverse.cli_entrypoints import build_sif_main
        rc = build_sif_main(["--slug", "pca", "--manifest", str(manifest), "--method", "docker-daemon"])
    assert rc != 0


def test_output_dir_respected(tmp_path):
    manifest = tmp_path / "model.yaml"
    _write_manifest(manifest, with_docker=True)
    custom_out = tmp_path / "my_sifs"

    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Build output line\n"])
    mock_proc.returncode = 0
    mock_proc.wait = MagicMock()

    with (
        patch("shutil.which", return_value="/usr/bin/apptainer"),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("multiverse.asset_registry.set_model_sif_path", return_value=True) as mock_reg,
    ):
        from multiverse.cli_entrypoints import build_sif_main
        rc = build_sif_main([
            "--slug", "pca",
            "--manifest", str(manifest),
            "--output-dir", str(custom_out),
            "--method", "docker-daemon",
        ])
    assert rc == 0
    mock_reg.assert_called_once()
    called_path = mock_reg.call_args[0][2]
    assert "my_sifs" in called_path


def test_model_yaml_unchanged_after_build(tmp_path):
    manifest = tmp_path / "model.yaml"
    _write_manifest(manifest, with_docker=True)
    original_content = manifest.read_text()

    mock_proc = MagicMock()
    mock_proc.stdout = iter([])
    mock_proc.returncode = 0
    mock_proc.wait = MagicMock()

    with (
        patch("shutil.which", return_value="/usr/bin/apptainer"),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("subprocess.Popen", return_value=mock_proc),
        patch("multiverse.asset_registry.set_model_sif_path", return_value=True),
    ):
        from multiverse.cli_entrypoints import build_sif_main
        rc = build_sif_main([
            "--slug", "pca",
            "--manifest", str(manifest),
            "--method", "docker-daemon",
        ])

    assert manifest.read_text() == original_content


def test_def_file_method_uses_def_path(tmp_path):
    manifest = tmp_path / "model.yaml"
    _write_manifest(manifest, with_docker=False, with_def_file=True)
    # Create the def file
    def_file = tmp_path / "Singularity.def"
    def_file.write_text("Bootstrap: docker\nFrom: ubuntu:22.04\n%runscript\nexec bash\n")

    mock_proc = MagicMock()
    mock_proc.stdout = iter([])
    mock_proc.returncode = 0
    mock_proc.wait = MagicMock()

    captured_cmd = []
    def capture_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return mock_proc

    with (
        patch("shutil.which", return_value="/usr/bin/apptainer"),
        patch("subprocess.Popen", side_effect=capture_popen),
        patch("multiverse.asset_registry.set_model_sif_path", return_value=True),
    ):
        from multiverse.cli_entrypoints import build_sif_main
        rc = build_sif_main([
            "--slug", "pca",
            "--manifest", str(manifest),
            "--method", "def-file",
        ])
    assert rc == 0
    assert str(def_file) in captured_cmd
