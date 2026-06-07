"""Move-8 exit-gate tests for registration hardening cutover.

Strategy v2 §8 acceptance: path escapes and elevated Docker flags are
rejected or require explicit opt-in through the **real** registration
entry points (``register_from_manifest`` and
``register_model_from_manifest``), not only through the
``multiverse/registration`` library.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse.registration.errors import (PathEscapeError,
                                            PrivilegedRegistrationError)

# ---------------------------------------------------------------------------
# 1. Dataset registration rejects path-escape
# ---------------------------------------------------------------------------


def test_dataset_register_from_manifest_rejects_path_escape(tmp_path: Path) -> None:
    """STRATEGY v2 §8 acceptance: a dataset.yaml with ``raw_files: rna:
    ../../../etc/passwd`` is refused at parse time."""
    from multiverse.ingestion import register_from_manifest

    dataset_dir = tmp_path / "store" / "datasets" / "evil"
    dataset_dir.mkdir(parents=True)
    manifest = dataset_dir / "dataset.yaml"
    manifest.write_text(
        "name: evil\n"
        "omics: [rna]\n"
        "raw_files:\n"
        "  rna: ../../../etc/passwd\n"
        "metadata_keys: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(PathEscapeError):
        register_from_manifest(str(manifest))


def test_dataset_register_from_manifest_rejects_absolute_outside(
    tmp_path: Path,
) -> None:
    from multiverse.ingestion import register_from_manifest

    dataset_dir = tmp_path / "store" / "datasets" / "bad_abs"
    dataset_dir.mkdir(parents=True)
    manifest = dataset_dir / "dataset.yaml"
    manifest.write_text(
        "name: bad\n"
        "omics: [rna]\n"
        "raw_files:\n"
        "  rna: /etc/passwd\n"
        "metadata_keys: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(PathEscapeError):
        register_from_manifest(str(manifest))


def test_dataset_register_does_not_touch_db_on_escape(
    tmp_path: Path, monkeypatch
) -> None:
    """The hardening check must run BEFORE any SQLite call so a malicious
    manifest cannot mutate the registry by partial side-effect."""
    from multiverse import registry_db
    from multiverse.ingestion import register_from_manifest

    touched = {"yes": False}

    def _exploding_init():
        touched["yes"] = True
        raise AssertionError("init_db must not run before hardening passes")

    monkeypatch.setattr(registry_db, "init_db", _exploding_init)

    dataset_dir = tmp_path / "store" / "datasets" / "evil"
    dataset_dir.mkdir(parents=True)
    manifest = dataset_dir / "dataset.yaml"
    manifest.write_text(
        "name: evil\n"
        "omics: [rna]\n"
        "raw_files:\n"
        "  rna: ../../../etc/passwd\n"
        "metadata_keys: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(PathEscapeError):
        register_from_manifest(str(manifest))
    assert touched["yes"] is False


# ---------------------------------------------------------------------------
# 2. Model registration rejects elevated Docker flags
# ---------------------------------------------------------------------------


def _write_minimal_model_manifest(dir_: Path, *, extra_docker: str = "") -> Path:
    """Build a minimal model.yaml that satisfies ``ModelManifest`` without
    requiring scvi/h5py."""
    (dir_).mkdir(parents=True, exist_ok=True)
    body = (
        "name: demo\n"
        'version: "1.0.0"\n'
        'supported_omics: ["rna"]\n'
        "runtime:\n"
        '  image: "multiverse-demo:1.0.0"\n'
        f"{extra_docker}"
    )
    path = dir_ / "model.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_model_register_rejects_privileged_flag(tmp_path: Path) -> None:
    from multiverse.models_ingest import register_model_from_manifest

    manifest = _write_minimal_model_manifest(
        tmp_path / "store" / "models" / "demo",
        extra_docker="docker:\n  privileged: true\n",
    )
    with pytest.raises(PrivilegedRegistrationError):
        register_model_from_manifest(str(manifest))


def test_model_register_rejects_network_host(tmp_path: Path) -> None:
    from multiverse.models_ingest import register_model_from_manifest

    manifest = _write_minimal_model_manifest(
        tmp_path / "store" / "models" / "demo",
        extra_docker='docker:\n  network: "host"\n',
    )
    with pytest.raises(PrivilegedRegistrationError):
        register_model_from_manifest(str(manifest))


def test_model_register_rejects_unauthorised_volume(tmp_path: Path) -> None:
    from multiverse.models_ingest import register_model_from_manifest

    manifest = _write_minimal_model_manifest(
        tmp_path / "store" / "models" / "demo",
        extra_docker='docker:\n  volumes: ["/etc:/etc:ro"]\n',
    )
    with pytest.raises(PrivilegedRegistrationError):
        register_model_from_manifest(str(manifest))


def test_model_register_rejects_path_escape(tmp_path: Path) -> None:
    from multiverse.models_ingest import register_model_from_manifest

    manifest = _write_minimal_model_manifest(
        tmp_path / "store" / "models" / "demo",
        extra_docker='raw_files:\n  weights: "../../../etc/passwd"\n',
    )
    with pytest.raises(PathEscapeError):
        register_model_from_manifest(str(manifest))


def test_model_register_does_not_touch_db_on_elevated(
    tmp_path: Path, monkeypatch
) -> None:
    from multiverse import registry_db
    from multiverse.models_ingest import register_model_from_manifest

    monkeypatch.setattr(
        registry_db,
        "init_db",
        lambda: (_ for _ in ()).throw(
            AssertionError("init_db must not run before hardening passes")
        ),
    )
    manifest = _write_minimal_model_manifest(
        tmp_path / "store" / "models" / "demo",
        extra_docker="docker:\n  privileged: true\n",
    )
    with pytest.raises(PrivilegedRegistrationError):
        register_model_from_manifest(str(manifest))


# ---------------------------------------------------------------------------
# 3. Explicit opt-in still allows elevated registration
# ---------------------------------------------------------------------------


def test_model_register_with_allow_elevated_passes_hardening(
    tmp_path: Path, monkeypatch
) -> None:
    """``allow_elevated=True`` short-circuits the privilege gate (the
    user has explicitly acknowledged the risk). Hardening should let
    the registration proceed into the SQLite path."""
    from multiverse import models_ingest, registry_db
    from multiverse.models_ingest import register_model_from_manifest

    # Skip the actual SQLite work so the test stays hot-path-clean.
    captured = {"called": False}

    def _fake_db_path() -> str:
        captured["called"] = True
        raise _ShortCircuit()

    class _ShortCircuit(Exception):
        pass

    monkeypatch.setattr(models_ingest, "init_db", _fake_db_path)
    monkeypatch.setattr(registry_db, "init_db", _fake_db_path)

    manifest = _write_minimal_model_manifest(
        tmp_path / "store" / "models" / "demo",
        extra_docker="docker:\n  privileged: true\n",
    )
    with pytest.raises(_ShortCircuit):
        register_model_from_manifest(str(manifest), allow_elevated=True)
    assert captured["called"] is True, "hardening must have allowed the call through"
