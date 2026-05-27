"""Milestone-2 exit-gate tests for validators and bundle writer.

Coverage:
    1. Validators deterministically pass on a good fixture and fail on a bad
       fixture per-rule (missing embedding, wrong shape, non-float, NaN-only,
       missing required metrics keys, truncated PNG).
    2. Strict mode upgrades selected warnings to refusals.
    3. ``write_bundle`` produces a complete, manifest-verified bundle.
    4. ``write_run_attempt_manifest`` produces a diagnosable record for
       non-success terminal states (S5 acceptance).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from multiverse.artifact import (
    ARTIFACT_MANIFEST_FILENAME,
    ArtifactManifest,
    BootContext,
    BundleInputs,
    ExpectedArtifact,
    ExpectedArtifactRole,
    ImageIdentity,
    IssueSeverity,
    ModelOutputContract,
    ProducedAt,
    ProducedBy,
    RunAttemptManifest,
    ValidationLevel,
    compute_logical_run_id,
    compute_manifest_hash,
    compute_params_hash,
    new_physical_attempt_id,
    produced_at_now,
    read_manifest,
    validate_output_bundle,
    write_bundle,
    write_run_attempt_manifest,
)


# ---------------------------------------------------------------------------
# Workspace fixture builders
# ---------------------------------------------------------------------------


# Single-pixel transparent PNG (real, byte-valid)
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000000500015a4d0e60000000004945"
    "4e44ae426082"
)


def _write_embedding(ws: Path, n_obs: int = 4, fill: str = "random",
                     dtype: np.dtype = np.float32, top_key: str = "latent") -> Path:
    ws.mkdir(parents=True, exist_ok=True)
    path = ws / "embeddings.h5"
    if fill == "random":
        arr = np.random.default_rng(0).standard_normal((n_obs, 4)).astype(dtype)
    elif fill == "nan":
        arr = np.full((n_obs, 4), np.nan, dtype=dtype)
    elif fill == "mixed":
        arr = np.random.default_rng(0).standard_normal((n_obs, 4)).astype(dtype)
        arr[0, 0] = np.nan
    else:
        raise ValueError(fill)
    with h5py.File(path, "w") as f:
        f.create_dataset(top_key, data=arr)
    return path


def _write_metrics(ws: Path, *, content: dict | None = None,
                   filename: str = "metrics.json") -> Path:
    path = ws / filename
    payload = content if content is not None else {"asw": 0.7, "ari": 0.6}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_umap(ws: Path, *, valid_header: bool = True,
                size_bytes: int | None = None) -> Path:
    path = ws / "umap.png"
    if valid_header:
        body = _TINY_PNG
    else:
        body = b"NOT A PNG"
    if size_bytes is not None:
        body = body + (b"\x00" * max(size_bytes - len(body), 0))
    path.write_bytes(body)
    return path


@pytest.fixture
def good_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_embedding(ws, n_obs=4)
    _write_metrics(ws)
    _write_umap(ws, size_bytes=1024)
    return ws


# ---------------------------------------------------------------------------
# 1. Per-rule validation
# ---------------------------------------------------------------------------


def test_good_workspace_passes_basic(good_workspace: Path) -> None:
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(good_workspace, contract, ValidationLevel.BASIC)
    assert report.passed, report.to_dict()
    assert {e.name for e in report.artifact_entries} == {
        "embeddings.h5",
        "metrics.json",
        "umap.png",
    }


def test_missing_embedding_is_refusal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    codes = {i.code for i in report.refusals}
    assert "EMBEDDING_MISSING" in codes
    assert not report.passed


def test_wrong_n_obs_is_refusal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=8)
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    codes = {i.code for i in report.refusals}
    assert codes == {"EMBEDDING_WRONG_N_OBS"}


def test_non_float_embedding_is_refusal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=4, dtype=np.dtype(np.int32))
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    codes = {i.code for i in report.refusals}
    assert "EMBEDDING_NOT_FLOAT" in codes


def test_all_nan_embedding_is_refusal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=4, fill="nan")
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    codes = {i.code for i in report.refusals}
    assert "EMBEDDING_TOO_MANY_NONFINITE" in codes


def test_extra_top_level_dataset_is_refusal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=4)
    # Reopen and add a stray dataset alongside 'latent'.
    with h5py.File(ws / "embeddings.h5", "a") as f:
        f.create_dataset("extras", data=np.zeros(2, dtype=np.float32))
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    codes = {i.code for i in report.refusals}
    assert "EMBEDDING_BAD_LAYOUT" in codes


def test_unparsable_metrics_is_refusal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=4)
    (ws / "metrics.json").write_text("not valid json {", encoding="utf-8")
    contract = ModelOutputContract(
        mv_contract_version="1",
        artifacts=[
            ExpectedArtifact.embedding(expected_n_obs=4),
            ExpectedArtifact.metrics(required=True),
        ],
    )
    report = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    codes = {i.code for i in report.refusals}
    assert "METRICS_UNPARSABLE" in codes


def test_metrics_missing_required_keys_is_warning_in_basic_refusal_in_strict(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=4)
    _write_metrics(ws, content={"asw": 0.5})
    contract = ModelOutputContract(
        mv_contract_version="1",
        artifacts=[
            ExpectedArtifact.embedding(expected_n_obs=4),
            ExpectedArtifact.metrics(required=True, schema_keys=("asw", "ari")),
        ],
    )

    basic = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    assert basic.passed, "basic mode must NOT refuse on missing optional metric keys"
    warnings = {i.code for i in basic.warnings}
    assert "METRICS_MISSING_KEYS" in warnings

    strict = validate_output_bundle(ws, contract, ValidationLevel.STRICT)
    refusals = {i.code for i in strict.refusals}
    assert "METRICS_MISSING_KEYS" in refusals


def test_bad_png_header_is_refusal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=4)
    _write_metrics(ws)
    _write_umap(ws, valid_header=False, size_bytes=4096)
    contract = ModelOutputContract(
        mv_contract_version="1",
        artifacts=[
            ExpectedArtifact.embedding(expected_n_obs=4),
            ExpectedArtifact.metrics(required=False),
            ExpectedArtifact.umap(required=True),
        ],
    )
    report = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    codes = {i.code for i in report.refusals}
    assert "UMAP_BAD_HEADER" in codes


def test_strict_mode_raises_finite_fraction_floor(tmp_path: Path) -> None:
    """A single NaN passes a 0.9 floor in basic but fails strict's 1.0 floor."""
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=4, fill="mixed")
    contract = ModelOutputContract(
        mv_contract_version="1",
        artifacts=[
            ExpectedArtifact.embedding(expected_n_obs=4, finite_fraction_min=0.9),
        ],
    )

    basic = validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    assert basic.passed

    strict = validate_output_bundle(ws, contract, ValidationLevel.STRICT)
    refusals = {i.code for i in strict.refusals}
    assert "EMBEDDING_TOO_MANY_NONFINITE" in refusals


def test_validation_is_read_only(tmp_path: Path) -> None:
    """A refusing validation must NOT mutate the workspace."""
    ws = tmp_path / "ws"
    _write_embedding(ws, n_obs=8)  # wrong n_obs → refusal
    snapshot = {p.name: p.read_bytes() for p in ws.iterdir()}
    contract = ModelOutputContract.default(expected_n_obs=4)
    validate_output_bundle(ws, contract, ValidationLevel.BASIC)
    after = {p.name: p.read_bytes() for p in ws.iterdir()}
    assert snapshot == after


# ---------------------------------------------------------------------------
# 3. Bundle writer
# ---------------------------------------------------------------------------


def _make_artifact_manifest(boot: BootContext) -> ArtifactManifest:
    image = ImageIdentity.registry_digest("sha256:" + "f" * 64)
    manifest_hash = compute_manifest_hash("jobs: []\n")
    params_hash = compute_params_hash({"n_latent": 4})
    fingerprint = {"slug": "demo", "n_obs": 4, "n_vars": 8}
    logical = compute_logical_run_id(
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
    )
    return ArtifactManifest(
        logical_run_id=logical,
        physical_attempt_id=new_physical_attempt_id(),
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
        produced_at=ProducedAt.from_dict(produced_at_now(boot)),
        produced_by=ProducedBy(mvd_version=boot.mvd_version),
        owner_token="owner-test",
    )


def test_bundle_round_trip(tmp_path: Path, good_workspace: Path) -> None:
    boot = BootContext.new(mvd_version="0.1.0-test")
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(good_workspace, contract, ValidationLevel.BASIC)
    assert report.passed

    manifest = _make_artifact_manifest(boot)
    manifest.artifacts = list(report.artifact_entries)

    manifest_input = tmp_path / "run_manifest.yaml"
    manifest_input.write_text("jobs: []\n", encoding="utf-8")
    log_file = tmp_path / "container.log"
    log_file.write_text("ok\n", encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    write_bundle(
        bundle_dir,
        BundleInputs(
            artifact_manifest=manifest,
            outputs={
                "embeddings.h5": good_workspace / "embeddings.h5",
                "metrics.json": good_workspace / "metrics.json",
                "umap.png": good_workspace / "umap.png",
            },
            inputs={"run_manifest.yaml": manifest_input},
            logs={"container.log": log_file},
            environment={"mvd_version": boot.mvd_version},
            validation_report=report.to_dict(),
        ),
    )

    assert (bundle_dir / ARTIFACT_MANIFEST_FILENAME).is_file()
    assert (bundle_dir / "outputs" / "embeddings.h5").is_file()
    assert (bundle_dir / "outputs" / "metrics.json").is_file()
    assert (bundle_dir / "outputs" / "umap.png").is_file()
    assert (bundle_dir / "inputs" / "run_manifest.yaml").is_file()
    assert (bundle_dir / "logs" / "container.log").is_file()
    assert (bundle_dir / "environment.json").is_file()
    assert (bundle_dir / "validation_report.json").is_file()
    assert (bundle_dir / "manifest.txt").is_file()
    assert (bundle_dir / "README.md").is_file()

    # Round-trip the manifest through the verified reader.
    loaded = read_manifest(bundle_dir)
    assert loaded.logical_run_id == manifest.logical_run_id
    assert {a.name for a in loaded.artifacts} == {
        "embeddings.h5",
        "metrics.json",
        "umap.png",
    }


def test_bundle_input_index_records_checksums(tmp_path: Path, good_workspace: Path) -> None:
    boot = BootContext.new(mvd_version="0.1.0-test")
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(good_workspace, contract)
    manifest = _make_artifact_manifest(boot)
    manifest.artifacts = list(report.artifact_entries)
    bundle_dir = tmp_path / "bundle"
    write_bundle(
        bundle_dir,
        BundleInputs(
            artifact_manifest=manifest,
            outputs={"embeddings.h5": good_workspace / "embeddings.h5"},
        ),
    )
    index = json.loads((bundle_dir / "manifest.txt").read_text())
    assert any(e["name"] == "embeddings.h5" for e in index["outputs"])
    for entry in index["outputs"]:
        assert len(entry["sha256"]) == 64


# ---------------------------------------------------------------------------
# 4. run_attempt_manifest writer
# ---------------------------------------------------------------------------


def test_run_attempt_manifest_round_trip(tmp_path: Path) -> None:
    boot = BootContext.new(mvd_version="0.1.0-test")
    attempt = RunAttemptManifest(
        physical_attempt_id=new_physical_attempt_id(),
        logical_run_id="x" * 64,
        manifest_hash="y" * 64,
        params_hash="z" * 64,
        image_identity=ImageIdentity.registry_digest("sha256:" + "a" * 64).to_dict(),
        mv_contract_version="1",
        final_state="EVALUATION_FAILED",
        failure_reason="post-flight validator refused: EMBEDDING_TOO_MANY_NONFINITE",
        produced_at=produced_at_now(boot),
        produced_by=ProducedBy(mvd_version=boot.mvd_version).to_dict(),
        state_transitions=[
            {"from": "TRAINING_SUCCEEDED", "to": "EVALUATING", "at": {}},
            {"from": "EVALUATING", "to": "EVALUATION_FAILED", "at": {}},
        ],
        recovery_hint="Re-run with --validators basic to inspect; review NaN sources.",
    )
    target = tmp_path / "failed"
    write_run_attempt_manifest(target, attempt)

    loaded = json.loads((target / "run_attempt_manifest.json").read_text())
    assert loaded["final_state"] == "EVALUATION_FAILED"
    assert loaded["recovery_hint"]
    assert loaded["state_transitions"]
    assert loaded["image_identity"]["kind"] == "registry_digest"
