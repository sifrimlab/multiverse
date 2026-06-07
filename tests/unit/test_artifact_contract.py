"""Milestone-1 exit-gate tests for the artifact contract library.

Coverage:
    1. Atomic manifest write + verified read round-trip.
    2. Checksum mismatch is detected and does not mutate the directory.
    3. Logical run ID is stable for identical inputs and changes for
       different inputs.
    4. Normalised bundle comparison is empty for two bundles produced from
       the same recipe.

These tests must not import MLflow, Optuna, Docker, Streamlit, or any other
non-contract dependency — the artifact module is the hot-path-clean kernel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from multiverse.artifact import (ARTIFACT_MANIFEST_FILENAME,
                                 ARTIFACT_MANIFEST_SHA256_FILENAME,
                                 ArtifactEntry, ArtifactManifest, BootContext,
                                 ChecksumMismatchError, ImageIdentity,
                                 ImageIdentityKind, ManifestCorruptError,
                                 ManifestMissingError, ProducedAt, ProducedBy,
                                 compute_logical_run_id, compute_manifest_hash,
                                 compute_params_hash, new_physical_attempt_id,
                                 produced_at_now, read_manifest, sha256_bytes,
                                 write_manifest)
from multiverse.artifact.manifest import normalize_for_equivalence

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def boot() -> BootContext:
    return BootContext.new(mvd_version="0.1.0-test", git_commit="deadbeef")


def _make_manifest(
    boot: BootContext,
    *,
    image_identity: ImageIdentity | None = None,
    params: dict | None = None,
    physical_attempt_id: str | None = None,
) -> ArtifactManifest:
    image_identity = image_identity or ImageIdentity.registry_digest(
        "sha256:" + "a" * 64
    )
    params = params or {"n_latent": 10, "n_layers": 2}
    manifest_hash = compute_manifest_hash("jobs: []\n")
    params_hash = compute_params_hash(params)
    fingerprint = {"slug": "demo", "n_obs": 100, "n_vars": 50}
    logical = compute_logical_run_id(
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image_identity,
        params_hash=params_hash,
        mv_contract_version="1",
    )
    return ArtifactManifest(
        logical_run_id=logical,
        physical_attempt_id=physical_attempt_id or new_physical_attempt_id(),
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image_identity,
        params_hash=params_hash,
        mv_contract_version="1",
        produced_at=ProducedAt.from_dict(produced_at_now(boot)),
        produced_by=ProducedBy(
            mvd_version=boot.mvd_version, git_commit=boot.git_commit
        ),
        artifacts=[
            ArtifactEntry(
                name="embeddings.h5",
                sha256="b" * 64,
                size=12345,
                role="embedding",
            ),
        ],
        owner_token="owner-test",
    )


# ---------------------------------------------------------------------------
# 1. Atomic write + verified read round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_every_field(tmp_path: Path, boot: BootContext) -> None:
    manifest = _make_manifest(boot)
    body_sha = write_manifest(tmp_path, manifest)

    body = (tmp_path / ARTIFACT_MANIFEST_FILENAME).read_bytes()
    assert sha256_bytes(body) == body_sha, "returned sha must match on-disk bytes"

    loaded = read_manifest(tmp_path)
    assert loaded.to_dict() == manifest.to_dict()


def test_write_is_idempotent_under_overwrite(tmp_path: Path, boot: BootContext) -> None:
    manifest_a = _make_manifest(boot)
    write_manifest(tmp_path, manifest_a)
    # Overwriting with a new manifest must atomically replace both the body
    # and the sidecar; the sidecar must never refer to the old body.
    manifest_b = _make_manifest(boot, params={"n_latent": 20})
    write_manifest(tmp_path, manifest_b)

    loaded = read_manifest(tmp_path)
    assert loaded.logical_run_id == manifest_b.logical_run_id
    assert loaded.logical_run_id != manifest_a.logical_run_id


def test_tmp_file_is_cleaned_up_after_successful_write(
    tmp_path: Path, boot: BootContext
) -> None:
    write_manifest(tmp_path, _make_manifest(boot))
    assert not (tmp_path / f"{ARTIFACT_MANIFEST_FILENAME}.tmp").exists()


# ---------------------------------------------------------------------------
# 2. Checksum mismatch detection without mutation (R4)
# ---------------------------------------------------------------------------


def test_truncating_manifest_is_detected_as_corruption(
    tmp_path: Path, boot: BootContext
) -> None:
    write_manifest(tmp_path, _make_manifest(boot))
    body_path = tmp_path / ARTIFACT_MANIFEST_FILENAME
    sidecar_path = tmp_path / ARTIFACT_MANIFEST_SHA256_FILENAME

    # Simulate the acceptance criterion verbatim: truncate the body after
    # promotion. Sidecar is intact so the mismatch must be visible.
    body_path.write_bytes(b"")

    with pytest.raises(ChecksumMismatchError) as exc_info:
        read_manifest(tmp_path)
    assert exc_info.value.path.endswith(ARTIFACT_MANIFEST_FILENAME)
    # The library must NOT have moved or renamed anything (R4 requires
    # repair only via explicit commands).
    assert body_path.exists()
    assert sidecar_path.exists()
    assert not (tmp_path / "artifact_manifest.corrupt.json").exists()


def test_manual_edit_of_manifest_is_detected(tmp_path: Path, boot: BootContext) -> None:
    write_manifest(tmp_path, _make_manifest(boot))
    body_path = tmp_path / ARTIFACT_MANIFEST_FILENAME

    # Over-eager user edits the manifest in place.
    data = json.loads(body_path.read_text())
    data["params_hash"] = "tampered"
    body_path.write_text(json.dumps(data, indent=2))

    with pytest.raises(ChecksumMismatchError):
        read_manifest(tmp_path)


def test_missing_sidecar_is_missing_not_corrupt(
    tmp_path: Path, boot: BootContext
) -> None:
    write_manifest(tmp_path, _make_manifest(boot))
    (tmp_path / ARTIFACT_MANIFEST_SHA256_FILENAME).unlink()
    with pytest.raises(ManifestMissingError):
        read_manifest(tmp_path)


def test_missing_manifest_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ManifestMissingError):
        read_manifest(tmp_path)


def test_bad_schema_version_is_corrupt(tmp_path: Path, boot: BootContext) -> None:
    write_manifest(tmp_path, _make_manifest(boot))
    body_path = tmp_path / ARTIFACT_MANIFEST_FILENAME
    sidecar_path = tmp_path / ARTIFACT_MANIFEST_SHA256_FILENAME

    data = json.loads(body_path.read_text())
    data["schema_version"] = "999"
    new_body = json.dumps(
        data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    body_path.write_bytes(new_body)
    sidecar_path.write_text(f"{sha256_bytes(new_body)}  {ARTIFACT_MANIFEST_FILENAME}\n")

    with pytest.raises(ManifestCorruptError):
        read_manifest(tmp_path)


# ---------------------------------------------------------------------------
# 3. Logical run ID stability
# ---------------------------------------------------------------------------


def test_logical_run_id_is_stable_for_identical_inputs() -> None:
    image = ImageIdentity.registry_digest("sha256:" + "c" * 64)
    fingerprint = {"slug": "demo", "n_obs": 100}
    params = {"a": 1, "b": [2, 3], "c": {"nested": True}}
    args = dict(
        manifest_hash="m" * 64,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=compute_params_hash(params),
        mv_contract_version="1",
    )
    assert compute_logical_run_id(**args) == compute_logical_run_id(**args)


def test_logical_run_id_changes_when_image_digest_changes() -> None:
    common = dict(
        manifest_hash="m" * 64,
        dataset_fingerprint={"slug": "demo"},
        params_hash=compute_params_hash({"a": 1}),
        mv_contract_version="1",
    )
    a = compute_logical_run_id(
        image_identity=ImageIdentity.registry_digest("sha256:" + "1" * 64),
        **common,
    )
    b = compute_logical_run_id(
        image_identity=ImageIdentity.registry_digest("sha256:" + "2" * 64),
        **common,
    )
    assert a != b, "rebuilt image at same tag must produce a different logical ID"


def test_logical_run_id_independent_of_dict_ordering() -> None:
    image = ImageIdentity.registry_digest("sha256:" + "d" * 64)
    a_params = {"a": 1, "b": 2, "c": 3}
    b_params = {"c": 3, "b": 2, "a": 1}
    # Hash inputs must be canonical so dict insertion order cannot leak in.
    assert compute_params_hash(a_params) == compute_params_hash(b_params)

    common = dict(
        manifest_hash="m" * 64,
        image_identity=image,
        mv_contract_version="1",
    )
    a = compute_logical_run_id(
        dataset_fingerprint={"slug": "demo", "n_obs": 1},
        params_hash=compute_params_hash(a_params),
        **common,
    )
    b = compute_logical_run_id(
        dataset_fingerprint={"n_obs": 1, "slug": "demo"},
        params_hash=compute_params_hash(b_params),
        **common,
    )
    assert a == b


def test_params_hash_rejects_nan() -> None:
    with pytest.raises(ValueError):
        compute_params_hash({"x": float("nan")})


# ---------------------------------------------------------------------------
# 4. Normalised bundle equivalence (R7 acceptance setup)
# ---------------------------------------------------------------------------


def test_normalize_for_equivalence_drops_nondeterministic_fields(
    boot: BootContext,
) -> None:
    a = _make_manifest(boot)
    b = _make_manifest(boot)
    assert a.physical_attempt_id != b.physical_attempt_id  # sanity

    norm_a = normalize_for_equivalence(a.to_dict())
    norm_b = normalize_for_equivalence(b.to_dict())
    assert (
        norm_a == norm_b
    ), "normalised manifests from identical recipes must be byte-equal"


def test_normalize_keeps_logical_run_id_and_artifact_hashes(
    boot: BootContext,
) -> None:
    a = _make_manifest(boot)
    norm = normalize_for_equivalence(a.to_dict())
    assert norm["logical_run_id"] == a.logical_run_id
    assert norm["artifacts"][0]["sha256"] == "b" * 64
    # Nondeterministic fields are gone.
    assert "physical_attempt_id" not in norm
    assert "produced_at" not in norm
    assert "state_transitions" not in norm


# ---------------------------------------------------------------------------
# 5. Image identity strict acceptability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identity,strict_ok",
    [
        (ImageIdentity.registry_digest("sha256:" + "a" * 64), True),
        (
            ImageIdentity.build_context_hash(
                "sha256:" + "b" * 64, "Dockerfile", "/ctx"
            ),
            True,
        ),
        (ImageIdentity.local_image_id("sha256:" + "c" * 64), False),
        (ImageIdentity.unverified_local("multiverse-pca:latest"), False),
    ],
)
def test_strict_acceptability_matches_R10(identity, strict_ok) -> None:
    assert identity.is_strict_acceptable == strict_ok


def test_image_identity_round_trips_through_dict() -> None:
    original = ImageIdentity.build_context_hash(
        "sha256:" + "e" * 64,
        dockerfile_path="container/Dockerfile",
        context_root="/abs/ctx",
    )
    restored = ImageIdentity.from_dict(original.to_dict())
    assert restored == original


def test_image_identity_rejects_empty_value() -> None:
    with pytest.raises(ValueError):
        ImageIdentity(kind=ImageIdentityKind.REGISTRY_DIGEST, value="")
