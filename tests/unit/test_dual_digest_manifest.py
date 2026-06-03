"""Tests for the M2 dual-digest manifest invariant and ImageIdentity
extensions."""

from __future__ import annotations

import pytest

from multiverse.artifact import (ArtifactManifest, ImageIdentity,
                                 ImageIdentityKind, ProducedAt, ProducedBy,
                                 verify_runtime_identity_matches_source)

pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# ImageIdentity.sif_digest + strict acceptability
# ---------------------------------------------------------------------------


def test_sif_with_built_from_is_strict_acceptable():
    rt = ImageIdentity.sif_digest(
        "sha256:fff",
        built_from="sha256:src",
        built_by="ci",
    )
    assert rt.kind is ImageIdentityKind.SIF_DIGEST
    assert rt.is_strict_acceptable


def test_sif_without_built_from_is_not_strict_acceptable():
    rt = ImageIdentity.sif_digest("sha256:fff")
    assert rt.kind is ImageIdentityKind.SIF_DIGEST
    assert not rt.is_strict_acceptable


def test_sif_round_trips_through_dict():
    rt = ImageIdentity.sif_digest("sha256:fff", built_from="sha256:src", built_by="ci")
    again = ImageIdentity.from_dict(rt.to_dict())
    assert again == rt


def test_pre_m2_manifest_without_runtime_field_still_loads():
    """Schema-compat: a manifest written before runtime_image_identity
    existed must still round-trip."""
    src = ImageIdentity.registry_digest("sha256:abc")
    legacy = {
        "schema_version": "1",
        "logical_run_id": "lr1",
        "physical_attempt_id": "pa1",
        "manifest_hash": "h",
        "dataset_fingerprint": {"slug": "d", "n_obs": 1},
        "image_identity": src.to_dict(),
        "params_hash": "p",
        "mv_contract_version": "v",
        "produced_at": {
            "wall": "2026-05-28T00:00:00+00:00",
            "monotonic_ns": 0,
            "tz": "UTC",
            "mvd_boot_id": "b",
        },
        "produced_by": {"mvd_version": "0.1.0-mvd", "git_commit": None},
        "artifacts": [],
        "state_transitions": [],
    }
    m = ArtifactManifest.from_dict(legacy)
    assert m.runtime_image_identity is None
    # Round-trip is stable.
    assert ArtifactManifest.from_dict(m.to_dict()).runtime_image_identity is None


# ---------------------------------------------------------------------------
# verify_runtime_identity_matches_source
# ---------------------------------------------------------------------------


def test_no_runtime_identity_passes():
    src = ImageIdentity.registry_digest("sha256:abc")
    verify_runtime_identity_matches_source(src, None)  # no raise


def test_runtime_must_be_sif():
    src = ImageIdentity.registry_digest("sha256:abc")
    bad = ImageIdentity.registry_digest("sha256:abc")
    with pytest.raises(ValueError, match="SIF_DIGEST"):
        verify_runtime_identity_matches_source(src, bad)


def test_runtime_missing_built_from_rejected():
    src = ImageIdentity.registry_digest("sha256:abc")
    bad = ImageIdentity.sif_digest("sha256:fff")
    with pytest.raises(ValueError, match="built_from"):
        verify_runtime_identity_matches_source(src, bad)


def test_runtime_built_from_must_match_source():
    src = ImageIdentity.registry_digest("sha256:abc")
    bad = ImageIdentity.sif_digest("sha256:fff", built_from="sha256:OTHER")
    with pytest.raises(ValueError, match="does not match"):
        verify_runtime_identity_matches_source(src, bad)


def test_matching_runtime_passes():
    src = ImageIdentity.registry_digest("sha256:abc")
    rt = ImageIdentity.sif_digest("sha256:fff", built_from="sha256:abc")
    verify_runtime_identity_matches_source(src, rt)


# ---------------------------------------------------------------------------
# Manifest serialization with runtime_image_identity
# ---------------------------------------------------------------------------


def _make_manifest(
    *,
    image: ImageIdentity,
    runtime: ImageIdentity | None = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        logical_run_id="lr",
        physical_attempt_id="pa",
        manifest_hash="h",
        dataset_fingerprint={"slug": "d", "n_obs": 10},
        image_identity=image,
        params_hash="p",
        mv_contract_version="v",
        produced_at=ProducedAt(
            wall="2026-05-28T00:00:00+00:00", monotonic_ns=0, tz="UTC", mvd_boot_id="b"
        ),
        produced_by=ProducedBy(mvd_version="0.1.0-mvd", git_commit=None),
        runtime_image_identity=runtime,
    )


def test_round_trip_with_runtime_identity():
    src = ImageIdentity.registry_digest("sha256:abc")
    rt = ImageIdentity.sif_digest("sha256:fff", built_from="sha256:abc", built_by="ci")
    m = _make_manifest(image=src, runtime=rt)
    again = ArtifactManifest.from_dict(m.to_dict())
    assert again.runtime_image_identity == rt
    assert again.image_identity == src


# ---------------------------------------------------------------------------
# Cross-backend reproducibility (STRATEGY M2 acceptance)
# ---------------------------------------------------------------------------


def test_docker_and_apptainer_manifests_compare_equal_on_source_invariants():
    """Two manifests for the same (model, dataset, params) produced via
    different backends must agree on (oci_digest, params_hash,
    dataset_fingerprint) — the trinity that makes cross-backend results
    comparable per STRATEGY M2."""
    oci = ImageIdentity.registry_digest("sha256:identical-source")
    fingerprint = {"slug": "ds-x", "n_obs": 1234, "n_vars": 567}

    # Docker manifest: no runtime identity.
    docker = _make_manifest(image=oci, runtime=None)
    docker.dataset_fingerprint = dict(fingerprint)

    # Apptainer manifest: SIF identity carries built_from = oci_digest.
    sif = ImageIdentity.sif_digest(
        "sha256:sif-bytes", built_from="sha256:identical-source", built_by="ci"
    )
    apptainer = _make_manifest(image=oci, runtime=sif)
    apptainer.dataset_fingerprint = dict(fingerprint)

    def trinity(m: ArtifactManifest):
        return (m.image_identity.value, m.params_hash, dict(m.dataset_fingerprint))

    assert trinity(docker) == trinity(apptainer)

    # And the runtime identity, when present, must pass the invariant.
    verify_runtime_identity_matches_source(
        apptainer.image_identity, apptainer.runtime_image_identity
    )
