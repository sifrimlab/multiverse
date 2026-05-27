"""Identity derivation (STRATEGY S16 / R10 / ADR 0001 §11).

Two identity surfaces are separated:

* **Logical run ID** — content-addressed; equal across attempts of the same
  recipe. Derived from the deterministic concatenation of manifest hash,
  dataset fingerprint, image identity value, params hash, and contract
  version.
* **Physical attempt ID** — UUID4; unique per execution.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Mapping

from .image_identity import ImageIdentity

# Stable separators so that two writers in two processes produce byte-equal
# canonical encodings.
PARAMS_HASH_CANONICAL_SEPARATORS = (",", ":")


def _canonical_json(obj: Any) -> bytes:
    """Canonical JSON encoding used for every hash input.

    ``sort_keys=True`` and tight separators ensure stability across Python
    versions and platforms.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=PARAMS_HASH_CANONICAL_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_params_hash(params: Mapping[str, Any]) -> str:
    """Return the hex sha256 over canonical-JSON-encoded params."""
    return hashlib.sha256(_canonical_json(dict(params))).hexdigest()


def compute_manifest_hash(manifest_text: str) -> str:
    """Return the hex sha256 of the *raw* manifest text.

    Computed over the on-disk bytes of ``run_manifest.yaml`` so that two
    manifests with the same logical content but different whitespace produce
    different hashes — comparing manifests semantically is the runner's job,
    not the hash function's.
    """
    return hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()


def compute_dataset_fingerprint_hash(fingerprint: Mapping[str, Any]) -> str:
    """Helper for callers that have a dict fingerprint and need its hash."""
    return hashlib.sha256(_canonical_json(dict(fingerprint))).hexdigest()


def compute_logical_run_id(
    *,
    manifest_hash: str,
    dataset_fingerprint: Mapping[str, Any],
    image_identity: ImageIdentity,
    params_hash: str,
    mv_contract_version: str,
) -> str:
    """Return the deterministic logical-run-ID for a recipe.

    The same inputs always hash to the same logical ID, regardless of OS,
    Python build, or process. This is the property tested in
    ``test_artifact_contract.py::test_logical_run_id_stability``.
    """
    fingerprint_hash = compute_dataset_fingerprint_hash(dataset_fingerprint)
    parts = (
        manifest_hash,
        fingerprint_hash,
        image_identity.value,
        params_hash,
        mv_contract_version,
    )
    payload = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_physical_attempt_id() -> str:
    """Generate a UUID4 physical attempt ID."""
    return str(uuid.uuid4())
