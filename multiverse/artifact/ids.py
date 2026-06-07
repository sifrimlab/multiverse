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
from typing import Any, Mapping, Optional

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
    """Return the hex sha256 over canonical-JSON-encoded params.

    Args:
        params: The model parameter mapping. Encoded with canonical JSON
            (sorted keys, tight separators) so equal params hash equally
            regardless of insertion order or process.

    Returns:
        The hex-encoded sha256 of the canonical params encoding; a component
        of the logical-run-ID.
    """
    return hashlib.sha256(_canonical_json(dict(params))).hexdigest()


def compute_manifest_hash(manifest_text: str) -> str:
    """Return the hex sha256 of the *raw* manifest text.

    Computed over the on-disk bytes of ``run_manifest.yaml`` so that two
    manifests with the same logical content but different whitespace produce
    different hashes — comparing manifests semantically is the runner's job,
    not the hash function's.

    Args:
        manifest_text: The exact text of the run manifest as read from disk.

    Returns:
        The hex-encoded sha256 of the UTF-8 manifest bytes.
    """
    return hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()


def compute_dataset_fingerprint_hash(fingerprint: Mapping[str, Any]) -> str:
    """Return the hex sha256 of a dataset fingerprint mapping.

    Args:
        fingerprint: The dataset fingerprint dict (shape, content digests, and
            other identity-bearing fields of the input data).

    Returns:
        The hex-encoded sha256 of the canonical-JSON fingerprint encoding.
    """
    return hashlib.sha256(_canonical_json(dict(fingerprint))).hexdigest()


def runtime_identity_fingerprint(
    *,
    seed: Any = None,
    preprocessing: Any = None,
    model_version: Any = None,
) -> dict:
    """Collect behaviour-affecting runtime fields outside ``model_params``.

    These fields live *outside* ``model_params`` but still change the
    scientific result or the execution recipe (STRATEGY: MVD completion key).

    Only non-empty values are included. An all-empty result means "no runtime
    fields" and, fed back into :func:`compute_logical_run_id`, yields a logical
    ID byte-identical to the historical four-part formula — so existing
    identities are preserved when none of these fields apply.

    Args:
        seed: RNG seed for the run, if one affects the result.
        preprocessing: Preprocessing recipe identifier or descriptor that
            alters the input pipeline.
        model_version: Model code/weights version when it is not already
            captured by the image identity.

    Returns:
        A dict containing only the non-empty fields among the three, suitable
        as the ``runtime_fingerprint`` argument to :func:`compute_logical_run_id`.
    """
    fingerprint: dict = {}
    if seed is not None:
        fingerprint["seed"] = seed
    if preprocessing:
        fingerprint["preprocessing"] = preprocessing
    if model_version:
        fingerprint["model_version"] = model_version
    return fingerprint


def compute_logical_run_id(
    *,
    manifest_hash: str,
    dataset_fingerprint: Mapping[str, Any],
    image_identity: ImageIdentity,
    params_hash: str,
    mv_contract_version: str,
    runtime_fingerprint: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the deterministic logical-run-ID for a recipe.

    The same inputs always hash to the same logical ID, regardless of OS,
    Python build, or process. This is the property tested in
    ``test_artifact_contract.py::test_logical_run_id_stability``.

    ``runtime_fingerprint`` carries behaviour-affecting fields that are not
    part of ``model_params`` (seed, preprocessing, model version — see
    :func:`runtime_identity_fingerprint`). When it is ``None`` or empty the
    payload is byte-identical to the original four-part formula, so adding the
    parameter does not change historical identities. There is exactly one
    canonical identity: every executor and the resume planner derive it here.

    Args:
        manifest_hash: Hash of the raw run manifest text (see
            :func:`compute_manifest_hash`).
        dataset_fingerprint: Identity-bearing fingerprint of the input data.
        image_identity: The source-of-truth image identity (its ``value`` is
            the bytes that participate in the hash).
        params_hash: Hash of the canonical model params.
        mv_contract_version: Model output contract version; bumping it
            deliberately invalidates cached identities.
        runtime_fingerprint: Optional behaviour-affecting runtime fields (seed,
            preprocessing, model version). ``None`` or empty reproduces the
            original four-part identity.

    Returns:
        The deterministic hex sha256 logical-run-ID for the recipe.
    """
    fingerprint_hash = compute_dataset_fingerprint_hash(dataset_fingerprint)
    parts = [
        manifest_hash,
        fingerprint_hash,
        image_identity.value,
        params_hash,
        mv_contract_version,
    ]
    if runtime_fingerprint:
        parts.append(compute_dataset_fingerprint_hash(dict(runtime_fingerprint)))
    payload = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_physical_attempt_id() -> str:
    """Generate a UUID4 physical attempt ID."""
    return str(uuid.uuid4())
