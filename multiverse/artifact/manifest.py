"""``artifact_manifest.json`` schema and atomic read/write.

The manifest is *not* the source of truth — the journal is (R3). But the
manifest is the durable, self-describing record alongside the artifact bytes,
and per R4 it is written as a three-step content-addressed commit with a
detached sha256 sidecar.

Readers MUST verify the sidecar before trusting the body, and MUST NOT mutate
the directory in response to a mismatch. Repair is reserved to
``multiverse doctor --repair`` and ``multiverse rebuild-index``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .checksums import (PathLike, atomic_write_bytes, fsync_path, sha256_bytes,
                        sha256_file)
from .errors import (ChecksumMismatchError, ManifestCorruptError,
                     ManifestMissingError)
from .image_identity import ImageIdentity

ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
ARTIFACT_MANIFEST_SHA256_FILENAME = "artifact_manifest.sha256"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArtifactEntry:
    """A single output artifact (e.g. ``embeddings.h5``)."""

    name: str
    sha256: str
    size: int
    produced: bool = True
    validated: bool = True
    role: Optional[str] = None  # e.g. "embedding", "metrics", "plot"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "sha256": self.sha256,
            "size": int(self.size),
            "produced": bool(self.produced),
            "validated": bool(self.validated),
        }
        if self.role is not None:
            out["role"] = self.role
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactEntry":
        return cls(
            name=str(data["name"]),
            sha256=str(data["sha256"]),
            size=int(data["size"]),
            produced=bool(data.get("produced", True)),
            validated=bool(data.get("validated", True)),
            role=data.get("role"),
        )

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        name: Optional[str] = None,
        role: Optional[str] = None,
        validated: bool = True,
    ) -> "ArtifactEntry":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"artifact not found: {p}")
        return cls(
            name=name or p.name,
            sha256=sha256_file(p),
            size=p.stat().st_size,
            produced=True,
            validated=validated,
            role=role,
        )


@dataclass
class StateTransition:
    """One primary-state change recorded in the artifact manifest timeline."""

    from_state: str
    to_state: str
    at: Dict[str, Any]  # produced_at-shaped struct from timestamps.py
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "from": self.from_state,
            "to": self.to_state,
            "at": dict(self.at),
        }
        if self.reason is not None:
            out["reason"] = self.reason
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateTransition":
        return cls(
            from_state=str(data["from"]),
            to_state=str(data["to"]),
            at=dict(data["at"]),
            reason=data.get("reason"),
        )


@dataclass
class ProducedAt:
    """Wall-clock and monotonic timestamps for manifest provenance."""

    wall: str
    monotonic_ns: int
    tz: str
    mvd_boot_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall": self.wall,
            "monotonic_ns": int(self.monotonic_ns),
            "tz": self.tz,
            "mvd_boot_id": self.mvd_boot_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProducedAt":
        return cls(
            wall=str(data["wall"]),
            monotonic_ns=int(data["monotonic_ns"]),
            tz=str(data["tz"]),
            mvd_boot_id=str(data["mvd_boot_id"]),
        )


@dataclass
class ProducedBy:
    mvd_version: str
    git_commit: Optional[str] = None
    degraded_capabilities: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    """Resolved owner of the run (G2). Absent from pre-G2 manifests."""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"mvd_version": self.mvd_version}
        if self.git_commit is not None:
            out["git_commit"] = self.git_commit
        if self.degraded_capabilities:
            out["degraded_capabilities"] = list(self.degraded_capabilities)
        if self.user_id is not None:
            out["user_id"] = self.user_id
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProducedBy":
        return cls(
            mvd_version=str(data["mvd_version"]),
            git_commit=data.get("git_commit"),
            degraded_capabilities=list(data.get("degraded_capabilities") or []),
            user_id=data.get("user_id"),
        )


@dataclass
class ResourceObservations:
    """Per-run resource observations (STRATEGY R11)."""

    peak_rss_bytes: Optional[int] = None
    peak_vram_bytes: Optional[int] = None
    oom_killed: bool = False
    broker_pressure_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"oom_killed": bool(self.oom_killed)}
        if self.peak_rss_bytes is not None:
            out["peak_rss_bytes"] = int(self.peak_rss_bytes)
        if self.peak_vram_bytes is not None:
            out["peak_vram_bytes"] = int(self.peak_vram_bytes)
        if self.broker_pressure_events:
            out["broker_pressure_events"] = [
                dict(ev) for ev in self.broker_pressure_events
            ]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResourceObservations":
        return cls(
            peak_rss_bytes=(
                int(data["peak_rss_bytes"])
                if data.get("peak_rss_bytes") is not None
                else None
            ),
            peak_vram_bytes=(
                int(data["peak_vram_bytes"])
                if data.get("peak_vram_bytes") is not None
                else None
            ),
            oom_killed=bool(data.get("oom_killed", False)),
            broker_pressure_events=list(data.get("broker_pressure_events") or []),
        )


@dataclass
class ArtifactManifest:
    """Self-describing record of a promoted artifact bundle.

    All hash/identity fields are required because the manifest is consumed by
    rebuild-index, export-import, and the GUI without further lookups.
    """

    logical_run_id: str
    physical_attempt_id: str
    manifest_hash: str
    dataset_fingerprint: Dict[str, Any]
    image_identity: ImageIdentity
    params_hash: str
    mv_contract_version: str
    produced_at: ProducedAt
    produced_by: ProducedBy
    artifacts: List[ArtifactEntry] = field(default_factory=list)
    state_transitions: List[StateTransition] = field(default_factory=list)
    owner_token: Optional[str] = None
    resource_observations: Optional[ResourceObservations] = None
    runtime_image_identity: Optional[ImageIdentity] = None
    """Set when the run executed via a derived image format (e.g. an
    Apptainer SIF built from an OCI source). ``image_identity`` remains
    the source-of-truth digest; ``runtime_image_identity`` records what
    actually ran. The two are linked by ``built_from``, verified at
    promotion (STRATEGY M2 dual-digest)."""
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA_VERSION

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "logical_run_id": self.logical_run_id,
            "physical_attempt_id": self.physical_attempt_id,
            "manifest_hash": self.manifest_hash,
            "dataset_fingerprint": dict(self.dataset_fingerprint),
            "image_identity": self.image_identity.to_dict(),
            "params_hash": self.params_hash,
            "mv_contract_version": self.mv_contract_version,
            "produced_at": self.produced_at.to_dict(),
            "produced_by": self.produced_by.to_dict(),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "state_transitions": [s.to_dict() for s in self.state_transitions],
        }
        if self.owner_token is not None:
            out["owner_token"] = self.owner_token
        if self.resource_observations is not None:
            out["resource_observations"] = self.resource_observations.to_dict()
        if self.runtime_image_identity is not None:
            out["runtime_image_identity"] = self.runtime_image_identity.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactManifest":
        schema_version = str(data.get("schema_version", ""))
        if schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ManifestCorruptError(
                f"unsupported manifest schema_version: {schema_version!r}, "
                f"expected {ARTIFACT_MANIFEST_SCHEMA_VERSION!r}"
            )
        try:
            return cls(
                schema_version=schema_version,
                logical_run_id=str(data["logical_run_id"]),
                physical_attempt_id=str(data["physical_attempt_id"]),
                manifest_hash=str(data["manifest_hash"]),
                dataset_fingerprint=dict(data["dataset_fingerprint"]),
                image_identity=ImageIdentity.from_dict(data["image_identity"]),
                params_hash=str(data["params_hash"]),
                mv_contract_version=str(data["mv_contract_version"]),
                produced_at=ProducedAt.from_dict(data["produced_at"]),
                produced_by=ProducedBy.from_dict(data["produced_by"]),
                artifacts=[
                    ArtifactEntry.from_dict(a) for a in data.get("artifacts", [])
                ],
                state_transitions=[
                    StateTransition.from_dict(s)
                    for s in data.get("state_transitions", [])
                ],
                owner_token=data.get("owner_token"),
                runtime_image_identity=(
                    ImageIdentity.from_dict(data["runtime_image_identity"])
                    if data.get("runtime_image_identity") is not None
                    else None
                ),
                resource_observations=(
                    ResourceObservations.from_dict(data["resource_observations"])
                    if data.get("resource_observations") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ManifestCorruptError(
                f"manifest missing required field: {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ManifestCorruptError(
                f"manifest field has wrong shape: {exc}"
            ) from exc

    def to_canonical_json_bytes(self) -> bytes:
        """Encode for on-disk write. Stable byte ordering across runs."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


# ---------------------------------------------------------------------------
# Atomic write / verified read
# ---------------------------------------------------------------------------


def write_manifest(
    artifact_dir: PathLike,
    manifest: ArtifactManifest,
    *,
    fsync: bool = True,
) -> str:
    """Atomically write ``artifact_manifest.json`` + ``artifact_manifest.sha256``.

    Sequence (R4):
        1. Write ``artifact_manifest.json.tmp`` and ``fsync`` the file.
        2. Compute sha256 over the bytes and write ``artifact_manifest.sha256``
           (atomic).
        3. Rename ``.tmp`` → ``artifact_manifest.json`` and ``fsync`` the
           parent directory inode.

    Returns the hex sha256 of the manifest body so callers can record it in
    the journal alongside the artifact directory path.
    """
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)

    body = manifest.to_canonical_json_bytes()
    body_sha = sha256_bytes(body)

    final = directory / ARTIFACT_MANIFEST_FILENAME
    tmp = directory / f"{ARTIFACT_MANIFEST_FILENAME}.tmp"
    sidecar = directory / ARTIFACT_MANIFEST_SHA256_FILENAME

    import os

    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, body)
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)

    sidecar_payload = f"{body_sha}  {ARTIFACT_MANIFEST_FILENAME}\n".encode("ascii")
    atomic_write_bytes(sidecar, sidecar_payload, fsync=fsync)

    os.replace(str(tmp), str(final))
    if fsync:
        fsync_path(directory)

    return body_sha


def read_manifest(artifact_dir: PathLike) -> ArtifactManifest:
    """Verified read of an artifact manifest.

    * Missing manifest → ``ManifestMissingError``.
    * Missing sidecar → ``ManifestMissingError`` (per R4, the sidecar is
      required; refusing to read without it prevents trusting an unsigned
      body).
    * Body sha256 ≠ sidecar → ``ChecksumMismatchError``. The function does
      NOT mutate the directory in this case.
    * Body parses but is structurally invalid → ``ManifestCorruptError``.
    """
    directory = Path(artifact_dir)
    body_path = directory / ARTIFACT_MANIFEST_FILENAME
    sidecar_path = directory / ARTIFACT_MANIFEST_SHA256_FILENAME

    if not body_path.is_file():
        raise ManifestMissingError(f"no artifact_manifest.json at {directory}")
    if not sidecar_path.is_file():
        raise ManifestMissingError(
            f"missing artifact_manifest.sha256 sidecar at {directory}"
        )

    expected_sha = _parse_sidecar(sidecar_path.read_text(encoding="ascii"))
    observed_sha = sha256_file(body_path)
    if expected_sha != observed_sha:
        raise ChecksumMismatchError(str(body_path), expected_sha, observed_sha)

    try:
        data = json.loads(body_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ManifestCorruptError(f"manifest is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestCorruptError("manifest top-level must be an object")

    return ArtifactManifest.from_dict(data)


def _parse_sidecar(text: str) -> str:
    """Parse the ``<sha>  <name>\\n`` sidecar format.

    Loose parsing is intentional: we only need the sha; spaces in the name
    are not supported because the sidecar is generated by us with a fixed
    filename.
    """
    line = text.strip().splitlines()[0] if text.strip() else ""
    if not line:
        raise ManifestCorruptError("artifact_manifest.sha256 is empty")
    parts = line.split()
    if not parts or len(parts[0]) != 64:
        raise ManifestCorruptError(
            f"artifact_manifest.sha256 has unexpected format: {line!r}"
        )
    return parts[0]


# ---------------------------------------------------------------------------
# Normalised comparison for daemon-vs-simple-mode bundle equivalence (R7)
# ---------------------------------------------------------------------------


_NONDETERMINISTIC_FIELDS: Sequence[Sequence[str]] = (
    ("physical_attempt_id",),
    ("owner_token",),
    ("produced_at",),
    ("produced_by", "git_commit"),
    ("produced_by", "mvd_version"),
    ("state_transitions",),
    ("resource_observations",),
)


def normalize_for_equivalence(manifest_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip fields that are expected to vary between two contract-equivalent
    bundles produced from the same recipe.

    Used by tests and by the R7 acceptance check (``simple-mode bundle
    equivalent to daemon bundle modulo approved nondeterministic fields``).
    """
    cleaned = json.loads(json.dumps(manifest_dict))  # deep-copy via JSON
    for path in _NONDETERMINISTIC_FIELDS:
        cursor = cleaned
        for key in path[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, dict):
            cursor.pop(path[-1], None)
    return cleaned
