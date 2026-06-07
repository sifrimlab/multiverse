"""Image identity as a union type (STRATEGY R10).

Image identity is one of four variants, declared at admission time and
recorded in the journal and artifact manifest. Strict (publication) mode
refuses anything but ``registry_digest`` or ``build_context_hash``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ImageIdentityKind(str, Enum):
    """The four image-identity variants admissible under R10.

    Ordered from most to least reproducible: a registry digest or
    build-context hash pins the image content exactly; an unverified local
    tag pins nothing and is refused under ``--strict``.
    """

    REGISTRY_DIGEST = "registry_digest"
    LOCAL_IMAGE_ID = "local_image_id"
    BUILD_CONTEXT_HASH = "build_context_hash"
    UNVERIFIED_LOCAL = "unverified_local"
    SIF_DIGEST = "sif_digest"
    """Apptainer/Singularity SIF file sha256. Carries a ``built_from``
    back-pointer to the source OCI digest so cross-backend manifests
    remain comparable (STRATEGY M2 / Addendum B)."""


_STRICT_KINDS = frozenset(
    {
        ImageIdentityKind.REGISTRY_DIGEST,
        ImageIdentityKind.BUILD_CONTEXT_HASH,
        ImageIdentityKind.SIF_DIGEST,
    }
)
"""SIF_DIGEST is strict-acceptable only when its ``built_from`` is set —
checked by :meth:`ImageIdentity.is_strict_acceptable`."""


@dataclass(frozen=True)
class ImageIdentity:
    """Tagged union for the four image-identity variants in R10.

    The ``value`` is the bytes participating in logical-run-ID derivation:

    * ``registry_digest`` → ``sha256:...`` returned by ``docker pull``.
    * ``local_image_id`` → docker image ID (``sha256:...``) of a locally
      tagged image with no registry provenance.
    * ``build_context_hash`` → deterministic hash of the build context that
      produced the local image (Dockerfile + tracked sources).
    * ``unverified_local`` → human-readable tag; emits a warning chip and
      is refused under ``--strict``.
    """

    kind: ImageIdentityKind
    value: str
    note: Optional[str] = None
    dockerfile_path: Optional[str] = None
    context_root: Optional[str] = None
    built_from: Optional[str] = None
    """Back-pointer for derived identities. Used by SIF_DIGEST to record
    the source OCI digest so a SIF run and an OCI run can be proven to
    come from the same image (STRATEGY M2)."""
    built_by: Optional[str] = None
    """One of ``"ci"``, ``"apptainer-pull-runtime"``, ``"author-supplied"``,
    or ``None``. Documentation only; not part of the strict-acceptability
    check (the built_from invariant covers that)."""

    def __post_init__(self) -> None:
        """Coerce ``kind`` from a raw string and validate that ``value`` is set."""
        if not isinstance(self.kind, ImageIdentityKind):
            object.__setattr__(self, "kind", ImageIdentityKind(self.kind))
        if not self.value or not isinstance(self.value, str):
            raise ValueError(
                "ImageIdentity.value must be a non-empty string identifying "
                "the image (digest, image id, build-context hash, or tag)"
            )

    @property
    def is_strict_acceptable(self) -> bool:
        """True iff this identity variant is acceptable under ``--strict``.

        ``SIF_DIGEST`` is strict-acceptable only when its ``built_from``
        is set; an SIF whose source OCI is unknown is no more reproducible
        than an ``unverified_local``.
        """
        if self.kind is ImageIdentityKind.SIF_DIGEST:
            return bool(self.built_from)
        return self.kind in _STRICT_KINDS

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-ready dict, omitting unset optional fields."""
        out: Dict[str, Any] = {"kind": self.kind.value, "value": self.value}
        if self.note is not None:
            out["note"] = self.note
        if self.dockerfile_path is not None:
            out["dockerfile_path"] = self.dockerfile_path
        if self.context_root is not None:
            out["context_root"] = self.context_root
        if self.built_from is not None:
            out["built_from"] = self.built_from
        if self.built_by is not None:
            out["built_by"] = self.built_by
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageIdentity":
        """Reconstruct an ``ImageIdentity`` from its ``to_dict`` form.

        Args:
            data: A dict previously produced by :meth:`to_dict`, typically
                deserialized from ``artifact_manifest.json``.

        Returns:
            The corresponding ``ImageIdentity`` instance.
        """
        return cls(
            kind=ImageIdentityKind(data["kind"]),
            value=str(data["value"]),
            note=data.get("note"),
            dockerfile_path=data.get("dockerfile_path"),
            context_root=data.get("context_root"),
            built_from=data.get("built_from"),
            built_by=data.get("built_by"),
        )

    # Convenience constructors --------------------------------------------------

    @classmethod
    def registry_digest(cls, digest: str) -> "ImageIdentity":
        """Build a strict-acceptable identity from a registry ``sha256:`` digest."""
        return cls(kind=ImageIdentityKind.REGISTRY_DIGEST, value=digest)

    @classmethod
    def local_image_id(cls, image_id: str) -> "ImageIdentity":
        """Build an identity from a local docker image ID with no registry provenance."""
        return cls(
            kind=ImageIdentityKind.LOCAL_IMAGE_ID,
            value=image_id,
            note="no registry digest available",
        )

    @classmethod
    def build_context_hash(
        cls,
        context_hash: str,
        dockerfile_path: str,
        context_root: str,
    ) -> "ImageIdentity":
        """Build a strict-acceptable identity from a deterministic build-context hash."""
        return cls(
            kind=ImageIdentityKind.BUILD_CONTEXT_HASH,
            value=context_hash,
            dockerfile_path=dockerfile_path,
            context_root=context_root,
        )

    @classmethod
    def unverified_local(cls, tag_or_id: str) -> "ImageIdentity":
        """Build a non-pinnable identity from a bare tag; refused under ``--strict``."""
        return cls(
            kind=ImageIdentityKind.UNVERIFIED_LOCAL,
            value=tag_or_id,
            note="not pinnable",
        )

    @classmethod
    def sif_digest(
        cls,
        sif_digest: str,
        *,
        built_from: Optional[str] = None,
        built_by: Optional[str] = None,
    ) -> "ImageIdentity":
        """Apptainer/Singularity SIF identity.

        ``built_from`` is the source OCI digest (``sha256:...``) the SIF
        was built from; without it, the identity is not strict-acceptable
        even though the SIF itself is content-addressed.

        ``built_by`` documents how the SIF was produced (``"ci"`` for a
        pre-built artifact, ``"apptainer-pull-runtime"`` for a runtime
        ``apptainer pull docker://...``).
        """
        return cls(
            kind=ImageIdentityKind.SIF_DIGEST,
            value=sif_digest,
            built_from=built_from,
            built_by=built_by,
        )


def verify_runtime_identity_matches_source(
    source: ImageIdentity, runtime: Optional[ImageIdentity]
) -> None:
    """Enforce the M2 dual-digest invariant.

    When ``runtime`` is supplied (Apptainer execution), assert:

    * ``runtime.kind == SIF_DIGEST`` — the only legal runtime-derived
      kind under the current design.
    * ``runtime.built_from == source.value`` — the SIF must point back
      to the source OCI digest that the manifest claims as truth.

    The dual-digest pair (OCI source digest + executed SIF sha256) ties the
    promoted result to both registry provenance and the binary that actually
    ran. Callers (promotion saga, doctor) treat a violation as fail-the-run.

    Args:
        source: The manifest's source-of-truth ``image_identity`` (the OCI
            digest the bundle claims to descend from).
        runtime: The identity of the image that actually executed. ``None``
            for non-Apptainer backends, in which case the check is a no-op.

    Raises:
        ValueError: If ``runtime`` is not a ``SIF_DIGEST``, lacks
            ``built_from``, or its ``built_from`` does not equal
            ``source.value``.
    """
    if runtime is None:
        return
    if runtime.kind is not ImageIdentityKind.SIF_DIGEST:
        raise ValueError(
            f"runtime_image_identity must be of kind SIF_DIGEST, got "
            f"{runtime.kind.value!r}"
        )
    if not runtime.built_from:
        raise ValueError(
            "runtime_image_identity (SIF) is missing built_from; cannot "
            "verify it derives from the manifest's image_identity"
        )
    if runtime.built_from != source.value:
        raise ValueError(
            f"runtime_image_identity.built_from={runtime.built_from!r} does "
            f"not match image_identity.value={source.value!r}; the SIF "
            "appears to have been built from a different source than the "
            "manifest claims"
        )
