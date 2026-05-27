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
    REGISTRY_DIGEST = "registry_digest"
    LOCAL_IMAGE_ID = "local_image_id"
    BUILD_CONTEXT_HASH = "build_context_hash"
    UNVERIFIED_LOCAL = "unverified_local"


_STRICT_KINDS = frozenset(
    {ImageIdentityKind.REGISTRY_DIGEST, ImageIdentityKind.BUILD_CONTEXT_HASH}
)


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

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ImageIdentityKind):
            object.__setattr__(self, "kind", ImageIdentityKind(self.kind))
        if not self.value or not isinstance(self.value, str):
            raise ValueError(
                "ImageIdentity.value must be a non-empty string identifying "
                "the image (digest, image id, build-context hash, or tag)"
            )

    @property
    def is_strict_acceptable(self) -> bool:
        """True iff this identity variant is acceptable under ``--strict``."""
        return self.kind in _STRICT_KINDS

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind.value, "value": self.value}
        if self.note is not None:
            out["note"] = self.note
        if self.dockerfile_path is not None:
            out["dockerfile_path"] = self.dockerfile_path
        if self.context_root is not None:
            out["context_root"] = self.context_root
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageIdentity":
        return cls(
            kind=ImageIdentityKind(data["kind"]),
            value=str(data["value"]),
            note=data.get("note"),
            dockerfile_path=data.get("dockerfile_path"),
            context_root=data.get("context_root"),
        )

    # Convenience constructors --------------------------------------------------

    @classmethod
    def registry_digest(cls, digest: str) -> "ImageIdentity":
        return cls(kind=ImageIdentityKind.REGISTRY_DIGEST, value=digest)

    @classmethod
    def local_image_id(cls, image_id: str) -> "ImageIdentity":
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
        return cls(
            kind=ImageIdentityKind.BUILD_CONTEXT_HASH,
            value=context_hash,
            dockerfile_path=dockerfile_path,
            context_root=context_root,
        )

    @classmethod
    def unverified_local(cls, tag_or_id: str) -> "ImageIdentity":
        return cls(
            kind=ImageIdentityKind.UNVERIFIED_LOCAL,
            value=tag_or_id,
            note="not pinnable",
        )
