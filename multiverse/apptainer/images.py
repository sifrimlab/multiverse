"""Image-reference classification and SIF digest helpers.

Apptainer accepts several reference shapes:

* A local SIF file (``/scratch/foo.sif``) — used directly.
* An OCI reference (``docker://registry/image:tag``) — converted to a
  SIF at ``apptainer pull`` time.
* A bare Docker tag (``myimage:tag``) — treated as
  ``docker-daemon://myimage:tag``; works only when a local Docker
  daemon is reachable. Rare on HPC and we surface a clear error if it
  fails.

The dual-digest manifest invariant (STRATEGY M2) requires the SIF
digest *and* the source OCI digest. ``apptainer pull`` does not by
itself record the source digest in a parseable way, so the engine
records both when it does the pull, and verifies them at promotion
time.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ApptainerImageKind(str, enum.Enum):
    SIF_FILE = "sif_file"
    OCI_REGISTRY = "oci_registry"
    DOCKER_DAEMON = "docker_daemon"


@dataclass(frozen=True)
class ApptainerImageRef:
    """Parsed image reference.

    ``raw`` is what the caller passed; ``kind`` and ``locator`` are the
    interpretation the engine will act on.
    """

    raw: str
    kind: ApptainerImageKind
    locator: str

    @property
    def is_local_file(self) -> bool:
        return self.kind is ApptainerImageKind.SIF_FILE


def classify_image_ref(image: str) -> ApptainerImageRef:
    """Best-effort classification of an image reference string."""
    s = image.strip()
    if not s:
        raise ValueError("image reference must be a non-empty string")
    lowered = s.lower()
    if lowered.startswith(("docker://", "oras://", "library://", "shub://", "http://", "https://")):
        return ApptainerImageRef(raw=s, kind=ApptainerImageKind.OCI_REGISTRY, locator=s)
    if lowered.startswith("docker-daemon://"):
        return ApptainerImageRef(raw=s, kind=ApptainerImageKind.DOCKER_DAEMON, locator=s)
    # Local file paths win when the file exists.
    candidate = Path(s).expanduser()
    if candidate.suffix.lower() == ".sif" or candidate.exists():
        return ApptainerImageRef(
            raw=s,
            kind=ApptainerImageKind.SIF_FILE,
            locator=str(candidate.resolve()) if candidate.exists() else str(candidate),
        )
    # Bare tags (``myimage:tag``) get the docker-daemon prefix. This is
    # the path most likely to fail on HPC, which is correct — bare tags
    # are not reproducible.
    return ApptainerImageRef(
        raw=s,
        kind=ApptainerImageKind.DOCKER_DAEMON,
        locator=f"docker-daemon://{s}",
    )


def compute_sif_digest(sif_path: Path) -> str:
    """Return the sha256 hex digest of a SIF file as ``sha256:<hex>``.

    Apptainer SIFs are content-addressable: byte-identical SIF files
    produce identical digests, which is what makes them safe to record
    in the manifest.
    """
    h = hashlib.sha256()
    with sif_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sif_cache_path_for(
    cache_root: Path, oci_digest: Optional[str], image_locator: str
) -> Path:
    """Where a pulled SIF should land in the per-state SIF cache.

    Prefer naming by source OCI digest (stable, content-addressed); fall
    back to a hash of the locator string when the digest is unknown.
    """
    if oci_digest and oci_digest.startswith("sha256:"):
        key = oci_digest.split(":", 1)[1]
    else:
        key = hashlib.sha256(image_locator.encode("utf-8")).hexdigest()
    return cache_root / f"{key}.sif"
