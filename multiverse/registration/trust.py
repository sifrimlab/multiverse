"""Trust levels for model registrations (STRATEGY S19).

* ``BUILTIN`` — registered from the repo's tracked ``store/models/<slug>/``
  directory by ``make register-models``.
* ``IMPORTED`` — user-supplied model directory. GUI surfaces a banner.

The classifier is heuristic: a model is BUILTIN iff its manifest path
resolves under the same directory as the ``multiverse`` package's tracked
models subtree. Tests pin both branches.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional


class TrustLevel(str, Enum):
    BUILTIN = "builtin"
    IMPORTED = "imported"


def classify_trust(
    manifest_path: Path,
    *,
    builtin_root: Optional[Path] = None,
) -> TrustLevel:
    """Classify a model registration by manifest path.

    ``builtin_root`` is the absolute path of the repository's tracked
    models directory (``<repo>/store/models``). Manifests resolving
    under it are BUILTIN; everything else is IMPORTED.
    """
    if builtin_root is None:
        return TrustLevel.IMPORTED
    try:
        Path(manifest_path).resolve(strict=False).relative_to(
            Path(builtin_root).resolve(strict=False)
        )
        return TrustLevel.BUILTIN
    except ValueError:
        return TrustLevel.IMPORTED
