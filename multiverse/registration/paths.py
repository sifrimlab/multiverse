"""Path normalisation for registration manifests (STRATEGY S19)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .errors import PathEscapeError


def safe_under_root(
    candidate: str | os.PathLike[str] | Path,
    *,
    root: str | os.PathLike[str] | Path,
) -> Path:
    """Resolve ``candidate`` with ``realpath`` and verify it stays under
    ``root``. Raise :class:`PathEscapeError` if it does not.

    Relative paths are resolved against ``root``.
    """
    root_resolved = Path(root).resolve(strict=False)
    target = Path(candidate)
    if not target.is_absolute():
        target = root_resolved / target
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PathEscapeError(
            f"path {str(candidate)!r} escapes store root {str(root)!r} "
            f"(resolves to {resolved})"
        ) from exc
    return resolved


def validate_paths_in_mapping(
    mapping: Mapping[str, Any],
    *,
    root: str | os.PathLike[str] | Path,
    keys: tuple[str, ...] = ("path", "paths", "raw_files"),
) -> Dict[str, Path]:
    """Walk a manifest dict, verify every path-bearing leaf, and return
    a flat ``key_dotted_path -> resolved_path`` map.

    The recursion accepts strings, lists of strings, and dicts of strings.
    Anything else under a flagged key raises ``PathEscapeError``.
    """
    out: Dict[str, Path] = {}

    def _walk(node: Any, path_breadcrumb: List[str], inside_flagged: bool) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                _walk(
                    v,
                    path_breadcrumb + [str(k)],
                    inside_flagged or (str(k) in keys),
                )
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, path_breadcrumb + [str(i)], inside_flagged)
        elif isinstance(node, str):
            if inside_flagged:
                resolved = safe_under_root(node, root=root)
                out[".".join(path_breadcrumb)] = resolved

    _walk(mapping, [], False)
    return out
