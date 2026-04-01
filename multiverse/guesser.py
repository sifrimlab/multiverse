"""Lightweight dataset heuristics: filename cues + shallow HDF5 metadata (no matrix I/O)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

import h5py

from .logging_utils import get_logger

logger = get_logger(__name__)

RAW_LIKE_EXTENSIONS = (".h5ad", ".h5mu")

# Filename tokens (lexical guesser). Order: modality-like tags first, then role tags.
_FILENAME_TAG_SPECS: Tuple[Tuple[Pattern[str], str], ...] = (
    (re.compile(r"(?i)rna"), "rna"),
    (re.compile(r"(?i)atac"), "atac"),
    (re.compile(r"(?i)adt"), "adt"),
    (re.compile(r"(?i)processed"), "processed"),
    # Substring "raw" (e.g. ``*_rna_raw.h5ad``); avoids missing ``raw`` when glued to underscores.
    (re.compile(r"(?i)raw"), "raw"),
)

# obs column matchers: first matching pattern wins its first key (alphabetically among ties).
# Patterns are tried in order; only columns listed in `keys` are considered.
_BATCH_KEY_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"(?i)donor"),
    re.compile(r"(?i)batch"),
    re.compile(r"(?i)sample"),
    re.compile(r"(?i)library"),
    re.compile(r"(?i)patient"),
    re.compile(r"(?i)replicate"),
    re.compile(r"(?i)\blane\b"),
    re.compile(r"(?i)gem[_-]?group"),
)

_CELL_TYPE_KEY_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(r"(?i)cell[_-]?type"),
    re.compile(r"(?i)ontology"),
    re.compile(r"(?i)annotation"),
    re.compile(r"(?i)\bcluster\b"),
    re.compile(r"(?i)\blabel\b"),
    re.compile(r"(?i)celltype"),
    re.compile(r"(?i)leiden"),
    re.compile(r"(?i)louvain"),
    re.compile(r"(?i)predicted"),
)

_OBS_KEYS_SKIP_MATCHING = frozenset({"_index", "index"})

_MODALITY_TAG_ORDER = ("rna", "atac", "adt")
_FALLBACK_MODALITY_STEM_RULES: Tuple[Tuple[str, str], ...] = (
    ("scrna", "rna"),
    ("gex", "rna"),
    ("gene", "rna"),
    ("rna", "rna"),
    ("atac", "atac"),
    ("peak", "atac"),
    ("adt", "adt"),
    ("protein", "adt"),
    ("cite", "adt"),
    ("hashing", "adt"),
)


def _filter_obs_keys_for_matching(raw_keys: Sequence[str]) -> List[str]:
    """Exclude index-like keys from metadata inference (still listed in ``obs_columns``)."""
    return [k for k in raw_keys if k not in _OBS_KEYS_SKIP_MATCHING]


def _pick_first_pattern_match(
    keys: Sequence[str],
    patterns: Tuple[Pattern[str], ...],
) -> Tuple[Optional[str], List[str]]:
    """First pattern (in order) that matches any key wins; primary is lexicographically first."""
    for pat in patterns:
        matches = sorted(k for k in keys if pat.search(k))
        if matches:
            primary = matches[0]
            alternatives = matches[1:]
            return primary, alternatives
    return None, []


def _modality_from_tags(tags: Sequence[str]) -> Optional[str]:
    """Pick rna/atac/adt from filename tags (spec order), ignoring processed/raw."""
    for m in _MODALITY_TAG_ORDER:
        if m in tags:
            return m
    return None


def _collect_all_pattern_matches(keys: Sequence[str], patterns: Tuple[Pattern[str], ...]) -> List[str]:
    """All keys matching any pattern, for logging cross-pattern alternatives."""
    seen = set()
    out: List[str] = []
    for pat in patterns:
        for k in keys:
            if pat.search(k) and k not in seen:
                seen.add(k)
                out.append(k)
    return sorted(out)


def _guess_modality_from_stem(stem: str) -> Optional[str]:
    lower = stem.lower()
    for needle, modality in _FALLBACK_MODALITY_STEM_RULES:
        if needle in lower:
            return modality
    return None


class DatasetHeuristics:
    """Infer modalities from filenames and ``obs`` column names via shallow HDF5 reads."""

    _BATCH_KEY_PATTERNS = _BATCH_KEY_PATTERNS
    _CELL_TYPE_KEY_PATTERNS = _CELL_TYPE_KEY_PATTERNS

    def _guess_from_filenames(self, directory: Path) -> Dict[str, Any]:
        """Classify files in ``directory`` using regex on names (rna, atac, adt, processed, raw)."""
        directory = Path(directory)
        per_file: List[Dict[str, Any]] = []
        if not directory.is_dir():
            return {"directory": str(directory), "files": [], "error": "not a directory"}

        for p in sorted(directory.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in RAW_LIKE_EXTENSIONS:
                continue
            tags: List[str] = []
            for pattern, tag in _FILENAME_TAG_SPECS:
                if pattern.search(p.name):
                    tags.append(tag)
            per_file.append({"path": p.name, "tags": tags})

        return {"directory": str(directory.resolve()), "files": per_file}

    def _shallow_peek_h5(self, filepath: Path) -> Dict[str, Any]:
        """Read only the ``obs`` group key names from an ``.h5ad`` / ``.h5mu`` file.

        Does not touch ``X``, ``layers``, ``raw``, ``obsm``, or modality matrices.
        """
        filepath = Path(filepath)
        if filepath.suffix.lower() not in RAW_LIKE_EXTENSIONS:
            raise ValueError(f"Expected .h5ad or .h5mu, got {filepath}")

        with h5py.File(filepath, "r") as f:
            if "obs" not in f:
                logger.warning("No 'obs' group in %s; cannot infer metadata keys.", filepath)
                return {
                    "filepath": str(filepath),
                    "obs_columns": [],
                    "batch_key": None,
                    "cell_type_key": None,
                    "batch_key_alternatives": [],
                    "cell_type_key_alternatives": [],
                }

            obs = f["obs"]
            raw_keys = [str(k) for k in obs.keys()]

        match_keys = _filter_obs_keys_for_matching(raw_keys)

        batch_key, batch_alts_same_pattern = _pick_first_pattern_match(
            match_keys, self._BATCH_KEY_PATTERNS
        )
        cell_type_key, cell_alts_same_pattern = _pick_first_pattern_match(
            match_keys, self._CELL_TYPE_KEY_PATTERNS
        )

        all_batchish = _collect_all_pattern_matches(match_keys, self._BATCH_KEY_PATTERNS)
        all_cellish = _collect_all_pattern_matches(match_keys, self._CELL_TYPE_KEY_PATTERNS)

        batch_alternatives = [k for k in all_batchish if k != batch_key]
        cell_type_alternatives = [k for k in all_cellish if k != cell_type_key]

        if batch_alternatives:
            logger.info(
                "batch_key=%r (alternatives: %s)",
                batch_key,
                batch_alternatives,
            )
        if cell_type_alternatives:
            logger.info(
                "cell_type_key=%r (alternatives: %s)",
                cell_type_key,
                cell_type_alternatives,
            )
        if batch_alts_same_pattern:
            logger.info("Other batch-like keys matching the same pattern tier: %s", batch_alts_same_pattern)
        if cell_alts_same_pattern:
            logger.info(
                "Other cell-type-like keys matching the same pattern tier: %s",
                cell_alts_same_pattern,
            )

        return {
            "filepath": str(filepath),
            "obs_columns": raw_keys,
            "batch_key": batch_key,
            "cell_type_key": cell_type_key,
            "batch_key_alternatives": batch_alternatives,
            "cell_type_key_alternatives": cell_type_alternatives,
        }

    def _pick_peek_target(self, directory: Path, lexical: Dict[str, Any]) -> Optional[Path]:
        """Prefer an RNA-tagged ``.h5ad``, else first ``.h5ad``, else first raw-like file."""
        directory = Path(directory)
        files = lexical.get("files") or []
        paths = [directory / e["path"] for e in files]
        if not paths:
            return None

        def has_tag(p: Path, tag: str) -> bool:
            for entry in files:
                if entry["path"] == p.name:
                    return tag in entry.get("tags", [])
            return False

        for p in paths:
            if p.suffix.lower() == ".h5ad" and has_tag(p, "rna"):
                return p
        for p in paths:
            if p.suffix.lower() == ".h5ad":
                return p
        return paths[0]

    def generate_manifest(self, directory: Path) -> Dict[str, Any]:
        """Combine filename heuristics and shallow ``obs`` peek into a YAML-ready manifest."""
        directory = Path(directory)
        lexical = self._guess_from_filenames(directory)
        peek_path = self._pick_peek_target(directory, lexical)
        peek: Dict[str, Any] = {}
        if peek_path is not None and peek_path.is_file():
            peek = self._shallow_peek_h5(peek_path)
        else:
            peek = {
                "filepath": None,
                "obs_columns": [],
                "batch_key": None,
                "cell_type_key": None,
                "batch_key_alternatives": [],
                "cell_type_key_alternatives": [],
            }

        name = directory.name or directory.resolve().name
        omics_tags = frozenset(_MODALITY_TAG_ORDER)

        raw_files: Dict[str, str] = {}
        for entry in lexical.get("files", []):
            fname = entry["path"]
            p = directory / fname
            if not p.is_file():
                continue
            if p.suffix.lower() not in RAW_LIKE_EXTENSIONS:
                continue
            tags = [t for t in entry.get("tags", []) if t in omics_tags]
            mod = _modality_from_tags(tags)
            if mod is None:
                mod = _guess_modality_from_stem(p.stem) or "rna"
            key = mod
            n = 2
            while key in raw_files:
                key = f"{mod}_{n}"
                n += 1
            raw_files[key] = f"data/{fname}"

        omics = sorted(raw_files.keys())

        metadata_keys: Dict[str, str] = {}
        if peek.get("batch_key"):
            metadata_keys["batch"] = peek["batch_key"]
        if peek.get("cell_type_key"):
            metadata_keys["cell_type"] = peek["cell_type_key"]
        if not metadata_keys:
            metadata_keys = {"batch": "batch", "cell_type": "cell_type"}

        return {
            "name": name,
            "omics": omics,
            "raw_files": dict(sorted(raw_files.items())),
            "metadata_keys": metadata_keys,
            "guesser_notes": {
                "filename_scan": lexical,
                "peek": peek,
            },
        }
