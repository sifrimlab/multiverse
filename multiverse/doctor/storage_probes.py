"""Storage capability probes (STRATEGY S11 / R8 / ADR §2).

Each probe returns one of four levels:

* ``supported`` — the capability behaves as expected.
* ``degraded`` — the capability works but with caveats; specific
  guarantees are disabled and recorded in the artifact manifest.
* ``dangerous`` — the capability is suspect (cloud-sync, network FS);
  daemon refuses to start unless ``--accept-degraded`` is passed.
* ``blocked`` — required capability is missing; daemon cannot start.

Probes never touch ``store/artifacts/``, ``store/workspaces/``, or
``store/journal/`` directly: they write into ``store/_probe/`` and clean
up after themselves.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from ..artifact.checksums import atomic_write_bytes


class StorageLevel(str, Enum):
    SUPPORTED = "supported"
    DEGRADED = "degraded"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


# Re-exported as module-level constants for ergonomics.
SUPPORTED = StorageLevel.SUPPORTED
DEGRADED = StorageLevel.DEGRADED
DANGEROUS = StorageLevel.DANGEROUS
BLOCKED = StorageLevel.BLOCKED


_PROBE_SUBDIR = "_probe"

# Cloud-sync markers heuristically detected per R8.
_CLOUD_SYNC_MARKERS = (
    ".dropbox",
    ".dropbox.cache",
    ".tmp.driveupload",
    ".onedrive",
    "icloud",
)


class CloudSyncMarkerError(RuntimeError):
    """Raised by ``cloud_sync_heuristic`` to make the marker explicit."""


@dataclass
class StorageProbeResult:
    name: str
    level: StorageLevel
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        out = {"name": self.name, "level": self.level.value}
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass
class StorageReport:
    root: Path
    results: List[StorageProbeResult] = field(default_factory=list)

    def by_name(self) -> Dict[str, StorageProbeResult]:
        return {r.name: r for r in self.results}

    @property
    def worst_level(self) -> StorageLevel:
        order = [SUPPORTED, DEGRADED, DANGEROUS, BLOCKED]
        seen = SUPPORTED
        for r in self.results:
            if order.index(r.level) > order.index(seen):
                seen = r.level
        return seen

    @property
    def degraded_capabilities(self) -> List[str]:
        return [r.name for r in self.results if r.level is DEGRADED]

    def can_start(self, *, accept_degraded: bool = False) -> bool:
        worst = self.worst_level
        if worst is BLOCKED:
            return False
        if worst is DANGEROUS:
            return accept_degraded
        return True


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@dataclass
class StorageProbe:
    name: str

    def run(self, root: Path) -> StorageProbeResult:  # pragma: no cover — interface
        raise NotImplementedError


def _probe_dir(root: Path) -> Path:
    p = root / _PROBE_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def probe_write_then_read(root: Path) -> StorageProbeResult:
    try:
        probe_dir = _probe_dir(root)
        probe = probe_dir / "rw"
        atomic_write_bytes(probe, b"mvd-probe-payload")
        observed = probe.read_bytes()
        probe.unlink()
        if observed == b"mvd-probe-payload":
            return StorageProbeResult("write_then_read", SUPPORTED)
        return StorageProbeResult(
            "write_then_read",
            BLOCKED,
            detail=f"readback mismatch: {observed!r}",
        )
    except (OSError, PermissionError) as exc:
        return StorageProbeResult(
            "write_then_read", BLOCKED, detail=f"{type(exc).__name__}: {exc}"
        )


def probe_atomic_rename(root: Path) -> StorageProbeResult:
    base = _probe_dir(root)
    src = base / "rename.src"
    dst = base / "rename.dst"
    try:
        src.write_bytes(b"x")
        if dst.exists():
            dst.unlink()
        os.replace(str(src), str(dst))
        if not dst.is_file() or src.exists():
            return StorageProbeResult(
                "atomic_rename", BLOCKED, detail="rename did not move bytes"
            )
        dst.unlink()
        return StorageProbeResult("atomic_rename", SUPPORTED)
    except OSError as exc:
        return StorageProbeResult(
            "atomic_rename", BLOCKED, detail=f"{type(exc).__name__}: {exc}"
        )


def probe_fsync_file(root: Path) -> StorageProbeResult:
    base = _probe_dir(root)
    target = base / "fsync_file"
    try:
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, b"hi")
            os.fsync(fd)
        finally:
            os.close(fd)
        target.unlink()
        return StorageProbeResult("fsync_file", SUPPORTED)
    except OSError as exc:
        return StorageProbeResult(
            "fsync_file", DANGEROUS, detail=f"{type(exc).__name__}: {exc}"
        )


def probe_fsync_dir(root: Path) -> StorageProbeResult:
    base = _probe_dir(root)
    try:
        fd = os.open(str(base), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return StorageProbeResult("fsync_dir", SUPPORTED)
    except OSError as exc:
        return StorageProbeResult(
            "fsync_dir",
            DEGRADED,
            detail=f"{type(exc).__name__}: {exc} (durability guarantee reduced)",
        )


def probe_case_sensitivity(root: Path) -> StorageProbeResult:
    base = _probe_dir(root)
    a = base / "Case.txt"
    b = base / "case.txt"
    try:
        a.write_bytes(b"a")
        b.write_bytes(b"b")
        a_bytes = a.read_bytes()
        b_bytes = b.read_bytes()
        a.unlink()
        if b.exists():
            b.unlink()
        if a_bytes != b_bytes:
            return StorageProbeResult("case_sensitivity", SUPPORTED)
        return StorageProbeResult(
            "case_sensitivity",
            DEGRADED,
            detail="case-insensitive FS: rename collisions possible",
        )
    except OSError as exc:
        return StorageProbeResult(
            "case_sensitivity", DEGRADED, detail=f"{type(exc).__name__}: {exc}"
        )


def probe_cloud_sync_heuristic(root: Path) -> StorageProbeResult:
    path = root.resolve()
    found = []
    for ancestor in [path, *path.parents]:
        for marker in _CLOUD_SYNC_MARKERS:
            if (ancestor / marker).exists():
                found.append(f"{ancestor}/{marker}")
    if found:
        return StorageProbeResult(
            "cloud_sync_heuristic",
            DANGEROUS,
            detail=f"cloud-sync markers detected: {found[:3]}",
        )
    # Also check ancestor names — Dropbox/OneDrive often appear in the path.
    lowered = str(path).lower()
    name_markers = ["dropbox", "onedrive", "google drive", "googledrive"]
    matched = [m for m in name_markers if m in lowered]
    if matched:
        return StorageProbeResult(
            "cloud_sync_heuristic",
            DANGEROUS,
            detail=f"path-name suggests cloud-sync: {matched}",
        )
    return StorageProbeResult("cloud_sync_heuristic", SUPPORTED)


def probe_free_space(
    root: Path, *, min_supported_gb: float = 5.0, min_degraded_gb: float = 1.0
) -> StorageProbeResult:
    try:
        usage = shutil.disk_usage(root)
    except OSError as exc:
        return StorageProbeResult(
            "free_space_reservation",
            DANGEROUS,
            detail=f"disk_usage failed: {exc}",
        )
    free_gb = usage.free / (1024**3)
    if free_gb >= min_supported_gb:
        return StorageProbeResult(
            "free_space_reservation",
            SUPPORTED,
            detail=f"{free_gb:.2f} GiB free",
        )
    if free_gb >= min_degraded_gb:
        return StorageProbeResult(
            "free_space_reservation",
            DEGRADED,
            detail=f"only {free_gb:.2f} GiB free (degraded headroom)",
        )
    return StorageProbeResult(
        "free_space_reservation",
        DANGEROUS,
        detail=f"only {free_gb:.2f} GiB free (<{min_degraded_gb} GiB threshold)",
    )


_PROBE_FUNCS = (
    probe_write_then_read,
    probe_atomic_rename,
    probe_fsync_file,
    probe_fsync_dir,
    probe_case_sensitivity,
    probe_cloud_sync_heuristic,
    probe_free_space,
)


def run_storage_probes(root: Path) -> StorageReport:
    """Run the standard probe matrix against ``root``.

    Always runs every probe — a ``BLOCKED`` early on does not stop the rest;
    the caller decides what to do with the report.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report = StorageReport(root=root)
    for probe_fn in _PROBE_FUNCS:
        try:
            report.results.append(probe_fn(root))
        except Exception as exc:  # pragma: no cover — defensive
            report.results.append(
                StorageProbeResult(
                    probe_fn.__name__.removeprefix("probe_"),
                    DANGEROUS,
                    detail=f"probe raised: {type(exc).__name__}: {exc}",
                )
            )
    # Clean the probe subdir so it doesn't accumulate.
    probe_dir = root / _PROBE_SUBDIR
    if probe_dir.is_dir():
        try:
            for child in probe_dir.iterdir():
                if child.is_file():
                    child.unlink()
            probe_dir.rmdir()
        except OSError:
            pass
    return report
