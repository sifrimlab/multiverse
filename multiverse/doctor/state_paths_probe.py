"""Doctor probe: refuse a ``state_root`` that points inside the package
directory (STRATEGY M1).

The pre-M1 bug was that ``BASE_DIR = dirname(dirname(__file__))`` made
the package directory itself the state directory. That fails on
read-only HPC installs and silently aliases data across users. This
probe enforces the M1 contract: the state root must live outside the
installed package tree.

The probe is read-only and cheap; it is included in the default
``multiverse doctor`` run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..state_paths import (PACKAGE_DIR, REPO_ROOT_GUESS, find_legacy_db,
                           is_inside_package_dir)
from .health_probes import (CleanupResult, LeakInventoryResult, ProbeOutcome,
                            ProbeReport)


def probe_state_root(state_root: Path, *, explicit: bool = False) -> ProbeReport:
    """Verify the configured state root is outside the package directory
    and (when ``state_root`` was *not* explicitly chosen) that no legacy
    ``multiverse_state.db`` is being orphaned.

    Pass ``explicit=True`` when the caller picked ``state_root`` via CLI
    flag or environment variable — they have expressed intent and we do
    not double-check the legacy location.
    """
    detail_lines: list[str] = []
    outcome = ProbeOutcome.PASS

    resolved: Optional[Path]
    try:
        resolved = Path(state_root).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        resolved = None
        outcome = ProbeOutcome.FAIL
        detail_lines.append(f"could not resolve {state_root!r}: {exc}")

    if resolved is not None:
        if is_inside_package_dir(resolved):
            outcome = ProbeOutcome.FAIL
            detail_lines.append(
                f"state_root {str(resolved)!r} is inside the package directory "
                f"({str(PACKAGE_DIR)!r}). Set MULTIVERSE_STATE_DIR or "
                f"state_root: in the config file to a user-writable path "
                f"(e.g. $HOME/.multiverse)."
            )
        else:
            detail_lines.append(f"state_root={str(resolved)!r}")

    if not explicit:
        legacy = find_legacy_db()
        if legacy is not None:
            legacy_str = str(legacy)
            resolved_legacy = legacy.resolve()
            if resolved is None or resolved_legacy != resolved / "multiverse_state.db":
                outcome = ProbeOutcome.FAIL
                detail_lines.append(
                    f"legacy database found at {legacy_str!r} (would be orphaned). "
                    "Run `multiverse migrate-state-dir` to relocate, or set "
                    f"MULTIVERSE_STATE_DIR to {str(legacy.parent)!r} to keep using the legacy path."
                )

    detail_lines.append(f"package_dir={str(PACKAGE_DIR)!r}")
    detail_lines.append(f"repo_root_guess={str(REPO_ROOT_GUESS)!r}")

    return ProbeReport(
        name="state_paths.state_root_outside_package",
        probe=outcome,
        cleanup=CleanupResult.CLEAN,
        leak=LeakInventoryResult.NONE,
        leak_count=0,
        detail=" | ".join(detail_lines),
    )
