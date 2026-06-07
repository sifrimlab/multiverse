"""State-directory and user-identity resolution (STRATEGY M1).

The kernel, projection plugins, doctor, and CLI all need to agree on
*where the persistent state lives* and *which user owns it*. Before M1
the answer was implicitly ``<package install dir>/store/...``, which
fails on HPC (read-only install) and burns the multi-user-future bridge
by giving every co-tenant the same paths.

This module is the single source of truth for both questions. Resolution
is pure (no I/O beyond ``os.environ`` and ``getpass.getuser``) so it is
cheap to call repeatedly and safe to monkey-patch in tests.

Precedence for ``state_root`` (first match wins):

1. ``$MULTIVERSE_STATE_DIR``
2. ``state_root:`` in the multiverse config file
3. ``$XDG_STATE_HOME/multiverse``
4. ``$HOME/.multiverse``

``user_id`` defaults to ``getpass.getuser()`` and may be overridden by
``$MULTIVERSE_USER_ID``. Today the field is informational; the multi-user
future just changes where it comes from. The point of capturing it now
is to avoid retrofitting tenancy into paths and journal records later.
"""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
"""The installed multiverse package directory. Used by the doctor probe
to refuse a state_root that points inside this directory."""

REPO_ROOT_GUESS = PACKAGE_DIR.parent
"""One level up from the package directory. On editable installs this is
the repo root; on wheel installs it is site-packages. Either way, it is
*not* a sensible default for state."""

CONFIG_FILENAME = "multiverse.config.yaml"
"""Name of the per-user config file. Searched in $XDG_CONFIG_HOME/multiverse
and $HOME/.config/multiverse."""

STATE_FILENAME = "multiverse_state.db"
"""SQLite filename for the combined state DB, including detecting a legacy
install whose state lived inside the package directory."""


@dataclass(frozen=True)
class StatePaths:
    """Bundle of paths derived from a single ``state_root``."""

    state_root: Path
    user_id: str

    @property
    def db_path(self) -> Path:
        return self.state_root / STATE_FILENAME

    @property
    def store_root(self) -> Path:
        return self.state_root / "store"

    @property
    def journal_root(self) -> Path:
        return self.state_root / "journal"

    @property
    def datasets_dir(self) -> Path:
        return self.store_root / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.store_root / "models"

    @property
    def artifacts_dir(self) -> Path:
        return self.store_root / "artifacts"

    @property
    def workspaces_dir(self) -> Path:
        return self.store_root / "workspaces"


def _candidate_config_paths(env: Mapping[str, str]) -> list[Path]:
    """Locations to search for the per-user config file, in priority order."""
    xdg = env.get("XDG_CONFIG_HOME")
    home = env.get("HOME")
    out: list[Path] = []
    if xdg:
        out.append(Path(xdg) / "multiverse" / CONFIG_FILENAME)
    if home:
        out.append(Path(home) / ".config" / "multiverse" / CONFIG_FILENAME)
        out.append(Path(home) / ".multiverse" / CONFIG_FILENAME)
    return out


def find_config_file(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Return the first existing config file, or None.

    Searches per-user XDG/home locations first. Falls back to the
    repo-root ``multiverse.config.yaml`` so that a project-level config
    (with ``state_root:`` set) is honoured for direct CLI invocations from
    the checkout directory.
    """
    e = env if env is not None else os.environ
    for p in _candidate_config_paths(e):
        if p.is_file():
            return p
    # Project-level config next to the package directory.
    legacy = REPO_ROOT_GUESS / CONFIG_FILENAME
    if legacy.is_file():
        return legacy
    return None


def _read_state_root_from_config(path: Path) -> Optional[Path]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    raw = data.get("state_root")
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def resolve_state_root(env: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve the state root per the documented precedence chain.

    The returned path is absolute and ``expanduser``-d. The directory is
    *not* created here; callers that need to write into it are expected
    to ``mkdir(parents=True, exist_ok=True)``.
    """
    e = env if env is not None else os.environ

    explicit = e.get("MULTIVERSE_STATE_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()

    cfg = find_config_file(e)
    if cfg is not None:
        from_cfg = _read_state_root_from_config(cfg)
        if from_cfg is not None:
            return from_cfg.resolve()

    xdg_state = e.get("XDG_STATE_HOME")
    if xdg_state:
        return (Path(xdg_state) / "multiverse").expanduser().resolve()

    home = e.get("HOME")
    if home:
        return (Path(home) / ".multiverse").resolve()

    # Last resort: current working directory. We deliberately do NOT fall
    # back to the package directory; that's the bug M1 exists to fix.
    return (Path.cwd() / ".multiverse").resolve()


def resolve_user_id(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the current user id.

    Defaults to ``getpass.getuser()``; overridable via
    ``$MULTIVERSE_USER_ID`` so tests and HPC job scripts can pin it
    explicitly. The value should be filesystem-safe; this function does not
    validate that.
    """
    e = env if env is not None else os.environ
    override = e.get("MULTIVERSE_USER_ID")
    if override:
        return override
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        # Some HPC compute nodes have no entry in /etc/passwd for the
        # SLURM user; fall back to numeric UID.
        return f"uid-{os.getuid()}" if hasattr(os, "getuid") else "unknown"


def resolve_paths(env: Optional[Mapping[str, str]] = None) -> StatePaths:
    """Resolve the full bundle of state paths plus user id."""
    return StatePaths(
        state_root=resolve_state_root(env),
        user_id=resolve_user_id(env),
    )


def is_inside_package_dir(path: Path) -> bool:
    """True if ``path`` is the package directory or one of its descendants
    (or the repo-root guess one level up). Used by the doctor probe."""
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    for anchor in (PACKAGE_DIR, REPO_ROOT_GUESS):
        try:
            resolved.relative_to(anchor)
        except ValueError:
            continue
        return True
    return False


def legacy_db_search_locations() -> list[Path]:
    """Locations where a misconfigured install might have left its SQLite
    DB inside the package tree (the pre-M1 anti-pattern).

    Used by the in-package-dir refusal check in
    ``registry_db.get_db_connection`` and by ``multiverse
    migrate-state-dir``. The first hit is what we report to the user.
    """
    return [
        REPO_ROOT_GUESS / STATE_FILENAME,
        PACKAGE_DIR / STATE_FILENAME,
    ]


def find_legacy_db() -> Optional[Path]:
    """Return the first in-package-dir DB found, or None."""
    for p in legacy_db_search_locations():
        if p.is_file():
            return p
    return None
