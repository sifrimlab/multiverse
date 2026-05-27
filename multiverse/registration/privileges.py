"""Detect elevated Docker flags in a model manifest (STRATEGY S19)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping


# Frozen list. Each entry is a (dotted-key, expected-value) pair; any of
# these in a manifest marks the model "elevated" and requires explicit
# user confirmation at registration.
PRIVILEGE_FLAGS = (
    ("privileged", True),
    ("network", "host"),
    ("pid", "host"),
    ("ipc", "host"),
    ("cap_add", "SYS_ADMIN"),
)


# Allowed volume mounts per the model container contract.
_ALLOWED_VOLUME_MOUNTS = frozenset({"/input", "/output", "/mvr-spec"})


@dataclass
class PrivilegeAudit:
    elevated: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "elevated": self.elevated,
            "reasons": list(self.reasons),
        }


def audit_docker_flags(manifest: Mapping[str, Any]) -> PrivilegeAudit:
    """Audit a model manifest's Docker section for privilege escalation.

    Looks under either ``manifest['docker']`` (the canonical place) or
    the top level — registration manifests in the wild use both.
    """
    audit = PrivilegeAudit()
    sections = []
    if isinstance(manifest.get("docker"), Mapping):
        sections.append(manifest["docker"])
    sections.append(manifest)

    for section in sections:
        for key, expected in PRIVILEGE_FLAGS:
            value = section.get(key)
            if value is None:
                continue
            if _matches(value, expected):
                audit.elevated = True
                audit.reasons.append(f"{key}={value!r}")

        # Volume mounts: anything outside the allow-list is elevated.
        volumes = section.get("volumes") or section.get("mounts") or []
        if isinstance(volumes, list):
            for entry in volumes:
                target = _container_target(entry)
                if target is not None and target not in _ALLOWED_VOLUME_MOUNTS:
                    audit.elevated = True
                    audit.reasons.append(
                        f"unauthorised volume target {target!r}"
                    )
    return audit


def _matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(value) is expected
    if isinstance(value, list):
        return expected in value
    return value == expected


def _container_target(entry: Any) -> str | None:
    """Extract the in-container mount target from a Docker volume spec.

    Accepted shapes:
        * ``"/host:/container[:ro]"`` (Docker --volume short syntax)
        * ``{"target": "/container", ...}`` (Docker mount long syntax)
    """
    if isinstance(entry, str):
        parts = entry.split(":")
        if len(parts) >= 2:
            return parts[1]
        return None
    if isinstance(entry, Mapping):
        target = entry.get("target") or entry.get("destination")
        return str(target) if target else None
    return None
