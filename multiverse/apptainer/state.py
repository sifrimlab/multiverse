"""Sidecar state file for the Apptainer engine.

Apptainer has no daemon and no first-class label store. To honor the
``ContainerEngine.list_by_labels`` contract, the engine maintains a JSON
sidecar at ``<state_dir>/apptainer-engine.json`` mapping a synthetic
``container_id`` to {pid, labels, image, started_at, exit_code, ...}.

The sidecar is single-writer by contract (kernel + supervisor are
single-threaded asyncio); we do not take a file lock.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional


SCHEMA_VERSION = 1
SIDECAR_FILENAME = "apptainer-engine.json"


@dataclass
class ApptainerContainerRecord:
    """One container's state in the sidecar."""

    container_id: str
    pid: Optional[int]
    image: str
    labels: Dict[str, str]
    command: list[str]
    name: Optional[str]
    started_at: float
    log_file: Optional[str] = None
    exit_code: Optional[int] = None
    oom_killed: bool = False
    finished_at: Optional[float] = None
    removed: bool = False
    sif_digest: Optional[str] = None
    mem_limit: Optional[str] = None
    """The ``--memory`` value passed at launch time, used by the OOM
    heuristic in the engine: exit_code==137 + mem_limit set → oom_killed."""
    """Recorded when the engine pulled or accepted a SIF, used by the
    dual-digest manifest."""
    source_oci_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "container_id": self.container_id,
            "pid": self.pid,
            "image": self.image,
            "labels": dict(self.labels),
            "command": list(self.command),
            "name": self.name,
            "started_at": self.started_at,
            "log_file": self.log_file,
            "exit_code": self.exit_code,
            "oom_killed": self.oom_killed,
            "finished_at": self.finished_at,
            "removed": self.removed,
            "sif_digest": self.sif_digest,
            "source_oci_digest": self.source_oci_digest,
            "mem_limit": self.mem_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ApptainerContainerRecord":
        return cls(
            container_id=str(data["container_id"]),
            pid=(int(data["pid"]) if data.get("pid") is not None else None),
            image=str(data["image"]),
            labels=dict(data.get("labels") or {}),
            command=list(data.get("command") or []),
            name=(str(data["name"]) if data.get("name") is not None else None),
            started_at=float(data["started_at"]),
            log_file=(str(data["log_file"]) if data.get("log_file") is not None else None),
            exit_code=(int(data["exit_code"]) if data.get("exit_code") is not None else None),
            oom_killed=bool(data.get("oom_killed", False)),
            finished_at=(
                float(data["finished_at"]) if data.get("finished_at") is not None else None
            ),
            removed=bool(data.get("removed", False)),
            sif_digest=(str(data["sif_digest"]) if data.get("sif_digest") is not None else None),
            source_oci_digest=(
                str(data["source_oci_digest"])
                if data.get("source_oci_digest") is not None
                else None
            ),
            mem_limit=(str(data["mem_limit"]) if data.get("mem_limit") is not None else None),
        )


@dataclass
class ApptainerSidecar:
    """Persistent dict-of-containers for the Apptainer engine."""

    path: Path
    containers: Dict[str, ApptainerContainerRecord] = field(default_factory=dict)

    @classmethod
    def load_or_empty(cls, path: Path) -> "ApptainerSidecar":
        if not path.is_file():
            return cls(path=path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(path=path)
        if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
            return cls(path=path)
        containers = {
            cid: ApptainerContainerRecord.from_dict(rec)
            for cid, rec in (data.get("containers") or {}).items()
        }
        return cls(path=path, containers=containers)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA_VERSION,
            "containers": {
                cid: rec.to_dict() for cid, rec in self.containers.items()
            },
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def put(self, record: ApptainerContainerRecord) -> None:
        self.containers[record.container_id] = record
        self.save()

    def get(self, container_id: str) -> Optional[ApptainerContainerRecord]:
        return self.containers.get(container_id)

    def matching_labels(
        self, labels: Dict[str, str]
    ) -> Iterator[ApptainerContainerRecord]:
        for rec in self.containers.values():
            if rec.removed:
                continue
            if all(rec.labels.get(k) == v for k, v in labels.items()):
                yield rec
