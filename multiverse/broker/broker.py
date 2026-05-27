"""ResourceBroker (STRATEGY S9 / R11).

The broker is single-threaded by contract (matches the kernel's asyncio
model). Tests instantiate one with an :class:`InMemoryHostObserver` and
manipulate ``observer.current`` to model pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from .observer import HostMetrics, HostObserver, ResourceRequest
from .pressure import PressureMode, PressureThresholds


class AdmissionOutcome(str, Enum):
    ADMITTED = "admitted"
    REJECTED_INSUFFICIENT = "rejected_insufficient"
    REJECTED_PRESSURE = "rejected_pressure"
    REJECTED_GPU_BUSY = "rejected_gpu_busy"


@dataclass
class AdmissionDecision:
    outcome: AdmissionOutcome
    physical_attempt_id: str
    detail: Optional[str] = None
    metrics_at_decision: Optional[HostMetrics] = None

    @property
    def admitted(self) -> bool:
        return self.outcome is AdmissionOutcome.ADMITTED


@dataclass
class ContinuousObservation:
    """Result of one ``observe()`` tick."""

    metrics: HostMetrics
    mode: PressureMode
    transitioned: bool = False
    pressure_events: List["PressureEvent"] = field(default_factory=list)


@dataclass
class PressureEvent:
    at_iso: str
    level: PressureMode
    resource: str
    utilization: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "at": self.at_iso,
            "level": self.level.value,
            "resource": self.resource,
            "utilization": float(self.utilization),
        }


@dataclass
class OomEvent:
    physical_attempt_id: str
    at_iso: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "physical_attempt_id": self.physical_attempt_id,
            "at": self.at_iso,
            "reason": "OOM_KILLED",
        }


@dataclass
class ReservationLedger:
    """Tracks admitted-but-not-yet-completed requests so the broker can
    subtract them from the live observer before answering the next
    admission."""

    by_attempt: Dict[str, ResourceRequest] = field(default_factory=dict)

    def reserve(self, attempt_id: str, request: ResourceRequest) -> None:
        self.by_attempt[attempt_id] = request

    def release(self, attempt_id: str) -> None:
        self.by_attempt.pop(attempt_id, None)

    def total_ram(self) -> int:
        return sum(r.ram_bytes for r in self.by_attempt.values())

    def total_vram_for(self, gpu_index: Optional[int]) -> int:
        if gpu_index is None:
            return 0
        return sum(
            r.vram_bytes
            for r in self.by_attempt.values()
            if r.gpu_index == gpu_index
        )

    def gpu_indices_in_use(self) -> set[int]:
        return {
            r.gpu_index
            for r in self.by_attempt.values()
            if r.gpu_index is not None
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


@dataclass
class ResourceBroker:
    """Admission + pressure observer.

    ``mode`` is a derived property; ``observe()`` recomputes it from the
    latest metrics. Transition events are emitted whenever the derived
    mode differs from the previous one — the kernel records these in the
    journal and the artifact manifest's ``resource_observations`` block.

    ``serialize_gpu_admissions`` controls R11's NVML-degraded mode: when
    NVML is absent, GPU jobs serialize one-at-a-time per GPU. Tests set
    this to True/False explicitly.
    """

    observer: HostObserver
    thresholds: PressureThresholds = field(default_factory=PressureThresholds)
    ledger: ReservationLedger = field(default_factory=ReservationLedger)
    serialize_gpu_admissions: bool = False

    _last_mode: PressureMode = PressureMode.NORMAL

    # ---- admission ----

    def admit(
        self,
        *,
        physical_attempt_id: str,
        request: ResourceRequest,
    ) -> AdmissionDecision:
        metrics = self.observer.observe()
        mode = self._classify(metrics)
        # In pressure / critical we refuse new admissions but never
        # preempt running ones.
        if mode is not PressureMode.NORMAL:
            return AdmissionDecision(
                outcome=AdmissionOutcome.REJECTED_PRESSURE,
                physical_attempt_id=physical_attempt_id,
                detail=f"pressure mode {mode.value}; admission paused",
                metrics_at_decision=metrics,
            )

        # GPU serialization mode: if NVML is unavailable, refuse a second
        # GPU job on the same index until the first releases its lease.
        if (
            self.serialize_gpu_admissions
            and request.gpu_index is not None
            and request.gpu_index in self.ledger.gpu_indices_in_use()
        ):
            return AdmissionDecision(
                outcome=AdmissionOutcome.REJECTED_GPU_BUSY,
                physical_attempt_id=physical_attempt_id,
                detail=f"GPU cuda:{request.gpu_index} already reserved (serialized mode)",
                metrics_at_decision=metrics,
            )

        # Compute *effective* free = live free - already-reserved.
        ram_effective = metrics.ram_free_bytes - self.ledger.total_ram()
        if ram_effective < request.ram_bytes:
            return AdmissionDecision(
                outcome=AdmissionOutcome.REJECTED_INSUFFICIENT,
                physical_attempt_id=physical_attempt_id,
                detail=(
                    f"need {request.ram_bytes} bytes RAM; "
                    f"effective free {ram_effective} (live {metrics.ram_free_bytes}, "
                    f"reserved {self.ledger.total_ram()})"
                ),
                metrics_at_decision=metrics,
            )

        if request.vram_bytes > 0 and request.gpu_index is not None:
            free = metrics.vram_free_per_gpu.get(request.gpu_index, 0)
            reserved = self.ledger.total_vram_for(request.gpu_index)
            if free - reserved < request.vram_bytes:
                return AdmissionDecision(
                    outcome=AdmissionOutcome.REJECTED_INSUFFICIENT,
                    physical_attempt_id=physical_attempt_id,
                    detail=(
                        f"need {request.vram_bytes} bytes VRAM on cuda:{request.gpu_index}; "
                        f"effective free {free - reserved}"
                    ),
                    metrics_at_decision=metrics,
                )

        for path, need in request.disk_bytes_per_path.items():
            free = metrics.disk_free_bytes_per_path.get(path, 0)
            if free < need:
                return AdmissionDecision(
                    outcome=AdmissionOutcome.REJECTED_INSUFFICIENT,
                    physical_attempt_id=physical_attempt_id,
                    detail=f"need {need} bytes free on {path!r}; have {free}",
                    metrics_at_decision=metrics,
                )

        self.ledger.reserve(physical_attempt_id, request)
        return AdmissionDecision(
            outcome=AdmissionOutcome.ADMITTED,
            physical_attempt_id=physical_attempt_id,
            metrics_at_decision=metrics,
        )

    def release(self, physical_attempt_id: str) -> None:
        self.ledger.release(physical_attempt_id)

    # ---- continuous observation ----

    def observe(self) -> ContinuousObservation:
        metrics = self.observer.observe()
        mode = self._classify(metrics)
        events: List[PressureEvent] = []
        if mode is not PressureMode.NORMAL:
            ram_util = (
                1.0 - (metrics.ram_free_bytes / metrics.ram_total_bytes)
                if metrics.ram_total_bytes
                else 0.0
            )
            events.append(
                PressureEvent(
                    at_iso=_now_iso(),
                    level=mode,
                    resource="ram",
                    utilization=ram_util,
                )
            )
        transitioned = mode is not self._last_mode
        self._last_mode = mode
        return ContinuousObservation(
            metrics=metrics, mode=mode, transitioned=transitioned, pressure_events=events
        )

    # ---- Docker events ----

    def classify_exit(
        self,
        *,
        physical_attempt_id: str,
        exit_code: Optional[int],
        oom_killed: bool,
    ) -> Optional[OomEvent]:
        """Translate a container-exit observation into a broker event.

        Returns an ``OomEvent`` iff the container was OOM-killed. The
        broker also releases the reservation regardless of how the
        container exited.
        """
        self.release(physical_attempt_id)
        if oom_killed:
            return OomEvent(physical_attempt_id=physical_attempt_id, at_iso=_now_iso())
        return None

    # ---- current state ----

    @property
    def mode(self) -> PressureMode:
        return self._classify(self.observer.observe())

    # ---- internals ----

    def _classify(self, metrics: HostMetrics) -> PressureMode:
        if metrics.ram_total_bytes:
            ram_util = 1.0 - (metrics.ram_free_bytes / metrics.ram_total_bytes)
            if ram_util >= self.thresholds.ram_critical:
                return PressureMode.CRITICAL
            if ram_util >= self.thresholds.ram_pressure:
                return PressureMode.PRESSURE
        # VRAM check: any GPU above the critical threshold dominates.
        for idx, total in metrics.vram_total_per_gpu.items():
            if total <= 0:
                continue
            free = metrics.vram_free_per_gpu.get(idx, 0)
            util = 1.0 - (free / total)
            if util >= self.thresholds.vram_critical:
                return PressureMode.CRITICAL
            if util >= self.thresholds.vram_pressure:
                return PressureMode.PRESSURE
        return PressureMode.NORMAL
