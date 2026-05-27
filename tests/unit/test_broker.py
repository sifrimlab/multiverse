"""Milestone-13 exit-gate tests for the resource broker.

Coverage:
    1. Admission honours the live observer + reservation ledger.
    2. Pressure mode pauses admission of NEW jobs; running ones untouched.
    3. CRITICAL mode also only pauses admission — never auto-kills (R11).
    4. OOM exits surface as ``OomEvent`` with reason ``OOM_KILLED``.
    5. GPU serialization mode refuses a second GPU job on the same index
       when NVML is unavailable.
    6. Continuous ``observe`` emits transition events on mode change and
       records the resource that triggered the transition.
    7. Reservation ledger releases on container exit regardless of OOM.
"""

from __future__ import annotations

from multiverse.broker import (
    AdmissionOutcome,
    HostMetrics,
    InMemoryHostObserver,
    PressureMode,
    PressureThresholds,
    ResourceBroker,
    ResourceRequest,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _metrics(*, ram_free=10 * 1024**3, ram_total=16 * 1024**3, vram=None):
    return HostMetrics(
        ram_free_bytes=ram_free,
        ram_total_bytes=ram_total,
        vram_free_per_gpu=vram or {},
        vram_total_per_gpu={gpu: total for gpu, total in (vram or {}).items()},
    )


def _broker(metrics: HostMetrics, **kwargs) -> ResourceBroker:
    return ResourceBroker(observer=InMemoryHostObserver(metrics), **kwargs)


# ---------------------------------------------------------------------------
# 1. Admission
# ---------------------------------------------------------------------------


def test_admit_succeeds_when_enough_ram() -> None:
    broker = _broker(_metrics(ram_free=8 * 1024**3, ram_total=16 * 1024**3))
    decision = broker.admit(
        physical_attempt_id="r1",
        request=ResourceRequest(ram_bytes=4 * 1024**3),
    )
    assert decision.admitted
    assert decision.outcome is AdmissionOutcome.ADMITTED
    # Reservation now affects subsequent admissions.
    decision2 = broker.admit(
        physical_attempt_id="r2",
        request=ResourceRequest(ram_bytes=5 * 1024**3),
    )
    assert decision2.outcome is AdmissionOutcome.REJECTED_INSUFFICIENT
    assert "reserved" in (decision2.detail or "")


def test_admit_returns_specific_failure_reason() -> None:
    # 50% RAM utilisation → NORMAL mode, but the requested amount exceeds
    # what's free, so the broker reports REJECTED_INSUFFICIENT (not
    # REJECTED_PRESSURE).
    broker = _broker(_metrics(ram_free=8 * 1024**3, ram_total=16 * 1024**3))
    decision = broker.admit(
        physical_attempt_id="r",
        request=ResourceRequest(ram_bytes=12 * 1024**3),
    )
    assert decision.outcome is AdmissionOutcome.REJECTED_INSUFFICIENT
    assert "RAM" in (decision.detail or "")


def test_release_returns_capacity() -> None:
    broker = _broker(_metrics(ram_free=10 * 1024**3, ram_total=16 * 1024**3))
    broker.admit(
        physical_attempt_id="r1",
        request=ResourceRequest(ram_bytes=6 * 1024**3),
    )
    rejected = broker.admit(
        physical_attempt_id="r2",
        request=ResourceRequest(ram_bytes=6 * 1024**3),
    )
    assert rejected.outcome is AdmissionOutcome.REJECTED_INSUFFICIENT
    broker.release("r1")
    accepted = broker.admit(
        physical_attempt_id="r3",
        request=ResourceRequest(ram_bytes=6 * 1024**3),
    )
    assert accepted.admitted


# ---------------------------------------------------------------------------
# 2-3. Pressure pauses admission; never kills (R11)
# ---------------------------------------------------------------------------


def test_pressure_mode_pauses_admission_without_killing() -> None:
    observer = InMemoryHostObserver(
        # 96% utilisation → CRITICAL (threshold 95%).
        _metrics(ram_free=640 * 1024**2, ram_total=16 * 1024**3)
    )
    broker = ResourceBroker(observer=observer)
    # Prior reservation simulates a running job — must NOT be released.
    broker.ledger.reserve("running_attempt", ResourceRequest(ram_bytes=512 * 1024**2))
    assert broker.mode is PressureMode.CRITICAL

    # New admission is rejected with REJECTED_PRESSURE.
    decision = broker.admit(
        physical_attempt_id="new",
        request=ResourceRequest(ram_bytes=128 * 1024**2),
    )
    assert decision.outcome is AdmissionOutcome.REJECTED_PRESSURE
    # Running reservation untouched (R11: never auto-kill).
    assert "running_attempt" in broker.ledger.by_attempt


def test_critical_mode_does_not_imply_release() -> None:
    """Acceptance: ``critical`` does not preempt running containers in
    local mode."""
    metrics = _metrics(ram_free=200 * 1024**2, ram_total=16 * 1024**3)
    broker = _broker(metrics)
    broker.ledger.reserve("a", ResourceRequest(ram_bytes=1 * 1024**3))
    broker.ledger.reserve("b", ResourceRequest(ram_bytes=1 * 1024**3))
    obs = broker.observe()
    assert obs.mode is PressureMode.CRITICAL
    # Running reservations not auto-killed.
    assert set(broker.ledger.by_attempt) == {"a", "b"}


# ---------------------------------------------------------------------------
# 4. OOM classification
# ---------------------------------------------------------------------------


def test_oom_exit_emits_oom_event() -> None:
    broker = _broker(_metrics())
    broker.ledger.reserve("victim", ResourceRequest(ram_bytes=1024))
    ev = broker.classify_exit(
        physical_attempt_id="victim", exit_code=137, oom_killed=True
    )
    assert ev is not None
    assert ev.to_dict()["reason"] == "OOM_KILLED"
    assert "victim" not in broker.ledger.by_attempt


def test_non_oom_exit_releases_without_event() -> None:
    broker = _broker(_metrics())
    broker.ledger.reserve("ok", ResourceRequest(ram_bytes=1024))
    ev = broker.classify_exit(
        physical_attempt_id="ok", exit_code=0, oom_killed=False
    )
    assert ev is None
    assert "ok" not in broker.ledger.by_attempt


# ---------------------------------------------------------------------------
# 5. GPU serialization (NVML absent)
# ---------------------------------------------------------------------------


def test_gpu_serialization_refuses_second_admission_on_same_gpu() -> None:
    metrics = _metrics(
        ram_free=10 * 1024**3,
        ram_total=16 * 1024**3,
        vram={0: 8 * 1024**3},  # 8 GiB free on cuda:0
    )
    broker = _broker(metrics, serialize_gpu_admissions=True)
    first = broker.admit(
        physical_attempt_id="g1",
        request=ResourceRequest(ram_bytes=1, vram_bytes=2 * 1024**3, gpu_index=0),
    )
    assert first.admitted
    second = broker.admit(
        physical_attempt_id="g2",
        request=ResourceRequest(ram_bytes=1, vram_bytes=2 * 1024**3, gpu_index=0),
    )
    assert second.outcome is AdmissionOutcome.REJECTED_GPU_BUSY


def test_gpu_no_serialization_allows_concurrent_when_vram_available() -> None:
    metrics = _metrics(
        ram_free=10 * 1024**3,
        ram_total=16 * 1024**3,
        vram={0: 16 * 1024**3},
    )
    broker = _broker(metrics, serialize_gpu_admissions=False)
    a = broker.admit(
        physical_attempt_id="g1",
        request=ResourceRequest(ram_bytes=1, vram_bytes=2 * 1024**3, gpu_index=0),
    )
    b = broker.admit(
        physical_attempt_id="g2",
        request=ResourceRequest(ram_bytes=1, vram_bytes=2 * 1024**3, gpu_index=0),
    )
    assert a.admitted and b.admitted


def test_vram_request_refused_when_insufficient() -> None:
    metrics = _metrics(
        ram_free=10 * 1024**3,
        ram_total=16 * 1024**3,
        vram={0: 1 * 1024**3},
    )
    broker = _broker(metrics)
    decision = broker.admit(
        physical_attempt_id="g",
        request=ResourceRequest(ram_bytes=1, vram_bytes=8 * 1024**3, gpu_index=0),
    )
    assert decision.outcome is AdmissionOutcome.REJECTED_INSUFFICIENT


# ---------------------------------------------------------------------------
# 6. Continuous observation emits transition events
# ---------------------------------------------------------------------------


def test_observe_records_transition_on_mode_change() -> None:
    observer = InMemoryHostObserver(_metrics(ram_free=8 * 1024**3, ram_total=16 * 1024**3))
    broker = ResourceBroker(observer=observer)

    a = broker.observe()
    assert a.mode is PressureMode.NORMAL
    assert a.transitioned is False
    assert a.pressure_events == []

    # 96% utilisation → CRITICAL.
    observer.current = _metrics(ram_free=640 * 1024**2, ram_total=16 * 1024**3)
    b = broker.observe()
    assert b.mode is PressureMode.CRITICAL
    assert b.transitioned is True
    assert b.pressure_events and b.pressure_events[0].resource == "ram"

    c = broker.observe()
    assert c.transitioned is False  # already CRITICAL


# ---------------------------------------------------------------------------
# 7. Custom thresholds work
# ---------------------------------------------------------------------------


def test_custom_thresholds() -> None:
    metrics = _metrics(ram_free=4 * 1024**3, ram_total=16 * 1024**3)  # 75% util
    strict = ResourceBroker(
        observer=InMemoryHostObserver(metrics),
        thresholds=PressureThresholds(ram_pressure=0.5, ram_critical=0.7),
    )
    assert strict.mode is PressureMode.CRITICAL
    lax = ResourceBroker(
        observer=InMemoryHostObserver(metrics),
        thresholds=PressureThresholds(ram_pressure=0.9, ram_critical=0.99),
    )
    assert lax.mode is PressureMode.NORMAL
