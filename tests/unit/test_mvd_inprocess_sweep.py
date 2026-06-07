"""Sweep routing + live registry in the GUI's in-process mvd controller.

The controller must route ``mode: sweep`` jobs to the background Optuna driver
(not submit them as plain single runs), return them as empty-attempt
placeholders so the cohort zip stays aligned, and expose a thread-safe registry
of trial attempts the GUI monitor can poll while a study runs.
"""

import threading

from multiverse.runner.mvd_inprocess import InProcessMvdController, SubmittedRun


def _bare_controller():
    """An InProcessMvdController without its background loop/kernel started."""
    ctrl = InProcessMvdController.__new__(InProcessMvdController)
    ctrl._sweep_lock = threading.Lock()
    ctrl._sweep_submissions = []
    ctrl._active_sweeps = 0
    return ctrl


def _sub(name):
    return SubmittedRun(
        attempt_id=f"att-{name}", job_name=name, dataset="ds", model="m",
        logical_run_id="",
    )


def test_submit_manifest_routes_sweeps_to_background_and_keeps_order(monkeypatch):
    ctrl = _bare_controller()
    # Pretend a stale sweep entry exists; a new launch must clear it.
    ctrl._sweep_submissions = [{"attempt_id": "stale"}]

    run_jobs_seen = {}
    sweep_jobs_seen = {}

    def fake_submit_manifest(**kwargs):
        run_jobs_seen["jobs"] = kwargs["pending_jobs"]
        return "RUN_COROUTINE"

    def fake_call(arg):
        assert arg == "RUN_COROUTINE"  # only run-job submission flows through _call
        return [_sub("A"), _sub("C")]

    def fake_start_sweeps(sweep_jobs, **kwargs):
        sweep_jobs_seen["jobs"] = sweep_jobs

    monkeypatch.setattr(ctrl, "_submit_manifest", fake_submit_manifest)
    monkeypatch.setattr(ctrl, "_call", fake_call)
    monkeypatch.setattr(ctrl, "_start_sweeps", fake_start_sweeps)

    pending = [
        {"model_name": "A", "mode": "run"},
        {"model_name": "B", "mode": "sweep"},
        {"model_name": "C"},  # no mode → run
        {"model_name": "D", "mode": "run", "_skipped": True},
    ]

    out = ctrl.submit_manifest(
        manifest_path="m.yaml", pending_jobs=pending, manifest_text="", seed=7
    )

    assert [j["model_name"] for j in sweep_jobs_seen["jobs"]] == ["B"]
    assert [j["model_name"] for j in run_jobs_seen["jobs"]] == ["A", "C"]
    # Original runnable order preserved; the sweep slot is an empty placeholder.
    assert [s.attempt_id for s in out] == ["att-A", "", "att-C"]
    # Stale registry was reset for the new launch.
    assert ctrl._sweep_submissions == []


def test_submit_manifest_all_run_jobs_does_not_start_sweeps(monkeypatch):
    ctrl = _bare_controller()
    monkeypatch.setattr(ctrl, "_submit_manifest", lambda **k: "CORO")
    monkeypatch.setattr(ctrl, "_call", lambda arg: [_sub("A")])

    def boom(*a, **k):
        raise AssertionError("_start_sweeps must not be called with no sweep jobs")

    monkeypatch.setattr(ctrl, "_start_sweeps", boom)

    out = ctrl.submit_manifest(
        manifest_path="m.yaml",
        pending_jobs=[{"model_name": "A", "mode": "run"}],
        manifest_text="",
        seed=None,
    )
    assert [s.attempt_id for s in out] == ["att-A"]


def test_submit_manifest_all_sweeps_skips_run_submit(monkeypatch):
    ctrl = _bare_controller()

    def boom(**k):
        raise AssertionError("_submit_manifest must not run with no run jobs")

    started = {}
    monkeypatch.setattr(ctrl, "_submit_manifest", boom)
    monkeypatch.setattr(ctrl, "_call", lambda arg: [])
    monkeypatch.setattr(
        ctrl, "_start_sweeps", lambda sweep_jobs, **k: started.update(n=len(sweep_jobs))
    )

    out = ctrl.submit_manifest(
        manifest_path="m.yaml",
        pending_jobs=[{"model_name": "B", "mode": "sweep"}],
        manifest_text="",
        seed=None,
    )
    # One placeholder for the sweep, with no attempt id yet.
    assert [s.attempt_id for s in out] == [""]
    assert started == {"n": 1}


def test_sweep_submissions_and_active_flag_are_threadsafe_snapshots():
    ctrl = _bare_controller()
    assert ctrl.has_active_sweeps() is False
    assert ctrl.sweep_submissions() == []

    ctrl._active_sweeps = 2
    ctrl._sweep_submissions.append({"attempt_id": "t0", "job_name": "x · trial 0"})
    assert ctrl.has_active_sweeps() is True

    snap = ctrl.sweep_submissions()
    assert snap == [{"attempt_id": "t0", "job_name": "x · trial 0"}]
    # Returned list is a copy — mutating it must not affect the controller.
    snap.append({"attempt_id": "leak"})
    assert len(ctrl._sweep_submissions) == 1
