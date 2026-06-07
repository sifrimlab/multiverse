import json

import pytest


def test_missing_dotted_key_prunes_trial():
    from multiverse.runner.tuner import _extract_metric

    optuna = pytest.importorskip("optuna")
    with pytest.raises(optuna.TrialPruned):
        _extract_metric({"outer": {}}, "outer.missing")


def test_nested_numeric_metric_resolves():
    from multiverse.runner.tuner import _extract_metric

    pytest.importorskip("optuna")
    assert _extract_metric({"outer": {"score": 0.4}}, "outer.score") == 0.4


def _write_metrics(tmp_path, name, payload):
    """Write metrics.json under <tmp>/<name>/outputs (bundle layout)."""
    outputs = tmp_path / name / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(tmp_path / name)


def test_objective_samples_runs_and_extracts_metric(tmp_path):
    from multiverse.runner.tuner import objective

    optuna = pytest.importorskip("optuna")

    job = {
        "search_space": {"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1}},
        "optimize_metric": "final.ari",
        "model_params": {"epochs": 5},
    }

    seen = {}

    def fake_runner(base_job, sampled, trial_number):
        # The sampled hyperparameter is surfaced and the base params preserved.
        seen["sampled"] = sampled
        seen["epochs"] = base_job["model_params"]["epochs"]
        art = _write_metrics(tmp_path, f"run_{trial_number}", {"final": {"ari": 0.73}})
        return {
            "primary_state": "ARTIFACT_SUCCESS",
            "artifact_dir": art,
            "physical_attempt_id": f"att-{trial_number}",
        }

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, job, run_trial=fake_runner), n_trials=1
    )

    assert study.best_value == 0.73
    assert "lr" in seen["sampled"]
    assert seen["epochs"] == 5
    assert study.best_trial.user_attrs["artifact_dir"].endswith("run_0")


def test_trials_run_sequentially_n_plus_one_after_n(tmp_path):
    """Optuna stays dynamic: trial N+1 must only start after trial N's
    objective returns a metric — no up-front trial list."""
    from multiverse.runner.tuner import objective

    optuna = pytest.importorskip("optuna")

    job = {
        "search_space": {"lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1}},
        "optimize_metric": "score",
    }

    events = []

    def ordered_runner(base_job, sampled, trial_number):
        events.append(("run_start", trial_number))
        art = _write_metrics(tmp_path, f"run_{trial_number}", {"score": float(trial_number)})
        events.append(("run_done", trial_number))
        return {
            "primary_state": "ARTIFACT_SUCCESS",
            "artifact_dir": art,
            "physical_attempt_id": f"att-{trial_number}",
        }

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, job, run_trial=ordered_runner), n_trials=3
    )

    # Each trial fully completes (run_start → run_done → metric read) before
    # the next trial's run begins.
    starts = [n for kind, n in events if kind == "run_start"]
    assert starts == [0, 1, 2]
    for i in range(len(events) - 1):
        if events[i] == ("run_start", 1):
            assert events[i - 1] == ("run_done", 0)
        if events[i] == ("run_start", 2):
            assert events[i - 1] == ("run_done", 1)
    assert len(study.trials) == 3


def test_objective_prunes_when_attempt_fails(tmp_path):
    from multiverse.runner.tuner import objective

    optuna = pytest.importorskip("optuna")

    job = {
        "search_space": {"k": {"type": "int", "low": 1, "high": 4}},
        "optimize_metric": "score",
    }

    def failing_runner(base_job, sampled, trial_number):
        return {"primary_state": "FAILED", "failure_reason": "container exited 1"}

    study = optuna.create_study(direction="maximize")
    # study.optimize converts a raised TrialPruned into a PRUNED trial rather
    # than re-raising, so the sweep keeps going instead of aborting.
    study.optimize(
        lambda t: objective(t, job, run_trial=failing_runner), n_trials=1
    )
    assert all(
        tr.state == optuna.trial.TrialState.PRUNED for tr in study.trials
    )


def test_objective_rejects_missing_optimize_metric():
    from multiverse.runner.tuner import objective

    pytest.importorskip("optuna")

    job = {"search_space": {"k": {"type": "int", "low": 1, "high": 2}}}
    with pytest.raises(ValueError):
        objective(object(), job, run_trial=lambda *a: {})


def test_default_runner_overrides_params_and_artifact_dir(monkeypatch):
    """The default trial runner merges sampled params and isolates artifacts."""
    import multiverse.runner.mvd_entrypoint as entry
    from multiverse.runner.tuner import _default_trial_runner

    pytest.importorskip("optuna")

    captured = {}

    def fake_attempt(**kwargs):
        captured.update(kwargs)
        return {"primary_state": "ARTIFACT_SUCCESS", "artifact_dir": "/x"}

    monkeypatch.setattr(entry, "run_trial_attempt", fake_attempt)

    job = {
        "model_params": {"epochs": 3},
        "artifact_dir_name": "exp_ds_model_hash",
        "_exec": {
            "state_root": "/state",
            "artifact_root": "/art",
            "manifest_hash": "abc",
            "seed": 7,
            "backend": "docker",
            "accept_degraded": True,
        },
    }
    runner = _default_trial_runner(job)
    runner(job, {"lr": 0.01}, 2)

    sent = captured["job"]
    assert sent["model_params"] == {"epochs": 3, "lr": 0.01}
    assert sent["artifact_dir_name"] == "exp_ds_model_hash_trial2"
    assert sent["mode"] == "run"
    assert "_exec" not in sent  # private context never reaches the executor
    assert captured["state_root"] == "/state"
    assert captured["seed"] == 7
    assert captured["backend"] == "docker"
    assert captured["accept_degraded"] is True
