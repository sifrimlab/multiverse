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


class _RecordingTrial:
    """Minimal Optuna-trial stand-in that records suggest_* call kwargs.

    ``_sample_param`` only delegates to the trial's suggest_* methods, so we can
    exercise its distribution routing without a real Optuna study.
    """

    def __init__(self):
        self.calls = []

    def suggest_int(self, name, low, high, **kwargs):
        self.calls.append(("int", name, low, high, kwargs))
        return low

    def suggest_float(self, name, low, high, **kwargs):
        self.calls.append(("float", name, low, high, kwargs))
        return low

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, choices))
        return choices[0]


def test_sample_param_float_routes_to_suggest_float_with_log():
    # GUI float sweep widget emits {"type": "float", ..., "log": bool}; the
    # tuner must accept it (Bug 2 compatibility), not raise.
    from multiverse.runner.tuner import _sample_param

    trial = _RecordingTrial()
    _sample_param(trial, "lr", {"type": "float", "low": 1e-4, "high": 1e-1, "log": True})

    assert trial.calls == [("float", "lr", 1e-4, 1e-1, {"log": True})]


def test_sample_param_float_linear_passes_log_false():
    from multiverse.runner.tuner import _sample_param

    trial = _RecordingTrial()
    _sample_param(trial, "dropout", {"type": "float", "low": 0.0, "high": 0.5})

    assert trial.calls == [("float", "dropout", 0.0, 0.5, {"log": False})]


def test_sample_param_int_log_omits_step():
    # Optuna forbids a custom step together with log=True, so the log path must
    # pass log=True and omit step entirely.
    from multiverse.runner.tuner import _sample_param

    trial = _RecordingTrial()
    _sample_param(trial, "n", {"type": "int", "low": 2, "high": 64, "log": True})

    assert trial.calls == [("int", "n", 2, 64, {"log": True})]


def test_sample_param_int_linear_uses_step():
    from multiverse.runner.tuner import _sample_param

    trial = _RecordingTrial()
    _sample_param(trial, "k", {"type": "int", "low": 2, "high": 8})

    assert trial.calls == [("int", "k", 2, 8, {"step": 1})]


def test_sample_param_rejects_unknown_type():
    from multiverse.runner.tuner import _sample_param

    with pytest.raises(ValueError):
        _sample_param(_RecordingTrial(), "x", {"type": "mystery"})


def test_build_trial_job_merges_params_and_isolates_artifacts():
    # Shared by both trial runners (default kernel-spawning + GUI in-process):
    # sampled params override base model_params, mode is forced to run, and the
    # trial gets its own artifact dir + trial/study env stamps.
    from multiverse.runner.tuner import build_trial_job

    base = {
        "model_params": {"epochs": 3, "n_latent": 0},
        "mode": "sweep",
        "artifact_dir_name": "exp_ds_model_hash",
        "output_path": "/art/exp_ds_model_hash",
        "_exec": {"state_root": "/state"},
    }
    trial = build_trial_job(base, {"n_latent": 8}, 2, study_name="study_x")

    assert trial["model_params"] == {"epochs": 3, "n_latent": 8}
    assert trial["mode"] == "run"
    assert trial["artifact_dir_name"] == "exp_ds_model_hash_trial2"
    assert trial["output_path"] == "/art/exp_ds_model_hash_trial2"
    assert trial["container_env_extra"]["MULTIVERSE_TRIAL_NUMBER"] == "2"
    assert trial["container_env_extra"]["MULTIVERSE_STUDY_NAME"] == "study_x"
    # The private execution context never leaks into the trial job.
    assert "_exec" not in trial


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
