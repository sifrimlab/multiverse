"""Tests for MLflow metric payload splitting and workspace discovery."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from multiverse.tracking import (finalize_parent_mlflow_run, find_metrics_json,
                                 load_run_metrics,
                                 log_successful_run_to_mlflow,
                                 sanitize_nan_inf, split_metrics_payload,
                                 start_parent_mlflow_run)


def test_split_metrics_payload_scalars_only():
    scalars, histories = split_metrics_payload({"loss": 0.42, "silhouette_score": 0.8})
    assert scalars == {"loss": 0.42, "silhouette_score": 0.8}
    assert histories == {}


def test_split_metrics_payload_with_history_block():
    payload = {
        "loss": 0.1,
        "history": {"loss": [0.9, 0.5, 0.1], "elbo_train": [100.0, 50.0, 10.0]},
    }
    scalars, histories = split_metrics_payload(payload)
    assert scalars["loss"] == 0.1
    assert histories["loss"] == [0.9, 0.5, 0.1]
    assert histories["elbo_train"] == [100.0, 50.0, 10.0]


def test_find_metrics_json_prefers_workspace_root(tmp_path):
    nested = tmp_path / "dataset" / "cobolt"
    nested.mkdir(parents=True)
    root_metrics = tmp_path / "metrics.json"
    nested_metrics = nested / "metrics.json"
    root_metrics.write_text(json.dumps({"loss": 0.2}), encoding="utf-8")
    nested_metrics.write_text(json.dumps({"loss": 0.9}), encoding="utf-8")
    assert find_metrics_json(str(tmp_path)) == str(root_metrics)


def test_find_metrics_json_falls_back_to_nested(tmp_path):
    nested = tmp_path / "dataset" / "totalvi"
    nested.mkdir(parents=True)
    nested_metrics = nested / "metrics.json"
    nested_metrics.write_text(json.dumps({"elbo_train": 1.0}), encoding="utf-8")
    assert find_metrics_json(str(tmp_path)) == str(nested_metrics)


def test_load_run_metrics_returns_empty_when_missing(tmp_path):
    assert load_run_metrics(str(tmp_path)) == {}


def test_log_successful_run_to_mlflow_logs_epoch_series(tmp_path):
    metrics = {
        "loss": 0.2,
        "history": {"loss": [0.8, 0.5, 0.2]},
    }
    fake_mlflow = MagicMock()
    fake_context = MagicMock()
    fake_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    fake_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
        log_successful_run_to_mlflow(
            job_context={"dataset_name": "pbmc", "experiment_name": "exp"},
            job_spec={"model_name": "cobolt", "hyperparameters": {"lr": 0.01}},
            metrics=metrics,
            artifacts_dir=str(tmp_path),
        )

    fake_mlflow.log_metric.assert_any_call("loss", 0.8, step=0)
    fake_mlflow.log_metric.assert_any_call("loss", 0.5, step=1)
    fake_mlflow.log_metric.assert_any_call("loss", 0.2, step=2)


def test_sanitize_nan_inf_replaces_nested_non_finite_values():
    payload = {
        "loss": float("nan"),
        "nested": {"pos": float("inf"), "neg": float("-inf"), "ok": 1.5},
        "history": [1.0, float("nan"), 2.0],
    }
    assert sanitize_nan_inf(payload) == {
        "loss": None,
        "nested": {"pos": None, "neg": None, "ok": 1.5},
        "history": [1.0, None, 2.0],
    }


def test_load_run_metrics_sanitizes_non_finite_values(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"loss": NaN, "score": Infinity, "ok": 0.75}', encoding="utf-8"
    )
    assert load_run_metrics(str(tmp_path)) == {"loss": None, "score": None, "ok": 0.75}


def test_split_metrics_payload_filters_non_finite_history():
    scalars, histories = split_metrics_payload(
        {"history": {"loss": [1.0, float("inf"), 0.5]}}
    )
    assert histories == {"loss": [1.0, 0.5]}
    assert scalars == {"loss": 0.5}


def test_start_parent_mlflow_run_uses_client_without_active_run():
    fake_client = MagicMock()
    fake_client.create_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-1")
    )
    fake_mlflow = MagicMock()
    fake_mlflow.tracking.MlflowClient.return_value = fake_client
    fake_mlflow.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="7")

    with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
        run_id = start_parent_mlflow_run(
            job_context={"experiment_name": "exp"},
            job_spec={
                "hyperparameters": {"lr": 0.01},
                "run_settings": {
                    "mlflow_tags": {"team": "bench"},
                    "optuna_trial_id": 3,
                },
            },
            run_name="pbmc-pca-run",
        )

    assert run_id == "run-1"
    fake_mlflow.start_run.assert_not_called()
    fake_client.create_run.assert_called_once_with(
        "7",
        tags={
            "mlflow.runName": "pbmc-pca-run",
            "team": "bench",
            "optuna_trial_id": "3",
        },
    )
    fake_client.log_param.assert_called_once_with("run-1", "lr", "0.01")
    assert any(
        call.args[:2] == ("run-1", "system/cpu_utilization_percentage")
        for call in fake_client.log_metric.call_args_list
    )


def test_finalize_parent_mlflow_run_uses_client_and_ignores_fluent_active_run(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok", encoding="utf-8")
    (tmp_path / ".promotion_complete").write_text("", encoding="utf-8")

    fake_client = MagicMock()
    fake_mlflow = MagicMock()
    fake_mlflow.tracking.MlflowClient.return_value = fake_client
    fake_mlflow.active_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="other-run")
    )

    with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
        finalize_parent_mlflow_run(
            run_id="run-1",
            job_context={"experiment_name": "exp"},
            job_spec={},
            metrics={"score": 0.7, "history": {"loss": [1.0, 0.5]}},
            artifacts_dir=str(tmp_path),
            status="FINISHED",
        )

    fake_mlflow.active_run.assert_not_called()
    fake_mlflow.start_run.assert_not_called()
    fake_mlflow.end_run.assert_not_called()
    fake_client.log_metric.assert_any_call("run-1", "loss", 1.0, step=0)
    fake_client.log_metric.assert_any_call("run-1", "loss", 0.5, step=1)
    fake_client.log_metric.assert_any_call("run-1", "score", 0.7, step=1)
    assert any(
        call.args[:2] == ("run-1", "system/cpu_utilization_percentage")
        for call in fake_client.log_metric.call_args_list
    )
    fake_client.log_artifact.assert_called_once_with(
        "run-1", str(artifact), artifact_path=None
    )
    fake_client.set_terminated.assert_called_once_with("run-1", status="FINISHED")


def test_finalize_parent_mlflow_run_replays_history_when_live_stream_missing(tmp_path):
    fake_client = MagicMock()
    fake_mlflow = MagicMock()
    fake_mlflow.tracking.MlflowClient.return_value = fake_client

    with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
        finalize_parent_mlflow_run(
            run_id="run-2",
            job_context={"experiment_name": "exp"},
            job_spec={},
            metrics={"history": {"loss": [0.9, 0.4], "ari": [0.1, 0.2]}},
            artifacts_dir=str(tmp_path),
        )

    fake_client.log_metric.assert_any_call("run-2", "loss", 0.9, step=0)
    fake_client.log_metric.assert_any_call("run-2", "loss", 0.4, step=1)
    fake_client.log_metric.assert_any_call("run-2", "ari", 0.1, step=0)
    fake_client.log_metric.assert_any_call("run-2", "ari", 0.2, step=1)


def test_finalize_parent_mlflow_run_logs_all_artifacts_recursively(tmp_path):
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    nested = tmp_path / "plots" / "umap"
    nested.mkdir(parents=True)
    (nested / "plot.png").write_text("png", encoding="utf-8")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".promotion_complete").write_text("", encoding="utf-8")

    fake_client = MagicMock()
    fake_mlflow = MagicMock()
    fake_mlflow.tracking.MlflowClient.return_value = fake_client

    with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
        finalize_parent_mlflow_run(
            run_id="run-artifacts",
            job_context={"experiment_name": "exp"},
            job_spec={},
            metrics={},
            artifacts_dir=str(tmp_path),
        )

    artifact_calls = [call.args for call in fake_client.log_artifact.call_args_list]
    assert ("run-artifacts", str(tmp_path / "root.txt")) in [
        args[:2] for args in artifact_calls
    ]
    assert fake_client.log_artifact.call_count == 3
    fake_client.log_artifact.assert_any_call(
        "run-artifacts", str(tmp_path / "root.txt"), artifact_path=None
    )
    fake_client.log_artifact.assert_any_call(
        "run-artifacts", str(metrics_dir / "metrics.json"), artifact_path="metrics"
    )
    fake_client.log_artifact.assert_any_call(
        "run-artifacts", str(nested / "plot.png"), artifact_path="plots/umap"
    )
