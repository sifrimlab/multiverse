"""Tests for MLflow metric payload splitting and workspace discovery."""
import json
from unittest.mock import MagicMock, patch

import pytest

from multiverse.tracking import (
    find_metrics_json,
    load_run_metrics,
    log_successful_run_to_mlflow,
    sanitize_nan_inf,
    split_metrics_payload,
)


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
    metrics_path.write_text('{"loss": NaN, "score": Infinity, "ok": 0.75}', encoding="utf-8")
    assert load_run_metrics(str(tmp_path)) == {"loss": None, "score": None, "ok": 0.75}


def test_split_metrics_payload_filters_non_finite_history():
    scalars, histories = split_metrics_payload({"history": {"loss": [1.0, float("inf"), 0.5]}})
    assert histories == {"loss": [1.0, 0.5]}
    assert scalars == {"loss": 0.5}
