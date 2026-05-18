import json

from multiverse.evaluate import aggregate_results
from multiverse.tracking import sanitize_nan_inf


def test_malformed_metrics_isolates_failure(tmp_path):
    output = tmp_path / "out"
    good = output / "good"
    bad = output / "bad"
    good.mkdir(parents=True); bad.mkdir()
    (good / "metrics.json").write_text(json.dumps({"score": 0.7}), encoding="utf-8")
    (bad / "metrics.json").write_text("{bad", encoding="utf-8")

    result = aggregate_results({"good": "success", "bad": "success"}, str(output))
    assert result == {"good": {"score": 0.7}}


def test_nan_sanitizer_replaces_with_none():
    assert sanitize_nan_inf({"x": float("nan"), "y": [float("inf")]}) == {"x": None, "y": [None]}


def test_write_then_load_round_trip_nan(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "metrics.json").write_text('{"score": NaN}', encoding="utf-8")
    result = aggregate_results({"model": "success"}, str(tmp_path))
    assert result == {"model": {"score": None}}
