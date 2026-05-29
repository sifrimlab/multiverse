from unittest.mock import MagicMock, patch

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
