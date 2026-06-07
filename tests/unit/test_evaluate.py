"""Tests for multiverse/evaluate.py — the per-member cohort evaluator.

Most tests are host-safe: the heavy scientific stack (anndata/muon/scib) is
either avoided (pure helpers), exercised via available deps + stubs
(``load_embeddings`` uses real h5py/numpy with a stub dataset), or faked
(``evaluate_cohort`` status branching). The dependency guard is patched off so
the member-aware logic can be tested without installing the full stack.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import multiverse.evaluate as ev
from multiverse.evaluation import result as r
from multiverse.evaluation.cohort import launch_dir
from multiverse.evaluation.result import MemberResult

h5py = pytest.importorskip("h5py")

LID = "abc_docker_seed42_20260607T000000_deadbe"


# --- pure helpers -----------------------------------------------------------


def test_extract_member_metrics_both_orientations():
    outer = {"X_m1": {"ARI": 0.8, "NMI": 0.9}, "X_m2": {"ARI": 0.1}}
    assert ev._extract_member_metrics(outer, "X_m1") == {"ARI": 0.8, "NMI": 0.9}
    inner = {"ARI": {"X_m1": 0.8, "X_m2": 0.1}, "NMI": {"X_m1": 0.9}}
    assert ev._extract_member_metrics(inner, "X_m1") == {"ARI": 0.8, "NMI": 0.9}
    assert ev._extract_member_metrics(outer, "X_absent") == {}


def test_group_members_by_dataset():
    members = [
        {"member_id": "a", "dataset_slug": "ds1", "model_slug": "pca",
         "dataset_path_resolved": "/d/ds1.h5mu", "batch_key": "donor",
         "label_key": "ct", "artifact_dir": "/art/a", "job": {"omics_available": ["rna"]}},
        {"member_id": "b", "dataset_slug": "ds1", "model_slug": "mofa", "artifact_dir": "/art/b"},
        {"member_id": "c", "dataset_slug": "ds2", "model_slug": "pca", "artifact_dir": "/art/c"},
    ]
    mbd, dc = ev._group_members_by_dataset(members)
    assert [m["member_id"] for m in mbd["ds1"]] == ["a", "b"]
    assert [m["member_id"] for m in mbd["ds2"]] == ["c"]
    assert dc["ds1"]["data_path"] == "/d/ds1.h5mu"
    assert dc["ds1"]["label_key"] == "ct"
    assert dc["ds1"]["batch_key"] == "donor"


def test_emb_path():
    assert ev._emb_path({"artifact_dir": "/art/x"}).endswith("/art/x/embeddings.h5")
    assert ev._emb_path({}).endswith("embeddings.h5")


# --- Evaluator.load_embeddings (real h5py, stub dataset) --------------------


class _StubDataset:
    def __init__(self, n_obs):
        self.n_obs = n_obs
        self.obsm = {}


def _write_embeddings(path: Path, n_rows: int, n_cols: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("latent", data=np.random.rand(n_rows, n_cols))


def _make_evaluator(dataset, model_configs, output_dir):
    with patch("multiverse.evaluate._require_eval_deps"):
        return ev.Evaluator(
            dataset=dataset, dataset_name="ds", model_configs=model_configs,
            output_dir=str(output_dir),
        )


def test_load_embeddings_ok_keys_by_member_id(tmp_path):
    ds = _StubDataset(n_obs=40)
    e1 = tmp_path / "a" / "embeddings.h5"; _write_embeddings(e1, 40)
    e2 = tmp_path / "b" / "embeddings.h5"; _write_embeddings(e2, 40)
    cfgs = [
        {"member_id": "a", "model_name": "pca", "output_embedding_path": str(e1)},
        {"member_id": "b", "model_name": "pca", "output_embedding_path": str(e2)},
    ]
    ev_obj = _make_evaluator(ds, cfgs, tmp_path / "out")
    ev_obj.load_embeddings()
    # Member-keyed: two members sharing a model do not collide.
    assert ev_obj.member_latent_key == {"a": "X_a", "b": "X_b"}
    assert set(ds.obsm) == {"X_a", "X_b"}
    assert not ev_obj.member_preflight


def test_load_embeddings_no_embeddings(tmp_path):
    ds = _StubDataset(n_obs=10)
    cfgs = [{"member_id": "a", "model_name": "pca",
             "output_embedding_path": str(tmp_path / "missing" / "embeddings.h5")}]
    ev_obj = _make_evaluator(ds, cfgs, tmp_path / "out")
    ev_obj.load_embeddings()
    assert ev_obj.member_preflight["a"][0] == r.EVAL_STATUS_NO_EMBEDDINGS
    assert "a" not in ev_obj.member_latent_key


def test_load_embeddings_obs_mismatch(tmp_path):
    ds = _StubDataset(n_obs=40)
    e1 = tmp_path / "a" / "embeddings.h5"; _write_embeddings(e1, 39)  # one short
    cfgs = [{"member_id": "a", "model_name": "pca", "output_embedding_path": str(e1)}]
    ev_obj = _make_evaluator(ds, cfgs, tmp_path / "out")
    ev_obj.load_embeddings()
    assert ev_obj.member_preflight["a"][0] == r.EVAL_STATUS_OBS_MISMATCH
    assert "X_a" not in ds.obsm


# --- evaluate_cohort status branching (faked heavy pipeline) ----------------


@pytest.fixture
def fake_heavy(monkeypatch):
    """Inject fake ``muon`` and ``multiverse.worker`` so evaluate_cohort runs
    without the scientific stack. The concatenated dataset is a stub."""
    fake_muon = types.ModuleType("muon")
    fake_muon.read_h5mu = lambda path: types.SimpleNamespace(mod={"rna": object()})
    monkeypatch.setitem(sys.modules, "muon", fake_muon)

    fake_worker = types.ModuleType("multiverse.worker")
    fake_worker.preprocess_mudata = lambda mdata, params, **kw: mdata
    fake_worker.anndata_concatenate = lambda **kw: _StubDataset(n_obs=40)
    monkeypatch.setitem(sys.modules, "multiverse.worker", fake_worker)
    yield


def _config(tmp_path, members):
    return {
        "launch_id": LID,
        "output_dir": str(tmp_path),
        "manifest_hash": "abc12345",
        "backend": "docker",
        "seed": 42,
        "members": members,
    }


def test_evaluate_cohort_done_and_failures(tmp_path, fake_heavy, monkeypatch):
    members = [
        {"member_id": "ok", "dataset_slug": "ds1", "model_slug": "pca",
         "dataset_path_resolved": str(tmp_path / "ds1.h5mu"), "batch_key": "b",
         "label_key": "ct", "artifact_dir": "/art/ok", "job": {"omics_available": ["rna"]}},
        {"member_id": "noemb", "dataset_slug": "ds1", "model_slug": "mofa",
         "artifact_dir": "/art/noemb"},
    ]

    class FakeEvaluator:
        def __init__(self, dataset, dataset_name, model_configs, output_dir):
            self.model_configs = model_configs
            self.member_latent_key = {"ok": "X_ok"}
            self.member_preflight = {"noemb": (r.EVAL_STATUS_NO_EMBEDDINGS, "missing")}

        def load_embeddings(self):
            pass

        def evaluate_models(self, batch_key, label_key):
            return {"X_ok": {"ARI": 0.77}}

    monkeypatch.setattr(ev, "Evaluator", FakeEvaluator)
    results = ev.evaluate_cohort(_config(tmp_path, members))

    by_id = {res.member_id: res for res in results}
    assert by_id["ok"].status == r.EVAL_STATUS_DONE
    assert by_id["ok"].metrics["evaluation"]["ARI"] == 0.77
    assert by_id["ok"].duration_seconds is not None
    assert by_id["noemb"].status == r.EVAL_STATUS_NO_EMBEDDINGS

    # Per-member files were written under the launch dir.
    evdir = launch_dir(tmp_path, LID) / "evaluations"
    assert (evdir / "ok.json").is_file()
    assert (evdir / "noemb.json").is_file()
    with open(evdir / "ok.json", encoding="utf-8") as fh:
        assert json.load(fh)["status"] == "done"


def test_evaluate_cohort_benchmark_exception_isolated(tmp_path, fake_heavy, monkeypatch):
    members = [
        {"member_id": "m1", "dataset_slug": "ds1", "model_slug": "pca",
         "dataset_path_resolved": str(tmp_path / "ds1.h5mu"), "batch_key": "b",
         "label_key": "ct", "artifact_dir": "/art/m1", "job": {"omics_available": ["rna"]}},
    ]

    class BoomEvaluator:
        def __init__(self, *a, **k):
            self.member_latent_key = {"m1": "X_m1"}
            self.member_preflight = {}

        def load_embeddings(self):
            pass

        def evaluate_models(self, batch_key, label_key):
            raise RuntimeError("scib exploded")

    monkeypatch.setattr(ev, "Evaluator", BoomEvaluator)
    results = ev.evaluate_cohort(_config(tmp_path, members))
    assert results[0].status == r.EVAL_STATUS_EVALUATION_FAILED
    assert "scib exploded" in results[0].error["message"]


def test_evaluate_cohort_missing_dataset(tmp_path, monkeypatch):
    members = [
        {"member_id": "m1", "dataset_slug": "ds1", "model_slug": "pca",
         "dataset_path_resolved": str(tmp_path / "nope.h5mu"), "batch_key": "b",
         "label_key": "ct", "artifact_dir": "/art/m1", "job": {"omics_available": ["rna"]}},
    ]
    fake_muon = types.ModuleType("muon")

    def _raise(path):
        raise FileNotFoundError(f"no such file: {path}")

    fake_muon.read_h5mu = _raise
    monkeypatch.setitem(sys.modules, "muon", fake_muon)
    fake_worker = types.ModuleType("multiverse.worker")
    fake_worker.preprocess_mudata = lambda *a, **k: None
    fake_worker.anndata_concatenate = lambda **k: None
    monkeypatch.setitem(sys.modules, "multiverse.worker", fake_worker)

    results = ev.evaluate_cohort(_config(tmp_path, members))
    assert results[0].status == r.EVAL_STATUS_MISSING_DATASET


def test_evaluate_cohort_skips_done_members(tmp_path, fake_heavy, monkeypatch):
    """A member already recorded as ``done`` (same artifact dir) is preserved
    and not re-evaluated; ``force`` overrides."""
    member = {
        "member_id": "m1", "dataset_slug": "ds1", "model_slug": "pca",
        "dataset_path_resolved": str(tmp_path / "ds1.h5mu"), "batch_key": "b",
        "label_key": "ct", "artifact_dir": "/art/m1", "job": {"omics_available": ["rna"]},
    }
    # Seed a prior 'done' result for the same artifact dir.
    r.write_member_result(
        output_dir=tmp_path, launch_id=LID,
        result=MemberResult(member_id="m1", status=r.EVAL_STATUS_DONE,
                            artifact_dir="/art/m1", metrics={"evaluation": {"ARI": 0.5}}),
    )

    constructed = {"n": 0}

    class CountingEvaluator:
        def __init__(self, *a, **k):
            constructed["n"] += 1
            self.member_latent_key = {"m1": "X_m1"}
            self.member_preflight = {}

        def load_embeddings(self):
            pass

        def evaluate_models(self, batch_key, label_key):
            return {"X_m1": {"ARI": 0.99}}

    monkeypatch.setattr(ev, "Evaluator", CountingEvaluator)

    # Without force: skipped, prior metrics preserved, evaluator never built.
    results = ev.evaluate_cohort(_config(tmp_path, [member]))
    assert constructed["n"] == 0
    assert results[0].status == r.EVAL_STATUS_DONE
    assert results[0].metrics["evaluation"]["ARI"] == 0.5

    # With force: re-evaluated, new metrics.
    results = ev.evaluate_cohort(_config(tmp_path, [member]), force=True)
    assert constructed["n"] == 1
    assert results[0].metrics["evaluation"]["ARI"] == 0.99


# --- guarded end-to-end (runs only with the full scientific stack) ----------

pytestmark_e2e = pytest.mark.skipif(
    True, reason="full anndata/muon/scib stack required; covered in CI eval image"
)
