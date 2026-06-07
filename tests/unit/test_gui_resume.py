"""GUI resume wiring for mvd manifest launches.

Resume decoration now happens in the Run tab before ``_launch_mvd_runs`` is
called. These tests keep the Streamlit/Docker surface fake and verify the small
helpers plus the launch function's current contract: it receives an already
resume-decorated plan, submits only runnable jobs, logs skipped jobs, and treats
an all-skipped plan as a successful no-op.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import yaml

import multiverse.gui as gui


def _job(slug: str, image: str, **overrides: Any) -> Dict[str, Any]:
    job = {
        "model_slug": slug,
        "model_name": slug,
        "model_image": image,
        "model_version": "1.0",
        "dataset_slug": "demo",
        "dataset_name": "demo",
        "dataset_path": "/tmp/data.h5mu",
        "dataset_n_obs": 100,
        "dataset_n_vars": 50,
        "model_params": {},
        "params_hash": "ph" + slug,
    }
    job.update(overrides)
    return job


class _FakeController:
    def __init__(self) -> None:
        self.received: List[Dict[str, Any]] = []

    def submit_manifest(self, *, manifest_path, pending_jobs, manifest_text, seed):
        self.received = list(pending_jobs)
        runnable = [j for j in pending_jobs if not j.get("_skipped")]
        return [
            SimpleNamespace(
                attempt_id=f"att-{i}",
                job_name=str(j.get("model_slug")),
                to_dict=lambda j=j, i=i: {
                    "attempt_id": f"att-{i}",
                    "job_name": str(j.get("model_slug")),
                    "dataset": "demo",
                    "model": str(j.get("model_slug")),
                },
            )
            for i, j in enumerate(runnable)
        ]


def _patch_controller(monkeypatch):
    fake = _FakeController()
    monkeypatch.setattr(
        "multiverse.runner.mvd_inprocess.get_controller",
        lambda **_kwargs: fake,
    )
    monkeypatch.setattr(gui, "st", SimpleNamespace(session_state={}))
    return fake


# ---------------------------------------------------------------------------
# GUI resume policy helpers
# ---------------------------------------------------------------------------


def test_launch_skip_cli_defers_to_manifest_when_user_did_not_toggle():
    assert gui._launch_skip_cli(user_set=False, checkbox=True) is None
    assert gui._launch_skip_cli(user_set=False, checkbox=False) is None


def test_launch_skip_cli_user_toggle_is_explicit_override():
    assert gui._launch_skip_cli(user_set=True, checkbox=True) is True
    assert gui._launch_skip_cli(user_set=True, checkbox=False) is False


def test_manifest_skip_completed_default_reads_manifest_global(tmp_path: Path):
    manifest = tmp_path / "run_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"globals": {"skip_completed": True}, "jobs": []}),
        encoding="utf-8",
    )

    assert gui._manifest_skip_completed_default(str(manifest), tmp_path) is True


def test_manifest_skip_completed_default_is_none_when_unset(tmp_path: Path):
    manifest = tmp_path / "run_manifest.yaml"
    manifest.write_text(yaml.safe_dump({"globals": {}, "jobs": []}), encoding="utf-8")

    assert gui._manifest_skip_completed_default(str(manifest), tmp_path) is None


def test_slugs_needing_build_ignores_skipped_jobs():
    skipped_missing = _job("pca", "missing-pca:1", _skipped=True)
    runnable_missing = _job("mofa", "missing-mofa:1")
    runnable_present = _job("cobolt", "present-cobolt:1")

    def image_status(image: str):
        return (
            image.startswith("present"),
            None if image.startswith("present") else "missing",
        )

    assert gui._slugs_needing_build(
        [skipped_missing, runnable_missing, runnable_present],
        backend="docker",
        force_rebuild=False,
        image_status_fn=image_status,
    ) == ["mofa"]


def test_slugs_needing_build_force_rebuild_uses_only_runnable_jobs():
    skipped = _job("pca", "pca:1", _skipped=True)
    runnable = _job("mofa", "mofa:1")

    assert gui._slugs_needing_build(
        [skipped, runnable], backend="docker", force_rebuild=True
    ) == ["mofa"]


# ---------------------------------------------------------------------------
# _launch_mvd_runs current contract
# ---------------------------------------------------------------------------


def test_launch_submits_only_runnable_jobs_and_logs_skipped(tmp_path, monkeypatch):
    fake = _patch_controller(monkeypatch)
    output_dir = tmp_path / "out"
    skipped_artifact = tmp_path / "done-artifact"
    skipped_artifact.mkdir()
    skipped = _job(
        "pca",
        "multiverse-pca:1.0.0",
        _skipped=True,
        _skip_reason="completed logical run already has ARTIFACT_SUCCESS",
        _completed_attempt_id="att-done",
        _completed_artifact_dir=str(skipped_artifact),
    )
    runnable = _job("mofa", "multiverse-mofa:1.0.0")

    gui._launch_mvd_runs(
        manifest_file=tmp_path / "run_manifest.yaml",
        output_dir=str(output_dir),
        seed=42,
        pending_jobs=[skipped, runnable],
        manifest_text="",
        repo_root=tmp_path,
    )

    assert [j["model_slug"] for j in fake.received] == ["pca", "mofa"]
    submissions = gui.st.session_state["_mvd_submissions"]
    assert [s["model"] for s in submissions] == ["mofa"]
    assert gui.st.session_state["is_running"] is True
    assert gui.st.session_state["run_finalized"] is False
    log_text = "\n".join(gui.st.session_state["_run_log_lines"])
    assert "demo_pca: SKIPPED" in log_text
    assert "att-done" in log_text
    assert str(skipped_artifact) in log_text
    assert "mofa: SUBMITTED" in log_text


def test_launch_all_skipped_is_successful_noop(tmp_path, monkeypatch):
    fake = _patch_controller(monkeypatch)
    output_dir = tmp_path / "out"
    skipped = _job(
        "pca",
        "multiverse-pca:1.0.0",
        _skipped=True,
        _skip_reason="completed logical run already has ARTIFACT_SUCCESS",
        _completed_attempt_id="att-done",
        _completed_artifact_dir=str(tmp_path / "done"),
    )

    gui._launch_mvd_runs(
        manifest_file=tmp_path / "run_manifest.yaml",
        output_dir=str(output_dir),
        seed=42,
        pending_jobs=[skipped],
        manifest_text="",
        repo_root=tmp_path,
    )

    assert fake.received == []
    assert gui.st.session_state["_mvd_submissions"] == []
    assert gui.st.session_state["is_running"] is False
    assert gui.st.session_state["run_finalized"] is True
    assert gui.st.session_state["_run_returncode"] == 0
    assert gui.st.session_state["_run_all_skipped"] is True
    log_text = "\n".join(gui.st.session_state["_run_log_lines"])
    assert "demo_pca: SKIPPED" in log_text
    assert "All jobs already completed" in log_text


def test_launch_without_decorated_skip_submits_job(tmp_path, monkeypatch):
    fake = _patch_controller(monkeypatch)
    job = _job("pca", "multiverse-pca:1.0.0")

    gui._launch_mvd_runs(
        manifest_file=tmp_path / "run_manifest.yaml",
        output_dir=str(tmp_path / "out"),
        seed=42,
        pending_jobs=[job],
        manifest_text="",
        repo_root=tmp_path,
    )

    assert fake.received[0].get("_skipped") is not True
    assert [s["model"] for s in gui.st.session_state["_mvd_submissions"]] == ["pca"]
