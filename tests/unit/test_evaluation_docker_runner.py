"""Tests for the containerized cohort evaluation runner.

These exercise the host-side logic only — readiness resolution is stubbed and
Docker is never invoked. The container contract itself is covered by the
in-container evaluation tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import multiverse.evaluation.docker_runner as dr
from multiverse.cli_entrypoints import evaluate_main
from multiverse.evaluation.cohort import (
    STATUS_READY,
    STATUS_RUNNING,
    filter_cohort_for_evaluation,
)


# ---------------------------------------------------------------------------
# resolve_image precedence
# ---------------------------------------------------------------------------


def test_resolve_image_default(monkeypatch):
    monkeypatch.delenv("MULTIVERSE_EVALUATION_IMAGE", raising=False)
    assert dr.resolve_image() == dr.DEFAULT_EVALUATION_IMAGE


def test_resolve_image_env_override(monkeypatch):
    monkeypatch.setenv("MULTIVERSE_EVALUATION_IMAGE", "custom:9")
    assert dr.resolve_image() == "custom:9"


def test_resolve_image_explicit_wins(monkeypatch):
    monkeypatch.setenv("MULTIVERSE_EVALUATION_IMAGE", "custom:9")
    assert dr.resolve_image("explicit:1") == "explicit:1"


# ---------------------------------------------------------------------------
# filter_cohort_for_evaluation
# ---------------------------------------------------------------------------


def test_filter_keeps_only_ready_members():
    cohort = {"output_dir": "/out", "members": ["raw"]}
    resolved = [
        {"member_id": "a", "readiness_status": STATUS_READY},
        {"member_id": "b", "readiness_status": STATUS_RUNNING},
        {"member_id": "c", "readiness_status": STATUS_READY},
    ]
    filtered = filter_cohort_for_evaluation(cohort, resolved)
    assert [m["member_id"] for m in filtered["members"]] == ["a", "c"]
    # Original cohort untouched.
    assert cohort["members"] == ["raw"]


# ---------------------------------------------------------------------------
# build_mounts
# ---------------------------------------------------------------------------


def test_build_mounts_modes_and_paths(tmp_path):
    cohort = {"output_dir": "/out"}
    members = [
        {"dataset_path_resolved": "/data/demo.h5mu", "artifact_dir": "/store/run1"},
    ]
    cfg = Path("/out/.multiverse/eval_config.json")
    mounts = {m.host_path: m for m in dr.build_mounts(cohort, members, cfg)}

    assert mounts["/out"].mode == "rw"
    assert mounts["/data/demo.h5mu"].mode == "ro"
    assert mounts["/store/run1"].mode == "ro"
    # Same path on both sides so config absolute paths resolve unchanged.
    for m in mounts.values():
        assert m.host_path == m.container_path


def test_build_docker_argv_force_appends_flag():
    mounts = [dr.Mount("/out", "/out", "rw")]
    cfg = Path("/out/.multiverse/eval_config.json")
    argv_plain = dr.build_docker_argv("img", mounts, cfg)
    argv_force = dr.build_docker_argv("img", mounts, cfg, force=True)
    assert "--force" not in argv_plain
    assert argv_force[-1] == "--force"


def test_launch_dir_is_writable_under_output_mount(tmp_path):
    """Per-member result files and evaluation_report.json are written under the
    launch dir, which must fall inside the rw output_dir mount so the container
    can write them back to the host."""
    from multiverse.evaluation.result import (evaluations_dir, report_path)

    out = tmp_path / "out"
    cohort = {"output_dir": str(out)}
    members = [{"dataset_path_resolved": "/data/demo.h5mu", "artifact_dir": "/store/run1"}]
    cfg = out / ".multiverse" / "eval_config.json"
    mounts = {m.host_path: m for m in dr.build_mounts(cohort, members, cfg)}

    rw_mounts = [Path(h) for h, m in mounts.items() if m.mode == "rw"]
    evdir = evaluations_dir(out, "L")
    rep = report_path(out, "L")
    assert any(evdir == p or evdir.is_relative_to(p) for p in rw_mounts)
    assert any(rep == p or rep.is_relative_to(p) for p in rw_mounts)


def test_build_mounts_dedupes_nested_paths():
    cohort = {"output_dir": "/out"}
    members = [
        # Both nested under /out -> covered by the rw output mount.
        {"dataset_path_resolved": "/out/nested.h5mu", "artifact_dir": "/out/sub/run"},
    ]
    cfg = Path("/out/.multiverse/eval_config.json")
    hosts = {m.host_path for m in dr.build_mounts(cohort, members, cfg)}
    assert hosts == {"/out"}


# ---------------------------------------------------------------------------
# build_docker_argv
# ---------------------------------------------------------------------------


def test_build_docker_argv_includes_mounts_env_and_config(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://host:25000")
    monkeypatch.delenv("MULTIVERSE_LOG_LEVEL", raising=False)
    mounts = [dr.Mount("/out", "/out", "rw")]
    cfg = Path("/out/.multiverse/eval_config.json")
    argv = dr.build_docker_argv("img:1", mounts, cfg)

    assert argv[:3] == ["docker", "run", "--rm"]
    assert "-v" in argv and "/out:/out:rw" in argv
    assert "MLFLOW_TRACKING_URI=http://host:25000" in argv
    assert argv[-3:] == ["img:1", "--config_path", str(cfg)]


# ---------------------------------------------------------------------------
# image build helpers
# ---------------------------------------------------------------------------


def test_build_image_argv_targets_repo_dockerfile(monkeypatch):
    monkeypatch.delenv("MULTIVERSE_EVALUATION_IMAGE", raising=False)
    argv = dr.build_image_argv()
    assert argv[:2] == ["docker", "build"]
    assert "-t" in argv and dr.DEFAULT_EVALUATION_IMAGE in argv
    joined = " ".join(argv)
    assert "evaluation.Dockerfile" in joined


def test_evaluation_dockerfile_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(dr, "_repo_root", lambda: tmp_path)
    with pytest.raises(dr.EvaluationError):
        dr.evaluation_dockerfile()


def test_ensure_image_builds_when_missing(monkeypatch):
    monkeypatch.setattr(dr, "docker_available", lambda: True)
    monkeypatch.setattr(dr, "image_present", lambda image: False)
    built = []
    monkeypatch.setattr(dr, "build_image", lambda image=None: built.append(image) or 0)
    dr.ensure_image("img:1")
    assert built == ["img:1"]


def test_ensure_image_force_rebuilds_when_present(monkeypatch):
    monkeypatch.setattr(dr, "docker_available", lambda: True)
    monkeypatch.setattr(dr, "image_present", lambda image: True)
    built = []
    monkeypatch.setattr(dr, "build_image", lambda image=None: built.append(image) or 0)
    dr.ensure_image("img:1", force_build=True)
    assert built == ["img:1"]


def test_ensure_image_skips_build_when_present(monkeypatch):
    monkeypatch.setattr(dr, "docker_available", lambda: True)
    monkeypatch.setattr(dr, "image_present", lambda image: True)
    built = []
    monkeypatch.setattr(dr, "build_image", lambda image=None: built.append(image) or 0)
    dr.ensure_image("img:1")
    assert built == []


def test_ensure_image_raises_when_docker_unavailable(monkeypatch):
    monkeypatch.setattr(dr, "docker_available", lambda: False)
    with pytest.raises(dr.EvaluationError):
        dr.ensure_image("img:1")


def test_ensure_image_raises_on_build_failure(monkeypatch):
    monkeypatch.setattr(dr, "docker_available", lambda: True)
    monkeypatch.setattr(dr, "image_present", lambda image: False)
    monkeypatch.setattr(dr, "build_image", lambda image=None: 1)
    with pytest.raises(dr.EvaluationError):
        dr.ensure_image("img:1")


# ---------------------------------------------------------------------------
# prepare_evaluation
# ---------------------------------------------------------------------------


def _write_cohort(tmp_path: Path) -> Path:
    cohort = {"output_dir": str(tmp_path), "launch_id": "L", "members": [{}, {}]}
    cpath = tmp_path / "cohort.json"
    cpath.write_text(json.dumps(cohort), encoding="utf-8")
    return cpath


def test_prepare_writes_filtered_config(tmp_path, monkeypatch):
    cpath = _write_cohort(tmp_path)
    resolved = [
        {
            "member_id": "a",
            "readiness_status": STATUS_READY,
            "dataset_path_resolved": str(tmp_path / "demo.h5mu"),
            "artifact_dir": str(tmp_path / "art"),
        },
        {"member_id": "b", "readiness_status": STATUS_RUNNING},
    ]
    monkeypatch.setattr(dr, "resolve_cohort_readiness", lambda c, **k: resolved)

    plan = dr.prepare_evaluation(cpath)

    assert plan.ready_count == 1
    written = json.loads((tmp_path / dr.EVAL_CONFIG_FILENAME).read_text())
    assert [m["member_id"] for m in written["members"]] == ["a"]
    assert plan.config_path == tmp_path / dr.EVAL_CONFIG_FILENAME
    assert plan.argv[-2:] == ["--config_path", str(plan.config_path)]


def test_prepare_raises_when_no_ready_members(tmp_path, monkeypatch):
    cpath = _write_cohort(tmp_path)
    monkeypatch.setattr(
        dr,
        "resolve_cohort_readiness",
        lambda c, **k: [{"readiness_status": STATUS_RUNNING}],
    )
    with pytest.raises(dr.EvaluationError):
        dr.prepare_evaluation(cpath)


def test_prepare_missing_cohort_raises(tmp_path):
    with pytest.raises(dr.EvaluationError):
        dr.prepare_evaluation(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# run_cohort_evaluation launch path
# ---------------------------------------------------------------------------


def test_run_cohort_evaluation_launches_and_returns_code(tmp_path, monkeypatch):
    cpath = _write_cohort(tmp_path)
    monkeypatch.setattr(
        dr,
        "resolve_cohort_readiness",
        lambda c, **k: [
            {
                "member_id": "a",
                "readiness_status": STATUS_READY,
                "dataset_path_resolved": str(tmp_path / "demo.h5mu"),
                "artifact_dir": str(tmp_path / "art"),
            }
        ],
    )

    launched = {}

    class _Completed:
        returncode = 0

    def _fake_run(argv, *a, **k):
        launched["argv"] = argv
        return _Completed()

    monkeypatch.setattr(dr.subprocess, "run", _fake_run)

    code = dr.run_cohort_evaluation(cpath, skip_preflight=True)
    assert code == 0
    assert launched["argv"][:3] == ["docker", "run", "--rm"]


def _ready_member(tmp_path: Path):
    return {
        "member_id": "a",
        "readiness_status": STATUS_READY,
        "dataset_path_resolved": str(tmp_path / "demo.h5mu"),
        "artifact_dir": str(tmp_path / "art"),
    }


def test_run_cohort_evaluation_auto_builds_missing_image(tmp_path, monkeypatch):
    cpath = _write_cohort(tmp_path)
    monkeypatch.setattr(
        dr, "resolve_cohort_readiness", lambda c, **k: [_ready_member(tmp_path)]
    )
    seen = {}
    monkeypatch.setattr(
        dr,
        "ensure_image",
        lambda image, *, force_build=False: seen.update(image=image, force=force_build),
    )

    class _Completed:
        returncode = 0

    monkeypatch.setattr(dr.subprocess, "run", lambda argv, *a, **k: _Completed())

    code = dr.run_cohort_evaluation(cpath, force_build=True)
    assert code == 0
    assert seen["force"] is True


def test_run_cohort_evaluation_no_build_uses_preflight(tmp_path, monkeypatch):
    cpath = _write_cohort(tmp_path)
    monkeypatch.setattr(
        dr, "resolve_cohort_readiness", lambda c, **k: [_ready_member(tmp_path)]
    )
    pf = {}
    monkeypatch.setattr(dr, "preflight", lambda image: pf.update(image=image))

    def _no_build(*a, **k):
        raise AssertionError("ensure_image should not be called with auto_build=False")

    monkeypatch.setattr(dr, "ensure_image", _no_build)

    class _Completed:
        returncode = 0

    monkeypatch.setattr(dr.subprocess, "run", lambda argv, *a, **k: _Completed())

    code = dr.run_cohort_evaluation(cpath, auto_build=False)
    assert code == 0
    assert pf["image"] == dr.DEFAULT_EVALUATION_IMAGE


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_cli_evaluate_dispatches_ready_only(tmp_path, monkeypatch):
    calls = {}

    def _fake(cohort_path, *, image, ready_members_only, force, auto_build, force_build):
        calls["cohort"] = cohort_path
        calls["image"] = image
        calls["ready"] = ready_members_only
        calls["force"] = force
        calls["auto_build"] = auto_build
        calls["force_build"] = force_build
        return 0

    monkeypatch.setattr(dr, "run_cohort_evaluation", _fake)
    code = evaluate_main(["--cohort", str(tmp_path / "c.json")])
    assert code == 0
    assert calls["ready"] is True
    assert calls["image"] is None
    # Auto-build on, force-build off, re-evaluation off by default.
    assert calls["auto_build"] is True
    assert calls["force_build"] is False
    assert calls["force"] is False


def test_cli_evaluate_force_reeval_flag(tmp_path, monkeypatch):
    calls = {}

    def _fake(cohort_path, *, image, ready_members_only, force, auto_build, force_build):
        calls["force"] = force
        return 0

    monkeypatch.setattr(dr, "run_cohort_evaluation", _fake)
    evaluate_main(["--cohort", str(tmp_path / "c.json"), "--force"])
    assert calls["force"] is True


def test_cli_evaluate_all_members_flag(tmp_path, monkeypatch):
    calls = {}

    def _fake(cohort_path, *, image, ready_members_only, force, auto_build, force_build):
        calls["ready"] = ready_members_only
        return 0

    monkeypatch.setattr(dr, "run_cohort_evaluation", _fake)
    evaluate_main(["--cohort", str(tmp_path / "c.json"), "--all-members"])
    assert calls["ready"] is False


def test_cli_evaluate_force_build_flag(tmp_path, monkeypatch):
    calls = {}

    def _fake(cohort_path, *, image, ready_members_only, force, auto_build, force_build):
        calls["force_build"] = force_build
        calls["auto_build"] = auto_build
        return 0

    monkeypatch.setattr(dr, "run_cohort_evaluation", _fake)
    evaluate_main(["--cohort", str(tmp_path / "c.json"), "--force-build"])
    assert calls["force_build"] is True
    assert calls["auto_build"] is True


def test_cli_evaluate_no_build_flag(tmp_path, monkeypatch):
    calls = {}

    def _fake(cohort_path, *, image, ready_members_only, force, auto_build, force_build):
        calls["auto_build"] = auto_build
        return 0

    monkeypatch.setattr(dr, "run_cohort_evaluation", _fake)
    evaluate_main(["--cohort", str(tmp_path / "c.json"), "--no-build"])
    assert calls["auto_build"] is False


def test_cli_evaluate_surfaces_evaluation_error(tmp_path, monkeypatch, capsys):
    def _fake(*a, **k):
        raise dr.EvaluationError("boom")

    monkeypatch.setattr(dr, "run_cohort_evaluation", _fake)
    code = evaluate_main(["--cohort", str(tmp_path / "c.json")])
    assert code == 2
    assert "boom" in capsys.readouterr().err
