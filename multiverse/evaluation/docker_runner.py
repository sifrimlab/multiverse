"""Containerized cohort evaluation.

The host stays thin: it never imports the heavy scientific stack (muon,
scib-metrics, scanpy, …). Instead it resolves a launch cohort, filters it to
the members whose artifacts are ready, writes a trimmed evaluation config, and
shells out to ``docker run`` against the prebuilt ``multiverse-evaluate`` image
(``make build-evaluate``). The container runs ``python -m multiverse.evaluate``
where the heavy dependencies actually live.

This module uses the Docker *CLI* (via ``subprocess``), never the Docker SDK,
so it is safe to import from the GUI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cohort import (
    STATUS_READY,
    filter_cohort_for_evaluation,
    resolve_cohort_readiness,
)

DEFAULT_EVALUATION_IMAGE = "multiverse-evaluate:latest"
"""Tag produced by ``make build-evaluate``."""

EVAL_CONFIG_FILENAME = "eval_config.json"
"""Trimmed, ready-members-only config written next to ``cohort.json`` and fed
to the container as ``--config_path``."""

EVAL_DOCKERFILE_RELATIVE = "docker-env/evaluation.Dockerfile"
"""Location of the evaluation image recipe, relative to the repo root."""

_ENV_PASSTHROUGH = ("MLFLOW_TRACKING_URI", "MULTIVERSE_LOG_LEVEL")


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot be prepared or launched."""


@dataclass
class EvaluationPlan:
    """Everything needed to launch (or merely inspect) a container run."""

    image: str
    config_path: Path
    argv: List[str]
    ready_count: int
    mounts: List["Mount"] = field(default_factory=list)


@dataclass(frozen=True)
class Mount:
    host_path: str
    container_path: str
    mode: str  # "ro" or "rw"


def resolve_image(image: Optional[str] = None) -> str:
    """Return the evaluation image tag.

    Precedence: explicit argument, ``MULTIVERSE_EVALUATION_IMAGE`` env var,
    then the default ``make build-evaluate`` tag.
    """
    return image or os.environ.get("MULTIVERSE_EVALUATION_IMAGE") or DEFAULT_EVALUATION_IMAGE


def _abs(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _is_within(child: str, parent: str) -> bool:
    """True if ``child`` is the same path as or nested under directory ``parent``."""
    if child == parent:
        return False
    try:
        Path(child).relative_to(parent)
        return True
    except ValueError:
        return False


def build_mounts(
    cohort: Dict[str, Any], members: List[Dict[str, Any]], config_path: Path
) -> List[Mount]:
    """Compute bind mounts for an evaluation run.

    Mounts every distinct absolute path the container needs at the *same*
    container path so the absolute paths already recorded in the config resolve
    unchanged. The output tree is read-write; datasets and artifact dirs are
    read-only. Paths nested under another mounted directory are dropped to avoid
    redundant, overlapping binds.
    """
    modes: Dict[str, str] = {}

    output_dir = cohort.get("output_dir")
    if output_dir:
        modes[_abs(str(output_dir))] = "rw"

    for member in members:
        dataset_path = member.get("dataset_path_resolved") or member.get("dataset_path")
        if dataset_path:
            modes.setdefault(_abs(str(dataset_path)), "ro")
        artifact_dir = member.get("artifact_dir")
        if artifact_dir:
            modes.setdefault(_abs(str(artifact_dir)), "ro")

    # The trimmed config lives under output_dir, so it is normally covered by
    # the output_dir mount; add it explicitly only if it is not.
    modes.setdefault(_abs(str(config_path)), "ro")

    # Drop any path nested under another mounted directory (parent mode wins).
    paths = sorted(modes, key=len)
    kept: Dict[str, str] = {}
    for path in paths:
        if any(_is_within(path, parent) for parent in kept):
            continue
        kept[path] = modes[path]

    return [Mount(host_path=p, container_path=p, mode=m) for p, m in sorted(kept.items())]


def build_docker_argv(
    image: str, mounts: List[Mount], config_path: Path, *, force: bool = False
) -> List[str]:
    """Assemble the ``docker run`` argv for an evaluation container.

    ``force`` passes ``--force`` to the container entrypoint so members already
    recorded as ``done`` are re-evaluated instead of skipped.
    """
    argv: List[str] = ["docker", "run", "--rm"]
    for mount in mounts:
        argv += ["-v", f"{mount.host_path}:{mount.container_path}:{mount.mode}"]
    for key in _ENV_PASSTHROUGH:
        value = os.environ.get(key)
        if value:
            argv += ["-e", f"{key}={value}"]
    argv += [image, "--config_path", str(config_path)]
    if force:
        argv += ["--force"]
    return argv


def _repo_root() -> Path:
    """Repo root for the editable checkout (multiverse/evaluation/ -> ../..)."""
    return Path(__file__).resolve().parents[2]


def evaluation_dockerfile() -> Path:
    """Return the evaluation Dockerfile path, raising if it is absent.

    The image can only be built from a source checkout; a wheel install has no
    Dockerfile, so we surface a clear error rather than a confusing build fail.
    """
    dockerfile = _repo_root() / EVAL_DOCKERFILE_RELATIVE
    if not dockerfile.is_file():
        raise EvaluationError(
            f"evaluation Dockerfile not found at {dockerfile}; build the image "
            "manually with `make build-evaluate` from a source checkout."
        )
    return dockerfile


def build_image_argv(image: Optional[str] = None) -> List[str]:
    """Assemble the ``docker build`` argv for the evaluation image.

    Mirrors the ``make build-evaluate`` target: build context is the repo root
    so the Dockerfile's ``COPY pyproject.toml multiverse/ ...`` directives
    resolve. Raises :class:`EvaluationError` if the Dockerfile is missing.
    """
    resolved_image = resolve_image(image)
    dockerfile = evaluation_dockerfile()
    root = _repo_root()
    return [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "-t",
        resolved_image,
        str(root),
    ]


def build_image(image: Optional[str] = None) -> int:
    """Build the evaluation image, streaming build output to stdout.

    Returns the ``docker build`` exit code.
    """
    argv = build_image_argv(image)
    return subprocess.run(argv).returncode


def ensure_image(image: str, *, force_build: bool = False) -> None:
    """Make sure the evaluation image is present, building it when needed.

    Builds when ``force_build`` is set or the image is absent. Raises
    :class:`EvaluationError` if Docker is unavailable or the build fails.
    """
    if not docker_available():
        raise EvaluationError(
            "Docker is not available; start the Docker daemon (the evaluation "
            "workflow runs inside a container)."
        )
    if force_build or not image_present(image):
        code = build_image(image)
        if code != 0:
            raise EvaluationError(
                f"failed to build evaluation image {image!r} (docker build "
                f"exited {code}); see the build output above."
            )


def docker_available() -> bool:
    """True if the Docker CLI is on PATH and the daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def image_present(image: str) -> bool:
    """True if the local Docker daemon has the evaluation image."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _load_cohort(cohort_path: Path) -> Dict[str, Any]:
    if not cohort_path.is_file():
        raise EvaluationError(f"cohort file not found: {cohort_path}")
    try:
        with open(cohort_path, encoding="utf-8") as fh:
            cohort = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"could not read cohort {cohort_path}: {exc}") from exc
    if not isinstance(cohort, dict):
        raise EvaluationError(f"cohort {cohort_path} is not a JSON object")
    return cohort


def prepare_evaluation(
    cohort_path: Path,
    *,
    image: Optional[str] = None,
    ready_members_only: bool = True,
    force: bool = False,
    mvd_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
    completed_runs: Optional[Dict[str, Dict[str, str]]] = None,
) -> EvaluationPlan:
    """Resolve a cohort, write the trimmed evaluation config, and build argv.

    Writes ``eval_config.json`` next to ``cohort.json`` containing only the
    members the container should evaluate. Raises :class:`EvaluationError` when
    no members are ready. ``force`` re-evaluates members already recorded as
    ``done`` instead of skipping them.
    """
    cohort_path = Path(cohort_path).expanduser().resolve()
    cohort = _load_cohort(cohort_path)

    members_with_status = resolve_cohort_readiness(
        cohort, mvd_snapshots=mvd_snapshots, completed_runs=completed_runs
    )
    if ready_members_only:
        eval_cohort = filter_cohort_for_evaluation(cohort, members_with_status)
        eval_members = eval_cohort["members"]
    else:
        eval_cohort = dict(cohort)
        eval_cohort["members"] = members_with_status
        eval_members = members_with_status

    if not eval_members:
        raise EvaluationError(
            "no members are ready for evaluation; train models first so their "
            "artifacts pass readiness checks"
        )

    config_path = cohort_path.parent / EVAL_CONFIG_FILENAME
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(eval_cohort, fh, indent=2)

    resolved_image = resolve_image(image)
    mounts = build_mounts(eval_cohort, eval_members, config_path)
    argv = build_docker_argv(resolved_image, mounts, config_path, force=force)

    return EvaluationPlan(
        image=resolved_image,
        config_path=config_path,
        argv=argv,
        ready_count=len(eval_members),
        mounts=mounts,
    )


def preflight(image: str) -> None:
    """Raise :class:`EvaluationError` if Docker or the image is unavailable."""
    if not docker_available():
        raise EvaluationError(
            "Docker is not available; start the Docker daemon (the evaluation "
            "workflow runs inside a container)."
        )
    if not image_present(image):
        raise EvaluationError(
            f"evaluation image {image!r} not found; build it with "
            "`make build-evaluate`."
        )


def run_cohort_evaluation(
    cohort_path: Path,
    *,
    image: Optional[str] = None,
    ready_members_only: bool = True,
    force: bool = False,
    auto_build: bool = True,
    force_build: bool = False,
    skip_preflight: bool = False,
) -> int:
    """Prepare and launch a containerized evaluation, streaming logs to stdout.

    By default the evaluation image is built automatically when it is missing
    (``auto_build``); pass ``force_build`` to rebuild it even when present, or
    ``auto_build=False`` to fail fast on a missing image instead. ``force``
    re-evaluates members already recorded as ``done``. Returns the container's
    exit code (non-zero on any failure).
    """
    plan = prepare_evaluation(
        cohort_path, image=image, ready_members_only=ready_members_only, force=force
    )
    if not skip_preflight:
        if auto_build or force_build:
            ensure_image(plan.image, force_build=force_build)
        else:
            preflight(plan.image)
    completed = subprocess.run(plan.argv)
    return completed.returncode
