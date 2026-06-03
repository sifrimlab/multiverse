"""``multiverse run --simple`` CLI surface.

Per R7 the simple-mode CLI shape is::

    multiverse run --simple <manifest.yaml> \\
      --out <bundle-dir> \\
      [--strict] \\
      [--validators basic|strict|developer] \\
      [--no-image-pull]

The Docker backend is imported lazily so that ``--help`` works on machines
without Docker. The synthetic backend is unavailable from the CLI by design;
tests construct ``SimpleModeRunner`` directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..artifact import ValidationLevel
from .manifest import SimpleManifestError, parse_simple_manifest
from .runner import JobStatus, SimpleModeResult, SimpleModeRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multiverse run --simple",
        description="Run multiverse jobs against the artifact contract without "
        "requiring the mvd daemon, MLflow, Optuna, the GUI, or the SQLite "
        "registry.",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to a simple-mode manifest YAML (schema documented in "
        "multiverse/simple/manifest.py).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output bundle root. One subdirectory per successful job; failed "
        "jobs land under <out>/_failed/<job>/.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Refuse non-strict-acceptable image identities and upgrade "
        "validator warnings to refusals (publication mode).",
    )
    parser.add_argument(
        "--validators",
        choices=["basic", "strict", "developer"],
        default=None,
        help="Validation level override. Defaults to basic; --strict pins it "
        "to strict regardless.",
    )
    parser.add_argument(
        "--no-image-pull",
        action="store_true",
        help="Do not attempt to pull the image even if a digest is declared. "
        "Local-only mode; required when running offline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed forwarded to the model backend; recorded in the manifest.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON summary on stdout in addition to "
        "the human-readable output.",
    )
    return parser


def _build_backend(no_image_pull: bool):
    """Lazy-import the Docker backend so the CLI can be inspected without it."""
    try:
        from .backends.docker import DockerBackend  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "The Docker backend is not available in this environment. "
            f"Install the 'docker' Python package. Underlying error: {exc}"
        )
    return DockerBackend(no_image_pull=no_image_pull)


def _emit_summary(result: SimpleModeResult, args: argparse.Namespace) -> None:
    n_success = sum(1 for o in result.outcomes if o.succeeded)
    n_failed = len(result.outcomes) - n_success
    print(
        f"simple-mode run complete: {n_success} succeeded, {n_failed} failed "
        f"(boot_id={result.boot_id})",
        file=sys.stderr,
    )
    for outcome in result.outcomes:
        if outcome.succeeded:
            print(
                f"  [ok]  {outcome.job_name}: bundle at {outcome.bundle_path}",
                file=sys.stderr,
            )
        else:
            print(
                f"  [!!]  {outcome.job_name}: {outcome.status.value} — see "
                f"{outcome.failure_dir}: {outcome.failure_reason}",
                file=sys.stderr,
            )
    if args.json:
        json.dump(
            {
                "boot_id": result.boot_id,
                "outcomes": [
                    {
                        "job": o.job_name,
                        "status": o.status.value,
                        "logical_run_id": o.logical_run_id,
                        "physical_attempt_id": o.physical_attempt_id,
                        "bundle_path": str(o.bundle_path) if o.bundle_path else None,
                        "failure_dir": str(o.failure_dir) if o.failure_dir else None,
                        "failure_reason": o.failure_reason,
                    }
                    for o in result.outcomes
                ],
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = parse_simple_manifest(args.manifest)
    except SimpleManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2

    validators = (
        ValidationLevel(args.validators) if args.validators else ValidationLevel.BASIC
    )

    backend = _build_backend(args.no_image_pull)

    runner = SimpleModeRunner(
        backend=backend,
        output_root=args.out,
        strict=args.strict,
        validators=validators,
        seed=args.seed,
    )
    result = runner.run(manifest)
    _emit_summary(result, args)

    any_failed = any(
        o.status is not JobStatus.ARTIFACT_SUCCESS for o in result.outcomes
    )
    return 1 if any_failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
