"""Docker image build utilities for model container contexts."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Optional

import docker
from rich.console import Console

from .logging_utils import get_logger
from .models_ingest import ModelManifest

logger = get_logger(__name__)
console = Console()


def _build_context_tar(context_path: Path, dockerfile_rel: str) -> io.BytesIO:
    """Return an in-memory tar of the build context with all UIDs/GIDs set to 0.

    Sending a tar with root ownership avoids lchown failures on NFS mounts where
    the user UID (e.g. 665170964 from LDAP/AD) is too large for the Docker overlay2
    storage driver's UID map.

    Only includes what each Dockerfile actually COPYs:
      - pyproject.toml + README.md  (package metadata)
      - multiverse/                 (orchestrator + worker SDK)
      - store/models/<model>/container/  (Dockerfile, environment.yml, run.py)
    """
    buf = io.BytesIO()

    def _strip_uid(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in info.name or info.name.endswith((".pyc", ".pyo")):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    def _add(tar: tarfile.TarFile, abs_path: Path, arcname: str) -> None:
        tar.add(str(abs_path), arcname=arcname, recursive=True, filter=_strip_uid)

    with tarfile.open(fileobj=buf, mode="w") as tar:
        # Package metadata
        for fname in ("pyproject.toml", "README.md"):
            meta_file = context_path / fname
            if meta_file.exists():
                tar.add(str(meta_file), arcname=fname, filter=_strip_uid)
        # Package source
        multiverse_dir = context_path / "multiverse"
        if multiverse_dir.exists():
            _add(tar, multiverse_dir, "multiverse")
        container_dir = (context_path / dockerfile_rel).parent
        container_arcdir = str(Path(dockerfile_rel).parent)
        _add(tar, container_dir, container_arcdir)

    buf.seek(0)
    return buf


def build_local_model(manifest: ModelManifest) -> Optional[str]:
    """Build a local Docker image for a model manifest."""
    if manifest.build is None:
        logger.info("Remote image expected, skipping local build")
        return None

    if not manifest.manifest_path:
        raise ValueError("Model manifest_path is required for local builds.")

    manifest_dir = Path(manifest.manifest_path).resolve().parent
    context_path = (manifest_dir / manifest.build.context).resolve()
    dockerfile_abs = (context_path / manifest.build.dockerfile).resolve()

    if not context_path.exists():
        raise FileNotFoundError(f"Build context not found: {context_path}")
    if not dockerfile_abs.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_abs}")

    dockerfile_rel = os.path.relpath(dockerfile_abs, context_path)
    client = docker.from_env()

    console.print(
        f"[cyan]Building local image[/cyan] [bold]{manifest.runtime.image}[/bold] "
        f"from {context_path} ({dockerfile_rel})"
    )
    context_tar = _build_context_tar(context_path, dockerfile_rel)
    image, logs = client.images.build(
        fileobj=context_tar,
        custom_context=True,
        dockerfile=dockerfile_rel,
        tag=manifest.runtime.image,
        rm=True,
        pull=False,
    )
    for chunk in logs:
        if isinstance(chunk, dict):
            if "stream" in chunk:
                console.print(chunk["stream"], end="")
            elif "error" in chunk:
                console.print(f"[red]{chunk['error']}[/red]")
            elif "status" in chunk:
                progress = chunk.get("progress", "")
                message = f"{chunk['status']} {progress}".strip()
                console.print(message)
        else:
            console.print(str(chunk), end="")

    logger.info(f"Built local image {manifest.runtime.image} ({image.short_id})")
    return manifest.runtime.image
