"""Artifact and log rendering helpers for the Streamlit GUI."""

from __future__ import annotations

import mimetypes
from pathlib import Path

import pandas as pd
import streamlit as st

from multiverse.gui_telemetry import track

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def find_umap_images(artifact_dir: Path) -> list[Path]:
    """Find UMAP preview images within an artifact bundle directory.

    Recurses through the directory and selects image files whose stem contains
    ``umap`` (case-insensitive), so the Results view can surface them inline.

    Args:
        artifact_dir: Directory holding a promoted artifact bundle's files.

    Returns:
        Image paths sorted by their path relative to ``artifact_dir``; empty
        when the directory is missing or holds no matching images.
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        return []
    images = [
        path
        for path in artifact_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "umap" in path.stem.lower()
    ]
    return sorted(images, key=lambda p: str(p.relative_to(artifact_dir)))


def render_download_button(
    path: Path,
    label: str | None = None,
    *,
    max_download_mb: float = 200,
) -> None:
    """Render a Streamlit download button for a single file.

    Files larger than the in-browser limit are not offered for download;
    instead the absolute path is shown so the user can fetch it directly off
    the shared filesystem. A successful click emits a ``download_clicked``
    telemetry event.

    Args:
        path: File to expose for download.
        label: Button caption; defaults to ``Download <filename>``.
        max_download_mb: Upper size bound for an in-browser download; above it
            the path is displayed instead of the button.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        st.caption(f"Unavailable: `{path}`")
        return

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_download_mb:
        st.caption(
            f"`{path.name}` is {size_mb:.1f} MB, larger than the "
            f"{max_download_mb:g} MB in-browser download limit."
        )
        st.code(str(path), language=None)
        return

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    button_label = label or f"Download {path.name}"
    clicked = st.download_button(
        button_label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        key=f"download::{path.resolve()}::{button_label}::{path.stat().st_mtime_ns}",
        help=f"{size_mb:.2f} MB",
    )
    if clicked:
        track(
            "download_clicked",
            artifact_kind=path.suffix.lower().lstrip(".") or "file",
            size_mb=round(size_mb, 4),
            path=str(path),
        )


def render_log_viewer(
    log_path: Path, *, default_tail: int = 200, with_filter: bool = True
) -> None:
    """Render a tailing, optionally filtered viewer for a log file.

    Args:
        log_path: Path to the log file to display.
        default_tail: Initial number of trailing lines to show.
        with_filter: When True, expose a case-insensitive substring filter that
            restricts the displayed (and counted) lines.
    """
    log_path = Path(log_path)
    if not log_path.exists() or not log_path.is_file():
        st.info(f"No log found at `{log_path}`.")
        return

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    c_tail, c_download = st.columns([3, 1])
    with c_tail:
        tail = st.number_input(
            "Lines",
            min_value=20,
            max_value=max(20, len(lines)),
            value=min(default_tail, max(20, len(lines))),
            step=20,
            key=f"log_tail::{log_path.resolve()}",
        )
    with c_download:
        st.write("")
        render_download_button(log_path, "Download log")

    if with_filter:
        query = st.text_input(
            "Filter log", key=f"log_filter::{log_path.resolve()}"
        ).strip()
    else:
        query = ""

    visible = lines
    if query:
        visible = [line for line in visible if query.lower() in line.lower()]
    st.code("\n".join(visible[-int(tail) :]) or "(empty log)", language=None)
    st.caption(
        f"Showing {min(int(tail), len(visible))} of {len(visible)} matching line(s)."
    )


def _render_image_preview(path: Path, *, max_preview_mb: float = 25) -> None:
    """Render an inline image preview, falling back to a download for large files."""
    path = Path(path)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= max_preview_mb:
        st.image(str(path), caption=path.name)
    else:
        st.caption(
            f"Preview skipped because `{path.name}` is {size_mb:.1f} MB, "
            f"larger than the {max_preview_mb:g} MB image preview limit."
        )
        st.code(str(path), language=None)
    render_download_button(path, f"Download {path.name}")


def render_artifact_tree(artifact_dir: Path, *, max_inline_mb: float = 200) -> None:
    """Render the file tree of an artifact bundle with inline previews.

    Lists every file with its size, then per-file shows an image preview, an
    inline text preview (truncated), or a download button depending on type and
    size. UMAP images are expanded by default.

    Args:
        artifact_dir: Directory of the promoted artifact bundle to display.
        max_inline_mb: Size ceiling above which inline previews are skipped in
            favour of a download button or a path notice.
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        st.warning(f"Artifact directory not found: `{artifact_dir}`")
        return

    files = sorted(path for path in artifact_dir.rglob("*") if path.is_file())
    if not files:
        st.info("No artifacts found in this directory.")
        return

    rows = []
    for path in files:
        rel = path.relative_to(artifact_dir)
        size_mb = path.stat().st_size / (1024 * 1024)
        rows.append({"File": str(rel), "Size MB": round(size_mb, 3), "Path": str(path)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    for path in files:
        rel = path.relative_to(artifact_dir)
        size_mb = path.stat().st_size / (1024 * 1024)
        suffix = path.suffix.lower()
        is_umap_image = suffix in IMAGE_SUFFIXES and "umap" in path.stem.lower()
        with st.expander(f"{rel} ({size_mb:.2f} MB)", expanded=is_umap_image):
            if suffix in IMAGE_SUFFIXES:
                _render_image_preview(path, max_preview_mb=max_inline_mb)
            else:
                render_download_button(path, f"Download artifact {rel}")
                if suffix in TEXT_SUFFIXES and size_mb <= max_inline_mb:
                    st.code(
                        path.read_text(encoding="utf-8", errors="replace")[:20000],
                        language=None,
                    )
                elif size_mb > max_inline_mb:
                    st.caption(
                        f"Preview skipped because the file is larger than {max_inline_mb:g} MB."
                    )
