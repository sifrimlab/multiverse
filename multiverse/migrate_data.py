"""Migrate legacy data folders into ``store/datasets/<slug>/data/``.

Inference is delegated to ``DatasetHeuristics``; this module performs only IO,
reporting, and migration safety checks.
"""

from __future__ import annotations

import argparse
import errno
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from .guesser import DatasetHeuristics, is_raw_like_file

console = Console()


@dataclass
class MigrationResult:
    source_dir: Path
    dest_dataset_dir: Path
    status: str  # migrated | skipped | verify | failed
    message: str
    files: List[Path] = field(default_factory=list)


def slugify_fs_safe(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._\-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "dataset"


def _slugify_relpath(rel_path: Path, fallback: str) -> str:
    parts = [slugify_fs_safe(part) for part in rel_path.parts if part not in (".", "")]
    if not parts:
        return slugify_fs_safe(fallback)
    return "-".join(parts)


def _list_raw_files_in_dir(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted([p for p in directory.iterdir() if is_raw_like_file(p)])


def iter_candidate_directories(source: Path) -> List[Path]:
    source = source.resolve()
    out: List[Path] = []
    for dirpath, _dirnames, _filenames in os.walk(source, followlinks=False):
        p = Path(dirpath)
        if _list_raw_files_in_dir(p):
            out.append(p)
    return sorted(out)


def _total_size(paths: Sequence[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())


def _ensure_disk_space(dest_root: Path, needed_bytes: int) -> tuple[bool, str]:
    check_path = dest_root if dest_root.exists() else dest_root.parent
    usage = shutil.disk_usage(check_path)
    if usage.free < needed_bytes:
        return (
            False,
            f"Insufficient free space: need {needed_bytes / (1024**3):.2f} GiB, "
            f"have {usage.free / (1024**3):.2f} GiB.",
        )
    return True, ""


def safe_copy_file(src: Path, dst: Path) -> str:
    """Try hard-link first; fallback to metadata-preserving copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as exc:
        if exc.errno in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EMLINK}:
            shutil.copy2(src, dst)
            return "copy2"
        raise


def _render_yaml_with_notes(manifest: Dict[str, Any]) -> str:
    notes = manifest.get("guesser_notes", {}) or {}
    peek = notes.get("peek", {}) or {}
    lines: List[str] = []

    batch_alts = peek.get("batch_key_alternatives") or []
    cell_alts = peek.get("cell_type_key_alternatives") or []
    if batch_alts or cell_alts:
        lines.append("# Heuristic alternatives found; verify metadata_keys below.")
        if batch_alts:
            lines.append(f"# batch alternatives: {', '.join(batch_alts)}")
        if cell_alts:
            lines.append(f"# cell_type alternatives: {', '.join(cell_alts)}")
        lines.append("")

    yaml_ready = {k: v for k, v in manifest.items() if k != "guesser_notes"}
    lines.append(
        yaml.safe_dump(
            yaml_ready,
            sort_keys=False,
            default_flow_style=False,
            width=10000,
        ).rstrip()
    )
    return "\n".join(lines) + "\n"


def _validate_manifest(manifest: Dict[str, Any]) -> Optional[str]:
    if not manifest.get("raw_files"):
        return "No raw files inferred by DatasetHeuristics."
    if not manifest.get("omics"):
        return "No omics inferred by DatasetHeuristics."
    return None


def _already_exists_guard(dataset_dir: Path) -> Optional[str]:
    if (dataset_dir / "dataset.yaml").exists():
        return "dataset.yaml already exists (manual work protected)."
    return None


def _print_dry_run_preview(dest_dataset: Path, yaml_text: str, files: Sequence[Path]) -> None:
    tree = Tree(f"[bold cyan]{dest_dataset}[/bold cyan]")
    tree.add("[green]data/[/green]")
    for f in files:
        tree.children[0].add(f.name)
    tree.add("dataset.yaml")
    console.print(tree)
    console.print("[dim]--- dataset.yaml preview ---[/dim]")
    console.print(yaml_text.rstrip())


def migrate_one(
    source_root: Path,
    source_dir: Path,
    dest_store: Path,
    guesser: DatasetHeuristics,
    *,
    dry_run: bool,
) -> MigrationResult:
    rel = source_dir.resolve().relative_to(source_root.resolve())
    slug = _slugify_relpath(rel, fallback=source_dir.name)
    dest_dataset = dest_store / "datasets" / slug
    dest_data = dest_dataset / "data"

    try:
        manifest = guesser.generate_manifest(source_dir)
    except Exception as exc:  # noqa: BLE001
        return MigrationResult(source_dir, dest_dataset, "failed", f"heuristic failed: {exc}")

    validation_error = _validate_manifest(manifest)
    if validation_error:
        return MigrationResult(source_dir, dest_dataset, "failed", validation_error)

    exists_error = _already_exists_guard(dest_dataset)
    if exists_error:
        return MigrationResult(source_dir, dest_dataset, "skipped", exists_error)

    yaml_text = _render_yaml_with_notes(manifest)
    files = _list_raw_files_in_dir(source_dir)
    needs_verify = bool(
        (manifest.get("guesser_notes", {}).get("peek", {}).get("batch_key_alternatives"))
        or (manifest.get("guesser_notes", {}).get("peek", {}).get("cell_type_key_alternatives"))
    )

    if dry_run:
        _print_dry_run_preview(dest_dataset, yaml_text, files)
        return MigrationResult(
            source_dir,
            dest_dataset,
            "verify" if needs_verify else "migrated",
            "dry-run preview",
            files,
        )

    dest_data.mkdir(parents=True, exist_ok=True)
    link_stats: Dict[str, int] = {"hardlink": 0, "copy2": 0}
    for src in files:
        dst = dest_data / src.name
        if dst.exists():
            return MigrationResult(source_dir, dest_dataset, "failed", f"target exists: {dst}")
        method = safe_copy_file(src, dst)
        link_stats[method] = link_stats.get(method, 0) + 1

    with (dest_dataset / "dataset.yaml").open("w", encoding="utf-8") as fh:
        fh.write(yaml_text)

    msg = f"hardlink={link_stats.get('hardlink', 0)}, copy2={link_stats.get('copy2', 0)}"
    status = "verify" if needs_verify else "migrated"
    if needs_verify:
        msg += " (metadata alternatives detected)"
    return MigrationResult(source_dir, dest_dataset, status, msg, files)


def run_migration(source: Path, dest_store: Path, *, dry_run: bool) -> int:
    source = source.resolve()
    dest_store = dest_store.resolve()
    guesser = DatasetHeuristics()

    if not source.is_dir() or not os.access(source, os.R_OK):
        console.print(f"[red]ERROR:[/red] cannot read source directory: {source}")
        return 1

    if not dry_run:
        try:
            dest_store.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]ERROR:[/red] cannot create destination: {exc}")
            return 1
        if not os.access(dest_store, os.W_OK):
            console.print(f"[red]ERROR:[/red] destination not writable: {dest_store}")
            return 1

    candidates = iter_candidate_directories(source)
    if not candidates:
        console.print("[yellow]No candidate directories found.[/yellow]")
        return 0

    if not dry_run:
        required = _total_size([f for d in candidates for f in _list_raw_files_in_dir(d)])
        ok, msg = _ensure_disk_space(dest_store, required)
        if not ok:
            console.print(f"[red]ERROR:[/red] {msg}")
            return 1

    results: List[MigrationResult] = []
    for directory in candidates:
        results.append(migrate_one(source, directory, dest_store, guesser, dry_run=dry_run))

    summary = Table(title="Migration Summary")
    summary.add_column("Source", style="cyan")
    summary.add_column("Destination", style="blue")
    summary.add_column("Status")
    summary.add_column("Details")

    style_for = {
        "migrated": "[green]migrated[/green]",
        "verify": "[yellow]needs-verify[/yellow]",
        "skipped": "[blue]skipped[/blue]",
        "failed": "[red]failed[/red]",
    }

    for r in results:
        summary.add_row(
            str(r.source_dir),
            str(r.dest_dataset_dir),
            style_for.get(r.status, r.status),
            r.message,
        )
    console.print(summary)

    failed = [r for r in results if r.status == "failed"]
    verify = [r for r in results if r.status == "verify"]
    console.print(
        f"[bold]Totals:[/bold] migrated={sum(r.status=='migrated' for r in results)}, "
        f"verify={len(verify)}, skipped={sum(r.status=='skipped' for r in results)}, failed={len(failed)}"
    )
    return 1 if failed else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate legacy folders into store/datasets/<slug>/data/ using DatasetHeuristics.",
    )
    parser.add_argument("--source", required=True, type=Path, help="Legacy root directory.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path.cwd() / "store",
        help="Destination store root (contains datasets/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned tree + dataset.yaml without writing files.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_migration(args.source, args.dest, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
