"""Migrate legacy data folders into the standardized Multi-verse store layout.

Layout: ``<dest>/datasets/<slug>/data/`` with a ``dataset.yaml`` manifest beside ``data/``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

RAW_EXTENSIONS = (".h5ad", ".h5mu")

# Filename keyword -> modality key in dataset.yaml
_MODALITY_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("rna", "rna"),
    ("gene", "rna"),
    ("gex", "rna"),
    ("scrna", "rna"),
    ("atac", "atac"),
    ("peak", "atac"),
    ("adt", "adt"),
    ("protein", "adt"),
    ("cite", "adt"),
    ("hashing", "adt"),
)


@dataclass
class MigrationCandidate:
    """A source directory that contains at least one raw data file."""

    source_dir: Path
    rel_path: Path  # relative to migration root, for naming / reporting
    data_files: List[Path] = field(default_factory=list)  # direct children only


@dataclass
class MigrationResult:
    slug: str
    dest_dataset_dir: Path
    success: bool
    message: str = ""


def slugify_fs_safe(name: str) -> str:
    """Produce a filesystem-safe slug: lowercase, spaces to hyphens, safe charset."""
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._\-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "dataset"
    if len(s) > 200:
        s = s[:200].rstrip("-")
    return s


def _list_raw_files_in_dir(dir_path: Path) -> List[Path]:
    """Return direct-child files with raw extensions (sorted for stable output)."""
    if not dir_path.is_dir():
        return []
    out: List[Path] = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS:
            out.append(p)
    return sorted(out)


def iter_candidate_directories(root: Path) -> List[MigrationCandidate]:
    """Recursively find directories that directly contain at least one .h5ad / .h5mu file."""
    root = root.resolve()
    candidates: List[MigrationCandidate] = []
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=True, followlinks=False):
        p = Path(dirpath)
        files = _list_raw_files_in_dir(p)
        if not files:
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = Path(".")
        candidates.append(
            MigrationCandidate(source_dir=p, rel_path=rel, data_files=files)
        )
    return sorted(candidates, key=lambda c: str(c.rel_path))


def _guess_modality_from_stem(stem: str) -> Optional[str]:
    lower = stem.lower()
    for needle, mod in _MODALITY_KEYWORDS:
        if needle in lower:
            return mod
    return None


def _read_h5mu_modalities(path: Path) -> List[str]:
    """Return modality keys from a MuData file (lightweight read when possible)."""
    try:
        import mudata as md

        try:
            m = md.read_h5mu(path, backed=True)
        except TypeError:
            m = md.read_h5mu(path, backed="r")
        try:
            return sorted(m.mod.keys())
        finally:
            f = getattr(m, "file", None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            close = getattr(m, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        import h5py

        with h5py.File(path, "r") as hf:
            if "mod" in hf:
                return sorted(hf["mod"].keys())
    except Exception as exc:
        raise RuntimeError(f"Could not read modalities from {path}: {exc}") from exc

    raise RuntimeError(f"No 'mod' group found in {path}")


def infer_modalities_and_mapping(
    files: Sequence[Path],
    *,
    prefer_auto: bool,
) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    """Return (omics list, raw_files map relative to dataset dir, error message).

    ``raw_files`` keys are modality names; values are paths under the dataset folder
    like ``data/filename.h5ad``.
    """
    if not files:
        return [], {}, "No .h5ad or .h5mu files in directory."

    if len(files) == 1:
        f = files[0]
        suffix = f.suffix.lower()
        dest_name = f.name
        rel = f"data/{dest_name}"
        if suffix == ".h5ad":
            return ["rna"], {"rna": rel}, None
        if suffix == ".h5mu":
            try:
                mods = _read_h5mu_modalities(f)
            except Exception as exc:  # noqa: BLE001 — surface as migration warning
                if prefer_auto:
                    return ["rna", "atac"], {"rna": rel, "atac": rel}, f"h5mu modality read failed ({exc}); using placeholder omics."
                return [], {}, str(exc)
            if not mods:
                return ["rna"], {"rna": rel}, "h5mu has no modalities; defaulting to rna."
            return mods, {m: rel for m in mods}, None

    # Multiple files: map by filename heuristic; collision resolution
    raw_files: Dict[str, str] = {}
    unmapped: List[Path] = []

    for f in sorted(files, key=lambda p: p.name.lower()):
        stem = f.stem
        mod = _guess_modality_from_stem(stem)
        if mod is None:
            unmapped.append(f)
            continue
        rel = f"data/{f.name}"
        key = mod
        n = 2
        while key in raw_files:
            key = f"{mod}_{n}"
            n += 1
        raw_files[key] = rel

    if unmapped and not prefer_auto:
        return [], {}, (
            "Could not map files to modalities by filename; "
            f"unmapped: {[p.name for p in unmapped]}"
        )

    # Auto: assign remaining keys file_1, file_2, ...
    n = 0
    for f in unmapped:
        rel = f"data/{f.name}"
        n += 1
        key = f"file_{n}"
        while key in raw_files:
            n += 1
            key = f"file_{n}"
        raw_files[key] = rel

    omics = sorted(raw_files.keys())
    return omics, raw_files, None


def build_dataset_yaml_dict(
    display_name: str,
    omics: List[str],
    raw_files: Dict[str, str],
    metadata_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build the manifest dict matching docs/DEVELOPER_GUIDE.md."""
    meta = metadata_keys or {
        "batch": "batch",
        "cell_type": "cell_type",
    }
    return {
        "name": display_name,
        "omics": omics,
        "raw_files": dict(sorted(raw_files.items())),
        "metadata_keys": meta,
    }


def dump_dataset_yaml(data: Dict[str, Any]) -> str:
    """Serialize with a stable, readable layout."""
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def validate_dataset_yaml_content(
    content: Dict[str, Any],
    dataset_root: Path,
) -> List[str]:
    """Return a list of validation errors; empty if OK."""
    errors: List[str] = []
    if not isinstance(content, dict):
        return ["Root must be a mapping."]
    for key in ("name", "omics", "raw_files"):
        if key not in content:
            errors.append(f"Missing key: {key}")
    if errors:
        return errors
    if not isinstance(content["name"], str):
        errors.append("'name' must be a string.")
    if not isinstance(content["omics"], list) or not all(
        isinstance(x, str) for x in content["omics"]
    ):
        errors.append("'omics' must be a list of strings.")
    if not isinstance(content["raw_files"], dict):
        errors.append("'raw_files' must be a mapping.")
    else:
        for mod, rel in content["raw_files"].items():
            if not isinstance(rel, str):
                errors.append(f"raw_files[{mod!r}] must be a string path.")
                continue
            p = dataset_root / rel
            if not p.exists():
                errors.append(f"Referenced path does not exist: {rel}")
    if "metadata_keys" in content and content["metadata_keys"] is not None:
        if not isinstance(content["metadata_keys"], dict):
            errors.append("'metadata_keys' must be a mapping or omitted.")
    return errors


def _unique_slug(base: str, used: set[str]) -> str:
    s = slugify_fs_safe(base)
    if s not in used:
        used.add(s)
        return s
    n = 2
    while True:
        cand = f"{s}-{n}"
        if cand not in used:
            used.add(cand)
            return cand
        n += 1


def _total_size(paths: Iterable[Path]) -> int:
    total = 0
    for p in paths:
        if p.is_file():
            total += p.stat().st_size
    return total


def _ensure_disk_space(dest_root: Path, needed_bytes: int) -> Tuple[bool, str]:
    check_path = dest_root if dest_root.exists() else dest_root.parent
    if not check_path.exists():
        check_path = Path.cwd()
    usage = shutil.disk_usage(check_path)
    if usage.free < needed_bytes:
        need_gb = needed_bytes / (1024**3)
        free_gb = usage.free / (1024**3)
        return False, f"Need ~{need_gb:.2f} GiB free; only {free_gb:.2f} GiB available on {dest_root}."
    return True, ""


def _check_access_read(path: Path) -> bool:
    return os.access(path, os.R_OK) and path.is_dir()


def _check_access_write(path: Path) -> bool:
    return os.access(path, os.W_OK)


def prompt_mapping_if_needed(
    files: Sequence[Path],
    existing_infer: Tuple[List[str], Dict[str, str], Optional[str]],
) -> Tuple[List[str], Dict[str, str]]:
    """Interactive: let user confirm or type modality per file. Returns omics, raw_files."""
    omics, raw_files, err = existing_infer
    if not err and raw_files:
        print("\nDetected layout:")
        for k, v in sorted(raw_files.items()):
            print(f"  {k} -> {v}")
        yn = input("Use this mapping? [Y/n]: ").strip().lower()
        if yn in ("", "y", "yes"):
            return omics, raw_files

    print("\nFiles in directory:")
    for i, p in enumerate(files):
        print(f"  [{i}] {p.name}")

    raw_files: Dict[str, str] = {}
    for p in files:
        default = _guess_modality_from_stem(p.stem) or "rna"
        mod = input(f"Modality key for {p.name!r} [{default}]: ").strip() or default
        key = mod
        n = 2
        while key in raw_files:
            key = f"{mod}_{n}"
            n += 1
        raw_files[key] = f"data/{p.name}"

    return sorted(raw_files.keys()), raw_files


def migrate_one(
    candidate: MigrationCandidate,
    dest_store: Path,
    *,
    dry_run: bool,
    mode: str,
    interactive: bool,
    auto: bool,
    used_slugs: set[str],
) -> MigrationResult:
    """Migrate a single candidate directory."""
    root_display = candidate.rel_path.as_posix()
    if root_display == ".":
        slug_base = candidate.source_dir.name
    else:
        slug_base = root_display.replace("/", "-")

    slug = _unique_slug(slug_base, used_slugs)
    dest_dataset = dest_store / "datasets" / slug
    dest_data = dest_dataset / "data"

    infer = infer_modalities_and_mapping(candidate.data_files, prefer_auto=auto)

    if interactive:
        omics, raw_files = prompt_mapping_if_needed(candidate.data_files, infer)
        warn = None
    else:
        omics, raw_files, warn = infer

    if not omics or not raw_files:
        msg = infer[2] or "Could not determine modalities."
        return MigrationResult(
            slug=slug,
            dest_dataset_dir=dest_dataset,
            success=False,
            message=msg,
        )

    display_name = root_display if root_display != "." else candidate.source_dir.name
    yaml_dict = build_dataset_yaml_dict(display_name.replace("-", " ").replace("/", " / "), omics, raw_files)
    yaml_text = dump_dataset_yaml(yaml_dict)

    if dry_run:
        print(f"\n[DRY-RUN] Would create: {dest_dataset}")
        print(yaml_text)
        if warn:
            print(f"Note: {warn}")
        return MigrationResult(
            slug=slug,
            dest_dataset_dir=dest_dataset,
            success=True,
            message=warn or "",
        )

    dest_data.mkdir(parents=True, exist_ok=True)
    for src in candidate.data_files:
        target = dest_data / src.name
        if mode == "copy":
            shutil.copy2(src, target)
        elif mode == "move":
            shutil.move(str(src), str(target))
        elif mode == "symlink":
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(src.resolve())
        else:
            raise ValueError(f"Unknown mode: {mode}")

    yaml_path = dest_dataset / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(yaml_text)

    loaded = yaml.safe_load(yaml_text)
    errs = validate_dataset_yaml_content(loaded, dest_dataset)
    if errs:
        return MigrationResult(
            slug=slug,
            dest_dataset_dir=dest_dataset,
            success=False,
            message="; ".join(errs),
        )

    msg = f"Migrated to {dest_dataset}"
    if warn:
        msg += f" ({warn})"
    return MigrationResult(slug=slug, dest_dataset_dir=dest_dataset, success=True, message=msg)


def run_migration(
    source: Path,
    dest_store: Path,
    *,
    dry_run: bool,
    mode: str,
    interactive: bool,
    auto: bool,
) -> int:
    source = source.resolve()
    dest_store = dest_store.resolve()

    if not _check_access_read(source):
        print(f"ERROR: Cannot read source directory: {source}", file=sys.stderr)
        return 1

    if not dry_run:
        dest_store.mkdir(parents=True, exist_ok=True)
        if not _check_access_write(dest_store):
            print(f"ERROR: Cannot write destination: {dest_store}", file=sys.stderr)
            return 1

    candidates = iter_candidate_directories(source)
    if not candidates:
        print(f"No directories under {source} contain .h5ad or .h5mu files.")
        return 0

    all_files = [f for c in candidates for f in c.data_files]
    total_bytes = _total_size(all_files)
    ok_space, space_msg = _ensure_disk_space(dest_store, total_bytes if mode != "symlink" else 1024)
    if not ok_space and not dry_run:
        print(f"ERROR: {space_msg}", file=sys.stderr)
        return 1

    used_slugs: set[str] = set()
    results: List[MigrationResult] = []
    failed_structure: List[str] = []

    for cand in candidates:
        if not dry_run and mode != "symlink":
            ok_space, space_msg = _ensure_disk_space(dest_store, _total_size(cand.data_files))
            if not ok_space:
                failed_structure.append(f"{cand.source_dir}: {space_msg}")
                continue

        res = migrate_one(
            cand,
            dest_store,
            dry_run=dry_run,
            mode=mode,
            interactive=interactive,
            auto=auto,
            used_slugs=used_slugs,
        )
        results.append(res)
        rel = cand.rel_path.as_posix()
        label = rel if rel != "." else "."
        if not res.success:
            failed_structure.append(f"{cand.source_dir}: {res.message}")
        elif res.message:
            print(f"[OK] {label} -> {res.slug}: {res.message}")
        else:
            print(f"[OK] {label} -> {res.slug}")

    print("\n=== Migration summary ===")
    print(f"Source: {source}")
    print(f"Destination store: {dest_store}")
    print(f"Dry run: {dry_run}")
    print(f"Mode: {mode}")
    ok_n = sum(1 for r in results if r.success)
    print(f"Succeeded: {ok_n} / {len(results)}")

    if failed_structure:
        print("\nDirectories that failed requirements or checks:")
        for line in failed_structure:
            print(f"  - {line}")
    else:
        print("\nNo failed directories.")

    return 0 if not failed_structure else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Migrate legacy folders to store/datasets/<slug>/data/ with dataset.yaml.",
    )
    p.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Root directory to scan for legacy data.",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Store root (will contain datasets/). Default: ./store (under the current working directory).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned dataset.yaml and paths without writing or copying.",
    )
    p.add_argument(
        "--mode",
        choices=("copy", "move", "symlink"),
        default="copy",
        help="How to place files into data/: copy (default), move, or symlink.",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Use heuristics only; do not prompt. On a TTY, prompts are used by default unless --auto is set.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    source: Path = args.source
    dest: Path
    if args.dest is not None:
        dest = args.dest
    else:
        dest = Path.cwd() / "store"

    auto = args.auto or not sys.stdin.isatty()
    interactive = sys.stdin.isatty() and not auto

    return run_migration(
        source,
        dest,
        dry_run=args.dry_run,
        mode=args.mode,
        interactive=interactive,
        auto=auto,
    )


if __name__ == "__main__":
    raise SystemExit(main())
