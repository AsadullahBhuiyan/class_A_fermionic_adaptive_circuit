#!/usr/bin/env python3
"""Create and verify a source-only repository snapshot without changing the source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BLOCK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def git_paths(source: Path, *args: str) -> list[Path]:
    completed = subprocess.run(
        ["git", "-C", str(source), "ls-files", "-z", *args],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [Path(os.fsdecode(item)) for item in completed.stdout.split(b"\0") if item]


def safe_relative(path: Path) -> None:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe repository-relative path: {path}")


def expand_selected_directories(source: Path, selected: set[Path]) -> set[Path]:
    """Expand nested working trees without copying their private .git metadata."""
    expanded: set[Path] = set()
    ignored_parts = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    for relative in selected:
        source_path = source / relative
        if not source_path.is_dir() or source_path.is_symlink():
            expanded.add(relative)
            continue
        if (source_path / ".git").exists():
            nested = set(git_paths(source_path, "--cached")) | set(
                git_paths(source_path, "--others", "--exclude-standard")
            )
            expanded.update(relative / item for item in nested)
            continue
        expanded.update(
            path.relative_to(source)
            for path in source_path.rglob("*")
            if (path.is_file() or path.is_symlink())
            and ignored_parts.isdisjoint(path.relative_to(source_path).parts)
        )
    return expanded


def create_snapshot(source: Path, destination: Path, *, resume: bool = False) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if not (source / ".git").exists():
        raise ValueError(f"source is not a Git working tree: {source}")
    if destination == source or source in destination.parents:
        raise ValueError("destination must not be inside the source repository")
    if destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    if destination.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    if resume and destination.exists() and (destination / "CLEAN_SNAPSHOT_MANIFEST.json").exists():
        raise FileExistsError("refusing to resume a snapshot that already has a final manifest")

    tracked = set(git_paths(source, "--cached"))
    selected = sorted(
        expand_selected_directories(
            source, tracked | set(git_paths(source, "--others", "--exclude-standard"))
        )
    )
    missing_tracked = sorted(path.as_posix() for path in tracked if not (source / path).exists())

    destination.mkdir(parents=True, exist_ok=resume)
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    started = time.time()

    for index, relative in enumerate(selected, start=1):
        safe_relative(relative)
        source_path = source / relative
        if not source_path.exists() and not source_path.is_symlink():
            continue
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.is_symlink():
            target = os.readlink(source_path)
            if destination_path.exists() or destination_path.is_symlink():
                if not destination_path.is_symlink() or os.readlink(destination_path) != target:
                    raise FileExistsError(f"nonmatching existing snapshot path: {destination_path}")
            else:
                destination_path.symlink_to(target)
            if os.readlink(destination_path) != target:
                raise IOError(f"symlink verification failed: {relative}")
            row: dict[str, Any] = {
                "path": relative.as_posix(),
                "kind": "symlink",
                "target": target,
            }
        elif source_path.is_file():
            source_hash = sha256_file(source_path)
            if destination_path.exists() or destination_path.is_symlink():
                if not destination_path.is_file() or destination_path.is_symlink():
                    raise FileExistsError(f"nonmatching existing snapshot path: {destination_path}")
                if destination_path.stat().st_size != source_path.stat().st_size:
                    raise FileExistsError(f"nonmatching existing snapshot size: {destination_path}")
            else:
                shutil.copy2(source_path, destination_path)
            destination_hash = sha256_file(destination_path)
            if source_hash != destination_hash:
                raise FileExistsError(f"nonmatching existing snapshot checksum: {relative}")
            size = source_path.stat().st_size
            total_bytes += size
            row = {
                "path": relative.as_posix(),
                "kind": "file",
                "bytes": size,
                "sha256": source_hash,
            }
        else:
            raise ValueError(f"unsupported selected path type: {source_path}")
        rows.append(row)
        print(f"[copy {index}/{len(selected)}] {relative}", flush=True)

    payload = {
        "schema_version": 1,
        "kind": "clean_repository_snapshot",
        "source": str(source),
        "destination": str(destination),
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "selection": "tracked files present in the working tree plus untracked non-ignored files",
        "resumed_partial_destination": resume,
        "file_or_symlink_count": len(rows),
        "regular_file_bytes": total_bytes,
        "absent_tracked_paths": missing_tracked,
        "files": rows,
    }
    manifest_path = destination / "CLEAN_SNAPSHOT_MANIFEST.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify matching files in an incomplete destination and copy only missing paths",
    )
    args = parser.parse_args(argv)
    payload = create_snapshot(args.source, args.destination, resume=args.resume)
    print(
        json.dumps(
            {
                "destination": payload["destination"],
                "file_or_symlink_count": payload["file_or_symlink_count"],
                "regular_file_bytes": payload["regular_file_bytes"],
                "absent_tracked_path_count": len(payload["absent_tracked_paths"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
