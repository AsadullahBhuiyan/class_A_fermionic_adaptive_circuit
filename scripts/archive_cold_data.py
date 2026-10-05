#!/usr/bin/env python3
"""Checksum, copy, and verify selected large datasets without deleting sources.

The historical ``cold_data_*`` schema names are retained for compatibility with
existing manifests; they do not by themselves classify a dataset as disposable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import string
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing temporary output: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_files(source: Path) -> list[Path]:
    if not source.is_dir():
        raise NotADirectoryError(source)
    ignored_parts = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ignored_parts.isdisjoint(path.relative_to(source).parts)
    )


def build_manifest(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    files = source_files(source)
    rows = []
    total = 0
    started = time.time()
    for index, path in enumerate(files, start=1):
        stat = path.stat()
        size = int(stat.st_size)
        digest = sha256_file(path)
        total += size
        rows.append(
            {
                "path": path.relative_to(source).as_posix(),
                "bytes": size,
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": digest,
            }
        )
        print(f"[hash {index}/{len(files)}] {size} {path}", flush=True)
    payload = {
        "schema_version": 1,
        "kind": "cold_data_source_manifest",
        "source": str(source),
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "file_count": len(rows),
        "total_bytes": total,
        "files": rows,
    }
    atomic_json(output, payload)
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("kind") != "cold_data_source_manifest":
        raise ValueError(f"unsupported manifest: {path}")
    return payload


def manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate manifest structure and return traversal-safe file rows."""
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != int(manifest.get("file_count", -1)):
        raise ValueError("manifest file_count does not match its files array")
    seen: set[str] = set()
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manifest file row is not an object")
        relative = Path(str(row.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate manifest path: {normalized}")
        seen.add(normalized)
        size = int(row.get("bytes", -1))
        digest = str(row.get("sha256", ""))
        if size < 0 or len(digest) != 64 or any(char not in string.hexdigits for char in digest):
            raise ValueError(f"invalid manifest metadata: {normalized}")
        total += size
    if total != int(manifest.get("total_bytes", -1)):
        raise ValueError("manifest total_bytes does not match its file rows")
    return rows


def contained_path(root: Path, relative: Path) -> Path:
    """Resolve an archive path and reject traversal through an existing symlink parent."""
    candidate = root / relative
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes archive root: {relative}")
    return candidate


def verify_source_snapshot(manifest_path: Path) -> dict[str, int]:
    """Require the current source inventory to match manifest path/size/mtime exactly."""
    manifest = load_manifest(manifest_path)
    rows = manifest_rows(manifest)
    source = Path(manifest["source"]).resolve()
    expected = {str(row["path"]): row for row in rows}
    actual = {path.relative_to(source).as_posix(): path for path in source_files(source)}
    errors: list[str] = []
    for relative in sorted(set(expected) - set(actual)):
        errors.append(f"missing:{relative}")
    for relative in sorted(set(actual) - set(expected)):
        errors.append(f"unexpected:{relative}")
    for relative in sorted(set(expected) & set(actual)):
        stat = actual[relative].stat()
        row = expected[relative]
        if stat.st_size != int(row["bytes"]):
            errors.append(f"size:{relative}")
        elif stat.st_mtime_ns != int(row["mtime_ns"]):
            errors.append(f"mtime:{relative}")
    if errors:
        preview = ", ".join(errors[:10])
        raise IOError(
            f"source no longer matches archive manifest {manifest_path}; "
            f"errors={len(errors)}; first={preview}"
        )
    return {
        "verified_file_count": len(rows),
        "verified_bytes": int(manifest["total_bytes"]),
    }


def copy_manifest(manifest_path: Path, destination_root: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    rows = manifest_rows(manifest)
    source = Path(manifest["source"]).resolve()
    destination_root = destination_root.resolve()
    if destination_root == source or source in destination_root.parents:
        raise ValueError("destination must not be inside the source dataset")
    destination_root.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for index, row in enumerate(rows, start=1):
        relative = Path(row["path"])
        source_path = source / relative
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(f"manifest source is missing or not a regular file: {source_path}")
        destination = contained_path(destination_root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = contained_path(destination_root, relative)
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() and destination.is_file() and destination.stat().st_size == int(row["bytes"]):
                if sha256_file(destination) == row["sha256"]:
                    skipped += 1
                    print(f"[verified existing {index}/{manifest['file_count']}] {destination}", flush=True)
                    continue
            raise FileExistsError(f"refusing to overwrite nonmatching destination: {destination}")
        temporary = destination.with_suffix(destination.suffix + f".partial.{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"refusing to remove or overwrite partial copy: {temporary}")
        shutil.copy2(source_path, temporary)
        if sha256_file(temporary) != row["sha256"]:
            raise IOError(f"checksum mismatch after copy: {source_path}")
        os.replace(temporary, destination)
        copied += 1
        print(f"[copied {index}/{manifest['file_count']}] {destination}", flush=True)
    return {"copied": copied, "verified_existing": skipped, "destination": str(destination_root)}


def verify_manifest(manifest_path: Path, destination_root: Path, receipt: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    rows = manifest_rows(manifest)
    destination_root = destination_root.resolve()
    if not destination_root.is_dir():
        raise NotADirectoryError(destination_root)
    errors = []
    checked_bytes = 0
    verified_count = 0
    expected = {str(row["path"]) for row in rows}
    for index, row in enumerate(rows, start=1):
        destination = contained_path(destination_root, Path(row["path"]))
        if destination.is_symlink():
            errors.append({"path": row["path"], "error": "symlink_not_regular_file"})
            continue
        if not destination.is_file():
            errors.append({"path": row["path"], "error": "missing"})
            continue
        if destination.stat().st_size != int(row["bytes"]):
            errors.append({"path": row["path"], "error": "size_mismatch"})
            continue
        digest = sha256_file(destination)
        if digest != row["sha256"]:
            errors.append({"path": row["path"], "error": "sha256_mismatch", "actual": digest})
            continue
        checked_bytes += int(row["bytes"])
        verified_count += 1
        print(f"[verify {index}/{manifest['file_count']}] {destination}", flush=True)
    for path in destination_root.rglob("*"):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative = path.relative_to(destination_root).as_posix()
        if relative not in expected:
            errors.append({"path": relative, "error": "unexpected_destination_entry"})
    payload = {
        "schema_version": 1,
        "kind": "cold_data_archive_receipt",
        "status": "verified" if not errors else "failed",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source": manifest["source"],
        "destination": str(destination_root),
        "verified_unix": time.time(),
        "verified_file_count": verified_count,
        "verified_bytes": checked_bytes,
        "errors": errors,
        "source_deletion_authorized": False,
    }
    atomic_json(receipt, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--source", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    copy = subparsers.add_parser("copy")
    copy.add_argument("--manifest", type=Path, required=True)
    copy.add_argument("--destination", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--destination", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "manifest":
        payload = build_manifest(args.source, args.output)
        print(json.dumps({key: payload[key] for key in ("file_count", "total_bytes", "elapsed_seconds")}, indent=2))
        return 0
    if args.command == "copy":
        print(json.dumps(copy_manifest(args.manifest, args.destination), indent=2))
        return 0
    payload = verify_manifest(args.manifest, args.destination, args.receipt)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "verified" else 2


if __name__ == "__main__":
    sys.exit(main())
