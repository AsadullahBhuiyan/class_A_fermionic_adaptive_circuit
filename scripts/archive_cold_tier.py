#!/usr/bin/env python3
"""Copy and independently verify selected large-result archive candidates.

Colab experiment packages and their generated data are intentionally excluded.  Their
data stays grouped with the notebooks and runners that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from archive_cold_data import (
    copy_manifest,
    load_manifest,
    manifest_rows,
    sha256_file,
    verify_manifest,
    verify_source_snapshot,
)


REPOSITORY = Path(__file__).resolve().parent.parent
MANIFEST_DIRECTORY = REPOSITORY / "PROJECT_ADMIN" / "archive_manifests"
ARCHIVE_NAME = "class_A_fermionic_adaptive_circuit_large_results_20260817"
DATASETS = {
    "cache_G_history_samples.json": Path("cache/G_history_samples"),
    "choi_covariance_cpu_data.json": Path(
        "00_WORKSPACE/LARGE_RESULTS/choi_covariance_cpu/cpu_data"
    ),
    "lyapunov_analysis_v2_cache.json": Path(
        "00_WORKSPACE/LARGE_RESULTS/lyapunov_analysis_v2/cache"
    ),
    "dw_convergence.json": Path("00_WORKSPACE/LARGE_RESULTS/dw_convergence"),
    "experiments.json": Path("00_WORKSPACE/LARGE_RESULTS/experiments"),
}
MINIMUM_FREE_RESERVE_BYTES = 10 * 1024**3
FREE_RESERVE_FRACTION = 0.02


def expected_archive_bytes() -> int:
    total = 0
    for manifest_name in DATASETS:
        payload = load_manifest(MANIFEST_DIRECTORY / manifest_name)
        manifest_rows(payload)
        total += int(payload["total_bytes"])
    return total


def safe_archive_subdirectory(archive_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive subdirectory: {relative}")
    archive_root = archive_root.resolve()
    candidate = archive_root / relative
    current = archive_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"archive subdirectory traverses a symlink: {current}")
    resolved = candidate.resolve()
    if resolved != archive_root and archive_root not in resolved.parents:
        raise ValueError(f"archive subdirectory escapes archive root: {candidate}")
    return candidate


def existing_expected_bytes(archive_root: Path) -> int:
    """Count size-matching expected files already occupying the resumable archive."""
    total = 0
    for manifest_name, relative_destination in DATASETS.items():
        payload = load_manifest(MANIFEST_DIRECTORY / manifest_name)
        dataset_root = safe_archive_subdirectory(archive_root, relative_destination)
        for row in manifest_rows(payload):
            path = dataset_root / row["path"]
            if path.is_file() and not path.is_symlink() and path.stat().st_size == int(row["bytes"]):
                total += int(row["bytes"])
    return total


def external_destination(path: Path, required_new_bytes: int) -> Path:
    destination = path.resolve()
    if not destination.is_dir():
        raise NotADirectoryError(
            f"destination root must already exist so its mount can be checked: {destination}"
        )
    if not os.access(destination, os.W_OK | os.X_OK):
        raise PermissionError(f"destination root is not writable: {destination}")
    if destination.stat().st_dev == REPOSITORY.stat().st_dev:
        raise ValueError("destination must be on a different filesystem from the repository")
    free = shutil.disk_usage(destination).free
    reserve = max(MINIMUM_FREE_RESERVE_BYTES, int(expected_archive_bytes() * FREE_RESERVE_FRACTION))
    if free < required_new_bytes + reserve:
        raise OSError(
            "insufficient archive space: "
            f"free={free}, required_new={required_new_bytes}, reserve={reserve}"
        )
    return destination


def load_verified_receipt(receipt: Path, manifest: Path, destination: Path) -> dict[str, object]:
    """Validate a prior immutable receipt so an interrupted tier run can resume."""
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    source_manifest_sha256 = sha256_file(manifest)
    required = {
        "schema_version": 1,
        "kind": "cold_data_archive_receipt",
        "status": "verified",
        "source_manifest_sha256": source_manifest_sha256,
        "destination": str(destination.resolve()),
        "source_deletion_authorized": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("verified_file_count") != manifest_payload.get("file_count"):
        mismatches["verified_file_count"] = {
            "expected": manifest_payload.get("file_count"),
            "actual": payload.get("verified_file_count"),
        }
    if payload.get("verified_bytes") != manifest_payload.get("total_bytes"):
        mismatches["verified_bytes"] = {
            "expected": manifest_payload.get("total_bytes"),
            "actual": payload.get("verified_bytes"),
        }
    if payload.get("errors") != []:
        mismatches["errors"] = {"expected": [], "actual": payload.get("errors")}
    if not destination.is_dir():
        mismatches["destination_exists"] = {"expected": True, "actual": False}
    if mismatches:
        raise ValueError(f"existing archive receipt is not reusable: {receipt}: {mismatches}")
    return payload


def run(destination_root: Path) -> dict[str, object]:
    unresolved_root = destination_root
    if not unresolved_root.is_dir():
        raise NotADirectoryError(unresolved_root)
    preliminary_root = unresolved_root.resolve() / ARCHIVE_NAME
    if preliminary_root.is_symlink():
        raise ValueError(f"archive root must not be a symlink: {preliminary_root}")
    for manifest_name in DATASETS:
        verify_source_snapshot(MANIFEST_DIRECTORY / manifest_name)
    total_bytes = expected_archive_bytes()
    reusable_bytes = existing_expected_bytes(preliminary_root) if preliminary_root.is_dir() else 0
    destination_root = external_destination(
        unresolved_root, required_new_bytes=max(0, total_bytes - reusable_bytes)
    )
    archive_root = destination_root / ARCHIVE_NAME
    if archive_root.exists() and (not archive_root.is_dir() or archive_root.is_symlink()):
        raise ValueError(f"archive root is not a safe directory: {archive_root}")
    receipt_directory = MANIFEST_DIRECTORY / "receipts"
    results = []

    for manifest_name, relative_destination in DATASETS.items():
        manifest = MANIFEST_DIRECTORY / manifest_name
        destination = safe_archive_subdirectory(archive_root, relative_destination)
        receipt = receipt_directory / f"{manifest.stem}.verified.json"
        if receipt.exists() or receipt.is_symlink():
            prior = load_verified_receipt(receipt, manifest, destination)
            print(f"\n=== reuse verified receipt {receipt.name} ===", flush=True)
            results.append(
                {
                    "manifest": str(manifest),
                    "manifest_sha256": sha256_file(manifest),
                    "destination": str(destination),
                    "receipt": str(receipt),
                    "copied": 0,
                    "verified_existing": int(prior["verified_file_count"]),
                    "verified_file_count": int(prior["verified_file_count"]),
                    "verified_bytes": int(prior["verified_bytes"]),
                    "receipt_reused": True,
                    "source_deletion_authorized": False,
                }
            )
            continue
        print(f"\n=== copy {manifest_name} -> {destination} ===", flush=True)
        copy_result = copy_manifest(manifest, destination)
        print(f"\n=== independent verification {manifest_name} ===", flush=True)
        verify_result = verify_manifest(manifest, destination, receipt)
        if verify_result["status"] != "verified":
            raise IOError(f"archive verification failed: {manifest_name}")
        results.append(
            {
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "destination": str(destination),
                "receipt": str(receipt),
                "copied": copy_result["copied"],
                "verified_existing": copy_result["verified_existing"],
                "verified_file_count": verify_result["verified_file_count"],
                "verified_bytes": verify_result["verified_bytes"],
                "receipt_reused": False,
                "source_deletion_authorized": False,
            }
        )

    return {
        "archive_root": str(archive_root),
        "dataset_count": len(results),
        "verified_bytes": sum(int(row["verified_bytes"]) for row in results),
        "source_deletion_authorized": False,
        "datasets": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.destination_root), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
