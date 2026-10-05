from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY / "scripts"))

from archive_cold_data import (  # noqa: E402
    build_manifest,
    copy_manifest,
    verify_manifest,
    verify_source_snapshot,
)
import archive_cold_tier  # noqa: E402
from archive_cold_tier import load_verified_receipt  # noqa: E402


def make_source(root: Path) -> Path:
    source = root / "source"
    (source / "nested").mkdir(parents=True)
    (source / "a.bin").write_bytes(b"alpha")
    (source / "nested/b.bin").write_bytes(b"beta")
    return source


def test_copy_verify_and_receipt_resume(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    manifest = tmp_path / "manifest.json"
    destination = tmp_path / "archive"
    receipt = tmp_path / "receipt.json"

    built = build_manifest(source, manifest)
    copied = copy_manifest(manifest, destination)
    verified = verify_manifest(manifest, destination, receipt)

    assert copied == {"copied": 2, "verified_existing": 0, "destination": str(destination)}
    assert verified["status"] == "verified"
    assert verified["verified_file_count"] == 2
    assert verified["verified_bytes"] == built["total_bytes"]
    assert load_verified_receipt(receipt, manifest, destination)["status"] == "verified"


def test_verification_rejects_unexpected_destination_file(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    manifest = tmp_path / "manifest.json"
    destination = tmp_path / "archive"
    receipt = tmp_path / "failed-receipt.json"
    build_manifest(source, manifest)
    copy_manifest(manifest, destination)
    (destination / "unexpected.bin").write_bytes(b"not in manifest")

    verified = verify_manifest(manifest, destination, receipt)

    assert verified["status"] == "failed"
    assert {row["error"] for row in verified["errors"]} == {"unexpected_destination_entry"}
    with pytest.raises(ValueError, match="not reusable"):
        load_verified_receipt(receipt, manifest, destination)


def test_copy_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    manifest = tmp_path / "manifest.json"
    build_manifest(source, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "../escape.bin"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe manifest path"):
        copy_manifest(manifest, tmp_path / "archive")


def test_external_destination_rejects_insufficient_free_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_device = archive_cold_tier.REPOSITORY.stat().st_dev
    original_stat = Path.stat

    def pretend_external_stat(path: Path, *args: object, **kwargs: object):
        result = original_stat(path, *args, **kwargs)
        if str(path) == str(tmp_path):
            values = list(result)
            values[2] = repository_device + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", pretend_external_stat)
    monkeypatch.setattr(
        archive_cold_tier.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=50, free=50),
    )

    with pytest.raises(OSError, match="insufficient archive space"):
        archive_cold_tier.external_destination(tmp_path, required_new_bytes=60)


def test_archive_subdirectory_rejects_symlink_parent(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    outside = tmp_path / "outside"
    archive_root.mkdir()
    outside.mkdir()
    (archive_root / "cache").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="traverses a symlink"):
        archive_cold_tier.safe_archive_subdirectory(
            archive_root, Path("cache/G_history_samples")
        )


def test_source_snapshot_rejects_post_manifest_change(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    manifest = tmp_path / "manifest.json"
    build_manifest(source, manifest)
    assert verify_source_snapshot(manifest)["verified_file_count"] == 2
    (source / "a.bin").write_bytes(b"changed after manifest")

    with pytest.raises(OSError, match="source no longer matches"):
        verify_source_snapshot(manifest)
