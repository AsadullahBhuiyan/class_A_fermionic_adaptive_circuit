from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "00_WORKSPACE" / "CURRENT" / "final_production_new_designs" / "_shared_src"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from drive_remote_commit import (  # noqa: E402
    DriveRemoteCommitter,
    RemoteCommitError,
    RemoteQuotaError,
)


class _Request:
    def __init__(self, callback):
        self.callback = callback

    def execute(self):
        return self.callback()


class _FakeFiles:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.next_id = 1
        self.create_failures = 0
        self.update_failures = 0
        self.force_bad_checksum = False

    def _id(self) -> str:
        value = f"file-{self.next_id}"
        self.next_id += 1
        return value

    def list(self, *, q: str, **_: Any):
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        name = re.search(r"name = '([^']*)'", q).group(1).replace("\\'", "'")
        return _Request(
            lambda: {
                "files": [
                    dict(item)
                    for item in self.items.values()
                    if item["name"] == name
                    and item.get("parents") == [parent]
                    and not item.get("trashed", False)
                ]
            }
        )

    def create(self, *, body: dict[str, Any], media_body=None, **_: Any):
        def run():
            if self.create_failures:
                self.create_failures -= 1
                raise OSError("simulated interrupted resumable upload")
            file_id = self._id()
            raw = b"" if media_body is None else Path(media_body).read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if self.force_bad_checksum and media_body is not None:
                sha = "0" * 64
            item = {
                "id": file_id,
                "name": body["name"],
                "parents": list(body.get("parents", ["root"])),
                "size": str(len(raw)),
                "sha256Checksum": sha,
                "md5Checksum": hashlib.md5(raw).hexdigest(),  # noqa: S324
                "trashed": False,
                "modifiedTime": "2026-08-31T00:00:00Z",
            }
            self.items[file_id] = item
            return dict(item)

        return _Request(run)

    def get(self, *, fileId: str, **_: Any):
        return _Request(lambda: dict(self.items[fileId]))

    def update(self, *, fileId: str, body: dict[str, Any], media_body=None, **_: Any):
        def run():
            if self.update_failures:
                self.update_failures -= 1
                raise OSError("simulated interrupted replacement upload")
            self.items[fileId].update(body)
            if media_body is not None:
                raw = Path(media_body).read_bytes()
                self.items[fileId].update({
                    "size": str(len(raw)),
                    "sha256Checksum": hashlib.sha256(raw).hexdigest(),
                    "md5Checksum": hashlib.md5(raw).hexdigest(),  # noqa: S324
                })
            return dict(self.items[fileId])

        return _Request(run)

    def delete(self, *, fileId: str):
        return _Request(lambda: self.items.pop(fileId) and {})


class _FakeAbout:
    def __init__(self, quota: dict[str, str]) -> None:
        self.quota = quota

    def get(self, **_: Any):
        return _Request(lambda: {"storageQuota": dict(self.quota)})


class _FakeService:
    def __init__(self, *, limit: int = 10_000_000, usage: int = 0) -> None:
        self.file_api = _FakeFiles()
        self.about_api = _FakeAbout({"limit": str(limit), "usage": str(usage)})

    def files(self):
        return self.file_api

    def about(self):
        return self.about_api


def _committer(tmp_path: Path, service: _FakeService) -> DriveRemoteCommitter:
    return DriveRemoteCommitter(
        service=service,
        drive_root=tmp_path,
        media_upload_factory=lambda path: str(path),
    )


def test_drivefs_visibility_is_not_durability(tmp_path: Path) -> None:
    visible = tmp_path / "campaign" / "shard.tar.gz"
    visible.parent.mkdir()
    visible.write_bytes(b"locally visible but absent from server")
    committer = _committer(tmp_path, _FakeService())
    with pytest.raises(RemoteCommitError, match="absent"):
        committer.verify_path(
            visible,
            expected_size=visible.stat().st_size,
            expected_sha256=hashlib.sha256(visible.read_bytes()).hexdigest(),
        )


def test_quota_exhaustion_stops_before_upload(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"123456")
    service = _FakeService(limit=100, usage=95)
    committer = _committer(tmp_path, service)
    with pytest.raises(RemoteQuotaError):
        committer.upload_verified(
            source, tmp_path / "out" / "source.bin", replace=False,
            required_headroom_bytes=1,
        )
    assert service.file_api.items == {}


def test_interrupted_resumable_upload_retries_and_server_verifies(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"durable payload")
    service = _FakeService()
    service.file_api.create_failures = 2
    committer = _committer(tmp_path, service)
    record = committer.upload_verified(
        source, tmp_path / "out" / "source.bin", replace=False
    )
    committer.verify_commit_record(record)
    assert record["remote_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_remote_checksum_mismatch_never_publishes_final_name(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    service = _FakeService()
    service.file_api.force_bad_checksum = True
    committer = _committer(tmp_path, service)
    with pytest.raises(RemoteCommitError, match="server verification failed"):
        committer.upload_verified(
            source, tmp_path / "out" / "source.bin", replace=False
        )
    assert all(item["name"] != "source.bin" for item in service.file_api.items.values())


def test_duplicate_remote_filename_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    service = _FakeService()
    committer = _committer(tmp_path, service)
    parent = committer.resolve_folder(("out",), create=True)
    for _ in range(2):
        request = service.file_api.create(
            body={"name": "source.bin", "parents": [parent]}, media_body=str(source)
        )
        request.execute()
    with pytest.raises(RemoteCommitError, match="ambiguous"):
        committer.upload_verified(
            source, tmp_path / "out" / "source.bin", replace=False
        )


def test_interrupted_pointer_replacement_preserves_previous_file(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    old.write_bytes(b"old-pointer")
    new = tmp_path / "new.json"
    new.write_bytes(b"new-pointer")
    service = _FakeService()
    committer = _committer(tmp_path, service)
    previous = committer.upload_verified(
        old, tmp_path / "out" / "latest.json", replace=False
    )
    service.file_api.update_failures = 3
    with pytest.raises(RemoteCommitError):
        committer.upload_verified(
            new, tmp_path / "out" / "latest.json", replace=True
        )
    committer.verify_commit_record(previous)
