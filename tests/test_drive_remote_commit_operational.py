from __future__ import annotations

import errno
import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_new_designs"
    / "drive_remote_commit.py"
)
SPEC = importlib.util.spec_from_file_location("classa_operational_drive", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
drive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drive)


class _Request:
    def __init__(self, callback: Any) -> None:
        self.callback = callback

    def execute(self) -> Any:
        return self.callback()


class _Files:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.next_id = 1
        self.fail_create_before_apply = 0
        self.lose_create_response_after_apply = 0
        self.lose_rename_response_after_apply = 0

    def _new_id(self) -> str:
        result = f"id-{self.next_id}"
        self.next_id += 1
        return result

    def list(self, *, q: str, **_: Any) -> _Request:
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        name = re.search(r"name = '([^']*)'", q).group(1).replace("\\'", "'")
        return _Request(
            lambda: {
                "files": [
                    dict(row)
                    for row in self.items.values()
                    if row["name"] == name
                    and row.get("parents") == [parent]
                    and not row.get("trashed", False)
                ]
            }
        )

    def create(
        self, *, body: dict[str, Any], media_body: Any = None, **_: Any
    ) -> _Request:
        def run() -> dict[str, Any]:
            if self.fail_create_before_apply:
                self.fail_create_before_apply -= 1
                raise OSError("transient before server apply")
            file_id = self._new_id()
            raw = b"" if media_body is None else Path(media_body).read_bytes()
            row = {
                "id": file_id,
                "name": body["name"],
                "parents": list(body.get("parents", ["root"])),
                "mimeType": body.get("mimeType", "application/octet-stream"),
                "size": str(len(raw)),
                "sha256Checksum": hashlib.sha256(raw).hexdigest(),
                "trashed": False,
                "modifiedTime": "2026-09-01T00:00:00Z",
            }
            self.items[file_id] = row
            if self.lose_create_response_after_apply:
                self.lose_create_response_after_apply -= 1
                raise OSError("response lost after server apply")
            return dict(row)

        return _Request(run)

    def get(self, *, fileId: str, **_: Any) -> _Request:
        def run() -> dict[str, Any]:
            if fileId not in self.items:
                raise OSError("404 not found")
            return dict(self.items[fileId])

        return _Request(run)

    def update(
        self,
        *,
        fileId: str,
        body: dict[str, Any],
        media_body: Any = None,
        **_: Any,
    ) -> _Request:
        def run() -> dict[str, Any]:
            row = self.items[fileId]
            row.update(body)
            if media_body is not None:
                raw = Path(media_body).read_bytes()
                row["size"] = str(len(raw))
                row["sha256Checksum"] = hashlib.sha256(raw).hexdigest()
            if self.lose_rename_response_after_apply:
                self.lose_rename_response_after_apply -= 1
                raise OSError("rename response lost after server apply")
            return dict(row)

        return _Request(run)

    def delete(self, *, fileId: str) -> _Request:
        return _Request(lambda: self.items.pop(fileId) and {})


class _About:
    def __init__(self, service: "_Service") -> None:
        self.service = service

    def get(self, **_: Any) -> _Request:
        return _Request(
            lambda: {
                "storageQuota": {
                    "limit": str(self.service.limit),
                    "usage": str(self.service.usage),
                }
            }
        )


class _Service:
    def __init__(self) -> None:
        self.file_api = _Files()
        self.limit = 10_000_000
        self.usage = 0
        self.about_api = _About(self)

    def files(self) -> _Files:
        return self.file_api

    def about(self) -> _About:
        return self.about_api


def _committer(tmp_path: Path, service: _Service) -> Any:
    return drive.DriveRemoteCommitter(
        service=service,
        drive_root=tmp_path,
        media_upload_factory=lambda path: str(path),
    )


def test_remote_paths_are_lexical_when_drivefs_stat_is_disconnected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _Service()
    committer = _committer(tmp_path, service)

    def disconnected_resolve(*_: Any, **__: Any) -> Path:
        raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")

    monkeypatch.setattr(Path, "resolve", disconnected_resolve)
    parts, name = committer.split_remote_path(tmp_path / "campaign" / "slot.json")
    assert parts == ("campaign",)
    assert name == "slot.json"


def test_exact_existing_commit_bypasses_exhausted_quota(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"already durable")
    service = _Service()
    committer = _committer(tmp_path, service)
    remote = tmp_path / "campaign" / "payload.bin"
    first = committer.upload_verified(source, remote, replace=False)
    service.usage = service.limit
    second = committer.upload_verified(
        source,
        remote,
        replace=False,
        required_headroom_bytes=service.limit,
    )
    assert second["remote_file_id"] == first["remote_file_id"]
    assert len([row for row in service.file_api.items.values() if row["name"] == remote.name]) == 1


def test_quota_failure_does_not_create_remote_folders(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"cannot fit")
    service = _Service()
    service.usage = service.limit
    committer = _committer(tmp_path, service)
    with pytest.raises(drive.RemoteQuotaError):
        committer.upload_verified(
            source,
            tmp_path / "new-campaign" / "payload.bin",
            replace=False,
            required_headroom_bytes=1,
        )
    assert service.file_api.items == {}


def test_create_applied_response_lost_is_reconciled_without_duplicate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"one server object")
    service = _Service()
    committer = _committer(tmp_path, service)
    # Folder creation is reconciled first, then exercise upload response loss.
    committer.resolve_folder(("campaign",), create=True)
    service.file_api.lose_create_response_after_apply = 1
    record = committer.upload_verified(
        source, tmp_path / "campaign" / "payload.bin", replace=False
    )
    committer.verify_record_for_path(
        record, tmp_path / "campaign" / "payload.bin"
    )
    named = [
        row
        for row in service.file_api.items.values()
        if row["name"] == "payload.bin"
    ]
    assert len(named) == 1


def test_rename_applied_response_lost_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"rename survives")
    service = _Service()
    committer = _committer(tmp_path, service)
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    committer.resolve_folder(("campaign",), create=True)
    service.file_api.lose_rename_response_after_apply = 6
    record = committer.upload_verified(
        source, tmp_path / "campaign" / "payload.bin", replace=False
    )
    assert record["remote_name"] == "payload.bin"
    assert committer.verify_record_for_path(
        record, tmp_path / "campaign" / "payload.bin"
    )["id"] == record["remote_file_id"]


def test_cross_parent_record_is_rejected_before_delete_or_accept(
    tmp_path: Path,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"bound parent")
    service = _Service()
    committer = _committer(tmp_path, service)
    record = committer.upload_verified(
        source, tmp_path / "pilot" / "payload.bin", replace=False
    )
    committer.resolve_folder(("production",), create=True)
    with pytest.raises(drive.RemoteCommitError, match="intended Drive path"):
        committer.verify_record_for_path(
            record, tmp_path / "production" / "payload.bin"
        )
    committer.verify_record_for_path(record, tmp_path / "pilot" / "payload.bin")


def test_transient_before_apply_retries_one_deterministic_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"retry safely")
    service = _Service()
    committer = _committer(tmp_path, service)
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    committer.resolve_folder(("campaign",), create=True)
    service.file_api.fail_create_before_apply = 2
    record = committer.upload_verified(
        source, tmp_path / "campaign" / "payload.bin", replace=False
    )
    assert record["remote_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not any(
        str(row["name"]).startswith(".uploading.")
        for row in service.file_api.items.values()
    )


def test_nonfolder_path_component_is_rejected(tmp_path: Path) -> None:
    service = _Service()
    service.file_api.items["not-a-folder"] = {
        "id": "not-a-folder",
        "name": "campaign",
        "parents": ["root"],
        "mimeType": "application/octet-stream",
        "size": "0",
        "sha256Checksum": hashlib.sha256(b"").hexdigest(),
        "trashed": False,
    }
    committer = _committer(tmp_path, service)
    with pytest.raises(drive.RemoteCommitError, match="not a folder"):
        committer.resolve_folder(("campaign",), create=False)


def test_duplicate_folder_component_is_rejected(tmp_path: Path) -> None:
    service = _Service()
    for file_id in ("folder-a", "folder-b"):
        service.file_api.items[file_id] = {
            "id": file_id,
            "name": "campaign",
            "parents": ["root"],
            "mimeType": drive.FOLDER_MIME_TYPE,
            "size": "0",
            "sha256Checksum": hashlib.sha256(b"").hexdigest(),
            "trashed": False,
        }
    committer = _committer(tmp_path, service)
    with pytest.raises(drive.RemoteCommitError, match="ambiguous Drive path"):
        committer.resolve_folder(("campaign",), create=False)


def test_resumable_upload_continues_same_session_after_chunk_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResumableRequest:
        def __init__(self) -> None:
            self.calls = 0

        def next_chunk(self, *, num_retries: int = 0):
            assert num_retries == 0
            self.calls += 1
            if self.calls == 1:
                return object(), None
            if self.calls == 2:
                raise OSError("connection dropped between upload chunks")
            return None, {"id": "durable-file"}

    request = ResumableRequest()
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    result = drive._execute_resumable_upload(
        lambda: request, deadline_seconds=60.0
    )
    assert result == {"id": "durable-file"}
    assert request.calls == 3


def test_resumable_upload_deadline_fails_without_recording_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingRequest:
        def next_chunk(self, *, num_retries: int = 0):
            assert num_retries == 0
            raise TimeoutError("socket timeout")

    moments = iter((0.0, 0.0, 2.0, 2.0, 2.0))
    monkeypatch.setattr(drive.time, "monotonic", lambda: next(moments, 2.0))
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    with pytest.raises(drive.RemoteCommitError, match="deadline reached"):
        drive._execute_resumable_upload(
            lambda: HangingRequest(), attempts=6, deadline_seconds=1.0
        )


def test_upload_verified_retries_transient_chunk_on_one_create_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "checkpoint.bin"
    source.write_bytes(b"checkpoint payload")
    service = _Service()
    committer = _committer(tmp_path, service)
    committer.resolve_folder(("campaign",), create=True)
    original_create = service.file_api.create
    created_requests = 0
    chunk_calls = 0

    class ResumableCreate:
        def __init__(self, body: dict[str, Any], media_body: Any) -> None:
            self.body = body
            self.media_body = media_body

        def next_chunk(self, *, num_retries: int = 0):
            nonlocal chunk_calls
            assert num_retries == 0
            chunk_calls += 1
            if chunk_calls == 1:
                raise OSError("transient upload chunk failure")
            raw = Path(self.media_body).read_bytes()
            file_id = service.file_api._new_id()
            row = {
                "id": file_id,
                "name": self.body["name"],
                "parents": list(self.body["parents"]),
                "mimeType": "application/octet-stream",
                "size": str(len(raw)),
                "sha256Checksum": hashlib.sha256(raw).hexdigest(),
                "trashed": False,
            }
            service.file_api.items[file_id] = row
            return None, dict(row)

    def create(*, body: dict[str, Any], media_body: Any = None, **kwargs: Any):
        nonlocal created_requests
        if media_body is None:
            return original_create(body=body, media_body=media_body, **kwargs)
        created_requests += 1
        return ResumableCreate(body, media_body)

    monkeypatch.setattr(service.file_api, "create", create)
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    record = committer.upload_verified(
        source,
        tmp_path / "campaign" / "checkpoint.bin",
        replace=False,
    )
    assert record["remote_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert created_requests == 1
    assert chunk_calls == 2
