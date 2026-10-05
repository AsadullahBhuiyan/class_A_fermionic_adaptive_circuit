"""Server-verified Google Drive commits for Colab production bundles.

DriveFS is deliberately treated as a cache.  A commit is durable only after the
Drive API reports the expected parent, name, byte count, and SHA-256 checksum.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable


REMOTE_COMMIT_SCHEMA = "classA_drive_api_commit_v1"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive")
FILE_FIELDS = "id,name,parents,size,md5Checksum,sha256Checksum,trashed,modifiedTime"


class RemoteCommitError(RuntimeError):
    """Raised when server-side durability cannot be proved."""


class RemoteQuotaError(RemoteCommitError):
    """Raised before upload when the account does not have enough free space."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _execute_with_retries(
    request_factory: Callable[[], Any], *, attempts: int = 6
) -> Any:
    delay = 1.0
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return request_factory().execute()
        except Exception as exc:  # Google transports expose several exception types.
            error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 8.0)
    raise RemoteCommitError(f"Google Drive API request failed: {error}") from error


def build_drive_service() -> Any:
    """Build an authenticated Drive v3 service using Colab application credentials."""
    try:
        import google.auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RemoteCommitError(
            "Google Drive API packages are unavailable; run this bundle in Colab"
        ) from exc
    credentials, _ = google.auth.default(scopes=[DRIVE_SCOPE])
    if credentials is None:
        raise RemoteCommitError(
            "Google Drive is not authorized; run google.colab.auth.authenticate_user()"
        )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


class DriveRemoteCommitter:
    """Small fail-closed wrapper around Drive v3 files/about endpoints."""

    def __init__(
        self,
        *,
        service: Any | None = None,
        drive_root: Path | str = DEFAULT_DRIVE_ROOT,
        media_upload_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.service = service if service is not None else build_drive_service()
        self.drive_root = Path(drive_root).resolve()
        self.media_upload_factory = media_upload_factory

    def _list_children(self, parent_id: str, name: str) -> list[dict[str, Any]]:
        query = (
            f"'{_escape_query(parent_id)}' in parents and "
            f"name = '{_escape_query(name)}' and trashed = false"
        )
        response = _execute_with_retries(
            lambda: self.service.files().list(
                q=query,
                spaces="drive",
                fields=f"files({FILE_FIELDS})",
                pageSize=100,
            )
        )
        return list(response.get("files", []))

    def _unique_child(
        self, parent_id: str, name: str, *, required: bool
    ) -> dict[str, Any] | None:
        matches = self._list_children(parent_id, name)
        if len(matches) > 1:
            raise RemoteCommitError(
                f"ambiguous Drive path: {len(matches)} files named {name!r}"
            )
        if not matches:
            if required:
                raise RemoteCommitError(f"Drive file is absent: {name}")
            return None
        return matches[0]

    def _create_folder(self, parent_id: str, name: str) -> dict[str, Any]:
        created = _execute_with_retries(
            lambda: self.service.files().create(
                body={
                    "name": name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [parent_id],
                },
                fields=FILE_FIELDS,
            )
        )
        return created

    def resolve_folder(self, relative_parts: tuple[str, ...], *, create: bool) -> str:
        parent_id = "root"
        for part in relative_parts:
            if not part or part in (".", ".."):
                raise RemoteCommitError(f"invalid Drive folder component: {part!r}")
            item = self._unique_child(parent_id, part, required=False)
            if item is None:
                if not create:
                    raise RemoteCommitError(
                        f"Drive folder is absent: {'/'.join(relative_parts)}"
                    )
                item = self._create_folder(parent_id, part)
            parent_id = str(item["id"])
        return parent_id

    def split_remote_path(self, path: Path | str) -> tuple[tuple[str, ...], str]:
        absolute = Path(path).resolve()
        try:
            relative = absolute.relative_to(self.drive_root)
        except ValueError as exc:
            raise RemoteCommitError(
                f"remote path must be below {self.drive_root}: {absolute}"
            ) from exc
        if not relative.name:
            raise RemoteCommitError("remote file path has no filename")
        return tuple(relative.parent.parts), relative.name

    def storage_quota(self) -> dict[str, int]:
        response = _execute_with_retries(
            lambda: self.service.about().get(fields="storageQuota")
        )
        raw = response.get("storageQuota", {})
        parsed: dict[str, int] = {}
        for key in ("limit", "usage", "usageInDrive", "usageInDriveTrash"):
            if raw.get(key) is not None:
                parsed[key] = int(raw[key])
        return parsed

    def require_quota(self, *, upload_bytes: int, required_headroom_bytes: int) -> None:
        quota = self.storage_quota()
        if "limit" not in quota:
            return
        free = quota["limit"] - quota.get("usage", 0)
        required = int(upload_bytes) + int(required_headroom_bytes)
        if free < required:
            raise RemoteQuotaError(
                "Google Drive account quota is insufficient: "
                f"free={free}, upload={upload_bytes}, required_headroom="
                f"{required_headroom_bytes}"
            )

    def metadata(self, file_id: str) -> dict[str, Any]:
        return _execute_with_retries(
            lambda: self.service.files().get(fileId=file_id, fields=FILE_FIELDS)
        )

    @staticmethod
    def _assert_metadata(
        metadata: dict[str, Any],
        *,
        name: str,
        parent_id: str,
        size: int,
        sha256: str,
    ) -> None:
        observed = {
            "name": metadata.get("name"),
            "parents": metadata.get("parents"),
            "size": int(metadata.get("size", -1)),
            "sha256": metadata.get("sha256Checksum"),
            "trashed": bool(metadata.get("trashed", False)),
        }
        expected = {
            "name": name,
            "parents": [parent_id],
            "size": int(size),
            "sha256": sha256,
            "trashed": False,
        }
        if observed != expected:
            raise RemoteCommitError(
                f"Drive server verification failed: observed={observed}, expected={expected}"
            )

    def verify_path(
        self, path: Path | str, *, expected_size: int, expected_sha256: str
    ) -> dict[str, Any]:
        parts, name = self.split_remote_path(path)
        parent_id = self.resolve_folder(parts, create=False)
        item = self._unique_child(parent_id, name, required=True)
        assert item is not None
        item = self.metadata(str(item["id"]))
        self._assert_metadata(
            item,
            name=name,
            parent_id=parent_id,
            size=expected_size,
            sha256=expected_sha256,
        )
        return item

    def path_commit_record(self, path: Path | str) -> dict[str, Any]:
        parts, name = self.split_remote_path(path)
        parent_id = self.resolve_folder(parts, create=False)
        item = self._unique_child(parent_id, name, required=True)
        assert item is not None
        item = self.metadata(str(item["id"]))
        if item.get("sha256Checksum") is None or item.get("size") is None:
            raise RemoteCommitError(f"Drive file lacks binary checksum metadata: {path}")
        return self.commit_record(item)

    def upload_verified(
        self,
        local_path: Path | str,
        remote_path: Path | str,
        *,
        replace: bool,
        required_headroom_bytes: int = 0,
    ) -> dict[str, Any]:
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        size = local_path.stat().st_size
        checksum = sha256_file(local_path)
        self.require_quota(
            upload_bytes=size, required_headroom_bytes=required_headroom_bytes
        )
        parts, final_name = self.split_remote_path(remote_path)
        parent_id = self.resolve_folder(parts, create=True)
        existing = self._unique_child(parent_id, final_name, required=False)
        if existing is not None and not replace:
            verified = self.metadata(str(existing["id"]))
            self._assert_metadata(
                verified,
                name=final_name,
                parent_id=parent_id,
                size=size,
                sha256=checksum,
            )
            return self.commit_record(verified)

        if self.media_upload_factory is None:
            try:
                from googleapiclient.http import MediaFileUpload
            except ImportError as exc:
                raise RemoteCommitError("googleapiclient upload support is unavailable") from exc
            media = MediaFileUpload(
                str(local_path), mimetype="application/octet-stream", resumable=True
            )
        else:
            media = self.media_upload_factory(local_path)
        if existing is not None:
            published = _execute_with_retries(
                lambda: self.service.files().update(
                    fileId=str(existing["id"]),
                    body={"name": final_name},
                    media_body=media,
                    fields=FILE_FIELDS,
                ),
                attempts=3,
            )
            published = self.metadata(str(published["id"]))
            self._assert_metadata(
                published,
                name=final_name,
                parent_id=parent_id,
                size=size,
                sha256=checksum,
            )
            return self.commit_record(published)
        temporary_name = f".uploading.{uuid.uuid4().hex}.{final_name}"
        temporary = _execute_with_retries(
            lambda: self.service.files().create(
                body={"name": temporary_name, "parents": [parent_id]},
                media_body=media,
                fields=FILE_FIELDS,
            ),
            attempts=3,
        )
        temporary_id = str(temporary["id"])
        temporary = self.metadata(temporary_id)
        self._assert_metadata(
            temporary,
            name=temporary_name,
            parent_id=parent_id,
            size=size,
            sha256=checksum,
        )
        published = _execute_with_retries(
            lambda: self.service.files().update(
                fileId=temporary_id,
                body={"name": final_name},
                fields=FILE_FIELDS,
            )
        )
        published = self.metadata(str(published["id"]))
        self._assert_metadata(
            published,
            name=final_name,
            parent_id=parent_id,
            size=size,
            sha256=checksum,
        )
        return self.commit_record(published)

    @staticmethod
    def commit_record(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": REMOTE_COMMIT_SCHEMA,
            "remote_file_id": str(metadata["id"]),
            "remote_name": str(metadata["name"]),
            "remote_parent_id": str(metadata["parents"][0]),
            "remote_bytes": int(metadata["size"]),
            "remote_sha256": str(metadata["sha256Checksum"]),
            "remote_verified_unix": time.time(),
        }

    def verify_commit_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("schema") != REMOTE_COMMIT_SCHEMA:
            raise RemoteCommitError("receipt has the wrong remote commit schema")
        metadata = self.metadata(str(record["remote_file_id"]))
        self._assert_metadata(
            metadata,
            name=str(record["remote_name"]),
            parent_id=str(record["remote_parent_id"]),
            size=int(record["remote_bytes"]),
            sha256=str(record["remote_sha256"]),
        )
        return metadata

    def download_bytes(self, file_id: str) -> bytes:
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise RemoteCommitError("googleapiclient download support is unavailable") from exc
        output = io.BytesIO()
        downloader = MediaIoBaseDownload(
            output, self.service.files().get_media(fileId=file_id)
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return output.getvalue()

    def delete_verified(self, record: dict[str, Any]) -> None:
        self.verify_commit_record(record)
        _execute_with_retries(
            lambda: self.service.files().delete(fileId=str(record["remote_file_id"]))
        )

    def delete_verified_if_present(self, record: dict[str, Any]) -> bool:
        """Idempotently finish a previously authorized post-commit cleanup."""
        try:
            self.verify_commit_record(record)
        except RemoteCommitError as exc:
            message = str(exc).lower()
            if "404" in message or "not found" in message:
                return False
            raise
        _execute_with_retries(
            lambda: self.service.files().delete(fileId=str(record["remote_file_id"]))
        )
        return True

    def delete_path_verified(
        self, path: Path | str, *, expected_size: int, expected_sha256: str
    ) -> None:
        metadata = self.verify_path(
            path, expected_size=expected_size, expected_sha256=expected_sha256
        )
        _execute_with_retries(
            lambda: self.service.files().delete(fileId=str(metadata["id"]))
        )

    def download_to(self, file_id: str, destination: Path | str) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = self.download_bytes(file_id)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        return destination


def stage_json(payload: Any, *, prefix: str) -> Path:
    stage_root = Path("/content/classA_remote_stage")
    stage_root.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f"{prefix}.", suffix=".json", dir=stage_root
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def publish_json(
    committer: DriveRemoteCommitter,
    payload: dict[str, Any],
    remote_path: Path | str,
    *,
    replace: bool,
    required_headroom_bytes: int = 0,
) -> dict[str, Any]:
    staged = stage_json(payload, prefix=Path(remote_path).name)
    try:
        return committer.upload_verified(
            staged,
            remote_path,
            replace=replace,
            required_headroom_bytes=required_headroom_bytes,
        )
    finally:
        staged.unlink(missing_ok=True)


def read_remote_json(committer: DriveRemoteCommitter, path: Path | str) -> dict[str, Any]:
    parts, name = committer.split_remote_path(path)
    parent_id = committer.resolve_folder(parts, create=False)
    item = committer._unique_child(parent_id, name, required=True)
    assert item is not None
    raw = committer.download_bytes(str(item["id"]))
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RemoteCommitError(f"remote JSON is not an object: {path}")
    return payload
