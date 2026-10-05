"""Mount-independent, server-verified Google Drive operations.

This is the operational copy used by the v4 deployment root.  The qualified
P1/H1 scientific snapshots keep their original ``src/drive_remote_commit.py``
bytes; wrappers may substitute this implementation at runtime without changing
the source identity recorded by existing archives and checkpoints.

DriveFS is never consulted for remote existence or durability.  A file is
accepted only when Drive v3 reports one exact name in the intended parent with
the expected byte count and SHA-256 checksum.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


REMOTE_COMMIT_SCHEMA = "classA_drive_api_commit_v1"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive")
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
FILE_FIELDS = (
    "id,name,parents,size,md5Checksum,sha256Checksum,trashed,modifiedTime,mimeType"
)
DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0
DEFAULT_OPERATION_DEADLINE_SECONDS = 15.0 * 60.0
DEFAULT_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_DOWNLOAD_RESERVE_BYTES = 64 * 1024 * 1024
MAX_JSON_DOWNLOAD_BYTES = 32 * 1024 * 1024


class RemoteCommitError(RuntimeError):
    """Raised when server-side durability cannot be proved."""


class RemoteQuotaError(RemoteCommitError):
    """Raised before a new upload when account quota is insufficient."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _lexical_absolute(path: Path | str) -> Path:
    """Normalize a path without touching the FUSE mount."""

    raw = os.fspath(path)
    return Path(os.path.normpath(os.path.abspath(raw)))


def _deadline_after(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    if float(seconds) <= 0.0:
        raise ValueError("operation deadline must be positive")
    return time.monotonic() + float(seconds)


def _execute_with_retries(
    request_factory: Callable[[], Any],
    *,
    attempts: int = 6,
    deadline_seconds: float | None = DEFAULT_OPERATION_DEADLINE_SECONDS,
) -> Any:
    """Execute a Drive request with bounded retries.

    The authenticated HTTP transport also has a per-request socket timeout, so
    a dead connection cannot hold a Colab session forever.
    """

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    deadline = _deadline_after(deadline_seconds)
    delay = 1.0
    error: Exception | None = None
    for attempt in range(attempts):
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            return request_factory().execute()
        except Exception as exc:  # Google transports expose several exception types.
            error = exc
            if attempt + 1 == attempts:
                break
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                sleep_seconds = min(delay, remaining)
            else:
                sleep_seconds = delay
            time.sleep(sleep_seconds)
            delay = min(delay * 2.0, 8.0)
    suffix = " (operation deadline reached)" if deadline is not None and time.monotonic() >= deadline else ""
    raise RemoteCommitError(f"Google Drive API request failed{suffix}: {error}") from error


def _execute_resumable_upload(
    request_factory: Callable[[], Any],
    *,
    attempts: int = 6,
    execute_attempts: int = 3,
    deadline_seconds: float | None = DEFAULT_OPERATION_DEADLINE_SECONDS,
) -> Any:
    """Advance one resumable request chunk by chunk under a hard deadline.

    Google discovery requests used by the local test doubles expose only
    ``execute``; those retain the ordinary bounded-retry path.  Real
    ``MediaFileUpload(resumable=True)`` requests expose ``next_chunk``.  We
    keep that same request object after a transient chunk failure so its
    server upload session is resumed instead of starting another object.
    """

    request = request_factory()
    if not callable(getattr(request, "next_chunk", None)):
        return _execute_with_retries(
            request_factory,
            attempts=execute_attempts,
            deadline_seconds=deadline_seconds,
        )
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    deadline = _deadline_after(deadline_seconds)
    failures = 0
    delay = 1.0
    error: Exception | None = None
    while deadline is None or time.monotonic() < deadline:
        try:
            _, response = request.next_chunk(num_retries=0)
        except Exception as exc:
            error = exc
            failures += 1
            if failures >= attempts:
                break
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0.0:
                break
            time.sleep(delay if remaining is None else min(delay, remaining))
            delay = min(delay * 2.0, 8.0)
            continue
        failures = 0
        delay = 1.0
        if response is not None:
            return response
    suffix = (
        " (operation deadline reached)"
        if deadline is not None and time.monotonic() >= deadline
        else ""
    )
    raise RemoteCommitError(f"Google Drive resumable upload failed{suffix}: {error}") from error


def build_drive_service() -> Any:
    """Build an authenticated Drive v3 service with a socket timeout."""

    try:
        import google.auth
        import google_auth_httplib2
        import httplib2
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
    authorized_http = google_auth_httplib2.AuthorizedHttp(
        credentials,
        http=httplib2.Http(timeout=DEFAULT_HTTP_TIMEOUT_SECONDS),
    )
    return build(
        "drive",
        "v3",
        http=authorized_http,
        cache_discovery=False,
    )


class DriveRemoteCommitter:
    """Fail-closed wrapper around Drive v3 files/about endpoints."""

    def __init__(
        self,
        *,
        service: Any | None = None,
        drive_root: Path | str = DEFAULT_DRIVE_ROOT,
        media_upload_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.service = service if service is not None else build_drive_service()
        self.drive_root = _lexical_absolute(drive_root)
        self.media_upload_factory = media_upload_factory

    def _list_children(self, parent_id: str, name: str) -> list[dict[str, Any]]:
        query = (
            f"'{_escape_query(parent_id)}' in parents and "
            f"name = '{_escape_query(name)}' and trashed = false"
        )
        rows: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            response = _execute_with_retries(
                lambda page_token=page_token: self.service.files().list(
                    q=query,
                    spaces="drive",
                    fields=f"nextPageToken,files({FILE_FIELDS})",
                    pageSize=100,
                    pageToken=page_token,
                )
            )
            rows.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return rows

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
        try:
            created = _execute_with_retries(
                lambda: self.service.files().create(
                    body={
                        "name": name,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [parent_id],
                    },
                    fields=FILE_FIELDS,
                ),
                # A create is not idempotent.  Reconcile its deterministic
                # server name after one attempt instead of blindly creating a
                # duplicate when only the response was lost.
                attempts=1,
            )
        except RemoteCommitError:
            # A create response can be lost after the server applies it.
            recovered = self._unique_child(parent_id, name, required=False)
            if recovered is None:
                raise
            created = recovered
        duplicates = self._list_children(parent_id, name)
        if len(duplicates) != 1:
            raise RemoteCommitError(
                f"Drive folder creation produced {len(duplicates)} entries named {name!r}"
            )
        item = duplicates[0]
        if item.get("mimeType") != FOLDER_MIME_TYPE:
            raise RemoteCommitError(f"Drive path component is not a folder: {name!r}")
        return item

    def resolve_folder(self, relative_parts: tuple[str, ...], *, create: bool) -> str:
        parent_id = "root"
        for part in relative_parts:
            if not part or part in (".", "..") or "/" in part:
                raise RemoteCommitError(f"invalid Drive folder component: {part!r}")
            item = self._unique_child(parent_id, part, required=False)
            if item is None:
                if not create:
                    raise RemoteCommitError(
                        f"Drive folder is absent: {'/'.join(relative_parts)}"
                    )
                item = self._create_folder(parent_id, part)
            if item.get("mimeType") != FOLDER_MIME_TYPE:
                raise RemoteCommitError(f"Drive path component is not a folder: {part!r}")
            parent_id = str(item["id"])
        return parent_id

    def split_remote_path(self, path: Path | str) -> tuple[tuple[str, ...], str]:
        absolute = _lexical_absolute(path)
        try:
            relative = absolute.relative_to(self.drive_root)
        except ValueError as exc:
            raise RemoteCommitError(
                f"remote path must be below {self.drive_root}: {absolute}"
            ) from exc
        if not relative.name or relative.name in (".", ".."):
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
        self._assert_metadata(
            item,
            name=name,
            parent_id=parent_id,
            size=int(item["size"]),
            sha256=str(item["sha256Checksum"]),
        )
        return self.commit_record(item)

    def verify_record_for_path(
        self, record: dict[str, Any], path: Path | str
    ) -> dict[str, Any]:
        """Bind a receipt record to an exact intended Drive path."""

        parts, name = self.split_remote_path(path)
        parent_id = self.resolve_folder(parts, create=False)
        if (
            str(record.get("remote_name")) != name
            or str(record.get("remote_parent_id")) != parent_id
        ):
            raise RemoteCommitError(
                "remote commit record is not bound to the intended Drive path"
            )
        exact = self._unique_child(parent_id, name, required=True)
        assert exact is not None
        if str(exact.get("id")) != str(record.get("remote_file_id")):
            raise RemoteCommitError(
                "remote commit record file ID differs from the intended Drive path"
            )
        return self.verify_commit_record(record)

    def _media(self, local_path: Path) -> Any:
        if self.media_upload_factory is not None:
            return self.media_upload_factory(local_path)
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise RemoteCommitError("googleapiclient upload support is unavailable") from exc
        return MediaFileUpload(
            str(local_path),
            mimetype="application/octet-stream",
            resumable=True,
            chunksize=DEFAULT_DOWNLOAD_CHUNK_BYTES,
        )

    def _verified_named_item(
        self,
        *,
        parent_id: str,
        name: str,
        size: int,
        checksum: str,
        required: bool,
    ) -> dict[str, Any] | None:
        item = self._unique_child(parent_id, name, required=required)
        if item is None:
            return None
        item = self.metadata(str(item["id"]))
        self._assert_metadata(
            item,
            name=name,
            parent_id=parent_id,
            size=size,
            sha256=checksum,
        )
        return item

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
        parts, final_name = self.split_remote_path(remote_path)
        quota_checked = False
        try:
            parent_id = self.resolve_folder(parts, create=False)
        except RemoteCommitError as exc:
            if "folder is absent" not in str(exc).lower():
                raise
            # Do not mutate even the folder tree when the account cannot hold
            # the pending file plus its required safety reserve.
            self.require_quota(
                upload_bytes=size,
                required_headroom_bytes=required_headroom_bytes,
            )
            quota_checked = True
            parent_id = self.resolve_folder(parts, create=True)

        # Idempotent recovery must work even when quota is now exhausted: if
        # the exact bytes already reached their final name, no upload is needed.
        existing = self._unique_child(parent_id, final_name, required=False)
        if existing is not None:
            existing = self.metadata(str(existing["id"]))
            try:
                self._assert_metadata(
                    existing,
                    name=final_name,
                    parent_id=parent_id,
                    size=size,
                    sha256=checksum,
                )
            except RemoteCommitError:
                if not replace:
                    raise
            else:
                return self.commit_record(existing)
            if not quota_checked:
                self.require_quota(
                    upload_bytes=size,
                    required_headroom_bytes=required_headroom_bytes,
                )
                quota_checked = True
            try:
                published = _execute_resumable_upload(
                    lambda: self.service.files().update(
                        fileId=str(existing["id"]),
                        body={"name": final_name},
                        media_body=self._media(local_path),
                        fields=FILE_FIELDS,
                    ),
                    attempts=3,
                )
            except RemoteCommitError:
                # The update response may have been lost after commit.
                published = self._unique_child(parent_id, final_name, required=True)
                assert published is not None
            published = self.metadata(str(published["id"]))
            self._assert_metadata(
                published,
                name=final_name,
                parent_id=parent_id,
                size=size,
                sha256=checksum,
            )
            return self.commit_record(published)

        temporary_name = f".uploading.{checksum[:24]}.{final_name}"
        temporary = self._unique_child(parent_id, temporary_name, required=False)
        if temporary is not None:
            temporary = self.metadata(str(temporary["id"]))
            self._assert_metadata(
                temporary,
                name=temporary_name,
                parent_id=parent_id,
                size=size,
                sha256=checksum,
            )
        else:
            if not quota_checked:
                self.require_quota(
                    upload_bytes=size,
                    required_headroom_bytes=required_headroom_bytes,
                )
                quota_checked = True
            upload_error: RemoteCommitError | None = None
            for upload_attempt in range(3):
                try:
                    temporary = _execute_resumable_upload(
                        lambda: self.service.files().create(
                            body={"name": temporary_name, "parents": [parent_id]},
                            media_body=self._media(local_path),
                            fields=FILE_FIELDS,
                        ),
                        # A create is not idempotent.  Reconcile its
                        # deterministic transaction name after one ordinary
                        # execute attempt.  A real resumable request still
                        # retries transient chunks on the same upload session.
                        execute_attempts=1,
                    )
                except RemoteCommitError as exc:
                    upload_error = exc
                    temporary = self._unique_child(
                        parent_id, temporary_name, required=False
                    )
                    if temporary is not None:
                        break
                    if upload_attempt < 2:
                        time.sleep(min(2**upload_attempt, 2))
                        continue
                    raise
                else:
                    break
            if temporary is None:
                assert upload_error is not None
                raise upload_error
            temporary = self.metadata(str(temporary["id"]))
            self._assert_metadata(
                temporary,
                name=temporary_name,
                parent_id=parent_id,
                size=size,
                sha256=checksum,
            )

        temporary_id = str(temporary["id"])
        try:
            published = _execute_with_retries(
                lambda: self.service.files().update(
                    fileId=temporary_id,
                    body={"name": final_name},
                    fields=FILE_FIELDS,
                )
            )
        except RemoteCommitError:
            # Reconcile the rename; never start a second upload blindly.
            published = self._unique_child(parent_id, final_name, required=False)
            if published is None:
                raise
        exact = self._list_children(parent_id, final_name)
        if len(exact) != 1:
            raise RemoteCommitError(
                f"final Drive publication produced {len(exact)} files named {final_name!r}"
            )
        published = self.metadata(str(exact[0]["id"]))
        if str(published["id"]) != temporary_id:
            raise RemoteCommitError(
                "final Drive path was won by a different concurrent upload"
            )
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
        required = (
            "remote_file_id",
            "remote_name",
            "remote_parent_id",
            "remote_bytes",
            "remote_sha256",
        )
        missing = [name for name in required if name not in record]
        if missing:
            raise RemoteCommitError(f"receipt remote commit is incomplete: {missing}")
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
        metadata = self.metadata(file_id)
        size = int(metadata.get("size", -1))
        if size < 0 or size > MAX_JSON_DOWNLOAD_BYTES:
            raise RemoteCommitError(
                f"refusing in-memory Drive download of {size} bytes; use download_to"
            )
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise RemoteCommitError("googleapiclient download support is unavailable") from exc
        output = io.BytesIO()
        downloader = MediaIoBaseDownload(
            output,
            self.service.files().get_media(fileId=file_id),
            chunksize=DEFAULT_DOWNLOAD_CHUNK_BYTES,
        )
        done = False
        while not done:
            error: Exception | None = None
            for attempt in range(6):
                try:
                    _, done = downloader.next_chunk()
                    break
                except Exception as exc:
                    error = exc
                    if attempt == 5:
                        raise RemoteCommitError(
                            f"Google Drive download failed: {error}"
                        ) from error
                    time.sleep(min(2**attempt, 8))
        raw = output.getvalue()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != metadata.get(
            "sha256Checksum"
        ):
            raise RemoteCommitError("downloaded Drive bytes failed size/SHA-256 verification")
        return raw

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

    def download_to(
        self,
        file_id: str,
        destination: Path | str,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        """Stream a Drive file to local disk, hash it, then atomically publish."""

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata(file_id)
        server_size = int(metadata.get("size", -1))
        server_sha = str(metadata.get("sha256Checksum", ""))
        expected_size = server_size if expected_size is None else int(expected_size)
        expected_sha256 = server_sha if expected_sha256 is None else str(expected_sha256)
        if server_size != expected_size or server_sha != expected_sha256:
            raise RemoteCommitError("Drive metadata differs from requested download identity")
        free = shutil.disk_usage(destination.parent).free
        if free < expected_size + DEFAULT_DOWNLOAD_RESERVE_BYTES:
            raise RemoteCommitError(
                "local disk has insufficient headroom for verified Drive download: "
                f"free={free}, file={expected_size}"
            )
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise RemoteCommitError("googleapiclient download support is unavailable") from exc
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary = Path(raw_temporary)
        try:
            with os.fdopen(descriptor, "w+b") as handle:
                downloader = MediaIoBaseDownload(
                    handle,
                    self.service.files().get_media(fileId=file_id),
                    chunksize=DEFAULT_DOWNLOAD_CHUNK_BYTES,
                )
                done = False
                while not done:
                    error: Exception | None = None
                    for attempt in range(6):
                        try:
                            _, done = downloader.next_chunk()
                            break
                        except Exception as exc:
                            error = exc
                            if attempt == 5:
                                raise RemoteCommitError(
                                    f"Google Drive download failed: {error}"
                                ) from error
                            time.sleep(min(2**attempt, 8))
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size != expected_size:
                raise RemoteCommitError("downloaded Drive file has the wrong byte count")
            if sha256_file(temporary) != expected_sha256:
                raise RemoteCommitError("downloaded Drive file failed SHA-256 verification")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination


def stage_json(payload: Any, *, prefix: str) -> Path:
    stage_root = Path("/content/classA_remote_stage")
    if not Path("/content").is_dir():
        stage_root = Path(tempfile.gettempdir()) / "classA_remote_stage"
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
    record = committer.path_commit_record(path)
    raw = committer.download_bytes(str(record["remote_file_id"]))
    if len(raw) != int(record["remote_bytes"]):
        raise RemoteCommitError(f"remote JSON has the wrong byte count: {path}")
    if hashlib.sha256(raw).hexdigest() != str(record["remote_sha256"]):
        raise RemoteCommitError(f"remote JSON failed SHA-256 verification: {path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RemoteCommitError(f"remote JSON is not an object: {path}")
    return payload
