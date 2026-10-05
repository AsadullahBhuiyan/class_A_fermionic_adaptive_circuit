"""Unhashed operational hardening for the active P1-v4 campaign.

P1 checkpoint identity hashes every file below the bundle's ``src`` directory.
This adapter deliberately lives outside that directory so transport and lease fixes
do not invalidate an already committed trajectory checkpoint.  It changes no
scientific configuration, engine argument, RNG state, observer state, or checkpoint
payload interpretation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Iterator


P1_OPERATIONAL_HARDENING_SCHEMA = "p1_v4_drivefs_independent_runtime_v1"
P1_API_LEASE_SCHEMA = "p1_api_shard_single_writer_lease_v1"
P1_MANIFEST_SIDECAR_SCHEMA = "p1_server_verified_manifest_sidecar_v1"
P1_API_LEASE_HEARTBEAT_SECONDS = 30.0
P1_API_LEASE_STALE_SECONDS = 900.0
P1_API_LEASE_SETTLE_SECONDS = 0.5


_OPERATIONAL_DRIVE: Any | None = None


def _operational_drive() -> Any:
    if _OPERATIONAL_DRIVE is None:
        raise RuntimeError("P1 operational Drive helper has not been installed")
    return _OPERATIONAL_DRIVE


def lexical_absolute(path: Path | str) -> Path:
    """Return an absolute normalized path without touching the filesystem."""
    raw = os.fspath(path)
    if not os.path.isabs(raw):
        raw = os.path.join(os.getcwd(), raw)
    return Path(os.path.normpath(raw))


def _is_below(path: Path | str, root: Path | str) -> bool:
    candidate = os.fspath(lexical_absolute(path))
    anchor = os.fspath(lexical_absolute(root))
    try:
        return os.path.commonpath((candidate, anchor)) == anchor
    except ValueError:
        return False


def _not_found(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "absent" in message or "404" in message or "not found" in message


class P1DriveRemoteCommitter:
    """Build a P1-path-scoped subclass of the root operational Drive helper."""

    @staticmethod
    def build(
        operational: Any,
        *,
        scope_path: Path | str | None = None,
        lease_getter: Any | None = None,
    ) -> type:
        class ScopedCommitter(operational.DriveRemoteCommitter):
            def __init__(
                self,
                *,
                service: Any | None = None,
                drive_root: Path | str = operational.DEFAULT_DRIVE_ROOT,
                media_upload_factory: Any | None = None,
                scope_path_override: Path | str | None = None,
            ) -> None:
                # The frozen superclass resolves DriveFS and can raise ENOTCONN.
                self.service = (
                    service if service is not None else operational.build_drive_service()
                )
                self.drive_root = lexical_absolute(drive_root)
                self.media_upload_factory = media_upload_factory
                selected = scope_path_override if scope_path_override is not None else scope_path
                self.scope_path = None if selected is None else lexical_absolute(selected)

            def split_remote_path(self, path: Path | str) -> tuple[tuple[str, ...], str]:
                absolute = lexical_absolute(path)
                try:
                    relative = absolute.relative_to(self.drive_root)
                except ValueError as exc:
                    raise operational.RemoteCommitError(
                        f"remote path must be below {self.drive_root}: {absolute}"
                    ) from exc
                if not relative.name or any(
                    part in ("", ".", "..") for part in relative.parts
                ):
                    raise operational.RemoteCommitError(f"invalid remote path: {path}")
                return tuple(relative.parent.parts), relative.name

            def upload_verified(
                self, local_path: Any, remote_path: Any, **kwargs: Any
            ) -> dict[str, Any]:
                lease = None if lease_getter is None else lease_getter()
                if lease is not None:
                    lease.assert_owned()
                result = super().upload_verified(local_path, remote_path, **kwargs)
                if lease is not None:
                    lease.assert_owned()
                return result

            def verify_record_for_path(
                self, record: dict[str, Any], path: Path | str
            ) -> dict[str, Any]:
                if record.get("schema") != operational.REMOTE_COMMIT_SCHEMA:
                    raise operational.RemoteCommitError(
                        "receipt has the wrong remote commit schema"
                    )
                metadata = self.verify_path(
                    path,
                    expected_size=int(record["remote_bytes"]),
                    expected_sha256=str(record["remote_sha256"]),
                )
                expected = {
                    "remote_file_id": str(metadata["id"]),
                    "remote_name": str(metadata["name"]),
                    "remote_parent_id": str(metadata["parents"][0]),
                    "remote_bytes": int(metadata["size"]),
                    "remote_sha256": str(metadata["sha256Checksum"]),
                }
                observed = {key: record.get(key) for key in expected}
                if observed != expected:
                    raise operational.RemoteCommitError(
                        "remote commit record is not bound to its intended path: "
                        f"observed={observed}, expected={expected}"
                    )
                return metadata

            def _intended_path(self, record: dict[str, Any]) -> Path:
                if self.scope_path is None:
                    raise operational.RemoteCommitError(
                        "scoped P1 verification lacks its output path"
                    )
                name = str(record.get("remote_name", ""))
                scope = self.scope_path
                if name == scope.name:
                    return scope
                if not scope.name.endswith(".tar.gz"):
                    raise operational.RemoteCommitError(
                        f"record {name!r} lies outside scoped path {scope}"
                    )
                run_id = scope.name.removesuffix(".tar.gz")
                if name == f"{scope.name}.manifest.json":
                    return scope.with_suffix(scope.suffix + ".manifest.json")
                allowed = (
                    name == f"{run_id}.latest.json"
                    or name == f"{run_id}.api_lease.json"
                    or (name.startswith(f"{run_id}.cycle_") and name.endswith(".npz"))
                )
                if not allowed or Path(name).name != name:
                    raise operational.RemoteCommitError(
                        f"record {name!r} lies outside P1 shard scope {run_id}"
                    )
                return scope.parent / "_cycle_checkpoints" / name

            def verify_commit_record(self, record: dict[str, Any]) -> dict[str, Any]:
                if self.scope_path is None:
                    return super().verify_commit_record(record)
                return self.verify_record_for_path(record, self._intended_path(record))

            def try_delete_record(self, record: dict[str, Any]) -> str:
                """Return deleted/absent/failed while refusing cross-path deletion."""
                intended = self._intended_path(record)
                try:
                    self.verify_record_for_path(record, intended)
                except operational.RemoteCommitError as exc:
                    if not _not_found(exc):
                        raise
                    try:
                        self.metadata(str(record.get("remote_file_id", "")))
                    except Exception as metadata_exc:
                        if _not_found(metadata_exc):
                            return "absent"
                        raise
                    raise operational.RemoteCommitError(
                        "intended P1 deletion path is absent but its recorded ID "
                        "still exists elsewhere"
                    ) from exc
                try:
                    operational._execute_with_retries(
                        lambda: self.service.files().delete(
                            fileId=str(record["remote_file_id"])
                        )
                    )
                except operational.RemoteCommitError as exc:
                    warnings.warn(
                        f"P1 postcommit cleanup remains pending: {exc}", RuntimeWarning
                    )
                    return "failed"
                return "deleted"

            def delete_verified(self, record: dict[str, Any]) -> None:
                self.try_delete_record(record)

            def delete_verified_if_present(self, record: dict[str, Any]) -> bool:
                return self.try_delete_record(record) == "deleted"

        return ScopedCommitter


def _list_all_children(committer: Any, parent_id: str) -> list[dict[str, Any]]:
    operational = _operational_drive()
    token: str | None = None
    rows: list[dict[str, Any]] = []
    while True:
        response = operational._execute_with_retries(
            lambda token=token: committer.service.files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                spaces="drive",
                fields=(
                    "nextPageToken,files(id,name,parents,size,sha256Checksum,"
                    "trashed,mimeType,createdTime,modifiedTime)"
                ),
                pageSize=100,
                pageToken=token,
            )
        )
        rows.extend(dict(row) for row in response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            return rows


class _ApiShardLease:
    def __init__(self, *, runner: Any, archive: Path, run_id: str) -> None:
        self.runner = runner
        self.archive = lexical_absolute(archive)
        self.run_id = str(run_id)
        self.token = uuid.uuid4().hex
        self.hostname = os.uname().nodename
        self.pid = os.getpid()
        self.directory, _ = runner._checkpoint_paths(
            archive=self.archive, run_id=self.run_id
        )
        self.path = self.directory / f"{self.run_id}.api_lease.json"
        self.committer = runner._remote_committer(self.archive)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lost_reason: str | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": P1_API_LEASE_SCHEMA,
            "operational_hardening": P1_OPERATIONAL_HARDENING_SCHEMA,
            "run_id": self.run_id,
            "owner_token": self.token,
            "hostname": self.hostname,
            "pid": self.pid,
            "updated_unix": time.time(),
        }

    def _read_optional(self, path: Path) -> dict[str, Any] | None:
        if lexical_absolute(path) != lexical_absolute(self.path):
            raise RuntimeError("P1 API lease reader was given an unrelated path")
        items = self._lease_items(create_parent=False)
        if not items:
            return None
        return self._reconcile_lease_items(items)

    def _lease_parent_id(self, *, create: bool) -> str:
        parts = tuple(self.directory.relative_to(self.committer.drive_root).parts)
        return self.committer.resolve_folder(parts, create=create)

    def _lease_items(self, *, create_parent: bool) -> list[dict[str, Any]]:
        operational = _operational_drive()
        try:
            parent_id = self._lease_parent_id(create=create_parent)
        except operational.RemoteCommitError as exc:
            if not create_parent and _not_found(exc):
                return []
            raise
        return self.committer._list_children(parent_id, self.path.name)

    def _validated_lease_item(
        self, item: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        operational = _operational_drive()
        parent_id = self._lease_parent_id(create=False)
        metadata = self.committer.metadata(str(item.get("id", "")))
        if (
            metadata.get("name") != self.path.name
            or metadata.get("parents") != [parent_id]
            or bool(metadata.get("trashed", False))
            or metadata.get("size") is None
            or metadata.get("sha256Checksum") is None
        ):
            raise operational.RemoteCommitError(
                "P1 API lease claim is not bound to the exact lease path"
            )
        raw = self.committer.download_bytes(str(metadata["id"]))
        latest = self.committer.metadata(str(metadata["id"]))
        fingerprint = lambda row: (
            str(row.get("id")),
            str(row.get("name")),
            tuple(row.get("parents", [])),
            int(row.get("size", -1)),
            str(row.get("sha256Checksum", "")),
            bool(row.get("trashed", False)),
        )
        if fingerprint(latest) != fingerprint(metadata):
            raise operational.RemoteCommitError(
                "P1 API lease claim changed while it was being verified"
            )
        payload = json.loads(raw.decode("utf-8"))
        token = payload.get("owner_token") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != P1_API_LEASE_SCHEMA
            or payload.get("run_id") != self.run_id
            or not isinstance(token, str)
            or re.fullmatch(r"[0-9a-f]{32}", token) is None
        ):
            raise operational.RemoteCommitError(
                "P1 API lease claim has an invalid schema/run/token"
            )
        return metadata, payload

    def _delete_lease_item(
        self, metadata: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        operational = _operational_drive()
        try:
            latest, latest_payload = self._validated_lease_item(metadata)
        except operational.RemoteCommitError as exc:
            if _not_found(exc):
                return
            raise
        if (
            (
                str(latest.get("id")),
                str(latest.get("name")),
                tuple(latest.get("parents", [])),
                int(latest.get("size", -1)),
                str(latest.get("sha256Checksum", "")),
            )
            != (
                str(metadata.get("id")),
                str(metadata.get("name")),
                tuple(metadata.get("parents", [])),
                int(metadata.get("size", -1)),
                str(metadata.get("sha256Checksum", "")),
            )
            or latest_payload != payload
        ):
            raise operational.RemoteCommitError(
                "P1 API lease loser changed before exact deletion"
            )
        try:
            operational._execute_with_retries(
                lambda: self.committer.service.files().delete(
                    fileId=str(latest["id"])
                )
            )
        except operational.RemoteCommitError as exc:
            if not _not_found(exc):
                raise

    def _reconcile_lease_items(
        self, items: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Elect one deterministic owner and remove every duplicate loser."""
        operational = _operational_drive()
        current = self._lease_items(create_parent=False) if items is None else items
        for _ in range(4):
            if not current:
                raise operational.RemoteCommitError("Drive file is absent: API lease")
            validated = [self._validated_lease_item(item) for item in current]
            winner = min(
                validated,
                key=lambda row: (
                    str(row[1]["owner_token"]), str(row[0]["id"])
                ),
            )
            for metadata, payload in validated:
                if str(metadata["id"]) != str(winner[0]["id"]):
                    self._delete_lease_item(metadata, payload)
            current = self._lease_items(create_parent=False)
            if len(current) == 1:
                final_metadata, final_payload = self._validated_lease_item(current[0])
                if str(final_metadata["id"]) != str(winner[0]["id"]):
                    raise operational.RemoteCommitError(
                        "P1 API lease election winner changed during reconciliation"
                    )
                return final_payload
        raise operational.RemoteCommitError(
            "P1 API lease duplicates did not converge to one deterministic winner"
        )

    def _guard_legacy_drivefs_lease(self) -> None:
        """Refuse a fresh lease held by a currently running pre-hotfix child."""
        operational = _operational_drive()
        parts = tuple(self.directory.relative_to(self.committer.drive_root).parts)
        try:
            parent_id = self.committer.resolve_folder(parts, create=False)
        except operational.RemoteCommitError as exc:
            if _not_found(exc):
                return
            raise
        legacy_name = f"{self.run_id}.lease"
        matches = self.committer._list_children(parent_id, legacy_name)
        if len(matches) > 1:
            raise operational.RemoteCommitError(
                f"ambiguous legacy P1 lease folder: {len(matches)} matches"
            )
        if not matches:
            return
        folder_id = str(matches[0]["id"])
        children = _list_all_children(self.committer, folder_id)
        if len(children) != 1 or children[0].get("name") != "lease.json":
            raise operational.RemoteCommitError(
                "legacy P1 lease folder contains unexpected entries; refusing cleanup"
            )
        lease_file = children[0]
        payload = json.loads(
            self.committer.download_bytes(str(lease_file["id"])).decode("utf-8")
        )
        if payload.get("run_id") != self.run_id:
            raise operational.RemoteCommitError("legacy P1 lease names another run")
        age = time.time() - float(payload.get("updated_unix", 0.0))
        if age < P1_API_LEASE_STALE_SECONDS:
            raise RuntimeError(
                "a pre-hotfix P1 child still owns the fresh legacy DriveFS lease: "
                f"owner={payload}"
            )
        latest = json.loads(
            self.committer.download_bytes(str(lease_file["id"])).decode("utf-8")
        )
        latest_age = time.time() - float(latest.get("updated_unix", 0.0))
        if (
            latest.get("run_id") != self.run_id
            or latest.get("owner_token") != payload.get("owner_token")
            or latest_age < P1_API_LEASE_STALE_SECONDS
        ):
            raise RuntimeError("legacy P1 lease changed during stale-lease recovery")
        operational._execute_with_retries(
            lambda: self.committer.service.files().delete(fileId=folder_id)
        )

    def _assert_payload_owned(self, payload: dict[str, Any] | None) -> None:
        if payload is None or payload.get("owner_token") != self.token:
            raise RuntimeError(
                f"P1 API shard lease ownership was lost: observed={payload}"
            )
        if payload.get("run_id") != self.run_id:
            raise RuntimeError("P1 API shard lease run identity changed")

    def assert_owned(self) -> None:
        if self._lost_reason is not None:
            raise RuntimeError(f"P1 API shard lease heartbeat failed: {self._lost_reason}")
        self._assert_payload_owned(self._read_optional(self.path))

    def _heartbeat(self) -> None:
        operational = _operational_drive()
        while not self._stop.wait(P1_API_LEASE_HEARTBEAT_SECONDS):
            try:
                self._assert_payload_owned(self._read_optional(self.path))
                operational.publish_json(
                    self.committer, self._payload(), self.path, replace=True
                )
                self._assert_payload_owned(self._read_optional(self.path))
            except Exception as exc:
                self._lost_reason = repr(exc)
                return

    def __enter__(self) -> "_ApiShardLease":
        operational = _operational_drive()
        self._guard_legacy_drivefs_lease()
        current_items = self._lease_items(create_parent=False)
        if current_items:
            current = self._reconcile_lease_items(current_items)
            exact_items = self._lease_items(create_parent=False)
            if len(exact_items) != 1:
                raise operational.RemoteCommitError(
                    "P1 stale lease path is not uniquely bound"
                )
            current_metadata, current = self._validated_lease_item(exact_items[0])
            age = time.time() - float(current.get("updated_unix", 0.0))
            if age < P1_API_LEASE_STALE_SECONDS:
                raise RuntimeError(
                    f"another P1 writer owns the API lease: owner={current}"
                )
            latest_items = self._lease_items(create_parent=False)
            if len(latest_items) != 1:
                raise RuntimeError("P1 API lease changed during stale recovery")
            latest_metadata, latest = self._validated_lease_item(latest_items[0])
            metadata_identity = lambda row: (
                str(row.get("id")),
                str(row.get("name")),
                tuple(row.get("parents", [])),
                int(row.get("size", -1)),
                str(row.get("sha256Checksum", "")),
            )
            if (
                metadata_identity(latest_metadata) != metadata_identity(current_metadata)
                or latest.get("owner_token") != current.get("owner_token")
                or float(latest.get("updated_unix", 0.0))
                != float(current.get("updated_unix", 0.0))
                or time.time() - float(latest.get("updated_unix", 0.0))
                < P1_API_LEASE_STALE_SECONDS
            ):
                raise RuntimeError("P1 API lease changed during stale recovery")
            # Never overwrite a stale canonical object in place.  Exact-delete
            # it, then enter the same absent-claim election used by fresh runs.
            self._delete_lease_item(latest_metadata, latest)
        try:
            operational.publish_json(
                self.committer,
                self._payload(),
                self.path,
                replace=False,
            )
        except operational.RemoteCommitError:
            # A response-lost or simultaneous absent claim may already have
            # produced one or more canonical-name objects.  Election below is
            # the only permitted recovery; no science starts on ambiguity.
            self._assert_payload_owned(self._reconcile_lease_items())
        if P1_API_LEASE_SETTLE_SECONDS > 0:
            time.sleep(P1_API_LEASE_SETTLE_SECONDS)
        self.assert_owned()
        self._thread = threading.Thread(
            target=self._heartbeat,
            name=f"p1-api-lease-{self.run_id[-8:]}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            current = self._read_optional(self.path)
            if current is not None and current.get("owner_token") == self.token:
                record = self.committer.path_commit_record(self.path)
                self.committer.delete_verified_if_present(record)
        except Exception as cleanup_exc:
            warnings.warn(
                f"P1 API lease cleanup remains for expiry recovery: {cleanup_exc}",
                RuntimeWarning,
            )


def _download_record_to_cache(
    *, runner: Any, committer: Any, path: Path, record: dict[str, Any]
) -> Path:
    committer.verify_record_for_path(record, path)
    cache_root = (
        Path("/content/classA_remote_cache/p1_archive_recovery")
        if Path("/content").exists()
        else Path(tempfile.gettempdir()) / "classA_remote_cache/p1_archive_recovery"
    )
    cache = cache_root / path.name
    if (
        not cache.is_file()
        or cache.stat().st_size != int(record["remote_bytes"])
        or runner.sha256_file(cache) != record["remote_sha256"]
    ):
        committer.download_to(str(record["remote_file_id"]), cache)
    if (
        cache.stat().st_size != int(record["remote_bytes"])
        or runner.sha256_file(cache) != record["remote_sha256"]
    ):
        raise RuntimeError("downloaded P1 archive failed size/SHA verification")
    return cache


def apply_p1_hardening(runner: Any, *, operational_drive: Any) -> None:
    """Patch operational globals while leaving identity-bearing source untouched."""
    global _OPERATIONAL_DRIVE
    if getattr(runner, "_P1_OPERATIONAL_HARDENING_APPLIED", False):
        if _OPERATIONAL_DRIVE is not operational_drive:
            raise RuntimeError("P1 hardening was already installed with another helper")
        return
    required = (
        "DEFAULT_DRIVE_ROOT",
        "DriveRemoteCommitter",
        "RemoteCommitError",
        "publish_json",
        "read_remote_json",
    )
    missing = [name for name in required if not hasattr(operational_drive, name)]
    if missing:
        raise RuntimeError(f"root operational Drive helper is incomplete: {missing}")
    _OPERATIONAL_DRIVE = operational_drive
    # Frozen runner functions resolve these imported names through their module
    # globals at call time.  Substituting all of them together keeps exception
    # handling and JSON publication on one implementation.
    runner.DEFAULT_DRIVE_ROOT = operational_drive.DEFAULT_DRIVE_ROOT
    runner.DriveRemoteCommitter = operational_drive.DriveRemoteCommitter
    runner.RemoteCommitError = operational_drive.RemoteCommitError
    def publish_small_json(
        committer: Any,
        payload: dict[str, Any],
        remote_path: Path | str,
        *,
        replace: bool,
        required_headroom_bytes: int = 0,
    ) -> dict[str, Any]:
        # Checkpoint generations and final archives retain the locked 1.34-GB
        # reserve on their direct upload calls.  Once those bytes are durable,
        # their tiny pointer/receipt JSON must be publishable using its actual
        # byte count; requiring a second full reserve creates an orphan window.
        del required_headroom_bytes
        return operational_drive.publish_json(
            committer,
            payload,
            remote_path,
            replace=replace,
            required_headroom_bytes=0,
        )

    runner.publish_json = publish_small_json
    runner.read_remote_json = operational_drive.read_remote_json
    active = threading.local()
    ScopedCommitter = P1DriveRemoteCommitter.build(
        operational_drive,
        lease_getter=lambda: getattr(active, "lease", None),
    )
    original_lease = runner._ShardLease
    original_verify = runner._verify_existing
    original_write_checkpoint = runner._write_cycle_checkpoint
    original_load_checkpoint = runner._load_cycle_checkpoint
    original_cleanup = runner._cleanup_cycle_checkpoint
    original_purge = runner._purge_checkpoint_orphans
    original_archive = runner._archive
    original_root_manifest = runner._root_manifest_from_archive
    original_preflight = runner._a100_preflight

    def server_required(path: Path | str) -> bool:
        return _is_below(path, runner.DEFAULT_DRIVE_ROOT)

    def committer_for(path: Path | str) -> Any:
        if not server_required(path):
            raise RuntimeError(f"not a production Google Drive path: {path}")
        return ScopedCommitter(
            drive_root=runner.DEFAULT_DRIVE_ROOT,
            scope_path_override=path,
        )

    class HardenedLease:
        def __init__(self, *, archive: Path, run_id: str) -> None:
            self.archive = lexical_absolute(archive)
            self.run_id = str(run_id)
            current = getattr(active, "lease", None)
            self.reentrant = current is not None
            if self.reentrant:
                if (
                    lexical_absolute(getattr(current, "archive", "")) != self.archive
                    or str(getattr(current, "run_id", "")) != self.run_id
                ):
                    raise RuntimeError(
                        "P1 attempted to nest a different shard inside an active lease"
                    )
                self.impl = current
                return
            self.impl = (
                _ApiShardLease(runner=runner, archive=self.archive, run_id=self.run_id)
                if server_required(self.archive)
                else original_lease(archive=self.archive, run_id=self.run_id)
            )

        def __enter__(self) -> "HardenedLease":
            if self.reentrant:
                checker = getattr(self.impl, "assert_owned", None)
                if checker is not None:
                    checker()
                return self
            self.impl.__enter__()
            active.lease = self.impl
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            if self.reentrant:
                checker = getattr(self.impl, "assert_owned", None)
                if checker is not None:
                    checker()
                return
            try:
                self.impl.__exit__(exc_type, exc, traceback)
            finally:
                active.lease = None

    def assert_lease() -> None:
        checker = getattr(getattr(active, "lease", None), "assert_owned", None)
        if checker is not None:
            checker()

    def drivefs_free_purge(
        *, directory: Path, run_id: str, keep_generation: Path | None
    ) -> None:
        if server_required(directory):
            return
        original_purge(
            directory=directory, run_id=run_id, keep_generation=keep_generation
        )

    def purge_remote_generations(
        *, archive: Path, run_id: str, keep_name: str | None,
        strict: bool = False
    ) -> None:
        if not server_required(archive):
            return
        committer = committer_for(archive)
        directory, _ = runner._checkpoint_paths(archive=archive, run_id=run_id)
        parts = tuple(directory.relative_to(committer.drive_root).parts)
        try:
            parent_id = committer.resolve_folder(parts, create=False)
        except operational_drive.RemoteCommitError as exc:
            if _not_found(exc):
                return
            raise
        failed: list[str] = []
        for item in _list_all_children(committer, parent_id):
            name = str(item.get("name", ""))
            if (
                name == keep_name
                or not name.startswith(f"{run_id}.cycle_")
                or not name.endswith(".npz")
                or item.get("sha256Checksum") is None
                or item.get("size") is None
            ):
                continue
            if committer.try_delete_record(committer.commit_record(item)) == "failed":
                failed.append(name)
        if failed and strict:
            raise RuntimeError(
                f"could not reclaim stale P1 checkpoint generations: {failed}"
            )

    def hardened_load_checkpoint(**kwargs: Any) -> dict[str, Any] | None:
        assert_lease()
        result = original_load_checkpoint(**kwargs)
        assert_lease()
        if server_required(kwargs["archive"]):
            keep_name = None
            if result is not None:
                keep_name = str(result.get("receipt", {}).get("checkpoint", "")) or None
            # Reconcile the generation-upload/pointer-publish crash window
            # before spending another GPU cycle or allocating another ~542 MB.
            purge_remote_generations(
                archive=kwargs["archive"],
                run_id=str(kwargs["run_id"]),
                keep_name=keep_name,
                strict=True,
            )
        return result

    def hardened_write_checkpoint(**kwargs: Any) -> dict[str, Any]:
        assert_lease()
        receipt = original_write_checkpoint(**kwargs)
        assert_lease()
        if server_required(kwargs["archive"]):
            try:
                purge_remote_generations(
                    archive=kwargs["archive"],
                    run_id=str(kwargs["run_id"]),
                    keep_name=str(receipt["checkpoint"]),
                )
            except Exception as exc:
                warnings.warn(
                    f"P1 superseded-generation cleanup remains pending: {exc}",
                    RuntimeWarning,
                )
        return receipt

    def manifest_sidecar_path(archive: Path) -> Path:
        return archive.with_suffix(archive.suffix + ".manifest.json")

    def commit_identity(record: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            record.get(key)
            for key in (
                "schema",
                "remote_file_id",
                "remote_name",
                "remote_parent_id",
                "remote_bytes",
                "remote_sha256",
            )
        )

    def sidecar_payload(
        *, archive: Path, archive_record: dict[str, Any], run_id: str,
        manifest: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema": P1_MANIFEST_SIDECAR_SCHEMA,
            "bundle": runner.BUNDLE,
            "run_id": str(run_id),
            "archive": archive.name,
            "archive_remote_commit": archive_record,
            "manifest_sha256": runner.sha256_json(manifest),
            "manifest": manifest,
        }

    def validate_sidecar(
        payload: dict[str, Any], *, archive: Path,
        archive_record: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        expected = {
            "schema": P1_MANIFEST_SIDECAR_SCHEMA,
            "bundle": runner.BUNDLE,
            "run_id": str(run_id),
            "archive": archive.name,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        recorded_archive = payload.get("archive_remote_commit")
        if (
            not isinstance(recorded_archive, dict)
            or commit_identity(recorded_archive) != commit_identity(archive_record)
        ):
            mismatches["archive_remote_commit"] = ("sidecar", "exact archive")
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict):
            mismatches["manifest"] = (type(manifest).__name__, "object")
        elif payload.get("manifest_sha256") != runner.sha256_json(manifest):
            mismatches["manifest_sha256"] = (
                payload.get("manifest_sha256"),
                runner.sha256_json(manifest),
            )
        if mismatches:
            raise RuntimeError(f"P1 manifest sidecar identity mismatch: {mismatches}")
        assert isinstance(manifest, dict)
        return manifest

    def publish_manifest_sidecar(
        *, archive: Path, archive_record: dict[str, Any], run_id: str,
        manifest: dict[str, Any]
    ) -> dict[str, Any]:
        committer = committer_for(archive)
        committer.verify_record_for_path(archive_record, archive)
        path = manifest_sidecar_path(archive)
        payload = sidecar_payload(
            archive=archive,
            archive_record=archive_record,
            run_id=run_id,
            manifest=manifest,
        )
        try:
            publish_small_json(committer, payload, path, replace=False)
        except operational_drive.RemoteCommitError as publish_exc:
            # Two absent-name publications can both reach Drive before either
            # create response is visible.  The root uploader then correctly
            # reports an ambiguous final name.  Recover only if the objects
            # now visible at that exact path are valid, byte-equivalent
            # sidecars; an absent path preserves the original transport error
            # and differing valid payloads fail closed below.
            try:
                verified = read_manifest_sidecar(
                    committer=committer,
                    path=path,
                    archive=archive,
                    archive_record=archive_record,
                    run_id=run_id,
                )
            except operational_drive.RemoteCommitError as recovery_exc:
                if _not_found(recovery_exc):
                    raise publish_exc
                raise
        else:
            verified = read_manifest_sidecar(
                committer=committer,
                path=path,
                archive=archive,
                archive_record=archive_record,
                run_id=run_id,
            )
        if verified != payload:
            raise RuntimeError(
                "P1 manifest sidecar differs from the manifest being published"
            )
        return verified

    def read_manifest_sidecar(
        *, committer: Any, path: Path, archive: Path,
        archive_record: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        """Read or safely converge identical same-name sidecar objects."""
        parts, name = committer.split_remote_path(path)
        try:
            parent_id = committer.resolve_folder(parts, create=False)
        except operational_drive.RemoteCommitError as exc:
            if _not_found(exc):
                raise operational_drive.RemoteCommitError(
                    f"Drive file is absent: {path}"
                ) from exc
            raise
        def read_item(item: dict[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
            metadata = committer.metadata(str(item.get("id", "")))
            if (
                metadata.get("name") != name
                or metadata.get("parents") != [parent_id]
                or bool(metadata.get("trashed", False))
                or metadata.get("size") is None
                or metadata.get("sha256Checksum") is None
            ):
                raise operational_drive.RemoteCommitError(
                    "P1 manifest sidecar is not bound to its exact Drive path"
                )
            raw = committer.download_bytes(str(metadata["id"]))
            latest = committer.metadata(str(metadata["id"]))
            identity = lambda row: (
                str(row.get("id")),
                str(row.get("name")),
                tuple(row.get("parents", [])),
                int(row.get("size", -1)),
                str(row.get("sha256Checksum", "")),
                bool(row.get("trashed", False)),
            )
            if identity(latest) != identity(metadata):
                raise operational_drive.RemoteCommitError(
                    "P1 manifest sidecar changed during verified read"
                )
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("P1 manifest sidecar is not a JSON object")
            validate_sidecar(
                payload,
                archive=archive,
                archive_record=archive_record,
                run_id=run_id,
            )
            return metadata, raw, payload

        # A losing in-flight upload can become visible just after the first
        # list.  Re-list after a short settle and repeat so convergence is not
        # declared while a second create is still completing.
        for _ in range(5):
            items = committer._list_children(parent_id, name)
            if not items:
                raise operational_drive.RemoteCommitError(
                    f"Drive file is absent: {path}"
                )
            validated = [read_item(item) for item in items]
            canonical_raw = validated[0][1]
            if any(raw != canonical_raw for _, raw, _ in validated[1:]):
                raise RuntimeError(
                    "P1 manifest sidecar has differing same-name duplicates; "
                    "refusing repair"
                )
            if len(validated) == 1:
                return validated[0][2]
            assert_lease()
            winner = min(validated, key=lambda row: str(row[0]["id"]))
            for metadata, raw, payload in validated:
                if str(metadata["id"]) == str(winner[0]["id"]):
                    continue
                latest_metadata, latest_raw, latest_payload = read_item(metadata)
                if (
                    str(latest_metadata["id"]) != str(metadata["id"])
                    or latest_raw != raw
                    or latest_payload != payload
                ):
                    raise operational_drive.RemoteCommitError(
                        "P1 duplicate sidecar changed before exact deletion"
                    )
                try:
                    operational_drive._execute_with_retries(
                        lambda file_id=str(metadata["id"]):
                        committer.service.files().delete(fileId=file_id)
                    )
                except operational_drive.RemoteCommitError as exc:
                    if not _not_found(exc):
                        raise
            assert_lease()
            if P1_API_LEASE_SETTLE_SECONDS > 0:
                time.sleep(P1_API_LEASE_SETTLE_SECONDS)
            remaining = committer._list_children(parent_id, name)
            if len(remaining) == 1:
                final_metadata, _, final_payload = read_item(remaining[0])
                if str(final_metadata["id"]) != str(winner[0]["id"]):
                    raise operational_drive.RemoteCommitError(
                        "P1 manifest sidecar reconciliation changed its winner"
                    )
                return final_payload
        raise operational_drive.RemoteCommitError(
            "P1 manifest sidecar duplicates did not converge to one object"
        )

    def hardened_root_manifest(archive: Path) -> dict[str, Any]:
        if not server_required(archive):
            return original_root_manifest(archive)
        derived_run_id = archive.name.removesuffix(".tar.gz")
        current_lease = getattr(active, "lease", None)
        if current_lease is None:
            with HardenedLease(archive=archive, run_id=derived_run_id):
                return hardened_root_manifest(archive)
        if (
            lexical_absolute(getattr(current_lease, "archive", ""))
            != lexical_absolute(archive)
            or str(getattr(current_lease, "run_id", "")) != derived_run_id
        ):
            raise RuntimeError(
                "P1 manifest read/backfill attempted outside its exact slot lease"
            )
        committer = committer_for(archive)
        receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        receipt = operational_drive.read_remote_json(committer, receipt_path)
        archive_record = receipt.get("archive_remote_commit")
        if not isinstance(archive_record, dict):
            raise RuntimeError("P1-v4 receipt lacks remote archive metadata")
        committer.verify_record_for_path(archive_record, archive)
        run_id = str(receipt.get("run_id", ""))
        if not run_id or archive.name != f"{run_id}.tar.gz":
            raise RuntimeError("P1-v4 receipt has an invalid run/archive identity")
        path = manifest_sidecar_path(archive)
        try:
            payload = read_manifest_sidecar(
                committer=committer,
                path=path,
                archive=archive,
                archive_record=archive_record,
                run_id=run_id,
            )
        except operational_drive.RemoteCommitError as exc:
            if not _not_found(exc):
                raise
            # One-time migration for a pre-sidecar archive.  The frozen reader
            # exact-binds the archive record through the scoped committer before
            # streaming it to local cache.
            manifest = original_root_manifest(archive)
            payload = publish_manifest_sidecar(
                archive=archive,
                archive_record=archive_record,
                run_id=run_id,
                manifest=manifest,
            )
        return validate_sidecar(
            payload,
            archive=archive,
            archive_record=archive_record,
            run_id=run_id,
        )

    def hardened_archive(scratch: Path, archive: Path, run_id: str) -> dict[str, Any]:
        assert_lease()
        receipt = original_archive(scratch, archive, run_id)
        assert_lease()
        if not server_required(archive):
            return receipt
        archive_record = receipt.get("archive_remote_commit")
        if not isinstance(archive_record, dict):
            raise RuntimeError("P1-v4 archive commit lacks remote archive metadata")
        raw_manifest = json.loads(
            (Path(scratch) / "manifest.json").read_text(encoding="utf-8")
        )
        if not isinstance(raw_manifest, dict):
            raise RuntimeError("P1 root manifest is not a JSON object")
        publish_manifest_sidecar(
            archive=archive,
            archive_record=archive_record,
            run_id=run_id,
            manifest=raw_manifest,
        )
        assert_lease()
        return receipt

    def hardened_verify_existing(
        archive: Path,
        *,
        expected_run_id: str | None = None,
        expected_run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not server_required(archive):
            return original_verify(
                archive,
                expected_run_id=expected_run_id,
                expected_run_config=expected_run_config,
            )
        verification_run_id = expected_run_id
        if verification_run_id is None:
            if not archive.name.endswith(".tar.gz"):
                raise RuntimeError("P1 remote archive has no derivable run identity")
            verification_run_id = archive.name.removesuffix(".tar.gz")
        current_lease = getattr(active, "lease", None)
        if current_lease is None:
            # The overwhelmingly common startup case is a genuinely absent
            # slot.  Prove both final names absent read-only so report scans do
            # not create/delete 120 pointless API lease objects.  Any orphan,
            # receipt, or final pair takes the serialized path below.
            committer = committer_for(archive)
            receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
            try:
                committer.path_commit_record(receipt_path)
            except operational_drive.RemoteCommitError as receipt_exc:
                if not _not_found(receipt_exc):
                    raise
                try:
                    committer.path_commit_record(archive)
                except operational_drive.RemoteCommitError as archive_exc:
                    if _not_found(archive_exc):
                        return None
                    raise
            # Remote status may need to repair an orphan receipt or backfill a
            # manifest sidecar.  Serialize that mutation with the same slot
            # lease used by science, then recurse into the already-leased body.
            with HardenedLease(archive=archive, run_id=verification_run_id):
                return hardened_verify_existing(
                    archive,
                    expected_run_id=expected_run_id,
                    expected_run_config=expected_run_config,
                )
        if (
            lexical_absolute(getattr(current_lease, "archive", ""))
            != lexical_absolute(archive)
            or str(getattr(current_lease, "run_id", "")) != verification_run_id
        ):
            raise RuntimeError(
                "P1 remote verification attempted outside its exact active slot lease"
            )
        try:
            verified = original_verify(
                archive,
                expected_run_id=expected_run_id,
                expected_run_config=expected_run_config,
            )
        except RuntimeError as exc:
            if "server archive exists without its remotely verified receipt" not in str(exc):
                raise
        else:
            if verified is not None and expected_run_id is not None:
                try:
                    hardened_cleanup(archive=archive, run_id=expected_run_id)
                except Exception as cleanup_exc:
                    warnings.warn(
                        "P1 final archive is durable, but checkpoint cleanup "
                        f"remains pending: {cleanup_exc}",
                        RuntimeWarning,
                    )
            return verified
        if expected_run_id is None or expected_run_config is None:
            raise RuntimeError(
                "P1 orphan archive recovery requires the exact requested run identity"
            )
        committer = committer_for(archive)
        assert_lease()
        record = committer.path_commit_record(archive)
        committer.verify_record_for_path(record, archive)
        cache = _download_record_to_cache(
            runner=runner, committer=committer, path=archive, record=record
        )
        cache_receipt = cache.with_suffix(cache.suffix + ".receipt.json")
        cache_receipt.unlink(missing_ok=True)
        local = original_verify(
            cache,
            expected_run_id=expected_run_id,
            expected_run_config=expected_run_config,
        )
        assert local is not None
        manifest = original_root_manifest(cache)
        cache_receipt.unlink(missing_ok=True)
        recovered = {
            **local,
            "schema_version": 2,
            "archive_remote_commit": record,
            "recovered_after_interrupted_remote_receipt_commit": True,
        }
        receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        operational_drive.publish_json(
            committer,
            recovered,
            receipt_path,
            replace=False,
            # The large archive is already durable.  This repair allocates only
            # the tiny receipt and must remain possible near account capacity.
            required_headroom_bytes=0,
        )
        assert_lease()
        publish_manifest_sidecar(
            archive=archive,
            archive_record=record,
            run_id=expected_run_id,
            manifest=manifest,
        )
        assert_lease()
        verified = original_verify(
            archive,
            expected_run_id=expected_run_id,
            expected_run_config=expected_run_config,
        )
        try:
            hardened_cleanup(archive=archive, run_id=expected_run_id)
        except Exception as cleanup_exc:
            warnings.warn(
                "P1 repaired final archive is durable, but checkpoint cleanup "
                f"remains pending: {cleanup_exc}",
                RuntimeWarning,
            )
        return verified

    def hardened_preflight(**kwargs: Any) -> dict[str, Any]:
        bundle_root = lexical_absolute(kwargs["bundle_root"])
        drive_root = lexical_absolute(kwargs["drive_root"])
        config = kwargs["config"]
        bootstrap_case = next(
            case
            for case in runner.expand_cases(config)
            if case["model"]["Nx"] == 64 and case["model"]["nshell"] == 1
        )
        _, archive, run_id, _ = runner._archive_paths(
            bundle_root=bundle_root,
            config=config,
            case=bootstrap_case,
            shard_index=0,
            drive_root=drive_root,
            mode="production",
        )
        if not server_required(archive):
            return original_preflight(**kwargs)
        with HardenedLease(archive=archive, run_id=run_id):
            # A second preflight may have observed absence before waiting for
            # this lease.  Recheck exact server state after acquisition so it
            # reuses the first safe receipt instead of publishing a duplicate.
            receipt_path = runner._preflight_receipt_path(
                drive_root=drive_root, config=config
            )
            committer = committer_for(receipt_path)
            try:
                record = committer.path_commit_record(receipt_path)
                committer.verify_record_for_path(record, receipt_path)
            except operational_drive.RemoteCommitError as exc:
                if not _not_found(exc):
                    raise
            else:
                return runner._require_safe_preflight(
                    drive_root=drive_root,
                    config=config,
                    bundle_root=bundle_root,
                )
            return original_preflight(**kwargs)

    def hardened_cleanup(*, archive: Path, run_id: str) -> None:
        if not server_required(archive):
            original_cleanup(archive=archive, run_id=run_id)
            return
        committer = committer_for(archive)
        # Checkpoint deletion is authorized only by an exact, server-visible
        # final archive and receipt at this shard path.  This keeps a stray
        # cleanup call or stale copied receipt from destroying the sole
        # resumable state.
        archive_record = committer.path_commit_record(archive)
        committer.verify_record_for_path(archive_record, archive)
        final_receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        final_receipt = operational_drive.read_remote_json(
            committer, final_receipt_path
        )
        declared_archive = final_receipt.get("archive_remote_commit")
        if (
            not isinstance(declared_archive, dict)
            or commit_identity(declared_archive) != commit_identity(archive_record)
            or final_receipt.get("run_id") != run_id
            or final_receipt.get("archive") != archive.name
            or final_receipt.get("archive_sha256") != archive_record["remote_sha256"]
            or int(final_receipt.get("archive_bytes", -1))
            != int(archive_record["remote_bytes"])
        ):
            raise RuntimeError(
                "P1 final archive receipt is not exact/current; retaining checkpoint"
            )
        _, pointer = runner._checkpoint_paths(archive=archive, run_id=run_id)
        try:
            receipt = operational_drive.read_remote_json(committer, pointer)
        except operational_drive.RemoteCommitError as exc:
            if _not_found(exc):
                # A prior cleanup may have removed the pointer but crashed
                # before sweeping superseded generations.  Finishing that
                # sweep is safe now that the final archive was reverified.
                try:
                    purge_remote_generations(
                        archive=archive, run_id=run_id, keep_name=None
                    )
                except Exception as cleanup_exc:
                    warnings.warn(
                        f"P1 orphan cleanup remains pending: {cleanup_exc}",
                        RuntimeWarning,
                    )
                return
            raise
        generation = receipt.get("checkpoint_remote_commit")
        if not isinstance(generation, dict):
            raise RuntimeError("P1-v4 checkpoint cleanup lacks remote generation metadata")
        result = committer.try_delete_record(generation)
        try:
            purge_remote_generations(archive=archive, run_id=run_id, keep_name=None)
        except Exception as exc:
            warnings.warn(f"P1 orphan cleanup remains pending: {exc}", RuntimeWarning)
        if result == "failed":
            return
        pointer_record = committer.path_commit_record(pointer)
        committer.try_delete_record(pointer_record)

    runner._server_commit_required = server_required
    runner._remote_committer = committer_for
    runner._ShardLease = HardenedLease
    runner._purge_checkpoint_orphans = drivefs_free_purge
    runner._write_cycle_checkpoint = hardened_write_checkpoint
    runner._load_cycle_checkpoint = hardened_load_checkpoint
    runner._archive = hardened_archive
    runner._root_manifest_from_archive = hardened_root_manifest
    runner._a100_preflight = hardened_preflight
    runner._verify_existing = hardened_verify_existing
    runner._cleanup_cycle_checkpoint = hardened_cleanup
    runner._P1_OPERATIONAL_HARDENING_APPLIED = True
    runner._P1_OPERATIONAL_HARDENING_SCHEMA = P1_OPERATIONAL_HARDENING_SCHEMA
    runner._P1_SCOPED_COMMITTER_CLASS = ScopedCommitter
    runner._P1_OPERATIONAL_PUBLISH_JSON = publish_small_json


@contextlib.contextmanager
def lexical_drive_resolve_context(drive_root: Path | str) -> Iterator[None]:
    """Keep the frozen runner's remaining ``Path.resolve`` calls off DriveFS."""
    original = Path.resolve
    root = lexical_absolute(drive_root)

    def safe_resolve(self: Path, strict: bool = False) -> Path:
        if _is_below(self, root):
            return lexical_absolute(self)
        return original(self, strict=strict)

    Path.resolve = safe_resolve  # type: ignore[method-assign]
    try:
        yield
    finally:
        Path.resolve = original  # type: ignore[method-assign]


def checkpoint_status(
    runner: Any,
    *,
    bundle_root: Path,
    drive_root: Path,
    mode: str,
    case_id: str | None,
    shard_index: int,
) -> dict[str, Any]:
    config = runner.load_config(bundle_root)
    cases = runner.expand_cases(config)
    selected = runner.case_index(cases)
    selected_id = case_id or str(cases[0]["case_id"])
    if selected_id not in selected:
        raise KeyError(f"unknown P1 case {selected_id!r}")
    case = selected[selected_id]
    _, archive, run_id, run_config = runner._archive_paths(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=int(shard_index),
        drive_root=drive_root,
        mode=mode,
    )
    final_archive = runner._verify_existing(
        archive,
        expected_run_id=run_id,
        expected_run_config=run_config,
    )
    identity = runner._checkpoint_identity(
        run_id=run_id,
        run_config=run_config,
        case=case,
        shard_index=int(shard_index),
        global_sample_ids=runner.global_sample_indices(case, int(shard_index)),
    )
    directory, pointer = runner._checkpoint_paths(archive=archive, run_id=run_id)
    committer = runner._remote_committer(archive)
    operational = _operational_drive()
    base = {
        "bundle": runner.BUNDLE,
        "case_id": selected_id,
        "shard_index": int(shard_index),
        "run_id": run_id,
        "pointer": str(pointer),
    }
    if final_archive is not None:
        return {
            "exists": False,
            **base,
            "final_archive_verified": True,
            "checkpoint_cleanup_reconciled": True,
        }
    try:
        payload = operational.read_remote_json(committer, pointer)
    except operational.RemoteCommitError as exc:
        if _not_found(exc):
            return {"exists": False, **base}
        raise
    expected = {
        "schema": runner.P1_CHECKPOINT_POINTER_SCHEMA,
        "checkpoint_schema": runner.P1_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "checkpoint_identity_sha256": runner.sha256_json(identity),
        "total_cycles": int(identity["total_cycles"]),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"P1 checkpoint pointer identity mismatch: {mismatches}")
    filename = str(payload.get("checkpoint", ""))
    if not filename or Path(filename).name != filename:
        raise RuntimeError("P1 checkpoint pointer has an invalid payload filename")
    record = payload.get("checkpoint_remote_commit")
    if not isinstance(record, dict):
        raise RuntimeError("P1-v4 pointer lacks its remote checkpoint commit")
    committer.verify_record_for_path(record, directory / filename)
    if (
        int(record.get("remote_bytes", -1)) != int(payload.get("checkpoint_bytes", -2))
        or record.get("remote_sha256") != payload.get("checkpoint_sha256")
    ):
        raise RuntimeError("P1-v4 checkpoint pointer disagrees with Drive metadata")
    completed = int(payload.get("completed_cycle", -1))
    total = int(payload.get("total_cycles", -1))
    if not 0 <= completed <= total:
        raise RuntimeError("P1 checkpoint has an invalid completed-cycle index")
    return {
        "exists": True,
        **base,
        "pointer_remote_commit": committer.path_commit_record(pointer),
        "completed_cycle": completed,
        "total_cycles": total,
        "checkpoint": filename,
        "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        "checkpoint_bytes": int(payload["checkpoint_bytes"]),
        "checkpoint_remote_commit": record,
        "resumable": True,
        "metadata_verified": True,
        "operational_hardening": P1_OPERATIONAL_HARDENING_SCHEMA,
    }


def repair_orphan_status(
    runner: Any,
    *,
    bundle_root: Path,
    drive_root: Path,
    mode: str,
    case_id: str | None,
    shard_index: int,
) -> dict[str, Any]:
    """Idempotently repair/backfill one P1 final archive under its slot lease."""
    config = runner.load_config(bundle_root)
    cases = runner.expand_cases(config)
    selected = runner.case_index(cases)
    selected_id = case_id or str(cases[0]["case_id"])
    if selected_id not in selected:
        raise KeyError(f"unknown P1 case {selected_id!r}")
    case = selected[selected_id]
    _, archive, run_id, run_config = runner._archive_paths(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=int(shard_index),
        drive_root=drive_root,
        mode=mode,
    )
    receipt = runner._verify_existing(
        archive,
        expected_run_id=run_id,
        expected_run_config=run_config,
    )
    if receipt is None:
        return {
            "status": "no_final_archive",
            "repaired": False,
            "archive": str(archive),
            "case_id": selected_id,
            "shard_index": int(shard_index),
            "run_id": run_id,
        }
    return {
        "status": "verified_or_repaired",
        "repaired": bool(
            receipt.get("recovered_after_interrupted_remote_receipt_commit")
        ),
        "archive": str(archive),
        "case_id": selected_id,
        "shard_index": int(shard_index),
        "run_id": run_id,
        "archive_sha256": receipt.get("archive_sha256"),
        "server_verified": True,
    }


def _operational_slot_parser(
    *, bundle_root: Path, flag: str, description: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--bundle-root", type=Path, default=bundle_root)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("production", "pilot", "smoke"), default="production"
    )
    parser.add_argument("--case-id")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(flag, action="store_true")
    return parser


def _run_fail_closed_a100_preflight(
    runner: Any, values: list[str]
) -> int | None:
    """Handle production qualification without the frozen broad fallback.

    The frozen CLI intentionally treated every read/validation failure as a
    missing receipt.  That makes a transient API failure, duplicate name, or
    corrupt receipt launch another costly L64 trajectory.  The wrapper permits
    a new qualification only after an explicit server-absent result.
    """
    args = runner.build_parser().parse_args(values)
    if args.list_cases or args.list_cases_json:
        return None
    bundle_root = lexical_absolute(args.bundle_root)
    drive_root = lexical_absolute(args.drive_root)
    config = runner.load_config(bundle_root)
    receipt_path = runner._preflight_receipt_path(
        drive_root=drive_root, config=config
    )
    if not runner._server_commit_required(receipt_path):
        return None
    committer = runner._remote_committer(receipt_path)
    explicit_absence = False
    try:
        record = committer.path_commit_record(receipt_path)
        committer.verify_record_for_path(record, receipt_path)
    except runner.RemoteCommitError as exc:
        if not _not_found(exc):
            raise
        explicit_absence = True

    if not explicit_absence:
        current = runner._require_safe_preflight(
            drive_root=drive_root,
            config=config,
            bundle_root=bundle_root,
        )
        current = dict(current)
        current["status"] = "reused_current_safe_receipt"
        current["receipt_path"] = str(receipt_path)
        print(json.dumps(current, indent=2, sort_keys=True))
        return 0

    result = runner._a100_preflight(
        bundle_root=bundle_root,
        config=config,
        drive_root=drive_root,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    if result.get("status") == "checkpointed_partial":
        runner._print_session_checkpoint(result)
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("safe", False):
        raise RuntimeError("P1 A100 qualification completed but was not safe")
    return 0


def run_hardened_main(runner: Any, argv: list[str] | None, *, bundle_root: Path) -> int:
    values = list(argv or [])
    if "--repair-orphan" in values:
        parser = _operational_slot_parser(
            bundle_root=bundle_root,
            flag="--repair-orphan",
            description="Repair one server-visible P1 archive/receipt/sidecar",
        )
        args = parser.parse_args(values)
        with lexical_drive_resolve_context(args.drive_root):
            result = repair_orphan_status(
                runner,
                bundle_root=lexical_absolute(args.bundle_root),
                drive_root=lexical_absolute(args.drive_root),
                mode=str(args.mode),
                case_id=args.case_id,
                shard_index=int(args.shard_index),
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    if "--checkpoint-status" in values:
        parser = _operational_slot_parser(
            bundle_root=bundle_root,
            flag="--checkpoint-status",
            description="Report one server-verified P1 cycle checkpoint",
        )
        args = parser.parse_args(values)
        with lexical_drive_resolve_context(args.drive_root):
            result = checkpoint_status(
                runner,
                bundle_root=lexical_absolute(args.bundle_root),
                drive_root=lexical_absolute(args.drive_root),
                mode=str(args.mode),
                case_id=args.case_id,
                shard_index=int(args.shard_index),
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    drive_root = Path("/content/drive/MyDrive")
    if "--drive-root" in values:
        index = values.index("--drive-root")
        if index + 1 >= len(values):
            raise ValueError("--drive-root requires a value")
        drive_root = Path(values[index + 1])
    with lexical_drive_resolve_context(drive_root):
        if "--a100-preflight" in values:
            handled = _run_fail_closed_a100_preflight(runner, values)
            if handled is not None:
                return handled
        return int(runner.main(values))
