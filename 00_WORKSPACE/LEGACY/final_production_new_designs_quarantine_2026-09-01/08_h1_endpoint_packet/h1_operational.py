"""Unhashed operational hardening for the immutable H1-v4 scientific bundle.

This module deliberately lives outside ``src``.  H1-v4 archives and the saved
A100 qualification pin every Python file below ``src``; changing those files
would invalidate the completed qualification.  The helpers here strengthen
transport/recovery behavior without changing the scientific implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Iterator


MOUNT_LOSS_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "ENOTCONN", None),
        getattr(errno, "EIO", None),
        getattr(errno, "ESTALE", None),
        getattr(errno, "ENODEV", None),
    )
    if value is not None
)
COMMIT_IDENTITY_FIELDS = (
    "schema",
    "remote_file_id",
    "remote_name",
    "remote_parent_id",
    "remote_bytes",
    "remote_sha256",
)
_ROOT_DRIVE_HELPER: Any | None = None
H1_API_LEASE_SCHEMA = "h1_api_shard_single_writer_lease_v1"
H1_REMOTE_STATUS_SIDECAR_SCHEMA = "h1_v4_remote_status_sidecar_v1"
H1_API_LEASE_HEARTBEAT_SECONDS = 30.0
H1_API_LEASE_STALE_SECONDS = 900.0
H1_API_LEASE_SETTLE_SECONDS = 0.5


class OrphanArchiveError(RuntimeError):
    """A final server archive exists, but its completion receipt does not."""

    def __init__(self, archive: Path, archive_record: dict[str, Any]) -> None:
        super().__init__(
            f"server H1-v4 archive exists without its receipt: {archive}; "
            "run the same bundle command with --repair-orphan"
        )
        self.archive = Path(archive)
        self.archive_record = dict(archive_record)


def _local_runtime_root() -> Path:
    """Return Colab's local-disk root, with a host-test fallback."""

    content_root = Path("/content")
    return content_root if content_root.is_dir() else Path(tempfile.gettempdir())


def lexical_absolute(path: Path | str) -> Path:
    """Normalize a path without consulting a disconnected DriveFS mount."""

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


@contextlib.contextmanager
def lexical_drive_resolve_context(drive_root: Path | str) -> Iterator[None]:
    """Keep the pinned runner's remaining ``Path.resolve`` calls off DriveFS."""

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


def _load_root_drive_helper() -> Any:
    """Load the deployment-root transport without importing the pinned copy."""

    global _ROOT_DRIVE_HELPER
    if _ROOT_DRIVE_HELPER is not None:
        return _ROOT_DRIVE_HELPER
    helper_path = Path(__file__).resolve().parent.parent / "drive_remote_commit.py"
    if not helper_path.is_file():
        raise RuntimeError(f"deployment-root Drive helper is missing: {helper_path}")
    spec = importlib.util.spec_from_file_location(
        "classA_h1_operational_drive_remote_commit", helper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load deployment-root Drive helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "DriveRemoteCommitter",
        "RemoteCommitError",
        "publish_json",
        "read_remote_json",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"deployment-root Drive helper is incomplete: {missing}")
    _ROOT_DRIVE_HELPER = module
    return module


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _is_absent_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "absent" in message or "not found" in message or "404" in message


def _is_concurrent_name_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "ambiguous" in message
        or re.search(r"produced\s+[2-9][0-9]*\s+files\s+named", message) is not None
        or "won by a different concurrent upload" in message
    )


def _commit_identity(record: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in COMMIT_IDENTITY_FIELDS if name not in record]
    if missing:
        raise RuntimeError(f"remote commit record is incomplete: {missing}")
    return {name: record[name] for name in COMMIT_IDENTITY_FIELDS}


def _refresh_exact_named_record(
    committer: Any,
    record: dict[str, Any],
    *,
    expected_name: str,
    expected_parent_id: str,
) -> dict[str, Any]:
    """Rebind an ID to its intended exact name and parent after refresh."""

    metadata = committer.metadata(str(record["remote_file_id"]))
    refreshed = committer.commit_record(metadata)
    if _commit_identity(refreshed) != _commit_identity(record):
        raise RuntimeError("Drive object changed between list and exact-ID refresh")
    if (
        str(refreshed["remote_name"]) != str(expected_name)
        or str(refreshed["remote_parent_id"]) != str(expected_parent_id)
    ):
        raise RuntimeError("Drive object moved outside the intended exact path")
    committer.verify_commit_record(refreshed)
    return refreshed


def _read_json_by_record(
    committer: Any, record: dict[str, Any], *, expected_name: str
) -> dict[str, Any]:
    identity = _commit_identity(record)
    if str(identity["remote_name"]) != str(expected_name):
        raise RuntimeError(
            "Drive record has the wrong final name: "
            f"{identity['remote_name']!r} != {expected_name!r}"
        )
    committer.verify_commit_record(record)
    raw = committer.download_bytes(str(identity["remote_file_id"]))
    if len(raw) != int(identity["remote_bytes"]):
        raise RuntimeError("downloaded Drive JSON has the wrong byte count")
    if hashlib.sha256(raw).hexdigest() != str(identity["remote_sha256"]):
        raise RuntimeError("downloaded Drive JSON failed its server checksum")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("remote H1 receipt is not a JSON object")
    return payload


def _converge_identical_json_duplicates(
    runner: Any, committer: Any, path: Path | str
) -> dict[str, Any]:
    """Elect one byte-identical JSON object after a concurrent absent upload."""

    path = Path(path)
    parts, name = committer.split_remote_path(path)
    try:
        parent_id = committer.resolve_folder(parts, create=False)
    except runner.RemoteCommitError as exc:
        if _is_absent_error(exc):
            raise RuntimeError(f"Drive JSON vanished during duplicate recovery: {path}") from exc
        raise
    for _ in range(5):
        records: list[tuple[dict[str, Any], bytes]] = []
        for item in committer._list_children(parent_id, name):
            try:
                metadata = committer.metadata(str(item["id"]))
                record = committer.commit_record(metadata)
                record = _refresh_exact_named_record(
                    committer,
                    record,
                    expected_name=name,
                    expected_parent_id=parent_id,
                )
                raw = committer.download_bytes(str(record["remote_file_id"]))
            except runner.RemoteCommitError as exc:
                if _is_absent_error(exc):
                    continue
                raise
            if len(raw) != int(record["remote_bytes"]):
                raise RuntimeError(f"duplicate Drive JSON has the wrong size: {path}")
            if hashlib.sha256(raw).hexdigest() != str(record["remote_sha256"]):
                raise RuntimeError(f"duplicate Drive JSON failed its checksum: {path}")
            # Require valid object-shaped JSON before treating the duplicate as
            # a harmless concurrent publication.
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"duplicate Drive JSON is not an object: {path}")
            records.append((record, raw))
        if not records:
            raise RuntimeError(f"Drive JSON vanished during duplicate recovery: {path}")
        if len({raw for _, raw in records}) != 1:
            raise RuntimeError(
                f"ambiguous Drive JSON contains non-identical payloads: {path}"
            )
        winner = min(records, key=lambda row: str(row[0]["remote_file_id"]))[0]
        winner_id = str(winner["remote_file_id"])
        for record, expected_raw in records:
            if str(record["remote_file_id"]) != winner_id:
                refreshed = _refresh_exact_named_record(
                    committer,
                    record,
                    expected_name=name,
                    expected_parent_id=parent_id,
                )
                raw = committer.download_bytes(str(refreshed["remote_file_id"]))
                if raw != expected_raw:
                    raise RuntimeError(
                        f"duplicate Drive JSON changed before exact-ID deletion: {path}"
                    )
                committer.delete_verified_if_present(refreshed)
        if H1_API_LEASE_SETTLE_SECONDS > 0:
            time.sleep(H1_API_LEASE_SETTLE_SECONDS)
        remaining = [
            item
            for item in committer._list_children(parent_id, name)
            if str(item["id"]) == winner_id
        ]
        all_current = committer._list_children(parent_id, name)
        if len(remaining) == 1 and len(all_current) == 1:
            current = committer.commit_record(
                committer.metadata(str(remaining[0]["id"]))
            )
            current = _refresh_exact_named_record(
                committer,
                current,
                expected_name=name,
                expected_parent_id=parent_id,
            )
            raw = committer.download_bytes(str(current["remote_file_id"]))
            if raw != records[0][1]:
                raise RuntimeError(
                    f"elected Drive JSON changed before exact return: {path}"
                )
            return current
    raise RuntimeError(f"identical Drive JSON duplicates did not converge: {path}")


def _path_record_with_identical_duplicate_recovery(
    runner: Any, committer: Any, path: Path | str
) -> dict[str, Any]:
    try:
        return committer.path_commit_record(path)
    except runner.RemoteCommitError as exc:
        if not _is_concurrent_name_error(exc):
            raise
        return _converge_identical_json_duplicates(runner, committer, path)


def _run_id_from_archive(archive: Path, bundle: str) -> str:
    suffix = ".tar.gz"
    if not archive.name.endswith(suffix):
        raise RuntimeError(f"H1 archive has the wrong suffix: {archive}")
    run_id = archive.name[: -len(suffix)]
    if not run_id.startswith(f"{bundle}_") or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", run_id
    ):
        raise RuntimeError(f"unsafe H1 run ID derived from {archive.name!r}")
    return run_id


def _local_scratch_root(bundle: str, run_id: str) -> Path:
    parent = _local_runtime_root() / "classA_final_production" / bundle
    target = parent / run_id
    target.relative_to(parent)
    return target


def cleanup_local_after_remote_verification(
    *, bundle: str, archive: Path, receipt: dict[str, Any]
) -> list[str]:
    """Remove local scratch/staging only after a bound remote receipt passed."""

    run_id = _run_id_from_archive(archive, bundle)
    if receipt.get("run_id") != run_id or receipt.get("archive") != archive.name:
        raise RuntimeError("refusing cleanup for a receipt that names another H1 run")
    removed: list[str] = []
    scratch = _local_scratch_root(bundle, run_id)
    if scratch.exists():
        shutil.rmtree(scratch)
        removed.append(str(scratch))
    stage_root = _local_runtime_root() / "classA_remote_stage" / "h1_archives"
    if stage_root.is_dir():
        for path in sorted(stage_root.glob(f".{run_id}.*.tar.gz")):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def verify_remote_existing(runner: Any, archive: Path | str) -> dict[str, Any] | None:
    """Verify an H1 archive/receipt pair at its exact expected Drive paths."""

    archive = Path(archive)
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    committer = runner._remote_committer(archive)
    try:
        receipt_record = committer.path_commit_record(receipt_path)
    except runner.RemoteCommitError as exc:
        if not _is_absent_error(exc):
            raise
        try:
            archive_record = committer.path_commit_record(archive)
        except runner.RemoteCommitError as archive_exc:
            if _is_absent_error(archive_exc):
                return None
            raise
        if str(archive_record.get("remote_name")) != archive.name:
            raise RuntimeError("orphan Drive archive has the wrong final name")
        raise OrphanArchiveError(archive, archive_record) from exc

    receipt = _read_json_by_record(
        committer, receipt_record, expected_name=receipt_path.name
    )
    try:
        archive_record = committer.path_commit_record(archive)
    except runner.RemoteCommitError as exc:
        if _is_absent_error(exc):
            raise RuntimeError(
                f"server H1-v4 receipt exists without its archive: {receipt_path}"
            ) from exc
        raise

    if str(archive_record.get("remote_name")) != archive.name:
        raise RuntimeError("Drive archive record has the wrong final name")
    if str(receipt_record.get("remote_parent_id")) != str(
        archive_record.get("remote_parent_id")
    ):
        raise RuntimeError("H1 archive and receipt are in different Drive parents")
    declared = receipt.get("archive_remote_commit")
    if not isinstance(declared, dict):
        raise RuntimeError("H1-v4 receipt lacks remote archive metadata")
    if _commit_identity(declared) != _commit_identity(archive_record):
        raise RuntimeError(
            "H1-v4 receipt archive record is not bound to the expected Drive path"
        )
    committer.verify_commit_record(archive_record)

    run_id = _run_id_from_archive(archive, runner.BUNDLE)
    if (
        receipt.get("run_id") != run_id
        or receipt.get("archive") != archive.name
        or receipt.get("archive_sha256") != archive_record.get("remote_sha256")
        or int(receipt.get("archive_bytes", -1))
        != int(archive_record.get("remote_bytes", -2))
    ):
        raise RuntimeError("H1-v4 receipt disagrees with its exact Drive archive")
    cleanup_local_after_remote_verification(
        bundle=runner.BUNDLE, archive=archive, receipt=receipt
    )
    return receipt


def _manifest_sidecar_path(archive: Path) -> Path:
    return archive.with_suffix(archive.suffix + ".status.json")


def publish_manifest_sidecar(
    runner: Any,
    *,
    archive: Path,
    receipt: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Publish a deterministic, archive-bound manifest for cheap status reads."""

    archive = Path(archive)
    committer = runner._remote_committer(archive)
    archive_record = committer.path_commit_record(archive)
    declared_archive = receipt.get("archive_remote_commit")
    if not isinstance(declared_archive, dict) or _commit_identity(
        archive_record
    ) != _commit_identity(declared_archive):
        raise RuntimeError("cannot sidecar a receipt misbound to its Drive archive")
    if hasattr(committer, "verify_record_for_path"):
        committer.verify_record_for_path(archive_record, archive)
    else:
        committer.verify_commit_record(archive_record)
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    receipt_record = committer.path_commit_record(receipt_path)
    server_receipt = _read_json_by_record(
        committer,
        receipt_record,
        expected_name=receipt_path.name,
    )
    expected_server_receipt = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_remote_commit"
    }
    if _normalized(server_receipt) != _normalized(expected_server_receipt):
        raise RuntimeError("cannot sidecar a receipt that differs from Drive")
    payload = {
        "schema": H1_REMOTE_STATUS_SIDECAR_SCHEMA,
        "bundle": runner.BUNDLE,
        "archive": archive.name,
        "run_id": str(receipt["run_id"]),
        "archive_remote_commit": _commit_identity(archive_record),
        "receipt_remote_commit": _commit_identity(receipt_record),
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "manifest": manifest,
    }
    sidecar = _manifest_sidecar_path(archive)
    try:
        commit = runner.publish_json(
            committer,
            payload,
            sidecar,
            replace=False,
            required_headroom_bytes=0,
        )
    except runner.RemoteCommitError as exc:
        if not _is_concurrent_name_error(exc):
            raise
        commit = _converge_identical_json_duplicates(runner, committer, sidecar)
    exact = _path_record_with_identical_duplicate_recovery(
        runner, committer, sidecar
    )
    if _commit_identity(exact) != _commit_identity(commit):
        raise RuntimeError("H1 status sidecar changed during publication")
    return {**payload, "sidecar_remote_commit": exact}


def read_manifest_sidecar(runner: Any, archive: Path) -> dict[str, Any] | None:
    """Return a fully bound sidecar, or ``None`` only on explicit absence."""

    archive = Path(archive)
    sidecar = _manifest_sidecar_path(archive)
    committer = runner._remote_committer(archive)
    try:
        sidecar_record = _path_record_with_identical_duplicate_recovery(
            runner, committer, sidecar
        )
    except runner.RemoteCommitError as exc:
        if _is_absent_error(exc):
            return None
        raise
    payload = _read_json_by_record(
        committer, sidecar_record, expected_name=sidecar.name
    )
    if (
        payload.get("schema") != H1_REMOTE_STATUS_SIDECAR_SCHEMA
        or payload.get("bundle") != runner.BUNDLE
        or payload.get("archive") != archive.name
    ):
        raise RuntimeError("H1 remote-status sidecar has the wrong identity")
    archive_record = payload.get("archive_remote_commit")
    receipt_record = payload.get("receipt_remote_commit")
    manifest = payload.get("manifest")
    if not all(isinstance(row, dict) for row in (archive_record, receipt_record, manifest)):
        raise RuntimeError("H1 remote-status sidecar is incomplete")
    current_archive = committer.path_commit_record(archive)
    if _commit_identity(current_archive) != _commit_identity(archive_record):
        raise RuntimeError("H1 remote-status sidecar names another archive commit")
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    current_receipt = committer.path_commit_record(receipt_path)
    if _commit_identity(current_receipt) != _commit_identity(receipt_record):
        raise RuntimeError("H1 remote-status sidecar names another receipt commit")
    receipt = _read_json_by_record(
        committer, current_receipt, expected_name=receipt_path.name
    )
    if _commit_identity(receipt.get("archive_remote_commit", {})) != _commit_identity(
        current_archive
    ):
        raise RuntimeError("H1 sidecar receipt is not bound to its archive")
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != payload.get("manifest_sha256"):
        raise RuntimeError("H1 remote-status sidecar manifest checksum failed")
    return manifest


def replay_lookup_with_mount_fallback(
    lookup: Callable[..., tuple[dict[str, Any] | None, list[str]]],
    /,
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Use the registered fresh-seed fallback when DriveFS loses its mount."""

    try:
        return lookup(**kwargs)
    except OSError as exc:
        if exc.errno not in MOUNT_LOSS_ERRNOS:
            raise
        reason = (
            "response_replay_drivefs_unavailable:"
            f"errno={exc.errno}:fresh_same_preregistered_seed"
        )
        print(
            "[H1 OPERATIONAL WARNING] "
            + json.dumps(
                {
                    "action": "fresh_same_preregistered_seed",
                    "reason": reason,
                    "source_replay_attempted": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return None, [reason]


def _root_manifest(local_archive: Path) -> dict[str, Any]:
    with tarfile.open(local_archive, "r:gz") as handle:
        matches = [
            member
            for member in handle.getmembers()
            if member.isfile() and member.name.lstrip("./") == "manifest.json"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"orphan archive must contain one root manifest, found {len(matches)}"
            )
        stream = handle.extractfile(matches[0])
        if stream is None:
            raise RuntimeError("orphan archive manifest is unreadable")
        payload = json.loads(stream.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("orphan archive manifest is not a JSON object")
    return payload


def _member_payload(handle: tarfile.TarFile, suffix: str) -> bytes:
    suffix = suffix.lstrip("/")
    matches = [
        member
        for member in handle.getmembers()
        if member.isfile() and member.name.lstrip("./").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"orphan archive expected one member ending in {suffix!r}, "
            f"found {len(matches)}"
        )
    stream = handle.extractfile(matches[0])
    if stream is None:
        raise RuntimeError(f"orphan archive member is unreadable: {suffix}")
    return stream.read()


def _validate_internal_products(
    local_archive: Path, manifest: dict[str, Any]
) -> None:
    products = manifest.get("products")
    if not isinstance(products, dict):
        raise RuntimeError("orphan archive manifest lacks products")
    packet = products.get("h1_endpoint_packet")
    record = products.get("ordered_born_record")
    if not isinstance(packet, dict) or not isinstance(record, dict):
        raise RuntimeError("orphan archive lacks H1 packet or ordered-record products")
    file_rows = packet.get("files")
    if not isinstance(file_rows, list) or len(file_rows) != 3:
        raise RuntimeError("orphan H1 packet product must declare exactly three files")
    with tarfile.open(local_archive, "r:gz") as handle:
        for row in file_rows:
            if not isinstance(row, dict):
                raise RuntimeError("orphan packet file declaration is invalid")
            name = Path(str(row.get("path", ""))).name
            if not name:
                raise RuntimeError("orphan packet file declaration has no name")
            raw = _member_payload(handle, f"h1_endpoint_packet/{name}")
            if len(raw) != int(row.get("bytes", -1)):
                raise RuntimeError(f"orphan packet member has wrong size: {name}")
            if hashlib.sha256(raw).hexdigest() != row.get("sha256"):
                raise RuntimeError(f"orphan packet member failed checksum: {name}")
        record_name = Path(str(record.get("path", ""))).name
        if record_name != "ordered_born_record.npz":
            raise RuntimeError("orphan ordered-record product has the wrong path")
        raw = _member_payload(handle, record_name)
        if len(raw) != int(record.get("bytes", -1)):
            raise RuntimeError("orphan ordered record has the wrong size")
        if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
            raise RuntimeError("orphan ordered record failed checksum")


def _validate_orphan_manifest(
    runner: Any,
    *,
    manifest: dict[str, Any],
    bundle_root: Path,
    config: dict[str, Any],
    case: dict[str, Any],
    shard_index: int,
    run_id: str,
    run_config: dict[str, Any],
) -> None:
    source_hashes = runner._source_hashes(bundle_root / "src")
    global_ids = list(
        range(int(shard_index) * runner.SHARD_SIZE, (int(shard_index) + 1) * runner.SHARD_SIZE)
    )
    expected = {
        "schema_version": 2,
        "status": "complete_local",
        "bundle": runner.BUNDLE,
        "sampling_revision": runner.REVISION,
        "audit_sha256": runner.AUDIT,
        "bundle_source_hashes_sha256": runner.sha256_json(source_hashes),
        "canonical_entry_point": runner.ENTRY_POINT,
        "canonical_engine_sha256": run_config["canonical_engine_sha256"],
        "run_config": run_config,
        "run_config_hash": runner.sha256_json(run_config),
        "root_seed": int(config["root_seed"]),
        "case_id": case["case_id"],
        "protocol": case["protocol"],
        "alpha_1": case["model"]["alpha_1"],
        "shard_index": int(shard_index),
        "global_sample_indices": global_ids,
        "shard_generator_seed": runner.shard_seed(
            int(config["root_seed"]), case["case_id"], shard_index
        ),
        "source_hashes": source_hashes,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if _normalized(manifest.get(key)) != _normalized(value)
    }
    if mismatches:
        raise RuntimeError(
            "orphan archive is not the exact current H1-v4 shard: "
            + ", ".join(sorted(mismatches))
        )
    numerical_status = manifest.get("numerical_status")
    if numerical_status not in {"pass", "warning"}:
        raise RuntimeError(
            f"orphan archive has non-completing numerical status {numerical_status!r}"
        )
    gpu = manifest.get("gpu_preflight", {})
    if "A100" not in str(gpu.get("device", "")).upper():
        raise RuntimeError("orphan production archive was not produced on an A100")
    if run_id != f"{runner.BUNDLE}_{runner.sha256_json(run_config)[:16]}":
        raise RuntimeError("orphan archive filename disagrees with its locked run config")


def _repair_orphan_core(
    runner: Any,
    *,
    bundle_root: Path | str,
    drive_root: Path | str,
    mode: str,
    case_id: str,
    shard_index: int,
    lease: "_H1ApiShardLease | None" = None,
) -> dict[str, Any]:
    """Validate an orphan final archive, publish its receipt, and verify it."""

    bundle_root = Path(bundle_root).resolve()
    drive_root = lexical_absolute(drive_root)
    config = runner.load_config(bundle_root)
    cases = {row["case_id"]: row for row in runner.expand_cases(config)}
    if case_id not in cases:
        raise KeyError(f"unknown H1-v4 case {case_id!r}")
    if not 0 <= int(shard_index) < 5:
        raise IndexError("H1-v4 shard index must lie in 0..4")
    case = cases[case_id]
    scratch, archive, run_id, run_config = runner._archive_paths(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=int(shard_index),
        drive_root=drive_root,
        mode=mode,
    )
    if not runner._server_commit_required(archive):
        raise RuntimeError("orphan repair is allowed only for a Google Drive API target")
    try:
        current = verify_remote_existing(runner, archive)
    except OrphanArchiveError as orphan:
        archive_record = orphan.archive_record
    else:
        if current is None:
            return {
                "status": "no_final_archive",
                "repaired": False,
                "archive": str(archive),
            }
        return {
            "status": "already_verified",
            "repaired": False,
            "archive": str(archive),
            "archive_sha256": current["archive_sha256"],
        }

    committer = runner._remote_committer(archive)
    archive_record = committer.path_commit_record(archive)
    if str(archive_record.get("remote_name")) != archive.name:
        raise RuntimeError("refusing to repair an archive under the wrong Drive name")
    repair_root = _local_runtime_root() / "classA_remote_repair" / "h1"
    repair_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{run_id}.", dir=repair_root) as raw:
        local_archive = Path(raw) / archive.name
        committer.download_to(str(archive_record["remote_file_id"]), local_archive)
        if local_archive.stat().st_size != int(archive_record["remote_bytes"]):
            raise RuntimeError("downloaded orphan archive has the wrong byte count")
        if runner.sha256_file(local_archive) != archive_record["remote_sha256"]:
            raise RuntimeError("downloaded orphan archive failed its server checksum")
        manifest = _root_manifest(local_archive)
        _validate_orphan_manifest(
            runner,
            manifest=manifest,
            bundle_root=bundle_root,
            config=config,
            case=case,
            shard_index=int(shard_index),
            run_id=run_id,
            run_config=run_config,
        )
        _validate_internal_products(local_archive, manifest)

    receipt = {
        "schema_version": 2,
        "run_id": run_id,
        "archive": archive.name,
        "archive_sha256": archive_record["remote_sha256"],
        "archive_bytes": int(archive_record["remote_bytes"]),
        "created_unix": float(manifest["created_unix"]),
        "archive_remote_commit": archive_record,
    }
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if lease is not None:
        lease.assert_owned()
    receipt_commit = runner.publish_json(
        committer,
        receipt,
        receipt_path,
        replace=False,
        required_headroom_bytes=runner.H1_REMOTE_REQUIRED_HEADROOM_BYTES,
    )
    if lease is not None:
        lease.assert_owned()
    publish_manifest_sidecar(
        runner,
        archive=archive,
        receipt=receipt,
        manifest=manifest,
    )
    if lease is not None:
        lease.assert_owned()
    verified = verify_remote_existing(runner, archive)
    if verified is None:
        raise RuntimeError("orphan receipt publication did not produce a durable pair")
    if verified.get("archive_sha256") != archive_record["remote_sha256"]:
        raise RuntimeError("repaired receipt resolved to a different archive")
    # ``verify_remote_existing`` performs cleanup only after the exact remote
    # pair has passed name/parent/file-ID/size/SHA binding.
    return {
        "status": "repaired_and_verified",
        "repaired": True,
        "archive": str(archive),
        "archive_sha256": verified["archive_sha256"],
        "receipt": str(receipt_path),
        "receipt_remote_commit": receipt_commit,
        "cleaned_scratch": not scratch.exists(),
    }


class _H1ApiShardLease:
    """Single-writer lease stored and heartbeated through the Drive API."""

    def __init__(
        self,
        *,
        runner: Any,
        archive: Path,
        run_id: str,
        case_id: str,
        shard_index: int,
    ) -> None:
        self.runner = runner
        self.archive = lexical_absolute(archive)
        self.run_id = str(run_id)
        self.case_id = str(case_id)
        self.shard_index = int(shard_index)
        self.token = uuid.uuid4().hex
        self.path = (
            self.archive.parent
            / "_operational_leases"
            / f"{self.run_id}.api_lease.json"
        )
        self.committer = runner._remote_committer(self.archive)
        # googleapiclient/httplib2 transports are not thread-safe.  Keep the
        # heartbeat on its own authorized service while the main thread uses
        # ``self.committer`` for science fencing and commits.
        self.heartbeat_committer = runner._remote_committer(self.archive)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lost_reason: str | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": H1_API_LEASE_SCHEMA,
            "bundle": self.runner.BUNDLE,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "shard_index": self.shard_index,
            "archive": self.archive.name,
            "owner_token": self.token,
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "updated_unix": time.time(),
        }

    def _read_optional_record(
        self, committer: Any | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        committer = self.committer if committer is None else committer
        try:
            record = committer.path_commit_record(self.path)
        except self.runner.RemoteCommitError as exc:
            if _is_absent_error(exc):
                return None, None
            raise
        return (
            _read_json_by_record(
                committer, record, expected_name=self.path.name
            ),
            record,
        )

    def _read_optional(self, committer: Any | None = None) -> dict[str, Any] | None:
        payload, _ = self._read_optional_record(committer)
        return payload

    def _assert_owned_payload(self, payload: dict[str, Any] | None) -> None:
        if payload is None or payload.get("owner_token") != self.token:
            raise RuntimeError(
                f"H1 API shard lease ownership was lost: observed={payload}"
            )
        expected = {
            "schema": H1_API_LEASE_SCHEMA,
            "bundle": self.runner.BUNDLE,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "shard_index": self.shard_index,
            "archive": self.archive.name,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"H1 API lease identity changed: {mismatches}")

    def _validate_claim_payload(self, payload: dict[str, Any]) -> None:
        expected = {
            "schema": H1_API_LEASE_SCHEMA,
            "bundle": self.runner.BUNDLE,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "shard_index": self.shard_index,
            "archive": self.archive.name,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        token = str(payload.get("owner_token", ""))
        if mismatches or not re.fullmatch(r"[0-9a-f]{32}", token):
            raise RuntimeError(
                f"H1 API lease contains an invalid concurrent claim: {mismatches}"
            )

    def _claim_records(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Read every same-name claim by ID, even while the path is ambiguous."""

        parts, name = self.committer.split_remote_path(self.path)
        try:
            parent_id = self.committer.resolve_folder(parts, create=False)
        except self.runner.RemoteCommitError as exc:
            if _is_absent_error(exc):
                return []
            raise
        items = self.committer._list_children(parent_id, name)
        claims: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for item in items:
            try:
                metadata = self.committer.metadata(str(item["id"]))
                record = self.committer.commit_record(metadata)
                record = _refresh_exact_named_record(
                    self.committer,
                    record,
                    expected_name=name,
                    expected_parent_id=parent_id,
                )
                raw = self.committer.download_bytes(str(record["remote_file_id"]))
            except self.runner.RemoteCommitError as exc:
                if _is_absent_error(exc):
                    continue
                raise
            if len(raw) != int(record["remote_bytes"]):
                raise RuntimeError("H1 API lease claim has the wrong byte count")
            if hashlib.sha256(raw).hexdigest() != str(record["remote_sha256"]):
                raise RuntimeError("H1 API lease claim failed its checksum")
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("H1 API lease claim is not a JSON object")
            self._validate_claim_payload(payload)
            claims.append((payload, record))
        return claims

    def _delete_claim(
        self, payload: dict[str, Any], record: dict[str, Any]
    ) -> None:
        parts, name = self.committer.split_remote_path(self.path)
        parent_id = self.committer.resolve_folder(parts, create=False)
        refreshed = _refresh_exact_named_record(
            self.committer,
            record,
            expected_name=name,
            expected_parent_id=parent_id,
        )
        raw = self.committer.download_bytes(str(refreshed["remote_file_id"]))
        if len(raw) != int(refreshed["remote_bytes"]) or hashlib.sha256(
            raw
        ).hexdigest() != str(refreshed["remote_sha256"]):
            raise RuntimeError("H1 API lease claim changed before deletion")
        current = json.loads(raw.decode("utf-8"))
        if not isinstance(current, dict):
            raise RuntimeError("H1 API lease claim is not a JSON object")
        self._validate_claim_payload(current)
        if _normalized(current) != _normalized(payload):
            raise RuntimeError("H1 API lease claim payload changed before deletion")
        self.committer.delete_verified_if_present(refreshed)

    def _reconcile_concurrent_claims(self) -> bool:
        """Elect the smallest token and converge to one exact same-name file."""

        for _ in range(5):
            claims = self._claim_records()
            own = [row for row in claims if row[0].get("owner_token") == self.token]
            if not own:
                return False
            winner = min(
                claims,
                key=lambda row: (
                    str(row[0]["owner_token"]),
                    str(row[1]["remote_file_id"]),
                ),
            )
            if winner[0]["owner_token"] != self.token:
                for payload, record in own:
                    self._delete_claim(payload, record)
                return False
            winner_id = str(winner[1]["remote_file_id"])
            for payload, record in claims:
                if str(record["remote_file_id"]) != winner_id:
                    self._delete_claim(payload, record)
            if H1_API_LEASE_SETTLE_SECONDS > 0:
                time.sleep(H1_API_LEASE_SETTLE_SECONDS)
            remaining = self._claim_records()
            if (
                len(remaining) == 1
                and remaining[0][0].get("owner_token") == self.token
                and str(remaining[0][1]["remote_file_id"]) == winner_id
            ):
                return True
        for payload, record in self._claim_records():
            try:
                if payload.get("owner_token") == self.token:
                    self._delete_claim(payload, record)
            except Exception:
                pass
        raise RuntimeError("H1 API lease could not converge to one unique claim")

    def _converge_existing_claims(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Repair an ambiguous lease path left by interrupted claimants."""

        for _ in range(5):
            claims = self._claim_records()
            if not claims:
                raise RuntimeError("ambiguous H1 API lease vanished during recovery")
            winner = min(
                claims,
                key=lambda row: (
                    str(row[0]["owner_token"]),
                    str(row[1]["remote_file_id"]),
                ),
            )
            winner_id = str(winner[1]["remote_file_id"])
            for payload, record in claims:
                if str(record["remote_file_id"]) != winner_id:
                    self._delete_claim(payload, record)
            if H1_API_LEASE_SETTLE_SECONDS > 0:
                time.sleep(H1_API_LEASE_SETTLE_SECONDS)
            remaining = self._claim_records()
            if (
                len(remaining) == 1
                and str(remaining[0][1]["remote_file_id"]) == winner_id
            ):
                return remaining[0]
        raise RuntimeError("ambiguous H1 API lease did not converge")

    def assert_owned(self, committer: Any | None = None) -> None:
        if self._lost_reason is not None:
            raise RuntimeError(
                f"H1 API shard lease heartbeat failed: {self._lost_reason}"
            )
        self._assert_owned_payload(self._read_optional(committer))

    def _heartbeat(self) -> None:
        while not self._stop.wait(H1_API_LEASE_HEARTBEAT_SECONDS):
            try:
                self.assert_owned(self.heartbeat_committer)
                self.runner.publish_json(
                    self.heartbeat_committer,
                    self._payload(),
                    self.path,
                    replace=True,
                    required_headroom_bytes=0,
                )
                self.assert_owned(self.heartbeat_committer)
            except Exception as exc:
                self._lost_reason = repr(exc)
                return

    def __enter__(self) -> "_H1ApiShardLease":
        try:
            current, current_record = self._read_optional_record()
        except self.runner.RemoteCommitError as exc:
            if "ambiguous" not in str(exc).lower():
                raise
            current, current_record = self._converge_existing_claims()
        if current is not None:
            expected = {
                "schema": H1_API_LEASE_SCHEMA,
                "bundle": self.runner.BUNDLE,
                "run_id": self.run_id,
                "case_id": self.case_id,
                "shard_index": self.shard_index,
                "archive": self.archive.name,
            }
            mismatches = {
                key: (current.get(key), value)
                for key, value in expected.items()
                if current.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"existing H1 API lease is misbound: {mismatches}")
            age = time.time() - float(current.get("updated_unix", 0.0))
            if age < H1_API_LEASE_STALE_SECONDS:
                raise RuntimeError(
                    f"another H1 writer owns the API lease: owner={current}"
                )
            assert current_record is not None
            latest, latest_record = self._read_optional_record()
            if (
                latest is None
                or latest_record is None
                or _commit_identity(latest_record) != _commit_identity(current_record)
                or latest.get("owner_token") != current.get("owner_token")
                or float(latest.get("updated_unix", 0.0))
                != float(current.get("updated_unix", 0.0))
                or time.time() - float(latest.get("updated_unix", 0.0))
                < H1_API_LEASE_STALE_SECONDS
            ):
                raise RuntimeError("H1 API lease changed during stale recovery")
            if hasattr(self.committer, "verify_record_for_path"):
                self.committer.verify_record_for_path(current_record, self.path)
            else:
                self.committer.verify_commit_record(current_record)
            self.committer.delete_verified_if_present(current_record)
        claim_error: Exception | None = None
        try:
            self.runner.publish_json(
                self.committer,
                self._payload(),
                self.path,
                replace=False,
                required_headroom_bytes=0,
            )
        except Exception as exc:
            # A simultaneous absent claim can make the final name ambiguous
            # after both uploads succeeded.  Reconcile by file ID before
            # deciding whether the publication really failed.
            claim_error = exc
        if H1_API_LEASE_SETTLE_SECONDS > 0:
            time.sleep(H1_API_LEASE_SETTLE_SECONDS)
        try:
            won = self._reconcile_concurrent_claims()
        except Exception:
            if claim_error is not None and not self._claim_records():
                raise claim_error
            raise
        if not won:
            raise RuntimeError("another H1 writer won the concurrent API lease claim")
        self.assert_owned()
        self._thread = threading.Thread(
            target=self._heartbeat,
            name=f"h1-api-lease-{self.run_id[-8:]}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            current = self._read_optional()
            if current is None or current.get("owner_token") != self.token:
                return
            record = self.committer.path_commit_record(self.path)
            if hasattr(self.committer, "verify_record_for_path"):
                self.committer.verify_record_for_path(record, self.path)
            else:
                self.committer.verify_commit_record(record)
            self.committer.delete_verified_if_present(record)
        except Exception as cleanup_exc:
            warnings.warn(
                f"H1 API lease cleanup remains for stale recovery: {cleanup_exc}",
                RuntimeWarning,
            )


def repair_orphan(
    runner: Any,
    *,
    bundle_root: Path | str,
    drive_root: Path | str,
    mode: str,
    case_id: str,
    shard_index: int,
) -> dict[str, Any]:
    """Repair only while holding the same single-writer lease as science."""

    bundle_root = Path(bundle_root).resolve()
    drive_root = lexical_absolute(drive_root)
    config = runner.load_config(bundle_root)
    cases = {row["case_id"]: row for row in runner.expand_cases(config)}
    if case_id not in cases:
        raise KeyError(f"unknown H1-v4 case {case_id!r}")
    if not 0 <= int(shard_index) < 5:
        raise IndexError("H1-v4 shard index must lie in 0..4")
    case = cases[case_id]
    _, archive, run_id, _ = runner._archive_paths(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=int(shard_index),
        drive_root=drive_root,
        mode=mode,
    )
    if not runner._server_commit_required(archive):
        raise RuntimeError("orphan repair is allowed only for a Google Drive API target")

    # Avoid a lease round trip for the overwhelmingly common absent/complete
    # cases.  A true archive-without-receipt is rechecked after acquiring the
    # slot lease, closing the live writer's archive-to-receipt race window.
    try:
        current = verify_remote_existing(runner, archive)
    except OrphanArchiveError:
        pass
    else:
        if current is None:
            return {
                "status": "no_final_archive",
                "repaired": False,
                "archive": str(archive),
            }
        return {
            "status": "already_verified",
            "repaired": False,
            "archive": str(archive),
            "archive_sha256": current["archive_sha256"],
        }

    lease = _H1ApiShardLease(
        runner=runner,
        archive=archive,
        run_id=run_id,
        case_id=str(case_id),
        shard_index=int(shard_index),
    )
    with lease:
        return _repair_orphan_core(
            runner,
            bundle_root=bundle_root,
            drive_root=drive_root,
            mode=mode,
            case_id=case_id,
            shard_index=int(shard_index),
            lease=lease,
        )


def install_operational_hardening(
    runner: Any, *, drive_helper: Any | None = None
) -> None:
    """Patch only transport boundaries; never modify the scientific sources."""

    drive_helper = drive_helper or _load_root_drive_helper()
    if getattr(runner, "_h1_operational_hardening_installed", False):
        if getattr(runner, "_h1_operational_drive_helper", None) is not drive_helper:
            raise RuntimeError("H1 hardening was already installed with another helper")
        return
    required = (
        "DEFAULT_DRIVE_ROOT",
        "DriveRemoteCommitter",
        "RemoteCommitError",
        "publish_json",
        "read_remote_json",
    )
    missing = [name for name in required if not hasattr(drive_helper, name)]
    if missing:
        raise RuntimeError(f"deployment-root Drive helper is incomplete: {missing}")
    runner.DEFAULT_DRIVE_ROOT = drive_helper.DEFAULT_DRIVE_ROOT
    runner.RemoteCommitError = drive_helper.RemoteCommitError
    runner.read_remote_json = drive_helper.read_remote_json
    original_verify = runner._verify_existing
    original_replay = runner._find_replay_source
    original_run_case = runner.run_case
    original_preflight = runner.a100_preflight
    original_archive = runner._archive
    original_root_manifest = runner._root_manifest_from_archive
    active = threading.local()

    class ScopedCommitter(drive_helper.DriveRemoteCommitter):
        """Recheck the active lease immediately before every scientific upload."""

        def upload_verified(self, local_path: Any, remote_path: Any, **kwargs: Any):
            lease = getattr(active, "lease", None)
            if lease is not None:
                lease.assert_owned()
            result = super().upload_verified(local_path, remote_path, **kwargs)
            if lease is not None:
                lease.assert_owned()
            return result

    def server_required(path: Path | str) -> bool:
        return _is_below(path, runner.DEFAULT_DRIVE_ROOT)

    def committer_for(path: Path | str) -> Any:
        if not server_required(path):
            raise RuntimeError(f"not a production Google Drive path: {path}")
        return ScopedCommitter(drive_root=runner.DEFAULT_DRIVE_ROOT)

    def metadata_publish(
        committer: Any,
        payload: dict[str, Any],
        remote_path: Path | str,
        *,
        replace: bool,
        required_headroom_bytes: int = 0,
    ) -> dict[str, Any]:
        # A large scientific archive retains its 1 GiB guard in `_archive`.
        # JSON receipts, leases, sidecars, and preflight records require only
        # their actual bytes so a completed archive cannot be stranded merely
        # because the reserve dipped between the two commits.
        return drive_helper.publish_json(
            committer,
            payload,
            remote_path,
            replace=replace,
            required_headroom_bytes=0,
        )

    @contextlib.contextmanager
    def lease_for(
        *,
        bundle_root: Path,
        config: dict[str, Any],
        case: dict[str, Any],
        shard_index: int,
        drive_root: Path,
        mode: str,
    ) -> Iterator[None]:
        _, archive, run_id, _ = runner._archive_paths(
            bundle_root=bundle_root,
            config=config,
            case=case,
            shard_index=int(shard_index),
            drive_root=drive_root,
            mode=mode,
        )
        if not server_required(archive):
            yield
            return
        current = getattr(active, "lease", None)
        if current is not None:
            if current.run_id != run_id:
                raise RuntimeError("nested H1 execution requested another shard lease")
            current.assert_owned()
            yield
            current.assert_owned()
            return
        lease = _H1ApiShardLease(
            runner=runner,
            archive=archive,
            run_id=run_id,
            case_id=str(case["case_id"]),
            shard_index=int(shard_index),
        )
        with lease:
            active.lease = lease
            try:
                yield
                lease.assert_owned()
            finally:
                active.lease = None

    def leased_run_case(
        *,
        bundle_root: Path,
        config: dict[str, Any],
        case: dict[str, Any],
        shard_index: int,
        drive_root: Path,
        mode: str,
        archive_result: bool = True,
    ) -> dict[str, Any]:
        with lease_for(
            bundle_root=bundle_root,
            config=config,
            case=case,
            shard_index=shard_index,
            drive_root=drive_root,
            mode=mode,
        ):
            return original_run_case(
                bundle_root=bundle_root,
                config=config,
                case=case,
                shard_index=shard_index,
                drive_root=drive_root,
                mode=mode,
                archive_result=archive_result,
            )

    def leased_preflight(
        *, bundle_root: Path, config: dict[str, Any], drive_root: Path
    ) -> dict[str, Any]:
        case = next(
            row
            for row in runner.expand_cases(config)
            if row["protocol"] == "soft" and row["model"]["alpha_1"] == 1.0
        )
        with lease_for(
            bundle_root=bundle_root,
            config=config,
            case=case,
            shard_index=0,
            drive_root=drive_root,
            mode="production",
        ):
            return original_preflight(
                bundle_root=bundle_root, config=config, drive_root=drive_root
            )

    def verified_existing(archive: Path | str) -> dict[str, Any] | None:
        if runner._server_commit_required(archive):
            return verify_remote_existing(runner, archive)
        return original_verify(Path(archive))

    def replay_lookup(**kwargs: Any) -> tuple[dict[str, Any] | None, list[str]]:
        return replay_lookup_with_mount_fallback(original_replay, **kwargs)

    def hardened_archive(
        scratch: Path, archive: Path, run_id: str
    ) -> dict[str, Any]:
        manifest_path = Path(scratch) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = original_archive(scratch, archive, run_id)
        if server_required(archive):
            publish_manifest_sidecar(
                runner,
                archive=Path(archive),
                receipt=receipt,
                manifest=manifest,
            )
            verified = verify_remote_existing(runner, archive)
            if verified is None:
                raise RuntimeError("H1 archive disappeared after sidecar publication")
        return receipt

    def hardened_root_manifest(path: Path) -> dict[str, Any]:
        if not server_required(path):
            return original_root_manifest(path)
        sidecar = read_manifest_sidecar(runner, Path(path))
        if sidecar is not None:
            return sidecar
        # Qualification archives committed before this operational adapter have
        # no sidecar.  Download exactly once through the frozen reader, then
        # publish a bound sidecar for every later status query.
        manifest = original_root_manifest(path)
        receipt = verify_remote_existing(runner, path)
        if receipt is None:
            raise RuntimeError("H1 archive vanished while backfilling status sidecar")
        publish_manifest_sidecar(
            runner,
            archive=Path(path),
            receipt=receipt,
            manifest=manifest,
        )
        return manifest

    runner.DriveRemoteCommitter = ScopedCommitter
    runner.publish_json = metadata_publish
    runner._server_commit_required = server_required
    runner._remote_committer = committer_for
    runner._verify_existing = verified_existing
    runner._find_replay_source = replay_lookup
    runner._archive = hardened_archive
    runner._root_manifest_from_archive = hardened_root_manifest
    runner.run_case = leased_run_case
    runner.a100_preflight = leased_preflight
    runner._h1_operational_drive_helper = drive_helper
    runner._h1_operational_hardening_installed = True


def strict_preflight_reuse(
    runner: Any,
    *,
    bundle_root: Path,
    drive_root: Path,
) -> dict[str, Any] | None:
    """Reuse only an exact valid receipt; compute only on explicit absence."""

    config = runner.load_config(bundle_root)
    path = runner._preflight_path(drive_root, config)
    committer = runner._remote_committer(path)
    try:
        record = committer.path_commit_record(path)
    except runner.RemoteCommitError as exc:
        if _is_absent_error(exc):
            return None
        raise
    payload = _read_json_by_record(committer, record, expected_name=path.name)
    validated = runner.require_safe_preflight(
        drive_root=drive_root, config=config
    )
    if _normalized(validated) != _normalized(payload):
        raise RuntimeError("H1 preflight changed during strict server validation")
    result = dict(validated)
    result["status"] = "reused_current_safe_receipt"
    result["receipt_path"] = str(path)
    result["receipt_remote_commit"] = record
    return result


def strict_h1_v3_migration(
    *, runner: Any, bundle_root: Path, drive_root: Path
) -> dict[str, Any]:
    """Reuse a valid migration ledger; rebuild only on explicit server absence."""

    import h1_v3_migration as migration

    drive_helper = _load_root_drive_helper()
    migration.DriveRemoteCommitter = drive_helper.DriveRemoteCommitter
    migration.RemoteCommitError = drive_helper.RemoteCommitError
    migration.read_remote_json = drive_helper.read_remote_json

    def metadata_publish(
        committer: Any,
        payload: dict[str, Any],
        path: Path | str,
        *,
        replace: bool,
        required_headroom_bytes: int = 0,
    ) -> dict[str, Any]:
        return drive_helper.publish_json(
            committer,
            payload,
            path,
            replace=replace,
            required_headroom_bytes=0,
        )

    migration.publish_json = metadata_publish
    committer = drive_helper.DriveRemoteCommitter(drive_root=drive_root)
    path = migration.verified_ledger_path(drive_root)
    try:
        record = committer.path_commit_record(path)
    except drive_helper.RemoteCommitError as exc:
        if not _is_absent_error(exc):
            raise
        lease = _H1ApiShardLease(
            runner=runner,
            archive=path,
            run_id=f"{runner.BUNDLE}_v3_migration_initializer",
            case_id="H1_V3_MIGRATION_LEDGER",
            shard_index=-1,
        )
        with lease:
            # Another initializer may have won immediately before our claim.
            # Recheck under the elected lease and build only on explicit
            # server absence.
            try:
                record = committer.path_commit_record(path)
            except drive_helper.RemoteCommitError as locked_exc:
                if not _is_absent_error(locked_exc):
                    raise
                lease.assert_owned()
                result = migration.publish_verified_ledger(
                    drive_root=drive_root,
                    bundle_root=bundle_root,
                    inspect_archives=True,
                )
                lease.assert_owned()
                record = committer.path_commit_record(path)
            else:
                _read_json_by_record(committer, record, expected_name=path.name)
                result = migration.load_current_verified_ledger(
                    drive_root=drive_root, bundle_root=bundle_root
                )
                lease.assert_owned()
    else:
        # A present-but-corrupt, ambiguous, stale, or transiently unreadable
        # ledger is never treated as permission to rebuild it.
        _read_json_by_record(committer, record, expected_name=path.name)
        result = migration.load_current_verified_ledger(
            drive_root=drive_root, bundle_root=bundle_root
        )

    allowlist = migration.load_allowlist(bundle_root)
    pinned = {str(row["run_id"]): row for row in allowlist["accepted_archives"]}
    source_root = (
        drive_root
        / "classA_final_production_outputs"
        / str(allowlist["source_revision"])
        / migration.BUNDLE
    )
    accepted = result.get("accepted_archives", [])
    if len(accepted) != 12:
        raise RuntimeError("strict H1 migration expected exactly 12 accepted archives")
    for row in accepted:
        expected = pinned.get(str(row.get("run_id")))
        if expected is None:
            raise RuntimeError("strict H1 migration contains an unknown run ID")
        for key in (
            "case_id",
            "shard_index",
            "global_sample_indices",
            "archive_bytes",
            "archive_sha256",
            "reuse_in_v4",
        ):
            if _normalized(row.get(key)) != _normalized(expected.get(key)):
                raise RuntimeError(f"strict H1 migration changed pinned {key}")
        archive = source_root / f"{migration.BUNDLE}_{row['run_id']}.tar.gz"
        receipt = archive.with_suffix(archive.suffix + ".receipt.json")
        committer.verify_record_for_path(row["archive_remote_commit"], archive)
        committer.verify_record_for_path(row["receipt_remote_commit"], receipt)
    rejected = {
        str(row.get("run_id")): row
        for row in result.get("rejected_receipt_only", [])
    }
    if set(rejected) != set(allowlist["rejected_receipt_only_run_ids"]):
        raise RuntimeError("strict H1 migration changed rejected receipt-only IDs")
    for run_id, row in rejected.items():
        archive = source_root / f"{migration.BUNDLE}_{run_id}.tar.gz"
        try:
            committer.path_commit_record(archive)
        except drive_helper.RemoteCommitError as exc:
            if not _is_absent_error(exc):
                raise
        else:
            raise RuntimeError(f"rejected H1-v3 archive now exists: {archive}")
        receipt = archive.with_suffix(archive.suffix + ".receipt.json")
        committer.verify_record_for_path(row["receipt_remote_commit"], receipt)
    exact_ledger = committer.path_commit_record(path)
    if _commit_identity(exact_ledger) != _commit_identity(record):
        raise RuntimeError("H1 migration ledger changed during strict validation")
    return {
        **result,
        "ledger_path": str(path),
        "ledger_remote_commit": exact_ledger,
        "strict_server_validation": True,
    }


def operational_main(
    argv: list[str], *, runner: Any, default_bundle_root: Path
) -> int | None:
    """Handle wrapper-only commands, returning ``None`` for scientific CLI use."""

    if "--migration-status" in argv:
        parser = argparse.ArgumentParser(
            description="Strictly verify or explicitly initialize H1-v3 migration"
        )
        parser.add_argument("--migration-status", action="store_true", required=True)
        parser.add_argument("--bundle-root", type=Path, default=default_bundle_root)
        parser.add_argument("--drive-root", type=Path, required=True)
        args = parser.parse_args(argv)
        result = strict_h1_v3_migration(
            runner=runner,
            bundle_root=Path(args.bundle_root).resolve(),
            drive_root=lexical_absolute(args.drive_root),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if "--a100-preflight" in argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--bundle-root", type=Path, default=default_bundle_root)
        parser.add_argument("--drive-root", type=Path, required=True)
        args, _ = parser.parse_known_args(argv)
        reused = strict_preflight_reuse(
            runner,
            bundle_root=Path(args.bundle_root).resolve(),
            drive_root=lexical_absolute(args.drive_root),
        )
        if reused is not None:
            print(json.dumps(reused, indent=2, sort_keys=True))
            return 0
        config = runner.load_config(Path(args.bundle_root).resolve())
        result = runner.a100_preflight(
            bundle_root=Path(args.bundle_root).resolve(),
            config=config,
            drive_root=lexical_absolute(args.drive_root),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result.get("safe", False):
            raise RuntimeError("H1 A100 qualification completed but was not safe")
        return 0
    if "--repair-orphan" not in argv:
        return None
    parser = argparse.ArgumentParser(description="Repair one verified H1 orphan archive")
    parser.add_argument("--repair-orphan", action="store_true", required=True)
    parser.add_argument("--bundle-root", type=Path, default=default_bundle_root)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("production", "pilot", "smoke"), default="production"
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args(argv)
    result = repair_orphan(
        runner,
        bundle_root=args.bundle_root,
        drive_root=args.drive_root,
        mode=args.mode,
        case_id=args.case_id,
        shard_index=args.shard_index,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
