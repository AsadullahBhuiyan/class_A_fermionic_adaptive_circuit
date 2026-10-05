#!/usr/bin/env python3
"""Run one checksum-resumable production bundle in a Colab session."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, TextIO

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
SHARED_SRC = ROOT / "_shared_src"
# Script execution usually pre-populates ``ROOT`` in ``sys.path``.  Merely
# checking membership would then put ``_shared_src`` ahead of it and silently
# import the frozen Drive transport.  Reorder both entries unconditionally:
# root operational helpers win; shared scientific utilities remain fallback.
for candidate in (str(ROOT), str(SHARED_SRC)):
    while candidate in sys.path:
        sys.path.remove(candidate)
sys.path.insert(0, str(SHARED_SRC))
sys.path.insert(0, str(ROOT))

from bundle_layout import bundle_path  # noqa: E402
from production_runtime import (  # noqa: E402
    APPROVED_LEGACY_AUDIT_SHA256,
    LEGACY_PRODUCTION_OUTPUT_COLLECTION,
    PILOT_OUTPUT_COLLECTION,
    PRIOR_V2_AUDIT_SHA256,
    PRIOR_V2_PRODUCTION_OUTPUT_COLLECTION,
    PRODUCTION_OUTPUT_COLLECTION,
    PRODUCTION_SAMPLES,
    SHARD_SIZE,
    V1_AUDIT_SHA256,
    V1_PRODUCTION_OUTPUT_COLLECTION,
    json_ready,
    sha256_file,
    verify_archive_receipt,
    write_json_atomic,
)
_DRIVE_HELPER_PATH = (
    ROOT / "drive_remote_commit.py"
    if (ROOT / "drive_remote_commit.py").is_file()
    else SHARED_SRC / "drive_remote_commit.py"
)
_DRIVE_HELPER_SPEC = importlib.util.spec_from_file_location(
    "classA_parent_operational_drive_remote_commit", _DRIVE_HELPER_PATH
)
if _DRIVE_HELPER_SPEC is None or _DRIVE_HELPER_SPEC.loader is None:
    raise RuntimeError(f"cannot load operational Drive helper: {_DRIVE_HELPER_PATH}")
_DRIVE_HELPER = importlib.util.module_from_spec(_DRIVE_HELPER_SPEC)
sys.modules[_DRIVE_HELPER_SPEC.name] = _DRIVE_HELPER
_DRIVE_HELPER_SPEC.loader.exec_module(_DRIVE_HELPER)
DriveRemoteCommitter = _DRIVE_HELPER.DriveRemoteCommitter
RemoteCommitError = _DRIVE_HELPER.RemoteCommitError
_execute_with_retries = _DRIVE_HELPER._execute_with_retries
read_remote_json = _DRIVE_HELPER.read_remote_json


RUNNABLE_BUNDLES = (
    "01_p1_chern_dynamics",
    "02_wall_cft_windows",
    "03_h1_modular_response",
    "08_h1_endpoint_packet",
)
P1_BUNDLE = "01_p1_chern_dynamics"
H1_ENDPOINT_BUNDLE = "08_h1_endpoint_packet"
SERVER_COMMIT_BUNDLES = frozenset((P1_BUNDLE, H1_ENDPOINT_BUNDLE))
P1_MANIFEST_SIDECAR_SCHEMA = "p1_server_verified_manifest_sidecar_v1"
H1_STATUS_SIDECAR_SCHEMA = "h1_v4_remote_status_sidecar_v1"
B1_BUNDLE = "06_b1_controller_frame"
SESSION_SCHEMA = "classA_colab_bundle_session_v1"
CHILD_TAIL_LINES = 200
P1_DEFAULT_SESSION_HOURS = 7.5
P1_RUNTIME_SAFETY_FACTOR = 1.25
P1_RUNTIME_MARGIN_SECONDS = 15.0 * 60.0
P1_L64_PHYSICAL_CYCLES = 64
P1_SESSION_CHECKPOINT_PREFIX = "[P1 SESSION CHECKPOINT] "
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
PROFILE_HOUR_CAPS = {
    "pilot_calibration": {
        "01_p1_chern_dynamics": 12.0,
        "02_wall_cft_windows": 12.0,
        "03_h1_modular_response": 12.0,
        "08_h1_endpoint_packet": 12.0,
    },
    "pilot_science": {
        "01_p1_chern_dynamics": 60.0,
        "02_wall_cft_windows": 60.0,
        "03_h1_modular_response": 60.0,
        "08_h1_endpoint_packet": 60.0,
    },
}


class ChildProcessFailure(RuntimeError):
    """A child command failed after its live output was preserved."""

    def __init__(
        self,
        *,
        command: list[str],
        returncode: int,
        output_tail: list[str],
        elapsed_seconds: float,
        stage: str | None = None,
    ) -> None:
        super().__init__(f"child command exited {returncode}: {shlex.join(command)}")
        self.command = list(command)
        self.returncode = int(returncode)
        self.output_tail = list(output_tail)
        self.elapsed_seconds = float(elapsed_seconds)
        self.stage = stage


class SessionBudgetReached(RuntimeError):
    """A clean durable-boundary stop requested before Colab's runtime cutoff."""


def _default_display_stream() -> TextIO:
    """Keep CLI status on stdout while coordinating notebook bars on stderr."""
    try:
        from IPython import get_ipython
    except ImportError:
        return sys.stdout
    return sys.stderr if get_ipython() is not None else sys.stdout


class SessionLogger:
    """Write readable Colab output and a local, non-gating session log.

    Session telemetry must never control a scientific child.  In particular, this
    class deliberately degrades after display or filesystem errors instead of
    propagating them through the streaming supervisor.
    """

    def __init__(self, path: Path, *, display: TextIO | None = None) -> None:
        self.path = path
        self.display = _default_display_stream() if display is None else display
        self.errors: list[dict[str, str]] = []
        self._handle: TextIO | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8", buffering=1)
        except Exception as exc:
            self.note_error("open", exc)

    def note_error(self, operation: str, exc: BaseException) -> None:
        self.errors.append(
            {
                "operation": str(operation),
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )

    def emit(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        try:
            tqdm.write(line, file=self.display)
        except Exception as exc:
            self.note_error("display", exc)
        if self._handle is not None:
            try:
                self._handle.write(line + "\n")
            except Exception as exc:
                self.note_error("write", exc)
                try:
                    self._handle.close()
                except Exception as close_exc:
                    self.note_error("close_after_write_failure", close_exc)
                self._handle = None

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        except Exception as exc:
            self.note_error("close", exc)
        finally:
            self._handle = None


def local_session_root(*, output_collection: str, bundle: str) -> Path:
    """Return a runtime-local telemetry directory, never a DriveFS path."""
    scratch_base = (
        Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    )
    relative = Path(str(output_collection))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe output collection for local telemetry: {output_collection!r}")
    return scratch_base / "classA_bundle_sessions" / relative / str(bundle)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(json_ready(value), sort_keys=True))


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


_REMOTE_COMMIT_IDENTITY_FIELDS = (
    "schema",
    "remote_file_id",
    "remote_name",
    "remote_parent_id",
    "remote_bytes",
    "remote_sha256",
)


def _remote_commit_identity(record: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field for field in _REMOTE_COMMIT_IDENTITY_FIELDS if field not in record
    ]
    if missing:
        raise RuntimeError(f"remote commit record is incomplete: {missing}")
    return {field: record[field] for field in _REMOTE_COMMIT_IDENTITY_FIELDS}


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.normpath(os.path.abspath(os.fspath(path))))


def _case_without_declared_samples(case: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalized(case)
    run = normalized.get("run")
    if isinstance(run, dict):
        run.pop("samples", None)
    return normalized


def _shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _expected_sample_indices(case: dict[str, Any], shard_index: int) -> list[int]:
    samples = int(case.get("run", {}).get("samples", SHARD_SIZE))
    width = int(
        case.get("execution", {}).get("samples_per_shard", SHARD_SIZE)
    )
    if width <= 0 or samples % width:
        raise ValueError(
            f"invalid samples_per_shard={width} for samples={samples}"
        )
    start = int(shard_index) * width
    return list(range(start, min(start + width, samples)))


def _root_manifest_from_archive(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.lstrip("./") == "manifest.json"
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"{path}: expected one root manifest, found {len(members)}"
            )
        handle = archive.extractfile(members[0])
        if handle is None:
            raise RuntimeError(f"{path}: unreadable root manifest")
        return json.loads(handle.read().decode("utf-8"))


def _scan_archive_directory(
    root: Path, *, label: str, logger: SessionLogger
) -> list[dict[str, Any]]:
    """Checksum and index one archive directory with visible progress."""
    if not root.is_dir():
        logger.emit(f"[RESUME] {label}: no archive directory at {root}")
        return []
    archives = sorted(root.glob("*.tar.gz"))
    receipts = sorted(root.glob("*.tar.gz.receipt.json"))
    archive_paths = {str(path) for path in archives}
    orphan_receipts = [
        path
        for path in receipts
        if str(path)[: -len(".receipt.json")] not in archive_paths
    ]
    if orphan_receipts:
        raise RuntimeError(
            "receipt exists without its archive: "
            + ", ".join(str(path) for path in orphan_receipts)
        )
    logger.emit(f"[RESUME] {label}: verifying {len(archives)} archive(s)")
    rows: list[dict[str, Any]] = []
    for archive in tqdm(
        archives,
        desc=f"verify {label}",
        unit="archive",
        dynamic_ncols=True,
        leave=False,
        file=logger.display,
    ):
        receipt = verify_archive_receipt(archive)
        manifest = _root_manifest_from_archive(archive)
        rows.append({"archive": archive, "receipt": receipt, "manifest": manifest})
    logger.emit(f"[RESUME] {label}: verified {len(rows)}/{len(archives)}")
    return rows


def _current_rows_for_bundle(
    root: Path,
    *,
    bundle: str,
    label: str,
    logger: SessionLogger,
) -> list[dict[str, Any]]:
    """Use DriveFS only for bundles without a server-commit contract."""
    if bundle in SERVER_COMMIT_BUNDLES:
        logger.emit(
            f"[RESUME] {label}: Drive API is authoritative; "
            "skipping the DriveFS current-output scan"
        )
        return []
    return _scan_archive_directory(root, label=label, logger=logger)


def _replace_remote_current_row(
    rows: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    case_id: str,
    shard_index: int,
) -> None:
    """Cache one exact remote row without retaining a stale row for its slot."""
    retained: list[dict[str, Any]] = []
    for candidate in rows:
        manifest = candidate.get("manifest", {})
        same_slot = (
            str(manifest.get("case_id", "")) == str(case_id)
            and int(manifest.get("shard_index", -1)) == int(shard_index)
        )
        if not same_slot:
            retained.append(candidate)
    rows[:] = [*retained, row]


def _current_match_reasons(
    row: dict[str, Any],
    *,
    bundle: str,
    case: dict[str, Any],
    shard_index: int,
    config: dict[str, Any],
    engine_hash: str,
    source_hashes: dict[str, str] | None = None,
) -> list[str]:
    manifest = row["manifest"]
    run_config = manifest.get("run_config", {})
    reasons: list[str] = []
    expected_seed = _shard_seed(int(config["root_seed"]), case["case_id"], shard_index)
    comparisons = (
        ("bundle", str(manifest.get("bundle", "")), str(bundle)),
        ("status", str(manifest.get("status", "")), "complete_local"),
        ("case_id", str(manifest.get("case_id", "")), str(case["case_id"])),
        ("shard_index", int(manifest.get("shard_index", -1)), int(shard_index)),
        ("root_seed", int(manifest.get("root_seed", -1)), int(config["root_seed"])),
        (
            "shard_generator_seed",
            int(manifest.get("shard_generator_seed", -1)),
            expected_seed,
        ),
        (
            "audit_sha256",
            str(manifest.get("audit_sha256", run_config.get("audit_sha256", ""))),
            str(config["audit_sha256"]),
        ),
        (
            "canonical_engine_sha256",
            str(
                run_config.get(
                    "canonical_engine_sha256",
                    manifest.get("canonical_engine_sha256", ""),
                )
            ),
            str(engine_hash),
        ),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            reasons.append(name)
    if _normalized(run_config.get("case", {})) != _normalized(case):
        reasons.append("case_configuration")
    if config.get("strict_source_hash_resume") is True:
        if _normalized(manifest.get("source_hashes", {})) != _normalized(
            source_hashes or {}
        ):
            reasons.append("source_hashes")
    if [int(value) for value in manifest.get("global_sample_indices", [])] != (
        _expected_sample_indices(case, shard_index)
    ):
        reasons.append("global_sample_indices")
    return reasons


def _find_current_archive(
    rows: list[dict[str, Any]],
    *,
    bundle: str,
    case: dict[str, Any],
    shard_index: int,
    config: dict[str, Any],
    engine_hash: str,
    source_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    same_slot = [
        row
        for row in rows
        if str(row["manifest"].get("case_id", "")) == str(case["case_id"])
        and int(row["manifest"].get("shard_index", -1)) == int(shard_index)
    ]
    matches = [
        row
        for row in same_slot
        if not _current_match_reasons(
            row,
            bundle=bundle,
            case=case,
            shard_index=shard_index,
            config=config,
            engine_hash=engine_hash,
            source_hashes=source_hashes,
        )
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple exact current archives for {bundle}/{case['case_id']} "
            f"shard {shard_index}: {[str(row['archive']) for row in matches]}"
        )
    mismatch_rows = []
    for row in same_slot:
        reasons = _current_match_reasons(
            row,
            bundle=bundle,
            case=case,
            shard_index=shard_index,
            config=config,
            engine_hash=engine_hash,
            source_hashes=source_hashes,
        )
        if reasons:
            mismatch_rows.append(
                {"archive": str(row["archive"]), "mismatch_fields": reasons}
            )
    return (matches[0] if matches else None), mismatch_rows


def _parent_verify_server_status(
    remote_row: dict[str, Any],
    *,
    drive_root: Path,
    profile: str,
    bundle: str,
    config: dict[str, Any],
    committer: DriveRemoteCommitter | None = None,
) -> dict[str, Any]:
    """Independently bind child status to the one intended Drive slot.

    Child status is useful evidence, but it is not the parent's durability
    authority.  The parent re-derives the deterministic filename, re-reads the
    exact receipt from Drive by path, and verifies both commit records before
    the row may advance the queue.
    """

    if bundle not in SERVER_COMMIT_BUNDLES:
        raise ValueError(f"{bundle} has no server status contract")
    if not remote_row.get("exists"):
        raise RuntimeError("cannot server-verify an absent status row")
    manifest = remote_row.get("manifest")
    child_receipt = remote_row.get("receipt")
    if not isinstance(manifest, dict) or not isinstance(child_receipt, dict):
        raise RuntimeError("remote status lacks a manifest or receipt object")
    run_config = manifest.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("remote status manifest lacks its run configuration")
    run_config_hash = _sha256_json(run_config)
    run_id = f"{bundle}_{run_config_hash[:16]}"
    if (
        manifest.get("run_config_hash") != run_config_hash
        or child_receipt.get("run_id") != run_id
    ):
        raise RuntimeError("remote status has a noncanonical run identity")

    collection_key = (
        "production_output_collection"
        if profile == "production"
        else "pilot_output_collection"
    )
    collection = Path(str(config[collection_key]))
    output_bundle = Path(str(config.get("output_bundle", bundle)))
    if (
        collection.is_absolute()
        or not collection.parts
        or ".." in collection.parts
        or output_bundle.is_absolute()
        or len(output_bundle.parts) != 1
        or output_bundle.name in ("", ".", "..")
    ):
        raise RuntimeError("server output collection/path is unsafe")
    expected_archive = (
        _lexical_absolute(drive_root)
        / collection
        / output_bundle
        / f"{run_id}.tar.gz"
    )
    if _lexical_absolute(Path(str(remote_row.get("archive", "")))) != expected_archive:
        raise RuntimeError("remote status archive is outside its locked output path")
    if child_receipt.get("archive") != expected_archive.name:
        raise RuntimeError("remote status receipt names another archive")
    receipt_path = expected_archive.with_suffix(
        expected_archive.suffix + ".receipt.json"
    )

    committer = committer or DriveRemoteCommitter(drive_root=drive_root)
    archive_record = committer.path_commit_record(expected_archive)
    committer.verify_record_for_path(archive_record, expected_archive)
    receipt_record_before = committer.path_commit_record(receipt_path)
    committer.verify_record_for_path(receipt_record_before, receipt_path)
    server_receipt = read_remote_json(committer, receipt_path)
    receipt_record_after = committer.path_commit_record(receipt_path)
    committer.verify_record_for_path(receipt_record_after, receipt_path)
    if _remote_commit_identity(receipt_record_before) != _remote_commit_identity(
        receipt_record_after
    ):
        raise RuntimeError("remote receipt changed during parent verification")
    if _normalized(server_receipt) != _normalized(child_receipt):
        raise RuntimeError("child status receipt differs from Drive readback")
    declared_archive = server_receipt.get("archive_remote_commit")
    if not isinstance(declared_archive, dict) or _remote_commit_identity(
        declared_archive
    ) != _remote_commit_identity(archive_record):
        raise RuntimeError("remote receipt is not bound to the intended archive")
    if (
        str(archive_record["remote_parent_id"])
        != str(receipt_record_after["remote_parent_id"])
        or server_receipt.get("archive_sha256")
        != archive_record["remote_sha256"]
        or int(server_receipt.get("archive_bytes", -1))
        != int(archive_record["remote_bytes"])
    ):
        raise RuntimeError("remote archive and receipt metadata disagree")

    sidecar_suffix, sidecar_schema = (
        (".manifest.json", P1_MANIFEST_SIDECAR_SCHEMA)
        if bundle == P1_BUNDLE
        else (".status.json", H1_STATUS_SIDECAR_SCHEMA)
    )
    sidecar_path = expected_archive.with_suffix(
        expected_archive.suffix + sidecar_suffix
    )
    sidecar_record_before = committer.path_commit_record(sidecar_path)
    committer.verify_record_for_path(sidecar_record_before, sidecar_path)
    sidecar = read_remote_json(committer, sidecar_path)
    sidecar_record_after = committer.path_commit_record(sidecar_path)
    committer.verify_record_for_path(sidecar_record_after, sidecar_path)
    if _remote_commit_identity(sidecar_record_before) != _remote_commit_identity(
        sidecar_record_after
    ):
        raise RuntimeError("remote manifest sidecar changed during parent verification")
    if (
        sidecar.get("schema") != sidecar_schema
        or sidecar.get("bundle") != bundle
        or sidecar.get("run_id") != run_id
        or sidecar.get("archive") != expected_archive.name
        or sidecar.get("manifest_sha256") != _sha256_json(manifest)
        or _normalized(sidecar.get("manifest")) != _normalized(manifest)
    ):
        raise RuntimeError("remote manifest sidecar disagrees with child status")
    if _remote_commit_identity(
        sidecar.get("archive_remote_commit", {})
    ) != _remote_commit_identity(archive_record):
        raise RuntimeError("remote manifest sidecar names another archive commit")
    if bundle == H1_ENDPOINT_BUNDLE and _remote_commit_identity(
        sidecar.get("receipt_remote_commit", {})
    ) != _remote_commit_identity(receipt_record_after):
        raise RuntimeError("H1 manifest sidecar names another receipt commit")

    normalized = dict(remote_row)
    normalized["archive"] = expected_archive
    normalized["archive_remote_commit"] = archive_record
    normalized["receipt_remote_commit"] = receipt_record_after
    normalized["manifest_sidecar_remote_commit"] = sidecar_record_after
    normalized["parent_server_verified"] = True
    return normalized


def _cache_remote_status_row(
    remote_row: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    bundle: str,
    case: dict[str, Any],
    shard_index: int,
    config: dict[str, Any],
    engine_hash: str,
    source_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Normalize and identity-check one server-verified child status row."""
    if not remote_row.get("exists"):
        return None, []
    normalized = dict(remote_row)
    normalized["archive"] = Path(normalized["archive"])
    _replace_remote_current_row(
        rows,
        normalized,
        case_id=str(case["case_id"]),
        shard_index=shard_index,
    )
    return _find_current_archive(
        rows,
        bundle=bundle,
        case=case,
        shard_index=shard_index,
        config=config,
        engine_hash=engine_hash,
        source_hashes=source_hashes,
    )


def _find_compatible_legacy_archive(
    rows_by_revision: dict[str, list[dict[str, Any]]],
    *,
    bundle: str,
    case: dict[str, Any],
    shard_index: int,
    config: dict[str, Any],
    engine_hash: str,
) -> dict[str, Any] | None:
    expected_indices = _expected_sample_indices(case, shard_index)
    expected_seed = _shard_seed(int(config["root_seed"]), case["case_id"], shard_index)
    current_case = _case_without_declared_samples(case)
    priorities = (
        ("prior_v2", {PRIOR_V2_AUDIT_SHA256}),
        ("v1", {V1_AUDIT_SHA256}),
        (
            "unversioned",
            APPROVED_LEGACY_AUDIT_SHA256
            - {PRIOR_V2_AUDIT_SHA256, V1_AUDIT_SHA256},
        ),
    )
    for revision, allowed_audits in priorities:
        matches: dict[str, dict[str, Any]] = {}
        for row in rows_by_revision.get(revision, []):
            manifest = row["manifest"]
            run_config = manifest.get("run_config", {})
            legacy_case = run_config.get("case", {})
            if str(manifest.get("bundle", "")) != str(bundle):
                continue
            if str(manifest.get("status", "")) != "complete_local":
                continue
            if str(legacy_case.get("case_id", "")) != str(case["case_id"]):
                continue
            if int(manifest.get("shard_index", -1)) != int(shard_index):
                continue
            if [
                int(value) for value in manifest.get("global_sample_indices", [])
            ] != expected_indices:
                continue
            if int(manifest.get("root_seed", -1)) != int(config["root_seed"]):
                continue
            if int(manifest.get("shard_generator_seed", -1)) != expected_seed:
                continue
            legacy_engine = str(
                run_config.get(
                    "canonical_engine_sha256",
                    manifest.get("canonical_engine_sha256", ""),
                )
            )
            if legacy_engine != str(engine_hash):
                continue
            legacy_audit = str(
                manifest.get("audit_sha256", run_config.get("audit_sha256", ""))
            )
            if legacy_audit not in allowed_audits:
                continue
            legacy_samples = int(legacy_case.get("run", {}).get("samples", 0))
            if legacy_samples < PRODUCTION_SAMPLES or legacy_samples % SHARD_SIZE:
                continue
            if _case_without_declared_samples(legacy_case) != current_case:
                continue
            matches.setdefault(str(row["receipt"]["archive_sha256"]), row)
        matched_rows = list(matches.values())
        matching_paths = [
            row
            for row in rows_by_revision.get(revision, [])
            if str(row["manifest"].get("case_id", "")) == str(case["case_id"])
            and int(row["manifest"].get("shard_index", -1)) == int(shard_index)
            and str(row["receipt"]["archive_sha256"]) in matches
        ]
        if len(matching_paths) > 1:
            raise RuntimeError(
                f"multiple compatible {revision} archives for "
                f"{case['case_id']} shard {shard_index}: "
                f"{[str(row['archive']) for row in matching_paths]}"
            )
        if not matched_rows:
            continue
        archive_hash, row = next(iter(matches.items()))
        manifest = row["manifest"]
        source_audit = str(
            manifest.get(
                "audit_sha256", manifest.get("run_config", {}).get("audit_sha256", "")
            )
        )
        return {
            "status": "compatible_legacy_superset",
            "archive": str(row["archive"]),
            "archive_sha256": archive_hash,
            "source_revision": revision,
            "legacy_audit_sha256": source_audit,
            "legacy_sample_count": int(
                manifest.get("run_config", {})
                .get("case", {})
                .get("run", {})
                .get("samples", 0)
            ),
        }
    return None


def _run_json(command: list[str], *, stage: str, logger: SessionLogger) -> Any:
    logger.emit(f"[{stage}] {shlex.join(command)}")
    started = time.monotonic()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    if completed.returncode:
        tail = (completed.stdout + "\n" + completed.stderr).splitlines()[
            -CHILD_TAIL_LINES:
        ]
        for line in tail:
            logger.emit(f"[CHILD] {line}")
        raise ChildProcessFailure(
            command=command,
            returncode=completed.returncode,
            output_tail=tail,
            elapsed_seconds=elapsed,
            stage=stage,
        )
    logger.emit(f"[{stage}] complete in {elapsed:.2f}s")
    return json.loads(completed.stdout)


def _child_environment() -> dict[str, str]:
    """Return an isolated child environment with terminal redraw bars disabled."""
    environment = os.environ.copy()
    environment["TQDM_DISABLE"] = "1"
    return environment


def _read_stable_child_lines(
    stream: BinaryIO, output: queue.Queue[str | None]
) -> None:
    """Read newline records while dropping carriage-return redraw frames."""
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    characters: list[str] = []
    transient = False
    pending_carriage_return = False

    def consume(text: str) -> None:
        nonlocal transient, pending_carriage_return
        for character in text:
            if pending_carriage_return:
                pending_carriage_return = False
                if character == "\n":
                    if characters and not transient:
                        output.put("".join(characters))
                    characters.clear()
                    transient = False
                    continue
                characters.clear()
                transient = True
            if character == "\r":
                pending_carriage_return = True
            elif character == "\n":
                if characters and not transient:
                    output.put("".join(characters))
                characters.clear()
                transient = False
            else:
                characters.append(character)

    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            consume(decoder.decode(chunk))
        consume(decoder.decode(b"", final=True))
        if characters and not transient and not pending_carriage_return:
            output.put("".join(characters))
    finally:
        output.put(None)


def _run_streaming(
    command: list[str],
    *,
    logger: SessionLogger,
    heartbeat_seconds: float,
    heartbeat: Callable[[int, float, str | None], None],
    on_start: Callable[[int], None] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> tuple[list[str], float]:
    """Tee one child live while retaining a bounded diagnostic tail."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=_child_environment(),
    )
    thread: threading.Thread | None = None
    tail: deque[str] = deque(maxlen=CHILD_TAIL_LINES)
    last_line: str | None = None
    next_heartbeat = started + heartbeat_seconds
    stream_closed = False
    try:
        if on_start is not None:
            on_start(process.pid)
        assert process.stdout is not None
        lines: queue.Queue[str | None] = queue.Queue()
        thread = threading.Thread(
            target=_read_stable_child_lines,
            args=(process.stdout, lines),
            name="bundle-child-output",
            daemon=True,
        )
        thread.start()
        while not stream_closed or process.poll() is None:
            timeout = max(0.05, min(0.5, next_heartbeat - time.monotonic()))
            try:
                item = lines.get(timeout=timeout)
            except queue.Empty:
                item = ""
            if item is None:
                stream_closed = True
            elif item:
                last_line = item
                tail.append(item)
                logger.emit(f"[CHILD] {item}")
                if on_line is not None:
                    on_line(item)
            now = time.monotonic()
            if now >= next_heartbeat and process.poll() is None:
                try:
                    heartbeat(process.pid, now - started, last_line)
                except Exception as exc:
                    logger.note_error("heartbeat_callback", exc)
                next_heartbeat = now + heartbeat_seconds
        returncode = process.wait()
        if thread is not None:
            thread.join(timeout=2.0)
        elapsed = time.monotonic() - started
        if returncode:
            raise ChildProcessFailure(
                command=command,
                returncode=returncode,
                output_tail=list(tail),
                elapsed_seconds=elapsed,
            )
        return list(tail), elapsed
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if thread is not None:
            thread.join(timeout=2.0)
        raise
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass


def _bundle_arguments(
    *, bundle: str, profile: str, drive_root: Path, provisional_width: int
) -> tuple[list[str], str]:
    mode = "production" if profile == "production" else "pilot"
    bundle_root = bundle_path(ROOT, bundle)
    base = [
        sys.executable,
        "-u",
        str(bundle_root / "run_bundle.py"),
        "--drive-root",
        str(drive_root),
        "--mode",
        mode,
    ]
    if mode == "pilot" and bundle != B1_BUNDLE:
        base += ["--pilot-width", str(provisional_width)]
    if mode == "production" and bundle in ("04_maxmix_master", "05_scans_and_controls"):
        decision_name = (
            "T1_gate_viable.json"
            if bundle == "04_maxmix_master"
            else "core_gate_decisions.json"
        )
        base += [
            "--gate-decisions-json",
            str(drive_root / PRODUCTION_OUTPUT_COLLECTION / bundle / decision_name),
        ]
    if mode == "production" and bundle == "05_scans_and_controls":
        m3_gate = (
            drive_root / PRODUCTION_OUTPUT_COLLECTION / bundle / "m3_bulk_gate.json"
        )
        if m3_gate.is_file():
            base += ["--m3-bulk-gate-json", str(m3_gate)]
    return base, mode


def _selected_shards(
    *,
    profile: str,
    pilot_level: str | None,
    pilot_plan: dict[str, Any],
    bundle: str,
    case_id: str,
    shard_count: int,
) -> list[int]:
    if profile == "production":
        return list(range(int(shard_count)))
    assert pilot_level is not None
    case_map = (
        pilot_plan.get("profile_case_shard_indices", {})
        .get(pilot_level, {})
        .get(bundle, {})
    )
    return [int(value) for value in case_map.get(case_id, pilot_plan["shard_indices"])]


def _session_cap(
    *, profile: str, bundles: list[str], override: float | None
) -> float | None:
    if override is not None:
        if override <= 0.0:
            raise ValueError("--max-session-hours must be positive")
        return float(override)
    if profile == "production" and bundles == [P1_BUNDLE]:
        return P1_DEFAULT_SESSION_HOURS
    if profile == "production":
        return None
    return sum(PROFILE_HOUR_CAPS[profile][bundle] for bundle in bundles)


def _p1_qualification_cycle_runtime_seconds(
    *,
    drive_root: Path,
    config: dict[str, Any],
    committer: DriveRemoteCommitter | None = None,
) -> float:
    path = (
        drive_root
        / str(config["production_output_collection"])
        / str(config.get("output_bundle", P1_BUNDLE))
        / "a100_preflight.json"
    )
    committer = committer or DriveRemoteCommitter(drive_root=drive_root)
    payload = read_remote_json(committer, path)
    committer.path_commit_record(path)
    if payload.get("safe") is not True:
        raise RuntimeError(f"P1 qualification receipt has no safe runtime: {path}")
    measured_cycles = int(
        payload.get("measured_cycles_per_trajectory", P1_L64_PHYSICAL_CYCLES)
    )
    elapsed = float(payload.get("elapsed_seconds", 0.0))
    cycle_seconds = float(
        payload.get("max_cycle_seconds")
        or payload.get("mean_cycle_seconds")
        or (elapsed / measured_cycles if measured_cycles > 0 else 0.0)
    )
    if measured_cycles <= 0 or elapsed <= 0.0 or cycle_seconds <= 0.0:
        raise RuntimeError(f"P1 qualification receipt has no safe cycle runtime: {path}")
    return cycle_seconds


def _remote_children(committer: DriveRemoteCommitter, parent_id: str) -> list[dict[str, Any]]:
    """List every non-trashed direct child using Drive API pagination."""
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        response = _execute_with_retries(
            lambda page_token=page_token: committer.service.files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken,files(id,name,mimeType,size,trashed)",
                pageSize=1000,
                pageToken=page_token,
            )
        )
        rows.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return rows


def _remote_tree_bytes(
    committer: DriveRemoteCommitter, folder_id: str
) -> tuple[int, dict[str, int]]:
    """Measure a Drive folder recursively without consulting DriveFS."""
    visited: set[str] = set()

    def measure(parent_id: str) -> int:
        if parent_id in visited:
            raise RuntimeError(f"Drive folder graph contains a cycle at {parent_id}")
        visited.add(parent_id)
        total = 0
        for item in _remote_children(committer, parent_id):
            if item.get("mimeType") == DRIVE_FOLDER_MIME_TYPE:
                total += measure(str(item["id"]))
            else:
                total += int(item.get("size", 0))
        return total

    children: dict[str, int] = {}
    total = 0
    for item in _remote_children(committer, folder_id):
        if item.get("mimeType") == DRIVE_FOLDER_MIME_TYPE:
            size = measure(str(item["id"]))
        else:
            size = int(item.get("size", 0))
        name = str(item.get("name", item.get("id", "unknown")))
        children[name] = children.get(name, 0) + size
        total += size
    return total, children


def server_storage_status(
    *,
    drive_root: Path,
    output_collection: str,
    working_limit_gb: float,
    absolute_edge_gb: float,
    required_headroom_gb: float,
    committer: DriveRemoteCommitter | None = None,
) -> dict[str, Any]:
    """Return a fail-closed server-authoritative storage decision for P1/H1."""
    decimal_gb = 1_000_000_000
    working_limit = round(float(working_limit_gb) * decimal_gb)
    absolute_edge = round(float(absolute_edge_gb) * decimal_gb)
    required_headroom = round(float(required_headroom_gb) * decimal_gb)
    if working_limit <= 0 or absolute_edge <= 0 or required_headroom < 0:
        raise ValueError("storage limits and headroom are invalid")
    if working_limit > absolute_edge:
        raise ValueError("working storage limit exceeds the absolute edge")
    relative = Path(str(output_collection))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe output collection: {output_collection!r}")
    committer = committer or DriveRemoteCommitter(drive_root=drive_root)
    quota = committer.storage_quota()
    try:
        folder_id = committer.resolve_folder(tuple(relative.parts), create=False)
    except RemoteCommitError as exc:
        if "folder is absent" not in str(exc).lower():
            raise
        used, children = 0, {}
    else:
        used, children = _remote_tree_bytes(committer, folder_id)
    projected = used + required_headroom
    account_free = (
        None
        if "limit" not in quota
        else int(quota["limit"]) - int(quota.get("usage", 0))
    )
    clear = bool(
        used < absolute_edge
        and projected <= working_limit
        and (account_free is None or account_free >= required_headroom)
    )
    return {
        "schema": "classA_drive_api_storage_guard_v1",
        "authoritative_backend": "google_drive_api_v3",
        "output_collection": str(output_collection),
        "used_bytes": used,
        "used_gb": used / decimal_gb,
        "required_headroom_bytes": required_headroom,
        "required_headroom_gb": required_headroom / decimal_gb,
        "projected_bytes": projected,
        "projected_gb": projected / decimal_gb,
        "working_limit_bytes": working_limit,
        "absolute_edge_bytes": absolute_edge,
        "clear_to_run": clear,
        "bundle_bytes": children,
        "account_storage_quota": quota,
        "account_free_bytes": account_free,
    }


def verify_h1_v3_migration_paths(
    *,
    drive_root: Path,
    config: dict[str, Any],
    migration: dict[str, Any],
    committer: DriveRemoteCommitter | None = None,
) -> None:
    """Bind every reused/rejected H1-v3 record to its exact source path."""
    policy = config.get("v3_compatibility", {})
    relative = Path(str(policy.get("source_collection", "")))
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise RuntimeError("H1-v3 source collection is missing or unsafe")
    source_root = drive_root / relative / H1_ENDPOINT_BUNDLE
    committer = committer or DriveRemoteCommitter(drive_root=drive_root)

    for row in migration.get("accepted_archives", []):
        run_id = str(row.get("run_id", ""))
        expected_name = f"{H1_ENDPOINT_BUNDLE}_{run_id}.tar.gz"
        if not re.fullmatch(r"[0-9a-f]{16}", run_id):
            raise RuntimeError(f"unsafe H1-v3 accepted run ID: {run_id!r}")
        if str(row.get("archive", "")) != expected_name:
            raise RuntimeError(
                f"H1-v3 accepted archive has the wrong name: {row.get('archive')!r}"
            )
        archive = source_root / expected_name
        receipt = archive.with_suffix(archive.suffix + ".receipt.json")
        committer.verify_record_for_path(row["archive_remote_commit"], archive)
        committer.verify_record_for_path(row["receipt_remote_commit"], receipt)

    for row in migration.get("rejected_receipt_only", []):
        run_id = str(row.get("run_id", ""))
        if not re.fullmatch(r"[0-9a-f]{16}", run_id):
            raise RuntimeError(f"unsafe H1-v3 rejected run ID: {run_id!r}")
        archive = source_root / f"{H1_ENDPOINT_BUNDLE}_{run_id}.tar.gz"
        receipt = archive.with_suffix(archive.suffix + ".receipt.json")
        committer.verify_record_for_path(row["receipt_remote_commit"], receipt)


def _remaining_child_runtime_seconds(
    *, elapsed_seconds: float, cap_hours: float | None
) -> float | None:
    if cap_hours is None:
        return None
    return max(
        0.0,
        float(cap_hours) * 3600.0
        - float(elapsed_seconds)
        - P1_RUNTIME_MARGIN_SECONDS,
    )


def _would_exceed_session_budget(
    *, elapsed_seconds: float, cap_hours: float | None, expected_seconds: float = 0.0
) -> bool:
    if cap_hours is None:
        return False
    reserve = 0.0
    if expected_seconds > 0.0:
        reserve = (
            P1_RUNTIME_SAFETY_FACTOR * float(expected_seconds)
            + P1_RUNTIME_MARGIN_SECONDS
        )
    return float(elapsed_seconds) + reserve >= float(cap_hours) * 3600.0


def _require_safe_qualification_payload(
    payload: Any, *, bundle: str
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{bundle} qualification receipt is not a JSON object")
    if payload.get("safe") is not True:
        raise RuntimeError(
            f"{bundle} qualification remains locked: safe={payload.get('safe')!r}"
        )
    return payload


def _qualification_unlock_evidence(
    *, archive_match: dict[str, Any] | None, safe_receipt: Any, bundle: str
) -> dict[str, Any]:
    if archive_match is None:
        raise RuntimeError(f"{bundle} qualification archive is not exact/current")
    return _require_safe_qualification_payload(safe_receipt, bundle=bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--bundle", choices=RUNNABLE_BUNDLES, required=True)
    parser.add_argument(
        "--case-prefix",
        action="append",
        default=[],
        help="run only case IDs beginning with this prefix; may be repeated",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="run only this exact case ID; may be repeated",
    )
    parser.add_argument(
        "--profile",
        choices=("pilot_calibration", "pilot_science", "production"),
        default="production",
    )
    parser.add_argument("--max-session-hours", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-report-only", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--working-limit-gb", type=float, default=12.0)
    parser.add_argument("--absolute-edge-gb", type=float, default=14.0)
    parser.add_argument("--required-headroom-gb", type=float, default=1.25)
    return parser


def _validated_drive_root(path: Path, *, bundle: str) -> Path:
    if bundle in SERVER_COMMIT_BUNDLES:
        # This is a logical namespace for Drive API operations.  Resolving or
        # probing it would consult DriveFS and can raise ENOTCONN after the
        # mount drops even though the server API remains healthy.
        return Path(os.path.normpath(os.path.abspath(os.fspath(path))))
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.heartbeat_seconds <= 0.0:
        raise ValueError("--heartbeat-seconds must be positive")
    if args.preflight_only and args.resume_report_only:
        raise ValueError(
            "use either --preflight-only or --resume-report-only, not both"
        )
    bundles = [str(args.bundle)]
    drive_root = _validated_drive_root(args.drive_root, bundle=str(args.bundle))
    for bundle in bundles:
        resolved_bundle = bundle_path(ROOT, bundle)
        if not (resolved_bundle / "run_bundle.py").is_file():
            raise FileNotFoundError(f"incomplete uploaded bundle: {resolved_bundle}")

    profile = str(args.profile)
    selected_bundle_config = json.loads(
        (bundle_path(ROOT, args.bundle) / "production_config.json").read_text(encoding="utf-8")
    )
    output_collection = str(
        selected_bundle_config.get(
            "production_output_collection" if profile == "production" else "pilot_output_collection",
            PRODUCTION_OUTPUT_COLLECTION if profile == "production" else PILOT_OUTPUT_COLLECTION,
        )
    )
    output_root = drive_root / output_collection
    session_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{args.bundle}_{os.getpid()}"
    )
    session_root = local_session_root(
        output_collection=output_collection, bundle=str(args.bundle)
    )
    session_path = session_root / f"{session_id}.json"
    log_path = session_root / f"{session_id}.log"
    logger = SessionLogger(log_path)
    runner_build_id = sha256_file(Path(__file__))[:16]
    session_started = time.monotonic()

    tasks: list[dict[str, Any]] = []
    flat_items: list[dict[str, Any]] = []
    bases: dict[str, list[str]] = {}
    bundle_configs: dict[str, dict[str, Any]] = {}
    bundle_cases: dict[str, dict[str, dict[str, Any]]] = {}
    bundle_engine_hashes: dict[str, str] = {}
    bundle_source_hashes: dict[str, dict[str, str]] = {}
    status_by_key: dict[tuple[str, str, int], str] = {}
    current_rows: dict[str, list[dict[str, Any]]] = {}
    completed: list[dict[str, Any]] = []
    verified_current: list[dict[str, Any]] = []
    compatible_legacy: list[dict[str, Any]] = []
    newly_completed: list[dict[str, Any]] = []
    preflight_completed: list[dict[str, Any]] = []
    a100_qualified: set[str] = set()
    identity_mismatches: list[dict[str, Any]] = []
    shard_wall_seconds: list[float] = []
    current: dict[str, Any] | None = None
    current_process: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    session_checkpoint: dict[str, Any] | None = None
    total_shards = 0
    credit_rate = 0.0
    session_cap: float | None = None
    p1_l64_expected_cycle_seconds: float | None = None
    h1_v3_compatibility: dict[tuple[str, int], dict[str, Any]] = {}
    queue_bar: Any | None = None
    parent_server_committer: DriveRemoteCommitter | None = None

    def server_committer() -> DriveRemoteCommitter:
        nonlocal parent_server_committer
        if parent_server_committer is None:
            parent_server_committer = DriveRemoteCommitter(drive_root=drive_root)
        return parent_server_committer

    def key_for(bundle: str, case_id: str, shard: int) -> tuple[str, str, int]:
        return bundle, case_id, int(shard)

    def output_bundle_for(bundle: str) -> str:
        return str(bundle_configs.get(bundle, {}).get("output_bundle", bundle))

    def per_bundle_summary() -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for bundle in bundles:
            bundle_items = [item for item in flat_items if item["bundle"] == bundle]
            counts = Counter(
                status_by_key.get(
                    key_for(bundle, item["case_id"], item["shard"]), "pending"
                )
                for item in bundle_items
            )
            result[bundle] = {
                "total": len(bundle_items),
                "verified_current": counts["verified_current"],
                "compatible_legacy": counts["compatible_legacy"],
                "newly_completed": counts["newly_completed"],
                "preflight_complete": counts["preflight_complete"],
                "pending": counts["pending"],
            }
        return result

    def report(status: str) -> None:
        elapsed_hours = (time.monotonic() - session_started) / 3600.0
        verified_total = len(verified_current) + len(compatible_legacy)
        processed = verified_total + len(newly_completed) + len(preflight_completed)
        pending = max(0, total_shards - processed)
        mean_seconds = (
            sum(shard_wall_seconds) / len(shard_wall_seconds)
            if shard_wall_seconds
            else None
        )
        payload = {
            "schema": SESSION_SCHEMA,
            "status": status,
            "runner_build_id": runner_build_id,
            "bundle_queue": args.bundle,
            "bundles": bundles,
            "profile": profile,
            "preflight_only": bool(args.preflight_only),
            "resume_report_only": bool(args.resume_report_only),
            "heartbeat_seconds": float(args.heartbeat_seconds),
            "processed_shards": processed,
            "total_shards": total_shards,
            "remaining_shards": pending,
            "verified_current_shards": len(verified_current),
            "compatible_legacy_shards": len(compatible_legacy),
            "newly_completed_shards": len(newly_completed),
            "preflight_completed_shards": len(preflight_completed),
            "pending_shards": pending,
            "bundle_summary": per_bundle_summary(),
            "output_bundles": {
                bundle: output_bundle_for(bundle) for bundle in bundles
            },
            "elapsed_hours": elapsed_hours,
            "credits": elapsed_hours * credit_rate,
            "mean_shard_minutes": None if mean_seconds is None else mean_seconds / 60.0,
            "rolling_eta_hours": (
                None if mean_seconds is None else mean_seconds * pending / 3600.0
            ),
            "session_cap_hours": session_cap,
            "p1_runtime_safety_factor": P1_RUNTIME_SAFETY_FACTOR,
            "p1_runtime_margin_seconds": P1_RUNTIME_MARGIN_SECONDS,
            "current": current,
            "current_process": current_process,
            "last_heartbeat_utc": (
                None
                if current_process is None
                else current_process.get("last_heartbeat_utc")
            ),
            "last_child_line": (
                None
                if current_process is None
                else current_process.get("last_child_line")
            ),
            "failure": failure,
            "session_checkpoint": session_checkpoint,
            "telemetry_errors": list(logger.errors),
            "identity_mismatches": identity_mismatches,
            "completed": completed,
            "session_json": str(session_path),
            "session_log": str(log_path),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _write_json_atomic(session_path, payload)
        except Exception as exc:
            logger.note_error("session_json", exc)

    def set_current(item: dict[str, Any], stage: str) -> None:
        nonlocal current
        current = {
            "ordinal": int(item["ordinal"]),
            "bundle": str(item["bundle"]),
            "case_id": str(item["case_id"]),
            "shard": int(item["shard"]),
            "stage": stage,
        }
        if queue_bar is not None:
            try:
                queue_bar.set_postfix_str(
                    f"{current['case_id']} shard={current['shard']} stage={stage}",
                    refresh=True,
                )
            except Exception as exc:
                logger.note_error("progress_postfix", exc)

    def advance_queue(count: int = 1) -> None:
        """Advance notebook-only progress without affecting durable work."""
        if queue_bar is None:
            return
        try:
            queue_bar.update(int(count))
        except Exception as exc:
            logger.note_error("progress_update", exc)

    def close_queue() -> None:
        """Close notebook-only progress without masking the queue outcome."""
        if queue_bar is None:
            return
        try:
            queue_bar.close()
        except Exception as exc:
            logger.note_error("progress_close", exc)

    def heartbeat(pid: int, elapsed: float, last_line: str | None) -> None:
        nonlocal current_process
        assert current is not None
        current_process = {
            "pid": int(pid),
            "started_utc": (
                current_process.get("started_utc") if current_process else None
            ),
            "elapsed_seconds": float(elapsed),
            "last_heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "last_child_line": last_line,
        }
        mean = (
            sum(shard_wall_seconds) / len(shard_wall_seconds)
            if shard_wall_seconds
            else None
        )
        pending = (
            total_shards
            - len(verified_current)
            - len(compatible_legacy)
            - len(newly_completed)
        )
        eta = "unknown" if mean is None else f"{mean * pending / 3600.0:.2f}h"
        logger.emit(
            f"[HEARTBEAT] item {current['ordinal']}/{total_shards} "
            f"{current['bundle']}/{current['case_id']} shard={current['shard']} "
            f"stage={current['stage']} elapsed={elapsed / 60.0:.1f}m "
            f"complete={total_shards - pending}/{total_shards} pending={pending} ETA={eta}"
        )
        report("queue_progress")

    def run_child(
        command: list[str],
        *,
        item: dict[str, Any],
        stage: str,
        on_line: Callable[[str], None] | None = None,
    ) -> float:
        nonlocal current_process
        set_current(item, stage)
        current_process = {
            "pid": None,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": 0.0,
            "last_heartbeat_utc": None,
            "last_child_line": None,
        }
        logger.emit(
            f"[{stage.upper()}] item {item['ordinal']}/{total_shards}: "
            f"{item['bundle']}/{item['case_id']} shard={item['shard']}"
        )

        def child_started(pid: int) -> None:
            assert current_process is not None
            current_process["pid"] = int(pid)
            report("child_started")

        _, elapsed = _run_streaming(
            command,
            logger=logger,
            heartbeat_seconds=float(args.heartbeat_seconds),
            heartbeat=heartbeat,
            on_start=child_started,
            on_line=on_line,
        )
        current_process = None
        return elapsed

    def p1_child_runtime_budget() -> float | None:
        elapsed_seconds = time.monotonic() - session_started
        remaining = _remaining_child_runtime_seconds(
            elapsed_seconds=elapsed_seconds,
            cap_hours=session_cap,
        )
        if remaining is not None and remaining <= 0.0:
            raise SessionBudgetReached(
                "P1 session budget reached after a durable cycle checkpoint; "
                "rerun this notebook to continue the active shard"
            )
        return remaining

    def run_p1_child(
        command: list[str], *, item: dict[str, Any], stage: str
    ) -> tuple[float, dict[str, Any] | None]:
        checkpoint_payload: dict[str, Any] | None = None

        def p1_line(line: str) -> None:
            nonlocal checkpoint_payload, session_checkpoint
            marker = line.find(P1_SESSION_CHECKPOINT_PREFIX)
            if marker < 0:
                return
            raw = line[marker + len(P1_SESSION_CHECKPOINT_PREFIX) :].strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid P1 session-checkpoint marker: {raw!r}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError("P1 session-checkpoint marker is not a JSON object")
            checkpoint_payload = payload
            session_checkpoint = dict(payload)

        budget = p1_child_runtime_budget()
        bounded_command = list(command)
        if budget is not None:
            bounded_command += ["--max-runtime-seconds", f"{budget:.6f}"]
        elapsed = run_child(
            bounded_command,
            item=item,
            stage=stage,
            on_line=p1_line,
        )
        return elapsed, checkpoint_payload

    def reconcile_server_slot(
        *,
        bundle: str,
        case: dict[str, Any],
        case_id: str,
        shard: int,
        stage_prefix: str,
        tolerate_repair_failure: bool = False,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Remotely verify one slot, repairing only a status failure.

        Normal complete and absent slots cost one read-only status query.  If
        status fails (including the final-archive/receipt crash window), the
        idempotent repair command runs once and status is retried before the
        parent makes any completion decision.
        """
        if bundle not in SERVER_COMMIT_BUNDLES:
            raise ValueError(f"{bundle} has no server reconciliation contract")
        status_command = [
            *bases[bundle],
            "--case-id",
            str(case_id),
            "--shard-index",
            str(shard),
            "--remote-status",
        ]
        initial_status_failure: ChildProcessFailure | None = None
        try:
            remote_row = _run_json(
                status_command,
                stage=f"{stage_prefix} STATUS {bundle}",
                logger=logger,
            )
        except ChildProcessFailure as exc:
            initial_status_failure = exc
            logger.emit(
                f"[{stage_prefix} STATUS WARNING] {bundle}/{case_id} "
                f"shard={shard}: {initial_status_failure}; attempting safe repair"
            )

        repair_failure: ChildProcessFailure | None = None
        repair_command = [
            *bases[bundle],
            "--case-id",
            str(case_id),
            "--shard-index",
            str(shard),
            "--repair-orphan",
        ]
        if initial_status_failure is not None:
            try:
                _run_json(
                    repair_command,
                    stage=f"{stage_prefix} REPAIR {bundle}",
                    logger=logger,
                )
            except ChildProcessFailure as exc:
                # Retry status even if repair failed.  A valid exact pair is
                # authoritative and must still be credited.
                repair_failure = exc
                logger.emit(
                    f"[{stage_prefix} REPAIR WARNING] {bundle}/{case_id} "
                    f"shard={shard}: {exc}"
                )
            remote_row = _run_json(
                status_command,
                stage=f"{stage_prefix} STATUS RETRY {bundle}",
                logger=logger,
            )
        if not isinstance(remote_row, dict):
            raise RuntimeError(
                f"{bundle} --remote-status did not return a JSON object"
            )
        matched: dict[str, Any] | None = None
        mismatches: list[dict[str, Any]] = []
        if remote_row.get("exists"):
            remote_row = _parent_verify_server_status(
                remote_row,
                drive_root=drive_root,
                profile=profile,
                bundle=bundle,
                config=bundle_configs[bundle],
                committer=server_committer(),
            )
            matched, mismatches = _cache_remote_status_row(
                remote_row,
                current_rows[bundle],
                bundle=bundle,
                case=case,
                shard_index=int(shard),
                config=bundle_configs[bundle],
                engine_hash=bundle_engine_hashes[bundle],
                source_hashes=bundle_source_hashes[bundle],
            )
        if repair_failure is not None and matched is None and not tolerate_repair_failure:
            raise repair_failure
        return matched, mismatches

    def refresh_p1_checkpoint_status(
        *, item: dict[str, Any], stage_prefix: str
    ) -> dict[str, Any] | None:
        """Best-effort server query for resumable P1 progress after a failure."""
        nonlocal session_checkpoint
        try:
            payload = _run_json(
                [
                    *bases[P1_BUNDLE],
                    "--case-id",
                    str(item["case_id"]),
                    "--shard-index",
                    str(item["shard"]),
                    "--checkpoint-status",
                ],
                stage=f"{stage_prefix} CHECKPOINT STATUS {P1_BUNDLE}",
                logger=logger,
            )
        except Exception as exc:
            logger.note_error("p1_checkpoint_status", exc)
            logger.emit(
                f"[{stage_prefix} CHECKPOINT WARNING] could not query "
                f"{item['case_id']} shard={item['shard']}: {exc}"
            )
            return None
        if not isinstance(payload, dict):
            exc = RuntimeError("P1 --checkpoint-status did not return a JSON object")
            logger.note_error("p1_checkpoint_status", exc)
            return None
        session_checkpoint = {
            "source": "server_checkpoint_status",
            **payload,
        }
        return payload

    def recover_committed_child_failure(
        *,
        exc: ChildProcessFailure,
        bundle: str,
        task: dict[str, Any],
        item: dict[str, Any],
        shard: int,
    ) -> bool:
        """Credit a child that failed only after its exact remote commit."""
        if bundle not in SERVER_COMMIT_BUNDLES:
            return False
        try:
            matched, mismatches = reconcile_server_slot(
                bundle=bundle,
                case=dict(task["case"]),
                case_id=str(task["case_id"]),
                shard=int(shard),
                stage_prefix="POST-FAILURE",
                tolerate_repair_failure=True,
            )
        except Exception as reconciliation_exc:
            logger.note_error("post_failure_reconciliation", reconciliation_exc)
            logger.emit(
                "[POST-FAILURE RECONCILIATION WARNING] "
                f"{bundle}/{task['case_id']} shard={shard}: {reconciliation_exc}"
            )
            matched = None
            mismatches = []
        if matched is not None:
            logger.emit(
                "[POST-FAILURE RECOVERED] child exited "
                f"{exc.returncode}, but {bundle}/{task['case_id']} shard={shard} "
                "is an exact server-verified commit"
            )
            return True
        if mismatches:
            identity_mismatches.append(
                {
                    "ordinal": item["ordinal"],
                    "bundle": bundle,
                    "case_id": task["case_id"],
                    "shard": int(shard),
                    "archives": mismatches,
                    "stage": "post_child_failure",
                }
            )
        if bundle == P1_BUNDLE:
            refresh_p1_checkpoint_status(item=item, stage_prefix="POST-FAILURE")
        return False

    try:
        logger.emit(
            f"[STARTUP] loading {args.bundle} profile={profile} runner={runner_build_id}"
        )
        logger.emit(f"[STARTUP] output root: {output_root}")
        pilot_plan = json.loads((ROOT / "pilot_plan.json").read_text(encoding="utf-8"))
        credit_rate = float(pilot_plan["credit_rate_per_a100_hour"])
        session_cap = _session_cap(
            profile=profile, bundles=bundles, override=args.max_session_hours
        )
        pilot_level = (
            profile.removeprefix("pilot_") if profile.startswith("pilot_") else None
        )

        for bundle in bundles:
            base, _ = _bundle_arguments(
                bundle=bundle,
                profile=profile,
                drive_root=drive_root,
                provisional_width=int(pilot_plan["provisional_width"]),
            )
            bases[bundle] = base
            rows = _run_json(
                [*base, "--list-cases-json"], stage=f"ENUMERATE {bundle}", logger=logger
            )
            shard_counts = {
                str(row["case_id"]): int(row["shard_count"]) for row in rows
            }
            cases_by_id = {str(row["case_id"]): dict(row["case"]) for row in rows}
            bundle_cases[bundle] = cases_by_id
            config = json.loads(
                (bundle_path(ROOT, bundle) / "production_config.json").read_text(encoding="utf-8")
            )
            bundle_configs[bundle] = config
            bundle_engine_hashes[bundle] = sha256_file(
                bundle_path(ROOT, bundle) / "src" / "classA_U1FGTN_gpu.py"
            )
            bundle_source_hashes[bundle] = {
                path.name: sha256_file(path)
                for path in sorted((bundle_path(ROOT, bundle) / "src").glob("*.py"))
            }
            selected_cases = (
                list(shard_counts)
                if profile == "production"
                else list(pilot_plan[pilot_level][bundle])
            )
            if args.case_id:
                selected_cases = list(
                    dict.fromkeys(str(value) for value in args.case_id)
                )
            if args.case_prefix:
                prefixes = tuple(str(value) for value in args.case_prefix)
                selected_cases = [
                    case_id
                    for case_id in selected_cases
                    if case_id.startswith(prefixes)
                ]
            unknown = sorted(set(selected_cases) - set(shard_counts))
            if unknown:
                raise KeyError(f"{bundle} pilot plan contains unknown cases: {unknown}")
            bundle_total = 0
            for case_id in selected_cases:
                shards = _selected_shards(
                    profile=profile,
                    pilot_level=pilot_level,
                    pilot_plan=pilot_plan,
                    bundle=bundle,
                    case_id=case_id,
                    shard_count=shard_counts[case_id],
                )
                invalid = [
                    value for value in shards if not 0 <= value < shard_counts[case_id]
                ]
                if invalid:
                    raise IndexError(
                        f"{bundle}/{case_id} has {shard_counts[case_id]} shards; invalid={invalid}"
                    )
                tasks.append(
                    {
                        "bundle": bundle,
                        "case_id": case_id,
                        "shards": shards,
                        "shard_count": shard_counts[case_id],
                        "case": cases_by_id[case_id],
                    }
                )
                bundle_total += len(shards)
            logger.emit(f"[ENUMERATE] {bundle}: {bundle_total} planned shard(s)")

        for task in tasks:
            task["ordinals"] = {}
            for shard in task["shards"]:
                item = {
                    "ordinal": len(flat_items) + 1,
                    "bundle": task["bundle"],
                    "case_id": task["case_id"],
                    "shard": int(shard),
                    "case": task["case"],
                }
                flat_items.append(item)
                task["ordinals"][int(shard)] = item["ordinal"]
        total_shards = len(flat_items)
        if not total_shards:
            raise ValueError("the selected bundle queue is empty")
        logger.emit(f"[STARTUP] {args.bundle}: {total_shards} planned shard(s)")

        if profile == "production" and H1_ENDPOINT_BUNDLE in bundles:
            migration = _run_json(
                [
                    sys.executable,
                    "-u",
                    str(bundle_path(ROOT, H1_ENDPOINT_BUNDLE) / "run_bundle.py"),
                    "--drive-root",
                    str(drive_root),
                    "--migration-status",
                ],
                stage="H1 V3 STRICT MIGRATION STATUS",
                logger=logger,
            )
            if (
                int(migration.get("accepted_count", -1)) != 12
                or int(migration.get("reusable_count", -1)) != 11
                or int(migration.get("rejected_receipt_only_count", -1)) != 8
            ):
                raise RuntimeError("H1-v3 server migration ledger is incomplete")
            verify_h1_v3_migration_paths(
                drive_root=drive_root,
                config=bundle_configs[H1_ENDPOINT_BUNDLE],
                migration=migration,
                committer=server_committer(),
            )
            h1_v3_compatibility = {
                (str(row["case_id"]), int(row["shard_index"])): row
                for row in migration["accepted_archives"]
                if bool(row.get("reuse_in_v4"))
            }
            logger.emit(
                "[H1 MIGRATION] 12 server-visible v3 archives validated; "
                "11 reusable slots credited and qualification slot forced to v4"
            )

        legacy_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for bundle in bundles:
            current_rows[bundle] = _current_rows_for_bundle(
                output_root / output_bundle_for(bundle),
                bundle=bundle,
                label=f"{bundle} current",
                logger=logger,
            )
            legacy_rows[bundle] = {"prior_v2": [], "v1": [], "unversioned": []}
            if profile == "production" and bundle not in SERVER_COMMIT_BUNDLES:
                legacy_rows[bundle]["prior_v2"] = _scan_archive_directory(
                    drive_root / PRIOR_V2_PRODUCTION_OUTPUT_COLLECTION / bundle,
                    label=f"{bundle} prior-v2",
                    logger=logger,
                )
                legacy_rows[bundle]["v1"] = _scan_archive_directory(
                    drive_root / V1_PRODUCTION_OUTPUT_COLLECTION / bundle,
                    label=f"{bundle} legacy-v1",
                    logger=logger,
                )
                legacy_rows[bundle]["unversioned"] = _scan_archive_directory(
                    drive_root / LEGACY_PRODUCTION_OUTPUT_COLLECTION / bundle,
                    label=f"{bundle} legacy-unversioned",
                    logger=logger,
                )

        for item in tqdm(
            flat_items,
            desc=f"match {args.bundle}",
            unit="shard",
            dynamic_ncols=True,
            leave=False,
            file=logger.display,
        ):
            bundle = str(item["bundle"])
            case = dict(item["case"])
            shard = int(item["shard"])
            key = key_for(bundle, item["case_id"], shard)
            current_row, mismatches = _find_current_archive(
                current_rows[bundle],
                bundle=bundle,
                case=case,
                shard_index=shard,
                config=bundle_configs[bundle],
                engine_hash=bundle_engine_hashes[bundle],
                source_hashes=bundle_source_hashes[bundle],
            )
            if current_row is None and bundle in (P1_BUNDLE, H1_ENDPOINT_BUNDLE):
                current_row, remote_mismatches = reconcile_server_slot(
                    bundle=bundle,
                    case=case,
                    case_id=str(item["case_id"]),
                    shard=shard,
                    stage_prefix="REMOTE RESUME",
                )
                mismatches.extend(remote_mismatches)
            if mismatches:
                identity_mismatches.append(
                    {
                        "ordinal": item["ordinal"],
                        "bundle": bundle,
                        "case_id": item["case_id"],
                        "shard": shard,
                        "archives": mismatches,
                    }
                )
            if current_row is not None:
                status_by_key[key] = "verified_current"
                record = {
                    "ordinal": item["ordinal"],
                    "bundle": bundle,
                    "case_id": item["case_id"],
                    "shard": shard,
                    "archive": str(current_row["archive"]),
                    "archive_sha256": current_row["receipt"]["archive_sha256"],
                }
                verified_current.append(record)
                completed.append(record)
                continue
            pinned_h1_v3 = (
                h1_v3_compatibility.get((str(item["case_id"]), shard))
                if bundle == H1_ENDPOINT_BUNDLE and profile == "production"
                else None
            )
            legacy = (
                {
                    "status": "compatible_h1_v3_server_verified",
                    "archive": str(
                        drive_root
                        / bundle_configs[bundle]["v3_compatibility"]["source_collection"]
                        / bundle
                        / pinned_h1_v3["archive"]
                    ),
                    "archive_sha256": pinned_h1_v3["archive_sha256"],
                    "source_revision": pinned_h1_v3.get("source_revision", "v3"),
                    "archive_remote_commit": pinned_h1_v3["archive_remote_commit"],
                    "receipt_remote_commit": pinned_h1_v3["receipt_remote_commit"],
                }
                if pinned_h1_v3 is not None
                else
                _find_compatible_legacy_archive(
                    legacy_rows[bundle],
                    bundle=bundle,
                    case=case,
                    shard_index=shard,
                    config=bundle_configs[bundle],
                    engine_hash=bundle_engine_hashes[bundle],
                )
                if profile == "production" and bundle != H1_ENDPOINT_BUNDLE
                else None
            )
            if legacy is not None:
                status_by_key[key] = "compatible_legacy"
                record = {
                    "ordinal": item["ordinal"],
                    "bundle": bundle,
                    "case_id": item["case_id"],
                    "shard": shard,
                    "reuse": legacy,
                }
                compatible_legacy.append(record)
                completed.append(record)
            else:
                status_by_key[key] = "pending"

        summary = per_bundle_summary()
        for bundle in bundles:
            row = summary[bundle]
            logger.emit(
                f"[SUMMARY] {bundle}: {row['total']} total, "
                f"{row['verified_current']} current, {row['compatible_legacy']} legacy, "
                f"{row['pending']} pending"
            )
        verified_total = len(verified_current) + len(compatible_legacy)
        logger.emit(
            f"[SUMMARY] {verified_total} verified complete "
            f"({len(verified_current)} current + {len(compatible_legacy)} legacy), "
            f"{total_shards - verified_total} pending"
        )
        pending_items = [
            item
            for item in flat_items
            if status_by_key[key_for(item["bundle"], item["case_id"], item["shard"])]
            == "pending"
        ]
        if pending_items:
            item = pending_items[0]
            logger.emit(
                f"[NEXT] item {item['ordinal']}/{total_shards}: "
                f"{item['bundle']}/{item['case_id']} shard={item['shard']}"
            )
        else:
            logger.emit("[NEXT] no pending numerical shards")
        report("resume_scan_complete")
        if args.resume_report_only:
            report("resume_report_complete")
            logger.emit(f"[SESSION] {session_path}")
            return 0

        last_task_for_bundle = {
            bundle: max(
                index for index, task in enumerate(tasks) if task["bundle"] == bundle
            )
            for bundle in bundles
        }

        def require_time_budget(*, expected_seconds: float = 0.0) -> None:
            elapsed_seconds = time.monotonic() - session_started
            if _would_exceed_session_budget(
                elapsed_seconds=elapsed_seconds,
                cap_hours=session_cap,
                expected_seconds=expected_seconds,
            ):
                projected_hours = (
                    elapsed_seconds
                    + (
                        P1_RUNTIME_SAFETY_FACTOR * expected_seconds
                        + P1_RUNTIME_MARGIN_SECONDS
                        if expected_seconds > 0.0
                        else 0.0
                    )
                ) / 3600.0
                raise SessionBudgetReached(
                    "bundle session budget reached at a durable cycle checkpoint "
                    f"(projected={projected_hours:.3f} h, cap={session_cap:.3f} h); "
                    "rerun this notebook to checksum-resume the remaining work"
                )

        def run_storage_guard(bundle: str, item: dict[str, Any]) -> None:
            required_headroom_gb = float(args.required_headroom_gb)
            if bundle == P1_BUNDLE:
                required_headroom_gb = max(required_headroom_gb, 1.25)
            if bundle in SERVER_COMMIT_BUNDLES:
                set_current(item, "storage")
                logger.emit(
                    f"[STORAGE] item {item['ordinal']}/{total_shards}: "
                    f"{item['bundle']}/{item['case_id']} shard={item['shard']}"
                )
                status = server_storage_status(
                    drive_root=drive_root,
                    output_collection=output_collection,
                    working_limit_gb=float(args.working_limit_gb),
                    absolute_edge_gb=float(args.absolute_edge_gb),
                    required_headroom_gb=required_headroom_gb,
                )
                for line in json.dumps(status, indent=2, sort_keys=True).splitlines():
                    logger.emit(f"[STORAGE API] {line}")
                if not status["clear_to_run"]:
                    raise RuntimeError(
                        "server-authoritative Drive storage guard blocked this launch"
                    )
                return
            command = [
                sys.executable,
                str(bundle_path(ROOT, bundle) / "src" / "drive_storage_guard.py"),
                "--output-root",
                str(output_root),
                "--drive-root",
                str(drive_root),
                "--working-limit-gb",
                str(args.working_limit_gb),
                "--absolute-edge-gb",
                str(args.absolute_edge_gb),
                "--required-headroom-gb",
                str(required_headroom_gb),
            ]
            run_child(command, item=item, stage="storage")

        def ingest_new_outputs(bundle: str) -> list[dict[str, Any]]:
            if bundle in SERVER_COMMIT_BUNDLES:
                logger.emit(
                    f"[VERIFY] {bundle}: Drive API remote-status is authoritative; "
                    "skipping the DriveFS ingest scan"
                )
                return []
            known = {str(row["archive"]) for row in current_rows[bundle]}
            root = output_root / output_bundle_for(bundle)
            candidates = [
                path for path in sorted(root.glob("*.tar.gz")) if str(path) not in known
            ]
            if candidates:
                logger.emit(
                    f"[VERIFY] {bundle}: checking {len(candidates)} newly created archive(s)"
                )
            new_rows: list[dict[str, Any]] = []
            for archive in tqdm(
                candidates,
                desc=f"verify new {bundle}",
                unit="archive",
                dynamic_ncols=True,
                leave=False,
                file=logger.display,
            ):
                new_rows.append(
                    {
                        "archive": archive,
                        "receipt": verify_archive_receipt(archive),
                        "manifest": _root_manifest_from_archive(archive),
                    }
                )
            current_rows[bundle].extend(new_rows)
            return new_rows

        def verify_new_outputs(
            bundle: str, task: dict[str, Any], shards: list[int]
        ) -> None:
            if bundle not in SERVER_COMMIT_BUNDLES:
                ingest_new_outputs(bundle)
            for shard in shards:
                item = next(
                    row
                    for row in flat_items
                    if row["bundle"] == bundle
                    and row["case_id"] == task["case_id"]
                    and int(row["shard"]) == int(shard)
                )
                if bundle in SERVER_COMMIT_BUNDLES:
                    matched, mismatches = reconcile_server_slot(
                        bundle=bundle,
                        case=dict(task["case"]),
                        case_id=str(task["case_id"]),
                        shard=int(shard),
                        stage_prefix="REMOTE VERIFY",
                    )
                    if matched is None:
                        reason = (
                            "Drive API reports no durable archive"
                            if not mismatches
                            else "Drive API reports a durable artifact with the wrong identity"
                        )
                        raise RuntimeError(
                            f"child finished, but {reason} for "
                            f"{bundle}/{task['case_id']} shard {shard}: {mismatches}"
                        )
                else:
                    matched, _ = _find_current_archive(
                        current_rows[bundle],
                        bundle=bundle,
                        case=task["case"],
                        shard_index=shard,
                        config=bundle_configs[bundle],
                        engine_hash=bundle_engine_hashes[bundle],
                        source_hashes=bundle_source_hashes[bundle],
                    )
                if matched is None:
                    raise RuntimeError(
                        f"child returned successfully but no exact verified archive exists for "
                        f"{bundle}/{task['case_id']} shard {shard}"
                    )
                key = key_for(bundle, task["case_id"], shard)
                status_by_key[key] = "newly_completed"
                record = {
                    "ordinal": item["ordinal"],
                    "bundle": bundle,
                    "case_id": task["case_id"],
                    "shard": shard,
                    "archive": str(matched["archive"]),
                    "archive_sha256": matched["receipt"]["archive_sha256"],
                }
                newly_completed.append(record)
                completed.append(record)
                logger.emit(
                    f"[DONE] item {item['ordinal']}/{total_shards}: archive and receipt verified"
                )

        def finalize_bundle(bundle: str, item: dict[str, Any]) -> None:
            if profile != "production":
                return
            if args.case_id or args.case_prefix:
                selected_ids = {str(row["case_id"]) for row in flat_items}
                complete_gate_filter = (
                    bundle == "01_bulk_width_gate"
                    and len(selected_ids) == 20
                    and all(case_id.startswith("W1_") for case_id in selected_ids)
                ) or (
                    bundle == "05_scans_and_controls"
                    and len(selected_ids) == 45
                    and all(case_id.startswith("M3BULK_") for case_id in selected_ids)
                )
                if not complete_gate_filter:
                    logger.emit(
                        "[FINALIZE] skipped: this filtered queue is not a complete gate matrix"
                    )
                    return
            if bundle == "01_bulk_width_gate":
                gate_output = output_root / bundle / "fixed_geometry_baseline_manifest.json"
                baseline_rows = [
                    row
                    for row in completed
                    if row["bundle"] == bundle and str(row["case_id"]).startswith("W1_")
                ]
                write_json_atomic(
                    gate_output,
                    {
                        "schema_version": 1,
                        "status": "complete_non_gating",
                        "fixed_Nx": 20,
                        "Ny": [20, 30, 40, 50, 60],
                        "case_count": 20,
                        "shard_count": 40,
                        "archives": baseline_rows,
                    },
                )
                logger.emit(f"[BASELINE] wrote {gate_output}")
                report("production_fixed_geometry_baseline_complete")
                return
            elif bundle == "05_scans_and_controls":
                gate_output = output_root / bundle / "m3_bulk_gate.json"
                command = [
                    sys.executable,
                    "-u",
                    str(bundle_path(ROOT, bundle) / "src" / "gate_analysis.py"),
                    "m3_bulk",
                    "--archive-root",
                    str(output_root / bundle),
                    "--output",
                    str(gate_output),
                ]
                status = (
                    "production_m3_bulk_gate_complete_rerun_bundle_for_wall_bracket"
                )
            else:
                return
            run_child(command, item=item, stage="gate-analysis")
            gate_payload = json.loads(gate_output.read_text(encoding="utf-8"))
            logger.emit(f"[GATE] {json.dumps(gate_payload, sort_keys=True)}")
            if gate_payload.get("status") != "accepted":
                raise RuntimeError(
                    f"{bundle} gate analysis did not accept the completed matrix: "
                    f"{gate_payload.get('status')} ({gate_output})"
                )
            report(status)

        try:
            queue_bar = tqdm(
                total=total_shards,
                initial=verified_total,
                desc=args.bundle,
                unit="shard",
                dynamic_ncols=True,
                file=logger.display,
            )
        except Exception as exc:
            logger.note_error("progress_open", exc)
            queue_bar = None
        try:
            for task_index, task in enumerate(tasks):
                bundle = str(task["bundle"])
                case_id = str(task["case_id"])
                pending_shards = [
                    int(shard)
                    for shard in task["shards"]
                    if status_by_key[key_for(bundle, case_id, shard)] == "pending"
                ]
                last_item = {
                    "ordinal": task["ordinals"][int(task["shards"][-1])],
                    "bundle": bundle,
                    "case_id": case_id,
                    "shard": int(task["shards"][-1]),
                }
                if not pending_shards:
                    if task_index == last_task_for_bundle[bundle]:
                        finalize_bundle(bundle, last_item)
                    continue
                base = bases[bundle]
                if (
                    profile == "production"
                    and not args.preflight_only
                    and bundle not in a100_qualified
                ):
                    bootstrap_task = next(
                        (
                            candidate
                            for candidate in tasks
                            if candidate["bundle"] == P1_BUNDLE
                            and candidate["case_id"] == "P1_CHERN_L64_nsh-1"
                        ),
                        None,
                    )
                    h1_qualification_task = next(
                        (
                            candidate
                            for candidate in tasks
                            if candidate["bundle"] == H1_ENDPOINT_BUNDLE
                            and candidate["case_id"] == "H1_N20x40_soft_a1-1"
                        ),
                        None,
                    )
                    qualification_task = bootstrap_task or h1_qualification_task
                    if bundle == P1_BUNDLE and bootstrap_task is None:
                        qualification_item = {
                            "ordinal": task["ordinals"][pending_shards[0]],
                            "bundle": bundle,
                            "case_id": "P1_CHERN_L64_nsh-1",
                            "shard": 0,
                        }
                    elif bundle == H1_ENDPOINT_BUNDLE and h1_qualification_task is None:
                        qualification_item = {
                            "ordinal": task["ordinals"][pending_shards[0]],
                            "bundle": bundle,
                            "case_id": "H1_N20x40_soft_a1-1",
                            "shard": 0,
                        }
                    else:
                        qualification_item = (
                            {
                                "ordinal": qualification_task["ordinals"][0],
                                "bundle": bundle,
                                "case_id": qualification_task["case_id"],
                                "shard": 0,
                            }
                            if qualification_task is not None
                            else {
                                "ordinal": task["ordinals"][pending_shards[0]],
                                "bundle": bundle,
                                "case_id": case_id,
                                "shard": pending_shards[0],
                            }
                        )
                    qualification_reconcile_task = {
                        "case_id": str(qualification_item["case_id"]),
                        "case": bundle_cases[bundle][str(qualification_item["case_id"])],
                    }
                    require_time_budget()
                    logger.emit(
                        f"[A100 QUALIFICATION] run or reuse the current safe receipt for {bundle}"
                    )
                    if bundle in (P1_BUNDLE, H1_ENDPOINT_BUNDLE):
                        run_storage_guard(bundle, qualification_item)
                    if bundle == P1_BUNDLE:
                        try:
                            _, checkpoint_payload = run_p1_child(
                                [*base, "--a100-preflight"],
                                item=qualification_item,
                                stage="a100-preflight",
                            )
                        except ChildProcessFailure as exc:
                            if not recover_committed_child_failure(
                                exc=exc,
                                bundle=bundle,
                                task=qualification_reconcile_task,
                                item=qualification_item,
                                shard=0,
                            ):
                                raise
                            checkpoint_payload = None
                        if checkpoint_payload is not None:
                            raise SessionBudgetReached(
                                "P1 qualification stopped after a durable cycle "
                                f"checkpoint ({checkpoint_payload}); rerun this notebook "
                                "to continue"
                            )
                    else:
                        try:
                            run_child(
                                [*base, "--a100-preflight"],
                                item=qualification_item,
                                stage="a100-preflight",
                            )
                        except ChildProcessFailure as exc:
                            if not recover_committed_child_failure(
                                exc=exc,
                                bundle=bundle,
                                task=qualification_reconcile_task,
                                item=qualification_item,
                                shard=0,
                            ):
                                raise
                    qualification_match, qualification_mismatches = (
                        reconcile_server_slot(
                            bundle=bundle,
                            case=dict(qualification_reconcile_task["case"]),
                            case_id=str(qualification_reconcile_task["case_id"]),
                            shard=0,
                            stage_prefix="POST-QUALIFICATION",
                        )
                    )
                    if qualification_match is None:
                        raise RuntimeError(
                            "A100 qualification child did not leave an exact "
                            f"server-verified archive for {bundle}/"
                            f"{qualification_reconcile_task['case_id']} shard 0: "
                            f"{qualification_mismatches}"
                        )
                    # An exact archive proves durability, not qualification
                    # safety.  Re-enter the wrapper after every attempt (and
                    # after response-loss recovery) so it must validate the
                    # separately committed safe A100 receipt before any
                    # production case can be unlocked.
                    qualification_receipt = _qualification_unlock_evidence(
                        archive_match=qualification_match,
                        safe_receipt=_run_json(
                            [*base, "--a100-preflight"],
                            stage=f"A100 SAFE RECEIPT VERIFY {bundle}",
                            logger=logger,
                        ),
                        bundle=bundle,
                    )
                    logger.emit(
                        "[A100 QUALIFICATION VERIFIED] "
                        f"{bundle}: safe receipt accepted "
                        f"({qualification_receipt.get('receipt_path', 'server')})"
                    )
                    a100_qualified.add(bundle)
                    ingest_new_outputs(bundle)
                    if bundle == P1_BUNDLE:
                        p1_l64_expected_cycle_seconds = _p1_qualification_cycle_runtime_seconds(
                            drive_root=drive_root,
                            config=bundle_configs[bundle],
                        )
                        if bootstrap_task is not None:
                            bootstrap_key = key_for(
                                bundle, bootstrap_task["case_id"], 0
                            )
                            if status_by_key[bootstrap_key] == "pending":
                                verify_new_outputs(bundle, bootstrap_task, [0])
                                advance_queue(1)
                        pending_shards = [
                            int(shard)
                            for shard in task["shards"]
                            if status_by_key[
                                key_for(bundle, case_id, shard)
                            ]
                            == "pending"
                        ]
                    elif bundle == H1_ENDPOINT_BUNDLE and h1_qualification_task is not None:
                        qualification_key = key_for(
                            bundle, h1_qualification_task["case_id"], 0
                        )
                        if status_by_key[qualification_key] == "pending":
                            verify_new_outputs(bundle, h1_qualification_task, [0])
                            advance_queue(1)
                        pending_shards = [
                            int(shard)
                            for shard in task["shards"]
                            if status_by_key[key_for(bundle, case_id, shard)]
                            == "pending"
                        ]
                    report("a100_preflight_complete")
                    if not pending_shards:
                        if task_index == last_task_for_bundle[bundle]:
                            finalize_bundle(bundle, last_item)
                        continue
                if bundle == B1_BUNDLE:
                    for shard in pending_shards:
                        item = {
                            "ordinal": task["ordinals"][shard],
                            "bundle": bundle,
                            "case_id": case_id,
                            "shard": shard,
                        }
                        command = [
                            *base,
                            "--case-id",
                            case_id,
                            "--shard-index",
                            str(shard),
                            "--preflight-only",
                        ]
                        run_child(command, item=item, stage="preflight")
                        if args.preflight_only:
                            status_by_key[key_for(bundle, case_id, shard)] = (
                                "preflight_complete"
                            )
                            preflight_completed.append(item)
                            completed.append({**item, "preflight": True})
                            advance_queue(1)
                    if args.preflight_only:
                        report("preflight_progress")
                        continue
                    require_time_budget()
                    first_item = {
                        "ordinal": task["ordinals"][pending_shards[0]],
                        "bundle": bundle,
                        "case_id": case_id,
                        "shard": pending_shards[0],
                    }
                    run_storage_guard(bundle, first_item)
                    command = [
                        *base,
                        "--case-id",
                        case_id,
                        "--shard-indices",
                        *[str(value) for value in pending_shards],
                    ]

                    def b1_line(line: str) -> None:
                        match = re.search(r"\[B1 start\].*?shard=(\d+)/(\d+)", line)
                        if match is None:
                            return
                        shard = int(match.group(1))
                        if shard not in task["ordinals"]:
                            return
                        set_current(
                            {
                                "ordinal": task["ordinals"][shard],
                                "bundle": bundle,
                                "case_id": case_id,
                                "shard": shard,
                            },
                            "running",
                        )
                        report("queue_progress")

                    elapsed = run_child(
                        command,
                        item=first_item,
                        stage="running",
                        on_line=b1_line,
                    )
                    per_shard = elapsed / len(pending_shards)
                    shard_wall_seconds.extend([per_shard] * len(pending_shards))
                    verify_new_outputs(bundle, task, pending_shards)
                    advance_queue(len(pending_shards))
                    report("queue_progress")
                else:
                    for shard in pending_shards:
                        item = {
                            "ordinal": task["ordinals"][shard],
                            "bundle": bundle,
                            "case_id": case_id,
                            "shard": shard,
                        }
                        require_time_budget()
                        command = [
                            *base,
                            "--case-id",
                            case_id,
                            "--shard-index",
                            str(shard),
                            "--preflight-only",
                        ]
                        run_child(command, item=item, stage="preflight")
                        if args.preflight_only:
                            status_by_key[key_for(bundle, case_id, shard)] = (
                                "preflight_complete"
                            )
                            preflight_completed.append(item)
                            completed.append({**item, "preflight": True})
                            advance_queue(1)
                            report("preflight_progress")
                            continue
                        run_storage_guard(bundle, item)
                        expected_seconds = 0.0
                        if bundle == P1_BUNDLE:
                            if p1_l64_expected_cycle_seconds is None:
                                p1_l64_expected_cycle_seconds = (
                                    _p1_qualification_cycle_runtime_seconds(
                                        drive_root=drive_root,
                                        config=bundle_configs[bundle],
                                    )
                                )
                            # Qualification measures the largest geometry.  Its
                            # maximum cycle time is a conservative launch bound
                            # for every smaller P1 geometry as well.
                            expected_seconds = p1_l64_expected_cycle_seconds
                        require_time_budget(expected_seconds=expected_seconds)
                        command = [
                            *base,
                            "--case-id",
                            case_id,
                            "--shard-index",
                            str(shard),
                        ]
                        if bundle == P1_BUNDLE:
                            try:
                                elapsed, checkpoint_payload = run_p1_child(
                                    command, item=item, stage="running"
                                )
                            except ChildProcessFailure as exc:
                                if not recover_committed_child_failure(
                                    exc=exc,
                                    bundle=bundle,
                                    task=task,
                                    item=item,
                                    shard=shard,
                                ):
                                    raise
                                elapsed = exc.elapsed_seconds
                                checkpoint_payload = None
                            if checkpoint_payload is not None:
                                raise SessionBudgetReached(
                                    "P1 stopped after a durable cycle checkpoint "
                                    f"({checkpoint_payload}); rerun this notebook to continue"
                                )
                        else:
                            try:
                                elapsed = run_child(
                                    command, item=item, stage="running"
                                )
                            except ChildProcessFailure as exc:
                                if not recover_committed_child_failure(
                                    exc=exc,
                                    bundle=bundle,
                                    task=task,
                                    item=item,
                                    shard=shard,
                                ):
                                    raise
                                elapsed = exc.elapsed_seconds
                        shard_wall_seconds.append(elapsed)
                        verify_new_outputs(bundle, task, [shard])
                        advance_queue(1)
                        report("queue_progress")
                if (
                    task_index == last_task_for_bundle[bundle]
                    and not args.preflight_only
                ):
                    finalize_bundle(bundle, last_item)
        finally:
            close_queue()

        current = None
        current_process = None
        report("preflight_complete" if args.preflight_only else "queue_complete")
        logger.emit(
            f"[COMPLETE] {args.bundle}: processed {total_shards}/{total_shards}"
        )
        logger.emit(f"[SESSION] {session_path}")
        return 0
    except SessionBudgetReached as exc:
        if current is not None and session_checkpoint is not None:
            current = {**current, "stage": "cycle_checkpointed"}
        else:
            current = None
        current_process = None
        report("session_budget_reached")
        logger.emit(f"[SESSION BUDGET] {exc}")
        logger.emit(f"[SESSION] {session_path}")
        return 0
    except BaseException as exc:
        if isinstance(exc, ChildProcessFailure):
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": exc.stage
                or (None if current is None else current.get("stage")),
                "ordinal": None if current is None else current.get("ordinal"),
                "bundle": None if current is None else current.get("bundle"),
                "case_id": None if current is None else current.get("case_id"),
                "shard": None if current is None else current.get("shard"),
                "command": exc.command,
                "command_text": shlex.join(exc.command),
                "returncode": exc.returncode,
                "elapsed_seconds": exc.elapsed_seconds,
                "output_tail": exc.output_tail,
                "session_json": str(session_path),
                "session_log": str(log_path),
            }
        else:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": None if current is None else current.get("stage"),
                "ordinal": None if current is None else current.get("ordinal"),
                "bundle": None if current is None else current.get("bundle"),
                "case_id": None if current is None else current.get("case_id"),
                "shard": None if current is None else current.get("shard"),
                "session_json": str(session_path),
                "session_log": str(log_path),
            }
        if session_checkpoint is not None:
            failure["p1_checkpoint_status"] = session_checkpoint
        report("failed")
        logger.emit("[FAILURE] " + json.dumps(failure, indent=2, sort_keys=True))
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
