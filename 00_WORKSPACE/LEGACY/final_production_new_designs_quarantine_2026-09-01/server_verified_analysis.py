#!/usr/bin/env python3
"""Materialize, analyze, and publish P1/H1 using Drive API truth only.

The script is intentionally outside every bundle ``src`` directory.  It keeps
the qualified scientific snapshots immutable while making the final analysis
independent of a healthy DriveFS mount.  Exact current slots are enumerated by
the bundle wrappers' ``--remote-status`` command, downloaded to local Colab
disk by file ID, analyzed locally, and uploaded as an immutable generation.
Only after every output is reverified does a replaceable analysis receipt point
at that generation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).absolute().parent
SHARED = ROOT / "_shared_src"
for candidate in (SHARED, ROOT):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

_DRIVE_HELPER_PATH = (
    ROOT / "drive_remote_commit.py"
    if (ROOT / "drive_remote_commit.py").is_file()
    else SHARED / "drive_remote_commit.py"
)
_DRIVE_SPEC = importlib.util.spec_from_file_location(
    "classA_server_analysis_drive_remote_commit", _DRIVE_HELPER_PATH
)
if _DRIVE_SPEC is None or _DRIVE_SPEC.loader is None:
    raise RuntimeError(f"cannot load operational Drive helper: {_DRIVE_HELPER_PATH}")
_DRIVE = importlib.util.module_from_spec(_DRIVE_SPEC)
_DRIVE_SPEC.loader.exec_module(_DRIVE)
DriveRemoteCommitter = _DRIVE.DriveRemoteCommitter
RemoteCommitError = _DRIVE.RemoteCommitError
publish_json = _DRIVE.publish_json
sha256_file = _DRIVE.sha256_file


P1 = "01_p1_chern_dynamics"
H1 = "08_h1_endpoint_packet"
ANALYSIS_RECEIPT_SCHEMA = "classA_server_verified_analysis_v1"
ANALYSIS_LEASE_SCHEMA = "classA_server_verified_analysis_lease_v1"
ANALYSIS_LEASE_HEARTBEAT_SECONDS = 30.0
ANALYSIS_LEASE_STALE_SECONDS = 900.0
ANALYSIS_LEASE_SETTLE_SECONDS = 0.5
REMOTE_COMMIT_FIELDS = (
    "schema",
    "remote_file_id",
    "remote_name",
    "remote_parent_id",
    "remote_bytes",
    "remote_sha256",
)
SPECS: dict[str, dict[str, Any]] = {
    P1: {
        "case_count": 12,
        "slot_count": 120,
        "analysis_script": "p1_chern_analysis.py",
        "analysis_directory": "P1_chern_analysis",
        "summary": "p1_chern_analysis_summary.json",
        "summary_statuses": {"complete"},
        "cache_directory": "p1_archives",
    },
    H1: {
        "case_count": 4,
        "slot_count": 20,
        "analysis_script": "h1_packet_analysis.py",
        "analysis_directory": "analysis_outputs",
        "summary": "h1_endpoint_analysis_summary.json",
        "summary_statuses": {"accepted", "not_accepted"},
        "cache_directory": "h1_archives",
    },
}


def _lexical(path: Path | str) -> Path:
    """Normalize without consulting a possibly disconnected FUSE mount."""

    return Path(os.path.normpath(os.path.abspath(os.fspath(path))))


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _commit_identity(record: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in REMOTE_COMMIT_FIELDS if name not in record]
    if missing:
        raise RuntimeError(f"remote commit record is incomplete: {missing}")
    return {name: record[name] for name in REMOTE_COMMIT_FIELDS}


def _is_absent(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "absent" in message or "not found" in message or "404" in message


def _refresh_exact_named_record(
    committer: Any,
    record: dict[str, Any],
    *,
    expected_name: str,
    expected_parent_id: str,
) -> dict[str, Any]:
    metadata = committer.metadata(str(record["remote_file_id"]))
    refreshed = committer.commit_record(metadata)
    if _commit_identity(refreshed) != _commit_identity(record):
        raise RuntimeError("analysis lease object changed during exact-ID refresh")
    if (
        str(refreshed["remote_name"]) != str(expected_name)
        or str(refreshed["remote_parent_id"]) != str(expected_parent_id)
    ):
        raise RuntimeError("analysis lease object moved outside its exact path")
    committer.verify_commit_record(refreshed)
    return refreshed


class _AnalysisApiLease:
    """Bundle-level single-writer fence for materialization through publish."""

    def __init__(
        self,
        *,
        bundle: str,
        config: dict[str, Any],
        path: Path,
        committer: Any,
        heartbeat_committer: Any | None = None,
    ) -> None:
        self.bundle = str(bundle)
        self.revision = str(config["sampling_revision"])
        self.audit = str(config["audit_sha256"])
        self.path = _lexical(path)
        self.committer = committer
        if heartbeat_committer is not None:
            self.heartbeat_committer = heartbeat_committer
        elif isinstance(committer, DriveRemoteCommitter):
            # googleapiclient's httplib2 transport is not thread-safe.  A
            # second service shares the already-authorized credentials while
            # isolating heartbeat requests from main-thread downloads/uploads.
            self.heartbeat_committer = DriveRemoteCommitter(
                drive_root=committer.drive_root
            )
        else:
            self.heartbeat_committer = committer
        self.token = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lost_reason: str | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_LEASE_SCHEMA,
            "bundle": self.bundle,
            "sampling_revision": self.revision,
            "audit_sha256": self.audit,
            "owner_token": self.token,
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "updated_unix": time.time(),
        }

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        expected = {
            "schema": ANALYSIS_LEASE_SCHEMA,
            "bundle": self.bundle,
            "sampling_revision": self.revision,
            "audit_sha256": self.audit,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        token = str(payload.get("owner_token", ""))
        if mismatches or len(token) != 32 or any(
            character not in "0123456789abcdef" for character in token
        ):
            raise RuntimeError(f"analysis API lease has invalid identity: {mismatches}")

    def _read_record(
        self, committer: Any | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        committer = self.committer if committer is None else committer
        try:
            record = committer.path_commit_record(self.path)
        except RemoteCommitError as exc:
            if _is_absent(exc):
                return None, None
            raise
        payload, exact = _remote_json_exact(
            committer, self.path, declared=record
        )
        self._validate_payload(payload)
        return payload, exact

    def _claims(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        parts, name = self.committer.split_remote_path(self.path)
        try:
            parent_id = self.committer.resolve_folder(parts, create=False)
        except RemoteCommitError as exc:
            if _is_absent(exc):
                return []
            raise
        claims: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for item in self.committer._list_children(parent_id, name):
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
            except RemoteCommitError as exc:
                if _is_absent(exc):
                    continue
                raise
            if len(raw) != int(record["remote_bytes"]):
                raise RuntimeError("analysis API lease has the wrong byte count")
            if hashlib.sha256(raw).hexdigest() != str(record["remote_sha256"]):
                raise RuntimeError("analysis API lease failed its checksum")
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("analysis API lease is not a JSON object")
            self._validate_payload(payload)
            claims.append((payload, record))
        return claims

    def _delete(self, payload: dict[str, Any], record: dict[str, Any]) -> None:
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
            raise RuntimeError("analysis API lease changed before exact-ID deletion")
        current = json.loads(raw.decode("utf-8"))
        if not isinstance(current, dict):
            raise RuntimeError("analysis API lease is not a JSON object")
        self._validate_payload(current)
        if _normalized(current) != _normalized(payload):
            raise RuntimeError("analysis API lease payload changed before deletion")
        self.committer.delete_verified_if_present(refreshed)

    def _elect(self, *, require_own_claim: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        for _ in range(5):
            claims = self._claims()
            if not claims:
                raise RuntimeError("analysis API lease vanished during election")
            own = [row for row in claims if row[0]["owner_token"] == self.token]
            if require_own_claim and not own:
                raise RuntimeError("another analysis runtime won the API lease claim")
            winner = min(
                claims,
                key=lambda row: (
                    str(row[0]["owner_token"]),
                    str(row[1]["remote_file_id"]),
                ),
            )
            if require_own_claim and winner[0]["owner_token"] != self.token:
                for payload, record in own:
                    self._delete(payload, record)
                raise RuntimeError("another analysis runtime won the API lease claim")
            winner_id = str(winner[1]["remote_file_id"])
            for payload, record in claims:
                if str(record["remote_file_id"]) != winner_id:
                    self._delete(payload, record)
            if ANALYSIS_LEASE_SETTLE_SECONDS > 0:
                time.sleep(ANALYSIS_LEASE_SETTLE_SECONDS)
            remaining = self._claims()
            if (
                len(remaining) == 1
                and str(remaining[0][1]["remote_file_id"]) == winner_id
            ):
                return remaining[0]
        raise RuntimeError("analysis API lease did not converge to one claimant")

    def assert_owned(self, committer: Any | None = None) -> None:
        if self._lost_reason is not None:
            raise RuntimeError(
                f"analysis API lease heartbeat failed: {self._lost_reason}"
            )
        payload, _ = self._read_record(committer)
        if payload is None or payload.get("owner_token") != self.token:
            raise RuntimeError(f"analysis API lease ownership was lost: {payload}")
        self._validate_payload(payload)

    def _heartbeat(self) -> None:
        while not self._stop.wait(ANALYSIS_LEASE_HEARTBEAT_SECONDS):
            try:
                self.assert_owned(self.heartbeat_committer)
                publish_json(
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

    def __enter__(self) -> "_AnalysisApiLease":
        try:
            current, current_record = self._read_record()
        except RemoteCommitError as exc:
            if "ambiguous" not in str(exc).lower():
                raise
            current, current_record = self._elect(require_own_claim=False)
        if current is not None:
            age = time.time() - float(current.get("updated_unix", 0.0))
            if age < ANALYSIS_LEASE_STALE_SECONDS:
                raise RuntimeError(
                    f"another analysis runtime owns the API lease: {current}"
                )
            assert current_record is not None
            latest, latest_record = self._read_record()
            if (
                latest is None
                or latest_record is None
                or _commit_identity(latest_record) != _commit_identity(current_record)
                or latest.get("owner_token") != current.get("owner_token")
                or float(latest.get("updated_unix", 0.0))
                != float(current.get("updated_unix", 0.0))
                or time.time() - float(latest.get("updated_unix", 0.0))
                < ANALYSIS_LEASE_STALE_SECONDS
            ):
                raise RuntimeError("analysis API lease changed during stale recovery")
            self.committer.verify_record_for_path(current_record, self.path)
            self.committer.delete_verified_if_present(current_record)
        publish_error: Exception | None = None
        try:
            publish_json(
                self.committer,
                self._payload(),
                self.path,
                replace=False,
                required_headroom_bytes=0,
            )
        except Exception as exc:
            publish_error = exc
        if ANALYSIS_LEASE_SETTLE_SECONDS > 0:
            time.sleep(ANALYSIS_LEASE_SETTLE_SECONDS)
        try:
            self._elect(require_own_claim=True)
        except Exception:
            if publish_error is not None and not self._claims():
                raise publish_error
            raise
        self.assert_owned()
        self._thread = threading.Thread(
            target=self._heartbeat,
            name=f"analysis-api-lease-{self.bundle}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            payload, record = self._read_record()
            if (
                payload is not None
                and record is not None
                and payload.get("owner_token") == self.token
            ):
                self.committer.verify_record_for_path(record, self.path)
                self.committer.delete_verified_if_present(record)
        except Exception as cleanup_exc:
            warnings.warn(
                "analysis API lease remains for stale recovery: "
                f"{cleanup_exc}",
                RuntimeWarning,
            )


def _exact_remote_record(
    committer: Any,
    path: Path,
    *,
    declared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = committer.path_commit_record(path)
    if str(current.get("remote_name")) != path.name:
        raise RuntimeError(f"Drive record has the wrong name for {path}")
    if declared is not None and _commit_identity(current) != _commit_identity(
        declared
    ):
        raise RuntimeError(f"Drive record changed or is misbound at {path}")
    if hasattr(committer, "verify_record_for_path"):
        committer.verify_record_for_path(current, path)
    else:
        committer.verify_commit_record(current)
    return current


def _remote_json_exact(
    committer: Any,
    path: Path,
    *,
    declared: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _exact_remote_record(committer, path, declared=declared)
    raw = committer.download_bytes(str(record["remote_file_id"]))
    if len(raw) != int(record["remote_bytes"]):
        raise RuntimeError(f"remote JSON has the wrong byte count: {path}")
    if hashlib.sha256(raw).hexdigest() != str(record["remote_sha256"]):
        raise RuntimeError(f"remote JSON failed SHA-256 verification: {path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"remote JSON is not an object: {path}")
    return payload, record


def _parse_json_stdout(output: str) -> Any:
    """Accept a final JSON value even if a library printed a warning first."""

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, Any]] = []
    for index, character in enumerate(output):
        if character not in "[{":
            continue
        try:
            value, consumed = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if not output[index + consumed :].strip():
            candidates.append((index, value))
    if not candidates:
        raise RuntimeError(f"child produced no final JSON value: {output[-2000:]}")
    return min(candidates, key=lambda row: row[0])[1]


def _run_wrapper_json(command: list[str]) -> Any:
    environment = os.environ.copy()
    environment["TQDM_DISABLE"] = "1"
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    if completed.returncode:
        tail = (completed.stdout + "\n" + completed.stderr).splitlines()[-80:]
        raise RuntimeError(
            f"analysis discovery child exited {completed.returncode}: "
            f"{' '.join(command)}\n" + "\n".join(tail)
        )
    return _parse_json_stdout(completed.stdout)


def _config(campaign_root: Path, bundle: str) -> dict[str, Any]:
    path = campaign_root / bundle / "production_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("bundle") != bundle:
        raise RuntimeError(f"bundle config identity changed: {path}")
    for key in (
        "sampling_revision",
        "audit_sha256",
        "production_output_collection",
        "output_bundle",
    ):
        if not payload.get(key):
            raise RuntimeError(f"bundle config lacks {key}: {path}")
    return payload


def _archive_root(drive_root: Path, config: dict[str, Any]) -> Path:
    return _lexical(drive_root) / str(config["production_output_collection"]) / str(
        config["output_bundle"]
    )


def _h1_shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _h1_current_contract(
    *,
    bundle_root: Path,
    config: dict[str, Any],
    case_row: dict[str, Any],
    shard_index: int,
) -> tuple[dict[str, Any], str]:
    """Reconstruct the frozen H1-v4 shard identity without importing its runner."""

    case = case_row.get("case")
    if not isinstance(case, dict):
        raise RuntimeError("H1 case enumeration lacks its locked case payload")
    source_hashes = {
        path.name: sha256_file(path)
        for path in sorted((bundle_root / "src").glob("*.py"))
    }
    engine_hash = source_hashes.get("classA_U1FGTN_gpu.py")
    if not isinstance(engine_hash, str):
        raise RuntimeError("H1 bundle lacks its canonical GPU engine source")
    run_config = {
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "canonical_engine_sha256": engine_hash,
        "bundle_source_hashes_sha256": _sha256_json(source_hashes),
        "shard_index": int(shard_index),
        "case": case,
        "trajectory_reuse_policy": config["trajectory_reuse"]["policy"],
    }
    run_id = f"{H1}_{_sha256_json(run_config)[:16]}"
    shard_size = int(config["locked_contract"]["sample_shard_size"])
    global_ids = list(
        range(int(shard_index) * shard_size, (int(shard_index) + 1) * shard_size)
    )
    expected = {
        "schema_version": 2,
        "status": "complete_local",
        "bundle": H1,
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "bundle_source_hashes_sha256": _sha256_json(source_hashes),
        "canonical_entry_point": config["locked_contract"]["canonical_entry_point"],
        "canonical_engine_sha256": engine_hash,
        "run_config": run_config,
        "run_config_hash": _sha256_json(run_config),
        "root_seed": int(config["root_seed"]),
        "case_id": case["case_id"],
        "protocol": case["protocol"],
        "alpha_1": case["model"]["alpha_1"],
        "shard_index": int(shard_index),
        "global_sample_indices": global_ids,
        "shard_generator_seed": _h1_shard_seed(
            int(config["root_seed"]), str(case["case_id"]), int(shard_index)
        ),
        "source_hashes": source_hashes,
    }
    return expected, run_id


def _status_manifest_is_current(
    *,
    bundle: str,
    config: dict[str, Any],
    case_row: dict[str, Any],
    shard_index: int,
    manifest: dict[str, Any],
    bundle_root: Path | None = None,
    archive: Path | None = None,
    receipt: dict[str, Any] | None = None,
) -> None:
    expected = {
        "status": "complete_local",
        "bundle": bundle,
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "case_id": case_row["case_id"],
        "shard_index": int(shard_index),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if _normalized(manifest.get(key)) != _normalized(value)
    }
    expected_case = case_row.get("case", {})
    observed_case = manifest.get("run_config", {}).get("case", {})
    if _normalized(observed_case) != _normalized(expected_case):
        mismatches["run_config.case"] = (observed_case, expected_case)
    if bundle == H1:
        if bundle_root is None or archive is None or not isinstance(receipt, dict):
            raise RuntimeError("H1 current status validation lacks its exact context")
        h1_expected, expected_run_id = _h1_current_contract(
            bundle_root=bundle_root,
            config=config,
            case_row=case_row,
            shard_index=int(shard_index),
        )
        for key, value in h1_expected.items():
            if _normalized(manifest.get(key)) != _normalized(value):
                mismatches[key] = (manifest.get(key), value)
        if archive.name != f"{expected_run_id}.tar.gz":
            mismatches["archive.name"] = (
                archive.name,
                f"{expected_run_id}.tar.gz",
            )
        if receipt.get("run_id") != expected_run_id:
            mismatches["receipt.run_id"] = (
                receipt.get("run_id"),
                expected_run_id,
            )
        numerical_status = manifest.get("numerical_status")
        if numerical_status not in {"pass", "warning"}:
            mismatches["numerical_status"] = (
                numerical_status,
                "pass or warning",
            )
        numerical = manifest.get("numerical_diagnostics")
        if not isinstance(numerical, dict):
            mismatches["numerical_diagnostics"] = (numerical, "object")
        else:
            actual_dtype = numerical.get("actual_dtype")
            if actual_dtype not in {"torch.complex128", "complex128"}:
                mismatches["numerical_diagnostics.actual_dtype"] = (
                    actual_dtype,
                    "torch.complex128",
                )
            probability_dtype = numerical.get("actual_probability_dtype")
            if probability_dtype not in {"torch.float64", "float64"}:
                mismatches["numerical_diagnostics.actual_probability_dtype"] = (
                    probability_dtype,
                    "torch.float64",
                )
        gpu = manifest.get("gpu_preflight")
        if not isinstance(gpu, dict):
            mismatches["gpu_preflight"] = (gpu, "A100 40 GB evidence")
        else:
            total_gib = gpu.get("total_gib")
            try:
                qualified_size = 35.0 <= float(total_gib) <= 45.0
            except (TypeError, ValueError):
                qualified_size = False
            if (
                "A100" not in str(gpu.get("device", "")).upper()
                or gpu.get("smoke_override") is not False
                or not qualified_size
            ):
                mismatches["gpu_preflight"] = (gpu, "production A100 40 GB")
    if mismatches:
        raise RuntimeError(
            f"remote-status returned a non-current {bundle} slot: {mismatches}"
        )


def _validate_current_status(
    *,
    committer: Any,
    bundle: str,
    config: dict[str, Any],
    case_row: dict[str, Any],
    shard_index: int,
    status: dict[str, Any],
    drive_root: Path,
    bundle_root: Path | None = None,
) -> dict[str, Any] | None:
    expected_parent = _archive_root(drive_root, config)
    archive = _lexical(str(status.get("archive", "")))
    if archive.parent != expected_parent or archive.suffixes[-2:] != [".tar", ".gz"]:
        raise RuntimeError(
            f"remote-status returned an archive outside the locked root: {archive}"
        )
    if not bool(status.get("exists")):
        return None
    receipt = status.get("receipt")
    manifest = status.get("manifest")
    if not isinstance(receipt, dict) or not isinstance(manifest, dict):
        raise RuntimeError("remote-status lacks its verified receipt or manifest")
    _status_manifest_is_current(
        bundle=bundle,
        config=config,
        case_row=case_row,
        shard_index=shard_index,
        manifest=manifest,
        bundle_root=bundle_root,
        archive=archive,
        receipt=receipt,
    )
    declared_archive = receipt.get("archive_remote_commit")
    if not isinstance(declared_archive, dict):
        raise RuntimeError("current receipt lacks archive remote metadata")
    archive_record = _exact_remote_record(
        committer, archive, declared=declared_archive
    )
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    server_receipt, receipt_record = _remote_json_exact(committer, receipt_path)
    if _normalized(server_receipt) != _normalized(receipt):
        raise RuntimeError("remote-status receipt differs from its exact Drive file")
    if (
        receipt.get("archive") != archive.name
        or receipt.get("archive_sha256") != archive_record["remote_sha256"]
        or int(receipt.get("archive_bytes", -1))
        != int(archive_record["remote_bytes"])
    ):
        raise RuntimeError("current receipt disagrees with its exact Drive archive")
    return {
        "source": "current_v4",
        "case_id": str(case_row["case_id"]),
        "shard_index": int(shard_index),
        "sampling_revision": str(config["sampling_revision"]),
        "archive_path": archive,
        "receipt_path": receipt_path,
        "archive_remote_commit": archive_record,
        "receipt_remote_commit": receipt_record,
        "receipt": receipt,
        "manifest": manifest,
    }


def enumerate_current_slots(
    *,
    campaign_root: Path,
    drive_root: Path,
    bundle: str,
    config: dict[str, Any],
    committer: Any,
    run_json: Callable[[list[str]], Any] = _run_wrapper_json,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any] | None]]:
    wrapper = campaign_root / bundle / "run_bundle.py"
    base = [
        sys.executable,
        "-u",
        str(wrapper),
        "--drive-root",
        str(_lexical(drive_root)),
        "--mode",
        "production",
    ]
    cases = run_json([*base, "--list-cases-json"])
    if not isinstance(cases, list) or len(cases) != SPECS[bundle]["case_count"]:
        raise RuntimeError(f"{bundle} wrapper returned the wrong case matrix")
    if len({str(row.get("case_id")) for row in cases}) != len(cases):
        raise RuntimeError(f"{bundle} wrapper returned duplicate case IDs")
    total = sum(int(row.get("shard_count", -1)) for row in cases)
    if total != SPECS[bundle]["slot_count"]:
        raise RuntimeError(f"{bundle} wrapper returned {total} slots, expected {SPECS[bundle]['slot_count']}")
    statuses: dict[tuple[str, int], dict[str, Any] | None] = {}
    ordinal = 0
    for case_row in cases:
        case_id = str(case_row["case_id"])
        for shard_index in range(int(case_row["shard_count"])):
            ordinal += 1
            print(
                f"[analysis remote-status {ordinal}/{total}] {case_id} shard {shard_index}",
                flush=True,
            )
            status = run_json(
                [
                    *base,
                    "--remote-status",
                    "--case-id",
                    case_id,
                    "--shard-index",
                    str(shard_index),
                ]
            )
            if not isinstance(status, dict):
                raise RuntimeError("remote-status output is not a JSON object")
            statuses[(case_id, shard_index)] = _validate_current_status(
                committer=committer,
                bundle=bundle,
                config=config,
                case_row=case_row,
                shard_index=shard_index,
                status=status,
                drive_root=drive_root,
                bundle_root=campaign_root / bundle,
            )
    return cases, statuses


def _h1_allowlist(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / "h1_v3_compatibility_ledger.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    accepted = payload.get("accepted_archives", [])
    rejected = payload.get("rejected_receipt_only_run_ids", [])
    if (
        payload.get("schema") != "h1_v3_server_compatibility_allowlist_v1"
        or len(accepted) != 12
        or len(rejected) != 8
        or sum(bool(row.get("reuse_in_v4")) for row in accepted) != 11
    ):
        raise RuntimeError("H1-v3 compatibility allowlist is not the locked 12/11/8 ledger")
    return payload


def load_h1_legacy_slots(
    *,
    campaign_root: Path,
    drive_root: Path,
    config: dict[str, Any],
    committer: Any,
) -> dict[tuple[str, int], dict[str, Any]]:
    bundle_root = campaign_root / H1
    allowlist = _h1_allowlist(bundle_root)
    v4_root = _archive_root(drive_root, config)
    ledger_path = v4_root / "migration" / "h1_v3_server_verified_ledger.json"
    ledger, ledger_record = _remote_json_exact(committer, ledger_path)
    if (
        ledger.get("schema") != "h1_v3_server_verified_compatibility_v1"
        or ledger.get("source_revision") != allowlist.get("source_revision")
        or ledger.get("source_audit_sha256") != allowlist.get("source_audit_sha256")
        or int(ledger.get("accepted_count", -1)) != 12
        or int(ledger.get("reusable_count", -1)) != 11
        or int(ledger.get("rejected_receipt_only_count", -1)) != 8
    ):
        raise RuntimeError("server H1-v3 migration ledger is incomplete or changed")
    pinned = {str(row["run_id"]): row for row in allowlist["accepted_archives"]}
    accepted = ledger.get("accepted_archives", [])
    if not isinstance(accepted, list) or len(accepted) != 12:
        raise RuntimeError("server H1-v3 accepted ledger has the wrong length")
    source_root = (
        _lexical(drive_root)
        / "classA_final_production_outputs"
        / str(allowlist["source_revision"])
        / H1
    )
    reusable: dict[tuple[str, int], dict[str, Any]] = {}
    for row in accepted:
        if not isinstance(row, dict):
            raise RuntimeError("server H1-v3 accepted ledger row is not an object")
        expected = pinned.get(str(row.get("run_id")))
        if expected is None:
            raise RuntimeError("server H1-v3 ledger contains an unknown run ID")
        for key in (
            "case_id",
            "shard_index",
            "global_sample_indices",
            "archive_bytes",
            "archive_sha256",
            "reuse_in_v4",
        ):
            if _normalized(row.get(key)) != _normalized(expected.get(key)):
                raise RuntimeError(f"server H1-v3 ledger changed {key}")
        archive = source_root / f"{H1}_{row['run_id']}.tar.gz"
        if row.get("archive") != archive.name:
            raise RuntimeError("server H1-v3 ledger changed its archive name")
        declared_archive = row.get("archive_remote_commit")
        declared_receipt = row.get("receipt_remote_commit")
        if not isinstance(declared_archive, dict) or not isinstance(
            declared_receipt, dict
        ):
            raise RuntimeError("server H1-v3 ledger lacks exact remote commit records")
        archive_record = _exact_remote_record(
            committer, archive, declared=declared_archive
        )
        if (
            int(archive_record["remote_bytes"]) != int(expected["archive_bytes"])
            or str(archive_record["remote_sha256"])
            != str(expected["archive_sha256"])
        ):
            raise RuntimeError(
                "server H1-v3 archive differs from the immutable compatibility allowlist"
            )
        receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        receipt_payload, receipt_record = _remote_json_exact(
            committer, receipt_path, declared=declared_receipt
        )
        if (
            receipt_payload.get("archive") != archive.name
            or receipt_payload.get("archive_sha256") != expected["archive_sha256"]
            or int(receipt_payload.get("archive_bytes", -1))
            != int(expected["archive_bytes"])
        ):
            raise RuntimeError(
                "server H1-v3 receipt disagrees with its immutable pinned archive"
            )
        if bool(row["reuse_in_v4"]):
            slot = (str(row["case_id"]), int(row["shard_index"]))
            if slot in reusable:
                raise RuntimeError(f"duplicate reusable H1-v3 slot: {slot}")
            reusable[slot] = {
                "source": "compatible_h1_v3_server_verified",
                "case_id": slot[0],
                "shard_index": slot[1],
                "sampling_revision": str(allowlist["source_revision"]),
                "archive_path": archive,
                "receipt_path": receipt_path,
                "archive_remote_commit": archive_record,
                "receipt_remote_commit": receipt_record,
                "allowlist_row": expected,
                "allowlist": allowlist,
                "ledger_remote_commit": ledger_record,
            }
    rejected_rows = ledger.get("rejected_receipt_only", [])
    rejected_by_id = {
        str(row.get("run_id")): row
        for row in rejected_rows
        if isinstance(row, dict)
    }
    if set(rejected_by_id) != set(allowlist["rejected_receipt_only_run_ids"]):
        raise RuntimeError("server H1-v3 rejected ledger changed its run IDs")
    for run_id, row in rejected_by_id.items():
        if row.get("reason") != "receipt_only_archive_absent_on_server":
            raise RuntimeError("server H1-v3 rejected ledger changed its reason")
        archive = source_root / f"{H1}_{run_id}.tar.gz"
        try:
            committer.path_commit_record(archive)
        except RemoteCommitError as exc:
            if not _is_absent(exc):
                raise
        else:
            raise RuntimeError(f"receipt-only H1-v3 archive appeared on Drive: {archive}")
        receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        declared_receipt = row.get("receipt_remote_commit")
        if not isinstance(declared_receipt, dict):
            raise RuntimeError(
                "server H1-v3 rejected ledger lacks its receipt remote commit"
            )
        _exact_remote_record(
            committer, receipt_path, declared=declared_receipt
        )
    if len(reusable) != 11:
        raise RuntimeError(f"expected 11 reusable H1-v3 slots, found {len(reusable)}")
    _exact_remote_record(committer, ledger_path, declared=ledger_record)
    return reusable


def build_input_plan(
    *,
    campaign_root: Path,
    drive_root: Path,
    bundle: str,
    config: dict[str, Any],
    committer: Any,
    run_json: Callable[[list[str]], Any] = _run_wrapper_json,
) -> list[dict[str, Any]]:
    cases, current = enumerate_current_slots(
        campaign_root=campaign_root,
        drive_root=drive_root,
        bundle=bundle,
        config=config,
        committer=committer,
        run_json=run_json,
    )
    legacy = (
        load_h1_legacy_slots(
            campaign_root=campaign_root,
            drive_root=drive_root,
            config=config,
            committer=committer,
        )
        if bundle == H1
        else {}
    )
    plan: list[dict[str, Any]] = []
    for case_row in cases:
        case_id = str(case_row["case_id"])
        for shard_index in range(int(case_row["shard_count"])):
            slot = (case_id, shard_index)
            current_row = current[slot]
            legacy_row = legacy.get(slot)
            if legacy_row is not None:
                if current_row is not None:
                    raise RuntimeError(
                        f"H1 slot {slot} has both a pinned legacy input and current v4 output"
                    )
                plan.append(legacy_row)
            elif current_row is None:
                raise RuntimeError(f"analysis input is not server complete: {bundle}/{slot}")
            else:
                plan.append(current_row)
    expected_current = 9 if bundle == H1 else SPECS[bundle]["slot_count"]
    actual_current = sum(row["source"] == "current_v4" for row in plan)
    if len(plan) != SPECS[bundle]["slot_count"] or actual_current != expected_current:
        raise RuntimeError(
            f"{bundle} input mix is wrong: total={len(plan)}, current={actual_current}"
        )
    return plan


def _root_manifest(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        matches = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lstrip("./") == "manifest.json"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{path}: expected exactly one root manifest")
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise RuntimeError(f"{path}: root manifest is unreadable")
        payload = json.loads(handle.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: root manifest is not an object")
    return payload


def _validate_legacy_manifest(manifest: dict[str, Any], row: dict[str, Any]) -> None:
    allowlist = row["allowlist"]
    pinned = row["allowlist_row"]
    expected = {
        "bundle": H1,
        "sampling_revision": allowlist["source_revision"],
        "audit_sha256": allowlist["source_audit_sha256"],
        "root_seed": int(allowlist["root_seed"]),
        "case_id": pinned["case_id"],
        "shard_index": int(pinned["shard_index"]),
        "global_sample_indices": pinned["global_sample_indices"],
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if _normalized(manifest.get(key)) != _normalized(value)
    }
    if mismatches:
        raise RuntimeError(f"downloaded H1-v3 manifest changed: {mismatches}")
    if manifest.get("status") != "complete_local" or manifest.get(
        "numerical_status"
    ) not in {"pass", "warning"}:
        raise RuntimeError("downloaded H1-v3 archive is incomplete or numerically unsafe")
    source_hashes = manifest.get("source_hashes", {})
    for name, expected_hash in allowlist["scientific_source_sha256"].items():
        if source_hashes.get(name) != expected_hash:
            raise RuntimeError(f"downloaded H1-v3 scientific source changed: {name}")
    case = manifest.get("run_config", {}).get("case", {})
    if case.get("case_id") != pinned["case_id"]:
        raise RuntimeError("downloaded H1-v3 run config names another case")
    if case.get("model", {}).get("dtype") != "complex128":
        raise RuntimeError("downloaded H1-v3 archive is not complex128")
    actual_dtype = manifest.get("numerical_diagnostics", {}).get("actual_dtype")
    if actual_dtype not in (None, "torch.complex128", "complex128"):
        raise RuntimeError("downloaded H1-v3 archive reports a wrong actual dtype")


def _download_record(
    *,
    committer: Any,
    remote_path: Path,
    record: dict[str, Any],
    destination: Path,
    cache_candidates: list[Path] | None = None,
) -> Path:
    record = _exact_remote_record(committer, remote_path, declared=record)
    expected_size = int(record["remote_bytes"])
    expected_sha = str(record["remote_sha256"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == expected_size and sha256_file(destination) == expected_sha:
            _exact_remote_record(committer, remote_path, declared=record)
            return destination
        destination.unlink()
    for cache in cache_candidates or []:
        try:
            valid = (
                cache.is_file()
                and cache.stat().st_size == expected_size
                and sha256_file(cache) == expected_sha
            )
        except OSError:
            valid = False
        if not valid:
            continue
        try:
            os.link(cache, destination)
        except OSError:
            shutil.copy2(cache, destination)
        _exact_remote_record(committer, remote_path, declared=record)
        return destination
    try:
        committer.download_to(
            str(record["remote_file_id"]),
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha,
        )
    except TypeError:
        committer.download_to(str(record["remote_file_id"]), destination)
    if destination.stat().st_size != expected_size or sha256_file(destination) != expected_sha:
        raise RuntimeError(f"downloaded Drive file failed verification: {remote_path}")
    _exact_remote_record(committer, remote_path, declared=record)
    return destination


def materialize_inputs(
    *,
    bundle: str,
    plan: list[dict[str, Any]],
    local_archive_root: Path,
    committer: Any,
) -> list[dict[str, Any]]:
    local_archive_root.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    materialized: list[dict[str, Any]] = []
    for ordinal, row in enumerate(plan, start=1):
        archive_name = str(row["archive_remote_commit"]["remote_name"])
        if archive_name in names:
            raise RuntimeError(f"two analysis inputs share archive name {archive_name!r}")
        names.add(archive_name)
        archive = local_archive_root / archive_name
        receipt = archive.with_suffix(archive.suffix + ".receipt.json")
        cache_candidates = [
            Path("/content/classA_remote_cache")
            / str(SPECS[bundle]["cache_directory"])
            / archive_name
        ]
        if row["source"] != "current_v4":
            cache_candidates.append(
                Path("/content/classA_remote_cache/h1_v3_analysis") / archive_name
            )
        print(
            f"[analysis download {ordinal}/{len(plan)}] {row['case_id']} "
            f"shard {row['shard_index']} ({row['source']})",
            flush=True,
        )
        _download_record(
            committer=committer,
            remote_path=row["archive_path"],
            record=row["archive_remote_commit"],
            destination=archive,
            cache_candidates=cache_candidates,
        )
        _download_record(
            committer=committer,
            remote_path=row["receipt_path"],
            record=row["receipt_remote_commit"],
            destination=receipt,
        )
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            receipt_payload.get("archive") != archive.name
            or receipt_payload.get("archive_sha256")
            != row["archive_remote_commit"]["remote_sha256"]
            or int(receipt_payload.get("archive_bytes", -1))
            != int(row["archive_remote_commit"]["remote_bytes"])
        ):
            raise RuntimeError(f"downloaded receipt disagrees with {archive.name}")
        manifest = _root_manifest(archive)
        if row["source"] == "current_v4":
            if _normalized(manifest) != _normalized(row["manifest"]):
                raise RuntimeError("downloaded current archive manifest changed after status")
            if _normalized(receipt_payload) != _normalized(row["receipt"]):
                raise RuntimeError("downloaded current receipt changed after status")
        else:
            _validate_legacy_manifest(manifest, row)
        materialized.append(
            {
                **row,
                "local_archive": str(archive),
                "local_receipt": str(receipt),
            }
        )
    return materialized


def _analysis_input_identity(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": row["source"],
            "case_id": row["case_id"],
            "shard_index": int(row["shard_index"]),
            "sampling_revision": row["sampling_revision"],
            "archive_remote_commit": _commit_identity(row["archive_remote_commit"]),
            "receipt_remote_commit": _commit_identity(row["receipt_remote_commit"]),
        }
        for row in plan
    ]


def _scientific_source_identity(bundle_root: Path) -> dict[str, Any]:
    rows = {
        path.name: sha256_file(path)
        for path in sorted((bundle_root / "src").glob("*.py"))
    }
    return {"files": rows, "aggregate_sha256": _sha256_json(rows)}


def run_local_analysis(
    *,
    bundle: str,
    bundle_root: Path,
    archive_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    spec = SPECS[bundle]
    analysis_script = bundle_root / "src" / str(spec["analysis_script"])
    command = [
        sys.executable,
        "-u",
        str(analysis_script),
        "--archive-root",
        str(archive_root),
        "--output-root",
        str(output_root),
    ]
    if bundle == H1:
        command.extend(["--bundle-root", str(bundle_root)])
        # Deliberately omit --drive-root: all eleven compatible v3 inputs have
        # already been server-verified and materialized locally.
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = "0"
    environment["PYTHONHASHSEED"] = "0"
    environment["MPLCONFIGDIR"] = str(output_root.parent / ".matplotlib")
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    if completed.returncode:
        tail = (completed.stdout + "\n" + completed.stderr).splitlines()[-100:]
        raise RuntimeError("local immutable analysis failed:\n" + "\n".join(tail))
    summary_path = output_root / str(spec["summary"])
    if not summary_path.is_file():
        raise RuntimeError(f"immutable analysis did not create {summary_path.name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in spec["summary_statuses"]:
        raise RuntimeError(f"immutable analysis returned invalid status {summary.get('status')!r}")
    source = {
        "path": str(analysis_script.relative_to(bundle_root)),
        "sha256": sha256_file(analysis_script),
        "bundle_source_identity": _scientific_source_identity(bundle_root),
    }
    return summary, source


def publish_analysis_generation(
    *,
    bundle: str,
    config: dict[str, Any],
    drive_root: Path,
    plan: list[dict[str, Any]],
    output_root: Path,
    summary: dict[str, Any],
    analysis_source: dict[str, Any],
    committer: Any,
    lease: _AnalysisApiLease | None = None,
) -> dict[str, Any]:
    if lease is not None:
        lease.assert_owned()
    local_outputs = sorted(path for path in output_root.rglob("*") if path.is_file())
    if not local_outputs:
        raise RuntimeError("analysis created no output files")
    output_identity = [
        {
            "relative_path": str(path.relative_to(output_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in local_outputs
    ]
    input_identity = _analysis_input_identity(plan)
    input_set_sha = _sha256_json(input_identity)
    output_set_sha = _sha256_json(output_identity)
    generation_id = _sha256_json(
        {
            "bundle": bundle,
            "sampling_revision": config["sampling_revision"],
            "audit_sha256": config["audit_sha256"],
            "analysis_source": analysis_source,
            "input_set_sha256": input_set_sha,
            "output_set_sha256": output_set_sha,
        }
    )[:24]
    analysis_root = _archive_root(drive_root, config) / str(
        SPECS[bundle]["analysis_directory"]
    )
    generation_root = analysis_root / "generations" / generation_id
    committed_outputs: list[dict[str, Any]] = []
    missing_output_bytes = 0
    for row in output_identity:
        remote = generation_root / row["relative_path"]
        try:
            existing = committer.path_commit_record(remote)
        except RemoteCommitError as exc:
            if not _is_absent(exc):
                raise
            missing_output_bytes += int(row["bytes"])
            continue
        _exact_remote_record(committer, remote, declared=existing)
        if (
            int(existing["remote_bytes"]) != int(row["bytes"])
            or existing["remote_sha256"] != row["sha256"]
        ):
            raise RuntimeError(
                f"existing immutable analysis generation differs at {remote}"
            )
    if missing_output_bytes:
        if lease is not None:
            lease.assert_owned()
        committer.require_quota(
            upload_bytes=missing_output_bytes,
            required_headroom_bytes=0,
        )
        if lease is not None:
            lease.assert_owned()
    for row, local in zip(output_identity, local_outputs):
        remote = generation_root / row["relative_path"]
        if lease is not None:
            lease.assert_owned()
        record = committer.upload_verified(
            local,
            remote,
            replace=False,
            required_headroom_bytes=0,
        )
        if lease is not None:
            lease.assert_owned()
        exact = _exact_remote_record(committer, remote, declared=record)
        if (
            int(exact["remote_bytes"]) != int(row["bytes"])
            or exact["remote_sha256"] != row["sha256"]
        ):
            raise RuntimeError(f"published analysis output changed: {remote}")
        committed_outputs.append(
            {
                **row,
                "remote_path": str(remote),
                "remote_commit": exact,
            }
        )
    receipt = {
        "schema": ANALYSIS_RECEIPT_SCHEMA,
        "status": "server_verified",
        "bundle": bundle,
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "generation_id": generation_id,
        "analysis_status": summary["status"],
        "analysis_source": analysis_source,
        "input_count": len(input_identity),
        "input_set_sha256": input_set_sha,
        "inputs": input_identity,
        "output_count": len(committed_outputs),
        "output_set_sha256": output_set_sha,
        "outputs": committed_outputs,
        "created_unix": time.time(),
    }
    receipt_path = analysis_root / "server_verified_analysis_receipt.json"
    if lease is not None:
        lease.assert_owned()
    receipt_commit = publish_json(
        committer,
        receipt,
        receipt_path,
        replace=True,
        required_headroom_bytes=0,
    )
    if lease is not None:
        lease.assert_owned()
    server_receipt, exact_receipt = _remote_json_exact(
        committer, receipt_path, declared=receipt_commit
    )
    if _normalized(server_receipt) != _normalized(receipt):
        raise RuntimeError("published analysis receipt failed Drive readback")
    for row in committed_outputs:
        _exact_remote_record(
            committer,
            Path(row["remote_path"]),
            declared=row["remote_commit"],
        )
    if lease is not None:
        lease.assert_owned()
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_remote_commit": exact_receipt,
    }


def _local_runtime_root() -> Path:
    content = Path("/content")
    return content if content.is_dir() else Path(tempfile.gettempdir())


def run_server_verified_analysis(
    *,
    campaign_root: Path,
    drive_root: Path,
    bundle: str,
    local_base: Path | None = None,
    keep_local: bool = False,
    committer: Any | None = None,
    run_json: Callable[[list[str]], Any] = _run_wrapper_json,
) -> dict[str, Any]:
    if bundle not in SPECS:
        raise KeyError(f"unsupported server-verified analysis bundle {bundle!r}")
    campaign_root = _lexical(campaign_root)
    drive_root = _lexical(drive_root)
    bundle_root = campaign_root / bundle
    config = _config(campaign_root, bundle)
    committer = committer or DriveRemoteCommitter(drive_root=drive_root)
    analysis_root = _archive_root(drive_root, config) / str(
        SPECS[bundle]["analysis_directory"]
    )
    lease = _AnalysisApiLease(
        bundle=bundle,
        config=config,
        path=(
            analysis_root
            / "_operational_leases"
            / "server_verified_analysis.api_lease.json"
        ),
        committer=committer,
    )
    with lease:
        plan = build_input_plan(
            campaign_root=campaign_root,
            drive_root=drive_root,
            bundle=bundle,
            config=config,
            committer=committer,
            run_json=run_json,
        )
        lease.assert_owned()
        plan_identity = _analysis_input_identity(plan)
        session_id = _sha256_json(
            {
                "bundle": bundle,
                "revision": config["sampling_revision"],
                "inputs": plan_identity,
            }
        )[:24]
        base = local_base or (_local_runtime_root() / "classA_server_verified_analysis")
        session = _lexical(base) / bundle / session_id
        archives = session / "archives"
        outputs = session / "outputs"
        try:
            materialize_inputs(
                bundle=bundle,
                plan=plan,
                local_archive_root=archives,
                committer=committer,
            )
            lease.assert_owned()
            summary, analysis_source = run_local_analysis(
                bundle=bundle,
                bundle_root=bundle_root,
                archive_root=archives,
                output_root=outputs,
            )
            lease.assert_owned()
            result = publish_analysis_generation(
                bundle=bundle,
                config=config,
                drive_root=drive_root,
                plan=plan,
                output_root=outputs,
                summary=summary,
                analysis_source=analysis_source,
                committer=committer,
                lease=lease,
            )
            lease.assert_owned()
        except Exception:
            print(f"[analysis local recovery preserved] {session}", file=sys.stderr)
            raise
    result["summary"] = summary
    result["local_session"] = str(session)
    if not keep_local:
        shutil.rmtree(session)
        result["local_session_cleaned"] = True
    else:
        result["local_session_cleaned"] = False
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, default=ROOT)
    parser.add_argument("--bundle", choices=tuple(SPECS), required=True)
    parser.add_argument("--local-base", type=Path)
    parser.add_argument("--keep-local", action="store_true")
    args = parser.parse_args(argv)
    result = run_server_verified_analysis(
        campaign_root=args.campaign_root,
        drive_root=args.drive_root,
        bundle=args.bundle,
        local_base=args.local_base,
        keep_local=args.keep_local,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
