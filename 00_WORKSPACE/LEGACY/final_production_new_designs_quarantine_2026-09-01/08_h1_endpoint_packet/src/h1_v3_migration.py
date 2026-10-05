"""Build and verify the pinned H1-v3 to H1-v4 compatibility ledger."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from drive_remote_commit import (
    DriveRemoteCommitter,
    RemoteCommitError,
    publish_json,
    read_remote_json,
    sha256_file,
)


BUNDLE = "08_h1_endpoint_packet"
V4_REVISION = "production_25sample_h1_endpoint_packet_v4"
V3_LEDGER_SCHEMA = "h1_v3_server_compatibility_allowlist_v1"
VERIFIED_LEDGER_SCHEMA = "h1_v3_server_verified_compatibility_v1"


def _root_manifest(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        rows = [
            member
            for member in archive.getmembers()
            if member.name.lstrip("./") == "manifest.json"
        ]
        if len(rows) != 1:
            raise RuntimeError(f"{path}: expected one root manifest, found {len(rows)}")
        handle = archive.extractfile(rows[0])
        if handle is None:
            raise RuntimeError(f"{path}: root manifest is unreadable")
        return json.loads(handle.read().decode("utf-8"))


def load_allowlist(bundle_root: Path | str) -> dict[str, Any]:
    path = Path(bundle_root) / "h1_v3_compatibility_ledger.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != V3_LEDGER_SCHEMA:
        raise RuntimeError("H1-v3 compatibility allowlist has the wrong schema")
    accepted = payload.get("accepted_archives", [])
    rejected = payload.get("rejected_receipt_only_run_ids", [])
    if len(accepted) != 12 or len(rejected) != 8:
        raise RuntimeError("H1-v3 allowlist must pin exactly 12 archives and 8 rejections")
    if sum(bool(row.get("reuse_in_v4")) for row in accepted) != 11:
        raise RuntimeError("H1-v4 must reuse exactly eleven H1-v3 shards")
    slots = {(row["case_id"], int(row["shard_index"])) for row in accepted}
    if len(slots) != 12:
        raise RuntimeError("H1-v3 allowlist contains duplicate case/shard slots")
    return payload


def _validate_manifest(
    manifest: dict[str, Any], row: dict[str, Any], allowlist: dict[str, Any]
) -> None:
    expected = {
        "bundle": BUNDLE,
        "sampling_revision": allowlist["source_revision"],
        "audit_sha256": allowlist["source_audit_sha256"],
        "root_seed": int(allowlist["root_seed"]),
        "case_id": row["case_id"],
        "shard_index": int(row["shard_index"]),
        "global_sample_indices": row["global_sample_indices"],
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"H1-v3 archive identity mismatch: {mismatches}")
    if manifest.get("status") != "complete_local":
        raise RuntimeError("H1-v3 migration rejects incomplete or failed archives")
    if manifest.get("numerical_status") not in {"pass", "warning"}:
        raise RuntimeError("H1-v3 migration rejects unsafe numerical status")
    source_hashes = manifest.get("source_hashes", {})
    for name, expected_hash in allowlist["scientific_source_sha256"].items():
        if source_hashes.get(name) != expected_hash:
            raise RuntimeError(f"H1-v3 scientific source changed: {name}")
    run_config = manifest.get("run_config", {})
    case = run_config.get("case", {})
    if case.get("case_id") != row["case_id"]:
        raise RuntimeError("H1-v3 run config names a different case")
    if case.get("model", {}).get("dtype") != "complex128":
        raise RuntimeError("H1-v3 migration requires the locked complex128 engine")
    numerical = manifest.get("numerical_diagnostics", {})
    actual_dtype = numerical.get("actual_dtype")
    if actual_dtype not in (None, "torch.complex128", "complex128"):
        raise RuntimeError(f"H1-v3 archive reports the wrong actual dtype: {actual_dtype}")


def build_verified_ledger(
    *,
    drive_root: Path,
    bundle_root: Path,
    committer: DriveRemoteCommitter | None = None,
    inspect_archives: bool = True,
) -> dict[str, Any]:
    allowlist = load_allowlist(bundle_root)
    committer = committer or DriveRemoteCommitter(drive_root=drive_root)
    source_root = (
        drive_root
        / "classA_final_production_outputs"
        / "production_25sample_h1_endpoint_packet_v3"
        / BUNDLE
    )
    accepted: list[dict[str, Any]] = []
    for row in allowlist["accepted_archives"]:
        archive = source_root / f"{BUNDLE}_{row['run_id']}.tar.gz"
        metadata = committer.verify_path(
            archive,
            expected_size=int(row["archive_bytes"]),
            expected_sha256=str(row["archive_sha256"]),
        )
        receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        receipt = read_remote_json(committer, receipt_path)
        receipt_record = committer.path_commit_record(receipt_path)
        if (
            receipt.get("archive") != archive.name
            or receipt.get("archive_sha256") != row["archive_sha256"]
            or int(receipt.get("archive_bytes", -1)) != int(row["archive_bytes"])
        ):
            raise RuntimeError(f"H1-v3 receipt disagrees with pinned archive: {archive}")
        if inspect_archives:
            with tempfile.TemporaryDirectory(prefix="h1_v3_migration_") as raw:
                local = Path(raw) / archive.name
                committer.download_to(str(metadata["id"]), local)
                if sha256_file(local) != row["archive_sha256"]:
                    raise RuntimeError(f"downloaded H1-v3 archive changed: {archive}")
                _validate_manifest(_root_manifest(local), row, allowlist)
        accepted.append(
            {
                **row,
                "archive": archive.name,
                "archive_remote_commit": committer.commit_record(metadata),
                "receipt_remote_commit": receipt_record,
            }
        )

    rejected: list[dict[str, Any]] = []
    for run_id in allowlist["rejected_receipt_only_run_ids"]:
        archive = source_root / f"{BUNDLE}_{run_id}.tar.gz"
        receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        try:
            committer.path_commit_record(archive)
        except RemoteCommitError as exc:
            if "absent" not in str(exc):
                raise
        else:
            raise RuntimeError(
                f"allowlisted receipt-only H1-v3 entry now has an archive: {archive}"
            )
        receipt_record = committer.path_commit_record(receipt_path)
        rejected.append(
            {
                "run_id": run_id,
                "reason": "receipt_only_archive_absent_on_server",
                "receipt_remote_commit": receipt_record,
            }
        )

    return {
        "schema": VERIFIED_LEDGER_SCHEMA,
        "target_revision": V4_REVISION,
        "source_revision": allowlist["source_revision"],
        "source_audit_sha256": allowlist["source_audit_sha256"],
        "accepted_count": len(accepted),
        "reusable_count": sum(bool(row["reuse_in_v4"]) for row in accepted),
        "rejected_receipt_only_count": len(rejected),
        "accepted_archives": accepted,
        "rejected_receipt_only": rejected,
        "created_unix": time.time(),
    }


def verified_ledger_path(drive_root: Path) -> Path:
    return (
        drive_root
        / "classA_final_production_outputs"
        / "production_25sample_h1_endpoint_packet_v4"
        / BUNDLE
        / "migration"
        / "h1_v3_server_verified_ledger.json"
    )


def publish_verified_ledger(
    *, drive_root: Path, bundle_root: Path, inspect_archives: bool = True
) -> dict[str, Any]:
    committer = DriveRemoteCommitter(drive_root=drive_root)
    ledger = build_verified_ledger(
        drive_root=drive_root,
        bundle_root=bundle_root,
        committer=committer,
        inspect_archives=inspect_archives,
    )
    destination = verified_ledger_path(drive_root)
    remote = publish_json(
        committer,
        ledger,
        destination,
        replace=True,
        required_headroom_bytes=1_073_741_824,
    )
    return {**ledger, "ledger_path": str(destination), "ledger_remote_commit": remote}


def load_current_verified_ledger(
    *, drive_root: Path, bundle_root: Path
) -> dict[str, Any]:
    allowlist = load_allowlist(bundle_root)
    committer = DriveRemoteCommitter(drive_root=drive_root)
    path = verified_ledger_path(drive_root)
    payload = read_remote_json(committer, path)
    committer.path_commit_record(path)
    if (
        payload.get("schema") != VERIFIED_LEDGER_SCHEMA
        or payload.get("accepted_count") != 12
        or payload.get("reusable_count") != 11
        or payload.get("rejected_receipt_only_count") != 8
    ):
        raise RuntimeError("server H1-v3 migration ledger is incomplete")
    pinned = {row["run_id"]: row for row in allowlist["accepted_archives"]}
    for row in payload.get("accepted_archives", []):
        expected = pinned.get(row.get("run_id"))
        if expected is None:
            raise RuntimeError("server H1-v3 migration ledger contains an unknown archive")
        for key in (
            "case_id", "shard_index", "global_sample_indices", "archive_bytes",
            "archive_sha256", "reuse_in_v4",
        ):
            if row.get(key) != expected.get(key):
                raise RuntimeError(f"server H1-v3 migration ledger changed {key}")
        committer.verify_commit_record(row["archive_remote_commit"])
        committer.verify_commit_record(row["receipt_remote_commit"])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument(
        "--bundle-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--skip-archive-inspection", action="store_true")
    parser.add_argument("--reuse-current", action="store_true")
    args = parser.parse_args(argv)
    if args.reuse_current:
        try:
            result = load_current_verified_ledger(
                drive_root=args.drive_root.resolve(),
                bundle_root=args.bundle_root.resolve(),
            )
        except (RemoteCommitError, RuntimeError, KeyError, ValueError):
            result = publish_verified_ledger(
                drive_root=args.drive_root.resolve(),
                bundle_root=args.bundle_root.resolve(),
                inspect_archives=not args.skip_archive_inspection,
            )
    else:
        result = publish_verified_ledger(
            drive_root=args.drive_root.resolve(),
            bundle_root=args.bundle_root.resolve(),
            inspect_archives=not args.skip_archive_inspection,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
