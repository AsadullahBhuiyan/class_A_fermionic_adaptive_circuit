from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "00_WORKSPACE" / "CURRENT" / "final_production_new_designs"
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))

import server_verified_analysis as analysis  # noqa: E402


class FakeRemoteError(analysis.RemoteCommitError):
    pass


class FakeCommitter:
    def __init__(self) -> None:
        self.by_path: dict[str, dict[str, Any]] = {}
        self.by_id: dict[str, dict[str, Any]] = {}
        self.raw_by_id: dict[str, bytes] = {}
        self.next_id = 1
        self.quota_calls: list[tuple[int, int]] = []
        self.upload_headrooms: list[int] = []
        self.block_quota = False

    def _record(self, path: Path, raw: bytes, *, file_id: str) -> dict[str, Any]:
        return {
            "schema": "classA_drive_api_commit_v1",
            "remote_file_id": file_id,
            "remote_name": path.name,
            "remote_parent_id": f"parent:{path.parent}",
            "remote_bytes": len(raw),
            "remote_sha256": hashlib.sha256(raw).hexdigest(),
            "remote_verified_unix": 1.0,
        }

    def add(self, path: Path | str, raw: bytes) -> dict[str, Any]:
        path = Path(path)
        file_id = f"file-{self.next_id}"
        self.next_id += 1
        record = self._record(path, raw, file_id=file_id)
        self.by_path[str(path)] = record
        self.by_id[file_id] = record
        self.raw_by_id[file_id] = bytes(raw)
        return dict(record)

    def path_commit_record(self, path: Path | str) -> dict[str, Any]:
        try:
            return dict(self.by_path[str(Path(path))])
        except KeyError as exc:
            raise FakeRemoteError(f"Drive file is absent: {path}") from exc

    def split_remote_path(self, path: Path | str) -> tuple[tuple[str, ...], str]:
        path = Path(path)
        return (str(path.parent),), path.name

    def resolve_folder(self, parts: tuple[str, ...], *, create: bool) -> str:
        assert len(parts) == 1
        return f"parent:{parts[0]}"

    def _list_children(self, parent_id: str, name: str) -> list[dict[str, Any]]:
        return [
            self.metadata(file_id)
            for file_id, row in self.by_id.items()
            if row["remote_parent_id"] == parent_id and row["remote_name"] == name
        ]

    def metadata(self, file_id: str) -> dict[str, Any]:
        try:
            row = self.by_id[file_id]
        except KeyError as exc:
            raise FakeRemoteError("Drive file is absent: id") from exc
        return {
            "id": row["remote_file_id"],
            "name": row["remote_name"],
            "parents": [row["remote_parent_id"]],
            "size": row["remote_bytes"],
            "sha256Checksum": row["remote_sha256"],
        }

    @staticmethod
    def commit_record(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": "classA_drive_api_commit_v1",
            "remote_file_id": metadata["id"],
            "remote_name": metadata["name"],
            "remote_parent_id": metadata["parents"][0],
            "remote_bytes": metadata["size"],
            "remote_sha256": metadata["sha256Checksum"],
            "remote_verified_unix": 1.0,
        }

    def verify_commit_record(self, record: dict[str, Any]) -> dict[str, Any]:
        current = self.by_id[str(record["remote_file_id"])]
        if analysis._commit_identity(current) != analysis._commit_identity(record):
            raise FakeRemoteError("remote commit mismatch")
        return dict(current)

    def verify_record_for_path(
        self, record: dict[str, Any], path: Path | str
    ) -> dict[str, Any]:
        current = self.path_commit_record(path)
        if analysis._commit_identity(current) != analysis._commit_identity(record):
            raise FakeRemoteError("remote path binding mismatch")
        return self.verify_commit_record(record)

    def download_bytes(self, file_id: str) -> bytes:
        return self.raw_by_id[file_id]

    def download_to(
        self,
        file_id: str,
        destination: Path | str,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        raw = self.download_bytes(file_id)
        if expected_size is not None:
            assert len(raw) == expected_size
        if expected_sha256 is not None:
            assert hashlib.sha256(raw).hexdigest() == expected_sha256
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return destination

    def require_quota(
        self, *, upload_bytes: int, required_headroom_bytes: int
    ) -> None:
        if self.block_quota:
            raise FakeRemoteError("quota exhausted")
        self.quota_calls.append((int(upload_bytes), int(required_headroom_bytes)))

    def upload_verified(
        self,
        local_path: Path | str,
        remote_path: Path | str,
        *,
        replace: bool,
        required_headroom_bytes: int = 0,
    ) -> dict[str, Any]:
        local_path, remote_path = Path(local_path), Path(remote_path)
        raw = local_path.read_bytes()
        self.upload_headrooms.append(int(required_headroom_bytes))
        key = str(remote_path)
        if key not in self.by_path:
            return self.add(remote_path, raw)
        current = self.by_path[key]
        if not replace and self.raw_by_id[current["remote_file_id"]] != raw:
            raise FakeRemoteError("immutable generation file changed")
        file_id = current["remote_file_id"]
        record = self._record(remote_path, raw, file_id=file_id)
        self.by_path[key] = record
        self.by_id[file_id] = record
        self.raw_by_id[file_id] = raw
        return dict(record)

    def delete_verified_if_present(self, record: dict[str, Any]) -> bool:
        file_id = str(record["remote_file_id"])
        current = self.by_id.pop(file_id, None)
        if current is None:
            return False
        self.raw_by_id.pop(file_id, None)
        for path, row in list(self.by_path.items()):
            if row["remote_file_id"] == file_id:
                del self.by_path[path]
        return True


class AmbiguousNameCommitter(FakeCommitter):
    def path_commit_record(self, path: Path | str) -> dict[str, Any]:
        path = Path(path)
        parent_id = f"parent:{path.parent}"
        matches = [
            row
            for row in self.by_id.values()
            if row["remote_parent_id"] == parent_id
            and row["remote_name"] == path.name
        ]
        if len(matches) > 1:
            raise FakeRemoteError("ambiguous Drive path")
        if not matches:
            raise FakeRemoteError(f"Drive file is absent: {path}")
        return dict(matches[0])


def tar_with_manifest(manifest: dict[str, Any]) -> bytes:
    raw = json.dumps(manifest, sort_keys=True).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        member = tarfile.TarInfo("manifest.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    return output.getvalue()


def current_plan_row(
    *, committer: FakeCommitter, drive_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    archive = drive_root / "outputs" / "bundle" / "run.tar.gz"
    archive_record = committer.add(archive, tar_with_manifest(manifest))
    receipt = {
        "schema_version": 2,
        "run_id": "run",
        "archive": archive.name,
        "archive_sha256": archive_record["remote_sha256"],
        "archive_bytes": archive_record["remote_bytes"],
        "archive_remote_commit": archive_record,
    }
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    receipt_record = committer.add(
        receipt_path,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {
        "source": "current_v4",
        "case_id": "case",
        "shard_index": 0,
        "sampling_revision": "revision-v4",
        "archive_path": archive,
        "receipt_path": receipt_path,
        "archive_remote_commit": archive_record,
        "receipt_remote_commit": receipt_record,
        "receipt": receipt,
        "manifest": manifest,
    }


def h1_current_status_fixture(
    *, tmp_path: Path, committer: FakeCommitter
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, dict[str, Any]]:
    bundle_root = CAMPAIGN / analysis.H1
    config = json.loads((bundle_root / "production_config.json").read_text())
    cases = analysis._run_wrapper_json(
        [
            sys.executable,
            "-u",
            str(bundle_root / "run_bundle.py"),
            "--drive-root",
            str(tmp_path / "logical-drive"),
            "--mode",
            "production",
            "--list-cases-json",
        ]
    )
    case_row = next(
        row
        for row in cases
        if row["case"]["protocol"] == "soft"
        and row["case"]["model"]["alpha_1"] == 1.0
    )
    expected, run_id = analysis._h1_current_contract(
        bundle_root=bundle_root,
        config=config,
        case_row=case_row,
        shard_index=0,
    )
    assert run_id == "08_h1_endpoint_packet_d417be5f10219c5c"
    manifest = {
        **expected,
        "numerical_status": "pass",
        "numerical_diagnostics": {
            "actual_dtype": "torch.complex128",
            "actual_probability_dtype": "torch.float64",
        },
        "gpu_preflight": {
            "device": "NVIDIA A100-SXM4-40GB",
            "total_gib": 39.6,
            "smoke_override": False,
        },
    }
    drive_root = tmp_path / "logical-drive"
    archive = analysis._archive_root(drive_root, config) / f"{run_id}.tar.gz"
    archive_record = committer.add(archive, tar_with_manifest(manifest))
    receipt = {
        "schema_version": 2,
        "run_id": run_id,
        "archive": archive.name,
        "archive_sha256": archive_record["remote_sha256"],
        "archive_bytes": archive_record["remote_bytes"],
        "archive_remote_commit": archive_record,
    }
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    committer.add(
        receipt_path,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    status = {
        "exists": True,
        "archive": str(archive),
        "receipt": receipt,
        "manifest": manifest,
    }
    return config, case_row, bundle_root, drive_root, status


def h1_legacy_fixture(
    *, tmp_path: Path, committer: FakeCommitter, monkeypatch: Any
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, list[dict[str, Any]]]:
    drive_root = tmp_path / "logical-drive"
    config = {
        "production_output_collection": "v4-outputs",
        "output_bundle": analysis.H1,
    }
    allowlist: dict[str, Any] = {
        "schema": "h1_v3_server_compatibility_allowlist_v1",
        "source_revision": "locked-v3",
        "source_audit_sha256": "locked-audit",
        "accepted_archives": [],
        "rejected_receipt_only_run_ids": [f"rejected-{index}" for index in range(8)],
    }
    source_root = (
        drive_root
        / "classA_final_production_outputs"
        / allowlist["source_revision"]
        / analysis.H1
    )
    accepted: list[dict[str, Any]] = []
    for index in range(12):
        run_id = f"{index:016x}"
        archive = source_root / f"{analysis.H1}_{run_id}.tar.gz"
        raw = f"locked archive {index}".encode()
        archive_record = committer.add(archive, raw)
        pinned = {
            "run_id": run_id,
            "case_id": f"case-{index}",
            "shard_index": index,
            "global_sample_indices": [index],
            "archive_bytes": len(raw),
            "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "reuse_in_v4": index != 0,
        }
        allowlist["accepted_archives"].append(pinned)
        receipt = {
            "archive": archive.name,
            "archive_bytes": pinned["archive_bytes"],
            "archive_sha256": pinned["archive_sha256"],
        }
        receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
        receipt_record = committer.add(
            receipt_path,
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
        )
        accepted.append(
            {
                **pinned,
                "archive": archive.name,
                "archive_remote_commit": archive_record,
                "receipt_remote_commit": receipt_record,
            }
        )
    rejected: list[dict[str, Any]] = []
    for run_id in allowlist["rejected_receipt_only_run_ids"]:
        receipt_path = source_root / f"{analysis.H1}_{run_id}.tar.gz.receipt.json"
        receipt_record = committer.add(receipt_path, b"{}\n")
        rejected.append(
            {
                "run_id": run_id,
                "reason": "receipt_only_archive_absent_on_server",
                "receipt_remote_commit": receipt_record,
            }
        )
    ledger = {
        "schema": "h1_v3_server_verified_compatibility_v1",
        "source_revision": allowlist["source_revision"],
        "source_audit_sha256": allowlist["source_audit_sha256"],
        "accepted_count": 12,
        "reusable_count": 11,
        "rejected_receipt_only_count": 8,
        "accepted_archives": accepted,
        "rejected_receipt_only": rejected,
    }
    ledger_path = (
        analysis._archive_root(drive_root, config)
        / "migration"
        / "h1_v3_server_verified_ledger.json"
    )
    committer.add(
        ledger_path,
        (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode(),
    )
    monkeypatch.setattr(analysis, "_h1_allowlist", lambda _: allowlist)
    return config, ledger, drive_root, ledger_path, accepted


def test_json_parser_accepts_warning_before_final_value() -> None:
    assert analysis._parse_json_stdout("warning first\n{\"exists\": true}\n") == {
        "exists": True
    }


def test_current_h1_status_requires_the_full_locked_scientific_contract(
    tmp_path: Path,
) -> None:
    committer = FakeCommitter()
    config, case_row, bundle_root, drive_root, status = h1_current_status_fixture(
        tmp_path=tmp_path, committer=committer
    )

    result = analysis._validate_current_status(
        committer=committer,
        bundle=analysis.H1,
        config=config,
        case_row=case_row,
        shard_index=0,
        status=status,
        drive_root=drive_root,
        bundle_root=bundle_root,
    )

    assert result is not None
    assert result["receipt"]["run_id"] == "08_h1_endpoint_packet_d417be5f10219c5c"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numerical_status", "hard_failure"),
        ("global_sample_indices", [99]),
        ("source_hashes", {"classA_U1FGTN_gpu.py": "stale"}),
    ],
)
def test_current_h1_status_rejects_unsafe_or_stale_manifest_fields(
    tmp_path: Path, field: str, value: Any
) -> None:
    committer = FakeCommitter()
    config, case_row, bundle_root, drive_root, status = h1_current_status_fixture(
        tmp_path=tmp_path, committer=committer
    )
    status["manifest"][field] = value

    with pytest.raises(RuntimeError, match="non-current"):
        analysis._validate_current_status(
            committer=committer,
            bundle=analysis.H1,
            config=config,
            case_row=case_row,
            shard_index=0,
            status=status,
            drive_root=drive_root,
            bundle_root=bundle_root,
        )


@pytest.mark.parametrize(
    "commit_field", ("archive_remote_commit", "receipt_remote_commit")
)
def test_h1_legacy_loader_requires_declared_commits_and_pinned_remote_identity(
    tmp_path: Path, monkeypatch: Any, commit_field: str
) -> None:
    committer = FakeCommitter()
    config, ledger, drive_root, ledger_path, accepted = h1_legacy_fixture(
        tmp_path=tmp_path, committer=committer, monkeypatch=monkeypatch
    )
    accepted[0].pop(commit_field)
    committer.add(
        ledger_path,
        (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode(),
    )

    with pytest.raises(RuntimeError, match="lacks exact remote commit"):
        analysis.load_h1_legacy_slots(
            campaign_root=CAMPAIGN,
            drive_root=drive_root,
            config=config,
            committer=committer,
        )


def test_h1_legacy_loader_accepts_exactly_eleven_fully_pinned_inputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    committer = FakeCommitter()
    config, _, drive_root, _, _ = h1_legacy_fixture(
        tmp_path=tmp_path, committer=committer, monkeypatch=monkeypatch
    )

    reusable = analysis.load_h1_legacy_slots(
        campaign_root=CAMPAIGN,
        drive_root=drive_root,
        config=config,
        committer=committer,
    )

    assert len(reusable) == 11
    assert all(
        row["archive_remote_commit"]["remote_sha256"]
        == row["allowlist_row"]["archive_sha256"]
        for row in reusable.values()
    )


def test_h1_legacy_loader_rejects_self_consistent_remote_replacement(
    tmp_path: Path, monkeypatch: Any
) -> None:
    committer = FakeCommitter()
    config, ledger, drive_root, ledger_path, accepted = h1_legacy_fixture(
        tmp_path=tmp_path, committer=committer, monkeypatch=monkeypatch
    )
    source_root = (
        drive_root
        / "classA_final_production_outputs"
        / ledger["source_revision"]
        / analysis.H1
    )
    archive = source_root / accepted[0]["archive"]
    replacement = b"self-consistent replacement"
    replacement_record = committer.add(archive, replacement)
    receipt = {
        "archive": archive.name,
        "archive_bytes": len(replacement),
        "archive_sha256": hashlib.sha256(replacement).hexdigest(),
    }
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    receipt_record = committer.add(
        receipt_path,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    accepted[0]["archive_remote_commit"] = replacement_record
    accepted[0]["receipt_remote_commit"] = receipt_record
    committer.add(
        ledger_path,
        (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode(),
    )

    with pytest.raises(RuntimeError, match="immutable compatibility allowlist"):
        analysis.load_h1_legacy_slots(
            campaign_root=CAMPAIGN,
            drive_root=drive_root,
            config=config,
            committer=committer,
        )


def test_h1_legacy_loader_binds_receipt_payload_to_pinned_archive(
    tmp_path: Path, monkeypatch: Any
) -> None:
    committer = FakeCommitter()
    config, ledger, drive_root, ledger_path, accepted = h1_legacy_fixture(
        tmp_path=tmp_path, committer=committer, monkeypatch=monkeypatch
    )
    source_root = (
        drive_root
        / "classA_final_production_outputs"
        / ledger["source_revision"]
        / analysis.H1
    )
    archive = source_root / accepted[0]["archive"]
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    bad_receipt = {
        "archive": archive.name,
        "archive_bytes": accepted[0]["archive_bytes"],
        "archive_sha256": "0" * 64,
    }
    accepted[0]["receipt_remote_commit"] = committer.add(
        receipt_path,
        (json.dumps(bad_receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    committer.add(
        ledger_path,
        (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode(),
    )

    with pytest.raises(RuntimeError, match="receipt disagrees"):
        analysis.load_h1_legacy_slots(
            campaign_root=CAMPAIGN,
            drive_root=drive_root,
            config=config,
            committer=committer,
        )


def test_analysis_api_lease_blocks_fresh_owner_and_recovers_stale(
    tmp_path: Path, monkeypatch: Any
) -> None:
    committer = FakeCommitter()
    config = {"sampling_revision": "revision-v4", "audit_sha256": "audit"}
    path = tmp_path / "drive" / "analysis" / "lease.json"
    monkeypatch.setattr(analysis, "ANALYSIS_LEASE_SETTLE_SECONDS", 0.0)
    first = analysis._AnalysisApiLease(
        bundle=analysis.P1, config=config, path=path, committer=committer
    )
    second = analysis._AnalysisApiLease(
        bundle=analysis.P1, config=config, path=path, committer=committer
    )
    with first:
        with pytest.raises(RuntimeError, match="another analysis runtime owns"):
            second.__enter__()

    stale = first._payload()
    stale["updated_unix"] = time.time() - analysis.ANALYSIS_LEASE_STALE_SECONDS - 1
    committer.add(
        path, (json.dumps(stale, indent=2, sort_keys=True) + "\n").encode()
    )
    recovered = analysis._AnalysisApiLease(
        bundle=analysis.P1, config=config, path=path, committer=committer
    )
    with recovered:
        recovered.assert_owned()
    with pytest.raises(FakeRemoteError):
        committer.path_commit_record(path)


def test_analysis_api_lease_elects_one_of_two_concurrent_absent_claims(
    tmp_path: Path, monkeypatch: Any
) -> None:
    committer = AmbiguousNameCommitter()
    config = {"sampling_revision": "revision-v4", "audit_sha256": "audit"}
    path = tmp_path / "drive" / "analysis" / "lease.json"
    first = analysis._AnalysisApiLease(
        bundle=analysis.H1, config=config, path=path, committer=committer
    )
    second = analysis._AnalysisApiLease(
        bundle=analysis.H1, config=config, path=path, committer=committer
    )
    first.token = "1" * 32
    second.token = "2" * 32
    for lease in (first, second):
        committer.add(
            path,
            (json.dumps(lease._payload(), indent=2, sort_keys=True) + "\n").encode(),
        )
    monkeypatch.setattr(analysis, "ANALYSIS_LEASE_SETTLE_SECONDS", 0.0)

    with pytest.raises(RuntimeError, match="another analysis runtime won"):
        second._elect(require_own_claim=True)
    winner, _ = first._elect(require_own_claim=True)

    assert winner["owner_token"] == first.token
    assert len(first._claims()) == 1


def test_analysis_lease_never_deletes_a_relocated_claim(tmp_path: Path) -> None:
    committer = FakeCommitter()
    config = {"sampling_revision": "revision-v4", "audit_sha256": "audit"}
    path = tmp_path / "drive" / "analysis" / "lease.json"
    lease = analysis._AnalysisApiLease(
        bundle=analysis.P1, config=config, path=path, committer=committer
    )
    payload = lease._payload()
    record = committer.add(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    file_id = str(record["remote_file_id"])
    committer.by_id[file_id]["remote_parent_id"] = "parent:relocated"

    with pytest.raises(RuntimeError, match="changed|moved"):
        lease._delete(payload, record)

    assert file_id in committer.by_id


def test_materialization_downloads_exact_archive_and_receipt_locally(
    tmp_path: Path,
) -> None:
    committer = FakeCommitter()
    manifest = {
        "bundle": analysis.P1,
        "status": "complete_local",
        "case_id": "case",
        "shard_index": 0,
    }
    row = current_plan_row(
        committer=committer, drive_root=tmp_path / "logical-drive", manifest=manifest
    )

    result = analysis.materialize_inputs(
        bundle=analysis.P1,
        plan=[row],
        local_archive_root=tmp_path / "local" / "archives",
        committer=committer,
    )

    archive = Path(result[0]["local_archive"])
    receipt = Path(result[0]["local_receipt"])
    assert analysis._root_manifest(archive) == manifest
    assert json.loads(receipt.read_text()) == row["receipt"]
    assert archive.is_relative_to(tmp_path / "local")


def test_analysis_outputs_publish_as_immutable_generation_then_verified_pointer(
    tmp_path: Path, monkeypatch: Any
) -> None:
    committer = FakeCommitter()
    manifest = {"bundle": analysis.P1, "case_id": "case", "shard_index": 0}
    drive_root = tmp_path / "logical-drive"
    row = current_plan_row(
        committer=committer, drive_root=drive_root, manifest=manifest
    )
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (output_root / "summary.json").write_text('{"status":"complete"}\n')
    (output_root / "figure.bin").write_bytes(b"figure")
    config = {
        "sampling_revision": "revision-v4",
        "audit_sha256": "audit",
        "production_output_collection": "outputs",
        "output_bundle": "bundle",
    }
    monkeypatch.setattr(analysis, "publish_json", analysis.publish_json)

    result = analysis.publish_analysis_generation(
        bundle=analysis.P1,
        config=config,
        drive_root=drive_root,
        plan=[row],
        output_root=output_root,
        summary={"status": "complete"},
        analysis_source={
            "path": "src/analysis.py",
            "sha256": "source",
            "bundle_source_identity": {"aggregate_sha256": "bundle"},
        },
        committer=committer,
    )

    assert result["status"] == "server_verified"
    assert result["output_count"] == 2
    assert "/generations/" in result["outputs"][0]["remote_path"]
    assert result["receipt_path"].endswith("server_verified_analysis_receipt.json")
    assert committer.quota_calls == [
        (sum(path.stat().st_size for path in output_root.iterdir()), 0)
    ]
    assert set(committer.upload_headrooms) == {0}
    server_receipt = json.loads(
        committer.download_bytes(result["receipt_remote_commit"]["remote_file_id"])
    )
    assert server_receipt["generation_id"] == result["generation_id"]


def test_existing_generation_can_republish_missing_receipt_without_output_quota(
    tmp_path: Path,
) -> None:
    committer = FakeCommitter()
    drive_root = tmp_path / "logical-drive"
    row = current_plan_row(
        committer=committer,
        drive_root=drive_root,
        manifest={"bundle": analysis.P1, "case_id": "case", "shard_index": 0},
    )
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (output_root / "summary.json").write_text('{"status":"complete"}\n')
    config = {
        "sampling_revision": "revision-v4",
        "audit_sha256": "audit",
        "production_output_collection": "outputs",
        "output_bundle": "bundle",
    }
    kwargs = {
        "bundle": analysis.P1,
        "config": config,
        "drive_root": drive_root,
        "plan": [row],
        "output_root": output_root,
        "summary": {"status": "complete"},
        "analysis_source": {
            "path": "src/analysis.py",
            "sha256": "source",
            "bundle_source_identity": {"aggregate_sha256": "bundle"},
        },
        "committer": committer,
    }
    first = analysis.publish_analysis_generation(**kwargs)
    receipt_path = first["receipt_path"]
    receipt_record = committer.by_path.pop(receipt_path)
    committer.by_id.pop(receipt_record["remote_file_id"])
    committer.raw_by_id.pop(receipt_record["remote_file_id"])
    quota_calls = list(committer.quota_calls)
    committer.block_quota = True

    recovered = analysis.publish_analysis_generation(**kwargs)

    assert recovered["generation_id"] == first["generation_id"]
    assert committer.quota_calls == quota_calls
    assert recovered["receipt_path"] in committer.by_path
