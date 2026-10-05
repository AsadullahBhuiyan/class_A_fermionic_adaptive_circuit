from __future__ import annotations

import errno
import hashlib
import contextlib
import importlib.util
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_new_designs"
    / "08_h1_endpoint_packet"
)
SRC = BUNDLE / "src"
_ISOLATED_MODULE_NAMES = {
    path.stem for path in SRC.glob("*.py") if path.stem != "__init__"
} | {
    "h1_operational",
    "h1_operational_test_isolated",
    "h1_packet_runner_test_isolated",
}


@contextlib.contextmanager
def isolated_h1_bundle_imports() -> Any:
    """Load frozen H1 modules without leaking paths/modules to other tests."""

    original_path = list(sys.path)
    saved_modules = {
        name: sys.modules.pop(name)
        for name in _ISOLATED_MODULE_NAMES
        if name in sys.modules
    }
    sys.path.insert(0, str(SRC))
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in _ISOLATED_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def load_isolated_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


with isolated_h1_bundle_imports():
    operational = load_isolated_module(
        "h1_operational_test_isolated", BUNDLE / "h1_operational.py"
    )
    runner = load_isolated_module(
        "h1_packet_runner_test_isolated", SRC / "h1_packet_runner.py"
    )


class FakeCommitter:
    def __init__(self) -> None:
        self.by_path: dict[str, dict[str, Any]] = {}
        self.by_id: dict[str, dict[str, Any]] = {}
        self.raw_by_id: dict[str, bytes] = {}
        self.downloaded_ids: list[str] = []
        self.next_id = 1

    def add(
        self,
        path: Path | str,
        raw: bytes,
        *,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        path = Path(path)
        file_id = f"file-{self.next_id}"
        self.next_id += 1
        record = {
            "schema": "classA_drive_api_commit_v1",
            "remote_file_id": file_id,
            "remote_name": path.name if name is None else name,
            "remote_parent_id": (
                f"parent:{path.parent}" if parent_id is None else parent_id
            ),
            "remote_bytes": len(raw),
            "remote_sha256": hashlib.sha256(raw).hexdigest(),
            "remote_verified_unix": 1.0,
        }
        self.by_path[str(path)] = record
        self.by_id[file_id] = record
        self.raw_by_id[file_id] = bytes(raw)
        return dict(record)

    def path_commit_record(self, path: Path | str) -> dict[str, Any]:
        try:
            return dict(self.by_path[str(Path(path))])
        except KeyError as exc:
            raise runner.RemoteCommitError(f"Drive file is absent: {path}") from exc

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
            if row["remote_parent_id"] == parent_id
            and row["remote_name"] == name
        ]

    def metadata(self, file_id: str) -> dict[str, Any]:
        try:
            row = self.by_id[file_id]
        except KeyError as exc:
            raise runner.RemoteCommitError("Drive file is absent: id") from exc
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
        try:
            current = self.by_id[str(record["remote_file_id"])]
        except KeyError as exc:
            raise runner.RemoteCommitError("Drive file is absent") from exc
        if operational._commit_identity(current) != operational._commit_identity(record):
            raise runner.RemoteCommitError("Drive server verification failed")
        return dict(current)

    def download_bytes(self, file_id: str) -> bytes:
        self.downloaded_ids.append(file_id)
        return self.raw_by_id[file_id]

    def download_to(self, file_id: str, destination: Path | str) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.download_bytes(file_id))
        return destination

    def put_json(
        self, path: Path | str, payload: dict[str, Any], *, replace: bool
    ) -> dict[str, Any]:
        path = Path(path)
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        key = str(path)
        if key not in self.by_path:
            return self.add(path, raw)
        if not replace:
            current = self.by_path[key]
            if self.raw_by_id[current["remote_file_id"]] != raw:
                raise runner.RemoteCommitError("existing Drive bytes differ")
            return dict(current)
        current = self.by_path[key]
        current["remote_bytes"] = len(raw)
        current["remote_sha256"] = hashlib.sha256(raw).hexdigest()
        self.raw_by_id[current["remote_file_id"]] = raw
        self.by_id[current["remote_file_id"]] = current
        return dict(current)

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
            raise runner.RemoteCommitError("ambiguous Drive path")
        if not matches:
            raise runner.RemoteCommitError(f"Drive file is absent: {path}")
        return dict(matches[0])


def fake_runner(committer: FakeCommitter) -> SimpleNamespace:
    return SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
    )


def receipt_bytes(
    *, archive: Path, archive_record: dict[str, Any], run_id: str
) -> bytes:
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "archive": archive.name,
        "archive_sha256": archive_record["remote_sha256"],
        "archive_bytes": archive_record["remote_bytes"],
        "created_unix": 1.0,
        "archive_remote_commit": archive_record,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def test_h1_scientific_source_digest_remains_the_saved_qualification_digest() -> None:
    assert runner.sha256_json(runner._source_hashes(SRC)) == (
        "30308f34f64a36a3ed110173679343e70c0e167a10859979f793c90aa1c22a31"
    )


def test_h1_test_imports_do_not_shadow_repository_src() -> None:
    assert str(BUNDLE) not in sys.path
    import src as repository_src

    repository_paths = {Path(path).resolve() for path in repository_src.__path__}
    assert (ROOT / "src").resolve() in repository_paths
    assert SRC.resolve() not in repository_paths


def test_replay_enotconn_uses_explicit_registered_fresh_seed_fallback() -> None:
    def disconnected(**_: Any):
        raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")

    source, reasons = operational.replay_lookup_with_mount_fallback(
        disconnected, drive_root=Path("/content/drive/MyDrive")
    )
    assert source is None
    assert reasons == [
        "response_replay_drivefs_unavailable:"
        f"errno={errno.ENOTCONN}:fresh_same_preregistered_seed"
    ]


def test_replay_non_mount_io_error_is_not_hidden() -> None:
    def denied(**_: Any):
        raise OSError(errno.EACCES, "permission denied")

    with pytest.raises(OSError) as exc:
        operational.replay_lookup_with_mount_fallback(denied)
    assert exc.value.errno == errno.EACCES


def test_exact_remote_pair_verifies_and_only_then_cleans_scratch(tmp_path: Path) -> None:
    run_id = f"{runner.BUNDLE}_operational_{tmp_path.name}"
    archive = tmp_path / "drive" / f"{run_id}.tar.gz"
    committer = FakeCommitter()
    archive_record = committer.add(archive, b"durable archive")
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    committer.add(
        receipt_path,
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ),
        parent_id=archive_record["remote_parent_id"],
    )
    scratch = operational._local_scratch_root(runner.BUNDLE, run_id)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "stale").write_text("safe to delete only after verification")

    receipt = operational.verify_remote_existing(fake_runner(committer), archive)
    assert receipt is not None
    assert receipt["archive_sha256"] == archive_record["remote_sha256"]
    assert not scratch.exists()


@pytest.mark.parametrize("field", ["remote_file_id", "remote_name", "remote_parent_id"])
def test_receipt_archive_record_must_bind_to_exact_expected_path(
    tmp_path: Path, field: str
) -> None:
    run_id = f"{runner.BUNDLE}_misbound_{field}_{tmp_path.name}"
    archive = tmp_path / "drive" / f"{run_id}.tar.gz"
    committer = FakeCommitter()
    archive_record = committer.add(archive, b"durable archive")
    declared = dict(archive_record)
    declared[field] = f"wrong-{field}"
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    committer.add(
        receipt_path,
        receipt_bytes(archive=archive, archive_record=declared, run_id=run_id),
        parent_id=archive_record["remote_parent_id"],
    )
    scratch = operational._local_scratch_root(runner.BUNDLE, run_id)
    scratch.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="not bound to the expected Drive path"):
        operational.verify_remote_existing(fake_runner(committer), archive)
    assert scratch.exists()
    scratch.rmdir()


def test_archive_without_receipt_is_distinct_repairable_orphan(tmp_path: Path) -> None:
    run_id = f"{runner.BUNDLE}_orphan_{tmp_path.name}"
    archive = tmp_path / "drive" / f"{run_id}.tar.gz"
    committer = FakeCommitter()
    archive_record = committer.add(archive, b"durable orphan")
    with pytest.raises(operational.OrphanArchiveError) as exc:
        operational.verify_remote_existing(fake_runner(committer), archive)
    assert exc.value.archive == archive
    assert exc.value.archive_record["remote_file_id"] == archive_record["remote_file_id"]


def add_tar_member(handle: tarfile.TarFile, name: str, raw: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(raw)
    handle.addfile(member, io.BytesIO(raw))


def valid_orphan_archive(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any],
    shard_index: int, drive_root: Path,
) -> tuple[Path, Path, str, bytes]:
    scratch, archive, run_id, run_config = runner._archive_paths(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=shard_index,
        drive_root=drive_root,
        mode="production",
    )
    packet_raw = {
        "common.npz": b"common",
        "packet_drift.npz": b"packet",
        "primary_profiles.npz": b"profiles",
    }
    record_raw = b"record"
    packet_files = [
        {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        for name, raw in packet_raw.items()
    ]
    source_hashes = runner._source_hashes(bundle_root / "src")
    manifest = {
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
        "root_seed": config["root_seed"],
        "case_id": case["case_id"],
        "protocol": case["protocol"],
        "alpha_1": case["model"]["alpha_1"],
        "shard_index": shard_index,
        "global_sample_indices": list(
            range(shard_index * runner.SHARD_SIZE, (shard_index + 1) * runner.SHARD_SIZE)
        ),
        "shard_generator_seed": runner.shard_seed(
            config["root_seed"], case["case_id"], shard_index
        ),
        "source_hashes": source_hashes,
        "numerical_status": "pass",
        "gpu_preflight": {"device": "NVIDIA A100-SXM4-40GB"},
        "products": {
            "h1_endpoint_packet": {"files": packet_files},
            "ordered_born_record": {
                "path": str(scratch / "shards/shard_000/ordered_born_record.npz"),
                "sha256": hashlib.sha256(record_raw).hexdigest(),
                "bytes": len(record_raw),
            },
        },
        "created_unix": 123.0,
    }
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as handle:
        add_tar_member(handle, "manifest.json", json.dumps(manifest).encode())
        for name, raw in packet_raw.items():
            add_tar_member(
                handle, f"shards/shard_{shard_index:03d}/h1_endpoint_packet/{name}", raw
            )
        add_tar_member(
            handle,
            f"shards/shard_{shard_index:03d}/ordered_born_record.npz",
            record_raw,
        )
    return scratch, archive, run_id, output.getvalue()


def test_repair_orphan_validates_publishes_verifies_then_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.load_config(BUNDLE)
    case = next(row for row in runner.expand_cases(config) if row["protocol"] == "hard")
    drive_root = tmp_path / "drive"
    scratch, archive, _, raw = valid_orphan_archive(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=4,
        drive_root=drive_root,
    )
    committer = FakeCommitter()
    archive_record = committer.add(archive, raw)

    def publish_json(
        selected: FakeCommitter,
        payload: dict[str, Any],
        path: Path | str,
        *,
        replace: bool,
        **_: Any,
    ) -> dict[str, Any]:
        assert selected is committer
        return selected.put_json(path, payload, replace=replace)

    monkeypatch.setattr(runner, "_server_commit_required", lambda _: True)
    monkeypatch.setattr(runner, "_remote_committer", lambda _: committer)
    monkeypatch.setattr(runner, "publish_json", publish_json)
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "complete-local-products").write_text("present")

    result = operational.repair_orphan(
        runner,
        bundle_root=BUNDLE,
        drive_root=drive_root,
        mode="production",
        case_id=case["case_id"],
        shard_index=4,
    )
    assert result["status"] == "repaired_and_verified"
    assert result["repaired"] is True
    assert result["archive_sha256"] == archive_record["remote_sha256"]
    assert result["cleaned_scratch"] is True
    assert not scratch.exists()
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    assert str(receipt_path) in committer.by_path
    assert operational.verify_remote_existing(runner, archive) is not None


def test_repair_orphan_is_successful_noop_when_final_archive_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.load_config(BUNDLE)
    case = next(row for row in runner.expand_cases(config) if row["protocol"] == "hard")
    drive_root = tmp_path / "drive"
    committer = FakeCommitter()
    monkeypatch.setattr(runner, "_server_commit_required", lambda _: True)
    monkeypatch.setattr(runner, "_remote_committer", lambda _: committer)

    result = operational.repair_orphan(
        runner,
        bundle_root=BUNDLE,
        drive_root=drive_root,
        mode="production",
        case_id=case["case_id"],
        shard_index=3,
    )

    assert set(result) == {"status", "repaired", "archive"}
    assert result["status"] == "no_final_archive"
    assert result["repaired"] is False
    assert result["archive"].endswith(".tar.gz")


def test_repair_orphan_never_races_a_fresh_scientific_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.load_config(BUNDLE)
    case = next(row for row in runner.expand_cases(config) if row["protocol"] == "hard")
    drive_root = tmp_path / "drive"
    _, archive, run_id, raw = valid_orphan_archive(
        bundle_root=BUNDLE,
        config=config,
        case=case,
        shard_index=2,
        drive_root=drive_root,
    )
    committer = FakeCommitter()
    committer.add(archive, raw)
    monkeypatch.setattr(runner, "_server_commit_required", lambda _: True)
    monkeypatch.setattr(runner, "_remote_committer", lambda _: committer)

    holder = operational._H1ApiShardLease(
        runner=runner,
        archive=archive,
        run_id=run_id,
        case_id=case["case_id"],
        shard_index=2,
    )
    lease_raw = (
        json.dumps(holder._payload(), indent=2, sort_keys=True) + "\n"
    ).encode()
    committer.add(holder.path, lease_raw)

    with pytest.raises(RuntimeError, match="another H1 writer owns"):
        operational.repair_orphan(
            runner,
            bundle_root=BUNDLE,
            drive_root=drive_root,
            mode="production",
            case_id=case["case_id"],
            shard_index=2,
        )

    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    with pytest.raises(runner.RemoteCommitError):
        committer.path_commit_record(receipt_path)


def test_manifest_sidecar_is_exactly_bound_and_avoids_archive_download(
    tmp_path: Path,
) -> None:
    run_id = f"{runner.BUNDLE}_sidecar_{tmp_path.name}"
    archive = tmp_path / "drive" / f"{run_id}.tar.gz"
    committer = FakeCommitter()
    archive_record = committer.add(archive, b"large archive bytes")
    receipt = json.loads(
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ).decode()
    )
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    committer.add(
        receipt_path,
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ),
        parent_id=archive_record["remote_parent_id"],
    )
    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
        publish_json=lambda selected, payload, path, replace, **_: selected.put_json(
            path, payload, replace=replace
        ),
    )
    manifest = {"bundle": runner.BUNDLE, "case_id": "case", "shard_index": 2}

    operational.publish_manifest_sidecar(
        fake, archive=archive, receipt=receipt, manifest=manifest
    )
    committer.downloaded_ids.clear()
    observed = operational.read_manifest_sidecar(fake, archive)

    assert observed == manifest
    assert archive_record["remote_file_id"] not in committer.downloaded_ids


def test_manifest_sidecar_converges_identical_concurrent_backfills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = f"{runner.BUNDLE}_sidecar_race_{tmp_path.name}"
    archive = tmp_path / "drive" / f"{run_id}.tar.gz"
    committer = AmbiguousNameCommitter()
    archive_record = committer.add(archive, b"large archive bytes")
    receipt = json.loads(
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ).decode()
    )
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    committer.add(
        receipt_path,
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ),
        parent_id=archive_record["remote_parent_id"],
    )
    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
        publish_json=lambda selected, payload, path, replace, **_: selected.put_json(
            path, payload, replace=replace
        ),
    )
    manifest = {"bundle": runner.BUNDLE, "case_id": "case", "shard_index": 2}
    operational.publish_manifest_sidecar(
        fake, archive=archive, receipt=receipt, manifest=manifest
    )
    sidecar = operational._manifest_sidecar_path(archive)
    first = committer.by_path[str(sidecar)]
    committer.add(
        sidecar,
        committer.raw_by_id[str(first["remote_file_id"])],
        parent_id=str(first["remote_parent_id"]),
    )
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)

    assert operational.read_manifest_sidecar(fake, archive) == manifest
    parts, name = committer.split_remote_path(sidecar)
    parent_id = committer.resolve_folder(parts, create=False)
    assert len(committer._list_children(parent_id, name)) == 1


def test_manifest_sidecar_recovers_publication_time_duplicate_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = f"{runner.BUNDLE}_sidecar_publish_race_{tmp_path.name}"
    archive = tmp_path / "drive" / f"{run_id}.tar.gz"
    committer = AmbiguousNameCommitter()
    archive_record = committer.add(archive, b"large archive bytes")
    receipt = json.loads(
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ).decode()
    )
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    committer.add(
        receipt_path,
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ),
        parent_id=archive_record["remote_parent_id"],
    )

    def racing_publish(
        selected: AmbiguousNameCommitter,
        payload: dict[str, Any],
        path: Path | str,
        **_: Any,
    ) -> dict[str, Any]:
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        selected.add(path, raw)
        selected.add(path, raw)
        raise runner.RemoteCommitError(
            f"final Drive publication produced 2 files named {Path(path).name!r}"
        )

    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
        publish_json=racing_publish,
    )
    manifest = {"bundle": runner.BUNDLE, "case_id": "case", "shard_index": 2}
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)

    published = operational.publish_manifest_sidecar(
        fake, archive=archive, receipt=receipt, manifest=manifest
    )

    assert published["manifest"] == manifest
    sidecar = operational._manifest_sidecar_path(archive)
    parts, name = committer.split_remote_path(sidecar)
    parent_id = committer.resolve_folder(parts, create=False)
    assert len(committer._list_children(parent_id, name)) == 1


def test_manifest_sidecar_fails_closed_on_nonidentical_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = f"{runner.BUNDLE}_sidecar_mismatch_{tmp_path.name}"
    archive = tmp_path / "drive" / f"{run_id}.tar.gz"
    committer = AmbiguousNameCommitter()
    archive_record = committer.add(archive, b"large archive bytes")
    receipt = json.loads(
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ).decode()
    )
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    committer.add(
        receipt_path,
        receipt_bytes(
            archive=archive, archive_record=archive_record, run_id=run_id
        ),
        parent_id=archive_record["remote_parent_id"],
    )
    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
        publish_json=lambda selected, payload, path, replace, **_: selected.put_json(
            path, payload, replace=replace
        ),
    )
    operational.publish_manifest_sidecar(
        fake,
        archive=archive,
        receipt=receipt,
        manifest={"bundle": runner.BUNDLE, "case_id": "case", "shard_index": 2},
    )
    sidecar = operational._manifest_sidecar_path(archive)
    first = committer.by_path[str(sidecar)]
    committer.add(
        sidecar,
        b'{"different": true}\n',
        parent_id=str(first["remote_parent_id"]),
    )
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)

    with pytest.raises(RuntimeError, match="non-identical payloads"):
        operational.read_manifest_sidecar(fake, archive)
    parts, name = committer.split_remote_path(sidecar)
    parent_id = committer.resolve_folder(parts, create=False)
    assert len(committer._list_children(parent_id, name)) == 2


def test_manifest_sidecar_never_deletes_a_relocated_loser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RelocatingCommitter(AmbiguousNameCommitter):
        target_id: str | None = None
        target_metadata_calls = 0

        def metadata(self, file_id: str) -> dict[str, Any]:
            if file_id == self.target_id:
                self.target_metadata_calls += 1
                if self.target_metadata_calls == 4:
                    self.by_id[file_id]["remote_parent_id"] = "parent:relocated"
            return super().metadata(file_id)

    committer = RelocatingCommitter()
    path = tmp_path / "drive" / "status.json"
    raw = b'{"same": true}\n'
    committer.add(path, raw)
    loser = committer.add(path, raw)
    committer.target_id = str(loser["remote_file_id"])
    fake = SimpleNamespace(RemoteCommitError=runner.RemoteCommitError)
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)

    with pytest.raises(RuntimeError, match="changed|moved"):
        operational._converge_identical_json_duplicates(fake, committer, path)

    assert str(loser["remote_file_id"]) in committer.by_id


def test_lexical_resolve_context_never_touches_disconnected_drive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.resolve
    drive_root = Path("/content/drive/MyDrive")

    def disconnected(self: Path, strict: bool = False) -> Path:
        if str(self).startswith(str(drive_root)):
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        return original(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", disconnected)
    with operational.lexical_drive_resolve_context(drive_root):
        assert (drive_root / "outputs/archive.tar.gz").resolve() == (
            drive_root / "outputs/archive.tar.gz"
        )


def test_h1_api_lease_blocks_concurrent_owner_and_recovers_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committer = FakeCommitter()

    def publish_json(
        selected: FakeCommitter,
        payload: dict[str, Any],
        path: Path | str,
        *,
        replace: bool,
        **_: Any,
    ) -> dict[str, Any]:
        return selected.put_json(path, payload, replace=replace)

    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
        publish_json=publish_json,
    )
    archive = tmp_path / "drive" / "run.tar.gz"
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)
    first = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    second = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    with first:
        with pytest.raises(RuntimeError, match="another H1 writer"):
            second.__enter__()

    stale = first._payload()
    stale["owner_token"] = "dead-runtime"
    stale["updated_unix"] = time.time() - operational.H1_API_LEASE_STALE_SECONDS - 1
    committer.put_json(first.path, stale, replace=False)
    recovered = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    with recovered:
        recovered.assert_owned()
    with pytest.raises(runner.RemoteCommitError):
        committer.path_commit_record(recovered.path)


def test_h1_api_lease_elects_one_concurrent_absent_claim_and_cleans_loser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DuplicateCommitter(FakeCommitter):
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
                raise runner.RemoteCommitError("ambiguous Drive path")
            if not matches:
                raise runner.RemoteCommitError(f"Drive file is absent: {path}")
            return dict(matches[0])

    committer = DuplicateCommitter()
    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
    )
    archive = tmp_path / "drive" / "run.tar.gz"
    first = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    second = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    first.token = "1" * 32
    second.token = "2" * 32
    for lease in (first, second):
        raw = (json.dumps(lease._payload(), indent=2, sort_keys=True) + "\n").encode()
        committer.add(lease.path, raw)
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)

    assert second._reconcile_concurrent_claims() is False
    assert first._reconcile_concurrent_claims() is True
    remaining = first._claim_records()
    assert len(remaining) == 1
    assert remaining[0][0]["owner_token"] == first.token


def test_h1_api_lease_startup_converges_interrupted_duplicate_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DuplicateCommitter(FakeCommitter):
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
                raise runner.RemoteCommitError("ambiguous Drive path")
            if not matches:
                raise runner.RemoteCommitError(f"Drive file is absent: {path}")
            return dict(matches[0])

    committer = DuplicateCommitter()
    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
    )
    archive = tmp_path / "drive" / "run.tar.gz"
    winner = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    interrupted_loser = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    arriving_runtime = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    winner.token = "1" * 32
    interrupted_loser.token = "2" * 32
    arriving_runtime.token = "3" * 32
    for lease in (winner, interrupted_loser):
        raw = (json.dumps(lease._payload(), indent=2, sort_keys=True) + "\n").encode()
        committer.add(lease.path, raw)
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)

    with pytest.raises(RuntimeError, match="another H1 writer owns"):
        arriving_runtime.__enter__()

    remaining = winner._claim_records()
    assert len(remaining) == 1
    assert remaining[0][0]["owner_token"] == winner.token


def test_h1_lease_never_deletes_a_relocated_claim(tmp_path: Path) -> None:
    committer = FakeCommitter()
    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
    )
    archive = tmp_path / "drive" / "run.tar.gz"
    lease = operational._H1ApiShardLease(
        runner=fake,
        archive=archive,
        run_id="run",
        case_id="case",
        shard_index=1,
    )
    payload = lease._payload()
    record = committer.add(
        lease.path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    file_id = str(record["remote_file_id"])
    committer.by_id[file_id]["remote_parent_id"] = "parent:relocated"

    with pytest.raises(RuntimeError, match="changed|moved"):
        lease._delete_claim(payload, record)

    assert file_id in committer.by_id


def test_h1_migration_initializer_lease_elects_one_of_two_claimants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committer = AmbiguousNameCommitter()
    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        RemoteCommitError=runner.RemoteCommitError,
        _remote_committer=lambda _: committer,
    )
    ledger = tmp_path / "drive" / "migration" / "ledger.json"
    first = operational._H1ApiShardLease(
        runner=fake,
        archive=ledger,
        run_id=f"{runner.BUNDLE}_v3_migration_initializer",
        case_id="H1_V3_MIGRATION_LEDGER",
        shard_index=-1,
    )
    second = operational._H1ApiShardLease(
        runner=fake,
        archive=ledger,
        run_id=f"{runner.BUNDLE}_v3_migration_initializer",
        case_id="H1_V3_MIGRATION_LEDGER",
        shard_index=-1,
    )
    first.token = "1" * 32
    second.token = "2" * 32
    for lease in (first, second):
        committer.add(
            lease.path,
            (json.dumps(lease._payload(), indent=2, sort_keys=True) + "\n").encode(),
        )
    monkeypatch.setattr(operational, "H1_API_LEASE_SETTLE_SECONDS", 0.0)

    assert second._reconcile_concurrent_claims() is False
    assert first._reconcile_concurrent_claims() is True
    remaining = first._claim_records()
    assert len(remaining) == 1
    assert remaining[0][0]["owner_token"] == first.token


def test_strict_preflight_computes_only_on_explicit_server_absence(
    tmp_path: Path,
) -> None:
    committer = FakeCommitter()
    path = tmp_path / "drive" / "a100_preflight.json"
    fake = SimpleNamespace(
        RemoteCommitError=runner.RemoteCommitError,
        load_config=lambda _: {},
        _preflight_path=lambda _drive, _config: path,
        _remote_committer=lambda _: committer,
        require_safe_preflight=lambda **_: (_ for _ in ()).throw(
            AssertionError("validation must not run for an absent receipt")
        ),
    )

    assert operational.strict_preflight_reuse(
        fake, bundle_root=tmp_path, drive_root=tmp_path / "drive"
    ) is None


def test_strict_preflight_never_recomputes_a_present_invalid_receipt(
    tmp_path: Path,
) -> None:
    committer = FakeCommitter()
    path = tmp_path / "drive" / "a100_preflight.json"
    payload = {"safe": True, "schema": "preflight"}
    committer.add(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    fake = SimpleNamespace(
        RemoteCommitError=runner.RemoteCommitError,
        load_config=lambda _: {},
        _preflight_path=lambda _drive, _config: path,
        _remote_committer=lambda _: committer,
        require_safe_preflight=lambda **_: (_ for _ in ()).throw(
            RuntimeError("present receipt is scientifically invalid")
        ),
    )

    with pytest.raises(RuntimeError, match="scientifically invalid"):
        operational.strict_preflight_reuse(
            fake, bundle_root=tmp_path, drive_root=tmp_path / "drive"
        )


def test_run_bundle_adapter_routes_wrapper_and_scientific_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper_path = BUNDLE / "run_bundle.py"
    with isolated_h1_bundle_imports():
        spec = importlib.util.spec_from_file_location(
            "tested_h1_run_bundle", wrapper_path
        )
        assert spec is not None and spec.loader is not None
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        scientific_calls: list[list[str]] = []

        monkeypatch.setattr(
            wrapper,
            "operational_main",
            lambda argv, **_: 0 if "--repair-orphan" in argv else None,
        )
        monkeypatch.setattr(
            wrapper._runner,
            "main",
            lambda argv: scientific_calls.append(list(argv)) or 17,
        )

        assert wrapper.main(["--repair-orphan"]) == 0
        assert scientific_calls == []
        assert wrapper.main(["--remote-status", "--case-id", "hard"]) == 17
        assert scientific_calls == [["--remote-status", "--case-id", "hard"]]


def test_install_uses_original_verifier_for_non_drive_paths(tmp_path: Path) -> None:
    calls: list[Path] = []
    fake = SimpleNamespace(
        _verify_existing=lambda path: calls.append(Path(path)) or {"local": True},
        _find_replay_source=lambda **_: (None, ["original"]),
        _server_commit_required=lambda _: False,
        run_case=lambda **_: {},
        a100_preflight=lambda **_: {},
        _archive=lambda *_, **__: {},
        _root_manifest_from_archive=lambda _: {},
    )
    operational.install_operational_hardening(fake)
    assert fake._verify_existing(tmp_path / "local.tar.gz") == {"local": True}
    assert calls == [tmp_path / "local.tar.gz"]
    assert fake._find_replay_source() == (None, ["original"])


def test_install_substitutes_the_unhashed_root_drive_transport() -> None:
    class RootCommitter:
        pass

    class RootCommitError(RuntimeError):
        pass

    def root_publish(*_: Any, **__: Any) -> None:
        return None

    def root_read(*_: Any, **__: Any) -> dict[str, Any]:
        return {}

    helper = SimpleNamespace(
        DEFAULT_DRIVE_ROOT=Path("/content/drive/MyDrive"),
        DriveRemoteCommitter=RootCommitter,
        RemoteCommitError=RootCommitError,
        publish_json=root_publish,
        read_remote_json=root_read,
    )
    fake = SimpleNamespace(
        _verify_existing=lambda _: None,
        _find_replay_source=lambda **_: (None, []),
        _server_commit_required=lambda _: False,
        run_case=lambda **_: {},
        a100_preflight=lambda **_: {},
        _archive=lambda *_, **__: {},
        _root_manifest_from_archive=lambda _: {},
    )

    operational.install_operational_hardening(fake, drive_helper=helper)

    assert issubclass(fake.DriveRemoteCommitter, RootCommitter)
    assert fake.RemoteCommitError is RootCommitError
    assert fake.publish_json is not root_publish
    assert fake.read_remote_json is root_read


def test_scoped_h1_upload_rechecks_lease_after_server_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads: list[tuple[Path, Path]] = []

    class RootCommitter:
        def __init__(self, *, drive_root: Path) -> None:
            self.drive_root = drive_root

        def upload_verified(
            self, local_path: Path, remote_path: Path, **_: Any
        ) -> dict[str, Any]:
            uploads.append((Path(local_path), Path(remote_path)))
            return {"published": True}

    class RootCommitError(RuntimeError):
        pass

    helper = SimpleNamespace(
        DEFAULT_DRIVE_ROOT=tmp_path / "drive",
        DriveRemoteCommitter=RootCommitter,
        RemoteCommitError=RootCommitError,
        publish_json=lambda *_, **__: {},
        read_remote_json=lambda *_, **__: {},
    )

    class ProbeLease:
        assertions = 0

        def __init__(self, **_: Any) -> None:
            self.run_id = "run"

        def __enter__(self) -> "ProbeLease":
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def assert_owned(self) -> None:
            self.assertions += 1
            if self.assertions == 2:
                raise RuntimeError("lease lost during upload")

    fake = SimpleNamespace(
        BUNDLE=runner.BUNDLE,
        _verify_existing=lambda _: None,
        _find_replay_source=lambda **_: (None, []),
        _server_commit_required=lambda _: True,
        _archive_paths=lambda **_: (
            tmp_path / "scratch",
            tmp_path / "drive" / "outputs" / "run.tar.gz",
            "run",
            {},
        ),
        run_case=None,
        a100_preflight=lambda **_: {},
        _archive=lambda *_, **__: {},
        _root_manifest_from_archive=lambda _: {},
    )

    def original_run_case(**_: Any) -> dict[str, Any]:
        selected = fake.DriveRemoteCommitter(drive_root=tmp_path / "drive")
        selected.upload_verified(
            tmp_path / "local.tar.gz", tmp_path / "drive" / "outputs" / "run.tar.gz"
        )
        return {}

    fake.run_case = original_run_case
    monkeypatch.setattr(operational, "_H1ApiShardLease", ProbeLease)
    operational.install_operational_hardening(fake, drive_helper=helper)

    with pytest.raises(RuntimeError, match="lease lost during upload"):
        fake.run_case(
            bundle_root=tmp_path,
            config={},
            case={"case_id": "case"},
            shard_index=0,
            drive_root=tmp_path / "drive",
            mode="production",
        )

    assert len(uploads) == 1
