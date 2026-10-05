from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs"
BUNDLE = DEPLOYMENT / "01_p1_chern_dynamics"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARDENING = _load("p1_runtime_hardening_under_test", DEPLOYMENT / "p1_runtime_hardening.py")
ROOT_DRIVE = _load("p1_root_drive_under_test", DEPLOYMENT / "drive_remote_commit.py")


class FakeRemoteError(RuntimeError):
    pass


class _Request:
    def __init__(self, callback: Any) -> None:
        self.callback = callback

    def execute(self) -> Any:
        return self.callback()


class _BackendFiles:
    def __init__(self, backend: "Backend") -> None:
        self.backend = backend

    def list(self, *, q: str, **_: Any) -> _Request:
        parent_match = re.search(r"'([^']+)' in parents", q)
        assert parent_match is not None
        parent = parent_match.group(1)
        name_match = re.search(r"name = '([^']*)'", q)
        name = None if name_match is None else name_match.group(1)

        def run() -> dict[str, Any]:
            rows = [
                dict(row)
                for row in self.backend.items.values()
                if row.get("parents") == [parent]
                and not row.get("trashed", False)
                and (name is None or row.get("name") == name)
            ]
            return {"files": rows}

        return _Request(run)

    def delete(self, *, fileId: str) -> _Request:
        def run() -> dict[str, Any]:
            if fileId in self.backend.fail_delete:
                self.backend.fail_delete.remove(fileId)
                raise FakeRemoteError("simulated delete timeout")
            if fileId not in self.backend.items:
                raise FakeRemoteError("404 not found")
            doomed = {fileId}
            changed = True
            while changed:
                changed = False
                for item_id, row in list(self.backend.items.items()):
                    if row.get("parents", [None])[0] in doomed and item_id not in doomed:
                        doomed.add(item_id)
                        changed = True
            for item_id in doomed:
                self.backend.raw.pop(item_id, None)
                self.backend.paths.pop(item_id, None)
                self.backend.items.pop(item_id, None)
            return {}

        return _Request(run)


class _Service:
    def __init__(self, backend: "Backend") -> None:
        self.file_api = _BackendFiles(backend)

    def files(self) -> _BackendFiles:
        return self.file_api


class Backend:
    def __init__(self, drive_root: Path) -> None:
        self.drive_root = Path(os.path.abspath(drive_root))
        self.service = _Service(self)
        self.items: dict[str, dict[str, Any]] = {}
        self.raw: dict[str, bytes] = {}
        self.paths: dict[str, Path] = {}
        self.folder_ids: dict[tuple[str, ...], str] = {(): "root"}
        self.next_id = 1
        self.fail_delete: set[str] = set()
        self.upload_headrooms: list[tuple[str, int]] = []

    def new_id(self) -> str:
        value = f"id-{self.next_id}"
        self.next_id += 1
        return value


class FakeCommitter:
    def __init__(
        self,
        *,
        service: _Service | None = None,
        drive_root: Path | str,
        media_upload_factory: Any | None = None,
    ) -> None:
        del media_upload_factory
        assert service is not None
        self.service = service
        self.drive_root = Path(os.path.abspath(drive_root))

    @property
    def backend(self) -> Backend:
        return self.service.file_api.backend

    def split_remote_path(self, path: Path | str) -> tuple[tuple[str, ...], str]:
        relative = Path(os.path.abspath(path)).relative_to(self.drive_root)
        return tuple(relative.parent.parts), relative.name

    def resolve_folder(self, parts: tuple[str, ...], *, create: bool) -> str:
        current: tuple[str, ...] = ()
        parent_id = "root"
        for part in parts:
            current = (*current, part)
            if current not in self.backend.folder_ids:
                if not create:
                    raise FakeRemoteError(f"Drive folder is absent: {'/'.join(parts)}")
                folder_id = self.backend.new_id()
                self.backend.folder_ids[current] = folder_id
                self.backend.items[folder_id] = {
                    "id": folder_id,
                    "name": part,
                    "parents": [parent_id],
                    "mimeType": "application/vnd.google-apps.folder",
                    "trashed": False,
                }
            parent_id = self.backend.folder_ids[current]
        return parent_id

    def _list_children(self, parent_id: str, name: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.backend.items.values()
            if row.get("parents") == [parent_id]
            and row.get("name") == name
            and not row.get("trashed", False)
        ]

    def metadata(self, file_id: str) -> dict[str, Any]:
        if file_id not in self.backend.items:
            raise FakeRemoteError("404 not found")
        return dict(self.backend.items[file_id])

    @staticmethod
    def commit_record(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": "classA_drive_api_commit_v1",
            "remote_file_id": str(metadata["id"]),
            "remote_name": str(metadata["name"]),
            "remote_parent_id": str(metadata["parents"][0]),
            "remote_bytes": int(metadata["size"]),
            "remote_sha256": str(metadata["sha256Checksum"]),
            "remote_verified_unix": 1.0,
        }

    def put(self, path: Path | str, raw: bytes, *, replace: bool) -> dict[str, Any]:
        path = Path(os.path.abspath(path))
        parts, name = self.split_remote_path(path)
        parent_id = self.resolve_folder(parts, create=True)
        matches = self._list_children(parent_id, name)
        if len(matches) > 1:
            raise FakeRemoteError("ambiguous Drive path")
        if matches and not replace:
            existing = self.commit_record(matches[0])
            if (
                existing["remote_bytes"] != len(raw)
                or existing["remote_sha256"] != hashlib.sha256(raw).hexdigest()
            ):
                raise FakeRemoteError("Drive server verification failed")
            return existing
        file_id = matches[0]["id"] if matches else self.backend.new_id()
        row = {
            "id": file_id,
            "name": name,
            "parents": [parent_id],
            "size": str(len(raw)),
            "sha256Checksum": hashlib.sha256(raw).hexdigest(),
            "trashed": False,
        }
        self.backend.items[file_id] = row
        self.backend.raw[file_id] = bytes(raw)
        self.backend.paths[file_id] = path
        return self.commit_record(row)

    def upload_verified(
        self,
        local_path: Path | str,
        remote_path: Path | str,
        *,
        replace: bool,
        required_headroom_bytes: int = 0,
    ) -> dict[str, Any]:
        self.backend.upload_headrooms.append(
            (Path(remote_path).name, int(required_headroom_bytes))
        )
        return self.put(remote_path, Path(local_path).read_bytes(), replace=replace)

    def path_commit_record(self, path: Path | str) -> dict[str, Any]:
        parts, name = self.split_remote_path(path)
        parent_id = self.resolve_folder(parts, create=False)
        matches = self._list_children(parent_id, name)
        if not matches:
            raise FakeRemoteError(f"Drive file is absent: {path}")
        if len(matches) != 1:
            raise FakeRemoteError("ambiguous Drive path")
        return self.commit_record(matches[0])

    def verify_path(
        self, path: Path | str, *, expected_size: int, expected_sha256: str
    ) -> dict[str, Any]:
        record = self.path_commit_record(path)
        if (
            record["remote_bytes"] != expected_size
            or record["remote_sha256"] != expected_sha256
        ):
            raise FakeRemoteError("Drive server verification failed")
        return self.metadata(record["remote_file_id"])

    def verify_commit_record(self, record: dict[str, Any]) -> dict[str, Any]:
        current = self.metadata(str(record["remote_file_id"]))
        if self.commit_record(current) | {"remote_verified_unix": 1.0} != (
            dict(record) | {"remote_verified_unix": 1.0}
        ):
            raise FakeRemoteError("Drive server verification failed")
        return current

    def verify_record_for_path(
        self, record: dict[str, Any], path: Path | str
    ) -> dict[str, Any]:
        current = self.path_commit_record(path)
        keys = (
            "schema",
            "remote_file_id",
            "remote_name",
            "remote_parent_id",
            "remote_bytes",
            "remote_sha256",
        )
        if tuple(current.get(key) for key in keys) != tuple(
            record.get(key) for key in keys
        ):
            raise FakeRemoteError("remote commit record is not bound to intended path")
        return self.metadata(current["remote_file_id"])

    def download_bytes(self, file_id: str) -> bytes:
        if file_id not in self.backend.raw:
            raise FakeRemoteError("404 not found")
        return self.backend.raw[file_id]

    def download_to(self, file_id: str, destination: Path | str, **_: Any) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.download_bytes(file_id))
        return destination

    def delete_verified_if_present(self, record: dict[str, Any]) -> bool:
        try:
            self.verify_commit_record(record)
        except FakeRemoteError as exc:
            if "404" in str(exc) or "not found" in str(exc):
                return False
            raise
        self.service.files().delete(fileId=str(record["remote_file_id"])).execute()
        return True


def fake_drive_module(backend: Backend) -> Any:
    def execute(factory: Any, **_: Any) -> Any:
        try:
            return factory().execute()
        except FakeRemoteError:
            raise
        except Exception as exc:
            raise FakeRemoteError(str(exc)) from exc

    def publish_json(
        committer: FakeCommitter,
        payload: dict[str, Any],
        path: Path | str,
        *,
        replace: bool,
        required_headroom_bytes: int = 0,
    ) -> dict[str, Any]:
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        committer.backend.upload_headrooms.append(
            (Path(path).name, int(required_headroom_bytes))
        )
        return committer.put(path, raw, replace=replace)

    def read_json(committer: FakeCommitter, path: Path | str) -> dict[str, Any]:
        record = committer.path_commit_record(path)
        payload = json.loads(committer.download_bytes(record["remote_file_id"]))
        assert isinstance(payload, dict)
        return payload

    return SimpleNamespace(
        DEFAULT_DRIVE_ROOT=backend.drive_root,
        REMOTE_COMMIT_SCHEMA="classA_drive_api_commit_v1",
        DriveRemoteCommitter=FakeCommitter,
        RemoteCommitError=FakeRemoteError,
        sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        build_drive_service=lambda: backend.service,
        _execute_with_retries=execute,
        publish_json=publish_json,
        read_remote_json=read_json,
    )


def _minimal_runner(backend: Backend, operational: Any) -> Any:
    def checkpoint_paths(*, archive: Path, run_id: str) -> tuple[Path, Path]:
        directory = archive.parent / "_cycle_checkpoints"
        return directory, directory / f"{run_id}.latest.json"

    runner = SimpleNamespace(
        BUNDLE="01_p1_chern_dynamics",
        DEFAULT_DRIVE_ROOT=backend.drive_root,
        DriveRemoteCommitter=FakeCommitter,
        RemoteCommitError=FakeRemoteError,
        publish_json=operational.publish_json,
        read_remote_json=operational.read_remote_json,
        P1_REMOTE_REQUIRED_HEADROOM_BYTES=1_342_177_280,
        _ShardLease=object,
        _verify_existing=lambda *_args, **_kwargs: None,
        _write_cycle_checkpoint=lambda **kwargs: kwargs,
        _load_cycle_checkpoint=lambda **_: None,
        _cleanup_cycle_checkpoint=lambda **_: None,
        _purge_checkpoint_orphans=lambda **_: None,
        _a100_preflight=lambda **_: {"safe": True},
        _checkpoint_paths=checkpoint_paths,
        sha256_json=lambda payload: hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
    runner._remote_committer = lambda _: FakeCommitter(
        service=backend.service, drive_root=backend.drive_root
    )
    return runner


def test_wrapper_preserves_cycle7_identity_and_substitutes_root_transport() -> None:
    script = f"""
import importlib.util, json, pathlib
p=pathlib.Path({str(BUNDLE / 'run_bundle.py')!r})
s=importlib.util.spec_from_file_location('p1_wrapper_identity_probe', p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
c=m._runner.load_config(m.BUNDLE_ROOT)
case=next(x for x in m._runner.expand_cases(c) if x['case_id']=='P1_CHERN_L64_nsh-1')
run_id=m._runner._archive_paths(bundle_root=m.BUNDLE_ROOT, config=c, case=case,
 shard_index=0, drive_root=pathlib.Path('/content/drive/MyDrive'), mode='production')[2]
print(json.dumps({{'run_id':run_id,'drive_module':m._runner.DriveRemoteCommitter.__module__,
 'helper':str(pathlib.Path(m._operational_drive.__file__).resolve())}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, text=True, capture_output=True
    )
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "01_p1_chern_dynamics_f85db6086b7bef4e"
    assert payload["drive_module"] == "_p1_root_operational_drive_remote_commit"
    assert payload["helper"] == str((DEPLOYMENT / "drive_remote_commit.py").resolve())
    expected = {
        "p1_chern_runner.py": "f75612f52dc58018a2fda767fe7e343367b4f899074cc216baf2e1bb2b986a0d",
        "drive_remote_commit.py": "1fd83bfe8d9487276e2b8793d4d2cdb8d1f20ff744911dfb617b09f3b439fe9d",
        "source_manifest.json": "9e335cb08742916cce22f57c141af80d2e539536f5d8cd48ac3510286fe739cf",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((BUNDLE / "src" / name).read_bytes()).hexdigest() == digest


def test_scoped_record_refuses_cross_parent_delete(tmp_path: Path) -> None:
    archive = tmp_path / "drive/production/run.tar.gz"
    Scoped = HARDENING.P1DriveRemoteCommitter.build(ROOT_DRIVE)
    deleted: list[str] = []
    committer = Scoped(
        service=SimpleNamespace(
            files=lambda: SimpleNamespace(
                delete=lambda **kwargs: _Request(
                    lambda: deleted.append(kwargs["fileId"])
                )
            )
        ),
        drive_root=tmp_path / "drive",
        scope_path_override=archive,
    )
    committer.verify_path = lambda *_args, **_kwargs: {
        "id": "right-id",
        "name": archive.name,
        "parents": ["right-parent"],
        "size": "4",
        "sha256Checksum": "a" * 64,
        "trashed": False,
    }
    wrong = {
        "schema": ROOT_DRIVE.REMOTE_COMMIT_SCHEMA,
        "remote_file_id": "wrong-id",
        "remote_name": archive.name,
        "remote_parent_id": "wrong-parent",
        "remote_bytes": 4,
        "remote_sha256": "a" * 64,
    }
    with pytest.raises(ROOT_DRIVE.RemoteCommitError, match="not bound"):
        committer.try_delete_record(wrong)
    assert deleted == []


@pytest.mark.parametrize("fresh", [True, False])
def test_api_lease_blocks_fresh_legacy_writer_and_cleans_only_stale_exact_folder(
    tmp_path: Path, fresh: bool
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    HARDENING._OPERATIONAL_DRIVE = operational
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"
    run_id = "run"
    directory, _ = runner._checkpoint_paths(archive=archive, run_id=run_id)
    committer = runner._remote_committer(archive)
    parent = committer.resolve_folder(
        tuple(directory.relative_to(backend.drive_root).parts), create=True
    )
    folder_id = backend.new_id()
    backend.items[folder_id] = {
        "id": folder_id,
        "name": f"{run_id}.lease",
        "parents": [parent],
        "mimeType": "application/vnd.google-apps.folder",
        "trashed": False,
    }
    payload = {
        "run_id": run_id,
        "owner_token": "old-owner",
        "updated_unix": time.time() - (1 if fresh else 2_000),
    }
    raw = json.dumps(payload).encode()
    child_id = backend.new_id()
    backend.items[child_id] = {
        "id": child_id,
        "name": "lease.json",
        "parents": [folder_id],
        "size": str(len(raw)),
        "sha256Checksum": hashlib.sha256(raw).hexdigest(),
        "trashed": False,
    }
    backend.raw[child_id] = raw
    lease = HARDENING._ApiShardLease(runner=runner, archive=archive, run_id=run_id)
    if fresh:
        with pytest.raises(RuntimeError, match="pre-hotfix P1 child"):
            lease._guard_legacy_drivefs_lease()
        assert folder_id in backend.items
    else:
        lease._guard_legacy_drivefs_lease()
        assert folder_id not in backend.items
        assert child_id not in backend.items


def test_api_lease_elects_one_simultaneous_absent_claim_and_cleans_loser(
    tmp_path: Path,
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    HARDENING._OPERATIONAL_DRIVE = operational
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"
    winner = HARDENING._ApiShardLease(runner=runner, archive=archive, run_id="run")
    loser = HARDENING._ApiShardLease(runner=runner, archive=archive, run_id="run")
    winner.token = "1" * 32
    loser.token = "2" * 32
    committer = winner.committer
    parent_id = committer.resolve_folder(
        tuple(winner.directory.relative_to(backend.drive_root).parts), create=True
    )

    for lease in (loser, winner):
        raw = (json.dumps(lease._payload(), sort_keys=True) + "\n").encode()
        file_id = backend.new_id()
        backend.items[file_id] = {
            "id": file_id,
            "name": winner.path.name,
            "parents": [parent_id],
            "size": str(len(raw)),
            "sha256Checksum": hashlib.sha256(raw).hexdigest(),
            "trashed": False,
        }
        backend.raw[file_id] = raw

    with pytest.raises(RuntimeError, match="ownership was lost"):
        loser.assert_owned()
    winner.assert_owned()
    remaining = committer._list_children(parent_id, winner.path.name)
    assert len(remaining) == 1
    payload = json.loads(committer.download_bytes(str(remaining[0]["id"])))
    assert payload["owner_token"] == winner.token


def test_two_stale_reclaimers_delete_then_elect_without_blind_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    HARDENING._OPERATIONAL_DRIVE = operational
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"
    first = HARDENING._ApiShardLease(runner=runner, archive=archive, run_id="run")
    second = HARDENING._ApiShardLease(runner=runner, archive=archive, run_id="run")
    first.token = "1" * 32
    second.token = "2" * 32
    old = dict(first._payload())
    old["owner_token"] = "f" * 32
    old["updated_unix"] = time.time() - 2_000
    first.committer.put(
        first.path,
        (json.dumps(old, sort_keys=True) + "\n").encode(),
        replace=False,
    )

    stale_delete_barrier = threading.Barrier(2)
    absent_claim_barrier = threading.Barrier(2)
    finish_barrier = threading.Barrier(2)
    base_publish = operational.publish_json

    def synchronized_publish(
        committer: FakeCommitter,
        payload: dict[str, Any],
        path: Path | str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if payload.get("owner_token") in {first.token, second.token}:
            absent_claim_barrier.wait(timeout=5)
        return base_publish(committer, payload, path, **kwargs)

    operational.publish_json = synchronized_publish
    monkeypatch.setattr(HARDENING, "P1_API_LEASE_SETTLE_SECONDS", 0.01)
    for lease in (first, second):
        original_delete = lease._delete_lease_item

        def synchronized_delete(
            metadata: dict[str, Any],
            payload: dict[str, Any],
            *,
            original_delete: Any = original_delete,
        ) -> None:
            if payload.get("owner_token") == old["owner_token"]:
                stale_delete_barrier.wait(timeout=5)
            original_delete(metadata, payload)

        lease._delete_lease_item = synchronized_delete

    outcomes: list[tuple[str, bool, str]] = []

    def reclaim(label: str, lease: Any) -> None:
        entered = False
        detail = ""
        try:
            lease.__enter__()
            entered = True
        except Exception as exc:  # Expected for the deterministic loser.
            detail = str(exc)
        outcomes.append((label, entered, detail))
        finish_barrier.wait(timeout=5)
        if entered:
            lease.__exit__(None, None, None)

    threads = [
        threading.Thread(target=reclaim, args=("first", first)),
        threading.Thread(target=reclaim, args=("second", second)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sum(entered for _, entered, _ in outcomes) == 1
    loser = next(detail for _, entered, detail in outcomes if not entered)
    assert "ownership was lost" in loser or "won" in loser or "absent" in loser


def test_stale_delete_refuses_same_token_heartbeat_refresh(tmp_path: Path) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    HARDENING._OPERATIONAL_DRIVE = operational
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"
    lease = HARDENING._ApiShardLease(runner=runner, archive=archive, run_id="run")
    stale = lease._payload()
    stale["owner_token"] = "a" * 32
    stale["updated_unix"] = time.time() - 2_000
    lease.committer.put(
        lease.path,
        (json.dumps(stale, sort_keys=True) + "\n").encode(),
        replace=False,
    )
    metadata, observed = lease._validated_lease_item(
        lease._lease_items(create_parent=False)[0]
    )
    refreshed = dict(stale)
    refreshed["updated_unix"] = time.time()
    lease.committer.put(
        lease.path,
        (json.dumps(refreshed, sort_keys=True) + "\n").encode(),
        replace=True,
    )
    with pytest.raises(FakeRemoteError, match="changed before exact deletion"):
        lease._delete_lease_item(metadata, observed)
    assert lease._read_optional(lease.path)["updated_unix"] == refreshed["updated_unix"]


def test_lease_loss_after_generation_upload_prevents_pointer_publication(
    tmp_path: Path,
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)

    class LosingLease:
        def __init__(self) -> None:
            self.assertions = 0

        def assert_owned(self) -> None:
            self.assertions += 1
            if self.assertions == 2:
                raise RuntimeError("lease ownership lost after upload")

    lease = LosingLease()
    Scoped = HARDENING.P1DriveRemoteCommitter.build(
        operational, lease_getter=lambda: lease
    )
    committer = Scoped(
        service=backend.service,
        drive_root=backend.drive_root,
        scope_path_override=backend.drive_root / "out/run.tar.gz",
    )
    local = tmp_path / "generation.npz"
    local.write_bytes(b"large completed generation")
    generation = backend.drive_root / "out/_cycle_checkpoints/run.cycle_7.npz"
    pointer = backend.drive_root / "out/_cycle_checkpoints/run.latest.json"

    def frozen_generation_then_pointer() -> None:
        committer.upload_verified(local, generation, replace=False)
        operational.publish_json(
            committer,
            {"checkpoint": generation.name},
            pointer,
            replace=True,
        )

    with pytest.raises(RuntimeError, match="lost after upload"):
        frozen_generation_then_pointer()
    assert committer.path_commit_record(generation)
    with pytest.raises(FakeRemoteError, match="absent"):
        committer.path_commit_record(pointer)


def test_two_absent_preflights_share_slot_lease_and_publish_one_safe_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"
    preflight_path = backend.drive_root / "out/a100_preflight.json"
    case = {"case_id": "P1_CHERN_L64_nsh-1", "model": {"Nx": 64, "nshell": 1}}
    config = {"production_output_collection": "out"}
    calculations: list[int] = []

    runner.expand_cases = lambda _: [case]
    runner._archive_paths = lambda **_: (tmp_path / "scratch", archive, "run", {})
    runner._preflight_receipt_path = lambda **_: preflight_path

    def original_preflight(**_: Any) -> dict[str, Any]:
        # Frozen `_a100_preflight` enters the decorated `_run_case` lease and
        # then publishes its receipt after `_run_case` returns.  The operational
        # outer lease must make this nested lease reentrant and remain active.
        with runner._ShardLease(archive=archive, run_id="run"):
            calculations.append(1)
        payload = {"safe": True, "created_unix": time.time(), "run_id": "run"}
        runner.publish_json(
            runner._remote_committer(preflight_path),
            payload,
            preflight_path,
            replace=False,
            required_headroom_bytes=runner.P1_REMOTE_REQUIRED_HEADROOM_BYTES,
        )
        return payload

    runner._a100_preflight = original_preflight
    runner._require_safe_preflight = lambda **_: operational.read_remote_json(
        runner._remote_committer(preflight_path), preflight_path
    )
    runner._archive = lambda *_: {}
    runner._root_manifest_from_archive = lambda _: {}

    claim_barrier = threading.Barrier(2)
    base_publish = operational.publish_json

    def synchronized_publish(
        committer: FakeCommitter,
        payload: dict[str, Any],
        path: Path | str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if payload.get("schema") == HARDENING.P1_API_LEASE_SCHEMA:
            claim_barrier.wait(timeout=5)
        return base_publish(committer, payload, path, **kwargs)

    operational.publish_json = synchronized_publish
    monkeypatch.setattr(HARDENING, "P1_API_LEASE_SETTLE_SECONDS", 0.01)
    HARDENING.apply_p1_hardening(runner, operational_drive=operational)
    outcomes: list[tuple[bool, str]] = []

    def qualify() -> None:
        try:
            result = runner._a100_preflight(
                bundle_root=tmp_path,
                config=config,
                drive_root=backend.drive_root,
                max_runtime_seconds=None,
            )
            outcomes.append((bool(result["safe"]), ""))
        except Exception as exc:  # The deterministic lease loser must stop.
            outcomes.append((False, str(exc)))

    threads = [threading.Thread(target=qualify), threading.Thread(target=qualify)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sum(success for success, _ in outcomes) == 1
    assert len(calculations) == 1
    committer = runner._remote_committer(preflight_path)
    parts, name = committer.split_remote_path(preflight_path)
    parent_id = committer.resolve_folder(parts, create=False)
    assert len(committer._list_children(parent_id, name)) == 1

    # A retry that observed absence before waiting must recheck after acquiring
    # the slot and reuse the now-safe receipt without a second calculation.
    operational.publish_json = base_publish
    reused = runner._a100_preflight(
        bundle_root=tmp_path,
        config=config,
        drive_root=backend.drive_root,
        max_runtime_seconds=None,
    )
    assert reused["safe"] is True
    assert len(calculations) == 1


def test_checkpoint_status_absent_is_success_and_present_is_exact_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    HARDENING._OPERATIONAL_DRIVE = operational
    committer = FakeCommitter(service=backend.service, drive_root=backend.drive_root)
    archive = backend.drive_root / "out/run.tar.gz"
    directory = archive.parent / "_cycle_checkpoints"
    pointer = directory / "run.latest.json"
    identity = {"total_cycles": 64, "fixed": True}

    runner = SimpleNamespace(
        BUNDLE="01_p1_chern_dynamics",
        P1_CHECKPOINT_POINTER_SCHEMA="pointer-v2",
        P1_CHECKPOINT_SCHEMA="checkpoint-v2",
        load_config=lambda _: {},
        expand_cases=lambda _: [{"case_id": "P1_CHERN_L64_nsh-1"}],
        case_index=lambda rows: {row["case_id"]: row for row in rows},
        _archive_paths=lambda **_: (tmp_path / "scratch", archive, "run", {}),
        _checkpoint_identity=lambda **_: identity,
        global_sample_indices=lambda *_: [0],
        _checkpoint_paths=lambda **_: (directory, pointer),
        _remote_committer=lambda _: committer,
        _verify_existing=lambda *_args, **_kwargs: None,
        sha256_json=lambda payload: hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    absent = HARDENING.checkpoint_status(
        runner,
        bundle_root=tmp_path,
        drive_root=backend.drive_root,
        mode="production",
        case_id="P1_CHERN_L64_nsh-1",
        shard_index=0,
    )
    assert absent["exists"] is False

    raw_generation = b"verified checkpoint"
    generation = directory / "run.cycle_000007.abcdef.npz"
    generation_record = committer.put(generation, raw_generation, replace=False)
    pointer_payload = {
        "schema": "pointer-v2",
        "checkpoint_schema": "checkpoint-v2",
        "run_id": "run",
        "checkpoint": generation.name,
        "checkpoint_sha256": generation_record["remote_sha256"],
        "checkpoint_bytes": generation_record["remote_bytes"],
        "checkpoint_identity_sha256": runner.sha256_json(identity),
        "completed_cycle": 7,
        "total_cycles": 64,
        "checkpoint_remote_commit": generation_record,
    }
    committer.put(
        pointer,
        (json.dumps(pointer_payload, sort_keys=True) + "\n").encode(),
        replace=False,
    )
    present = HARDENING.checkpoint_status(
        runner,
        bundle_root=tmp_path,
        drive_root=backend.drive_root,
        mode="production",
        case_id="P1_CHERN_L64_nsh-1",
        shard_index=0,
    )
    assert present["exists"] is True
    assert present["completed_cycle"] == 7
    assert present["checkpoint_remote_commit"] == generation_record

    rc = HARDENING.run_hardened_main(
        runner,
        [
            "--drive-root",
            str(backend.drive_root),
            "--case-id",
            "P1_CHERN_L64_nsh-1",
            "--shard-index",
            "0",
            "--checkpoint-status",
        ],
        bundle_root=tmp_path,
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["completed_cycle"] == 7


def test_repair_orphan_cli_is_supported_and_returns_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "drive/out/run.tar.gz"
    receipt = {
        "archive_sha256": "a" * 64,
        "recovered_after_interrupted_remote_receipt_commit": True,
    }
    runner = SimpleNamespace(
        load_config=lambda _: {},
        expand_cases=lambda _: [{"case_id": "P1_CHERN_L64_nsh-1"}],
        case_index=lambda rows: {row["case_id"]: row for row in rows},
        _archive_paths=lambda **_: (tmp_path / "scratch", archive, "run", {}),
        _verify_existing=lambda *_args, **_kwargs: receipt,
    )
    rc = HARDENING.run_hardened_main(
        runner,
        [
            "--drive-root",
            str(tmp_path / "drive"),
            "--case-id",
            "P1_CHERN_L64_nsh-1",
            "--repair-orphan",
        ],
        bundle_root=tmp_path,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "verified_or_repaired"
    assert payload["repaired"] is True
    assert payload["server_verified"] is True


def test_fail_closed_preflight_only_qualifies_on_explicit_absence(tmp_path: Path) -> None:
    class StatusCommitter:
        mode = "transient"

        def path_commit_record(self, _path: Path) -> dict[str, Any]:
            if self.mode == "absent":
                raise FakeRemoteError("Drive file is absent")
            if self.mode == "transient":
                raise FakeRemoteError("Google Drive API request failed: socket timeout")
            return {"remote_file_id": "preflight"}

        def verify_record_for_path(self, record: dict[str, Any], _path: Path) -> dict[str, Any]:
            return record

    committer = StatusCommitter()
    qualification_calls: list[bool] = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=tmp_path)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--a100-preflight", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-cases-json", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float)
    runner = SimpleNamespace(
        build_parser=lambda: parser,
        load_config=lambda _: {},
        _preflight_receipt_path=lambda **_: tmp_path / "drive/preflight.json",
        _server_commit_required=lambda _: True,
        _remote_committer=lambda _: committer,
        RemoteCommitError=FakeRemoteError,
        _require_safe_preflight=lambda **_: (_ for _ in ()).throw(
            RuntimeError("corrupt preflight")
        ),
        _a100_preflight=lambda **_: qualification_calls.append(True)
        or {"safe": True},
        _print_session_checkpoint=lambda _: None,
    )
    values = ["--drive-root", str(tmp_path / "drive"), "--a100-preflight"]
    with pytest.raises(FakeRemoteError, match="socket timeout"):
        HARDENING._run_fail_closed_a100_preflight(runner, values)
    assert qualification_calls == []
    committer.mode = "present"
    with pytest.raises(RuntimeError, match="corrupt preflight"):
        HARDENING._run_fail_closed_a100_preflight(runner, values)
    assert qualification_calls == []
    committer.mode = "absent"
    assert HARDENING._run_fail_closed_a100_preflight(runner, values) == 0
    assert qualification_calls == [True]


def test_sidecar_avoids_archive_download_and_small_json_uses_zero_reserve(
    tmp_path: Path
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    runner = _minimal_runner(backend, operational)
    root_reads: list[Path] = []
    manifest = {"status": "complete_local", "run_config": {"x": 1}}
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "manifest.json").write_text(json.dumps(manifest))
    archive = backend.drive_root / "out/run.tar.gz"

    def original_archive(_scratch: Path, selected: Path, run_id: str) -> dict[str, Any]:
        committer = FakeCommitter(service=backend.service, drive_root=backend.drive_root)
        archive_record = committer.put(selected, b"large archive", replace=False)
        receipt = {
            "schema_version": 2,
            "run_id": run_id,
            "archive": selected.name,
            "archive_sha256": archive_record["remote_sha256"],
            "archive_bytes": archive_record["remote_bytes"],
            "archive_remote_commit": archive_record,
        }
        operational.publish_json(
            committer,
            receipt,
            selected.with_suffix(selected.suffix + ".receipt.json"),
            replace=False,
            required_headroom_bytes=runner.P1_REMOTE_REQUIRED_HEADROOM_BYTES,
        )
        return receipt

    runner._archive = original_archive
    runner._root_manifest_from_archive = lambda path: root_reads.append(Path(path)) or manifest
    HARDENING.apply_p1_hardening(runner, operational_drive=operational)
    receipt = runner._archive(scratch, archive, "run")
    assert receipt["archive"] == archive.name
    assert runner._root_manifest_from_archive(archive) == manifest
    assert root_reads == []
    sidecar = archive.with_suffix(archive.suffix + ".manifest.json")
    committer = runner._remote_committer(archive)
    assert committer.path_commit_record(sidecar)["remote_name"] == sidecar.name
    assert (sidecar.name, 0) in backend.upload_headrooms

    recorded: list[int] = []
    original_publish = operational.publish_json

    def observe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        recorded.append(int(kwargs["required_headroom_bytes"]))
        return original_publish(*args, **kwargs)

    operational.publish_json = observe
    runner._P1_OPERATIONAL_PUBLISH_JSON(
        committer,
        {"pointer": True},
        archive.parent / "pointer.json",
        replace=False,
        required_headroom_bytes=runner.P1_REMOTE_REQUIRED_HEADROOM_BYTES,
    )
    assert recorded == [0]


def test_sidecar_recovers_identical_publication_race_and_converges_one_name(
    tmp_path: Path,
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"
    manifest = {"status": "complete_local", "run_config": {"x": 1}}
    committer = runner._remote_committer(archive)
    archive_record = committer.put(archive, b"large archive", replace=False)
    operational.publish_json(
        committer,
        {
            "run_id": "run",
            "archive": archive.name,
            "archive_sha256": archive_record["remote_sha256"],
            "archive_bytes": archive_record["remote_bytes"],
            "archive_remote_commit": archive_record,
        },
        archive.with_suffix(archive.suffix + ".receipt.json"),
        replace=False,
    )
    runner._archive = lambda *_: {}
    runner._root_manifest_from_archive = lambda _: dict(manifest)
    HARDENING.apply_p1_hardening(runner, operational_drive=operational)
    base_publish = operational.publish_json

    def racing_publish(
        selected: FakeCommitter,
        payload: dict[str, Any],
        path: Path | str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not Path(path).name.endswith(".manifest.json"):
            return base_publish(selected, payload, path, **kwargs)
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        parts, name = selected.split_remote_path(path)
        parent_id = selected.resolve_folder(parts, create=True)
        for _ in range(2):
            file_id = backend.new_id()
            backend.items[file_id] = {
                "id": file_id,
                "name": name,
                "parents": [parent_id],
                "size": str(len(raw)),
                "sha256Checksum": hashlib.sha256(raw).hexdigest(),
                "trashed": False,
            }
            backend.raw[file_id] = raw
        raise FakeRemoteError(
            f"final Drive publication produced 2 files named {name!r}"
        )

    operational.publish_json = racing_publish
    assert runner._root_manifest_from_archive(archive) == manifest
    sidecar = archive.with_suffix(archive.suffix + ".manifest.json")
    parts, name = committer.split_remote_path(sidecar)
    parent_id = committer.resolve_folder(parts, create=False)
    assert len(committer._list_children(parent_id, name)) == 1


def test_sidecar_fails_closed_and_preserves_differing_valid_duplicates(
    tmp_path: Path,
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"
    manifest = {"status": "complete_local", "run_config": {"x": 1}}
    committer = runner._remote_committer(archive)
    archive_record = committer.put(archive, b"large archive", replace=False)
    operational.publish_json(
        committer,
        {
            "run_id": "run",
            "archive": archive.name,
            "archive_sha256": archive_record["remote_sha256"],
            "archive_bytes": archive_record["remote_bytes"],
            "archive_remote_commit": archive_record,
        },
        archive.with_suffix(archive.suffix + ".receipt.json"),
        replace=False,
    )
    runner._archive = lambda *_: {}
    runner._root_manifest_from_archive = lambda _: dict(manifest)
    HARDENING.apply_p1_hardening(runner, operational_drive=operational)
    assert runner._root_manifest_from_archive(archive) == manifest

    sidecar = archive.with_suffix(archive.suffix + ".manifest.json")
    first_record = committer.path_commit_record(sidecar)
    first_raw = committer.download_bytes(first_record["remote_file_id"])
    second_payload = json.loads(first_raw.decode("utf-8"))
    second_payload["manifest"] = {
        "status": "complete_local",
        "run_config": {"x": 2},
    }
    second_payload["manifest_sha256"] = runner.sha256_json(
        second_payload["manifest"]
    )
    second_raw = (
        json.dumps(second_payload, indent=2, sort_keys=True) + "\n"
    ).encode()
    second_id = backend.new_id()
    backend.items[second_id] = {
        "id": second_id,
        "name": sidecar.name,
        "parents": [first_record["remote_parent_id"]],
        "size": str(len(second_raw)),
        "sha256Checksum": hashlib.sha256(second_raw).hexdigest(),
        "trashed": False,
    }
    backend.raw[second_id] = second_raw

    with pytest.raises(RuntimeError, match="differing same-name duplicates"):
        runner._root_manifest_from_archive(archive)
    assert len(
        committer._list_children(first_record["remote_parent_id"], sidecar.name)
    ) == 2


def test_server_orphan_archive_repairs_receipt_and_sidecar_without_recompute(
    tmp_path: Path,
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"
    committer = runner._remote_committer(archive)
    archive_record = committer.put(archive, b"completed scientific archive", replace=False)
    manifest = {"status": "complete_local", "source_hashes": {"engine": "same"}}

    def original_verify(
        selected: Path,
        *,
        expected_run_id: str | None = None,
        expected_run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del expected_run_config
        selected = Path(selected)
        if str(selected).startswith(str(backend.drive_root)):
            receipt_path = selected.with_suffix(selected.suffix + ".receipt.json")
            try:
                return operational.read_remote_json(committer, receipt_path)
            except FakeRemoteError as exc:
                if "absent" not in str(exc):
                    raise
                committer.path_commit_record(selected)
                raise RuntimeError(
                    f"server archive exists without its remotely verified receipt: {selected}"
                ) from exc
        return {
            "schema_version": 1,
            "run_id": expected_run_id,
            "archive": selected.name,
            "archive_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
            "archive_bytes": selected.stat().st_size,
        }

    runner._verify_existing = original_verify
    runner._archive = lambda *_: {}
    runner._root_manifest_from_archive = lambda _: dict(manifest)
    HARDENING.apply_p1_hardening(runner, operational_drive=operational)
    repaired = runner._verify_existing(
        archive, expected_run_id="run", expected_run_config={"identity": "same"}
    )
    assert repaired is not None
    assert repaired["recovered_after_interrupted_remote_receipt_commit"] is True
    assert repaired["archive_remote_commit"] == archive_record
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    sidecar_path = archive.with_suffix(archive.suffix + ".manifest.json")
    assert committer.path_commit_record(receipt_path)
    assert committer.path_commit_record(sidecar_path)
    assert (receipt_path.name, 0) in backend.upload_headrooms
    assert (sidecar_path.name, 0) in backend.upload_headrooms


def test_checkpoint_cleanup_requires_final_archive_and_retries_delete_failure(
    tmp_path: Path
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    runner = _minimal_runner(backend, operational)
    runner._archive = lambda *_: {}
    runner._root_manifest_from_archive = lambda _: {}
    HARDENING.apply_p1_hardening(runner, operational_drive=operational)
    archive = backend.drive_root / "out/run.tar.gz"
    directory, pointer = runner._checkpoint_paths(archive=archive, run_id="run")
    committer = runner._remote_committer(archive)
    generation = directory / "run.cycle_000007.hash.npz"
    generation_record = committer.put(generation, b"checkpoint", replace=False)
    operational.publish_json(
        committer,
        {"checkpoint_remote_commit": generation_record},
        pointer,
        replace=False,
    )
    with pytest.raises(FakeRemoteError, match="absent"):
        runner._cleanup_cycle_checkpoint(archive=archive, run_id="run")
    assert committer.path_commit_record(pointer)
    assert committer.path_commit_record(generation)

    final_record = committer.put(archive, b"final archive", replace=False)
    operational.publish_json(
        committer,
        {
            "run_id": "run",
            "archive": archive.name,
            "archive_sha256": final_record["remote_sha256"],
            "archive_bytes": final_record["remote_bytes"],
            "archive_remote_commit": final_record,
        },
        archive.with_suffix(archive.suffix + ".receipt.json"),
        replace=False,
    )
    backend.fail_delete.add(generation_record["remote_file_id"])
    with pytest.warns(RuntimeWarning, match="cleanup remains pending"):
        runner._cleanup_cycle_checkpoint(archive=archive, run_id="run")
    assert committer.path_commit_record(pointer)
    runner._cleanup_cycle_checkpoint(archive=archive, run_id="run")
    with pytest.raises(FakeRemoteError, match="absent"):
        committer.path_commit_record(pointer)
    with pytest.raises(FakeRemoteError, match="absent"):
        committer.path_commit_record(generation)


def test_generation_uploaded_before_pointer_crash_is_reclaimed_before_gpu_resume(
    tmp_path: Path,
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    runner = _minimal_runner(backend, operational)
    runner._archive = lambda *_: {}
    runner._root_manifest_from_archive = lambda _: {}
    HARDENING.apply_p1_hardening(runner, operational_drive=operational)
    archive = backend.drive_root / "out/run.tar.gz"
    directory, _ = runner._checkpoint_paths(archive=archive, run_id="run")
    committer = runner._remote_committer(archive)
    orphan = directory / "run.cycle_000008.orphan.npz"
    committer.put(orphan, b"uploaded but pointer never published", replace=False)
    assert runner._load_cycle_checkpoint(archive=archive, run_id="run") is None
    with pytest.raises(FakeRemoteError, match="absent"):
        committer.path_commit_record(orphan)


def test_remote_status_reconciles_crash_after_final_receipt_before_cleanup(
    tmp_path: Path,
) -> None:
    backend = Backend(tmp_path / "drive")
    operational = fake_drive_module(backend)
    runner = _minimal_runner(backend, operational)
    archive = backend.drive_root / "out/run.tar.gz"

    def original_verify(
        selected: Path, *, expected_run_id: str | None = None, **_: Any
    ) -> dict[str, Any] | None:
        committer = runner._remote_committer(selected)
        try:
            receipt = operational.read_remote_json(
                committer, selected.with_suffix(selected.suffix + ".receipt.json")
            )
        except FakeRemoteError as exc:
            if "absent" in str(exc):
                return None
            raise
        record = receipt["archive_remote_commit"]
        committer.verify_record_for_path(record, selected)
        assert receipt["run_id"] == expected_run_id
        return receipt

    runner._verify_existing = original_verify
    runner._archive = lambda *_: {}
    runner._root_manifest_from_archive = lambda _: {}
    HARDENING.apply_p1_hardening(runner, operational_drive=operational)
    committer = runner._remote_committer(archive)
    directory, pointer = runner._checkpoint_paths(archive=archive, run_id="run")
    generation = directory / "run.cycle_000064.hash.npz"
    generation_record = committer.put(generation, b"final checkpoint", replace=False)
    operational.publish_json(
        committer,
        {"checkpoint_remote_commit": generation_record},
        pointer,
        replace=False,
    )
    archive_record = committer.put(archive, b"final archive", replace=False)
    final_receipt = {
        "run_id": "run",
        "archive": archive.name,
        "archive_sha256": archive_record["remote_sha256"],
        "archive_bytes": archive_record["remote_bytes"],
        "archive_remote_commit": archive_record,
    }
    operational.publish_json(
        committer,
        final_receipt,
        archive.with_suffix(archive.suffix + ".receipt.json"),
        replace=False,
    )
    backend.fail_delete.add(generation_record["remote_file_id"])
    with pytest.warns(RuntimeWarning, match="cleanup remains pending"):
        assert runner._verify_existing(
            archive, expected_run_id="run", expected_run_config={}
        ) == final_receipt
    assert committer.path_commit_record(pointer)
    assert runner._verify_existing(
        archive, expected_run_id="run", expected_run_config={}
    ) == final_receipt
    with pytest.raises(FakeRemoteError, match="absent"):
        committer.path_commit_record(pointer)
    with pytest.raises(FakeRemoteError, match="absent"):
        committer.path_commit_record(generation)
