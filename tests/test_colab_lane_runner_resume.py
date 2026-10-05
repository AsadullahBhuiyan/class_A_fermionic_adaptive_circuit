from __future__ import annotations

import importlib.util
import errno
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = (
    REPO
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_new_designs"
)
RUNNER_PATH = CAMPAIGN / "colab_bundle_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("tested_colab_bundle_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _archive(root: Path, name: str, manifest: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.tar.gz"
    raw = json.dumps(manifest).encode("utf-8")
    with tarfile.open(path, "w:gz") as handle:
        member = tarfile.TarInfo("manifest.json")
        member.size = len(raw)
        handle.addfile(member, io.BytesIO(raw))
    receipt = {
        "schema_version": 1,
        "run_id": name,
        "archive": path.name,
        "archive_sha256": RUNNER.sha256_file(path),
        "archive_bytes": path.stat().st_size,
    }
    Path(str(path) + ".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _case(samples: int = 25) -> dict:
    return {
        "case_id": "CASE",
        "model": {"Nx": 4, "Ny": 6},
        "run": {"samples": samples, "cycles": 12},
    }


def _manifest(*, audit: str, samples: int, engine: str = "engine") -> dict:
    case = _case(samples)
    return {
        "bundle": "bundle",
        "status": "complete_local",
        "case_id": "CASE",
        "shard_index": 0,
        "root_seed": 7,
        "shard_generator_seed": RUNNER._shard_seed(7, "CASE", 0),
        "global_sample_indices": [0, 1, 2, 3, 4],
        "audit_sha256": audit,
        "run_config": {
            "case": case,
            "audit_sha256": audit,
            "canonical_engine_sha256": engine,
        },
    }


def test_current_archive_scan_and_exact_match(tmp_path: Path) -> None:
    audit = "a" * 64
    _archive(tmp_path / "bundle", "run", _manifest(audit=audit, samples=25))
    logger = RUNNER.SessionLogger(tmp_path / "session.log")
    try:
        rows = RUNNER._scan_archive_directory(
            tmp_path / "bundle", label="test", logger=logger
        )
    finally:
        logger.close()
    matched, mismatches = RUNNER._find_current_archive(
        rows,
        bundle="bundle",
        case=_case(25),
        shard_index=0,
        config={"root_seed": 7, "audit_sha256": audit},
        engine_hash="engine",
    )
    assert matched is not None
    assert mismatches == []


def test_server_committed_current_scan_ignores_drivefs_archive_without_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "08_h1_endpoint_packet"
    _archive(root, "remote", _manifest(audit="a" * 64, samples=25))
    Path(str(root / "remote.tar.gz") + ".receipt.json").unlink()
    logger = RUNNER.SessionLogger(tmp_path / "session.log", display=io.StringIO())
    try:
        rows = RUNNER._current_rows_for_bundle(
            root,
            bundle=RUNNER.H1_ENDPOINT_BUNDLE,
            label="H1 current",
            logger=logger,
        )
    finally:
        logger.close()
    assert rows == []
    assert "Drive API is authoritative" in (tmp_path / "session.log").read_text()


def test_non_server_bundle_still_fails_on_incomplete_drivefs_pair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "02_wall_cft_windows"
    _archive(root, "local", _manifest(audit="a" * 64, samples=25))
    Path(str(root / "local.tar.gz") + ".receipt.json").unlink()
    logger = RUNNER.SessionLogger(tmp_path / "session.log", display=io.StringIO())
    try:
        with pytest.raises(RuntimeError, match="incomplete archive/receipt pair"):
            RUNNER._current_rows_for_bundle(
                root,
                bundle="02_wall_cft_windows",
                label="local current",
                logger=logger,
            )
    finally:
        logger.close()


def test_remote_status_row_replaces_stale_slot_and_is_identity_checked() -> None:
    audit = "c" * 64
    stale_manifest = _manifest(audit=audit, samples=25)
    stale = {
        "archive": Path("stale.tar.gz"),
        "receipt": {"archive_sha256": "stale"},
        "manifest": stale_manifest,
    }
    other_manifest = _manifest(audit=audit, samples=25)
    other_manifest["case_id"] = "OTHER"
    other = {
        "archive": Path("other.tar.gz"),
        "receipt": {"archive_sha256": "other"},
        "manifest": other_manifest,
    }
    remote = {
        "exists": True,
        "archive": "/content/drive/MyDrive/remote.tar.gz",
        "receipt": {"archive_sha256": "remote"},
        "manifest": _manifest(audit=audit, samples=25),
    }
    rows = [stale, other]
    matched, mismatches = RUNNER._cache_remote_status_row(
        remote,
        rows,
        bundle="bundle",
        case=_case(25),
        shard_index=0,
        config={"root_seed": 7, "audit_sha256": audit},
        engine_hash="engine",
    )
    assert matched is not None
    assert matched["receipt"]["archive_sha256"] == "remote"
    assert mismatches == []
    assert [str(row["archive"]) for row in rows] == [
        "other.tar.gz",
        "/content/drive/MyDrive/remote.tar.gz",
    ]

    wrong = dict(remote)
    wrong["manifest"] = _manifest(audit="d" * 64, samples=25)
    matched, mismatches = RUNNER._cache_remote_status_row(
        wrong,
        rows,
        bundle="bundle",
        case=_case(25),
        shard_index=0,
        config={"root_seed": 7, "audit_sha256": audit},
        engine_hash="engine",
    )
    assert matched is None
    assert mismatches[0]["mismatch_fields"] == ["audit_sha256"]


def test_absent_remote_status_never_populates_current_rows() -> None:
    rows: list[dict] = []
    matched, mismatches = RUNNER._cache_remote_status_row(
        {"exists": False, "archive": "missing.tar.gz"},
        rows,
        bundle="bundle",
        case=_case(25),
        shard_index=0,
        config={"root_seed": 7, "audit_sha256": "a" * 64},
        engine_hash="engine",
    )
    assert matched is None
    assert mismatches == []
    assert rows == []


def test_parent_independently_binds_remote_status_to_exact_archive_and_receipt(
    tmp_path: Path,
) -> None:
    schema = "classA_drive_api_commit_v1"
    run_config = {"case": _case(25), "shard_index": 0}
    run_hash = RUNNER._sha256_json(run_config)
    bundle = RUNNER.P1_BUNDLE
    run_id = f"{bundle}_{run_hash[:16]}"
    archive = tmp_path / "outputs" / bundle / f"{run_id}.tar.gz"
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    sidecar_path = archive.with_suffix(archive.suffix + ".manifest.json")

    def record(file_id: str, name: str, size: int, sha: str) -> dict:
        return {
            "schema": schema,
            "remote_file_id": file_id,
            "remote_name": name,
            "remote_parent_id": "output-parent",
            "remote_bytes": size,
            "remote_sha256": sha,
        }

    archive_record = record("archive-id", archive.name, 123, "a" * 64)
    receipt = {
        "run_id": run_id,
        "archive": archive.name,
        "archive_bytes": 123,
        "archive_sha256": "a" * 64,
        "archive_remote_commit": archive_record,
    }
    receipt_raw = json.dumps(receipt, sort_keys=True).encode()
    receipt_record = record(
        "receipt-id",
        receipt_path.name,
        len(receipt_raw),
        hashlib.sha256(receipt_raw).hexdigest(),
    )
    manifest = {
        "run_config": run_config,
        "run_config_hash": run_hash,
    }
    sidecar = {
        "schema": RUNNER.P1_MANIFEST_SIDECAR_SCHEMA,
        "bundle": bundle,
        "run_id": run_id,
        "archive": archive.name,
        "archive_remote_commit": archive_record,
        "manifest_sha256": RUNNER._sha256_json(manifest),
        "manifest": manifest,
    }
    sidecar_raw = json.dumps(sidecar, sort_keys=True).encode()
    sidecar_record = record(
        "sidecar-id",
        sidecar_path.name,
        len(sidecar_raw),
        hashlib.sha256(sidecar_raw).hexdigest(),
    )

    class Committer:
        def __init__(self) -> None:
            self.records = {
                str(archive): archive_record,
                str(receipt_path): receipt_record,
                str(sidecar_path): sidecar_record,
            }

        def path_commit_record(self, path: Path) -> dict:
            return dict(self.records[str(path)])

        def verify_record_for_path(self, declared: dict, path: Path) -> None:
            assert RUNNER._remote_commit_identity(declared) == RUNNER._remote_commit_identity(
                self.records[str(path)]
            )

        def download_bytes(self, file_id: str) -> bytes:
            if file_id == "receipt-id":
                return receipt_raw
            if file_id == "sidecar-id":
                return sidecar_raw
            raise AssertionError(file_id)

    remote = {
        "exists": True,
        "archive": str(archive),
        "receipt": receipt,
        "manifest": manifest,
    }
    verified = RUNNER._parent_verify_server_status(
        remote,
        drive_root=tmp_path,
        profile="production",
        bundle=bundle,
        config={
            "production_output_collection": "outputs",
            "output_bundle": bundle,
        },
        committer=Committer(),
    )
    assert verified["archive"] == archive
    assert verified["parent_server_verified"] is True
    assert verified["archive_remote_commit"] == archive_record
    assert verified["receipt_remote_commit"] == receipt_record
    assert verified["manifest_sidecar_remote_commit"] == sidecar_record

    forged = dict(remote)
    forged["archive"] = str(tmp_path / "pilot" / archive.name)
    with pytest.raises(RuntimeError, match="locked output path"):
        RUNNER._parent_verify_server_status(
            forged,
            drive_root=tmp_path,
            profile="production",
            bundle=bundle,
            config={
                "production_output_collection": "outputs",
                "output_bundle": bundle,
            },
            committer=Committer(),
        )

    wrong_receipt = Committer()
    wrong_receipt.records[str(receipt_path)] = {
        **receipt_record,
        "remote_file_id": "wrong-receipt-id",
    }
    with pytest.raises((AssertionError, RuntimeError)):
        RUNNER._parent_verify_server_status(
            remote,
            drive_root=tmp_path,
            profile="production",
            bundle=bundle,
            config={
                "production_output_collection": "outputs",
                "output_bundle": bundle,
            },
            committer=wrong_receipt,
        )


def test_parent_accepts_real_shaped_h1_manifest_without_manifest_run_id(
    tmp_path: Path,
) -> None:
    bundle = RUNNER.H1_ENDPOINT_BUNDLE
    run_config = {
        "sampling_revision": "production_25sample_h1_endpoint_packet_v4",
        "audit_sha256": "b" * 64,
        "canonical_engine_sha256": "c" * 64,
        "bundle_source_hashes_sha256": "d" * 64,
        "shard_index": 0,
        "case": _case(25),
        "trajectory_reuse_policy": "fresh_same_seed",
    }
    run_hash = RUNNER._sha256_json(run_config)
    run_id = f"{bundle}_{run_hash[:16]}"
    archive = tmp_path / "h1-output" / bundle / f"{run_id}.tar.gz"
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    sidecar_path = archive.with_suffix(archive.suffix + ".status.json")
    schema = "classA_drive_api_commit_v1"

    def commit(file_id: str, path: Path, raw: bytes) -> dict:
        return {
            "schema": schema,
            "remote_file_id": file_id,
            "remote_name": path.name,
            "remote_parent_id": "h1-parent",
            "remote_bytes": len(raw),
            "remote_sha256": hashlib.sha256(raw).hexdigest(),
        }

    archive_raw = b"archive"
    archive_record = commit("h1-archive", archive, archive_raw)
    receipt = {
        "run_id": run_id,
        "archive": archive.name,
        "archive_bytes": len(archive_raw),
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archive_remote_commit": archive_record,
    }
    receipt_raw = json.dumps(receipt, sort_keys=True).encode()
    receipt_record = commit("h1-receipt", receipt_path, receipt_raw)
    manifest = {
        "run_config": run_config,
        "run_config_hash": run_hash,
        # Frozen v4 manifests intentionally do not duplicate run_id here.
    }
    sidecar = {
        "schema": RUNNER.H1_STATUS_SIDECAR_SCHEMA,
        "bundle": bundle,
        "run_id": run_id,
        "archive": archive.name,
        "archive_remote_commit": archive_record,
        "receipt_remote_commit": receipt_record,
        "manifest_sha256": RUNNER._sha256_json(manifest),
        "manifest": manifest,
    }
    sidecar_raw = json.dumps(sidecar, sort_keys=True).encode()
    sidecar_record = commit("h1-sidecar", sidecar_path, sidecar_raw)

    class Committer:
        records = {
            str(archive): archive_record,
            str(receipt_path): receipt_record,
            str(sidecar_path): sidecar_record,
        }
        downloads = {
            "h1-receipt": receipt_raw,
            "h1-sidecar": sidecar_raw,
        }

        def path_commit_record(self, path: Path) -> dict:
            return dict(self.records[str(path)])

        def verify_record_for_path(self, record: dict, path: Path) -> None:
            assert RUNNER._remote_commit_identity(record) == RUNNER._remote_commit_identity(
                self.records[str(path)]
            )

        def download_bytes(self, file_id: str) -> bytes:
            return self.downloads[file_id]

    verified = RUNNER._parent_verify_server_status(
        {
            "exists": True,
            "archive": str(archive),
            "receipt": receipt,
            "manifest": manifest,
        },
        drive_root=tmp_path,
        profile="production",
        bundle=bundle,
        config={
            "production_output_collection": "h1-output",
            "output_bundle": bundle,
        },
        committer=Committer(),
    )
    assert verified["parent_server_verified"] is True


def test_server_bundle_post_child_verification_never_ingests_drivefs() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    start = source.index("        def verify_new_outputs(")
    end = source.index("        def finalize_bundle(", start)
    verification = source[start:end]
    assert "if bundle not in SERVER_COMMIT_BUNDLES:" in verification
    assert 'stage_prefix="REMOTE VERIFY"' in verification
    assert "reconcile_server_slot(" in verification
    assert "Drive API reports a durable artifact with the wrong identity" in verification


def test_server_reconciliation_repairs_only_after_status_failure() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    start = source.index("    def reconcile_server_slot(")
    end = source.index("    def refresh_p1_checkpoint_status(", start)
    reconciliation = source[start:end]
    assert reconciliation.index('"--remote-status"') < reconciliation.index(
        '"--repair-orphan"'
    )
    assert "if initial_status_failure is not None:" in reconciliation
    assert "STATUS RETRY" in reconciliation
    assert "return matched, mismatches" in reconciliation
    assert "raise repair_failure" in reconciliation

    recovery_start = source.index("    def recover_committed_child_failure(")
    recovery_end = source.index("    try:", recovery_start)
    recovery = source[recovery_start:recovery_end]
    assert "Credit a child that failed only after its exact remote commit" in recovery


def test_progress_ui_operations_are_best_effort() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    set_current = source[
        source.index("    def set_current(") : source.index("    def heartbeat(")
    ]
    assert 'logger.note_error("progress_postfix"' in set_current
    assert 'logger.note_error("progress_update"' in set_current
    assert 'logger.note_error("progress_close"' in set_current
    queue_open = source[
        source.index("            queue_bar = tqdm(") - 40 : source.index(
            "            for task_index", source.index("            queue_bar = tqdm(")
        )
    ]
    assert 'logger.note_error("progress_open"' in queue_open
    assert "advance_queue(" in source
    assert "close_queue()" in source


def test_strict_bundle_resume_rejects_any_source_hash_change(tmp_path: Path) -> None:
    audit = "b" * 64
    manifest = _manifest(audit=audit, samples=25)
    manifest["source_hashes"] = {"observer.py": "old"}
    path = _archive(tmp_path / "bundle", "strict", manifest)
    row = {
        "archive": path,
        "receipt": json.loads(Path(str(path) + ".receipt.json").read_text()),
        "manifest": RUNNER._root_manifest_from_archive(path),
    }
    matched, mismatches = RUNNER._find_current_archive(
        [row],
        bundle="bundle",
        case=_case(25),
        shard_index=0,
        config={
            "root_seed": 7,
            "audit_sha256": audit,
            "strict_source_hash_resume": True,
        },
        engine_hash="engine",
        source_hashes={"observer.py": "new"},
    )
    assert matched is None
    assert mismatches[0]["mismatch_fields"] == ["source_hashes"]


def test_variable_width_sample_indices_and_session_guard() -> None:
    case = _case(25)
    case["execution"] = {"samples_per_shard": 1}
    assert RUNNER._expected_sample_indices(case, 7) == [7]
    assert RUNNER._session_cap(
        profile="production",
        bundles=["01_p1_chern_dynamics"],
        override=None,
    ) == 7.5
    assert not RUNNER._would_exceed_session_budget(
        elapsed_seconds=2.0 * 3600.0,
        cap_hours=7.5,
        expected_seconds=3.0 * 3600.0,
    )
    assert RUNNER._would_exceed_session_budget(
        elapsed_seconds=4.0 * 3600.0,
        cap_hours=7.5,
        expected_seconds=3.0 * 3600.0,
    )


def test_p1_qualification_budget_uses_one_cycle_and_remaining_child_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "production_output_collection": "outputs/v3",
        "output_bundle": "01_p1_chern_dynamics",
    }
    receipt = (
        tmp_path
        / config["production_output_collection"]
        / config["output_bundle"]
        / "a100_preflight.json"
    )
    base = {
        "safe": True,
        "elapsed_seconds": 64_000.0,
        "measured_cycles_per_trajectory": 64,
    }

    class FakeCommitter:
        def __init__(self, payload: dict) -> None:
            self.payload = payload
            self.verified: list[Path] = []

        def path_commit_record(self, path: Path) -> dict:
            self.verified.append(path)
            return {"remote_name": path.name}

    committer = FakeCommitter(
        {**base, "mean_cycle_seconds": 1_000.0, "max_cycle_seconds": 1_250.0}
    )
    monkeypatch.setattr(
        RUNNER,
        "read_remote_json",
        lambda selected, path: selected.payload,
    )
    assert RUNNER._p1_qualification_cycle_runtime_seconds(
        drive_root=tmp_path, config=config, committer=committer
    ) == 1_250.0
    assert committer.verified == [receipt]

    committer.payload = base
    assert RUNNER._p1_qualification_cycle_runtime_seconds(
        drive_root=tmp_path, config=config, committer=committer
    ) == 1_000.0

    assert RUNNER._remaining_child_runtime_seconds(
        elapsed_seconds=0.0, cap_hours=None
    ) is None
    assert RUNNER._remaining_child_runtime_seconds(
        elapsed_seconds=2.0 * 3600.0, cap_hours=7.5
    ) == pytest.approx(5.5 * 3600.0 - RUNNER.P1_RUNTIME_MARGIN_SECONDS)
    assert RUNNER._remaining_child_runtime_seconds(
        elapsed_seconds=8.0 * 3600.0, cap_hours=7.5
    ) == 0.0


def test_every_p1_geometry_uses_the_qualified_max_cycle_launch_bound() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    start = source.index("                        expected_seconds = 0.0")
    end = source.index(
        "                        require_time_budget(expected_seconds=expected_seconds)",
        start,
    )
    budget = source[start:end]
    assert "if bundle == P1_BUNDLE:" in budget
    assert "p1_l64_expected_cycle_seconds" in budget
    assert 'get("Nx"' not in budget
    command = source[end : source.index("                        shard_wall_seconds", end)]
    assert '"--case-id",\n                            case_id,' in command
    assert '"--case-id",\n                            "--case-id"' not in command


def test_local_session_root_never_uses_drivefs() -> None:
    root = RUNNER.local_session_root(
        output_collection="classA_outputs/production_v4",
        bundle=RUNNER.H1_ENDPOINT_BUNDLE,
    )
    assert root.is_absolute()
    assert "/content/drive" not in str(root)
    assert root.parts[-3:] == (
        "classA_outputs",
        "production_v4",
        RUNNER.H1_ENDPOINT_BUNDLE,
    )


def test_server_storage_guard_uses_remote_tree_and_account_quota(
    tmp_path: Path,
) -> None:
    class Request:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def execute(self) -> dict:
            return self.payload

    class Files:
        def __init__(self, children: dict[str, list[dict]]) -> None:
            self.children = children

        def list(self, **kwargs: object) -> Request:
            query = str(kwargs["q"])
            parent_id = query.split("'", 2)[1]
            return Request({"files": self.children.get(parent_id, [])})

    class Service:
        def __init__(self, children: dict[str, list[dict]]) -> None:
            self._files = Files(children)

        def files(self) -> Files:
            return self._files

    class Committer:
        def __init__(self) -> None:
            self.service = Service(
                {
                    "collection": [
                        {
                            "id": "bundle-folder",
                            "name": RUNNER.P1_BUNDLE,
                            "mimeType": RUNNER.DRIVE_FOLDER_MIME_TYPE,
                        },
                        {"id": "other", "name": "metadata.json", "size": "2"},
                    ],
                    "bundle-folder": [
                        {"id": "archive", "name": "run.tar.gz", "size": "6"},
                        {
                            "id": "receipt",
                            "name": "run.tar.gz.receipt.json",
                            "size": "1",
                        },
                    ],
                }
            )

        def storage_quota(self) -> dict[str, int]:
            return {"limit": 100, "usage": 10}

        def resolve_folder(
            self, relative_parts: tuple[str, ...], *, create: bool
        ) -> str:
            assert relative_parts == ("outputs", "v4")
            assert create is False
            return "collection"

    clear = RUNNER.server_storage_status(
        drive_root=tmp_path,
        output_collection="outputs/v4",
        working_limit_gb=0.000000012,
        absolute_edge_gb=0.000000014,
        required_headroom_gb=0.000000002,
        committer=Committer(),
    )
    assert clear["authoritative_backend"] == "google_drive_api_v3"
    assert clear["used_bytes"] == 9
    assert clear["bundle_bytes"] == {RUNNER.P1_BUNDLE: 7, "metadata.json": 2}
    assert clear["account_free_bytes"] == 90
    assert clear["clear_to_run"] is True

    blocked = RUNNER.server_storage_status(
        drive_root=tmp_path,
        output_collection="outputs/v4",
        working_limit_gb=0.000000010,
        absolute_edge_gb=0.000000014,
        required_headroom_gb=0.000000002,
        committer=Committer(),
    )
    assert blocked["projected_bytes"] == 11
    assert blocked["clear_to_run"] is False


def test_h1_v3_migration_records_are_bound_to_exact_source_paths(
    tmp_path: Path,
) -> None:
    class Committer:
        def __init__(self) -> None:
            self.calls: list[tuple[dict, Path]] = []

        def verify_record_for_path(self, record: dict, path: Path) -> None:
            self.calls.append((record, path))

    accepted_id = "0123456789abcdef"
    rejected_id = "fedcba9876543210"
    accepted_archive = f"{RUNNER.H1_ENDPOINT_BUNDLE}_{accepted_id}.tar.gz"
    migration = {
        "accepted_archives": [
            {
                "run_id": accepted_id,
                "archive": accepted_archive,
                "archive_remote_commit": {"id": "archive"},
                "receipt_remote_commit": {"id": "receipt"},
            }
        ],
        "rejected_receipt_only": [
            {
                "run_id": rejected_id,
                "receipt_remote_commit": {"id": "rejected-receipt"},
            }
        ],
    }
    committer = Committer()
    config = {"v3_compatibility": {"source_collection": "outputs/h1_v3"}}
    RUNNER.verify_h1_v3_migration_paths(
        drive_root=tmp_path,
        config=config,
        migration=migration,
        committer=committer,
    )
    source = tmp_path / "outputs/h1_v3" / RUNNER.H1_ENDPOINT_BUNDLE
    assert committer.calls == [
        ({"id": "archive"}, source / accepted_archive),
        ({"id": "receipt"}, source / f"{accepted_archive}.receipt.json"),
        (
            {"id": "rejected-receipt"},
            source
            / f"{RUNNER.H1_ENDPOINT_BUNDLE}_{rejected_id}.tar.gz.receipt.json",
        ),
    ]

    migration["accepted_archives"][0]["archive"] = "copied_elsewhere.tar.gz"
    with pytest.raises(RuntimeError, match="wrong name"):
        RUNNER.verify_h1_v3_migration_paths(
            drive_root=tmp_path,
            config=config,
            migration=migration,
            committer=Committer(),
        )


def test_parent_routes_h1_migration_through_strict_operational_wrapper() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    start = source.index('if profile == "production" and H1_ENDPOINT_BUNDLE')
    migration_block = source[start : source.index("legacy_rows:", start)]
    assert 'bundle_path(ROOT, H1_ENDPOINT_BUNDLE) / "run_bundle.py"' in migration_block
    assert '"--migration-status"' in migration_block
    assert '"h1_v3_migration.py"' not in migration_block
    assert '"--reuse-current"' not in migration_block


@pytest.mark.parametrize("bundle", [RUNNER.P1_BUNDLE, RUNNER.H1_ENDPOINT_BUNDLE])
def test_exact_qualification_archive_alone_never_unlocks_production(
    bundle: str,
) -> None:
    exact_archive = {"archive": "exact.tar.gz"}
    for unsafe in (None, {}, {"safe": False}):
        with pytest.raises(RuntimeError, match="qualification"):
            RUNNER._qualification_unlock_evidence(
                archive_match=exact_archive,
                safe_receipt=unsafe,
                bundle=bundle,
            )
    assert RUNNER._qualification_unlock_evidence(
        archive_match=exact_archive,
        safe_receipt={"safe": True, "status": "reused_current_safe_receipt"},
        bundle=bundle,
    )["safe"] is True


def test_parent_revalidates_safe_receipt_before_marking_a100_qualified() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    start = source.index("qualification_match, qualification_mismatches")
    end = source.index("a100_qualified.add(bundle)", start)
    block = source[start:end]
    assert "A100 SAFE RECEIPT VERIFY" in block
    assert "_qualification_unlock_evidence" in block


def test_server_bundle_drive_root_never_probes_disconnected_drivefs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def disconnected(*_args, **_kwargs):
        raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")

    monkeypatch.setattr(Path, "resolve", disconnected)
    monkeypatch.setattr(Path, "is_dir", disconnected)
    logical = RUNNER._validated_drive_root(
        Path("/content/drive/MyDrive"), bundle=RUNNER.P1_BUNDLE
    )
    assert logical == Path("/content/drive/MyDrive")


def test_legacy_bundle_drive_root_still_requires_a_real_directory(
    tmp_path: Path,
) -> None:
    assert RUNNER._validated_drive_root(tmp_path, bundle="00_validation") == tmp_path
    with pytest.raises(FileNotFoundError):
        RUNNER._validated_drive_root(
            tmp_path / "absent", bundle="00_validation"
        )


def test_session_logger_mount_loss_is_nonfatal(tmp_path: Path) -> None:
    class DisconnectedHandle:
        def write(self, _value: str) -> None:
            raise OSError(errno.ENOTCONN, "DriveFS disconnected")

        def close(self) -> None:
            raise OSError(errno.ENOTCONN, "DriveFS disconnected")

    display = io.StringIO()
    logger = RUNNER.SessionLogger(tmp_path / "session.log", display=display)
    assert logger._handle is not None
    logger._handle.close()
    logger._handle = DisconnectedHandle()
    logger.emit("child remains authoritative")
    logger.close()

    assert "child remains authoritative" in display.getvalue()
    assert [row["operation"] for row in logger.errors] == [
        "write",
        "close_after_write_failure",
    ]

def test_corrupt_archive_fails_before_resume(tmp_path: Path) -> None:
    path = _archive(
        tmp_path / "bundle", "run", _manifest(audit="a" * 64, samples=25)
    )
    path.write_bytes(path.read_bytes() + b"corrupt")
    logger = RUNNER.SessionLogger(tmp_path / "session.log")
    try:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            RUNNER._scan_archive_directory(
                tmp_path / "bundle", label="test", logger=logger
            )
    finally:
        logger.close()


def test_orphan_receipt_fails_before_resume(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "missing.tar.gz.receipt.json").write_text("{}", encoding="utf-8")
    logger = RUNNER.SessionLogger(tmp_path / "session.log")
    try:
        with pytest.raises(RuntimeError, match="receipt exists without its archive"):
            RUNNER._scan_archive_directory(root, label="test", logger=logger)
    finally:
        logger.close()


def test_approved_legacy_superset_matches_without_becoming_current(tmp_path: Path) -> None:
    audit = next(
        value
        for value in RUNNER.APPROVED_LEGACY_AUDIT_SHA256
        if value not in {RUNNER.V1_AUDIT_SHA256, RUNNER.PRIOR_V2_AUDIT_SHA256}
    )
    path = _archive(tmp_path, "legacy", _manifest(audit=audit, samples=25))
    row = {
        "archive": path,
        "receipt": json.loads(Path(str(path) + ".receipt.json").read_text()),
        "manifest": RUNNER._root_manifest_from_archive(path),
    }
    match = RUNNER._find_compatible_legacy_archive(
        {"prior_v2": [], "v1": [], "unversioned": [row]},
        bundle="bundle",
        case=_case(25),
        shard_index=0,
        config={"root_seed": 7, "audit_sha256": "current"},
        engine_hash="engine",
    )
    assert match is not None
    assert match["status"] == "compatible_legacy_superset"
    assert match["legacy_sample_count"] == 25


def test_streaming_child_emits_heartbeats_and_retains_failure_tail(tmp_path: Path) -> None:
    logger = RUNNER.SessionLogger(tmp_path / "session.log")
    heartbeats: list[tuple[int, float, str | None]] = []
    try:
        with pytest.raises(RUNNER.ChildProcessFailure) as error:
            RUNNER._run_streaming(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; print('working'); time.sleep(.08); "
                    "print('failed'); raise SystemExit(7)",
                ],
                logger=logger,
                heartbeat_seconds=0.02,
                heartbeat=lambda pid, elapsed, line: heartbeats.append((pid, elapsed, line)),
            )
    finally:
        logger.close()
    assert heartbeats
    assert error.value.returncode == 7
    assert "working" in error.value.output_tail
    assert "failed" in error.value.output_tail


def test_streaming_child_survives_telemetry_disconnect(tmp_path: Path) -> None:
    logger = RUNNER.SessionLogger(tmp_path / "session.log", display=io.StringIO())

    def disconnected_heartbeat(*_args: object) -> None:
        raise OSError(errno.ENOTCONN, "DriveFS disconnected")

    try:
        tail, _ = RUNNER._run_streaming(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('started'); time.sleep(.08); print('committed')",
            ],
            logger=logger,
            heartbeat_seconds=0.01,
            heartbeat=disconnected_heartbeat,
        )
    finally:
        logger.close()

    assert tail == ["started", "committed"]
    assert any(row["operation"] == "heartbeat_callback" for row in logger.errors)


def test_streaming_child_uses_isolated_tqdm_environment_and_drops_redraws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TQDM_DISABLE", "parent-value")
    display = io.StringIO()
    logger = RUNNER.SessionLogger(tmp_path / "session.log", display=display)
    observed_lines: list[str] = []
    try:
        tail, _ = RUNNER._run_streaming(
            [
                sys.executable,
                "-u",
                "-c",
                "import os, sys; "
                "print('child_tqdm=' + os.environ['TQDM_DISABLE']); "
                "sys.stdout.write('redraw 1\\rredraw 2\\rredraw 3\\n'); "
                "sys.stdout.flush(); print('stable status')",
            ],
            logger=logger,
            heartbeat_seconds=1.0,
            heartbeat=lambda *_: None,
            on_line=observed_lines.append,
        )
    finally:
        logger.close()

    assert os.environ["TQDM_DISABLE"] == "parent-value"
    assert tail == ["child_tqdm=1", "stable status"]
    assert observed_lines == tail
    assert "redraw" not in display.getvalue()
    assert "redraw" not in (tmp_path / "session.log").read_text(encoding="utf-8")


def test_session_logger_preserves_cli_stdout_outside_notebook(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = RUNNER.SessionLogger(tmp_path / "session.log")
    try:
        logger.emit("visible status")
    finally:
        logger.close()
    captured = capsys.readouterr()
    assert "visible status" in captured.out
    assert captured.err == ""


def test_streaming_child_is_reaped_when_output_handler_interrupts(tmp_path: Path) -> None:
    logger = RUNNER.SessionLogger(tmp_path / "session.log", display=io.StringIO())
    child_pids: list[int] = []

    def interrupt(_line: str) -> None:
        raise KeyboardInterrupt("test interruption")

    try:
        with pytest.raises(KeyboardInterrupt, match="test interruption"):
            RUNNER._run_streaming(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; print('started'); time.sleep(30)",
                ],
                logger=logger,
                heartbeat_seconds=1.0,
                heartbeat=lambda *_: None,
                on_start=child_pids.append,
                on_line=interrupt,
            )
    finally:
        logger.close()

    assert len(child_pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(child_pids[0], 0)


def test_streaming_child_is_reaped_when_on_start_reporting_fails(
    tmp_path: Path,
) -> None:
    logger = RUNNER.SessionLogger(tmp_path / "session.log", display=io.StringIO())
    child_pids: list[int] = []

    def fail_after_start(pid: int) -> None:
        child_pids.append(pid)
        raise OSError(errno.ENOTCONN, "telemetry target disconnected")

    try:
        with pytest.raises(OSError, match="telemetry target disconnected"):
            RUNNER._run_streaming(
                [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
                logger=logger,
                heartbeat_seconds=1.0,
                heartbeat=lambda *_: None,
                on_start=fail_after_start,
            )
    finally:
        logger.close()

    assert len(child_pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(child_pids[0], 0)


def test_generated_long_runs_call_exported_main_in_colab_kernel() -> None:
    generator_path = CAMPAIGN / "_maintenance" / "build_colab_notebooks.py"
    spec = importlib.util.spec_from_file_location(
        "tested_in_kernel_bundle_notebook_generator", generator_path
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    for bundle in generator.BUNDLES:
        notebook = generator.bundle_notebook(bundle)
        cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        if bundle in {
            "01_p1_chern_dynamics",
            "02_wall_cft_windows",
            "03_h1_modular_response",
            "08_h1_endpoint_packet",
        }:
            long_run_cells = [cell for cell in cells if "[bundle dashboard]" in cell]
            assert len(long_run_cells) == 1
            assert "runner_module.main(runner_argv)" in long_run_cells[0]
            assert "subprocess.run(command" not in long_run_cells[0]
            qualification_cell = next(
                cell for cell in cells if "RUN_A100_PREFLIGHT = False" in cell
            )
            assert "preflight_module.main(preflight_argv)" in qualification_cell
            assert "str(bundle_root / 'run_bundle.py')" not in qualification_cell
        elif bundle in {"04_maxmix_operator_cft", "05_pure_tangent_stability"}:
            long_run_cells = [cell for cell in cells if "RUN_QUEUE = False" in cell]
            assert len(long_run_cells) == 1
            assert "runner_module.main(runner_argv)" in long_run_cells[0]
            assert "subprocess.run(command" not in long_run_cells[0]
        else:
            pilot_cell = next(cell for cell in cells if "RUN_PILOT = False" in cell)
            production_cell = next(
                cell for cell in cells if "RUN_PRODUCTION = False" in cell
            )
            assert "runner_module.main(pilot_argv)" in pilot_cell
            assert "runner_module.main(production_argv)" in production_cell
            assert "subprocess.run(pilot_command" not in pilot_cell
            assert "subprocess.run(production_command" not in production_cell

        preflight_cell = next(
            cell for cell in cells if "RUN_A100_PREFLIGHT = True" in cell
        ) if bundle in {
            "04_maxmix_operator_cft",
            "05_pure_tangent_stability",
            "07_log_gram_alpha_scan",
        } else None
        if preflight_cell is not None:
            assert "subprocess.run(" in preflight_cell


def test_all_bundle_notebooks_expose_resume_dashboard_and_heartbeat() -> None:
    paths = [
        CAMPAIGN / bundle / "run_production_bundle.ipynb"
        for bundle in (
            "01_p1_chern_dynamics",
            "02_wall_cft_windows",
            "03_h1_modular_response",
            "08_h1_endpoint_packet",
        )
    ]
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        assert "HEARTBEAT_SECONDS = 60" in source
        assert "RESUME_REPORT_ONLY = True" in source
        assert "--heartbeat-seconds" in source
        assert "runner_build_id" in source
        assert "saved bundle failure" in source
        assert "runtime.unassign()" in source
        if path.parent.name in {
            RUNNER.P1_BUNDLE,
            RUNNER.H1_ENDPOINT_BUNDLE,
        }:
            assert "DRIVE_CAMPAIGN_ROOT" in source
            assert "LOCAL_CAMPAIGN_ROOT" in source
            assert "CAMPAIGN_ROOT = LOCAL_CAMPAIGN_ROOT" in source
            assert "_verified_deployment" in source
            assert "deployment_files" in source
            assert "runtime_deployment_files" in source
            assert "notebook_manifest_files" in source
            assert "notebook records are audit-only because Colab mutates open notebook outputs" in source
            assert "EXPECTED_OPERATIONAL_RELEASE" in source
            assert "_drive_service.files().get_media" in source
            assert "deployment_manifest_path.read_text" not in source
            assert "shutil.copy2(source, destination)" not in source
            assert "runner_module.local_session_root" in source
            assert "saved bundle failure: local telemetry" in source
            assert "runner_module.server_storage_status" in source
        if path.parent.name == "01_p1_chern_dynamics":
            assert "MAX_SESSION_GPU_HOURS_OVERRIDE = 7.5" in source
            assert "sample-0 trajectory through atomic Drive checkpoints" in source


def test_v4_deployment_includes_root_remote_helper_and_manifest_hash(
    tmp_path: Path,
) -> None:
    builder_path = CAMPAIGN / "_maintenance" / "build_v4_deployment.py"
    spec = importlib.util.spec_from_file_location("tested_v4_builder", builder_path)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    destination = tmp_path / "final_production_new_designs_v4"
    manifest = builder.build(destination)
    assert manifest["operational_release"] == builder.V4_OPERATIONAL_RELEASE
    helper = destination / "drive_remote_commit.py"
    assert helper.is_file()
    record = manifest["files"]["drive_remote_commit.py"]
    assert record["bytes"] == helper.stat().st_size
    assert record["sha256"] == builder.sha256(helper)
    assert (destination / "p1_runtime_hardening.py").is_file()
    assert (destination / "server_verified_analysis.py").is_file()
    assert (destination / "production_runtime.py").is_file()
    assert (destination / "colab_bundle_runner.py").is_file()


def test_server_bootstrap_does_not_hash_gate_mutable_open_notebooks() -> None:
    expected_notebooks = {
        "01_p1_chern_dynamics/run_production_bundle.ipynb",
        "08_h1_endpoint_packet/run_production_bundle.ipynb",
    }
    for bundle in (RUNNER.P1_BUNDLE, RUNNER.H1_ENDPOINT_BUNDLE):
        notebook = json.loads(
            (CAMPAIGN / bundle / "run_production_bundle.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        for relative in expected_notebooks:
            assert relative in source
        assert (
            "runtime_deployment_files = {relative: expected for relative, expected "
            "in deployment_files.items() if relative not in notebook_manifest_files}"
        ) in source
        assert source.count(
            "for relative, expected in runtime_deployment_files.items():"
        ) == 2
        assert "_drive_download_verified(remote_row, expected)" in source
        assert "notebook records are audit-only because Colab mutates open notebook outputs" in source


def test_p1_h1_analysis_cells_use_server_verified_materialization() -> None:
    for bundle, toggle in (
        ("01_p1_chern_dynamics", "RUN_P1_ANALYSIS = False"),
        ("08_h1_endpoint_packet", "RUN_H1_ENDPOINT_ANALYSIS = False"),
    ):
        notebook = json.loads(
            (CAMPAIGN / bundle / "run_production_bundle.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = next(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if toggle in "".join(cell.get("source", []))
        )
        assert "server_verified_analysis.py" in source
        assert "--campaign-root" in source
        assert f"src/{'p1_chern_analysis.py' if bundle.startswith('01_') else 'h1_packet_analysis.py'}" not in source


def test_remote_commit_probe_precedes_any_real_p1_h1_preflight() -> None:
    for bundle in (RUNNER.P1_BUNDLE, RUNNER.H1_ENDPOINT_BUNDLE):
        notebook = json.loads(
            (CAMPAIGN / bundle / "run_production_bundle.ipynb").read_text(
                encoding="utf-8"
            )
        )
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        config = next(i for i, source in enumerate(sources) if f"BUNDLE = {bundle!r}" in source)
        probe = next(i for i, source in enumerate(sources) if "RUN_REMOTE_COMMIT_PROBE" in source)
        preflight = next(i for i, source in enumerate(sources) if "RUN_A100_PREFLIGHT = False" in source)
        queue = next(i for i, source in enumerate(sources) if "[bundle dashboard]" in source)
        assert config < probe < preflight < queue


def test_bundle_notebooks_match_the_canonical_generator() -> None:
    generator_path = CAMPAIGN / "_maintenance" / "build_colab_notebooks.py"
    spec = importlib.util.spec_from_file_location("tested_bundle_notebook_generator", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    for bundle in generator.BUNDLES:
        path = generator.bundle_path(CAMPAIGN, bundle) / "run_production_bundle.ipynb"
        assert json.loads(path.read_text(encoding="utf-8")) == generator.bundle_notebook(bundle)
    assert not (CAMPAIGN / "COLAB_LANES").exists()
    assert not (CAMPAIGN / "colab_lane_runner.py").exists()


def test_h1_analysis_cell_does_not_depend_on_queue_cell_state() -> None:
    notebook = json.loads(
        (CAMPAIGN / "03_h1_modular_response" / "run_production_bundle.ipynb").read_text(
            encoding="utf-8"
        )
    )
    analysis_source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "RUN_H1_ANALYSIS = False" in "".join(cell.get("source", []))
    )
    assert "H1_ANALYSIS_PROFILE = 'production'" in analysis_source
    assert "MYDRIVE / bundle_config[collection_key] / OUTPUT_BUNDLE" in analysis_source
    assert "output_root" not in analysis_source

def test_production_queue_automatically_runs_or_reuses_a100_qualification() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    qualification = source.index('[A100 QUALIFICATION]')
    lightweight = source.index('"--preflight-only"', qualification)
    running = source.index('stage="running"', lightweight)
    assert qualification < lightweight < running
    assert 'stage="a100-preflight"' in source

    for bundle in (
        "01_p1_chern_dynamics",
        "02_wall_cft_windows",
        "03_h1_modular_response",
        "08_h1_endpoint_packet",
    ):
        notebook = json.loads(
            (CAMPAIGN / bundle / "run_production_bundle.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        expected_qualification_text = (
            "runs or resumes this qualification automatically"
            if bundle == "01_p1_chern_dynamics"
            else "runs or reuses this qualification automatically"
        )
        assert expected_qualification_text in source


def test_all_redesigned_notebooks_have_safe_launch_defaults() -> None:
    expected = {
        "01_p1_chern_dynamics",
        "02_wall_cft_windows",
        "03_h1_modular_response",
        "04_maxmix_operator_cft",
        "05_pure_tangent_stability",
        "07_log_gram_alpha_scan",
        "08_h1_endpoint_packet",
    }
    paths = sorted(CAMPAIGN.glob("*/run_production_bundle.ipynb"))
    assert {path.parent.name for path in paths} == expected
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        assert notebook["metadata"]["colab"]["gpuType"] == "A100"
        assert "runtime.unassign()" in source
    for bundle in ("04_maxmix_operator_cft", "05_pure_tangent_stability"):
        source = (CAMPAIGN / bundle / "run_production_bundle.ipynb").read_text()
        assert "RUN_A100_PREFLIGHT = True" in source
        assert "RUN_QUEUE = False" in source
    log_gram = (CAMPAIGN / "07_log_gram_alpha_scan/run_production_bundle.ipynb").read_text()
    assert "RUN_A100_PREFLIGHT = True" in log_gram
    assert "RUN_PILOT = False" in log_gram
    assert "RUN_PRODUCTION = False" in log_gram
