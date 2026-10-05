#!/usr/bin/env python3
"""Run one dependency-aware sequence of production bundles in a Colab session."""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Callable

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
SHARED_SRC = ROOT / "_shared_src"
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))

from production_runtime import (  # noqa: E402
    APPROVED_LEGACY_AUDIT_SHA256,
    LEGACY_PRODUCTION_OUTPUT_COLLECTION,
    PILOT_OUTPUT_COLLECTION,
    PRODUCTION_OUTPUT_COLLECTION,
    PRODUCTION_SAMPLES,
    SHARD_SIZE,
    V1_AUDIT_SHA256,
    V1_PRODUCTION_OUTPUT_COLLECTION,
    json_ready,
    sha256_file,
    verify_archive_receipt,
)


LANES = {
    "lane_1_baselines": ("01_bulk_width_gate", "06_b1_controller_frame"),
    "lane_2_parents_flux": ("02_pure_wall_master", "03_chirality_replay"),
    "lane_3_maxmix_scans": ("04_maxmix_master", "05_scans_and_controls"),
}
B1_BUNDLE = "06_b1_controller_frame"
SESSION_SCHEMA = "classA_colab_lane_session_v2"
CHILD_TAIL_LINES = 200
PROFILE_HOUR_CAPS = {
    "pilot_calibration": {
        "01_bulk_width_gate": 7.0,
        "02_pure_wall_master": 10.0,
        "03_chirality_replay": 6.0,
        "04_maxmix_master": 3.0,
        "05_scans_and_controls": 5.0,
        "06_b1_controller_frame": 2.0,
    },
    "pilot_science": {
        "01_bulk_width_gate": 32.0,
        "02_pure_wall_master": 30.0,
        "03_chirality_replay": 20.0,
        "04_maxmix_master": 14.0,
        "05_scans_and_controls": 25.0,
        "06_b1_controller_frame": 8.0,
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
        super().__init__(
            f"child command exited {returncode}: {shlex.join(command)}"
        )
        self.command = list(command)
        self.returncode = int(returncode)
        self.output_tail = list(output_tail)
        self.elapsed_seconds = float(elapsed_seconds)
        self.stage = stage


class SessionLogger:
    """Write readable Colab output and an append-only persistent session log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def emit(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        tqdm.write(line)
        self._handle.write(line + "\n")

    def close(self) -> None:
        self._handle.close()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(json_ready(value), sort_keys=True))


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
    if samples == 1:
        return [0]
    start = int(shard_index) * SHARD_SIZE
    return list(range(start, min(start + SHARD_SIZE, samples)))


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
    ):
        receipt = verify_archive_receipt(archive)
        manifest = _root_manifest_from_archive(archive)
        rows.append({"archive": archive, "receipt": receipt, "manifest": manifest})
    logger.emit(f"[RESUME] {label}: verified {len(rows)}/{len(archives)}")
    return rows


def _current_match_reasons(
    row: dict[str, Any],
    *,
    bundle: str,
    case: dict[str, Any],
    shard_index: int,
    config: dict[str, Any],
    engine_hash: str,
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
        )
    ]
    hashes = {str(row["receipt"]["archive_sha256"]): row for row in matches}
    if len(hashes) > 1:
        raise RuntimeError(
            f"multiple distinct current archives for {bundle}/{case['case_id']} "
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
        )
        if reasons:
            mismatch_rows.append(
                {"archive": str(row["archive"]), "mismatch_fields": reasons}
            )
    return (next(iter(hashes.values())) if hashes else None), mismatch_rows


def _find_compatible_legacy_archive(
    rows_by_revision: dict[str, list[dict[str, Any]]],
    *,
    bundle: str,
    case: dict[str, Any],
    shard_index: int,
    config: dict[str, Any],
    engine_hash: str,
) -> dict[str, Any] | None:
    expected_indices = list(
        range(int(shard_index) * SHARD_SIZE, (int(shard_index) + 1) * SHARD_SIZE)
    )
    expected_seed = _shard_seed(int(config["root_seed"]), case["case_id"], shard_index)
    current_case = _case_without_declared_samples(case)
    priorities = (
        ("v1", {V1_AUDIT_SHA256}),
        ("unversioned", APPROVED_LEGACY_AUDIT_SHA256 - {V1_AUDIT_SHA256}),
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
        if len(matches) > 1:
            raise RuntimeError(
                f"multiple distinct compatible {revision} archives for "
                f"{case['case_id']} shard {shard_index}: "
                f"{[str(row['archive']) for row in matches.values()]}"
            )
        if not matches:
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
                manifest.get("run_config", {}).get("case", {}).get("run", {}).get("samples", 0)
            ),
        }
    return None


def _run_json(command: list[str], *, stage: str, logger: SessionLogger) -> Any:
    logger.emit(f"[{stage}] {shlex.join(command)}")
    started = time.monotonic()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    if completed.returncode:
        tail = (completed.stdout + "\n" + completed.stderr).splitlines()[-CHILD_TAIL_LINES:]
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
        text=True,
        bufsize=1,
    )
    if on_start is not None:
        on_start(process.pid)
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        try:
            for line in process.stdout:
                lines.put(line.rstrip("\n"))
        finally:
            lines.put(None)

    thread = threading.Thread(target=reader, name="lane-child-output", daemon=True)
    thread.start()
    tail: deque[str] = deque(maxlen=CHILD_TAIL_LINES)
    last_line: str | None = None
    next_heartbeat = started + heartbeat_seconds
    stream_closed = False
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
            heartbeat(process.pid, now - started, last_line)
            next_heartbeat = now + heartbeat_seconds
    returncode = process.wait()
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


def _bundle_arguments(
    *, bundle: str, profile: str, drive_root: Path, provisional_width: int
) -> tuple[list[str], str]:
    mode = "production" if profile == "production" else "pilot"
    bundle_root = ROOT / bundle
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
    if mode == "production" and bundle not in ("01_bulk_width_gate", B1_BUNDLE):
        base += [
            "--accepted-width-json",
            str(
                drive_root
                / PRODUCTION_OUTPUT_COLLECTION
                / "01_bulk_width_gate"
                / "accepted_width.json"
            ),
        ]
    if mode == "production" and bundle in ("04_maxmix_master", "05_scans_and_controls"):
        decision_name = (
            "T1_gate_viable.json" if bundle == "04_maxmix_master" else "core_gate_decisions.json"
        )
        base += [
            "--gate-decisions-json",
            str(drive_root / PRODUCTION_OUTPUT_COLLECTION / bundle / decision_name),
        ]
    if mode == "production" and bundle == "05_scans_and_controls":
        m3_gate = drive_root / PRODUCTION_OUTPUT_COLLECTION / bundle / "m3_bulk_gate.json"
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
    if profile == "production":
        return None
    return sum(PROFILE_HOUR_CAPS[profile][bundle] for bundle in bundles)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--lane", choices=tuple(LANES), required=True)
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
    parser.add_argument("--required-headroom-gb", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.heartbeat_seconds <= 0.0:
        raise ValueError("--heartbeat-seconds must be positive")
    if args.preflight_only and args.resume_report_only:
        raise ValueError("use either --preflight-only or --resume-report-only, not both")
    drive_root = args.drive_root.resolve()
    if not drive_root.is_dir():
        raise FileNotFoundError(drive_root)
    bundles = list(LANES[args.lane])
    for bundle in bundles:
        if not (ROOT / bundle / "run_bundle.py").is_file():
            raise FileNotFoundError(f"incomplete uploaded bundle: {ROOT / bundle}")

    profile = str(args.profile)
    output_collection = (
        PRODUCTION_OUTPUT_COLLECTION if profile == "production" else PILOT_OUTPUT_COLLECTION
    )
    output_root = drive_root / output_collection
    session_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{args.lane}_{os.getpid()}"
    )
    session_root = output_root / "_lane_sessions"
    session_path = session_root / f"{session_id}.json"
    log_path = session_root / f"{session_id}.log"
    logger = SessionLogger(log_path)
    runner_build_id = sha256_file(Path(__file__))[:16]
    session_started = time.monotonic()

    tasks: list[dict[str, Any]] = []
    flat_items: list[dict[str, Any]] = []
    bases: dict[str, list[str]] = {}
    bundle_configs: dict[str, dict[str, Any]] = {}
    bundle_engine_hashes: dict[str, str] = {}
    status_by_key: dict[tuple[str, str, int], str] = {}
    current_rows: dict[str, list[dict[str, Any]]] = {}
    completed: list[dict[str, Any]] = []
    verified_current: list[dict[str, Any]] = []
    compatible_legacy: list[dict[str, Any]] = []
    newly_completed: list[dict[str, Any]] = []
    preflight_completed: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []
    shard_wall_seconds: list[float] = []
    current: dict[str, Any] | None = None
    current_process: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    total_shards = 0
    credit_rate = 0.0
    session_cap: float | None = None

    def key_for(bundle: str, case_id: str, shard: int) -> tuple[str, str, int]:
        return bundle, case_id, int(shard)

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
            "lane": args.lane,
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
            "elapsed_hours": elapsed_hours,
            "credits": elapsed_hours * credit_rate,
            "mean_shard_minutes": None if mean_seconds is None else mean_seconds / 60.0,
            "rolling_eta_hours": (
                None if mean_seconds is None else mean_seconds * pending / 3600.0
            ),
            "session_cap_hours": session_cap,
            "current": current,
            "current_process": current_process,
            "last_heartbeat_utc": (
                None if current_process is None else current_process.get("last_heartbeat_utc")
            ),
            "last_child_line": (
                None if current_process is None else current_process.get("last_child_line")
            ),
            "failure": failure,
            "identity_mismatches": identity_mismatches,
            "completed": completed,
            "session_json": str(session_path),
            "session_log": str(log_path),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(session_path, payload)

    def set_current(item: dict[str, Any], stage: str) -> None:
        nonlocal current
        current = {
            "ordinal": int(item["ordinal"]),
            "bundle": str(item["bundle"]),
            "case_id": str(item["case_id"]),
            "shard": int(item["shard"]),
            "stage": stage,
        }

    def heartbeat(pid: int, elapsed: float, last_line: str | None) -> None:
        nonlocal current_process
        assert current is not None
        current_process = {
            "pid": int(pid),
            "started_utc": current_process.get("started_utc") if current_process else None,
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

    try:
        logger.emit(
            f"[STARTUP] loading {args.lane} profile={profile} runner={runner_build_id}"
        )
        logger.emit(f"[STARTUP] output root: {output_root}")
        pilot_plan = json.loads((ROOT / "pilot_plan.json").read_text(encoding="utf-8"))
        credit_rate = float(pilot_plan["credit_rate_per_a100_hour"])
        session_cap = _session_cap(
            profile=profile, bundles=bundles, override=args.max_session_hours
        )
        pilot_level = profile.removeprefix("pilot_") if profile.startswith("pilot_") else None

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
            shard_counts = {str(row["case_id"]): int(row["shard_count"]) for row in rows}
            cases_by_id = {str(row["case_id"]): dict(row["case"]) for row in rows}
            config = json.loads(
                (ROOT / bundle / "production_config.json").read_text(encoding="utf-8")
            )
            bundle_configs[bundle] = config
            bundle_engine_hashes[bundle] = sha256_file(
                ROOT / bundle / "src" / "classA_U1FGTN_gpu.py"
            )
            selected_cases = (
                list(shard_counts)
                if profile == "production"
                else list(pilot_plan[pilot_level][bundle])
            )
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
                invalid = [value for value in shards if not 0 <= value < shard_counts[case_id]]
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
            raise ValueError("the selected lane queue is empty")
        logger.emit(f"[STARTUP] {args.lane}: {total_shards} planned shard(s)")

        legacy_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for bundle in bundles:
            current_rows[bundle] = _scan_archive_directory(
                output_root / bundle,
                label=f"{bundle} current",
                logger=logger,
            )
            legacy_rows[bundle] = {"v1": [], "unversioned": []}
            if profile == "production":
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
            desc=f"match {args.lane}",
            unit="shard",
            dynamic_ncols=True,
            leave=False,
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
            )
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
            legacy = (
                _find_compatible_legacy_archive(
                    legacy_rows[bundle],
                    bundle=bundle,
                    case=case,
                    shard_index=shard,
                    config=bundle_configs[bundle],
                    engine_hash=bundle_engine_hashes[bundle],
                )
                if profile == "production"
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
            bundle: max(index for index, task in enumerate(tasks) if task["bundle"] == bundle)
            for bundle in bundles
        }

        def require_time_budget() -> None:
            elapsed_hours = (time.monotonic() - session_started) / 3600.0
            if session_cap is not None and elapsed_hours >= session_cap:
                raise RuntimeError(
                    f"lane session cap reached ({elapsed_hours:.3f}/{session_cap:.3f} h); "
                    "restart this lane to resume verified remaining work"
                )

        def run_storage_guard(bundle: str, item: dict[str, Any]) -> None:
            command = [
                sys.executable,
                str(ROOT / bundle / "src" / "drive_storage_guard.py"),
                "--output-root",
                str(output_root),
                "--working-limit-gb",
                str(args.working_limit_gb),
                "--absolute-edge-gb",
                str(args.absolute_edge_gb),
                "--required-headroom-gb",
                str(args.required_headroom_gb),
            ]
            run_child(command, item=item, stage="storage")

        def verify_new_outputs(bundle: str, task: dict[str, Any], shards: list[int]) -> None:
            known = {str(row["archive"]) for row in current_rows[bundle]}
            root = output_root / bundle
            candidates = [path for path in sorted(root.glob("*.tar.gz")) if str(path) not in known]
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
            ):
                new_rows.append(
                    {
                        "archive": archive,
                        "receipt": verify_archive_receipt(archive),
                        "manifest": _root_manifest_from_archive(archive),
                    }
                )
            current_rows[bundle].extend(new_rows)
            for shard in shards:
                item = next(
                    row
                    for row in flat_items
                    if row["bundle"] == bundle
                    and row["case_id"] == task["case_id"]
                    and int(row["shard"]) == int(shard)
                )
                matched, _ = _find_current_archive(
                    current_rows[bundle],
                    bundle=bundle,
                    case=task["case"],
                    shard_index=shard,
                    config=bundle_configs[bundle],
                    engine_hash=bundle_engine_hashes[bundle],
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
            if bundle == "01_bulk_width_gate":
                gate_output = output_root / bundle / "accepted_width.json"
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / bundle / "src" / "gate_analysis.py"),
                    "width",
                    "--archive-root",
                    str(output_root / bundle),
                    "--output",
                    str(gate_output),
                    "--b0-gate",
                    str(ROOT / bundle / "b0_accepted_width.json"),
                ]
                status = "production_width_gate_complete"
            elif bundle == "05_scans_and_controls":
                gate_output = output_root / bundle / "m3_bulk_gate.json"
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / bundle / "src" / "gate_analysis.py"),
                    "m3_bulk",
                    "--archive-root",
                    str(output_root / bundle),
                    "--output",
                    str(gate_output),
                ]
                status = "production_m3_bulk_gate_complete_rerun_lane_for_wall_bracket"
            else:
                return
            run_child(command, item=item, stage="gate-analysis")
            logger.emit(f"[GATE] {gate_output.read_text(encoding='utf-8')}")
            report(status)

        queue_bar = tqdm(
            total=total_shards,
            initial=verified_total,
            desc=args.lane,
            unit="shard",
            dynamic_ncols=True,
        )
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
                            status_by_key[key_for(bundle, case_id, shard)] = "preflight_complete"
                            preflight_completed.append(item)
                            completed.append({**item, "preflight": True})
                            queue_bar.update(1)
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
                    queue_bar.update(len(pending_shards))
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
                            status_by_key[key_for(bundle, case_id, shard)] = "preflight_complete"
                            preflight_completed.append(item)
                            completed.append({**item, "preflight": True})
                            queue_bar.update(1)
                            report("preflight_progress")
                            continue
                        run_storage_guard(bundle, item)
                        command = [
                            *base,
                            "--case-id",
                            case_id,
                            "--shard-index",
                            str(shard),
                        ]
                        elapsed = run_child(command, item=item, stage="running")
                        shard_wall_seconds.append(elapsed)
                        verify_new_outputs(bundle, task, [shard])
                        queue_bar.update(1)
                        report("queue_progress")
                if task_index == last_task_for_bundle[bundle] and not args.preflight_only:
                    finalize_bundle(bundle, last_item)
        finally:
            queue_bar.close()

        current = None
        current_process = None
        report("preflight_complete" if args.preflight_only else "queue_complete")
        logger.emit(f"[COMPLETE] {args.lane}: processed {total_shards}/{total_shards}")
        logger.emit(f"[SESSION] {session_path}")
        return 0
    except BaseException as exc:
        if isinstance(exc, ChildProcessFailure):
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": exc.stage or (None if current is None else current.get("stage")),
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
        report("failed")
        logger.emit("[FAILURE] " + json.dumps(failure, indent=2, sort_keys=True))
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
