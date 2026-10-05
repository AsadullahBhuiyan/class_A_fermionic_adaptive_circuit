#!/usr/bin/env python3
"""Verify and finish only the original 25-trajectory P1 matrix.

This runner is deliberately pinned to the transferred ``production_25sample_v1``
identity.  It never enumerates W1 and never writes into the new lean campaign tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import tarfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campaign_cases import expand_cases  # noqa: E402
from production_runtime import (  # noqa: E402
    SHARD_SIZE,
    load_config,
    sha256_file,
    sha256_json,
    verify_archive_receipt,
)


EXPECTED_AUDIT = "d23d313fd8d6b11074ae0a351b8c8f9fa720cbd0866dac739ba0beca9fc1338f"
EXPECTED_ENGINE = "2a51e13bd960cd3f45f8d63bd6114ddbc02dffe21a94dd5993234a47d5eefb41"
EXPECTED_SAMPLING_REVISION = "production_25sample_v1"
EXPECTED_CASES = 48
EXPECTED_SHARDS = 240


def _seed(root_seed: int, case_id: str, shard: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _manifest(path: Path) -> dict[str, Any]:
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
            raise RuntimeError(f"{path}: root manifest is unreadable")
        return json.loads(handle.read().decode("utf-8"))


def _run_config(
    config: dict[str, Any], case: dict[str, Any], shard: int
) -> dict[str, Any]:
    start = int(shard) * SHARD_SIZE
    return {
        "mode": "production",
        "case": case,
        "shard_index": int(shard),
        "sample_start": start,
        "sample_stop": start + SHARD_SIZE,
        "shard_generator_seed": _seed(config["root_seed"], case["case_id"], shard),
        "canonical_engine_sha256": EXPECTED_ENGINE,
        "audit_sha256": EXPECTED_AUDIT,
    }


def _expected_archive(
    output_root: Path, config: dict[str, Any], case: dict[str, Any], shard: int
) -> Path:
    run_id = f"{config['bundle']}_{sha256_json(_run_config(config, case, shard))[:16]}"
    return output_root / f"{run_id}.tar.gz"


def _validate_manifest(
    manifest: dict[str, Any], config: dict[str, Any], case: dict[str, Any], shard: int
) -> list[str]:
    run_config = manifest.get("run_config", {})
    expected = _run_config(config, case, shard)
    checks = {
        "bundle": manifest.get("bundle") == config["bundle"],
        "status": manifest.get("status") == "complete_local",
        "case_id": manifest.get("case_id") == case["case_id"],
        "shard_index": int(manifest.get("shard_index", -1)) == int(shard),
        "root_seed": int(manifest.get("root_seed", -1)) == int(config["root_seed"]),
        "audit_sha256": manifest.get("audit_sha256") == EXPECTED_AUDIT,
        "case_configuration": run_config.get("case") == case,
        "run_configuration": run_config == expected,
        "global_sample_indices": [
            int(value) for value in manifest.get("global_sample_indices", [])
        ]
        == list(range(shard * SHARD_SIZE, (shard + 1) * SHARD_SIZE)),
    }
    return [name for name, passed in checks.items() if not passed]


def _stream(command: list[str], heartbeat_seconds: float) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    messages: queue.Queue[str | None] = queue.Queue()
    tail: deque[str] = deque(maxlen=200)

    def reader() -> None:
        for line in process.stdout:
            messages.put(line.rstrip("\n"))
        messages.put(None)

    threading.Thread(target=reader, daemon=True).start()
    started = time.monotonic()
    next_heartbeat = started + heartbeat_seconds
    closed = False
    while not closed:
        timeout = max(0.05, min(1.0, next_heartbeat - time.monotonic()))
        try:
            line = messages.get(timeout=timeout)
        except queue.Empty:
            line = ""
        if line is None:
            closed = True
        elif line:
            tail.append(line)
            tqdm.write(line)
        now = time.monotonic()
        if now >= next_heartbeat and process.poll() is None:
            tqdm.write(
                f"[HEARTBEAT] pid={process.pid} elapsed={(now - started) / 60:.1f}m "
                f"last={tail[-1] if tail else '<no child output yet>'}"
            )
            next_heartbeat = now + heartbeat_seconds
    returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"child exited {returncode}: {' '.join(command)}\n" + "\n".join(tail)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="compute missing P1 shards")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive")

    config = load_config(ROOT)
    engine = sha256_file(SRC / "classA_U1FGTN_gpu.py")
    if config.get("audit_sha256") != EXPECTED_AUDIT:
        raise RuntimeError("the frozen P1 audit hash changed")
    if config.get("sampling_revision") != EXPECTED_SAMPLING_REVISION:
        raise RuntimeError("the frozen P1 sampling revision changed")
    if int(config["locked_contract"]["samples"]) != 25:
        raise RuntimeError("the frozen P1 contract is not S=25")
    if engine != EXPECTED_ENGINE:
        raise RuntimeError(f"the frozen P1 engine changed: {engine}")

    cases = [case for case in expand_cases(config) if case["campaign"] == "P1"]
    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} P1 cases, found {len(cases)}")
    tasks = [(case, shard) for case in cases for shard in range(5)]
    if len(tasks) != EXPECTED_SHARDS:
        raise RuntimeError(f"expected {EXPECTED_SHARDS} P1 shards, found {len(tasks)}")

    output_root = (
        args.drive_root.resolve()
        / "classA_final_production_outputs"
        / "01_bulk_width_gate"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    archives = sorted(output_root.glob("*.tar.gz"))
    receipts = sorted(output_root.glob("*.tar.gz.receipt.json"))
    archive_names = {path.name for path in archives}
    orphans = [
        path
        for path in receipts
        if path.name.removesuffix(".receipt.json") not in archive_names
    ]
    if orphans:
        raise RuntimeError(
            f"receipt(s) without archive: {[str(path) for path in orphans]}"
        )

    print(f"[STARTUP] exact P1 completion source: {ROOT}")
    print(f"[STARTUP] output: {output_root}")
    rows: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = {}
    for archive in tqdm(archives, desc="verify existing receipts", unit="archive"):
        verify_archive_receipt(archive)
        manifest = _manifest(archive)
        slot = (str(manifest.get("case_id", "")), int(manifest.get("shard_index", -1)))
        rows.setdefault(slot, []).append((archive, manifest))

    complete: list[tuple[dict[str, Any], int]] = []
    pending: list[tuple[dict[str, Any], int]] = []
    for case, shard in tqdm(tasks, desc="match P1 matrix", unit="shard"):
        slot_rows = rows.get((case["case_id"], shard), [])
        exact = [
            (path, manifest)
            for path, manifest in slot_rows
            if not _validate_manifest(manifest, config, case, shard)
        ]
        if len(exact) > 1:
            raise RuntimeError(
                f"duplicate exact P1 outputs for {case['case_id']} shard {shard}: "
                f"{[str(path) for path, _ in exact]}"
            )
        if exact:
            complete.append((case, shard))
        else:
            mismatches = [
                {
                    "path": str(path),
                    "fields": _validate_manifest(manifest, config, case, shard),
                }
                for path, manifest in slot_rows
            ]
            if mismatches:
                raise RuntimeError(
                    f"identity mismatch at {case['case_id']} shard {shard}: {mismatches}"
                )
            pending.append((case, shard))

    print(f"[SUMMARY] P1: {len(complete)}/{len(tasks)} checksum-verified")
    print(f"[SUMMARY] remaining: {len(pending)} shard(s)")
    if pending:
        print(f"[NEXT] {pending[0][0]['case_id']} shard {pending[0][1]}")
        print(f"[ETA] approximately {len(pending) * 24.9 / 60:.2f} A100 hours")
    if not args.run:
        print(
            "[REPORT ONLY] No numerical work launched. Add --run after reviewing this report."
        )
        return 0

    bar = tqdm(
        total=len(tasks), initial=len(complete), desc="P1 completion", unit="shard"
    )
    base = [
        sys.executable,
        "-u",
        str(ROOT / "run_bundle.py"),
        "--drive-root",
        str(args.drive_root.resolve()),
        "--mode",
        "production",
    ]
    for ordinal, (case, shard) in enumerate(pending, start=len(complete) + 1):
        label = f"{case['case_id']} shard {shard}"
        print(f"[PREFLIGHT] {ordinal}/{len(tasks)} {label}")
        _stream(
            [
                *base,
                "--case-id",
                case["case_id"],
                "--shard-index",
                str(shard),
                "--preflight-only",
            ],
            args.heartbeat_seconds,
        )
        print(f"[RUNNING] {ordinal}/{len(tasks)} {label}")
        _stream(
            [*base, "--case-id", case["case_id"], "--shard-index", str(shard)],
            args.heartbeat_seconds,
        )
        expected_archive = _expected_archive(output_root, config, case, shard)
        verify_archive_receipt(expected_archive)
        manifest = _manifest(expected_archive)
        mismatches = _validate_manifest(manifest, config, case, shard)
        if mismatches:
            raise RuntimeError(
                f"new archive identity mismatch {expected_archive}: {mismatches}"
            )
        print(f"[DONE] {ordinal}/{len(tasks)} {label}; checksum verified")
        bar.update(1)
    bar.close()
    print(f"[COMPLETE] P1 {len(tasks)}/{len(tasks)} verified; W1 was not enumerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
