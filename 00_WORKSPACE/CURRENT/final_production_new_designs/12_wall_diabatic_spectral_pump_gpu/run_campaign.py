#!/usr/bin/env python3
"""Run only the primary wall-diabatized spectral pump on an A100.

Input endpoint files remain read-only on Drive.  Each result is computed in
local scratch, validated against the unchanged CPU schema, copied to a Drive
temporary path, read back and checksummed, then atomically renamed.  The small
completion JSON is published last.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

# Prevent concurrent endpoint lanes from multiplying host BLAS threads.
for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
import torch
from tqdm.auto import tqdm

import spectral_cpu_reference as core
import gpu_backend


BUNDLE_ROOT = Path(__file__).resolve().parent
REVISION = "wall_diabatic_spectral_pump_gpu_primary_v1"
BENCHMARK_SCHEMA = "wall_diabatic_spectral_pump_a100_gate_v1"
CONCURRENCY_SCHEMA = "wall_diabatic_spectral_pump_a100_concurrency_v1"
SOURCE_FILES = ("run_campaign.py", "gpu_backend.py", "spectral_cpu_reference.py", "campaign_config.json")
CONCURRENCY_CANDIDATES = (1, 2, 4, 8)
MINIMUM_GPU_HEADROOM_BYTES = 8 * 1024**3


def hashes() -> dict[str, str]:
    return {name: core.sha256_path(BUNDLE_ROOT / name) for name in SOURCE_FILES}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_from_scratch(scratch_root: Path, drive_root: Path, task: dict[str, Any]) -> None:
    local_result, local_completion = core.result_paths(scratch_root, task)
    drive_result, drive_completion = core.result_paths(drive_root, task)
    drive_result.parent.mkdir(parents=True, exist_ok=True)
    temporary = drive_result.with_name(f".{drive_result.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(local_result, temporary)
        if temporary.stat().st_size != local_result.stat().st_size:
            raise IOError("Drive readback byte count differs from local result")
        if core.sha256_path(temporary) != core.sha256_path(local_result):
            raise IOError("Drive readback checksum differs from local result")
        os.replace(temporary, drive_result)
        completion_payload = json.loads(local_completion.read_text(encoding="utf-8"))
        atomic_json(drive_completion, completion_payload)
        if core.sha256_path(drive_result) != completion_payload["result"]["sha256"]:
            raise IOError("stable Drive result changed after publication")
    finally:
        temporary.unlink(missing_ok=True)


def discover_cached(
    rows: list[dict[str, Any]], config: dict[str, Any], new_root: Path
) -> tuple[dict[str, core.EndpointRef], dict[str, str]]:
    sources = core.source_rows(config, include_bridge=True)
    refs: dict[str, core.EndpointRef] = {}
    missing: dict[str, str] = {}
    cache: dict[tuple[str, str, int], core.EndpointRef] = {}
    for task in rows:
        source = sources[task["cell"]]
        sample = int(task["sample_id"])
        member = sample % 5 if source["kind"] == "five_sample_shards" else 0
        key = (task["cell"], task["wall"], sample - member)
        try:
            base = cache.get(key)
            if base is None:
                base = core.endpoint_ref(source, task["wall"], sample, new_root)
                cache[key] = base
            ref = base
            if source["kind"] == "five_sample_shards":
                ref = core.EndpointRef(
                    base.frame_path, base.completion_path, member,
                    base.result_sha256, base.result_bytes, base.completion_sha256,
                    base.source_config_hash, base.source_schema,
                )
            refs[task["task_id"]] = ref
        except Exception as exc:
            missing[task["task_id"]] = f"{type(exc).__name__}: {exc}"
    return refs, missing


def parity_gate(
    config: dict[str, Any], task: dict[str, Any], ref: core.EndpointRef,
    receipt_path: Path, gpu_info: dict[str, Any], force: bool,
) -> dict[str, Any]:
    identity = {
        "schema": BENCHMARK_SCHEMA,
        "revision": REVISION,
        "config_hash": core.scientific_config_hash(config),
        "source_hashes": hashes(),
        "endpoint_dependency": ref.dependency(),
    }
    if receipt_path.is_file() and not force:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if all(receipt.get(key) == value for key, value in identity.items()) and receipt.get("status") == "approved":
            print("[benchmark] reusing verified A100 parity receipt", flush=True)
            return receipt

    frame = core.load_endpoint(ref, int(task["Nx"]), int(task["Ny"]))
    projector = frame @ frame.conj().T
    h0 = np.eye(frame.shape[0], dtype=np.complex128) - 2.0 * projector
    _, y, dy = core._coordinates(int(task["Nx"]), int(task["Ny"]))
    points = (0.37, np.pi, 5.41)
    cpu, gpu = [], []
    original_parent = core._parent_eigensystem
    for phi in points:
        cpu.append(original_parent(h0, dy, phi, int(task["Ny"]), y, "uniform"))
    gpu_backend.install(core)
    torch.cuda.reset_peak_memory_stats()
    gpu_backend.synchronize()
    started = time.perf_counter()
    for phi in points:
        gpu.append(core._parent_eigensystem(h0, dy, phi, int(task["Ny"]), y, "uniform"))
    gpu_backend.synchronize()
    seconds = time.perf_counter() - started
    rank = frame.shape[1]
    eigenvalue_error = max(float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(cpu, gpu))
    projector_errors = []
    for index, (cpu_pair, gpu_pair) in enumerate(zip(cpu, gpu)):
        # At phi=pi the occupied/empty wall pair is intentionally almost
        # degenerate.  Its two-dimensional cluster projector is stable even
        # when the arbitrary basis or rank-cut member is not.
        selected = slice(rank - 1, rank + 1) if index == 1 else slice(0, rank)
        cpu_vectors, gpu_vectors = cpu_pair[1][:, selected], gpu_pair[1][:, selected]
        projector_errors.append(float(np.linalg.norm(
            cpu_vectors @ cpu_vectors.conj().T
            - gpu_vectors @ gpu_vectors.conj().T,
            ord="fro",
        ) / np.sqrt(frame.shape[0])))
    projector_error = max(projector_errors)
    receipt = {
        **identity,
        "status": "approved" if eigenvalue_error <= 1e-10 and projector_error <= 1e-10 else "rejected",
        "gpu": gpu_info,
        "dtype": "complex128",
        "matrix_dimension": frame.shape[0],
        "matrix_count": len(points),
        "gpu_seconds": seconds,
        "seconds_per_full_eigh_at_benchmark_dimension": seconds / len(points),
        "indicative_primary_eigh_hours_at_benchmark_dimension": (
            (seconds / len(points)) * 1544 * 1600 / 3600
        ),
        "projection_warning": (
            "eigensolver-only lower-bound at the benchmark dimension; larger Nx, "
            "sequential continuation, transfers, SVDs, and Drive I/O are excluded"
        ),
        "maximum_eigenvalue_error": eigenvalue_error,
        "maximum_stable_subspace_projector_error": projector_error,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "completed_unix": time.time(),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if receipt["status"] != "approved":
        raise RuntimeError("A100 complex128 parity gate failed; production was not started")
    return receipt


def _eigh_probe_worker(payload: tuple[int, int, int]) -> dict[str, Any]:
    dimension, repeats, seed = payload
    info = gpu_backend.require_a100()
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    real = torch.randn((dimension, dimension), dtype=torch.float64, device="cuda:0")
    imag = torch.randn((dimension, dimension), dtype=torch.float64, device="cuda:0")
    matrix = torch.complex(real, imag)
    matrix = 0.5 * (matrix + matrix.mH)
    torch.linalg.eigh(matrix)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(int(repeats)):
        torch.linalg.eigh(matrix)
    torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "gpu": info,
    }


def concurrency_gate(
    receipt_path: Path,
    *,
    dimension: int,
    config_hash: str,
    source_hashes: dict[str, str],
    gpu_info: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    identity = {
        "schema": CONCURRENCY_SCHEMA,
        "revision": REVISION,
        "config_hash": config_hash,
        "source_hashes": source_hashes,
        "matrix_dimension": int(dimension),
    }
    if receipt_path.is_file() and not force:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if all(receipt.get(key) == value for key, value in identity.items()) and receipt.get("status") == "approved":
            print(
                f"[concurrency] reusing selected lane count={receipt['selected_lanes']}",
                flush=True,
            )
            return receipt

    context = mp.get_context("spawn")
    candidates: list[dict[str, Any]] = []
    for lanes in CONCURRENCY_CANDIDATES:
        print(f"[concurrency] benchmarking {lanes} simultaneous A100 lanes", flush=True)
        try:
            with ProcessPoolExecutor(max_workers=lanes, mp_context=context) as pool:
                futures = [
                    pool.submit(_eigh_probe_worker, (dimension, 2, 2026091000 + index))
                    for index in range(lanes)
                ]
                results = [future.result() for future in futures]
            wall_seconds = max(float(row["seconds"]) for row in results)
            total_reserved = sum(int(row["reserved_bytes"]) for row in results)
            throughput = (2.0 * lanes) / wall_seconds
            headroom = int(gpu_info["total_memory_bytes"]) - total_reserved
            candidates.append({
                "lanes": lanes,
                "status": "accepted" if headroom >= MINIMUM_GPU_HEADROOM_BYTES else "rejected_headroom",
                "throughput_eigh_per_second": throughput,
                "maximum_worker_seconds": wall_seconds,
                "total_reserved_bytes": total_reserved,
                "estimated_headroom_bytes": headroom,
                "maximum_worker_peak_allocated_bytes": max(
                    int(row["peak_allocated_bytes"]) for row in results
                ),
            })
        except Exception as exc:
            candidates.append({
                "lanes": lanes,
                "status": "rejected_error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            torch.cuda.empty_cache()
    accepted = [row for row in candidates if row["status"] == "accepted"]
    if not accepted:
        receipt = {
            **identity, "status": "rejected", "gpu": gpu_info,
            "candidates": candidates, "completed_unix": time.time(),
        }
        atomic_json(receipt_path, receipt)
        raise RuntimeError("no A100 concurrency candidate retained 8 GiB headroom")
    selected = max(accepted, key=lambda row: float(row["throughput_eigh_per_second"]))
    single = next(row for row in accepted if int(row["lanes"]) == 1)
    receipt = {
        **identity,
        "status": "approved",
        "gpu": gpu_info,
        "dtype": "complex128",
        "candidates": candidates,
        "selected_lanes": int(selected["lanes"]),
        "measured_eigh_throughput_gain_vs_one_lane": (
            float(selected["throughput_eigh_per_second"])
            / float(single["throughput_eigh_per_second"])
        ),
        "completed_unix": time.time(),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return receipt


class _BatchedProgressSink:
    def __init__(self, task_id: str, progress_queue: Any, stride: int = 16) -> None:
        self.task_id = task_id
        self.progress_queue = progress_queue
        self.stride = int(stride)
        self.pending = 0

    def put(self, count: int) -> None:
        self.pending += int(count)
        if self.pending >= self.stride:
            self.progress_queue.put((self.task_id, self.pending))
            self.pending = 0

    def flush(self) -> None:
        if self.pending:
            self.progress_queue.put((self.task_id, self.pending))
            self.pending = 0


def _gpu_task_worker(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    started = time.perf_counter()
    sink = _BatchedProgressSink(task["task_id"], payload["progress_queue"])
    try:
        torch.set_num_threads(1)
        gpu_backend.require_a100()
        gpu_backend.install(core)
        ref = payload["ref"]
        frame = core.load_endpoint(ref, int(task["Nx"]), int(task["Ny"]))
        arrays = core.compute_wall_diabatic_pump(
            frame, task, payload["config"], queue=sink
        )
        sink.flush()
        gpu_backend.synchronize()
        elapsed = time.perf_counter() - started
        scratch_root = Path(payload["scratch_root"])
        core.publish_pair(
            scratch_root, task, arrays, payload["config_hash"],
            payload["source_hashes"], ref, elapsed,
        )
        ok, reason, _ = core.verify_pair(
            scratch_root, task, payload["config"], payload["config_hash"],
            payload["source_hashes"], ref,
        )
        if not ok:
            raise RuntimeError(f"local result verification failed: {reason}")
        return {"ok": True, "task": task, "elapsed_seconds": elapsed}
    except BaseException as exc:
        sink.flush()
        return {
            "ok": False,
            "task": task,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE_ROOT / "campaign_config.json")
    parser.add_argument("--legacy-endpoint-root", type=Path, required=True)
    parser.add_argument("--new-endpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/content/wall_spectral_gpu_scratch"))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--rerun-benchmark", action="store_true")
    parser.add_argument("--include-bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-batches", type=int)
    parser.add_argument(
        "--lanes", type=int,
        help="override the benchmark-selected lane count for a bounded diagnostic run",
    )
    args = parser.parse_args()

    config = core.load_config(args.config.resolve())
    core.validate_config(config)
    core.PROJECT_ROOT = args.legacy_endpoint_root.resolve()
    core.SOURCE_PATHS = {
        "gpu_campaign_runner": (BUNDLE_ROOT / "run_campaign.py").resolve(),
        "gpu_backend": (BUNDLE_ROOT / "gpu_backend.py").resolve(),
        "cpu_reference": (BUNDLE_ROOT / "spectral_cpu_reference.py").resolve(),
    }
    output_root, scratch_root = args.output_root.resolve(), args.scratch_root.resolve()
    rows = core.tasks(config, include_bridge=bool(args.include_bridge))
    for task in rows:
        task["execution_backend"] = "torch_cuda_a100_complex128"
        task["gpu_campaign_revision"] = REVISION
    refs, missing = discover_cached(rows, config, args.new_endpoint_root.resolve())
    source_hashes, config_hash = core.source_hashes(), core.scientific_config_hash(config)
    verified, pending, invalid = [], [], {}
    for task in rows:
        ref = refs.get(task["task_id"])
        if ref is None:
            continue
        ok, reason, _ = core.verify_pair(output_root, task, config, config_hash, source_hashes, ref)
        (verified if ok else pending).append(task)
        if not ok and "missing result/completion" not in reason:
            invalid[task["task_id"]] = reason

    print(f"[campaign] {REVISION}")
    print(f"[workload] tasks={len(rows)} verified={len(verified)} pending={len(pending)}")
    print(f"[sources] available={len(refs)}/{len(rows)} unavailable={len(missing)}")
    print(f"[bridge] included={bool(args.include_bridge)}")
    print(f"[legacy endpoints] {core.PROJECT_ROOT}")
    print(f"[new/bridge endpoints] {args.new_endpoint_root.resolve()}")
    print(f"[output] {output_root}")
    print(f"[scratch] {scratch_root}")
    if missing:
        for task_id, reason in list(missing.items())[:12]:
            print(f"[missing] {task_id}: {reason}")
        if len(missing) > 12:
            print(f"[missing] ... and {len(missing) - 12} more")
    if invalid:
        print(f"[warning] {len(invalid)} existing outputs failed verification and will rerun")
    if args.report_only:
        return 0
    if missing:
        raise RuntimeError("endpoint preflight failed; production was not started")

    gpu_info = gpu_backend.require_a100()
    benchmark_task = next(
        task for task in rows
        if int(task["Nx"]) == 32 and task["task_id"] in refs
    )
    parity_gate(
        config, benchmark_task, refs[benchmark_task["task_id"]],
        output_root / "benchmarks/a100_complex128_parity.json",
        gpu_info, args.rerun_benchmark,
    )
    concurrency = concurrency_gate(
        output_root / "benchmarks/a100_concurrency.json",
        dimension=2 * int(benchmark_task["Nx"]) * int(benchmark_task["Ny"]),
        config_hash=config_hash,
        source_hashes=source_hashes,
        gpu_info=gpu_info,
        force=args.rerun_benchmark,
    )
    if args.benchmark_only:
        print(
            "[benchmark-only] parity and concurrency gates passed; "
            "no production task was run",
            flush=True,
        )
        return 0

    core.write_identity(config, output_root)
    atomic_json(output_root / "gpu_campaign_identity.json", {
        "schema": "wall_diabatic_spectral_pump_gpu_identity_v1",
        "revision": REVISION,
        "execution_backend": "torch_cuda_a100_complex128",
        "gpu": gpu_info,
        "dtype": "complex128",
        "config_hash": config_hash,
        "source_hashes": source_hashes,
        "sensitivity_included": False,
        "selected_lanes": int(args.lanes or concurrency["selected_lanes"]),
    })
    lanes = int(args.lanes or concurrency["selected_lanes"])
    if lanes not in CONCURRENCY_CANDIDATES:
        raise ValueError(f"lanes must be one of {CONCURRENCY_CANDIDATES}")
    calibration_path = output_root / "benchmarks/full_batch_timing.json"
    calibration_identity = {
        "schema": "wall_diabatic_spectral_pump_full_batch_timing_v1",
        "revision": REVISION,
        "config_hash": config_hash,
        "source_hashes": source_hashes,
        "lanes": lanes,
    }
    if args.max_new_batches is None:
        calibration = (
            json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration_path.is_file() else {}
        )
        if not all(calibration.get(key) == value for key, value in calibration_identity.items()):
            raise RuntimeError(
                "run one bounded worst-case batch with --max-new-batches 1 before "
                "unlimited production"
            )
        if not bool(calibration.get("all_tasks_under_one_hour")):
            raise RuntimeError(
                "the worst-case batch exceeded one hour per task; sub-task rolling "
                "checkpoints are required before unlimited production"
            )
    run_limit = None if args.max_new_batches is None else lanes * int(args.max_new_batches)
    if run_limit is None:
        run_rows = pending
    else:
        # Exercise the largest matrices first so the bounded batch is a real
        # checkpoint-safety and throughput gate, not an optimistic small-N run.
        run_rows = sorted(
            pending,
            key=lambda task: (-int(task["Nx"]), task["task_id"]),
        )[:run_limit]
    print(
        f"[launch] lanes={lanes} new_tasks={len(run_rows)} "
        f"batches={(len(run_rows) + lanes - 1) // lanes}",
        flush=True,
    )
    if not run_rows:
        print("[done] all selected tasks already verify", flush=True)
        return 0

    context = mp.get_context("spawn")
    failures: list[dict[str, Any]] = []
    elapsed_rows: list[float] = []
    batch_started = time.perf_counter()
    with context.Manager() as manager:
        progress_queue = manager.Queue()
        point_total = sum(4 * (int(task["grid_intervals"]) + 1) for task in run_rows)
        with tqdm(total=len(run_rows), desc="A100 endpoint batches", unit="task", position=0) as task_bar, \
                tqdm(total=point_total, desc="continued flux points", unit="point", position=1) as point_bar, \
                ProcessPoolExecutor(max_workers=lanes, mp_context=context) as pool:
            future_map = {
                pool.submit(_gpu_task_worker, {
                    "task": task,
                    "ref": refs[task["task_id"]],
                    "config": config,
                    "config_hash": config_hash,
                    "source_hashes": source_hashes,
                    "scratch_root": str(scratch_root),
                    "progress_queue": progress_queue,
                }): task
                for task in run_rows
            }
            remaining = set(future_map)
            while remaining:
                done, remaining = wait(
                    remaining, timeout=0.5, return_when=FIRST_COMPLETED
                )
                while True:
                    try:
                        _task_id, increment = progress_queue.get_nowait()
                    except Exception:
                        break
                    point_bar.update(int(increment))
                for future in done:
                    result = future.result()
                    task = result["task"]
                    if not result["ok"]:
                        failures.append(result)
                        atomic_json(core.failure_path(output_root, task), {
                            "task_id": task["task_id"],
                            "failed_unix": time.time(),
                            "error": result["error"],
                            "traceback": result["traceback"],
                        })
                        task_bar.write(f"[failure] {task['task_id']}: {result['error']}")
                        task_bar.update(1)
                        continue
                    ref = refs[task["task_id"]]
                    publish_from_scratch(scratch_root, output_root, task)
                    ok, reason, _ = core.verify_pair(
                        output_root, task, config, config_hash, source_hashes, ref
                    )
                    if not ok:
                        raise RuntimeError(f"Drive result verification failed: {reason}")
                    local_result, local_completion = core.result_paths(scratch_root, task)
                    local_result.unlink(missing_ok=True)
                    local_completion.unlink(missing_ok=True)
                    core.failure_path(output_root, task).unlink(missing_ok=True)
                    elapsed = float(result["elapsed_seconds"])
                    elapsed_rows.append(elapsed)
                    task_bar.write(
                        f"[complete] {task['task_id']} elapsed_min={elapsed / 60:.2f}"
                    )
                    task_bar.update(1)
            while True:
                try:
                    _task_id, increment = progress_queue.get_nowait()
                except Exception:
                    break
                point_bar.update(int(increment))
    if failures:
        raise RuntimeError(
            f"{len(failures)} task/boundary failures; inspect the output failures directory"
        )
    batch_wall_seconds = time.perf_counter() - batch_started
    if args.max_new_batches is not None:
        timing = {
            **calibration_identity,
            "task_ids": [task["task_id"] for task in run_rows],
            "Nx_values": sorted({int(task["Nx"]) for task in run_rows}),
            "task_elapsed_seconds": elapsed_rows,
            "maximum_task_seconds": max(elapsed_rows),
            "batch_wall_seconds": batch_wall_seconds,
            "tasks_per_hour": len(run_rows) * 3600.0 / batch_wall_seconds,
            "projected_1600_task_hours": (
                1600.0 * batch_wall_seconds / (len(run_rows) * 3600.0)
            ),
            "all_tasks_under_one_hour": max(elapsed_rows) <= 3600.0,
            "completed_unix": time.time(),
        }
        atomic_json(calibration_path, timing)
        print("[full-batch timing]", json.dumps(timing, indent=2, sort_keys=True), flush=True)
    print(f"[done] completed {len(run_rows)} new tasks; all publications verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
