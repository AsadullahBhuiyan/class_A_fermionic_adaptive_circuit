#!/usr/bin/env python3
"""Benchmark-only gate for a possible future A100 spectral eigensolver port.

This script never changes the production backend.  It emits an approval
receipt only when complex128 GPU projectors agree with the CPU reference and
the measured throughput exceeds the locked two-fold speedup threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import run_wall_diabatic_spectral_pump_s100 as core  # noqa: E402


SCHEMA = "wall_diabatic_a100_spectral_benchmark_v1"
DEFAULT_RECEIPT = core.DEFAULT_OUTPUT / "benchmarks/a100_spectral_port_benchmark.json"


def approved(receipt: dict[str, Any]) -> bool:
    return bool(
        receipt.get("schema") == SCHEMA
        and receipt.get("status") == "approved"
        and receipt.get("dtype") == "complex128"
        and receipt.get("cpu_reference") == "same-host-56-affinity-cpus"
        and int(receipt.get("cpu_threads", 0)) == 56
        and int(receipt.get("available_affinity_cpus", 0)) >= 56
        and float(receipt.get("maximum_eigenvalue_error", np.inf)) <= 1e-10
        and float(receipt.get("maximum_projector_error", np.inf)) <= 1e-10
        and float(receipt.get("throughput_speedup_vs_local56", 0.0)) >= 2.0
    )


def representative_matrices() -> tuple[list[np.ndarray], int, dict[str, Any]]:
    config = core.load_config(core.DEFAULT_CONFIG)
    core.validate_config(config)
    task = next(
        row for row in core.tasks(config, include_bridge=False)
        if row["cell"] == "nsh1_N20x24" and row["wall"] == "soft" and row["sample_id"] == 0
    )
    source = core.source_rows(config, include_bridge=False)[task["cell"]]
    ref = core.endpoint_ref(source, "soft", 0, core.DEFAULT_NEW_ENDPOINT_ROOT)
    frame = core.load_endpoint(ref, 20, 24)
    projector = frame @ frame.conj().T
    h0 = np.eye(frame.shape[0], dtype=np.complex128) - 2.0 * projector
    _, y, dy = core._coordinates(20, 24)
    matrices = [core.twisted_parent(h0, dy, phi, 24, y=y) for phi in (0.37, np.pi, 5.41)]
    return matrices, frame.shape[1], ref.dependency()


def run_benchmark(cpu_threads: int, repeats: int) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"schema": SCHEMA, "status": "cpu_only", "reason": "torch is unavailable"}
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0).upper():
        return {
            "schema": SCHEMA, "status": "cpu_only",
            "reason": "a 40-GB-class A100 CUDA runtime is unavailable",
        }
    available_cpus = len(os.sched_getaffinity(0))
    if int(cpu_threads) != 56 or available_cpus < 56:
        return {
            "schema": SCHEMA,
            "status": "rejected",
            "reason": (
                "the spectral gate requires a same-host 56-affinity-CPU "
                "reference; a smaller Colab CPU is not the local-56 baseline"
            ),
            "cpu_reference": "same-host-56-affinity-cpus",
            "cpu_threads": int(cpu_threads),
            "available_affinity_cpus": int(available_cpus),
            "production_backend_changed": False,
        }
    properties = torch.cuda.get_device_properties(0)
    if int(properties.total_memory) < 38 * 1024**3:
        return {"schema": SCHEMA, "status": "cpu_only", "reason": "A100 memory is below 38 GiB"}
    matrices, rank, dependency = representative_matrices()
    cpu_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    started = time.perf_counter()
    with threadpool_limits(limits=cpu_threads):
        for _ in range(repeats):
            for matrix in matrices:
                values, vectors = np.linalg.eigh(matrix)
                cpu_outputs.append((values, vectors[:, :rank] @ vectors[:, :rank].conj().T))
    cpu_seconds = time.perf_counter() - started
    device = torch.device("cuda:0")
    gpu_matrices = [torch.as_tensor(matrix, dtype=torch.complex128, device=device) for matrix in matrices]
    torch.linalg.eigh(gpu_matrices[0])
    torch.cuda.synchronize(device)
    gpu_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    started = time.perf_counter()
    for _ in range(repeats):
        for matrix in gpu_matrices:
            values, vectors = torch.linalg.eigh(matrix)
            projector = vectors[:, :rank] @ vectors[:, :rank].mH
            gpu_outputs.append((values.cpu().numpy(), projector.cpu().numpy()))
    torch.cuda.synchronize(device)
    gpu_seconds = time.perf_counter() - started
    eigenvalue_error = max(
        float(np.max(np.abs(cpu[0] - gpu[0]))) for cpu, gpu in zip(cpu_outputs, gpu_outputs)
    )
    projector_error = max(
        float(np.linalg.norm(cpu[1] - gpu[1], ord="fro") / np.sqrt(cpu[1].shape[0]))
        for cpu, gpu in zip(cpu_outputs, gpu_outputs)
    )
    speedup = cpu_seconds / gpu_seconds
    receipt = {
        "schema": SCHEMA, "status": "candidate", "dtype": "complex128",
        "gpu_name": torch.cuda.get_device_name(0), "gpu_total_bytes": int(properties.total_memory),
        "matrix_dimension": matrices[0].shape[0], "matrix_count": len(matrices),
        "repeats": repeats, "cpu_threads": cpu_threads,
        "available_affinity_cpus": available_cpus,
        "cpu_reference": "same-host-56-affinity-cpus",
        "cpu_seconds": cpu_seconds, "gpu_seconds": gpu_seconds,
        "throughput_speedup_vs_local56": speedup,
        "maximum_eigenvalue_error": eigenvalue_error,
        "maximum_projector_error": projector_error,
        "endpoint_dependency": dependency,
        "production_backend_changed": False,
    }
    receipt["status"] = "approved" if approved({**receipt, "status": "approved"}) else "rejected"
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "run"), nargs="?", default="report")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--cpu-threads", type=int, default=56)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    if args.stage == "report":
        payload = (
            json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt_path.is_file() else {
                "schema": SCHEMA, "status": "cpu_only",
                "reason": "no A100 benchmark receipt; production remains CPU-only",
            }
        )
    else:
        payload = run_benchmark(args.cpu_threads, args.repeats)
        payload["completed_unix"] = time.time()
        core._atomic_json(receipt_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_approved and not approved(payload):
        raise RuntimeError("A100 spectral port gate is not approved; keep production CPU-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
