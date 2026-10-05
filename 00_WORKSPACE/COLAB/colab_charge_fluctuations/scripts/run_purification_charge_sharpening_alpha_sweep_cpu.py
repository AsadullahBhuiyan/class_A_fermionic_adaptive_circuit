#!/usr/bin/env python3
"""Run the CPU purification charge-sharpening alpha sweep.

The canonical CPU Markov-circuit entry point owns all dynamics. This runner
parallelizes independent trajectory blocks, streams scalar observables through
``cycle_observer``, and writes resumable per-trajectory checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SRC_ROOT_TEXT = str(SRC_ROOT)
if SRC_ROOT_TEXT in sys.path:
    sys.path.remove(SRC_ROOT_TEXT)
sys.path.insert(0, SRC_ROOT_TEXT)

from fgtn.classA_U1FGTN import classA_U1FGTN


CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
CAMPAIGN_NAME = "purification_charge_sharpening_alpha_sweep"
DEFAULT_CAMPAIGN_ID = "N16_alpha-sweep_nsh1_dwtrunc1_init-maxmix_S10_cycles-2Ny"
PROTOCOLS = ("perfect_correction", "postselection")
NY_VALUES = (16, 24, 32)
ALPHA_VALUES = (1.0, 1.5, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0)
NX = 16
ALPHA_TRIVIAL = 30.0
NSHELL = 1
PERFECT_CORRECTION_SAMPLES = 10
SHARPENING_THRESHOLD = 1e-2
WORKER_MEMORY_ESTIMATE_BYTES = 768 * 1024**2


@dataclass(frozen=True)
class TrajectorySpec:
    protocol: str
    nx: int
    ny: int
    alpha_topological_region: float
    alpha_trivial_region: float
    dw_truncation: bool
    cycles: int
    sample_index: int
    seed: int
    checkpoint_path: str

    @property
    def config_id(self) -> str:
        alpha = format_alpha(self.alpha_topological_region)
        return f"N{self.nx}x{self.ny}_alpha1-{alpha}_{self.protocol}"

    @property
    def observable_region(self) -> str:
        return "domain_wall_slab" if self.dw_truncation else "full_top_layer"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_alpha(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json_atomic(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(jsonable(payload), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def save_npz_atomic(path: Path | str, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def save_dataframe_atomic_csv(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_csv(temp_name, index=False)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_checkpoint(path: Path | str, expected_cycles: int | None = None) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key] for key in payload.files}
    if expected_cycles is not None:
        expected = (int(expected_cycles),)
        for key in ("cycles", "total_entropy", "total_charge_variance"):
            if result[key].shape != expected:
                raise ValueError(f"{path}: {key} has shape {result[key].shape}, expected {expected}.")
    return result


def trajectory_observables(G: np.ndarray, active_indices: np.ndarray, entropy_eps: float = 1e-12) -> tuple[float, float]:
    active = np.asarray(G[np.ix_(active_indices, active_indices)], dtype=np.complex128)
    occupation = 0.5 * (active + np.eye(active.shape[0], dtype=np.complex128))
    occupation = 0.5 * (occupation + occupation.conj().T)
    eigvals = np.linalg.eigvalsh(occupation)
    if not np.all(np.isfinite(eigvals)):
        raise FloatingPointError("Occupation eigenvalues contain non-finite values.")
    clipped = np.clip(eigvals, entropy_eps, 1.0 - entropy_eps)
    entropy = -float(np.sum(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)))
    charge_variance = float(np.real(np.trace(occupation)) - np.sum(np.abs(occupation) ** 2))
    if charge_variance < -1e-7:
        raise FloatingPointError(f"Total charge variance is unexpectedly negative: {charge_variance:.6e}")
    return entropy, max(0.0, charge_variance)


def projector_diagnostics(model: classA_U1FGTN, alpha: float) -> dict[str, Any]:
    kx = 2.0 * np.pi * np.fft.fftfreq(model.Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(model.Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    nmag = np.sqrt(np.sin(KX) ** 2 + np.sin(KY) ** 2 + (float(alpha) - np.cos(KX) - np.cos(KY)) ** 2)
    arrays = [model.WF_Ap, model.WF_Bp, model.WF_Am, model.WF_Bm]
    norms = np.concatenate([np.linalg.norm(arr, axis=0).reshape(-1) for arr in arrays])
    return {
        "alpha_topological_region": float(alpha),
        "critical_point_alpha_equals_2": bool(np.isclose(alpha, 2.0)),
        "minimum_bloch_norm": float(np.min(nmag)),
        "projectors_all_finite": bool(all(np.all(np.isfinite(arr)) for arr in arrays)),
        "projector_center_norm_min": float(np.min(norms)),
        "projector_center_norm_max": float(np.max(norms)),
    }


def worker_initialize(affinity: list[int]) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu) for cpu in affinity))


def run_trajectory_block(spec_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    specs = [TrajectorySpec(**payload) for payload in spec_payloads]
    first = specs[0]
    if any(
        (spec.protocol, spec.nx, spec.ny, spec.alpha_topological_region, spec.dw_truncation)
        != (first.protocol, first.nx, first.ny, first.alpha_topological_region, first.dw_truncation)
        for spec in specs
    ):
        raise ValueError("Every trajectory block must contain one protocol/configuration.")

    model = classA_U1FGTN(
        first.nx,
        first.ny,
        DW=True,
        nshell=NSHELL,
        alpha_1=first.alpha_topological_region,
        alpha_2=first.alpha_trivial_region,
        trial_orbitals="X",
        dw_truncation=first.dw_truncation,
    )
    model.construct_OW_projectors(
        nshell=NSHELL,
        DW=True,
        trial_orbitals="X",
        dw_truncation=first.dw_truncation,
    )
    active_indices = model.active_top_layer_indices(meas_slab_only=True)
    diagnostics = projector_diagnostics(model, first.alpha_topological_region)
    results: list[dict[str, Any]] = []

    with threadpool_limits(limits=1):
        for spec in specs:
            checkpoint_path = Path(spec.checkpoint_path)
            if checkpoint_path.exists():
                load_checkpoint(checkpoint_path, spec.cycles)
                results.append(
                    {
                        "config_id": spec.config_id,
                        "sample_index": spec.sample_index,
                        "seed": spec.seed,
                        "checkpoint_path": str(checkpoint_path),
                        "status": "skipped",
                    }
                )
                continue

            entropy = np.full(spec.cycles, np.nan, dtype=np.float64)
            charge_variance = np.full(spec.cycles, np.nan, dtype=np.float64)

            def observe(*, cycle: int, G: np.ndarray, **_: Any) -> None:
                if int(cycle) == 0:
                    return
                s_total, q_var = trajectory_observables(G, active_indices)
                entropy[int(cycle) - 1] = s_total
                charge_variance[int(cycle) - 1] = q_var

            np.random.seed(int(spec.seed) & 0xFFFFFFFF)
            trajectory_started = time.perf_counter()
            model.run_markov_circuit(
                G_history=False,
                progress=False,
                cycles=spec.cycles,
                postselect=spec.protocol == "postselection",
                perfect_correction=spec.protocol == "perfect_correction",
                samples=1,
                parallelize_samples=False,
                init_mode="maxmix",
                save=False,
                n_a=0.5,
                sequence="raster_y",
                meas_slab_only=True,
                cycle_observer=observe,
            )
            if not np.all(np.isfinite(entropy)) or not np.all(np.isfinite(charge_variance)):
                raise FloatingPointError(f"{spec.config_id} sample {spec.sample_index}: incomplete observables.")
            elapsed = time.perf_counter() - trajectory_started
            save_npz_atomic(
                checkpoint_path,
                cycles=np.arange(1, spec.cycles + 1, dtype=np.int64),
                total_entropy=entropy,
                total_charge_variance=charge_variance,
                sample_index=np.asarray(spec.sample_index, dtype=np.int64),
                seed=np.asarray(spec.seed, dtype=np.uint32),
                elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
                protocol=np.asarray(spec.protocol),
                config_id=np.asarray(spec.config_id),
                dw_truncation=np.asarray(spec.dw_truncation),
                observable_region=np.asarray(spec.observable_region),
                canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY_POINT),
            )
            results.append(
                {
                    "config_id": spec.config_id,
                    "sample_index": spec.sample_index,
                    "seed": spec.seed,
                    "checkpoint_path": str(checkpoint_path),
                    "elapsed_seconds": elapsed,
                    "status": "completed",
                }
            )

    process = psutil.Process()
    return {
        "results": results,
        "diagnostics": diagnostics,
        "worker_pid": os.getpid(),
        "worker_rss_bytes": process.memory_info().rss,
        "block_elapsed_seconds": time.perf_counter() - started,
    }


def throughput_probe(repetitions: int) -> float:
    rng = np.random.default_rng(8128 + int(repetitions))
    matrix = rng.standard_normal((128, 128))
    matrix = 0.5 * (matrix + matrix.T)
    checksum = 0.0
    with threadpool_limits(limits=1):
        for _ in range(int(repetitions)):
            checksum += float(np.linalg.eigvalsh(matrix)[0])
            matrix[0, 0] += 1e-12
    return checksum


def parse_alpha_csv(values: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in values.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse comma-separated alpha values: {values!r}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("--alpha-values must contain at least one float.")
    if not all(np.isfinite(parsed)):
        raise argparse.ArgumentTypeError("--alpha-values must be finite floats.")
    return parsed


def resolve_alpha_values(args: argparse.Namespace) -> tuple[float, ...]:
    if args.alpha_values is not None and args.alpha_linspace is not None:
        raise ValueError("Use either --alpha-values or --alpha-linspace, not both.")
    if args.alpha_values is not None:
        return tuple(float(value) for value in args.alpha_values)
    if args.alpha_linspace is not None:
        start, stop, count = args.alpha_linspace
        count_int = int(count)
        if count_int < 2:
            raise ValueError("--alpha-linspace COUNT must be at least 2.")
        return tuple(float(value) for value in np.linspace(float(start), float(stop), count_int))
    return tuple(float(value) for value in ALPHA_VALUES)


def sample_cpu_usage(
    available_cpus: list[int],
    *,
    samples: int,
    interval: float,
) -> dict[str, Any]:
    samples = max(1, int(samples))
    interval = max(0.05, float(interval))
    usage_rows = []
    psutil.cpu_percent(interval=None, percpu=True)
    for _ in range(samples):
        usage = psutil.cpu_percent(interval=interval, percpu=True)
        usage_rows.append([float(usage[cpu]) for cpu in available_cpus])
    usage_array = np.asarray(usage_rows, dtype=np.float64)
    return {
        "samples": samples,
        "interval_seconds": interval,
        "available_cpus": available_cpus,
        "per_cpu_average_percent": {
            str(cpu): float(usage_array[:, index].mean()) for index, cpu in enumerate(available_cpus)
        },
        "per_cpu_max_percent": {
            str(cpu): float(usage_array[:, index].max()) for index, cpu in enumerate(available_cpus)
        },
    }


def resolve_cpu_pool(args: argparse.Namespace, available_cpus: list[int]) -> tuple[list[int], dict[str, Any]]:
    if args.cpu_selection == "highest":
        return available_cpus, {
            "cpu_selection": "highest",
            "scheduler_affinity": available_cpus,
        }

    usage = sample_cpu_usage(
        available_cpus,
        samples=args.cpu_free_samples,
        interval=args.cpu_free_sample_interval,
    )
    threshold = float(args.cpu_free_threshold)
    max_by_cpu = {int(cpu): float(value) for cpu, value in usage["per_cpu_max_percent"].items()}
    free_cpus = [cpu for cpu in available_cpus if max_by_cpu[cpu] <= threshold]
    if len(free_cpus) < int(args.min_free_workers):
        raise RuntimeError(
            f"Only {len(free_cpus)} CPUs are below {threshold:g}% usage; "
            f"refusing to launch because --min-free-workers={args.min_free_workers}."
        )
    return free_cpus, {
        "cpu_selection": "free",
        "scheduler_affinity": available_cpus,
        "free_usage_threshold_percent": threshold,
        "min_free_workers": int(args.min_free_workers),
        "free_cpus": free_cpus,
        "busy_cpus": [cpu for cpu in available_cpus if cpu not in set(free_cpus)],
        "usage_sampling": usage,
    }


def benchmark_parallel_throughput(candidate: int, affinity: list[int]) -> dict[str, Any]:
    started = time.perf_counter()
    context = get_context("spawn")
    task_count = max(candidate, 12)
    with ProcessPoolExecutor(
        max_workers=candidate,
        mp_context=context,
        initializer=worker_initialize,
        initargs=(affinity,),
    ) as executor:
        list(executor.map(throughput_probe, [3] * task_count))
    elapsed = time.perf_counter() - started
    return {
        "workers": candidate,
        "tasks": task_count,
        "elapsed_seconds": elapsed,
        "throughput_tasks_per_second": task_count / elapsed,
    }


def choose_parallelism(args: argparse.Namespace, available_cpus: list[int]) -> tuple[int, int, dict[str, Any]]:
    available_memory = int(psutil.virtual_memory().available)
    memory_cap = max(1, int((0.70 * available_memory) // WORKER_MEMORY_ESTIMATE_BYTES))
    hard_cap = max(1, min(len(available_cpus), memory_cap, int(args.max_workers)))

    if args.workers != "auto":
        workers = max(1, min(int(args.workers), hard_cap))
        block_size = int(args.block_size) if args.block_size != "auto" else 5
        return workers, block_size, {
            "mode": "explicit_workers",
            "available_memory_bytes": available_memory,
            "estimated_worker_memory_bytes": WORKER_MEMORY_ESTIMATE_BYTES,
            "memory_worker_cap": memory_cap,
            "hard_worker_cap": hard_cap,
        }

    candidates = sorted({value for value in (10, 16, 24, 32, 40, 48, hard_cap) if 1 <= value <= hard_cap})
    if hard_cap < 10:
        candidates = [hard_cap]
    benchmark_rows = []
    for candidate in tqdm(candidates, desc="worker calibration", unit="candidate"):
        benchmark_rows.append(benchmark_parallel_throughput(candidate, available_cpus))
    peak = max(row["throughput_tasks_per_second"] for row in benchmark_rows)
    near_peak = [row["workers"] for row in benchmark_rows if row["throughput_tasks_per_second"] >= 0.95 * peak]
    workers = max(near_peak)

    if args.block_size == "auto":
        block_benchmarks = []
        for block_size in (1, 2, 5):
            started = time.perf_counter()
            throughput_probe(block_size + 2)
            elapsed = time.perf_counter() - started
            block_benchmarks.append(
                {
                    "block_size": block_size,
                    "elapsed_seconds": elapsed,
                    "trajectories_per_second": block_size / elapsed,
                }
            )
        block_peak = max(row["trajectories_per_second"] for row in block_benchmarks)
        block_size = max(
            row["block_size"]
            for row in block_benchmarks
            if row["trajectories_per_second"] >= 0.95 * block_peak
        )
    else:
        block_size = int(args.block_size)
        block_benchmarks = []

    return workers, block_size, {
        "mode": "auto_benchmark",
        "available_memory_bytes": available_memory,
        "estimated_worker_memory_bytes": WORKER_MEMORY_ESTIMATE_BYTES,
        "memory_worker_cap": memory_cap,
        "hard_worker_cap": hard_cap,
        "worker_benchmarks": benchmark_rows,
        "block_benchmarks": block_benchmarks,
        "selection_rule": "highest count within 5% of peak throughput and below 70% memory cap",
    }


def config_dir(output_root: Path, campaign_id: str, spec: TrajectorySpec) -> Path:
    return output_root / spec.protocol / "campaigns" / campaign_id / "runs" / spec.config_id


def build_specs(args: argparse.Namespace, output_root: Path, campaign_id: str) -> list[TrajectorySpec]:
    ny_values = (4,) if args.smoke else NY_VALUES
    alpha_values = tuple(float(value) for value in args.resolved_alpha_values)
    nx = 4 if args.smoke else NX
    sample_count = 2 if args.smoke else PERFECT_CORRECTION_SAMPLES
    specs = []
    for protocol in args.protocols:
        for ny in ny_values:
            cycles = 2 if args.smoke else 2 * ny
            for alpha in alpha_values:
                samples = sample_count if protocol == "perfect_correction" else 1
                for sample_index in range(samples):
                    seed = stable_seed(campaign_id, protocol, nx, ny, alpha, sample_index)
                    placeholder = TrajectorySpec(
                        protocol=protocol,
                        nx=nx,
                        ny=ny,
                        alpha_topological_region=alpha,
                        alpha_trivial_region=ALPHA_TRIVIAL,
                        dw_truncation=bool(args.dw_truncation),
                        cycles=cycles,
                        sample_index=sample_index,
                        seed=seed,
                        checkpoint_path="",
                    )
                    checkpoint = config_dir(output_root, campaign_id, placeholder) / "checkpoints" / f"trajectory_{sample_index:03d}.npz"
                    specs.append(
                        TrajectorySpec(
                            **{**placeholder.__dict__, "checkpoint_path": str(checkpoint)}
                        )
                    )
    return specs


def chunk_specs(specs: Iterable[TrajectorySpec], block_size: int) -> list[list[TrajectorySpec]]:
    by_config: dict[str, list[TrajectorySpec]] = {}
    for spec in specs:
        by_config.setdefault(spec.config_id, []).append(spec)
    blocks = []
    for config_specs in by_config.values():
        ordered = sorted(config_specs, key=lambda spec: spec.sample_index)
        for start in range(0, len(ordered), block_size):
            blocks.append(ordered[start : start + block_size])
    return blocks


def aggregate_config(config_specs: list[TrajectorySpec], diagnostics: dict[str, Any] | None = None) -> dict[str, Any] | None:
    paths = [Path(spec.checkpoint_path) for spec in sorted(config_specs, key=lambda item: item.sample_index)]
    if not all(path.exists() for path in paths):
        return None
    payloads = [load_checkpoint(path, config_specs[0].cycles) for path in paths]
    entropy = np.stack([payload["total_entropy"] for payload in payloads], axis=0)
    variance = np.stack([payload["total_charge_variance"] for payload in payloads], axis=0)
    seeds = np.asarray([int(payload["seed"]) for payload in payloads], dtype=np.uint32)
    sample_indices = np.asarray([int(payload["sample_index"]) for payload in payloads], dtype=np.int64)
    elapsed = np.asarray([float(payload["elapsed_seconds"]) for payload in payloads], dtype=np.float64)
    first = config_specs[0]
    run_dir = Path(first.checkpoint_path).parent.parent
    existing_summary_path = run_dir / "run_summary.json"
    if diagnostics is None and existing_summary_path.exists():
        with existing_summary_path.open("r", encoding="utf-8") as fh:
            diagnostics = json.load(fh).get("projector_diagnostics")
    cycles = np.arange(1, first.cycles + 1, dtype=np.int64)
    save_npz_atomic(
        run_dir / "trajectory_observables.npz",
        cycles=cycles,
        sample_indices=sample_indices,
        seeds=seeds,
        total_entropy=entropy,
        total_charge_variance=variance,
        protocol=np.asarray(first.protocol),
        config_id=np.asarray(first.config_id),
        dw_truncation=np.asarray(first.dw_truncation),
        observable_region=np.asarray(first.observable_region),
        canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY_POINT),
    )
    rows = []
    for sample_pos, sample_index in enumerate(sample_indices):
        for cycle_pos, cycle in enumerate(cycles):
            rows.append(
                {
                    "config_id": first.config_id,
                    "protocol": first.protocol,
                    "Nx": first.nx,
                    "Ny": first.ny,
                    "alpha_topological_region": first.alpha_topological_region,
                    "alpha_trivial_region": first.alpha_trivial_region,
                    "dw_truncation": bool(first.dw_truncation),
                    "observable_region": first.observable_region,
                    "sample_index": int(sample_index),
                    "seed": int(seeds[sample_pos]),
                    "cycle": int(cycle),
                    "total_entropy": float(entropy[sample_pos, cycle_pos]),
                    "total_charge_variance": float(variance[sample_pos, cycle_pos]),
                }
            )
    save_dataframe_atomic_csv(pd.DataFrame(rows), run_dir / "scalar_metrics.csv")
    summary = {
        "config_id": first.config_id,
        "protocol": first.protocol,
        "Nx": first.nx,
        "Ny": first.ny,
        "cycles": first.cycles,
        "samples_actual": len(paths),
        "alpha_topological_region": first.alpha_topological_region,
        "alpha_trivial_region": first.alpha_trivial_region,
        "dw_truncation": bool(first.dw_truncation),
        "observable_region": first.observable_region,
        "init_mode": "maxmix",
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "parallelize_sites": False,
        "covariance_history_saved": False,
        "final_entropy_mean": float(np.mean(entropy[:, -1])),
        "final_entropy_sample_std": float(np.std(entropy[:, -1], ddof=1)) if len(paths) > 1 else 0.0,
        "final_charge_variance_mean": float(np.mean(variance[:, -1])),
        "final_charge_variance_sample_std": float(np.std(variance[:, -1], ddof=1)) if len(paths) > 1 else 0.0,
        "final_sharpening_fraction": float(np.mean(variance[:, -1] < SHARPENING_THRESHOLD)),
        "trajectory_elapsed_seconds": elapsed.tolist(),
        "projector_diagnostics": diagnostics,
        "completed_at": utc_now(),
    }
    write_json_atomic(run_dir / "run_summary.json", summary)
    return summary


def write_campaign_metadata(
    *,
    output_root: Path,
    campaign_id: str,
    specs: list[TrajectorySpec],
    worker_metadata: dict[str, Any],
    diagnostics_by_config: dict[str, Any],
) -> None:
    for protocol in sorted({spec.protocol for spec in specs}):
        protocol_specs = [spec for spec in specs if spec.protocol == protocol]
        campaign_root = output_root / protocol / "campaigns" / campaign_id
        grouped: dict[str, list[TrajectorySpec]] = {}
        for spec in protocol_specs:
            grouped.setdefault(spec.config_id, []).append(spec)
        results = []
        for config_id, config_specs in grouped.items():
            summary = aggregate_config(config_specs, diagnostics_by_config.get(config_id))
            first = config_specs[0]
            results.append(
                {
                    "config_id": config_id,
                    "protocol": protocol,
                    "Nx": first.nx,
                    "Ny": first.ny,
                    "alpha_topological_region": first.alpha_topological_region,
                    "dw_truncation": bool(first.dw_truncation),
                    "observable_region": first.observable_region,
                    "cycles": first.cycles,
                    "samples_expected": len(config_specs),
                    "samples_completed": sum(Path(spec.checkpoint_path).exists() for spec in config_specs),
                    "complete": summary is not None,
                    "run_dir_relative": str((Path(first.checkpoint_path).parent.parent).relative_to(output_root)),
                }
            )
        manifest = {
            "campaign_name": CAMPAIGN_NAME,
            "campaign_id": campaign_id,
            "protocol": protocol,
            "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
            "parallelize_sites": False,
            "dw_truncation": bool(protocol_specs[0].dw_truncation),
            "alpha_grid": sorted({float(spec.alpha_topological_region) for spec in protocol_specs}),
            "observable_collection_mode": "cpu_cycle_observer_active_slab",
            "observable_region": protocol_specs[0].observable_region,
            "covariance_history_saved": False,
            "worker_metadata": worker_metadata,
            "results": results,
            "complete": all(result["complete"] for result in results),
            "updated_at": utc_now(),
        }
        write_json_atomic(campaign_root / "campaign_manifest.json", manifest)
        save_dataframe_atomic_csv(pd.DataFrame(results), campaign_root / "run_index.csv")
        write_json_atomic(
            output_root / protocol / "latest_campaign.json",
            {
                "campaign_id": campaign_id,
                "campaign_manifest_path": str(campaign_root / "campaign_manifest.json"),
                "complete": manifest["complete"],
                "updated_at": utc_now(),
            },
        )


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("charge_sharpening_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "colab_charge_fluctuations" / "cpu_data" / CAMPAIGN_NAME)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--alpha-values", type=parse_alpha_csv, default=None, help="Comma-separated alpha_1 values.")
    parser.add_argument(
        "--alpha-linspace",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "COUNT"),
        default=None,
        help="Use np.linspace(START, STOP, int(COUNT)) for alpha_1 values.",
    )
    parser.add_argument(
        "--dw-truncation",
        type=int,
        choices=(0, 1),
        default=1,
        help="Set to 1 for domain-wall truncation/slab observables, or 0 for the full active top layer.",
    )
    parser.add_argument(
        "--cpu-selection",
        choices=("highest", "free"),
        default="highest",
        help="Select CPUs from scheduler affinity by highest numbering or current low utilization.",
    )
    parser.add_argument("--cpu-free-threshold", type=float, default=20.0)
    parser.add_argument("--cpu-free-samples", type=int, default=3)
    parser.add_argument("--cpu-free-sample-interval", type=float, default=0.5)
    parser.add_argument("--min-free-workers", type=int, default=1)
    parser.add_argument("--workers", default="auto", help="'auto' or an explicit positive integer")
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--block-size", default="auto", help="'auto' or an explicit positive integer")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers != "auto":
        int(args.workers)
    if args.block_size != "auto":
        int(args.block_size)
    args.resolved_alpha_values = resolve_alpha_values(args)
    campaign_id = f"{args.campaign_id}_smoke" if args.smoke and not args.campaign_id.endswith("_smoke") else args.campaign_id
    output_root = args.output_root.resolve()
    log_path = output_root / "logs" / f"{campaign_id}.log"
    logger = configure_logging(log_path)
    available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))

    logger.info("Starting campaign %s", campaign_id)
    logger.info("Canonical dynamics entry point: %s", CANONICAL_ENTRY_POINT)
    logger.info("dw_truncation=%s, observable_region=%s", bool(args.dw_truncation), "domain_wall_slab" if args.dw_truncation else "full_top_layer")
    logger.info("Alpha grid (%d values): %s", len(args.resolved_alpha_values), list(args.resolved_alpha_values))
    logger.info("Scheduler CPU affinity: %s", available_cpus)
    candidate_cpus, cpu_selection_metadata = resolve_cpu_pool(args, available_cpus)
    logger.info("CPU selection mode=%s, candidate CPUs=%s", args.cpu_selection, candidate_cpus)
    workers, block_size, calibration = choose_parallelism(args, candidate_cpus)
    selected_cpus = candidate_cpus[-workers:]
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(selected_cpus))
    logger.info("Selected workers=%d, block_size=%d, CPUs=%s", workers, block_size, selected_cpus)

    specs = build_specs(args, output_root, campaign_id)
    expected_pc = sum(spec.protocol == "perfect_correction" for spec in specs)
    expected_ps = sum(spec.protocol == "postselection" for spec in specs)
    logger.info("Trajectory counts: perfect_correction=%d, postselection=%d, total=%d", expected_pc, expected_ps, len(specs))
    expected_counts = {
        "perfect_correction": (
            (0 if "perfect_correction" not in set(args.protocols) else len(NY_VALUES) * len(args.resolved_alpha_values) * PERFECT_CORRECTION_SAMPLES)
        ),
        "postselection": (
            (0 if "postselection" not in set(args.protocols) else len(NY_VALUES) * len(args.resolved_alpha_values))
        ),
    }
    if not args.smoke and (expected_pc, expected_ps) != (
        expected_counts["perfect_correction"],
        expected_counts["postselection"],
    ):
        raise AssertionError(
            "Unexpected production trajectory counts: "
            f"actual={(expected_pc, expected_ps)}, expected={expected_counts}"
        )

    worker_metadata = {
        "workers": workers,
        "block_size": block_size,
        "scheduler_affinity_before_selection": available_cpus,
        "candidate_cpus_after_selection": candidate_cpus,
        "selected_highest_numbered_cpus": selected_cpus,
        "selected_cpus": selected_cpus,
        "cpu_selection": cpu_selection_metadata,
        "thread_limits_per_worker": 1,
        "dw_truncation": bool(args.dw_truncation),
        "observable_region": "domain_wall_slab" if args.dw_truncation else "full_top_layer",
        "alpha_grid": list(args.resolved_alpha_values),
        "expected_trajectory_counts": expected_counts,
        "calibration": calibration,
        "log_path": str(log_path),
    }
    write_json_atomic(output_root / "logs" / f"{campaign_id}_worker_metadata.json", worker_metadata)
    diagnostics_by_config: dict[str, Any] = {}
    write_campaign_metadata(
        output_root=output_root,
        campaign_id=campaign_id,
        specs=specs,
        worker_metadata=worker_metadata,
        diagnostics_by_config=diagnostics_by_config,
    )

    pending_specs = [spec for spec in specs if not Path(spec.checkpoint_path).exists()]
    skipped = len(specs) - len(pending_specs)
    blocks = chunk_specs(pending_specs, block_size)
    logger.info("Resume scan: skipped=%d, pending=%d, task_blocks=%d", skipped, len(pending_specs), len(blocks))
    if not blocks:
        logger.info("Campaign already complete.")
        return 0
    config_ids = sorted({spec.config_id for spec in specs})
    completed_config_ids = {
        config_id
        for config_id in config_ids
        if all(Path(spec.checkpoint_path).exists() for spec in specs if spec.config_id == config_id)
    }

    context = get_context("spawn")
    retries: dict[tuple[str, tuple[int, ...]], int] = {}
    failures: list[dict[str, Any]] = []
    completed_trajectories = skipped
    bars = {
        "overall": tqdm(total=len(specs), initial=skipped, desc="all trajectories", unit="trajectory", position=0),
        "perfect_correction": tqdm(
            total=expected_pc,
            initial=sum(spec.protocol == "perfect_correction" and Path(spec.checkpoint_path).exists() for spec in specs),
            desc="perfect correction",
            unit="trajectory",
            position=1,
        ),
        "postselection": tqdm(
            total=expected_ps,
            initial=sum(spec.protocol == "postselection" and Path(spec.checkpoint_path).exists() for spec in specs),
            desc="postselection",
            unit="trajectory",
            position=2,
        ),
        "configs": tqdm(
            total=len(config_ids),
            initial=len(completed_config_ids),
            desc="complete configs",
            unit="config",
            position=3,
        ),
        "blocks": tqdm(total=len(blocks), desc="task blocks", unit="block", position=4),
    }

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=worker_initialize,
            initargs=(selected_cpus,),
        ) as executor:
            futures = {executor.submit(run_trajectory_block, [spec.__dict__ for spec in block]): block for block in blocks}
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    block = futures.pop(future)
                    key = (block[0].config_id, tuple(spec.sample_index for spec in block))
                    try:
                        payload = future.result()
                    except Exception as exc:
                        attempt = retries.get(key, 0)
                        logger.error("Block failed config=%s samples=%s attempt=%d: %s", key[0], key[1], attempt + 1, exc)
                        logger.error(traceback.format_exc())
                        if attempt < args.max_retries:
                            retries[key] = attempt + 1
                            futures[executor.submit(run_trajectory_block, [spec.__dict__ for spec in block])] = block
                        else:
                            failures.append({"config_id": key[0], "sample_indices": key[1], "error": repr(exc)})
                            bars["blocks"].update(1)
                        continue

                    diagnostics_by_config[block[0].config_id] = payload["diagnostics"]
                    new_count = sum(result["status"] == "completed" for result in payload["results"])
                    skip_count = sum(result["status"] == "skipped" for result in payload["results"])
                    count = new_count + skip_count
                    completed_trajectories += count
                    bars["overall"].update(count)
                    bars[block[0].protocol].update(count)
                    bars["blocks"].update(1)
                    grouped = [spec for spec in specs if spec.config_id == block[0].config_id]
                    summary = aggregate_config(grouped, diagnostics_by_config.get(block[0].config_id))
                    if summary is not None and block[0].config_id not in completed_config_ids:
                        completed_config_ids.add(block[0].config_id)
                        bars["configs"].update(1)
                    logger.info(
                        "Block finished config=%s samples=%s elapsed=%.2fs rss=%.2f GiB "
                        "completed=%d/%d pending=%d failed_blocks=%d complete_configs=%d/%d",
                        block[0].config_id,
                        [spec.sample_index for spec in block],
                        payload["block_elapsed_seconds"],
                        payload["worker_rss_bytes"] / 1024**3,
                        completed_trajectories,
                        len(specs),
                        len(specs) - completed_trajectories,
                        len(failures),
                        len(completed_config_ids),
                        len(config_ids),
                    )
                    write_campaign_metadata(
                        output_root=output_root,
                        campaign_id=campaign_id,
                        specs=specs,
                        worker_metadata=worker_metadata,
                        diagnostics_by_config=diagnostics_by_config,
                    )
    finally:
        for bar in bars.values():
            bar.close()

    write_campaign_metadata(
        output_root=output_root,
        campaign_id=campaign_id,
        specs=specs,
        worker_metadata=worker_metadata,
        diagnostics_by_config=diagnostics_by_config,
    )
    if failures:
        failure_path = output_root / "logs" / f"{campaign_id}_failures.json"
        write_json_atomic(failure_path, failures)
        logger.error("Campaign finished with %d failed blocks. See %s", len(failures), failure_path)
        return 1
    logger.info("Campaign complete. Output root: %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
