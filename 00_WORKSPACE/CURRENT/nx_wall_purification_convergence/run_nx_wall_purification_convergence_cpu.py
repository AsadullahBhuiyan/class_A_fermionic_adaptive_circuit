#!/usr/bin/env python3
"""Run the fixed-Ny transverse-width wall-purification campaign on CPUs.

All circuit dynamics are delegated to the canonical
``classA_U1FGTN.run_markov_circuit`` entry point.  Independent trajectories
are parallelized across processes, observables are streamed cycle by cycle,
and exact RNG/covariance restart states are saved atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN


CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
CAMPAIGN_NAME = "nx_wall_purification_convergence"
DEFAULT_CAMPAIGN_ID = "Nx20-24-28_Ny20_nsh1_dwtrunc1_init-maxmix_S25_C100"
DEFAULT_NX_VALUES = (20, 24, 28)
DEFAULT_NY = 20
DEFAULT_CYCLES = 100
DEFAULT_SAMPLES = 25
EXPECTED_WALLS = {20: (4, 16), 24: (4, 20), 28: (5, 23)}
NSHELL = 1
ALPHA_TOPOLOGICAL = 1.0
ALPHA_TRIVIAL = 30.0
TRIAL_ORBITALS = "X"
CHECKPOINT_SCHEMA = 1
RESULT_SCHEMA = 1
WORKER_MEMORY_ESTIMATE_BYTES = 1536 * 1024**2
OBSERVABLE_KEYS = (
    "entropy_profile_bits",
    "wall_entropy_bits_per_cell",
    "bulk_entropy_bits_per_cell",
    "total_entropy_bits",
    "total_entropy_per_active_mode_bits",
    "total_entropy_per_total_mode_bits",
    "total_charge_variance",
    "charge_variance_per_active_mode",
    "charge_variance_per_total_mode",
    "active_purity_deficit_rms",
    "active_covariance_frobenius_rms",
    "full_covariance_frobenius_per_total_mode",
    "hermiticity_residual",
    "occupation_eigenvalue_min",
    "occupation_eigenvalue_max",
    "contour_sum_error_bits",
)


@dataclass(frozen=True)
class TrajectorySpec:
    campaign_id: str
    nx: int
    ny: int
    cycles: int
    sample_index: int
    seed: int
    checkpoint_stride: int
    result_path: str
    restart_path: str

    @property
    def config_id(self) -> str:
        return (
            f"N{self.nx}x{self.ny}_nsh{NSHELL}_dwtrunc1_"
            "init-maxmix_perfect-correction"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json_atomic(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_npz_atomic(path: Path | str, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_pickle_atomic(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_pickle(path: Path | str) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(CAMPAIGN_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def expected_domain_wall(nx: int) -> tuple[int, int]:
    half = int(nx) // 2
    half_width = max(1, int(nx) // 3)
    return max(0, half - half_width), min(int(nx), half + half_width + 1) - 1


def geometry_metadata(model: classA_U1FGTN) -> dict[str, Any]:
    walls = tuple(int(value) for value in model.DW_loc)
    expected = expected_domain_wall(model.Nx)
    if walls != expected:
        raise AssertionError(f"N_x={model.Nx}: walls={walls}, expected={expected}")
    if model.Nx in EXPECTED_WALLS and walls != EXPECTED_WALLS[model.Nx]:
        raise AssertionError(
            f"N_x={model.Nx}: standard geometry changed from {EXPECTED_WALLS[model.Nx]} to {walls}"
        )
    x_left, x_right = walls
    bulk_rows = np.arange(x_left + 2, x_right - 1, dtype=np.int64)
    if bulk_rows.size == 0:
        raise ValueError(f"N_x={model.Nx}: active-bulk mask is empty")
    active = model.active_top_layer_indices(meas_slab_only=True)
    return {
        "Nx": int(model.Nx),
        "Ny": int(model.Ny),
        "domain_wall_locations": list(walls),
        "wall_rows": list(walls),
        "bulk_rows": bulk_rows.tolist(),
        "active_slab_x_min": int(x_left),
        "active_slab_x_max": int(x_right),
        "active_slab_width_sites": int(x_right - x_left + 1),
        "active_mode_count": int(active.size),
        "total_mode_count": int(2 * model.Nx * model.Ny),
        "domain_wall_rule": getattr(model, "DW_slab_half_width_rule", "max(1, Nx // 3)"),
    }


def allocate_observables(cycles: int, nx: int) -> dict[str, np.ndarray]:
    arrays = {
        key: np.full(cycles + 1, np.nan, dtype=np.float64)
        for key in OBSERVABLE_KEYS
        if key != "entropy_profile_bits"
    }
    arrays["entropy_profile_bits"] = np.full((cycles + 1, nx), np.nan, dtype=np.float64)
    return arrays


def compute_cycle_observables(
    G_shifted: np.ndarray,
    *,
    active_indices: np.ndarray,
    nx: int,
    ny: int,
    walls: tuple[int, int],
    bulk_rows: np.ndarray,
) -> dict[str, Any]:
    """Compute all streamed observables from the shifted correlation matrix."""
    full = np.asarray(G_shifted, dtype=np.complex128)
    hermiticity_residual = float(np.max(np.abs(full - full.conj().T)))
    if not np.all(np.isfinite(full)):
        raise FloatingPointError("Shifted correlation matrix contains non-finite entries")
    if hermiticity_residual > 1e-8:
        raise FloatingPointError(f"Hermiticity residual is {hermiticity_residual:.3e}")

    active = np.asarray(full[np.ix_(active_indices, active_indices)], dtype=np.complex128)
    active = 0.5 * (active + active.conj().T)
    n_active = int(active.shape[0])
    n_total = int(2 * nx * ny)
    x_left, x_right = walls
    slab_width = x_right - x_left + 1
    if n_active != 2 * slab_width * ny:
        raise AssertionError(
            f"Active dimension {n_active} != 2*{slab_width}*{ny}"
        )

    occupation = 0.5 * (np.eye(n_active, dtype=np.complex128) + active)
    occupation = 0.5 * (occupation + occupation.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(occupation)
    eigenvalues = np.real_if_close(eigenvalues).astype(np.float64)
    eigenvalue_min = float(np.min(eigenvalues))
    eigenvalue_max = float(np.max(eigenvalues))
    if eigenvalue_min < -1e-7 or eigenvalue_max > 1.0 + 1e-7:
        raise FloatingPointError(
            f"Occupation spectrum outside [0,1]: [{eigenvalue_min:.6e}, {eigenvalue_max:.6e}]"
        )

    clipped = np.clip(eigenvalues, 1e-12, 1.0 - 1e-12)
    entropy_eigenvalues_nats = -(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    )
    entropy_eigenvalues_bits = entropy_eigenvalues_nats / np.log(2.0)
    diagonal_entropy_bits = (np.abs(eigenvectors) ** 2) @ entropy_eigenvalues_bits
    contour_active = diagonal_entropy_bits.reshape(2, slab_width, ny, order="F").sum(axis=0)
    profile = np.full(nx, np.nan, dtype=np.float64)
    profile[x_left : x_right + 1] = np.mean(contour_active, axis=1)
    total_entropy_bits = float(np.sum(entropy_eigenvalues_bits))
    contour_sum_error = float(abs(np.sum(contour_active) - total_entropy_bits))
    if contour_sum_error > 1e-7 * max(1.0, total_entropy_bits):
        raise FloatingPointError(f"Entropy-contour sum mismatch: {contour_sum_error:.3e} bits")

    charge_variance = float(
        np.real(np.trace(occupation)) - np.sum(np.abs(occupation) ** 2)
    )
    if charge_variance < -1e-7:
        raise FloatingPointError(f"Negative charge variance: {charge_variance:.6e}")
    charge_variance = max(0.0, charge_variance)
    purity_residual = active @ active - np.eye(n_active, dtype=np.complex128)

    return {
        "entropy_profile_bits": profile,
        "wall_entropy_bits_per_cell": float(np.mean(profile[list(walls)])),
        "bulk_entropy_bits_per_cell": float(np.mean(profile[bulk_rows])),
        "total_entropy_bits": total_entropy_bits,
        "total_entropy_per_active_mode_bits": total_entropy_bits / n_active,
        "total_entropy_per_total_mode_bits": total_entropy_bits / n_total,
        "total_charge_variance": charge_variance,
        "charge_variance_per_active_mode": charge_variance / n_active,
        "charge_variance_per_total_mode": charge_variance / n_total,
        "active_purity_deficit_rms": float(np.linalg.norm(purity_residual, "fro") / math.sqrt(n_active)),
        "active_covariance_frobenius_rms": float(np.linalg.norm(active, "fro") / math.sqrt(n_active)),
        "full_covariance_frobenius_per_total_mode": float(np.linalg.norm(full, "fro") / n_total),
        "hermiticity_residual": hermiticity_residual,
        "occupation_eigenvalue_min": eigenvalue_min,
        "occupation_eigenvalue_max": eigenvalue_max,
        "contour_sum_error_bits": contour_sum_error,
    }


def validate_cycle_zero(observed: dict[str, Any], geometry: dict[str, Any]) -> None:
    n_active = int(geometry["active_mode_count"])
    expected = {
        "total_entropy_bits": float(n_active),
        "total_charge_variance": float(n_active / 4.0),
        "active_purity_deficit_rms": 1.0,
        "active_covariance_frobenius_rms": 0.0,
        "wall_entropy_bits_per_cell": 2.0,
        "bulk_entropy_bits_per_cell": 2.0,
    }
    for key, value in expected.items():
        if not np.isclose(float(observed[key]), value, atol=2e-8, rtol=2e-8):
            raise AssertionError(
                f"Cycle-zero max-mix check failed for {key}: {observed[key]} != {value}"
            )


def store_observation(arrays: dict[str, np.ndarray], cycle: int, values: dict[str, Any]) -> None:
    for key in OBSERVABLE_KEYS:
        arrays[key][cycle] = values[key]


def _load_partial_checkpoint(
    restart_path: Path,
    *,
    spec: TrajectorySpec,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any] | None, float]:
    if not restart_path.exists():
        return None, 0.0
    payload = load_pickle(restart_path)
    if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported restart schema in {restart_path}")
    if payload.get("config_id") != spec.config_id or int(payload.get("sample_index", -1)) != spec.sample_index:
        raise ValueError(f"Restart checkpoint does not match {spec.config_id} sample {spec.sample_index}")
    state = payload["markov_checkpoint_state"]
    completed = int(state["completed_cycles"])
    if completed >= spec.cycles:
        return state, float(payload.get("elapsed_seconds_accumulated", 0.0))
    saved = payload["observables"]
    for key in OBSERVABLE_KEYS:
        source = np.asarray(saved[key])
        arrays[key][: source.shape[0]] = source
    return state, float(payload.get("elapsed_seconds_accumulated", 0.0))


def result_reaches_target(path: Path | str, target_cycles: int) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            cycles = np.asarray(payload["cycles"], dtype=np.int64)
            return cycles.shape == (target_cycles + 1,) and int(cycles[-1]) == target_cycles
    except (OSError, KeyError, ValueError):
        return False


def projector_diagnostics(model: classA_U1FGTN) -> dict[str, Any]:
    arrays = [model.WF_Ap, model.WF_Bp, model.WF_Am, model.WF_Bm]
    norms = np.concatenate([np.linalg.norm(array, axis=0).reshape(-1) for array in arrays])
    return {
        "projectors_all_finite": bool(all(np.all(np.isfinite(array)) for array in arrays)),
        "projector_center_norm_min": float(np.min(norms)),
        "projector_center_norm_max": float(np.max(norms)),
    }


def run_trajectory(spec: TrajectorySpec, model: classA_U1FGTN) -> dict[str, Any]:
    result_path = Path(spec.result_path)
    restart_path = Path(spec.restart_path)
    if result_reaches_target(result_path, spec.cycles):
        return {
            "status": "skipped",
            "config_id": spec.config_id,
            "sample_index": spec.sample_index,
            "result_path": str(result_path),
        }

    geometry = geometry_metadata(model)
    active_indices = model.active_top_layer_indices(meas_slab_only=True)
    walls = tuple(int(value) for value in geometry["wall_rows"])
    bulk_rows = np.asarray(geometry["bulk_rows"], dtype=np.int64)
    arrays = allocate_observables(spec.cycles, spec.nx)
    checkpoint_state, elapsed_before = _load_partial_checkpoint(
        restart_path, spec=spec, arrays=arrays
    )
    if checkpoint_state is not None and int(checkpoint_state["completed_cycles"]) >= spec.cycles:
        raise RuntimeError(
            f"Restart checkpoint reached cycle {checkpoint_state['completed_cycles']} but result is missing; "
            "extend --cycles or inspect the checkpoint."
        )

    segment_started = time.perf_counter()

    def observe(*, cycle: int, G: np.ndarray, **_: Any) -> None:
        values = compute_cycle_observables(
            G,
            active_indices=active_indices,
            nx=spec.nx,
            ny=spec.ny,
            walls=walls,
            bulk_rows=bulk_rows,
        )
        if int(cycle) == 0:
            validate_cycle_zero(values, geometry)
        store_observation(arrays, int(cycle), values)

    def checkpoint_observer(*, cycle: int, state: dict[str, Any]) -> None:
        cycle = int(cycle)
        if cycle % spec.checkpoint_stride != 0 and cycle != spec.cycles:
            return
        partial = {key: np.array(value[: cycle + 1], copy=True) for key, value in arrays.items()}
        save_pickle_atomic(
            restart_path,
            {
                "schema_version": CHECKPOINT_SCHEMA,
                "campaign_id": spec.campaign_id,
                "config_id": spec.config_id,
                "sample_index": spec.sample_index,
                "seed": spec.seed,
                "target_cycles_when_written": spec.cycles,
                "saved_at": utc_now(),
                "elapsed_seconds_accumulated": elapsed_before + time.perf_counter() - segment_started,
                "markov_checkpoint_state": state,
                "observables": partial,
            },
        )

    with threadpool_limits(limits=1):
        model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=spec.cycles,
            perfect_correction=True,
            samples=1,
            parallelize_samples=False,
            init_mode="maxmix",
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=True,
            random_seed=spec.seed,
            cycle_observer=observe,
            checkpoint_state=checkpoint_state,
            checkpoint_observer=checkpoint_observer,
        )

    for key, value in arrays.items():
        if key == "entropy_profile_bits":
            active_slice = value[:, geometry["active_slab_x_min"] : geometry["active_slab_x_max"] + 1]
            if not np.all(np.isfinite(active_slice)):
                raise FloatingPointError(f"Incomplete active entropy profile for {spec.config_id}")
        elif not np.all(np.isfinite(value)):
            raise FloatingPointError(f"Incomplete observable {key} for {spec.config_id}")

    elapsed = elapsed_before + time.perf_counter() - segment_started
    save_npz_atomic(
        result_path,
        schema_version=np.asarray(RESULT_SCHEMA, dtype=np.int64),
        cycles=np.arange(spec.cycles + 1, dtype=np.int64),
        sample_index=np.asarray(spec.sample_index, dtype=np.int64),
        seed=np.asarray(spec.seed, dtype=np.uint32),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
        config_id=np.asarray(spec.config_id),
        canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY_POINT),
        domain_wall_locations=np.asarray(walls, dtype=np.int64),
        bulk_rows=bulk_rows,
        active_mode_count=np.asarray(geometry["active_mode_count"], dtype=np.int64),
        total_mode_count=np.asarray(geometry["total_mode_count"], dtype=np.int64),
        **arrays,
    )
    return {
        "status": "completed",
        "config_id": spec.config_id,
        "sample_index": spec.sample_index,
        "result_path": str(result_path),
        "restart_path": str(restart_path),
        "elapsed_seconds": elapsed,
    }


def worker_initialize(affinity: list[int]) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu) for cpu in affinity))


def run_trajectory_block(spec_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    specs = [TrajectorySpec(**payload) for payload in spec_payloads]
    first = specs[0]
    if any((spec.nx, spec.ny) != (first.nx, first.ny) for spec in specs):
        raise ValueError("A worker block may contain only one geometry")
    model = classA_U1FGTN(
        first.nx,
        first.ny,
        DW=True,
        nshell=NSHELL,
        alpha_1=ALPHA_TOPOLOGICAL,
        alpha_2=ALPHA_TRIVIAL,
        trial_orbitals=TRIAL_ORBITALS,
        dw_truncation=True,
        # This completed campaign predates the CPU/GPU default alignment and
        # explicitly retains its recorded one-third-width geometry.
        dw_interval=EXPECTED_WALLS[first.nx],
    )
    model.construct_OW_projectors(
        nshell=NSHELL,
        DW=True,
        trial_orbitals=TRIAL_ORBITALS,
        dw_truncation=True,
    )
    geometry = geometry_metadata(model)
    diagnostics = projector_diagnostics(model)
    started = time.perf_counter()
    results = [run_trajectory(spec, model) for spec in specs]
    return {
        "results": results,
        "geometry": geometry,
        "projector_diagnostics": diagnostics,
        "worker_pid": os.getpid(),
        "worker_rss_bytes": psutil.Process().memory_info().rss,
        "elapsed_seconds": time.perf_counter() - started,
    }


def sample_cpu_usage(
    cpus: list[int], *, samples: int, interval: float
) -> dict[str, Any]:
    psutil.cpu_percent(interval=None, percpu=True)
    rows = []
    for _ in range(max(1, int(samples))):
        usage = psutil.cpu_percent(interval=max(0.05, float(interval)), percpu=True)
        rows.append([float(usage[cpu]) for cpu in cpus])
    values = np.asarray(rows, dtype=np.float64)
    return {
        "samples": int(values.shape[0]),
        "interval_seconds": max(0.05, float(interval)),
        "per_cpu_average_percent": {
            str(cpu): float(values[:, position].mean()) for position, cpu in enumerate(cpus)
        },
        "per_cpu_max_percent": {
            str(cpu): float(values[:, position].max()) for position, cpu in enumerate(cpus)
        },
    }


def choose_cpu_pool(args: argparse.Namespace, available: list[int]) -> tuple[list[int], dict[str, Any]]:
    if args.cpu_selection == "highest":
        return available, {"mode": "highest", "available": available}
    usage = sample_cpu_usage(
        available,
        samples=args.cpu_free_samples,
        interval=args.cpu_free_sample_interval,
    )
    maximum = {int(cpu): float(value) for cpu, value in usage["per_cpu_max_percent"].items()}
    selected = [cpu for cpu in available if maximum[cpu] <= args.cpu_free_threshold]
    if len(selected) < args.min_free_workers:
        raise RuntimeError(
            f"Only {len(selected)} CPUs are below {args.cpu_free_threshold:g}% utilization; "
            f"need at least {args.min_free_workers}."
        )
    return selected, {
        "mode": "free",
        "available": available,
        "selected": selected,
        "busy": [cpu for cpu in available if cpu not in set(selected)],
        "threshold_percent": args.cpu_free_threshold,
        "usage": usage,
    }


def choose_worker_count(args: argparse.Namespace, cpus: list[int], pending_count: int) -> tuple[int, dict[str, Any]]:
    available_memory = int(psutil.virtual_memory().available)
    memory_cap = max(1, int(0.70 * available_memory // WORKER_MEMORY_ESTIMATE_BYTES))
    hard_cap = max(1, min(len(cpus), memory_cap, args.max_workers, max(1, pending_count)))
    workers = hard_cap if args.workers == "auto" else max(1, min(int(args.workers), hard_cap))
    return workers, {
        "workers": workers,
        "available_memory_bytes": available_memory,
        "estimated_worker_memory_bytes": WORKER_MEMORY_ESTIMATE_BYTES,
        "memory_worker_cap": memory_cap,
        "hard_worker_cap": hard_cap,
    }


def config_directory(output_root: Path, campaign_id: str, config_id: str) -> Path:
    return output_root / "campaigns" / campaign_id / "runs" / config_id


def build_specs(args: argparse.Namespace, output_root: Path) -> list[TrajectorySpec]:
    specs: list[TrajectorySpec] = []
    for nx in args.nx_values:
        config_id = (
            f"N{nx}x{args.ny}_nsh{NSHELL}_dwtrunc1_"
            "init-maxmix_perfect-correction"
        )
        run_dir = config_directory(output_root, args.campaign_id, config_id)
        for sample_index in range(args.samples):
            seed = stable_seed(
                args.campaign_id,
                nx,
                args.ny,
                NSHELL,
                ALPHA_TOPOLOGICAL,
                "maxmix",
                "perfect_correction",
                sample_index,
            )
            specs.append(
                TrajectorySpec(
                    campaign_id=args.campaign_id,
                    nx=int(nx),
                    ny=int(args.ny),
                    cycles=int(args.cycles),
                    sample_index=sample_index,
                    seed=seed,
                    checkpoint_stride=int(args.checkpoint_stride),
                    result_path=str(run_dir / "trajectories" / f"trajectory_{sample_index:03d}.npz"),
                    restart_path=str(run_dir / "restart" / f"trajectory_{sample_index:03d}.pkl"),
                )
            )
    return specs


def chunk_specs(specs: Iterable[TrajectorySpec], block_size: int) -> list[list[TrajectorySpec]]:
    grouped: dict[str, list[TrajectorySpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.config_id, []).append(spec)
    blocks: list[list[TrajectorySpec]] = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: item.sample_index)
        for start in range(0, len(ordered), block_size):
            blocks.append(ordered[start : start + block_size])
    return blocks


def load_trajectory_result(path: Path, expected_cycles: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key] for key in payload.files}
    if result["cycles"].shape != (expected_cycles + 1,) or int(result["cycles"][-1]) != expected_cycles:
        raise ValueError(f"{path} does not reach cycle {expected_cycles}")
    return result


def aggregate_config(config_specs: list[TrajectorySpec], geometry: dict[str, Any] | None = None) -> dict[str, Any] | None:
    completed = [
        spec for spec in sorted(config_specs, key=lambda item: item.sample_index)
        if result_reaches_target(spec.result_path, spec.cycles)
    ]
    if not completed:
        return None
    payloads = [load_trajectory_result(Path(spec.result_path), spec.cycles) for spec in completed]
    first = completed[0]
    run_dir = Path(first.result_path).parent.parent
    stacked = {
        key: np.stack([np.asarray(payload[key]) for payload in payloads], axis=0)
        for key in OBSERVABLE_KEYS
    }
    sample_indices = np.asarray([spec.sample_index for spec in completed], dtype=np.int64)
    seeds = np.asarray([spec.seed for spec in completed], dtype=np.uint32)
    elapsed = np.asarray([float(payload["elapsed_seconds"]) for payload in payloads])
    save_npz_atomic(
        run_dir / "trajectory_observables.npz",
        schema_version=np.asarray(RESULT_SCHEMA, dtype=np.int64),
        cycles=np.arange(first.cycles + 1, dtype=np.int64),
        sample_indices=sample_indices,
        seeds=seeds,
        domain_wall_locations=np.asarray(payloads[0]["domain_wall_locations"], dtype=np.int64),
        bulk_rows=np.asarray(payloads[0]["bulk_rows"], dtype=np.int64),
        canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY_POINT),
        **stacked,
    )

    scalar_keys = [key for key in OBSERVABLE_KEYS if key != "entropy_profile_bits"]
    csv_path = run_dir / "scalar_metrics.csv"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{csv_path.name}.", suffix=".tmp", dir=run_dir
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            fields = ["config_id", "Nx", "Ny", "sample_index", "seed", "cycle", *scalar_keys]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sample_position, spec in enumerate(completed):
                for cycle in range(first.cycles + 1):
                    writer.writerow(
                        {
                            "config_id": first.config_id,
                            "Nx": first.nx,
                            "Ny": first.ny,
                            "sample_index": spec.sample_index,
                            "seed": spec.seed,
                            "cycle": cycle,
                            **{
                                key: float(stacked[key][sample_position, cycle])
                                for key in scalar_keys
                            },
                        }
                    )
        os.replace(temporary_name, csv_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    if geometry is None:
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            geometry = json.loads(summary_path.read_text(encoding="utf-8")).get("geometry")
    summary = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": first.campaign_id,
        "config_id": first.config_id,
        "Nx": first.nx,
        "Ny": first.ny,
        "cycles": first.cycles,
        "samples_requested": len(config_specs),
        "samples_completed": len(completed),
        "complete": len(completed) == len(config_specs),
        "init_mode": "maxmix",
        "protocol": "perfect_correction",
        "sequence": "raster_y",
        "n_a": 0.5,
        "nshell": NSHELL,
        "alpha_topological_region": ALPHA_TOPOLOGICAL,
        "alpha_trivial_region": ALPHA_TRIVIAL,
        "dw_truncation": True,
        "meas_slab_only": True,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "covariance_history_saved": False,
        "geometry": geometry,
        "final_wall_entropy_mean_bits_per_cell": float(np.mean(stacked["wall_entropy_bits_per_cell"][:, -1])),
        "final_bulk_entropy_mean_bits_per_cell": float(np.mean(stacked["bulk_entropy_bits_per_cell"][:, -1])),
        "trajectory_elapsed_seconds": elapsed.tolist(),
        "updated_at": utc_now(),
    }
    write_json_atomic(run_dir / "run_summary.json", summary)
    return summary


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_campaign_manifest(
    *,
    output_root: Path,
    args: argparse.Namespace,
    specs: list[TrajectorySpec],
    worker_metadata: dict[str, Any],
    geometry_by_config: dict[str, dict[str, Any]],
    projector_by_config: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    campaign_root = output_root / "campaigns" / args.campaign_id
    grouped: dict[str, list[TrajectorySpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.config_id, []).append(spec)
    results = []
    for config_id, config_specs in sorted(grouped.items()):
        summary = aggregate_config(config_specs, geometry_by_config.get(config_id))
        first = config_specs[0]
        results.append(
            {
                "config_id": config_id,
                "Nx": first.nx,
                "Ny": first.ny,
                "cycles": first.cycles,
                "samples_requested": len(config_specs),
                "samples_completed": sum(
                    result_reaches_target(spec.result_path, spec.cycles) for spec in config_specs
                ),
                "complete": bool(summary and summary["complete"]),
                "run_directory": str(Path(first.result_path).parent.parent),
                "geometry": geometry_by_config.get(config_id),
                "projector_diagnostics": projector_by_config.get(config_id),
            }
        )
    source_path = REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py"
    runner_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "campaign_name": CAMPAIGN_NAME,
        "campaign_id": args.campaign_id,
        "status": "failed" if failures else ("complete" if all(row["complete"] for row in results) else "incomplete"),
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "source_hashes": {
            str(source_path.relative_to(REPO_ROOT)): sha256_file(source_path),
            str(runner_path.relative_to(REPO_ROOT)): sha256_file(runner_path),
        },
        "git": git_metadata(),
        "scientific_configuration": {
            "Nx_values": list(args.nx_values),
            "Ny": args.ny,
            "cycles": args.cycles,
            "samples": args.samples,
            "init_mode": "maxmix",
            "protocol": "perfect_correction",
            "sequence": "raster_y",
            "n_a": 0.5,
            "nshell": NSHELL,
            "alpha_topological_region": ALPHA_TOPOLOGICAL,
            "alpha_trivial_region": ALPHA_TRIVIAL,
            "trial_orbitals": TRIAL_ORBITALS,
            "dw_truncation": True,
            "meas_slab_only": True,
            "checkpoint_stride": args.checkpoint_stride,
        },
        "observable_collection": "CPU cycle_observer over active domain-wall slab; no covariance history",
        "worker_metadata": worker_metadata,
        "results": results,
        "failures": failures,
        "updated_at": utc_now(),
    }
    write_json_atomic(campaign_root / "campaign_manifest.json", manifest)
    write_json_atomic(
        output_root / "latest_campaign.json",
        {
            "campaign_id": args.campaign_id,
            "manifest": str(campaign_root / "campaign_manifest.json"),
            "status": manifest["status"],
            "updated_at": utc_now(),
        },
    )
    return manifest


def parse_nx_values(values: list[str]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed or any(value < 4 for value in parsed):
        raise argparse.ArgumentTypeError("--nx-values must contain integers >= 4")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "results")
    parser.add_argument("--nx-values", nargs="+", default=[str(value) for value in DEFAULT_NX_VALUES])
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--checkpoint-stride", type=int, default=5)
    parser.add_argument("--workers", default="auto", help="'auto' or a positive integer")
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--cpu-selection", choices=("free", "highest"), default="free")
    parser.add_argument("--cpu-free-threshold", type=float, default=20.0)
    parser.add_argument("--cpu-free-samples", type=int, default=3)
    parser.add_argument("--cpu-free-sample-interval", type=float, default=0.5)
    parser.add_argument("--min-free-workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.nx_values = parse_nx_values(list(args.nx_values))
    if args.smoke:
        args.cycles = 3
        args.samples = 1
        if args.campaign_id == DEFAULT_CAMPAIGN_ID:
            args.campaign_id = f"{DEFAULT_CAMPAIGN_ID}_smoke"
    for name in ("ny", "cycles", "samples", "checkpoint_stride", "max_workers", "block_size"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers != "auto" and int(args.workers) < 1:
        parser.error("--workers must be 'auto' or a positive integer")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    log_path = output_root / "campaigns" / args.campaign_id / "logs" / "runner.log"
    logger = configure_logging(log_path)
    specs = build_specs(args, output_root)
    pending = [spec for spec in specs if not result_reaches_target(spec.result_path, spec.cycles)]
    logger.info("Campaign %s", args.campaign_id)
    logger.info("Canonical dynamics entry point: %s", CANONICAL_ENTRY_POINT)
    logger.info(
        "Configuration Nx=%s Ny=%d cycles=%d samples=%d pending=%d",
        list(args.nx_values), args.ny, args.cycles, args.samples, len(pending),
    )

    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    cpu_pool, cpu_metadata = choose_cpu_pool(args, available)
    workers, parallel_metadata = choose_worker_count(args, cpu_pool, len(pending))
    selected_cpus = cpu_pool[-workers:]
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(selected_cpus))
    worker_metadata = {
        "scheduler_affinity": available,
        "cpu_selection": cpu_metadata,
        "selected_cpus": selected_cpus,
        "one_blas_thread_per_worker": True,
        **parallel_metadata,
        "log_path": str(log_path),
    }
    logger.info("Selected %d workers on CPUs %s", workers, selected_cpus)

    geometry_by_config: dict[str, dict[str, Any]] = {}
    projector_by_config: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    write_campaign_manifest(
        output_root=output_root,
        args=args,
        specs=specs,
        worker_metadata=worker_metadata,
        geometry_by_config=geometry_by_config,
        projector_by_config=projector_by_config,
        failures=failures,
    )
    if not pending:
        logger.info("Every trajectory already reaches the requested cycle horizon.")
        return 0

    blocks = chunk_specs(pending, args.block_size)
    retries: dict[tuple[str, tuple[int, ...]], int] = {}
    context = get_context("spawn")
    progress = tqdm(total=len(pending), desc="trajectories", unit="trajectory")
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=worker_initialize,
            initargs=(selected_cpus,),
        ) as executor:
            futures = {
                executor.submit(run_trajectory_block, [asdict(spec) for spec in block]): block
                for block in blocks
            }
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    block = futures.pop(future)
                    key = (block[0].config_id, tuple(spec.sample_index for spec in block))
                    try:
                        payload = future.result()
                    except Exception as error:
                        attempt = retries.get(key, 0)
                        logger.error(
                            "Block failed config=%s samples=%s attempt=%d: %s",
                            key[0], key[1], attempt + 1, error,
                        )
                        logger.error(traceback.format_exc())
                        if attempt < args.max_retries:
                            retries[key] = attempt + 1
                            futures[
                                executor.submit(run_trajectory_block, [asdict(spec) for spec in block])
                            ] = block
                        else:
                            failures.append(
                                {"config_id": key[0], "sample_indices": list(key[1]), "error": repr(error)}
                            )
                        continue
                    geometry_by_config[block[0].config_id] = payload["geometry"]
                    projector_by_config[block[0].config_id] = payload["projector_diagnostics"]
                    progress.update(len(payload["results"]))
                    logger.info(
                        "Finished %s samples=%s elapsed=%.1fs rss=%.2f GiB",
                        block[0].config_id,
                        [item.sample_index for item in block],
                        payload["elapsed_seconds"],
                        payload["worker_rss_bytes"] / 1024**3,
                    )
                    write_campaign_manifest(
                        output_root=output_root,
                        args=args,
                        specs=specs,
                        worker_metadata=worker_metadata,
                        geometry_by_config=geometry_by_config,
                        projector_by_config=projector_by_config,
                        failures=failures,
                    )
    finally:
        progress.close()

    manifest = write_campaign_manifest(
        output_root=output_root,
        args=args,
        specs=specs,
        worker_metadata=worker_metadata,
        geometry_by_config=geometry_by_config,
        projector_by_config=projector_by_config,
        failures=failures,
    )
    if failures:
        failure_path = output_root / "campaigns" / args.campaign_id / "logs" / "failures.json"
        write_json_atomic(failure_path, failures)
        logger.error("Campaign ended with %d failed blocks; see %s", len(failures), failure_path)
        return 1
    logger.info("Campaign status: %s", manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
