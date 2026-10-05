#!/usr/bin/env python3
"""Run no-feedback domain-wall alpha sweeps for Choi transfer gaps on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import get_context
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN


CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
EXPERIMENT_NAME = "no_feedback_alpha_sweep_transfer"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results_v2"
DEFAULT_NX = 20
DEFAULT_NY = 20
DEFAULT_SAMPLES = 10
DEFAULT_NSHELL = 1
DEFAULT_ALPHA_2 = 30.0
DEFAULT_ALPHA_VALUES = (1.0, 1.5, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0)
DEFAULT_N_A = 0.5
DEFAULT_P_GAIN = 0.0
DEFAULT_P_LOSS = 0.0
DEFAULT_MAX_WORKERS_PER_RUN = 12
TRIAL_ORBITALS = "X"
WORKER_MEMORY_ESTIMATE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class SweepConfig:
    dw_label: str
    DW: bool
    dw_truncation: bool
    meas_slab_only: bool
    alpha_1: float
    alpha_2: float


@dataclass(frozen=True)
class TrajectorySpec:
    init_mode_label: str
    canonical_init_mode: str
    dw_label: str
    nx: int
    ny: int
    cycles: int
    nshell: float | int | None
    alpha_1: float
    alpha_2: float
    sample_index: int
    seed: int
    output_path: str
    cpu_affinity: tuple[int, ...]
    overwrite: bool
    save_final_g: bool
    track_global_entropy: bool


class ChoiSpectrumObserver:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.sigma_ll: np.ndarray | None = None

    def __call__(
        self,
        *,
        cycle: int,
        sigma_ll: np.ndarray,
        min_abs_d: float,
        choi_active_mask: np.ndarray | None = None,
        choi_failure_records: tuple[dict[str, Any], ...] = (),
        active_top_layer_indices: np.ndarray | None = None,
        full_nlayer: int | None = None,
        **_: Any,
    ) -> None:
        ll = np.asarray(sigma_ll)
        active_mask = (
            np.ones((ll.shape[0],), dtype=bool)
            if choi_active_mask is None
            else np.asarray(choi_active_mask, dtype=bool).reshape(-1)
        )
        self.sigma_ll = np.array(ll[0], copy=True)
        self.records.append(
            {
                "cycle": int(cycle),
                "min_abs_d": float(min_abs_d),
                "choi_active": bool(active_mask[0]),
                "choi_failure_records": [dict(record) for record in choi_failure_records],
                "choi_basis_size": int(ll.shape[-1]),
                "full_nlayer": None if full_nlayer is None else int(full_nlayer),
                "active_top_layer_indices": None
                if active_top_layer_indices is None
                else np.asarray(active_top_layer_indices, dtype=np.int64).tolist(),
            }
        )
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def alpha_tag(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
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


def write_csv_atomic(path: Path | str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def parse_cpu_list(text: str | None) -> list[int] | None:
    if text is None or str(text).strip() == "":
        return None
    cpus: list[int] = []
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            stop = int(right)
            if stop < start:
                raise ValueError(f"Invalid CPU range: {part!r}")
            cpus.extend(range(start, stop + 1))
        else:
            cpus.append(int(part))
    deduped = sorted(set(cpus))
    if not deduped:
        raise ValueError("--cpu-list did not contain any CPUs.")
    return deduped


def format_cpu_list(cpus: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(int(cpu)) for cpu in cpus)


def get_affinity_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        except Exception:
            pass
    count = os.cpu_count() or 1
    return list(range(count))


def sample_cpu_usage(interval: float = 0.75) -> dict[int, float]:
    try:
        import psutil

        usage = psutil.cpu_percent(interval=float(interval), percpu=True)
        return {idx: float(value) for idx, value in enumerate(usage)}
    except Exception:
        return {}


def free_cpu_pool(limit: int | None = None, interval: float = 0.75) -> list[int]:
    allowed = get_affinity_cpus()
    usage = sample_cpu_usage(interval=interval)
    ordered = sorted(allowed, key=lambda cpu: (usage.get(cpu, 100.0), cpu)) if usage else allowed
    if limit is not None:
        ordered = ordered[: max(1, int(limit))]
    return ordered


def choose_parallelism(samples: int, cpu_pool: list[int], max_workers: int | None) -> int:
    if max_workers is not None:
        requested = max(1, int(max_workers))
    else:
        requested = min(len(cpu_pool), DEFAULT_MAX_WORKERS_PER_RUN)
        try:
            import psutil

            avail = psutil.virtual_memory().available
            mem_cap = max(1, int(avail // WORKER_MEMORY_ESTIMATE_BYTES))
            requested = min(requested, mem_cap)
        except Exception:
            pass
    return max(1, min(int(samples), len(cpu_pool), DEFAULT_MAX_WORKERS_PER_RUN, requested))


def current_git_commit() -> str | None:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def canonical_init_mode(label: str) -> str:
    if label == "random":
        return "default"
    if label == "maxmix":
        return "maxmix"
    raise ValueError(f"Unsupported init mode: {label!r}")


def parse_alpha_values(text: str | None) -> list[float]:
    if text is None or str(text).strip() == "":
        return list(DEFAULT_ALPHA_VALUES)
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("--alpha-values did not contain any values.")
    for value in values:
        if not np.isfinite(value):
            raise ValueError(f"Invalid alpha value: {value!r}")
    return values


def parse_ny_values(text: str | None, default_ny: int) -> list[int]:
    if text is None or str(text).strip() == "":
        return [int(default_ny)]
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("--ny-values did not contain any values.")
    for value in values:
        if value < 1:
            raise ValueError(f"Invalid Ny value: {value!r}")
    return values


def global_gaussian_entropy(G: np.ndarray, entropy_eps: float = 1e-12) -> float:
    covariance = np.asarray(G, dtype=np.complex128)
    occupation = 0.5 * (covariance + np.eye(covariance.shape[0], dtype=np.complex128))
    occupation = 0.5 * (occupation + occupation.conj().T)
    eigvals = np.linalg.eigvalsh(occupation)
    if not np.all(np.isfinite(eigvals)):
        raise FloatingPointError("Occupation eigenvalues contain non-finite values.")
    clipped = np.clip(eigvals, float(entropy_eps), 1.0 - float(entropy_eps))
    return -float(np.sum(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)))


def parse_dwtrunc_values(text: str) -> list[int]:
    text = str(text).strip().lower()
    if text == "both":
        return [0, 1]
    if text in ("0", "false", "dwtrunc0"):
        return [0]
    if text in ("1", "true", "dwtrunc1"):
        return [1]
    raise ValueError("--dwtrunc must be one of 0, 1, or both.")


def sweep_config(dwtrunc: int, alpha_1: float, alpha_2: float) -> SweepConfig:
    flag = bool(int(dwtrunc))
    return SweepConfig(
        dw_label=f"dwtrunc{int(flag)}",
        DW=True,
        dw_truncation=flag,
        meas_slab_only=flag,
        alpha_1=float(alpha_1),
        alpha_2=float(alpha_2),
    )


def particle_transfer_spectrum_from_sigma_ll(
    sigma_ll: np.ndarray,
    *,
    cycle: int,
    endpoint_tol: float = 1e-12,
    spectral_tol: float = 1e-8,
) -> dict[str, Any]:
    if int(cycle) <= 0:
        raise ValueError("cycle must be positive for transfer spectra.")
    herm = 0.5 * (np.asarray(sigma_ll, dtype=np.complex128) + np.asarray(sigma_ll, dtype=np.complex128).conj().T)
    a_values = np.linalg.eigvalsh(herm)
    raw_min = float(np.min(a_values))
    raw_max = float(np.max(a_values))
    if raw_min < -1.0 - float(spectral_tol) or raw_max > 1.0 + float(spectral_tol):
        raise FloatingPointError(
            f"Sigma_LL eigenvalues outside [-1,1] beyond tolerance: [{raw_min:.6e}, {raw_max:.6e}]"
        )
    clipped = np.clip(a_values.real, -1.0, 1.0)
    particle_zeros = clipped >= 1.0 - float(endpoint_tol)
    particle_poles = clipped <= -1.0 + float(endpoint_tol)
    finite = ~(particle_zeros | particle_poles)
    exponents = np.empty_like(clipped, dtype=np.float64)
    exponents[particle_zeros] = -np.inf
    exponents[particle_poles] = np.inf
    exponents[finite] = (
        np.log1p(-clipped[finite]) - np.log1p(clipped[finite])
    ) / (2.0 * float(cycle))
    order = np.argsort(exponents)
    exponents_sorted = exponents[order]
    a_sorted = clipped[order]
    finite_sorted = np.isfinite(exponents_sorted)
    singular_values = np.empty_like(exponents_sorted)
    with np.errstate(over="ignore", under="ignore"):
        singular_values[:] = np.exp(float(cycle) * exponents_sorted)
    finite_exponents = exponents_sorted[finite_sorted]
    gap = float(np.min(np.abs(finite_exponents))) if finite_exponents.size else float("nan")
    return {
        "transfer_exponents": exponents_sorted,
        "transfer_singular_values": singular_values,
        "transfer_a_eigenvalues": a_sorted,
        "transfer_gap": gap,
        "finite_eigenstate_count": int(np.count_nonzero(finite)),
        "particle_zero_count": int(np.count_nonzero(particle_zeros)),
        "particle_pole_count": int(np.count_nonzero(particle_poles)),
        "a_eigenvalue_min": raw_min,
        "a_eigenvalue_max": raw_max,
    }


def load_sample_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def sample_file_is_valid(path: Path | str) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = load_sample_npz(path)
        required = {
            "sample_seed",
            "sample_index",
            "alpha_1",
            "transfer_singular_values",
            "transfer_exponents",
            "transfer_gap",
            "finite_eigenstate_count",
            "choi_min_abs_d",
            "choi_basis_size",
        }
        return required.issubset(payload)
    except Exception:
        return False


def run_one_sample(spec_payload: dict[str, Any]) -> dict[str, Any]:
    spec = TrajectorySpec(**spec_payload)
    output_path = Path(spec.output_path)
    if output_path.exists() and not spec.overwrite and sample_file_is_valid(output_path):
        return {
            "sample_index": spec.sample_index,
            "output_path": str(output_path),
            "status": "cached",
            "elapsed_s": 0.0,
            "transfer_gap": float(load_sample_npz(output_path)["transfer_gap"]),
        }

    started = time.perf_counter()
    np.random.seed(int(spec.seed) & 0xFFFFFFFF)
    if spec.cpu_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu) for cpu in spec.cpu_affinity))

    config = sweep_config(1 if spec.dw_label == "dwtrunc1" else 0, spec.alpha_1, spec.alpha_2)
    with threadpool_limits(limits=1):
        model = classA_U1FGTN(
            spec.nx,
            spec.ny,
            DW=config.DW,
            nshell=spec.nshell,
            alpha_1=config.alpha_1,
            alpha_2=config.alpha_2,
            trial_orbitals=TRIAL_ORBITALS,
            dw_truncation=config.dw_truncation,
        )
        model.construct_OW_projectors(
            nshell=spec.nshell,
            DW=config.DW,
            trial_orbitals=TRIAL_ORBITALS,
            dw_truncation=config.dw_truncation,
        )

        entropy_cycles: list[int] = []
        global_entropy: list[float] = []

        def observe_cycle(*, cycle: int, G: np.ndarray, **_: Any) -> None:
            if not spec.track_global_entropy:
                return None
            entropy_cycles.append(int(cycle))
            global_entropy.append(global_gaussian_entropy(G))
            return None

        choi_observer = ChoiSpectrumObserver()
        result = model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=spec.cycles,
            samples=1,
            save=False,
            n_a=DEFAULT_N_A,
            p_gain=DEFAULT_P_GAIN,
            p_loss=DEFAULT_P_LOSS,
            perfect_correction=False,
            sequence="raster_y",
            meas_slab_only=config.meas_slab_only,
            parallelize_samples=False,
            init_mode=spec.canonical_init_mode,
            cycle_observer=observe_cycle if spec.track_global_entropy else None,
            track_choi=True,
            choi_observer=choi_observer,
            choi_observer_cycles=[spec.cycles],
            choi_failure_mode="censor",
        )
        if choi_observer.sigma_ll is None:
            raise RuntimeError("Choi observer did not capture final Sigma_LL.")

        spectrum = particle_transfer_spectrum_from_sigma_ll(choi_observer.sigma_ll, cycle=spec.cycles)
        final_record = choi_observer.records[-1] if choi_observer.records else {}

        sample_metadata = {
            "experiment": EXPERIMENT_NAME,
            "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
            "sample_index": int(spec.sample_index),
            "sample_seed": int(spec.seed),
            "init_mode": spec.init_mode_label,
            "canonical_init_mode": spec.canonical_init_mode,
            "sweep_config": asdict(config),
            "Nx": int(spec.nx),
            "Ny": int(spec.ny),
            "cycles": int(spec.cycles),
            "nshell": spec.nshell,
            "n_a": float(DEFAULT_N_A),
            "p_gain": float(DEFAULT_P_GAIN),
            "p_loss": float(DEFAULT_P_LOSS),
            "p_gain_effective": float(DEFAULT_P_GAIN),
            "p_loss_effective": float(DEFAULT_P_LOSS),
            "perfect_correction": False,
            "trial_orbitals": TRIAL_ORBITALS,
            "sequence": "raster_y",
            "cpu_affinity": list(spec.cpu_affinity),
            "started_at": utc_now(),
            "save_final_g": bool(spec.save_final_g),
            "track_global_entropy": bool(spec.track_global_entropy),
            "choi_basis": "reduced_topological_slab" if config.meas_slab_only else "full_top_layer",
        }

        arrays: dict[str, Any] = {
            "alpha_1": np.asarray(config.alpha_1, dtype=np.float64),
            "alpha_2": np.asarray(config.alpha_2, dtype=np.float64),
            "dw_truncation": np.asarray(config.dw_truncation, dtype=bool),
            "meas_slab_only": np.asarray(config.meas_slab_only, dtype=bool),
            "sample_seed": np.asarray(int(spec.seed), dtype=np.uint32),
            "sample_index": np.asarray(int(spec.sample_index), dtype=np.int64),
            "cycle_count": np.asarray(int(spec.cycles), dtype=np.int64),
            "Nx": np.asarray(int(spec.nx), dtype=np.int64),
            "Ny": np.asarray(int(spec.ny), dtype=np.int64),
            "transfer_singular_values": spectrum["transfer_singular_values"],
            "transfer_exponents": spectrum["transfer_exponents"],
            "transfer_a_eigenvalues": spectrum["transfer_a_eigenvalues"],
            "transfer_gap": np.asarray(spectrum["transfer_gap"], dtype=np.float64),
            "finite_eigenstate_count": np.asarray(spectrum["finite_eigenstate_count"], dtype=np.int64),
            "particle_zero_count": np.asarray(spectrum["particle_zero_count"], dtype=np.int64),
            "particle_pole_count": np.asarray(spectrum["particle_pole_count"], dtype=np.int64),
            "a_eigenvalue_min": np.asarray(spectrum["a_eigenvalue_min"], dtype=np.float64),
            "a_eigenvalue_max": np.asarray(spectrum["a_eigenvalue_max"], dtype=np.float64),
            "choi_min_abs_d": np.asarray(final_record.get("min_abs_d", np.nan), dtype=np.float64),
            "choi_active_final": np.asarray(final_record.get("choi_active", False), dtype=bool),
            "choi_basis_size": np.asarray(final_record.get("choi_basis_size", 0), dtype=np.int64),
            "choi_full_nlayer": np.asarray(
                -1 if final_record.get("full_nlayer") is None else final_record.get("full_nlayer"),
                dtype=np.int64,
            ),
            "choi_failure_records_json": np.asarray(
                json.dumps(jsonable(final_record.get("choi_failure_records", [])), sort_keys=True)
            ),
            "metadata_json": np.asarray(json.dumps(jsonable(sample_metadata), sort_keys=True)),
        }
        if spec.track_global_entropy:
            arrays["entropy_cycle"] = np.asarray(entropy_cycles, dtype=np.int64)
            arrays["global_entropy"] = np.asarray(global_entropy, dtype=np.float64)
        if spec.save_final_g:
            arrays["final_G"] = np.asarray(result["G_final"][0], dtype=np.complex128)

        save_npz_atomic(output_path, **arrays)

    return {
        "sample_index": spec.sample_index,
        "output_path": str(output_path),
        "status": "completed",
        "elapsed_s": time.perf_counter() - started,
        "transfer_gap": float(spectrum["transfer_gap"]),
    }


def run_dir_for(output_root: Path, init_mode: str, ny: int, dw_label: str, alpha_1: float) -> Path:
    return output_root / init_mode / f"Ny_{int(ny)}" / dw_label / f"alpha_{alpha_tag(alpha_1)}"


def build_specs(
    args: argparse.Namespace,
    *,
    config: SweepConfig,
    run_dir: Path,
    cpu_pool: list[int],
) -> list[TrajectorySpec]:
    samples_dir = run_dir / "samples"
    specs = []
    for sample_idx in range(int(args.samples)):
        seed = stable_seed(
            EXPERIMENT_NAME,
            args.init_mode,
            config.dw_label,
            config.alpha_1,
            config.alpha_2,
            args.nx,
            args.ny,
            args.cycles,
            args.nshell,
            sample_idx,
        )
        specs.append(
            TrajectorySpec(
                init_mode_label=args.init_mode,
                canonical_init_mode=canonical_init_mode(args.init_mode),
                dw_label=config.dw_label,
                nx=int(args.nx),
                ny=int(args.ny),
                cycles=int(args.cycles),
                nshell=args.nshell,
                alpha_1=float(config.alpha_1),
                alpha_2=float(config.alpha_2),
                sample_index=int(sample_idx),
                seed=int(seed),
                output_path=str(samples_dir / f"sample_{sample_idx:04d}.npz"),
                cpu_affinity=tuple(cpu_pool),
                overwrite=bool(args.overwrite),
                save_final_g=bool(args.save_final_G),
                track_global_entropy=bool(args.track_global_entropy),
            )
        )
    return specs


def aggregate_alpha(run_dir: Path, *, samples: int, track_global_entropy: bool) -> dict[str, Any]:
    sample_paths = [run_dir / "samples" / f"sample_{idx:04d}.npz" for idx in range(int(samples))]
    missing = [str(path) for path in sample_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot aggregate; missing sample files: {missing[:5]}")

    sample_indices: list[int] = []
    seeds: list[int] = []
    singular_values: list[np.ndarray] = []
    exponents: list[np.ndarray] = []
    gaps: list[float] = []
    finite_counts: list[int] = []
    min_abs_d: list[float] = []
    active_final: list[bool] = []
    basis_size: list[int] = []
    entropy_cycles: np.ndarray | None = None
    entropy_values: list[np.ndarray] = []
    alpha_1 = float("nan")
    alpha_2 = float("nan")
    nx = -1
    ny = -1
    cycles = -1

    for path in sample_paths:
        payload = load_sample_npz(path)
        alpha_1 = float(payload["alpha_1"])
        alpha_2 = float(payload["alpha_2"])
        nx = int(payload["Nx"])
        ny = int(payload["Ny"])
        cycles = int(payload["cycle_count"])
        sample_indices.append(int(payload["sample_index"]))
        seeds.append(int(payload["sample_seed"]))
        singular_values.append(np.asarray(payload["transfer_singular_values"], dtype=np.float64))
        exponents.append(np.asarray(payload["transfer_exponents"], dtype=np.float64))
        gaps.append(float(payload["transfer_gap"]))
        finite_counts.append(int(payload["finite_eigenstate_count"]))
        min_abs_d.append(float(payload["choi_min_abs_d"]))
        active_final.append(bool(payload["choi_active_final"]))
        basis_size.append(int(payload["choi_basis_size"]))
        if track_global_entropy:
            if "global_entropy" not in payload or "entropy_cycle" not in payload:
                raise ValueError(f"{path}: missing maxmix entropy arrays.")
            cycle_arr = np.asarray(payload["entropy_cycle"], dtype=np.int64)
            entropy_arr = np.asarray(payload["global_entropy"], dtype=np.float64)
            if entropy_cycles is None:
                entropy_cycles = cycle_arr
            elif not np.array_equal(entropy_cycles, cycle_arr):
                raise ValueError(f"{path}: entropy_cycle does not match previous samples.")
            entropy_values.append(entropy_arr)

    singular_stack = np.stack(singular_values, axis=0)
    exponent_stack = np.stack(exponents, axis=0)
    gap_arr = np.asarray(gaps, dtype=np.float64)
    gap_std = float(np.std(gap_arr, ddof=1)) if gap_arr.size > 1 else 0.0
    payload = {
        "Nx": nx,
        "Ny": ny,
        "cycles": cycles,
        "alpha_1": alpha_1,
        "alpha_2": alpha_2,
        "sample_count": int(gap_arr.size),
        "transfer_gap_mean": float(np.nanmean(gap_arr)),
        "transfer_gap_std": gap_std,
        "transfer_gap_sem": float(gap_std / math.sqrt(gap_arr.size)) if gap_arr.size > 1 else 0.0,
        "transfer_gap_min": float(np.nanmin(gap_arr)),
        "transfer_gap_max": float(np.nanmax(gap_arr)),
        "finite_eigenstate_count_mean": float(np.mean(finite_counts)),
        "choi_active_fraction": float(np.mean(active_final)),
        "choi_basis_size": int(basis_size[0]) if basis_size else 0,
        "choi_min_abs_d_min": float(np.nanmin(np.asarray(min_abs_d, dtype=np.float64))),
    }
    save_npz_atomic(
        run_dir / "transfer_spectrum_summary.npz",
        sample_index=np.asarray(sample_indices, dtype=np.int64),
        sample_seed=np.asarray(seeds, dtype=np.uint32),
        Nx=np.asarray(nx, dtype=np.int64),
        Ny=np.asarray(ny, dtype=np.int64),
        cycles=np.asarray(cycles, dtype=np.int64),
        alpha_1=np.asarray(alpha_1, dtype=np.float64),
        alpha_2=np.asarray(alpha_2, dtype=np.float64),
        transfer_singular_values=singular_stack,
        transfer_exponents=exponent_stack,
        transfer_gap=gap_arr,
        finite_eigenstate_count=np.asarray(finite_counts, dtype=np.int64),
        choi_min_abs_d=np.asarray(min_abs_d, dtype=np.float64),
        choi_active_final=np.asarray(active_final, dtype=bool),
        choi_basis_size=np.asarray(basis_size, dtype=np.int64),
        transfer_singular_values_mean=np.mean(singular_stack, axis=0),
        transfer_exponents_mean=np.mean(exponent_stack, axis=0),
    )
    write_csv_atomic(
        run_dir / "transfer_gap_summary.csv",
        [payload],
        [
            "Nx",
            "Ny",
            "cycles",
            "alpha_1",
            "alpha_2",
            "sample_count",
            "transfer_gap_mean",
            "transfer_gap_std",
            "transfer_gap_sem",
            "transfer_gap_min",
            "transfer_gap_max",
            "finite_eigenstate_count_mean",
            "choi_active_fraction",
            "choi_basis_size",
            "choi_min_abs_d_min",
        ],
    )

    if track_global_entropy:
        if entropy_cycles is None:
            raise ValueError("Entropy tracking requested but no entropy arrays were loaded.")
        entropy_stack = np.stack(entropy_values, axis=0)
        entropy_std = np.std(entropy_stack, axis=0, ddof=1) if entropy_stack.shape[0] > 1 else np.zeros_like(entropy_stack[0])
        entropy_sem = entropy_std / math.sqrt(entropy_stack.shape[0]) if entropy_stack.shape[0] > 1 else entropy_std
        entropy_rows: list[dict[str, Any]] = []
        entropy_summary_rows: list[dict[str, Any]] = []
        for sample_pos, sample_idx in enumerate(sample_indices):
            for cycle, value in zip(entropy_cycles.tolist(), entropy_stack[sample_pos].tolist()):
                entropy_rows.append(
                    {
                        "sample_index": int(sample_idx),
                        "sample_seed": int(seeds[sample_pos]),
                        "cycle": int(cycle),
                        "global_entropy": float(value),
                    }
                )
        for pos, cycle in enumerate(entropy_cycles.tolist()):
            entropy_summary_rows.append(
                {
                    "cycle": int(cycle),
                    "global_entropy_mean": float(np.mean(entropy_stack[:, pos])),
                    "global_entropy_std": float(entropy_std[pos]),
                    "global_entropy_sem": float(entropy_sem[pos]),
                    "sample_count": int(entropy_stack.shape[0]),
                }
            )
        write_csv_atomic(
            run_dir / "global_entropy_vs_cycle.csv",
            entropy_rows,
            ["sample_index", "sample_seed", "cycle", "global_entropy"],
        )
        write_csv_atomic(
            run_dir / "global_entropy_summary.csv",
            entropy_summary_rows,
            ["cycle", "global_entropy_mean", "global_entropy_std", "global_entropy_sem", "sample_count"],
        )
        save_npz_atomic(
            run_dir / "global_entropy_summary.npz",
            sample_index=np.asarray(sample_indices, dtype=np.int64),
            sample_seed=np.asarray(seeds, dtype=np.uint32),
            entropy_cycle=entropy_cycles,
            global_entropy=entropy_stack,
            global_entropy_mean=np.mean(entropy_stack, axis=0),
            global_entropy_std=entropy_std,
            global_entropy_sem=entropy_sem,
        )
    return payload


def write_run_config(args: argparse.Namespace, output_root: Path, cpu_pool: list[int], alpha_values: list[float], ny_values: list[int]) -> None:
    payload = {
        "experiment": EXPERIMENT_NAME,
        "created_at": utc_now(),
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "init_mode": args.init_mode,
        "canonical_init_mode": canonical_init_mode(args.init_mode),
        "dwtrunc_requested": args.dwtrunc,
        "alpha_values": alpha_values,
        "alpha_2": float(args.alpha_2),
        "Nx": int(args.nx),
        "Ny": int(args.ny),
        "ny_values": ny_values,
        "cycles": int(args.cycles) if args.cycles is not None else "2*Ny",
        "samples": int(args.samples),
        "nshell": args.nshell,
        "n_a": float(DEFAULT_N_A),
        "p_gain": float(DEFAULT_P_GAIN),
        "p_loss": float(DEFAULT_P_LOSS),
        "p_gain_effective": float(DEFAULT_P_GAIN),
        "p_loss_effective": float(DEFAULT_P_LOSS),
        "perfect_correction": False,
        "trial_orbitals": TRIAL_ORBITALS,
        "cpu_pool": cpu_pool,
        "max_workers_requested": int(args.max_workers) if args.max_workers is not None else None,
        "max_workers_effective_cap": DEFAULT_MAX_WORKERS_PER_RUN,
        "git_commit": current_git_commit(),
        "save_final_G": bool(args.save_final_G),
        "track_global_entropy": bool(args.track_global_entropy),
        "choi_observer_cycles": "final cycle for each Ny",
        "choi_failure_mode": "censor",
        "gap_definition": "min(abs(finite transfer exponents)) from final-cycle Choi Sigma_LL",
    }
    write_json_atomic(output_root / args.init_mode / "run_config.json", payload)


def run_specs(specs: list[TrajectorySpec], *, max_workers: int, desc: str) -> None:
    failures: list[dict[str, Any]] = []
    ctx = get_context("spawn")
    executor: ProcessPoolExecutor | None = None
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        futures = {executor.submit(run_one_sample, asdict(spec)): spec for spec in specs}
        with tqdm(total=len(futures), desc=desc, unit="sample") as pbar:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    spec = futures[future]
                    try:
                        result = future.result()
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"sample={result['sample_index']} {result['status']} gap={result['transfer_gap']:.4g}",
                            refresh=False,
                        )
                        print(
                            f"[sample] {spec.init_mode_label}/{spec.dw_label}/alpha={spec.alpha_1:g} "
                            f"sample={result['sample_index']:04d} {result['status']} "
                            f"gap={result['transfer_gap']:.6g} -> {result['output_path']}",
                            flush=True,
                        )
                    except Exception as exc:
                        tb = traceback.format_exc()
                        failures.append(
                            {
                                "sample_index": spec.sample_index,
                                "output_path": spec.output_path,
                                "error": repr(exc),
                                "traceback": tb,
                            }
                        )
                        pbar.update(1)
                        print(f"[error] sample={spec.sample_index:04d} failed: {exc!r}", flush=True)
                        print(tb, flush=True)
        if failures:
            failure_path = Path(specs[0].output_path).parents[1] / "failures.json"
            write_json_atomic(failure_path, failures)
            raise RuntimeError(f"{len(failures)} samples failed; see {failure_path}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def run_sweep(args: argparse.Namespace) -> int:
    alpha_values = parse_alpha_values(args.alpha_values)
    ny_values = parse_ny_values(args.ny_values, args.ny)
    cycle_override = args.cycles
    dw_values = parse_dwtrunc_values(args.dwtrunc)
    output_root = Path(args.output_root).resolve()
    requested_cpu_pool = parse_cpu_list(args.cpu_list)
    discovered_pool = requested_cpu_pool if requested_cpu_pool is not None else free_cpu_pool()
    max_workers = choose_parallelism(args.samples, discovered_pool, args.max_workers)
    cpu_pool = discovered_pool[:max_workers]

    print("=" * 80, flush=True)
    print(f"[run] {EXPERIMENT_NAME}", flush=True)
    print(f"[run] init_mode={args.init_mode} canonical_init={canonical_init_mode(args.init_mode)}", flush=True)
    print(f"[run] dwtrunc={args.dwtrunc} alpha_values={','.join(f'{value:g}' for value in alpha_values)} alpha_2={args.alpha_2:g}", flush=True)
    print(f"[run] Nx={args.nx} Ny_values={','.join(str(v) for v in ny_values)} cycles={'2*Ny' if args.cycles is None else args.cycles} samples={args.samples}", flush=True)
    print(f"[run] nshell={args.nshell} p_gain=0 p_loss=0 save_final_G={args.save_final_G} track_global_entropy={args.track_global_entropy}", flush=True)
    print(f"[run] output={output_root / args.init_mode}", flush=True)
    print(f"[run] cpu_pool={format_cpu_list(cpu_pool)} max_workers={max_workers}", flush=True)
    print("=" * 80, flush=True)

    write_run_config(args, output_root, cpu_pool, alpha_values, ny_values)
    if args.preflight_only:
        print("[preflight] Completed config write; exiting because --preflight-only was set.", flush=True)
        return 0

    all_rows: list[dict[str, Any]] = []
    for ny in tqdm(ny_values, desc=f"{args.init_mode}: Ny", unit="size"):
        args.ny = int(ny)
        args.cycles = int(cycle_override) if cycle_override is not None else 2 * int(args.ny)
        for dwtrunc in tqdm(dw_values, desc=f"{args.init_mode}/Ny{ny}: dwtrunc", unit="config"):
            for alpha_1 in tqdm(alpha_values, desc=f"{args.init_mode}/Ny{ny}/dwtrunc{dwtrunc}: alpha", unit="alpha"):
                config = sweep_config(dwtrunc, alpha_1, args.alpha_2)
                run_dir = run_dir_for(output_root, args.init_mode, args.ny, config.dw_label, alpha_1)
                samples_dir = run_dir / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
                specs = build_specs(args, config=config, run_dir=run_dir, cpu_pool=cpu_pool)

                if not args.overwrite and not args.resume:
                    existing = [spec.output_path for spec in specs if Path(spec.output_path).exists()]
                    if existing:
                        raise FileExistsError(
                            "Found existing sample outputs. Pass --resume to reuse them or --overwrite to replace. "
                            f"First: {existing[0]}"
                        )

                print(
                    f"[alpha] init={args.init_mode} Ny={args.ny} {config.dw_label} alpha_1={alpha_1:g} "
                    f"output={run_dir}",
                    flush=True,
                )
                run_specs(
                    specs,
                    max_workers=max_workers,
                    desc=f"{args.init_mode}/Ny{args.ny}/{config.dw_label}/alpha={alpha_1:g}",
                )
                summary = aggregate_alpha(run_dir, samples=args.samples, track_global_entropy=args.track_global_entropy)
                summary.update(
                    {
                        "init_mode": args.init_mode,
                        "dw_label": config.dw_label,
                        "dw_truncation": bool(config.dw_truncation),
                        "meas_slab_only": bool(config.meas_slab_only),
                        "run_dir": str(run_dir),
                    }
                )
                all_rows.append(summary)
                print(
                    f"[aggregate] {args.init_mode}/Ny{args.ny}/{config.dw_label}/alpha={alpha_1:g} "
                    f"gap_mean={summary['transfer_gap_mean']:.6g} gap_sem={summary['transfer_gap_sem']:.3g}",
                    flush=True,
                )

    fieldnames = [
        "init_mode",
        "Nx",
        "Ny",
        "cycles",
        "dw_label",
        "dw_truncation",
        "meas_slab_only",
        "alpha_1",
        "alpha_2",
        "sample_count",
        "transfer_gap_mean",
        "transfer_gap_std",
        "transfer_gap_sem",
        "transfer_gap_min",
        "transfer_gap_max",
        "finite_eigenstate_count_mean",
        "choi_active_fraction",
        "choi_basis_size",
        "choi_min_abs_d_min",
        "run_dir",
    ]
    summary_path = output_root / args.init_mode / "transfer_gap_by_alpha.csv"
    write_csv_atomic(summary_path, all_rows, fieldnames)
    print(f"[done] Wrote sweep summary to {summary_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-mode", choices=("random", "maxmix"), default="random")
    parser.add_argument("--dwtrunc", choices=("0", "1", "both"), default="both")
    parser.add_argument("--alpha-values", type=str, default=None, help="Comma-separated alpha_1 values.")
    parser.add_argument("--alpha-2", type=float, default=DEFAULT_ALPHA_2)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--ny-values", type=str, default=None, help="Comma-separated Ny values; defaults to --ny.")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--nshell", type=float, default=DEFAULT_NSHELL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--cpu-list", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-final-G", dest="save_final_G", action="store_true")
    parser.add_argument("--track-global-entropy", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--list-free-cpus", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_free_cpus:
        print(format_cpu_list(free_cpu_pool()))
        return 0
    if args.samples < 1:
        raise ValueError("--samples must be positive.")
    if args.nx < 1 or args.ny < 1:
        raise ValueError("--nx and --ny must be positive.")
    if args.cycles is not None and args.cycles < 1:
        raise ValueError("--cycles must be positive.")
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
