#!/usr/bin/env python3
"""Run no-feedback domain-wall Chern/transfer campaigns on CPU.

The canonical CPU Markov-circuit entry point owns all dynamics. This script
parallelizes independent samples externally so cycle observers and Choi tracking
can remain serial inside each worker.
"""

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
from multiprocessing import get_context
from pathlib import Path
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
EXPERIMENT_NAME = "no_feedback_domain_wall_chern"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_NX = 20
DEFAULT_NY = 20
DEFAULT_SAMPLES = 10
DEFAULT_NSHELL = 1
DEFAULT_RADIUS = 6.0
DEFAULT_N_A = 0.5
DEFAULT_P_GAIN = 0.0
DEFAULT_P_LOSS = 0.0
DEFAULT_MAX_WORKERS_PER_RUN = 12
TRIAL_ORBITALS = "X"
WORKER_MEMORY_ESTIMATE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    DW: bool
    dw_truncation: bool
    meas_slab_only: bool
    alpha_1: float
    alpha_2: float
    track_transfer_spectrum: bool


CAMPAIGNS: dict[str, CampaignConfig] = {
    "uniform_alpha1_no_dw": CampaignConfig(
        name="uniform_alpha1_no_dw",
        DW=False,
        dw_truncation=False,
        meas_slab_only=False,
        alpha_1=1.0,
        alpha_2=30.0,
        track_transfer_spectrum=False,
    ),
    "domain_wall_untruncated_alpha1_alpha30": CampaignConfig(
        name="domain_wall_untruncated_alpha1_alpha30",
        DW=True,
        dw_truncation=False,
        meas_slab_only=False,
        alpha_1=1.0,
        alpha_2=30.0,
        track_transfer_spectrum=True,
    ),
    "domain_wall_truncated_alpha1_alpha30": CampaignConfig(
        name="domain_wall_truncated_alpha1_alpha30",
        DW=True,
        dw_truncation=True,
        meas_slab_only=True,
        alpha_1=1.0,
        alpha_2=30.0,
        track_transfer_spectrum=True,
    ),
}


@dataclass(frozen=True)
class TrajectorySpec:
    campaign: str
    init_mode_label: str
    canonical_init_mode: str
    nx: int
    ny: int
    cycles: int
    nshell: float | int | None
    radius: float
    sample_index: int
    seed: int
    output_path: str
    cpu_affinity: tuple[int, ...]
    overwrite: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if usage:
        ordered = sorted(allowed, key=lambda cpu: (usage.get(cpu, 100.0), cpu))
    else:
        ordered = allowed
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
    return max(1, min(int(samples), len(cpu_pool), requested))


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


def disk_mask(nx: int, ny: int, xref: int, yref: int, radius: float) -> np.ndarray:
    inside = np.zeros((int(nx), int(ny)), dtype=bool)
    rr = float(radius) * float(radius)
    ymax = int(math.floor(float(radius)))
    for dy in range(-ymax, ymax + 1):
        y = int(yref) + dy
        if y < 0 or y >= ny:
            continue
        max_dx = int(math.floor(math.sqrt(max(0.0, rr - dy * dy))))
        x0 = max(0, int(xref) - max_dx)
        x1 = min(int(nx) - 1, int(xref) + max_dx)
        if x0 <= x1:
            inside[x0 : x1 + 1, y] = True
    return inside


def assert_radius_inside_domain_wall(config: CampaignConfig, nx: int, ny: int, radius: float) -> dict[str, Any]:
    if not config.DW:
        return {"checked": False, "reason": "campaign_has_no_domain_wall"}
    model = classA_U1FGTN(
        nx,
        ny,
        DW=True,
        nshell=DEFAULT_NSHELL,
        alpha_1=config.alpha_1,
        alpha_2=config.alpha_2,
        trial_orbitals=TRIAL_ORBITALS,
        dw_truncation=config.dw_truncation,
    )
    xref = int(nx) // 2
    yref = int(ny) // 2
    mask = disk_mask(nx, ny, xref=xref, yref=yref, radius=radius)
    xs, _ = np.nonzero(mask)
    x_min, x_max = sorted(int(x) for x in model.DW_loc)
    outside = xs[(xs < x_min) | (xs > x_max)]
    payload = {
        "checked": True,
        "xref": xref,
        "yref": yref,
        "radius": float(radius),
        "DW_loc": [x_min, x_max],
        "disk_x_min": int(np.min(xs)) if xs.size else None,
        "disk_x_max": int(np.max(xs)) if xs.size else None,
        "disk_site_count": int(np.count_nonzero(mask)),
        "slab_contained": bool(outside.size == 0),
    }
    if outside.size:
        raise ValueError(
            f"radius={radius:g} disk centered at ({xref},{yref}) leaves DW slab "
            f"x={x_min}..{x_max}; disk x-range is {payload['disk_x_min']}..{payload['disk_x_max']}."
        )
    return payload


def fermi_projector_diagnostics(alpha: float, radius: float, grid_n: int = 512) -> dict[str, Any]:
    n = int(grid_n)
    k = 2.0 * np.pi * np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = float(alpha) - np.cos(kx) - np.cos(ky)
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    blocks_k = (
        0.5 * (1.0 - nz),
        -0.5 * (nx - 1j * ny),
        -0.5 * (nx + 1j * ny),
        0.5 * (1.0 + nz),
    )
    blocks_r = [np.fft.fftshift(np.fft.ifft2(block)) for block in blocks_k]
    coords = np.arange(-n // 2, n // 2)
    rx, ry = np.meshgrid(coords, coords, indexing="ij")
    rr = np.sqrt(rx * rx + ry * ry)
    norm2 = sum(np.abs(block) ** 2 for block in blocks_r)
    total = float(norm2.sum())
    tail_fraction = float(norm2[rr > float(radius)].sum() / total)

    shell_r: list[float] = []
    shell_envelope: list[float] = []
    max_shell = min(15, n // 4)
    for shell in range(0, max_shell):
        mask = (rr >= shell) & (rr < shell + 1)
        if np.any(mask):
            value = float(np.sqrt(np.max(norm2[mask])))
            shell_r.append(shell + 0.5)
            shell_envelope.append(value)

    fit_indices = [idx for idx, r_val in enumerate(shell_r) if 3.0 <= r_val <= 14.5 and shell_envelope[idx] > 0.0]
    if len(fit_indices) >= 2:
        xfit = np.asarray([shell_r[idx] for idx in fit_indices], dtype=np.float64)
        yfit = np.log(np.asarray([shell_envelope[idx] for idx in fit_indices], dtype=np.float64))
        slope, intercept = np.polyfit(xfit, yfit, deg=1)
        xi_exp = float(-1.0 / slope) if slope < 0 else float("inf")
    else:
        slope = float("nan")
        intercept = float("nan")
        xi_exp = float("nan")

    dk = 2.0 * np.pi / n
    nhat = np.stack([nx, ny, nz], axis=0)
    dnhat_x = (np.roll(nhat, -1, axis=1) - np.roll(nhat, 1, axis=1)) / (2.0 * dk)
    dnhat_y = (np.roll(nhat, -1, axis=2) - np.roll(nhat, 1, axis=2)) / (2.0 * dk)
    xi2_sq = float(
        np.mean(0.5 * (np.sum(dnhat_x * dnhat_x, axis=0) + np.sum(dnhat_y * dnhat_y, axis=0)))
    )
    xi2 = float(math.sqrt(max(0.0, xi2_sq)))
    return {
        "alpha": float(alpha),
        "grid_n": n,
        "radius": float(radius),
        "tail_fraction_beyond_radius": tail_fraction,
        "xi_exp_shell_envelope": xi_exp,
        "xi_exp_fit_slope": float(slope),
        "xi_exp_fit_intercept": float(intercept),
        "xi_2": xi2,
        "xi_2_squared": xi2_sq,
        "radius_over_xi_exp": float(float(radius) / xi_exp) if np.isfinite(xi_exp) and xi_exp > 0 else None,
        "radius_over_xi_2": float(float(radius) / xi2) if xi2 > 0 else None,
        "shell_r": shell_r,
        "shell_envelope": shell_envelope,
    }


def retained_ow_norm_diagnostics(nx: int, ny: int, alpha: float, nshell: float | int | None) -> dict[str, Any]:
    model = classA_U1FGTN(
        nx,
        ny,
        DW=False,
        nshell=None,
        alpha_1=float(alpha),
        alpha_2=30.0,
        trial_orbitals=TRIAL_ORBITALS,
        dw_truncation=False,
    )
    model.construct_OW_projectors(
        nshell=None,
        DW=False,
        trial_orbitals=TRIAL_ORBITALS,
        dw_truncation=False,
    )
    center = (int(nx) // 2, int(ny) // 2)
    mask: list[bool] = []
    for y in range(int(ny)):
        for x in range(int(nx)):
            dxw = ((x - center[0] + int(nx) // 2) % int(nx)) - int(nx) // 2
            dyw = ((y - center[1] + int(ny) // 2) % int(ny)) - int(ny) // 2
            keep = bool(
                classA_U1FGTN._ow_support_mask(
                    np.asarray(dxw),
                    np.asarray(dyw),
                    nshell,
                ).item()
            )
            mask.extend([keep, keep])
    mask_arr = np.asarray(mask, dtype=bool)
    arrays = {
        "Ap": model.WF_Ap,
        "Bp": model.WF_Bp,
        "Am": model.WF_Am,
        "Bm": model.WF_Bm,
    }
    retained = {}
    for label, array in arrays.items():
        chi = np.asarray(array[:, center[0], center[1]], dtype=np.complex128)
        denom = float(np.sum(np.abs(chi) ** 2))
        retained[label] = float(np.sum(np.abs(chi[mask_arr]) ** 2) / denom) if denom > 0 else float("nan")
    support_sites = int(np.count_nonzero(mask_arr) // 2)
    support_components = int(np.count_nonzero(mask_arr))
    return {
        "alpha": float(alpha),
        "nshell": None if nshell is None else float(nshell),
        "center": [center[0], center[1]],
        "support_shape": "untruncated" if nshell is None else f"{2 * int(nshell) + 1}x{2 * int(nshell) + 1}",
        "support_sites": support_sites,
        "support_components": support_components,
        "retained_norm_by_channel": retained,
        "retained_norm_min": float(min(retained.values())),
        "retained_norm_max": float(max(retained.values())),
        "form_factor_zero_winding_evidence": (
            "form_factor_analysis/docs/windowed_chern.tex reports direct continuum-k "
            "evaluation preserving the inherited overlap zero/winding for nshell=1; "
            "the decay length is supporting locality evidence, not a proof by itself."
        ),
    }


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


def sample_file_is_valid(path: Path | str, expect_transfer: bool, cycles: int) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = load_sample_npz(path)
        required = {"cycle", "real_space_chern", "final_local_chern_marker", "sample_seed", "final_G"}
        if expect_transfer:
            required.update({"transfer_singular_values", "transfer_exponents", "transfer_gap"})
        if not required.issubset(payload):
            return False
        return tuple(payload["cycle"].shape) == (int(cycles) + 1,)
    except Exception:
        return False


def run_one_sample(spec_payload: dict[str, Any]) -> dict[str, Any]:
    spec = TrajectorySpec(**spec_payload)
    config = CAMPAIGNS[spec.campaign]
    output_path = Path(spec.output_path)
    if output_path.exists() and not spec.overwrite and sample_file_is_valid(
        output_path,
        expect_transfer=config.track_transfer_spectrum,
        cycles=spec.cycles,
    ):
        return {
            "sample_index": spec.sample_index,
            "output_path": str(output_path),
            "status": "cached",
            "elapsed_s": 0.0,
        }

    started = time.perf_counter()
    np.random.seed(int(spec.seed) & 0xFFFFFFFF)
    if spec.cpu_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu) for cpu in spec.cpu_affinity))

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

        cycles_seen: list[int] = []
        chern_seen: list[float] = []

        def observe_cycle(*, cycle: int, G: np.ndarray, **_: Any) -> None:
            value = model.real_space_chern_number(
                G,
                xref=spec.nx // 2,
                yref=spec.ny // 2,
                radius=spec.radius,
            )
            cycles_seen.append(int(cycle))
            chern_seen.append(float(np.real(value)))

        choi_observer = ChoiSpectrumObserver() if config.track_transfer_spectrum else None
        run_kwargs = {
            "G_history": False,
            "progress": False,
            "cycles": spec.cycles,
            "samples": 1,
            "save": False,
            "n_a": DEFAULT_N_A,
            "p_gain": DEFAULT_P_GAIN,
            "p_loss": DEFAULT_P_LOSS,
            "perfect_correction": False,
            "sequence": "raster_y",
            "meas_slab_only": config.meas_slab_only,
            "parallelize_samples": False,
            "init_mode": spec.canonical_init_mode,
            "cycle_observer": observe_cycle,
            "track_choi": config.track_transfer_spectrum,
            "choi_observer": choi_observer,
            "choi_observer_cycles": [spec.cycles] if config.track_transfer_spectrum else None,
            "choi_failure_mode": "censor",
        }
        result = model.run_markov_circuit(**run_kwargs)
        final_g = np.asarray(result["G_final"][0], dtype=np.complex128)
        local_marker = model.local_chern_marker_flat(final_g, apply_tanh=True)

        sample_metadata = {
            "experiment": EXPERIMENT_NAME,
            "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
            "campaign": spec.campaign,
            "campaign_config": asdict(config),
            "sample_index": int(spec.sample_index),
            "sample_seed": int(spec.seed),
            "init_mode": spec.init_mode_label,
            "canonical_init_mode": spec.canonical_init_mode,
            "Nx": int(spec.nx),
            "Ny": int(spec.ny),
            "cycles": int(spec.cycles),
            "nshell": spec.nshell,
            "radius": float(spec.radius),
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
        }

        arrays: dict[str, Any] = {
            "cycle": np.asarray(cycles_seen, dtype=np.int64),
            "real_space_chern": np.asarray(chern_seen, dtype=np.float64),
            "final_local_chern_marker": np.asarray(local_marker, dtype=np.float64),
            "final_G": final_g,
            "sample_seed": np.asarray(int(spec.seed), dtype=np.uint32),
            "sample_index": np.asarray(int(spec.sample_index), dtype=np.int64),
            "metadata_json": np.asarray(json.dumps(jsonable(sample_metadata), sort_keys=True)),
        }
        if config.track_transfer_spectrum:
            if choi_observer is None or choi_observer.sigma_ll is None:
                raise RuntimeError("Choi observer did not capture final Sigma_LL.")
            spectrum = particle_transfer_spectrum_from_sigma_ll(
                choi_observer.sigma_ll,
                cycle=spec.cycles,
            )
            final_record = choi_observer.records[-1] if choi_observer.records else {}
            arrays.update(
                {
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
                }
            )
        save_npz_atomic(output_path, **arrays)

    return {
        "sample_index": spec.sample_index,
        "output_path": str(output_path),
        "status": "completed",
        "elapsed_s": time.perf_counter() - started,
    }


def aggregate_outputs(run_dir: Path, config: CampaignConfig, samples: int, cycles: int) -> None:
    sample_paths = [run_dir / "samples" / f"sample_{idx:04d}.npz" for idx in range(int(samples))]
    missing = [str(path) for path in sample_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot aggregate; missing sample files: {missing[:5]}")

    chern_rows: list[dict[str, Any]] = []
    chern_arrays: list[np.ndarray] = []
    local_markers: list[np.ndarray] = []
    sample_indices: list[int] = []
    seeds: list[int] = []
    transfer_singular_values: list[np.ndarray] = []
    transfer_exponents: list[np.ndarray] = []
    transfer_gaps: list[float] = []
    finite_counts: list[int] = []
    choi_min_abs_d: list[float] = []
    choi_active_final: list[bool] = []
    choi_basis_size: list[int] = []

    for path in sample_paths:
        payload = load_sample_npz(path)
        idx = int(payload["sample_index"])
        seed = int(payload["sample_seed"])
        sample_indices.append(idx)
        seeds.append(seed)
        cycle = np.asarray(payload["cycle"], dtype=np.int64)
        chern = np.asarray(payload["real_space_chern"], dtype=np.float64)
        if cycle.shape != (int(cycles) + 1,) or chern.shape != (int(cycles) + 1,):
            raise ValueError(f"{path}: invalid cycle/chern shapes {cycle.shape} {chern.shape}")
        chern_arrays.append(chern)
        local_markers.append(np.asarray(payload["final_local_chern_marker"], dtype=np.float64))
        for cyc, value in zip(cycle.tolist(), chern.tolist()):
            chern_rows.append(
                {
                    "sample_index": idx,
                    "sample_seed": seed,
                    "cycle": int(cyc),
                    "real_space_chern": float(value),
                }
            )
        if config.track_transfer_spectrum:
            transfer_singular_values.append(np.asarray(payload["transfer_singular_values"], dtype=np.float64))
            transfer_exponents.append(np.asarray(payload["transfer_exponents"], dtype=np.float64))
            transfer_gaps.append(float(payload["transfer_gap"]))
            finite_counts.append(int(payload["finite_eigenstate_count"]))
            choi_min_abs_d.append(float(payload["choi_min_abs_d"]))
            choi_active_final.append(bool(payload["choi_active_final"]))
            choi_basis_size.append(int(payload["choi_basis_size"]))

    chern_stack = np.stack(chern_arrays, axis=0)
    cycle_axis = np.arange(int(cycles) + 1, dtype=np.int64)
    summary_rows: list[dict[str, Any]] = []
    for pos, cyc in enumerate(cycle_axis.tolist()):
        values = chern_stack[:, pos]
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(
            {
                "cycle": int(cyc),
                "real_space_chern_mean": float(np.mean(values)),
                "real_space_chern_std": std,
                "real_space_chern_sem": float(std / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                "sample_count": int(len(values)),
            }
        )

    write_csv_atomic(
        run_dir / "chern_vs_cycle.csv",
        chern_rows,
        ["sample_index", "sample_seed", "cycle", "real_space_chern"],
    )
    write_csv_atomic(
        run_dir / "chern_summary.csv",
        summary_rows,
        ["cycle", "real_space_chern_mean", "real_space_chern_std", "real_space_chern_sem", "sample_count"],
    )

    local_stack = np.stack(local_markers, axis=0)
    local_std = np.std(local_stack, axis=0, ddof=1) if local_stack.shape[0] > 1 else np.zeros_like(local_stack[0])
    save_npz_atomic(
        run_dir / "local_chern_marker_summary.npz",
        sample_index=np.asarray(sample_indices, dtype=np.int64),
        sample_seed=np.asarray(seeds, dtype=np.uint32),
        final_local_chern_marker=local_stack,
        final_local_chern_marker_mean=np.mean(local_stack, axis=0),
        final_local_chern_marker_std=local_std,
        final_local_chern_marker_sem=local_std / math.sqrt(local_stack.shape[0]) if local_stack.shape[0] > 1 else local_std,
    )

    if config.track_transfer_spectrum:
        singular_stack = np.stack(transfer_singular_values, axis=0)
        exponent_stack = np.stack(transfer_exponents, axis=0)
        save_npz_atomic(
            run_dir / "transfer_spectrum_summary.npz",
            sample_index=np.asarray(sample_indices, dtype=np.int64),
            sample_seed=np.asarray(seeds, dtype=np.uint32),
            transfer_singular_values=singular_stack,
            transfer_exponents=exponent_stack,
            transfer_gap=np.asarray(transfer_gaps, dtype=np.float64),
            finite_eigenstate_count=np.asarray(finite_counts, dtype=np.int64),
            choi_min_abs_d=np.asarray(choi_min_abs_d, dtype=np.float64),
            choi_active_final=np.asarray(choi_active_final, dtype=bool),
            choi_basis_size=np.asarray(choi_basis_size, dtype=np.int64),
            transfer_singular_values_mean=np.mean(singular_stack, axis=0),
            transfer_exponents_mean=np.mean(exponent_stack, axis=0),
        )


def run_preflight(args: argparse.Namespace, config: CampaignConfig, run_dir: Path, cpu_pool: list[int]) -> dict[str, Any]:
    print("[preflight] Computing projector locality diagnostics for alpha=1 ...", flush=True)
    projector_diag = fermi_projector_diagnostics(config.alpha_1, radius=args.radius)
    print(
        "[preflight] radius={radius:g}, xi_exp={xi:.4g}, tail_fraction={tail:.4e}".format(
            radius=args.radius,
            xi=projector_diag["xi_exp_shell_envelope"],
            tail=projector_diag["tail_fraction_beyond_radius"],
        ),
        flush=True,
    )
    print("[preflight] Computing nshell retained-norm diagnostics ...", flush=True)
    ow_diag = retained_ow_norm_diagnostics(args.nx, args.ny, config.alpha_1, args.nshell)
    print(
        "[preflight] nshell={nshell}, support={support}, retained_norm_min={ret:.6f}".format(
            nshell=args.nshell,
            support=ow_diag["support_shape"],
            ret=ow_diag["retained_norm_min"],
        ),
        flush=True,
    )
    geometry_diag = assert_radius_inside_domain_wall(config, args.nx, args.ny, args.radius)
    if geometry_diag.get("checked"):
        print(
            "[preflight] Domain-wall slab {dw}; disk x-range {xmin}..{xmax}; contained={ok}".format(
                dw=geometry_diag["DW_loc"],
                xmin=geometry_diag["disk_x_min"],
                xmax=geometry_diag["disk_x_max"],
                ok=geometry_diag["slab_contained"],
            ),
            flush=True,
        )
    payload = {
        "experiment": EXPERIMENT_NAME,
        "created_at": utc_now(),
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "campaign": config.name,
        "campaign_config": asdict(config),
        "init_mode": args.init_mode,
        "canonical_init_mode": canonical_init_mode(args.init_mode),
        "Nx": int(args.nx),
        "Ny": int(args.ny),
        "cycles": int(args.cycles),
        "samples": int(args.samples),
        "nshell": args.nshell,
        "radius": float(args.radius),
        "n_a": float(DEFAULT_N_A),
        "p_gain": float(DEFAULT_P_GAIN),
        "p_loss": float(DEFAULT_P_LOSS),
        "p_gain_effective": float(DEFAULT_P_GAIN),
        "p_loss_effective": float(DEFAULT_P_LOSS),
        "perfect_correction": False,
        "trial_orbitals": TRIAL_ORBITALS,
        "cpu_pool": cpu_pool,
        "max_workers": int(args.max_workers) if args.max_workers is not None else None,
        "git_commit": current_git_commit(),
        "projector_radius_diagnostics": projector_diag,
        "ow_projector_diagnostics": ow_diag,
        "domain_wall_geometry_diagnostics": geometry_diag,
    }
    write_json_atomic(run_dir / "run_config.json", payload)
    return payload


def canonical_init_mode(label: str) -> str:
    if label == "random":
        return "default"
    if label == "maxmix":
        return "maxmix"
    raise ValueError(f"Unsupported init mode: {label!r}")


def run_campaign(args: argparse.Namespace) -> int:
    config = CAMPAIGNS[args.campaign]
    args.cycles = int(args.cycles) if args.cycles is not None else 2 * int(args.ny)
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / config.name / args.init_mode
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    requested_cpu_pool = parse_cpu_list(args.cpu_list)
    cpu_pool = requested_cpu_pool if requested_cpu_pool is not None else free_cpu_pool()
    max_workers = choose_parallelism(args.samples, cpu_pool, args.max_workers)
    cpu_pool = cpu_pool[:max_workers]

    print("=" * 80, flush=True)
    print(f"[run] {EXPERIMENT_NAME}", flush=True)
    print(f"[run] campaign={config.name} init_mode={args.init_mode} canonical_init={canonical_init_mode(args.init_mode)}", flush=True)
    print(f"[run] Nx={args.nx} Ny={args.ny} cycles={args.cycles} samples={args.samples}", flush=True)
    print(f"[run] nshell={args.nshell} radius={args.radius} p_gain=0 p_loss=0", flush=True)
    print(f"[run] output={run_dir}", flush=True)
    print(f"[run] cpu_pool={format_cpu_list(cpu_pool)} max_workers={max_workers}", flush=True)
    print("=" * 80, flush=True)

    run_preflight(args, config, run_dir, cpu_pool)
    if args.preflight_only:
        print("[preflight] Completed; exiting because --preflight-only was set.", flush=True)
        return 0

    specs: list[TrajectorySpec] = []
    for sample_idx in range(int(args.samples)):
        seed = stable_seed(
            EXPERIMENT_NAME,
            config.name,
            args.init_mode,
            args.nx,
            args.ny,
            args.cycles,
            args.nshell,
            args.radius,
            sample_idx,
        )
        specs.append(
            TrajectorySpec(
                campaign=config.name,
                init_mode_label=args.init_mode,
                canonical_init_mode=canonical_init_mode(args.init_mode),
                nx=int(args.nx),
                ny=int(args.ny),
                cycles=int(args.cycles),
                nshell=args.nshell,
                radius=float(args.radius),
                sample_index=int(sample_idx),
                seed=int(seed),
                output_path=str(samples_dir / f"sample_{sample_idx:04d}.npz"),
                cpu_affinity=tuple(cpu_pool),
                overwrite=bool(args.overwrite),
            )
        )

    if not args.overwrite and not args.resume:
        existing = [spec.output_path for spec in specs if Path(spec.output_path).exists()]
        if existing:
            raise FileExistsError(
                f"Found existing sample outputs. Pass --resume to reuse them or --overwrite to replace. First: {existing[0]}"
            )

    failures: list[dict[str, Any]] = []
    completed = 0
    ctx = get_context("spawn")
    executor: ProcessPoolExecutor | None = None
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        futures = {
            executor.submit(run_one_sample, asdict(spec)): spec
            for spec in specs
        }
        with tqdm(total=len(futures), desc=f"{config.name}/{args.init_mode}", unit="sample") as pbar:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    spec = futures[future]
                    try:
                        result = future.result()
                        completed += 1
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"sample={result['sample_index']} {result['status']} elapsed={result['elapsed_s']:.1f}s",
                            refresh=False,
                        )
                        print(
                            f"[sample] {config.name}/{args.init_mode} sample={result['sample_index']:04d} "
                            f"{result['status']} -> {result['output_path']}",
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
            write_json_atomic(run_dir / "failures.json", failures)
            raise RuntimeError(f"{len(failures)} samples failed; see {run_dir / 'failures.json'}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    print(f"[aggregate] Loading {args.samples} per-sample files from {samples_dir}", flush=True)
    aggregate_outputs(run_dir, config, args.samples, args.cycles)
    print(f"[done] Aggregates written under {run_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), required=False)
    parser.add_argument("--init-mode", choices=("random", "maxmix"), default="random")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--nshell", type=float, default=DEFAULT_NSHELL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--cpu-list", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--list-free-cpus", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_free_cpus:
        cpus = free_cpu_pool()
        print(format_cpu_list(cpus))
        return 0
    if args.campaign is None:
        parser.error("--campaign is required unless --list-free-cpus is used")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.nx <= 0 or args.ny <= 0:
        parser.error("--nx and --ny must be positive")
    if args.radius <= 0:
        parser.error("--radius must be positive")
    if args.cycles is not None and args.cycles <= 0:
        parser.error("--cycles must be positive")
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
