from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages


HELPER_VERSION = "streaming_covariance_protocol_characterization_gpu_v1"
PROTOCOL_ORDER = ("perfect_correction", "postselect")
TRACE_IMAG_TOL = 1e-8
HERM_TOL = 1e-9


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp_path.replace(path)


def save_dataframe_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def rel_to_root(path: Path | str | None, root: Path | str) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return None


def expected_samples_for_config(cfg: dict[str, Any]) -> int:
    postselect_probability = float(
        cfg.get("postselect_probability", 1.0 if bool(cfg.get("postselect", False)) else 0.0)
    )
    return 1 if bool(cfg.get("postselect", False)) or postselect_probability == 1.0 else int(cfg["samples"])


def case_output_dir(runs_root: Path, cfg: dict[str, Any]) -> Path:
    return Path(runs_root) / (
        f"N{int(cfg['Nx'])}x{int(cfg['Ny'])}_nsh{int(cfg['nshell'])}_"
        f"init-{cfg['init_mode']}_{cfg['protocol']}"
    )


def domain_wall_metadata(model: Any) -> dict[str, Any]:
    dw_loc = [int(x) for x in getattr(model, "DW_loc", [])]
    payload: dict[str, Any] = {"dw_loc": dw_loc}
    if len(dw_loc) == 2:
        payload["topological_x_range"] = [dw_loc[0], dw_loc[1]]
        payload["trivial_x_segments"] = [
            [0, max(-1, dw_loc[0] - 1)],
            [min(int(model.Nx), dw_loc[1] + 1), int(model.Nx) - 1],
        ]
    else:
        payload["topological_x_range"] = None
        payload["trivial_x_segments"] = None
    return payload


def cycle_group_label(cycle_label: int, early_cycle_max: int, cycles: int) -> str:
    cycle_label = int(cycle_label)
    early_cycle_max = int(early_cycle_max)
    cycles = int(cycles)
    if cycle_label < 1 or cycle_label > cycles:
        raise ValueError(f"cycle_label must satisfy 1 <= cycle_label <= {cycles}; got {cycle_label}")
    if cycle_label <= early_cycle_max:
        return f"cycles_1_{early_cycle_max}"
    return f"cycles_{early_cycle_max + 1}_{cycles}"


def geometry_key(nx: int, ny: int) -> str:
    return f"N{int(nx)}x{int(ny)}"


def config_id(nx: int, ny: int, protocol: str) -> str:
    return f"{geometry_key(nx, ny)}_{str(protocol)}"


def fit_line_with_error(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return {
            "slope": np.nan,
            "slope_err": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
        }
    try:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    except Exception:
        coeffs = np.polyfit(x, y, 1)
        slope_err = np.nan
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "r2": float(r2),
    }


def fit_entropy_log_chord_batch(
    entropy_curves: np.ndarray,
    *,
    ay_values: np.ndarray,
    ny: int,
    fit_ay_min: int,
) -> dict[str, np.ndarray]:
    entropy_curves = np.asarray(entropy_curves, dtype=np.float64)
    ay_values = np.asarray(ay_values, dtype=np.int64)
    fit_mask = (
        (ay_values >= int(fit_ay_min))
        & (ay_values <= int(ny) // 2)
        & (np.sin(np.pi * ay_values / float(ny)) > 0.0)
    )
    fit_x = np.full(ay_values.shape, np.nan, dtype=np.float64)
    fit_x[fit_mask] = np.log(np.sin(np.pi * ay_values[fit_mask] / float(ny)))
    batch_count = int(entropy_curves.shape[0])
    out = {
        "fit_point_count": np.zeros((batch_count,), dtype=np.int64),
        "slope": np.full((batch_count,), np.nan, dtype=np.float64),
        "intercept": np.full((batch_count,), np.nan, dtype=np.float64),
        "slope_err": np.full((batch_count,), np.nan, dtype=np.float64),
        "r2": np.full((batch_count,), np.nan, dtype=np.float64),
    }
    for idx in range(batch_count):
        y = entropy_curves[idx]
        valid = fit_mask & np.isfinite(y)
        out["fit_point_count"][idx] = int(np.count_nonzero(valid))
        if np.count_nonzero(valid) < 3:
            continue
        fit = fit_line_with_error(fit_x[valid], y[valid])
        out["slope"][idx] = float(fit["slope"])
        out["intercept"][idx] = float(fit["intercept"])
        out["slope_err"][idx] = float(fit["slope_err"])
        out["r2"][idx] = float(fit["r2"])
    return out


def strip_mode_indices(
    *,
    nx: int,
    ny: int,
    ay: int,
    y0_values: Iterable[int],
    device: torch.device | str,
) -> torch.Tensor:
    nx = int(nx)
    ny = int(ny)
    ay = int(ay)
    y0_values = torch.as_tensor(list(y0_values), dtype=torch.long, device=device)
    if ay == 0:
        return torch.empty((int(y0_values.numel()), 0), dtype=torch.long, device=device)
    dy = torch.arange(ay, dtype=torch.long, device=device)
    x = torch.arange(nx, dtype=torch.long, device=device)
    orbital = torch.arange(2, dtype=torch.long, device=device)
    y = (y0_values[:, None] + dy[None, :]) % ny
    idx = 2 * x[None, None, :, None] + 2 * nx * y[:, :, None, None] + orbital[None, None, None, :]
    return idx.reshape(int(y0_values.numel()), ay * nx * 2).contiguous()


def gather_restricted_covariance(G_batch: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if G_batch.ndim != 3:
        raise ValueError(f"Expected G_batch shape (S,N,N), got {tuple(G_batch.shape)}")
    if idx.ndim != 2:
        raise ValueError(f"Expected idx shape (Y0,M), got {tuple(idx.shape)}")
    sample_count, nrow, ncol = G_batch.shape
    if nrow != ncol:
        raise ValueError(f"Expected square covariance matrices, got {tuple(G_batch.shape)}")
    y0_count, mode_count = idx.shape
    if mode_count == 0:
        return torch.empty((sample_count * y0_count, 0, 0), dtype=G_batch.dtype, device=G_batch.device)
    idx_flat = idx.unsqueeze(0).expand(sample_count, -1, -1).reshape(sample_count * y0_count, mode_count)
    G_view = G_batch.unsqueeze(1).expand(-1, y0_count, -1, -1).reshape(sample_count * y0_count, nrow, ncol)
    batch_idx = torch.arange(sample_count * y0_count, dtype=torch.long, device=G_batch.device)
    return G_view[batch_idx[:, None, None], idx_flat[:, :, None], idx_flat[:, None, :]]


def _is_eigh_convergence_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "linalg.eigh" in msg and ("failed to converge" in msg or "ill-conditioned" in msg)


def _eigh_with_fallback(occ: torch.Tensor) -> tuple[torch.Tensor, int]:
    try:
        evals = torch.linalg.eigvalsh(occ)
        return evals, 0
    except Exception as exc:
        if not _is_eigh_convergence_error(exc):
            raise
    if occ.ndim == 3 and int(occ.shape[0]) > 1:
        mid = int(occ.shape[0]) // 2
        evals_l, fall_l = _eigh_with_fallback(occ[:mid])
        evals_r, fall_r = _eigh_with_fallback(occ[mid:])
        return torch.cat([evals_l, evals_r], dim=0), fall_l + fall_r
    occ_cpu = occ.detach().cpu()
    try:
        evals_cpu = torch.linalg.eigvalsh(occ_cpu)
    except Exception:
        n = int(occ_cpu.shape[-1])
        jitter = 100.0 * torch.finfo(occ_cpu.real.dtype).eps
        eye = torch.eye(n, dtype=occ_cpu.dtype, device=occ_cpu.device)
        evals_cpu = torch.linalg.eigvalsh(occ_cpu + jitter * eye)
    return evals_cpu.to(occ.device), 1


def entropy_total_batch_torch(
    sub_G: torch.Tensor,
    *,
    eps: float = 1e-12,
    validate: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if sub_G.ndim != 3:
        raise ValueError(f"Expected sub_G shape (B,M,M), got {tuple(sub_G.shape)}")
    batch_count, nrow, ncol = sub_G.shape
    if nrow != ncol:
        raise ValueError(f"Expected restricted covariance shape (B,M,M), got {tuple(sub_G.shape)}")
    if nrow == 0:
        return torch.zeros((batch_count,), dtype=torch.float64, device=sub_G.device), {
            "max_hermiticity_error": 0.0,
            "min_occupation_eval": 0.0,
            "max_occupation_eval": 0.0,
            "eigh_cpu_fallback_count": 0,
        }
    if validate and not torch.isfinite(sub_G).all():
        raise FloatingPointError("Non-finite restricted covariance entries.")
    herm_error = torch.amax(torch.abs(sub_G - sub_G.conj().transpose(-2, -1))).detach()
    eye = torch.eye(nrow, dtype=sub_G.dtype, device=sub_G.device)
    occ = 0.5 * (sub_G + eye.unsqueeze(0))
    occ = 0.5 * (occ + occ.conj().transpose(-2, -1))
    evals, fallback_count = _eigh_with_fallback(occ)
    evals_real = evals.real
    evals_clamped = torch.clamp(evals_real, float(eps), 1.0 - float(eps))
    weights = -(evals_clamped * torch.log(evals_clamped) + (1.0 - evals_clamped) * torch.log(1.0 - evals_clamped))
    totals = weights.sum(dim=-1).to(torch.float64)
    return totals, {
        "max_hermiticity_error": float(herm_error.detach().cpu()),
        "min_occupation_eval": float(torch.min(evals_real).detach().cpu()),
        "max_occupation_eval": float(torch.max(evals_real).detach().cpu()),
        "eigh_cpu_fallback_count": int(fallback_count),
    }


def _cuda_memory_info(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "device_name": "", "free_bytes": None, "total_bytes": None}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "free_gib": float(free_bytes / 1024**3),
        "total_gib": float(total_bytes / 1024**3),
    }


def _normalize_int_candidates(candidates: Iterable[Any], maximum: int) -> list[int]:
    values = []
    for candidate in candidates:
        if isinstance(candidate, str):
            if candidate.lower() != "ny":
                raise ValueError(f"Unsupported candidate string {candidate!r}; expected 'ny'.")
            value = int(maximum)
        else:
            value = int(candidate)
        if 1 <= value <= int(maximum):
            values.append(value)
    return sorted(set(values))


def estimate_eigh_bytes(*, sample_chunk: int, y0_chunk: int, nx: int, ay: int, multiplier: float = 8.0) -> int:
    matrix_size = 2 * int(nx) * int(ay)
    batch_count = int(sample_chunk) * int(y0_chunk)
    return int(batch_count * matrix_size * matrix_size * 16 * float(multiplier))


def autotune_entropy_chunks_from_batch(
    G_batch: torch.Tensor,
    *,
    nx: int,
    ny: int,
    ay: int,
    eps: float = 1e-12,
    sample_chunk_candidates: Iterable[int] = (1, 2, 4, 8, 16),
    y0_chunk_candidates: Iterable[Any] = (1, 2, 4, 6, 8, 12, 16, 24, 32, "ny"),
    repeat: int = 1,
    memory_safety_fraction: float = 0.85,
    eigh_memory_multiplier: float = 8.0,
) -> dict[str, Any]:
    device = G_batch.device
    nx = int(nx)
    ny = int(ny)
    ay = int(ay)
    sample_count = int(G_batch.shape[0])
    sample_candidates = _normalize_int_candidates(sample_chunk_candidates, sample_count)
    y0_candidates = _normalize_int_candidates(y0_chunk_candidates, ny)
    memory_before = _cuda_memory_info(device)
    if ay == 0:
        return {
            "selected_sample_chunk": max(sample_candidates) if sample_candidates else 1,
            "selected_y0_chunk": 0,
            "trials": [],
            "memory_before": memory_before,
            "memory_after": _cuda_memory_info(device),
            "ay": ay,
            "repeat": int(repeat),
        }
    total_bytes = memory_before.get("total_bytes") or 0
    free_bytes = memory_before.get("free_bytes") or total_bytes
    prefilter_limit = int(0.75 * free_bytes) if free_bytes else None
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    repeat = max(1, int(repeat))
    for sample_chunk in sample_candidates:
        for y0_chunk in y0_candidates:
            est_bytes = estimate_eigh_bytes(
                sample_chunk=sample_chunk,
                y0_chunk=y0_chunk,
                nx=nx,
                ay=ay,
                multiplier=eigh_memory_multiplier,
            )
            trial: dict[str, Any] = {
                "sample_chunk": int(sample_chunk),
                "y0_chunk": int(y0_chunk),
                "estimated_eigh_bytes": int(est_bytes),
                "status": "pending",
            }
            if prefilter_limit is not None and est_bytes > prefilter_limit:
                trial["status"] = "prefilter_skip"
                trials.append(trial)
                continue
            elapsed_ms = []
            peak_bytes = 0
            status = "ok"
            error = None
            for _ in range(repeat):
                idx = sub_G = totals = None
                try:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                    start_time = time.perf_counter()
                    if device.type == "cuda":
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    idx = strip_mode_indices(nx=nx, ny=ny, ay=ay, y0_values=range(y0_chunk), device=device)
                    sub_G = gather_restricted_covariance(G_batch[:sample_chunk], idx)
                    totals, _ = entropy_total_batch_torch(sub_G, eps=eps, validate=True)
                    if device.type == "cuda":
                        end_event.record()
                        torch.cuda.synchronize(device)
                        elapsed_ms.append(float(start_event.elapsed_time(end_event)))
                        peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated(device)))
                    else:
                        elapsed_ms.append(float((time.perf_counter() - start_time) * 1000.0))
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        status = "oom"
                        error = str(exc).splitlines()[0]
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        break
                    raise
                finally:
                    del idx, sub_G, totals
            trial["status"] = status
            trial["error"] = error
            if elapsed_ms:
                trial["mean_elapsed_ms"] = float(np.mean(elapsed_ms))
                trial["min_elapsed_ms"] = float(np.min(elapsed_ms))
            trial["peak_bytes"] = int(peak_bytes)
            trial["peak_gib"] = float(peak_bytes / 1024**3)
            trials.append(trial)
            if status == "ok":
                peak_ok = True
                if total_bytes:
                    peak_ok = peak_bytes < int(float(memory_safety_fraction) * total_bytes)
                if peak_ok:
                    score = float(trial.get("mean_elapsed_ms", np.inf))
                    if best is None or score < float(best.get("mean_elapsed_ms", np.inf)):
                        best = trial
    if best is None:
        raise RuntimeError(f"No safe entropy chunks found for ay={ay}. Trials: {trials}")
    return {
        "selected_sample_chunk": int(best["sample_chunk"]),
        "selected_y0_chunk": int(best["y0_chunk"]),
        "selected_trial": best,
        "trials": trials,
        "memory_before": memory_before,
        "memory_after": _cuda_memory_info(device),
        "ay": ay,
        "repeat": int(repeat),
        "memory_safety_fraction": float(memory_safety_fraction),
        "eigh_memory_multiplier": float(eigh_memory_multiplier),
    }


def build_chern_partition_indices(
    *,
    nx: int,
    ny: int,
    xref: int | None = None,
    yref: int | None = None,
    radius: float | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    nx = int(nx)
    ny = int(ny)
    xref = nx // 2 if xref is None else int(xref)
    yref = ny // 2 if yref is None else int(yref)
    radius = 0.4 * min(nx, ny) if radius is None else float(radius)
    if radius <= 0:
        raise ValueError(f"radius must be positive; got {radius}")
    inside = np.zeros((nx, ny), dtype=bool)
    a_mask = np.zeros_like(inside)
    b_mask = np.zeros_like(inside)
    c_mask = np.zeros_like(inside)
    rr = radius * radius
    ymax = int(math.floor(radius))
    a2 = 2.0 * np.pi / 3.0
    a4 = 4.0 * np.pi / 3.0
    for dy in range(-ymax, ymax + 1):
        y = yref + dy
        if y < 0 or y >= ny:
            continue
        max_dx = int(math.floor(math.sqrt(rr - dy * dy)))
        x0 = max(0, xref - max_dx)
        x1 = min(nx - 1, xref + max_dx)
        if x0 > x1:
            continue
        inside[x0 : x1 + 1, y] = True
        dxs = np.arange(x0, x1 + 1) - xref
        dys = np.full_like(dxs, dy)
        theta = np.mod(np.arctan2(dys, dxs), 2 * np.pi)
        a_mask[x0 : x1 + 1, y] = (theta >= 0.0) & (theta < a2)
        b_mask[x0 : x1 + 1, y] = (theta >= a2) & (theta < a4)
        c_mask[x0 : x1 + 1, y] = (theta >= a4) & (theta < 2 * np.pi)

    def idx_from_mask(mask: np.ndarray) -> np.ndarray:
        xs, ys = np.nonzero(mask)
        idx0 = 0 + 2 * xs + 2 * nx * ys
        idx1 = 1 + 2 * xs + 2 * nx * ys
        return np.sort(np.concatenate([idx0, idx1])).astype(np.int64, copy=False)

    i_a = idx_from_mask(a_mask)
    i_b = idx_from_mask(b_mask)
    i_c = idx_from_mask(c_mask)
    torch_device = torch.device(device)
    return {
        "nx": nx,
        "ny": ny,
        "xref": xref,
        "yref": yref,
        "radius": radius,
        "A": torch.as_tensor(i_a, dtype=torch.long, device=torch_device),
        "B": torch.as_tensor(i_b, dtype=torch.long, device=torch_device),
        "C": torch.as_tensor(i_c, dtype=torch.long, device=torch_device),
    }


def real_space_chern_batch_torch(
    G_batch: torch.Tensor,
    partitions: dict[str, Any],
) -> torch.Tensor:
    if G_batch.ndim != 3:
        raise ValueError(f"Expected G_batch shape (B,N,N), got {tuple(G_batch.shape)}")
    sample_count, nrow, ncol = G_batch.shape
    if nrow != ncol:
        raise ValueError(f"Expected square covariance batch, got {tuple(G_batch.shape)}")
    eye = torch.eye(nrow, dtype=G_batch.dtype, device=G_batch.device)
    P = torch.conj(0.5 * (G_batch + eye.unsqueeze(0)))
    i_a = partitions["A"].to(G_batch.device)
    i_b = partitions["B"].to(G_batch.device)
    i_c = partitions["C"].to(G_batch.device)

    def gather(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        out = torch.index_select(P, 1, rows)
        return torch.index_select(out, 2, cols)

    p_ca = gather(i_c, i_a)
    p_ab = gather(i_a, i_b)
    p_bc = gather(i_b, i_c)
    p_ac = gather(i_a, i_c)
    p_cb = gather(i_c, i_b)
    p_ba = gather(i_b, i_a)
    t1 = torch.diagonal(torch.bmm(torch.bmm(p_ca, p_ab), p_bc), dim1=-2, dim2=-1).sum(dim=-1)
    t2 = torch.diagonal(torch.bmm(torch.bmm(p_ac, p_cb), p_ba), dim1=-2, dim2=-1).sum(dim=-1)
    y = 12.0 * math.pi * 1j * (t1 - t2)
    return y.real.to(torch.float64)


class StreamingCovarianceCharacterizer:
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        samples_expected: int,
        protocol: str,
        nshell: int,
        fit_ay_min: int = 8,
        early_cycle_max: int = 20,
        snapshot_cycles: Iterable[int] = (20, 21, 30, 40, 50),
        trace_imag_tol: float = TRACE_IMAG_TOL,
        herm_tol: float = HERM_TOL,
        dw_loc: Iterable[int] | None = None,
        entropy_eps: float = 1e-12,
        autotune_entropy: bool = True,
        sample_chunk_candidates: Iterable[int] = (1, 2, 4, 8, 16),
        y0_chunk_candidates: Iterable[Any] = (1, 2, 4, 6, 8, 12, 16, 24, 32, "ny"),
        autotune_repeat: int = 1,
        memory_safety_fraction: float = 0.85,
        eigh_memory_multiplier: float = 8.0,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples_expected = int(samples_expected)
        self.protocol = str(protocol)
        self.nshell = int(nshell)
        self.fit_ay_min = int(fit_ay_min)
        self.early_cycle_max = int(early_cycle_max)
        self.snapshot_cycles = tuple(int(cycle) for cycle in snapshot_cycles)
        self.trace_imag_tol = float(trace_imag_tol)
        self.herm_tol = float(herm_tol)
        self.entropy_eps = float(entropy_eps)
        self.autotune_entropy = bool(autotune_entropy)
        self.sample_chunk_candidates = tuple(sample_chunk_candidates)
        self.y0_chunk_candidates = tuple(y0_chunk_candidates)
        self.autotune_repeat = int(autotune_repeat)
        self.memory_safety_fraction = float(memory_safety_fraction)
        self.eigh_memory_multiplier = float(eigh_memory_multiplier)
        self.nlayer = 2 * self.nx * self.ny
        self.ay_values = np.arange(self.ny // 2 + 1, dtype=np.int64)
        self.actual_samples: int | None = None

        shape = (self.samples_expected, self.cycles)
        curve_shape = (self.samples_expected, self.cycles, len(self.ay_values))
        self.entropy_curves = np.full(curve_shape, np.nan, dtype=np.float64)
        self.fit_point_count = np.zeros(shape, dtype=np.int64)
        self.entropy_slope = np.full(shape, np.nan, dtype=np.float64)
        self.entropy_intercept = np.full(shape, np.nan, dtype=np.float64)
        self.entropy_slope_err = np.full(shape, np.nan, dtype=np.float64)
        self.entropy_r2 = np.full(shape, np.nan, dtype=np.float64)
        self.real_space_chern = np.full(shape, np.nan, dtype=np.float64)
        self.normalized_trace = np.full(shape, np.nan, dtype=np.float64)
        self.trace_imag_abs = np.full(shape, np.nan, dtype=np.float64)
        self.hermitian_max_err = np.full(shape, np.nan, dtype=np.float64)
        self.frob_successive_delta = np.full(shape, np.nan, dtype=np.float64)

        self.observer_stats = {
            "entropy_autotune": {},
            "entropy_production_chunks": {},
            "restricted_eigh_cpu_fallback_count": 0,
            "restricted_occupation_eval_min": np.inf,
            "restricted_occupation_eval_max": -np.inf,
            "restricted_max_hermiticity_error": 0.0,
            "full_batch_trace_imag_max_abs": 0.0,
            "full_batch_hermitian_max_err": 0.0,
        }
        self._prev_G: torch.Tensor | None = None
        self._progress_bar = None
        self._chern_partitions_cache: dict[str, dict[str, Any]] = {}

        dw_loc_list = [int(x) for x in dw_loc] if dw_loc is not None else []
        if len(dw_loc_list) == 2:
            self.xref = int(math.floor((dw_loc_list[0] + dw_loc_list[1]) / 2))
        else:
            self.xref = self.nx // 2
        self.yref = self.ny // 2
        self.radius = 0.4 * min(self.nx, self.ny)

    def _chern_partitions_for_device(self, device: torch.device) -> dict[str, Any]:
        key = str(device)
        if key not in self._chern_partitions_cache:
            self._chern_partitions_cache[key] = build_chern_partition_indices(
                nx=self.nx,
                ny=self.ny,
                xref=self.xref,
                yref=self.yref,
                radius=self.radius,
                device=device,
            )
        return self._chern_partitions_cache[key]

    def _validate_top_layer_batch(self, G_batch: torch.Tensor) -> None:
        if G_batch.ndim != 3:
            raise ValueError(f"Expected G_batch with shape (B,N,N), got {tuple(G_batch.shape)}")
        batch, nlayer, nlayer_2 = G_batch.shape
        if nlayer != nlayer_2 or nlayer != self.nlayer:
            raise ValueError(f"Expected covariance shape (B,{self.nlayer},{self.nlayer}), got {tuple(G_batch.shape)}")
        if not torch.isfinite(G_batch).all():
            bad = torch.nonzero(~torch.isfinite(G_batch), as_tuple=False)[0].detach().cpu().tolist()
            raise FloatingPointError(f"Non-finite covariance batch entry encountered at index {bad}")

    def _fit_entropy_batch(self, entropy_batch: np.ndarray) -> dict[str, np.ndarray]:
        return fit_entropy_log_chord_batch(
            entropy_batch,
            ay_values=self.ay_values,
            ny=self.ny,
            fit_ay_min=self.fit_ay_min,
        )

    def _compute_entropy_curves_batch(self, G_batch: torch.Tensor) -> np.ndarray:
        batch_count = int(G_batch.shape[0])
        curves = np.zeros((batch_count, len(self.ay_values)), dtype=np.float64)
        for ay_idx, ay in enumerate(self.ay_values):
            ay_int = int(ay)
            if ay_int == 0:
                continue
            if ay_int not in self.observer_stats["entropy_autotune"]:
                if self.autotune_entropy:
                    autotune_info = autotune_entropy_chunks_from_batch(
                        G_batch,
                        nx=self.nx,
                        ny=self.ny,
                        ay=ay_int,
                        eps=self.entropy_eps,
                        sample_chunk_candidates=self.sample_chunk_candidates,
                        y0_chunk_candidates=self.y0_chunk_candidates,
                        repeat=self.autotune_repeat,
                        memory_safety_fraction=self.memory_safety_fraction,
                        eigh_memory_multiplier=self.eigh_memory_multiplier,
                    )
                else:
                    autotune_info = {
                        "selected_sample_chunk": 1,
                        "selected_y0_chunk": 1,
                        "trials": [],
                        "ay": ay_int,
                    }
                self.observer_stats["entropy_autotune"][str(ay_int)] = autotune_info

            sample_chunk = max(1, int(self.observer_stats["entropy_autotune"][str(ay_int)]["selected_sample_chunk"]))
            y0_chunk = max(1, int(self.observer_stats["entropy_autotune"][str(ay_int)]["selected_y0_chunk"]))
            final_sample_chunk = sample_chunk
            final_y0_chunk = y0_chunk
            oom_retries = 0
            entropy_sum = torch.zeros((batch_count,), dtype=torch.float64, device=G_batch.device)
            sample_start = 0
            while sample_start < batch_count:
                sample_stop = min(batch_count, sample_start + final_sample_chunk)
                y0_start = 0
                while y0_start < self.ny:
                    try:
                        G_sample = G_batch[sample_start:sample_stop]
                        y0_stop = min(self.ny, y0_start + final_y0_chunk)
                        idx = strip_mode_indices(
                            nx=self.nx,
                            ny=self.ny,
                            ay=ay_int,
                            y0_values=range(y0_start, y0_stop),
                            device=G_batch.device,
                        )
                        sub_G = gather_restricted_covariance(G_sample, idx)
                        totals, metrics = entropy_total_batch_torch(sub_G, eps=self.entropy_eps, validate=True)
                        totals = totals.reshape(sample_stop - sample_start, y0_stop - y0_start)
                        entropy_sum[sample_start:sample_stop] += totals.sum(dim=1)
                        self.observer_stats["restricted_eigh_cpu_fallback_count"] += int(
                            metrics["eigh_cpu_fallback_count"]
                        )
                        self.observer_stats["restricted_occupation_eval_min"] = min(
                            float(self.observer_stats["restricted_occupation_eval_min"]),
                            float(metrics["min_occupation_eval"]),
                        )
                        self.observer_stats["restricted_occupation_eval_max"] = max(
                            float(self.observer_stats["restricted_occupation_eval_max"]),
                            float(metrics["max_occupation_eval"]),
                        )
                        self.observer_stats["restricted_max_hermiticity_error"] = max(
                            float(self.observer_stats["restricted_max_hermiticity_error"]),
                            float(metrics["max_hermiticity_error"]),
                        )
                        y0_start = y0_stop
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        oom_retries += 1
                        if G_batch.device.type == "cuda":
                            torch.cuda.empty_cache()
                        if final_y0_chunk > 1:
                            final_y0_chunk = max(1, final_y0_chunk // 2)
                        elif final_sample_chunk > 1:
                            if y0_start != 0:
                                raise RuntimeError(
                                    "OOM after partially processing a sample chunk. "
                                    "Lower the selected sample_chunk before resuming."
                                ) from exc
                            final_sample_chunk = max(1, final_sample_chunk // 2)
                            sample_stop = min(batch_count, sample_start + final_sample_chunk)
                        else:
                            raise
                sample_start = sample_stop
            self.observer_stats["entropy_production_chunks"][str(ay_int)] = {
                "selected_sample_chunk": int(sample_chunk),
                "selected_y0_chunk": int(y0_chunk),
                "final_sample_chunk": int(final_sample_chunk),
                "final_y0_chunk": int(final_y0_chunk),
                "oom_retries": int(oom_retries),
            }
            curves[:, ay_idx] = (entropy_sum / float(self.ny)).detach().cpu().numpy().astype(np.float64, copy=False)
        return curves

    def observe(self, *, cycle: int, G: torch.Tensor, batch_index: int, batch_start: int, batch_count: int) -> None:
        del batch_index
        G_work = G.detach().clone()
        self._validate_top_layer_batch(G_work)
        cycle = int(cycle)
        batch_start = int(batch_start)
        batch_count = int(batch_count)
        sl = slice(batch_start, batch_start + batch_count)
        if cycle == 0:
            self._prev_G = G_work
            return
        if self._prev_G is None:
            raise RuntimeError("Observer state is missing the previous-cycle covariance batch.")

        cycle_idx = cycle - 1
        herm_vals = torch.amax(torch.abs(G_work - G_work.conj().transpose(-2, -1)), dim=(-2, -1)).to(torch.float64)
        herm_max = float(torch.max(herm_vals).detach().cpu())
        self.observer_stats["full_batch_hermitian_max_err"] = max(
            float(self.observer_stats["full_batch_hermitian_max_err"]),
            herm_max,
        )
        if herm_max > self.herm_tol:
            raise FloatingPointError(
                f"Full covariance Hermitian error exceeded tolerance at cycle={cycle}: "
                f"{herm_max:.6e} > {self.herm_tol:.6e}"
        )
        self.hermitian_max_err[sl, cycle_idx] = herm_vals.detach().cpu().numpy().astype(np.float64, copy=False)

        trace_vals = torch.diagonal(G_work, dim1=-2, dim2=-1).sum(dim=-1)
        trace_imag_abs = torch.abs(trace_vals.imag).to(torch.float64)
        trace_imag_max = float(torch.max(trace_imag_abs).detach().cpu())
        self.observer_stats["full_batch_trace_imag_max_abs"] = max(
            float(self.observer_stats["full_batch_trace_imag_max_abs"]),
            trace_imag_max,
        )
        if trace_imag_max > self.trace_imag_tol:
            raise FloatingPointError(
                f"Trace imaginary part exceeded tolerance at cycle={cycle}: "
                f"{trace_imag_max:.6e} > {self.trace_imag_tol:.6e}"
            )
        self.normalized_trace[sl, cycle_idx] = (
            (trace_vals.real / float(self.nlayer)).detach().cpu().numpy().astype(np.float64, copy=False)
        )
        self.trace_imag_abs[sl, cycle_idx] = trace_imag_abs.detach().cpu().numpy().astype(np.float64, copy=False)

        prev_g = self._prev_G
        delta = torch.linalg.vector_norm((G_work - prev_g).reshape(batch_count, -1), dim=-1).to(torch.float64)
        self.frob_successive_delta[sl, cycle_idx] = delta.detach().cpu().numpy().astype(np.float64, copy=False)
        self._prev_G = None
        del prev_g

        entropy_batch = self._compute_entropy_curves_batch(G_work)
        self.entropy_curves[sl, cycle_idx, :] = entropy_batch
        fit = self._fit_entropy_batch(entropy_batch)
        self.fit_point_count[sl, cycle_idx] = fit["fit_point_count"]
        self.entropy_slope[sl, cycle_idx] = fit["slope"]
        self.entropy_intercept[sl, cycle_idx] = fit["intercept"]
        self.entropy_slope_err[sl, cycle_idx] = fit["slope_err"]
        self.entropy_r2[sl, cycle_idx] = fit["r2"]

        partitions = self._chern_partitions_for_device(G_work.device)
        chern_vals = real_space_chern_batch_torch(G_work, partitions)
        self.real_space_chern[sl, cycle_idx] = chern_vals.detach().cpu().numpy().astype(np.float64, copy=False)

        self._prev_G = G_work
        if self._progress_bar is not None:
            self._progress_bar.update(batch_count)

    def make_cycle_observer(self, progress_bar=None):
        self._progress_bar = progress_bar

        def _observer(*, cycle, G, batch_index, batch_start, batch_count):
            self.observe(
                cycle=cycle,
                G=G,
                batch_index=batch_index,
                batch_start=batch_start,
                batch_count=batch_count,
            )

        return _observer

    def finalize(self, actual_samples: int) -> None:
        actual_samples = int(actual_samples)
        if actual_samples <= 0 or actual_samples > self.samples_expected:
            raise ValueError(
                f"actual_samples must satisfy 1 <= actual_samples <= {self.samples_expected}; got {actual_samples}"
            )
        self.actual_samples = actual_samples
        sample_slice = slice(0, actual_samples)
        required_arrays = {
            "entropy_curves": self.entropy_curves[sample_slice],
            "fit_point_count": self.fit_point_count[sample_slice],
            "entropy_slope": self.entropy_slope[sample_slice],
            "entropy_intercept": self.entropy_intercept[sample_slice],
            "entropy_slope_err": self.entropy_slope_err[sample_slice],
            "entropy_r2": self.entropy_r2[sample_slice],
            "real_space_chern": self.real_space_chern[sample_slice],
            "normalized_trace": self.normalized_trace[sample_slice],
            "trace_imag_abs": self.trace_imag_abs[sample_slice],
            "hermitian_max_err": self.hermitian_max_err[sample_slice],
            "frob_successive_delta": self.frob_successive_delta[sample_slice],
        }
        for name, value in required_arrays.items():
            if name == "fit_point_count":
                continue
            if np.isnan(np.asarray(value, dtype=np.float64)).any():
                raise RuntimeError(f"{name} contains unfilled NaN entries after finalize().")
        if np.any(self.trace_imag_abs[sample_slice] > self.trace_imag_tol):
            raise RuntimeError("trace_imag_abs exceeded tolerance in saved characterization data.")
        if np.any(self.hermitian_max_err[sample_slice] > self.herm_tol):
            raise RuntimeError("hermitian_max_err exceeded tolerance in saved characterization data.")

    def metrics_dataframe(self) -> pd.DataFrame:
        if self.actual_samples is None:
            raise RuntimeError("Call finalize(actual_samples=...) before exporting data.")
        rows = []
        for sample_index in range(self.actual_samples):
            for cycle_label in range(1, self.cycles + 1):
                cycle_idx = cycle_label - 1
                rows.append(
                    {
                        "config_id": config_id(self.nx, self.ny, self.protocol),
                        "geometry_key": geometry_key(self.nx, self.ny),
                        "protocol": self.protocol,
                        "Nx": self.nx,
                        "Ny": self.ny,
                        "nshell": self.nshell,
                        "sample_index": int(sample_index),
                        "cycle_label": int(cycle_label),
                        "cycle_group": cycle_group_label(cycle_label, self.early_cycle_max, self.cycles),
                        "fit_point_count": int(self.fit_point_count[sample_index, cycle_idx]),
                        "entropy_slope": float(self.entropy_slope[sample_index, cycle_idx]),
                        "entropy_intercept": float(self.entropy_intercept[sample_index, cycle_idx]),
                        "entropy_slope_err": float(self.entropy_slope_err[sample_index, cycle_idx]),
                        "entropy_r2": float(self.entropy_r2[sample_index, cycle_idx]),
                        "real_space_chern": float(self.real_space_chern[sample_index, cycle_idx]),
                        "normalized_trace": float(self.normalized_trace[sample_index, cycle_idx]),
                        "trace_imag_abs": float(self.trace_imag_abs[sample_index, cycle_idx]),
                        "hermitian_max_err": float(self.hermitian_max_err[sample_index, cycle_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def frob_dataframe(self) -> pd.DataFrame:
        if self.actual_samples is None:
            raise RuntimeError("Call finalize(actual_samples=...) before exporting data.")
        rows = []
        for sample_index in range(self.actual_samples):
            for cycle_label in range(1, self.cycles + 1):
                cycle_idx = cycle_label - 1
                rows.append(
                    {
                        "config_id": config_id(self.nx, self.ny, self.protocol),
                        "geometry_key": geometry_key(self.nx, self.ny),
                        "protocol": self.protocol,
                        "Nx": self.nx,
                        "Ny": self.ny,
                        "nshell": self.nshell,
                        "sample_index": int(sample_index),
                        "cycle_label": int(cycle_label),
                        "cycle_prev_label": int(cycle_label - 1),
                        "cycle_group": cycle_group_label(cycle_label, self.early_cycle_max, self.cycles),
                        "frob_successive_delta": float(self.frob_successive_delta[sample_index, cycle_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def entropy_curve_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        if self.actual_samples is None:
            raise RuntimeError("Call finalize(actual_samples=...) before exporting data.")
        return {
            "entropy_curves": self.entropy_curves[: self.actual_samples].copy(),
            "ay_values": self.ay_values.copy(),
            "cycle_labels": np.arange(1, self.cycles + 1, dtype=np.int64),
            "sample_indices": np.arange(self.actual_samples, dtype=np.int64),
            "config_json": np.asarray(json.dumps(config, sort_keys=True)),
            "helper_version": np.asarray(HELPER_VERSION),
        }

    def run_summary_metrics(self) -> dict[str, Any]:
        restricted_eval_min = self.observer_stats["restricted_occupation_eval_min"]
        restricted_eval_max = self.observer_stats["restricted_occupation_eval_max"]
        return {
            "observer_stats": {
                "entropy_autotune": self.observer_stats["entropy_autotune"],
                "entropy_production_chunks": self.observer_stats["entropy_production_chunks"],
                "restricted_eigh_cpu_fallback_count": int(self.observer_stats["restricted_eigh_cpu_fallback_count"]),
                "restricted_occupation_eval_min": (
                    None if not np.isfinite(restricted_eval_min) else float(restricted_eval_min)
                ),
                "restricted_occupation_eval_max": (
                    None if not np.isfinite(restricted_eval_max) else float(restricted_eval_max)
                ),
                "restricted_max_hermiticity_error": float(
                    self.observer_stats["restricted_max_hermiticity_error"]
                ),
                "full_batch_trace_imag_max_abs": float(self.observer_stats["full_batch_trace_imag_max_abs"]),
                "full_batch_hermitian_max_err": float(self.observer_stats["full_batch_hermitian_max_err"]),
            }
        }


def build_entropy_curve_payload(
    metrics_df: pd.DataFrame,
    run_curve_records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    curve_keys: list[str] = []
    offsets: list[int] = []
    counts: list[int] = []
    ay_flat: list[np.ndarray] = []
    entropy_flat: list[np.ndarray] = []
    offset = 0
    for record in run_curve_records:
        cfg_id = str(record["config_id"])
        entropy_curves = np.asarray(record["entropy_curves"], dtype=np.float64)
        ay_values = np.asarray(record["ay_values"], dtype=np.int64)
        actual_samples = int(record["actual_samples"])
        cycles = int(record["cycles"])
        for sample_index in range(actual_samples):
            for cycle_label in range(1, cycles + 1):
                curve = entropy_curves[sample_index, cycle_label - 1]
                curve_keys.append(f"{cfg_id}|sample={sample_index:03d}|cycle={cycle_label:02d}")
                offsets.append(offset)
                counts.append(int(curve.shape[0]))
                ay_flat.append(ay_values)
                entropy_flat.append(np.asarray(curve, dtype=np.float64))
                offset += int(curve.shape[0])
    return {
        "curve_keys": np.asarray(curve_keys, dtype=f"<U{max(len(k) for k in curve_keys)}"),
        "curve_offsets": np.asarray(offsets, dtype=np.int64),
        "curve_counts": np.asarray(counts, dtype=np.int64),
        "ay_values_flat": np.concatenate(ay_flat).astype(np.int64, copy=False),
        "entropy_values_flat": np.concatenate(entropy_flat).astype(np.float64, copy=False),
        "curve_table_columns": np.asarray(metrics_df.columns.tolist(), dtype="<U64"),
        "helper_version": np.asarray(HELPER_VERSION),
    }


def summarize_distribution(
    df: pd.DataFrame,
    value_col: str,
    metric_name: str,
    group_kind: str,
    *,
    cycle_label: int | None = None,
) -> pd.DataFrame:
    work = df[np.isfinite(df[value_col])].copy()
    if cycle_label is not None:
        work = work[work["cycle_label"] == int(cycle_label)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "group_kind",
                "config_id",
                "geometry_key",
                "protocol",
                "Nx",
                "Ny",
                "cycle_label",
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    grouped = (
        work.groupby(["config_id", "geometry_key", "protocol", "Nx", "Ny"], dropna=False)[value_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    grouped.insert(0, "metric", metric_name)
    grouped.insert(1, "group_kind", group_kind)
    grouped["cycle_label"] = np.nan if cycle_label is None else int(cycle_label)
    return grouped[
        [
            "metric",
            "group_kind",
            "config_id",
            "geometry_key",
            "protocol",
            "Nx",
            "Ny",
            "cycle_label",
            "count",
            "mean",
            "std",
            "min",
            "max",
        ]
    ]


def build_ensemble_summaries(
    metrics_df: pd.DataFrame,
    frob_df: pd.DataFrame,
    *,
    early_cycle_max: int,
    cycles: int,
    snapshot_cycles: Iterable[int],
) -> pd.DataFrame:
    snapshot_cycles = tuple(int(cycle) for cycle in snapshot_cycles)
    parts = []
    frob_grouped = (
        frob_df.groupby(["config_id", "geometry_key", "protocol", "Nx", "Ny", "cycle_label"], dropna=False)[
            "frob_successive_delta"
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    frob_grouped.insert(0, "metric", "frob_successive_delta")
    frob_grouped.insert(1, "group_kind", "cycle_snapshot")
    parts.append(
        frob_grouped[
            [
                "metric",
                "group_kind",
                "config_id",
                "geometry_key",
                "protocol",
                "Nx",
                "Ny",
                "cycle_label",
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
    )
    early_df = metrics_df[metrics_df["cycle_label"] <= int(early_cycle_max)].copy()
    late_df = metrics_df[metrics_df["cycle_label"] > int(early_cycle_max)].copy()
    early_label = f"cycles_1_{int(early_cycle_max)}"
    late_label = f"cycles_{int(early_cycle_max) + 1}_{int(cycles)}"
    for value_col, metric_name in (
        ("entropy_slope", "entropy_slope"),
        ("real_space_chern", "real_space_chern"),
        ("normalized_trace", "normalized_trace"),
    ):
        parts.append(summarize_distribution(early_df, value_col, metric_name, early_label))
        parts.append(summarize_distribution(late_df, value_col, metric_name, late_label))
        for cycle_label_value in snapshot_cycles:
            parts.append(
                summarize_distribution(
                    metrics_df,
                    value_col,
                    metric_name,
                    "cycle_snapshot",
                    cycle_label=cycle_label_value,
                )
            )
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["metric", "group_kind", "Nx", "Ny", "protocol", "cycle_label"], na_position="last")
        .reset_index(drop=True)
    )


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def compute_hist_bins(value_sets: list[np.ndarray]) -> np.ndarray:
    arrays = [finite_values(values) for values in value_sets if finite_values(values).size > 0]
    if not arrays:
        return np.linspace(-0.5, 0.5, 2)
    merged = np.concatenate(arrays)
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if np.isclose(vmin, vmax):
        delta = max(1e-6, abs(vmin) * 0.05, 0.05)
        return np.array([vmin - delta, vmax + delta], dtype=float)
    return np.linspace(vmin, vmax, 16)


def histogram_panel(ax, values: np.ndarray, bins: np.ndarray, title: str, xlabel: str) -> None:
    values = finite_values(values)
    if values.size == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        return
    ax.hist(values, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="black")
    mean_val = float(np.mean(values))
    ax.axvline(mean_val, color="#DD8452", lw=2)
    ax.set_title(f"{title}\nmean={mean_val:.6g}, n={values.size}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)


def create_overview_figure(
    geometry_name: str,
    metrics_df: pd.DataFrame,
    frob_df: pd.DataFrame,
    *,
    protocol_order: Iterable[str] = PROTOCOL_ORDER,
    early_cycle_max: int,
    cycles: int,
) -> plt.Figure:
    late_label = f"cycles {int(early_cycle_max) + 1}-{int(cycles)}"
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    late_df = metrics_df[
        (metrics_df["geometry_key"] == geometry_name) & (metrics_df["cycle_label"] > int(early_cycle_max))
    ].copy()
    frob_sub = frob_df[frob_df["geometry_key"] == geometry_name].copy()
    slope_bins = compute_hist_bins(
        [late_df[late_df["protocol"] == protocol]["entropy_slope"].to_numpy() for protocol in protocol_order]
    )
    chern_bins = compute_hist_bins(
        [late_df[late_df["protocol"] == protocol]["real_space_chern"].to_numpy() for protocol in protocol_order]
    )
    trace_bins = compute_hist_bins(
        [late_df[late_df["protocol"] == protocol]["normalized_trace"].to_numpy() for protocol in protocol_order]
    )
    for col, protocol in enumerate(protocol_order):
        ax = axes[0, col]
        line_df = frob_sub[frob_sub["protocol"] == protocol].copy()
        stats = line_df.groupby("cycle_label", dropna=False)["frob_successive_delta"].agg(["mean", "std"]).reindex(
            range(1, int(cycles) + 1)
        )
        x = np.asarray(stats.index, dtype=int)
        mean = stats["mean"].to_numpy(dtype=float)
        std = np.nan_to_num(stats["std"].to_numpy(dtype=float), nan=0.0)
        ax.plot(x, mean, marker="o", markersize=2.2, lw=1.2, color="#4C72B0")
        ax.fill_between(x, mean - std, mean + std, color="#4C72B0", alpha=0.2)
        ax.set_title(f"{geometry_name} {protocol}\nFrobenius successive delta")
        ax.set_xlabel("cycle label c for ||G_c - G_(c-1)||_F")
        ax.set_ylabel("mean ± std")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

        proto_df = late_df[late_df["protocol"] == protocol].copy()
        histogram_panel(
            axes[1, col],
            proto_df["entropy_slope"].to_numpy(),
            slope_bins,
            f"{geometry_name} {protocol}\nlate slope histogram ({late_label})",
            "entropy slope",
        )
        histogram_panel(
            axes[2, col],
            proto_df["real_space_chern"].to_numpy(),
            chern_bins,
            f"{geometry_name} {protocol}\nlate Chern histogram ({late_label})",
            "real-space Chern number",
        )
        histogram_panel(
            axes[3, col],
            proto_df["normalized_trace"].to_numpy(),
            trace_bins,
            f"{geometry_name} {protocol}\nlate normalized trace histogram ({late_label})",
            "tr(G) / Nlayer",
        )
    fig.suptitle(f"Streaming covariance characterization overview: {geometry_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def create_snapshot_histogram_figure(
    metrics_df: pd.DataFrame,
    *,
    geometry_name: str,
    protocol: str,
    metric_col: str,
    metric_label: str,
    snapshot_cycles: Iterable[int],
) -> plt.Figure:
    snapshot_cycles = tuple(int(cycle) for cycle in snapshot_cycles)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    page_df = metrics_df[
        (metrics_df["geometry_key"] == geometry_name)
        & (metrics_df["protocol"] == protocol)
        & (metrics_df["cycle_label"].isin(snapshot_cycles))
    ].copy()
    bins = compute_hist_bins(
        [page_df[page_df["cycle_label"] == cycle_label][metric_col].to_numpy() for cycle_label in snapshot_cycles]
    )
    for ax, cycle_label in zip(axes, snapshot_cycles):
        cycle_df = page_df[page_df["cycle_label"] == cycle_label]
        histogram_panel(
            ax,
            cycle_df[metric_col].to_numpy(),
            bins,
            f"{geometry_name} {protocol}\ncycle {cycle_label}",
            metric_label,
        )
    for ax in axes[len(snapshot_cycles) :]:
        ax.axis("off")
    fig.suptitle(f"{geometry_name} {protocol}: {metric_label} snapshot distributions", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def create_late_ensemble_compare_figure(
    metrics_df: pd.DataFrame,
    *,
    protocol: str,
    metric_col: str,
    metric_label: str,
    geometry_order: Iterable[tuple[int, int]],
    early_cycle_max: int,
    cycles: int,
) -> plt.Figure:
    late_label = f"cycles {int(early_cycle_max) + 1}-{int(cycles)}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    geometry_order = tuple((int(nx), int(ny)) for nx, ny in geometry_order)
    page_df = metrics_df[
        (metrics_df["protocol"] == protocol)
        & (metrics_df["cycle_label"] > int(early_cycle_max))
        & (metrics_df["geometry_key"].isin([geometry_key(nx, ny) for nx, ny in geometry_order]))
    ].copy()
    bins = compute_hist_bins(
        [page_df[page_df["geometry_key"] == geometry_key(nx, ny)][metric_col].to_numpy() for nx, ny in geometry_order]
    )
    for ax, (nx, ny) in zip(axes, geometry_order):
        key = geometry_key(nx, ny)
        sub = page_df[page_df["geometry_key"] == key]
        histogram_panel(
            ax,
            sub[metric_col].to_numpy(),
            bins,
            f"{key} {protocol}\nlate ensemble ({late_label})",
            metric_label,
        )
    fig.suptitle(f"{protocol}: late ensemble {metric_label} comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def write_pdf_report(
    pdf_path: Path,
    metrics_df: pd.DataFrame,
    frob_df: pd.DataFrame,
    *,
    geometry_order: Iterable[tuple[int, int]],
    protocol_order: Iterable[str] = PROTOCOL_ORDER,
    early_cycle_max: int,
    cycles: int,
    snapshot_cycles: Iterable[int],
    progress_bar=None,
) -> None:
    geometry_order = tuple((int(nx), int(ny)) for nx, ny in geometry_order)
    protocol_order = tuple(str(protocol) for protocol in protocol_order)
    snapshot_cycles = tuple(int(cycle) for cycle in snapshot_cycles)
    page_jobs: list[tuple[str, dict[str, Any]]] = []
    for nx, ny in geometry_order:
        page_jobs.append(("overview", {"geometry_name": geometry_key(nx, ny)}))
    for metric_col, metric_label in (
        ("entropy_slope", "entropy slope"),
        ("real_space_chern", "real-space Chern number"),
        ("normalized_trace", "tr(G) / Nlayer"),
    ):
        for protocol in protocol_order:
            page_jobs.append(
                (
                    "snapshot",
                    {
                        "geometry_name": geometry_key(geometry_order[0][0], geometry_order[0][1]),
                        "protocol": protocol,
                        "metric_col": metric_col,
                        "metric_label": metric_label,
                    },
                )
            )
    for metric_col, metric_label in (
        ("entropy_slope", "entropy slope"),
        ("real_space_chern", "real-space Chern number"),
        ("normalized_trace", "tr(G) / Nlayer"),
    ):
        for protocol in protocol_order:
            page_jobs.append(
                (
                    "late",
                    {
                        "protocol": protocol,
                        "metric_col": metric_col,
                        "metric_label": metric_label,
                    },
                )
            )
    if progress_bar is not None:
        progress_bar.reset(total=len(page_jobs))
    with PdfPages(pdf_path) as pdf:
        for kind, kwargs in page_jobs:
            if kind == "overview":
                fig = create_overview_figure(
                    kwargs["geometry_name"],
                    metrics_df,
                    frob_df,
                    protocol_order=protocol_order,
                    early_cycle_max=early_cycle_max,
                    cycles=cycles,
                )
            elif kind == "snapshot":
                fig = create_snapshot_histogram_figure(
                    metrics_df,
                    geometry_name=kwargs["geometry_name"],
                    protocol=kwargs["protocol"],
                    metric_col=kwargs["metric_col"],
                    metric_label=kwargs["metric_label"],
                    snapshot_cycles=snapshot_cycles,
                )
            else:
                fig = create_late_ensemble_compare_figure(
                    metrics_df,
                    protocol=kwargs["protocol"],
                    metric_col=kwargs["metric_col"],
                    metric_label=kwargs["metric_label"],
                    geometry_order=geometry_order,
                    early_cycle_max=early_cycle_max,
                    cycles=cycles,
                )
            pdf.savefig(fig)
            plt.close(fig)
            if progress_bar is not None:
                progress_bar.update(1)


def run_internal_checks() -> None:
    idx_a = strip_mode_indices(nx=4, ny=7, ay=3, y0_values=[2], device="cpu").detach().cpu().numpy()[0]
    idx_b = strip_mode_indices(nx=4, ny=7, ay=3, y0_values=[9], device="cpu").detach().cpu().numpy()[0]
    if not np.array_equal(idx_a, idx_b):
        raise AssertionError("Wrapped strip-mode indexing failed modulo equivalence check")

    empty = torch.empty((3, 0, 0), dtype=torch.complex128)
    totals, _ = entropy_total_batch_torch(empty)
    if not np.allclose(totals.detach().cpu().numpy(), 0.0):
        raise AssertionError("Ay=0 entropy totals must be exactly zero")

    fit = fit_entropy_log_chord_batch(
        np.asarray([[1.0, 1.1, 1.2]], dtype=np.float64),
        ay_values=np.asarray([8, 9, 10], dtype=np.int64),
        ny=20,
        fit_ay_min=8,
    )
    if not np.isfinite(fit["slope"][0]):
        raise AssertionError("Entropy fit should return a finite slope with >=3 fit points")

    g = torch.zeros((2, 2 * 4 * 6, 2 * 4 * 6), dtype=torch.complex128)
    partitions = build_chern_partition_indices(nx=4, ny=6, device="cpu")
    cherns = real_space_chern_batch_torch(g, partitions).detach().cpu().numpy()
    if not np.allclose(cherns, 0.0):
        raise AssertionError("Zero covariance should yield zero real-space Chern in this partition test")
