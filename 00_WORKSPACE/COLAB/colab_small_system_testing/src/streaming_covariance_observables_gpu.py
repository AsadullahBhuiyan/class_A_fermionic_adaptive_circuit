from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


HELPER_VERSION = "streaming_covariance_observables_gpu_v3"
PROTOCOL_ORDER = ("perfect_correction", "postselect")
TRACE_IMAG_TOL = 1e-8
HERM_TOL = 1e-9
SUCCESS_PROB_TOL = 1e-8


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


def geometry_key(nx: int, ny: int) -> str:
    return f"N{int(nx)}x{int(ny)}"


def config_id(nx: int, ny: int, protocol: str) -> str:
    return f"{geometry_key(nx, ny)}_{str(protocol)}"


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
    g_view = G_batch.unsqueeze(1).expand(-1, y0_count, -1, -1).reshape(sample_count * y0_count, nrow, ncol)
    batch_idx = torch.arange(sample_count * y0_count, dtype=torch.long, device=G_batch.device)
    return g_view[batch_idx[:, None, None], idx_flat[:, :, None], idx_flat[:, None, :]]


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
    sub_g: torch.Tensor,
    *,
    eps: float = 1e-12,
    validate: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if sub_g.ndim != 3:
        raise ValueError(f"Expected sub_G shape (B,M,M), got {tuple(sub_g.shape)}")
    batch_count, nrow, ncol = sub_g.shape
    if nrow != ncol:
        raise ValueError(f"Expected restricted covariance shape (B,M,M), got {tuple(sub_g.shape)}")
    if nrow == 0:
        return torch.zeros((batch_count,), dtype=torch.float64, device=sub_g.device), {
            "max_hermiticity_error": 0.0,
            "min_occupation_eval": 0.0,
            "max_occupation_eval": 0.0,
            "eigh_cpu_fallback_count": 0,
        }
    if validate and not torch.isfinite(sub_g).all():
        raise FloatingPointError("Non-finite restricted covariance entries.")
    herm_error = torch.amax(torch.abs(sub_g - sub_g.conj().transpose(-2, -1))).detach()
    eye = torch.eye(nrow, dtype=sub_g.dtype, device=sub_g.device)
    occ = 0.5 * (sub_g + eye.unsqueeze(0))
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
    g_batch: torch.Tensor,
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
    device = g_batch.device
    nx = int(nx)
    ny = int(ny)
    ay = int(ay)
    sample_count = int(g_batch.shape[0])
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
                idx = sub_g = totals = None
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
                    sub_g = gather_restricted_covariance(g_batch[:sample_chunk], idx)
                    totals, _ = entropy_total_batch_torch(sub_g, eps=eps, validate=True)
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
                    del idx, sub_g, totals
            trial["status"] = status
            trial["error"] = error
            if elapsed_ms:
                trial["mean_elapsed_ms"] = float(np.mean(elapsed_ms))
                trial["min_elapsed_ms"] = float(np.min(elapsed_ms))
            trial["peak_bytes"] = int(peak_bytes)
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

    torch_device = torch.device(device)
    return {
        "nx": nx,
        "ny": ny,
        "xref": xref,
        "yref": yref,
        "radius": radius,
        "inside_mask": inside.astype(bool, copy=True),
        "A": torch.as_tensor(idx_from_mask(a_mask), dtype=torch.long, device=torch_device),
        "B": torch.as_tensor(idx_from_mask(b_mask), dtype=torch.long, device=torch_device),
        "C": torch.as_tensor(idx_from_mask(c_mask), dtype=torch.long, device=torch_device),
    }


def real_space_chern_batch_torch(g_batch: torch.Tensor, partitions: dict[str, Any]) -> torch.Tensor:
    if g_batch.ndim != 3:
        raise ValueError(f"Expected G_batch shape (B,N,N), got {tuple(g_batch.shape)}")
    _, nrow, ncol = g_batch.shape
    if nrow != ncol:
        raise ValueError(f"Expected square covariance batch, got {tuple(g_batch.shape)}")
    eye = torch.eye(nrow, dtype=g_batch.dtype, device=g_batch.device)
    p = torch.conj(0.5 * (g_batch + eye.unsqueeze(0)))
    i_a = partitions["A"].to(g_batch.device)
    i_b = partitions["B"].to(g_batch.device)
    i_c = partitions["C"].to(g_batch.device)

    def gather(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        out = torch.index_select(p, 1, rows)
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


def build_square_correlator_pair_indices(
    *,
    nx: int,
    ny: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    torch_device = torch.device(device)
    ry = torch.arange(ny // 2 + 1, dtype=torch.long, device=torch_device)
    x = torch.arange(nx, dtype=torch.long, device=torch_device)
    y = torch.arange(ny, dtype=torch.long, device=torch_device)
    mu = torch.arange(2, dtype=torch.long, device=torch_device)
    nu = torch.arange(2, dtype=torch.long, device=torch_device)
    left = (
        mu.view(1, 1, 2, 1)
        + 2 * x.view(nx, 1, 1, 1)
        + 2 * nx * y.view(1, ny, 1, 1)
    ).expand(nx, ny, 2, 2)
    right_y = (y[None, :] + ry[:, None]) % ny
    right = (
        nu.view(1, 1, 1, 1, 2)
        + 2 * x.view(1, nx, 1, 1, 1)
        + 2 * nx * right_y.view(int(ry.numel()), 1, ny, 1, 1)
    )
    right = right.expand(int(ry.numel()), nx, ny, 2, 2)
    left = left.unsqueeze(0).expand(int(ry.numel()), nx, ny, 2, 2)
    return {
        "ry_values": ry,
        "left_indices": left.reshape(int(ry.numel()), -1).contiguous(),
        "right_indices": right.reshape(int(ry.numel()), -1).contiguous(),
    }


def xavg_square_correlator_batch_torch(
    g_batch: torch.Tensor,
    pair_indices: dict[str, torch.Tensor],
    *,
    nx: int,
    ny: int,
) -> torch.Tensor:
    if g_batch.ndim != 3:
        raise ValueError(f"Expected G_batch shape (B,N,N), got {tuple(g_batch.shape)}")
    batch_count, nrow, ncol = g_batch.shape
    if nrow != ncol:
        raise ValueError(f"Expected square covariance batch, got {tuple(g_batch.shape)}")
    left = pair_indices["left_indices"].to(g_batch.device)
    right = pair_indices["right_indices"].to(g_batch.device)
    ry_count, pair_count = left.shape
    eye = torch.eye(nrow, dtype=g_batch.dtype, device=g_batch.device)
    occ = 0.5 * (g_batch + eye.unsqueeze(0))
    values = occ[:, left.reshape(-1), right.reshape(-1)]
    values = values.reshape(batch_count, ry_count, pair_count)
    norm = float(2 * int(nx) * int(ny))
    return (torch.abs(values) ** 2).sum(dim=-1).to(torch.float64) / norm


def local_charge_cell_batch_torch(g_batch: torch.Tensor, *, nx: int, ny: int) -> torch.Tensor:
    diag = torch.diagonal(g_batch, dim1=-2, dim2=-1)
    occ = 0.5 * (diag.real.to(torch.float64) + 1.0)
    occ = occ.reshape(int(g_batch.shape[0]), int(ny), int(nx), 2)
    return occ.sum(dim=-1).transpose(1, 2).contiguous()


def local_chern_marker_single_torch(g_top: torch.Tensor, *, nx: int, ny: int) -> torch.Tensor:
    nlayer = 2 * int(nx) * int(ny)
    eye = torch.eye(nlayer, dtype=g_top.dtype, device=g_top.device)
    p = torch.conj(0.5 * (g_top + eye))
    x_grid = torch.arange(1, int(nx) + 1, dtype=p.real.dtype, device=g_top.device)
    y_grid = torch.arange(1, int(ny) + 1, dtype=p.real.dtype, device=g_top.device)
    x_vec = x_grid.repeat(int(ny)).repeat_interleave(2).to(p.dtype)
    y_vec = y_grid.repeat_interleave(int(nx)).repeat_interleave(2).to(p.dtype)
    pxp = (p * x_vec.unsqueeze(0)) @ p
    pyp = (p * y_vec.unsqueeze(0)) @ p
    t_diag = torch.sum((pxp * y_vec.unsqueeze(0)) * p.transpose(0, 1), dim=1)
    u_diag = torch.sum((pyp * x_vec.unsqueeze(0)) * p.transpose(0, 1), dim=1)
    marker_orb = ((2.0 * math.pi * 1j) * (t_diag - u_diag)).real.to(torch.float64)
    return marker_orb.reshape(int(ny), int(nx), 2).sum(dim=-1).transpose(0, 1).contiguous()


def local_chern_marker_batch_torch(g_batch: torch.Tensor, *, nx: int, ny: int) -> torch.Tensor:
    batch_count = int(g_batch.shape[0])
    if batch_count == 0:
        return torch.empty((0, int(nx), int(ny)), dtype=torch.float64, device=g_batch.device)
    nlayer = int(g_batch.shape[-1])
    eye = torch.eye(nlayer, dtype=g_batch.dtype, device=g_batch.device)
    p = torch.conj(0.5 * (g_batch + eye.unsqueeze(0)))
    x_grid = torch.arange(1, int(nx) + 1, dtype=p.real.dtype, device=g_batch.device)
    y_grid = torch.arange(1, int(ny) + 1, dtype=p.real.dtype, device=g_batch.device)
    x_vec = x_grid.repeat(int(ny)).repeat_interleave(2).to(p.dtype)
    y_vec = y_grid.repeat_interleave(int(nx)).repeat_interleave(2).to(p.dtype)
    pxp = torch.bmm(p * x_vec.view(1, 1, -1), p)
    pyp = torch.bmm(p * y_vec.view(1, 1, -1), p)
    t_diag = torch.sum((pxp * y_vec.view(1, 1, -1)) * p.transpose(-2, -1), dim=-1)
    u_diag = torch.sum((pyp * x_vec.view(1, 1, -1)) * p.transpose(-2, -1), dim=-1)
    marker_orb = ((2.0 * math.pi * 1j) * (t_diag - u_diag)).real.to(torch.float64)
    return marker_orb.reshape(batch_count, int(ny), int(nx), 2).sum(dim=-1).transpose(1, 2).contiguous()


def _square_corr_reference_numpy(g_top: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    nlayer = 2 * int(nx) * int(ny)
    c = 0.5 * (np.asarray(g_top, dtype=np.complex128) + np.eye(nlayer, dtype=np.complex128))
    ry_values = np.arange(int(ny) // 2 + 1, dtype=np.int64)
    out = np.empty((ry_values.size,), dtype=np.float64)
    for r_idx, ry in enumerate(ry_values):
        total = 0.0
        for x0 in range(int(nx)):
            for y0 in range(int(ny)):
                yp = (y0 + int(ry)) % int(ny)
                for mu in range(2):
                    i = mu + 2 * x0 + 2 * int(nx) * y0
                    for nu in range(2):
                        j = nu + 2 * x0 + 2 * int(nx) * yp
                        total += abs(c[i, j]) ** 2
        out[r_idx] = total / float(2 * int(nx) * int(ny))
    return out


def run_internal_checks() -> None:
    idx_a = strip_mode_indices(nx=4, ny=7, ay=3, y0_values=[2], device="cpu").detach().cpu().numpy()[0]
    idx_b = strip_mode_indices(nx=4, ny=7, ay=3, y0_values=[9], device="cpu").detach().cpu().numpy()[0]
    if not np.array_equal(idx_a, idx_b):
        raise AssertionError("Wrapped strip-mode indexing failed modulo equivalence check")

    empty = torch.empty((3, 0, 0), dtype=torch.complex128)
    totals, _ = entropy_total_batch_torch(empty)
    if not np.allclose(totals.detach().cpu().numpy(), 0.0):
        raise AssertionError("Ay=0 entropy totals must be exactly zero")

    zero = torch.zeros((2, 2 * 4 * 6, 2 * 4 * 6), dtype=torch.complex128)
    partitions = build_chern_partition_indices(nx=4, ny=6, device="cpu")
    cherns = real_space_chern_batch_torch(zero, partitions).detach().cpu().numpy()
    if not np.allclose(cherns, 0.0):
        raise AssertionError("Zero covariance should yield zero real-space Chern in this partition test")

    pair_indices = build_square_correlator_pair_indices(nx=3, ny=4, device="cpu")
    g_test = torch.zeros((1, 2 * 3 * 4, 2 * 3 * 4), dtype=torch.complex128)
    diag_vals = torch.tensor([1.0, -1.0] * (3 * 4), dtype=torch.float64)
    g_test[0] = torch.diag(diag_vals.to(torch.complex128))
    corr_ref = _square_corr_reference_numpy(g_test[0].detach().cpu().numpy(), nx=3, ny=4)
    corr_fast = xavg_square_correlator_batch_torch(g_test, pair_indices, nx=3, ny=4)[0].detach().cpu().numpy()
    if not np.allclose(corr_ref, corr_fast, atol=1e-12, rtol=1e-12):
        raise AssertionError("Square-correlator batch implementation failed the CPU reference check")

    marker = local_chern_marker_single_torch(g_test[0], nx=3, ny=4).detach().cpu().numpy()
    marker_batch = local_chern_marker_batch_torch(g_test, nx=3, ny=4)[0].detach().cpu().numpy()
    if marker.shape != (3, 4):
        raise AssertionError(f"Unexpected local Chern marker shape {marker.shape}")
    if not np.allclose(marker, marker_batch, atol=1e-12, rtol=1e-12):
        raise AssertionError("Batched local Chern marker implementation disagrees with the single-sample path")
    try:
        from fgtn.classA_U1FGTN import classA_U1FGTN

        cpu_model = classA_U1FGTN(Nx=3, Ny=4, DW=False, nshell=1)
        rng = np.random.default_rng(0)
        random_real = rng.standard_normal((2 * 3 * 4, 2 * 3 * 4))
        random_imag = rng.standard_normal((2 * 3 * 4, 2 * 3 * 4))
        g_ref = random_real + 1j * random_imag
        g_ref = 0.5 * (g_ref + g_ref.conj().T)
        g_ref_torch = torch.as_tensor(g_ref, dtype=torch.complex128)
        marker_ref = cpu_model.local_chern_marker_flat(
            g_ref,
            apply_tanh=False,
        )
        marker_fast = local_chern_marker_single_torch(g_ref_torch, nx=3, ny=4).detach().cpu().numpy()
        if not np.allclose(marker_fast, marker_ref, atol=1e-8, rtol=1e-8):
            raise AssertionError("Local Chern marker implementation failed the CPU reference check")
        chern_ref = cpu_model.real_space_chern_number(
            g_ref,
            xref=cpu_model.Nx // 2,
            yref=cpu_model.Ny // 2,
            radius=0.4 * min(cpu_model.Nx, cpu_model.Ny),
        )
        partitions_ref = build_chern_partition_indices(nx=3, ny=4, device="cpu")
        chern_fast = real_space_chern_batch_torch(g_ref_torch.unsqueeze(0), partitions_ref)[0].detach().cpu().item()
        if not np.allclose(chern_fast, chern_ref, atol=1e-8, rtol=1e-8):
            raise AssertionError("Real-space Chern implementation failed the CPU reference check")
    except ImportError:
        pass

    charges = local_charge_cell_batch_torch(g_test, nx=3, ny=4).detach().cpu().numpy()
    if charges.shape != (1, 3, 4):
        raise AssertionError(f"Unexpected local charge shape {charges.shape}")
    if not np.allclose(charges.sum(axis=(1, 2)), 12.0):
        raise AssertionError("Local charge cell sum must match the total occupancy")

    observer = StreamingCovarianceObservables(
        nx=4,
        ny=4,
        cycles=2,
        samples_expected=2,
        protocol="perfect_correction",
        nshell=1,
    )
    site_observer = observer.make_site_observer()
    site_observer(
        cycle=1,
        site_ids=np.asarray([0, 5], dtype=np.int64),
        sample_indices=np.asarray([0, 1], dtype=np.int64),
        batch_index=0,
        batch_start=0,
        batch_count=2,
        s_Ap=np.asarray([1.0, 0.8], dtype=np.float64),
        s_Am=np.asarray([0.9, 0.7], dtype=np.float64),
        s_Bp=np.asarray([0.8, 0.6], dtype=np.float64),
        s_Bm=np.asarray([0.7, 0.5], dtype=np.float64),
    )
    if not np.isclose(observer.frustration_min[0, 0, 0, 0], 0.3):
        raise AssertionError("frustration_min did not match the min-success definition")
    if not np.isclose(observer.frustration_avg[1, 0, 1, 1], 0.35):
        raise AssertionError("frustration_avg did not match the mean-success definition")
    if not np.isclose(observer.frustration_prod[0, 0, 0, 0], 1.0 - 1.0 * 0.9 * 0.8 * 0.7):
        raise AssertionError("frustration_prod did not match the product-success definition")

    postselect_observer = StreamingCovarianceObservables(
        nx=4,
        ny=4,
        cycles=1,
        samples_expected=1,
        protocol="postselect",
        nshell=1,
    )
    if postselect_observer.measurement_frustration_enabled:
        raise AssertionError("postselect should not enable measurement-frustration recording")
    try:
        postselect_observer.make_site_observer()
    except RuntimeError:
        pass
    else:
        raise AssertionError("postselect should not expose a measurement-frustration site observer")


class StreamingCovarianceObservables:
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        samples_expected: int,
        protocol: str,
        nshell: int,
        dw_loc: Iterable[int] | None = None,
        entropy_eps: float = 1e-12,
        autotune_entropy: bool = True,
        sample_chunk_candidates: Iterable[int] = (1, 2, 4, 8, 16),
        y0_chunk_candidates: Iterable[Any] = (1, 2, 4, 6, 8, 12, 16, 24, 32, "ny"),
        autotune_repeat: int = 1,
        memory_safety_fraction: float = 0.85,
        eigh_memory_multiplier: float = 8.0,
        trace_imag_tol: float = TRACE_IMAG_TOL,
        herm_tol: float = HERM_TOL,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples_expected = int(samples_expected)
        self.protocol = str(protocol)
        self.nshell = int(nshell)
        self.measurement_frustration_enabled = self.protocol == "perfect_correction"
        self.entropy_eps = float(entropy_eps)
        self.autotune_entropy = bool(autotune_entropy)
        self.sample_chunk_candidates = tuple(sample_chunk_candidates)
        self.y0_chunk_candidates = tuple(y0_chunk_candidates)
        self.autotune_repeat = int(autotune_repeat)
        self.memory_safety_fraction = float(memory_safety_fraction)
        self.eigh_memory_multiplier = float(eigh_memory_multiplier)
        self.trace_imag_tol = float(trace_imag_tol)
        self.herm_tol = float(herm_tol)
        self.nlayer = 2 * self.nx * self.ny
        self.actual_samples: int | None = None

        shape = (self.samples_expected, self.cycles)
        curve_shape = (self.samples_expected, self.cycles, self.ny // 2 + 1)
        map_shape = (self.samples_expected, self.cycles, self.nx, self.ny)
        self.frob_norm = np.full(shape, np.nan, dtype=np.float64)
        self.frob_successive_delta = np.full(shape, np.nan, dtype=np.float64)
        self.real_space_chern = np.full(shape, np.nan, dtype=np.float64)
        self.entropy_y0avg_vs_ay = np.full(curve_shape, np.nan, dtype=np.float64)
        self.xavg_square_correlator_vs_ry = np.full(curve_shape, np.nan, dtype=np.float64)
        self.local_chern_marker = np.full(map_shape, np.nan, dtype=np.float64)
        self.local_charge_cell = np.full(map_shape, np.nan, dtype=np.float64)
        if self.measurement_frustration_enabled:
            self.s_Ap = np.full(map_shape, np.nan, dtype=np.float64)
            self.s_Am = np.full(map_shape, np.nan, dtype=np.float64)
            self.s_Bp = np.full(map_shape, np.nan, dtype=np.float64)
            self.s_Bm = np.full(map_shape, np.nan, dtype=np.float64)
            self.frustration_min = np.full(map_shape, np.nan, dtype=np.float64)
            self.frustration_avg = np.full(map_shape, np.nan, dtype=np.float64)
            self.frustration_prod = np.full(map_shape, np.nan, dtype=np.float64)
            self.postselect_forced_site = np.zeros(map_shape, dtype=np.bool_)
            self._measurement_fill_count = np.zeros(map_shape, dtype=np.uint8)
        else:
            self.s_Ap = None
            self.s_Am = None
            self.s_Bp = None
            self.s_Bm = None
            self.frustration_min = None
            self.frustration_avg = None
            self.frustration_prod = None
            self.postselect_forced_site = None
            self._measurement_fill_count = None

        self.observer_stats = {
            "entropy_autotune": {},
            "entropy_production_chunks": {},
            "restricted_eigh_cpu_fallback_count": 0,
            "restricted_occupation_eval_min": np.inf,
            "restricted_occupation_eval_max": -np.inf,
            "restricted_max_hermiticity_error": 0.0,
            "full_batch_trace_imag_max_abs": 0.0,
            "full_batch_hermitian_max_err": 0.0,
            "local_chern_marker_final_sample_chunk": None,
            "local_chern_marker_oom_retries": 0,
            "s_Ap_min": np.inf,
            "s_Ap_max": -np.inf,
            "s_Am_min": np.inf,
            "s_Am_max": -np.inf,
            "s_Bp_min": np.inf,
            "s_Bp_max": -np.inf,
            "s_Bm_min": np.inf,
            "s_Bm_max": -np.inf,
            "frustration_min_min": np.inf,
            "frustration_min_max": -np.inf,
            "frustration_avg_min": np.inf,
            "frustration_avg_max": -np.inf,
            "frustration_prod_min": np.inf,
            "frustration_prod_max": -np.inf,
            "measurement_success_missing_entries": 0,
            "measurement_success_duplicate_entries": 0,
            "postselect_forced_site_count": 0,
        }
        self._prev_g: torch.Tensor | None = None
        self._progress_bar = None
        self._chern_partitions_cache: dict[str, dict[str, Any]] = {}
        self._square_corr_pair_cache: dict[str, dict[str, torch.Tensor]] = {}

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

    def _square_correlator_pairs_for_device(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        if key not in self._square_corr_pair_cache:
            self._square_corr_pair_cache[key] = build_square_correlator_pair_indices(
                nx=self.nx,
                ny=self.ny,
                device=device,
            )
        return self._square_corr_pair_cache[key]

    def _validate_top_layer_batch(self, g_batch: torch.Tensor) -> None:
        if g_batch.ndim != 3:
            raise ValueError(f"Expected G_batch with shape (B,N,N), got {tuple(g_batch.shape)}")
        _, nlayer, nlayer_2 = g_batch.shape
        if nlayer != nlayer_2 or nlayer != self.nlayer:
            raise ValueError(f"Expected covariance shape (B,{self.nlayer},{self.nlayer}), got {tuple(g_batch.shape)}")
        if not torch.isfinite(g_batch).all():
            bad = torch.nonzero(~torch.isfinite(g_batch), as_tuple=False)[0].detach().cpu().tolist()
            raise FloatingPointError(f"Non-finite covariance batch entry encountered at index {bad}")

    def _compute_entropy_curves_batch(self, g_batch: torch.Tensor) -> np.ndarray:
        batch_count = int(g_batch.shape[0])
        curves = np.zeros((batch_count, self.ny // 2 + 1), dtype=np.float64)
        for ay_int in range(1, self.ny // 2 + 1):
            key = str(ay_int)
            if key not in self.observer_stats["entropy_autotune"]:
                if self.autotune_entropy:
                    autotune_info = autotune_entropy_chunks_from_batch(
                        g_batch,
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
                self.observer_stats["entropy_autotune"][key] = autotune_info

            sample_chunk = max(1, int(self.observer_stats["entropy_autotune"][key]["selected_sample_chunk"]))
            y0_chunk = max(1, int(self.observer_stats["entropy_autotune"][key]["selected_y0_chunk"]))
            final_sample_chunk = sample_chunk
            final_y0_chunk = y0_chunk
            oom_retries = 0
            entropy_sum = torch.zeros((batch_count,), dtype=torch.float64, device=g_batch.device)
            sample_start = 0
            while sample_start < batch_count:
                sample_stop = min(batch_count, sample_start + final_sample_chunk)
                y0_start = 0
                while y0_start < self.ny:
                    try:
                        g_sample = g_batch[sample_start:sample_stop]
                        y0_stop = min(self.ny, y0_start + final_y0_chunk)
                        idx = strip_mode_indices(
                            nx=self.nx,
                            ny=self.ny,
                            ay=ay_int,
                            y0_values=range(y0_start, y0_stop),
                            device=g_batch.device,
                        )
                        sub_g = gather_restricted_covariance(g_sample, idx)
                        totals, metrics = entropy_total_batch_torch(sub_g, eps=self.entropy_eps, validate=True)
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
                        if g_batch.device.type == "cuda":
                            torch.cuda.empty_cache()
                        if final_y0_chunk > 1:
                            final_y0_chunk = max(1, final_y0_chunk // 2)
                        elif final_sample_chunk > 1:
                            if y0_start != 0:
                                raise RuntimeError(
                                    "OOM after partially processing a sample chunk. Lower sample_chunk before retrying."
                                ) from exc
                            final_sample_chunk = max(1, final_sample_chunk // 2)
                            sample_stop = min(batch_count, sample_start + final_sample_chunk)
                        else:
                            raise
                sample_start = sample_stop
            self.observer_stats["entropy_production_chunks"][key] = {
                "selected_sample_chunk": int(sample_chunk),
                "selected_y0_chunk": int(y0_chunk),
                "final_sample_chunk": int(final_sample_chunk),
                "final_y0_chunk": int(final_y0_chunk),
                "oom_retries": int(oom_retries),
            }
            curves[:, ay_int] = (entropy_sum / float(self.ny)).detach().cpu().numpy().astype(np.float64, copy=False)
        return curves

    def _compute_local_chern_marker_batch(self, g_batch: torch.Tensor) -> np.ndarray:
        batch_count = int(g_batch.shape[0])
        final_chunk = batch_count
        oom_retries = 0
        outputs = np.empty((batch_count, self.nx, self.ny), dtype=np.float64)
        start = 0
        while start < batch_count:
            stop = min(batch_count, start + final_chunk)
            try:
                marker = local_chern_marker_batch_torch(g_batch[start:stop], nx=self.nx, ny=self.ny)
                outputs[start:stop] = marker.detach().cpu().numpy().astype(np.float64, copy=False)
                start = stop
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                oom_retries += 1
                if g_batch.device.type == "cuda":
                    torch.cuda.empty_cache()
                if final_chunk > 1:
                    final_chunk = max(1, final_chunk // 2)
                    continue
                raise
        self.observer_stats["local_chern_marker_final_sample_chunk"] = int(final_chunk)
        self.observer_stats["local_chern_marker_oom_retries"] = int(
            self.observer_stats["local_chern_marker_oom_retries"]
        ) + int(oom_retries)
        return outputs

    @staticmethod
    def _as_numpy_float64(values: Any, *, name: str, expected_size: int) -> np.ndarray:
        if torch.is_tensor(values):
            arr = values.detach().cpu().numpy()
        else:
            arr = np.asarray(values)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size != int(expected_size):
            raise ValueError(f"{name} expected {expected_size} values, got shape {arr.shape}")
        if not np.isfinite(arr).all():
            raise FloatingPointError(f"{name} contains non-finite values")
        tol = float(SUCCESS_PROB_TOL)
        if float(np.min(arr)) < -tol or float(np.max(arr)) > 1.0 + tol:
            raise FloatingPointError(
                f"{name} left the probability interval: [{float(np.min(arr)):.6e}, {float(np.max(arr)):.6e}]"
            )
        return arr

    def observe_site(
        self,
        *,
        cycle: int,
        site_ids: Any,
        sample_indices: Any | None = None,
        sample_offsets: Any | None = None,
        batch_index: int,
        batch_start: int,
        batch_count: int,
        s_Ap: Any,
        s_Am: Any,
        s_Bp: Any,
        s_Bm: Any,
        postselect_mask: Any | None = None,
    ) -> None:
        del batch_index, batch_start
        if not self.measurement_frustration_enabled:
            raise RuntimeError("Measurement-frustration recording is only enabled for perfect_correction runs.")
        cycle = int(cycle)
        if cycle < 1 or cycle > self.cycles:
            raise ValueError(f"cycle must satisfy 1 <= cycle <= {self.cycles}; got {cycle}")
        batch_count = int(batch_count)
        cycle_idx = cycle - 1
        def _as_numpy_int64(values: Any) -> np.ndarray:
            if torch.is_tensor(values):
                arr_local = values.detach().cpu().numpy()
            else:
                arr_local = np.asarray(values)
            return np.asarray(arr_local, dtype=np.int64).reshape(-1)
        if sample_indices is None:
            if sample_offsets is None:
                sample_indices_arr = np.arange(batch_count, dtype=np.int64)
            else:
                sample_indices_arr = _as_numpy_int64(sample_offsets)
        else:
            sample_indices_arr = _as_numpy_int64(sample_indices)
        if sample_indices_arr.size != batch_count:
            raise ValueError(
                f"sample_indices size must match batch_count={batch_count}; got {sample_indices_arr.size}"
            )
        if np.any(sample_indices_arr < 0) or np.any(sample_indices_arr >= self.samples_expected):
            raise IndexError("sample_indices are out of bounds for the configured sample count")

        if torch.is_tensor(site_ids):
            site_ids_arr = site_ids.detach().cpu().numpy()
        else:
            site_ids_arr = np.asarray(site_ids)
        site_ids_arr = np.asarray(site_ids_arr, dtype=np.int64).reshape(-1)
        if site_ids_arr.size != batch_count:
            raise ValueError(f"site_ids size must match batch_count={batch_count}; got {site_ids_arr.size}")
        x_coords = site_ids_arr % self.nx
        y_coords = site_ids_arr // self.nx
        if np.any(y_coords < 0) or np.any(y_coords >= self.ny):
            raise IndexError("site_ids map outside the configured lattice geometry")

        s_ap_arr = self._as_numpy_float64(s_Ap, name="s_Ap", expected_size=batch_count)
        s_am_arr = self._as_numpy_float64(s_Am, name="s_Am", expected_size=batch_count)
        s_bp_arr = self._as_numpy_float64(s_Bp, name="s_Bp", expected_size=batch_count)
        s_bm_arr = self._as_numpy_float64(s_Bm, name="s_Bm", expected_size=batch_count)
        if postselect_mask is None:
            postselect_mask_arr = np.zeros((batch_count,), dtype=np.bool_)
        elif torch.is_tensor(postselect_mask):
            postselect_mask_arr = postselect_mask.detach().cpu().numpy().astype(np.bool_, copy=False).reshape(-1)
        else:
            postselect_mask_arr = np.asarray(postselect_mask, dtype=np.bool_).reshape(-1)
        if postselect_mask_arr.size != batch_count:
            raise ValueError(
                f"postselect_mask size must match batch_count={batch_count}; got {postselect_mask_arr.size}"
            )

        frustration_min = 1.0 - np.minimum.reduce([s_ap_arr, s_am_arr, s_bp_arr, s_bm_arr])
        frustration_avg = 1.0 - 0.25 * (s_ap_arr + s_am_arr + s_bp_arr + s_bm_arr)
        frustration_prod = 1.0 - (s_ap_arr * s_am_arr * s_bp_arr * s_bm_arr)

        prev_counts = self._measurement_fill_count[sample_indices_arr, cycle_idx, x_coords, y_coords]
        duplicate_entries = int(np.count_nonzero(prev_counts))
        self.observer_stats["measurement_success_duplicate_entries"] = int(
            self.observer_stats["measurement_success_duplicate_entries"]
        ) + duplicate_entries
        self._measurement_fill_count[sample_indices_arr, cycle_idx, x_coords, y_coords] = prev_counts + 1

        self.s_Ap[sample_indices_arr, cycle_idx, x_coords, y_coords] = s_ap_arr
        self.s_Am[sample_indices_arr, cycle_idx, x_coords, y_coords] = s_am_arr
        self.s_Bp[sample_indices_arr, cycle_idx, x_coords, y_coords] = s_bp_arr
        self.s_Bm[sample_indices_arr, cycle_idx, x_coords, y_coords] = s_bm_arr
        self.frustration_min[sample_indices_arr, cycle_idx, x_coords, y_coords] = frustration_min
        self.frustration_avg[sample_indices_arr, cycle_idx, x_coords, y_coords] = frustration_avg
        self.frustration_prod[sample_indices_arr, cycle_idx, x_coords, y_coords] = frustration_prod
        self.postselect_forced_site[sample_indices_arr, cycle_idx, x_coords, y_coords] = postselect_mask_arr
        self.observer_stats["postselect_forced_site_count"] = int(
            self.observer_stats["postselect_forced_site_count"]
        ) + int(np.count_nonzero(postselect_mask_arr))

        for key, values in (
            ("s_Ap", s_ap_arr),
            ("s_Am", s_am_arr),
            ("s_Bp", s_bp_arr),
            ("s_Bm", s_bm_arr),
            ("frustration_min", frustration_min),
            ("frustration_avg", frustration_avg),
            ("frustration_prod", frustration_prod),
        ):
            self.observer_stats[f"{key}_min"] = min(float(self.observer_stats[f"{key}_min"]), float(np.min(values)))
            self.observer_stats[f"{key}_max"] = max(float(self.observer_stats[f"{key}_max"]), float(np.max(values)))

    def observe(self, *, cycle: int, G: torch.Tensor, batch_index: int, batch_start: int, batch_count: int) -> None:
        del batch_index
        g_work = G.detach().clone()
        self._validate_top_layer_batch(g_work)
        cycle = int(cycle)
        batch_start = int(batch_start)
        batch_count = int(batch_count)
        sl = slice(batch_start, batch_start + batch_count)
        if cycle == 0:
            self._prev_g = g_work
            return
        if self._prev_g is None:
            raise RuntimeError("Observer state is missing the previous-cycle covariance batch.")

        cycle_idx = cycle - 1
        herm_vals = torch.amax(torch.abs(g_work - g_work.conj().transpose(-2, -1)), dim=(-2, -1)).to(torch.float64)
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

        trace_vals = torch.diagonal(g_work, dim1=-2, dim2=-1).sum(dim=-1)
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

        self.frob_norm[sl, cycle_idx] = (
            torch.linalg.vector_norm(g_work.reshape(batch_count, -1), dim=-1).to(torch.float64).detach().cpu().numpy()
        )

        prev_g = self._prev_g
        delta = torch.linalg.vector_norm((g_work - prev_g).reshape(batch_count, -1), dim=-1).to(torch.float64)
        self.frob_successive_delta[sl, cycle_idx] = delta.detach().cpu().numpy().astype(np.float64, copy=False)
        self._prev_g = None
        del prev_g

        entropy_batch = self._compute_entropy_curves_batch(g_work)
        self.entropy_y0avg_vs_ay[sl, cycle_idx, :] = entropy_batch

        partitions = self._chern_partitions_for_device(g_work.device)
        chern_vals = real_space_chern_batch_torch(g_work, partitions)
        self.real_space_chern[sl, cycle_idx] = chern_vals.detach().cpu().numpy().astype(np.float64, copy=False)

        pair_indices = self._square_correlator_pairs_for_device(g_work.device)
        corr_vals = xavg_square_correlator_batch_torch(g_work, pair_indices, nx=self.nx, ny=self.ny)
        self.xavg_square_correlator_vs_ry[sl, cycle_idx, :] = (
            corr_vals.detach().cpu().numpy().astype(np.float64, copy=False)
        )

        self.local_chern_marker[sl, cycle_idx, :, :] = self._compute_local_chern_marker_batch(g_work)

        charge_cell = local_charge_cell_batch_torch(g_work, nx=self.nx, ny=self.ny)
        self.local_charge_cell[sl, cycle_idx, :, :] = charge_cell.detach().cpu().numpy().astype(np.float64, copy=False)

        self._prev_g = g_work
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

    def make_site_observer(self):
        if not self.measurement_frustration_enabled:
            raise RuntimeError("Measurement-frustration site recording is disabled for this protocol.")

        def _observer(
            *,
            cycle,
            site_ids,
            batch_index,
            batch_start,
            batch_count,
            s_Ap,
            s_Am,
            s_Bp,
            s_Bm,
            sample_indices=None,
            sample_offsets=None,
            postselect_mask=None,
        ):
            self.observe_site(
                cycle=cycle,
                site_ids=site_ids,
                batch_index=batch_index,
                batch_start=batch_start,
                batch_count=batch_count,
                s_Ap=s_Ap,
                s_Am=s_Am,
                s_Bp=s_Bp,
                s_Bm=s_Bm,
                sample_indices=sample_indices,
                sample_offsets=sample_offsets,
                postselect_mask=postselect_mask,
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
            "frob_norm": self.frob_norm[sample_slice],
            "frob_successive_delta": self.frob_successive_delta[sample_slice],
            "real_space_chern": self.real_space_chern[sample_slice],
            "entropy_y0avg_vs_ay": self.entropy_y0avg_vs_ay[sample_slice],
            "xavg_square_correlator_vs_ry": self.xavg_square_correlator_vs_ry[sample_slice],
            "local_chern_marker": self.local_chern_marker[sample_slice],
            "local_charge_cell": self.local_charge_cell[sample_slice],
        }
        if self.measurement_frustration_enabled:
            required_arrays.update(
                {
                    "s_Ap": self.s_Ap[sample_slice],
                    "s_Am": self.s_Am[sample_slice],
                    "s_Bp": self.s_Bp[sample_slice],
                    "s_Bm": self.s_Bm[sample_slice],
                    "frustration_min": self.frustration_min[sample_slice],
                    "frustration_avg": self.frustration_avg[sample_slice],
                    "frustration_prod": self.frustration_prod[sample_slice],
                    "postselect_forced_site": self.postselect_forced_site[sample_slice],
                }
            )
        for name, value in required_arrays.items():
            if np.isnan(np.asarray(value, dtype=np.float64)).any():
                raise RuntimeError(f"{name} contains unfilled NaN entries after finalize().")
        if self.measurement_frustration_enabled:
            counts = self._measurement_fill_count[sample_slice]
            missing_entries = int(np.count_nonzero(counts == 0))
            duplicate_entries = int(np.count_nonzero(counts > 1))
            self.observer_stats["measurement_success_missing_entries"] = missing_entries
            self.observer_stats["measurement_success_duplicate_entries"] = duplicate_entries
            if missing_entries or duplicate_entries:
                raise RuntimeError(
                    "Measurement-frustration site recording did not fill every site exactly once per sample-cycle: "
                    f"missing={missing_entries}, duplicate={duplicate_entries}"
                )
            expected_min = 1.0 - np.minimum.reduce(
                [
                    self.s_Ap[sample_slice],
                    self.s_Am[sample_slice],
                    self.s_Bp[sample_slice],
                    self.s_Bm[sample_slice],
                ]
            )
            expected_avg = 1.0 - 0.25 * (
                self.s_Ap[sample_slice]
                + self.s_Am[sample_slice]
                + self.s_Bp[sample_slice]
                + self.s_Bm[sample_slice]
            )
            expected_prod = 1.0 - (
                self.s_Ap[sample_slice]
                * self.s_Am[sample_slice]
                * self.s_Bp[sample_slice]
                * self.s_Bm[sample_slice]
            )
            if not np.allclose(self.frustration_min[sample_slice], expected_min):
                raise RuntimeError("frustration_min does not match the stored success-probability maps")
            if not np.allclose(self.frustration_avg[sample_slice], expected_avg):
                raise RuntimeError("frustration_avg does not match the stored success-probability maps")
            if not np.allclose(self.frustration_prod[sample_slice], expected_prod):
                raise RuntimeError("frustration_prod does not match the stored success-probability maps")
        if not np.allclose(self.entropy_y0avg_vs_ay[sample_slice, :, 0], 0.0):
            raise RuntimeError("Ay=0 entropy slice must remain exactly zero.")

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
                        "frob_norm": float(self.frob_norm[sample_index, cycle_idx]),
                        "frob_successive_delta": float(self.frob_successive_delta[sample_index, cycle_idx]),
                        "real_space_chern": float(self.real_space_chern[sample_index, cycle_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def _base_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        if self.actual_samples is None:
            raise RuntimeError("Call finalize(actual_samples=...) before exporting data.")
        return {
            "cycle_labels": np.arange(1, self.cycles + 1, dtype=np.int64),
            "sample_indices": np.arange(self.actual_samples, dtype=np.int64),
            "config_json": np.asarray(json.dumps(config, sort_keys=True)),
            "helper_version": np.asarray(HELPER_VERSION),
        }

    def entropy_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "entropy_y0avg_vs_ay": self.entropy_y0avg_vs_ay[: self.actual_samples].copy(),
            "ay_values": np.arange(self.ny // 2 + 1, dtype=np.int64),
        }

    def square_correlator_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "xavg_square_correlator_vs_ry": self.xavg_square_correlator_vs_ry[: self.actual_samples].copy(),
            "ry_values": np.arange(self.ny // 2 + 1, dtype=np.int64),
            "formula": np.asarray(
                "C=(G+I)/2; xavg_corr(ry)=(1/(2*Nx*Ny))*sum_x,y,mu,nu |C[x,y,mu;x,y+ry,nu]|^2"
            ),
        }

    def local_chern_marker_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "local_chern_marker": self.local_chern_marker[: self.actual_samples].copy(),
            "x_coords": np.arange(self.nx, dtype=np.int64),
            "y_coords": np.arange(self.ny, dtype=np.int64),
        }

    def local_charge_cell_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._base_payload(config=config),
            "local_charge_cell": self.local_charge_cell[: self.actual_samples].copy(),
            "x_coords": np.arange(self.nx, dtype=np.int64),
            "y_coords": np.arange(self.ny, dtype=np.int64),
            "convention": np.asarray("sum_orbital diag((G+I)/2)"),
        }

    def measurement_frustration_npz_payload(self, *, config: dict[str, Any]) -> dict[str, Any]:
        if not self.measurement_frustration_enabled:
            raise RuntimeError("Measurement-frustration payloads are only available for perfect_correction runs.")
        return {
            **self._base_payload(config=config),
            "s_Ap": self.s_Ap[: self.actual_samples].copy(),
            "s_Am": self.s_Am[: self.actual_samples].copy(),
            "s_Bp": self.s_Bp[: self.actual_samples].copy(),
            "s_Bm": self.s_Bm[: self.actual_samples].copy(),
            "frustration_min": self.frustration_min[: self.actual_samples].copy(),
            "frustration_avg": self.frustration_avg[: self.actual_samples].copy(),
            "frustration_prod": self.frustration_prod[: self.actual_samples].copy(),
            "postselect_forced_site": self.postselect_forced_site[: self.actual_samples].copy(),
            "x_coords": np.arange(self.nx, dtype=np.int64),
            "y_coords": np.arange(self.ny, dtype=np.int64),
            "frustration_order": np.asarray("Ap -> Am -> Bp -> Bm"),
            "formula_s_Ap": np.asarray("1 - p_occ(Ap | state before Ap)"),
            "formula_s_Am": np.asarray("p_occ(Am | state after full Ap step)"),
            "formula_s_Bp": np.asarray("1 - p_occ(Bp | state after full Ap, Am steps)"),
            "formula_s_Bm": np.asarray("p_occ(Bm | state after full Ap, Am, Bp steps)"),
            "formula_frustration_min": np.asarray("1 - min(s_Ap, s_Am, s_Bp, s_Bm)"),
            "formula_frustration_avg": np.asarray("1 - (s_Ap + s_Am + s_Bp + s_Bm)/4"),
            "formula_frustration_prod": np.asarray("1 - s_Ap*s_Am*s_Bp*s_Bm"),
            "formula_postselect_forced_site": np.asarray("True when the site update used the forced post-selected map"),
            "protocol_metadata": np.asarray("saved only for perfect_correction and partial post-selection diagnostics"),
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
                "local_chern_marker_final_sample_chunk": self.observer_stats["local_chern_marker_final_sample_chunk"],
                "local_chern_marker_oom_retries": int(self.observer_stats["local_chern_marker_oom_retries"]),
                "measurement_frustration_enabled": bool(self.measurement_frustration_enabled),
                "postselect_forced_site_count": int(self.observer_stats["postselect_forced_site_count"]),
                "s_Ap_min": None if not np.isfinite(self.observer_stats["s_Ap_min"]) else float(self.observer_stats["s_Ap_min"]),
                "s_Ap_max": None if not np.isfinite(self.observer_stats["s_Ap_max"]) else float(self.observer_stats["s_Ap_max"]),
                "s_Am_min": None if not np.isfinite(self.observer_stats["s_Am_min"]) else float(self.observer_stats["s_Am_min"]),
                "s_Am_max": None if not np.isfinite(self.observer_stats["s_Am_max"]) else float(self.observer_stats["s_Am_max"]),
                "s_Bp_min": None if not np.isfinite(self.observer_stats["s_Bp_min"]) else float(self.observer_stats["s_Bp_min"]),
                "s_Bp_max": None if not np.isfinite(self.observer_stats["s_Bp_max"]) else float(self.observer_stats["s_Bp_max"]),
                "s_Bm_min": None if not np.isfinite(self.observer_stats["s_Bm_min"]) else float(self.observer_stats["s_Bm_min"]),
                "s_Bm_max": None if not np.isfinite(self.observer_stats["s_Bm_max"]) else float(self.observer_stats["s_Bm_max"]),
                "frustration_min_min": None if not np.isfinite(self.observer_stats["frustration_min_min"]) else float(self.observer_stats["frustration_min_min"]),
                "frustration_min_max": None if not np.isfinite(self.observer_stats["frustration_min_max"]) else float(self.observer_stats["frustration_min_max"]),
                "frustration_avg_min": None if not np.isfinite(self.observer_stats["frustration_avg_min"]) else float(self.observer_stats["frustration_avg_min"]),
                "frustration_avg_max": None if not np.isfinite(self.observer_stats["frustration_avg_max"]) else float(self.observer_stats["frustration_avg_max"]),
                "frustration_prod_min": None if not np.isfinite(self.observer_stats["frustration_prod_min"]) else float(self.observer_stats["frustration_prod_min"]),
                "frustration_prod_max": None if not np.isfinite(self.observer_stats["frustration_prod_max"]) else float(self.observer_stats["frustration_prod_max"]),
                "measurement_success_missing_entries": int(self.observer_stats["measurement_success_missing_entries"]),
                "measurement_success_duplicate_entries": int(self.observer_stats["measurement_success_duplicate_entries"]),
            }
        }
