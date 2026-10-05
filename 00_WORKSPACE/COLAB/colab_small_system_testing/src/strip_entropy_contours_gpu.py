from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


HELPER_VERSION = "strip_entropy_contours_gpu_robust_eigh_v2"


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


def rel_to_root(path: Path, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return None


def strip_mode_indices(
    *,
    nx: int,
    ny: int,
    ay: int,
    y0_values,
    device: torch.device | str,
) -> torch.Tensor:
    """Return mode indices ordered as dy, x, orbital for each y0."""
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
    # (Y0, Ay, Nx, 2), flattened in local-strip order: dy, x, orbital.
    idx = 2 * x[None, None, :, None] + 2 * nx * y[:, :, None, None] + orbital[None, None, None, :]
    return idx.reshape(int(y0_values.numel()), ay * nx * 2).contiguous()


def gather_restricted_covariance(G_batch: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather G[:, idx, idx] for all y0 indices.

    G_batch has shape (S, N, N), idx has shape (Y0, M), and the result has
    shape (S * Y0, M, M).
    """
    if G_batch.ndim != 3:
        raise ValueError(f"Expected G_batch with shape (S,N,N), got {tuple(G_batch.shape)}")
    if idx.ndim != 2:
        raise ValueError(f"Expected idx with shape (Y0,M), got {tuple(idx.shape)}")
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


def _eigh_with_fallback(occ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run Hermitian eigendecomposition, falling back around rare CUDA batched failures."""
    try:
        evals, vecs = torch.linalg.eigh(occ)
        return evals, vecs, 0
    except Exception as exc:
        if not _is_eigh_convergence_error(exc):
            raise

    if occ.ndim == 3 and int(occ.shape[0]) > 1:
        mid = int(occ.shape[0]) // 2
        evals_left, vecs_left, fallbacks_left = _eigh_with_fallback(occ[:mid])
        evals_right, vecs_right, fallbacks_right = _eigh_with_fallback(occ[mid:])
        return (
            torch.cat([evals_left, evals_right], dim=0),
            torch.cat([vecs_left, vecs_right], dim=0),
            fallbacks_left + fallbacks_right,
        )

    target_device = occ.device
    occ_cpu = occ.detach().cpu()
    try:
        evals_cpu, vecs_cpu = torch.linalg.eigh(occ_cpu)
    except Exception:
        n = int(occ_cpu.shape[-1])
        jitter = 100.0 * torch.finfo(occ_cpu.real.dtype).eps
        eye = torch.eye(n, dtype=occ_cpu.dtype, device=occ_cpu.device)
        evals_cpu, vecs_cpu = torch.linalg.eigh(occ_cpu + jitter * eye)
    return evals_cpu.to(target_device), vecs_cpu.to(target_device), 1


def entropy_contour_batch_torch(
    sub_G: torch.Tensor,
    *,
    nx: int,
    ay: int,
    eps: float = 1e-12,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute strip entanglement contours from restricted covariance matrices."""
    nx = int(nx)
    ay = int(ay)
    if sub_G.ndim != 3:
        raise ValueError(f"Expected sub_G with shape (B,M,M), got {tuple(sub_G.shape)}")
    batch_count, nrow, ncol = sub_G.shape
    expected = 2 * nx * ay
    if (nrow, ncol) != (expected, expected):
        raise ValueError(f"Expected restricted shape (B,{expected},{expected}), got {tuple(sub_G.shape)}")
    if ay == 0:
        contour = torch.zeros((batch_count, nx, 0), dtype=torch.float64, device=sub_G.device)
        total = torch.zeros((batch_count,), dtype=torch.float64, device=sub_G.device)
        return contour, total, {
            "max_hermiticity_error": 0.0,
            "min_occupation_eval": 0.0,
            "max_occupation_eval": 0.0,
            "max_total_mismatch": 0.0,
        }
    if validate and not torch.isfinite(sub_G).all():
        raise FloatingPointError("Non-finite restricted covariance entries.")

    herm_error = torch.max(torch.abs(sub_G - sub_G.conj().transpose(-2, -1))).detach()
    eye = torch.eye(expected, dtype=sub_G.dtype, device=sub_G.device)
    occ = 0.5 * (sub_G + eye.unsqueeze(0))
    occ = 0.5 * (occ + occ.conj().transpose(-2, -1))
    evals, vecs, eigh_cpu_fallback_count = _eigh_with_fallback(occ)
    evals_real = evals.real
    min_eval = torch.min(evals_real).detach()
    max_eval = torch.max(evals_real).detach()
    evals_clamped = torch.clamp(evals_real, float(eps), 1.0 - float(eps))
    weights = -(evals_clamped * torch.log(evals_clamped) + (1.0 - evals_clamped) * torch.log(1.0 - evals_clamped))
    diag_f = torch.sum(torch.abs(vecs) ** 2 * weights.unsqueeze(-2), dim=-1).real
    contour = diag_f.reshape(batch_count, ay, nx, 2).sum(dim=-1).transpose(1, 2).contiguous()
    total = weights.sum(dim=-1)
    total_from_contour = contour.sum(dim=(1, 2))
    mismatch = torch.max(torch.abs(total - total_from_contour)).detach()
    if validate:
        if not torch.isfinite(contour).all() or not torch.isfinite(total).all():
            raise FloatingPointError("Non-finite entropy contour output.")
        if torch.min(contour) < -1e-8:
            raise FloatingPointError("Negative entropy contour entry below tolerance.")

    metrics = {
        "max_hermiticity_error": float(herm_error.detach().cpu()),
        "min_occupation_eval": float(min_eval.detach().cpu()),
        "max_occupation_eval": float(max_eval.detach().cpu()),
        "max_total_mismatch": float(mismatch.detach().cpu()),
        "eigh_cpu_fallback_count": int(eigh_cpu_fallback_count),
    }
    return contour.to(torch.float64), total.to(torch.float64), metrics


def _normalize_y0_chunk_candidates(candidates, ny: int) -> list[int]:
    ny = int(ny)
    if candidates is None:
        candidates = [1, 2, 4, 8, 12, 16, 24, 32, ny]
    normalized = []
    for candidate in candidates:
        if isinstance(candidate, str):
            if candidate.lower() != "ny":
                raise ValueError(f"Unknown y0 chunk candidate {candidate!r}; only 'ny' is supported.")
            value = ny
        else:
            value = int(candidate)
        if 1 <= value <= ny:
            normalized.append(value)
    return sorted(set(normalized))


def _cuda_memory_snapshot(device: torch.device) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_gib": float(free_bytes / 1024**3),
        "total_gib": float(total_bytes / 1024**3),
    }


def autotune_y0_chunk_for_ay(
    *,
    shard_path: Path,
    nx: int,
    ny: int,
    ay: int,
    device: torch.device | str = "cuda:0",
    eps: float = 1e-12,
    candidates=None,
    repeat: int = 2,
    validate: bool = True,
    progress=None,
) -> dict[str, Any]:
    """Benchmark y0 chunk sizes on the real first snapshot and select the fastest safe value."""
    shard_path = Path(shard_path)
    shard = np.load(shard_path, mmap_mode="r")
    sample_count, time_count, nlayer, nlayer_2 = shard.shape
    expected_nlayer = 2 * int(nx) * int(ny)
    if nlayer != nlayer_2 or nlayer != expected_nlayer:
        raise ValueError(
            f"Expected shard shape (S,T,{expected_nlayer},{expected_nlayer}), got {shard.shape}"
        )
    if time_count < 1:
        raise ValueError("Cannot autotune with an empty snapshot shard.")
    if str(shard.dtype) != "complex128":
        raise ValueError(f"Expected complex128 shard, got {shard.dtype}.")

    ay = int(ay)
    if ay < 0 or ay > int(ny):
        raise ValueError("Ay must be in 0..Ny.")
    candidate_values = _normalize_y0_chunk_candidates(candidates, int(ny))
    if ay == 0:
        return {
            "selected_y0_chunk": 0,
            "candidates": candidate_values,
            "trials": [],
            "repeat": int(repeat),
            "sample_count": int(sample_count),
            "benchmark_snapshot_index": 0,
            "device": str(device),
            "device_name": "",
            "memory_before": {},
            "memory_after": {},
        }

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("autotune_y0_chunk_for_ay requires a CUDA device.")

    repeat = max(1, int(repeat))
    device_name = torch.cuda.get_device_name(device)
    memory_before = _cuda_memory_snapshot(device)
    G_t = torch.as_tensor(np.array(shard[:, 0], dtype=np.complex128, copy=True), dtype=torch.complex128, device=device)
    torch.cuda.synchronize(device)

    trials = []
    iterator = candidate_values if progress is None else progress(
        candidate_values,
        desc=f"Autotune Ay={ay}",
        leave=False,
        unit="chunk",
    )
    for chunk in iterator:
        y0_values = list(range(min(int(ny), int(chunk))))
        elapsed_ms = []
        peak_gib = 0.0
        status = "ok"
        error = None
        for _ in range(repeat):
            idx = sub_G = contour_flat = total_flat = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                idx = strip_mode_indices(nx=nx, ny=ny, ay=ay, y0_values=y0_values, device=device)
                sub_G = gather_restricted_covariance(G_t, idx)
                contour_flat, total_flat, _ = entropy_contour_batch_torch(
                    sub_G,
                    nx=nx,
                    ay=ay,
                    eps=eps,
                    validate=validate,
                )
                end.record()
                torch.cuda.synchronize(device)
                elapsed_ms.append(float(start.elapsed_time(end)))
                peak_gib = max(peak_gib, float(torch.cuda.max_memory_allocated(device) / 1024**3))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    status = "oom"
                    error = str(exc).splitlines()[0]
                    torch.cuda.empty_cache()
                    break
                raise
            finally:
                del idx, sub_G, contour_flat, total_flat

        trial = {
            "candidate_y0_chunk": int(chunk),
            "status": status,
            "elapsed_ms": elapsed_ms,
            "mean_ms": float(np.mean(elapsed_ms)) if elapsed_ms else None,
            "min_ms": float(np.min(elapsed_ms)) if elapsed_ms else None,
            "peak_cuda_gib": float(peak_gib),
            "batch_count": int(sample_count) * int(chunk),
            "matrix_size": int(2 * int(nx) * ay),
        }
        if error is not None:
            trial["error"] = error
        trials.append(trial)

    del G_t
    torch.cuda.empty_cache()
    memory_after = _cuda_memory_snapshot(device)
    safe_trials = [trial for trial in trials if trial["status"] == "ok" and trial["mean_ms"] is not None]
    if not safe_trials:
        raise RuntimeError(f"No safe y0 chunk candidates for Ay={ay}.")
    selected = min(safe_trials, key=lambda trial: (trial["mean_ms"], -trial["candidate_y0_chunk"]))
    return {
        "selected_y0_chunk": int(selected["candidate_y0_chunk"]),
        "candidates": candidate_values,
        "trials": trials,
        "repeat": int(repeat),
        "sample_count": int(sample_count),
        "benchmark_snapshot_index": 0,
        "benchmark_y0_start": 0,
        "benchmark_y0_stop": int(selected["candidate_y0_chunk"]),
        "selected_mean_ms": float(selected["mean_ms"]),
        "selected_min_ms": float(selected["min_ms"]),
        "selected_peak_cuda_gib": float(selected["peak_cuda_gib"]),
        "device": str(device),
        "device_name": device_name,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }


def _empty_ay_result(*, snapshot_cycles, sample_count: int, nx: int, ay: int, ny: int) -> dict[str, np.ndarray | int | float | list]:
    time_count = len(snapshot_cycles)
    shape = (time_count, int(nx), int(ay))
    return {
        "entropy_contour_avg": np.zeros(shape, dtype=np.float64),
        "entropy_contour_std": np.zeros(shape, dtype=np.float64),
        "entropy_contour_sem": np.zeros(shape, dtype=np.float64),
        "entropy_contour_y0_sample_sum": np.zeros(shape, dtype=np.float64),
        "entropy_contour_y0_sample_sumsq": np.zeros(shape, dtype=np.float64),
        "entropy_total_mean": np.zeros((time_count,), dtype=np.float64),
        "entropy_total_std": np.zeros((time_count,), dtype=np.float64),
        "entropy_total_sem": np.zeros((time_count,), dtype=np.float64),
        "entropy_total_y0_sample_sum": np.zeros((time_count,), dtype=np.float64),
        "entropy_total_y0_sample_sumsq": np.zeros((time_count,), dtype=np.float64),
        "snapshot_cycles": np.asarray(snapshot_cycles, dtype=np.int64),
        "x": np.arange(int(nx), dtype=np.int64),
        "strip_dy": np.arange(int(ay), dtype=np.int64),
        "Ay": np.asarray(int(ay), dtype=np.int64),
        "sample_count": np.asarray(int(sample_count), dtype=np.int64),
        "y0_count": np.asarray(int(ny), dtype=np.int64),
        "observation_count": np.asarray(int(sample_count) * int(ny), dtype=np.int64),
        "selected_y0_chunk": int(0),
        "final_y0_chunk": int(0),
        "peak_cuda_gib": float(0.0),
        "validation_metrics": [],
    }


def compute_strip_entropy_for_ay(
    *,
    shard_path: Path,
    nx: int,
    ny: int,
    ay: int,
    snapshot_cycles,
    device: torch.device | str = "cuda:0",
    eps: float = 1e-12,
    y0_chunk_max: int = 8,
    validate: bool = True,
    progress=None,
) -> dict[str, Any]:
    """Compute sample/y0-averaged strip entropy contours for one Ay."""
    shard_path = Path(shard_path)
    shard = np.load(shard_path, mmap_mode="r")
    sample_count, time_count, nlayer, nlayer_2 = shard.shape
    expected_nlayer = 2 * int(nx) * int(ny)
    if nlayer != nlayer_2 or nlayer != expected_nlayer:
        raise ValueError(
            f"Expected shard shape (S,T,{expected_nlayer},{expected_nlayer}), got {shard.shape}"
        )
    if time_count != len(snapshot_cycles):
        raise ValueError(f"Expected {len(snapshot_cycles)} snapshots, got {time_count}.")
    if str(shard.dtype) != "complex128":
        raise ValueError(f"Expected complex128 shard, got {shard.dtype}.")

    ay = int(ay)
    if ay < 0 or ay > int(ny):
        raise ValueError("Ay must be in 0..Ny.")
    if ay == 0:
        return _empty_ay_result(snapshot_cycles=snapshot_cycles, sample_count=sample_count, nx=nx, ay=ay, ny=ny)

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("compute_strip_entropy_for_ay requires a CUDA device.")
    torch.cuda.reset_peak_memory_stats(device)

    contour_obs_sum = np.zeros((time_count, int(nx), ay), dtype=np.float64)
    contour_obs_sumsq = np.zeros((time_count, int(nx), ay), dtype=np.float64)
    total_obs_sum = np.zeros((time_count,), dtype=np.float64)
    total_obs_sumsq = np.zeros((time_count,), dtype=np.float64)
    metrics = []
    selected_y0_chunk = min(int(ny), int(max(1, y0_chunk_max)))
    min_chunk_used = selected_y0_chunk

    iterator = range(time_count) if progress is None else progress(range(time_count), desc=f"Snapshots Ay={ay}", leave=False)
    for t_idx in iterator:
        G_t = torch.as_tensor(np.array(shard[:, t_idx], dtype=np.complex128, copy=True), dtype=torch.complex128, device=device)
        y0_start = 0
        y0_chunk = selected_y0_chunk
        while y0_start < int(ny):
            stop = min(int(ny), y0_start + y0_chunk)
            y0_values = list(range(y0_start, stop))
            idx = sub_G = contour_flat = total_flat = contour = total = None
            try:
                idx = strip_mode_indices(nx=nx, ny=ny, ay=ay, y0_values=y0_values, device=device)
                sub_G = gather_restricted_covariance(G_t, idx)
                contour_flat, total_flat, block_metrics = entropy_contour_batch_torch(
                    sub_G,
                    nx=nx,
                    ay=ay,
                    eps=eps,
                    validate=validate,
                )
            except RuntimeError as exc:
                if device.type == "cuda" and "out of memory" in str(exc).lower() and y0_chunk > 1:
                    del idx, sub_G, contour_flat, total_flat, contour, total
                    torch.cuda.empty_cache()
                    y0_chunk = max(1, y0_chunk // 2)
                    min_chunk_used = min(min_chunk_used, y0_chunk)
                    continue
                raise

            y0_count = len(y0_values)
            contour = contour_flat.reshape(sample_count, y0_count, int(nx), ay)
            total = total_flat.reshape(sample_count, y0_count)
            contour_cpu = contour.detach().cpu().numpy()
            total_cpu = total.detach().cpu().numpy()

            contour_obs_sum[t_idx] += contour_cpu.sum(axis=(0, 1))
            contour_obs_sumsq[t_idx] += np.square(contour_cpu).sum(axis=(0, 1))
            total_obs_sum[t_idx] += total_cpu.sum()
            total_obs_sumsq[t_idx] += np.square(total_cpu).sum()
            metrics.append({"t_idx": int(t_idx), "cycle": int(snapshot_cycles[t_idx]), "y0_start": int(y0_start), "y0_stop": int(stop), **block_metrics})

            del idx, sub_G, contour_flat, total_flat, contour, total
            y0_start = stop

        del G_t
        torch.cuda.empty_cache()

    observation_count = int(sample_count) * int(ny)
    entropy_contour_avg = contour_obs_sum / float(observation_count)
    if observation_count > 1:
        contour_var = (contour_obs_sumsq - np.square(contour_obs_sum) / float(observation_count)) / float(observation_count - 1)
        entropy_contour_std = np.sqrt(np.maximum(contour_var, 0.0))
        entropy_contour_sem = entropy_contour_std / np.sqrt(observation_count)
    else:
        entropy_contour_std = np.zeros_like(entropy_contour_avg)
        entropy_contour_sem = np.zeros_like(entropy_contour_avg)

    entropy_total_mean = total_obs_sum / float(observation_count)
    if observation_count > 1:
        total_var = (total_obs_sumsq - np.square(total_obs_sum) / float(observation_count)) / float(observation_count - 1)
        entropy_total_std = np.sqrt(np.maximum(total_var, 0.0))
        entropy_total_sem = entropy_total_std / np.sqrt(observation_count)
    else:
        entropy_total_std = np.zeros_like(entropy_total_mean)
        entropy_total_sem = np.zeros_like(entropy_total_mean)

    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return {
        "entropy_contour_avg": entropy_contour_avg,
        "entropy_contour_std": entropy_contour_std,
        "entropy_contour_sem": entropy_contour_sem,
        "entropy_contour_y0_sample_sum": contour_obs_sum,
        "entropy_contour_y0_sample_sumsq": contour_obs_sumsq,
        "entropy_total_mean": entropy_total_mean,
        "entropy_total_std": entropy_total_std,
        "entropy_total_sem": entropy_total_sem,
        "entropy_total_y0_sample_sum": total_obs_sum,
        "entropy_total_y0_sample_sumsq": total_obs_sumsq,
        "snapshot_cycles": np.asarray(snapshot_cycles, dtype=np.int64),
        "x": np.arange(int(nx), dtype=np.int64),
        "strip_dy": np.arange(ay, dtype=np.int64),
        "Ay": np.asarray(ay, dtype=np.int64),
        "sample_count": np.asarray(sample_count, dtype=np.int64),
        "y0_count": np.asarray(int(ny), dtype=np.int64),
        "observation_count": np.asarray(observation_count, dtype=np.int64),
        "selected_y0_chunk": int(selected_y0_chunk),
        "final_y0_chunk": int(min_chunk_used),
        "peak_cuda_gib": float(peak),
        "validation_metrics": metrics,
    }


def validate_saved_strip_product(path: Path, *, time_count: int, nx: int, ay: int, tol: float = 1e-8) -> None:
    with np.load(path, allow_pickle=False) as data:
        contour = np.asarray(data["entropy_contour_avg"], dtype=np.float64)
        total = np.asarray(data["entropy_total_mean"], dtype=np.float64)
        expected_shape = (int(time_count), int(nx), int(ay))
        if contour.shape != expected_shape:
            raise ValueError(f"{path}: expected contour shape {expected_shape}, got {contour.shape}")
        if total.shape != (int(time_count),):
            raise ValueError(f"{path}: expected total shape {(int(time_count),)}, got {total.shape}")
        if not np.all(np.isfinite(contour)) or not np.all(np.isfinite(total)):
            raise FloatingPointError(f"{path}: non-finite values in saved product.")
        if contour.size and np.nanmin(contour) < -tol:
            raise FloatingPointError(f"{path}: negative contour entry below tolerance.")
        mismatch = np.max(np.abs(contour.sum(axis=(1, 2)) - total)) if int(ay) > 0 else np.max(np.abs(total))
        if mismatch > tol:
            raise ValueError(f"{path}: total/contour mismatch {mismatch:g} exceeds {tol:g}.")
