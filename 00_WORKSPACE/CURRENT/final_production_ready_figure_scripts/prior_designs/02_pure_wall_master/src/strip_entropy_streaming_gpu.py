from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


HELPER_VERSION = "strip_entropy_streaming_gpu_a100_v2"


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


def rel_to_root(path: Path | str | None, root: Path | str) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return None


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
            "x": x,
            "y": y,
            "y_fit": np.full_like(y, np.nan),
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
        "x": x,
        "y": y,
        "y_fit": y_fit,
    }


def log_sin_chord(ay: np.ndarray | int | float, ny: int) -> np.ndarray:
    values = np.asarray(ay, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = values > 0
    out[mask] = np.log(np.sin(np.pi * values[mask] / float(ny)))
    return out


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


def _eigh_with_fallback(occ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    try:
        evals, vecs = torch.linalg.eigh(occ)
        return evals, vecs, 0
    except Exception as exc:
        if not _is_eigh_convergence_error(exc):
            raise
    if occ.ndim == 3 and int(occ.shape[0]) > 1:
        mid = int(occ.shape[0]) // 2
        evals_l, vecs_l, fall_l = _eigh_with_fallback(occ[:mid])
        evals_r, vecs_r, fall_r = _eigh_with_fallback(occ[mid:])
        return torch.cat([evals_l, evals_r], dim=0), torch.cat([vecs_l, vecs_r], dim=0), fall_l + fall_r
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
    nx = int(nx)
    ay = int(ay)
    if sub_G.ndim != 3:
        raise ValueError(f"Expected sub_G shape (B,M,M), got {tuple(sub_G.shape)}")
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
            "eigh_cpu_fallback_count": 0,
        }
    if validate and not torch.isfinite(sub_G).all():
        raise FloatingPointError("Non-finite restricted covariance entries.")
    herm_error = torch.max(torch.abs(sub_G - sub_G.conj().transpose(-2, -1))).detach()
    eye = torch.eye(expected, dtype=sub_G.dtype, device=sub_G.device)
    occ = 0.5 * (sub_G + eye.unsqueeze(0))
    occ = 0.5 * (occ + occ.conj().transpose(-2, -1))
    evals, vecs, fallback_count = _eigh_with_fallback(occ)
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
    return contour.to(torch.float64), total.to(torch.float64), {
        "max_hermiticity_error": float(herm_error.detach().cpu()),
        "min_occupation_eval": float(min_eval.detach().cpu()),
        "max_occupation_eval": float(max_eval.detach().cpu()),
        "max_total_mismatch": float(mismatch.detach().cpu()),
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


def estimate_eigh_bytes(*, sample_chunk: int, y0_chunk: int, nx: int, ay: int, multiplier: float = 8.0) -> int:
    matrix_size = 2 * int(nx) * int(ay)
    batch_count = int(sample_chunk) * int(y0_chunk)
    return int(batch_count * matrix_size * matrix_size * 16 * float(multiplier))


def autotune_contour_chunks_from_batch(
    G_batch: torch.Tensor,
    *,
    nx: int,
    ny: int,
    ay: int,
    eps: float = 1e-12,
    sample_chunk_candidates: Iterable[int] = (1, 2, 4, 8, 16),
    y0_chunk_candidates: Iterable[Any] = (1, 2, 4, 6, 8, 12, 16, 24, 32, "ny"),
    repeat: int = 2,
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
                idx = sub_G = contour = total = None
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
                    contour, total, _ = entropy_contour_batch_torch(sub_G, nx=nx, ay=ay, eps=eps, validate=True)
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
                    del idx, sub_G, contour, total
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
        raise RuntimeError(f"No safe contour chunks found for ay={ay}. Trials: {trials}")
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


class StripContourAccumulator:
    def __init__(self, *, nx: int, ny: int, ay_values: Iterable[int], cycles: Iterable[int], config: dict[str, Any]):
        self.nx = int(nx)
        self.ny = int(ny)
        self.ay_values = [int(ay) for ay in ay_values]
        self.cycles = [int(cycle) for cycle in cycles]
        self.cycle_to_index = {cycle: idx for idx, cycle in enumerate(self.cycles)}
        self.config = dict(config)
        self.sum = {ay: np.zeros((len(self.cycles), self.nx, ay), dtype=np.float64) for ay in self.ay_values}
        self.sumsq = {ay: np.zeros((len(self.cycles), self.nx, ay), dtype=np.float64) for ay in self.ay_values}
        self.total_sum = {ay: np.zeros((len(self.cycles),), dtype=np.float64) for ay in self.ay_values}
        self.total_sumsq = {ay: np.zeros((len(self.cycles),), dtype=np.float64) for ay in self.ay_values}
        self.count = {ay: np.zeros((len(self.cycles),), dtype=np.int64) for ay in self.ay_values}
        self.autotune = {}
        self.production_chunks = {}
        self.metrics = {}

    @staticmethod
    def _json_array_to_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        text = np.asarray(value).item()
        if text in ("", None):
            return {}
        return json.loads(str(text))

    @staticmethod
    def _restore_int_keys(payload: dict[str, Any]) -> dict[Any, Any]:
        restored = {}
        for key, value in payload.items():
            if isinstance(key, str):
                try:
                    restored[int(key)] = value
                    continue
                except ValueError:
                    pass
            restored[key] = value
        return restored

    def _validate_checkpoint_axes(self, data: np.lib.npyio.NpzFile, path: Path) -> None:
        nx = int(np.asarray(data["nx"]).item())
        ny = int(np.asarray(data["ny"]).item())
        ay_values = [int(value) for value in np.asarray(data["ay_values"]).tolist()]
        cycles = [int(value) for value in np.asarray(data["cycles"]).tolist()]
        if nx != self.nx:
            raise ValueError(f"Checkpoint {path} has nx={nx}, expected {self.nx}.")
        if ny != self.ny:
            raise ValueError(f"Checkpoint {path} has ny={ny}, expected {self.ny}.")
        if ay_values != self.ay_values:
            raise ValueError(f"Checkpoint {path} has ay_values={ay_values}, expected {self.ay_values}.")
        if cycles != self.cycles:
            raise ValueError(f"Checkpoint {path} has cycles={cycles}, expected {self.cycles}.")

    def _validate_checkpoint_arrays(
        self,
        data: np.lib.npyio.NpzFile,
        path: Path,
        *,
        require_contours: bool,
    ) -> None:
        for ay in self.ay_values:
            suffix = f"Ay{ay:03d}"
            required = [f"count_{suffix}", f"total_sum_{suffix}", f"total_sumsq_{suffix}"]
            if require_contours:
                required.extend([f"sum_{suffix}", f"sumsq_{suffix}"])
            missing = [key for key in required if key not in data.files]
            if missing:
                raise ValueError(
                    f"Checkpoint {path} is missing {missing}. "
                    "Use require_contours=False only for legacy reduced checkpoints."
                )
            expected_count_shape = (len(self.cycles),)
            for key in (f"count_{suffix}", f"total_sum_{suffix}", f"total_sumsq_{suffix}"):
                if tuple(data[key].shape) != expected_count_shape:
                    raise ValueError(
                        f"Checkpoint {path} key {key} has shape {data[key].shape}, "
                        f"expected {expected_count_shape}."
                    )
            if f"sum_{suffix}" in data.files:
                expected_contour_shape = (len(self.cycles), self.nx, ay)
                for key in (f"sum_{suffix}", f"sumsq_{suffix}"):
                    if tuple(data[key].shape) != expected_contour_shape:
                        raise ValueError(
                            f"Checkpoint {path} key {key} has shape {data[key].shape}, "
                            f"expected {expected_contour_shape}."
                        )

    @classmethod
    def load_checkpoint(cls, path: Path, *, require_contours: bool = True) -> "StripContourAccumulator":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            config = cls._json_array_to_dict(data["config_json"]) if "config_json" in data.files else {}
            acc = cls(
                nx=int(np.asarray(data["nx"]).item()),
                ny=int(np.asarray(data["ny"]).item()),
                ay_values=[int(value) for value in np.asarray(data["ay_values"]).tolist()],
                cycles=[int(value) for value in np.asarray(data["cycles"]).tolist()],
                config=config,
            )
            acc._validate_checkpoint_arrays(data, path, require_contours=require_contours)
            for ay in acc.ay_values:
                suffix = f"Ay{ay:03d}"
                acc.count[ay] = np.asarray(data[f"count_{suffix}"], dtype=np.int64).copy()
                acc.total_sum[ay] = np.asarray(data[f"total_sum_{suffix}"], dtype=np.float64).copy()
                acc.total_sumsq[ay] = np.asarray(data[f"total_sumsq_{suffix}"], dtype=np.float64).copy()
                if f"sum_{suffix}" in data.files and f"sumsq_{suffix}" in data.files:
                    acc.sum[ay] = np.asarray(data[f"sum_{suffix}"], dtype=np.float64).copy()
                    acc.sumsq[ay] = np.asarray(data[f"sumsq_{suffix}"], dtype=np.float64).copy()
            if "autotune_json" in data.files:
                acc.autotune = cls._restore_int_keys(cls._json_array_to_dict(data["autotune_json"]))
            if "production_chunks_json" in data.files:
                acc.production_chunks = cls._json_array_to_dict(data["production_chunks_json"])
            if "metrics_json" in data.files:
                acc.metrics = cls._json_array_to_dict(data["metrics_json"])
            return acc

    def merge_checkpoint(self, path: Path, *, require_contours: bool = True) -> None:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self._validate_checkpoint_axes(data, path)
            self._validate_checkpoint_arrays(data, path, require_contours=require_contours)
            for ay in self.ay_values:
                suffix = f"Ay{ay:03d}"
                self.count[ay] += np.asarray(data[f"count_{suffix}"], dtype=np.int64)
                self.total_sum[ay] += np.asarray(data[f"total_sum_{suffix}"], dtype=np.float64)
                self.total_sumsq[ay] += np.asarray(data[f"total_sumsq_{suffix}"], dtype=np.float64)
                if f"sum_{suffix}" in data.files and f"sumsq_{suffix}" in data.files:
                    self.sum[ay] += np.asarray(data[f"sum_{suffix}"], dtype=np.float64)
                    self.sumsq[ay] += np.asarray(data[f"sumsq_{suffix}"], dtype=np.float64)

            if "autotune_json" in data.files:
                incoming_autotune = self._restore_int_keys(self._json_array_to_dict(data["autotune_json"]))
                for key, value in incoming_autotune.items():
                    self.autotune.setdefault(key, value)
            if "production_chunks_json" in data.files:
                incoming_chunks = self._json_array_to_dict(data["production_chunks_json"])
                for key, value in incoming_chunks.items():
                    self.production_chunks.setdefault(key, value)
            if "metrics_json" in data.files:
                incoming_metrics = self._json_array_to_dict(data["metrics_json"])
                for key, values in incoming_metrics.items():
                    current = self.metrics.setdefault(key, {})
                    for metric_key, metric_value in values.items():
                        if metric_key == "max_total_mismatch":
                            current[metric_key] = max(float(current.get(metric_key, 0.0)), float(metric_value))
                        elif metric_key == "eigh_cpu_fallback_count":
                            current[metric_key] = int(current.get(metric_key, 0)) + int(metric_value)
                        else:
                            current[metric_key] = metric_value

    def update(
        self,
        *,
        cycle: int,
        G: torch.Tensor,
        eps: float = 1e-12,
        autotune: bool = True,
        sample_chunk_candidates: Iterable[int] = (1, 2, 4, 8, 16),
        y0_chunk_candidates: Iterable[Any] = (1, 2, 4, 6, 8, 12, 16, 24, 32, "ny"),
        autotune_repeat: int = 2,
        memory_safety_fraction: float = 0.85,
        eigh_memory_multiplier: float = 8.0,
    ) -> None:
        cycle = int(cycle)
        if cycle not in self.cycle_to_index:
            return
        t_idx = self.cycle_to_index[cycle]
        if G.ndim != 3:
            raise ValueError(f"Expected G with shape (B,N,N), got {tuple(G.shape)}")
        for ay in self.ay_values:
            if ay == 0:
                observation_count = int(G.shape[0]) * self.ny
                self.count[ay][t_idx] += observation_count
                continue
            if ay not in self.autotune:
                if autotune:
                    self.autotune[ay] = autotune_contour_chunks_from_batch(
                        G,
                        nx=self.nx,
                        ny=self.ny,
                        ay=ay,
                        eps=eps,
                        sample_chunk_candidates=sample_chunk_candidates,
                        y0_chunk_candidates=y0_chunk_candidates,
                        repeat=autotune_repeat,
                        memory_safety_fraction=memory_safety_fraction,
                        eigh_memory_multiplier=eigh_memory_multiplier,
                    )
                else:
                    self.autotune[ay] = {
                        "selected_sample_chunk": 1,
                        "selected_y0_chunk": 1,
                        "trials": [],
                        "ay": ay,
                    }
            sample_chunk = max(1, int(self.autotune[ay]["selected_sample_chunk"]))
            y0_chunk = max(1, int(self.autotune[ay]["selected_y0_chunk"]))
            final_sample_chunk = sample_chunk
            final_y0_chunk = y0_chunk
            oom_retries = 0
            sample_start = 0
            while sample_start < int(G.shape[0]):
                sample_stop = min(int(G.shape[0]), sample_start + final_sample_chunk)
                y0_start = 0
                while y0_start < self.ny:
                    try:
                        G_sample = G[sample_start:sample_stop]
                        y0_stop = min(self.ny, y0_start + final_y0_chunk)
                        idx = strip_mode_indices(
                            nx=self.nx,
                            ny=self.ny,
                            ay=ay,
                            y0_values=range(y0_start, y0_stop),
                            device=G.device,
                        )
                        sub_G = gather_restricted_covariance(G_sample, idx)
                        contour, total, metrics = entropy_contour_batch_torch(
                            sub_G,
                            nx=self.nx,
                            ay=ay,
                            eps=eps,
                            validate=True,
                        )
                        self.sum[ay][t_idx] += contour.sum(dim=0).detach().cpu().numpy()
                        self.sumsq[ay][t_idx] += (contour**2).sum(dim=0).detach().cpu().numpy()
                        self.total_sum[ay][t_idx] += float(total.sum().detach().cpu())
                        self.total_sumsq[ay][t_idx] += float((total**2).sum().detach().cpu())
                        self.count[ay][t_idx] += int(total.numel())
                        key = str(ay)
                        current = self.metrics.setdefault(key, {})
                        current["max_total_mismatch"] = max(
                            float(current.get("max_total_mismatch", 0.0)),
                            float(metrics.get("max_total_mismatch", 0.0)),
                        )
                        current["eigh_cpu_fallback_count"] = int(current.get("eigh_cpu_fallback_count", 0)) + int(
                            metrics.get("eigh_cpu_fallback_count", 0)
                        )
                        y0_start = y0_stop
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        oom_retries += 1
                        if G.device.type == "cuda":
                            torch.cuda.empty_cache()
                        if final_y0_chunk > 1:
                            final_y0_chunk = max(1, final_y0_chunk // 2)
                        elif final_sample_chunk > 1:
                            if y0_start != 0:
                                raise RuntimeError(
                                    "OOM after partially processing a sample chunk. "
                                    "Lower the selected sample_chunk before resuming to avoid duplicate accumulation."
                                ) from exc
                            final_sample_chunk = max(1, final_sample_chunk // 2)
                            sample_stop = min(int(G.shape[0]), sample_start + final_sample_chunk)
                        else:
                            raise
                sample_start = sample_stop
            self.production_chunks[str(ay)] = {
                "selected_sample_chunk": int(sample_chunk),
                "selected_y0_chunk": int(y0_chunk),
                "final_sample_chunk": int(final_sample_chunk),
                "final_y0_chunk": int(final_y0_chunk),
                "oom_retries": int(oom_retries),
            }

    def finalize_ay(self, ay: int) -> dict[str, np.ndarray]:
        ay = int(ay)
        count = self.count[ay].astype(np.float64)
        if np.any(count <= 0):
            missing = [int(self.cycles[idx]) for idx, value in enumerate(count) if value <= 0]
            raise RuntimeError(f"Missing observations for Ay={ay}, cycles={missing}")
        shape = (len(self.cycles), self.nx, ay)
        mean = np.zeros(shape, dtype=np.float64)
        std = np.zeros(shape, dtype=np.float64)
        sem = np.zeros(shape, dtype=np.float64)
        total_mean = np.zeros((len(self.cycles),), dtype=np.float64)
        total_std = np.zeros((len(self.cycles),), dtype=np.float64)
        total_sem = np.zeros((len(self.cycles),), dtype=np.float64)
        for idx, nobs in enumerate(count):
            if ay > 0:
                mean[idx] = self.sum[ay][idx] / nobs
                var = np.maximum(self.sumsq[ay][idx] / nobs - mean[idx] ** 2, 0.0)
                std[idx] = np.sqrt(var)
                sem[idx] = std[idx] / math.sqrt(nobs)
            total_mean[idx] = self.total_sum[ay][idx] / nobs
            total_var = max(float(self.total_sumsq[ay][idx] / nobs - total_mean[idx] ** 2), 0.0)
            total_std[idx] = math.sqrt(total_var)
            total_sem[idx] = total_std[idx] / math.sqrt(nobs)
        return {
            "entropy_contour_avg": mean,
            "entropy_contour_std": std,
            "entropy_contour_sem": sem,
            "entropy_contour_sum": self.sum[ay].copy(),
            "entropy_contour_sumsq": self.sumsq[ay].copy(),
            "entropy_total_mean": total_mean,
            "entropy_total_std": total_std,
            "entropy_total_sem": total_sem,
            "entropy_total_sum": self.total_sum[ay].copy(),
            "entropy_total_sumsq": self.total_sumsq[ay].copy(),
            "sample_counts_by_cycle": self.count[ay].copy(),
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "Ay": np.asarray(ay, dtype=np.int64),
            "x": np.arange(self.nx, dtype=np.int64),
            "strip_dy": np.arange(ay, dtype=np.int64),
        }

    def save_checkpoint(self, path: Path, *, include_contours: bool = True) -> None:
        payload: dict[str, Any] = {
            "nx": np.asarray(self.nx, dtype=np.int64),
            "ny": np.asarray(self.ny, dtype=np.int64),
            "ay_values": np.asarray(self.ay_values, dtype=np.int64),
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "config_json": np.asarray(json.dumps(self.config, sort_keys=True)),
            "autotune_json": np.asarray(json.dumps(self.autotune, sort_keys=True)),
            "production_chunks_json": np.asarray(json.dumps(self.production_chunks, sort_keys=True)),
            "metrics_json": np.asarray(json.dumps(self.metrics, sort_keys=True)),
            "checkpoint_format": np.asarray("full_contour" if include_contours else "reduced_totals"),
            "helper_version": np.asarray(HELPER_VERSION),
        }
        for ay in self.ay_values:
            if include_contours:
                payload[f"sum_Ay{ay:03d}"] = self.sum[ay]
                payload[f"sumsq_Ay{ay:03d}"] = self.sumsq[ay]
            payload[f"count_Ay{ay:03d}"] = self.count[ay]
            payload[f"total_sum_Ay{ay:03d}"] = self.total_sum[ay]
            payload[f"total_sumsq_Ay{ay:03d}"] = self.total_sumsq[ay]
        save_npz_atomic(path, **payload)


def write_contour_batch_files(
    *,
    accumulator: StripContourAccumulator,
    output_dir: Path,
    ay_batch_size: int,
    cycle_batch_size: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    ay_values = list(accumulator.ay_values)
    cycles = list(accumulator.cycles)
    batch_index = 0
    for cycle_start_idx in range(0, len(cycles), int(cycle_batch_size)):
        cycle_stop_idx = min(len(cycles), cycle_start_idx + int(cycle_batch_size))
        cycle_values = cycles[cycle_start_idx:cycle_stop_idx]
        cycle_dir = output_dir / f"cycles_{cycle_values[0]:03d}_{cycle_values[-1]:03d}"
        for ay_start_idx in range(0, len(ay_values), int(ay_batch_size)):
            ay_batch = ay_values[ay_start_idx:ay_start_idx + int(ay_batch_size)]
            arrays: dict[str, Any] = {
                "cycles": np.asarray(cycle_values, dtype=np.int64),
                "Ay_values": np.asarray(ay_batch, dtype=np.int64),
                "config_json": np.asarray(json.dumps(config, sort_keys=True)),
                "batch_json": np.asarray(json.dumps({
                    "cycle_start": int(cycle_values[0]),
                    "cycle_stop": int(cycle_values[-1]),
                    "Ay_start": int(ay_batch[0]),
                    "Ay_stop": int(ay_batch[-1]),
                    "helper_version": HELPER_VERSION,
                }, sort_keys=True)),
                "autotune_json": np.asarray(json.dumps({
                    str(ay): accumulator.autotune.get(ay, {}) for ay in ay_batch
                }, sort_keys=True)),
                "production_chunks_json": np.asarray(json.dumps({
                    str(ay): accumulator.production_chunks.get(str(ay), {}) for ay in ay_batch
                }, sort_keys=True)),
            }
            for ay in ay_batch:
                product = accumulator.finalize_ay(ay)
                for key, value in product.items():
                    if key in {"cycles", "Ay", "x", "strip_dy"}:
                        arrays[f"{key}_Ay{ay:03d}"] = value
                    elif value.ndim > 0 and value.shape[0] == len(cycles):
                        arrays[f"{key}_Ay{ay:03d}"] = value[cycle_start_idx:cycle_stop_idx]
                    else:
                        arrays[f"{key}_Ay{ay:03d}"] = value
            filename = f"batch_{batch_index:05d}_Ay{ay_batch[0]:03d}_Ay{ay_batch[-1]:03d}.npz"
            path = cycle_dir / filename
            save_npz_atomic(path, **arrays)
            records.append({
                "filename": str(path),
                "cycle_start": int(cycle_values[0]),
                "cycle_stop": int(cycle_values[-1]),
                "Ay_start": int(ay_batch[0]),
                "Ay_stop": int(ay_batch[-1]),
                "Ay_values": [int(ay) for ay in ay_batch],
                "cycles": [int(cycle) for cycle in cycle_values],
            })
            batch_index += 1
    return records


def entropy_curve_rows_from_accumulator(accumulator: StripContourAccumulator) -> list[dict[str, Any]]:
    rows = []
    for ay in accumulator.ay_values:
        product = accumulator.finalize_ay(ay)
        for idx, cycle in enumerate(accumulator.cycles):
            rows.append({
                "cycle": int(cycle),
                "Ay": int(ay),
                "log_sin_pi_Ay_over_Ny": float(log_sin_chord(ay, accumulator.ny)) if ay > 0 else np.nan,
                "entropy_mean": float(product["entropy_total_mean"][idx]),
                "entropy_std": float(product["entropy_total_std"][idx]),
                "entropy_sem": float(product["entropy_total_sem"][idx]),
                "observation_count": int(product["sample_counts_by_cycle"][idx]),
            })
    return rows


def fit_full_x_log_chord(
    rows,
    *,
    nx: int,
    ny: int,
    nshell: int,
    cycle: int,
    fit_ay_min: int,
    target_slope: float = 1.0 / 3.0,
) -> dict[str, Any]:
    import pandas as pd

    df = pd.DataFrame(rows)
    sub = df[(df["cycle"] == int(cycle)) & (df["Ay"] > 0)].sort_values("Ay")
    if sub.empty:
        raise ValueError(f"No entropy rows for cycle={cycle}.")
    fit_ay_max = int(ny) // 2
    fit_sub = sub[(sub["Ay"] >= int(fit_ay_min)) & (sub["Ay"] <= fit_ay_max)]
    if len(fit_sub) < 3:
        raise ValueError(f"Not enough fit points for cycle={cycle}, Ny={ny}: {len(fit_sub)}")
    x_fit = fit_sub["log_sin_pi_Ay_over_Ny"].to_numpy(dtype=np.float64)
    y_fit = fit_sub["entropy_mean"].to_numpy(dtype=np.float64)
    fit = fit_line_with_error(x_fit, y_fit)
    slope = float(fit["slope"])
    intercept = float(fit["intercept"])
    x_all = sub["log_sin_pi_Ay_over_Ny"].to_numpy(dtype=np.float64)
    return {
        "case_id": f"N{int(nx)}x{int(ny)}_nsh{int(nshell)}_perfect_correction",
        "Nx": int(nx),
        "Ny": int(ny),
        "nshell": int(nshell),
        "cycle": int(cycle),
        "x_interval_kind": "full_x",
        "Ay_fit_min": int(fit_ay_min),
        "Ay_fit_max": int(fit_ay_max),
        "n_fit_points": int(len(fit_sub)),
        "slope": slope,
        "target_slope": float(target_slope),
        "slope_err": float(fit["slope_err"]),
        "intercept": intercept,
        "r2": float(fit["r2"]),
        "percent_diff": float(100.0 * abs(slope - target_slope) / abs(target_slope)) if np.isfinite(slope) else np.nan,
        "fit_coordinate": "log_sin_pi_Ay_over_Ny",
        "fit_x_min": float(np.min(x_fit)),
        "fit_x_max": float(np.max(x_fit)),
        "plotted_x_min": float(np.min(x_all)),
        "plotted_x_max": float(np.max(x_all)),
        "plotted_Ay_min": int(sub["Ay"].min()),
        "plotted_Ay_max": int(sub["Ay"].max()),
    }
