"""Analysis helpers for CPU trajectory CFT extraction.

The helpers here are intentionally lightweight: they operate on NumPy arrays,
CSV rows, and JSON manifests produced by ``run_cpu_cft_sweep.py``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def finite_size_linear_fit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray | None = None) -> dict[str, float]:
    """Fit ``y = intercept + slope * x`` and return slope/intercept diagnostics."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=np.float64)
        mask &= np.isfinite(yerr) & (yerr > 0.0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"intercept": np.nan, "slope": np.nan, "intercept_se": np.nan, "slope_se": np.nan, "n": int(x.size)}
    if x.size == 2:
        coeff = np.polyfit(x, y, deg=1)
        cov = np.full((2, 2), np.nan, dtype=np.float64)
    elif yerr is None:
        coeff, cov = np.polyfit(x, y, deg=1, cov=True)
    else:
        coeff, cov = np.polyfit(x, y, deg=1, w=1.0 / yerr[mask], cov=True)
    slope, intercept = coeff
    slope_se = float(np.sqrt(cov[0, 0])) if cov.shape == (2, 2) else np.nan
    intercept_se = float(np.sqrt(cov[1, 1])) if cov.shape == (2, 2) else np.nan
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "intercept_se": intercept_se,
        "slope_se": slope_se,
        "n": int(x.size),
    }


def bootstrap_stat(
    values: Iterable[float],
    stat="mean",
    n_boot: int = 512,
    seed: int = 1234,
) -> dict[str, float]:
    """Bootstrap a one-dimensional mean or median."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"value": np.nan, "se": np.nan, "n": 0}
    center = float(np.mean(arr) if stat == "mean" else np.median(arr))
    if arr.size == 1 or n_boot <= 1:
        return {"value": center, "se": 0.0, "n": int(arr.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    samples = arr[idx]
    boot = np.mean(samples, axis=1) if stat == "mean" else np.median(samples, axis=1)
    return {"value": center, "se": float(np.std(boot, ddof=1)), "n": int(arr.size)}


def choi_finite_rapidities(eigenvalues, cycle: int, endpoint_tol: float = 1e-8) -> dict[str, np.ndarray | int]:
    """Classify Choi covariance eigenvalues and return finite rapidity magnitudes."""
    vals = np.asarray(eigenvalues, dtype=np.float64)
    endpoint = np.isfinite(vals) & (np.abs(np.abs(vals) - 1.0) < float(endpoint_tol))
    finite_mask = np.isfinite(vals) & (~endpoint) & (np.abs(vals) < 1.0)
    clipped = np.clip(vals[finite_mask], -1.0 + float(endpoint_tol), 1.0 - float(endpoint_tol))
    rapidities = np.sort(np.abs(np.arctanh(clipped) / float(cycle)))
    return {
        "rapidities": rapidities,
        "endpoint_plus": int(np.count_nonzero(endpoint & (vals >= 0.0))),
        "endpoint_minus": int(np.count_nonzero(endpoint & (vals < 0.0))),
        "finite_count": int(np.count_nonzero(finite_mask)),
    }


def min_abs_gap(values, floor: float = 0.0) -> float:
    """Smallest finite absolute value above ``floor``."""
    arr = np.asarray(values, dtype=np.float64)
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > float(floor)]
    if arr.size == 0:
        return np.nan
    return float(np.min(arr))


def additive_fock_gap(values, rank: int = 1, floor: float = 0.0) -> float:
    """Sum the ``rank`` smallest finite positive one-body gaps."""
    rank = int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive")
    arr = np.asarray(values, dtype=np.float64)
    arr = np.sort(np.abs(arr[np.isfinite(arr)]))
    arr = arr[arr > float(floor)]
    if arr.size < rank:
        return np.nan
    return float(np.sum(arr[:rank]))


def load_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv_dicts(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, payload: dict) -> None:
    def _sanitize(value):
        if isinstance(value, dict):
            return {key: _sanitize(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_sanitize(val) for val in value]
        if isinstance(value, tuple):
            return [_sanitize(val) for val in value]
        if isinstance(value, np.generic):
            return _sanitize(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize(payload), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def fit_campaign(scalars: list[dict[str, object]], alpha: float = 1.0) -> dict[str, object]:
    """Fit ``c_eff`` and additive tangent/Choi Fock-sector gaps."""
    rows = [row for row in scalars if np.isfinite(float(row["L"]))]
    L = np.asarray([float(row["L"]) for row in rows], dtype=np.float64)
    inv_l2 = 1.0 / (L * L)
    f0 = np.asarray([float(row["f0"]) for row in rows], dtype=np.float64)
    ceff_fit = finite_size_linear_fit(inv_l2, f0)
    out = {
        "ceff": float(-6.0 * ceff_fit["slope"] / np.pi),
        "ceff_fit": ceff_fit,
        "alpha": float(alpha),
    }
    labels = {
        "tangent": "x_tangent_fock",
        "choi": "x_choi_fock",
    }
    for label, col in (("tangent", "tangent_fock_gap"), ("choi", "choi_fock_gap")):
        gap = np.asarray([float(row[col]) for row in rows], dtype=np.float64)
        y = gap / (float(alpha) * L)
        fit = finite_size_linear_fit(inv_l2, y)
        out[labels[label]] = float(fit["slope"] / (2.0 * np.pi))
        out[f"{labels[label]}_fit"] = fit
    return out
