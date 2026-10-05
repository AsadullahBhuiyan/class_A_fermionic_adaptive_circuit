"""Shared numerical helpers for the Kraus boundary-contour analysis."""

from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh


NX = 20
LEFT_WALL = 5
RIGHT_WALL = 15
WALL_RADIUS = 2
SOFT_MODE_COUNT = 16
LEADING_LEVEL_COUNT = 64
OCCUPATION_TOL = 1.0e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_completion_pair(result_path: Path) -> dict[str, Any]:
    completion_path = result_path.with_suffix(".complete.json")
    if not completion_path.is_file():
        raise FileNotFoundError(f"missing completion JSON for {result_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("result_filename") != result_path.name:
        raise RuntimeError(f"completion filename mismatch for {result_path}")
    if int(completion.get("result_bytes", -1)) != result_path.stat().st_size:
        raise RuntimeError(f"completion byte-count mismatch for {result_path}")
    actual = sha256_file(result_path)
    if completion.get("result_sha256") != actual:
        raise RuntimeError(f"completion SHA-256 mismatch for {result_path}")
    return completion


def wall_cells(center: int, radius: int = WALL_RADIUS) -> np.ndarray:
    return np.asarray(
        [(int(center) + offset) % NX for offset in range(-radius, radius + 1)],
        dtype=np.int64,
    )


LEFT_CELLS = wall_cells(LEFT_WALL)
RIGHT_CELLS = wall_cells(RIGHT_WALL)
WALL_CELLS = np.unique(np.concatenate((LEFT_CELLS, RIGHT_CELLS)))


def orbital_to_x_profile(values: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    """Sum canonical ``(y,x,orbital)`` one-body weights over y and orbital."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 2 * int(nx) * int(ny):
        raise ValueError("orbital vector has the wrong physical dimension")
    return values.reshape(*values.shape[:-1], int(ny), int(nx), 2).sum(
        axis=(-3, -1)
    )


def spectral_snapshot(
    centered_covariance: np.ndarray,
    *,
    nx: int,
    ny: int,
    soft_count: int = SOFT_MODE_COUNT,
) -> dict[str, np.ndarray | float | int]:
    """Diagonalize one centered covariance and construct exact x contours."""

    G = np.asarray(centered_covariance, dtype=np.complex128)
    expected = 2 * int(nx) * int(ny)
    if G.shape != (expected, expected):
        raise ValueError(f"expected covariance {(expected, expected)}, got {G.shape}")
    hermiticity = float(np.max(np.abs(G - G.conj().T)))
    if hermiticity > 1.0e-8:
        raise FloatingPointError(f"covariance Hermiticity residual {hermiticity:.3e}")
    C = 0.5 * (0.5 * (G + G.conj().T) + np.eye(expected))
    occupations, vectors = eigh(
        C, check_finite=False, overwrite_a=True, driver="evd"
    )
    bound_residual = float(
        max(0.0, -float(occupations.min()), float(occupations.max()) - 1.0)
    )
    if bound_residual > OCCUPATION_TOL:
        raise FloatingPointError(
            f"occupation bound residual {bound_residual:.3e} exceeds tolerance"
        )
    clipped = np.clip(occupations, 0.0, 1.0)
    logmax = np.log(np.maximum(clipped, 1.0 - clipped))
    probabilities = np.abs(vectors) ** 2
    spectral_orbital = probabilities @ logmax
    spectral_x = orbital_to_x_profile(spectral_orbital, nx=nx, ny=ny)

    cap = (clipped <= 1.0e-15) | (clipped >= 1.0 - 1.0e-15)
    flip_costs = np.full(clipped.shape, np.inf, dtype=np.float64)
    finite = ~cap
    flip_costs[finite] = np.abs(
        np.log(clipped[finite]) - np.log1p(-clipped[finite])
    )
    soft_indices = np.argsort(flip_costs, kind="stable")[: int(soft_count)]
    soft_probabilities = probabilities[:, soft_indices].T
    soft_profiles = orbital_to_x_profile(soft_probabilities, nx=nx, ny=ny)
    profile_closure = float(np.max(np.abs(soft_profiles.sum(axis=1) - 1.0)))
    if profile_closure > 1.0e-10:
        raise FloatingPointError(f"soft-mode profile closure {profile_closure:.3e}")
    left_weights = soft_profiles[:, LEFT_CELLS].sum(axis=1)
    right_weights = soft_profiles[:, RIGHT_CELLS].sum(axis=1)
    return {
        "occupations": occupations,
        "spectral_x": spectral_x,
        "spectral_total": float(logmax.sum()),
        "soft_indices": soft_indices.astype(np.int64),
        "soft_occupations": occupations[soft_indices],
        "soft_costs": flip_costs[soft_indices],
        "soft_profiles": soft_profiles,
        "soft_wall_weights": np.stack((left_weights, right_weights), axis=-1),
        "cap_count": int(np.count_nonzero(cap)),
        "hermiticity_residual": hermiticity,
        "occupation_bound_residual": bound_residual,
    }


def leading_subset_sums(
    costs: np.ndarray, count: int = LEADING_LEVEL_COUNT
) -> tuple[np.ndarray, np.ndarray]:
    """Return the smallest distinct subset sums and their bit masks."""

    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 1 or not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("costs must be a finite nonnegative vector")
    levels: list[tuple[float, int]] = [(0.0, 0)]
    for mode, cost in enumerate(costs):
        additions = [(value + float(cost), mask | (1 << mode)) for value, mask in levels]
        levels = heapq.nsmallest(int(count), levels + additions, key=lambda item: item[0])
    values = np.asarray([item[0] for item in levels], dtype=np.float64)
    masks = np.asarray([item[1] for item in levels], dtype=np.uint64)
    order = np.argsort(values, kind="stable")
    return values[order], masks[order]


def mask_gap_contour(
    mask: int, costs: np.ndarray, profiles: np.ndarray
) -> np.ndarray:
    selected = [mode for mode in range(len(costs)) if int(mask) & (1 << mode)]
    if not selected:
        return np.zeros(profiles.shape[-1], dtype=np.float64)
    selected_array = np.asarray(selected, dtype=np.int64)
    return np.einsum(
        "m,mx->x", costs[selected_array], profiles[selected_array], optimize=True
    )


def fit_slope(cycles: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    cycles = np.asarray(cycles, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = (cycles >= float(lo)) & (cycles <= float(hi)) & np.isfinite(values)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.polyfit(cycles[mask], values[mask], 1)[0])
