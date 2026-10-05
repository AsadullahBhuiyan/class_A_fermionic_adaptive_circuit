from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def gauge_vector(nx: int, ny: int, axis: str) -> np.ndarray:
    """Large-gauge transition in the physical row ordering mu+2*x+2*Nx*y."""
    row = np.arange(2 * int(nx) * int(ny), dtype=np.int64)
    cell = row // 2
    coordinate = cell % nx if axis == "x" else cell // nx
    length = nx if axis == "x" else ny
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'.")
    return np.exp(2j * np.pi * coordinate / length)


def exact_target_frame(model: Any, *, singular_tolerance: float = 1e-10) -> np.ndarray:
    """Return the occupied target frame spanned by the full lower-band OW family."""
    lower = np.concatenate(
        [
            model.WF_Am.reshape(2 * model.Nx * model.Ny, -1, order="F"),
            model.WF_Bm.reshape(2 * model.Nx * model.Ny, -1, order="F"),
        ],
        axis=1,
    )
    left, singular_values, _ = np.linalg.svd(lower, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > float(singular_tolerance)))
    expected = model.Nx * model.Ny
    if rank != expected:
        raise RuntimeError(f"Exact target rank is {rank}, expected {expected}.")
    return np.ascontiguousarray(left[:, :rank])


def projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Frobenius projector distance without constructing either dense projector."""
    if left.shape[1] == right.shape[1]:
        u, _, vh = np.linalg.svd(left.conj().T @ right, full_matrices=False)
        alignment = vh.conj().T @ u.conj().T
        residual = np.linalg.norm(left - right @ alignment, ord="fro")
        return float(residual / np.sqrt(left.shape[1] + right.shape[1]))
    overlap_sq = float(np.linalg.norm(left.conj().T @ right, ord="fro") ** 2)
    value = max(0.0, left.shape[1] + right.shape[1] - 2.0 * overlap_sq)
    return float(np.sqrt(value / max(1, left.shape[1] + right.shape[1])))


@dataclass(frozen=True)
class LinkResult:
    phase: complex
    singular_values: np.ndarray
    minimum_singular_value: float
    log_abs_determinant: float
    valid: bool


def polar_link(
    left: np.ndarray,
    right: np.ndarray,
    *,
    singular_value_tolerance: float,
) -> LinkResult:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Neighboring frames must be two-dimensional with equal row counts.")
    if left.shape[1] != right.shape[1]:
        return LinkResult(0j, np.empty(0), 0.0, float("-inf"), False)
    overlap = left.conj().T @ right
    u, singular_values, vh = np.linalg.svd(overlap, full_matrices=False)
    minimum = float(np.min(singular_values)) if singular_values.size else 1.0
    positive = singular_values[singular_values > 0]
    log_abs = float(np.sum(np.log(positive))) if positive.size == singular_values.size else float("-inf")
    polar = u @ vh
    phase, _ = np.linalg.slogdet(polar)
    valid = bool(
        np.isfinite(minimum)
        and np.isfinite(log_abs)
        and np.isfinite(phase.real)
        and np.isfinite(phase.imag)
        and minimum > float(singular_value_tolerance)
    )
    return LinkResult(complex(phase), singular_values, minimum, log_abs, valid)


def analyze_frame_surface(
    frames: np.ndarray,
    *,
    nx: int,
    ny: int,
    singular_value_tolerance: float,
    expected_chern: float = 1.0,
    integer_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Analyze a rectangular object array of occupied frames on the twist torus."""
    frames = np.asarray(frames, dtype=object)
    if frames.ndim != 2:
        raise ValueError("frames must be a two-dimensional object array.")
    gx = gauge_vector(nx, ny, "x")[:, None]
    gy = gauge_vector(nx, ny, "y")[:, None]
    ranks = np.asarray([[frames[i, j].shape[1] for j in range(frames.shape[1])] for i in range(frames.shape[0])])
    constant_rank = bool(np.all(ranks == ranks.flat[0]))
    shape = frames.shape
    link_x = np.zeros(shape, dtype=np.complex128)
    link_y = np.zeros(shape, dtype=np.complex128)
    minimum_x = np.zeros(shape, dtype=np.float64)
    minimum_y = np.zeros(shape, dtype=np.float64)
    logdet_x = np.full(shape, -np.inf, dtype=np.float64)
    logdet_y = np.full(shape, -np.inf, dtype=np.float64)
    valid_x = np.zeros(shape, dtype=bool)
    valid_y = np.zeros(shape, dtype=bool)
    singular_x = np.empty(shape, dtype=object)
    singular_y = np.empty(shape, dtype=object)

    for i in range(shape[0]):
        for j in range(shape[1]):
            next_x = frames[(i + 1) % shape[0], j]
            next_y = frames[i, (j + 1) % shape[1]]
            if i == shape[0] - 1:
                next_x = gx * next_x
            if j == shape[1] - 1:
                next_y = gy * next_y
            result_x = polar_link(
                frames[i, j], next_x, singular_value_tolerance=singular_value_tolerance
            )
            result_y = polar_link(
                frames[i, j], next_y, singular_value_tolerance=singular_value_tolerance
            )
            link_x[i, j], link_y[i, j] = result_x.phase, result_y.phase
            minimum_x[i, j], minimum_y[i, j] = (
                result_x.minimum_singular_value,
                result_y.minimum_singular_value,
            )
            logdet_x[i, j], logdet_y[i, j] = (
                result_x.log_abs_determinant,
                result_y.log_abs_determinant,
            )
            valid_x[i, j], valid_y[i, j] = result_x.valid, result_y.valid
            singular_x[i, j], singular_y[i, j] = (
                result_x.singular_values,
                result_y.singular_values,
            )

    all_links_valid = bool(np.all(valid_x) and np.all(valid_y))
    plaquette = np.full(shape, np.nan, dtype=np.float64)
    chern = float("nan")
    if constant_rank and all_links_valid:
        loop = (
            link_x
            * np.roll(link_y, -1, axis=0)
            * np.conj(np.roll(link_x, -1, axis=1))
            * np.conj(link_y)
        )
        # The occupied-frame convention is complex-conjugate to the lower-band
        # Bloch convention.  This orientation matches the repository's positive
        # real-space Chern marker for the alpha=1 target.
        plaquette = -np.angle(loop)
        chern = float(np.sum(plaquette) / (2.0 * np.pi))

    if not constant_rank:
        classification = "undefined_rank_mismatch"
    elif not all_links_valid:
        classification = "undefined_singular_links"
    elif not np.isfinite(chern):
        classification = "numerical_failure"
    elif abs(chern - round(chern)) > float(integer_tolerance):
        classification = "numerical_failure"
    elif abs(chern - float(expected_chern)) <= float(integer_tolerance):
        classification = "defined_C1"
    else:
        classification = "defined_wrong_integer"

    return {
        "classification": classification,
        "chern": chern,
        "ranks": ranks,
        "constant_rank": constant_rank,
        "link_x": link_x,
        "link_y": link_y,
        "minimum_singular_x": minimum_x,
        "minimum_singular_y": minimum_y,
        "log_abs_determinant_x": logdet_x,
        "log_abs_determinant_y": logdet_y,
        "valid_x": valid_x,
        "valid_y": valid_y,
        "singular_values_x": singular_x,
        "singular_values_y": singular_y,
        "plaquette_phase": plaquette,
        "valid_link_fraction": float((np.count_nonzero(valid_x) + np.count_nonzero(valid_y)) / (2 * valid_x.size)),
    }
