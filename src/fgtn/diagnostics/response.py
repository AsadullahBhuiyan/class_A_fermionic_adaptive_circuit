from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .common import RegionMasks, local_charge_map


def unit_cell_reset(
    G: np.ndarray,
    *,
    nx: int,
    ny: int,
    x: int,
    y: int,
    occupied: bool,
) -> np.ndarray:
    """Apply G -> Q G Q +/- P on both onsite orbitals of one unit cell."""
    G = np.asarray(G, dtype=np.complex128)
    nlayer = 2 * int(nx) * int(ny)
    if G.shape != (nlayer, nlayer):
        raise ValueError(f"Expected covariance shape {(nlayer, nlayer)}, got {G.shape}.")
    x, y = int(x) % int(nx), int(y) % int(ny)
    indices = np.asarray([2 * x + 2 * int(nx) * y, 1 + 2 * x + 2 * int(nx) * y], dtype=np.int64)
    out = np.array(0.5 * (G + G.conj().T), copy=True)
    out[indices, :] = 0.0
    out[:, indices] = 0.0
    out[np.ix_(indices, indices)] = (1.0 if occupied else -1.0) * np.eye(2)
    return out


def validate_covariance(G: np.ndarray, *, tolerance: float = 1e-10) -> dict[str, float]:
    G = np.asarray(G, dtype=np.complex128)
    hermiticity = float(np.linalg.norm(G - G.conj().T, ord="fro"))
    eigenvalues = np.linalg.eigvalsh(0.5 * (G + G.conj().T))
    bound_violation = float(max(0.0, -1.0 - eigenvalues[0], eigenvalues[-1] - 1.0))
    if hermiticity > tolerance or bound_violation > tolerance:
        raise FloatingPointError(
            f"Invalid covariance: hermiticity={hermiticity:.3e}, bound_violation={bound_violation:.3e}."
        )
    return {
        "hermiticity_residual": hermiticity,
        "spectral_bound_violation": bound_violation,
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
    }


class ChargeMapRecorder:
    def __init__(self, *, samples: int, cycles: int, nx: int, ny: int) -> None:
        self.samples = int(samples)
        self.cycles = int(cycles)
        self.nx = int(nx)
        self.ny = int(ny)
        self.charge = np.full(
            (self.samples, self.cycles + 1, self.nx, self.ny), np.nan, dtype=np.float64
        )

    def __call__(
        self,
        *,
        cycle: int,
        G: np.ndarray,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        if int(batch_count) != 1:
            raise ValueError("The CPU charge-map recorder expects serial trajectories.")
        sample = int(batch_start)
        matrix = np.asarray(G, dtype=np.complex128)
        if matrix.ndim == 3:
            matrix = matrix[0]
        self.charge[sample, int(cycle)] = local_charge_map(matrix, nx=self.nx, ny=self.ny)


class SingleTrajectoryChargeRecorder(ChargeMapRecorder):
    def __call__(self, *, cycle: int, G: np.ndarray, **kwargs: Any) -> None:
        matrix = np.asarray(G, dtype=np.complex128)
        if matrix.ndim == 3:
            matrix = matrix[0]
        self.charge[0, int(cycle)] = local_charge_map(matrix, nx=self.nx, ny=self.ny)


def _wall_masks(
    *,
    nx: int,
    ny: int,
    wall_x: Iterable[int],
    width: int,
    active: np.ndarray,
) -> np.ndarray:
    result = []
    for wall in wall_x:
        mask = np.zeros((nx, ny), dtype=bool)
        for x in range(nx):
            if abs(x - int(wall)) < int(width):
                mask[x, :] = active[x, :]
        result.append(mask)
    return np.stack(result, axis=0)


def analyze_paired_response(
    delta_charge: np.ndarray,
    *,
    regions: RegionMasks,
    y0: int,
    wall_x: Iterable[int] | None = None,
    fit_max_cycle: int | None = None,
    norm_fraction_cutoff: float = 0.2,
) -> dict[str, Any]:
    delta = np.asarray(delta_charge, dtype=np.float64)
    if delta.ndim != 4:
        raise ValueError("delta_charge must have shape (samples,time,nx,ny).")
    samples, times, nx, ny = delta.shape
    walls = tuple(int(value) for value in (regions.wall_x if wall_x is None else wall_x))
    if not walls:
        walls = (nx // 2,)
    masks = _wall_masks(
        nx=nx,
        ny=ny,
        wall_x=walls,
        width=regions.interface_width,
        active=regions.active,
    )
    profiles = np.zeros((samples, times, len(walls), ny), dtype=np.float64)
    for wall_index, mask in enumerate(masks):
        profiles[:, :, wall_index] = np.sum(delta * mask[None, None, :, :], axis=2)
    displacement = ((np.arange(ny) - int(y0) + ny // 2) % ny) - ny // 2
    response_norm = np.sum(np.abs(profiles), axis=-1)
    signed_first_moment = np.divide(
        np.sum(profiles * displacement[None, None, None, :], axis=-1),
        response_norm,
        out=np.full(response_norm.shape, np.nan, dtype=np.float64),
        where=response_norm > 1e-14,
    )
    absolute_center = np.divide(
        np.sum(np.abs(profiles) * displacement[None, None, None, :], axis=-1),
        response_norm,
        out=np.full(response_norm.shape, np.nan, dtype=np.float64),
        where=response_norm > 1e-14,
    )
    centered = displacement[None, None, None, :] - absolute_center[..., None]
    width = np.sqrt(
        np.divide(
            np.sum(np.abs(profiles) * centered * centered, axis=-1),
            response_norm,
            out=np.full(response_norm.shape, np.nan, dtype=np.float64),
            where=response_norm > 1e-14,
        )
    )
    total_norm = np.sum(np.abs(delta) * regions.active[None, None, :, :], axis=(2, 3))
    wall_weight = np.divide(
        np.sum(np.abs(delta)[:, :, None, :, :] * masks[None, None, :, :, :], axis=(3, 4)),
        total_norm[:, :, None],
        out=np.full(response_norm.shape, np.nan, dtype=np.float64),
        where=total_norm[:, :, None] > 1e-14,
    )

    max_cycle = min(times - 1, max(2, ny // 4)) if fit_max_cycle is None else min(times - 1, int(fit_max_cycle))
    velocity = np.full((samples, len(walls)), np.nan, dtype=np.float64)
    fit_point_count = np.zeros((samples, len(walls)), dtype=np.int64)
    relaxed_fit = np.zeros((samples, len(walls)), dtype=bool)
    for sample in range(samples):
        for wall_index in range(len(walls)):
            reference = response_norm[sample, 0, wall_index]
            base_valid = (
                (np.arange(times) <= max_cycle)
                & np.isfinite(signed_first_moment[sample, :, wall_index])
                & (response_norm[sample, :, wall_index] > 1e-14)
            )
            base_valid[0] = False
            valid = base_valid.copy()
            if reference > 0.0:
                valid &= response_norm[sample, :, wall_index] >= float(norm_fraction_cutoff) * reference
            if np.count_nonzero(valid) < 2 and np.count_nonzero(base_valid) >= 2:
                valid = base_valid
                relaxed_fit[sample, wall_index] = True
            fit_point_count[sample, wall_index] = int(np.count_nonzero(valid))
            if fit_point_count[sample, wall_index] >= 2:
                velocity[sample, wall_index] = np.polyfit(
                    np.flatnonzero(valid), signed_first_moment[sample, valid, wall_index], 1
                )[0]

    finite_velocity_count = np.sum(np.isfinite(velocity), axis=0)
    velocity_mean = np.divide(
        np.nansum(velocity, axis=0),
        finite_velocity_count,
        out=np.full((len(walls),), np.nan, dtype=np.float64),
        where=finite_velocity_count > 0,
    )
    velocity_sem = np.full((len(walls),), np.nan, dtype=np.float64)
    for wall_index in range(len(walls)):
        finite_values = velocity[np.isfinite(velocity[:, wall_index]), wall_index]
        if finite_values.size > 1:
            velocity_sem[wall_index] = np.std(finite_values, ddof=1) / np.sqrt(finite_values.size)

    return {
        "wall_x": np.asarray(walls, dtype=np.int64),
        "wall_masks": masks,
        "y0": np.asarray(int(y0), dtype=np.int64),
        "periodic_displacement": displacement,
        "wall_profiles": profiles,
        "response_norm": response_norm,
        "signed_first_moment": signed_first_moment,
        "absolute_center": absolute_center,
        "response_width": width,
        "wall_weight": wall_weight,
        "velocity_per_sample": velocity,
        "velocity_mean": velocity_mean,
        "velocity_sem": velocity_sem,
        "velocity_fit_point_count": fit_point_count,
        "velocity_relaxed_fit": relaxed_fit,
        "fit_max_cycle": np.asarray(max_cycle, dtype=np.int64),
    }
