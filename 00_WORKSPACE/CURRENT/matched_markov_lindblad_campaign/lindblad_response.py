"""Local density-kick response for the matched continuous Lindblad arm.

The public estimator uses the physical PRR correlation convention

    G_ij = Tr(rho c_i^dagger c_j).

For a local phase kick at ``(source_x, source_y)`` its exact central-difference
initial condition is ``i*sinc(epsilon)*[P_s,G]``.  Without number dephasing the
homogeneous propagator is a one-particle contraction and the response remains
rank four.  With number dephasing that factorization is lost; the implementation
then evolves all independent y-momentum-transfer sectors ``X_q(k)=G(k,k-q)``.
Only their diagonal sums are retained, so no dense response history is stored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DensityKickResponse:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def periodic_displacements(ny: int) -> np.ndarray:
    values = np.arange(int(ny), dtype=np.float64)
    return ((values + float(ny) / 2.0) % float(ny)) - float(ny) / 2.0


def fit_positive_lobe_velocity(
    times: np.ndarray,
    centers: np.ndarray,
    norms: np.ndarray,
    *,
    fit_time_min: float,
    fit_time_max: float,
) -> tuple[float, float]:
    """OLS slope and deterministic fit quality for one source/wall profile."""

    times = np.asarray(times, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    norms = np.asarray(norms, dtype=np.float64)
    threshold = max(1e-8 * float(np.max(norms, initial=0.0)), 1e-15)
    selected = (
        (times >= float(fit_time_min))
        & (times <= float(fit_time_max))
        & np.isfinite(centers)
        & (norms > threshold)
    )
    if np.count_nonzero(selected) < 3:
        return math.nan, math.nan
    x, y = times[selected], centers[selected]
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / denominator if denominator > 0.0 else 1.0
    return float(slope), float(r2)


def local_kick_q_sector(
    baseline_q0: np.ndarray,
    *,
    q_index: int,
    source_x: Iterable[int],
    source_y: Iterable[int],
    nx: int,
    epsilon: float,
) -> np.ndarray:
    """Return ``i*sinc(epsilon)[P_s,G]`` in one q sector.

    The leading axis enumerates paired ``(source_x,source_y)`` inputs.  ``P_s``
    contains both physical orbitals of the selected unit cell.
    """

    blocks = np.asarray(baseline_q0, dtype=np.complex128)
    ny, dimension, dimension_two = blocks.shape
    if dimension != dimension_two or dimension != 2 * int(nx):
        raise ValueError("baseline_q0 must have shape (Ny,2*Nx,2*Nx)")
    xs = np.asarray(list(source_x), dtype=np.int64).reshape(-1)
    ys = np.asarray(list(source_y), dtype=np.int64).reshape(-1)
    if xs.size == 0 or xs.shape != ys.shape:
        raise ValueError("source_x and source_y must be nonempty paired arrays")
    if np.any((xs < 0) | (xs >= int(nx))) or np.any((ys < 0) | (ys >= ny)):
        raise ValueError("source coordinates lie outside the lattice")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a positive finite scalar")

    projectors = np.zeros((xs.size, dimension, dimension), dtype=np.complex128)
    for source_index, x in enumerate(xs):
        indices = 2 * int(x) + np.arange(2)
        projectors[source_index, indices, indices] = 1.0
    q_index = int(q_index) % ny
    shifted = blocks[(np.arange(ny) - q_index) % ny]
    phase = np.exp(-2j * np.pi * q_index * ys / float(ny))
    sinc = float(np.sinc(epsilon / np.pi))
    commutator = (
        projectors[:, None] @ shifted[None]
        - blocks[None] @ projectors[:, None]
    )
    return (1j * sinc / float(ny)) * phase[:, None, None, None] * commutator


def density_from_q_diagonals(q_diagonals: np.ndarray, *, nx: int) -> np.ndarray:
    """Reconstruct ``chi(source,time,x,y)`` from ``sum_k diag X_q(k)``."""

    values = np.asarray(q_diagonals, dtype=np.complex128)
    if values.ndim != 4:
        raise ValueError("q_diagonals must have shape (source,q,time,2*Nx)")
    source_count, ny, time_count, dimension = values.shape
    if dimension != 2 * int(nx):
        raise ValueError("the q diagonal orbital dimension must equal 2*Nx")
    density_y = np.fft.ifft(values, axis=1)
    scale = max(float(np.max(np.abs(density_y), initial=0.0)), 1e-300)
    imaginary_residual = float(np.max(np.abs(density_y.imag), initial=0.0)) / scale
    if imaginary_residual > 2e-9:
        raise FloatingPointError(
            "q-sector reconstruction is not real; relative imaginary residual="
            f"{imaginary_residual:.3e}"
        )
    density = density_y.real.reshape(
        source_count, ny, time_count, int(nx), 2
    ).sum(axis=-1)
    return np.transpose(density, (0, 2, 3, 1))


def _rank_four_no_dephasing_response(
    generator: Any,
    baseline_q0: np.ndarray,
    *,
    source_x: np.ndarray,
    source_y: np.ndarray,
    epsilon: float,
    times: np.ndarray,
) -> np.ndarray:
    """Exact no-dephasing response using propagated rank-four factors."""

    ny, dimension = int(generator.ny), int(generator.block_dimension)
    operators = generator.momentum_frame_operators
    damping = 0.5 * (
        operators["A_minus"]
        + operators["B_minus"]
        + operators["A_plus"]
        + operators["B_plus"]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(damping)
    eigenvectors_dagger = np.swapaxes(eigenvectors.conj(), -1, -2)

    source_count = source_x.size
    source_real = np.zeros(
        (source_count, ny, dimension, 2), dtype=np.complex128
    )
    for source_index, (x, y) in enumerate(zip(source_x, source_y, strict=True)):
        for orbital in (0, 1):
            source_real[source_index, int(y), 2 * int(x) + orbital, orbital] = 1.0
    source_k = np.fft.fft(source_real, axis=1, norm="ortho")
    correlation_source_k = baseline_q0[None] @ source_k
    sinc = float(np.sinc(float(epsilon) / np.pi))
    response = np.empty(
        (source_count, times.size, int(generator.nx), ny), dtype=np.float64
    )
    for time_index, time_value in enumerate(times):
        decay = np.exp(-eigenvalues * float(time_value))
        propagator = (eigenvectors * decay[:, None, :]) @ eigenvectors_dagger
        evolved_source_k = propagator[None] @ source_k
        evolved_correlation_source_k = propagator[None] @ correlation_source_k
        evolved_source = np.fft.ifft(evolved_source_k, axis=1, norm="ortho")
        evolved_correlation_source = np.fft.ifft(
            evolved_correlation_source_k, axis=1, norm="ortho"
        )
        diagonal = 2.0 * sinc * np.imag(
            np.sum(evolved_correlation_source * evolved_source.conj(), axis=-1)
        )
        cell_density = diagonal.reshape(
            source_count, ny, int(generator.nx), 2
        ).sum(axis=-1)
        response[:, time_index] = np.transpose(cell_density, (0, 2, 1))
    return response


def _q_sector_rk4_response(
    generator: Any,
    baseline_q0: np.ndarray,
    *,
    source_x: np.ndarray,
    source_y: np.ndarray,
    epsilon: float,
    times: np.ndarray,
    integration_dt: float,
    include_number_dephasing: bool,
    q_batch_size: int,
) -> np.ndarray:
    """Stream all q sectors and retain only diagonal density information."""

    integration_dt = float(integration_dt)
    steps = np.rint(times / integration_dt).astype(np.int64)
    if not np.allclose(
        steps * integration_dt, times, rtol=1e-11, atol=1e-13
    ):
        raise ValueError("response times must align with the Lindblad RK4 step")
    save_lookup = {int(step): index for index, step in enumerate(steps)}
    ny, dimension = int(generator.ny), int(generator.block_dimension)
    q_diagonal = np.empty(
        (source_x.size, ny, times.size, dimension), dtype=np.complex128
    )

    q_batch_size = int(q_batch_size)
    if q_batch_size < 1:
        raise ValueError("q_batch_size must be positive")

    def save(state: np.ndarray, q_values: np.ndarray, time_index: int) -> None:
        diagonal = np.diagonal(state, axis1=-2, axis2=-1)
        summed = np.sum(diagonal, axis=-2)
        q_diagonal[:, q_values, time_index] = np.swapaxes(summed, 0, 1)

    for q_start in range(0, ny, q_batch_size):
        q_values = np.arange(q_start, min(q_start + q_batch_size, ny), dtype=np.int64)
        state = np.stack(
            [
                local_kick_q_sector(
                    baseline_q0,
                    q_index=int(q_index),
                    source_x=source_x,
                    source_y=source_y,
                    nx=int(generator.nx),
                    epsilon=epsilon,
                )
                for q_index in q_values
            ],
            axis=0,
        )
        save(state, q_values, 0)
        action = lambda value: generator.q_sectors_homogeneous_rhs(
            value,
            q_indices=q_values,
            include_number_dephasing=include_number_dephasing,
        )
        for step in range(1, int(steps[-1]) + 1):
            k1 = action(state)
            k2 = action(state + 0.5 * integration_dt * k1)
            k3 = action(state + 0.5 * integration_dt * k2)
            k4 = action(state + integration_dt * k3)
            state = state + (integration_dt / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            if not np.all(np.isfinite(state)):
                raise FloatingPointError(
                    f"non-finite density response in q={q_values.tolist()}, step={step}"
                )
            if step in save_lookup:
                save(state, q_values, save_lookup[step])
    return density_from_q_diagonals(q_diagonal, nx=int(generator.nx))


def local_density_kick_response(
    generator: Any,
    baseline_q0: np.ndarray,
    *,
    walls: Iterable[int],
    source_ys: Iterable[int],
    epsilon: float,
    times: Iterable[float],
    integration_dt: float,
    include_number_dephasing: bool,
    wall_window_columns: int,
    fit_time_min: float,
    fit_time_max: float,
    algorithm: str = "auto",
    q_batch_size: int = 1,
    epsilon_multipliers: Iterable[float] = (0.5, 1.0, 2.0),
) -> DensityKickResponse:
    """Evaluate response profiles and the campaign's velocity estimators."""

    baseline = np.asarray(baseline_q0, dtype=np.complex128)
    expected = (
        int(generator.ny),
        int(generator.block_dimension),
        int(generator.block_dimension),
    )
    if baseline.shape != expected:
        raise ValueError(f"baseline_q0 must have shape {expected}")
    times_array = np.asarray(list(times), dtype=np.float64)
    if (
        times_array.size == 0
        or not np.isclose(times_array[0], 0.0)
        or np.any(np.diff(times_array) <= 0.0)
    ):
        raise ValueError("response times must be strictly increasing and begin at zero")
    walls_array = np.asarray(list(walls), dtype=np.int64)
    source_y_array = np.asarray(list(source_ys), dtype=np.int64)
    if walls_array.size == 0 or source_y_array.size == 0:
        raise ValueError("walls and source_ys must be nonempty")
    paired_x = np.tile(walls_array, source_y_array.size)
    paired_y = np.repeat(source_y_array, walls_array.size)

    selected_algorithm = str(algorithm).lower()
    if selected_algorithm == "auto":
        selected_algorithm = (
            "q_sector_rk4"
            if bool(include_number_dephasing)
            else "rank_four_exact"
        )
    if selected_algorithm == "rank_four_exact":
        if include_number_dephasing:
            raise ValueError("rank-four propagation is invalid with number dephasing")
        density = _rank_four_no_dephasing_response(
            generator,
            baseline,
            source_x=paired_x,
            source_y=paired_y,
            epsilon=float(epsilon),
            times=times_array,
        )
    elif selected_algorithm == "q_sector_rk4":
        density = _q_sector_rk4_response(
            generator,
            baseline,
            source_x=paired_x,
            source_y=paired_y,
            epsilon=float(epsilon),
            times=times_array,
            integration_dt=float(integration_dt),
            include_number_dephasing=bool(include_number_dephasing),
            q_batch_size=int(q_batch_size),
        )
    else:
        raise ValueError("algorithm must be auto, rank_four_exact, or q_sector_rk4")

    # A density phase has no instantaneous density response.  Enforce this
    # exact commutator identity instead of retaining roundoff-level FFT noise.
    density[:, np.isclose(times_array, 0.0, rtol=0.0, atol=1e-14)] = 0.0

    nx, ny = int(generator.nx), int(generator.ny)
    source_count, wall_count = source_y_array.size, walls_array.size
    density = density.reshape(source_count, wall_count, times_array.size, nx, ny)
    raw_profile = np.empty(
        (source_count, wall_count, times_array.size, ny), dtype=np.float64
    )
    norm = np.sum(np.abs(density), axis=(-2, -1))
    retention = np.full_like(norm, np.nan)
    centers = np.full_like(norm, np.nan)
    aligned = np.empty_like(raw_profile)
    displacement = periodic_displacements(ny)
    radius = int(wall_window_columns)
    if radius < 0:
        raise ValueError("wall_window_columns must be nonnegative")
    for source_index, source_y in enumerate(source_y_array):
        for wall_index, wall in enumerate(walls_array):
            columns = sorted(
                {(int(wall) + delta) % nx for delta in range(-radius, radius + 1)}
            )
            cell_density = density[source_index, wall_index]
            profile = np.sum(cell_density[:, columns, :], axis=1)
            raw_profile[source_index, wall_index] = profile
            aligned_profile = np.roll(profile, -int(source_y), axis=-1)
            aligned[source_index, wall_index] = aligned_profile
            wall_norm = np.sum(np.abs(cell_density[:, columns, :]), axis=(1, 2))
            retention[source_index, wall_index] = np.divide(
                wall_norm,
                norm[source_index, wall_index],
                out=np.full(times_array.size, np.nan),
                where=norm[source_index, wall_index] > 0.0,
            )
            positive = np.maximum(aligned_profile, 0.0)
            denominator = np.sum(positive, axis=-1)
            centers[source_index, wall_index] = np.divide(
                positive @ displacement,
                denominator,
                out=np.full(times_array.size, np.nan),
                where=denominator > 1e-300,
            )

    velocity = np.full((source_count, wall_count), np.nan)
    velocity_r2 = np.full_like(velocity, np.nan)
    for source_index in range(source_count):
        for wall_index in range(wall_count):
            velocity[source_index, wall_index], velocity_r2[source_index, wall_index] = (
                fit_positive_lobe_velocity(
                    times_array,
                    centers[source_index, wall_index],
                    norm[source_index, wall_index],
                    fit_time_min=fit_time_min,
                    fit_time_max=fit_time_max,
                )
            )

    epsilon = float(epsilon)
    epsilon_multipliers = np.asarray(
        list(epsilon_multipliers), dtype=np.float64
    ).reshape(-1)
    if (
        epsilon_multipliers.size == 0
        or not np.all(np.isfinite(epsilon_multipliers))
        or np.any(epsilon_multipliers <= 0.0)
    ):
        raise ValueError("epsilon_multipliers must be positive finite values")
    reference = float(np.sinc(epsilon / np.pi))
    epsilon_error = np.abs(
        np.sinc(epsilon * epsilon_multipliers / np.pi) / reference - 1.0
    )
    arrays = {
        "response_times": times_array,
        "response_source_y": source_y_array,
        "response_density_source_wall_time_y": raw_profile,
        "response_density_ty_mean_source": np.mean(aligned, axis=0),
        "response_norm_time": norm,
        "response_wall_retention_time": retention,
        "response_positive_center_time": centers,
        "response_velocity_source_wall": velocity,
        "response_velocity_r2_source_wall": velocity_r2,
        "response_epsilon_relative_error": epsilon_error,
    }
    metadata = {
        "response_probe": "exact_plus_minus_epsilon_local_density_phase_unitary",
        "response_initial_condition": "+i*sinc(epsilon)*[P_s,G_late]",
        "response_algorithm": selected_algorithm,
        "response_q_sector_definition": "X_q(k)=delta_G(k,k-q)",
        "response_baseline": "G_late_cycle_average",
        "response_epsilon": epsilon,
        "response_epsilon_multipliers": epsilon_multipliers.tolist(),
        "response_integration_dt": (
            float(integration_dt) if selected_algorithm == "q_sector_rk4" else None
        ),
        "response_q_batch_size": (
            int(q_batch_size) if selected_algorithm == "q_sector_rk4" else None
        ),
        "response_fit_window": [float(fit_time_min), float(fit_time_max)],
        "response_velocity_center_method": "positive_signed_response_center",
        "response_norm": "full-density L1 norm",
        "response_retention": "three-column wall L1 norm divided by full-density L1 norm",
    }
    return DensityKickResponse(arrays=arrays, metadata=metadata)


__all__ = [
    "DensityKickResponse",
    "density_from_q_diagonals",
    "fit_positive_lobe_velocity",
    "local_density_kick_response",
    "local_kick_q_sector",
    "periodic_displacements",
]
