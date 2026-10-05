from __future__ import annotations

from typing import Iterable

import numpy as np


def late_cycle_average(
    states: np.ndarray,
    cycles: Iterable[int],
    *,
    start_cycle: int,
    end_cycle: int,
) -> np.ndarray:
    """Average dense correlation matrices over an inclusive late-cycle window.

    ``states`` may have shape ``(cycle,N,N)`` or ``(sample,cycle,N,N)``.  The
    returned object retains the sample axis when one was supplied.
    """

    values = np.asarray(states, dtype=np.complex128)
    coordinates = np.asarray(list(cycles), dtype=np.int64)
    cycle_axis = 1 if values.ndim == 4 else 0
    if values.ndim not in (3, 4):
        raise ValueError("states must have shape (cycle,N,N) or (sample,cycle,N,N)")
    if values.shape[cycle_axis] != coordinates.size:
        raise ValueError("cycle coordinate length does not match the state history")
    if np.unique(coordinates).size != coordinates.size:
        raise ValueError("cycle coordinates contain duplicates")
    selected = (coordinates >= int(start_cycle)) & (coordinates <= int(end_cycle))
    expected = np.arange(int(start_cycle), int(end_cycle) + 1, dtype=np.int64)
    if not np.array_equal(coordinates[selected], expected):
        raise ValueError("late-cycle window is incomplete or out of order")
    return np.mean(np.take(values, np.flatnonzero(selected), axis=cycle_axis), axis=cycle_axis)


def _as_y_block(G: np.ndarray, nx: int, ny: int) -> np.ndarray:
    matrix = np.asarray(G, dtype=np.complex128)
    n = 2 * int(nx) * int(ny)
    if matrix.shape != (n, n):
        raise ValueError(f"expected a ({n},{n}) correlation matrix, got {matrix.shape}")
    # The canonical class flattens i=mu+2*x+2*Nx*y, i.e. Fortran order on
    # (mu,x,y).  Work below in the explicit (y,x,mu;y',x',mu') layout.
    canonical = matrix.reshape(2, int(nx), int(ny), 2, int(nx), int(ny), order="F")
    return np.transpose(canonical, (2, 1, 0, 5, 4, 3))


def _from_y_block(block: np.ndarray, nx: int, ny: int) -> np.ndarray:
    values = np.asarray(block, dtype=np.complex128)
    expected = (int(ny), int(nx), 2, int(ny), int(nx), 2)
    if values.shape != expected:
        raise ValueError(f"expected y-block shape {expected}, got {values.shape}")
    canonical = np.transpose(values, (2, 1, 0, 5, 4, 3))
    return canonical.reshape(2 * int(nx) * int(ny), 2 * int(nx) * int(ny), order="F")


def y_translate(G: np.ndarray, nx: int, ny: int, shift: int = 1) -> np.ndarray:
    block = _as_y_block(G, nx, ny)
    shifted = np.roll(np.roll(block, int(shift), axis=0), int(shift), axis=3)
    return _from_y_block(shifted, nx, ny)


def translation_residual(G: np.ndarray, nx: int, ny: int) -> float:
    matrix = np.asarray(G, dtype=np.complex128)
    denominator = max(float(np.linalg.norm(matrix)), 1e-300)
    return float(np.linalg.norm(y_translate(matrix, nx, ny) - matrix) / denominator)


def exact_y_twirl(G: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Return ``Ny^-1 sum_s T_y^s G T_y^-s`` without assuming symmetry."""

    block = _as_y_block(G, nx, ny)
    twirled = np.zeros_like(block)
    for shift in range(int(ny)):
        twirled += np.roll(np.roll(block, shift, axis=0), shift, axis=3)
    twirled /= float(ny)
    matrix = _from_y_block(twirled, nx, ny)
    return 0.5 * (matrix + matrix.conj().T)


def ky_blocks_from_twirled(G: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Fourier transform an exactly twirled correlation matrix into ``ky`` blocks."""

    matrix = 0.5 * (np.asarray(G, dtype=np.complex128) + np.asarray(G, dtype=np.complex128).conj().T)
    twirled = matrix if translation_residual(matrix, nx, ny) <= 1e-12 else exact_y_twirl(matrix, nx, ny)
    block = _as_y_block(twirled, nx, ny)
    transformed = np.fft.fft(block, axis=0, norm="ortho")
    transformed = np.fft.ifft(transformed, axis=3, norm="ortho")
    ky_blocks = np.empty((ny, 2 * nx, 2 * nx), dtype=np.complex128)
    for k in range(ny):
        diagonal = transformed[k, :, :, k, :, :].reshape(2 * nx, 2 * nx)
        ky_blocks[k] = 0.5 * (diagonal + diagonal.conj().T)
    return 2.0 * np.pi * np.fft.fftfreq(ny), ky_blocks


def ky_spectrum_and_x_weights(
    G: np.ndarray, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ky, blocks = ky_blocks_from_twirled(G, nx, ny)
    occupations, eigenvectors = np.linalg.eigh(blocks)
    x_weights = np.sum(
        np.abs(eigenvectors.reshape(ny, nx, 2, 2 * nx)) ** 2,
        axis=2,
    )
    return ky, occupations.real, x_weights.real


def global_spectrum_and_momentum_weights(
    G: np.ndarray, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    matrix = 0.5 * (np.asarray(G) + np.asarray(G).conj().T)
    occupations, eigenvectors = np.linalg.eigh(matrix)
    modes = eigenvectors.reshape(ny, nx, 2, 2 * nx * ny)
    modes_k = np.fft.fft(modes, axis=0, norm="ortho")
    weights = np.sum(np.abs(modes_k) ** 2, axis=(1, 2)).T
    return occupations.real, weights.real


def select_wall_branches(
    occupations: np.ndarray,
    x_weights: np.ndarray,
    wall_locations: tuple[int, int] | list[int],
    *,
    threshold: float = 0.25,
    wall_penalty: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    branches, selected_weights = [], []
    nx = int(x_weights.shape[1])
    for wall in wall_locations:
        columns = sorted({(int(wall) + delta) % nx for delta in (-1, 0, 1)})
        weights = np.sum(x_weights[:, columns, :], axis=1)
        eligible = weights >= float(threshold)
        cost = np.where(
            eligible,
            np.abs(occupations - 0.5) + float(wall_penalty) * (1.0 - weights),
            np.inf,
        )
        mode = np.argmin(cost, axis=1)
        missing = ~np.any(eligible, axis=1)
        mode[missing] = np.argmax(weights[missing], axis=1)
        index = np.arange(occupations.shape[0])
        branches.append(occupations[index, mode])
        selected_weights.append(weights[index, mode])
    return np.asarray(branches), np.asarray(selected_weights)


def enrich_terminal_arrays(case: dict, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add standardized terminal products without changing adapter-owned arrays."""

    nx, ny = int(case["model"]["Nx"]), int(case["model"]["Ny"])
    required = {"G_final", "G_late_cycle_average"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"adapter result is missing checkpoint states: {sorted(missing)}")
    result = dict(arrays)
    late = np.asarray(result["G_late_cycle_average"], dtype=np.complex128)
    final = np.asarray(result["G_final"], dtype=np.complex128)
    if late.ndim == 2:
        late = late[None, ...]
    if final.ndim == 2:
        final = final[None, ...]
    expected_samples = len(case["dynamics"]["sample_ids"])
    if late.shape[0] != expected_samples or final.shape[0] != expected_samples:
        raise ValueError("checkpoint states do not retain the declared sample axis")
    result["G_late_cycle_average"] = late
    result["G_final"] = final
    residuals, twirls, occ_ky_all, x_weights_all = [], [], [], []
    branches_all, branch_weights_all = [], []
    global_occ_all, momentum_weights_all = [], []
    ky = None
    for sample_late in late:
        residual = translation_residual(sample_late, nx, ny)
        residuals.append(residual)
        twirled = exact_y_twirl(sample_late, nx, ny)
        twirls.append(twirled)
        ky, occupations_ky, x_weights = ky_spectrum_and_x_weights(twirled, nx, ny)
        occ_ky_all.append(occupations_ky)
        x_weights_all.append(x_weights)
        branches, branch_weights = select_wall_branches(
            occupations_ky, x_weights, case["model"]["wall_locations"]
        )
        branches_all.append(branches)
        branch_weights_all.append(branch_weights)
        if residual <= 1e-11:
            # For the exactly translation-invariant Lindblad arm the full dense
            # eigensolve is both redundant and much more expensive than the Ny
            # independent 2*Nx blocks.  Preserve the same terminal schema with an
            # exact one-hot momentum label for each block eigenmode.
            global_occupations = occupations_ky.reshape(-1)
            momentum_weights = np.zeros((2 * nx * ny, ny), dtype=float)
            momentum_weights[np.arange(2 * nx * ny), np.repeat(np.arange(ny), 2 * nx)] = 1.0
        else:
            global_occupations, momentum_weights = global_spectrum_and_momentum_weights(
                sample_late, nx, ny
            )
        global_occ_all.append(global_occupations)
        momentum_weights_all.append(momentum_weights)
    result["late_translation_residual"] = np.asarray(residuals)
    result["G_late_cycle_average_twirl"] = np.asarray(twirls)
    result["ky"] = np.asarray(ky)
    result["twirled_ky_natural_occupations"] = np.asarray(occ_ky_all)
    result["twirled_ky_x_weights"] = np.asarray(x_weights_all)
    result["wall_branch_occupations"] = np.asarray(branches_all)
    result["wall_branch_weights"] = np.asarray(branch_weights_all)
    result["global_natural_occupations"] = np.asarray(global_occ_all)
    result["momentum_weights_for_global_modes"] = np.asarray(momentum_weights_all)
    return result
