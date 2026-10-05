from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .common import RegionMasks, mode_localization


def exact_choi_spectrum(
    sigma_ll: np.ndarray,
    *,
    cycle: int,
    n_eigenstates: int = 3,
    endpoint_tol: float = 1e-12,
    spectral_tol: float = 1e-10,
) -> dict[str, Any]:
    """Extract regularized particle-transfer exponents from one Sigma_LL block."""
    sigma_ll = np.asarray(sigma_ll, dtype=np.complex128)
    if sigma_ll.ndim != 2 or sigma_ll.shape[0] != sigma_ll.shape[1]:
        raise ValueError("sigma_ll must be a square matrix.")
    if int(cycle) <= 0:
        raise ValueError("cycle must be positive.")
    hermitian = 0.5 * (sigma_ll + sigma_ll.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    raw_min, raw_max = float(values[0]), float(values[-1])
    if raw_min < -1.0 - spectral_tol or raw_max > 1.0 + spectral_tol:
        raise FloatingPointError(
            f"Sigma_LL spectrum [{raw_min:.6e}, {raw_max:.6e}] leaves the pure-Choi interval."
        )
    clipped = np.clip(values, -1.0, 1.0)
    particle_poles = clipped <= -1.0 + endpoint_tol
    particle_zeros = clipped >= 1.0 - endpoint_tol
    finite = ~(particle_poles | particle_zeros)
    exponents = np.empty_like(clipped)
    exponents[particle_poles] = np.inf
    exponents[particle_zeros] = -np.inf
    exponents[finite] = (
        np.log1p(-clipped[finite]) - np.log1p(clipped[finite])
    ) / (2.0 * float(cycle))
    order = np.argsort(exponents)
    spectrum = exponents[order]
    finite_indices = np.flatnonzero(finite)
    finite_spectrum = np.sort(exponents[finite_indices])
    near_order = finite_indices[np.argsort(np.abs(exponents[finite_indices]))]
    selected = near_order[: min(int(n_eigenstates), near_order.size)]
    selected_vectors = vectors[:, selected]
    residuals = (
        np.linalg.norm(hermitian @ selected_vectors - selected_vectors * clipped[selected][None, :], axis=0)
        if selected.size
        else np.empty((0,), dtype=np.float64)
    )
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    return {
        "spectrum": spectrum,
        "finite_spectrum": finite_spectrum,
        "gap": float(np.min(np.abs(exponents[finite_indices]))) if finite_indices.size else np.nan,
        "near_gap_exponents": exponents[selected],
        "near_gap_a_eigenvalues": clipped[selected],
        "near_gap_residuals": residuals,
        "eigenstates": selected_vectors,
        "selected_indices": inverse_order[selected],
        "finite_eigenstate_count": int(finite_indices.size),
        "particle_zero_count": int(np.count_nonzero(particle_zeros)),
        "particle_pole_count": int(np.count_nonzero(particle_poles)),
        "a_eigenvalue_min": raw_min,
        "a_eigenvalue_max": raw_max,
    }


class ChoiSpectrumRecorder:
    def __init__(
        self,
        *,
        samples: int,
        cycles: Iterable[int],
        dimension: int,
        n_eigenstates: int = 3,
        endpoint_tol: float = 1e-12,
        spectral_tol: float = 1e-10,
    ) -> None:
        self.samples = int(samples)
        self.cycles = tuple(int(cycle) for cycle in cycles)
        self.dimension = int(dimension)
        self.n_eigenstates = int(n_eigenstates)
        self.endpoint_tol = float(endpoint_tol)
        self.spectral_tol = float(spectral_tol)
        if not self.cycles or tuple(sorted(set(self.cycles))) != self.cycles:
            raise ValueError("Choi observer cycles must be sorted and unique.")
        self._cycle_index = {cycle: index for index, cycle in enumerate(self.cycles)}
        shape = (self.samples, len(self.cycles))
        near_shape = shape + (self.n_eigenstates,)
        self.gap = np.full(shape, np.nan, dtype=np.float64)
        self.spectrum = np.full(shape + (self.dimension,), np.nan, dtype=np.float64)
        self.near_gap_exponents = np.full(near_shape, np.nan, dtype=np.float64)
        self.near_gap_a_eigenvalues = np.full(near_shape, np.nan, dtype=np.float64)
        self.near_gap_residuals = np.full(near_shape, np.nan, dtype=np.float64)
        self.finite_count = np.full(shape, -1, dtype=np.int64)
        self.zero_count = np.full(shape, -1, dtype=np.int64)
        self.pole_count = np.full(shape, -1, dtype=np.int64)
        self.active = np.zeros(shape, dtype=bool)
        self.final_eigenvectors = np.full(
            (self.samples, self.dimension, self.n_eigenstates),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        self.final_eigenvector_cycle = np.full((self.samples,), -1, dtype=np.int64)
        self.failure_records: list[dict[str, Any]] = []
        self.active_indices: np.ndarray | None = None

    def __call__(
        self,
        *,
        cycle: int,
        sigma_ll: np.ndarray,
        batch_start: int,
        batch_count: int,
        choi_active_mask: np.ndarray | None = None,
        choi_failure_records: Iterable[dict[str, Any]] = (),
        active_top_layer_indices: np.ndarray | None = None,
        **_: Any,
    ) -> None:
        if int(cycle) not in self._cycle_index:
            raise ValueError(f"Unexpected Choi observation cycle {cycle}.")
        cycle_index = self._cycle_index[int(cycle)]
        blocks = np.asarray(sigma_ll, dtype=np.complex128)
        if blocks.shape != (int(batch_count), self.dimension, self.dimension):
            raise ValueError(f"Unexpected Sigma_LL shape {blocks.shape}.")
        if active_top_layer_indices is not None:
            active_indices = np.asarray(active_top_layer_indices, dtype=np.int64)
            if self.active_indices is None:
                self.active_indices = active_indices.copy()
            elif not np.array_equal(self.active_indices, active_indices):
                raise ValueError("The active Choi basis changed during a run.")
        active = (
            np.ones((int(batch_count),), dtype=bool)
            if choi_active_mask is None
            else np.asarray(choi_active_mask, dtype=bool).reshape(-1)
        )
        for offset in range(int(batch_count)):
            sample = int(batch_start) + offset
            if not active[offset]:
                continue
            result = exact_choi_spectrum(
                blocks[offset],
                cycle=int(cycle),
                n_eigenstates=self.n_eigenstates,
                endpoint_tol=self.endpoint_tol,
                spectral_tol=self.spectral_tol,
            )
            self.active[sample, cycle_index] = True
            self.gap[sample, cycle_index] = result["gap"]
            finite_count = result["finite_eigenstate_count"]
            self.spectrum[sample, cycle_index, :finite_count] = result["finite_spectrum"]
            count = result["near_gap_exponents"].size
            self.near_gap_exponents[sample, cycle_index, :count] = result["near_gap_exponents"]
            self.near_gap_a_eigenvalues[sample, cycle_index, :count] = result["near_gap_a_eigenvalues"]
            self.near_gap_residuals[sample, cycle_index, :count] = result["near_gap_residuals"]
            self.finite_count[sample, cycle_index] = result["finite_eigenstate_count"]
            self.zero_count[sample, cycle_index] = result["particle_zero_count"]
            self.pole_count[sample, cycle_index] = result["particle_pole_count"]
            if count:
                self.final_eigenvectors[sample] = np.nan + 1j * np.nan
                self.final_eigenvectors[sample, :, :count] = result["eigenstates"]
                self.final_eigenvector_cycle[sample] = int(cycle)
        self.failure_records.extend(dict(record) for record in choi_failure_records)
        return None

    def payload(self) -> dict[str, Any]:
        return {
            "choi_cycles": np.asarray(self.cycles, dtype=np.int64),
            "choi_gap": self.gap,
            "choi_spectrum": self.spectrum,
            "choi_near_gap_exponents": self.near_gap_exponents,
            "choi_near_gap_a_eigenvalues": self.near_gap_a_eigenvalues,
            "choi_near_gap_residuals": self.near_gap_residuals,
            "choi_finite_count": self.finite_count,
            "choi_zero_count": self.zero_count,
            "choi_pole_count": self.pole_count,
            "choi_active": self.active,
            "choi_final_eigenvectors": self.final_eigenvectors,
            "choi_final_eigenvector_cycle": self.final_eigenvector_cycle,
            "choi_active_indices": np.asarray([], dtype=np.int64) if self.active_indices is None else self.active_indices,
        }


class LyapunovSpectrumRecorder:
    def __init__(self, *, samples: int, cycles: int, nvec: int, vector_dimension: int) -> None:
        self.samples = int(samples)
        self.cycles = int(cycles)
        self.nvec = int(nvec)
        self.vector_dimension = int(vector_dimension)
        self.spectrum = np.full((self.samples, self.cycles, self.nvec), np.nan, dtype=np.float64)
        self.gap = np.full((self.samples, self.cycles), np.nan, dtype=np.float64)
        self.final_vector = np.full(
            (self.samples, self.vector_dimension), np.nan + 1j * np.nan, dtype=np.complex128
        )
        self.final_value = np.full((self.samples,), np.nan, dtype=np.float64)
        self.final_index = np.full((self.samples,), -1, dtype=np.int64)
        self.null_count = np.full((self.samples,), -1, dtype=np.int64)

    def __call__(
        self,
        *,
        cycle: int,
        spectra: np.ndarray,
        batch_start: int,
        batch_count: int,
        lyapunov_min_abs_vector: np.ndarray | None = None,
        lyapunov_min_abs_value: np.ndarray | None = None,
        lyapunov_min_abs_index: np.ndarray | None = None,
        lyapunov_null_counts: np.ndarray | None = None,
        **_: Any,
    ) -> None:
        cycle_index = int(cycle) - 1
        values = np.asarray(spectra, dtype=np.float64)
        if values.shape != (int(batch_count), self.nvec):
            raise ValueError(f"Unexpected Lyapunov spectrum shape {values.shape}.")
        sl = slice(int(batch_start), int(batch_start) + int(batch_count))
        self.spectrum[sl, cycle_index] = values
        self.gap[sl, cycle_index] = np.min(np.abs(values), axis=1)
        if lyapunov_min_abs_vector is not None:
            vectors = np.asarray(lyapunov_min_abs_vector, dtype=np.complex128)
            if vectors.shape != (int(batch_count), self.vector_dimension):
                raise ValueError(f"Unexpected Lyapunov vector shape {vectors.shape}.")
            self.final_vector[sl] = vectors
            self.final_value[sl] = np.asarray(lyapunov_min_abs_value, dtype=np.float64)
            self.final_index[sl] = np.asarray(lyapunov_min_abs_index, dtype=np.int64)
            self.null_count[sl] = np.asarray(lyapunov_null_counts, dtype=np.int64)

    def payload(self) -> dict[str, Any]:
        return {
            "lyapunov_spectrum": self.spectrum,
            "lyapunov_gap": self.gap,
            "lyapunov_final_vector": self.final_vector,
            "lyapunov_final_value": self.final_value,
            "lyapunov_final_index": self.final_index,
            "lyapunov_null_count": self.null_count,
        }


def localize_vector_batch(
    vectors: np.ndarray,
    *,
    nx: int,
    ny: int,
    regions: RegionMasks,
    active_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.complex128)
    if vectors.ndim == 2:
        vectors = vectors[:, :, None]
    if vectors.ndim != 3:
        raise ValueError("vectors must have shape (samples,dimension[,modes]).")
    samples, _, modes = vectors.shape
    cell = np.full((samples, modes, nx, ny), np.nan, dtype=np.float64)
    x_profile = np.full((samples, modes, nx), np.nan, dtype=np.float64)
    ipr = np.full((samples, modes), np.nan, dtype=np.float64)
    region_weight = np.full((samples, modes, len(regions.names)), np.nan, dtype=np.float64)
    for sample in range(samples):
        for mode in range(modes):
            vector = vectors[sample, :, mode]
            if not np.all(np.isfinite(vector)):
                continue
            result = mode_localization(
                vector,
                nx=nx,
                ny=ny,
                regions=regions,
                active_indices=active_indices,
            )
            cell[sample, mode] = result["cell_weight"]
            x_profile[sample, mode] = result["x_profile"]
            ipr[sample, mode] = result["ipr"]
            for region_index, name in enumerate(regions.names):
                region_weight[sample, mode, region_index] = result[f"{name}_weight"]
    return {
        "cell_weight": cell,
        "x_profile": x_profile,
        "ipr": ipr,
        "region_weight": region_weight,
        "region_names": np.asarray(regions.names),
    }
