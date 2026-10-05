"""Endpoint-only active-slab occupation and finite-time Lyapunov diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "hard_wall_purification_endpoint_spectrum_v1"
CAP_TOLERANCE = 1.0e-9
EIGENVECTOR_ENDPOINT_MARGIN = 0.1
NEAREST_MODE_COUNT = 16


def centered_to_lyapunov(
    centered: np.ndarray,
    *,
    cycles: int,
    cap_tolerance: float = CAP_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map centered occupations ``a=2*nu-1`` to signed finite-time rates."""
    values = np.asarray(centered, dtype=np.float64)
    if int(cycles) <= 0:
        raise ValueError("cycles must be positive")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("centered occupation spectrum is nonfinite")
    low = float(np.min(values, initial=0.0))
    high = float(np.max(values, initial=0.0))
    if low < -1.0 - float(cap_tolerance) or high > 1.0 + float(cap_tolerance):
        raise FloatingPointError(
            f"centered occupation spectrum left [-1,1]: [{low:.6e},{high:.6e}]"
        )
    clipped = np.clip(values, -1.0, 1.0)
    pole_cap = clipped <= -1.0 + float(cap_tolerance)
    zero_cap = clipped >= 1.0 - float(cap_tolerance)
    finite = ~(pole_cap | zero_cap)
    rates = np.empty_like(clipped)
    rates[pole_cap] = np.inf
    rates[zero_cap] = -np.inf
    rates[finite] = (
        np.log1p(-clipped[finite]) - np.log1p(clipped[finite])
    ) / (2.0 * float(cycles))
    return rates, finite, zero_cap, pole_cap


def phase_fix_columns(vectors: np.ndarray) -> np.ndarray:
    """Fix each vector's phase using its largest-magnitude component."""
    value = np.asarray(vectors, dtype=np.complex128).copy()
    if value.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if value.shape[1] == 0:
        return value
    pivots = np.argmax(np.abs(value), axis=0)
    columns = np.arange(value.shape[1])
    pivot_values = value[pivots, columns]
    magnitudes = np.abs(pivot_values)
    if np.any(magnitudes == 0.0):
        raise FloatingPointError("cannot phase-fix a zero eigenvector")
    value *= (np.conjugate(pivot_values) / magnitudes)[None, :]
    return value


def spectrum_scalars(
    centered: np.ndarray,
    *,
    cycles: int,
    cap_tolerance: float = CAP_TOLERANCE,
) -> dict[str, Any]:
    """Return the registered sample-wise gap scalars for one sorted spectrum."""
    values = np.asarray(centered, dtype=np.float64)
    rates, finite, zero_cap, pole_cap = centered_to_lyapunov(
        values, cycles=cycles, cap_tolerance=cap_tolerance
    )
    finite_rates = rates[finite]
    half_gap = (
        float(np.min(np.abs(finite_rates))) if finite_rates.size else float("inf")
    )
    positive = finite_rates[finite_rates > 0.0]
    negative = finite_rates[finite_rates < 0.0]
    two_sided = (
        float(np.min(positive) - np.max(negative))
        if positive.size and negative.size
        else float("nan")
    )
    return {
        "rates": rates,
        "finite": finite,
        "zero_cap": zero_cap,
        "pole_cap": pole_cap,
        "lyapunov_half_gap": half_gap,
        "lyapunov_two_sided_gap": two_sided,
        "centered_half_gap": float(np.min(np.abs(values))),
        "finite_mode_count": int(np.count_nonzero(finite)),
        "positive_cap_count": int(np.count_nonzero(zero_cap)),
        "negative_cap_count": int(np.count_nonzero(pole_cap)),
    }


class EndpointSpectrumObserver:
    """Compute one active-slab eigensystem after the dynamics is durable."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        samples: int,
        active_indices: torch.Tensor,
        full_mode_count: int,
        sample_indices: np.ndarray,
        sample_chunk: int,
        cap_tolerance: float = CAP_TOLERANCE,
        eigenvector_endpoint_margin: float = EIGENVECTOR_ENDPOINT_MARGIN,
        nearest_mode_count: int = NEAREST_MODE_COUNT,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples = int(samples)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.active_indices = torch.as_tensor(
            active_indices, dtype=torch.long, device=active_indices.device
        ).reshape(-1)
        self.full_mode_count = int(full_mode_count)
        self.sample_chunk = int(sample_chunk)
        self.cap_tolerance = float(cap_tolerance)
        self.eigenvector_endpoint_margin = float(eigenvector_endpoint_margin)
        self.nearest_mode_count = int(nearest_mode_count)
        self.n_eff = int(self.active_indices.numel())
        if self.nx != 20 or self.cycles != 2 * self.ny:
            raise ValueError("production requires Nx=20 and cycles=2*Ny")
        if self.n_eff != 22 * self.ny:
            raise ValueError(f"hard-wall active dimension must be 22*Ny, got {self.n_eff}")
        if self.sample_indices.shape != (self.samples,):
            raise ValueError("sample_indices must identify every resident trajectory")
        if self.sample_chunk <= 0 or self.nearest_mode_count <= 0:
            raise ValueError("sample_chunk and nearest_mode_count must be positive")
        if not 0.0 < self.eigenvector_endpoint_margin < 1.0:
            raise ValueError("eigenvector endpoint margin must lie in (0,1)")
        self._computed = False
        self._allocate()

    def _allocate(self) -> None:
        s = self.samples
        k = min(self.nearest_mode_count, self.n_eff)
        self.lyapunov_half_gap = np.full(s, np.nan)
        self.lyapunov_two_sided_gap = np.full(s, np.nan)
        self.centered_half_gap = np.full(s, np.nan)
        self.finite_mode_count = np.zeros(s, dtype=np.int64)
        self.positive_cap_count = np.zeros(s, dtype=np.int64)
        self.negative_cap_count = np.zeros(s, dtype=np.int64)
        self.nearest_centered = np.full((s, k), np.nan)
        self.nearest_lyapunov = np.full((s, k), np.nan)
        self.nearest_full_indices = np.full((s, k), -1, dtype=np.int64)
        self.hermiticity_residual = np.full(s, np.nan)
        self.spectral_bound_residual = np.full(s, np.nan)
        self.nearest_eigensolver_residual = np.full(s, np.nan)
        self.selected_eigensolver_residual = np.full(s, np.nan)
        self.full_centered_spectrum = (
            np.full((s, self.n_eff), np.nan) if self.ny == 50 else None
        )
        self.selected_vectors: list[np.ndarray] = [
            np.empty((self.n_eff, 0), dtype=np.complex128) for _ in range(s)
        ]
        self.selected_indices: list[np.ndarray] = [
            np.empty((0,), dtype=np.int64) for _ in range(s)
        ]
        self.selected_centered: list[np.ndarray] = [
            np.empty((0,), dtype=np.float64) for _ in range(s)
        ]
        self.selected_lyapunov: list[np.ndarray] = [
            np.empty((0,), dtype=np.float64) for _ in range(s)
        ]
        self.selected_residuals: list[np.ndarray] = [
            np.empty((0,), dtype=np.float64) for _ in range(s)
        ]

    def checkpoint_payload(self) -> dict[str, np.ndarray]:
        return {}

    def restore_checkpoint(
        self, payload: dict[str, np.ndarray], *, completed_cycle: int
    ) -> None:
        if payload:
            raise ValueError("endpoint-only observer checkpoint must be empty")
        if completed_cycle < 0 or completed_cycle > self.cycles:
            raise ValueError("invalid completed cycle")

    def validate(self, *, completed_cycle: int, require_endpoint: bool = False) -> None:
        if completed_cycle < 0 or completed_cycle > self.cycles:
            raise ValueError("completed cycle lies outside the campaign horizon")
        if require_endpoint and not self._computed:
            raise RuntimeError("endpoint spectrum has not been computed")

    def compute_endpoint(self, G: np.ndarray | torch.Tensor) -> None:
        if self._computed:
            raise RuntimeError("endpoint spectrum was already computed")
        full = torch.as_tensor(G, device=self.active_indices.device)
        expected = (self.samples, self.full_mode_count, self.full_mode_count)
        if tuple(full.shape) != expected or full.dtype != torch.complex128:
            raise ValueError(f"expected complex128 covariance shape {expected}")
        keep_bound = 1.0 - self.eigenvector_endpoint_margin
        with torch.inference_mode():
            for start in range(0, self.samples, self.sample_chunk):
                stop = min(self.samples, start + self.sample_chunk)
                chunk = full[start:stop]
                active = chunk.index_select(1, self.active_indices).index_select(
                    2, self.active_indices
                )
                herm = torch.amax(torch.abs(active - active.mH), dim=(-2, -1)).real
                active = 0.5 * (active + active.mH)
                values, vectors = torch.linalg.eigh(active)
                for local in range(stop - start):
                    row = start + local
                    raw = values[local].detach().cpu().numpy().astype(np.float64, copy=False)
                    low = float(raw[0])
                    high = float(raw[-1])
                    bound_residual = max(0.0, -1.0 - low, high - 1.0)
                    if bound_residual > self.cap_tolerance:
                        raise FloatingPointError(
                            f"sample {self.sample_indices[row]} left [-1,1] by {bound_residual:.3e}"
                        )
                    centered = np.clip(raw, -1.0, 1.0)
                    scalars = spectrum_scalars(
                        centered,
                        cycles=self.cycles,
                        cap_tolerance=self.cap_tolerance,
                    )
                    rates = np.asarray(scalars.pop("rates"), dtype=np.float64)
                    finite = np.asarray(scalars.pop("finite"), dtype=bool)
                    scalars.pop("zero_cap")
                    scalars.pop("pole_cap")
                    self.lyapunov_half_gap[row] = scalars["lyapunov_half_gap"]
                    self.lyapunov_two_sided_gap[row] = scalars["lyapunov_two_sided_gap"]
                    self.centered_half_gap[row] = scalars["centered_half_gap"]
                    self.finite_mode_count[row] = scalars["finite_mode_count"]
                    self.positive_cap_count[row] = scalars["positive_cap_count"]
                    self.negative_cap_count[row] = scalars["negative_cap_count"]
                    self.hermiticity_residual[row] = float(herm[local].item())
                    self.spectral_bound_residual[row] = bound_residual

                    finite_indices = np.flatnonzero(finite)
                    ordered = finite_indices[
                        np.argsort(np.abs(rates[finite_indices]), kind="mergesort")
                    ]
                    nearest = ordered[: self.nearest_mode_count]
                    count = nearest.size
                    self.nearest_centered[row, :count] = centered[nearest]
                    self.nearest_lyapunov[row, :count] = rates[nearest]
                    self.nearest_full_indices[row, :count] = nearest
                    if count:
                        nearest_t = torch.as_tensor(
                            nearest, dtype=torch.long, device=vectors.device
                        )
                        nearest_vectors = vectors[local].index_select(1, nearest_t)
                        nearest_values = values[local].index_select(0, nearest_t)
                        residual = torch.linalg.vector_norm(
                            active[local] @ nearest_vectors
                            - nearest_vectors * nearest_values[None, :],
                            dim=0,
                        )
                        self.nearest_eigensolver_residual[row] = float(
                            torch.max(residual).item()
                        )
                    else:
                        self.nearest_eigensolver_residual[row] = 0.0

                    if self.ny == 50:
                        assert self.full_centered_spectrum is not None
                        self.full_centered_spectrum[row] = centered
                        selected = np.flatnonzero(np.abs(centered) <= keep_bound)
                        selected_t = torch.as_tensor(
                            selected, dtype=torch.long, device=vectors.device
                        )
                        selected_vectors_t = vectors[local].index_select(1, selected_t)
                        selected_values_t = values[local].index_select(0, selected_t)
                        selected_residual_t = torch.linalg.vector_norm(
                            active[local] @ selected_vectors_t
                            - selected_vectors_t * selected_values_t[None, :],
                            dim=0,
                        )
                        selected_vectors = phase_fix_columns(
                            selected_vectors_t.detach().cpu().numpy()
                        )
                        self.selected_vectors[row] = selected_vectors
                        self.selected_indices[row] = selected.astype(np.int64, copy=False)
                        self.selected_centered[row] = centered[selected]
                        self.selected_lyapunov[row] = rates[selected]
                        selected_residuals = (
                            selected_residual_t.detach().cpu().numpy().astype(np.float64, copy=False)
                        )
                        self.selected_residuals[row] = selected_residuals
                        self.selected_eigensolver_residual[row] = (
                            float(np.max(selected_residuals))
                            if selected_residuals.size
                            else 0.0
                        )
        self._computed = True

    def result_payload(self, sample_slice: slice) -> dict[str, np.ndarray]:
        self.validate(completed_cycle=self.cycles, require_endpoint=True)
        positions = np.arange(self.samples, dtype=np.int64)[sample_slice]
        payload: dict[str, np.ndarray] = {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "sample_indices": self.sample_indices[positions],
            "active_top_layer_indices": self.active_indices.detach().cpu().numpy(),
            "endpoint_cycle": np.asarray(self.cycles, dtype=np.int64),
            "lyapunov_half_gap": self.lyapunov_half_gap[positions],
            "lyapunov_two_sided_gap": self.lyapunov_two_sided_gap[positions],
            "centered_half_gap": self.centered_half_gap[positions],
            "finite_mode_count": self.finite_mode_count[positions],
            "positive_cap_count": self.positive_cap_count[positions],
            "negative_cap_count": self.negative_cap_count[positions],
            "nearest_centered_eigenvalues": self.nearest_centered[positions],
            "nearest_lyapunov_exponents": self.nearest_lyapunov[positions],
            "nearest_full_spectrum_indices": self.nearest_full_indices[positions],
            "hermiticity_residual": self.hermiticity_residual[positions],
            "spectral_bound_residual": self.spectral_bound_residual[positions],
            "nearest_eigensolver_residual": self.nearest_eigensolver_residual[positions],
            "cap_tolerance": np.asarray(self.cap_tolerance, dtype=np.float64),
            "lyapunov_formula": np.asarray("lambda=(log(1-a)-log(1+a))/(2*T)"),
        }
        if self.ny == 50:
            assert self.full_centered_spectrum is not None
            vectors = [self.selected_vectors[int(row)] for row in positions]
            indices = [self.selected_indices[int(row)] for row in positions]
            centered = [self.selected_centered[int(row)] for row in positions]
            rates = [self.selected_lyapunov[int(row)] for row in positions]
            residuals = [self.selected_residuals[int(row)] for row in positions]
            counts = np.asarray([value.shape[0] for value in indices], dtype=np.int64)
            offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
            payload.update(
                {
                    "full_centered_occupation_spectrum": self.full_centered_spectrum[positions],
                    "selected_mode_offsets": offsets,
                    "selected_mode_counts": counts,
                    "selected_mode_sample_indices": np.repeat(
                        self.sample_indices[positions], counts
                    ),
                    "selected_full_spectrum_indices": np.concatenate(indices),
                    "selected_centered_eigenvalues": np.concatenate(centered),
                    "selected_lyapunov_exponents": np.concatenate(rates),
                    "selected_eigensolver_residuals": np.concatenate(residuals),
                    "selected_eigenvectors": np.concatenate(vectors, axis=1),
                    "selected_eigenvector_endpoint_margin": np.asarray(
                        self.eigenvector_endpoint_margin, dtype=np.float64
                    ),
                    "selected_eigenvector_rule": np.asarray("abs(a)<=0.9"),
                    "selected_eigenvector_phase_rule": np.asarray(
                        "largest-magnitude component real nonnegative"
                    ),
                    "selected_eigensolver_residual": self.selected_eigensolver_residual[
                        positions
                    ],
                }
            )
        return payload
