"""Batched active-space many-body Lyapunov observer for the A100 pilot."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "maxmix_active_spectrum_gpu_observer_v2"
# At production dimension (N_eff=880), CUDA complex128 ``eigh`` can place an
# exact covariance cap a few 1e-10 outside [0, 1].  Use one tolerance for both
# the physicality guard and exact-cap classification so the two checks cannot
# contradict one another.  Values beyond this scale still fail closed.
CAP_TOLERANCE = 1.0e-9


def spectrum_checkpoint_cycles(ny: int) -> np.ndarray:
    """Return the preregistered spectrum checkpoints through ``2*Ny``."""
    ny = int(ny)
    if ny <= 0 or ny % 2:
        raise ValueError("Ny must be a positive even integer")
    return np.asarray(
        sorted(set(range(0, 2 * ny + 1, 4)) | {ny, 3 * ny // 2, 2 * ny}),
        dtype=np.int64,
    )


def natural_spectrum_factors(
    occupations: np.ndarray, *, cap_tolerance: float = CAP_TOLERANCE
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the preferred log weight, flip costs, and exact-cap mask."""
    nu = np.asarray(occupations, dtype=np.float64)
    if nu.ndim != 1 or not np.all(np.isfinite(nu)):
        raise ValueError("occupations must be one finite one-dimensional array")
    if np.min(nu, initial=0.0) < -cap_tolerance or np.max(
        nu, initial=1.0
    ) > 1.0 + cap_tolerance:
        raise FloatingPointError("occupation spectrum lies outside [0,1]")
    empty_cap = nu <= cap_tolerance
    full_cap = nu >= 1.0 - cap_tolerance
    caps = empty_cap | full_cap
    interior = ~caps
    preferred = np.zeros_like(nu)
    preferred[interior] = np.maximum(nu[interior], 1.0 - nu[interior])
    log_preferred = float(np.log(preferred[interior]).sum())
    costs = np.full(nu.shape, np.inf, dtype=np.float64)
    costs[interior] = np.abs(
        np.log(nu[interior]) - np.log1p(-nu[interior])
    )
    return log_preferred, costs, caps


def lowest_subset_sums(costs: np.ndarray, count: int) -> np.ndarray:
    """Return the ``count`` smallest subset sums of finite nonnegative costs."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    values = np.sort(np.asarray(costs, dtype=np.float64))
    values = values[np.isfinite(values)]
    if np.any(values < 0):
        raise ValueError("flip costs must be nonnegative")
    sums = np.asarray([0.0], dtype=np.float64)
    for value in values:
        merged = np.concatenate((sums, sums + value))
        if merged.size > count:
            keep = np.argpartition(merged, count - 1)[:count]
            merged = merged[keep]
        sums = np.sort(merged)
        if sums.size == count and value > sums[-1]:
            break
    if sums.size < count:
        sums = np.pad(sums, (0, count - sums.size), constant_values=np.inf)
    return sums[:count]


def leading_log_sigma2_levels(
    occupations: np.ndarray,
    log_z: float,
    *,
    count: int = 64,
    cap_tolerance: float = CAP_TOLERANCE,
) -> np.ndarray:
    """Reconstruct leading ``log(sigma**2)`` values without Fock enumeration."""
    log_preferred, costs, _ = natural_spectrum_factors(
        occupations, cap_tolerance=cap_tolerance
    )
    return float(log_z) + log_preferred - lowest_subset_sums(costs, count)


def _binary_entropy(occupations: np.ndarray) -> float:
    nu = np.asarray(occupations, dtype=np.float64)
    mask = (nu > 0.0) & (nu < 1.0)
    return float(
        -np.sum(
            nu[mask] * np.log(nu[mask])
            + (1.0 - nu[mask]) * np.log1p(-nu[mask])
        )
    )


class BatchedActiveSpectrumObserver:
    """Accumulate record weights every cycle and selected active spectra."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        samples: int,
        active_indices: torch.Tensor,
        full_mode_count: int,
        wall_locations: tuple[int, int],
        soft_mode_count: int = 16,
        leading_level_count: int = 64,
        cap_tolerance: float = CAP_TOLERANCE,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples = int(samples)
        self.wall_locations = tuple(int(value) for value in wall_locations)
        self.soft_mode_count = int(soft_mode_count)
        self.leading_level_count = int(leading_level_count)
        self.cap_tolerance = float(cap_tolerance)
        self.active_indices = torch.as_tensor(
            active_indices, dtype=torch.long, device=active_indices.device
        ).reshape(-1)
        self.full_mode_count = int(full_mode_count)
        if self.cycles != 2 * self.ny:
            raise ValueError("production requires cycles=2*Ny")
        if self.wall_locations != (5, 15):
            raise ValueError("production requires hard walls at x=5,15")
        if int(self.active_indices.numel()) != 22 * self.ny:
            raise AssertionError("active transfer dimension must equal 22*Ny")
        mask = torch.ones(
            self.full_mode_count, dtype=torch.bool, device=self.active_indices.device
        )
        mask[self.active_indices] = False
        self.exterior_indices = torch.nonzero(mask, as_tuple=False).flatten()
        self.spectrum_cycles = spectrum_checkpoint_cycles(self.ny)
        self._spectrum_position = {
            int(cycle): index for index, cycle in enumerate(self.spectrum_cycles)
        }
        x_coordinates = (self.active_indices // 2) % self.nx
        self._x_projector = torch.nn.functional.one_hot(
            x_coordinates, num_classes=self.nx
        ).to(dtype=torch.float64)
        self._allocate()

    @property
    def expected_sites_per_cycle(self) -> int:
        return int((self.wall_locations[1] - self.wall_locations[0] + 1) * self.ny)

    def _allocate(self) -> None:
        total_cycles = self.cycles + 1
        checkpoints = int(self.spectrum_cycles.size)
        n_eff = int(self.active_indices.numel())
        soft = min(self.soft_mode_count, n_eff)
        self.cycle_seen = np.zeros((self.samples, total_cycles), dtype=bool)
        self.measurement_log_probability = np.zeros(
            (self.samples, total_cycles), dtype=np.float64
        )
        self.cumulative_log_probability = np.full(
            (self.samples, total_cycles), np.nan, dtype=np.float64
        )
        self.site_event_count = np.zeros(
            (self.samples, total_cycles), dtype=np.int64
        )
        self.channel_event_count = np.zeros(
            (self.samples, total_cycles), dtype=np.int64
        )
        self.spectrum_seen = np.zeros((self.samples, checkpoints), dtype=bool)
        self.occupations = np.full(
            (self.samples, checkpoints, n_eff), np.nan, dtype=np.float64
        )
        self.cap_mask = np.zeros(
            (self.samples, checkpoints, n_eff), dtype=bool
        )
        self.entropy_nats = np.full(
            (self.samples, checkpoints), np.nan, dtype=np.float64
        )
        self.charge_variance = np.full_like(self.entropy_nats, np.nan)
        self.log_z = np.full_like(self.entropy_nats, np.nan)
        self.leading_log_sigma2 = np.full(
            (self.samples, checkpoints, self.leading_level_count),
            np.nan,
            dtype=np.float64,
        )
        self.soft_mode_occupations = np.full(
            (self.samples, checkpoints, soft), np.nan, dtype=np.float64
        )
        self.soft_mode_flip_costs = np.full_like(
            self.soft_mode_occupations, np.nan
        )
        self.soft_mode_x_profiles = np.full(
            (self.samples, checkpoints, soft, self.nx),
            np.nan,
            dtype=np.float64,
        )
        self.soft_mode_wall_weights = np.full(
            (self.samples, checkpoints, soft, 2), np.nan, dtype=np.float64
        )
        scalar_shape = (self.samples, checkpoints)
        self.hermiticity_residual = np.full(scalar_shape, np.nan)
        self.eigensolver_residual = np.full(scalar_shape, np.nan)
        self.eigenvector_gram_residual = np.full(scalar_shape, np.nan)
        self.occupation_bound_residual = np.full(scalar_shape, np.nan)
        self.active_exterior_coupling_residual = np.full(scalar_shape, np.nan)
        self.exterior_product_residual = np.full(scalar_shape, np.nan)

    def record_event(
        self,
        *,
        cycle: int,
        sample_offsets: torch.Tensor,
        conditional_log_probability: torch.Tensor,
        **_: Any,
    ) -> None:
        cycle = int(cycle)
        if cycle < 1 or cycle > self.cycles:
            raise ValueError(f"record event has invalid cycle {cycle}")
        rows = torch.as_tensor(sample_offsets, dtype=torch.long).detach().cpu().numpy()
        conditional = (
            torch.as_tensor(conditional_log_probability, dtype=torch.float64)
            .detach()
            .cpu()
            .numpy()
        )
        if conditional.ndim != 2 or conditional.shape[0] != rows.size:
            raise ValueError("conditional log-probability payload has invalid shape")
        if np.any(rows < 0) or np.any(rows >= self.samples):
            raise IndexError("record sample offset is outside this task")
        if not np.all(np.isfinite(conditional)):
            raise FloatingPointError("record contains a nonfinite log probability")
        self.measurement_log_probability[rows, cycle] += conditional.sum(axis=1)
        self.site_event_count[rows, cycle] += 1
        self.channel_event_count[rows, cycle] += conditional.shape[1]

    def observe(
        self,
        *,
        cycle: int,
        G: torch.Tensor,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        cycle = int(cycle)
        if int(batch_start) != 0 or int(batch_count) != self.samples:
            raise RuntimeError("observer requires one engine batch per five-sample task")
        if np.any(self.cycle_seen[:, cycle]):
            raise RuntimeError(f"duplicate cycle observation at cycle {cycle}")
        if cycle == 0:
            self.cumulative_log_probability[:, 0] = 0.0
        else:
            if np.any(self.site_event_count[:, cycle] != self.expected_sites_per_cycle):
                raise RuntimeError(f"cycle {cycle} has incomplete site records")
            if np.any(
                self.channel_event_count[:, cycle]
                != 4 * self.expected_sites_per_cycle
            ):
                raise RuntimeError(f"cycle {cycle} has incomplete channel records")
            self.cumulative_log_probability[:, cycle] = (
                self.cumulative_log_probability[:, cycle - 1]
                + self.measurement_log_probability[:, cycle]
            )
        self.cycle_seen[:, cycle] = True
        position = self._spectrum_position.get(cycle)
        if position is not None:
            self._observe_spectrum(position, cycle, G)

    def _observe_spectrum(
        self, position: int, cycle: int, G: torch.Tensor
    ) -> None:
        full = torch.as_tensor(G)
        if full.dtype != torch.complex128:
            raise TypeError(f"production requires complex128 covariance, got {full.dtype}")
        if tuple(full.shape) != (
            self.samples,
            self.full_mode_count,
            self.full_mode_count,
        ):
            raise ValueError(f"unexpected covariance shape {tuple(full.shape)}")
        if not torch.isfinite(full).all():
            raise FloatingPointError("covariance contains nonfinite values")
        hermiticity = torch.amax(
            torch.abs(full - full.mH), dim=(-2, -1)
        )
        if torch.any(hermiticity > 1.0e-8):
            raise FloatingPointError("covariance Hermiticity residual exceeded 1e-8")
        active = full.index_select(1, self.active_indices).index_select(
            2, self.active_indices
        )
        active = 0.5 * (active + active.mH)
        eye = torch.eye(
            active.shape[-1], dtype=torch.complex128, device=active.device
        )
        correlation = 0.5 * (active + eye[None, :, :])
        correlation = 0.5 * (correlation + correlation.mH)
        eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
        if torch.any(eigenvalues < -self.cap_tolerance) or torch.any(
            eigenvalues > 1.0 + self.cap_tolerance
        ):
            extrema = (
                float(eigenvalues.min().item()),
                float(eigenvalues.max().item()),
            )
            raise FloatingPointError(
                "active occupation spectrum lies outside the roundoff allowance: "
                f"min={extrema[0]:.17g}, max={extrema[1]:.17g}, "
                f"tolerance={self.cap_tolerance:.3g}"
            )
        eigenvalues_np = eigenvalues.detach().cpu().numpy().astype(np.float64)
        bound_residual = np.maximum(
            np.maximum(-eigenvalues_np.min(axis=1), 0.0),
            np.maximum(eigenvalues_np.max(axis=1) - 1.0, 0.0),
        )
        eigenvalues_np[np.abs(eigenvalues_np) <= self.cap_tolerance] = 0.0
        eigenvalues_np[
            np.abs(eigenvalues_np - 1.0) <= self.cap_tolerance
        ] = 1.0

        exterior = full.index_select(1, self.exterior_indices).index_select(
            2, self.exterior_indices
        )
        coupling = full.index_select(1, self.active_indices).index_select(
            2, self.exterior_indices
        )
        exterior_diag = torch.diagonal(exterior, dim1=-2, dim2=-1)
        exterior_offdiag = exterior - torch.diag_embed(exterior_diag)
        coupling_scale = math.sqrt(max(1, coupling.shape[-1] * coupling.shape[-2]))
        exterior_scale = max(1, exterior.shape[-1])

        for sample in range(self.samples):
            nu = eigenvalues_np[sample]
            _, costs, caps = natural_spectrum_factors(
                nu, cap_tolerance=self.cap_tolerance
            )
            order = np.argsort(costs, kind="stable")[: self.soft_mode_count]
            order_t = torch.as_tensor(order, dtype=torch.long, device=full.device)
            vectors = eigenvectors[sample].index_select(1, order_t)
            weights = torch.abs(vectors).square().transpose(0, 1).to(torch.float64)
            profiles = weights @ self._x_projector
            left, right = self.wall_locations
            wall_weights = torch.stack(
                (
                    profiles[:, left : left + 2].sum(dim=1),
                    profiles[:, right - 1 : right + 1].sum(dim=1),
                ),
                dim=1,
            )
            selected_nu = eigenvalues[sample].index_select(0, order_t)
            residual = correlation[sample] @ vectors - vectors * selected_nu[None, :]
            gram = vectors.mH @ vectors
            gram_eye = torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            log_z = 22 * self.ny * math.log(2.0) + float(
                self.cumulative_log_probability[sample, cycle]
            )
            self.occupations[sample, position] = nu
            self.cap_mask[sample, position] = caps
            self.entropy_nats[sample, position] = _binary_entropy(nu)
            self.charge_variance[sample, position] = float(
                np.sum(nu * (1.0 - nu))
            )
            self.log_z[sample, position] = log_z
            self.leading_log_sigma2[sample, position] = leading_log_sigma2_levels(
                nu,
                log_z,
                count=self.leading_level_count,
                cap_tolerance=self.cap_tolerance,
            )
            self.soft_mode_occupations[sample, position] = nu[order]
            self.soft_mode_flip_costs[sample, position] = costs[order]
            self.soft_mode_x_profiles[sample, position] = (
                profiles.detach().cpu().numpy()
            )
            self.soft_mode_wall_weights[sample, position] = (
                wall_weights.detach().cpu().numpy()
            )
            self.hermiticity_residual[sample, position] = float(
                hermiticity[sample].item()
            )
            self.eigensolver_residual[sample, position] = float(
                torch.linalg.vector_norm(residual, dim=0).max().item()
            )
            self.eigenvector_gram_residual[sample, position] = float(
                torch.abs(gram - gram_eye).max().item()
            )
            self.occupation_bound_residual[sample, position] = float(
                bound_residual[sample]
            )
            self.active_exterior_coupling_residual[sample, position] = float(
                torch.linalg.matrix_norm(coupling[sample], ord="fro").item()
                / coupling_scale
            )
            diagonal_error = torch.abs(torch.abs(exterior_diag[sample]) - 1.0).max()
            offdiag_error = (
                torch.linalg.matrix_norm(exterior_offdiag[sample], ord="fro")
                / exterior_scale
            )
            self.exterior_product_residual[sample, position] = float(
                torch.maximum(diagonal_error, offdiag_error).item()
            )
            self.spectrum_seen[sample, position] = True

        if cycle == 0:
            if not np.allclose(eigenvalues_np, 0.5, atol=2.0e-10, rtol=0.0):
                raise AssertionError("cycle-zero active spectrum is not maximally mixed")
            if not np.allclose(
                self.leading_log_sigma2[:, position], 0.0, atol=2.0e-10
            ):
                raise AssertionError("cycle-zero transfer spectrum is not the identity")

    def validate(self) -> None:
        if not np.all(self.cycle_seen):
            raise RuntimeError("observer cycle history is incomplete")
        if not np.all(self.spectrum_seen):
            raise RuntimeError("observer spectrum history is incomplete")
        if np.any(self.site_event_count[:, 1:] != self.expected_sites_per_cycle):
            raise RuntimeError("site-event counts are incomplete")
        if np.any(
            self.channel_event_count[:, 1:] != 4 * self.expected_sites_per_cycle
        ):
            raise RuntimeError("channel-event counts are incomplete")
        if not np.all(np.isfinite(self.cumulative_log_probability)):
            raise FloatingPointError("cumulative record weights are incomplete")
        increments = np.diff(self.cumulative_log_probability, axis=1)
        if not np.allclose(
            increments,
            self.measurement_log_probability[:, 1:],
            atol=2.0e-10,
            rtol=2.0e-12,
        ):
            raise AssertionError("cycle record weights do not sum to cumulative weight")

    def result_arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        names = (
            "cycle_seen",
            "measurement_log_probability",
            "cumulative_log_probability",
            "site_event_count",
            "channel_event_count",
            "spectrum_seen",
            "occupations",
            "cap_mask",
            "entropy_nats",
            "charge_variance",
            "log_z",
            "leading_log_sigma2",
            "soft_mode_occupations",
            "soft_mode_flip_costs",
            "soft_mode_x_profiles",
            "soft_mode_wall_weights",
            "hermiticity_residual",
            "eigensolver_residual",
            "eigenvector_gram_residual",
            "occupation_bound_residual",
            "active_exterior_coupling_residual",
            "exterior_product_residual",
        )
        result = {name: np.array(getattr(self, name), copy=True) for name in names}
        result["cycles"] = np.arange(self.cycles + 1, dtype=np.int64)
        result["normalized_cycles"] = result["cycles"] / float(self.ny)
        result["spectrum_cycles"] = self.spectrum_cycles.copy()
        return result
