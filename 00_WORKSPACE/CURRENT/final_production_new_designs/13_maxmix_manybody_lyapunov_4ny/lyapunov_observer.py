"""Checkpointable hard/soft many-body-spectrum observer for bundle 13."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "maxmix_active_spectrum_gpu_observer_4ny_v1"
# At production dimension (N_eff=880), CUDA complex128 ``eigh`` can place an
# exact covariance cap a few 1e-10 outside [0, 1].  Use one tolerance for both
# the physicality guard and exact-cap classification so the two checks cannot
# contradict one another.  Values beyond this scale still fail closed.
CAP_TOLERANCE = 1.0e-9


def spectrum_checkpoint_cycles(ny: int) -> np.ndarray:
    """Return stride-four checkpoints through ``4*Ny`` and fit boundaries."""
    ny = int(ny)
    if ny <= 0 or ny % 2:
        raise ValueError("Ny must be a positive even integer")
    return np.asarray(
        sorted(set(range(0, 4 * ny + 1, 4)) | {ny, 2 * ny, 3 * ny, 4 * ny}),
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
        construction: str,
        sample_indices: np.ndarray,
        sample_chunk: int,
        soft_mode_count: int = 16,
        leading_level_count: int = 64,
        cap_tolerance: float = CAP_TOLERANCE,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.samples = int(samples)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.sample_chunk = int(sample_chunk)
        self.construction = str(construction)
        self.wall_locations = tuple(int(value) for value in wall_locations)
        self.soft_mode_count = int(soft_mode_count)
        self.leading_level_count = int(leading_level_count)
        self.cap_tolerance = float(cap_tolerance)
        self.active_indices = torch.as_tensor(
            active_indices, dtype=torch.long, device=active_indices.device
        ).reshape(-1)
        self.full_mode_count = int(full_mode_count)
        if self.cycles != 4 * self.ny:
            raise ValueError("production requires cycles=4*Ny")
        if self.wall_locations != (5, 15):
            raise ValueError("production requires walls at x=5,15")
        if self.construction not in {"hard", "soft"}:
            raise ValueError("construction must be hard or soft")
        if self.sample_indices.shape != (self.samples,):
            raise ValueError("sample_indices must identify every resident trajectory")
        if self.sample_chunk < 1:
            raise ValueError("sample_chunk must be positive")
        expected_active = (22 if self.construction == "hard" else 40) * self.ny
        if int(self.active_indices.numel()) != expected_active:
            raise AssertionError(
                f"active transfer dimension must equal {expected_active}"
            )
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
        self._record_log_probability_gpu: torch.Tensor | None = None
        self._site_event_count_gpu: torch.Tensor | None = None
        self._channel_event_count_gpu: torch.Tensor | None = None

    @property
    def expected_sites_per_cycle(self) -> int:
        if self.construction == "hard":
            return int(
                (self.wall_locations[1] - self.wall_locations[0] + 1) * self.ny
            )
        return int(self.nx * self.ny)

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
        if np.any(self.cycle_seen[:, cycle]):
            raise RuntimeError(f"record event arrived after cycle {cycle} was finalized")
        conditional = torch.as_tensor(
            conditional_log_probability, dtype=torch.float64
        )
        rows = torch.as_tensor(
            sample_offsets, dtype=torch.long, device=conditional.device
        ).reshape(-1)
        if conditional.ndim != 2 or conditional.shape[0] != rows.numel():
            raise ValueError("conditional log-probability payload has invalid shape")
        # Do not synchronize CUDA for every site. ``index_add_`` rejects an
        # invalid sample offset, and finiteness is checked once per cycle when
        # the accumulated record weight is copied to CPU.
        if self._record_log_probability_gpu is None:
            shape = (self.samples, self.cycles + 1)
            self._record_log_probability_gpu = torch.zeros(
                shape, dtype=torch.float64, device=conditional.device
            )
            self._site_event_count_gpu = torch.zeros(
                shape, dtype=torch.int64, device=conditional.device
            )
            self._channel_event_count_gpu = torch.zeros(
                shape, dtype=torch.int64, device=conditional.device
            )
        elif self._record_log_probability_gpu.device != conditional.device:
            raise RuntimeError("record observer device changed during execution")
        assert self._site_event_count_gpu is not None
        assert self._channel_event_count_gpu is not None
        self._record_log_probability_gpu[:, cycle].index_add_(
            0, rows, conditional.sum(dim=1)
        )
        self._site_event_count_gpu[:, cycle].index_add_(
            0, rows, torch.ones_like(rows, dtype=torch.int64)
        )
        self._channel_event_count_gpu[:, cycle].index_add_(
            0,
            rows,
            torch.full_like(rows, int(conditional.shape[1]), dtype=torch.int64),
        )

    def _finalize_record_cycle(self, cycle: int) -> None:
        if cycle == 0:
            self.measurement_log_probability[:, 0] = 0.0
            self.cumulative_log_probability[:, 0] = 0.0
            return
        if self._record_log_probability_gpu is None:
            raise RuntimeError(f"cycle {cycle} has no record-observer events")
        assert self._site_event_count_gpu is not None
        assert self._channel_event_count_gpu is not None
        measurement = (
            self._record_log_probability_gpu[:, cycle].detach().cpu().numpy()
        )
        site_count = self._site_event_count_gpu[:, cycle].detach().cpu().numpy()
        channel_count = (
            self._channel_event_count_gpu[:, cycle].detach().cpu().numpy()
        )
        if not np.all(np.isfinite(measurement)):
            raise FloatingPointError("record contains a nonfinite log probability")
        if np.any(site_count != self.expected_sites_per_cycle):
            raise RuntimeError(f"cycle {cycle} has incomplete site records")
        if np.any(channel_count != 4 * self.expected_sites_per_cycle):
            raise RuntimeError(f"cycle {cycle} has incomplete channel records")
        self.measurement_log_probability[:, cycle] = measurement
        self.site_event_count[:, cycle] = site_count
        self.channel_event_count[:, cycle] = channel_count
        self.cumulative_log_probability[:, cycle] = (
            self.cumulative_log_probability[:, cycle - 1] + measurement
        )

    def observe(
        self,
        *,
        cycle: int,
        G: torch.Tensor,
        batch_start: int = 0,
        batch_count: int | None = None,
        **_: Any,
    ) -> None:
        cycle = int(cycle)
        if int(batch_start) != 0 or (
            batch_count is not None and int(batch_count) != self.samples
        ):
            raise RuntimeError("observer requires one engine batch for all resident samples")
        if np.any(self.cycle_seen[:, cycle]):
            raise RuntimeError(f"duplicate cycle observation at cycle {cycle}")
        self._finalize_record_cycle(cycle)
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
        for start in range(0, self.samples, self.sample_chunk):
            stop = min(self.samples, start + self.sample_chunk)
            chunk = full[start:stop]
            hermiticity = torch.amax(torch.abs(chunk - chunk.mH), dim=(-2, -1))
            if bool(torch.any(hermiticity > 1.0e-8).item()):
                raise FloatingPointError(
                    "covariance Hermiticity residual exceeded 1e-8"
                )
            active = chunk.index_select(1, self.active_indices).index_select(
                2, self.active_indices
            )
            active = 0.5 * (active + active.mH)
            eye = torch.eye(
                active.shape[-1], dtype=torch.complex128, device=active.device
            )
            correlation = 0.5 * (active + eye[None, :, :])
            correlation = 0.5 * (correlation + correlation.mH)
            eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
            if bool(torch.any(eigenvalues < -self.cap_tolerance).item()) or bool(
                torch.any(eigenvalues > 1.0 + self.cap_tolerance).item()
            ):
                raise FloatingPointError(
                    "active occupation spectrum lies outside the roundoff allowance: "
                    f"min={float(eigenvalues.min().item()):.17g}, "
                    f"max={float(eigenvalues.max().item()):.17g}, "
                    f"tolerance={self.cap_tolerance:.3g}"
                )
            raw_nu = eigenvalues.detach().cpu().numpy().astype(np.float64)
            bound_residual = np.maximum(
                np.maximum(-raw_nu.min(axis=1), 0.0),
                np.maximum(raw_nu.max(axis=1) - 1.0, 0.0),
            )
            saved_nu = raw_nu.copy()
            saved_nu[np.abs(saved_nu) <= self.cap_tolerance] = 0.0
            saved_nu[np.abs(saved_nu - 1.0) <= self.cap_tolerance] = 1.0

            if int(self.exterior_indices.numel()):
                exterior = chunk.index_select(1, self.exterior_indices).index_select(
                    2, self.exterior_indices
                )
                coupling = chunk.index_select(1, self.active_indices).index_select(
                    2, self.exterior_indices
                )
                exterior_diag = torch.diagonal(exterior, dim1=-2, dim2=-1)
                exterior_offdiag = exterior - torch.diag_embed(exterior_diag)
                coupling_scale = math.sqrt(
                    max(1, coupling.shape[-1] * coupling.shape[-2])
                )
                exterior_scale = max(1, exterior.shape[-1])
            else:
                exterior = coupling = exterior_diag = exterior_offdiag = None
                coupling_scale = exterior_scale = 1

            for local_sample in range(stop - start):
                sample = start + local_sample
                nu = saved_nu[local_sample]
                _, costs, caps = natural_spectrum_factors(
                    nu, cap_tolerance=self.cap_tolerance
                )
                order = np.argsort(costs, kind="stable")[: self.soft_mode_count]
                order_t = torch.as_tensor(
                    order, dtype=torch.long, device=full.device
                )
                vectors = eigenvectors[local_sample].index_select(1, order_t)
                weights = (
                    torch.abs(vectors).square().transpose(0, 1).to(torch.float64)
                )
                profiles = weights @ self._x_projector
                left, right = self.wall_locations
                wall_weights = torch.stack(
                    (
                        profiles[:, left : left + 2].sum(dim=1),
                        profiles[:, right - 1 : right + 1].sum(dim=1),
                    ),
                    dim=1,
                )
                selected_nu = eigenvalues[local_sample].index_select(0, order_t)
                residual = (
                    correlation[local_sample] @ vectors
                    - vectors * selected_nu[None, :]
                )
                gram = vectors.mH @ vectors
                gram_eye = torch.eye(
                    gram.shape[0], dtype=gram.dtype, device=gram.device
                )
                log_z = int(self.active_indices.numel()) * math.log(2.0) + float(
                    self.cumulative_log_probability[sample, cycle]
                )
                self.occupations[sample, position] = nu
                self.cap_mask[sample, position] = caps
                self.entropy_nats[sample, position] = _binary_entropy(nu)
                self.charge_variance[sample, position] = float(
                    np.sum(nu * (1.0 - nu))
                )
                self.log_z[sample, position] = log_z
                self.leading_log_sigma2[sample, position] = (
                    leading_log_sigma2_levels(
                        nu,
                        log_z,
                        count=self.leading_level_count,
                        cap_tolerance=self.cap_tolerance,
                    )
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
                    hermiticity[local_sample].item()
                )
                self.eigensolver_residual[sample, position] = float(
                    torch.linalg.vector_norm(residual, dim=0).max().item()
                )
                self.eigenvector_gram_residual[sample, position] = float(
                    torch.abs(gram - gram_eye).max().item()
                )
                self.occupation_bound_residual[sample, position] = float(
                    bound_residual[local_sample]
                )
                if coupling is None:
                    coupling_residual = exterior_residual = 0.0
                else:
                    coupling_residual = float(
                        torch.linalg.matrix_norm(
                            coupling[local_sample], ord="fro"
                        ).item()
                        / coupling_scale
                    )
                    assert exterior_diag is not None
                    assert exterior_offdiag is not None
                    diagonal_error = torch.abs(
                        torch.abs(exterior_diag[local_sample]) - 1.0
                    ).max()
                    offdiag_error = (
                        torch.linalg.matrix_norm(
                            exterior_offdiag[local_sample], ord="fro"
                        )
                        / exterior_scale
                    )
                    exterior_residual = float(
                        torch.maximum(diagonal_error, offdiag_error).item()
                    )
                self.active_exterior_coupling_residual[sample, position] = (
                    coupling_residual
                )
                self.exterior_product_residual[sample, position] = exterior_residual
                self.spectrum_seen[sample, position] = True

        if cycle == 0:
            if not np.allclose(
                self.occupations[:, position], 0.5, atol=2.0e-10, rtol=0.0
            ):
                raise AssertionError("cycle-zero active spectrum is not maximally mixed")
            if not np.allclose(
                self.leading_log_sigma2[:, position], 0.0, atol=2.0e-10
            ):
                raise AssertionError("cycle-zero transfer spectrum is not the identity")

    ARRAY_NAMES = (
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

    def validate(self, *, completed_cycle: int | None = None) -> None:
        completed = self.cycles if completed_cycle is None else int(completed_cycle)
        if completed < 0 or completed > self.cycles:
            raise ValueError("completed_cycle lies outside the configured horizon")
        expected_cycles = np.arange(self.cycles + 1) <= completed
        if not np.all(self.cycle_seen == expected_cycles[None, :]):
            raise RuntimeError("observer cycle history is not a contiguous prefix")
        expected_spectra = self.spectrum_cycles <= completed
        if not np.all(self.spectrum_seen == expected_spectra[None, :]):
            raise RuntimeError("observer spectrum history is not a contiguous prefix")
        if np.any(
            self.site_event_count[:, 1 : completed + 1]
            != self.expected_sites_per_cycle
        ):
            raise RuntimeError("site-event counts are incomplete")
        if np.any(
            self.channel_event_count[:, 1 : completed + 1]
            != 4 * self.expected_sites_per_cycle
        ):
            raise RuntimeError("channel-event counts are incomplete")
        if not np.all(
            np.isfinite(self.cumulative_log_probability[:, : completed + 1])
        ):
            raise FloatingPointError("cumulative record weights are incomplete")
        increments = np.diff(
            self.cumulative_log_probability[:, : completed + 1], axis=1
        )
        if not np.allclose(
            increments,
            self.measurement_log_probability[:, 1 : completed + 1],
            atol=2.0e-10,
            rtol=2.0e-12,
        ):
            raise AssertionError("cycle record weights do not sum to cumulative weight")

    def checkpoint_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "observer_sample_indices": self.sample_indices,
        }
        payload.update(
            {f"observer_{name}": np.asarray(getattr(self, name)) for name in self.ARRAY_NAMES}
        )
        return payload

    def restore_checkpoint(
        self, payload: dict[str, np.ndarray], *, completed_cycle: int
    ) -> None:
        if str(np.asarray(payload["observer_schema"]).item()) != OBSERVER_SCHEMA:
            raise RuntimeError("observer checkpoint schema mismatch")
        if not np.array_equal(
            np.asarray(payload["observer_sample_indices"], dtype=np.int64),
            self.sample_indices,
        ):
            raise RuntimeError("observer checkpoint sample IDs mismatch")
        for name in self.ARRAY_NAMES:
            saved = np.asarray(payload[f"observer_{name}"])
            target = getattr(self, name)
            if saved.shape != target.shape or saved.dtype != target.dtype:
                raise RuntimeError(f"observer checkpoint {name} shape/dtype mismatch")
            target[...] = saved
        self.validate(completed_cycle=completed_cycle)

    def result_arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        result = {
            name: np.array(getattr(self, name), copy=True)
            for name in self.ARRAY_NAMES
        }
        result["cycles"] = np.arange(self.cycles + 1, dtype=np.int64)
        result["normalized_cycles"] = result["cycles"] / float(self.ny)
        result["spectrum_cycles"] = self.spectrum_cycles.copy()
        result["sample_indices"] = self.sample_indices.copy()
        result["construction"] = np.asarray(self.construction)
        result["transfer_mode_count"] = np.asarray(
            int(self.active_indices.numel()), dtype=np.int64
        )
        result["log_probability_origin"] = np.asarray(
            "after_born_conditioned_exterior_preparation"
            if self.construction == "hard"
            else "global_maxmix_cycle_zero"
        )
        result["log_z_formula"] = np.asarray(
            "log_Z=N_eff*log(2)+cumulative_log_probability"
        )
        result["squared_singular_value_convention"] = np.asarray(
            "ell_i=log(sigma_i^2)"
        )
        return result

    def result_payload(self, sample_slice: slice) -> dict[str, np.ndarray]:
        self.validate()
        per_sample = {
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
            "sample_indices",
        }
        result = {
            name: np.asarray(getattr(self, name))
            for name in self.ARRAY_NAMES
        }
        result.update(
            {
                "cycles": np.arange(self.cycles + 1, dtype=np.int64),
                "normalized_cycles": np.arange(self.cycles + 1, dtype=np.float64)
                / float(self.ny),
                "spectrum_cycles": self.spectrum_cycles,
                "sample_indices": self.sample_indices,
                "construction": np.asarray(self.construction),
                "transfer_mode_count": np.asarray(
                    int(self.active_indices.numel()), dtype=np.int64
                ),
                "log_probability_origin": np.asarray(
                    "after_born_conditioned_exterior_preparation"
                    if self.construction == "hard"
                    else "global_maxmix_cycle_zero"
                ),
                "log_z_formula": np.asarray(
                    "log_Z=N_eff*log(2)+cumulative_log_probability"
                ),
                "squared_singular_value_convention": np.asarray(
                    "ell_i=log(sigma_i^2)"
                ),
            }
        )
        return {
            key: np.array(
                value[sample_slice] if key in per_sample else value,
                copy=True,
            )
            for key, value in result.items()
        }
