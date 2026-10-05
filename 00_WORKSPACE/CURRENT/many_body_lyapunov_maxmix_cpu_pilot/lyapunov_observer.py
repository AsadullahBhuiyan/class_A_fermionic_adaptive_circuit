"""Compact active-space spectrum observer for the Nx=16 hard/soft CPU run."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


OBSERVER_SCHEMA = "maxmix_active_spectrum_observer_v2"
CAP_TOLERANCE = 1.0e-12


def spectrum_checkpoint_cycles(ny: int, cycles_multiplier: int = 4) -> np.ndarray:
    """Return stride-four checkpoints plus integer-``Ny`` fit boundaries."""
    ny = int(ny)
    cycles_multiplier = int(cycles_multiplier)
    if ny <= 0 or ny % 2:
        raise ValueError("Ny must be a positive even integer")
    if cycles_multiplier < 1:
        raise ValueError("cycles_multiplier must be positive")
    terminal = cycles_multiplier * ny
    boundaries = {multiple * ny for multiple in range(1, cycles_multiplier + 1)}
    if cycles_multiplier == 2:
        boundaries.add(3 * ny // 2)
    return np.asarray(
        sorted(set(range(0, terminal + 1, 4)) | boundaries),
        dtype=np.int64,
    )


def natural_spectrum_factors(
    occupations: np.ndarray, *, cap_tolerance: float = CAP_TOLERANCE
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return log preferred weight, squared-singular flip costs, and cap mask.

    The returned costs use ``ell=log(sigma**2)``. Exact occupied/empty caps
    have infinite flip cost and are never replaced by finite clipped values.
    """
    nu = np.asarray(occupations, dtype=np.float64)
    if nu.ndim != 1 or not np.all(np.isfinite(nu)):
        raise ValueError("occupations must be one finite one-dimensional array")
    if np.min(nu, initial=0.0) < -cap_tolerance or np.max(nu, initial=1.0) > 1 + cap_tolerance:
        raise FloatingPointError("occupation spectrum lies outside [0,1]")
    empty_cap = nu <= cap_tolerance
    full_cap = nu >= 1.0 - cap_tolerance
    caps = empty_cap | full_cap
    interior = ~caps
    preferred = np.zeros_like(nu)
    preferred[interior] = np.maximum(nu[interior], 1.0 - nu[interior])
    log_preferred = float(np.log(preferred[interior]).sum())
    costs = np.full(nu.shape, np.inf, dtype=np.float64)
    costs[interior] = np.abs(np.log(nu[interior]) - np.log1p(-nu[interior]))
    return log_preferred, costs, caps


def lowest_subset_sums(costs: np.ndarray, count: int) -> np.ndarray:
    """Return the ``count`` smallest subset sums of nonnegative finite costs."""
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
    """Reconstruct leading many-body ``log(sigma_i**2)`` levels."""
    log_preferred, costs, _ = natural_spectrum_factors(
        occupations, cap_tolerance=cap_tolerance
    )
    return float(log_z) + log_preferred - lowest_subset_sums(costs, count)


def _binary_entropy(occupations: np.ndarray) -> float:
    nu = np.asarray(occupations, dtype=np.float64)
    mask = (nu > 0.0) & (nu < 1.0)
    return float(
        -np.sum(nu[mask] * np.log(nu[mask]) + (1.0 - nu[mask]) * np.log1p(-nu[mask]))
    )


@dataclass(frozen=True)
class ActiveGeometry:
    nx: int
    ny: int
    active_indices: np.ndarray
    wall_locations: tuple[int, int]

    @property
    def active_mode_count(self) -> int:
        return int(self.active_indices.size)

    @property
    def expected_sites_per_cycle(self) -> int:
        return int((self.wall_locations[1] - self.wall_locations[0] + 1) * self.ny)


class ActiveSpectrumObserver:
    """Record probability every cycle and active spectra at selected cycles."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        active_indices: np.ndarray,
        wall_locations: tuple[int, int],
        construction: str = "hard",
        spectrum_cycles: np.ndarray | None = None,
        soft_mode_count: int = 16,
        leading_level_count: int = 64,
        cap_tolerance: float = CAP_TOLERANCE,
    ) -> None:
        self.geometry = ActiveGeometry(
            nx=int(nx),
            ny=int(ny),
            active_indices=np.asarray(active_indices, dtype=np.int64),
            wall_locations=tuple(int(value) for value in wall_locations),
        )
        self.cycles = int(cycles)
        self.construction = str(construction)
        self.spectrum_cycles = (
            spectrum_checkpoint_cycles(ny, self.cycles // int(ny))
            if spectrum_cycles is None
            else np.asarray(spectrum_cycles, dtype=np.int64)
        )
        self.soft_mode_count = int(soft_mode_count)
        self.leading_level_count = int(leading_level_count)
        self.cap_tolerance = float(cap_tolerance)
        if self.construction not in {"hard", "soft"}:
            raise ValueError("construction must be hard or soft")
        if self.cycles <= 0 or self.cycles % int(ny):
            raise ValueError("cycles must be a positive integer multiple of Ny")
        if not np.array_equal(self.spectrum_cycles, np.unique(self.spectrum_cycles)):
            raise ValueError("spectrum cycles must be sorted and unique")
        if self.spectrum_cycles[0] != 0 or self.spectrum_cycles[-1] != self.cycles:
            raise ValueError("spectrum cycles must include zero and the final cycle")
        if np.any((self.spectrum_cycles < 0) | (self.spectrum_cycles > self.cycles)):
            raise ValueError("spectrum cycle outside trajectory range")
        expected_active = (
            2 * (self.geometry.wall_locations[1] - self.geometry.wall_locations[0] + 1) * int(ny)
            if self.construction == "hard"
            else 2 * int(nx) * int(ny)
        )
        if self.geometry.active_mode_count != expected_active:
            raise AssertionError(
                f"active transfer dimension {self.geometry.active_mode_count} != 22*Ny={expected_active}"
            )
        self._spectrum_position = {
            int(cycle): index for index, cycle in enumerate(self.spectrum_cycles)
        }
        self._allocate()

    def _allocate(self) -> None:
        total = self.cycles + 1
        checkpoints = self.spectrum_cycles.size
        n_eff = self.geometry.active_mode_count
        soft = min(self.soft_mode_count, n_eff)
        self.cycle_seen = np.zeros(total, dtype=bool)
        self.cumulative_log_probability = np.full(total, np.nan, dtype=np.float64)
        self.measurement_log_probability = np.zeros(total, dtype=np.float64)
        self.correction_log_probability = np.zeros(total, dtype=np.float64)
        self.site_event_count = np.zeros(total, dtype=np.int64)
        self.spectrum_seen = np.zeros(checkpoints, dtype=bool)
        self.occupations = np.full((checkpoints, n_eff), np.nan, dtype=np.float64)
        self.cap_mask = np.zeros((checkpoints, n_eff), dtype=bool)
        self.entropy_nats = np.full(checkpoints, np.nan, dtype=np.float64)
        self.charge_variance = np.full(checkpoints, np.nan, dtype=np.float64)
        self.log_z = np.full(checkpoints, np.nan, dtype=np.float64)
        self.leading_log_sigma2 = np.full(
            (checkpoints, self.leading_level_count), np.nan, dtype=np.float64
        )
        self.soft_mode_occupations = np.full((checkpoints, soft), np.nan, dtype=np.float64)
        self.soft_mode_flip_costs = np.full((checkpoints, soft), np.nan, dtype=np.float64)
        self.soft_mode_x_profiles = np.full(
            (checkpoints, soft, self.geometry.nx), np.nan, dtype=np.float64
        )
        self.soft_mode_wall_weights = np.full((checkpoints, soft, 2), np.nan, dtype=np.float64)
        self.hermiticity_residual = np.full(checkpoints, np.nan, dtype=np.float64)
        self.eigensolver_residual = np.full(checkpoints, np.nan, dtype=np.float64)
        self.eigenvector_gram_residual = np.full(checkpoints, np.nan, dtype=np.float64)
        self.active_exterior_coupling_residual = np.full(checkpoints, np.nan, dtype=np.float64)
        self.exterior_product_residual = np.full(checkpoints, np.nan, dtype=np.float64)

    @property
    def expected_sites_per_cycle(self) -> int:
        if self.construction == "hard":
            return self.geometry.expected_sites_per_cycle
        return int(self.geometry.nx * self.geometry.ny)

    def record_site(
        self,
        *,
        cycle: int,
        measurement_log_weight: float,
        correction_log_weight: float,
        cumulative_log_weight: float,
        forced_postselect: bool,
        **_: Any,
    ) -> None:
        cycle = int(cycle)
        if cycle < 1 or cycle > self.cycles:
            raise ValueError(f"site event has invalid cycle {cycle}")
        if forced_postselect:
            raise AssertionError("pilot forbids postselection")
        self.site_event_count[cycle] += 1
        self.measurement_log_probability[cycle] += float(measurement_log_weight)
        self.correction_log_probability[cycle] += float(correction_log_weight)
        self.cumulative_log_probability[cycle] = float(cumulative_log_weight)

    def observe(self, *, cycle: int, G: np.ndarray, **_: Any) -> None:
        cycle = int(cycle)
        if self.cycle_seen[cycle]:
            raise RuntimeError(f"duplicate observation at cycle {cycle}")
        if cycle == 0:
            self.cumulative_log_probability[0] = 0.0
        elif self.site_event_count[cycle] != self.expected_sites_per_cycle:
            raise RuntimeError(
                f"cycle {cycle}: received {self.site_event_count[cycle]} site events, "
                f"expected {self.expected_sites_per_cycle}"
            )
        self.cycle_seen[cycle] = True
        if cycle in self._spectrum_position:
            self._observe_spectrum(self._spectrum_position[cycle], cycle, G)

    def _observe_spectrum(self, position: int, cycle: int, G: np.ndarray) -> None:
        raw = np.asarray(G)
        if raw.dtype != np.dtype(np.complex128):
            raise TypeError(f"pilot requires complex128 covariance, received {raw.dtype}")
        full = np.asarray(raw, dtype=np.complex128)
        if full.ndim != 2 or full.shape[0] != full.shape[1] or not np.all(np.isfinite(full)):
            raise FloatingPointError("full shifted covariance is invalid")
        hermiticity = float(np.max(np.abs(full - full.conj().T)))
        if hermiticity > 1.0e-8:
            raise FloatingPointError(f"Hermiticity residual {hermiticity:.3e}")
        active_indices = self.geometry.active_indices
        active = full[np.ix_(active_indices, active_indices)]
        active = 0.5 * (active + active.conj().T)
        occupation = 0.5 * (
            np.eye(active.shape[0], dtype=np.complex128) + active
        )
        occupation = 0.5 * (occupation + occupation.conj().T)
        nu, vectors = np.linalg.eigh(occupation)
        nu = np.asarray(np.real_if_close(nu), dtype=np.float64)
        if nu.min() < -1.0e-9 or nu.max() > 1.0 + 1.0e-9:
            raise FloatingPointError(
                f"occupation spectrum outside [0,1]: [{nu.min():.3e}, {nu.max():.3e}]"
            )
        nu[np.abs(nu) <= self.cap_tolerance] = 0.0
        nu[np.abs(nu - 1.0) <= self.cap_tolerance] = 1.0
        log_preferred, costs, caps = natural_spectrum_factors(
            nu, cap_tolerance=self.cap_tolerance
        )
        del log_preferred
        log_z = self.geometry.active_mode_count * math.log(2.0) + float(
            self.cumulative_log_probability[cycle]
        )
        order = np.argsort(costs, kind="stable")[: self.soft_mode_occupations.shape[1]]
        selected_vectors = vectors[:, order]
        x_coordinates = (active_indices // 2) % self.geometry.nx
        profiles = np.stack(
            [
                np.bincount(
                    x_coordinates,
                    weights=np.abs(selected_vectors[:, column]) ** 2,
                    minlength=self.geometry.nx,
                )
                for column in range(selected_vectors.shape[1])
            ],
            axis=0,
        )
        left, right = self.geometry.wall_locations
        wall_weights = np.stack(
            (
                profiles[:, [left, left + 1]].sum(axis=1),
                profiles[:, [right - 1, right]].sum(axis=1),
            ),
            axis=1,
        )
        residuals = occupation @ selected_vectors - selected_vectors * nu[order][None, :]
        gram = selected_vectors.conj().T @ selected_vectors
        all_indices = np.arange(full.shape[0], dtype=np.int64)
        exterior_indices = np.setdiff1d(all_indices, active_indices, assume_unique=True)
        coupling = full[np.ix_(active_indices, exterior_indices)]
        exterior = full[np.ix_(exterior_indices, exterior_indices)]
        exterior_offdiag = exterior - np.diag(np.diag(exterior))
        exterior_residual = max(
            float(np.max(np.abs(np.abs(np.diag(exterior)) - 1.0), initial=0.0)),
            float(np.linalg.norm(exterior_offdiag, "fro") / max(1, exterior.shape[0])),
        )
        self.occupations[position] = nu
        self.cap_mask[position] = caps
        self.entropy_nats[position] = _binary_entropy(nu)
        self.charge_variance[position] = float(np.sum(nu * (1.0 - nu)))
        self.log_z[position] = log_z
        self.leading_log_sigma2[position] = leading_log_sigma2_levels(
            nu, log_z, count=self.leading_level_count, cap_tolerance=self.cap_tolerance
        )
        self.soft_mode_occupations[position] = nu[order]
        self.soft_mode_flip_costs[position] = costs[order]
        self.soft_mode_x_profiles[position] = profiles
        self.soft_mode_wall_weights[position] = wall_weights
        self.hermiticity_residual[position] = hermiticity
        self.eigensolver_residual[position] = float(
            np.max(np.linalg.norm(residuals, axis=0), initial=0.0)
        )
        self.eigenvector_gram_residual[position] = float(
            np.max(np.abs(gram - np.eye(gram.shape[0])), initial=0.0)
        )
        self.active_exterior_coupling_residual[position] = float(
            np.linalg.norm(coupling, "fro") / math.sqrt(max(1, coupling.size))
        )
        self.exterior_product_residual[position] = exterior_residual
        self.spectrum_seen[position] = True
        if cycle == 0:
            if not np.allclose(nu, 0.5, atol=2.0e-10, rtol=0.0):
                raise AssertionError("cycle-zero active spectrum is not maximally mixed")
            if not np.allclose(self.leading_log_sigma2[position], 0.0, atol=2.0e-10):
                raise AssertionError("cycle-zero transfer operator is not the identity spectrum")

    def validate(self, *, completed_cycle: int | None = None) -> None:
        completed = self.cycles if completed_cycle is None else int(completed_cycle)
        if not np.all(self.cycle_seen[: completed + 1]):
            raise RuntimeError("observer cycle prefix is incomplete")
        if np.any(self.cycle_seen[completed + 1 :]):
            raise RuntimeError("observer contains cycles beyond checkpoint")
        expected_spectra = self.spectrum_cycles <= completed
        if not np.array_equal(self.spectrum_seen, expected_spectra):
            raise RuntimeError("observer spectrum prefix is incomplete or noncontiguous")
        expected_sites = self.expected_sites_per_cycle
        if completed and not np.all(self.site_event_count[1 : completed + 1] == expected_sites):
            raise RuntimeError("record event counts are incomplete")
        if not np.all(np.isfinite(self.cumulative_log_probability[: completed + 1])):
            raise FloatingPointError("record log probability is incomplete")
        if np.max(np.abs(self.correction_log_probability[: completed + 1]), initial=0.0) > 1.0e-12:
            raise AssertionError("perfect correction unexpectedly contributed stochastic log weight")
        if completed:
            increments = np.diff(self.cumulative_log_probability[: completed + 1])
            branch_totals = (
                self.measurement_log_probability[1 : completed + 1]
                + self.correction_log_probability[1 : completed + 1]
            )
            if not np.allclose(increments, branch_totals, atol=2.0e-10, rtol=2.0e-12):
                raise AssertionError("per-cycle record weights do not sum to cumulative log probability")

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        names = (
            "cycle_seen",
            "cumulative_log_probability",
            "measurement_log_probability",
            "correction_log_probability",
            "site_event_count",
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
            "active_exterior_coupling_residual",
            "exterior_product_residual",
        )
        return {f"observer_{name}": np.array(getattr(self, name), copy=True) for name in names}

    def restore_checkpoint_arrays(
        self, arrays: Mapping[str, np.ndarray], *, completed_cycle: int
    ) -> None:
        for key, source in self.checkpoint_arrays().items():
            if key not in arrays:
                raise ValueError(f"checkpoint is missing {key}")
            target = getattr(self, key.removeprefix("observer_"))
            value = np.asarray(arrays[key])
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"checkpoint {key} has incompatible shape or dtype")
            target[...] = value
        self.validate(completed_cycle=completed_cycle)

    def result_arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        arrays = self.checkpoint_arrays()
        return {key.removeprefix("observer_"): value for key, value in arrays.items()}
