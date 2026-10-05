from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from production_runtime import save_npz_atomic, sha256_file

try:
    from streaming_covariance_observables_gpu import (
        build_chern_partition_indices,
        entropy_total_batch_torch,
        gather_restricted_covariance,
        local_chern_marker_batch_torch,
        real_space_chern_batch_torch,
        strip_mode_indices,
        xavg_square_correlator_batch_torch,
        build_square_correlator_pair_indices,
    )
except ImportError:
    build_chern_partition_indices = None


OBSERVER_SCHEMA = "selected_covariance_observables_v3"


def covariance_storage_bytes(
    *, nx: int, ny: int, samples: int, snapshots: int, representation: str = "dense"
) -> int:
    nlayer = 2 * int(nx) * int(ny)
    if representation == "dense":
        elements = nlayer * nlayer
    elif representation == "hermitian_triangle":
        elements = nlayer * (nlayer + 1) // 2
    else:
        raise ValueError("representation must be dense or hermitian_triangle")
    return int(samples) * int(snapshots) * elements * np.dtype(np.complex128).itemsize


def _occupation_batch(G: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
    C = 0.5 * (G + eye.unsqueeze(0))
    return 0.5 * (C + C.conj().transpose(-2, -1))


def _strip_spectral_curves(
    G: torch.Tensor, *, nx: int, ny: int, y0_chunk: int = 4, eps: float = 1e-12
) -> dict[str, np.ndarray]:
    if build_chern_partition_indices is None:
        raise ImportError("streaming_covariance_observables_gpu.py is required for strip entropy")
    samples = int(G.shape[0])
    curves = {
        key: np.zeros((samples, ny // 2 + 1), dtype=np.float64)
        for key in ("entropy", "charge_mean", "charge_k2", "charge_k3", "charge_k4")
    }
    for ay in range(1, ny // 2 + 1):
        totals = {
            key: torch.zeros((samples,), dtype=torch.float64, device=G.device)
            for key in curves
        }
        for y0_start in range(0, ny, int(y0_chunk)):
            y0_stop = min(ny, y0_start + int(y0_chunk))
            idx = strip_mode_indices(
                nx=nx,
                ny=ny,
                ay=ay,
                y0_values=range(y0_start, y0_stop),
                device=G.device,
            )
            restricted = gather_restricted_covariance(G, idx)
            matrix_size = int(restricted.shape[-1])
            eye = torch.eye(matrix_size, dtype=restricted.dtype, device=restricted.device)
            occupation = 0.5 * (restricted + eye[None])
            occupation = 0.5 * (occupation + occupation.conj().transpose(-2, -1))
            nu = torch.linalg.eigvalsh(occupation).real
            clipped = torch.clamp(nu, eps, 1.0 - eps)
            q = nu * (1.0 - nu)
            values = {
                "entropy": -torch.sum(
                    clipped * torch.log(clipped)
                    + (1.0 - clipped) * torch.log(1.0 - clipped), dim=1
                ),
                "charge_mean": torch.sum(nu, dim=1),
                "charge_k2": torch.sum(q, dim=1),
                "charge_k3": torch.sum(q * (1.0 - 2.0 * nu), dim=1),
                "charge_k4": torch.sum(q * (1.0 - 6.0 * q), dim=1),
            }
            for key, value in values.items():
                totals[key] += value.reshape(samples, y0_stop - y0_start).sum(dim=1)
        for key in curves:
            curves[key][:, ay] = (totals[key] / float(ny)).detach().cpu().numpy()
    return curves


def _bott_index_from_occupation(
    C: torch.Tensor, *, nx: int, ny: int, purity_threshold: float = 1e-8
) -> tuple[float, float]:
    evals, vectors = torch.linalg.eigh(C)
    purity_gap = float(torch.min(torch.abs(evals - 0.5)).detach().cpu())
    if purity_gap <= float(purity_threshold):
        return float("nan"), purity_gap
    occupied = vectors[:, evals > 0.5]
    nlayer = int(C.shape[0])
    mode = torch.arange(nlayer, device=C.device)
    x = (mode // 2) % int(nx)
    y = (mode // (2 * int(nx))) % int(ny)
    phase_x = torch.exp(2j * math.pi * x.to(C.real.dtype) / float(nx)).to(C.dtype)
    phase_y = torch.exp(2j * math.pi * y.to(C.real.dtype) / float(ny)).to(C.dtype)
    ux = occupied.conj().T @ (phase_x[:, None] * occupied)
    uy = occupied.conj().T @ (phase_y[:, None] * occupied)

    def polar_unitary(matrix: torch.Tensor) -> torch.Tensor:
        left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
        return left @ right_h

    ux = polar_unitary(ux)
    uy = polar_unitary(uy)
    commutator = uy @ ux @ uy.conj().T @ ux.conj().T
    phases = torch.angle(torch.linalg.eigvals(commutator))
    bott = float((torch.sum(phases) / (2.0 * math.pi)).detach().cpu())
    return bott, purity_gap


class SelectedCovarianceObserver:
    """Per-trajectory observables at declared cycles, with no cycle history."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        samples: int,
        physical_cycles: int,
        observation_cycles: Iterable[int],
        covariance_cycles: Iterable[int] = (),
        strip_entropy_cycles: Iterable[int] = (),
        local_marker_cycles: Iterable[int] = (),
        bott_cycles: Iterable[int] = (),
        compute_correlator: bool = True,
        low_mode_count: int = 16,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.nlayer = 2 * self.nx * self.ny
        self.samples = int(samples)
        self.physical_cycles = int(physical_cycles)
        if self.physical_cycles < 1:
            raise ValueError("physical_cycles must be positive")
        self.cycles = sorted({int(value) for value in observation_cycles})
        self.cycle_index = {cycle: index for index, cycle in enumerate(self.cycles)}
        self.covariance_cycles = {int(value) for value in covariance_cycles}
        self.strip_entropy_cycles = {int(value) for value in strip_entropy_cycles}
        self.local_marker_cycles = {int(value) for value in local_marker_cycles}
        self.bott_cycles = {int(value) for value in bott_cycles}
        self.compute_correlator = bool(compute_correlator)
        self.low_mode_count = min(int(low_mode_count), self.nlayer)
        unknown = (
            self.covariance_cycles
            | self.strip_entropy_cycles
            | self.local_marker_cycles
            | self.bott_cycles
        ) - set(self.cycles)
        if unknown:
            raise ValueError(f"specialized observer cycles not in observation_cycles: {sorted(unknown)}")
        nobs = len(self.cycles)
        scalar = (self.samples, nobs)
        self.seen = np.zeros(scalar, dtype=np.bool_)
        self.convergence_cycles = np.arange(1, self.physical_cycles + 1, dtype=np.int64)
        self.successive_covariance_frobenius_per_dimension = np.full(
            (self.samples, self.physical_cycles), np.nan, dtype=np.float64
        )
        self._convergence_seen = np.zeros(
            (self.samples, self.physical_cycles), dtype=np.bool_
        )
        self._convergence_device_values: torch.Tensor | None = None
        self._convergence_device_seen: torch.Tensor | None = None
        self._previous_covariance: torch.Tensor | None = None
        self._previous_batch_start: int | None = None
        self._previous_cycle: int | None = None
        self.purity_gap = np.full(scalar, np.nan, dtype=np.float64)
        self.total_entropy = np.full(scalar, np.nan, dtype=np.float64)
        self.total_charge_variance = np.full(scalar, np.nan, dtype=np.float64)
        self.real_space_chern = np.full(scalar, np.nan, dtype=np.float64)
        self.bott_index = np.full(scalar, np.nan, dtype=np.float64)
        self.density = np.full((self.samples, nobs, self.nx, self.ny), np.nan, dtype=np.float64)
        self.entropy_contour = np.full_like(self.density, np.nan)
        self.occupation_spectra: dict[int, np.ndarray] = {}
        self.strip_entropy: dict[int, np.ndarray] = {}
        self.strip_charge_cumulants: dict[int, dict[str, np.ndarray]] = {}
        self.local_marker: dict[int, np.ndarray] = {}
        self.correlator: dict[int, np.ndarray] = {}
        self.low_mode_values = np.full((self.samples, nobs, self.low_mode_count), np.nan, dtype=np.float64)
        self.low_mode_x_weight = np.full(
            (self.samples, nobs, self.low_mode_count, self.nx), np.nan, dtype=np.float64
        )
        self.covariance_triangle: dict[int, np.ndarray] = {}
        self._triangle_indices = np.triu_indices(self.nlayer)
        self._chern_cache: dict[str, Any] = {}
        self._corr_cache: dict[str, Any] = {}

    def _materialize_convergence_cpu(self) -> None:
        if self._convergence_device_values is None:
            return
        self.successive_covariance_frobenius_per_dimension = (
            self._convergence_device_values.detach().cpu().numpy()
        )
        self._convergence_seen = (
            self._convergence_device_seen.detach().cpu().numpy()
        )
        self._convergence_device_values = None
        self._convergence_device_seen = None

    def __call__(
        self, *, cycle: int, G: torch.Tensor, batch_start: int, batch_count: int, **_: Any
    ) -> None:
        cycle = int(cycle)
        start = int(batch_start)
        stop = start + int(batch_count)
        if stop > self.samples:
            raise IndexError("cycle observer emitted samples outside this shard")
        work = G.detach()
        if cycle == 0:
            if self._previous_covariance is not None:
                raise RuntimeError("received a new covariance batch before the prior batch completed")
            self._previous_covariance = work.clone()
            self._previous_batch_start = start
            self._previous_cycle = 0
            if work.device.type == "cuda" and self._convergence_device_values is None:
                self._convergence_device_values = torch.full(
                    (self.samples, self.physical_cycles),
                    torch.nan,
                    dtype=torch.float64,
                    device=work.device,
                )
                self._convergence_device_seen = torch.zeros(
                    (self.samples, self.physical_cycles),
                    dtype=torch.bool,
                    device=work.device,
                )
        else:
            if not 1 <= cycle <= self.physical_cycles:
                raise IndexError(
                    f"convergence cycle {cycle} is outside 1..{self.physical_cycles}"
                )
            if (
                self._previous_covariance is None
                or self._previous_batch_start != start
                or self._previous_cycle != cycle - 1
            ):
                raise RuntimeError(
                    "successive-covariance convergence observations are not cycle-contiguous"
                )
            if self._convergence_device_values is None:
                if np.any(self._convergence_seen[start:stop, cycle - 1]):
                    raise RuntimeError(
                        f"duplicate convergence observation at cycle {cycle}"
                    )
            # Reuse the one retained covariance buffer for the difference so the
            # diagnostic does not allocate a second dense batch-sized matrix.
            difference = self._previous_covariance.sub_(work)
            normalized_frobenius = torch.linalg.vector_norm(
                difference.reshape(int(batch_count), -1), dim=1
            ) / float(self.nlayer)
            if self._convergence_device_values is None:
                self.successive_covariance_frobenius_per_dimension[
                    start:stop, cycle - 1
                ] = normalized_frobenius.detach().cpu().numpy()
                self._convergence_seen[start:stop, cycle - 1] = True
            else:
                self._convergence_device_values[
                    start:stop, cycle - 1
                ] = normalized_frobenius.to(torch.float64)
                self._convergence_device_seen[start:stop, cycle - 1] = True
            if cycle == self.physical_cycles:
                self._previous_covariance = None
                self._previous_batch_start = None
                self._previous_cycle = None
            else:
                self._previous_covariance.copy_(work)
                self._previous_cycle = cycle
        if cycle not in self.cycle_index:
            return
        obs = self.cycle_index[cycle]
        if np.any(self.seen[start:stop, obs]):
            raise RuntimeError(f"duplicate selected covariance observation at cycle {cycle}")
        C = _occupation_batch(work)
        evals, eigenvectors = torch.linalg.eigh(C)
        evals = evals.real
        clipped = torch.clamp(evals, 1e-12, 1.0 - 1e-12)
        entropy = -torch.sum(
            clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped), dim=1
        )
        variance = torch.sum(evals * (1.0 - evals), dim=1)
        diag = torch.diagonal(C, dim1=-2, dim2=-1).real.reshape(
            int(batch_count), self.ny, self.nx, 2
        )
        density = diag.sum(dim=-1).transpose(1, 2)
        self.purity_gap[start:stop, obs] = (
            torch.min(torch.abs(evals - 0.5), dim=1).values.detach().cpu().numpy()
        )
        self.total_entropy[start:stop, obs] = entropy.detach().cpu().numpy()
        self.total_charge_variance[start:stop, obs] = variance.detach().cpu().numpy()
        self.density[start:stop, obs] = density.detach().cpu().numpy()
        mode_entropy = -(
            clipped * torch.log(clipped)
            + (1.0 - clipped) * torch.log(1.0 - clipped)
        )
        contour = torch.matmul(eigenvectors.abs().square(), mode_entropy[:, :, None]).squeeze(-1)
        contour = contour.reshape(int(batch_count), self.ny, self.nx, 2).sum(dim=-1).transpose(1, 2)
        self.entropy_contour[start:stop, obs] = contour.detach().cpu().numpy()
        self.occupation_spectra.setdefault(
            cycle, np.full((self.samples, self.nlayer), np.nan, dtype=np.float64)
        )[start:stop] = evals.detach().cpu().numpy()
        low_order = torch.argsort(torch.abs(evals - 0.5), dim=1)[:, : self.low_mode_count]
        low_values = torch.gather(evals, 1, low_order)
        low_vectors = torch.gather(
            eigenvectors,
            2,
            low_order[:, None, :].expand(-1, self.nlayer, -1),
        )
        x_weight = (
            low_vectors.abs()
            .square()
            .reshape(int(batch_count), self.ny, self.nx, 2, self.low_mode_count)
            .sum(dim=(1, 3))
            .permute(0, 2, 1)
        )
        self.low_mode_values[start:stop, obs] = low_values.detach().cpu().numpy()
        self.low_mode_x_weight[start:stop, obs] = x_weight.detach().cpu().numpy()
        if build_chern_partition_indices is not None:
            key = str(work.device)
            if key not in self._chern_cache:
                self._chern_cache[key] = build_chern_partition_indices(
                    nx=self.nx, ny=self.ny, device=work.device
                )
            self.real_space_chern[start:stop, obs] = (
                real_space_chern_batch_torch(work, self._chern_cache[key]).detach().cpu().numpy()
            )
            if self.compute_correlator:
                if key not in self._corr_cache:
                    self._corr_cache[key] = build_square_correlator_pair_indices(
                        nx=self.nx, ny=self.ny, device=work.device
                    )
                self.correlator.setdefault(
                    cycle,
                    np.full((self.samples, self.ny // 2 + 1), np.nan, dtype=np.float64),
                )[start:stop] = (
                    xavg_square_correlator_batch_torch(
                        work, self._corr_cache[key], nx=self.nx, ny=self.ny
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

        if cycle in self.covariance_cycles:
            triangle = work[:, self._triangle_indices[0], self._triangle_indices[1]]
            self.covariance_triangle.setdefault(
                cycle,
                np.full(
                    (self.samples, len(self._triangle_indices[0])),
                    np.nan + 1j * np.nan,
                    dtype=np.complex128,
                ),
            )[start:stop] = triangle.cpu().numpy()
        if cycle in self.strip_entropy_cycles:
            curves = _strip_spectral_curves(work, nx=self.nx, ny=self.ny)
            self.strip_entropy.setdefault(
                cycle, np.full((self.samples, self.ny // 2 + 1), np.nan, dtype=np.float64)
            )[start:stop] = curves["entropy"]
            cycle_cumulants = self.strip_charge_cumulants.setdefault(cycle, {})
            for key in ("charge_mean", "charge_k2", "charge_k3", "charge_k4"):
                cycle_cumulants.setdefault(
                    key,
                    np.full((self.samples, self.ny // 2 + 1), np.nan, dtype=np.float64),
                )[start:stop] = curves[key]
        if cycle in self.local_marker_cycles:
            self.local_marker.setdefault(
                cycle,
                np.full((self.samples, self.nx, self.ny), np.nan, dtype=np.float64),
            )[start:stop] = local_chern_marker_batch_torch(
                work, nx=self.nx, ny=self.ny
            ).detach().cpu().numpy()
        if cycle in self.bott_cycles:
            for local_index in range(int(batch_count)):
                value, _ = _bott_index_from_occupation(
                    C[local_index], nx=self.nx, ny=self.ny
                )
                self.bott_index[start + local_index, obs] = value
        self.seen[start:stop, obs] = True

    def validate(self) -> dict[str, Any]:
        self._materialize_convergence_cpu()
        if not self._convergence_seen.all():
            missing = np.argwhere(~self._convergence_seen)
            raise RuntimeError(
                "missing successive-covariance convergence observations; "
                f"first missing indices: {missing[:8].tolist()}"
            )
        if not np.isfinite(self.successive_covariance_frobenius_per_dimension).all():
            raise FloatingPointError(
                "successive covariance Frobenius diagnostic contains non-finite values"
            )
        if self._previous_covariance is not None:
            raise RuntimeError("transient previous covariance was not released")
        if not self.seen.all():
            missing = np.argwhere(~self.seen)
            raise RuntimeError(f"missing selected observations; first missing indices: {missing[:8].tolist()}")
        for name in (
            "purity_gap", "total_entropy", "total_charge_variance",
            "real_space_chern", "entropy_contour",
        ):
            values = getattr(self, name)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"{name} contains non-finite values")
        for cycle, values in self.strip_entropy.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(f"strip entropy at cycle {cycle} is incomplete")
        for cycle, cumulants in self.strip_charge_cumulants.items():
            if not all(np.isfinite(values).all() for values in cumulants.values()):
                raise FloatingPointError(f"strip charge cumulants at cycle {cycle} are incomplete")
        return {
            "schema": OBSERVER_SCHEMA,
            "samples": self.samples,
            "cycles": self.cycles,
            "physical_cycles": self.physical_cycles,
            "convergence_diagnostic": "frobenius(G_cycle-G_previous_cycle)/nlayer",
            "convergence_normalization_dimension": self.nlayer,
            "convergence_transient_previous_covariance_bytes_peak": (
                self.samples
                * self.nlayer
                * self.nlayer
                * np.dtype(np.complex128).itemsize
            ),
            "nlayer": self.nlayer,
            "covariance_representation": "transient_only_not_archived",
            "dense_bytes_per_covariance": covariance_storage_bytes(
                nx=self.nx, ny=self.ny, samples=1, snapshots=1, representation="dense"
            ),
            "triangle_bytes_per_covariance": covariance_storage_bytes(
                nx=self.nx,
                ny=self.ny,
                samples=1,
                snapshots=1,
                representation="hermitian_triangle",
            ),
        }

    def save(
        self,
        path: Path | str,
        *,
        config: dict[str, Any],
        include_transient_covariances: bool = False,
    ) -> dict[str, Any]:
        diagnostics = self.validate()
        payload: dict[str, Any] = {
            "schema": np.asarray(OBSERVER_SCHEMA),
            "config_json": np.asarray(json.dumps(config, sort_keys=True)),
            "cycles": np.asarray(self.cycles, dtype=np.int64),
            "convergence_cycles": self.convergence_cycles,
            "successive_covariance_frobenius_per_dimension": (
                self.successive_covariance_frobenius_per_dimension
            ),
            "purity_gap": self.purity_gap,
            "total_entropy": self.total_entropy,
            "total_charge_variance": self.total_charge_variance,
            "real_space_chern": self.real_space_chern,
            "bott_index": self.bott_index,
            "density": self.density,
            "entropy_contour": self.entropy_contour,
            "low_mode_occupation": self.low_mode_values,
            "low_mode_x_weight": self.low_mode_x_weight,
            "triangle_rows": np.asarray(self._triangle_indices[0], dtype=np.int32),
            "triangle_cols": np.asarray(self._triangle_indices[1], dtype=np.int32),
        }
        for cycle, values in self.occupation_spectra.items():
            payload[f"occupation_spectrum_cycle_{cycle:04d}"] = values
        for cycle, values in self.strip_entropy.items():
            payload[f"strip_entropy_cycle_{cycle:04d}"] = values
        for cycle, cumulants in self.strip_charge_cumulants.items():
            for key, values in cumulants.items():
                payload[f"strip_{key}_cycle_{cycle:04d}"] = values
        for cycle, values in self.local_marker.items():
            payload[f"local_chern_marker_cycle_{cycle:04d}"] = values
        for cycle, values in self.correlator.items():
            payload[f"square_correlator_cycle_{cycle:04d}"] = values
        if include_transient_covariances:
            for cycle, values in self.covariance_triangle.items():
                payload[f"covariance_triangle_cycle_{cycle:04d}"] = values
        path = Path(path)
        save_npz_atomic(path, **payload)
        return {
            **diagnostics,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "transient_covariances_included": bool(include_transient_covariances),
        }


def reconstruct_hermitian_covariance(
    triangle: np.ndarray, rows: np.ndarray, cols: np.ndarray, nlayer: int
) -> np.ndarray:
    output = np.zeros((int(nlayer), int(nlayer)), dtype=np.complex128)
    output[rows, cols] = triangle
    output[cols, rows] = np.conj(triangle)
    diagonal = np.arange(int(nlayer))
    output[diagonal, diagonal] = output[diagonal, diagonal].real
    return output
