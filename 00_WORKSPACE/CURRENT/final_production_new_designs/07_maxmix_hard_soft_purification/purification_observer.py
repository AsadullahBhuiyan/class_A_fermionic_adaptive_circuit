"""Cycle-resolved purification observables and record weight for bundle 07."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "maxmix_purification_observer_v2"
ENTROPY_LOG_BASE = "natural"
OCCUPATION_ROUNDOFF_TOL = 1.0e-9
HERMITICITY_TOL = 1.0e-9
LOG_PROBABILITY_DTYPE = "float64"
LOG_PROBABILITY_CONVENTION = "cumulative_realized_born_log_probability"


def _cell_sum(values: torch.Tensor, *, nx: int, ny: int) -> torch.Tensor:
    """Sum two orbitals per cell and return ``(..., nx, ny)``."""
    leading = values.shape[:-1]
    return values.reshape(*leading, ny, nx, 2).sum(dim=-1).transpose(-2, -1)


@dataclass(frozen=True)
class CycleObservables:
    occupation_spectrum: np.ndarray
    entropy_contour: np.ndarray
    charge_variance_contour: np.ndarray
    total_entropy: np.ndarray
    total_charge: np.ndarray
    total_charge_variance: np.ndarray
    hermiticity_residual: np.ndarray
    entropy_closure_error: np.ndarray
    charge_closure_error: np.ndarray


def covariance_observables(
    G: torch.Tensor,
    *,
    nx: int,
    ny: int,
    sample_chunk: int,
    entropy_eps: float = 1.0e-12,
) -> CycleObservables:
    """Evaluate spectra and additive contours from centered covariance ``G``.

    The occupation matrix is ``C=(G+I)/2``.  One Hermitian eigendecomposition
    supplies the raw occupation spectrum, the global mixed-state entropy
    contour, and the intrinsic charge-fluctuation contour ``diag(C(1-C))``.
    """
    if G.ndim != 3:
        raise ValueError(f"G must have shape (samples,N,N), got {tuple(G.shape)}")
    samples, n, n2 = map(int, G.shape)
    expected = 2 * int(nx) * int(ny)
    if n != n2 or n != expected:
        raise ValueError(f"expected covariance shape (samples,{expected},{expected})")
    if G.dtype != torch.complex128:
        raise ValueError(f"expected complex128 covariance, got {G.dtype}")
    if sample_chunk <= 0:
        raise ValueError("sample_chunk must be positive")

    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "occupation_spectrum",
            "entropy_contour",
            "charge_variance_contour",
            "total_entropy",
            "total_charge",
            "total_charge_variance",
            "hermiticity_residual",
            "entropy_closure_error",
            "charge_closure_error",
        )
    }
    with torch.inference_mode():
        for start in range(0, samples, int(sample_chunk)):
            chunk = G[start : start + int(sample_chunk)]
            if not bool(torch.isfinite(chunk).all().item()):
                raise FloatingPointError("covariance contains non-finite entries")
            herm = torch.amax(torch.abs(chunk - chunk.mH), dim=(-2, -1)).real
            if float(torch.max(herm).item()) > HERMITICITY_TOL:
                raise FloatingPointError("covariance Hermiticity residual exceeded tolerance")
            chunk = 0.5 * (chunk + chunk.mH)
            eye = torch.eye(n, dtype=chunk.dtype, device=chunk.device)
            C = 0.5 * (chunk + eye)
            nu, vectors = torch.linalg.eigh(C)
            if not bool(torch.isfinite(nu).all().item()):
                raise FloatingPointError("occupation spectrum contains non-finite entries")
            nu_min = float(torch.min(nu).item())
            nu_max = float(torch.max(nu).item())
            if nu_min < -OCCUPATION_ROUNDOFF_TOL or nu_max > 1.0 + OCCUPATION_ROUNDOFF_TOL:
                raise FloatingPointError(
                    f"occupation spectrum left [0,1] beyond roundoff: [{nu_min:.3e},{nu_max:.3e}]"
                )
            clipped = torch.clamp(nu, min=float(entropy_eps), max=1.0 - float(entropy_eps))
            entropy_weight = -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped))
            variance_weight = nu * (1.0 - nu)
            probabilities = torch.abs(vectors).square().real
            entropy_orbital = torch.einsum("bia,ba->bi", probabilities, entropy_weight)
            variance_orbital = torch.einsum("bia,ba->bi", probabilities, variance_weight)
            entropy_cell = _cell_sum(entropy_orbital, nx=nx, ny=ny)
            variance_cell = _cell_sum(variance_orbital, nx=nx, ny=ny)
            total_entropy = torch.sum(entropy_weight, dim=-1)
            total_charge = torch.sum(nu, dim=-1)
            total_variance = torch.sum(variance_weight, dim=-1)
            entropy_closure = torch.abs(torch.sum(entropy_cell, dim=(-2, -1)) - total_entropy)
            variance_closure = torch.abs(torch.sum(variance_cell, dim=(-2, -1)) - total_variance)

            values = {
                "occupation_spectrum": nu,
                "entropy_contour": entropy_cell,
                "charge_variance_contour": variance_cell,
                "total_entropy": total_entropy,
                "total_charge": total_charge,
                "total_charge_variance": total_variance,
                "hermiticity_residual": herm,
                "entropy_closure_error": entropy_closure,
                "charge_closure_error": variance_closure,
            }
            for key, value in values.items():
                arrays[key].append(value.detach().cpu().numpy().astype(np.float64, copy=False))

    return CycleObservables(**{key: np.concatenate(value, axis=0) for key, value in arrays.items()})


class PurificationObserver:
    """Trajectory-resolved observer with an NPZ-safe rolling state."""

    COVARIANCE_ARRAY_NAMES = (
        "occupation_spectrum",
        "entropy_contour",
        "charge_variance_contour",
        "total_entropy",
        "total_charge",
        "total_charge_variance",
        "hermiticity_residual",
        "entropy_closure_error",
        "charge_closure_error",
    )
    ARRAY_NAMES = COVARIANCE_ARRAY_NAMES + (
        "measurement_log_probability",
        "cumulative_log_probability",
        "site_event_count",
        "channel_event_count",
        "seen",
    )

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        cycles: int,
        sample_indices: np.ndarray,
        sample_chunk: int,
        construction: str,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.sample_chunk = int(sample_chunk)
        self.samples = int(self.sample_indices.size)
        self.nlayer = 2 * self.nx * self.ny
        self.construction = str(construction)
        if self.construction not in ("hard", "soft"):
            raise ValueError("construction must be hard or soft")
        if self.construction == "hard":
            x_left, x_right = self.nx // 4, 3 * self.nx // 4
            active_width = x_right - x_left + 1
            self.expected_sites_per_cycle = active_width * self.ny
            self.transfer_mode_count = 2 * active_width * self.ny
            self.log_probability_origin = (
                "after_born_conditioned_exterior_preparation"
            )
        else:
            self.expected_sites_per_cycle = self.nx * self.ny
            self.transfer_mode_count = 2 * self.nx * self.ny
            self.log_probability_origin = "global_maxmix_cycle_zero"
        self.expected_channels_per_cycle = 4 * self.expected_sites_per_cycle
        t = self.cycles + 1
        self.occupation_spectrum = np.full((self.samples, t, self.nlayer), np.nan)
        self.entropy_contour = np.full((self.samples, t, self.nx, self.ny), np.nan)
        self.charge_variance_contour = np.full((self.samples, t, self.nx, self.ny), np.nan)
        self.total_entropy = np.full((self.samples, t), np.nan)
        self.total_charge = np.full((self.samples, t), np.nan)
        self.total_charge_variance = np.full((self.samples, t), np.nan)
        self.hermiticity_residual = np.full((self.samples, t), np.nan)
        self.entropy_closure_error = np.full((self.samples, t), np.nan)
        self.charge_closure_error = np.full((self.samples, t), np.nan)
        self.measurement_log_probability = np.full(
            (self.samples, t), np.nan, dtype=np.float64
        )
        self.cumulative_log_probability = np.full(
            (self.samples, t), np.nan, dtype=np.float64
        )
        self.site_event_count = np.zeros((self.samples, t), dtype=np.int64)
        self.channel_event_count = np.zeros((self.samples, t), dtype=np.int64)
        self.seen = np.zeros(t, dtype=bool)
        self._record_log_probability_gpu: torch.Tensor | None = None
        self._site_event_count_gpu: torch.Tensor | None = None
        self._channel_event_count_gpu: torch.Tensor | None = None

    def _ensure_record_buffers(self, device: torch.device) -> None:
        shape = (self.samples, self.cycles + 1)
        if self._record_log_probability_gpu is None:
            self._record_log_probability_gpu = torch.zeros(
                shape, dtype=torch.float64, device=device
            )
            self._site_event_count_gpu = torch.zeros(
                shape, dtype=torch.int64, device=device
            )
            self._channel_event_count_gpu = torch.zeros(
                shape, dtype=torch.int64, device=device
            )
        elif self._record_log_probability_gpu.device != device:
            raise RuntimeError("record observer device changed during execution")

    def record_event(
        self,
        *,
        cycle: int,
        sample_offsets: torch.Tensor,
        conditional_log_probability: torch.Tensor,
        **_: Any,
    ) -> None:
        """Accumulate realized Born weights without synchronizing to the CPU."""
        cycle = int(cycle)
        if cycle < 1 or cycle > self.cycles:
            raise ValueError(f"record event cycle {cycle} is outside 1..{self.cycles}")
        if self.seen[cycle]:
            raise RuntimeError(f"record event arrived after cycle {cycle} was finalized")
        conditional = torch.as_tensor(conditional_log_probability)
        if conditional.ndim != 2:
            raise ValueError("conditional log-probability payload must be two-dimensional")
        if conditional.dtype != torch.float64:
            conditional = conditional.to(dtype=torch.float64)
        rows = torch.as_tensor(
            sample_offsets, dtype=torch.long, device=conditional.device
        ).reshape(-1)
        if int(rows.numel()) != int(conditional.shape[0]):
            raise ValueError("record sample offsets do not match probability rows")
        if int(rows.numel()) == 0:
            return
        self._ensure_record_buffers(conditional.device)
        assert self._record_log_probability_gpu is not None
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
            raise FloatingPointError("record contains a non-finite log probability")
        if np.any(site_count != self.expected_sites_per_cycle):
            raise RuntimeError(
                f"cycle {cycle} has incomplete site records: expected "
                f"{self.expected_sites_per_cycle}"
            )
        if np.any(channel_count != self.expected_channels_per_cycle):
            raise RuntimeError(
                f"cycle {cycle} has incomplete channel records: expected "
                f"{self.expected_channels_per_cycle}"
            )
        self.measurement_log_probability[:, cycle] = measurement
        self.site_event_count[:, cycle] = site_count
        self.channel_event_count[:, cycle] = channel_count
        self.cumulative_log_probability[:, cycle] = (
            self.cumulative_log_probability[:, cycle - 1] + measurement
        )

    def observe(self, *, cycle: int, G: torch.Tensor) -> None:
        cycle = int(cycle)
        if cycle < 0 or cycle > self.cycles:
            raise ValueError(f"cycle {cycle} is outside 0..{self.cycles}")
        if self.seen[cycle]:
            raise RuntimeError(f"cycle {cycle} was observed more than once")
        self._finalize_record_cycle(cycle)
        result = covariance_observables(
            G, nx=self.nx, ny=self.ny, sample_chunk=self.sample_chunk
        )
        for name in self.COVARIANCE_ARRAY_NAMES:
            getattr(self, name)[:, cycle] = getattr(result, name)
        self.seen[cycle] = True

    def validate(self, *, completed_cycle: int, final: bool = False) -> None:
        completed_cycle = int(completed_cycle)
        if completed_cycle < 0 or completed_cycle > self.cycles:
            raise ValueError("completed_cycle is outside the configured horizon")
        expected = np.arange(self.cycles + 1) <= completed_cycle
        if not np.array_equal(self.seen, expected):
            raise RuntimeError("observer cycles are not a contiguous completed prefix")
        for name in self.COVARIANCE_ARRAY_NAMES + (
            "measurement_log_probability",
            "cumulative_log_probability",
        ):
            array = getattr(self, name)
            if not np.isfinite(array[:, : completed_cycle + 1]).all():
                raise FloatingPointError(f"{name} contains incomplete/non-finite observations")
        if np.any(self.site_event_count[:, 0] != 0) or np.any(
            self.channel_event_count[:, 0] != 0
        ):
            raise RuntimeError("cycle-zero record event counts must vanish")
        if completed_cycle:
            if np.any(
                self.site_event_count[:, 1 : completed_cycle + 1]
                != self.expected_sites_per_cycle
            ):
                raise RuntimeError("site record counts are incomplete")
            if np.any(
                self.channel_event_count[:, 1 : completed_cycle + 1]
                != self.expected_channels_per_cycle
            ):
                raise RuntimeError("channel record counts are incomplete")
            increments = np.diff(
                self.cumulative_log_probability[:, : completed_cycle + 1], axis=1
            )
            if not np.allclose(
                increments,
                self.measurement_log_probability[:, 1 : completed_cycle + 1],
                rtol=2.0e-13,
                atol=1.0e-11,
            ):
                raise RuntimeError("cumulative record probability does not close")
        if final and completed_cycle != self.cycles:
            raise RuntimeError("final observer validation requires the full cycle horizon")

    def checkpoint_payload(self) -> dict[str, np.ndarray]:
        payload = {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "observer_sample_indices": self.sample_indices,
        }
        payload.update({f"observer_{name}": getattr(self, name) for name in self.ARRAY_NAMES})
        return payload

    def restore_checkpoint(self, payload: dict[str, np.ndarray], *, completed_cycle: int) -> None:
        if str(np.asarray(payload["observer_schema"]).item()) != OBSERVER_SCHEMA:
            raise RuntimeError("observer checkpoint schema mismatch")
        if not np.array_equal(payload["observer_sample_indices"], self.sample_indices):
            raise RuntimeError("observer checkpoint sample IDs mismatch")
        for name in self.ARRAY_NAMES:
            saved = np.asarray(payload[f"observer_{name}"])
            target = getattr(self, name)
            if saved.shape != target.shape or saved.dtype != target.dtype:
                raise RuntimeError(f"observer checkpoint {name} shape/dtype mismatch")
            target[...] = saved
        self.validate(completed_cycle=completed_cycle)

    def result_payload(self, sample_slice: slice) -> dict[str, np.ndarray]:
        return {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "entropy_log_base": np.asarray(ENTROPY_LOG_BASE),
            "cycles": np.arange(self.cycles + 1, dtype=np.int64),
            "sample_indices": self.sample_indices[sample_slice],
            "occupation_spectrum": self.occupation_spectrum[sample_slice],
            "entropy_contour": self.entropy_contour[sample_slice],
            "charge_variance_contour": self.charge_variance_contour[sample_slice],
            "total_entropy": self.total_entropy[sample_slice],
            "total_charge": self.total_charge[sample_slice],
            "total_charge_variance": self.total_charge_variance[sample_slice],
            "hermiticity_residual": self.hermiticity_residual[sample_slice],
            "entropy_closure_error": self.entropy_closure_error[sample_slice],
            "charge_closure_error": self.charge_closure_error[sample_slice],
            "measurement_log_probability": self.measurement_log_probability[sample_slice],
            "cumulative_log_probability": self.cumulative_log_probability[sample_slice],
            "site_event_count": self.site_event_count[sample_slice],
            "channel_event_count": self.channel_event_count[sample_slice],
            "log_probability_dtype": np.asarray(LOG_PROBABILITY_DTYPE),
            "log_probability_convention": np.asarray(LOG_PROBABILITY_CONVENTION),
            "log_probability_origin": np.asarray(self.log_probability_origin),
            "transfer_mode_count": np.asarray(self.transfer_mode_count, dtype=np.int64),
            "log_z_formula": np.asarray("log_Z=N_eff*log(2)+cumulative_log_probability"),
            "occupation_matrix_formula": np.asarray("C=(G+I)/2"),
            "charge_contour_formula": np.asarray("diag(C(I-C)), summed over cell orbitals"),
        }
