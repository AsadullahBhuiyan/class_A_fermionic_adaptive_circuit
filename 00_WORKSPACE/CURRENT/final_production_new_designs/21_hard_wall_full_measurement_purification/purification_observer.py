"""Full-layer cycle spectra, optional contours, and endpoint slow eigenmodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import torch


OBSERVER_SCHEMA = "full_measurement_purification_observer_v1"
ENTROPY_LOG_BASE = "natural"
OCCUPATION_ROUNDOFF_TOL = 1.0e-9
HERMITICITY_TOL = 1.0e-9
LOG_PROBABILITY_DTYPE = "float64"
LOG_PROBABILITY_CONVENTION = "cumulative_realized_born_log_probability"


class OccupationSpectrumError(FloatingPointError):
    """A CPU-confirmed violation, with one unmodified state for diagnosis."""

    def __init__(self, diagnostics, covariance, occupations):
        self.diagnostics = diagnostics
        self.covariance = covariance
        self.occupations = occupations
        super().__init__(
            "CPU-confirmed occupation bound violation: "
            f"min={diagnostics['nu_min']:.17g}, max={diagnostics['nu_max']:.17g}, "
            f"excess={diagnostics['bound_excess']:.9e}, "
            f"tolerance={OCCUPATION_ROUNDOFF_TOL:.9e}, "
            f"sample_offset={diagnostics['sample_offset']}. "
            "State and tolerance are unchanged; retain the last verified checkpoint."
        )


def _cpu_recheck(C, raw_G, nu, vectors, *, sample_start):
    """Independently re-solve suspect matrices, never project the evolving state.

    LAPACK evr provides a separate implementation from the GPU eigensolver.
    Ordinary accepted spectra and all RNG states are left untouched.
    """
    from scipy.linalg import eigh
    from threadpoolctl import threadpool_limits

    bad = (nu.min(dim=-1).values < -OCCUPATION_ROUNDOFF_TOL) | (
        nu.max(dim=-1).values > 1 + OCCUPATION_ROUNDOFF_TOL
    )
    for index in torch.nonzero(bad, as_tuple=False).flatten().tolist():
        cpu_C = C[index].detach().cpu().numpy()
        with threadpool_limits(limits=1, user_api="blas"):
            solved = eigh(cpu_C, eigvals_only=vectors is None, driver="evr")
        values = solved if vectors is None else solved[0]
        lower, upper = float(values.min()), float(values.max())
        excess = max(0., -lower, upper - 1.)
        if not np.isfinite(values).all() or excess > OCCUPATION_ROUNDOFF_TOL:
            diagnostics = dict(nu_min=lower, nu_max=upper, bound_excess=excess,
                tolerance=OCCUPATION_ROUNDOFF_TOL, sample_offset=sample_start + index,
                primary_min=float(nu[index].min()), primary_max=float(nu[index].max()),
                recheck_backend="scipy.linalg.eigh(driver=evr)")
            raise OccupationSpectrumError(diagnostics,
                raw_G[index].detach().cpu().numpy().copy(), values.copy())
        warnings.warn(
            f"Occupation spectrum for sample offset {sample_start + index} "
            "passed independent CPU recheck at unchanged 1e-9 tolerance; "
            "using CPU eigenpairs for this observation only.", RuntimeWarning,
        )
        nu[index] = torch.as_tensor(values, device=nu.device, dtype=nu.dtype)
        if vectors is not None:
            vectors[index] = torch.as_tensor(solved[1], device=vectors.device, dtype=vectors.dtype)
    return nu, vectors


def _cell_sum(values: torch.Tensor, *, nx: int, ny: int) -> torch.Tensor:
    """Sum two orbitals per cell and return ``(..., nx, ny)``."""
    leading = values.shape[:-1]
    return values.reshape(*leading, ny, nx, 2).sum(dim=-1).transpose(-2, -1)


@dataclass(frozen=True)
class CycleObservables:
    occupation_spectrum: np.ndarray
    entropy_contour: np.ndarray | None
    charge_variance_contour: np.ndarray | None
    total_entropy: np.ndarray
    total_charge: np.ndarray
    total_charge_variance: np.ndarray
    hermiticity_residual: np.ndarray
    entropy_closure_error: np.ndarray | None
    charge_closure_error: np.ndarray | None


def covariance_observables(
    G: torch.Tensor,
    *,
    nx: int,
    ny: int,
    sample_chunk: int,
    entropy_eps: float = 1.0e-12,
    save_contours: bool = True,
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
    contour_names = ("entropy_contour", "charge_variance_contour",
                     "entropy_closure_error", "charge_closure_error")
    if not save_contours:
        arrays = {key: value for key, value in arrays.items() if key not in contour_names}
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
            if save_contours:
                nu, vectors = torch.linalg.eigh(C)
            else:
                nu = torch.linalg.eigvalsh(C)
                vectors = None
            if not bool(torch.isfinite(nu).all().item()):
                raise FloatingPointError("occupation spectrum contains non-finite entries")
            nu_min = float(torch.min(nu).item())
            nu_max = float(torch.max(nu).item())
            if nu_min < -OCCUPATION_ROUNDOFF_TOL or nu_max > 1.0 + OCCUPATION_ROUNDOFF_TOL:
                nu, vectors = _cpu_recheck(C, G[start : start + int(sample_chunk)],
                    nu, vectors, sample_start=start)
            clipped = torch.clamp(nu, min=float(entropy_eps), max=1.0 - float(entropy_eps))
            entropy_weight = -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped))
            variance_weight = nu * (1.0 - nu)
            total_entropy = torch.sum(entropy_weight, dim=-1)
            total_charge = torch.sum(nu, dim=-1)
            total_variance = torch.sum(variance_weight, dim=-1)
            values = {
                "occupation_spectrum": nu,
                "total_entropy": total_entropy,
                "total_charge": total_charge,
                "total_charge_variance": total_variance,
                "hermiticity_residual": herm,
            }
            if save_contours:
                probabilities = torch.abs(vectors).square().real
                entropy_cell = _cell_sum(torch.einsum("bia,ba->bi", probabilities, entropy_weight), nx=nx, ny=ny)
                variance_cell = _cell_sum(torch.einsum("bia,ba->bi", probabilities, variance_weight), nx=nx, ny=ny)
                values.update(
                    entropy_contour=entropy_cell, charge_variance_contour=variance_cell,
                    entropy_closure_error=torch.abs(entropy_cell.sum(dim=(-2, -1)) - total_entropy),
                    charge_closure_error=torch.abs(variance_cell.sum(dim=(-2, -1)) - total_variance),
                )
            for key, value in values.items():
                arrays[key].append(value.detach().cpu().numpy().astype(np.float64, copy=False))

    result = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    if not save_contours:
        result.update({key: None for key in contour_names})
    return CycleObservables(**result)


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
        save_contours: bool = True,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.sample_chunk = int(sample_chunk)
        self.samples = int(self.sample_indices.size)
        self.nlayer = 2 * self.nx * self.ny
        self.construction = str(construction)
        if self.construction != "hard":
            raise ValueError("only hard walls are allowed in this campaign")
        self.save_contours = bool(save_contours)
        self.expected_sites_per_cycle = self.nx * self.ny
        self.transfer_mode_count = 2 * self.nx * self.ny
        self.log_probability_origin = "global_maxmix_cycle_zero_no_exterior_preparation"
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
        if not self.save_contours:
            removed = ("entropy_contour", "charge_variance_contour", "entropy_closure_error", "charge_closure_error")
            self.COVARIANCE_ARRAY_NAMES = tuple(k for k in self.COVARIANCE_ARRAY_NAMES if k not in removed)
            self.ARRAY_NAMES = tuple(k for k in self.ARRAY_NAMES if k not in removed)
            for key in removed:
                delattr(self, key)
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
            G, nx=self.nx, ny=self.ny, sample_chunk=self.sample_chunk,
            save_contours=self.save_contours,
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
            "observer_save_contours": np.asarray(self.save_contours),
        }
        payload.update({f"observer_{name}": getattr(self, name) for name in self.ARRAY_NAMES})
        return payload

    def restore_checkpoint(self, payload: dict[str, np.ndarray], *, completed_cycle: int) -> None:
        if str(np.asarray(payload["observer_schema"]).item()) != OBSERVER_SCHEMA:
            raise RuntimeError("observer checkpoint schema mismatch")
        if not np.array_equal(payload["observer_sample_indices"], self.sample_indices):
            raise RuntimeError("observer checkpoint sample IDs mismatch")
        if bool(payload["observer_save_contours"].item()) != self.save_contours:
            raise RuntimeError("observer checkpoint contour contract mismatch")
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
            **({"entropy_contour": self.entropy_contour[sample_slice],
                "charge_variance_contour": self.charge_variance_contour[sample_slice],
                "entropy_closure_error": self.entropy_closure_error[sample_slice],
                "charge_closure_error": self.charge_closure_error[sample_slice]}
               if self.save_contours else {}),
            "total_entropy": self.total_entropy[sample_slice],
            "total_charge": self.total_charge[sample_slice],
            "total_charge_variance": self.total_charge_variance[sample_slice],
            "hermiticity_residual": self.hermiticity_residual[sample_slice],
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


def extract_slowest_modes(G: np.ndarray, *, nx: int, ny: int, cycles: int,
                         device: str, sample_chunk: int) -> dict[str, np.ndarray]:
    """Minimum-|lambda| full-system mode per sample; never the most negative rate.

    Rates are log((1-nu)/nu)/(2T), with the same finite-level cutoff as the
    previous purification figure. Ties are flagged: the returned vector in a
    tied subspace is not a unique physical mode. Raw final covariances remain
    available for alternative tolerances/subspace analysis.
    """
    from tqdm.auto import tqdm

    G = np.asarray(G)
    n = 2 * nx * ny
    if G.dtype != np.complex128 or G.ndim != 3 or G.shape[1:] != (n, n):
        raise ValueError("invalid endpoint covariance")
    if cycles <= 0 or sample_chunk <= 0:
        raise ValueError("positive endpoint time and sample chunk required")
    count = len(G)
    out = {
        "slow_mode_vector": np.full((count, n), complex(np.nan, np.nan), dtype=np.complex128),
        "slow_mode_density": np.full((count, nx, ny), np.nan),
        "slow_mode_occupation": np.full(count, np.nan),
        "slow_mode_signed_rate": np.full(count, np.nan),
        "slow_mode_abs_rate": np.full(count, np.nan),
        "slow_mode_spectrum_index": np.full(count, -1, dtype=np.int64),
        "slow_mode_residual": np.full(count, np.nan),
        "slow_mode_min_abs_multiplicity": np.zeros(count, dtype=np.int64),
        "slow_mode_resolved": np.zeros(count, dtype=bool),
    }
    for start in tqdm(range(0, count, sample_chunk), desc="endpoint eigenmodes", unit="chunk", leave=False):
        g = torch.as_tensor(G[start:start + sample_chunk], device=device)
        if not bool(torch.isfinite(g).all()) or float((g - g.mH).abs().max()) > HERMITICITY_TOL:
            raise ValueError("nonfinite/non-Hermitian endpoint covariance")
        c = (g + g.mH) / 4 + torch.eye(n, dtype=g.dtype, device=g.device) / 2
        nu, u = torch.linalg.eigh(c)
        if not bool(torch.isfinite(nu).all()) or float(nu.min()) < -OCCUPATION_ROUNDOFF_TOL or float(nu.max()) > 1 + OCCUPATION_ROUNDOFF_TOL:
            raise ValueError("nonphysical endpoint occupations")
        for local in range(len(g)):
            row = start + local
            values = nu[local]
            valid = (values > 1e-9) & (values < 1 - 1e-9)
            indices = torch.nonzero(valid).flatten()
            if indices.numel() == 0:
                continue  # Explicit unresolved flag; do not invent a finite gap.
            rates = (torch.log1p(-values[valid]) - torch.log(values[valid])) / (2 * cycles)
            pick = int(rates.abs().argmin())
            j = int(indices[pick])
            vector = u[local, :, j]
            pivot = vector[vector.abs().argmax()]
            vector = vector * pivot.conj() / pivot.abs()
            density = _cell_sum(vector.abs().square(), nx=nx, ny=ny)
            residual = torch.linalg.vector_norm(c[local] @ vector - values[j] * vector)
            if float(residual) > 1e-9 or abs(float(density.sum()) - 1) > 1e-10:
                raise RuntimeError("slow-mode eigenvector failed residual/normalization validation")
            out["slow_mode_vector"][row] = vector.cpu().numpy()
            out["slow_mode_density"][row] = density.cpu().numpy()
            out["slow_mode_occupation"][row] = float(values[j])
            out["slow_mode_signed_rate"][row] = float(rates[pick])
            out["slow_mode_abs_rate"][row] = float(rates[pick].abs())
            out["slow_mode_spectrum_index"][row] = j
            out["slow_mode_residual"][row] = float(residual)
            out["slow_mode_min_abs_multiplicity"][row] = int(torch.isclose(
                rates.abs(), rates[pick].abs(), atol=1e-10, rtol=1e-10).sum())
            out["slow_mode_resolved"][row] = True
    return out
