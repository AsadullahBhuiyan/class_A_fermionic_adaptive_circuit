from __future__ import annotations

from typing import Any

import numpy as np

from .edge_modes import PhysicalEdgeFrame


def _as_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _orbital_mask(cell_mask: np.ndarray) -> np.ndarray:
    cell_mask = np.asarray(cell_mask, dtype=bool)
    nx, ny = cell_mask.shape
    out = np.zeros((2 * nx * ny,), dtype=bool)
    for x, y in zip(*np.nonzero(cell_mask)):
        out[2 * int(x) + 2 * nx * int(y) : 2 * int(x) + 2 * nx * int(y) + 2] = True
    return out


def _wall_cell_mask(nx: int, ny: int, wall_x: int, width: int) -> np.ndarray:
    x = np.arange(int(nx), dtype=np.int64)
    distance = np.minimum((x - int(wall_x)) % int(nx), (int(wall_x) - x) % int(nx))
    return np.broadcast_to((distance < int(width))[:, None], (int(nx), int(ny))).copy()


def _stable_core_step(core_hat: np.ndarray, log_scale: float, r_factor: np.ndarray) -> tuple[np.ndarray, float, int]:
    product = np.asarray(r_factor, dtype=np.complex128) @ np.asarray(core_hat, dtype=np.complex128)
    norm = float(np.linalg.norm(product, ord="fro"))
    if not np.isfinite(norm) or norm == 0.0:
        return np.zeros_like(product), -np.inf, 1
    return product / norm, float(log_scale + np.log(norm)), 0


def _log_singular_values(core_hat: np.ndarray, log_scale: float, *, rank_tol: float) -> tuple[np.ndarray, int]:
    singular = np.linalg.svd(np.asarray(core_hat, dtype=np.complex128), compute_uv=False)
    threshold = float(rank_tol) * max(1.0, float(singular[0]) if singular.size else 1.0)
    valid = np.isfinite(singular) & (singular > threshold)
    logs = np.full(singular.shape, np.nan, dtype=np.float64)
    logs[valid] = float(log_scale) + np.log(singular[valid])
    return logs, int(np.count_nonzero(valid))


def _isotropy_defect(log_singular: np.ndarray, rank: int) -> float:
    if int(rank) < 2 or not np.all(np.isfinite(log_singular[:2])):
        return np.nan
    difference = abs(float(log_singular[0] - log_singular[1]))
    return float(np.tanh(difference))


def _unwrap_with_gaps(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float64)
    out = np.full_like(phase, np.nan)
    finite = np.flatnonzero(np.isfinite(phase))
    if finite.size == 0:
        return out
    split = np.flatnonzero(np.diff(finite) > 1) + 1
    for group in np.split(finite, split):
        out[group] = np.unwrap(phase[group])
    return out


class TangentChannelRecorder:
    """Stream compact diagnostics from the canonical tangent-frame callback.

    This recorder consumes the ``lyapunov_frame_observer`` payload emitted by
    ``run_markov_circuit``.  It never differentiates through a Born draw.  The
    physical images of the predetermined input modes are reconstructed as
    ``Q_t B_t`` using the phase-consistent QR factor and a stabilized restricted
    core.  Only the final ``Q_t`` is retained; the time series contains compact
    two-by-two factors and scalar localization/privacy diagnostics.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        samples: int,
        observation_cycles: int,
        edge_frame: PhysicalEdgeFrame,
        active_indices: np.ndarray | None = None,
        rank_tol: float = 1e-12,
        coherence_tol: float = 1e-14,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.samples = int(samples)
        self.observation_cycles = int(observation_cycles)
        self.edge_frame = edge_frame
        self.vector_dimension = int(edge_frame.frame.shape[0])
        self.active_indices = (
            np.arange(self.vector_dimension, dtype=np.int64)
            if active_indices is None
            else np.asarray(active_indices, dtype=np.int64).reshape(-1)
        )
        self.rank_tol = float(rank_tol)
        self.coherence_tol = float(coherence_tol)
        if self.samples <= 0 or self.observation_cycles <= 0:
            raise ValueError("samples and observation_cycles must be positive.")
        if edge_frame.frame.shape != (2 * self.nx * self.ny, 2):
            raise ValueError(
                f"Expected a two-column full edge frame of shape {(2 * self.nx * self.ny, 2)}, "
                f"got {edge_frame.frame.shape}."
            )
        if self.rank_tol <= 0.0 or self.coherence_tol <= 0.0:
            raise ValueError("rank_tol and coherence_tol must be positive.")

        shape = (self.samples, self.observation_cycles)
        matrix_shape = shape + (2, 2)
        self.absolute_cycle = np.full(shape, -1, dtype=np.int64)
        self.observed = np.zeros(shape, dtype=bool)
        self.active = np.zeros(shape, dtype=bool)
        self.qr_r = np.full(matrix_shape, np.nan + 1j * np.nan, dtype=np.complex128)
        self.log_diag = np.full(shape + (2,), np.nan, dtype=np.float64)
        self.cycle_null_mask = np.zeros(shape + (2,), dtype=bool)
        self.null_count = np.full(shape, -1, dtype=np.int64)
        self.min_branch_probability = np.full(shape, np.nan, dtype=np.float64)
        self.min_abs_born_denominator = np.full(shape, np.nan, dtype=np.float64)
        self.invalid_branch_count = np.full(shape, -1, dtype=np.int64)

        self.core_hat = np.full(matrix_shape, np.nan + 1j * np.nan, dtype=np.complex128)
        self.core_log_scale = np.full(shape, np.nan, dtype=np.float64)
        self.core_rank = np.full(shape, -1, dtype=np.int64)
        self.core_null_count = np.full(shape, -1, dtype=np.int64)
        self.log_singular_values = np.full(shape + (2,), np.nan, dtype=np.float64)
        self.mean_survival_exponent = np.full(shape, np.nan, dtype=np.float64)
        self.isotropy_defect = np.full(shape, np.nan, dtype=np.float64)

        self.target_wall_retention = np.full(shape + (2,), np.nan, dtype=np.float64)
        self.opposite_wall_leakage = np.full(shape + (2,), np.nan, dtype=np.float64)
        self.interface_retention = np.full(shape + (2,), np.nan, dtype=np.float64)
        self.interference_harmonic = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
        self.interference_coherence = np.full(shape, np.nan, dtype=np.float64)

        self.record_fisher_hat = np.full(shape + (3, 3), np.nan, dtype=np.float64)
        self.record_fisher_log_scale = np.full(shape, np.nan, dtype=np.float64)
        self.record_fisher_endpoint_count = np.full(shape, -1, dtype=np.int64)
        self.record_fisher_infinite = np.zeros(shape, dtype=bool)
        self.record_fisher_max_log_eigenvalue = np.full(shape, np.nan, dtype=np.float64)
        self.record_fisher_max_log_density = np.full(shape, np.nan, dtype=np.float64)

        self.final_frame = np.full(
            (self.samples, self.vector_dimension, 2),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        self._local_core_hat = np.broadcast_to(
            np.eye(2, dtype=np.complex128), (self.samples, 2, 2)
        ).copy()
        self._local_core_log_scale = np.zeros((self.samples,), dtype=np.float64)
        self._local_core_null_count = np.zeros((self.samples,), dtype=np.int64)
        self.failure_records: list[dict[str, Any]] = []
        self._failure_signatures: set[str] = set()

        target_x = edge_frame.wall_x[0] if edge_frame.wall == "left" else edge_frame.wall_x[1]
        opposite_x = edge_frame.wall_x[1] if edge_frame.wall == "left" else edge_frame.wall_x[0]
        target_cell = _wall_cell_mask(self.nx, self.ny, target_x, edge_frame.interface_width)
        opposite_cell = _wall_cell_mask(self.nx, self.ny, opposite_x, edge_frame.interface_width)
        self.target_orbital_mask = _orbital_mask(target_cell)
        self.opposite_orbital_mask = _orbital_mask(opposite_cell)
        self.interface_orbital_mask = self.target_orbital_mask | self.opposite_orbital_mask
        self.delta_momentum = float(
            np.angle(np.exp(1j * (edge_frame.momentum[1] - edge_frame.momentum[0])))
        )
        if abs(self.delta_momentum) <= 1e-14:
            raise ValueError("The two edge modes must have distinct momenta.")
        self._y_coordinate = np.repeat(np.arange(self.ny, dtype=np.float64), 2 * self.nx)
        self._harmonic_phase = np.exp(-1j * self.delta_momentum * self._y_coordinate)
        self.initial_interference_harmonic = self._interference(edge_frame.frame)[0]

    def _expand_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.complex128)
        if frame.shape[1:] == (self.vector_dimension, 2):
            return frame
        if frame.shape[1:] == (self.active_indices.size, 2):
            expanded = np.zeros((frame.shape[0], self.vector_dimension, 2), dtype=np.complex128)
            expanded[:, self.active_indices, :] = frame
            return expanded
        raise ValueError(
            f"Unexpected tangent frame shape {frame.shape}; expected row dimension "
            f"{self.vector_dimension} or {self.active_indices.size}."
        )

    def _interference(self, images: np.ndarray) -> tuple[complex, float]:
        first, second = images[:, 0], images[:, 1]
        target = self.target_orbital_mask
        norm_first = float(np.sum(np.abs(first[target]) ** 2))
        norm_second = float(np.sum(np.abs(second[target]) ** 2))
        denominator = np.sqrt(norm_first * norm_second)
        if denominator <= self.coherence_tol:
            return complex(np.nan, np.nan), np.nan
        harmonic = np.sum(first[target].conj() * second[target] * self._harmonic_phase[target])
        return complex(harmonic), float(abs(harmonic) / denominator)

    @staticmethod
    def _mode_weights(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = np.full((images.shape[1],), np.nan, dtype=np.float64)
        for mode in range(images.shape[1]):
            norm = float(np.sum(np.abs(images[:, mode]) ** 2))
            if norm > 0.0 and np.isfinite(norm):
                result[mode] = float(np.sum(np.abs(images[mask, mode]) ** 2) / norm)
        return result

    def _record_failure_records(self, records: Any, *, batch_start: int) -> None:
        for raw in records or ():
            record = dict(raw)
            offset = int(record.get("sample_offset", 0))
            record.setdefault("sample_index", int(batch_start) + offset)
            signature = repr(sorted(record.items(), key=lambda item: str(item[0])))
            if signature not in self._failure_signatures:
                self._failure_signatures.add(signature)
                self.failure_records.append(record)

    def __call__(
        self,
        *,
        cycle: int,
        lyapunov_cycle: int,
        lyapunov_frame: Any,
        lyapunov_qr_r: Any,
        lyapunov_log_diag: Any,
        lyapunov_cycle_null_mask: Any,
        lyapunov_null_counts: Any,
        lyapunov_active_mask: Any,
        lyapunov_min_branch_probability: Any,
        lyapunov_min_abs_born_denominator: Any,
        lyapunov_invalid_branch_count: Any,
        batch_start: int,
        batch_count: int,
        lyapunov_failure_records: Any = (),
        lyapunov_core_hat: Any | None = None,
        lyapunov_core_log_scale: Any | None = None,
        lyapunov_core_null_count: Any | None = None,
        lyapunov_record_fisher_hat: Any | None = None,
        lyapunov_record_fisher_log_scale: Any | None = None,
        lyapunov_record_fisher_endpoint_count: Any | None = None,
        lyapunov_record_fisher_infinite: Any | None = None,
        **_: Any,
    ) -> None:
        elapsed = int(lyapunov_cycle)
        time_index = elapsed - 1
        if not (0 <= time_index < self.observation_cycles):
            raise ValueError(
                f"lyapunov_cycle={elapsed} lies outside 1..{self.observation_cycles}."
            )
        start, count = int(batch_start), int(batch_count)
        stop = start + count
        if start < 0 or stop > self.samples:
            raise ValueError(f"Batch slice {start}:{stop} exceeds {self.samples} samples.")
        sl = slice(start, stop)

        frame = self._expand_frame(_as_numpy(lyapunov_frame, dtype=np.complex128))
        r_factor = _as_numpy(lyapunov_qr_r, dtype=np.complex128)
        if r_factor.shape != (count, 2, 2):
            raise ValueError(f"Unexpected QR-factor shape {r_factor.shape}.")
        if frame.shape != (count, self.vector_dimension, 2):
            raise ValueError(f"Unexpected tangent-frame shape {frame.shape}.")

        self.absolute_cycle[sl, time_index] = int(cycle)
        self.observed[sl, time_index] = True
        self.qr_r[sl, time_index] = r_factor
        self.log_diag[sl, time_index] = _as_numpy(lyapunov_log_diag, dtype=np.float64)
        self.cycle_null_mask[sl, time_index] = _as_numpy(lyapunov_cycle_null_mask, dtype=bool)
        self.null_count[sl, time_index] = _as_numpy(lyapunov_null_counts, dtype=np.int64)
        active = _as_numpy(lyapunov_active_mask, dtype=bool).reshape(-1)
        if active.shape != (count,):
            raise ValueError(f"Unexpected active-mask shape {active.shape}.")
        self.active[sl, time_index] = active
        self.min_branch_probability[sl, time_index] = _as_numpy(
            lyapunov_min_branch_probability, dtype=np.float64
        )
        self.min_abs_born_denominator[sl, time_index] = _as_numpy(
            lyapunov_min_abs_born_denominator, dtype=np.float64
        )
        self.invalid_branch_count[sl, time_index] = _as_numpy(
            lyapunov_invalid_branch_count, dtype=np.int64
        )

        for local, sample in enumerate(range(start, stop)):
            local_hat, local_scale, local_null = _stable_core_step(
                self._local_core_hat[sample],
                self._local_core_log_scale[sample],
                r_factor[local],
            )
            self._local_core_hat[sample] = local_hat
            self._local_core_log_scale[sample] = local_scale
            self._local_core_null_count[sample] += int(local_null)

        if lyapunov_core_hat is None:
            core_hat = self._local_core_hat[sl].copy()
            core_log_scale = self._local_core_log_scale[sl].copy()
            core_null_count = self._local_core_null_count[sl].copy()
        else:
            core_hat = _as_numpy(lyapunov_core_hat, dtype=np.complex128)
            core_log_scale = _as_numpy(lyapunov_core_log_scale, dtype=np.float64).reshape(-1)
            core_null_count = _as_numpy(lyapunov_core_null_count, dtype=np.int64).reshape(-1)
            if core_hat.shape != (count, 2, 2):
                raise ValueError(f"Unexpected restricted-core shape {core_hat.shape}.")

        self.core_hat[sl, time_index] = core_hat
        self.core_log_scale[sl, time_index] = core_log_scale
        self.core_null_count[sl, time_index] = core_null_count
        for local, sample in enumerate(range(start, stop)):
            if not active[local]:
                self.final_frame[sample] = frame[local]
                continue
            logs, rank = _log_singular_values(core_hat[local], core_log_scale[local], rank_tol=self.rank_tol)
            self.log_singular_values[sample, time_index] = logs
            self.core_rank[sample, time_index] = rank
            if rank == 2:
                self.mean_survival_exponent[sample, time_index] = float(np.mean(logs) / elapsed)
            self.isotropy_defect[sample, time_index] = _isotropy_defect(logs, rank)

            images = frame[local] @ core_hat[local]
            self.target_wall_retention[sample, time_index] = self._mode_weights(
                images, self.target_orbital_mask
            )
            self.opposite_wall_leakage[sample, time_index] = self._mode_weights(
                images, self.opposite_orbital_mask
            )
            self.interface_retention[sample, time_index] = self._mode_weights(
                images, self.interface_orbital_mask
            )
            harmonic, coherence = self._interference(images)
            self.interference_harmonic[sample, time_index] = harmonic
            self.interference_coherence[sample, time_index] = coherence
            self.final_frame[sample] = frame[local]

        if lyapunov_record_fisher_hat is not None:
            fisher_hat = _as_numpy(lyapunov_record_fisher_hat, dtype=np.float64)
            fisher_scale = _as_numpy(lyapunov_record_fisher_log_scale, dtype=np.float64).reshape(-1)
            fisher_endpoint = _as_numpy(
                lyapunov_record_fisher_endpoint_count, dtype=np.int64
            ).reshape(-1)
            fisher_infinite = _as_numpy(lyapunov_record_fisher_infinite, dtype=bool).reshape(-1)
            if fisher_hat.shape != (count, 3, 3):
                raise ValueError(f"Unexpected record-Fisher shape {fisher_hat.shape}.")
            self.record_fisher_hat[sl, time_index] = fisher_hat
            self.record_fisher_log_scale[sl, time_index] = fisher_scale
            self.record_fisher_endpoint_count[sl, time_index] = fisher_endpoint
            self.record_fisher_infinite[sl, time_index] = fisher_infinite
            for local, sample in enumerate(range(start, stop)):
                if not active[local]:
                    continue
                hermitian = 0.5 * (fisher_hat[local] + fisher_hat[local].T)
                maximum = float(max(0.0, np.max(np.linalg.eigvalsh(hermitian))))
                if fisher_infinite[local]:
                    self.record_fisher_max_log_eigenvalue[sample, time_index] = np.inf
                    self.record_fisher_max_log_density[sample, time_index] = np.inf
                elif maximum > self.rank_tol and np.isfinite(fisher_scale[local]):
                    value = float(fisher_scale[local] + np.log(maximum))
                    self.record_fisher_max_log_eigenvalue[sample, time_index] = value
                    self.record_fisher_max_log_density[sample, time_index] = value - np.log(elapsed)

        self._record_failure_records(lyapunov_failure_records, batch_start=start)

    def phase_displacement(self) -> np.ndarray:
        """Return unwrapped interference-packet displacement along the wall."""

        displacement = np.full(self.interference_harmonic.shape, np.nan, dtype=np.float64)
        reference_phase = float(np.angle(self.initial_interference_harmonic))
        for sample in range(self.samples):
            phase = np.angle(self.interference_harmonic[sample]) - reference_phase
            phase[self.interference_coherence[sample] <= self.coherence_tol] = np.nan
            displacement[sample] = -_unwrap_with_gaps(phase) / self.delta_momentum
        return displacement

    def velocity_per_sample(self, *, minimum_points: int = 3) -> np.ndarray:
        displacement = self.phase_displacement()
        velocity = np.full((self.samples,), np.nan, dtype=np.float64)
        for sample in range(self.samples):
            valid = self.observed[sample] & np.isfinite(displacement[sample])
            if np.count_nonzero(valid) >= int(minimum_points):
                velocity[sample] = float(
                    np.polyfit(
                        np.arange(1, self.observation_cycles + 1, dtype=np.float64)[valid],
                        displacement[sample, valid],
                        1,
                    )[0]
                )
        return velocity

    def assert_complete(self, *, allow_censored: bool = True) -> None:
        if not np.all(self.observed):
            raise RuntimeError(
                f"The tangent observer missed {int(np.count_nonzero(~self.observed))} sample-cycle entries."
            )
        if not allow_censored and not np.all(self.active):
            raise RuntimeError("At least one tangent trajectory was censored by an invalid fixed branch.")

    def payload(self) -> dict[str, np.ndarray]:
        return {
            "tangent_absolute_cycle": self.absolute_cycle,
            "tangent_observed": self.observed,
            "tangent_active": self.active,
            "tangent_qr_r": self.qr_r,
            "tangent_log_diag": self.log_diag,
            "tangent_cycle_null_mask": self.cycle_null_mask,
            "tangent_null_count": self.null_count,
            "tangent_min_branch_probability": self.min_branch_probability,
            "tangent_min_abs_born_denominator": self.min_abs_born_denominator,
            "tangent_invalid_branch_count": self.invalid_branch_count,
            "tangent_core_hat": self.core_hat,
            "tangent_core_log_scale": self.core_log_scale,
            "tangent_core_rank": self.core_rank,
            "tangent_core_null_count": self.core_null_count,
            "tangent_log_singular_values": self.log_singular_values,
            "tangent_mean_survival_exponent": self.mean_survival_exponent,
            "tangent_isotropy_defect": self.isotropy_defect,
            "tangent_target_wall_retention": self.target_wall_retention,
            "tangent_opposite_wall_leakage": self.opposite_wall_leakage,
            "tangent_interface_retention": self.interface_retention,
            "tangent_interference_harmonic": self.interference_harmonic,
            "tangent_interference_coherence": self.interference_coherence,
            "tangent_phase_displacement": self.phase_displacement(),
            "tangent_velocity_per_sample": self.velocity_per_sample(),
            "tangent_record_fisher_hat": self.record_fisher_hat,
            "tangent_record_fisher_log_scale": self.record_fisher_log_scale,
            "tangent_record_fisher_endpoint_count": self.record_fisher_endpoint_count,
            "tangent_record_fisher_infinite": self.record_fisher_infinite,
            "tangent_record_fisher_max_log_eigenvalue": self.record_fisher_max_log_eigenvalue,
            "tangent_record_fisher_max_log_density": self.record_fisher_max_log_density,
            "tangent_final_frame": self.final_frame,
            "tangent_target_orbital_mask": self.target_orbital_mask,
            "tangent_opposite_orbital_mask": self.opposite_orbital_mask,
            "tangent_interface_orbital_mask": self.interface_orbital_mask,
            "tangent_delta_momentum": np.asarray(self.delta_momentum, dtype=np.float64),
            "tangent_initial_interference_harmonic": np.asarray(
                self.initial_interference_harmonic, dtype=np.complex128
            ),
        }

    def summary_rows(self) -> list[dict[str, Any]]:
        velocity = self.velocity_per_sample()
        rows: list[dict[str, Any]] = []
        for sample in range(self.samples):
            final = np.flatnonzero(self.observed[sample])
            index = int(final[-1]) if final.size else -1
            rows.append(
                {
                    "sample_index": sample,
                    "final_lyapunov_cycle": index + 1,
                    "active_final": bool(self.active[sample, index]) if index >= 0 else False,
                    "mean_survival_exponent_final": (
                        float(self.mean_survival_exponent[sample, index]) if index >= 0 else np.nan
                    ),
                    "isotropy_defect_final": (
                        float(self.isotropy_defect[sample, index]) if index >= 0 else np.nan
                    ),
                    "target_wall_retention_final": (
                        float(np.nanmean(self.target_wall_retention[sample, index])) if index >= 0 else np.nan
                    ),
                    "opposite_wall_leakage_final": (
                        float(np.nanmean(self.opposite_wall_leakage[sample, index])) if index >= 0 else np.nan
                    ),
                    "wall_velocity": float(velocity[sample]),
                    "record_fisher_max_log_density_final": (
                        float(self.record_fisher_max_log_density[sample, index]) if index >= 0 else np.nan
                    ),
                    "invalid_branch_count_final": (
                        int(self.invalid_branch_count[sample, index]) if index >= 0 else -1
                    ),
                    "tangent_null_count_final": int(self.null_count[sample, index]) if index >= 0 else -1,
                    "core_rank_final": int(self.core_rank[sample, index]) if index >= 0 else -1,
                }
            )
        return rows

