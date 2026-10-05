"""Frame-native translated-cut endpoint-packet observables for H1-v3."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from h1_io import save_npz_atomic, sha256_file


SCHEMA = "h1_translated_endpoint_packet_v3"


def checkpoint_cycles(ny: int) -> list[int]:
    """Six inclusive equally spaced checkpoints from Ny through 2*Ny."""

    ny = int(ny)
    if ny <= 0:
        raise ValueError("Ny must be positive")
    return np.rint(np.linspace(ny, 2 * ny, 6)).astype(np.int64).tolist()


def wall_columns(center: int, nx: int, width: int) -> np.ndarray:
    """Return an odd number of periodic columns centered on an actual wall."""

    center, nx, width = int(center), int(nx), int(width)
    if nx <= 0 or width <= 0 or width % 2 != 1 or width > nx:
        raise ValueError("wall widths must be positive odd integers no larger than Nx")
    half = width // 2
    return np.asarray([(center + dx) % nx for dx in range(-half, half + 1)])


def translated_half_indices(
    *, nx: int, ny: int, cut_origin: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """A translated half cylinder in relative-y,x,orbital order."""

    nx, ny, cut_origin = int(nx), int(ny), int(cut_origin)
    if nx <= 0 or ny <= 0 or ny % 2:
        raise ValueError("H1-v3 requires positive Nx and even Ny")
    values = [
        mu + 2 * x + 2 * nx * ((cut_origin + y_relative) % ny)
        for y_relative in range(ny // 2)
        for x in range(nx)
        for mu in (0, 1)
    ]
    return torch.as_tensor(values, dtype=torch.long, device=device)


def reduced_correlation_from_frame(
    frame: torch.Tensor, *, rank: int, indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build and diagonalize a reduced correlation matrix without full covariance."""

    if frame.ndim != 2:
        raise ValueError("frame must have shape (mode,capacity)")
    rank = int(rank)
    if not 0 <= rank <= frame.shape[1]:
        raise ValueError("invalid occupied-frame rank")
    rows = frame.index_select(0, indices.to(frame.device))[:, :rank]
    correlation = rows @ rows.mH
    error = float(torch.max(torch.abs(correlation - correlation.mH)).item())
    correlation = 0.5 * (correlation + correlation.mH)
    occupations, vectors = torch.linalg.eigh(correlation)
    return correlation, occupations.real, vectors, error


def localized_source_matrix(
    *, nx: int, ay: int, walls: Iterable[int], source_widths: Iterable[int],
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, np.ndarray]:
    """Normalized wall/endpoint packets and their (width,wall,endpoint) index."""

    nx, ay = int(nx), int(ay)
    wall_values = tuple(int(value) % nx for value in walls)
    widths = tuple(int(value) for value in source_widths)
    if len(wall_values) != 2 or len(set(wall_values)) != 2:
        raise ValueError("H1-v3 requires exactly two distinct physical walls")
    sources: list[torch.Tensor] = []
    index: list[tuple[int, int, int]] = []
    for width_index, width in enumerate(widths):
        for wall_index, wall in enumerate(wall_values):
            columns = wall_columns(wall, nx, width)
            for endpoint_index, endpoint in enumerate((0, ay - 1)):
                source = torch.zeros(
                    2 * nx * ay, dtype=dtype, device=device
                )
                local_indices = [
                    mu + 2 * int(x) + 2 * nx * endpoint
                    for x in columns for mu in (0, 1)
                ]
                source[torch.as_tensor(local_indices, device=device)] = (
                    1.0 / np.sqrt(len(local_indices))
                )
                sources.append(source)
                index.append((width_index, wall_index, endpoint_index))
    matrix = torch.column_stack(sources)
    return matrix, np.asarray(index, dtype=np.int64)


def packet_probabilities_from_eigensystem(
    *, occupations: torch.Tensor, vectors: torch.Tensor,
    source_matrix: torch.Tensor, modular_times: torch.Tensor,
    epsilon: float, nx: int, ay: int,
) -> torch.Tensor:
    """Return probability with shape (time,relative-y,x,source)."""

    nu = occupations.clamp(float(epsilon), 1.0 - float(epsilon))
    energies = torch.log((1.0 - nu) / nu)
    coefficients = vectors.mH @ source_matrix.to(vectors.device, vectors.dtype)
    phase = torch.exp(-1j * modular_times[:, None] * energies[None, :])
    evolved = torch.einsum(
        "ij,tjs->tis", vectors, phase[:, :, None] * coefficients[None]
    )
    return evolved.abs().square().real.reshape(
        len(modular_times), int(ay), int(nx), 2, source_matrix.shape[1]
    ).sum(dim=3)


def paired_endpoint_drift(centers: np.ndarray, ay: int) -> np.ndarray:
    """Apply the legacy symmetry-cancelling endpoint pairing on the last two axes."""

    values = np.asarray(centers, dtype=np.float64)
    if values.shape[-2] != 2:
        raise ValueError("endpoint center axis must have length two")
    return 0.5 * (values[..., 0, :] + values[..., 1, :] - (int(ay) - 1))


def linear_fit_last_axis(
    times: np.ndarray, values: np.ndarray, window: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit every leading-index curve independently over one locked time window."""

    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    lo, hi = (float(value) for value in window)
    mask = (times >= lo - 1e-12) & (times <= hi + 1e-12)
    count = int(np.count_nonzero(mask))
    if count < 3 or values.shape[-1] != len(times):
        raise ValueError("fit window must contain at least three points on the time axis")
    x = times[mask]
    y = values[..., mask]
    centered_x = x - x.mean()
    denominator = float(centered_x @ centered_x)
    slopes = np.sum((y - y.mean(axis=-1, keepdims=True)) * centered_x, axis=-1) / denominator
    intercepts = y.mean(axis=-1) - slopes * x.mean()
    predicted = intercepts[..., None] + slopes[..., None] * x
    residual = np.sum((y - predicted) ** 2, axis=-1)
    variance = np.sum((y - y.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    r2 = np.ones_like(slopes)
    np.divide(residual, variance, out=r2, where=variance > 0.0)
    r2 = np.where(variance > 0.0, 1.0 - r2, 1.0)
    return slopes, r2, count


class H1EndpointPacketObserver:
    """Evaluate the nonlinear endpoint estimator separately on every translated cut."""

    def __init__(
        self, *, nx: int, ny: int, wall_x: Iterable[int],
        checkpoints: Iterable[int], global_sample_ids: Iterable[int],
        modular_times: Iterable[float], spectral_clip_eps: Iterable[float],
        source_widths: Iterable[int], retention_widths: Iterable[int],
        primary_epsilon: float, primary_source_width: int,
        primary_retention_width: int, fixed_time: float,
        fit_windows: Iterable[Iterable[float]], orientation_signs: Iterable[int],
        minimum_primary_retention: float,
        raw_norm_warning_tolerance: float,
        raw_norm_hard_failure_tolerance: float,
        post_normalization_norm_tolerance: float,
        gram_diagnostic_trigger: float,
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.ay = self.ny // 2
        self.wall_x = np.asarray(sorted(int(value) % self.nx for value in wall_x))
        self.checkpoints = np.asarray(list(checkpoints), dtype=np.int64)
        self.sample_ids = np.asarray(list(global_sample_ids), dtype=np.int64)
        self.times = np.asarray(list(modular_times), dtype=np.float64)
        self.eps = np.asarray(list(spectral_clip_eps), dtype=np.float64)
        self.source_widths = np.asarray(list(source_widths), dtype=np.int64)
        self.retention_widths = np.asarray(list(retention_widths), dtype=np.int64)
        self.fit_windows = np.asarray(
            [[float(value) for value in window] for window in fit_windows],
            dtype=np.float64,
        )
        self.orientation = np.asarray(list(orientation_signs), dtype=np.int64)
        self.minimum_primary_retention = float(minimum_primary_retention)
        self.raw_norm_warning_tolerance = float(raw_norm_warning_tolerance)
        self.raw_norm_hard_failure_tolerance = float(
            raw_norm_hard_failure_tolerance
        )
        self.post_normalization_norm_tolerance = float(
            post_normalization_norm_tolerance
        )
        self.gram_diagnostic_trigger = float(gram_diagnostic_trigger)
        if self.ny % 2 or self.wall_x.shape != (2,) or len(set(self.wall_x)) != 2:
            raise ValueError("H1-v3 requires even Ny and two distinct model walls")
        for width in np.concatenate((self.source_widths, self.retention_widths)):
            wall_columns(0, self.nx, int(width))
        if self.orientation.tolist() != [-1, 1]:
            raise ValueError("wall orientation must remain the exact-benchmark map [-1,+1]")
        if not np.all(np.diff(self.times) > 0.0) or not np.isclose(self.times[0], 0.0):
            raise ValueError("modular times must be strictly increasing from zero")
        numerical_tolerances = {
            "raw_norm_warning_tolerance": self.raw_norm_warning_tolerance,
            "raw_norm_hard_failure_tolerance": self.raw_norm_hard_failure_tolerance,
            "post_normalization_norm_tolerance": self.post_normalization_norm_tolerance,
            "gram_diagnostic_trigger": self.gram_diagnostic_trigger,
        }
        invalid_tolerances = [
            name for name, value in numerical_tolerances.items()
            if not np.isfinite(value) or value <= 0.0
        ]
        if invalid_tolerances:
            raise ValueError(
                f"H1-v3 numerical tolerances must be positive and finite: "
                f"{invalid_tolerances}"
            )
        if self.raw_norm_warning_tolerance >= self.raw_norm_hard_failure_tolerance:
            raise ValueError(
                "raw norm warning tolerance must be below the hard failure tolerance"
            )
        if self.gram_diagnostic_trigger > self.raw_norm_warning_tolerance:
            raise ValueError(
                "Gram diagnostic trigger must not exceed the raw norm warning tolerance"
            )
        self.primary_eps = self._unique_index(self.eps, float(primary_epsilon))
        self.primary_source = self._unique_index(
            self.source_widths, int(primary_source_width)
        )
        self.primary_retention = self._unique_index(
            self.retention_widths, int(primary_retention_width)
        )
        self.fixed_time_index = self._unique_index(self.times, float(fixed_time))
        self.cut_origins = np.arange(self.ny, dtype=np.int64)

        ns, nc, ncut = len(self.sample_ids), len(self.checkpoints), self.ny
        ne, nsw, nrw = len(self.eps), len(self.source_widths), len(self.retention_widths)
        nfit, nt = len(self.fit_windows), len(self.times)
        self.seen = np.zeros((ns, nc), dtype=np.bool_)
        self.seconds = np.zeros((ns, nc), dtype=np.float64)
        self.hermiticity_error = np.full((ns, nc, ncut), np.nan)
        self.occupations = np.full((ns, nc, ncut, 2 * self.nx * self.ay), np.nan)
        self.clip_counts = np.zeros((ns, nc, ncut, ne, 2), dtype=np.int32)
        self.paired_drift = np.full(
            (ns, nc, ncut, ne, nsw, nrw, 2, nt), np.nan
        )
        self.wall_delta = np.full((ns, nc, ncut, ne, nsw, nrw, 2), np.nan)
        self.wall_velocity = np.full(
            (ns, nc, ncut, ne, nsw, nrw, nfit, 2), np.nan
        )
        self.wall_velocity_r2 = np.full_like(self.wall_velocity, np.nan)
        self.handed_delta = np.full((ns, nc, ncut, ne, nsw, nrw), np.nan)
        self.handed_velocity = np.full(
            (ns, nc, ncut, ne, nsw, nrw, nfit), np.nan
        )
        self.minimum_retention = np.full(
            (ns, nc, ncut, ne, nsw, nrw, 2), np.nan
        )
        packet_shape = (ns, nc, ncut, ne, nsw, 2, 2)
        self.raw_packet_total_norm = np.full(packet_shape + (nt,), np.nan)
        self.max_raw_norm_error = np.full(packet_shape, np.nan)
        self.max_raw_norm_drift = np.full(packet_shape, np.nan)
        self.max_post_normalization_norm_error = np.full(packet_shape, np.nan)
        # Keep the v2 attribute as a read-only alias for callers that inspect an
        # observer before serialization.  The stable v3 NPZ names are explicit.
        self.max_norm_drift = self.max_raw_norm_drift
        self.conditional_eigenvector_gram_residual = np.full(
            (ns, nc, ncut), np.nan
        )
        self.conditional_eigenvector_gram_residual_computed = np.zeros(
            (ns, nc, ncut), dtype=np.bool_
        )
        self.actual_frame_dtype: str | None = None
        self.actual_probability_dtype: str | None = None
        primary_shape = (ns, nc, ncut, 2, 2, nt)
        self.primary_endpoint_centers = np.full(primary_shape, np.nan)
        self.primary_endpoint_retention = np.full(primary_shape, np.nan)
        self.primary_conditional_profiles = np.zeros(
            (ns, nc, 2, 2, nt, self.ay), dtype=np.float64
        )
        _, self.source_index = localized_source_matrix(
            nx=self.nx, ay=self.ay, walls=self.wall_x,
            source_widths=self.source_widths,
        )

    @staticmethod
    def _unique_index(values: np.ndarray, target: float | int) -> int:
        # A zero absolute tolerance is essential for logarithmically separated
        # spectral cutoffs: NumPy's default atol=1e-8 makes 1e-8, 1e-10, and
        # 1e-12 all appear close to a target of 1e-10.
        matches = np.flatnonzero(np.isclose(
            values.astype(float), float(target), rtol=1e-12, atol=0.0
        ))
        if len(matches) != 1:
            raise ValueError(f"primary value {target!r} is not unique in {values.tolist()}")
        return int(matches[0])

    def __call__(
        self, *, cycle: int, state: Any, batch_start: int,
        batch_count: int, **_: Any,
    ) -> None:
        matches = np.flatnonzero(self.checkpoints == int(cycle))
        if not len(matches):
            return
        checkpoint_index = int(matches[0])
        start, stop = int(batch_start), int(batch_start) + int(batch_count)
        if stop > len(self.sample_ids) or np.any(self.seen[start:stop, checkpoint_index]):
            raise RuntimeError("invalid or duplicate H1-v3 observer batch")
        if not hasattr(state, "frame") or not hasattr(state, "ranks"):
            raise TypeError("H1-v3 endpoint observer requires occupied-frame state")
        diagnostic_horizon = max(
            float(self.times[self.fixed_time_index]), float(np.max(self.fit_windows[:, 1]))
        )
        retention_time_mask = self.times <= diagnostic_horizon + 1e-12

        for local in range(start, stop):
            began = time.perf_counter()
            batch_local = local - start
            frame = state.frame[batch_local].detach()
            frame_dtype = str(frame.dtype)
            if self.actual_frame_dtype is None:
                self.actual_frame_dtype = frame_dtype
            elif self.actual_frame_dtype != frame_dtype:
                raise TypeError(
                    "H1-v3 observer received inconsistent frame dtypes: "
                    f"{self.actual_frame_dtype!r} then {frame_dtype!r}"
                )
            rank = int(state.ranks[batch_local].item())
            source_matrix, source_index = localized_source_matrix(
                nx=self.nx, ay=self.ay, walls=self.wall_x,
                source_widths=self.source_widths,
                dtype=frame.dtype, device=frame.device,
            )
            time_device = torch.as_tensor(
                self.times, dtype=frame.real.dtype, device=frame.device
            )
            y_device = torch.arange(
                self.ay, dtype=frame.real.dtype, device=frame.device
            )
            for cut_index, cut_origin in enumerate(self.cut_origins):
                indices = translated_half_indices(
                    nx=self.nx, ny=self.ny, cut_origin=int(cut_origin),
                    device=frame.device,
                )
                correlation, occupations, vectors, hermiticity = (
                    reduced_correlation_from_frame(frame, rank=rank, indices=indices)
                )
                self.hermiticity_error[local, checkpoint_index, cut_index] = hermiticity
                self.occupations[local, checkpoint_index, cut_index] = (
                    occupations.detach().cpu().numpy()
                )
                for eps_index, eps in enumerate(self.eps):
                    self.clip_counts[local, checkpoint_index, cut_index, eps_index, 0] = int(
                        torch.count_nonzero(occupations < eps).item()
                    )
                    self.clip_counts[local, checkpoint_index, cut_index, eps_index, 1] = int(
                        torch.count_nonzero(occupations > 1.0 - eps).item()
                    )
                    probability = packet_probabilities_from_eigensystem(
                        occupations=occupations, vectors=vectors,
                        source_matrix=source_matrix, modular_times=time_device,
                        epsilon=float(eps), nx=self.nx, ay=self.ay,
                    )
                    probability_dtype = str(probability.dtype)
                    if self.actual_probability_dtype is None:
                        self.actual_probability_dtype = probability_dtype
                    elif self.actual_probability_dtype != probability_dtype:
                        raise TypeError(
                            "H1-v3 observer received inconsistent probability dtypes: "
                            f"{self.actual_probability_dtype!r} then "
                            f"{probability_dtype!r}"
                        )

                    raw_norms = probability.sum(dim=(1, 2))
                    if (
                        not bool(torch.isfinite(raw_norms).all().item())
                        or bool(torch.any(raw_norms <= 0.0).item())
                    ):
                        raise FloatingPointError(
                            "H1-v3 packet propagation produced a non-finite or "
                            "nonpositive raw total norm"
                        )
                    raw_norm_error = torch.abs(raw_norms - 1.0)
                    raw_norm_drift = torch.abs(raw_norms - raw_norms[:1])
                    raw_norms_np = raw_norms.detach().cpu().numpy()
                    packet_slot = (
                        local, checkpoint_index, cut_index, eps_index
                    )
                    packet_axes = (len(self.source_widths), 2, 2)
                    self.raw_packet_total_norm[packet_slot] = raw_norms_np.reshape(
                        (len(self.times), *packet_axes)
                    ).transpose(1, 2, 3, 0)
                    self.max_raw_norm_error[packet_slot] = (
                        raw_norm_error.max(dim=0).values.detach().cpu().numpy().reshape(
                            packet_axes
                        )
                    )
                    self.max_raw_norm_drift[packet_slot] = (
                        raw_norm_drift.max(dim=0).values.detach().cpu().numpy().reshape(
                            packet_axes
                        )
                    )

                    gram_slot = (local, checkpoint_index, cut_index)
                    if (
                        float(raw_norm_error.max().item())
                        > self.gram_diagnostic_trigger
                        and not np.any(
                            self.conditional_eigenvector_gram_residual_computed[
                                local, checkpoint_index
                            ]
                        )
                    ):
                        identity = torch.eye(
                            vectors.shape[-1], dtype=vectors.dtype,
                            device=vectors.device,
                        )
                        gram_residual = torch.max(
                            torch.abs(vectors.mH @ vectors - identity)
                        )
                        if not bool(torch.isfinite(gram_residual).item()):
                            raise FloatingPointError(
                                "H1-v3 eigenvector Gram diagnostic is non-finite"
                            )
                        self.conditional_eigenvector_gram_residual[gram_slot] = float(
                            gram_residual.item()
                        )
                        self.conditional_eigenvector_gram_residual_computed[
                            gram_slot
                        ] = True

                    probability = probability / raw_norms[:, None, None, :]
                    post_norms = probability.sum(dim=(1, 2))
                    if (
                        not bool(torch.isfinite(post_norms).all().item())
                        or bool(torch.any(post_norms <= 0.0).item())
                    ):
                        raise FloatingPointError(
                            "H1-v3 normalized packet has a non-finite or "
                            "nonpositive total norm"
                        )
                    self.max_post_normalization_norm_error[packet_slot] = (
                        torch.abs(post_norms - 1.0)
                        .max(dim=0).values.detach().cpu().numpy().reshape(packet_axes)
                    )

                    centers = torch.empty(
                        (len(self.source_widths), len(self.retention_widths), 2, 2, len(self.times)),
                        dtype=frame.real.dtype, device=frame.device,
                    )
                    retention = torch.empty_like(centers)
                    for source_column, (source_width_index, wall_index, endpoint_index) in enumerate(source_index):
                        for retention_index, retention_width in enumerate(self.retention_widths):
                            columns = torch.as_tensor(
                                wall_columns(
                                    int(self.wall_x[wall_index]), self.nx,
                                    int(retention_width),
                                ),
                                dtype=torch.long, device=frame.device,
                            )
                            profile = probability[:, :, columns, source_column].sum(dim=2)
                            retained = profile.sum(dim=1)
                            center = torch.sum(profile * y_device[None], dim=1) / retained.clamp_min(1e-300)
                            slot = (
                                int(source_width_index), int(retention_index),
                                int(wall_index), int(endpoint_index),
                            )
                            retention[slot] = retained
                            centers[slot] = center
                            if (
                                eps_index == self.primary_eps
                                and source_width_index == self.primary_source
                                and retention_index == self.primary_retention
                            ):
                                conditional = profile / retained[:, None].clamp_min(1e-300)
                                self.primary_conditional_profiles[
                                    local, checkpoint_index, wall_index, endpoint_index
                                ] += conditional.detach().cpu().numpy() / len(self.cut_origins)

                    centers_np = centers.detach().cpu().numpy()
                    retention_np = retention.detach().cpu().numpy()
                    drift = paired_endpoint_drift(centers_np, self.ay)
                    base = (local, checkpoint_index, cut_index, eps_index)
                    self.paired_drift[base] = drift
                    delta = drift[..., self.fixed_time_index] - drift[..., 0]
                    self.wall_delta[base] = delta
                    self.handed_delta[base] = np.mean(
                        delta * self.orientation[None, None, :], axis=-1
                    )
                    self.minimum_retention[base] = np.min(
                        retention_np[..., retention_time_mask], axis=(-1, -2)
                    )
                    for fit_index, fit_window in enumerate(self.fit_windows):
                        slope, r2, _ = linear_fit_last_axis(
                            self.times, drift, fit_window
                        )
                        self.wall_velocity[base + (slice(None), slice(None), fit_index)] = slope
                        self.wall_velocity_r2[base + (slice(None), slice(None), fit_index)] = r2
                        self.handed_velocity[base + (slice(None), slice(None), fit_index)] = np.mean(
                            slope * self.orientation[None, None, :], axis=-1
                        )
                    if eps_index == self.primary_eps:
                        self.primary_endpoint_centers[
                            local, checkpoint_index, cut_index
                        ] = centers_np[self.primary_source, self.primary_retention]
                        self.primary_endpoint_retention[
                            local, checkpoint_index, cut_index
                        ] = retention_np[self.primary_source, self.primary_retention]
                    del probability, centers, retention
                del correlation, occupations, vectors
                if (cut_index + 1) % 10 == 0 or cut_index + 1 == len(self.cut_origins):
                    print(
                        f"[H1-v3 packet] sample={int(self.sample_ids[local])} "
                        f"checkpoint={int(cycle)} cuts={cut_index + 1}/{len(self.cut_origins)}",
                        flush=True,
                    )
            self.seconds[local, checkpoint_index] = time.perf_counter() - began
            self.seen[local, checkpoint_index] = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _raw_norm_error_argmax(self) -> dict[str, Any]:
        errors = np.abs(self.raw_packet_total_norm - 1.0)
        index = tuple(int(value) for value in np.unravel_index(
            int(np.argmax(errors)), errors.shape
        ))
        (
            sample_index,
            checkpoint_index,
            cut_index,
            epsilon_index,
            source_width_index,
            wall_index,
            endpoint_index,
            time_index,
        ) = index
        return {
            "array_index": list(index),
            "error": float(errors[index]),
            "sample_index": sample_index,
            "global_sample_id": int(self.sample_ids[sample_index]),
            "checkpoint_index": checkpoint_index,
            "checkpoint_cycle": int(self.checkpoints[checkpoint_index]),
            "cut_index": cut_index,
            "cut_origin": int(self.cut_origins[cut_index]),
            "epsilon_index": epsilon_index,
            "spectral_clip_epsilon": float(self.eps[epsilon_index]),
            "source_width_index": source_width_index,
            "source_width_columns": int(self.source_widths[source_width_index]),
            "wall_index": wall_index,
            "wall_x": int(self.wall_x[wall_index]),
            "wall_orientation_sign": int(self.orientation[wall_index]),
            "endpoint_index": endpoint_index,
            "endpoint_relative_y": int((0, self.ay - 1)[endpoint_index]),
            "modular_time_index": time_index,
            "modular_time": float(self.times[time_index]),
            "raw_total_norm": float(self.raw_packet_total_norm[index]),
        }

    def validate(self) -> dict[str, Any]:
        if not self.seen.all():
            raise RuntimeError("one or more trajectory-checkpoint H1-v3 products are missing")
        required = {
            "occupations": self.occupations,
            "paired_drift": self.paired_drift,
            "wall_delta": self.wall_delta,
            "wall_velocity": self.wall_velocity,
            "wall_velocity_r2": self.wall_velocity_r2,
            "handed_delta": self.handed_delta,
            "handed_velocity": self.handed_velocity,
            "minimum_retention": self.minimum_retention,
            "raw_packet_total_norm": self.raw_packet_total_norm,
            "max_raw_norm_error": self.max_raw_norm_error,
            "max_raw_norm_drift": self.max_raw_norm_drift,
            "max_post_normalization_norm_error": (
                self.max_post_normalization_norm_error
            ),
            "primary_endpoint_centers": self.primary_endpoint_centers,
            "primary_endpoint_retention": self.primary_endpoint_retention,
            "primary_conditional_profiles": self.primary_conditional_profiles,
        }
        bad = [name for name, value in required.items() if not np.isfinite(value).all()]
        if bad:
            raise FloatingPointError(f"non-finite H1-v3 products: {bad}")
        if np.any(self.raw_packet_total_norm <= 0.0):
            raise FloatingPointError("H1-v3 raw packet norms must remain positive")
        computed_gram = self.conditional_eigenvector_gram_residual_computed
        gram_values = self.conditional_eigenvector_gram_residual
        if np.any(~np.isfinite(gram_values[computed_gram])):
            raise FloatingPointError("non-finite H1-v3 conditional Gram diagnostics")
        if np.any(np.isfinite(gram_values[~computed_gram])):
            raise RuntimeError("H1-v3 Gram diagnostic value exists without its computed flag")
        if self.actual_frame_dtype is None or self.actual_probability_dtype is None:
            raise RuntimeError("H1-v3 observer did not record its actual tensor dtypes")

        dtype_contract_failure = bool(
            self.actual_frame_dtype != "torch.complex128"
            or self.actual_probability_dtype != "torch.float64"
        )

        hermiticity = float(np.max(self.hermiticity_error))
        maximum_raw_norm_error = float(np.max(self.max_raw_norm_error))
        maximum_raw_norm_drift = float(np.max(self.max_raw_norm_drift))
        maximum_post_normalization_norm_error = float(
            np.max(self.max_post_normalization_norm_error)
        )
        primary_retention = float(np.min(
            self.minimum_retention[
                ..., self.primary_eps, self.primary_source,
                self.primary_retention, :,
            ]
        ))
        raw_norm_warning = bool(
            maximum_raw_norm_error > self.raw_norm_warning_tolerance
        )
        raw_norm_hard_failure = bool(
            maximum_raw_norm_error > self.raw_norm_hard_failure_tolerance
        )
        post_normalization_norm_failure = bool(
            maximum_post_normalization_norm_error
            > self.post_normalization_norm_tolerance
        )
        hermiticity_failure = bool(hermiticity > 1e-10)
        numerical_status = (
            "hard_failure"
            if (
                raw_norm_hard_failure
                or post_normalization_norm_failure
                or hermiticity_failure
                or dtype_contract_failure
            )
            else "warning" if raw_norm_warning else "pass"
        )
        if np.any(computed_gram):
            maximum_conditional_gram_residual: float | None = float(
                np.max(gram_values[computed_gram])
            )
        else:
            maximum_conditional_gram_residual = None
        raw_norm_error_argmax = self._raw_norm_error_argmax()
        return {
            "schema": SCHEMA,
            "samples": len(self.sample_ids),
            "checkpoints": self.checkpoints.tolist(),
            "translated_cuts_per_trajectory_checkpoint": len(self.cut_origins),
            "packets_per_cut_and_epsilon": int(len(self.source_widths) * 4),
            "maximum_hermiticity_error": hermiticity,
            "maximum_raw_packet_norm_error": maximum_raw_norm_error,
            "maximum_raw_packet_norm_drift": maximum_raw_norm_drift,
            "maximum_post_normalization_norm_error": (
                maximum_post_normalization_norm_error
            ),
            # Temporary compatibility for v2 readers; this is explicitly the raw
            # pre-normalization temporal drift in the v3 schema.
            "maximum_packet_norm_drift": maximum_raw_norm_drift,
            "maximum_conditional_eigenvector_gram_residual": (
                maximum_conditional_gram_residual
            ),
            "conditional_eigenvector_gram_diagnostics": int(
                np.count_nonzero(computed_gram)
            ),
            "raw_norm_warning_tolerance": self.raw_norm_warning_tolerance,
            "raw_norm_hard_failure_tolerance": (
                self.raw_norm_hard_failure_tolerance
            ),
            "post_normalization_norm_tolerance": (
                self.post_normalization_norm_tolerance
            ),
            "gram_diagnostic_trigger": self.gram_diagnostic_trigger,
            "raw_norm_warning": raw_norm_warning,
            "raw_norm_hard_failure": raw_norm_hard_failure,
            "post_normalization_norm_failure": post_normalization_norm_failure,
            "hermiticity_failure": hermiticity_failure,
            "dtype_contract_failure": dtype_contract_failure,
            "numerical_status": numerical_status,
            "raw_norm_error_argmax": raw_norm_error_argmax,
            "actual_dtype": self.actual_frame_dtype,
            "actual_probability_dtype": self.actual_probability_dtype,
            "minimum_primary_retention": primary_retention,
            "primary_retention_pass": bool(
                primary_retention >= self.minimum_primary_retention
            ),
            "observer_seconds": float(self.seconds.sum()),
        }

    def save(self, directory: Path | str, *, config: dict[str, Any]) -> dict[str, Any]:
        diagnostics = self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        common = directory / "common.npz"
        packet = directory / "packet_drift.npz"
        profiles = directory / "primary_profiles.npz"
        save_npz_atomic(
            common,
            schema=np.asarray(SCHEMA),
            config_json=np.asarray(json.dumps(config, sort_keys=True)),
            checkpoints=self.checkpoints,
            global_sample_ids=self.sample_ids,
            cut_origins=self.cut_origins,
            modular_times=self.times,
            wall_x=self.wall_x,
            wall_orientation_signs=self.orientation,
            source_width_columns=self.source_widths,
            retention_width_columns=self.retention_widths,
            spectral_clip_eps=self.eps,
            fit_windows=self.fit_windows,
            primary_epsilon_index=np.asarray(self.primary_eps, dtype=np.int64),
            primary_source_width_index=np.asarray(self.primary_source, dtype=np.int64),
            primary_retention_width_index=np.asarray(self.primary_retention, dtype=np.int64),
            fixed_time_index=np.asarray(self.fixed_time_index, dtype=np.int64),
            source_index=self.source_index,
            translated_restricted_occupation_spectrum=self.occupations,
            spectral_clip_counts=self.clip_counts,
            restricted_correlation_hermiticity_error=self.hermiticity_error,
            conditional_eigenvector_gram_residual=(
                self.conditional_eigenvector_gram_residual
            ),
            conditional_eigenvector_gram_residual_computed=(
                self.conditional_eigenvector_gram_residual_computed
            ),
            actual_dtype=np.asarray(self.actual_frame_dtype),
            actual_probability_dtype=np.asarray(self.actual_probability_dtype),
            raw_norm_warning_tolerance=np.asarray(
                self.raw_norm_warning_tolerance, dtype=np.float64
            ),
            raw_norm_hard_failure_tolerance=np.asarray(
                self.raw_norm_hard_failure_tolerance, dtype=np.float64
            ),
            post_normalization_norm_tolerance=np.asarray(
                self.post_normalization_norm_tolerance, dtype=np.float64
            ),
            gram_diagnostic_trigger=np.asarray(
                self.gram_diagnostic_trigger, dtype=np.float64
            ),
            observer_seconds=self.seconds,
        )
        save_npz_atomic(
            packet,
            schema=np.asarray(SCHEMA),
            paired_endpoint_wall_drift=self.paired_drift,
            wall_delta_at_fixed_time=self.wall_delta,
            oriented_handed_delta=self.handed_delta,
            wall_velocity=self.wall_velocity,
            wall_velocity_r2=self.wall_velocity_r2,
            oriented_handed_velocity=self.handed_velocity,
            minimum_wall_retention=self.minimum_retention,
            raw_packet_total_norm=self.raw_packet_total_norm,
            maximum_raw_packet_norm_error=self.max_raw_norm_error,
            maximum_raw_packet_norm_drift=self.max_raw_norm_drift,
            maximum_post_normalization_norm_error=(
                self.max_post_normalization_norm_error
            ),
            # Compatibility alias for readers of the superseded v2 product.
            maximum_packet_norm_drift=self.max_raw_norm_drift,
            raw_norm_error_argmax_json=np.asarray(json.dumps(
                diagnostics["raw_norm_error_argmax"], sort_keys=True
            )),
            raw_norm_warning=np.asarray(
                diagnostics["raw_norm_warning"], dtype=np.bool_
            ),
            raw_norm_hard_failure=np.asarray(
                diagnostics["raw_norm_hard_failure"], dtype=np.bool_
            ),
            post_normalization_norm_failure=np.asarray(
                diagnostics["post_normalization_norm_failure"], dtype=np.bool_
            ),
            dtype_contract_failure=np.asarray(
                diagnostics["dtype_contract_failure"], dtype=np.bool_
            ),
            numerical_status=np.asarray(diagnostics["numerical_status"]),
        )
        save_npz_atomic(
            profiles,
            schema=np.asarray(SCHEMA),
            primary_endpoint_center=self.primary_endpoint_centers,
            primary_endpoint_retention=self.primary_endpoint_retention,
            primary_cut_mean_conditional_profile=self.primary_conditional_profiles,
        )
        files = [
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (common, packet, profiles)
        ]
        return {
            **diagnostics,
            "path": str(directory),
            "files": files,
            "bytes": int(sum(row["bytes"] for row in files)),
            "retarded_response_archived": False,
        }
