from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from production_runtime import save_npz_atomic, sha256_file


H1_SCHEMA = "h1_live_parent_modular_transport_v1"
H2_SCHEMA = "h2_live_parent_born_response_v1"


class CompositeCycleObserver:
    def __init__(self, *observers: Any) -> None:
        self.observers = tuple(observer for observer in observers if observer is not None)

    def __call__(self, **payload: Any) -> None:
        for observer in self.observers:
            observer(**payload)


class TransientStateBank:
    """CPU-local live-state bank; its arrays are never serialized."""

    def __init__(self, *, samples: int, cycles: Iterable[int]) -> None:
        self.samples = int(samples)
        self.cycles = {int(cycle) for cycle in cycles}
        self.states: dict[int, np.ndarray] = {}
        self.seen: dict[int, np.ndarray] = {}

    def __call__(
        self, *, cycle: int, G: Any, batch_start: int, batch_count: int, **_: Any
    ) -> None:
        cycle = int(cycle)
        if cycle not in self.cycles:
            return
        start, stop = int(batch_start), int(batch_start) + int(batch_count)
        value = G.detach().cpu().numpy()
        if cycle not in self.states:
            self.states[cycle] = np.empty(
                (self.samples, value.shape[-2], value.shape[-1]), dtype=np.complex128
            )
            self.seen[cycle] = np.zeros((self.samples,), dtype=np.bool_)
        if np.any(self.seen[cycle][start:stop]):
            raise RuntimeError(f"duplicate transient covariance at cycle {cycle}")
        self.states[cycle][start:stop] = value
        self.seen[cycle][start:stop] = True

    def validate(self) -> None:
        missing = {
            cycle: np.flatnonzero(~self.seen.get(cycle, np.zeros(self.samples, dtype=bool))).tolist()
            for cycle in sorted(self.cycles)
            if cycle not in self.seen or not np.all(self.seen[cycle])
        }
        if missing:
            raise RuntimeError(f"transient state bank is incomplete: {missing}")

    def pop(self, cycle: int) -> np.ndarray:
        cycle = int(cycle)
        if cycle not in self.states or not np.all(self.seen[cycle]):
            raise KeyError(f"complete transient state for cycle {cycle} is unavailable")
        self.seen.pop(cycle, None)
        return self.states.pop(cycle)

    def get(self, cycle: int) -> np.ndarray:
        cycle = int(cycle)
        if cycle not in self.states or not np.all(self.seen[cycle]):
            raise KeyError(f"complete transient state for cycle {cycle} is unavailable")
        return self.states[cycle]

    def clear(self) -> None:
        self.states.clear()
        self.seen.clear()

    @property
    def transient_bytes(self) -> int:
        return int(sum(value.nbytes for value in self.states.values()))


def _wall_x(nx: int) -> tuple[int, int]:
    half = int(nx) // 2
    width = max(1, int(nx) // 4)
    return max(0, half - width), min(int(nx), half + width + 1) - 1


def _half_window_indices(nx: int, ny: int, y0: int) -> np.ndarray:
    return np.asarray(
        [
            orbital + 2 * x + 2 * nx * ((y0 + yrel) % ny)
            for yrel in range(ny // 2)
            for x in range(nx)
            for orbital in (0, 1)
        ],
        dtype=np.int64,
    )


def _packet_basis(nx: int, ny_sub: int, wall_x: tuple[int, int], device: Any, dtype: Any) -> Any:
    import torch

    packets = torch.zeros((2 * nx * ny_sub, 4), dtype=dtype, device=device)
    column = 0
    for x in wall_x:
        for yrel in (0, ny_sub - 1):
            for orbital in (0, 1):
                packets[orbital + 2 * x + 2 * nx * yrel, column] = 1 / math.sqrt(2)
            column += 1
    return packets


def _linear_velocity(times: np.ndarray, displacement: np.ndarray, valid: np.ndarray) -> tuple[float, float, int]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(displacement)
    if np.count_nonzero(mask) < 5:
        return float("nan"), float("nan"), int(np.count_nonzero(mask))
    x = times[mask]
    y = displacement[mask]
    design = np.column_stack((np.ones_like(x), x))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if denom <= 0 else 1.0 - float(np.sum(residual**2)) / denom
    return float(beta[1]), r2, int(x.size)


def compute_h1_live_products(
    *,
    state_bank: TransientStateBank,
    nx: int,
    ny: int,
    cycles: Iterable[int],
    config: dict[str, Any],
    path: Path | str,
    device: str,
    smoke: bool = False,
) -> dict[str, Any]:
    import torch

    cycles = [int(cycle) for cycle in cycles]
    eps_values = np.asarray(config["spectral_clip_eps"], dtype=np.float64)
    widths = np.asarray(config["wall_half_widths"], dtype=np.int64)
    times = np.linspace(
        float(config["modular_time_start"]),
        float(config["modular_time_stop"]),
        int(config["modular_time_points"]),
        dtype=np.float64,
    )
    if smoke:
        times = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    wall_x = _wall_x(nx)
    ny_sub = ny // 2
    y0_values = list(range(ny)) if not smoke else list(range(min(2, ny)))
    samples = state_bank.states[cycles[0]].shape[0]
    dim = 2 * nx * ny_sub
    shape = (samples, len(cycles), len(eps_values), 2, 2)
    velocities = np.zeros(shape + (len(widths),), dtype=np.float64)
    velocity_r2 = np.zeros_like(velocities)
    fit_points = np.zeros_like(velocities, dtype=np.int16)
    velocity_cut_count = np.zeros_like(velocities, dtype=np.int16)
    spectra = np.full((samples, len(cycles), len(y0_values), dim), np.nan, dtype=np.float64)
    clip_counts = np.zeros((samples, len(cycles), len(y0_values), len(eps_values), 2), dtype=np.int32)
    primary_eps = int(np.argmin(np.abs(eps_values - float(config["primary_spectral_clip_eps"]))))
    primary_width = int(np.argmin(np.abs(widths - int(config["primary_wall_half_width"]))))
    density_sum = np.zeros(
        (samples, len(cycles), 2, 2, len(times), ny_sub), dtype=np.float64
    )
    retention_sum = np.zeros(
        (samples, len(cycles), 2, 2, len(times)), dtype=np.float64
    )
    cut_count = np.zeros((samples, len(cycles)), dtype=np.int16)

    time_device = torch.as_tensor(times, dtype=torch.float64, device=device)
    packets = _packet_basis(nx, ny_sub, wall_x, device, torch.complex128)
    y_coordinates = np.arange(ny_sub, dtype=np.float64)
    wall_masks = {
        (wall_index, width_index): np.asarray(
            [
                min((x - xcenter) % nx, (xcenter - x) % nx) <= width
                for x in range(nx)
            ]
        )
        for wall_index, xcenter in enumerate(wall_x)
        for width_index, width in enumerate(widths)
    }

    # The ensemble shard is the GPU batch.  Batched Hermitian diagonalization and
    # packet propagation avoid launching the expensive H1 path once per trajectory.
    for cycle_index, cycle in enumerate(
        tqdm(cycles, desc="H1 observation cycles", unit="cycle-state")
    ):
        full = torch.as_tensor(
            state_bank.get(cycle), dtype=torch.complex128, device=device
        )
        for y0_index, y0 in enumerate(
            tqdm(y0_values, desc=f"H1 translated cuts t={cycle}", unit="cut", leave=False)
        ):
            idx = torch.as_tensor(
                _half_window_indices(nx, ny, y0), dtype=torch.long, device=device
            )
            restricted = full.index_select(1, idx).index_select(2, idx)
            restricted = 0.5 * (restricted + restricted.conj().transpose(-2, -1))
            gvals, vectors = torch.linalg.eigh(restricted)
            gvals = gvals.real
            spectra[:, cycle_index, y0_index] = gvals.detach().cpu().numpy()
            coeff = vectors.conj().transpose(-2, -1) @ packets[None, :, :]
            for eps_index, eps in enumerate(eps_values):
                low = gvals < (-1.0 + eps)
                high = gvals > (1.0 - eps)
                clip_counts[:, cycle_index, y0_index, eps_index, 0] = (
                    low.sum(dim=1).detach().cpu().numpy()
                )
                clip_counts[:, cycle_index, y0_index, eps_index, 1] = (
                    high.sum(dim=1).detach().cpu().numpy()
                )
                hvals = -2.0 * torch.atanh(
                    torch.clamp(gvals, -1.0 + eps, 1.0 - eps)
                )
                phase = torch.exp(-1j * time_device[None, :, None] * hvals[:, None, :])
                evolved = torch.einsum(
                    "bij,btjk->btik", vectors, phase[:, :, :, None] * coeff[:, None]
                )
                probability_np = (
                    evolved.abs()
                    .square()
                    .reshape(samples, len(times), ny_sub, nx, 2, 4)
                    .sum(dim=4)
                    .detach()
                    .cpu()
                    .numpy()
                )
                for sample in range(samples):
                    for wall_index, _xcenter in enumerate(wall_x):
                        for endpoint in range(2):
                            packet_index = 2 * wall_index + endpoint
                            y_density = probability_np[sample, ..., packet_index].sum(axis=2)
                            source_y = 0 if endpoint == 0 else ny_sub - 1
                            density_norm = np.maximum(y_density.sum(axis=1), 1e-300)
                            displacement = (
                                y_density @ (y_coordinates - source_y)
                            ) / density_norm
                            for width_index, _width in enumerate(widths):
                                retention = probability_np[
                                    sample,
                                    :,
                                    :,
                                    wall_masks[(wall_index, width_index)],
                                    packet_index,
                                ].sum(axis=(1, 2))
                                valid = (
                                    (retention >= float(config["minimum_wall_retention"]))
                                    & (np.abs(displacement) <= ny_sub / 3)
                                )
                                velocity, r2, count = _linear_velocity(
                                    times, displacement, valid
                                )
                                slot = (
                                    sample,
                                    cycle_index,
                                    eps_index,
                                    wall_index,
                                    endpoint,
                                    width_index,
                                )
                                if np.isfinite(velocity):
                                    velocities[slot] += velocity
                                    velocity_r2[slot] += r2
                                    fit_points[slot] += count
                                    velocity_cut_count[slot] += 1
                                if (
                                    eps_index == primary_eps
                                    and width_index == primary_width
                                ):
                                    density_sum[
                                        sample, cycle_index, wall_index, endpoint
                                    ] += y_density
                                    retention_sum[
                                        sample, cycle_index, wall_index, endpoint
                                    ] += retention
            cut_count[:, cycle_index] += 1
        del full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    divisor = np.maximum(cut_count[:, :, None, None, None, None], 1)
    density_mean = density_sum / divisor
    retention_mean = retention_sum / np.maximum(cut_count[:, :, None, None, None], 1)
    # The per-cut velocity arrays above are accumulators; normalize them by translated cuts.
    norm = np.maximum(velocity_cut_count, 1)
    velocities = velocities / norm
    velocity_r2 = velocity_r2 / norm
    fit_points = np.rint(fit_points / norm).astype(np.int16)
    velocities[velocity_cut_count == 0] = np.nan
    velocity_r2[velocity_cut_count == 0] = np.nan
    path = Path(path)
    save_npz_atomic(
        path,
        schema=np.asarray(H1_SCHEMA),
        cycles=np.asarray(cycles, dtype=np.int64),
        y0_values=np.asarray(y0_values, dtype=np.int64),
        modular_times=times,
        spectral_clip_eps=eps_values,
        wall_half_widths=widths,
        wall_x=np.asarray(wall_x, dtype=np.int64),
        modular_restricted_spectra=spectra,
        spectral_clip_counts=clip_counts,
        velocity=velocities,
        velocity_r2=velocity_r2,
        fit_points=fit_points,
        velocity_translated_cut_count=velocity_cut_count,
        primary_shifted_density=density_mean,
        primary_wall_retention=retention_mean,
        translated_cut_count=cut_count,
        gpu_trajectory_batch_size=np.asarray(samples, dtype=np.int64),
        gpu_batching_contract=np.asarray(
            "batched_eigh_and_packet_propagation_over_complete_shard_v1"
        ),
    )
    return {"schema": H1_SCHEMA, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


class WallDensityObserver:
    def __init__(self, *, samples: int, nx: int, ny: int, cycles: int, wall_half_width: int) -> None:
        self.samples, self.nx, self.ny, self.cycles = int(samples), int(nx), int(ny), int(cycles)
        self.wall_x = _wall_x(nx)
        self.wall_half_width = int(wall_half_width)
        self.density = np.full((samples, cycles + 1, 2, ny), np.nan, dtype=np.float64)
        self._density_device: Any | None = None

    def materialize_cpu(self) -> None:
        if self._density_device is None:
            return
        self.density = self._density_device.detach().cpu().numpy()
        self._density_device = None

    def __call__(self, *, cycle: int, G: Any, batch_start: int, batch_count: int, **_: Any) -> None:
        import torch

        cycle = int(cycle)
        if not 0 <= cycle <= self.cycles:
            return
        Cdiag = 0.5 * (torch.diagonal(G, dim1=-2, dim2=-1).real + 1.0)
        spatial = Cdiag.reshape(int(batch_count), self.ny, self.nx, 2).sum(dim=-1).transpose(1, 2)
        start, stop = int(batch_start), int(batch_start) + int(batch_count)
        if G.device.type == "cuda" and self._density_device is None:
            self._density_device = torch.full(
                (self.samples, self.cycles + 1, 2, self.ny),
                torch.nan,
                dtype=torch.float64,
                device=G.device,
            )
        for wall, xcenter in enumerate(self.wall_x):
            xidx = [
                x for x in range(self.nx)
                if min((x - xcenter) % self.nx, (xcenter - x) % self.nx) <= self.wall_half_width
            ]
            values = spatial[:, xidx, :].sum(dim=1).to(torch.float64)
            if self._density_device is None:
                self.density[start:stop, cycle, wall] = values.detach().cpu().numpy()
            else:
                self._density_device[start:stop, cycle, wall] = values


class CumulativeBranchLogObserver:
    def __init__(self, *, samples: int, cycles: int) -> None:
        self.samples, self.cycles = int(samples), int(cycles)
        self.per_cycle: Any = None

    def __call__(
        self,
        *,
        cycle: int,
        sample_indices: Any,
        channel_labels: tuple[str, ...],
        conditional_log_probability: Any,
        **_: Any,
    ) -> None:
        if hasattr(conditional_log_probability, "detach"):
            import torch

            device = conditional_log_probability.device
            if self.per_cycle is None:
                self.per_cycle = torch.zeros(
                    (self.samples, self.cycles),
                    dtype=torch.float64,
                    device=device,
                )
            indices = sample_indices.to(dtype=torch.long, device=device)
            self.per_cycle[indices, int(cycle) - 1] += torch.sum(
                conditional_log_probability[:, : len(channel_labels)], dim=1
            )
            return
        if self.per_cycle is None:
            self.per_cycle = np.zeros(
                (self.samples, self.cycles), dtype=np.float64
            )
        values = np.asarray(conditional_log_probability)
        indices = np.asarray(sample_indices, dtype=np.int64)
        self.per_cycle[indices, int(cycle) - 1] += np.sum(
            values[:, : len(channel_labels)], axis=1
        )

    @property
    def cumulative(self) -> np.ndarray:
        if hasattr(self.per_cycle, "detach"):
            self.per_cycle = self.per_cycle.detach().cpu().numpy()
        if self.per_cycle is None:
            raise RuntimeError("branch-log observer was never emitted")
        return np.concatenate(
            (np.zeros((self.samples, 1), dtype=np.float64), np.cumsum(self.per_cycle, axis=1)), axis=1
        )


def _kick_covariances(G: np.ndarray, *, nx: int, ny: int, x: int, y: int, epsilon: float) -> np.ndarray:
    output = np.array(G, copy=True)
    phase = np.ones((output.shape[-1],), dtype=np.complex128)
    for orbital in (0, 1):
        phase[orbital + 2 * int(x) + 2 * int(nx) * (int(y) % int(ny))] = np.exp(-1j * epsilon)
    output *= phase[None, :, None]
    output *= phase.conj()[None, None, :]
    return output


def compute_h2_live_products(
    *,
    model: Any,
    state_bank: TransientStateBank,
    record_writer: Any,
    origins: Iterable[int],
    config: dict[str, Any],
    path: Path | str,
    meas_slab_only: bool,
    smoke: bool = False,
) -> dict[str, Any]:
    origins = [int(origin) for origin in origins]
    nx, ny = int(model.Nx), int(model.Ny)
    samples = int(record_writer.samples)
    horizon = max(1, ny // 2)
    if smoke:
        horizon = min(horizon, 2)
        origins = origins[:1]
    epsilon = float(config["finite_difference_epsilon"])
    source_count = int(config["source_positions"])
    if smoke:
        source_count = min(source_count, 2)
    source_y = np.unique(np.rint(np.arange(source_count) * ny / source_count).astype(np.int64))
    wall_x = _wall_x(nx)
    response_shape = (samples, len(origins), 2, horizon + 1, 2, ny)
    conditional_mean = np.zeros(response_shape, dtype=np.float64)
    score_mean = np.zeros(response_shape, dtype=np.float64)
    total_mean = np.zeros(response_shape, dtype=np.float64)
    conditional_m2 = np.zeros(response_shape, dtype=np.float64)
    total_m2 = np.zeros(response_shape, dtype=np.float64)
    source_n = np.zeros((len(origins), 2), dtype=np.int16)
    covariance_bytes_per_row = (
        int(state_bank.states[origins[0]].shape[-1]) ** 2
        * np.dtype(np.complex128).itemsize
    )
    gpu_covariance_batch_target_bytes = int(
        config.get("gpu_covariance_batch_target_bytes", 2_000_000_000)
    )
    source_positions_per_gpu_batch = max(
        1,
        min(
            len(source_y),
            gpu_covariance_batch_target_bytes
            // max(1, 2 * samples * covariance_bytes_per_row),
        ),
    )

    for origin_index, origin in enumerate(
        tqdm(origins, desc="H2 response origins", unit="origin")
    ):
        G0 = state_bank.get(origin)
        schedule = record_writer.site_ids[:, origin : origin + horizon]
        outcomes = record_writer.outcomes[:, origin : origin + horizon]
        base_density = WallDensityObserver(
            samples=samples,
            nx=nx,
            ny=ny,
            cycles=horizon,
            wall_half_width=int(config["source_wall_half_width"]),
        )
        model.run_markov_circuit(
            cycles=horizon,
            samples=samples,
            batch_size=samples,
            init_mode="default",
            G_init=G0,
            sequence="random",
            meas_slab_only=meas_slab_only,
            perfect_correction=True,
            postselect=False,
            G_history=False,
            save=False,
            return_data=False,
            frozen_schedule=schedule,
            frozen_outcomes=outcomes,
            cycle_observer=base_density,
        )
        base_density.materialize_cpu()
        for source_wall, xsource in enumerate(wall_x):
            chunk_starts = range(0, len(source_y), source_positions_per_gpu_batch)
            for chunk_start in tqdm(
                chunk_starts,
                desc=f"H2 sources t={origin}, wall={source_wall}",
                unit="GPU batch",
                leave=False,
            ):
                source_chunk = source_y[
                    chunk_start : chunk_start + source_positions_per_gpu_batch
                ]
                covariance_blocks = []
                for ysource in source_chunk:
                    covariance_blocks.extend(
                        (
                            _kick_covariances(
                                G0,
                                nx=nx,
                                ny=ny,
                                x=xsource,
                                y=int(ysource),
                                epsilon=epsilon,
                            ),
                            _kick_covariances(
                                G0,
                                nx=nx,
                                ny=ny,
                                x=xsource,
                                y=int(ysource),
                                epsilon=-epsilon,
                            ),
                        )
                    )
                Gpm = np.concatenate(covariance_blocks, axis=0)
                repeat_count = 2 * len(source_chunk)
                schedule_pm = np.concatenate([schedule] * repeat_count, axis=0)
                outcomes_pm = np.concatenate([outcomes] * repeat_count, axis=0)
                continuation_samples = int(Gpm.shape[0])
                density = WallDensityObserver(
                    samples=continuation_samples,
                    nx=nx,
                    ny=ny,
                    cycles=horizon,
                    wall_half_width=int(config["source_wall_half_width"]),
                )
                branch = CumulativeBranchLogObserver(
                    samples=continuation_samples, cycles=horizon
                )
                model.run_markov_circuit(
                    cycles=horizon,
                    samples=continuation_samples,
                    batch_size=continuation_samples,
                    init_mode="default",
                    G_init=Gpm,
                    sequence="random",
                    meas_slab_only=meas_slab_only,
                    perfect_correction=True,
                    postselect=False,
                    G_history=False,
                    save=False,
                    return_data=False,
                    frozen_schedule=schedule_pm,
                    frozen_outcomes=outcomes_pm,
                    cycle_observer=density,
                    record_observer=branch,
                )
                density.materialize_cpu()
                branch_cumulative = branch.cumulative
                for source_offset, ysource in enumerate(source_chunk):
                    plus_start = 2 * source_offset * samples
                    minus_start = plus_start + samples
                    plus_slice = slice(plus_start, plus_start + samples)
                    minus_slice = slice(minus_start, minus_start + samples)
                    d_observable = (
                        density.density[plus_slice]
                        - density.density[minus_slice]
                    ) / (2 * epsilon)
                    d_logp = (
                        branch_cumulative[plus_slice]
                        - branch_cumulative[minus_slice]
                    ) / (2 * epsilon)
                    score_term = base_density.density * d_logp[:, :, None, None]
                    total = d_observable + score_term
                    # Translate source positions before averaging within each trajectory.
                    d_observable = np.roll(
                        d_observable, -int(ysource), axis=-1
                    )
                    score_term = np.roll(score_term, -int(ysource), axis=-1)
                    total = np.roll(total, -int(ysource), axis=-1)
                    n = int(source_n[origin_index, source_wall])
                    for mean, m2, value in (
                        (
                            conditional_mean[:, origin_index, source_wall],
                            conditional_m2[:, origin_index, source_wall],
                            d_observable,
                        ),
                        (
                            total_mean[:, origin_index, source_wall],
                            total_m2[:, origin_index, source_wall],
                            total,
                        ),
                    ):
                        delta = value - mean
                        mean += delta / float(n + 1)
                        m2 += delta * (value - mean)
                    delta = score_term - score_mean[:, origin_index, source_wall]
                    score_mean[:, origin_index, source_wall] += delta / float(n + 1)
                    source_n[origin_index, source_wall] = n + 1

    path = Path(path)
    save_npz_atomic(
        path,
        schema=np.asarray(H2_SCHEMA),
        origins=np.asarray(origins, dtype=np.int64),
        response_cycles=np.arange(horizon + 1, dtype=np.int64),
        source_y=source_y,
        source_wall_x=np.asarray(wall_x, dtype=np.int64),
        finite_difference_epsilon=np.asarray(epsilon),
        source_positions_per_gpu_batch=np.asarray(
            source_positions_per_gpu_batch, dtype=np.int64
        ),
        gpu_covariance_batch_target_bytes=np.asarray(
            gpu_covariance_batch_target_bytes, dtype=np.int64
        ),
        gpu_trajectory_batch_size=np.asarray(samples, dtype=np.int64),
        gpu_batching_contract=np.asarray(
            "source_chunk_times_plus_minus_times_complete_shard_v1"
        ),
        conditional_derivative=conditional_mean,
        likelihood_score_term=score_mean,
        total_response=total_mean,
        conditional_source_sample_variance=conditional_m2 / np.maximum(source_n[None, :, :, None, None, None] - 1, 1),
        total_source_sample_variance=total_m2 / np.maximum(source_n[None, :, :, None, None, None] - 1, 1),
        source_count=source_n,
    )
    return {
        "schema": H2_SCHEMA,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "source_positions_per_gpu_batch": source_positions_per_gpu_batch,
        "gpu_covariance_batch_target_bytes": gpu_covariance_batch_target_bytes,
    }
