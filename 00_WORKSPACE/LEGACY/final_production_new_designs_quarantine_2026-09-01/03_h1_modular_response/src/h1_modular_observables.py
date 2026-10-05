"""Frame-native retarded modular response and packet-drift observables for H1."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from h1_io import save_npz_atomic, sha256_file


SCHEMA = "h1_retarded_modular_response_and_packet_v1"


def checkpoint_cycles(ny: int) -> list[int]:
    """Six inclusive equally spaced checkpoints from Ny through 2*Ny."""

    ny = int(ny)
    if ny <= 0:
        raise ValueError("Ny must be positive")
    return np.rint(np.linspace(ny, 2 * ny, 6)).astype(np.int64).tolist()


def wall_x_positions(nx: int) -> tuple[int, int]:
    half = int(nx) // 2
    width = max(1, int(nx) // 4)
    return max(0, half - width), min(int(nx), half + width + 1) - 1


def retained_upper_half_indices(
    *, nx: int, ny: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Modes in A=[0,Nx) x [Ny//2,Ny), ordered by relative y,x,orbital."""

    nx, ny = int(nx), int(ny)
    if nx <= 0 or ny <= 0 or ny % 2:
        raise ValueError("H1 requires positive Nx and even Ny")
    values = [
        mu + 2 * x + 2 * nx * y
        for y in range(ny // 2, ny)
        for x in range(nx)
        for mu in (0, 1)
    ]
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _source_seed(root_seed: int, sample_id: int, checkpoint: int, wall: int) -> int:
    raw = (
        f"H1-source-v1:{int(root_seed)}:{int(sample_id)}:"
        f"{int(checkpoint)}:{int(wall)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def sample_wall_sources(
    *, root_seed: int, nx: int, ny: int, sample_id: int,
    checkpoint: int, source_count: int = 10,
) -> np.ndarray:
    """Return (wall,source,xy), independent of the physical case parameters."""

    nx, ny, source_count = int(nx), int(ny), int(source_count)
    candidates = np.arange(ny // 2, ny, dtype=np.int64)
    if source_count <= 0 or source_count > len(candidates):
        raise ValueError("source_count must lie in 1..Ny//2")
    result = np.empty((2, source_count, 2), dtype=np.int64)
    for wall, x in enumerate(wall_x_positions(nx)):
        rng = np.random.default_rng(
            _source_seed(root_seed, sample_id, checkpoint, wall)
        )
        result[wall, :, 0] = x
        result[wall, :, 1] = rng.choice(
            candidates, size=source_count, replace=False
        )
    return result


def reduced_correlation_from_frame(
    frame: torch.Tensor, *, rank: int, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return C_A, its occupations/eigenvectors, and Hermiticity error."""

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


def _source_local_indices(nx: int, y_relative: int, device: Any) -> torch.Tensor:
    base = 2 * int(nx) * int(y_relative)
    # x is added by the caller because it differs between walls.
    return torch.as_tensor([base, base + 1], dtype=torch.long, device=device)


def _linear_slope(
    times: np.ndarray, values: np.ndarray, fit_window: tuple[float, float],
    valid: np.ndarray | None = None,
) -> tuple[float, float, int]:
    mask = (
        (times >= float(fit_window[0]) - 1e-12)
        & (times <= float(fit_window[1]) + 1e-12)
        & np.isfinite(values)
    )
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    count = int(np.count_nonzero(mask))
    if count < 5:
        return float("nan"), float("nan"), count
    x, y = times[mask], values[mask]
    design = np.column_stack((np.ones_like(x), x))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = (
        float("nan") if denominator <= 0
        else 1.0 - float(np.sum(residual**2)) / denominator
    )
    return float(beta[1]), r2, count


def _aligned_profile(profile: np.ndarray, source_y: int) -> np.ndarray:
    """Place relative-y data on d=-(Ay-1),...,Ay-1 without wraparound."""

    profile = np.asarray(profile)
    ay = profile.shape[-1]
    result = np.zeros(profile.shape[:-1] + (2 * ay - 1,), dtype=profile.dtype)
    for y in range(ay):
        result[..., y - int(source_y) + ay - 1] = profile[..., y]
    return result


def _aligned_field(field: np.ndarray, source_y: int) -> np.ndarray:
    field = np.asarray(field)
    ay, nx = field.shape[-2:]
    result = np.zeros(
        field.shape[:-2] + (2 * ay - 1, nx), dtype=field.dtype
    )
    for y in range(ay):
        result[..., y - int(source_y) + ay - 1, :] = field[..., y, :]
    return result


def retarded_density_from_eigensystem(
    *, occupations: torch.Tensor, vectors: torch.Tensor,
    source_indices: torch.Tensor, modular_times: torch.Tensor, epsilon: float,
) -> torch.Tensor:
    """Signed density response to -i[P_source,C_A] at all modular times."""

    physical_nu = occupations.clamp(0.0, 1.0)
    modular_nu = occupations.clamp(float(epsilon), 1.0 - float(epsilon))
    energies = torch.log((1.0 - modular_nu) / modular_nu)
    coeff = vectors.mH.index_select(1, source_indices.to(vectors.device))
    phase = torch.exp(-1j * modular_times[:, None] * energies[None, :])
    evolved = torch.einsum("ij,tjk->tik", vectors, phase[:, :, None] * coeff[None])
    covariance_evolved = torch.einsum(
        "ij,tjk->tik",
        vectors,
        (physical_nu[None, :, None] * phase[:, :, None]) * coeff[None],
    )
    return 2.0 * torch.imag(
        torch.sum(evolved * covariance_evolved.conj(), dim=-1)
    )


def packet_probability_from_eigensystem(
    *, occupations: torch.Tensor, vectors: torch.Tensor,
    source_indices: torch.Tensor, modular_times: torch.Tensor, epsilon: float,
) -> torch.Tensor:
    """Probability of the existing equal-orbital localized H1 packet."""

    nu = occupations.clamp(float(epsilon), 1.0 - float(epsilon))
    energies = torch.log((1.0 - nu) / nu)
    packet = torch.zeros(
        vectors.shape[0], dtype=vectors.dtype, device=vectors.device
    )
    packet[source_indices.to(vectors.device)] = 1.0 / np.sqrt(2.0)
    coeff = vectors.mH @ packet
    phase = torch.exp(-1j * modular_times[:, None] * energies[None, :])
    evolved = torch.einsum("ij,tj->ti", vectors, phase * coeff[None])
    return evolved.abs().square().real


def dense_retarded_reference(
    correlation: torch.Tensor, source_indices: Iterable[int],
    times: Iterable[float], epsilon: float,
) -> torch.Tensor:
    """Dense reference entry point used by tests and finite-kick validation."""

    correlation = 0.5 * (correlation + correlation.mH)
    occupations, vectors = torch.linalg.eigh(correlation)
    return retarded_density_from_eigensystem(
        occupations=occupations.real,
        vectors=vectors,
        source_indices=torch.as_tensor(
            list(source_indices), dtype=torch.long, device=correlation.device
        ),
        modular_times=torch.as_tensor(
            list(times), dtype=correlation.real.dtype, device=correlation.device
        ),
        epsilon=float(epsilon),
    )


class H1ModularObserver:
    """Checkpoint observer retaining only response and packet scientific products."""

    def __init__(
        self, *, nx: int, ny: int, checkpoints: Iterable[int],
        global_sample_ids: Iterable[int], root_seed: int, source_count: int,
        response_times: Iterable[float], packet_times: Iterable[float],
        spectral_clip_eps: Iterable[float], wall_half_widths: Iterable[int],
        primary_epsilon: float, primary_wall_half_width: int,
        fit_window: tuple[float, float], minimum_packet_retention: float = 0.7,
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.ay = self.ny // 2
        self.checkpoints = np.asarray(list(checkpoints), dtype=np.int64)
        self.sample_ids = np.asarray(list(global_sample_ids), dtype=np.int64)
        self.root_seed = int(root_seed)
        self.source_count = int(source_count)
        self.response_times = np.asarray(list(response_times), dtype=np.float64)
        self.packet_times = np.asarray(list(packet_times), dtype=np.float64)
        self.eps = np.asarray(list(spectral_clip_eps), dtype=np.float64)
        self.widths = np.asarray(list(wall_half_widths), dtype=np.int64)
        self.primary_eps = int(np.argmin(np.abs(self.eps - float(primary_epsilon))))
        self.primary_width = int(
            np.argmin(np.abs(self.widths - int(primary_wall_half_width)))
        )
        self.fit_window = (float(fit_window[0]), float(fit_window[1]))
        self.minimum_packet_retention = float(minimum_packet_retention)
        if self.checkpoints.tolist() != checkpoint_cycles(self.ny):
            raise ValueError("checkpoint schedule differs from inclusive six-point H1 contract")
        if self.source_count > self.ay:
            raise ValueError("too many distinct retained-half wall sources")
        if self.response_times.tolist() != np.arange(0.0, 4.0 + 0.025, 0.05).tolist():
            raise ValueError("retarded modular-time grid changed")
        if self.packet_times.tolist() != np.arange(0.0, 8.0 + 0.025, 0.05).tolist():
            raise ValueError("packet modular-time grid changed")

        ns, nc, ne, nw = (
            len(self.sample_ids), len(self.checkpoints), len(self.eps), len(self.widths)
        )
        nr, npacket = len(self.response_times), len(self.packet_times)
        nsrc, nd = self.source_count, 2 * self.ay - 1
        self.seen = np.zeros((ns, nc), dtype=np.bool_)
        self.seconds = np.zeros((ns, nc), dtype=np.float64)
        self.hermiticity_error = np.full((ns, nc), np.nan, dtype=np.float64)
        self.sources = np.full((ns, nc, 2, nsrc, 2), -1, dtype=np.int64)
        self.occupations = np.full((ns, nc, 2 * self.nx * self.ay), np.nan)
        self.clip_counts = np.zeros((ns, nc, ne, 2), dtype=np.int32)

        self.retarded_density = np.full(
            (ns, nc, 2, nsrc, nr, self.ay, self.nx), np.nan
        )
        self.retarded_aligned_profile = np.full(
            (ns, nc, ne, nw, 2, nsrc, nr, nd), np.nan
        )
        self.retarded_dipole = np.full(
            (ns, nc, ne, nw, 2, nsrc, nr), np.nan
        )
        self.retarded_transverse_leakage = np.full_like(
            self.retarded_dipole, np.nan
        )
        self.retarded_total_charge = np.full(
            (ns, nc, ne, 2, nsrc, nr), np.nan
        )
        self.retarded_source_slope = np.full(
            (ns, nc, ne, nw, 2, nsrc), np.nan
        )
        self.retarded_source_slope_r2 = np.full_like(
            self.retarded_source_slope, np.nan
        )
        self.retarded_source_mean_aligned_density = np.full(
            (ns, nc, 2, nr, nd, self.nx), np.nan
        )
        self.retarded_source_mean_aligned_profile = np.full(
            (ns, nc, ne, nw, 2, nr, nd), np.nan
        )
        self.wall_velocity = np.full((ns, nc, ne, nw, 2), np.nan)
        self.wall_velocity_r2 = np.full_like(self.wall_velocity, np.nan)
        self.handed_response = np.full((ns, nc, ne, nw), np.nan)

        self.packet_aligned_profile = np.full(
            (ns, nc, 2, nsrc, npacket, nd), np.nan
        )
        self.packet_displacement = np.full(
            (ns, nc, ne, 2, nsrc, npacket), np.nan
        )
        self.packet_retention = np.full(
            (ns, nc, ne, nw, 2, nsrc, npacket), np.nan
        )
        self.packet_source_velocity = np.full(
            (ns, nc, ne, nw, 2, nsrc), np.nan
        )
        self.packet_source_fit_points = np.zeros(
            (ns, nc, ne, nw, 2, nsrc), dtype=np.int16
        )
        self.packet_source_mean_profile = np.full(
            (ns, nc, 2, npacket, nd), np.nan
        )
        self.packet_wall_velocity = np.full((ns, nc, ne, nw, 2), np.nan)
        self.packet_wall_fit_points = np.zeros(
            (ns, nc, ne, nw, 2), dtype=np.int16
        )
        self.packet_handed_velocity = np.full((ns, nc, ne, nw), np.nan)

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
            raise RuntimeError("invalid or duplicate H1 observer batch")
        if not hasattr(state, "frame") or not hasattr(state, "ranks"):
            raise TypeError("H1 modular observer requires occupied-frame state")
        for local in range(start, stop):
            began = time.perf_counter()
            batch_local = local - start
            frame = state.frame[batch_local].detach()
            rank = int(state.ranks[batch_local].item())
            indices = retained_upper_half_indices(
                nx=self.nx, ny=self.ny, device=frame.device
            )
            correlation, occupations, vectors, hermiticity = (
                reduced_correlation_from_frame(frame, rank=rank, indices=indices)
            )
            self.hermiticity_error[local, checkpoint_index] = hermiticity
            self.occupations[local, checkpoint_index] = (
                occupations.detach().cpu().numpy()
            )
            sources = sample_wall_sources(
                root_seed=self.root_seed, nx=self.nx, ny=self.ny,
                sample_id=int(self.sample_ids[local]), checkpoint=int(cycle),
                source_count=self.source_count,
            )
            self.sources[local, checkpoint_index] = sources
            response_time_device = torch.as_tensor(
                self.response_times, dtype=frame.real.dtype, device=frame.device
            )
            packet_time_device = torch.as_tensor(
                self.packet_times, dtype=frame.real.dtype, device=frame.device
            )
            for eps_index, eps in enumerate(self.eps):
                self.clip_counts[local, checkpoint_index, eps_index, 0] = int(
                    torch.count_nonzero(occupations < eps).item()
                )
                self.clip_counts[local, checkpoint_index, eps_index, 1] = int(
                    torch.count_nonzero(occupations > 1.0 - eps).item()
                )
                for wall, wall_x in enumerate(wall_x_positions(self.nx)):
                    for source_index, (_x, source_y_absolute) in enumerate(sources[wall]):
                        source_y = int(source_y_absolute) - self.ny // 2
                        base = 2 * int(wall_x) + 2 * self.nx * source_y
                        source_modes = _source_local_indices(
                            self.nx, source_y, frame.device
                        ) + 2 * int(wall_x)
                        if source_modes.tolist() != [base, base + 1]:
                            raise AssertionError("source-index construction changed")
                        response_orbital = retarded_density_from_eigensystem(
                            occupations=occupations, vectors=vectors,
                            source_indices=source_modes,
                            modular_times=response_time_device, epsilon=float(eps),
                        )
                        response = (
                            response_orbital.reshape(
                                len(self.response_times), self.ay, self.nx, 2
                            ).sum(dim=-1).detach().cpu().numpy()
                        )
                        if eps_index == self.primary_eps:
                            self.retarded_density[
                                local, checkpoint_index, wall, source_index
                            ] = response
                            self.retarded_source_mean_aligned_density[
                                local, checkpoint_index, wall
                            ] = np.nan_to_num(
                                self.retarded_source_mean_aligned_density[
                                    local, checkpoint_index, wall
                                ], nan=0.0
                            ) + _aligned_field(response, source_y) / self.source_count
                        self.retarded_total_charge[
                            local, checkpoint_index, eps_index, wall, source_index
                        ] = response.sum(axis=(1, 2))
                        for width_index, width in enumerate(self.widths):
                            xmask = np.asarray([
                                min((x - wall_x) % self.nx, (wall_x - x) % self.nx)
                                <= int(width)
                                for x in range(self.nx)
                            ])
                            profile = response[..., xmask].sum(axis=-1)
                            aligned = _aligned_profile(profile, source_y)
                            displacement_axis = np.arange(
                                -(self.ay - 1), self.ay, dtype=np.float64
                            )
                            dipole = aligned @ displacement_axis
                            denominator = np.maximum(
                                np.sum(np.abs(response), axis=(1, 2)), 1e-300
                            )
                            leakage = (
                                np.sum(np.abs(response[..., ~xmask]), axis=(1, 2))
                                / denominator
                            )
                            slot = (
                                local, checkpoint_index, eps_index, width_index,
                                wall, source_index,
                            )
                            self.retarded_aligned_profile[slot] = aligned
                            self.retarded_dipole[slot] = dipole
                            self.retarded_transverse_leakage[slot] = leakage
                            velocity, r2, _count = _linear_slope(
                                self.response_times, dipole, self.fit_window
                            )
                            self.retarded_source_slope[slot] = velocity
                            self.retarded_source_slope_r2[slot] = r2

                        probability_orbital = packet_probability_from_eigensystem(
                            occupations=occupations, vectors=vectors,
                            source_indices=source_modes,
                            modular_times=packet_time_device, epsilon=float(eps),
                        )
                        probability = (
                            probability_orbital.reshape(
                                len(self.packet_times), self.ay, self.nx, 2
                            ).sum(dim=-1).detach().cpu().numpy()
                        )
                        longitudinal = probability.sum(axis=-1)
                        aligned_packet = _aligned_profile(longitudinal, source_y)
                        displacement_axis = np.arange(
                            -(self.ay - 1), self.ay, dtype=np.float64
                        )
                        displacement = aligned_packet @ displacement_axis
                        packet_slot = (
                            local, checkpoint_index, eps_index, wall, source_index
                        )
                        self.packet_displacement[packet_slot] = displacement
                        if eps_index == self.primary_eps:
                            self.packet_aligned_profile[
                                local, checkpoint_index, wall, source_index
                            ] = aligned_packet
                            self.packet_source_mean_profile[
                                local, checkpoint_index, wall
                            ] = np.nan_to_num(
                                self.packet_source_mean_profile[
                                    local, checkpoint_index, wall
                                ], nan=0.0
                            ) + aligned_packet / self.source_count
                        for width_index, width in enumerate(self.widths):
                            xmask = np.asarray([
                                min((x - wall_x) % self.nx, (wall_x - x) % self.nx)
                                <= int(width)
                                for x in range(self.nx)
                            ])
                            retention = probability[..., xmask].sum(axis=(1, 2))
                            pslot = (
                                local, checkpoint_index, eps_index, width_index,
                                wall, source_index,
                            )
                            self.packet_retention[pslot] = retention
                            valid = (
                                (retention >= self.minimum_packet_retention)
                                & (np.abs(displacement) <= self.ay / 3.0)
                            )
                            velocity, _r2, count = _linear_slope(
                                self.packet_times, displacement,
                                (0.0, 8.0), valid=valid,
                            )
                            self.packet_source_velocity[pslot] = velocity
                            self.packet_source_fit_points[pslot] = count

            for eps_index in range(len(self.eps)):
                for width_index in range(len(self.widths)):
                    for wall in range(2):
                        mean_profile = self.retarded_aligned_profile[
                            local, checkpoint_index, eps_index, width_index, wall
                        ].mean(axis=0)
                        self.retarded_source_mean_aligned_profile[
                            local, checkpoint_index, eps_index, width_index, wall
                        ] = mean_profile
                        displacement_axis = np.arange(
                            -(self.ay - 1), self.ay, dtype=np.float64
                        )
                        mean_dipole = mean_profile @ displacement_axis
                        velocity, r2, _count = _linear_slope(
                            self.response_times, mean_dipole, self.fit_window
                        )
                        self.wall_velocity[
                            local, checkpoint_index, eps_index, width_index, wall
                        ] = velocity
                        self.wall_velocity_r2[
                            local, checkpoint_index, eps_index, width_index, wall
                        ] = r2
                        mean_packet_displacement = self.packet_displacement[
                            local, checkpoint_index, eps_index, wall
                        ].mean(axis=0)
                        mean_packet_retention = self.packet_retention[
                            local, checkpoint_index, eps_index, width_index, wall
                        ].mean(axis=0)
                        valid = (
                            (mean_packet_retention >= self.minimum_packet_retention)
                            & (np.abs(mean_packet_displacement) <= self.ay / 3.0)
                        )
                        packet_velocity, _r2, count = _linear_slope(
                            self.packet_times, mean_packet_displacement,
                            (0.0, 8.0), valid=valid,
                        )
                        self.packet_wall_velocity[
                            local, checkpoint_index, eps_index, width_index, wall
                        ] = packet_velocity
                        self.packet_wall_fit_points[
                            local, checkpoint_index, eps_index, width_index, wall
                        ] = count
                    self.handed_response[
                        local, checkpoint_index, eps_index, width_index
                    ] = 0.5 * (
                        self.wall_velocity[
                            local, checkpoint_index, eps_index, width_index, 0
                        ]
                        - self.wall_velocity[
                            local, checkpoint_index, eps_index, width_index, 1
                        ]
                    )
                    self.packet_handed_velocity[
                        local, checkpoint_index, eps_index, width_index
                    ] = 0.5 * (
                        self.packet_wall_velocity[
                            local, checkpoint_index, eps_index, width_index, 0
                        ]
                        - self.packet_wall_velocity[
                            local, checkpoint_index, eps_index, width_index, 1
                        ]
                    )
            self.seconds[local, checkpoint_index] = time.perf_counter() - began
            self.seen[local, checkpoint_index] = True
            del correlation, occupations, vectors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def validate(self) -> dict[str, Any]:
        if not self.seen.all():
            raise RuntimeError("one or more trajectory-checkpoint H1 products are missing")
        required_finite = {
            "occupations": self.occupations,
            "retarded_density": self.retarded_density,
            "retarded_aligned_profile": self.retarded_aligned_profile,
            "retarded_dipole": self.retarded_dipole,
            "retarded_transverse_leakage": self.retarded_transverse_leakage,
            "retarded_total_charge": self.retarded_total_charge,
            "wall_velocity": self.wall_velocity,
            "handed_response": self.handed_response,
            "packet_aligned_profile": self.packet_aligned_profile,
            "packet_displacement": self.packet_displacement,
            "packet_retention": self.packet_retention,
        }
        bad = [name for name, value in required_finite.items() if not np.isfinite(value).all()]
        if bad:
            raise FloatingPointError(f"non-finite H1 products: {bad}")
        charge_residual = float(np.max(np.abs(self.retarded_total_charge)))
        if charge_residual > 1e-8:
            raise FloatingPointError(
                f"retarded response violates charge conservation: {charge_residual:.3e}"
            )
        if float(np.max(self.hermiticity_error)) > 1e-10:
            raise FloatingPointError("restricted correlation is not Hermitian")
        return {
            "schema": SCHEMA,
            "samples": len(self.sample_ids),
            "checkpoints": self.checkpoints.tolist(),
            "sources_per_wall_checkpoint": self.source_count,
            "retarded_total_charge_max": charge_residual,
            "covariance_hermiticity_error_max": float(np.max(self.hermiticity_error)),
            "observer_seconds": float(self.seconds.sum()),
        }

    def save(self, directory: Path | str, *, config: dict[str, Any]) -> dict[str, Any]:
        diagnostics = self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        common = directory / "common.npz"
        retarded_fields = directory / "retarded_fields.npz"
        retarded_summary = directory / "retarded_summary.npz"
        packet = directory / "packet_drift.npz"
        save_npz_atomic(
            common,
            schema=np.asarray(SCHEMA),
            config_json=np.asarray(json.dumps(config, sort_keys=True)),
            checkpoints=self.checkpoints,
            global_sample_ids=self.sample_ids,
            retained_indices=retained_upper_half_indices(
                nx=self.nx, ny=self.ny
            ).cpu().numpy(),
            retained_y=np.arange(self.ny // 2, self.ny, dtype=np.int64),
            wall_x=np.asarray(wall_x_positions(self.nx), dtype=np.int64),
            source_xy=self.sources,
            modular_times=self.response_times,
            packet_modular_times=self.packet_times,
            spectral_clip_eps=self.eps,
            wall_half_widths=self.widths,
            primary_epsilon_index=np.asarray(self.primary_eps, dtype=np.int64),
            primary_wall_width_index=np.asarray(self.primary_width, dtype=np.int64),
            restricted_occupation_spectrum=self.occupations,
            spectral_clip_counts=self.clip_counts,
            covariance_hermiticity_error=self.hermiticity_error,
            observer_seconds=self.seconds,
        )
        save_npz_atomic(
            retarded_fields,
            schema=np.asarray(SCHEMA),
            retarded_density_xy=self.retarded_density,
        )
        save_npz_atomic(
            retarded_summary,
            schema=np.asarray(SCHEMA),
            retarded_aligned_wall_profile=self.retarded_aligned_profile,
            retarded_wall_dipole=self.retarded_dipole,
            retarded_transverse_leakage=self.retarded_transverse_leakage,
            retarded_total_charge=self.retarded_total_charge,
            retarded_source_slope=self.retarded_source_slope,
            retarded_source_slope_r2=self.retarded_source_slope_r2,
            retarded_source_mean_aligned_density=self.retarded_source_mean_aligned_density,
            retarded_source_mean_aligned_profile=self.retarded_source_mean_aligned_profile,
            retarded_wall_velocity=self.wall_velocity,
            retarded_wall_velocity_r2=self.wall_velocity_r2,
            handed_response=self.handed_response,
        )
        save_npz_atomic(
            packet,
            schema=np.asarray(SCHEMA),
            packet_aligned_longitudinal_profile=self.packet_aligned_profile,
            packet_displacement=self.packet_displacement,
            packet_wall_retention=self.packet_retention,
            packet_source_velocity=self.packet_source_velocity,
            packet_source_fit_points=self.packet_source_fit_points,
            packet_source_mean_profile=self.packet_source_mean_profile,
            packet_wall_velocity=self.packet_wall_velocity,
            packet_wall_fit_points=self.packet_wall_fit_points,
            packet_handed_velocity=self.packet_handed_velocity,
        )
        files = []
        for path in (common, retarded_fields, retarded_summary, packet):
            files.append({
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
        return {
            **diagnostics,
            "path": str(directory),
            "files": files,
            "bytes": int(sum(row["bytes"] for row in files)),
            "static_susceptibility_archived": False,
        }
