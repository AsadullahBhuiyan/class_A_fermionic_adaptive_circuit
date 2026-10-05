"""Compact every-cycle observables for the uniform bulk validation campaign."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "uniform_bulk_chern_charge_observer_v1"


def build_chern_partition_indices(
    *,
    nx: int,
    ny: int,
    xref: int | None = None,
    yref: int | None = None,
    radius: float | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build the legacy central three-sector real-space Chern partition."""

    nx = int(nx)
    ny = int(ny)
    xref = nx // 2 if xref is None else int(xref)
    yref = ny // 2 if yref is None else int(yref)
    radius = 0.4 * min(nx, ny) if radius is None else float(radius)
    if radius <= 0:
        raise ValueError(f"radius must be positive; got {radius}")

    a_mask = np.zeros((nx, ny), dtype=bool)
    b_mask = np.zeros_like(a_mask)
    c_mask = np.zeros_like(a_mask)
    radius_squared = radius * radius
    angle_2pi_over_3 = 2.0 * np.pi / 3.0
    angle_4pi_over_3 = 4.0 * np.pi / 3.0
    for dy in range(-int(math.floor(radius)), int(math.floor(radius)) + 1):
        y = yref + dy
        if y < 0 or y >= ny:
            continue
        max_dx = int(math.floor(math.sqrt(radius_squared - dy * dy)))
        x0 = max(0, xref - max_dx)
        x1 = min(nx - 1, xref + max_dx)
        if x0 > x1:
            continue
        dxs = np.arange(x0, x1 + 1) - xref
        dys = np.full_like(dxs, dy)
        theta = np.mod(np.arctan2(dys, dxs), 2.0 * np.pi)
        a_mask[x0 : x1 + 1, y] = (theta >= 0.0) & (
            theta < angle_2pi_over_3
        )
        b_mask[x0 : x1 + 1, y] = (theta >= angle_2pi_over_3) & (
            theta < angle_4pi_over_3
        )
        c_mask[x0 : x1 + 1, y] = (theta >= angle_4pi_over_3) & (
            theta < 2.0 * np.pi
        )

    def indices(mask: np.ndarray) -> torch.Tensor:
        xs, ys = np.nonzero(mask)
        orbital_zero = 2 * xs + 2 * nx * ys
        orbital_one = 1 + 2 * xs + 2 * nx * ys
        values = np.sort(np.concatenate((orbital_zero, orbital_one))).astype(
            np.int64, copy=False
        )
        return torch.as_tensor(values, dtype=torch.long, device=device)

    return {
        "nx": nx,
        "ny": ny,
        "xref": xref,
        "yref": yref,
        "radius": radius,
        "A": indices(a_mask),
        "B": indices(b_mask),
        "C": indices(c_mask),
    }


def frame_real_space_chern(
    frame: torch.Tensor, partitions: dict[str, Any]
) -> torch.Tensor:
    """Evaluate the legacy tripartition estimator from ``P*=V*V^T``."""

    if frame.ndim != 3:
        raise ValueError(f"frame must have shape (B,N,R), got {tuple(frame.shape)}")
    projector_frame = frame.conj()
    i_a = partitions["A"].to(frame.device)
    i_b = partitions["B"].to(frame.device)
    i_c = partitions["C"].to(frame.device)
    w_a = projector_frame.index_select(1, i_a)
    w_b = projector_frame.index_select(1, i_b)
    w_c = projector_frame.index_select(1, i_c)
    p_ca = w_c @ w_a.mH
    p_ab = w_a @ w_b.mH
    p_bc = w_b @ w_c.mH
    p_ac = w_a @ w_c.mH
    p_cb = w_c @ w_b.mH
    p_ba = w_b @ w_a.mH
    term_abc = torch.diagonal(p_ca @ p_ab @ p_bc, dim1=-2, dim2=-1).sum(
        dim=-1
    )
    term_acb = torch.diagonal(p_ac @ p_cb @ p_ba, dim1=-2, dim2=-1).sum(
        dim=-1
    )
    return (12.0 * math.pi * 1j * (term_abc - term_acb)).real.to(torch.float64)


class CompactChernChargeObserver:
    """Collect one batch's Chern estimator and global charge at every cycle."""

    def __init__(self, *, size: int, physical_cycles: int, samples: int) -> None:
        self.size = int(size)
        self.physical_cycles = int(physical_cycles)
        self.samples = int(samples)
        if self.size <= 0 or self.physical_cycles <= 0 or self.samples <= 0:
            raise ValueError("size, physical_cycles, and samples must be positive")
        self.cycles = np.arange(self.physical_cycles + 1, dtype=np.int64)
        self.seen = np.zeros(self.physical_cycles + 1, dtype=np.bool_)
        self.real_space_chern = np.full(
            (self.samples, self.physical_cycles + 1), np.nan, dtype=np.float64
        )
        self.global_charge = np.full(
            (self.samples, self.physical_cycles + 1), np.nan, dtype=np.float64
        )
        self.particle_number = np.full(
            (self.samples, self.physical_cycles + 1), -1, dtype=np.int64
        )
        self._partitions: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        cycle: int,
        state: Any,
        batch_index: int,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        del batch_index
        cycle = int(cycle)
        if not 0 <= cycle <= self.physical_cycles:
            raise IndexError(f"cycle {cycle} is outside 0..{self.physical_cycles}")
        if int(batch_start) != 0 or int(batch_count) != self.samples:
            raise ValueError(
                "the compact observer requires one complete durable batch per task"
            )
        if self.seen[cycle]:
            raise RuntimeError(f"duplicate observation at cycle {cycle}")
        if not hasattr(state, "frame") or not hasattr(state, "ranks"):
            raise TypeError("the compact observer requires native occupied-frame state")

        frame = state.frame.detach()
        if tuple(frame.shape[:2]) != (self.samples, 2 * self.size * self.size):
            raise ValueError(f"unexpected occupied-frame shape {tuple(frame.shape)}")
        if self._partitions is None:
            self._partitions = build_chern_partition_indices(
                nx=self.size,
                ny=self.size,
                device=frame.device,
            )

        rank = state.ranks.detach().cpu().numpy().astype(np.int64, copy=False)
        charge = (
            frame.abs().square().sum(dim=(-2, -1)).real.detach().cpu().numpy()
        )
        chern = frame_real_space_chern(frame, self._partitions).detach().cpu().numpy()
        self.real_space_chern[:, cycle] = chern
        self.global_charge[:, cycle] = charge
        self.particle_number[:, cycle] = rank
        self.seen[cycle] = True

    def validate(self) -> None:
        if not bool(np.all(self.seen)):
            missing = self.cycles[~self.seen].tolist()
            raise RuntimeError(f"missing cycle observations: {missing}")
        if not np.isfinite(self.real_space_chern).all():
            raise FloatingPointError("real-space Chern history contains nonfinite values")
        if not np.isfinite(self.global_charge).all():
            raise FloatingPointError("global-charge history contains nonfinite values")
        residual = np.max(np.abs(self.global_charge - self.particle_number))
        if residual > 1.0e-8:
            raise FloatingPointError(
                f"occupied-frame charge/rank mismatch exceeds tolerance: {residual}"
            )

    def payload(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "cycles": self.cycles.copy(),
            "normalized_cycles": self.cycles.astype(np.float64) / float(self.size),
            "real_space_chern": self.real_space_chern.copy(),
            "global_charge": self.global_charge.copy(),
            "particle_number": self.particle_number.copy(),
            "half_filling_offset": self.particle_number - self.size * self.size,
            "partition_xref": np.asarray(self.size // 2, dtype=np.int64),
            "partition_yref": np.asarray(self.size // 2, dtype=np.int64),
            "partition_radius": np.asarray(0.4 * self.size, dtype=np.float64),
        }
