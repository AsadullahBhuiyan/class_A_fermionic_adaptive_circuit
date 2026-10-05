"""Compact frame-native observables for the domain-wall correlator campaign."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "domain_wall_frame_correlator_observer_v1"


def x_resolved_square_correlator_from_frame(
    frame: torch.Tensor,
    ranks: torch.Tensor,
    *,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """Return the legacy squared correlator resolved by x and y separation.

    The result has shape ``(batch, nx, ny // 2 + 1)`` and uses

    ``sum_{y,mu,nu} |C[(x,y,mu),(x,y+r,nu)]|^2 / (2*ny)``.

    Averaging the result over ``x`` therefore gives the legacy
    ``xavg_square_correlator_vs_ry`` estimator exactly, without constructing
    the dense correlation matrix.
    """

    nx, ny = int(nx), int(ny)
    if frame.ndim != 3:
        raise ValueError(f"frame must have shape (B,N,R), got {tuple(frame.shape)}")
    batch, dimension, capacity = (int(value) for value in frame.shape)
    if dimension != 2 * nx * ny:
        raise ValueError(
            f"frame dimension {dimension} differs from 2*nx*ny={2 * nx * ny}"
        )
    if tuple(ranks.shape) != (batch,):
        raise ValueError(f"ranks must have shape ({batch},), got {tuple(ranks.shape)}")
    ranks = ranks.to(device=frame.device, dtype=torch.long)
    if bool(torch.any(ranks < 0)) or bool(torch.any(ranks > capacity)):
        raise ValueError("occupied-frame ranks lie outside the saved capacity")

    column = torch.arange(capacity, device=frame.device)
    active = frame * (column[None, None, :] < ranks[:, None, None])
    rows = active.reshape(batch, ny, nx, 2, capacity).permute(0, 2, 1, 3, 4)
    result = torch.empty(
        (batch, nx, ny // 2 + 1), dtype=frame.real.dtype, device=frame.device
    )
    normalization = float(2 * ny)
    for separation in range(ny // 2 + 1):
        right = torch.roll(rows, shifts=-separation, dims=2)
        overlaps = torch.einsum("bxymk,bxynk->bxymn", rows, right.conj())
        result[:, :, separation] = (
            overlaps.abs().square().sum(dim=(2, 3, 4)) / normalization
        )
    return result.to(torch.float64)


class DomainWallCorrelatorObserver:
    """Collect correlators and total charge for one immutable sample batch."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        physical_cycles: int,
        sample_ids: np.ndarray,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.physical_cycles = int(physical_cycles)
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        if self.nx <= 0 or self.ny <= 0 or self.physical_cycles <= 0:
            raise ValueError("nx, ny, and physical_cycles must be positive")
        if self.sample_ids.ndim != 1 or not len(self.sample_ids):
            raise ValueError("sample_ids must be a nonempty one-dimensional array")
        if len(np.unique(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")

        self.cycles = np.arange(self.physical_cycles + 1, dtype=np.int64)
        self.ry_values = np.arange(self.ny // 2 + 1, dtype=np.int64)
        self.x_values = np.arange(self.nx, dtype=np.int64)
        self.seen = np.zeros(self.physical_cycles + 1, dtype=np.bool_)
        self.x_resolved = np.full(
            (
                len(self.sample_ids),
                self.physical_cycles + 1,
                self.nx,
                len(self.ry_values),
            ),
            np.nan,
            dtype=np.float64,
        )
        self.global_charge = np.full(
            (len(self.sample_ids), self.physical_cycles + 1),
            -1,
            dtype=np.int64,
        )
        self.maximum_charge_orthonormality_residual = 0.0

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
        if int(batch_start) != 0 or int(batch_count) != len(self.sample_ids):
            raise ValueError("observer requires one complete durable batch per task")
        if self.seen[cycle]:
            raise RuntimeError(f"duplicate observation at cycle {cycle}")
        if not hasattr(state, "frame") or not hasattr(state, "ranks"):
            raise TypeError("observer requires the native occupied-frame state")

        frame = state.frame.detach()
        ranks = state.ranks.detach()
        expected_dimension = 2 * self.nx * self.ny
        if tuple(frame.shape[:2]) != (len(self.sample_ids), expected_dimension):
            raise ValueError(f"unexpected occupied-frame shape {tuple(frame.shape)}")
        if frame.dtype != torch.complex128:
            raise TypeError(f"occupied frame must be complex128, got {frame.dtype}")

        correlator = x_resolved_square_correlator_from_frame(
            frame, ranks, nx=self.nx, ny=self.ny
        )
        rank_array = ranks.cpu().numpy().astype(np.int64, copy=False)
        charge_from_frame = (
            frame.abs().square().sum(dim=(-2, -1)).real.detach().cpu().numpy()
        )
        residual = float(np.max(np.abs(charge_from_frame - rank_array)))
        self.maximum_charge_orthonormality_residual = max(
            self.maximum_charge_orthonormality_residual, residual
        )
        if residual > 1.0e-8:
            raise FloatingPointError(
                "occupied-frame charge/rank mismatch exceeds tolerance: "
                f"{residual:.6e}"
            )
        values = correlator.detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise FloatingPointError("square correlator contains nonfinite values")
        self.x_resolved[:, cycle] = values
        self.global_charge[:, cycle] = rank_array
        self.seen[cycle] = True

    def checkpoint_payload(self) -> dict[str, np.ndarray]:
        """Return the complete compact observer state for exact continuation."""

        return {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "observer_nx": np.asarray(self.nx, dtype=np.int64),
            "observer_ny": np.asarray(self.ny, dtype=np.int64),
            "observer_physical_cycles": np.asarray(
                self.physical_cycles, dtype=np.int64
            ),
            "observer_sample_ids": self.sample_ids.copy(),
            "observer_seen": self.seen.copy(),
            "observer_x_resolved": self.x_resolved.copy(),
            "observer_global_charge": self.global_charge.copy(),
            "observer_maximum_charge_residual": np.asarray(
                self.maximum_charge_orthonormality_residual, dtype=np.float64
            ),
        }

    def restore_checkpoint(
        self, payload: dict[str, np.ndarray], *, completed_cycle: int
    ) -> None:
        """Validate and restore a checkpointed observer prefix."""

        scalar_expectations = {
            "observer_schema": OBSERVER_SCHEMA,
            "observer_nx": self.nx,
            "observer_ny": self.ny,
            "observer_physical_cycles": self.physical_cycles,
        }
        for key, expected in scalar_expectations.items():
            if key not in payload or np.asarray(payload[key]).item() != expected:
                raise ValueError(f"checkpoint observer identity mismatch: {key}")
        if not np.array_equal(
            np.asarray(payload.get("observer_sample_ids"), dtype=np.int64),
            self.sample_ids,
        ):
            raise ValueError("checkpoint observer sample IDs mismatch")

        seen = np.asarray(payload.get("observer_seen"), dtype=np.bool_)
        x_resolved = np.asarray(payload.get("observer_x_resolved"))
        global_charge = np.asarray(payload.get("observer_global_charge"))
        if seen.shape != self.seen.shape:
            raise ValueError("checkpoint observer seen-mask shape mismatch")
        if x_resolved.shape != self.x_resolved.shape or x_resolved.dtype != np.float64:
            raise ValueError("checkpoint x-resolved correlator shape/dtype mismatch")
        if global_charge.shape != self.global_charge.shape or global_charge.dtype != np.int64:
            raise ValueError("checkpoint global-charge shape/dtype mismatch")
        completed_cycle = int(completed_cycle)
        expected_seen = np.arange(self.physical_cycles + 1) <= completed_cycle
        if not np.array_equal(seen, expected_seen):
            raise ValueError("checkpoint observer cycles are not a contiguous prefix")
        if not np.isfinite(x_resolved[:, : completed_cycle + 1]).all():
            raise FloatingPointError("checkpoint observer prefix contains nonfinite values")
        if np.any(global_charge[:, : completed_cycle + 1] < 0):
            raise ValueError("checkpoint observer prefix contains invalid charge")
        if completed_cycle < self.physical_cycles:
            if not np.isnan(x_resolved[:, completed_cycle + 1 :]).all():
                raise ValueError("checkpoint correlator suffix must be unobserved")
            if not np.all(global_charge[:, completed_cycle + 1 :] == -1):
                raise ValueError("checkpoint charge suffix must be unobserved")

        self.seen[:] = seen
        self.x_resolved[:] = x_resolved
        self.global_charge[:] = global_charge
        residual = float(
            np.asarray(payload.get("observer_maximum_charge_residual", 0.0)).item()
        )
        if not np.isfinite(residual) or residual < 0 or residual > 1.0e-8:
            raise ValueError("checkpoint charge residual is invalid")
        self.maximum_charge_orthonormality_residual = residual

    def validate(self) -> None:
        if not bool(np.all(self.seen)):
            raise RuntimeError(f"missing cycle observations: {self.cycles[~self.seen].tolist()}")
        if not np.isfinite(self.x_resolved).all():
            raise FloatingPointError("square-correlator history contains nonfinite values")
        if np.any(self.global_charge < 0):
            raise ValueError("global-charge history contains invalid values")

    def result_payload(self) -> dict[str, np.ndarray]:
        self.validate()
        xavg = self.x_resolved.mean(axis=2)
        return {
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "cycles": self.cycles.copy(),
            "normalized_cycles": self.cycles.astype(np.float64) / float(self.ny),
            "ry_values": self.ry_values.copy(),
            "x_values": self.x_values.copy(),
            "global_sample_indices": self.sample_ids.copy(),
            "x_resolved_square_correlator": self.x_resolved.copy(),
            "xavg_square_correlator_vs_ry": xavg,
            "global_charge": self.global_charge.copy(),
            "maximum_charge_orthonormality_residual": np.asarray(
                self.maximum_charge_orthonormality_residual, dtype=np.float64
            ),
        }
