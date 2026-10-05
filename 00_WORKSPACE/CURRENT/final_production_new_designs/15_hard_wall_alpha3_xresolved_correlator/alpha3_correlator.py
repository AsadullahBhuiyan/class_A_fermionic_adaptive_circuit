"""Compact per-sample endpoint correlators; occupied frames are checkpoint-only."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "hard_wall_alpha3_xresolved_endpoint_observer_v1"


def x_resolved_square_correlator_from_frame(
    frame: torch.Tensor,
    ranks: torch.Tensor,
    *,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """Evaluate the legacy squared correlator without forming a covariance.

    The returned tensor has shape ``(samples, nx, ny // 2 + 1)`` and entries

    ``sum_{y,mu,nu} |C[(x,y,mu),(x,y+r,nu)]|^2 / (2*ny)``.

    Averaging over ``x`` therefore reproduces the legacy x-averaged estimator.
    """

    nx, ny = int(nx), int(ny)
    if frame.ndim != 3:
        raise ValueError(f"frame must have shape (B,N,R), got {tuple(frame.shape)}")
    samples, dimension, capacity = (int(value) for value in frame.shape)
    if dimension != 2 * nx * ny:
        raise ValueError(
            f"frame dimension {dimension} differs from 2*nx*ny={2 * nx * ny}"
        )
    if tuple(ranks.shape) != (samples,):
        raise ValueError(
            f"ranks must have shape ({samples},), got {tuple(ranks.shape)}"
        )
    if frame.dtype != torch.complex128:
        raise TypeError(f"occupied frame must be complex128, got {frame.dtype}")

    ranks = ranks.to(device=frame.device, dtype=torch.long)
    if bool(torch.any(ranks < 0)) or bool(torch.any(ranks > capacity)):
        raise ValueError("occupied-frame ranks lie outside the saved capacity")

    column = torch.arange(capacity, device=frame.device)
    active = frame * (column[None, None, :] < ranks[:, None, None])
    rows = active.reshape(samples, ny, nx, 2, capacity).permute(0, 2, 1, 3, 4)
    result = torch.empty(
        (samples, nx, ny // 2 + 1),
        dtype=frame.real.dtype,
        device=frame.device,
    )
    normalization = float(2 * ny)
    for separation in range(ny // 2 + 1):
        right = torch.roll(rows, shifts=-separation, dims=2)
        overlaps = torch.einsum("bxymk,bxynk->bxymn", rows, right.conj())
        result[:, :, separation] = (
            overlaps.abs().square().sum(dim=(2, 3, 4)) / normalization
        )
    return result.to(torch.float64)


def endpoint_result_payload(*, frame, ranks, nx, ny, global_sample_indices):
    ids = np.asarray(global_sample_indices, dtype=np.int64)
    if ids.shape != (frame.shape[0],) or len(np.unique(ids)) != len(ids):
        raise ValueError("sample indices must be unique and match the batch")
    values = x_resolved_square_correlator_from_frame(frame, ranks, nx=nx, ny=ny)
    values = values.detach().cpu().numpy()
    charge = ranks.detach().cpu().numpy().astype(np.int64)
    active = frame * (torch.arange(frame.shape[-1], device=frame.device)[None, None, :] < ranks[:, None, None])
    residual = float(np.max(np.abs(active.abs().square().sum((-2,-1)).cpu().numpy()-charge), initial=0))
    if not np.isfinite(residual) or residual > 1e-8:
        raise FloatingPointError(f"occupied-frame charge/rank residual: {residual}")
    payload = {
        "observer_schema": np.asarray(OBSERVER_SCHEMA),
        "cycles": np.asarray([2*ny], dtype=np.int64),
        "normalized_cycles": np.asarray([2.], dtype=np.float64),
        "x_values": np.arange(nx, dtype=np.int64),
        "ry_values": np.arange(ny//2+1, dtype=np.int64),
        "wall_locations": np.asarray([nx//4, 3*nx//4], dtype=np.int64),
        "global_sample_indices": ids,
        "x_resolved_square_correlator": values[:, None, :, :],
        "xavg_square_correlator_vs_ry": values.mean(axis=1)[:, None, :],
        "global_charge": charge[:, None],
        "half_filling_offset": (charge-nx*ny)[:, None],
        "maximum_charge_orthonormality_residual": np.asarray(residual),
    }
    validate_endpoint_payload(payload, nx=nx, ny=ny, global_sample_indices=ids)
    return payload


def validate_endpoint_payload(payload, *, nx, ny, global_sample_indices):
    ids = np.asarray(global_sample_indices, dtype=np.int64)
    count = len(ids)
    expected = {
        "observer_schema": OBSERVER_SCHEMA,
        "cycles": np.array([2*ny]), "normalized_cycles": np.array([2.]),
        "x_values": np.arange(nx), "ry_values": np.arange(ny//2+1),
        "wall_locations": np.array([nx//4,3*nx//4]),
        "global_sample_indices": ids,
    }
    for key, value in expected.items():
        if key not in payload or not np.array_equal(payload[key], value):
            raise ValueError(f"endpoint identity mismatch: {key}")
    shapes = {
        "x_resolved_square_correlator": (count,1,nx,ny//2+1),
        "xavg_square_correlator_vs_ry": (count,1,ny//2+1),
        "global_charge": (count,1), "half_filling_offset": (count,1),
    }
    for key, shape in shapes.items():
        arr = np.asarray(payload[key])
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"endpoint shape/finiteness mismatch: {key}")
        integer = key in ("global_charge", "half_filling_offset")
        if arr.dtype != (np.int64 if integer else np.float64):
            raise ValueError(f"endpoint dtype mismatch: {key}")
    x = payload["x_resolved_square_correlator"]
    if np.any(x < 0):
        raise ValueError("negative squared correlator")
    if not np.allclose(x.mean(axis=2), payload["xavg_square_correlator_vs_ry"], rtol=2e-14, atol=1e-15):
        raise ValueError("x average does not match x-resolved correlator")
    charge = payload["global_charge"]
    if np.any(charge < 0) or np.any(charge > 2*nx*ny):
        raise ValueError("charge outside physical range")
    if not np.array_equal(payload["half_filling_offset"], charge-nx*ny):
        raise ValueError("half-filling offset mismatch")
    residual = float(np.asarray(payload["maximum_charge_orthonormality_residual"]).item())
    if not np.isfinite(residual) or not 0 <= residual <= 1e-8:
        raise ValueError("invalid charge/rank residual")
