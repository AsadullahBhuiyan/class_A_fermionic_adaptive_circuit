"""Final-time correlators, frames, and half-system covariance data."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


OBSERVER_SCHEMA = "hard_wall_xresolved_endpoint_frame_halfcov_observer_v1"


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


def half_system_covariance_and_occupations_from_frame(
    frame: torch.Tensor,
    ranks: torch.Tensor,
    *,
    nx: int,
    ny: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return ``C_A`` and its spectrum for ``A=[0,Nx)x[0,Ny//2)``.

    The occupied-frame row ordering is ``(y, x, orbital)``.  Selecting the
    first ``Ny//2`` rows in ``y`` therefore selects the requested half system.
    Eigenvalues are sorted ascending by ``torch.linalg.eigvalsh``.  Values
    outside ``[0,1]`` by no more than roundoff are clipped only in the saved
    spectrum; the raw extrema remain in the diagnostics.
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
    ay = ny // 2
    region_rows = active.reshape(samples, ny, nx, 2, capacity)[:, :ay]
    region_rows = region_rows.reshape(samples, 2 * nx * ay, capacity)
    covariance = region_rows @ region_rows.mH
    hermiticity_residual = float(
        torch.max(torch.abs(covariance - covariance.mH)).detach().cpu().item()
    )
    if not bool(torch.isfinite(covariance).all()):
        raise FloatingPointError("half-system covariance contains nonfinite values")
    if hermiticity_residual > 1.0e-10:
        raise FloatingPointError(
            "half-system covariance Hermiticity residual exceeds tolerance: "
            f"{hermiticity_residual:.6e}"
        )

    occupations_raw = torch.linalg.eigvalsh(covariance).to(torch.float64)
    raw_minimum = float(occupations_raw.min().detach().cpu().item())
    raw_maximum = float(occupations_raw.max().detach().cpu().item())
    if not bool(torch.isfinite(occupations_raw).all()):
        raise FloatingPointError("half-system occupations contain nonfinite values")
    if raw_minimum < -1.0e-10 or raw_maximum > 1.0 + 1.0e-10:
        raise FloatingPointError(
            "half-system occupations lie outside the physical interval: "
            f"min={raw_minimum:.6e}, max={raw_maximum:.6e}"
        )
    occupations = occupations_raw.clamp(0.0, 1.0)
    return covariance, occupations, {
        "maximum_hermiticity_residual": hermiticity_residual,
        "raw_occupation_minimum": raw_minimum,
        "raw_occupation_maximum": raw_maximum,
    }


def endpoint_result_payload(
    *,
    frame: torch.Tensor,
    ranks: torch.Tensor,
    nx: int,
    ny: int,
    global_sample_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return one five-trajectory endpoint payload with native final states."""

    sample_indices = np.asarray(global_sample_indices, dtype=np.int64)
    if sample_indices.ndim != 1 or len(sample_indices) != int(frame.shape[0]):
        raise ValueError("global sample indices do not match the frame batch")
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError("global sample indices must be unique")

    correlator = x_resolved_square_correlator_from_frame(frame, ranks, nx=nx, ny=ny)
    half_covariance, half_occupations, half_diagnostics = (
        half_system_covariance_and_occupations_from_frame(
            frame, ranks, nx=nx, ny=ny
        )
    )
    rank_array = ranks.detach().cpu().numpy().astype(np.int64, copy=False)
    charge_from_frame = (
        frame.abs().square().sum(dim=(-2, -1)).real.detach().cpu().numpy()
    )
    maximum_charge_residual = float(
        np.max(np.abs(charge_from_frame - rank_array), initial=0.0)
    )
    if maximum_charge_residual > 1.0e-8:
        raise FloatingPointError(
            "occupied-frame charge/rank mismatch exceeds tolerance: "
            f"{maximum_charge_residual:.6e}"
        )

    values = correlator.detach().cpu().numpy().astype(np.float64, copy=False)
    occupied_frame = frame.detach().cpu().numpy().astype(np.complex128, copy=True)
    half_covariance_values = (
        half_covariance.detach().cpu().numpy().astype(np.complex128, copy=True)
    )
    half_occupation_values = (
        half_occupations.detach().cpu().numpy().astype(np.float64, copy=True)
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("endpoint correlator contains nonfinite values")
    xavg = values.mean(axis=1)
    cycles = np.asarray([2 * int(ny)], dtype=np.int64)
    return {
        "observer_schema": np.asarray(OBSERVER_SCHEMA),
        "cycles": cycles,
        "normalized_cycles": cycles.astype(np.float64) / float(ny),
        "ry_values": np.arange(int(ny) // 2 + 1, dtype=np.int64),
        "x_values": np.arange(int(nx), dtype=np.int64),
        "wall_locations": np.asarray([5, 15], dtype=np.int64),
        "global_sample_indices": sample_indices.copy(),
        "x_resolved_square_correlator": values[:, None, :, :],
        "xavg_square_correlator_vs_ry": xavg[:, None, :],
        "global_charge": rank_array[:, None],
        "half_filling_offset": (rank_array - int(nx) * int(ny))[:, None],
        "occupied_frame": occupied_frame,
        "occupied_ranks": rank_array.copy(),
        "half_system_Ay": np.asarray(int(ny) // 2, dtype=np.int64),
        "half_system_region_bounds": np.asarray(
            [0, int(nx), 0, int(ny) // 2], dtype=np.int64
        ),
        "half_system_covariance": half_covariance_values[:, None, :, :],
        "half_system_occupation_spectrum": half_occupation_values[:, None, :],
        "half_system_covariance_convention": np.asarray(
            "C_A=F_A@F_A_dagger;A=[0,Nx)x[0,Ny//2)"
        ),
        "half_system_occupation_ordering": np.asarray("ascending"),
        "frame_covariance_convention": np.asarray("C=F@F_dagger;G=2C-I"),
        "maximum_half_system_hermiticity_residual": np.asarray(
            half_diagnostics["maximum_hermiticity_residual"], dtype=np.float64
        ),
        "half_system_raw_occupation_minimum": np.asarray(
            half_diagnostics["raw_occupation_minimum"], dtype=np.float64
        ),
        "half_system_raw_occupation_maximum": np.asarray(
            half_diagnostics["raw_occupation_maximum"], dtype=np.float64
        ),
        "maximum_charge_orthonormality_residual": np.asarray(
            maximum_charge_residual, dtype=np.float64
        ),
    }


def validate_endpoint_payload(
    payload: dict[str, Any],
    *,
    nx: int,
    ny: int,
    global_sample_indices: np.ndarray,
) -> None:
    """Validate the scientific arrays in one result shard."""

    sample_indices = np.asarray(global_sample_indices, dtype=np.int64)
    required = {
        "observer_schema",
        "cycles",
        "normalized_cycles",
        "ry_values",
        "x_values",
        "wall_locations",
        "global_sample_indices",
        "x_resolved_square_correlator",
        "xavg_square_correlator_vs_ry",
        "global_charge",
        "half_filling_offset",
        "occupied_frame",
        "occupied_ranks",
        "half_system_Ay",
        "half_system_region_bounds",
        "half_system_covariance",
        "half_system_occupation_spectrum",
        "half_system_covariance_convention",
        "half_system_occupation_ordering",
        "frame_covariance_convention",
        "maximum_half_system_hermiticity_residual",
        "half_system_raw_occupation_minimum",
        "half_system_raw_occupation_maximum",
        "maximum_charge_orthonormality_residual",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"endpoint payload missing fields: {missing}")
    if str(np.asarray(payload["observer_schema"]).item()) != OBSERVER_SCHEMA:
        raise ValueError("endpoint observer schema mismatch")
    expected_cycles = np.asarray([2 * int(ny)], dtype=np.int64)
    if not np.array_equal(payload["cycles"], expected_cycles):
        raise ValueError("endpoint cycle label mismatch")
    if not np.array_equal(payload["normalized_cycles"], np.asarray([2.0])):
        raise ValueError("endpoint normalized-cycle label mismatch")
    if not np.array_equal(
        payload["ry_values"], np.arange(int(ny) // 2 + 1, dtype=np.int64)
    ):
        raise ValueError("endpoint y-separation labels mismatch")
    if not np.array_equal(payload["x_values"], np.arange(int(nx), dtype=np.int64)):
        raise ValueError("endpoint x labels mismatch")
    if not np.array_equal(payload["wall_locations"], np.asarray([5, 15])):
        raise ValueError("endpoint wall locations mismatch")
    if not np.array_equal(payload["global_sample_indices"], sample_indices):
        raise ValueError("endpoint sample indices mismatch")

    expected_shape = (len(sample_indices), 1, int(nx), int(ny) // 2 + 1)
    x_resolved = np.asarray(payload["x_resolved_square_correlator"])
    xavg = np.asarray(payload["xavg_square_correlator_vs_ry"])
    charge = np.asarray(payload["global_charge"])
    offset = np.asarray(payload["half_filling_offset"])
    occupied_frame = np.asarray(payload["occupied_frame"])
    occupied_ranks = np.asarray(payload["occupied_ranks"])
    half_system_covariance = np.asarray(payload["half_system_covariance"])
    half_system_occupations = np.asarray(payload["half_system_occupation_spectrum"])
    if x_resolved.shape != expected_shape or x_resolved.dtype != np.float64:
        raise ValueError("endpoint x-resolved shape/dtype mismatch")
    if xavg.shape != expected_shape[:2] + (expected_shape[3],):
        raise ValueError("endpoint x-average shape mismatch")
    if charge.shape != expected_shape[:2] or charge.dtype != np.int64:
        raise ValueError("endpoint charge shape/dtype mismatch")
    if offset.shape != charge.shape or offset.dtype != np.int64:
        raise ValueError("endpoint half-filling offset shape/dtype mismatch")
    expected_frame_shape = (
        len(sample_indices),
        2 * int(nx) * int(ny),
        int(occupied_frame.shape[-1]) if occupied_frame.ndim == 3 else -1,
    )
    if (
        occupied_frame.shape != expected_frame_shape
        or occupied_frame.dtype != np.complex128
    ):
        raise ValueError("endpoint occupied-frame shape/dtype mismatch")
    if occupied_frame.ndim != 3 or occupied_frame.shape[-1] > occupied_frame.shape[-2]:
        raise ValueError("endpoint occupied-frame capacity is invalid")
    if (
        occupied_ranks.shape != (len(sample_indices),)
        or occupied_ranks.dtype != np.int64
    ):
        raise ValueError("endpoint occupied-ranks shape/dtype mismatch")
    ay = int(ny) // 2
    subsystem_modes = 2 * int(nx) * ay
    if int(np.asarray(payload["half_system_Ay"]).item()) != ay:
        raise ValueError("endpoint half-system Ay mismatch")
    if not np.array_equal(
        payload["half_system_region_bounds"], np.asarray([0, int(nx), 0, ay])
    ):
        raise ValueError("endpoint half-system region mismatch")
    if (
        half_system_covariance.shape
        != (len(sample_indices), 1, subsystem_modes, subsystem_modes)
        or half_system_covariance.dtype != np.complex128
    ):
        raise ValueError("endpoint half-system covariance shape/dtype mismatch")
    if (
        half_system_occupations.shape != (len(sample_indices), 1, subsystem_modes)
        or half_system_occupations.dtype != np.float64
    ):
        raise ValueError("endpoint half-system occupation shape/dtype mismatch")
    if not np.isfinite(half_system_covariance).all():
        raise FloatingPointError("endpoint half-system covariance is nonfinite")
    if not np.isfinite(half_system_occupations).all():
        raise FloatingPointError("endpoint half-system occupations are nonfinite")
    if np.any(half_system_occupations < 0.0) or np.any(half_system_occupations > 1.0):
        raise ValueError("endpoint half-system occupations leave [0,1]")
    if np.any(np.diff(half_system_occupations, axis=-1) < -1.0e-14):
        raise ValueError("endpoint half-system occupations are not sorted ascending")
    if not np.allclose(
        half_system_covariance,
        half_system_covariance.swapaxes(-1, -2).conj(),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("endpoint half-system covariance is not Hermitian")
    covariance_trace = np.trace(half_system_covariance, axis1=-2, axis2=-1).real
    occupation_trace = half_system_occupations.sum(axis=-1)
    if not np.allclose(covariance_trace, occupation_trace, rtol=0.0, atol=1.0e-8):
        raise ValueError("half-system covariance trace and occupations disagree")
    if str(np.asarray(payload["half_system_covariance_convention"]).item()) != (
        "C_A=F_A@F_A_dagger;A=[0,Nx)x[0,Ny//2)"
    ):
        raise ValueError("endpoint half-system covariance convention mismatch")
    if str(np.asarray(payload["half_system_occupation_ordering"]).item()) != "ascending":
        raise ValueError("endpoint half-system occupation ordering mismatch")
    if str(np.asarray(payload["frame_covariance_convention"]).item()) != (
        "C=F@F_dagger;G=2C-I"
    ):
        raise ValueError("endpoint frame covariance convention mismatch")
    if not np.isfinite(occupied_frame).all():
        raise FloatingPointError("endpoint occupied frame contains nonfinite values")
    if np.any(occupied_ranks < 0) or np.any(occupied_ranks > occupied_frame.shape[-1]):
        raise ValueError("endpoint occupied ranks lie outside frame capacity")
    if not np.array_equal(occupied_ranks[:, None], charge):
        raise ValueError("endpoint occupied ranks do not match global charge")
    if not np.isfinite(x_resolved).all() or not np.isfinite(xavg).all():
        raise FloatingPointError("endpoint correlator contains nonfinite values")
    if not np.allclose(xavg, x_resolved.mean(axis=2), rtol=2.0e-13, atol=2.0e-13):
        raise ValueError("endpoint x average does not match x-resolved data")
    if not np.array_equal(offset, charge - int(nx) * int(ny)):
        raise ValueError("endpoint half-filling offset does not match charge")
    residual = float(
        np.asarray(payload["maximum_charge_orthonormality_residual"]).item()
    )
    if not np.isfinite(residual) or residual < 0 or residual > 1.0e-8:
        raise ValueError("endpoint charge residual is invalid")
    half_residual = float(
        np.asarray(payload["maximum_half_system_hermiticity_residual"]).item()
    )
    raw_minimum = float(
        np.asarray(payload["half_system_raw_occupation_minimum"]).item()
    )
    raw_maximum = float(
        np.asarray(payload["half_system_raw_occupation_maximum"]).item()
    )
    if not np.isfinite(half_residual) or not 0.0 <= half_residual <= 1.0e-10:
        raise ValueError("endpoint half-system Hermiticity residual is invalid")
    if not np.isfinite(raw_minimum) or raw_minimum < -1.0e-10:
        raise ValueError("endpoint raw half-system occupation minimum is invalid")
    if not np.isfinite(raw_maximum) or raw_maximum > 1.0 + 1.0e-10:
        raise ValueError("endpoint raw half-system occupation maximum is invalid")
