"""Per-trajectory endpoint entropy and log-chord central-charge estimator.

The estimator uses the occupied frame directly.  For a pure Gaussian state
``C = F F^dagger`` and the restricted correlation eigenvalues are the squared
singular values of the corresponding rows of ``F``.  No dense full-system
covariance is constructed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


ENDPOINT_CFT_SCHEMA = "state_projector_endpoint_cft_v1"


def log_chord(ay: np.ndarray, *, ny: int) -> np.ndarray:
    ay = np.asarray(ay, dtype=np.float64)
    if np.any(ay <= 0) or np.any(ay >= int(ny)):
        raise ValueError("log-chord widths must lie strictly between zero and Ny")
    return np.log((float(ny) / math.pi) * np.sin(math.pi * ay / float(ny)))


def periodic_strip_indices(
    *, nx: int, ny: int, y0_start: int, y0_stop: int, ay: int
) -> torch.Tensor:
    """Indices for full-x strips, ordered by relative y, x, then orbital."""

    nx, ny, ay = int(nx), int(ny), int(ay)
    origins = torch.arange(int(y0_start), int(y0_stop), dtype=torch.long)
    dy = torch.arange(ay, dtype=torch.long)
    x = torch.arange(nx, dtype=torch.long)
    orbital = torch.arange(2, dtype=torch.long)
    y = (origins[:, None] + dy[None, :]) % ny
    return (
        2 * nx * y[:, :, None, None]
        + 2 * x[None, None, :, None]
        + orbital[None, None, None, :]
    ).reshape(len(origins), 2 * nx * ay)


def y0_averaged_entropy_curve(
    frame: np.ndarray,
    *,
    nx: int,
    ny: int,
    y0_chunk: int = 4,
    occupation_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Return one trajectory's full-x strip entropy averaged over every y0."""

    frame = np.asarray(frame)
    nx, ny, y0_chunk = int(nx), int(ny), int(y0_chunk)
    if frame.ndim != 2 or frame.shape[0] != 2 * nx * ny:
        raise ValueError("occupied frame has the wrong physical dimension")
    if frame.dtype != np.complex128:
        raise TypeError("endpoint CFT estimator requires a complex128 frame")
    if y0_chunk <= 0:
        raise ValueError("y0_chunk must be positive")
    if not np.isfinite(frame).all():
        raise FloatingPointError("occupied frame contains nonfinite values")

    tensor = torch.from_numpy(frame)
    ay_values = np.arange(ny // 2 + 1, dtype=np.int64)
    entropy = np.zeros(len(ay_values), dtype=np.float64)
    occupation_min, occupation_max = np.inf, -np.inf
    tolerance = float(occupation_tolerance)

    with torch.inference_mode():
        for width_index, ay in enumerate(ay_values[1:], start=1):
            total = 0.0
            for y0_start in range(0, ny, y0_chunk):
                y0_stop = min(ny, y0_start + y0_chunk)
                indices = periodic_strip_indices(
                    nx=nx, ny=ny, y0_start=y0_start, y0_stop=y0_stop, ay=int(ay)
                )
                rows = tensor.index_select(0, indices.reshape(-1)).reshape(
                    y0_stop - y0_start, 2 * nx * int(ay), frame.shape[1]
                )
                raw = torch.linalg.svdvals(rows).square().real
                block_min = float(raw.amin())
                block_max = float(raw.amax())
                occupation_min = min(occupation_min, block_min)
                occupation_max = max(occupation_max, block_max)
                if block_min < -tolerance or block_max > 1.0 + tolerance:
                    raise FloatingPointError(
                        "restricted-frame occupation lies outside [0,1] tolerance: "
                        f"min={block_min:.6e}, max={block_max:.6e}"
                    )
                probability = raw.clamp(0.0, 1.0)
                complement = 1.0 - probability
                block_entropy = -torch.xlogy(probability, probability) - torch.xlogy(
                    complement, complement
                )
                total += float(block_entropy.sum())
            entropy[width_index] = total / float(ny)

    return {
        "ay_values": ay_values,
        "entropy": entropy,
        "occupation_min": float(occupation_min),
        "occupation_max": float(occupation_max),
    }


def fit_central_charge(
    ay_values: np.ndarray,
    entropy: np.ndarray,
    *,
    ny: int,
    fit_ay_min: int = 8,
) -> dict[str, Any]:
    """Fit S(Ay)=intercept+(c_eff/3) log[(Ny/pi) sin(pi Ay/Ny)]."""

    ay_values = np.asarray(ay_values, dtype=np.int64)
    entropy = np.asarray(entropy, dtype=np.float64)
    if ay_values.shape != entropy.shape or entropy.ndim != 1:
        raise ValueError("Ay and entropy must be matching one-dimensional arrays")
    selected = (ay_values >= int(fit_ay_min)) & (ay_values <= int(ny) // 2)
    if int(selected.sum()) < 3:
        raise ValueError("central-charge fit requires at least three widths")
    fit_ay = ay_values[selected]
    x = log_chord(fit_ay, ny=int(ny))
    y = entropy[selected]
    if not np.isfinite(y).all():
        raise FloatingPointError("entropy fit input contains nonfinite values")
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = slope * x + intercept
    residuals = y - fitted
    rss = float(np.dot(residuals, residuals))
    total = float(np.dot(y - y.mean(), y - y.mean()))
    degrees = int(len(y) - 2)
    slope_stderr = (
        float(np.sqrt((rss / degrees) / np.dot(x - x.mean(), x - x.mean())))
        if degrees > 0 and np.dot(x - x.mean(), x - x.mean()) > 0
        else float("nan")
    )
    return {
        "fit_ay": fit_ay,
        "fit_log_chord": x,
        "fit_entropy": y,
        "fit_residuals": residuals,
        "slope": float(slope),
        "slope_stderr": slope_stderr,
        "intercept": float(intercept),
        "c_eff": float(3.0 * slope),
        "c_eff_stderr": float(3.0 * slope_stderr),
        "r2": float("nan") if total == 0.0 else float(1.0 - rss / total),
        "rss": rss,
        "fit_point_count": int(len(y)),
        "fit_ay_min": int(fit_ay[0]),
        "fit_ay_max": int(fit_ay[-1]),
    }


def endpoint_cft_payload(frame: np.ndarray, *, nx: int, ny: int, contract: dict[str, Any]) -> dict[str, Any]:
    curve = y0_averaged_entropy_curve(
        frame,
        nx=int(nx),
        ny=int(ny),
        y0_chunk=int(contract["y0_chunk"]),
        occupation_tolerance=float(contract["occupation_tolerance"]),
    )
    fit = fit_central_charge(
        curve["ay_values"], curve["entropy"], ny=int(ny), fit_ay_min=int(contract["fit_ay_min"])
    )
    return {
        "endpoint_cft_schema": np.asarray(ENDPOINT_CFT_SCHEMA),
        "endpoint_entropy_ay": curve["ay_values"],
        "endpoint_entropy_y0_averaged": curve["entropy"],
        "endpoint_entropy_y0_count": np.asarray(int(ny), dtype=np.int64),
        "endpoint_entropy_fit_ay": fit["fit_ay"],
        "endpoint_entropy_fit_log_chord": fit["fit_log_chord"],
        "endpoint_entropy_fit_values": fit["fit_entropy"],
        "endpoint_entropy_fit_residuals": fit["fit_residuals"],
        "endpoint_entropy_fit_slope": np.asarray(fit["slope"]),
        "endpoint_entropy_fit_slope_stderr": np.asarray(fit["slope_stderr"]),
        "endpoint_entropy_fit_intercept": np.asarray(fit["intercept"]),
        "endpoint_c_eff": np.asarray(fit["c_eff"]),
        "endpoint_c_eff_stderr": np.asarray(fit["c_eff_stderr"]),
        "endpoint_entropy_fit_r2": np.asarray(fit["r2"]),
        "endpoint_entropy_fit_rss": np.asarray(fit["rss"]),
        "endpoint_entropy_fit_point_count": np.asarray(fit["fit_point_count"], dtype=np.int64),
        "endpoint_entropy_fit_ay_min": np.asarray(fit["fit_ay_min"], dtype=np.int64),
        "endpoint_entropy_fit_ay_max": np.asarray(fit["fit_ay_max"], dtype=np.int64),
        "endpoint_entropy_occupation_min": np.asarray(curve["occupation_min"]),
        "endpoint_entropy_occupation_max": np.asarray(curve["occupation_max"]),
    }


def validate_endpoint_cft_payload(payload: dict[str, Any], *, ny: int, contract: dict[str, Any]) -> None:
    if str(np.asarray(payload["endpoint_cft_schema"]).item()) != ENDPOINT_CFT_SCHEMA:
        raise ValueError("endpoint CFT schema mismatch")
    expected_ay = np.arange(int(ny) // 2 + 1, dtype=np.int64)
    if not np.array_equal(np.asarray(payload["endpoint_entropy_ay"]), expected_ay):
        raise ValueError("endpoint entropy width axis mismatch")
    entropy = np.asarray(payload["endpoint_entropy_y0_averaged"], dtype=np.float64)
    if entropy.shape != expected_ay.shape or not np.isfinite(entropy).all() or entropy[0] != 0.0:
        raise ValueError("endpoint entropy curve is invalid")
    fit_ay = np.arange(int(contract["fit_ay_min"]), int(ny) // 2 + 1, dtype=np.int64)
    if not np.array_equal(np.asarray(payload["endpoint_entropy_fit_ay"]), fit_ay):
        raise ValueError("endpoint entropy fit-width axis mismatch")
    for key in (
        "endpoint_entropy_fit_log_chord",
        "endpoint_entropy_fit_values",
        "endpoint_entropy_fit_residuals",
    ):
        value = np.asarray(payload[key], dtype=np.float64)
        if value.shape != fit_ay.shape or not np.isfinite(value).all():
            raise ValueError(f"invalid endpoint CFT field: {key}")
    for key in (
        "endpoint_entropy_fit_slope",
        "endpoint_entropy_fit_slope_stderr",
        "endpoint_entropy_fit_intercept",
        "endpoint_c_eff",
        "endpoint_c_eff_stderr",
        "endpoint_entropy_fit_r2",
        "endpoint_entropy_fit_rss",
        "endpoint_entropy_occupation_min",
        "endpoint_entropy_occupation_max",
    ):
        if np.asarray(payload[key]).shape != () or not np.isfinite(float(np.asarray(payload[key]))):
            raise ValueError(f"invalid endpoint CFT scalar: {key}")
    if int(np.asarray(payload["endpoint_entropy_y0_count"])) != int(ny):
        raise ValueError("endpoint entropy did not average every y0")
    if int(np.asarray(payload["endpoint_entropy_fit_point_count"])) != len(fit_ay):
        raise ValueError("endpoint entropy fit point count mismatch")
    if int(np.asarray(payload["endpoint_entropy_fit_ay_min"])) != int(fit_ay[0]):
        raise ValueError("endpoint entropy fit minimum mismatch")
    if int(np.asarray(payload["endpoint_entropy_fit_ay_max"])) != int(fit_ay[-1]):
        raise ValueError("endpoint entropy fit maximum mismatch")
