"""Scientific construction and statistics for the boundary anisotropy campaign."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


CONFIG_SCHEMA = "boundary_only_reference_anisotropy_cpu_config_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
STREAM_CODES = {
    "engine": 101,
    "post": 211,
    "probe": 307,
    "position": 401,
    "bootstrap": 503,
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def configuration_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("configuration schema mismatch")
    if config["geometry"] != {
        "Nx": 16,
        "Ny_values": [16, 20, 24, 28, 32],
        "domain_wall_interval": [4, 12],
        "controller_twist_y": 1.0e-7,
        "initial_twist_y": 0.0,
    }:
        raise ValueError("geometry differs from the locked campaign")
    if config["initial_state"] != {
        "hamiltonian": "periodic_translation_invariant_qwz",
        "mass": 1.0,
        "filling_fraction": 0.5,
        "construction": "fill_exact_negative_energy_bloch_band",
    }:
        raise ValueError("initial-state contract mismatch")
    controller = config["controller"]
    expected = {
        "DW": True,
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "trial_orbitals": "X",
        "dw_truncation": True,
        "measurement_x": [4, 12],
        "measurement_slab_only": False,
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "state_representation": "physical_frame",
        "dtype": "complex128",
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
    }
    if controller != expected:
        raise ValueError("boundary-controller contract mismatch")
    targets = list(config["production"]["sample_targets"])
    if targets != list(range(100, 401, 50)):
        raise ValueError("production sample targets must be 100..400 in increments of 50")


def stable_seed(root_seed: int, ny: int, sample: int, stream: str) -> int:
    if stream not in STREAM_CODES:
        raise KeyError(stream)
    sequence = np.random.SeedSequence(
        [int(root_seed), int(ny), int(sample), int(STREAM_CODES[stream])]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def qwz_negative_band_frame(
    nx: int, ny: int, *, mass: float = 1.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the exact half-filled periodic QWZ negative-band Slater frame."""

    nx, ny = int(nx), int(ny)
    sites = nx * ny
    frame = np.empty((2 * sites, sites), dtype=np.complex128)
    x = np.arange(nx, dtype=np.float64)[:, None]
    y = np.arange(ny, dtype=np.float64)[None, :]
    energies: list[float] = []
    column = 0
    for mx in range(nx):
        kx = 2.0 * np.pi * mx / nx
        for my in range(ny):
            ky = 2.0 * np.pi * my / ny
            h = np.asarray(
                [
                    [mass - np.cos(kx) - np.cos(ky), np.sin(kx) - 1j * np.sin(ky)],
                    [np.sin(kx) + 1j * np.sin(ky), -mass + np.cos(kx) + np.cos(ky)],
                ],
                dtype=np.complex128,
            )
            eigenvalues, eigenvectors = np.linalg.eigh(h)
            spinor = eigenvectors[:, 0]
            phase = np.exp(1j * (kx * x + ky * y)) / np.sqrt(sites)
            for orbital in (0, 1):
                rows = orbital + 2 * np.arange(nx)[:, None] + 2 * nx * np.arange(ny)[None, :]
                frame[rows.reshape(-1, order="F"), column] = (
                    phase * spinor[orbital]
                ).reshape(-1, order="F")
            energies.append(float(eigenvalues[0]))
            column += 1
    gram = frame.conj().T @ frame
    projector = frame @ frame.conj().T
    tx_rows = np.empty(2 * sites, dtype=np.int64)
    ty_rows = np.empty(2 * sites, dtype=np.int64)
    for yy in range(ny):
        for xx in range(nx):
            for orbital in (0, 1):
                row = orbital + 2 * xx + 2 * nx * yy
                tx_rows[row] = orbital + 2 * ((xx + 1) % nx) + 2 * nx * yy
                ty_rows[row] = orbital + 2 * xx + 2 * nx * ((yy + 1) % ny)
    diagnostics = {
        "rank": float(frame.shape[1]),
        "gram_residual": float(
            np.linalg.norm(gram - np.eye(sites), ord="fro") / np.sqrt(sites)
        ),
        "translation_x_residual": float(np.max(np.abs(projector - projector[np.ix_(tx_rows, tx_rows)]))),
        "translation_y_residual": float(np.max(np.abs(projector - projector[np.ix_(ty_rows, ty_rows)]))),
        "single_particle_gap": float(-2.0 * max(energies)),
        "particle_number": float(np.trace(projector).real),
    }
    return np.ascontiguousarray(frame), diagnostics


def wall_measurement_site_ids(nx: int, ny: int, walls: Sequence[int]) -> np.ndarray:
    values = [int(x) + int(nx) * y for y in range(int(ny)) for x in walls]
    return np.asarray(values, dtype=np.int64)


def ordered_crossings(
    temporal_separations: Sequence[float],
    temporal_values: Sequence[float],
    spatial_value: float,
) -> list[dict[str, float]]:
    separations = np.asarray(temporal_separations, dtype=np.float64)
    values = np.asarray(temporal_values, dtype=np.float64)
    if separations.ndim != 1 or values.shape != separations.shape:
        raise ValueError("temporal coordinate/value shape mismatch")
    order = np.argsort(separations)
    separations, values = separations[order], values[order]
    target = float(spatial_value)
    if not np.isfinite(target) or target <= 0.0 or not np.all(np.isfinite(values)):
        return []
    difference = values - target
    if difference[0] <= 0.0:
        return []
    output: list[dict[str, float]] = []
    for index in range(separations.size - 1):
        if difference[index] == 0.0:
            output.append(
                {
                    "lower": float(separations[index]),
                    "upper": float(separations[index]),
                    "time_star": float(separations[index]),
                }
            )
        elif difference[index] > 0.0 and difference[index + 1] < 0.0:
            fraction = difference[index] / (difference[index] - difference[index + 1])
            output.append(
                {
                    "lower": float(separations[index]),
                    "upper": float(separations[index + 1]),
                    "time_star": float(
                        separations[index]
                        + fraction * (separations[index + 1] - separations[index])
                    ),
                }
            )
    return output


def anisotropy(ny: int, time_star: float) -> float:
    if not np.isfinite(time_star) or float(time_star) <= 0.0:
        return float("nan")
    return float(np.arcsinh(1.0) * int(ny) / (np.pi * float(time_star)))
