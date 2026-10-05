#!/usr/bin/env python3
"""Quenched log-polar analysis for the completed hard/soft max-mix campaigns.

The hard construction lives on the active slab ``x=5,...,15`` after its
Born-conditioned product exterior is removed.  The soft construction retains
the full ``x=0,...,19`` transfer sector.  If ``G`` is the appropriate centered
covariance, the output-side polar generator and its Gaussian Hamiltonian are

    A = log(P_L) = arctanh(G),       h = -2 A.

The endpoint is nearly pure, so ``A`` is evaluated as a capped spectral
function.  The logarithm is always taken trajectory by trajectory before the
linear y twirl and the quenched sample average.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


ANALYSIS_SCHEMA = "domain_wall_quenched_log_polar_analysis_v2"
CACHE_SCHEMA = "domain_wall_log_polar_sample_cache_v2"
CACHE_COMPLETION_SCHEMA = "domain_wall_log_polar_cache_completion_v2"
EXPECTED_REVISION = "maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2"
EXPECTED_RESULT_SCHEMA = "maxmix_purification_result_v2"
EXPECTED_COMPLETION_SCHEMA = "maxmix_purification_completion_v2"
EXPECTED_CONFIG_HASH = "2dc0ba9a2a3ec8cc79eebec19bda0a6bcbaf8f3efcd7b47da99eab8decbcffce"
EXPECTED_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
EXPECTED_NX = 20
ACTIVE_X = tuple(range(5, 16))
SOFT_ACTIVE_X = tuple(range(EXPECTED_NX))
WALL_X = (5, 15)
WALL_WINDOWS = ((4, 5, 6), (14, 15, 16))
OCCUPATION_ROUNDOFF_TOL = 1.0e-9
NUMERICAL_TOL = 1.0e-10

BUNDLE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (
    BUNDLE_ROOT
    / "gpu_data"
    / EXPECTED_REVISION
    / "hard"
)
DEFAULT_OUTPUT_ROOT = (
    BUNDLE_ROOT
    / "analysis_outputs"
    / "hard_quenched_log_polar_v1"
)
SOFT_REVISION = "maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3"
SOFT_RESULT_SCHEMA = "maxmix_purification_result_v3"
SOFT_COMPLETION_SCHEMA = "maxmix_purification_completion_v3"
SOFT_CONFIG_HASH = "a0169e31d63dc77f08da34b491a88516a05cdb1dd4f98badd19e1735c180b821"
DEFAULT_SOFT_DATA_ROOT = BUNDLE_ROOT / "gpu_data" / SOFT_REVISION / "soft"
DEFAULT_SOFT_OUTPUT_ROOT = (
    BUNDLE_ROOT / "analysis_outputs" / "soft_quenched_log_polar_v1"
)


@dataclass(frozen=True)
class GeometrySpec:
    construction: str
    revision: str
    result_schema: str
    completion_schema: str
    configuration_hash: str
    active_x: tuple[int, ...]
    data_root: Path
    output_root: Path
    require_product_exterior: bool

    def transfer_mode_count(self, ny: int) -> int:
        return 2 * len(self.active_x) * int(ny)


HARD_SPEC = GeometrySpec(
    construction="hard",
    revision=EXPECTED_REVISION,
    result_schema=EXPECTED_RESULT_SCHEMA,
    completion_schema=EXPECTED_COMPLETION_SCHEMA,
    configuration_hash=EXPECTED_CONFIG_HASH,
    active_x=ACTIVE_X,
    data_root=DEFAULT_DATA_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
    require_product_exterior=True,
)
SOFT_SPEC = GeometrySpec(
    construction="soft",
    revision=SOFT_REVISION,
    result_schema=SOFT_RESULT_SCHEMA,
    completion_schema=SOFT_COMPLETION_SCHEMA,
    configuration_hash=SOFT_CONFIG_HASH,
    active_x=SOFT_ACTIVE_X,
    data_root=DEFAULT_SOFT_DATA_ROOT,
    output_root=DEFAULT_SOFT_OUTPUT_ROOT,
    require_product_exterior=False,
)


def geometry_spec(construction: str) -> GeometrySpec:
    if construction == "hard":
        return HARD_SPEC
    if construction == "soft":
        return SOFT_SPEC
    raise ValueError("construction must be 'hard' or 'soft'")


@dataclass(frozen=True)
class SampleRecord:
    ny: int
    sample_index: int
    sample_offset: int
    result_path: Path
    completion_path: Path
    result_sha256: str
    construction: str = "hard"

    @property
    def task_id(self) -> str:
        return f"Ny{self.ny:03d}_sample{self.sample_index:03d}"


@dataclass(frozen=True)
class AnalysisConfig:
    construction: str = "hard"
    caps: tuple[float, ...] = (8.0, 10.0, 12.0)
    default_cap: float = 10.0
    renyi_orders: tuple[int, ...] = (1, 2, 3)
    twists: tuple[float, ...] = (-1.0e-7, 1.0e-7)
    fit_ay_min: int = 8
    sensitivity_ay_min: int = 6
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 2026091907
    jackknife_groups: int = 10
    wall_weight_min: float = 0.5

    def __post_init__(self) -> None:
        geometry_spec(self.construction)
        caps = tuple(float(value) for value in self.caps)
        orders = tuple(int(value) for value in self.renyi_orders)
        twists = tuple(float(value) for value in self.twists)
        if not caps or any(value <= 0 for value in caps):
            raise ValueError("caps must be positive")
        if float(self.default_cap) not in caps:
            raise ValueError("default_cap must be one of caps")
        if orders != tuple(sorted(set(orders))) or any(value < 1 for value in orders):
            raise ValueError("renyi_orders must be unique increasing positive integers")
        if len(twists) != 2 or not np.isclose(twists[0], -twists[1]):
            raise ValueError("twists must be a symmetric nonzero pair")
        if self.bootstrap_replicates < 1 or self.jackknife_groups < 2:
            raise ValueError("bootstrap and jackknife counts must be positive")

    def identity(self) -> dict[str, Any]:
        spec = geometry_spec(self.construction)
        return {
            "analysis_schema": ANALYSIS_SCHEMA,
            "caps": list(self.caps),
            "default_cap": self.default_cap,
            "renyi_orders": list(self.renyi_orders),
            "twists": list(self.twists),
            "fit_ay_min": self.fit_ay_min,
            "sensitivity_ay_min": self.sensitivity_ay_min,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "jackknife_groups": self.jackknife_groups,
            "wall_weight_min": self.wall_weight_min,
            "construction": self.construction,
            "source_revision": spec.revision,
            "active_x": list(spec.active_x),
            "wall_x": list(WALL_X),
        }

    @property
    def identity_hash(self) -> str:
        raw = json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(data: Any, key: str) -> Any:
    return np.asarray(data[key]).item()


def active_indices(nx: int, ny: int, active_x: Sequence[int] = ACTIVE_X) -> np.ndarray:
    """Return y-major active-slab indices in the repository orbital convention."""
    return np.asarray(
        [orbital + 2 * x + 2 * nx * y for y in range(ny) for x in active_x for orbital in range(2)],
        dtype=np.int64,
    )


def discover_samples(
    data_root: Path | None = None,
    *,
    construction: str = "hard",
    ny_values: Sequence[int] = (20, 30, 40),
    verify_hashes: bool = True,
    require_complete: bool = True,
) -> list[SampleRecord]:
    """Verify one immutable campaign inventory and return sample records."""
    spec = geometry_spec(construction)
    data_root = spec.data_root if data_root is None else Path(data_root)
    records: list[SampleRecord] = []
    for ny in tuple(int(value) for value in ny_values):
        directory = data_root / f"Ny{ny:03d}"
        completions = sorted(directory.glob("*.complete.json"))
        if require_complete and len(completions) != 20:
            raise RuntimeError(f"{directory}: expected 20 completions, found {len(completions)}")
        seen: set[int] = set()
        for completion_path in completions:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            expected = {
                "schema": spec.completion_schema,
                "sampling_revision": spec.revision,
                "configuration_hash": spec.configuration_hash,
                "canonical_dynamics_entry_point": EXPECTED_ENTRY_POINT,
                "construction": spec.construction,
                "Nx": EXPECTED_NX,
                "Ny": ny,
                "cycles": 4 * ny,
                "dtype": "complex128",
            }
            for key, value in expected.items():
                if completion.get(key) != value:
                    raise RuntimeError(f"{completion_path}: {key} mismatch")
            result_path = directory / str(completion["result_filename"])
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            if result_path.stat().st_size != int(completion["result_bytes"]):
                raise RuntimeError(f"{result_path}: byte-count mismatch")
            result_sha = str(completion["result_sha256"])
            if verify_hashes and sha256_file(result_path) != result_sha:
                raise RuntimeError(f"{result_path}: SHA-256 mismatch")
            with np.load(result_path, allow_pickle=False) as saved:
                scalar_expected = {
                    "result_schema": spec.result_schema,
                    "sampling_revision": spec.revision,
                    "configuration_hash": spec.configuration_hash,
                    "canonical_dynamics_entry_point": EXPECTED_ENTRY_POINT,
                    "construction": spec.construction,
                    "Nx": EXPECTED_NX,
                    "Ny": ny,
                    "transfer_mode_count": spec.transfer_mode_count(ny),
                    "centered_covariance_convention": "G=2C-I",
                }
                for key, value in scalar_expected.items():
                    if _scalar(saved, key) != value:
                        raise RuntimeError(f"{result_path}: {key} mismatch")
                sample_indices = np.asarray(saved["sample_indices"], dtype=np.int64)
            declared = np.asarray(completion["sample_indices"], dtype=np.int64)
            if not np.array_equal(sample_indices, declared):
                raise RuntimeError(f"{result_path}: sample identity mismatch")
            for offset, sample_index in enumerate(sample_indices.tolist()):
                if sample_index in seen:
                    raise RuntimeError(f"Ny={ny}: duplicate sample {sample_index}")
                seen.add(sample_index)
                records.append(
                    SampleRecord(
                        ny=ny,
                        sample_index=int(sample_index),
                        sample_offset=offset,
                        result_path=result_path,
                        completion_path=completion_path,
                        result_sha256=result_sha,
                        construction=spec.construction,
                    )
                )
        if require_complete and seen != set(range(100)):
            missing = sorted(set(range(100)) - seen)
            raise RuntimeError(f"Ny={ny}: incomplete sample coverage; missing={missing}")
    return sorted(records, key=lambda row: (row.ny, row.sample_index))


def extract_active_covariance(
    full_g: np.ndarray,
    *,
    nx: int,
    ny: int,
    active_x: Sequence[int] = ACTIVE_X,
    require_product_exterior: bool = True,
    coupling_tolerance: float = NUMERICAL_TOL,
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract and validate the decoupled hard-wall active covariance."""
    full_g = np.asarray(full_g)
    full_n = 2 * nx * ny
    if full_g.dtype != np.complex128 or full_g.shape != (full_n, full_n):
        raise ValueError(f"full G must have shape {(full_n, full_n)} and dtype complex128")
    hermiticity = float(np.max(np.abs(full_g - full_g.conj().T)))
    if hermiticity > OCCUPATION_ROUNDOFF_TOL:
        raise FloatingPointError(f"full covariance Hermiticity residual {hermiticity:.3e}")
    full_g = 0.5 * (full_g + full_g.conj().T)
    active = active_indices(nx, ny, active_x)
    exterior = np.setdiff1d(np.arange(full_n, dtype=np.int64), active)
    coupling = float(
        np.max(np.abs(full_g[np.ix_(active, exterior)]), initial=0.0)
    )
    if coupling > coupling_tolerance:
        raise FloatingPointError(f"active/exterior coupling {coupling:.3e}")
    exterior_g = full_g[np.ix_(exterior, exterior)]
    if exterior.size:
        exterior_purity = float(
            np.max(
                np.abs(
                    exterior_g @ exterior_g
                    - np.eye(exterior.size, dtype=np.complex128)
                ),
                initial=0.0,
            )
        )
        exterior_y = exterior // (2 * nx)
        different_y = exterior_y[:, None] != exterior_y[None, :]
        exterior_inter_y = float(
            np.max(np.abs(exterior_g[different_y]), initial=0.0)
        )
    else:
        exterior_purity = 0.0
        exterior_inter_y = 0.0
    if require_product_exterior and exterior_purity > coupling_tolerance:
        raise FloatingPointError(
            f"exterior covariance is not a pure product sector: {exterior_purity:.3e}"
        )
    if require_product_exterior and exterior_inter_y > coupling_tolerance:
        raise FloatingPointError(
            f"exterior has inter-y correlations: {exterior_inter_y:.3e}"
        )
    active_g = full_g[np.ix_(active, active)]
    active_hermiticity = float(np.max(np.abs(active_g - active_g.conj().T)))
    return active_g, {
        "full_hermiticity_residual": hermiticity,
        "active_hermiticity_residual": active_hermiticity,
        "active_exterior_coupling": coupling,
        "exterior_purity_residual": exterior_purity,
        "exterior_inter_y_coupling": exterior_inter_y,
    }


def capped_log_polar(
    g: np.ndarray, cap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return capped ``A=arctanh(G)``, ``sign(G)``, and the raw spectrum."""
    matrix = np.asarray(g, dtype=np.complex128)
    matrix = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(matrix)
    bound = max(
        float(-1.0 - values.min(initial=-1.0)),
        float(values.max(initial=1.0) - 1.0),
        0.0,
    )
    if bound > OCCUPATION_ROUNDOFF_TOL:
        raise FloatingPointError(f"centered covariance leaves [-1,1] by {bound:.3e}")
    cap = float(cap)
    g_limit = float(np.tanh(cap))
    mapped = np.arctanh(np.clip(values, -g_limit, g_limit))
    mapped = np.clip(mapped, -cap, cap)
    polar = (vectors * mapped[None, :]) @ vectors.conj().T
    signs = np.where(values >= 0.0, 1.0, -1.0)
    sign_matrix = (vectors * signs[None, :]) @ vectors.conj().T
    polar = 0.5 * (polar + polar.conj().T)
    sign_matrix = 0.5 * (sign_matrix + sign_matrix.conj().T)
    interior = np.abs(values) < g_limit
    return polar, sign_matrix, values, {
        "occupation_bound_residual": bound / 2.0,
        "polar_hermiticity_residual": float(np.max(np.abs(polar - polar.conj().T))),
        "interior_mode_count": int(np.count_nonzero(interior)),
        "capped_mode_count": int(values.size - np.count_nonzero(interior)),
    }


def y_twirl_displacements(matrix: np.ndarray, ny: int) -> np.ndarray:
    """Twirl a y-major matrix and return blocks indexed by ``delta=y-y'``."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % int(ny):
        raise ValueError("matrix dimension must be square and divisible by Ny")
    ny = int(ny)
    block = matrix.shape[0] // ny
    tensor = matrix.reshape(ny, block, ny, block)
    y = np.arange(ny)
    displacements = np.empty((ny, block, block), dtype=np.complex128)
    for delta in range(ny):
        displacements[delta] = tensor[(y + delta) % ny, :, y, :].mean(axis=0)
    return displacements


def displacements_to_matrix(displacements: np.ndarray) -> np.ndarray:
    displacements = np.asarray(displacements, dtype=np.complex128)
    ny, block, block_2 = displacements.shape
    if block != block_2:
        raise ValueError("displacement blocks must be square")
    matrix = np.empty((ny, block, ny, block), dtype=np.complex128)
    for y in range(ny):
        for yp in range(ny):
            matrix[y, :, yp, :] = displacements[(y - yp) % ny]
    return matrix.reshape(ny * block, ny * block)


def displacement_hermiticity_residual(displacements: np.ndarray) -> float:
    matrix = displacements_to_matrix(displacements)
    return float(np.max(np.abs(matrix - matrix.conj().T)))


def displacement_translation_residual(displacements: np.ndarray) -> float:
    matrix = displacements_to_matrix(displacements)
    ny = int(displacements.shape[0])
    block = int(displacements.shape[1])
    tensor = matrix.reshape(ny, block, ny, block)
    shifted = np.roll(np.roll(tensor, 1, axis=0), 1, axis=2)
    return float(np.max(np.abs(tensor - shifted)))


def momentum_blocks(
    displacements: np.ndarray, *, twist: float = 0.0, centered: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a Hermitian circulant kernel at physical or twisted momenta.

    The Ny/2 displacement of an even ring is split equally between the two
    orientations.  This reproduces the discrete Fourier transform at zero
    twist while remaining Hermitian for the infinitesimal twist regulator.
    """
    disp = np.asarray(displacements, dtype=np.complex128)
    ny, block, block_2 = disp.shape
    if block != block_2:
        raise ValueError("displacement blocks must be square")
    momenta = (2.0 * np.pi * np.arange(ny, dtype=np.float64) + float(twist)) / ny
    blocks = np.empty((ny, block, block), dtype=np.complex128)
    for ki, momentum in enumerate(momenta):
        value = disp[0].copy()
        stop = (ny - 1) // 2
        for delta in range(1, stop + 1):
            value += np.exp(-1j * momentum * delta) * disp[delta]
            value += np.exp(1j * momentum * delta) * disp[delta].conj().T
        if ny % 2 == 0:
            half = 0.5 * (disp[ny // 2] + disp[ny // 2].conj().T)
            value += math.cos(momentum * ny / 2.0) * half
        blocks[ki] = 0.5 * (value + value.conj().T)
    if centered:
        wrapped = (momenta + np.pi) % (2.0 * np.pi) - np.pi
        order = np.argsort(wrapped)
        return wrapped[order], blocks[order]
    return momenta, blocks


def fourier_reconstruction_residual(displacements: np.ndarray) -> float:
    """Check exact physical-momentum Fourier reconstruction of the twirl."""
    _, blocks = momentum_blocks(displacements, twist=0.0, centered=False)
    reconstructed = np.fft.ifft(blocks, axis=0)
    return float(np.max(np.abs(reconstructed - displacements)))


def spectral_function_displacements(
    displacements: np.ndarray,
    function: str,
    *,
    cap: float = 10.0,
) -> np.ndarray:
    """Apply a Hermitian spectral function to a block-circulant matrix."""
    _, blocks = momentum_blocks(displacements, twist=0.0, centered=False)
    output = np.empty_like(blocks)
    for index, block in enumerate(blocks):
        values, vectors = np.linalg.eigh(block)
        if function == "arctanh":
            limit = float(np.tanh(cap))
            mapped = np.arctanh(np.clip(values, -limit, limit))
        elif function == "sign":
            mapped = np.where(values >= 0.0, 1.0, -1.0)
        else:
            raise ValueError(f"unknown spectral function {function!r}")
        output[index] = (vectors * mapped[None, :]) @ vectors.conj().T
    return np.fft.ifft(output, axis=0)


def _projectors_to_correlation(projectors: np.ndarray, momenta: np.ndarray) -> np.ndarray:
    projectors = np.asarray(projectors, dtype=np.complex128)
    momenta = np.asarray(momenta, dtype=np.float64)
    ny, block, _ = projectors.shape
    correlation = np.empty((ny, block, ny, block), dtype=np.complex128)
    for y in range(ny):
        for yp in range(ny):
            phase = np.exp(1j * momenta * (y - yp))
            correlation[y, :, yp, :] = np.einsum("k,kab->ab", phase, projectors) / ny
    result = correlation.reshape(ny * block, ny * block)
    return 0.5 * (result + result.conj().T)


def half_filled_ground_state(
    h_displacements: np.ndarray, *, twist: float = 1.0e-7
) -> tuple[np.ndarray, dict[str, float]]:
    """Fill the lowest half of a translation-invariant one-body Hamiltonian."""
    momenta, blocks = momentum_blocks(h_displacements, twist=twist, centered=False)
    ny, block, _ = blocks.shape
    values = np.empty((ny, block), dtype=np.float64)
    vectors = np.empty((ny, block, block), dtype=np.complex128)
    for ki, matrix in enumerate(blocks):
        values[ki], vectors[ki] = np.linalg.eigh(matrix)
    count = ny * block // 2
    order = np.argsort(values.reshape(-1), kind="stable")
    occupied = np.zeros(ny * block, dtype=bool)
    occupied[order[:count]] = True
    occupied = occupied.reshape(ny, block)
    projectors = np.empty_like(blocks)
    for ki in range(ny):
        frame = vectors[ki][:, occupied[ki]]
        projectors[ki] = frame @ frame.conj().T
    correlation = _projectors_to_correlation(projectors, momenta)
    identity_error = float(np.max(np.abs(correlation @ correlation - correlation)))
    charge_error = abs(float(np.trace(correlation).real) - count)
    return correlation, {
        "projector_idempotency_residual": identity_error,
        "half_filling_charge_residual": charge_error,
        "occupied_modes": int(count),
        "fermi_energy_lower": float(values.reshape(-1)[order[count - 1]]),
        "fermi_energy_upper": float(values.reshape(-1)[order[count]]),
        "fermi_gap": float(values.reshape(-1)[order[count]] - values.reshape(-1)[order[count - 1]]),
    }


def strip_indices(width: int, ny: int, ay: int, y0: int = 0) -> np.ndarray:
    if ay < 1 or ay > ny:
        raise ValueError("Ay must lie in [1, Ny]")
    ys = (np.arange(ay, dtype=np.int64) + int(y0)) % ny
    return np.concatenate([np.arange(y * width, (y + 1) * width, dtype=np.int64) for y in ys])


def renyi_entropy(eigenvalues: np.ndarray, order: int) -> float:
    values = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), 0.0, 1.0)
    if int(order) == 1:
        mask = (values > 0.0) & (values < 1.0)
        v = values[mask]
        return float(-np.sum(v * np.log(v) + (1.0 - v) * np.log1p(-v)))
    order = int(order)
    tiny = np.finfo(np.float64).tiny
    left = order * np.log(np.clip(values, tiny, 1.0))
    right = order * np.log(np.clip(1.0 - values, tiny, 1.0))
    return float(np.sum(np.logaddexp(left, right)) / (1.0 - order))


def entropy_profiles(
    correlation: np.ndarray,
    *,
    width: int,
    ny: int,
    orders: Sequence[int] = (1, 2, 3),
    average_origins: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return S_q(Ay) for Ay=1,...,Ny/2."""
    correlation = np.asarray(correlation, dtype=np.complex128)
    expected = width * ny
    if correlation.shape != (expected, expected):
        raise ValueError(f"correlation shape must be {(expected, expected)}")
    ay_values = np.arange(1, ny // 2 + 1, dtype=np.int64)
    result = np.empty((len(orders), ay_values.size), dtype=np.float64)
    max_origin_spread = 0.0
    for ai, ay in enumerate(ay_values):
        origins = range(ny) if average_origins else (0,)
        values = []
        for y0 in origins:
            indices = strip_indices(width, ny, int(ay), int(y0))
            eigenvalues = np.linalg.eigvalsh(correlation[np.ix_(indices, indices)])
            values.append([renyi_entropy(eigenvalues, order) for order in orders])
        values_array = np.asarray(values, dtype=np.float64)
        result[:, ai] = values_array.mean(axis=0)
        max_origin_spread = max(
            max_origin_spread,
            float(np.max(np.ptp(values_array, axis=0), initial=0.0)),
        )
    return result, {"entropy_origin_spread": max_origin_spread}


def log_chord(ny: int, ay_values: np.ndarray) -> np.ndarray:
    ay = np.asarray(ay_values, dtype=np.float64)
    return np.log((float(ny) / np.pi) * np.sin(np.pi * ay / float(ny)))


def fit_entropy_curve(
    entropy: np.ndarray,
    *,
    ny: int,
    order: int,
    ay_min: int = 8,
    drop_endpoint: bool = False,
) -> dict[str, float]:
    ay = np.arange(1, ny // 2 + 1, dtype=np.int64)
    upper = ny // 2 - (1 if drop_endpoint else 0)
    mask = (ay >= int(ay_min)) & (ay <= upper)
    if np.count_nonzero(mask) < 3:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "r_squared": math.nan,
            "c_per_wall": math.nan,
            "n_points": int(np.count_nonzero(mask)),
        }
    x = log_chord(ny, ay[mask])
    y = np.asarray(entropy, dtype=np.float64)[mask]
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ np.asarray([slope, intercept])
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    c_per_wall = 6.0 * float(slope) / (1.0 + 1.0 / int(order))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "c_per_wall": c_per_wall,
        "n_points": int(np.count_nonzero(mask)),
    }


def _wall_columns_local(
    active_x: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    global_x = np.asarray(active_x, dtype=np.int64)
    return tuple(
        np.flatnonzero(np.isin(global_x, np.asarray(window, dtype=np.int64)))
        for window in WALL_WINDOWS
    )  # type: ignore[return-value]


def _track_low_energy_modes(
    blocks: np.ndarray, low_mode_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Track the modes nearest zero continuously by eigenvector overlap."""
    blocks = np.asarray(blocks, dtype=np.complex128)
    nk, dimension, _ = blocks.shape
    if not 1 <= int(low_mode_count) <= dimension:
        raise ValueError("low_mode_count must lie between one and the block dimension")
    all_values = np.empty((nk, dimension), dtype=np.float64)
    all_vectors = np.empty((nk, dimension, dimension), dtype=np.complex128)
    for ki, block in enumerate(blocks):
        all_values[ki], all_vectors[ki] = np.linalg.eigh(block)
    # ``momentum_blocks(..., centered=True)`` sorts k from -pi to pi.  Starting
    # at k=0 makes "nearest zero" unambiguous before propagating the labels in
    # both directions through avoided and exact crossings.
    start = nk // 2
    initial = np.argsort(np.abs(all_values[start]), kind="stable")[:low_mode_count]
    initial = initial[np.argsort(all_values[start, initial], kind="stable")]
    tracked_values = np.empty((nk, low_mode_count), dtype=np.float64)
    tracked_vectors = np.empty((nk, dimension, low_mode_count), dtype=np.complex128)
    tracked_values[start] = all_values[start, initial]
    tracked_vectors[start] = all_vectors[start][:, initial]

    def propagate(indices: Sequence[int], previous_index: int) -> None:
        previous = tracked_vectors[previous_index]
        for ki in indices:
            overlap = np.abs(previous.conj().T @ all_vectors[ki]) ** 2
            rows, columns = linear_sum_assignment(-overlap)
            chosen = np.empty(low_mode_count, dtype=np.int64)
            chosen[rows] = columns
            current = all_vectors[ki][:, chosen]
            # Fix arbitrary eigenvector phases to make cached tracked modes
            # deterministic without changing any spectrum or projector.
            phase = np.einsum("ij,ij->j", previous.conj(), current)
            nonzero = np.abs(phase) > 0
            current[:, nonzero] *= (phase[nonzero] / np.abs(phase[nonzero])).conj()
            tracked_values[ki] = all_values[ki, chosen]
            tracked_vectors[ki] = current
            previous = current

    propagate(range(start + 1, nk), start)
    propagate(range(start - 1, -1, -1), start)
    return tracked_values, tracked_vectors


def _localize_wall_doublet(
    h: np.ndarray,
    tracked_values: np.ndarray,
    tracked_vectors: np.ndarray,
    active_x: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chosen = np.argsort(np.abs(tracked_values), kind="stable")[:2]
    basis = tracked_vectors[:, chosen]
    x_coordinates = np.repeat(np.asarray(active_x, dtype=np.float64), 2)
    projected_x = basis.conj().T @ (x_coordinates[:, None] * basis)
    _, rotation = np.linalg.eigh(0.5 * (projected_x + projected_x.conj().T))
    localized = basis @ rotation
    centers = np.real(np.einsum("ij,i,ij->j", localized.conj(), x_coordinates, localized))
    energies = np.real(np.diag(localized.conj().T @ h @ localized))
    cost = np.abs(centers[None, :] - np.asarray(WALL_X, dtype=np.float64)[:, None])
    rows, cols = linear_sum_assignment(cost)
    assignment = np.empty(2, dtype=np.int64)
    assignment[rows] = cols
    localized = localized[:, assignment]
    centers = centers[assignment]
    energies = energies[assignment]
    profile = np.abs(localized.reshape(len(active_x), 2, 2)) ** 2
    profile = profile.sum(axis=1).T
    windows = _wall_columns_local(active_x)
    weights = np.asarray([profile[wall, windows[wall]].sum() for wall in range(2)])
    return energies, centers, weights, profile


def band_observables(
    h_displacements: np.ndarray,
    *,
    low_mode_count: int = 6,
    velocity_points: int = 5,
    wall_weight_min: float = 0.5,
    active_x: Sequence[int] = ACTIVE_X,
) -> dict[str, np.ndarray]:
    """Return low bands, wall-localized branches, and local crossing fits."""
    ky, blocks = momentum_blocks(h_displacements, centered=True)
    nk, dimension, _ = blocks.shape
    low_energy, low_vectors = _track_low_energy_modes(blocks, low_mode_count)
    wall_energy = np.empty((nk, 2), dtype=np.float64)
    wall_center = np.empty((nk, 2), dtype=np.float64)
    wall_weight = np.empty((nk, 2), dtype=np.float64)
    wall_profile = np.empty((nk, 2, len(active_x)), dtype=np.float64)
    for ki, block in enumerate(blocks):
        energy, center, weight, profile = _localize_wall_doublet(
            block, low_energy[ki], low_vectors[ki], active_x
        )
        wall_energy[ki] = energy
        wall_center[ki] = center
        wall_weight[ki] = weight
        wall_profile[ki] = profile
    crossing = np.full(2, np.nan, dtype=np.float64)
    velocity = np.full(2, np.nan, dtype=np.float64)
    velocity_r2 = np.full(2, np.nan, dtype=np.float64)
    for wall in range(2):
        eligible = wall_weight[:, wall] >= float(wall_weight_min)
        candidates = np.flatnonzero(eligible)
        if candidates.size < 3:
            candidates = np.arange(nk)
        center_index = int(candidates[np.argmin(np.abs(wall_energy[candidates, wall]))])
        delta = (ky - ky[center_index] + np.pi) % (2.0 * np.pi) - np.pi
        allowed = np.flatnonzero(eligible) if np.count_nonzero(eligible) >= 3 else np.arange(nk)
        chosen = allowed[np.argsort(np.abs(delta[allowed]))[: min(velocity_points, allowed.size)]]
        if chosen.size < 3:
            continue
        design = np.column_stack((delta[chosen], np.ones(chosen.size)))
        slope, intercept = np.linalg.lstsq(design, wall_energy[chosen, wall], rcond=None)[0]
        prediction = design @ np.asarray([slope, intercept])
        residual = float(np.sum((wall_energy[chosen, wall] - prediction) ** 2))
        total = float(np.sum((wall_energy[chosen, wall] - wall_energy[chosen, wall].mean()) ** 2))
        crossing[wall] = (ky[center_index] - intercept / slope + np.pi) % (2 * np.pi) - np.pi if slope else math.nan
        velocity[wall] = slope
        velocity_r2[wall] = 1.0 - residual / total if total > 0 else 1.0
    return {
        "ky": ky,
        "low_energy": low_energy,
        "wall_energy": wall_energy,
        "wall_center": wall_center,
        "wall_weight": wall_weight,
        "wall_profile": wall_profile,
        "crossing": crossing,
        "velocity": velocity,
        "velocity_r_squared": velocity_r2,
        "block_dimension": np.asarray(dimension, dtype=np.int64),
    }


def _cache_paths(cache_root: Path, record: SampleRecord) -> tuple[Path, Path]:
    directory = Path(cache_root) / f"Ny{record.ny:03d}"
    stem = f"sample_{record.sample_index:03d}"
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def _atomic_save_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_is_valid(cache_root: Path, record: SampleRecord, config: AnalysisConfig) -> bool:
    npz_path, json_path = _cache_paths(cache_root, record)
    if not npz_path.is_file() or not json_path.is_file():
        return False
    try:
        completion = json.loads(json_path.read_text(encoding="utf-8"))
        expected = {
            "schema": CACHE_COMPLETION_SCHEMA,
            "analysis_config_hash": config.identity_hash,
            "construction": config.construction,
            "source_result_sha256": record.result_sha256,
            "Ny": record.ny,
            "sample_index": record.sample_index,
            "result_filename": npz_path.name,
            "result_bytes": npz_path.stat().st_size,
        }
        if any(completion.get(key) != value for key, value in expected.items()):
            return False
        if sha256_file(npz_path) != completion.get("result_sha256"):
            return False
        with np.load(npz_path, allow_pickle=False) as saved:
            return (
                _scalar(saved, "cache_schema") == CACHE_SCHEMA
                and _scalar(saved, "analysis_config_hash") == config.identity_hash
                and _scalar(saved, "construction") == config.construction
                and int(_scalar(saved, "sample_index")) == record.sample_index
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def analyze_sample_covariance(
    full_g: np.ndarray,
    record: SampleRecord,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute all compact per-sample products from one saved endpoint."""
    spec = geometry_spec(config.construction)
    if record.construction != spec.construction:
        raise ValueError(
            f"record construction {record.construction!r} does not match analysis "
            f"construction {spec.construction!r}"
        )
    active_g, base_diagnostics = extract_active_covariance(
        full_g,
        nx=EXPECTED_NX,
        ny=record.ny,
        active_x=spec.active_x,
        require_product_exterior=spec.require_product_exterior,
    )
    width = 2 * len(spec.active_x)
    active_c = 0.5 * (active_g + np.eye(active_g.shape[0], dtype=np.complex128))
    endpoint_entropy, endpoint_diag = entropy_profiles(
        active_c,
        width=width,
        ny=record.ny,
        orders=config.renyi_orders,
        average_origins=True,
    )
    g_disp = y_twirl_displacements(active_g, record.ny)
    base_diagnostics.update(
        {
            "g_twirl_hermiticity_residual": displacement_hermiticity_residual(g_disp),
            "g_twirl_translation_residual": displacement_translation_residual(g_disp),
            "g_fourier_reconstruction_residual": fourier_reconstruction_residual(g_disp),
        }
    )
    base_diagnostics["endpoint_origin_variation"] = endpoint_diag[
        "entropy_origin_spread"
    ]
    cap_products = []
    sign_disp = None
    spectra = None
    cap_diagnostics = []
    typical_entropy = np.empty(
        (len(config.caps), len(config.twists), len(config.renyi_orders), record.ny // 2),
        dtype=np.float64,
    )
    ground_diagnostics: list[list[dict[str, float]]] = []
    for cap_index, cap in enumerate(config.caps):
        polar, sign_matrix, values, diagnostics = capped_log_polar(active_g, cap)
        if sign_disp is None:
            sign_disp = y_twirl_displacements(sign_matrix, record.ny)
            spectra = values
        cap_products.append(y_twirl_displacements(polar, record.ny))
        diagnostics.update(
            {
                "a_twirl_hermiticity_residual": displacement_hermiticity_residual(cap_products[-1]),
                "a_twirl_translation_residual": displacement_translation_residual(cap_products[-1]),
                "a_fourier_reconstruction_residual": fourier_reconstruction_residual(cap_products[-1]),
            }
        )
        cap_diagnostics.append(diagnostics)
        twist_diagnostics = []
        for twist_index, twist in enumerate(config.twists):
            correlation, ground_diag = half_filled_ground_state(
                -2.0 * cap_products[-1], twist=twist
            )
            entropy, entropy_diag = entropy_profiles(
                correlation,
                width=width,
                ny=record.ny,
                orders=config.renyi_orders,
                average_origins=False,
            )
            # Translation invariance is checked explicitly on three additional origins.
            check_origins = (0, 1, record.ny // 3, record.ny - 1)
            origin_spread = 0.0
            for ay in range(1, record.ny // 2 + 1):
                values_by_origin = []
                for y0 in check_origins:
                    indices = strip_indices(width, record.ny, ay, y0)
                    eigenvalues = np.linalg.eigvalsh(correlation[np.ix_(indices, indices)])
                    values_by_origin.append(
                        [renyi_entropy(eigenvalues, order) for order in config.renyi_orders]
                    )
                origin_spread = max(origin_spread, float(np.max(np.ptp(values_by_origin, axis=0))))
            ground_diag.update(entropy_diag)
            ground_diag["entropy_origin_spread"] = origin_spread
            typical_entropy[cap_index, twist_index] = entropy
            twist_diagnostics.append(ground_diag)
        ground_diagnostics.append(twist_diagnostics)
    assert sign_disp is not None and spectra is not None
    payload: dict[str, Any] = {
        "cache_schema": np.asarray(CACHE_SCHEMA),
        "analysis_config_hash": np.asarray(config.identity_hash),
        "construction": np.asarray(config.construction),
        "source_result_sha256": np.asarray(record.result_sha256),
        "Ny": np.asarray(record.ny, dtype=np.int64),
        "sample_index": np.asarray(record.sample_index, dtype=np.int64),
        "caps": np.asarray(config.caps, dtype=np.float64),
        "twists": np.asarray(config.twists, dtype=np.float64),
        "renyi_orders": np.asarray(config.renyi_orders, dtype=np.int64),
        "active_x": np.asarray(spec.active_x, dtype=np.int64),
        "g_displacements": g_disp,
        "sign_g_displacements": sign_disp,
        "a_displacements": np.asarray(cap_products),
        "active_g_eigenvalues": spectra,
        "endpoint_entropy": endpoint_entropy,
        "typical_ground_entropy": typical_entropy,
        "base_diagnostics_json": np.asarray(json.dumps(base_diagnostics, sort_keys=True)),
        "cap_diagnostics_json": np.asarray(json.dumps(cap_diagnostics, sort_keys=True)),
        "ground_diagnostics_json": np.asarray(json.dumps(ground_diagnostics, sort_keys=True)),
    }
    return payload


def write_sample_cache(
    cache_root: Path,
    record: SampleRecord,
    config: AnalysisConfig,
    payload: dict[str, Any],
) -> Path:
    npz_path, json_path = _cache_paths(cache_root, record)
    _atomic_save_npz(npz_path, payload)
    digest = sha256_file(npz_path)
    _atomic_write_json(
        json_path,
        {
            "schema": CACHE_COMPLETION_SCHEMA,
            "analysis_config_hash": config.identity_hash,
            "construction": config.construction,
            "source_result_sha256": record.result_sha256,
            "Ny": record.ny,
            "sample_index": record.sample_index,
            "result_filename": npz_path.name,
            "result_bytes": npz_path.stat().st_size,
            "result_sha256": digest,
        },
    )
    return npz_path


def build_sample_caches(
    records: Sequence[SampleRecord],
    *,
    cache_root: Path,
    config: AnalysisConfig,
    progress: Any = None,
    max_samples_per_ny: int | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Build resumable one-trajectory caches, opening each source shard once."""
    selected: list[SampleRecord] = []
    counts: dict[int, int] = {}
    for record in records:
        used = counts.get(record.ny, 0)
        if max_samples_per_ny is None or used < int(max_samples_per_ny):
            selected.append(record)
            counts[record.ny] = used + 1
    grouped: dict[Path, list[SampleRecord]] = {}
    for record in selected:
        grouped.setdefault(record.result_path, []).append(record)
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    skipped = completed = 0
    bar = (
        progress(total=len(selected), desc="log-polar trajectories", unit="trajectory")
        if progress is not None
        else None
    )
    for result_path, shard_records in grouped.items():
        pending = [row for row in shard_records if not cache_is_valid(cache_root, row, config)]
        skipped += len(shard_records) - len(pending)
        if bar is not None:
            bar.update(len(shard_records) - len(pending))
        if not pending:
            continue
        with np.load(result_path, allow_pickle=False) as saved:
            final = np.asarray(saved["G_final"])
            if final.dtype != np.complex128:
                raise RuntimeError(f"{result_path}: G_final is not complex128")
            if workers == 1:
                products = [
                    (record, analyze_sample_covariance(final[record.sample_offset], record, config))
                    for record in pending
                ]
            else:
                with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
                    futures = [
                        (
                            record,
                            executor.submit(
                                analyze_sample_covariance,
                                final[record.sample_offset],
                                record,
                                config,
                            ),
                        )
                        for record in pending
                    ]
                    products = [(record, future.result()) for record, future in futures]
            for record, payload in products:
                write_sample_cache(cache_root, record, config, payload)
                completed += 1
                if bar is not None:
                    bar.update(1)
    if bar is not None:
        bar.close()
    return {"selected": len(selected), "completed": completed, "skipped": skipped}


def load_sample_cache(cache_root: Path, record: SampleRecord, config: AnalysisConfig) -> dict[str, Any]:
    if not cache_is_valid(cache_root, record, config):
        raise RuntimeError(f"invalid or missing cache for {record.task_id}")
    npz_path, _ = _cache_paths(cache_root, record)
    with np.load(npz_path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _mean_and_interval(values: np.ndarray, rng: np.random.Generator, replicates: int) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    if values.shape[0] == 1:
        return {"mean": mean, "sem": np.zeros_like(mean), "low": mean, "high": mean}
    sem = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    draws = np.empty((replicates,) + mean.shape, dtype=np.float64)
    for draw in range(replicates):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        draws[draw] = values[indices].mean(axis=0)
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"mean": mean, "sem": sem, "low": low, "high": high}


def _fit_variants(curve: np.ndarray, *, ny: int, order: int, config: AnalysisConfig) -> list[tuple[str, dict[str, float]]]:
    return [
        ("primary", fit_entropy_curve(curve, ny=ny, order=order, ay_min=config.fit_ay_min)),
        ("aymin6", fit_entropy_curve(curve, ny=ny, order=order, ay_min=config.sensitivity_ay_min)),
        (
            "drop_endpoint",
            fit_entropy_curve(
                curve, ny=ny, order=order, ay_min=config.fit_ay_min, drop_endpoint=True
            ),
        ),
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    seen = set(fields)
    for row in rows[1:]:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_analysis(
    records: Sequence[SampleRecord],
    *,
    cache_root: Path,
    output_root: Path,
    config: AnalysisConfig,
    max_samples_per_ny: int | None = None,
) -> dict[str, Any]:
    """Aggregate cached trajectories, write tables, and return notebook-ready arrays."""
    spec = geometry_spec(config.construction)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected: dict[int, list[tuple[SampleRecord, dict[str, Any]]]] = {}
    for record in records:
        if record.construction != spec.construction:
            raise ValueError(
                f"cannot pool {record.construction!r} into {spec.construction!r} analysis"
            )
        rows = selected.setdefault(record.ny, [])
        if max_samples_per_ny is None or len(rows) < int(max_samples_per_ny):
            rows.append((record, load_sample_cache(cache_root, record, config)))
    rng = np.random.default_rng(config.bootstrap_seed)
    spectrum_rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    notebook_arrays: dict[str, Any] = {}

    for ny, items in sorted(selected.items()):
        samples = len(items)
        if samples == 0:
            continue
        width = 2 * len(spec.active_x)
        caps = np.asarray(items[0][1]["caps"], dtype=np.float64)
        twists = np.asarray(items[0][1]["twists"], dtype=np.float64)
        orders = np.asarray(items[0][1]["renyi_orders"], dtype=np.int64)
        endpoint = np.stack([row["endpoint_entropy"] for _, row in items])
        typical = np.stack([row["typical_ground_entropy"] for _, row in items])
        a_all = np.stack([row["a_displacements"] for _, row in items])
        g_all = np.stack([row["g_displacements"] for _, row in items])
        sign_all = np.stack([row["sign_g_displacements"] for _, row in items])

        for order_index, order in enumerate(orders.tolist()):
            stats = _mean_and_interval(endpoint[:, order_index], rng, config.bootstrap_replicates)
            for ay_index in range(ny // 2):
                entropy_rows.append(
                    {
                        "family": "actual_endpoint",
                        "Ny": ny,
                        "samples": samples,
                        "cap": "",
                        "twist": "",
                        "renyi_order": order,
                        "Ay": ay_index + 1,
                        "mean": stats["mean"][ay_index],
                        "sem": stats["sem"][ay_index],
                        "ci_low": stats["low"][ay_index],
                        "ci_high": stats["high"][ay_index],
                    }
                )
            for variant, fit in _fit_variants(stats["mean"], ny=ny, order=order, config=config):
                if variant == "primary":
                    sample_variant = [
                        fit_entropy_curve(curve, ny=ny, order=order, ay_min=config.fit_ay_min)
                        for curve in endpoint[:, order_index]
                    ]
                elif variant == "aymin6":
                    sample_variant = [
                        fit_entropy_curve(curve, ny=ny, order=order, ay_min=config.sensitivity_ay_min)
                        for curve in endpoint[:, order_index]
                    ]
                else:
                    sample_variant = [
                        fit_entropy_curve(
                            curve,
                            ny=ny,
                            order=order,
                            ay_min=config.fit_ay_min,
                            drop_endpoint=True,
                        )
                        for curve in endpoint[:, order_index]
                    ]
                sample_fits = np.asarray(
                    [row["c_per_wall"] for row in sample_variant], dtype=np.float64
                )
                finite_fits = sample_fits[np.isfinite(sample_fits)]
                fit_stats = (
                    _mean_and_interval(
                        finite_fits[:, None], rng, config.bootstrap_replicates
                    )
                    if finite_fits.size
                    else None
                )
                fit_rows.append(
                    {
                        "family": "actual_endpoint",
                        "Ny": ny,
                        "samples": samples,
                        "cap": "",
                        "twist": "",
                        "renyi_order": order,
                        "fit_variant": variant,
                        **fit,
                        "ensemble_c_mean": "" if fit_stats is None else float(fit_stats["mean"][0]),
                        "ensemble_c_ci_low": "" if fit_stats is None else float(fit_stats["low"][0]),
                        "ensemble_c_ci_high": "" if fit_stats is None else float(fit_stats["high"][0]),
                    }
                )

        mean_g_disp = g_all.mean(axis=0)
        mean_sign_disp = sign_all.mean(axis=0)
        cap_index_default = int(np.flatnonzero(np.isclose(caps, config.default_cap))[0])
        for cap_index, cap in enumerate(caps.tolist()):
            mean_a_disp = a_all[:, cap_index].mean(axis=0)
            quenched_h_disp = -2.0 * mean_a_disp
            quenched_band = band_observables(
                quenched_h_disp,
                wall_weight_min=config.wall_weight_min,
                active_x=spec.active_x,
            )
            key = f"Ny{ny:03d}_cap{cap:g}"
            notebook_arrays[f"{key}_quenched_ky"] = quenched_band["ky"]
            notebook_arrays[f"{key}_quenched_low_energy"] = quenched_band["low_energy"]
            notebook_arrays[f"{key}_quenched_wall_energy"] = quenched_band["wall_energy"]
            notebook_arrays[f"{key}_quenched_wall_weight"] = quenched_band["wall_weight"]
            notebook_arrays[f"{key}_quenched_velocity"] = quenched_band["velocity"]
            notebook_arrays[f"{key}_quenched_crossing"] = quenched_band["crossing"]
            for wall in range(2):
                crossing_index = int(
                    np.argmin(np.abs(quenched_band["wall_energy"][:, wall]))
                )
                velocity_rows.append(
                    {
                        "method": "quenched_mean",
                        "Ny": ny,
                        "samples": samples,
                        "cap": cap,
                        "wall": wall,
                        "crossing": quenched_band["crossing"][wall],
                        "velocity": quenched_band["velocity"][wall],
                        "velocity_rate": quenched_band["velocity"][wall] / (4 * ny),
                        "velocity_r_squared": quenched_band["velocity_r_squared"][wall],
                        "crossing_wall_weight": quenched_band["wall_weight"][crossing_index, wall],
                        "velocity_ci_low": "",
                        "velocity_ci_high": "",
                    }
                )
            for ki, momentum in enumerate(quenched_band["ky"]):
                for band in range(quenched_band["low_energy"].shape[1]):
                    spectrum_rows.append(
                        {
                            "method": "quenched_mean_low",
                            "Ny": ny,
                            "samples": samples,
                            "cap": cap,
                            "ky": momentum,
                            "band": band,
                            "wall": "",
                            "energy": quenched_band["low_energy"][ki, band],
                            "rate": quenched_band["low_energy"][ki, band] / (4 * ny),
                            "wall_weight": "",
                            "wall_center": "",
                        }
                    )
                for wall in range(2):
                    spectrum_rows.append(
                        {
                            "method": "quenched_mean_wall",
                            "Ny": ny,
                            "samples": samples,
                            "cap": cap,
                            "ky": momentum,
                            "band": "",
                            "wall": wall,
                            "energy": quenched_band["wall_energy"][ki, wall],
                            "rate": quenched_band["wall_energy"][ki, wall] / (4 * ny),
                            "wall_weight": quenched_band["wall_weight"][ki, wall],
                            "wall_center": quenched_band["wall_center"][ki, wall],
                        }
                    )

            sample_bands = [
                band_observables(
                    -2.0 * a_all[sample, cap_index],
                    wall_weight_min=config.wall_weight_min,
                    active_x=spec.active_x,
                )
                for sample in range(samples)
            ]
            typical_wall_energy = np.stack([row["wall_energy"] for row in sample_bands])
            typical_wall_weight = np.stack([row["wall_weight"] for row in sample_bands])
            typical_velocity = np.stack([row["velocity"] for row in sample_bands])
            notebook_arrays[f"{key}_typical_wall_energy_mean"] = typical_wall_energy.mean(axis=0)
            notebook_arrays[f"{key}_typical_wall_energy_sem"] = typical_wall_energy.std(axis=0, ddof=1) / math.sqrt(samples) if samples > 1 else np.zeros_like(typical_wall_energy[0])
            notebook_arrays[f"{key}_typical_velocity"] = typical_velocity
            for wall in range(2):
                finite_velocity = typical_velocity[:, wall][
                    np.isfinite(typical_velocity[:, wall])
                ]
                velocity_stats = (
                    _mean_and_interval(
                        finite_velocity[:, None], rng, config.bootstrap_replicates
                    )
                    if finite_velocity.size
                    else None
                )
                crossing_values = np.asarray(
                    [row["crossing"][wall] for row in sample_bands], dtype=np.float64
                )
                crossing_weights = np.asarray(
                    [
                        row["wall_weight"][
                            int(np.argmin(np.abs(row["wall_energy"][:, wall]))), wall
                        ]
                        for row in sample_bands
                    ],
                    dtype=np.float64,
                )
                velocity_rows.append(
                    {
                        "method": "typical_sample",
                        "Ny": ny,
                        "samples": samples,
                        "cap": cap,
                        "wall": wall,
                        "crossing": float(np.nanmedian(crossing_values)),
                        "velocity": "" if velocity_stats is None else float(velocity_stats["mean"][0]),
                        "velocity_rate": "" if velocity_stats is None else float(velocity_stats["mean"][0]) / (4 * ny),
                        "velocity_r_squared": "",
                        "crossing_wall_weight": float(np.nanmean(crossing_weights)),
                        "velocity_ci_low": "" if velocity_stats is None else float(velocity_stats["low"][0]),
                        "velocity_ci_high": "" if velocity_stats is None else float(velocity_stats["high"][0]),
                    }
                )
            for ki, momentum in enumerate(sample_bands[0]["ky"]):
                for wall in range(2):
                    spectrum_rows.append(
                        {
                            "method": "typical_wall_mean",
                            "Ny": ny,
                            "samples": samples,
                            "cap": cap,
                            "ky": momentum,
                            "band": "",
                            "wall": wall,
                            "energy": typical_wall_energy[:, ki, wall].mean(),
                            "rate": typical_wall_energy[:, ki, wall].mean() / (4 * ny),
                            "wall_weight": typical_wall_weight[:, ki, wall].mean(),
                            "wall_center": "",
                        }
                    )

            for twist_index, twist in enumerate(twists.tolist()):
                mean_correlation, mean_ground_diag = half_filled_ground_state(
                    quenched_h_disp, twist=twist
                )
                mean_entropy, mean_entropy_diag = entropy_profiles(
                    mean_correlation,
                    width=width,
                    ny=ny,
                    orders=orders,
                    average_origins=False,
                )
                mean_ground_diag.update(mean_entropy_diag)
                diagnostic_rows.append(
                    {
                        "family": "quenched_mean",
                        "Ny": ny,
                        "sample_index": "",
                        "cap": cap,
                        "twist": twist,
                        **mean_ground_diag,
                    }
                )
                for order_index, order in enumerate(orders.tolist()):
                    typical_stats = _mean_and_interval(
                        typical[:, cap_index, twist_index, order_index],
                        rng,
                        config.bootstrap_replicates,
                    )
                    for family, curve, stats in (
                        ("quenched_mean", mean_entropy[order_index], None),
                        ("typical_sample", typical_stats["mean"], typical_stats),
                    ):
                        for ay_index in range(ny // 2):
                            entropy_rows.append(
                                {
                                    "family": family,
                                    "Ny": ny,
                                    "samples": samples,
                                    "cap": cap,
                                    "twist": twist,
                                    "renyi_order": order,
                                    "Ay": ay_index + 1,
                                    "mean": curve[ay_index],
                                    "sem": 0.0 if stats is None else stats["sem"][ay_index],
                                    "ci_low": curve[ay_index] if stats is None else stats["low"][ay_index],
                                    "ci_high": curve[ay_index] if stats is None else stats["high"][ay_index],
                                }
                            )
                        for variant, fit in _fit_variants(curve, ny=ny, order=order, config=config):
                            fit_stats = None
                            if family == "typical_sample":
                                per_sample = []
                                for sample_curve in typical[:, cap_index, twist_index, order_index]:
                                    if variant == "primary":
                                        sample_fit = fit_entropy_curve(
                                            sample_curve,
                                            ny=ny,
                                            order=order,
                                            ay_min=config.fit_ay_min,
                                        )
                                    elif variant == "aymin6":
                                        sample_fit = fit_entropy_curve(
                                            sample_curve,
                                            ny=ny,
                                            order=order,
                                            ay_min=config.sensitivity_ay_min,
                                        )
                                    else:
                                        sample_fit = fit_entropy_curve(
                                            sample_curve,
                                            ny=ny,
                                            order=order,
                                            ay_min=config.fit_ay_min,
                                            drop_endpoint=True,
                                        )
                                    if np.isfinite(sample_fit["c_per_wall"]):
                                        per_sample.append(sample_fit["c_per_wall"])
                                if per_sample:
                                    fit_stats = _mean_and_interval(
                                        np.asarray(per_sample)[:, None],
                                        rng,
                                        config.bootstrap_replicates,
                                    )
                            fit_rows.append(
                                {
                                    "family": family,
                                    "Ny": ny,
                                    "samples": samples,
                                    "cap": cap,
                                    "twist": twist,
                                    "renyi_order": order,
                                    "fit_variant": variant,
                                    **fit,
                                    "ensemble_c_mean": "" if fit_stats is None else float(fit_stats["mean"][0]),
                                    "ensemble_c_ci_low": "" if fit_stats is None else float(fit_stats["low"][0]),
                                    "ensemble_c_ci_high": "" if fit_stats is None else float(fit_stats["high"][0]),
                                }
                            )

        # Mean-operator jackknife at the default cap.  Curves are retained for uncertainty.
        group_ids = np.asarray([record.sample_index % config.jackknife_groups for record, _ in items])
        jackknife_curves = []
        for group in range(config.jackknife_groups):
            keep = group_ids != group
            if not np.any(keep):
                continue
            h_disp = -2.0 * a_all[keep, cap_index_default].mean(axis=0)
            correlation, _ = half_filled_ground_state(h_disp, twist=float(twists[-1]))
            curve, _ = entropy_profiles(
                correlation, width=width, ny=ny, orders=orders, average_origins=False
            )
            jackknife_curves.append(curve)
        jackknife_curves_array = np.asarray(jackknife_curves)
        notebook_arrays[f"Ny{ny:03d}_quenched_jackknife_entropy"] = jackknife_curves_array
        if jackknife_curves_array.size:
            group_count = jackknife_curves_array.shape[0]
            for order_index, order in enumerate(orders.tolist()):
                for ay_index in range(ny // 2):
                    values = jackknife_curves_array[:, order_index, ay_index]
                    estimate = next(
                        row for row in entropy_rows
                        if row["family"] == "quenched_mean"
                        and row["Ny"] == ny
                        and np.isclose(float(row["cap"]), config.default_cap)
                        and np.isclose(float(row["twist"]), float(twists[-1]))
                        and row["renyi_order"] == order
                        and row["Ay"] == ay_index + 1
                    )
                    standard_error = math.sqrt(
                        (group_count - 1) / group_count
                        * float(np.sum((values - values.mean()) ** 2))
                    )
                    estimate["sem"] = standard_error
                    estimate["ci_low"] = float(estimate["mean"]) - 1.96 * standard_error
                    estimate["ci_high"] = float(estimate["mean"]) + 1.96 * standard_error
                for variant in ("primary", "aymin6", "drop_endpoint"):
                    jackknife_fits = []
                    for sample_curve in jackknife_curves_array[:, order_index]:
                        if variant == "primary":
                            fit = fit_entropy_curve(
                                sample_curve, ny=ny, order=order, ay_min=config.fit_ay_min
                            )
                        elif variant == "aymin6":
                            fit = fit_entropy_curve(
                                sample_curve,
                                ny=ny,
                                order=order,
                                ay_min=config.sensitivity_ay_min,
                            )
                        else:
                            fit = fit_entropy_curve(
                                sample_curve,
                                ny=ny,
                                order=order,
                                ay_min=config.fit_ay_min,
                                drop_endpoint=True,
                            )
                        if np.isfinite(fit["c_per_wall"]):
                            jackknife_fits.append(fit["c_per_wall"])
                    if not jackknife_fits:
                        continue
                    row = next(
                        row for row in fit_rows
                        if row["family"] == "quenched_mean"
                        and row["Ny"] == ny
                        and np.isclose(float(row["cap"]), config.default_cap)
                        and np.isclose(float(row["twist"]), float(twists[-1]))
                        and row["renyi_order"] == order
                        and row["fit_variant"] == variant
                    )
                    values = np.asarray(jackknife_fits, dtype=np.float64)
                    standard_error = math.sqrt(
                        (values.size - 1) / values.size
                        * float(np.sum((values - values.mean()) ** 2))
                    )
                    row["ensemble_c_mean"] = row["c_per_wall"]
                    row["ensemble_c_ci_low"] = float(row["c_per_wall"]) - 1.96 * standard_error
                    row["ensemble_c_ci_high"] = float(row["c_per_wall"]) + 1.96 * standard_error

        # Annealed and flattened controls at the default cap.
        annealed_a_disp = spectral_function_displacements(
            mean_g_disp, "arctanh", cap=config.default_cap
        )
        controls = {
            "annealed_covariance": -2.0 * annealed_a_disp,
            "flattened_parent": -mean_sign_disp,
        }
        for method, h_disp in controls.items():
            band = band_observables(
                h_disp,
                wall_weight_min=config.wall_weight_min,
                active_x=spec.active_x,
            )
            correlation, ground_diag = half_filled_ground_state(
                h_disp, twist=float(twists[-1])
            )
            curve, entropy_diag = entropy_profiles(
                correlation, width=width, ny=ny, orders=orders, average_origins=False
            )
            ground_diag.update(entropy_diag)
            diagnostic_rows.append(
                {
                    "family": method,
                    "Ny": ny,
                    "sample_index": "",
                    "cap": config.default_cap,
                    "twist": float(twists[-1]),
                    **ground_diag,
                }
            )
            notebook_arrays[f"Ny{ny:03d}_{method}_ky"] = band["ky"]
            notebook_arrays[f"Ny{ny:03d}_{method}_wall_energy"] = band["wall_energy"]
            for order_index, order in enumerate(orders.tolist()):
                for ay_index in range(ny // 2):
                    entropy_rows.append(
                        {
                            "family": method,
                            "Ny": ny,
                            "samples": samples,
                            "cap": config.default_cap,
                            "twist": float(twists[-1]),
                            "renyi_order": order,
                            "Ay": ay_index + 1,
                            "mean": curve[order_index, ay_index],
                            "sem": 0.0,
                            "ci_low": curve[order_index, ay_index],
                            "ci_high": curve[order_index, ay_index],
                        }
                    )
                for variant, fit in _fit_variants(curve[order_index], ny=ny, order=order, config=config):
                    fit_rows.append(
                        {
                            "family": method,
                            "Ny": ny,
                            "samples": samples,
                            "cap": config.default_cap,
                            "twist": float(twists[-1]),
                            "renyi_order": order,
                            "fit_variant": variant,
                            **fit,
                            "ensemble_c_mean": "",
                            "ensemble_c_ci_low": "",
                            "ensemble_c_ci_high": "",
                        }
                    )

        for record, payload in items:
            base = json.loads(str(_scalar(payload, "base_diagnostics_json")))
            caps_diag = json.loads(str(_scalar(payload, "cap_diagnostics_json")))
            ground_diag = json.loads(str(_scalar(payload, "ground_diagnostics_json")))
            for cap_index, cap in enumerate(caps.tolist()):
                for twist_index, twist in enumerate(twists.tolist()):
                    diagnostic_rows.append(
                        {
                            "family": "typical_sample",
                            "Ny": ny,
                            "sample_index": record.sample_index,
                            "cap": cap,
                            "twist": twist,
                            **base,
                            **caps_diag[cap_index],
                            **ground_diag[cap_index][twist_index],
                        }
                    )

    _write_csv(output_root / "spectrum_rows.csv", spectrum_rows)
    _write_csv(output_root / "velocity_summary.csv", velocity_rows)
    _write_csv(output_root / "entropy_curves.csv", entropy_rows)
    _write_csv(output_root / "entropy_fit_summary.csv", fit_rows)
    _write_csv(output_root / "numerical_diagnostics.csv", diagnostic_rows)
    np.savez_compressed(output_root / "notebook_arrays.npz", **notebook_arrays)

    claim_ledger = evaluate_claims(
        spectrum_rows=spectrum_rows,
        velocity_rows=velocity_rows,
        fit_rows=fit_rows,
        diagnostic_rows=diagnostic_rows,
        config=config,
    )
    summary = {
        "analysis_schema": ANALYSIS_SCHEMA,
        "analysis_config": config.identity(),
        "analysis_config_hash": config.identity_hash,
        "source_revision": spec.revision,
        "construction": spec.construction,
        "Nx": EXPECTED_NX,
        "Ny_values": sorted(selected),
        "sample_counts": {str(ny): len(items) for ny, items in sorted(selected.items())},
        "estimator_order": "trajectory log -> y twirl -> quenched sample mean",
        "polar_convention": "T=P_L U; A=log(P_L)=arctanh(G_active); h=-2A",
        "active_x": list(spec.active_x),
        "claim_ledger": claim_ledger,
        "output_files": [
            "spectrum_rows.csv",
            "velocity_summary.csv",
            "entropy_curves.csv",
            "entropy_fit_summary.csv",
            "numerical_diagnostics.csv",
            "notebook_arrays.npz",
        ],
    }
    _atomic_write_json(output_root / "analysis_summary.json", summary)
    return {"summary": summary, "arrays": notebook_arrays}


def evaluate_claims(
    *,
    spectrum_rows: Sequence[dict[str, Any]],
    velocity_rows: Sequence[dict[str, Any]],
    fit_rows: Sequence[dict[str, Any]],
    diagnostic_rows: Sequence[dict[str, Any]],
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Apply preregistered numerical, edge, and shared-criticality gates."""
    diagnostics = [row for row in diagnostic_rows if row.get("family") in {"typical_sample", "quenched_mean"}]
    numerical_fields = (
        "full_hermiticity_residual",
        "active_hermiticity_residual",
        "active_exterior_coupling",
        "exterior_purity_residual",
        "exterior_inter_y_coupling",
        "g_twirl_hermiticity_residual",
        "g_twirl_translation_residual",
        "g_fourier_reconstruction_residual",
        "polar_hermiticity_residual",
        "a_twirl_hermiticity_residual",
        "a_twirl_translation_residual",
        "a_fourier_reconstruction_residual",
        "projector_idempotency_residual",
        "half_filling_charge_residual",
        "entropy_origin_spread",
    )
    maxima = {
        field: max((float(row[field]) for row in diagnostics if field in row), default=math.nan)
        for field in numerical_fields
    }
    numerical_pass = all(
        np.isfinite(value) and value <= NUMERICAL_TOL for value in maxima.values()
    )

    edge_by_size: dict[str, Any] = {}
    for ny in (20, 30, 40):
        cap_results = []
        for cap in config.caps:
            per_wall = [
                next(
                    (
                        {
                            "weight": float(row["crossing_wall_weight"]),
                            "velocity": float(row["velocity"]),
                            "crossing": float(row["crossing"]),
                        }
                        for row in velocity_rows
                        if row["method"] == "quenched_mean"
                        and row["Ny"] == ny
                        and np.isclose(float(row["cap"]), cap)
                        and int(row["wall"]) == wall
                    ),
                    {"weight": math.nan, "velocity": math.nan, "crossing": math.nan},
                )
                for wall in (0, 1)
            ]
            passed = (
                all(row["weight"] >= config.wall_weight_min for row in per_wall)
                and np.prod([row["velocity"] for row in per_wall]) < 0
                and all(np.isfinite(row["crossing"]) for row in per_wall)
            )
            cap_results.append({"cap": cap, "walls": per_wall, "pass": bool(passed)})
        crossing_spans = []
        for wall in (0, 1):
            crossings = np.asarray(
                [row["walls"][wall]["crossing"] for row in cap_results],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(crossings)):
                crossing_spans.append(math.inf)
                continue
            ordered = np.sort((crossings + 2.0 * np.pi) % (2.0 * np.pi))
            gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
            crossing_spans.append(float(2.0 * np.pi - gaps.max()))
        crossing_tolerance = 2.0 * np.pi / ny
        edge_by_size[str(ny)] = {
            "caps": cap_results,
            "crossing_spans": crossing_spans,
            "crossing_stability_tolerance": crossing_tolerance,
            "cap_stable_pass": bool(
                cap_results
                and all(row["pass"] for row in cap_results)
                and all(span <= crossing_tolerance for span in crossing_spans)
            ),
        }

    def primary_fit(family: str, ny: int, order: int) -> dict[str, float]:
        candidates = [
            row for row in fit_rows
            if row["family"] == family
            and row["Ny"] == ny
            and row["renyi_order"] == order
            and row["fit_variant"] == "primary"
            and (family == "actual_endpoint" or np.isclose(float(row["cap"]), config.default_cap))
            and (family == "actual_endpoint" or np.isclose(float(row["twist"]), max(config.twists)))
        ]
        if not candidates:
            return {"value": math.nan, "ci_low": math.nan, "ci_high": math.nan}
        row = candidates[0]
        low = row.get("ensemble_c_ci_low", math.nan)
        high = row.get("ensemble_c_ci_high", math.nan)
        return {
            "value": float(row["c_per_wall"]),
            "ci_low": float(low) if low != "" else math.nan,
            "ci_high": float(high) if high != "" else math.nan,
        }

    c_estimates = {
        family: {
            str(order): primary_fit(family, 40, order)
            for order in config.renyi_orders
        }
        for family in ("quenched_mean", "typical_sample", "actual_endpoint")
    }
    c1 = [c_estimates[family]["1"]["value"] for family in c_estimates]
    c1_lows = [c_estimates[family]["1"]["ci_low"] for family in c_estimates]
    c1_highs = [c_estimates[family]["1"]["ci_high"] for family in c_estimates]
    c1_compatible = (
        all(np.isfinite(value) for value in c1_lows + c1_highs)
        and max(c1_lows) <= min(c1_highs)
    )
    cap_stability: dict[str, Any] = {}
    twist_stability: dict[str, Any] = {}
    for family in ("quenched_mean", "typical_sample"):
        cap_stability[family] = {}
        for order in config.renyi_orders:
            cap_values = []
            for cap in config.caps:
                candidates = [
                    row
                    for row in fit_rows
                    if row["family"] == family
                    and row["Ny"] == 40
                    and row["renyi_order"] == order
                    and row["fit_variant"] == "primary"
                    and np.isclose(float(row["cap"]), cap)
                    and np.isclose(float(row["twist"]), max(config.twists))
                ]
                cap_values.append(
                    float(candidates[0]["c_per_wall"]) if candidates else math.nan
                )
            spread = (
                float(np.ptp(cap_values))
                if np.all(np.isfinite(cap_values))
                else math.inf
            )
            cap_stability[family][str(order)] = {
                "values": cap_values,
                "spread": spread,
                "pass": bool(spread <= 0.15),
            }
        twist_values = []
        for twist in config.twists:
            candidates = [
                row
                for row in fit_rows
                if row["family"] == family
                and row["Ny"] == 40
                and row["renyi_order"] == 1
                and row["fit_variant"] == "primary"
                and np.isclose(float(row["cap"]), config.default_cap)
                and np.isclose(float(row["twist"]), twist)
            ]
            twist_values.append(
                float(candidates[0]["c_per_wall"]) if candidates else math.nan
            )
        twist_difference = (
            float(abs(twist_values[1] - twist_values[0]))
            if np.all(np.isfinite(twist_values))
            else math.inf
        )
        twist_stability[family] = {
            "values": twist_values,
            "difference": twist_difference,
            "pass": bool(twist_difference <= 0.15),
        }
    cap_stable = all(
        row["pass"]
        for family_rows in cap_stability.values()
        for row in family_rows.values()
    )
    twist_stable = all(row["pass"] for row in twist_stability.values())
    shared_pass = (
        numerical_pass
        and all(np.isfinite(c1_value) and abs(c1_value - 1.0) <= 0.15 for c1_value in c1)
        and c1_compatible
        and cap_stable
        and twist_stable
        and all(
            np.isfinite(c_estimates[family][str(order)]["value"])
            and abs(c_estimates[family][str(order)]["value"] - 1.0) <= 0.15
            for family in c_estimates
            for order in (2, 3)
        )
    )
    edge_release = bool(
        numerical_pass
        and edge_by_size
        and all(row["cap_stable_pass"] for row in edge_by_size.values())
    )
    polar_cap_identifiable = bool(
        cap_stable
        and edge_by_size
        and all(
            all(
                span <= row["crossing_stability_tolerance"]
                for span in row["crossing_spans"]
            )
            for row in edge_by_size.values()
        )
    )
    return {
        "numerical": {"pass": bool(numerical_pass), "maximum_residuals": maxima},
        "edge_modes": {"release": edge_release, "by_size": edge_by_size},
        "shared_criticality": {
            "pass": bool(shared_pass),
            "largest_size_c_per_wall": c_estimates,
            "joint_c1_interval_compatibility": bool(c1_compatible),
            "cap_stability": cap_stability,
            "twist_stability": twist_stability,
            "absolute_tolerance": 0.15,
        },
        "polar_cap_identifiability": {
            "pass": polar_cap_identifiable,
            "recommendation": (
                "The capped endpoint is stable across Lambda=8,10,12."
                if polar_cap_identifiable
                else "The 4Ny endpoints are too saturated to determine the log-polar "
                "spectrum robustly; acquire intermediate covariances or stabilized polar modes."
            ),
        },
        "saturation_policy": (
            "No polar-spectrum claim is released unless every requested cap passes. "
            "Failure is classified as endpoint non-identifiability, not absence of edge physics."
        ),
    }


__all__ = [
    "ACTIVE_X",
    "ANALYSIS_SCHEMA",
    "AnalysisConfig",
    "BUNDLE_ROOT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SOFT_DATA_ROOT",
    "DEFAULT_SOFT_OUTPUT_ROOT",
    "GeometrySpec",
    "HARD_SPEC",
    "SOFT_ACTIVE_X",
    "SOFT_SPEC",
    "SampleRecord",
    "active_indices",
    "aggregate_analysis",
    "analyze_sample_covariance",
    "band_observables",
    "build_sample_caches",
    "cache_is_valid",
    "capped_log_polar",
    "discover_samples",
    "displacement_hermiticity_residual",
    "displacement_translation_residual",
    "displacements_to_matrix",
    "entropy_profiles",
    "extract_active_covariance",
    "fit_entropy_curve",
    "fourier_reconstruction_residual",
    "geometry_spec",
    "half_filled_ground_state",
    "log_chord",
    "momentum_blocks",
    "renyi_entropy",
    "spectral_function_displacements",
    "strip_indices",
    "y_twirl_displacements",
]
