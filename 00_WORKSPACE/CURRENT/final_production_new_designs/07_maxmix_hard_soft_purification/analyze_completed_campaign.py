#!/usr/bin/env python3
"""Verify and analyze the completed hard-v2/soft-v3 purification ensembles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_ROOT.parents[3]
DEFAULT_REMOTE_INVENTORY = (
    REPO_ROOT
    / "PROJECT_ADMIN"
    / "drive_import_manifests"
    / "completed_campaigns_20260908.remote.json"
)
DEFAULT_OUTPUT_ROOT = (
    BUNDLE_ROOT / "analysis_outputs" / "hard_v2_soft_v3_4ny_analysis_v1"
)
LEGACY_2NY_ANALYSIS_ROOT = (
    REPO_ROOT
    / "00_WORKSPACE"
    / "CURRENT"
    / "final_production_new_designs"
    / "04_maxmix_manybody_lyapunov_pilot"
    / "analysis_outputs"
    / "maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2"
)
LEGACY_2NY_SUMMARY_SHA256 = (
    "145224e94c35c7b3d3945d84d3e70745fcddcf7e1eeb28e43b027f7ace09df48"
)
LEGACY_2NY_TEMPORAL_SHA256 = (
    "ac8e47ff09e45e7f83567b0b2643475676e8d83c9a303381db979289eb2318be"
)

ANALYSIS_SCHEMA = "maxmix_hard_soft_purification_4ny_analysis_v1"
BOOTSTRAP_SEED = 2026090907
BOOTSTRAP_REPLICATES = 2000
SUBSET_REPEATS = 200
LEADING_LEVEL_COUNT = 64
FIT_LEVEL_COUNT = 5
CAP_TOLERANCE = 1.0e-9
TEMPORAL_RELATIVE_LIMIT = 0.10
RECORD_CLOSURE_LIMIT = 0.01
FINITE_SIZE_SENSITIVITY_LIMIT = 0.20
MINIMUM_RESOLVED_FRACTION = 0.95
NY_VALUES = (20, 30, 40)
WINDOWS = {
    "W1_Ny_to_2Ny": (1.0, 2.0),
    "W2_2Ny_to_3Ny": (2.0, 3.0),
    "W3_3Ny_to_4Ny": (3.0, 4.0),
}


@dataclass(frozen=True)
class CampaignSpec:
    key: str
    construction: str
    revision: str
    completion_schema: str
    result_schema: str
    configuration_hash: str
    relative_root: str
    source_hashes: dict[str, str]

    @property
    def data_root(self) -> Path:
        return REPO_ROOT / self.relative_root

    @property
    def transfer_modes_per_ny(self) -> int:
        return 22 if self.construction == "hard" else 40

    @property
    def sites_per_ny(self) -> int:
        return 11 if self.construction == "hard" else 20

    @property
    def log_probability_origin(self) -> str:
        return (
            "after_born_conditioned_exterior_preparation"
            if self.construction == "hard"
            else "global_maxmix_cycle_zero"
        )


COMMON_SOURCE_HASHES = {
    "purification_observer.py": "be6b6df0c7ea11adf3a19ed20c596d433048acd9278a7bbd98c86707e0b415ae",
    "src/classA_U1FGTN_gpu.py": "53de96bced6839b485afe04fe4aaa15d2e42c249a9cf14f6ca0c931f55409700",
    "src/occupied_frame_gpu.py": "bfc10cefea98ce00184a88b5c375eadfcda8e3dc6954d9f66183d566008951b0",
}

CAMPAIGNS = (
    CampaignSpec(
        key="purification_hard_v2",
        construction="hard",
        revision="maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2",
        completion_schema="maxmix_purification_completion_v2",
        result_schema="maxmix_purification_result_v2",
        configuration_hash="2dc0ba9a2a3ec8cc79eebec19bda0a6bcbaf8f3efcd7b47da99eab8decbcffce",
        relative_root=(
            "00_WORKSPACE/CURRENT/final_production_new_designs/"
            "07_maxmix_hard_soft_purification/gpu_data/"
            "maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2"
        ),
        source_hashes={
            **COMMON_SOURCE_HASHES,
            "run_campaign.py": "a479138da1287558aafb84d68c9be54fe8c155fe9e96ff75930f821480462cc1",
        },
    ),
    CampaignSpec(
        key="purification_soft_v3",
        construction="soft",
        revision="maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3",
        completion_schema="maxmix_purification_completion_v3",
        result_schema="maxmix_purification_result_v3",
        configuration_hash="a0169e31d63dc77f08da34b491a88516a05cdb1dd4f98badd19e1735c180b821",
        relative_root=(
            "00_WORKSPACE/CURRENT/final_production_new_designs/"
            "07_maxmix_hard_soft_purification/gpu_data/"
            "maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3"
        ),
        source_hashes={
            **COMMON_SOURCE_HASHES,
            "run_campaign.py": "966af05bf9f68c57e28326c3c7e50d797726a3a2e3ef3d0fa5d370ed550af02d",
        },
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: Any, key: str) -> Any:
    return np.asarray(data[key]).item()


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(low), float(high)


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("a slope requires aligned finite arrays with at least two values")
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("slope coordinates are degenerate")
    return float(np.dot(centered, y - y.mean()) / denominator)


def linear_slope_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    """Return a slope only when the complete preregistered window is resolved."""
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        return math.nan
    return linear_slope(x, y)


def lowest_subset_sums(costs: np.ndarray, count: int) -> np.ndarray:
    """Return the smallest subset sums of finite nonnegative flip costs."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    values = np.sort(np.asarray(costs, dtype=np.float64))
    values = values[np.isfinite(values)]
    if np.any(values < 0.0):
        raise ValueError("flip costs must be nonnegative")
    sums = np.asarray([0.0], dtype=np.float64)
    for value in values:
        merged = np.concatenate((sums, sums + value))
        if merged.size > count:
            merged = merged[np.argpartition(merged, count - 1)[:count]]
        sums = np.sort(merged)
        if sums.size == count and value >= sums[-1]:
            break
    if sums.size < count:
        sums = np.pad(sums, (0, count - sums.size), constant_values=np.inf)
    return sums[:count]


def leading_log_sigma2_levels(
    occupations: np.ndarray,
    log_z: float,
    *,
    count: int = LEADING_LEVEL_COUNT,
    cap_tolerance: float = CAP_TOLERANCE,
) -> np.ndarray:
    """Reconstruct leading ``log(sigma**2)`` levels without Fock enumeration."""
    nu = np.asarray(occupations, dtype=np.float64)
    if nu.ndim != 1 or not np.all(np.isfinite(nu)):
        raise ValueError("occupations must be one finite vector")
    if float(nu.min(initial=0.0)) < -cap_tolerance or float(
        nu.max(initial=1.0)
    ) > 1.0 + cap_tolerance:
        raise FloatingPointError("occupation spectrum leaves [0,1] beyond roundoff")
    caps = (nu <= cap_tolerance) | (nu >= 1.0 - cap_tolerance)
    interior = ~caps
    preferred = np.maximum(nu[interior], 1.0 - nu[interior])
    log_preferred = float(np.log(preferred).sum())
    costs = np.full(nu.shape, np.inf, dtype=np.float64)
    costs[interior] = np.abs(
        np.log(nu[interior]) - np.log1p(-nu[interior])
    )
    return float(log_z) + log_preferred - lowest_subset_sums(costs, count)


def _fit_line(x: np.ndarray, y: np.ndarray, *, intercept: bool) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack((np.ones(x.size), x)) if intercept else x[:, None]
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "intercept": float(coefficients[0]) if intercept else 0.0,
        "coefficient": float(coefficients[-1]),
        "r_squared": float(1.0 - residual / total) if total else 1.0,
    }


def _fit_subleading(x: np.ndarray, y: np.ndarray, *, intercept: bool) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = (
        np.column_stack((np.ones(x.size), x, x**2))
        if intercept
        else np.column_stack((x, x**2))
    )
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    offset = 1 if intercept else 0
    return {
        "intercept": float(coefficients[0]) if intercept else 0.0,
        "coefficient": float(coefficients[offset]),
        "subleading_coefficient": float(coefficients[offset + 1]),
    }


def _bootstrap_mean_ci(
    values: np.ndarray, *, count: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    chosen = rng.integers(0, values.size, size=(count, values.size))
    return percentile_interval(values[chosen].mean(axis=1))


def _inventory_campaigns(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "completed_drive_campaign_remote_inventory_v1":
        raise RuntimeError("remote inventory schema mismatch")
    return {row["campaign"]: row for row in payload["campaigns"]}


def _expected_batch(ny: int, sample_index: int) -> tuple[int, int, int]:
    batch_size = 100 if ny in (20, 30) else 40
    start = (int(sample_index) // batch_size) * batch_size
    stop = min(100, start + batch_size)
    return start // batch_size, start, stop


def _execution_seed(construction: str, ny: int, start: int, stop: int) -> int:
    label = f"2026090407|{construction}|Ny={ny}|samples={start}:{stop}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little")


def _validate_completion(
    payload: dict[str, Any],
    spec: CampaignSpec,
    result_path: Path,
    *,
    expected_result_bytes: int,
    expected_result_sha256: str,
) -> None:
    expected = {
        "schema": spec.completion_schema,
        "sampling_revision": spec.revision,
        "configuration_hash": spec.configuration_hash,
        "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
        "construction": spec.construction,
        "Nx": 20,
        "dtype": "complex128",
        "result_filename": result_path.name,
        "result_bytes": int(expected_result_bytes),
        "result_sha256": expected_result_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{result_path.name}: completion mismatch for {key}")
    if payload.get("source_hashes") != spec.source_hashes:
        raise RuntimeError(f"{result_path.name}: historical source hash mismatch")
    ny = int(payload["Ny"])
    if ny not in NY_VALUES or int(payload["cycles"]) != 4 * ny:
        raise RuntimeError(f"{result_path.name}: geometry/depth mismatch")
    indices = [int(value) for value in payload["sample_indices"]]
    if len(indices) != 5 or indices != list(range(indices[0], indices[0] + 5)):
        raise RuntimeError(f"{result_path.name}: sample shard is not contiguous size five")
    batch_index, start, stop = _expected_batch(ny, indices[0])
    expected_task = (
        f"{spec.construction}_Ny{ny:03d}_exec{batch_index:03d}_"
        f"samples{start:03d}-{stop - 1:03d}"
    )
    if payload["execution_batch_id"] != expected_task:
        raise RuntimeError(f"{result_path.name}: execution batch identity mismatch")
    if int(payload["execution_seed"]) != _execution_seed(spec.construction, ny, start, stop):
        raise RuntimeError(f"{result_path.name}: execution seed mismatch")


def _validate_product(
    data: Any,
    *,
    spec: CampaignSpec,
    completion: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    ny = int(completion["Ny"])
    cycles = 4 * ny
    samples = len(completion["sample_indices"])
    modes = 40 * ny
    expected_cycles = np.arange(cycles + 1, dtype=np.int64)
    scalar_expectations = {
        "result_schema": spec.result_schema,
        "sampling_revision": spec.revision,
        "configuration_hash": spec.configuration_hash,
        "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
        "construction": spec.construction,
        "Nx": 20,
        "Ny": ny,
        "observer_schema": "maxmix_purification_observer_v2",
        "log_probability_dtype": "float64",
        "log_probability_convention": "cumulative_realized_born_log_probability",
        "log_probability_origin": spec.log_probability_origin,
        "transfer_mode_count": spec.transfer_modes_per_ny * ny,
        "log_z_formula": "log_Z=N_eff*log(2)+cumulative_log_probability",
        "centered_covariance_convention": "G=2C-I",
    }
    for key, value in scalar_expectations.items():
        if scalar(data, key) != value:
            raise RuntimeError(f"{result_path.name}: product mismatch for {key}")
    if not np.array_equal(data["cycles"], expected_cycles):
        raise RuntimeError(f"{result_path.name}: cycle grid mismatch")
    if not np.array_equal(data["sample_indices"], completion["sample_indices"]):
        raise RuntimeError(f"{result_path.name}: sample identity mismatch")

    shapes = {
        "occupation_spectrum": (samples, cycles + 1, modes),
        "entropy_contour": (samples, cycles + 1, 20, ny),
        "charge_variance_contour": (samples, cycles + 1, 20, ny),
        "total_entropy": (samples, cycles + 1),
        "total_charge": (samples, cycles + 1),
        "total_charge_variance": (samples, cycles + 1),
        "hermiticity_residual": (samples, cycles + 1),
        "entropy_closure_error": (samples, cycles + 1),
        "charge_closure_error": (samples, cycles + 1),
        "measurement_log_probability": (samples, cycles + 1),
        "cumulative_log_probability": (samples, cycles + 1),
        "site_event_count": (samples, cycles + 1),
        "channel_event_count": (samples, cycles + 1),
        "G_final": (samples, modes, modes),
    }
    maxima: dict[str, float] = {}
    for key, shape in shapes.items():
        array = np.asarray(data[key])
        if array.shape != shape:
            raise RuntimeError(f"{result_path.name}: {key} shape mismatch")
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{result_path.name}: {key} contains nonfinite values")
    if np.asarray(data["G_final"]).dtype != np.complex128:
        raise RuntimeError(f"{result_path.name}: final covariance is not complex128")

    omega = np.asarray(data["cumulative_log_probability"], dtype=np.float64)
    increments = np.asarray(data["measurement_log_probability"], dtype=np.float64)
    record_error = float(np.max(np.abs(np.diff(omega, axis=1) - increments[:, 1:])))
    if record_error > 2.0e-10:
        raise RuntimeError(f"{result_path.name}: record accumulation mismatch")
    if not np.array_equal(omega[:, 0], np.zeros(samples)) or not np.array_equal(
        increments[:, 0], np.zeros(samples)
    ):
        raise RuntimeError(f"{result_path.name}: record origin is not zero")
    sites = np.asarray(data["site_event_count"], dtype=np.int64)
    channels = np.asarray(data["channel_event_count"], dtype=np.int64)
    if np.any(sites[:, 0]) or np.any(channels[:, 0]):
        raise RuntimeError(f"{result_path.name}: cycle-zero event count is nonzero")
    expected_sites = spec.sites_per_ny * ny
    if not np.all(sites[:, 1:] == expected_sites) or not np.all(
        channels[:, 1:] == 4 * expected_sites
    ):
        raise RuntimeError(f"{result_path.name}: measurement event count mismatch")

    occupations = np.asarray(data["occupation_spectrum"], dtype=np.float64)
    occupation_bound = max(
        0.0,
        float(-occupations.min(initial=0.0)),
        float(occupations.max(initial=1.0) - 1.0),
    )
    if occupation_bound > CAP_TOLERANCE:
        raise RuntimeError(f"{result_path.name}: occupation bound residual too large")
    t0 = occupations[:, 0]
    half_count = np.sum(np.abs(t0 - 0.5) <= CAP_TOLERANCE, axis=1)
    if not np.all(half_count == spec.transfer_modes_per_ny * ny):
        raise RuntimeError(f"{result_path.name}: identity-input transfer dimension mismatch")

    g_final = np.asarray(data["G_final"], dtype=np.complex128)
    g_hermiticity = float(np.max(np.abs(g_final - np.swapaxes(g_final.conj(), -1, -2))))
    if g_hermiticity > 1.0e-9:
        raise RuntimeError(f"{result_path.name}: final covariance is non-Hermitian")
    maxima.update(
        {
            "record_accumulation_error": record_error,
            "occupation_bound_residual": occupation_bound,
            "hermiticity_residual": float(np.max(data["hermiticity_residual"])),
            "entropy_closure_error": float(np.max(data["entropy_closure_error"])),
            "charge_closure_error": float(np.max(data["charge_closure_error"])),
            "final_covariance_hermiticity_residual": g_hermiticity,
        }
    )
    return maxima


def reconstruct_levels(occupations: np.ndarray, omega: np.ndarray, n_eff: int) -> np.ndarray:
    occupations = np.asarray(occupations, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    samples, times, _ = occupations.shape
    levels = np.empty((samples, times, LEADING_LEVEL_COUNT), dtype=np.float64)
    for sample in range(samples):
        for position in range(times):
            levels[sample, position] = leading_log_sigma2_levels(
                occupations[sample, position],
                n_eff * math.log(2.0) + float(omega[sample, position]),
            )
    if not np.allclose(levels[:, 0], 0.0, rtol=0.0, atol=2.0e-9):
        raise RuntimeError("cycle-zero many-body singular spectrum is not the identity")
    return levels


def load_and_verify(
    *, inventory_path: Path, verify_hashes: bool = True
) -> tuple[dict[tuple[str, int], dict[str, np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    inventory = _inventory_campaigns(inventory_path)
    grouped: dict[tuple[str, int], dict[str, list[np.ndarray]]] = {}
    numerical_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {
        "remote_inventory": str(inventory_path.resolve()),
        "remote_inventory_sha256": sha256_file(inventory_path),
        "verify_file_hashes": bool(verify_hashes),
        "campaigns": {},
    }
    for spec in CAMPAIGNS:
        record = inventory.get(spec.key)
        if record is None:
            raise RuntimeError(f"remote inventory omits {spec.key}")
        if Path(record["canonical_local_destination"]) != Path(spec.relative_root):
            raise RuntimeError(f"{spec.key}: canonical destination mismatch")
        if int(record["result_completion_pairs"]) != 60:
            raise RuntimeError(f"{spec.key}: expected exactly 60 result pairs")
        discovered = set(spec.data_root.rglob("*.npz")) | set(
            spec.data_root.rglob("*.complete.json")
        )
        expected_paths: set[Path] = set()
        coverage = {ny: set() for ny in NY_VALUES}
        for remote in record["records"]:
            result_path = spec.data_root / remote["relative_result_path"]
            completion_path = spec.data_root / remote["relative_completion_path"]
            expected_paths.update((result_path, completion_path))
            for path, bytes_key, hash_key in (
                (result_path, "result_bytes", "result_sha256"),
                (completion_path, "completion_bytes", "completion_sha256"),
            ):
                if not path.is_file() or path.stat().st_size != int(remote[bytes_key]):
                    raise RuntimeError(f"{spec.key}: missing or wrong-sized {path}")
                if verify_hashes and sha256_file(path) != remote[hash_key]:
                    raise RuntimeError(f"{spec.key}: checksum mismatch for {path}")
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            _validate_completion(
                completion,
                spec,
                result_path,
                expected_result_bytes=int(remote["result_bytes"]),
                expected_result_sha256=remote["result_sha256"],
            )
            ny = int(completion["Ny"])
            coverage[ny].update(int(value) for value in completion["sample_indices"])
            with np.load(result_path, allow_pickle=False) as data:
                maxima = _validate_product(
                    data, spec=spec, completion=completion, result_path=result_path
                )
                occupations = np.asarray(data["occupation_spectrum"], dtype=np.float64)
                omega = np.asarray(data["cumulative_log_probability"], dtype=np.float64)
                levels = reconstruct_levels(
                    occupations, omega, spec.transfer_modes_per_ny * ny
                )
                key = (spec.construction, ny)
                store = grouped.setdefault(key, {})
                entropy_contour = np.asarray(data["entropy_contour"], dtype=np.float64)
                variance_contour = np.asarray(
                    data["charge_variance_contour"], dtype=np.float64
                )
                fields = {
                    "sample_indices": np.asarray(data["sample_indices"], dtype=np.int64),
                    "levels": levels,
                    "omega": omega,
                    "total_entropy": np.asarray(data["total_entropy"], dtype=np.float64),
                    "total_charge": np.asarray(data["total_charge"], dtype=np.float64),
                    "total_charge_variance": np.asarray(
                        data["total_charge_variance"], dtype=np.float64
                    ),
                    "endpoint_entropy_x": entropy_contour[:, -1].sum(axis=-1),
                    "endpoint_variance_x": variance_contour[:, -1].sum(axis=-1),
                }
                for name, value in fields.items():
                    store.setdefault(name, []).append(value)
                numerical_rows.append(
                    {
                        "campaign": spec.key,
                        "construction": spec.construction,
                        "sampling_revision": spec.revision,
                        "Ny": ny,
                        "result": str(result_path.relative_to(REPO_ROOT)),
                        **maxima,
                    }
                )
        if discovered != expected_paths:
            unexpected = sorted(str(path.relative_to(spec.data_root)) for path in discovered - expected_paths)
            missing = sorted(str(path.relative_to(spec.data_root)) for path in expected_paths - discovered)
            raise RuntimeError(
                f"{spec.key}: noncanonical result tree; unexpected={unexpected}, missing={missing}"
            )
        for ny, indices in coverage.items():
            if indices != set(range(100)):
                raise RuntimeError(f"{spec.key}: Ny={ny} sample coverage mismatch")
        provenance["campaigns"][spec.construction] = {
            "campaign_key": spec.key,
            "sampling_revision": spec.revision,
            "configuration_hash": spec.configuration_hash,
            "source_hashes": spec.source_hashes,
            "result_completion_pairs": 60,
            "trajectories": 300,
            "total_remote_bytes": int(record["total_remote_bytes"]),
            "data_root": str(spec.data_root),
        }
    arrays = {
        key: {name: np.concatenate(values, axis=0) for name, values in store.items()}
        for key, store in grouped.items()
    }
    for construction in ("hard", "soft"):
        for ny in NY_VALUES:
            key = (construction, ny)
            if key not in arrays or arrays[key]["sample_indices"].shape != (100,):
                raise RuntimeError(f"missing complete analysis group {key}")
            order = np.argsort(arrays[key]["sample_indices"])
            for name in arrays[key]:
                arrays[key][name] = arrays[key][name][order]
            if not np.array_equal(arrays[key]["sample_indices"], np.arange(100)):
                raise RuntimeError(f"noncanonical sample ordering for {key}")
    provenance["verified_result_pairs"] = 120
    provenance["verified_trajectories"] = 600
    return arrays, numerical_rows, provenance


def _window_mask(ny: int, cycles: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    start, stop = bounds
    return (cycles >= start * ny) & (cycles <= stop * ny)


def trajectory_slopes(
    arrays: dict[tuple[str, int], dict[str, np.ndarray]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for (construction, ny), data in sorted(arrays.items()):
        cycles = np.arange(4 * ny + 1, dtype=np.float64)
        result: dict[str, np.ndarray] = {}
        for window, bounds in WINDOWS.items():
            mask = _window_mask(ny, cycles, bounds)
            level_slopes = np.asarray(
                [
                    [linear_slope_or_nan(cycles[mask], trajectory[mask, level]) for level in range(FIT_LEVEL_COUNT)]
                    for trajectory in data["levels"]
                ]
            )
            omega_slopes = np.asarray(
                [linear_slope(cycles[mask], trajectory[mask]) for trajectory in data["omega"]]
            )
            result[f"{window}_levels"] = level_slopes
            result[f"{window}_omega"] = omega_slopes
            for sample in range(100):
                row = {
                    "construction": construction,
                    "Ny": ny,
                    "sample_index": sample,
                    "window": window,
                    "window_start": bounds[0] * ny,
                    "window_stop": bounds[1] * ny,
                    "omega_slope": float(omega_slopes[sample]),
                    "omega_minus_lambda0": float(omega_slopes[sample] - level_slopes[sample, 0]),
                    "endpoint_record_rate": float(-data["omega"][sample, -1] / (4 * ny)),
                }
                for level in range(FIT_LEVEL_COUNT):
                    row[f"lambda{level}"] = float(level_slopes[sample, level])
                    if level:
                        row[f"gap{level}"] = float(
                            level_slopes[sample, 0] - level_slopes[sample, level]
                        )
                rows.append(row)
        fitted[(construction, ny)] = result
    return rows, fitted


def temporal_stability(
    fitted: dict[tuple[str, int], dict[str, np.ndarray]],
    *,
    bootstrap_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("W1_Ny_to_2Ny", "W2_2Ny_to_3Ny"),
        ("W2_2Ny_to_3Ny", "W3_3Ny_to_4Ny"),
    )
    for (construction, ny), data in sorted(fitted.items()):
        for earlier, later in comparisons:
            first = data[f"{earlier}_levels"]
            second = data[f"{later}_levels"]
            for level in range(FIT_LEVEL_COUNT):
                if level == 0:
                    name = "lambda0"
                    first_values, second_values = first[:, 0], second[:, 0]
                else:
                    name = f"gap{level}"
                    first_values = first[:, 0] - first[:, level]
                    second_values = second[:, 0] - second[:, level]
                shift = second_values - first_values
                valid = np.isfinite(first_values) & np.isfinite(second_values)
                resolved = int(valid.sum())
                resolved_fraction = resolved / int(valid.size)
                if resolved:
                    first_valid = first_values[valid]
                    second_valid = second_values[valid]
                    shift_valid = shift[valid]
                    low, high = _bootstrap_mean_ci(
                        shift_valid, count=bootstrap_count, rng=rng
                    )
                    denominator = abs(float(second_valid.mean()))
                    relative = (
                        abs(float(shift_valid.mean())) / denominator
                        if denominator
                        else math.inf
                    )
                    earlier_mean = float(first_valid.mean())
                    earlier_sem = sem(first_valid)
                    later_mean = float(second_valid.mean())
                    later_sem = sem(second_valid)
                    shift_mean = float(shift_valid.mean())
                else:
                    low = high = math.nan
                    relative = math.inf
                    earlier_mean = earlier_sem = later_mean = later_sem = shift_mean = math.nan
                rows.append(
                    {
                        "construction": construction,
                        "Ny": ny,
                        "metric": name,
                        "earlier_window": earlier,
                        "later_window": later,
                        "resolved_trajectories": resolved,
                        "resolved_fraction": resolved_fraction,
                        "minimum_resolved_fraction": MINIMUM_RESOLVED_FRACTION,
                        "earlier_mean": earlier_mean,
                        "earlier_sem": earlier_sem,
                        "later_mean": later_mean,
                        "later_sem": later_sem,
                        "shift": shift_mean,
                        "shift_ci_low": low,
                        "shift_ci_high": high,
                        "shift_ci_contains_zero": bool(low <= 0.0 <= high),
                        "relative_shift": float(relative),
                        "relative_limit": TEMPORAL_RELATIVE_LIMIT,
                        "passes": bool(
                            resolved_fraction >= MINIMUM_RESOLVED_FRACTION
                            and low <= 0.0 <= high
                            and relative <= TEMPORAL_RELATIVE_LIMIT
                        ),
                    }
                )
    return rows


def legacy_depth_comparison(
    current_temporal_rows: list[dict[str, Any]],
    *,
    legacy_root: Path = LEGACY_2NY_ANALYSIS_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare independent 2Ny and 4Ny gate ledgers without pooling samples."""
    summary_path = legacy_root / "analysis_summary.json"
    temporal_path = legacy_root / "temporal_convergence.csv"
    if sha256_file(summary_path) != LEGACY_2NY_SUMMARY_SHA256:
        raise RuntimeError("legacy 2Ny analysis summary checksum mismatch")
    if sha256_file(temporal_path) != LEGACY_2NY_TEMPORAL_SHA256:
        raise RuntimeError("legacy 2Ny temporal ledger checksum mismatch")

    legacy_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_identity = {
        "schema": "maxmix_manybody_lyapunov_analysis_v2",
        "sampling_revision": "maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2",
        "verified_tasks": 160,
        "verified_trajectories": 800,
        "bootstrap_replicates": 2000,
    }
    for key, expected in expected_identity.items():
        if legacy_summary.get(key) != expected:
            raise RuntimeError(
                f"legacy 2Ny analysis identity mismatch for {key}: "
                f"{legacy_summary.get(key)!r} != {expected!r}"
            )
    if legacy_summary.get("acceptance_decisions", {}).get("T_2Ny_temporally_sufficient") is not False:
        raise RuntimeError("legacy 2Ny temporal acceptance decision unexpectedly changed")

    with temporal_path.open(newline="", encoding="utf-8") as handle:
        legacy_rows = list(csv.DictReader(handle))
    if len(legacy_rows) != 40:
        raise RuntimeError(f"expected 40 legacy temporal rows, found {len(legacy_rows)}")

    current_index = {
        (int(row["Ny"]), str(row["metric"])): row
        for row in current_temporal_rows
        if row["construction"] == "hard"
        and row["earlier_window"] == "W2_2Ny_to_3Ny"
    }
    legacy_index = {
        (int(row["Ny"]), str(row["metric"])): row for row in legacy_rows
    }
    comparison_rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        for metric in ("lambda0", "gap1", "gap2", "gap3", "gap4"):
            key = (ny, metric)
            if key not in legacy_index or key not in current_index:
                raise RuntimeError(f"missing depth-comparison row {key}")
            old = legacy_index[key]
            new = current_index[key]
            comparison_rows.extend(
                (
                    {
                        "ensemble": "independent_hard_wall_2Ny_v2",
                        "depth_multiple": 2,
                        "Ny": ny,
                        "metric": metric,
                        "earlier_window": "Ny_to_3Ny_over_2",
                        "later_window": "3Ny_over_2_to_2Ny",
                        "resolved_fraction": 1.0,
                        "relative_shift": float(old["relative_shift"]),
                        "shift_ci_low": float(old["shift_ci_low"]),
                        "shift_ci_high": float(old["shift_ci_high"]),
                        "shift_ci_contains_zero": old["shift_ci_contains_zero"] == "True",
                        "passes": old["passes"] == "True",
                        "samples": 100,
                        "pooled_with_4Ny": False,
                    },
                    {
                        "ensemble": "independent_hard_wall_4Ny_v2",
                        "depth_multiple": 4,
                        "Ny": ny,
                        "metric": metric,
                        "earlier_window": str(new["earlier_window"]),
                        "later_window": str(new["later_window"]),
                        "resolved_fraction": float(new["resolved_fraction"]),
                        "relative_shift": float(new["relative_shift"]),
                        "shift_ci_low": float(new["shift_ci_low"]),
                        "shift_ci_high": float(new["shift_ci_high"]),
                        "shift_ci_contains_zero": bool(new["shift_ci_contains_zero"]),
                        "passes": bool(new["passes"]),
                        "samples": 100,
                        "pooled_with_2Ny": False,
                    },
                )
            )

    lambda_rows = [row for row in comparison_rows if row["metric"] == "lambda0"]
    comparison_summary = {
        "purpose": "independent depth diagnostic only; no trajectory pooling",
        "legacy_analysis_schema": legacy_summary["schema"],
        "legacy_sampling_revision": legacy_summary["sampling_revision"],
        "legacy_verified_trajectories": legacy_summary["verified_trajectories"],
        "legacy_summary_path": str(summary_path.relative_to(REPO_ROOT)),
        "legacy_summary_sha256": LEGACY_2NY_SUMMARY_SHA256,
        "legacy_temporal_path": str(temporal_path.relative_to(REPO_ROOT)),
        "legacy_temporal_sha256": LEGACY_2NY_TEMPORAL_SHA256,
        "shared_Ny": list(NY_VALUES),
        "lambda0_pass_by_depth": {
            str(depth): {
                str(row["Ny"]): bool(row["passes"])
                for row in lambda_rows
                if row["depth_multiple"] == depth
            }
            for depth in (2, 4)
        },
    }
    return comparison_rows, comparison_summary


def record_closure(
    fitted: dict[tuple[str, int], dict[str, np.ndarray]],
    *,
    bootstrap_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (construction, ny), data in sorted(fitted.items()):
        for window in WINDOWS:
            levels = data[f"{window}_levels"][:, 0]
            omega = data[f"{window}_omega"]
            difference = omega - levels
            low, high = _bootstrap_mean_ci(difference, count=bootstrap_count, rng=rng)
            relative = abs(float(difference.mean())) / abs(float(levels.mean()))
            rows.append(
                {
                    "construction": construction,
                    "Ny": ny,
                    "window": window,
                    "samples": 100,
                    "lambda0_mean": float(levels.mean()),
                    "lambda0_sem": sem(levels),
                    "omega_slope_mean": float(omega.mean()),
                    "omega_slope_sem": sem(omega),
                    "omega_minus_lambda0": float(difference.mean()),
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "relative_closure": float(relative),
                    "relative_limit": RECORD_CLOSURE_LIMIT,
                    "passes": bool(relative < RECORD_CLOSURE_LIMIT),
                }
            )
    return rows


def _finite_fit_from_means(ny: np.ndarray, means: np.ndarray) -> dict[str, Any]:
    x = 1.0 / ny.astype(np.float64) ** 2
    f0 = -means[:, 0] / ny
    primary = _fit_line(x, f0, intercept=True)
    subleading = _fit_subleading(x, f0, intercept=True)
    omit = _fit_line(x[1:], f0[1:], intercept=True)
    gaps = []
    for level in range(1, FIT_LEVEL_COUNT):
        density = (means[:, 0] - means[:, level]) / ny
        constrained = _fit_line(x, density, intercept=False)
        unconstrained = _fit_line(x, density, intercept=True)
        sub = _fit_subleading(x, density, intercept=False)
        omit_gap = _fit_line(x[1:], density[1:], intercept=False)
        gaps.append(
            {
                "level": level,
                "Ai": constrained["coefficient"],
                "alpha_x": constrained["coefficient"] / (2.0 * math.pi),
                "unconstrained_intercept": unconstrained["intercept"],
                "unconstrained_Ai": unconstrained["coefficient"],
                "subleading_Ai": sub["coefficient"],
                "subleading_Bi": sub["subleading_coefficient"],
                "omit_Ny20_Ai": omit_gap["coefficient"],
                "density": density.tolist(),
            }
        )
    return {
        "Ny": ny.astype(int).tolist(),
        "inverse_Ny_squared": x.tolist(),
        "f0_tilde": f0.tolist(),
        "primary": {
            "f_infinity": primary["intercept"],
            "A0": primary["coefficient"],
            "alpha_c_eff": -6.0 * primary["coefficient"] / math.pi,
            "r_squared": primary["r_squared"],
        },
        "subleading": {
            "f_infinity": subleading["intercept"],
            "A0": subleading["coefficient"],
            "B0": subleading["subleading_coefficient"],
            "exactly_determined_three_parameter_fit": True,
        },
        "omit_Ny20": {
            "f_infinity": omit["intercept"],
            "A0": omit["coefficient"],
            "exactly_determined_two_point_fit": True,
        },
        "gaps": gaps,
    }


def finite_size_analysis(
    fitted: dict[tuple[str, int], dict[str, np.ndarray]],
    temporal_rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    ny = np.asarray(NY_VALUES, dtype=np.float64)
    for construction in ("hard", "soft"):
        lambda_means = np.asarray(
            [
                fitted[(construction, int(value))]["W3_3Ny_to_4Ny_levels"][:, 0].mean()
                for value in ny
            ]
        )
        means = np.repeat(lambda_means[:, None], FIT_LEVEL_COUNT, axis=1)
        central = _finite_fit_from_means(ny, means)
        boot_a0 = np.empty(bootstrap_count)
        for replicate in range(bootstrap_count):
            sampled_lambda = []
            for value in ny:
                values = fitted[(construction, int(value))]["W3_3Ny_to_4Ny_levels"][:, 0]
                chosen = rng.integers(0, values.size, size=values.size)
                sampled_lambda.append(values[chosen].mean())
            sampled = np.repeat(np.asarray(sampled_lambda)[:, None], FIT_LEVEL_COUNT, axis=1)
            fit = _finite_fit_from_means(ny, sampled)
            boot_a0[replicate] = fit["primary"]["A0"]
        a0_low, a0_high = percentile_interval(boot_a0)
        a0 = central["primary"]["A0"]
        alternatives = (central["subleading"]["A0"], central["omit_Ny20"]["A0"])
        sensitivity = max(abs(value - a0) / abs(a0) for value in alternatives) if a0 else math.inf
        temporal_lambda_pass = all(
            row["passes"]
            for row in temporal_rows
            if row["construction"] == construction
            and row["metric"] == "lambda0"
            and row["earlier_window"] == "W2_2Ny_to_3Ny"
        )
        central["primary"].update(
            {
                "A0_ci95": [a0_low, a0_high],
                "alpha_c_eff_ci95": [
                    -6.0 * a0_high / math.pi,
                    -6.0 * a0_low / math.pi,
                ],
                "sensitivity_fraction": float(sensitivity),
                "temporal_gate_passes": bool(temporal_lambda_pass),
                "release_gate_passes": bool(
                    temporal_lambda_pass
                    and a0 < 0.0
                    and a0_high < 0.0
                    and sensitivity <= FINITE_SIZE_SENSITIVITY_LIMIT
                ),
            }
        )
        rows.extend(
            [
                {
                    "construction": construction,
                    "quantity": "leading",
                    "model": "f_inf+A0/Ny^2",
                    "coefficient": a0,
                    "ci_low": a0_low,
                    "ci_high": a0_high,
                    "derived_name": "alpha_c_eff",
                    "derived_value": central["primary"]["alpha_c_eff"],
                    "release_gate_passes": central["primary"]["release_gate_passes"],
                },
                {
                    "construction": construction,
                    "quantity": "leading",
                    "model": "f_inf+A0/Ny^2+B0/Ny^4 (exactly determined)",
                    "coefficient": central["subleading"]["A0"],
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "derived_name": "alpha_c_eff",
                    "derived_value": -6.0 * central["subleading"]["A0"] / math.pi,
                    "release_gate_passes": False,
                },
                {
                    "construction": construction,
                    "quantity": "leading",
                    "model": "f_inf+A0/Ny^2 omit Ny20 (two point)",
                    "coefficient": central["omit_Ny20"]["A0"],
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "derived_name": "alpha_c_eff",
                    "derived_value": -6.0 * central["omit_Ny20"]["A0"] / math.pi,
                    "release_gate_passes": False,
                },
            ]
        )
        central["gaps"] = []
        x = 1.0 / ny**2
        for level in range(1, FIT_LEVEL_COUNT):
            values_by_ny = []
            resolved_fractions = []
            for value in ny:
                slopes = fitted[(construction, int(value))]["W3_3Ny_to_4Ny_levels"]
                gaps = slopes[:, 0] - slopes[:, level]
                finite_gaps = gaps[np.isfinite(gaps)]
                values_by_ny.append(finite_gaps)
                resolved_fractions.append(float(finite_gaps.size / gaps.size))
            enough = all(values.size > 0 for values in values_by_ny)
            if enough:
                density = np.asarray([values.mean() for values in values_by_ny]) / ny
                primary_gap = _fit_line(x, density, intercept=False)
                unconstrained_gap = _fit_line(x, density, intercept=True)
                sub_gap = _fit_subleading(x, density, intercept=False)
                omit_gap = _fit_line(x[1:], density[1:], intercept=False)
                boot_ai = np.empty(bootstrap_count)
                for replicate in range(bootstrap_count):
                    boot_density = []
                    for value, values in zip(ny, values_by_ny):
                        selected = rng.integers(0, values.size, size=values.size)
                        boot_density.append(values[selected].mean() / value)
                    boot_ai[replicate] = _fit_line(
                        x, np.asarray(boot_density), intercept=False
                    )["coefficient"]
                low, high = percentile_interval(boot_ai)
                ai = primary_gap["coefficient"]
                gap_sensitivity = max(
                    abs(sub_gap["coefficient"] - ai),
                    abs(omit_gap["coefficient"] - ai),
                ) / abs(ai) if ai else math.inf
                gap = {
                    "level": level,
                    "Ai": ai,
                    "Ai_ci95": [low, high],
                    "alpha_x": ai / (2.0 * math.pi),
                    "alpha_x_ci95": [low / (2.0 * math.pi), high / (2.0 * math.pi)],
                    "unconstrained_intercept": unconstrained_gap["intercept"],
                    "unconstrained_Ai": unconstrained_gap["coefficient"],
                    "subleading_Ai": sub_gap["coefficient"],
                    "subleading_Bi": sub_gap["subleading_coefficient"],
                    "omit_Ny20_Ai": omit_gap["coefficient"],
                    "density": density.tolist(),
                    "resolved_fraction_by_Ny": resolved_fractions,
                }
            else:
                low = high = math.nan
                ai = math.nan
                gap_sensitivity = math.inf
                gap = {
                    "level": level,
                    "Ai": None,
                    "Ai_ci95": None,
                    "alpha_x": None,
                    "alpha_x_ci95": None,
                    "unconstrained_intercept": None,
                    "unconstrained_Ai": None,
                    "subleading_Ai": None,
                    "subleading_Bi": None,
                    "omit_Ny20_Ai": None,
                    "density": None,
                    "resolved_fraction_by_Ny": resolved_fractions,
                }
            temporal_gap_pass = all(
                row["passes"]
                for row in temporal_rows
                if row["construction"] == construction
                and row["metric"] == f"gap{level}"
                and row["earlier_window"] == "W2_2Ny_to_3Ny"
            )
            gap["sensitivity_fraction"] = float(gap_sensitivity)
            gap["temporal_gate_passes"] = bool(temporal_gap_pass)
            gap["release_gate_passes"] = bool(
                enough
                and min(resolved_fractions) >= MINIMUM_RESOLVED_FRACTION
                and temporal_gap_pass
                and low > 0.0
                and gap_sensitivity <= FINITE_SIZE_SENSITIVITY_LIMIT
            )
            gap["x_over_c_eff"] = (
                -ai / (12.0 * a0)
                if central["primary"]["release_gate_passes"] and gap["release_gate_passes"]
                else None
            )
            rows.append(
                {
                    "construction": construction,
                    "quantity": f"gap{level}",
                    "model": "Ai/Ny^2",
                    "coefficient": ai,
                    "ci_low": low,
                    "ci_high": high,
                    "derived_name": "alpha_x",
                    "derived_value": gap["alpha_x"],
                    "resolved_fraction_Ny20": resolved_fractions[0],
                    "resolved_fraction_Ny30": resolved_fractions[1],
                    "resolved_fraction_Ny40": resolved_fractions[2],
                    "release_gate_passes": gap["release_gate_passes"],
                }
            )
            central["gaps"].append(gap)
        summaries[construction] = central
    return summaries, rows


def sample_convergence(
    fitted: dict[tuple[str, int], dict[str, np.ndarray]],
    *,
    rng: np.random.Generator,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ny = np.asarray(NY_VALUES, dtype=np.float64)
    for construction in ("hard", "soft"):
        for count in (25, 50, 75, 100):
            draws = 1 if count == 100 else repeats
            estimates = {"A0": [], **{f"Ai{level}": [] for level in range(1, FIT_LEVEL_COUNT)}}
            for _ in range(draws):
                selected_lambda = []
                selected_gaps = {level: [] for level in range(1, FIT_LEVEL_COUNT)}
                for value in ny:
                    samples = fitted[(construction, int(value))]["W3_3Ny_to_4Ny_levels"]
                    selected = rng.choice(samples.shape[0], size=count, replace=False)
                    selected_lambda.append(samples[selected, 0].mean())
                    for level in range(1, FIT_LEVEL_COUNT):
                        gaps = samples[:, 0] - samples[:, level]
                        finite = gaps[np.isfinite(gaps)]
                        if finite.size >= count:
                            chosen_gap = rng.choice(finite.size, size=count, replace=False)
                            selected_gaps[level].append(finite[chosen_gap].mean())
                dummy = np.repeat(np.asarray(selected_lambda)[:, None], FIT_LEVEL_COUNT, axis=1)
                fit = _finite_fit_from_means(ny, dummy)
                estimates["A0"].append(fit["primary"]["A0"])
                for level in range(1, FIT_LEVEL_COUNT):
                    if len(selected_gaps[level]) == len(ny):
                        density = np.asarray(selected_gaps[level]) / ny
                        estimates[f"Ai{level}"].append(
                            _fit_line(1.0 / ny**2, density, intercept=False)["coefficient"]
                        )
            for metric, values in estimates.items():
                values = np.asarray(values, dtype=np.float64)
                low, high = (
                    percentile_interval(values)
                    if values.size > 1
                    else ((float(values[0]), float(values[0])) if values.size else (math.nan, math.nan))
                )
                rows.append(
                    {
                        "construction": construction,
                        "samples_per_Ny": count,
                        "metric": metric,
                        "mean": float(values.mean()) if values.size else None,
                        "ci_low": low,
                        "ci_high": high,
                        "subset_repeats": int(values.size),
                        "available": bool(values.size),
                        "selection": (
                            "random_without_replacement"
                            if count < 100
                            else "full_ensemble"
                        ),
                    }
                )
    return rows


def cycle_and_endpoint_summaries(
    arrays: dict[tuple[str, int], dict[str, np.ndarray]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cycle_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    wall_columns = np.asarray([4, 5, 6, 14, 15, 16], dtype=np.int64)
    for (construction, ny), data in sorted(arrays.items()):
        cycles = np.arange(4 * ny + 1)
        ell0 = data["levels"][:, :, 0]
        for cycle in cycles:
            cycle_rows.append(
                {
                    "construction": construction,
                    "Ny": ny,
                    "cycle": int(cycle),
                    "normalized_cycle": float(cycle / ny),
                    "ell0_mean": float(ell0[:, cycle].mean()),
                    "ell0_sem": sem(ell0[:, cycle]),
                    "omega_mean": float(data["omega"][:, cycle].mean()),
                    "omega_sem": sem(data["omega"][:, cycle]),
                    "entropy_mean": float(data["total_entropy"][:, cycle].mean()),
                    "entropy_sem": sem(data["total_entropy"][:, cycle]),
                    "charge_variance_mean": float(
                        data["total_charge_variance"][:, cycle].mean()
                    ),
                    "charge_variance_sem": sem(data["total_charge_variance"][:, cycle]),
                    "charge_mean": float(data["total_charge"][:, cycle].mean()),
                    "charge_sem": sem(data["total_charge"][:, cycle]),
                }
            )
        entropy_profile = data["endpoint_entropy_x"]
        variance_profile = data["endpoint_variance_x"]
        entropy_total = entropy_profile.sum(axis=1)
        variance_total = variance_profile.sum(axis=1)
        entropy_wall_fraction = np.divide(
            entropy_profile[:, wall_columns].sum(axis=1),
            entropy_total,
            out=np.full(100, np.nan),
            where=entropy_total > 1.0e-12,
        )
        variance_wall_fraction = np.divide(
            variance_profile[:, wall_columns].sum(axis=1),
            variance_total,
            out=np.full(100, np.nan),
            where=variance_total > 1.0e-12,
        )
        endpoint_rows.append(
            {
                "construction": construction,
                "Ny": ny,
                "samples": 100,
                "cycle": 4 * ny,
                "entropy_mean": float(entropy_total.mean()),
                "entropy_sem": sem(entropy_total),
                "charge_variance_mean": float(variance_total.mean()),
                "charge_variance_sem": sem(variance_total),
                "charge_mean": float(data["total_charge"][:, -1].mean()),
                "charge_sem": sem(data["total_charge"][:, -1]),
                "wall_window_columns": "4,5,6,14,15,16",
                "entropy_wall_fraction_mean": float(np.nanmean(entropy_wall_fraction)),
                "variance_wall_fraction_mean": float(np.nanmean(variance_wall_fraction)),
                "wall_fraction_is_descriptive_not_mode_resolved": True,
            }
        )
        for x in range(20):
            profile_rows.append(
                {
                    "construction": construction,
                    "Ny": ny,
                    "x": x,
                    "endpoint_entropy_profile_mean": float(entropy_profile[:, x].mean()),
                    "endpoint_entropy_profile_sem": sem(entropy_profile[:, x]),
                    "endpoint_variance_profile_mean": float(variance_profile[:, x].mean()),
                    "endpoint_variance_profile_sem": sem(variance_profile[:, x]),
                }
            )
    return cycle_rows, endpoint_rows, profile_rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    cleaned = []
    for row in rows:
        cleaned.append(
            {
                key: (
                    None
                    if isinstance(value, (float, np.floating))
                    and not math.isfinite(float(value))
                    else value
                )
                for key, value in row.items()
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in cleaned:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cleaned)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, (np.integer, np.bool_)):
            return value.item()
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(float(value)) else None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


COLORS = {20: "#d62728", 30: "#2ca02c", 40: "#1f77b4"}
MARKERS = {20: "^", 30: "s", 40: "o"}
LINESTYLES = {20: ":", 30: "--", 40: "-"}


def _panel_label(axis: Any, label: str) -> None:
    axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold", va="bottom")


def make_main_figure(
    *,
    output_root: Path,
    fitted: dict[tuple[str, int], dict[str, np.ndarray]],
    temporal_rows: list[dict[str, Any]],
    closure_rows: list[dict[str, Any]],
    finite: dict[str, Any],
    cycle_rows: list[dict[str, Any]],
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.65), constrained_layout=True)
    window_centers = np.asarray([1.5, 2.5, 3.5])
    window_names = list(WINDOWS)
    for construction, filled, construction_label in (
        ("hard", True, "hard"),
        ("soft", False, "soft"),
    ):
        for ny in NY_VALUES:
            means = np.asarray(
                [
                    -fitted[(construction, ny)][f"{window}_levels"][:, 0].mean() / ny
                    for window in window_names
                ]
            )
            errors = np.asarray(
                [
                    sem(-fitted[(construction, ny)][f"{window}_levels"][:, 0] / ny)
                    for window in window_names
                ]
            )
            axes[0, 0].errorbar(
                window_centers,
                means,
                yerr=errors,
                color=COLORS[ny],
                marker=MARKERS[ny],
                linestyle=LINESTYLES[ny] if filled else "none",
                markerfacecolor=COLORS[ny] if filled else "white",
                markeredgecolor=COLORS[ny],
                capsize=2,
                label=f"{construction_label}, $N_y={ny}$",
            )
    axes[0, 0].set(xlabel="fit-window center $t/N_y$", ylabel=r"$-\langle\lambda_0\rangle/N_y$")
    axes[0, 0].legend(fontsize=6, ncol=2)
    _panel_label(axes[0, 0], "(a)")

    selected_temporal = [
        row for row in temporal_rows if row["earlier_window"] == "W2_2Ny_to_3Ny"
    ]
    metric_x = {name: index for index, name in enumerate(("lambda0", "gap1", "gap2", "gap3", "gap4"))}
    offsets = {20: -0.16, 30: 0.0, 40: 0.16}
    for row in selected_temporal:
        if not math.isfinite(float(row["relative_shift"])):
            continue
        x = metric_x[row["metric"]] + offsets[row["Ny"]]
        marker = MARKERS[row["Ny"]]
        axes[0, 1].scatter(
            x,
            row["relative_shift"],
            color=COLORS[row["Ny"]] if row["construction"] == "hard" else "white",
            edgecolor=COLORS[row["Ny"]],
            marker=marker,
            s=20,
        )
    axes[0, 1].axhline(TEMPORAL_RELATIVE_LIMIT, color="0.35", linestyle="--", linewidth=0.8)
    axes[0, 1].set_xticks(range(5), [r"$\lambda_0$", r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_3$", r"$\Delta_4$"])
    axes[0, 1].set(ylabel=r"relative $W_2\to W_3$ shift")
    axes[0, 1].set_yscale("log")
    _panel_label(axes[0, 1], "(b)")

    late_closure = [row for row in closure_rows if row["window"] == "W3_3Ny_to_4Ny"]
    all_values = []
    for row in late_closure:
        x = -row["lambda0_mean"]
        y = -row["omega_slope_mean"]
        all_values.extend((x, y))
        axes[0, 2].errorbar(
            x,
            y,
            xerr=row["lambda0_sem"],
            yerr=row["omega_slope_sem"],
            marker=MARKERS[row["Ny"]],
            color=COLORS[row["Ny"]],
            markerfacecolor=COLORS[row["Ny"]] if row["construction"] == "hard" else "white",
            linestyle="none",
            capsize=2,
        )
    lower, upper = min(all_values), max(all_values)
    axes[0, 2].plot([lower, upper], [lower, upper], color="0.3", linestyle="--", linewidth=0.8)
    axes[0, 2].set(xlabel=r"$-\langle\lambda_0\rangle_{W_3}$", ylabel=r"$-\langle d\omega/dt\rangle_{W_3}$")
    _panel_label(axes[0, 2], "(c)")

    dense_x = np.linspace(0.0, 1.0 / 20**2 * 1.05, 200)
    for construction, marker, linestyle in (("hard", "o", "-"), ("soft", "s", "--")):
        result = finite[construction]
        x = np.asarray(result["inverse_Ny_squared"])
        y = np.asarray(result["f0_tilde"])
        fit = result["primary"]
        axes[1, 0].plot(dense_x, fit["f_infinity"] + fit["A0"] * dense_x, linestyle=linestyle, color="0.25")
        axes[1, 0].scatter(x, y, marker=marker, facecolor="0.25" if construction == "hard" else "white", edgecolor="0.25", label=construction)
    axes[1, 0].set(xlabel=r"$1/N_y^2$", ylabel=r"$-\langle\lambda_0\rangle/N_y$")
    axes[1, 0].legend()
    _panel_label(axes[1, 0], "(d)")

    positions = np.arange(1, FIT_LEVEL_COUNT)
    for construction, offset, marker in (("hard", -0.10, "o"), ("soft", 0.10, "s")):
        selected = [
            row
            for row in temporal_rows
            if row["construction"] == construction
            and row["earlier_window"] == "W2_2Ny_to_3Ny"
            and row["metric"].startswith("gap")
        ]
        for ny in NY_VALUES:
            local = [row for row in selected if row["Ny"] == ny]
            axes[1, 1].plot(
                positions + offset,
                [row["resolved_fraction"] for row in local],
                marker=MARKERS[ny],
                color=COLORS[ny],
                markerfacecolor=COLORS[ny] if construction == "hard" else "white",
                linestyle="none",
                markersize=4,
            )
    axes[1, 1].axhline(MINIMUM_RESOLVED_FRACTION, color="0.5", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set(xlabel="ordered gap $i$", ylabel="$W_2/W_3$ resolved fraction", ylim=(-0.03, 1.04))
    _panel_label(axes[1, 1], "(e)")

    for construction, linestyle in (("hard", "-"), ("soft", "--")):
        for ny in NY_VALUES:
            selected = [row for row in cycle_rows if row["construction"] == construction and row["Ny"] == ny]
            x = np.asarray([row["normalized_cycle"] for row in selected])
            y = np.asarray([row["entropy_mean"] / ny for row in selected])
            axes[1, 2].plot(x, y, color=COLORS[ny], linestyle=linestyle, linewidth=1.0)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set(xlabel=r"cycle $t/N_y$", ylabel=r"$\langle S\rangle/N_y$")
    _panel_label(axes[1, 2], "(f)")

    figure_dir = output_root / "figure_assets"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "purification_lyapunov_main.pdf")
    fig.savefig(figure_dir / "purification_lyapunov_main.png", dpi=300)
    plt.close(fig)


def make_diagnostic_figure(
    *,
    output_root: Path,
    cycle_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    numerical_rows: list[dict[str, Any]],
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.6), constrained_layout=True)
    for construction, linestyle in (("hard", "-"), ("soft", "--")):
        for ny in NY_VALUES:
            selected = [row for row in cycle_rows if row["construction"] == construction and row["Ny"] == ny]
            x = np.asarray([row["normalized_cycle"] for row in selected])
            y = np.asarray([row["charge_variance_mean"] / ny for row in selected])
            axes[0, 0].plot(x, y, color=COLORS[ny], linestyle=linestyle, linewidth=1.0)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel=r"cycle $t/N_y$", ylabel=r"$\langle\mathrm{Var}_Q\rangle/N_y$")
    _panel_label(axes[0, 0], "(a)")

    for construction, linestyle in (("hard", "-"), ("soft", "--")):
        for ny in NY_VALUES:
            selected = [row for row in profile_rows if row["construction"] == construction and row["Ny"] == ny]
            axes[0, 1].plot(
                [row["x"] for row in selected],
                [row["endpoint_entropy_profile_mean"] for row in selected],
                color=COLORS[ny],
                linestyle=linestyle,
                linewidth=1.0,
            )
    axes[0, 1].axvline(5, color="0.5", linestyle=":", linewidth=0.7)
    axes[0, 1].axvline(15, color="0.5", linestyle=":", linewidth=0.7)
    axes[0, 1].set(xlabel="$x$", ylabel=r"endpoint entropy summed over $y$")
    _panel_label(axes[0, 1], "(b)")

    for construction, marker in (("hard", "o"), ("soft", "s")):
        selected = [row for row in sample_rows if row["construction"] == construction and row["metric"] == "A0"]
        axes[1, 0].errorbar(
            [row["samples_per_Ny"] for row in selected],
            [row["mean"] for row in selected],
            yerr=np.asarray(
                [
                    [row["mean"] - row["ci_low"] for row in selected],
                    [row["ci_high"] - row["mean"] for row in selected],
                ]
            ),
            marker=marker,
            markerfacecolor="0.2" if construction == "hard" else "white",
            color="0.2",
            capsize=2,
            label=construction,
        )
    axes[1, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axes[1, 0].set(xlabel="trajectories per $N_y$", ylabel="$A_0$")
    axes[1, 0].legend()
    _panel_label(axes[1, 0], "(c)")

    residual_names = (
        "record_accumulation_error",
        "occupation_bound_residual",
        "entropy_closure_error",
        "charge_closure_error",
        "final_covariance_hermiticity_residual",
    )
    for index, construction in enumerate(("hard", "soft")):
        values = [
            max(row[name] for row in numerical_rows if row["construction"] == construction)
            for name in residual_names
        ]
        axes[1, 1].scatter(
            np.arange(len(residual_names)) + (-0.08 if construction == "hard" else 0.08),
            np.maximum(values, 1.0e-18),
            marker="o" if construction == "hard" else "s",
            facecolor="0.2" if construction == "hard" else "white",
            edgecolor="0.2",
            label=construction,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(
        np.arange(len(residual_names)), ["record", "bounds", "$S$ closure", "$Q$ closure", "$G$ herm."], rotation=20
    )
    axes[1, 1].set(ylabel="maximum residual")
    axes[1, 1].legend()
    _panel_label(axes[1, 1], "(d)")

    figure_dir = output_root / "figure_assets"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "purification_lyapunov_diagnostics.pdf")
    fig.savefig(figure_dir / "purification_lyapunov_diagnostics.png", dpi=300)
    plt.close(fig)


def run_analysis(
    *,
    inventory_path: Path,
    output_root: Path,
    bootstrap_count: int = BOOTSTRAP_REPLICATES,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    arrays, numerical_rows, provenance = load_and_verify(
        inventory_path=inventory_path, verify_hashes=verify_hashes
    )
    trajectory_rows, fitted = trajectory_slopes(arrays)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    temporal_rows = temporal_stability(
        fitted, bootstrap_count=bootstrap_count, rng=rng
    )
    depth_rows, depth_summary = legacy_depth_comparison(temporal_rows)
    closure_rows = record_closure(fitted, bootstrap_count=bootstrap_count, rng=rng)
    finite, finite_rows = finite_size_analysis(
        fitted, temporal_rows, bootstrap_count=bootstrap_count, rng=rng
    )
    sample_rows = sample_convergence(fitted, rng=rng, repeats=SUBSET_REPEATS)
    cycle_rows, endpoint_rows, profile_rows = cycle_and_endpoint_summaries(arrays)

    late_temporal = [
        row for row in temporal_rows if row["earlier_window"] == "W2_2Ny_to_3Ny"
    ]
    late_closure = [row for row in closure_rows if row["window"] == "W3_3Ny_to_4Ny"]
    summary: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": bootstrap_count,
        "independent_sampling_unit": "whole Born trajectory",
        "provenance": provenance,
        "analysis_contract": {
            "leading_levels_reconstructed": LEADING_LEVEL_COUNT,
            "fitted_levels": FIT_LEVEL_COUNT,
            "squared_singular_value_convention": "ell_i=log(sigma_i^2)",
            "windows": {key: list(value) for key, value in WINDOWS.items()},
            "temporal_relative_limit": TEMPORAL_RELATIVE_LIMIT,
            "record_closure_limit": RECORD_CLOSURE_LIMIT,
            "finite_size_sensitivity_limit": FINITE_SIZE_SENSITIVITY_LIMIT,
            "minimum_resolved_fraction": MINIMUM_RESOLVED_FRACTION,
            "hard_N_eff": "22*Ny",
            "soft_N_eff": "40*Ny",
            "exact_caps": "infinite flip cost; no finite clipping",
        },
        "temporal_acceptance": {
            construction: {
                str(ny): {
                    row["metric"]: bool(row["passes"])
                    for row in late_temporal
                    if row["construction"] == construction and row["Ny"] == ny
                }
                for ny in NY_VALUES
            }
            for construction in ("hard", "soft")
        },
        "all_temporal_metrics_pass": {
            construction: bool(
                all(row["passes"] for row in late_temporal if row["construction"] == construction)
            )
            for construction in ("hard", "soft")
        },
        "legacy_2Ny_depth_comparison": depth_summary,
        "late_record_closure": {
            construction: {
                str(row["Ny"]): row
                for row in late_closure
                if row["construction"] == construction
            }
            for construction in ("hard", "soft")
        },
        "finite_size": finite,
        "claim_policy": {
            "absolute_c_eff_released": False,
            "absolute_operator_dimensions_released": False,
            "reason": "alpha is not independently calibrated and only three circumferences are available",
            "mode_labels_released": False,
            "mode_label_reason": "occupation eigenvectors and symmetry labels were not saved",
        },
        "endpoint_purification": endpoint_rows,
        "maximum_numerical_residuals": {
            construction: {
                key: max(row[key] for row in numerical_rows if row["construction"] == construction)
                for key in (
                    "record_accumulation_error",
                    "occupation_bound_residual",
                    "hermiticity_residual",
                    "entropy_closure_error",
                    "charge_closure_error",
                    "final_covariance_hermiticity_residual",
                )
            }
            for construction in ("hard", "soft")
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "trajectory_window_slopes.csv", trajectory_rows)
    write_csv(output_root / "temporal_stability.csv", temporal_rows)
    write_csv(output_root / "depth_comparison.csv", depth_rows)
    write_csv(output_root / "record_weight_closure.csv", closure_rows)
    write_csv(output_root / "finite_size_fits.csv", finite_rows)
    write_csv(output_root / "sample_convergence.csv", sample_rows)
    write_csv(output_root / "cycle_summary.csv", cycle_rows)
    write_csv(output_root / "endpoint_purification.csv", endpoint_rows)
    write_csv(output_root / "endpoint_profiles.csv", profile_rows)
    write_csv(output_root / "numerical_diagnostics.csv", numerical_rows)
    write_json(output_root / "analysis_summary.json", summary)
    make_main_figure(
        output_root=output_root,
        fitted=fitted,
        temporal_rows=temporal_rows,
        closure_rows=closure_rows,
        finite=finite,
        cycle_rows=cycle_rows,
    )
    make_diagnostic_figure(
        output_root=output_root,
        cycle_rows=cycle_rows,
        profile_rows=profile_rows,
        sample_rows=sample_rows,
        numerical_rows=numerical_rows,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_REMOTE_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="Development-only shortcut; production analysis must rehash every imported file.",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("bootstrap-replicates must be at least 100")
    summary = run_analysis(
        inventory_path=args.inventory.resolve(),
        output_root=args.output_root.resolve(),
        bootstrap_count=args.bootstrap_replicates,
        verify_hashes=not args.skip_file_hashes,
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "verified_trajectories": summary["provenance"]["verified_trajectories"],
                "all_temporal_metrics_pass": summary["all_temporal_metrics_pass"],
                "output_root": str(args.output_root.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
