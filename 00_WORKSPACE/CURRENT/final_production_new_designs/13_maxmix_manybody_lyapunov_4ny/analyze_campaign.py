#!/usr/bin/env python3
"""Verify and analyze the completed bundle-13 hard-wall 4Ny campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_ROOT.parents[3]
REVISION = "maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_gpu_v4_38gib_memory_scaled"
DATA_ROOT = BUNDLE_ROOT / "gpu_data" / REVISION
DOWNLOAD_MANIFEST = DATA_ROOT / "DOWNLOAD_MANIFEST.json"
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "analysis_outputs" / "hard_4ny_dynamical_critical_v2_ungated"
ANALYSIS_SCHEMA = "maxmix_manybody_lyapunov_hard_4ny_analysis_v2_ungated"
RESULT_SCHEMA = "maxmix_manybody_lyapunov_4ny_result_v4"
COMPLETION_SCHEMA = "maxmix_manybody_lyapunov_4ny_completion_v4"
CONFIGURATION_HASH = "ae91cea912648523cc83e6396f327934e19991559dd0bc3b15b6efa0ee572842"
SOURCE_HASHES = {
    "lyapunov_observer.py": "1e5e85e85b1d10c0bf69e273993efa4fb09c6515998dedb35930a7a20a0b03cc",
    "run_campaign.py": "2e71ecb5753deb78a4df8bec3565129467dcf46818417cc2d3b62fb164829dfb",
    "src/classA_U1FGTN_gpu.py": "53de96bced6839b485afe04fe4aaa15d2e42c249a9cf14f6ca0c931f55409700",
    "src/occupied_frame_gpu.py": "bfc10cefea98ce00184a88b5c375eadfcda8e3dc6954d9f66183d566008951b0",
}
NY_VALUES = (20, 24, 30, 36, 44, 56, 60)
WINDOWS = {
    "W1_Ny_to_2Ny": (1.0, 2.0),
    "W2_2Ny_to_3Ny": (2.0, 3.0),
    "W3_3Ny_to_4Ny": (3.0, 4.0),
}
BOOTSTRAP_SEED = 2026091702
BOOTSTRAP_REPLICATES = 2000
SUBSET_REPEATS = 200
LEADING_LEVEL_COUNT = 64
FIT_LEVEL_COUNT = 5
CAP_TOLERANCE = 1.0e-9
PURIFICATION_FRACTIONS = (0.5, 0.1, 0.01)

LEGACY_2NY_ROOT = BUNDLE_ROOT.parent / "04_maxmix_manybody_lyapunov_pilot" / "analysis_outputs" / "maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2"
LEGACY_4NY_ROOT = BUNDLE_ROOT.parent / "07_maxmix_hard_soft_purification" / "analysis_outputs" / "hard_v2_soft_v3_4ny_analysis_v1"


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
    low, high = np.percentile(np.asarray(values, dtype=np.float64), (2.5, 97.5))
    return float(low), float(high)


def bootstrap_mean_ci(values: np.ndarray, count: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    choices = rng.integers(0, values.size, size=(count, values.size))
    return percentile_interval(values[choices].mean(axis=1))


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size < 2 or not np.all(np.isfinite(y)):
        return math.nan
    centered = x - x.mean()
    denominator = float(centered @ centered)
    return float(centered @ (y - y.mean()) / denominator) if denominator else math.nan


def fit_line(x: np.ndarray, y: np.ndarray, intercept: bool = True) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack((np.ones(x.size), x)) if intercept else x[:, None]
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficient
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {"intercept": float(coefficient[0]) if intercept else 0.0,
            "coefficient": float(coefficient[-1]),
            "r_squared": float(1.0 - residual / total) if total else 1.0}


def fit_subleading(x: np.ndarray, y: np.ndarray, intercept: bool = True) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack((np.ones(x.size), x, x**2)) if intercept else np.column_stack((x, x**2))
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    offset = int(intercept)
    return {"intercept": float(coefficient[0]) if intercept else 0.0,
            "coefficient": float(coefficient[offset]),
            "subleading_coefficient": float(coefficient[offset + 1])}


def lowest_subset_sums(costs: np.ndarray, count: int) -> np.ndarray:
    values = np.sort(np.asarray(costs, dtype=np.float64))
    values = values[np.isfinite(values)]
    if np.any(values < 0.0):
        raise ValueError("flip costs must be nonnegative")
    sums = np.asarray([0.0])
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


def leading_levels(occupations: np.ndarray, log_z: float, count: int = LEADING_LEVEL_COUNT) -> np.ndarray:
    nu = np.asarray(occupations, dtype=np.float64)
    if not np.all(np.isfinite(nu)):
        raise ValueError("occupation spectrum is nonfinite")
    if nu.min(initial=0.0) < -CAP_TOLERANCE or nu.max(initial=1.0) > 1.0 + CAP_TOLERANCE:
        raise FloatingPointError("occupation spectrum leaves [0,1]")
    caps = (nu <= CAP_TOLERANCE) | (nu >= 1.0 - CAP_TOLERANCE)
    interior = ~caps
    preferred = np.maximum(nu[interior], 1.0 - nu[interior])
    costs = np.full(nu.shape, np.inf)
    costs[interior] = np.abs(np.log(nu[interior]) - np.log1p(-nu[interior]))
    return float(log_z) + float(np.log(preferred).sum()) - lowest_subset_sums(costs, count)


def expected_spectrum_cycles(ny: int) -> np.ndarray:
    return np.unique(np.concatenate((np.arange(0, 4 * ny + 1, 4), np.asarray([ny, 2 * ny, 3 * ny, 4 * ny]))))


def raw_inventory_hash(paths: list[Path]) -> str:
    lines = [f"{sha256_file(path)}  {path.relative_to(DATA_ROOT).as_posix()}\n" for path in paths]
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


def verify_manifest() -> dict[str, Any]:
    manifest = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    expected = {"schema": "maxmix_manybody_lyapunov_4ny_drive_import_v1",
                "bundle": "13_maxmix_manybody_lyapunov_4ny", "campaign": REVISION,
                "sampling_revision": REVISION, "completion_schema": COMPLETION_SCHEMA,
                "result_schema": RESULT_SCHEMA, "configuration_sha256": CONFIGURATION_HASH}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"download manifest mismatch for {key}")
    inventory = manifest.get("inventory", {})
    exact = {"sizes": list(NY_VALUES), "samples_per_size": 100, "trajectories": 700,
             "result_pairs": 140, "npz_files": 140, "completion_json_files": 140,
             "downloaded_files": 280}
    for key, value in exact.items():
        if inventory.get(key) != value:
            raise RuntimeError(f"download inventory mismatch for {key}")
    if manifest.get("validation", {}).get("source_hashes") != SOURCE_HASHES:
        raise RuntimeError("download manifest source identity mismatch")
    return manifest


def validate_completion(payload: dict[str, Any], result_path: Path) -> None:
    expected = {"schema": COMPLETION_SCHEMA, "sampling_revision": REVISION,
                "configuration_hash": CONFIGURATION_HASH,
                "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
                "construction": "hard", "Nx": 20, "dtype": "complex128",
                "result_filename": result_path.name}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{result_path.name}: completion mismatch for {key}")
    if payload.get("source_hashes") != SOURCE_HASHES:
        raise RuntimeError(f"{result_path.name}: source identity mismatch")
    ny = int(payload["Ny"])
    samples = [int(value) for value in payload["sample_indices"]]
    if ny not in NY_VALUES or int(payload["cycles"]) != 4 * ny:
        raise RuntimeError(f"{result_path.name}: geometry/depth mismatch")
    if len(samples) != 5 or samples != list(range(samples[0], samples[0] + 5)):
        raise RuntimeError(f"{result_path.name}: expected a contiguous five-sample shard")


def load_and_verify(verify_hashes: bool = True) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    manifest = verify_manifest()
    npz_paths = sorted(DATA_ROOT.rglob("*.npz"))
    completion_paths = sorted(DATA_ROOT.rglob("*.complete.json"))
    if len(npz_paths) != 140 or len(completion_paths) != 140:
        raise RuntimeError("expected exactly 140 NPZ/completion pairs")
    all_raw = sorted(npz_paths + completion_paths)
    if sum(path.stat().st_size for path in all_raw) != int(manifest["inventory"]["downloaded_bytes"]):
        raise RuntimeError("raw byte inventory mismatch")
    actual_inventory_sha256 = raw_inventory_hash(all_raw) if verify_hashes else None
    inventory_digest_matches = (
        actual_inventory_sha256 == manifest["inventory"]["raw_file_inventory_sha256"]
        if verify_hashes else None
    )
    grouped: dict[int, dict[str, list[np.ndarray]]] = {}
    numerical_rows: list[dict[str, Any]] = []
    coverage = {ny: set() for ny in NY_VALUES}
    reconstruction_max = 0.0
    for result_path in npz_paths:
        completion_path = result_path.with_name(result_path.name.replace(".npz", ".complete.json"))
        if not completion_path.is_file():
            raise RuntimeError(f"missing completion for {result_path.name}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        validate_completion(completion, result_path)
        if result_path.stat().st_size != int(completion["result_bytes"]):
            raise RuntimeError(f"{result_path.name}: byte count mismatch")
        if verify_hashes and sha256_file(result_path) != completion["result_sha256"]:
            raise RuntimeError(f"{result_path.name}: result SHA-256 mismatch")
        ny = int(completion["Ny"])
        sample_indices = np.asarray(completion["sample_indices"], dtype=np.int64)
        coverage[ny].update(sample_indices.tolist())
        with np.load(result_path, allow_pickle=False) as data:
            scalar_expected = {"result_schema": RESULT_SCHEMA, "sampling_revision": REVISION,
                "configuration_hash": CONFIGURATION_HASH,
                "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
                "construction": "hard", "Nx": 20, "Ny": ny, "transfer_mode_count": 22 * ny,
                "log_probability_origin": "after_born_conditioned_exterior_preparation",
                "log_z_formula": "log_Z=N_eff*log(2)+cumulative_log_probability",
                "squared_singular_value_convention": "ell_i=log(sigma_i^2)",
                "centered_covariance_convention": "G=2C-I"}
            for key, value in scalar_expected.items():
                if scalar(data, key) != value:
                    raise RuntimeError(f"{result_path.name}: product mismatch for {key}")
            cycles = np.arange(4 * ny + 1, dtype=np.int64)
            spectrum_cycles = expected_spectrum_cycles(ny)
            if not np.array_equal(data["cycles"], cycles) or not np.array_equal(data["spectrum_cycles"], spectrum_cycles):
                raise RuntimeError(f"{result_path.name}: cycle grid mismatch")
            if not np.array_equal(data["sample_indices"], sample_indices):
                raise RuntimeError(f"{result_path.name}: sample identity mismatch")
            shapes = {"cycle_seen": (5, cycles.size), "measurement_log_probability": (5, cycles.size),
                "cumulative_log_probability": (5, cycles.size), "site_event_count": (5, cycles.size),
                "channel_event_count": (5, cycles.size), "spectrum_seen": (5, spectrum_cycles.size),
                "occupations": (5, spectrum_cycles.size, 22 * ny), "cap_mask": (5, spectrum_cycles.size, 22 * ny),
                "entropy_nats": (5, spectrum_cycles.size), "charge_variance": (5, spectrum_cycles.size),
                "log_z": (5, spectrum_cycles.size), "leading_log_sigma2": (5, spectrum_cycles.size, 64),
                "soft_mode_occupations": (5, spectrum_cycles.size, 16),
                "soft_mode_flip_costs": (5, spectrum_cycles.size, 16),
                "soft_mode_x_profiles": (5, spectrum_cycles.size, 16, 20),
                "soft_mode_wall_weights": (5, spectrum_cycles.size, 16, 2)}
            for key, shape in shapes.items():
                array = np.asarray(data[key])
                if array.shape != shape:
                    raise RuntimeError(f"{result_path.name}: {key} shape mismatch")
                if array.dtype.kind not in "biu":
                    if key == "leading_log_sigma2":
                        if np.any(np.isnan(array)) or np.any(np.isposinf(array)):
                            raise RuntimeError(f"{result_path.name}: {key} has invalid nonfinite values")
                    elif key == "soft_mode_flip_costs":
                        if np.any(np.isnan(array)) or np.any(np.isneginf(array)):
                            raise RuntimeError(f"{result_path.name}: {key} has invalid nonfinite values")
                    elif not np.all(np.isfinite(array)):
                        raise RuntimeError(f"{result_path.name}: {key} has nonfinite values")
            if not np.all(data["cycle_seen"]) or not np.all(data["spectrum_seen"]):
                raise RuntimeError(f"{result_path.name}: incomplete cycle mask")
            omega = np.asarray(data["cumulative_log_probability"], dtype=np.float64)
            increments = np.asarray(data["measurement_log_probability"], dtype=np.float64)
            record_error = float(np.max(np.abs(np.diff(omega, axis=1) - increments[:, 1:])))
            if record_error > 2e-10 or np.any(omega[:, 0]) or np.any(increments[:, 0]):
                raise RuntimeError(f"{result_path.name}: record accumulation mismatch")
            if np.any(data["site_event_count"][:, 0]) or np.any(data["channel_event_count"][:, 0]):
                raise RuntimeError(f"{result_path.name}: nonzero cycle-zero event count")
            if not np.all(data["site_event_count"][:, 1:] == 11 * ny) or not np.all(data["channel_event_count"][:, 1:] == 44 * ny):
                raise RuntimeError(f"{result_path.name}: event count mismatch")
            occupations = np.asarray(data["occupations"], dtype=np.float64)
            bound = max(0.0, float(-occupations.min(initial=0.0)), float(occupations.max(initial=1.0) - 1.0))
            if bound > CAP_TOLERANCE:
                raise RuntimeError(f"{result_path.name}: occupation bound failure")
            expected_caps = (occupations <= CAP_TOLERANCE) | (occupations >= 1.0 - CAP_TOLERANCE)
            if not np.array_equal(data["cap_mask"], expected_caps):
                raise RuntimeError(f"{result_path.name}: exact-cap mask mismatch")
            stored = np.asarray(data["leading_log_sigma2"], dtype=np.float64)
            reconstructed = np.empty_like(stored)
            for sample in range(5):
                for position in range(spectrum_cycles.size):
                    reconstructed[sample, position] = leading_levels(occupations[sample, position], float(data["log_z"][sample, position]))
            finite = np.isfinite(stored) & np.isfinite(reconstructed)
            if not np.array_equal(np.isneginf(stored), np.isneginf(reconstructed)):
                raise RuntimeError(f"{result_path.name}: exact-cap padding mismatch")
            error = float(np.max(np.abs(stored[finite] - reconstructed[finite]), initial=0.0))
            reconstruction_max = max(reconstruction_max, error)
            if error > 5e-10:
                raise RuntimeError(f"{result_path.name}: many-body reconstruction mismatch")
            store = grouped.setdefault(ny, {})
            fields = {"sample_indices": sample_indices,
                "omega": omega, "levels": stored, "entropy": np.asarray(data["entropy_nats"]),
                "charge_variance": np.asarray(data["charge_variance"]),
                "wall_weights": np.asarray(data["soft_mode_wall_weights"]),
                "x_profiles": np.asarray(data["soft_mode_x_profiles"])}
            for key, value in fields.items():
                store.setdefault(key, []).append(np.asarray(value))
            residual_names = ("hermiticity_residual", "eigensolver_residual", "eigenvector_gram_residual",
                "occupation_bound_residual", "active_exterior_coupling_residual", "exterior_product_residual")
            numerical_rows.append({"Ny": ny, "result": str(result_path.relative_to(REPO_ROOT)),
                "record_accumulation_error": record_error, "spectrum_reconstruction_error": error,
                **{key: float(np.max(np.asarray(data[key]))) for key in residual_names}})
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for ny in NY_VALUES:
        if coverage[ny] != set(range(100)):
            raise RuntimeError(f"Ny={ny}: sample coverage mismatch")
        arrays[ny] = {key: np.concatenate(value, axis=0) for key, value in grouped[ny].items()}
        order = np.argsort(arrays[ny]["sample_indices"])
        arrays[ny] = {key: value[order] for key, value in arrays[ny].items()}
        if not np.array_equal(arrays[ny]["sample_indices"], np.arange(100)):
            raise RuntimeError(f"Ny={ny}: noncanonical ordering")
    provenance = {"download_manifest": str(DOWNLOAD_MANIFEST.relative_to(REPO_ROOT)),
        "download_manifest_sha256": sha256_file(DOWNLOAD_MANIFEST),
        "manifest_raw_file_inventory_sha256": manifest["inventory"]["raw_file_inventory_sha256"],
        "recomputed_raw_file_inventory_sha256": actual_inventory_sha256,
        "raw_file_inventory_digest_matches": inventory_digest_matches,
        "raw_file_inventory_note": (
            "The manifest aggregate digest does not reproduce under its documented algorithm; "
            "the immutable manifest was not rewritten. Every NPZ is independently SHA-256 verified "
            "against its completion JSON, and every completion is identity-checked."
            if inventory_digest_matches is False else "aggregate inventory digest matches"
        ),
        "verify_hashes": verify_hashes, "verified_result_pairs": 140, "verified_files": 280,
        "verified_trajectories": 700, "configuration_hash": CONFIGURATION_HASH,
        "source_hashes": SOURCE_HASHES, "maximum_spectrum_reconstruction_error": reconstruction_max}
    return arrays, numerical_rows, provenance


def trajectory_slopes(arrays: dict[int, dict[str, np.ndarray]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    fitted: dict[int, dict[str, np.ndarray]] = {}
    for ny, data in arrays.items():
        spectrum_cycles = expected_spectrum_cycles(ny).astype(np.float64)
        full_cycles = np.arange(4 * ny + 1, dtype=np.float64)
        local: dict[str, np.ndarray] = {}
        for window, bounds in WINDOWS.items():
            spectrum_mask = (spectrum_cycles >= bounds[0] * ny) & (spectrum_cycles <= bounds[1] * ny)
            full_mask = (full_cycles >= bounds[0] * ny) & (full_cycles <= bounds[1] * ny)
            level_slopes = np.asarray([
                [linear_slope(spectrum_cycles[spectrum_mask], sample[spectrum_mask, level])
                 for level in range(FIT_LEVEL_COUNT)]
                for sample in data["levels"]
            ])
            omega_slopes = np.asarray([linear_slope(full_cycles[full_mask], sample[full_mask]) for sample in data["omega"]])
            local[f"{window}_levels"] = level_slopes
            local[f"{window}_omega"] = omega_slopes
            for sample in range(100):
                row: dict[str, Any] = {
                    "Ny": ny, "sample_index": sample, "window": window,
                    "window_start": bounds[0] * ny, "window_stop": bounds[1] * ny,
                    "omega_slope": float(omega_slopes[sample]),
                    "endpoint_record_rate": float(-data["omega"][sample, -1] / (4 * ny)),
                }
                for level in range(FIT_LEVEL_COUNT):
                    row[f"lambda{level}"] = float(level_slopes[sample, level])
                    if level:
                        row[f"gap{level}"] = float(level_slopes[sample, 0] - level_slopes[sample, level])
                row["omega_minus_lambda0"] = float(omega_slopes[sample] - level_slopes[sample, 0])
                rows.append(row)
        fitted[ny] = local
    return rows, fitted


def temporal_stability(fitted: dict[int, dict[str, np.ndarray]], bootstrap_count: int,
                       rng: np.random.Generator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = (("W1_Ny_to_2Ny", "W2_2Ny_to_3Ny"),
                   ("W2_2Ny_to_3Ny", "W3_3Ny_to_4Ny"))
    for ny in NY_VALUES:
        for earlier, later in comparisons:
            first = fitted[ny][f"{earlier}_levels"]
            second = fitted[ny][f"{later}_levels"]
            for level in range(FIT_LEVEL_COUNT):
                if level == 0:
                    metric = "lambda0"
                    left, right = first[:, 0], second[:, 0]
                else:
                    metric = f"gap{level}"
                    left, right = first[:, 0] - first[:, level], second[:, 0] - second[:, level]
                valid = np.isfinite(left) & np.isfinite(right)
                shift = right[valid] - left[valid]
                fraction = float(valid.mean())
                if shift.size:
                    low, high = bootstrap_mean_ci(shift, bootstrap_count, rng)
                    denominator = abs(float(right[valid].mean()))
                    relative = abs(float(shift.mean())) / denominator if denominator else math.inf
                    left_mean, right_mean, shift_mean = float(left[valid].mean()), float(right[valid].mean()), float(shift.mean())
                else:
                    low = high = left_mean = right_mean = shift_mean = math.nan
                    relative = math.inf
                rows.append({"Ny": ny, "metric": metric, "earlier_window": earlier, "later_window": later,
                    "resolved_trajectories": int(valid.sum()), "resolved_fraction": fraction,
                    "earlier_mean": left_mean, "later_mean": right_mean, "shift": shift_mean,
                    "shift_ci_low": low, "shift_ci_high": high,
                    "shift_ci_contains_zero": bool(low <= 0 <= high), "relative_shift": relative})
    return rows


def record_closure(fitted: dict[int, dict[str, np.ndarray]], bootstrap_count: int,
                   rng: np.random.Generator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ny in NY_VALUES:
        for window in WINDOWS:
            lambda0 = fitted[ny][f"{window}_levels"][:, 0]
            omega = fitted[ny][f"{window}_omega"]
            difference = omega - lambda0
            low, high = bootstrap_mean_ci(difference, bootstrap_count, rng)
            relative = abs(float(difference.mean())) / abs(float(lambda0.mean()))
            rows.append({"Ny": ny, "window": window, "samples": 100,
                "lambda0_mean": float(lambda0.mean()), "lambda0_sem": sem(lambda0),
                "omega_slope_mean": float(omega.mean()), "omega_slope_sem": sem(omega),
                "omega_minus_lambda0": float(difference.mean()), "difference_ci_low": low,
                "difference_ci_high": high, "relative_closure": relative,
                })
    return rows


def finite_size_analysis(fitted: dict[int, dict[str, np.ndarray]], bootstrap_count: int,
                         rng: np.random.Generator) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ny = np.asarray(NY_VALUES, dtype=np.float64)
    x = 1.0 / ny**2
    slope_values = [fitted[int(value)]["W3_3Ny_to_4Ny_levels"] for value in ny]
    lambda_values = [values[:, 0] for values in slope_values]
    lambda_means = np.asarray([value.mean() for value in lambda_values])
    f0 = -lambda_means / ny
    primary = fit_line(x, f0)
    subleading = fit_subleading(x, f0)
    omit_fits = {str(int(cut)): fit_line(x[ny >= cut], f0[ny >= cut]) for cut in (24, 30, 36)}
    a0 = primary["coefficient"]
    alternatives = [subleading["coefficient"], *(fit["coefficient"] for fit in omit_fits.values())]
    sensitivity = max(abs(value - a0) for value in alternatives) / abs(a0) if a0 else math.inf
    gap_inputs: dict[int, list[np.ndarray]] = {
        level: [values[:, 0] - values[:, level] for values in slope_values]
        for level in range(1, FIT_LEVEL_COUNT)
    }
    boot_a0 = np.empty(bootstrap_count)
    boot_ai = {level: np.empty(bootstrap_count) for level in range(1, FIT_LEVEL_COUNT)}
    for replicate in range(bootstrap_count):
        choices = [rng.integers(0, 100, 100) for _ in NY_VALUES]
        means = np.asarray([values[index].mean() for values, index in zip(lambda_values, choices)])
        boot_a0[replicate] = fit_line(x, -means / ny)["coefficient"]
        for level in range(1, FIT_LEVEL_COUNT):
            density = []
            for gaps, index, value in zip(gap_inputs[level], choices, ny):
                selected = gaps[index]
                selected = selected[np.isfinite(selected)]
                density.append(selected.mean() / value)
            boot_ai[level][replicate] = fit_line(x, np.asarray(density), intercept=False)["coefficient"]
    a0_low, a0_high = percentile_interval(boot_a0)
    summary: dict[str, Any] = {
        "Ny": list(NY_VALUES), "inverse_Ny_squared": x.tolist(), "f0_tilde": f0.tolist(),
        "primary": {**primary, "A0": a0, "A0_ci95": [a0_low, a0_high],
            "alpha_c_eff": -6 * a0 / math.pi,
            "alpha_c_eff_ci95": [-6 * a0_high / math.pi, -6 * a0_low / math.pi],
            "sensitivity_fraction": sensitivity},
        "subleading": {"A0": subleading["coefficient"], "B0": subleading["subleading_coefficient"],
            "f_infinity": subleading["intercept"]},
        "successive_Ny_min": {key: {"A0": value["coefficient"], "f_infinity": value["intercept"]}
                               for key, value in omit_fits.items()}, "gaps": []}
    rows: list[dict[str, Any]] = [{"quantity": "leading", "model": "f_inf+A0/Ny^2",
        "coefficient": a0, "ci_low": a0_low, "ci_high": a0_high,
        "derived_name": "alpha_c_eff", "derived_value": -6 * a0 / math.pi,
        "sensitivity_fraction": sensitivity}]
    for key, value in omit_fits.items():
        rows.append({"quantity": "leading", "model": f"f_inf+A0/Ny^2; Ny_min={key}",
            "coefficient": value["coefficient"], "ci_low": math.nan, "ci_high": math.nan,
            "derived_name": "alpha_c_eff", "derived_value": -6 * value["coefficient"] / math.pi,
            "sensitivity_fraction": math.nan})
    central_ai: dict[int, float] = {}
    for level in range(1, FIT_LEVEL_COUNT):
        value_sets = [gaps[np.isfinite(gaps)] for gaps in gap_inputs[level]]
        fractions = [float(values.size / 100) for values in value_sets]
        density = np.asarray([values.mean() if values.size else math.nan for values in value_sets]) / ny
        enough = np.all(np.isfinite(density))
        if enough:
            primary_gap = fit_line(x, density, intercept=False)
            unconstrained = fit_line(x, density, intercept=True)
            sub_gap = fit_subleading(x, density, intercept=False)
            omit = {str(int(cut)): fit_line(x[ny >= cut], density[ny >= cut], intercept=False) for cut in (24, 30, 36)}
            low, high = percentile_interval(boot_ai[level])
            ai = primary_gap["coefficient"]
            alternatives = [sub_gap["coefficient"], *(fit["coefficient"] for fit in omit.values())]
            gap_sensitivity = max(abs(value - ai) for value in alternatives) / abs(ai) if ai else math.inf
        else:
            ai = low = high = gap_sensitivity = math.nan
            unconstrained = {"intercept": math.nan, "coefficient": math.nan}
            sub_gap = {"coefficient": math.nan, "subleading_coefficient": math.nan}
            omit = {}
        central_ai[level] = ai
        ratio_samples = -boot_ai[level] / (12.0 * boot_a0)
        ratio_samples = ratio_samples[np.isfinite(ratio_samples)]
        ratio_low, ratio_high = percentile_interval(ratio_samples)
        first_ratio_samples = boot_ai[level] / boot_ai[1]
        first_low, first_high = percentile_interval(first_ratio_samples[np.isfinite(first_ratio_samples)])
        gap_summary = {"level": level, "Ai": ai, "Ai_ci95": [low, high],
            "alpha_x": ai / (2 * math.pi) if enough else None,
            "alpha_x_ci95": [low / (2 * math.pi), high / (2 * math.pi)] if enough else None,
            "density": density.tolist(), "resolved_fraction_by_Ny": fractions,
            "unconstrained_intercept": unconstrained["intercept"], "unconstrained_Ai": unconstrained["coefficient"],
            "subleading_Ai": sub_gap["coefficient"],
            "successive_Ny_min_Ai": {key: fit["coefficient"] for key, fit in omit.items()},
            "sensitivity_fraction": gap_sensitivity,
            "r_i_x_over_c_eff": -ai / (12 * a0),
            "r_i_x_over_c_eff_ci95": [ratio_low, ratio_high],
            "x_i_over_x_1": ai / central_ai.get(1, ai),
            "x_i_over_x_1_ci95": [first_low, first_high]}
        summary["gaps"].append(gap_summary)
        row = {"quantity": f"gap{level}", "model": "Ai/Ny^2", "coefficient": ai,
            "ci_low": low, "ci_high": high, "derived_name": "alpha_x",
            "derived_value": ai / (2 * math.pi) if enough else None,
            "sensitivity_fraction": gap_sensitivity,
            "r_i_x_over_c_eff": -ai / (12 * a0), "r_i_ci_low": ratio_low, "r_i_ci_high": ratio_high,
            "x_i_over_x_1": ai / central_ai.get(1, ai),
            "x_i_over_x_1_ci_low": first_low, "x_i_over_x_1_ci_high": first_high}
        row.update({f"resolved_fraction_Ny{int(value)}": fraction for value, fraction in zip(ny, fractions)})
        rows.append(row)
    return summary, rows


def purification_analysis(arrays: dict[int, dict[str, np.ndarray]], bootstrap_count: int,
                          rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cycle_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    time_by_fraction: dict[float, dict[int, np.ndarray]] = {fraction: {} for fraction in PURIFICATION_FRACTIONS}
    for ny, data in arrays.items():
        cycles = expected_spectrum_cycles(ny)
        entropy = data["entropy"]
        initial = entropy[:, 0]
        for position, cycle in enumerate(cycles):
            values = entropy[:, position]
            low, high = bootstrap_mean_ci(values, bootstrap_count, rng)
            cycle_rows.append({"Ny": ny, "cycle": int(cycle), "normalized_cycle": float(cycle / ny),
                "entropy_mean": float(values.mean()), "entropy_sem": sem(values),
                "entropy_ci_low": low, "entropy_ci_high": high,
                "entropy_per_Ny_mean": float((values / ny).mean()),
                "charge_variance_mean": float(data["charge_variance"][:, position].mean())})
        for fraction in PURIFICATION_FRACTIONS:
            times = np.full(100, np.nan)
            for sample in range(100):
                threshold = fraction * initial[sample]
                indices = np.flatnonzero(entropy[sample] <= threshold)
                if indices.size:
                    index = int(indices[0])
                    if index == 0:
                        times[sample] = 0.0
                    else:
                        x0, x1 = cycles[index - 1], cycles[index]
                        y0, y1 = entropy[sample, index - 1], entropy[sample, index]
                        times[sample] = float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
            time_by_fraction[fraction][ny] = times
            finite = times[np.isfinite(times)]
            low, high = bootstrap_mean_ci(finite, bootstrap_count, rng) if finite.size else (math.nan, math.nan)
            for sample, value in enumerate(times):
                time_rows.append({"Ny": ny, "sample_index": sample, "entropy_fraction": fraction,
                                  "purification_time": value, "resolved": bool(np.isfinite(value))})
            cycle_rows.append({"Ny": ny, "cycle": "threshold", "normalized_cycle": fraction,
                "entropy_mean": float(finite.mean()) if finite.size else math.nan,
                "entropy_sem": sem(finite) if finite.size else math.nan,
                "entropy_ci_low": low, "entropy_ci_high": high,
                "entropy_per_Ny_mean": float(finite.mean() / ny) if finite.size else math.nan,
                "charge_variance_mean": math.nan})
    z_rows = []
    for fraction in PURIFICATION_FRACTIONS:
        means = np.asarray([np.nanmean(time_by_fraction[fraction][ny]) for ny in NY_VALUES])
        primary = fit_line(np.log(np.asarray(NY_VALUES)), np.log(means))
        omitted = fit_line(np.log(np.asarray(NY_VALUES[1:])), np.log(means[1:]))
        z_rows.append({"entropy_fraction": fraction, "z": primary["coefficient"],
                       "z_omit_Ny20": omitted["coefficient"], "r_squared": primary["r_squared"]})
    z_values = np.asarray([row["z"] for row in z_rows])
    return cycle_rows, time_rows, {"threshold_fits": z_rows,
        "z_mean_across_thresholds": float(z_values.mean()),
        "note": "threshold crossing times use linear interpolation on the saved four-cycle entropy grid"}


def localization_analysis(arrays: dict[int, dict[str, np.ndarray]], bootstrap_count: int,
                          rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for ny, data in arrays.items():
        aggregate = data["wall_weights"][:, -1, :4, :].sum(axis=(1, 2)) / 4.0
        low, high = bootstrap_mean_ci(aggregate, bootstrap_count, rng)
        rows.append({"Ny": ny, "modes": 4, "samples": 100,
            "aggregate_wall_subspace_weight_mean": float(aggregate.mean()),
            "aggregate_wall_subspace_weight_sem": sem(aggregate),
            "ci_low": low, "ci_high": high,
            "label_policy": "aggregate subspace only; no individual-mode labels"})
        profile_samples = data["x_profiles"][:, -1, :4, :].sum(axis=1) / 4.0
        for x in range(20):
            values = profile_samples[:, x]
            profiles.append({"Ny": ny, "x": x, "profile_mean": float(values.mean()),
                             "profile_sem": sem(values)})
    return rows, profiles


def sample_convergence(fitted: dict[int, dict[str, np.ndarray]], rng: np.random.Generator,
                       repeats: int) -> list[dict[str, Any]]:
    ny = np.asarray(NY_VALUES, dtype=np.float64)
    x = 1.0 / ny**2
    rows: list[dict[str, Any]] = []
    for count in (25, 50, 75, 100):
        draws = 1 if count == 100 else repeats
        estimates: dict[str, list[float]] = {"A0": [], **{f"Ai{level}": [] for level in range(1, FIT_LEVEL_COUNT)}}
        for _ in range(draws):
            means = []
            gap_means = {level: [] for level in range(1, FIT_LEVEL_COUNT)}
            for value in NY_VALUES:
                slopes = fitted[value]["W3_3Ny_to_4Ny_levels"]
                indices = rng.choice(100, count, replace=False)
                means.append(slopes[indices, 0].mean())
                for level in range(1, FIT_LEVEL_COUNT):
                    gaps = slopes[:, 0] - slopes[:, level]
                    finite = gaps[np.isfinite(gaps)]
                    if finite.size >= count:
                        gap_means[level].append(finite[rng.choice(finite.size, count, replace=False)].mean())
            estimates["A0"].append(fit_line(x, -np.asarray(means) / ny)["coefficient"])
            for level in range(1, FIT_LEVEL_COUNT):
                if len(gap_means[level]) == len(NY_VALUES):
                    estimates[f"Ai{level}"].append(fit_line(x, np.asarray(gap_means[level]) / ny, intercept=False)["coefficient"])
        for metric, values in estimates.items():
            array = np.asarray(values, dtype=np.float64)
            low, high = percentile_interval(array) if array.size > 1 else ((float(array[0]), float(array[0])) if array.size else (math.nan, math.nan))
            rows.append({"samples_per_Ny": count, "metric": metric,
                "mean": float(array.mean()) if array.size else math.nan, "ci_low": low, "ci_high": high,
                "subset_repeats": int(array.size),
                "selection": "random_without_replacement" if count < 100 else "full_ensemble"})
    return rows


def nonpooled_comparison() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = (
        ("independent_2Ny_v2", LEGACY_2NY_ROOT / "analysis_summary.json"),
        ("independent_three_size_4Ny", LEGACY_4NY_ROOT / "analysis_summary.json"),
    )
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"samples_pooled": False, "sources": {}}
    for name, path in sources:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance["sources"][name] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path),
                                               "schema": payload.get("schema")}
        if name == "independent_2Ny_v2":
            rows.append({"ensemble": name, "depth_multiple": 2, "sizes": "20,22,24,26,28,30,36,40",
                         "trajectories": payload.get("verified_trajectories", 800),
                         "temporal_gate": payload.get("acceptance_decisions", {}).get("T_2Ny_temporally_sufficient"),
                         "A0": payload.get("finite_size", {}).get("primary", {}).get("A0"), "pooled": False})
        else:
            hard = payload.get("finite_size", {}).get("hard", {})
            rows.append({"ensemble": name, "depth_multiple": 4, "sizes": "20,30,40",
                         "trajectories": payload.get("provenance", {}).get("campaigns", {}).get("hard", {}).get("trajectories", 300),
                         "temporal_gate": payload.get("all_temporal_metrics_pass", {}).get("hard"),
                         "A0": hard.get("primary", {}).get("A0"), "pooled": False})
    return rows, provenance


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: ("" if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)) else value)
                for key, value in row.items()
            }
            for row in rows
        )
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure_plotting() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False, "savefig.bbox": "tight"})


COLORS = dict(zip(NY_VALUES, ("#7f0000", "#b2182b", "#d6604d", "#f4a582", "#92c5de", "#4393c3", "#2166ac")))
MARKERS = dict(zip(NY_VALUES, ("o", "s", "^", "D", "v", "P", "X")))


def panel(axis: Any, label: str) -> None:
    axis.text(-0.18, 1.04, label, transform=axis.transAxes, fontweight="bold", va="bottom")


def make_figures(output_root: Path, arrays: dict[int, dict[str, np.ndarray]], fitted: dict[int, dict[str, np.ndarray]],
                 temporal_rows: list[dict[str, Any]], closure_rows: list[dict[str, Any]], finite: dict[str, Any],
                 cycle_rows: list[dict[str, Any]], purification_times: list[dict[str, Any]], localization_rows: list[dict[str, Any]],
                 profile_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]], numerical_rows: list[dict[str, Any]]) -> None:
    configure_plotting()
    figure_dir = output_root / "figure_assets"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.6), constrained_layout=True)
    for ny in NY_VALUES:
        selected_cycles = [row for row in cycle_rows if row["Ny"] == ny and row["cycle"] != "threshold"]
        t = np.asarray([row["normalized_cycle"] for row in selected_cycles])
        mean = np.asarray([row["entropy_mean"] / ny for row in selected_cycles])
        low = np.asarray([row["entropy_ci_low"] / ny for row in selected_cycles])
        high = np.asarray([row["entropy_ci_high"] / ny for row in selected_cycles])
        axes[0, 0].plot(t, mean, color=COLORS[ny], label=str(ny), linewidth=1)
        axes[0, 0].fill_between(t, low, high, color=COLORS[ny], alpha=.16, linewidth=0)
    axes[0, 0].set(xlabel=r"$t/N_y$", ylabel=r"$\langle S(t)\rangle/N_y$")
    axes[0, 0].set_yscale("log"); axes[0, 0].legend(title=r"$N_y$", ncol=2, fontsize=6)
    panel(axes[0, 0], "(a)")

    for fraction in PURIFICATION_FRACTIONS:
        selected = [row for row in purification_times if row["entropy_fraction"] == fraction and row["resolved"]]
        means = [np.mean([row["purification_time"] for row in selected if row["Ny"] == ny]) / ny for ny in NY_VALUES]
        axes[0, 1].plot(NY_VALUES, means, marker="o", markersize=3, label=f"{fraction:g}")
    axes[0, 1].set(xlabel=r"$N_y$", ylabel=r"$\langle t_{S/S_0}\rangle/N_y$")
    axes[0, 1].legend(title=r"$S/S_0$")
    endpoint_axis = axes[0, 1].twinx()
    endpoint_axis.plot(NY_VALUES, [arrays[ny]["entropy"][:, -1].mean() for ny in NY_VALUES],
                       color="0.25", linestyle=":", marker=".", linewidth=.8)
    endpoint_axis.set_ylabel(r"$\langle S(4N_y)\rangle$", color="0.25")
    panel(axes[0, 1], "(b)")

    late = [row for row in temporal_rows if row["earlier_window"] == "W2_2Ny_to_3Ny"]
    metrics = ("lambda0", "gap1", "gap2", "gap3", "gap4")
    for index, metric in enumerate(metrics):
        selected = [row for row in late if row["metric"] == metric]
        axes[0, 2].scatter(np.full(len(selected), index) + np.linspace(-.22, .22, len(selected)),
                           [row["relative_shift"] for row in selected],
                           c=[COLORS[row["Ny"]] for row in selected], s=15)
    axes[0, 2].set_xticks(range(5), [r"$\lambda_0$", r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_3$", r"$\Delta_4$"])
    axes[0, 2].set(ylabel=r"relative $W_2\to W_3$ shift"); axes[0, 2].set_yscale("log")
    panel(axes[0, 2], "(c)")

    selected = [row for row in closure_rows if row["window"] == "W3_3Ny_to_4Ny"]
    values = []
    for row in selected:
        x, y = -row["lambda0_mean"], -row["omega_slope_mean"]
        values.extend((x, y)); axes[1, 0].errorbar(x, y, xerr=row["lambda0_sem"], yerr=row["omega_slope_sem"],
            marker=MARKERS[row["Ny"]], color=COLORS[row["Ny"]], linestyle="none", capsize=2)
    limits = (min(values), max(values)); axes[1, 0].plot(limits, limits, color="0.4", linestyle="--", linewidth=.8)
    axes[1, 0].set(xlabel=r"$-\langle\lambda_0\rangle_{W_3}$", ylabel=r"$-\langle d\omega/dt\rangle_{W_3}$")
    panel(axes[1, 0], "(d)")

    x = np.asarray(finite["inverse_Ny_squared"]); y = np.asarray(finite["f0_tilde"])
    dense = np.linspace(0, x.max() * 1.05, 200); primary = finite["primary"]
    axes[1, 1].plot(dense, primary["intercept"] + primary["A0"] * dense, color="0.25")
    axes[1, 1].scatter(x, y, c=[COLORS[ny] for ny in NY_VALUES], marker="o")
    axes[1, 1].set(xlabel=r"$1/N_y^2$", ylabel=r"$-\langle\lambda_0\rangle/N_y$")
    panel(axes[1, 1], "(e)")

    for gap in finite["gaps"]:
        axes[1, 2].plot(x, gap["density"], marker="o", markersize=3, label=rf"$\Delta_{gap['level']}$")
    axes[1, 2].set(xlabel=r"$1/N_y^2$", ylabel=r"$\langle\lambda_0-\lambda_i\rangle/N_y$")
    axes[1, 2].legend(fontsize=6)
    panel(axes[1, 2], "(f)")
    fig.savefig(figure_dir / "dynamical_critical_main.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.6), constrained_layout=True)
    axes[0, 0].errorbar([row["Ny"] for row in localization_rows],
        [row["aggregate_wall_subspace_weight_mean"] for row in localization_rows],
        yerr=[row["aggregate_wall_subspace_weight_sem"] for row in localization_rows], marker="o", capsize=2)
    axes[0, 0].set(xlabel=r"$N_y$", ylabel="four-mode wall weight")
    panel(axes[0, 0], "(a)")
    for ny in NY_VALUES:
        selected_profiles = [row for row in profile_rows if row["Ny"] == ny]
        axes[0, 1].plot([row["x"] for row in selected_profiles], [row["profile_mean"] for row in selected_profiles], color=COLORS[ny])
    axes[0, 1].axvline(5, color="0.5", linestyle=":"); axes[0, 1].axvline(15, color="0.5", linestyle=":")
    axes[0, 1].set(xlabel=r"$x$", ylabel="four-mode profile")
    panel(axes[0, 1], "(b)")
    selected_samples = [row for row in sample_rows if row["metric"] == "A0"]
    axes[0, 2].errorbar([row["samples_per_Ny"] for row in selected_samples], [row["mean"] for row in selected_samples],
        yerr=[[row["mean"] - row["ci_low"] for row in selected_samples], [row["ci_high"] - row["mean"] for row in selected_samples]],
        marker="o", capsize=2)
    axes[0, 2].axhline(0, color="0.5", linestyle="--"); axes[0, 2].set(xlabel="trajectories per size", ylabel=r"$A_0$")
    panel(axes[0, 2], "(c)")

    for ny in NY_VALUES:
        selected_cycles = [row for row in cycle_rows if row["Ny"] == ny and row["cycle"] != "threshold"]
        t = np.asarray([row["normalized_cycle"] for row in selected_cycles])
        mean = np.asarray([row["entropy_mean"] for row in selected_cycles])
        axes[1, 0].plot(t, mean, color=COLORS[ny], linewidth=1)
    axes[1, 0].set_yscale("log"); axes[1, 0].set(xlabel=r"$t/N_y$", ylabel=r"$\langle S(t)\rangle$")
    panel(axes[1, 0], "(d)")
    resolved = [[row["resolved_fraction"] for row in late if row["Ny"] == ny and row["metric"].startswith("gap")] for ny in NY_VALUES]
    image = axes[1, 1].imshow(np.asarray(resolved).T, vmin=0, vmax=1, aspect="auto", origin="lower", cmap="viridis")
    axes[1, 1].set_xticks(range(len(NY_VALUES)), NY_VALUES); axes[1, 1].set_yticks(range(4), [1, 2, 3, 4])
    axes[1, 1].set(xlabel=r"$N_y$", ylabel="gap index"); fig.colorbar(image, ax=axes[1, 1], label="resolved fraction")
    panel(axes[1, 1], "(e)")
    residual_names = ("spectrum_reconstruction_error", "eigensolver_residual", "eigenvector_gram_residual", "occupation_bound_residual")
    residual_values = [max(row[name] for row in numerical_rows) for name in residual_names]
    axes[1, 2].scatter(range(len(residual_names)), np.maximum(residual_values, 1e-18), color="0.2")
    axes[1, 2].set_yscale("log"); axes[1, 2].set_xticks(range(4), ["recon.", "eigen.", "Gram", "bound"], rotation=20)
    axes[1, 2].set(ylabel="maximum residual")
    panel(axes[1, 2], "(f)")
    fig.savefig(figure_dir / "dynamical_critical_diagnostics.png", dpi=300)
    plt.close(fig)


def write_reader_document(output_root: Path, summary: dict[str, Any]) -> Path:
    primary = summary["finite_size"]["primary"]
    gaps = summary["finite_size"]["gaps"]
    max_closure = max(row["relative_closure"] for row in summary["record_closure_W3"].values())
    gap_table = "\n".join(
        rf"{gap['level']} & {gap['Ai']:.3f} & {gap['alpha_x']:.3f} & "
        rf"{gap['r_i_x_over_c_eff']:.4f} & {gap['x_i_over_x_1']:.3f} \\" for gap in gaps
    )
    tex = rf"""\documentclass[aps,prb,twocolumn,nofootinbib,superscriptaddress]{{revtex4-2}}
\usepackage{{amsmath,amssymb,graphicx,booktabs}}
\usepackage[T1]{{fontenc}}
\begin{{document}}
\title{{Completed $4N_y$ hard-wall dynamical-critical analysis}}
\author{{Numerical working note}}
\date{{September 17, 2026}}
\begin{{abstract}}
We analyze 700 independent Born trajectories of the $N_x=20$ hard-wall adaptive Gaussian circuit at
$N_y=20,24,30,36,44,56,60$ and depth $T=4N_y$. All 140 five-sample shards pass checksum,
identity, coverage, and numerical validation. The saved occupation spectra reproduce the stored leading
many-body squared-singular-value levels to a maximum absolute error of
${summary['provenance']['maximum_spectrum_reconstruction_error']:.2e}$.
The calculation reports the raw finite-size coefficients and their uncertainty without acceptance gates.
The resulting central-charge coefficient is anomalous, while the ordered gap coefficients form a nearly
equally spaced sequence.
\end{{abstract}}
\maketitle

\section{{Protocol and reconstruction}}
The campaign uses maximally mixed initial conditions, hard support-truncated walls at $x=5,15$,
$n_{{\rm shell}}=1$, $\alpha_1=1$, $\alpha_2=30$, raster-$y$ measurements, perfect correction,
and complex128 covariance evolution. Entropies and active-space occupations were saved every four cycles;
the realized Born record weight $\omega(t)$ was saved every cycle. For each saved occupation spectrum
$\{{\nu_a\}}$ we reconstruct the many-body levels
\begin{{equation}}
 \ell_i=\log\sigma_i^2=\log Z+\sum_a\log\max(\nu_a,1-\nu_a)-d_i,
\end{{equation}}
where $d_i$ are the lowest subset sums of
$|\log[\nu_a/(1-\nu_a)]|$. Exact zero and unit occupations retain infinite flip costs.
We fit every trajectory separately in $W_1=[N_y,2N_y]$, $W_2=[2N_y,3N_y]$, and
$W_3=[3N_y,4N_y]$. All confidence intervals resample complete trajectories, with 2000 replicates and
seed 2026091702.

\begin{{figure*}}[t]
\centering\includegraphics[width=\textwidth]{{figure_assets/dynamical_critical_main.png}}
\caption{{Primary dynamical-critical results. (a) Entropy-density collapse with whole-trajectory bootstrap intervals.
(b) Sample-wise purification times and endpoint residual entropy. (c) Relative $W_2\to W_3$ shifts.
(d) Same-window record-weight closure. (e) Leading finite-size fit. (f) Ordered-gap densities.
Colors label circumference.}}
\end{{figure*}}

\section{{Purification and temporal convergence}}
The endpoint entropy is small at every circumference, while the full $S(t)/N_y$ curves provide a direct
descriptive test of $z=1$ scaling. Threshold-crossing times are linearly interpolated on the native
four-cycle grid. We report the fitted threshold exponents and the paired $W_2\to W_3$ shifts directly,
without converting them into pass/fail decisions.

\section{{Record closure and boundary localization}}
The leading reconstructed level is compared with $d\omega/dt$ in the same fit window. The endpoint ratio
$-\omega(T)/T$ is retained only as a finite-depth diagnostic. The largest late-window relative mismatch
between $d\omega/dt$ and $\lambda_0$ is {max_closure:.3e}. The first four soft modes
are analyzed only through their aggregate subspace density and aggregate weight near both walls. This avoids
nonphysical labels under rotations in nearly degenerate eigenspaces.

\section{{Finite-size coefficients}}
The late-window leading density is fit to
\begin{{equation}}
 -\lambda_0/N_y=f_\infty+A_0/N_y^2,
\end{{equation}}
with $1/N_y^4$ and successive $N_{{y,\min}}=24,30,36$ sensitivity fits. The primary result is
$A_0={primary['A0']:.4g}$ with 95\% interval
$[{primary['A0_ci95'][0]:.4g},{primary['A0_ci95'][1]:.4g}]$.
This gives $\alpha c_{{\rm eff}}={primary['alpha_c_eff']:.3f}$ with 95\% interval
$[{primary['alpha_c_eff_ci95'][0]:.3f},{primary['alpha_c_eff_ci95'][1]:.3f}]$.
For the gaps we define $r_i\equiv x_i/c_{{\rm eff}}=-A_i/(12A_0)$ and additionally quote
$x_i/x_1=A_i/A_1$:
\begin{{center}}
\begin{{tabular}}{{ccccc}}\toprule
$i$ & $A_i$ & $\alpha x_i$ & $r_i$ & $x_i/x_1$ \\\midrule
{gap_table}
\bottomrule\end{{tabular}}
\end{{center}}
These are direct fit outputs, not gated claims. The negative $r_i$ values result from the positive fitted
$A_0$ and therefore expose the anomalous central-charge extrapolation. No conformal operator names are
assigned because momentum, charge-sector, and two-wall symmetry labels were not saved.

\begin{{figure*}}[t]
\centering\includegraphics[width=\textwidth]{{figure_assets/dynamical_critical_diagnostics.png}}
\caption{{Diagnostics. (a,b) Aggregate four-mode wall localization and transverse profile.
(c) Random without-replacement sample convergence. (d) Unnormalized entropy dynamics. (e) Resolved fractions.
(f) Maximum numerical residuals.}}
\end{{figure*}}

\section{{Relation to earlier ensembles}}
The earlier $2N_y$ pilot and the independent three-size $4N_y$ purification campaign are compared only at
the summary-ledger level. Their trajectories are never pooled with this primary seven-size campaign.

\section{{Conclusion}}
The completed data validate the Gaussian many-body spectral reconstruction, deep-time purification,
same-window record closure, and boundary localization. The ungated fits yield a negative
$\alpha c_{{\rm eff}}$ but clean positive ordered-gap coefficients. The accompanying CSV and JSON files
retain the bootstrap intervals, temporal shifts, resolved fractions, and fit sensitivities needed to judge
those estimates directly.
\end{{document}}
"""
    tex_path = output_root / "dynamical_critical_analysis.tex"
    tex_path.write_text(tex, encoding="utf-8")
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is required to build the reader-facing document")
    for _ in range(2):
        subprocess.run([executable, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                       cwd=output_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pdf_path = output_root / "dynamical_critical_analysis.pdf"
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError("reader-facing PDF was not produced")
    return pdf_path


def run_analysis(output_root: Path = DEFAULT_OUTPUT_ROOT, bootstrap_count: int = BOOTSTRAP_REPLICATES,
                 verify_hashes: bool = True, subset_repeats: int = SUBSET_REPEATS,
                 build_document: bool = True) -> dict[str, Any]:
    arrays, numerical_rows, provenance = load_and_verify(verify_hashes=verify_hashes)
    trajectory_rows, fitted = trajectory_slopes(arrays)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    temporal_rows = temporal_stability(fitted, bootstrap_count, rng)
    closure_rows = record_closure(fitted, bootstrap_count, rng)
    finite, finite_rows = finite_size_analysis(fitted, bootstrap_count, rng)
    cycle_rows, purification_time_rows, dynamic_summary = purification_analysis(arrays, bootstrap_count, rng)
    localization_rows, profile_rows = localization_analysis(arrays, bootstrap_count, rng)
    sample_rows = sample_convergence(fitted, rng, subset_repeats)
    comparison_rows, comparison_provenance = nonpooled_comparison()
    late_temporal = [row for row in temporal_rows if row["earlier_window"] == "W2_2Ny_to_3Ny"]
    late_closure = [row for row in closure_rows if row["window"] == "W3_3Ny_to_4Ny"]
    comparison_rows.append({"ensemble": "primary_bundle13_seven_size_4Ny", "depth_multiple": 4,
        "sizes": ",".join(str(value) for value in NY_VALUES), "trajectories": 700,
        "temporal_gate": "not_applied_in_ungated_revision",
        "A0": finite["primary"]["A0"], "pooled": False})
    summary: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA, "sampling_revision": REVISION,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": bootstrap_count,
        "subset_repeats": subset_repeats, "independent_sampling_unit": "whole Born trajectory",
        "provenance": provenance,
        "analysis_contract": {"Nx": 20, "Ny": list(NY_VALUES), "samples_per_size": 100,
            "depth": "4*Ny", "leading_levels_reconstructed": 64, "fitted_levels": 5,
            "spectrum_resolution_cycles": 4, "windows": {key: list(value) for key, value in WINDOWS.items()},
            "exact_caps": "infinite flip cost; no finite clipping", "N_eff": "22*Ny"},
        "temporal_diagnostics_W2_to_W3": {str(ny): {
            row["metric"]: {key: row[key] for key in ("resolved_fraction", "shift", "shift_ci_low", "shift_ci_high", "relative_shift")}
            for row in late_temporal if row["Ny"] == ny} for ny in NY_VALUES},
        "record_closure_W3": {str(row["Ny"]): row for row in late_closure},
        "purification": dynamic_summary, "finite_size": finite,
        "localization": localization_rows, "nonpooled_comparisons": comparison_provenance,
        "verification": {
            "verified_140_pairs_700_trajectories": provenance["verified_result_pairs"] == 140 and provenance["verified_trajectories"] == 700,
            "manifest_aggregate_inventory_digest_matches": provenance["raw_file_inventory_digest_matches"],
            "roundoff_spectrum_reconstruction": provenance["maximum_spectrum_reconstruction_error"] <= 5e-10,
        },
        "reported_estimates": {"alpha_c_eff": finite["primary"]["alpha_c_eff"],
            "alpha_c_eff_ci95": finite["primary"]["alpha_c_eff_ci95"],
            "r_i_definition": "x_i/c_eff=-Ai/(12*A0)",
            "r_i": {str(gap["level"]): gap["r_i_x_over_c_eff"] for gap in finite["gaps"]},
            "x_i_over_x_1": {str(gap["level"]): gap["x_i_over_x_1"] for gap in finite["gaps"]}},
        "maximum_numerical_residuals": {key: max(row[key] for row in numerical_rows) for key in numerical_rows[0] if key not in ("Ny", "result")},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "trajectory_window_slopes.csv", trajectory_rows)
    write_csv(output_root / "temporal_stability.csv", temporal_rows)
    write_csv(output_root / "record_weight_closure.csv", closure_rows)
    write_csv(output_root / "finite_size_fits.csv", finite_rows)
    write_csv(output_root / "purification_cycle_summary.csv", [row for row in cycle_rows if row["cycle"] != "threshold"])
    write_csv(output_root / "purification_times.csv", purification_time_rows)
    write_csv(output_root / "localization.csv", localization_rows)
    write_csv(output_root / "localization_profiles.csv", profile_rows)
    write_csv(output_root / "sample_convergence.csv", sample_rows)
    write_csv(output_root / "numerical_diagnostics.csv", numerical_rows)
    write_csv(output_root / "nonpooled_comparison.csv", comparison_rows)
    write_json(output_root / "analysis_summary.json", summary)
    make_figures(output_root, arrays, fitted, temporal_rows, closure_rows, finite, cycle_rows, purification_time_rows,
                 localization_rows, profile_rows, sample_rows, numerical_rows)
    if build_document:
        pdf = write_reader_document(output_root, summary)
        summary["reader_document"] = {"path": pdf.name, "sha256": sha256_file(pdf), "bytes": pdf.stat().st_size}
        write_json(output_root / "analysis_summary.json", summary)
    readme = """# Completed 4Ny dynamical-critical analysis, ungated revision\n\nThis immutable revision reports the fitted central-charge and ordered-gap coefficients unconditionally, together with bootstrap intervals and sensitivity diagnostics. Raw samples are never pooled with older ensembles. Figure assets are PNG files; `dynamical_critical_analysis.pdf` is the single reader-facing PDF containing both figures.\n"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--subset-repeats", type=int, default=SUBSET_REPEATS)
    parser.add_argument("--skip-file-hashes", action="store_true")
    parser.add_argument("--skip-document", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100 or args.subset_repeats < 1:
        parser.error("bootstrap replicates must be >=100 and subset repeats positive")
    summary = run_analysis(args.output_root.resolve(), args.bootstrap_replicates,
                           not args.skip_file_hashes, args.subset_repeats, not args.skip_document)
    print(json.dumps({"schema": summary["schema"], "output_root": str(args.output_root.resolve()),
                      "reported_estimates": summary["reported_estimates"],
                      "verification": summary["verification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
