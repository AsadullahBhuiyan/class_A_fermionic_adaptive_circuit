#!/usr/bin/env python3
"""Verify and analyze the completed v2 many-body Lyapunov campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
from run_campaign import (  # noqa: E402
    EXPECTED_TRAJECTORIES,
    config_hash,
    expand_tasks,
    source_hashes,
    task_paths,
    validate_config,
    verified_complete,
)
from lyapunov_observer import (  # noqa: E402
    OBSERVER_SCHEMA,
    leading_log_sigma2_levels,
)


ANALYSIS_SCHEMA = "maxmix_manybody_lyapunov_analysis_v2"
IMPORT_MANIFEST_SCHEMA = "maxmix_manybody_lyapunov_drive_import_v1"
RESULT_SCHEMA = "maxmix_manybody_lyapunov_gpu_task_v2"
BOOTSTRAP_DEFAULT = 2000
FIRST_LEVEL_COUNT = 5
RELATIVE_SHIFT_THRESHOLD = 0.10
FINITE_SIZE_STABILITY_THRESHOLD = 0.20
SOFT_SUBSPACE_SIZE = 4
NUMERICAL_LIMITS = {
    "hermiticity_residual": 1.0e-8,
    "eigensolver_residual": 1.0e-8,
    "eigenvector_gram_residual": 1.0e-8,
    "occupation_bound_residual": 1.0e-9,
    "active_exterior_coupling_residual": 1.0e-8,
    "exterior_product_residual": 1.0e-8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: Any, key: str) -> Any:
    return np.asarray(data[key]).item()


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.shape != x.shape or not np.all(np.isfinite(y)):
        raise ValueError("a slope needs at least two finite aligned observations")
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("a slope needs at least two distinct coordinates")
    return float(np.dot(centered, y - y.mean()) / denominator)


def reconstructed_levels_match(
    reconstructed: np.ndarray, stored: np.ndarray
) -> bool:
    """Accept roundoff only; matching exact -inf padding is allowed."""
    tolerance = 8.0 * np.finfo(np.float64).eps
    return bool(
        np.allclose(
            np.asarray(reconstructed, dtype=np.float64),
            np.asarray(stored, dtype=np.float64),
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        )
    )


def trajectory_window_slopes(
    cycles: np.ndarray,
    levels: np.ndarray,
    *,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    cycles = np.asarray(cycles, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    middle_mask = (cycles >= ny) & (cycles <= 1.5 * ny)
    late_mask = (cycles >= 1.5 * ny) & (cycles <= 2 * ny)
    fit_count = min(FIRST_LEVEL_COUNT, levels.shape[1])
    middle = np.asarray(
        [linear_slope(cycles[middle_mask], levels[middle_mask, i]) for i in range(fit_count)]
    )
    late = np.asarray(
        [linear_slope(cycles[late_mask], levels[late_mask, i]) for i in range(fit_count)]
    )
    return middle, late


def record_window_slopes(
    cycles: np.ndarray, omega: np.ndarray, *, ny: int
) -> tuple[float, float]:
    cycles = np.asarray(cycles, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    middle = (cycles >= ny) & (cycles <= 1.5 * ny)
    late = (cycles >= 1.5 * ny) & (cycles <= 2 * ny)
    return linear_slope(cycles[middle], omega[middle]), linear_slope(
        cycles[late], omega[late]
    )


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(low), float(high)


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0


def _metric_values(slopes: np.ndarray, index: int) -> np.ndarray:
    if index == 0:
        return slopes[:, 0]
    return slopes[:, 0] - slopes[:, index]


def paired_convergence(
    middle: np.ndarray,
    late: np.ndarray,
    *,
    bootstrap_count: int,
    rng: np.random.Generator,
    relative_threshold: float,
) -> list[dict[str, Any]]:
    middle = np.asarray(middle, dtype=np.float64)
    late = np.asarray(late, dtype=np.float64)
    count = middle.shape[0]
    chosen = rng.integers(0, count, size=(bootstrap_count, count))
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(("lambda0", "gap1", "gap2", "gap3", "gap4")):
        middle_values = _metric_values(middle, index)
        late_values = _metric_values(late, index)
        shifts = late_values - middle_values
        boot = shifts[chosen].mean(axis=1)
        low, high = percentile_interval(boot)
        shift = float(shifts.mean())
        reference = abs(float(late_values.mean()))
        relative = abs(shift) / reference if reference else math.inf
        rows.append(
            {
                "metric": name,
                "middle_mean": float(middle_values.mean()),
                "middle_sem": sem(middle_values),
                "late_mean": float(late_values.mean()),
                "late_sem": sem(late_values),
                "shift": shift,
                "shift_ci_low": low,
                "shift_ci_high": high,
                "shift_ci_contains_zero": bool(low <= 0.0 <= high),
                "relative_shift": float(relative),
                "relative_threshold": float(relative_threshold),
                "passes": bool(low <= 0.0 <= high and relative <= relative_threshold),
            }
        )
    return rows


def _fit_line(x: np.ndarray, y: np.ndarray, *, intercept: bool) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack((np.ones(x.size), x)) if intercept else x[:, None]
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - y.mean()) ** 2))
    return {
        "intercept": float(coefficients[0]) if intercept else 0.0,
        "coefficient": float(coefficients[-1]),
        "r_squared": float(1.0 - residual_sum / total_sum) if total_sum else 1.0,
    }


def _fit_quadratic_correction(x: np.ndarray, y: np.ndarray, *, intercept: bool) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = (
        np.column_stack((np.ones(x.size), x, x**2))
        if intercept
        else np.column_stack((x, x**2))
    )
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    if intercept:
        return {
            "intercept": float(coefficients[0]),
            "coefficient": float(coefficients[1]),
            "subleading_coefficient": float(coefficients[2]),
        }
    return {
        "intercept": 0.0,
        "coefficient": float(coefficients[0]),
        "subleading_coefficient": float(coefficients[1]),
    }


def fit_finite_size(ny_values: np.ndarray, lambda_values: np.ndarray) -> dict[str, Any]:
    """Compatibility/public helper for the pre-registered finite-size forms."""
    ny = np.asarray(ny_values, dtype=np.float64)
    lam = np.asarray(lambda_values, dtype=np.float64)
    x = 1.0 / ny**2
    f0 = -lam[:, 0] / ny
    primary = _fit_line(x, f0, intercept=True)
    subleading = _fit_quadratic_correction(x, f0, intercept=True)
    omit = _fit_line(x[1:], f0[1:], intercept=True)
    a0 = primary["coefficient"]
    gaps = []
    for index in range(1, min(FIRST_LEVEL_COUNT, lam.shape[1])):
        density = (lam[:, 0] - lam[:, index]) / ny
        constrained = _fit_line(x, density, intercept=False)
        diagnostic = _fit_line(x, density, intercept=True)
        ai = constrained["coefficient"]
        gaps.append(
            {
                "level": index,
                "intercept": diagnostic["intercept"],
                "Ai": ai,
                "alpha_x_typ": ai / (2.0 * math.pi),
                "x_typ_over_c_eff": -ai / (12.0 * a0) if a0 else math.nan,
                "unconstrained_Ai": diagnostic["coefficient"],
            }
        )
    return {
        "Ny": ny.astype(int).tolist(),
        "inverse_Ny_squared": x.tolist(),
        "f0_tilde": f0.tolist(),
        "primary": {
            "intercept": primary["intercept"],
            "A0": a0,
            "alpha_c_eff": -6.0 * a0 / math.pi,
            "r_squared": primary["r_squared"],
        },
        "with_inverse_Ny_fourth": {
            "intercept": subleading["intercept"],
            "A0": subleading["coefficient"],
            "B0": subleading["subleading_coefficient"],
        },
        "omit_Ny20": {"intercept": omit["intercept"], "A0": omit["coefficient"]},
        "gaps": gaps,
    }


def _raw_inventory_digest(paths: list[Path], collection_root: Path) -> str:
    lines = []
    for path in sorted(paths, key=lambda item: item.relative_to(collection_root).as_posix()):
        relative = path.relative_to(collection_root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def load_historical_provenance(
    *, results_root: Path, config: dict[str, Any], allow_incomplete: bool
) -> dict[str, Any]:
    collection_root = results_root.parent if results_root.name == "results" else results_root
    manifest_path = collection_root / "DOWNLOAD_MANIFEST.json"
    current_hashes = source_hashes()
    if not manifest_path.is_file():
        if not allow_incomplete:
            raise RuntimeError(f"complete historical analysis requires {manifest_path}")
        return {
            "manifest_path": None,
            "historical_manifest_used": False,
            "executed_source_hashes": current_hashes,
            "current_source_hashes": current_hashes,
            "source_hash_drift": {},
            "collection_root": collection_root,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != IMPORT_MANIFEST_SCHEMA:
        raise RuntimeError("download manifest schema mismatch")
    identity = manifest.get("campaign_identity", {})
    if identity.get("sampling_revision") != config["sampling_revision"]:
        raise RuntimeError("download manifest revision mismatch")
    if identity.get("configuration_sha256") != config_hash(config):
        raise RuntimeError("download manifest configuration hash mismatch")
    pinned = identity.get("source_hashes")
    if not isinstance(pinned, dict) or set(pinned) != set(current_hashes):
        raise RuntimeError("download manifest executed-source hash set is incomplete")
    drift = {
        name: {"executed": pinned[name], "current": current_hashes[name]}
        for name in pinned
        if pinned[name] != current_hashes[name]
    }
    return {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "historical_manifest_used": True,
        "executed_source_hashes": pinned,
        "current_source_hashes": current_hashes,
        "source_hash_drift": drift,
        "collection_root": collection_root,
    }


def _validate_real_product(data: Any, task: Any, config: dict[str, Any]) -> dict[str, float]:
    expected_cycles = np.arange(task.cycles + 1, dtype=np.int64)
    spectrum_cycles = np.asarray(data["spectrum_cycles"], dtype=np.int64)
    if scalar(data, "schema") != RESULT_SCHEMA or scalar(data, "observer_schema") != OBSERVER_SCHEMA:
        raise RuntimeError(f"{task.task_id}: result schema mismatch")
    scalar_expectations = {
        "task_id": task.task_id,
        "Nx": 20,
        "Ny": task.ny,
        "cycles_total": task.cycles,
        "batch_index": task.batch_index,
        "sample_start": task.sample_start,
        "sample_stop": task.sample_stop,
        "batch_seed": task.seed,
        "active_mode_count": 22 * task.ny,
        "dtype": "complex128",
        "init_mode": "maxmix",
        "sequence": "raster_y",
        "singular_value_convention": "ell=log(sigma^2)",
    }
    for key, expected in scalar_expectations.items():
        if scalar(data, key) != expected:
            raise RuntimeError(f"{task.task_id}: result identity mismatch for {key}")
    if not np.array_equal(data["global_sample_indices"], task.global_sample_indices):
        raise RuntimeError(f"{task.task_id}: global sample indices mismatch")
    if not np.array_equal(data["cycles"], expected_cycles):
        raise RuntimeError(f"{task.task_id}: every-cycle grid mismatch")
    if not np.allclose(data["normalized_cycles"], expected_cycles / task.ny, rtol=0, atol=0):
        raise RuntimeError(f"{task.task_id}: normalized cycle grid mismatch")
    if not np.all(data["cycle_seen"]) or not np.all(data["spectrum_seen"]):
        raise RuntimeError(f"{task.task_id}: observation mask is incomplete")
    expected_sites = 11 * task.ny
    if not np.all(data["site_event_count"][:, 1:] == expected_sites):
        raise RuntimeError(f"{task.task_id}: site-event history is incomplete")
    if not np.all(data["channel_event_count"][:, 1:] == 4 * expected_sites):
        raise RuntimeError(f"{task.task_id}: channel-event history is incomplete")
    omega = np.asarray(data["cumulative_log_probability"], dtype=np.float64)
    increments = np.diff(omega, axis=1)
    if not np.allclose(
        increments,
        np.asarray(data["measurement_log_probability"])[:, 1:],
        atol=2.0e-10,
        rtol=2.0e-12,
    ):
        raise RuntimeError(f"{task.task_id}: record-weight accumulation mismatch")
    occupations = np.asarray(data["occupations"], dtype=np.float64)
    caps = np.asarray(data["cap_mask"], dtype=bool)
    cap_tolerance = float(config["observer"]["cap_tolerance"])
    expected_caps = (occupations == 0.0) | (occupations == 1.0)
    if not np.array_equal(caps, expected_caps):
        raise RuntimeError(f"{task.task_id}: exact-cap mask mismatch")
    if np.any(occupations < 0.0) or np.any(occupations > 1.0):
        raise RuntimeError(f"{task.task_id}: snapped occupations leave [0,1]")
    if not np.allclose(occupations[:, 0], 0.5, atol=2.0e-10, rtol=0):
        raise RuntimeError(f"{task.task_id}: t=0 is not the identity spectrum")
    stored = np.asarray(data["leading_log_sigma2"], dtype=np.float64)
    if np.any(np.isnan(stored)) or np.any(np.isposinf(stored)):
        raise RuntimeError(f"{task.task_id}: invalid saved many-body levels")
    if not np.allclose(stored[:, 0], 0.0, atol=2.0e-10, rtol=0):
        raise RuntimeError(f"{task.task_id}: t=0 many-body levels are not zero")
    reconstructed = np.stack(
        [
            np.stack(
                [
                    leading_log_sigma2_levels(
                        occupations[sample, position],
                        float(data["log_z"][sample, position]),
                        count=int(config["observer"]["leading_level_count"]),
                        cap_tolerance=cap_tolerance,
                    )
                    for position in range(spectrum_cycles.size)
                ]
            )
            for sample in range(task.sample_count)
        ]
    )
    if not reconstructed_levels_match(reconstructed, stored):
        raise RuntimeError(f"{task.task_id}: exact level reconstruction mismatch")
    for key, limit in NUMERICAL_LIMITS.items():
        values = np.asarray(data[key], dtype=np.float64)
        if not np.all(np.isfinite(values)) or float(values.max()) > limit:
            raise RuntimeError(f"{task.task_id}: numerical gate failed for {key}")
    return {key: float(np.max(data[key])) for key in NUMERICAL_LIMITS}


def _append(store: dict[str, list[np.ndarray]], key: str, value: Any) -> None:
    store.setdefault(key, []).append(np.asarray(value))


def load_and_verify_campaign(
    *,
    config: dict[str, Any],
    results_root: Path,
    allow_incomplete: bool = False,
) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tasks = expand_tasks(config)
    provenance = load_historical_provenance(
        results_root=results_root, config=config, allow_incomplete=allow_incomplete
    )
    hashes = provenance["executed_source_hashes"]
    config_sha = config_hash(config)
    available = []
    pair_paths: list[Path] = []
    for task in tasks:
        valid, reason = verified_complete(
            output_root=results_root,
            task=task,
            config_sha256=config_sha,
            hashes=hashes,
        )
        if valid:
            available.append(task)
            pair_paths.extend(task_paths(results_root, task))
        elif not allow_incomplete:
            raise RuntimeError(f"historical pair verification failed for {task.task_id}: {reason}")
    if not allow_incomplete and len(available) != len(tasks):
        raise RuntimeError(f"campaign is incomplete: {len(available)}/{len(tasks)}")
    if provenance["historical_manifest_used"] and not allow_incomplete:
        manifest = provenance["manifest"]
        inventory = manifest["inventory"]
        if len(pair_paths) != int(inventory["downloaded_files"]):
            raise RuntimeError("downloaded file count disagrees with manifest")
        discovered = sorted((results_root / "results").rglob("*.npz")) + sorted(
            (results_root / "results").rglob("*.complete.json")
        )
        if {path.resolve() for path in discovered} != {path.resolve() for path in pair_paths}:
            raise RuntimeError("results tree contains missing or unexpected canonical products")
        digest = _raw_inventory_digest(pair_paths, provenance["collection_root"])
        if digest != inventory["raw_file_inventory_sha256"]:
            raise RuntimeError("raw-file inventory digest mismatch")
    by_ny_lists: dict[int, dict[str, list[np.ndarray]]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    numerical_rows: list[dict[str, Any]] = []
    strict = provenance["historical_manifest_used"] and not allow_incomplete
    for task in available:
        result_path, _ = task_paths(results_root, task)
        with np.load(result_path, allow_pickle=False) as data:
            maxima = _validate_real_product(data, task, config) if strict else {}
            spectrum_cycles = np.asarray(data["spectrum_cycles"], dtype=np.int64)
            occupations = np.asarray(data["occupations"], dtype=np.float64)
            levels = np.stack(
                [
                    np.stack(
                        [
                            leading_log_sigma2_levels(
                                occupations[sample, position],
                                float(data["log_z"][sample, position]),
                                count=int(config["observer"]["leading_level_count"]),
                                cap_tolerance=float(config["observer"]["cap_tolerance"]),
                            )
                            for position in range(spectrum_cycles.size)
                        ]
                    )
                    for sample in range(task.sample_count)
                ]
            )
            stored = np.asarray(data["leading_log_sigma2"], dtype=np.float64)
            if not reconstructed_levels_match(levels, stored):
                raise RuntimeError(f"offline level reconstruction disagrees for {task.task_id}")
            omega = np.asarray(data["cumulative_log_probability"], dtype=np.float64)
            cycles = np.arange(task.cycles + 1, dtype=np.int64)
            store = by_ny_lists.setdefault(task.ny, {})
            for local, sample_index in enumerate(task.global_sample_indices):
                middle, late = trajectory_window_slopes(spectrum_cycles, levels[local], ny=task.ny)
                omega_middle, omega_late = record_window_slopes(cycles, omega[local], ny=task.ny)
                for key, value in (
                    ("middle_slopes", middle),
                    ("late_slopes", late),
                    ("levels", levels[local]),
                    ("cumulative_log_probability", omega[local]),
                    ("entropy_nats", data["entropy_nats"][local] if "entropy_nats" in data else np.full(spectrum_cycles.size, np.nan)),
                    ("charge_variance", data["charge_variance"][local] if "charge_variance" in data else np.full(spectrum_cycles.size, np.nan)),
                    ("cap_fraction", np.asarray(data["cap_mask"][local]).mean(axis=1) if "cap_mask" in data else np.full(spectrum_cycles.size, np.nan)),
                    ("soft_mode_x_profiles", data["soft_mode_x_profiles"][local]),
                    ("soft_mode_wall_weights", data["soft_mode_wall_weights"][local] if "soft_mode_wall_weights" in data else np.full((spectrum_cycles.size, 16, 2), np.nan)),
                ):
                    _append(store, key, value)
                trajectory_rows.append(
                    {
                        "task_id": task.task_id,
                        "Ny": task.ny,
                        "sample_index": int(sample_index),
                        "batch_seed": int(task.seed),
                        "endpoint_record_rate": float(-omega[local, -1] / task.cycles),
                        "middle_omega_slope": omega_middle,
                        "late_omega_slope": omega_late,
                        "late_record_level_closure": float(omega_late - late[0]),
                        **{f"middle_lambda_{i}": float(middle[i]) for i in range(FIRST_LEVEL_COUNT)},
                        **{f"late_lambda_{i}": float(late[i]) for i in range(FIRST_LEVEL_COUNT)},
                        **{f"middle_gap_{i}": float(middle[0] - middle[i]) for i in range(1, FIRST_LEVEL_COUNT)},
                        **{f"late_gap_{i}": float(late[0] - late[i]) for i in range(1, FIRST_LEVEL_COUNT)},
                    }
                )
            store["spectrum_cycles"] = [spectrum_cycles]
            numerical_rows.append(
                {
                    "task_id": task.task_id,
                    "Ny": task.ny,
                    "batch_index": task.batch_index,
                    **maxima,
                    "finite_leading_levels_min": int(np.isfinite(stored).sum(axis=2).min()),
                    "exact_cap_count_max": int(np.asarray(data["cap_mask"]).sum(axis=2).max()) if "cap_mask" in data else 0,
                }
            )
    by_ny = {
        ny: {
            key: (values[0] if key == "spectrum_cycles" else np.stack(values))
            for key, values in store.items()
        }
        for ny, store in by_ny_lists.items()
    }
    provenance["verified_tasks"] = len(available)
    provenance["requested_tasks"] = len(tasks)
    provenance["verified_files"] = 2 * len(available)
    return by_ny, trajectory_rows, numerical_rows, provenance


def bootstrap_finite_size(
    by_ny: dict[int, dict[str, np.ndarray]], *, count: int, rng: np.random.Generator
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ny = np.asarray(sorted(by_ny), dtype=np.int64)
    means = np.stack([by_ny[int(value)]["late_slopes"].mean(axis=0) for value in ny])
    central = fit_finite_size(ny, means)
    bootstrap_a0 = np.empty(count)
    bootstrap_ai = np.empty((count, 4))
    bootstrap_intercepts = np.empty((count, 4))
    for replicate in range(count):
        sampled = []
        for value in ny:
            slopes = by_ny[int(value)]["late_slopes"]
            selected = rng.integers(0, slopes.shape[0], size=slopes.shape[0])
            sampled.append(slopes[selected].mean(axis=0))
        values = np.stack(sampled)
        fit = fit_finite_size(ny, values)
        bootstrap_a0[replicate] = fit["primary"]["A0"]
        for level in range(4):
            bootstrap_ai[replicate, level] = fit["gaps"][level]["Ai"]
            density = (values[:, 0] - values[:, level + 1]) / ny
            bootstrap_intercepts[replicate, level] = _fit_line(
                1.0 / ny**2, density, intercept=True
            )["intercept"]
    a0_low, a0_high = percentile_interval(bootstrap_a0)
    lmin_rows = []
    for start in range(0, max(1, len(ny) - 3)):
        selected = means[start:]
        local = fit_finite_size(ny[start:], selected)
        lmin_rows.append(
            {
                "fit_family": "leading_free_energy",
                "Ny_min": int(ny[start]),
                "points": int(len(ny) - start),
                "intercept": local["primary"]["intercept"],
                "coefficient": local["primary"]["A0"],
                "derived_value": local["primary"]["alpha_c_eff"],
                "derived_name": "alpha_c_eff",
                "model": "intercept+A0/Ny^2",
            }
        )
    a0 = central["primary"]["A0"]
    alternatives = [row["coefficient"] for row in lmin_rows[1:]] + [central["with_inverse_Ny_fourth"]["A0"]]
    sensitivity = max((abs(value - a0) / abs(a0) for value in alternatives), default=0.0) if a0 else math.inf
    central_gate = {
        "A0_negative": bool(a0 < 0.0),
        "bootstrap_interval_excludes_zero_with_negative_sign": bool(a0_high < 0.0),
        "maximum_sensitivity_fraction": float(sensitivity),
        "sensitivity_threshold": FINITE_SIZE_STABILITY_THRESHOLD,
        "sensitivity_passes": bool(sensitivity <= FINITE_SIZE_STABILITY_THRESHOLD),
    }
    central_gate["passes"] = bool(
        central_gate["A0_negative"]
        and central_gate["bootstrap_interval_excludes_zero_with_negative_sign"]
        and central_gate["sensitivity_passes"]
    )
    central["primary"]["A0_ci95"] = [a0_low, a0_high]
    central["primary"]["alpha_c_eff_ci95"] = [
        -6.0 * a0_high / math.pi,
        -6.0 * a0_low / math.pi,
    ]
    central["primary"]["acceptance"] = central_gate
    rows = list(lmin_rows)
    rows.append(
        {
            "fit_family": "leading_free_energy",
            "Ny_min": int(ny[0]),
            "points": int(len(ny)),
            "intercept": central["with_inverse_Ny_fourth"]["intercept"],
            "coefficient": central["with_inverse_Ny_fourth"]["A0"],
            "derived_value": -6.0 * central["with_inverse_Ny_fourth"]["A0"] / math.pi,
            "derived_name": "alpha_c_eff",
            "model": "intercept+A0/Ny^2+B0/Ny^4",
        }
    )
    for level, gap in enumerate(central["gaps"], start=1):
        ci = percentile_interval(bootstrap_ai[:, level - 1])
        intercept_ci = percentile_interval(bootstrap_intercepts[:, level - 1])
        density = (means[:, 0] - means[:, level]) / ny
        subleading = _fit_quadratic_correction(1.0 / ny**2, density, intercept=False)
        omit = _fit_line(1.0 / ny[1:] ** 2, density[1:], intercept=False)
        ai = gap["Ai"]
        stability = max(abs(subleading["coefficient"] - ai), abs(omit["coefficient"] - ai)) / abs(ai) if ai else math.inf
        gap["Ai_ci95"] = list(ci)
        gap["alpha_x_typ_ci95"] = [ci[0] / (2 * math.pi), ci[1] / (2 * math.pi)]
        gap["unconstrained_intercept_ci95"] = list(intercept_ci)
        gap["subleading_Ai"] = subleading["coefficient"]
        gap["omit_Ny20_Ai"] = omit["coefficient"]
        gap["finite_size_stability_fraction"] = float(stability)
        gap["finite_size_passes"] = bool(stability <= FINITE_SIZE_STABILITY_THRESHOLD and intercept_ci[0] <= 0 <= intercept_ci[1])
        rows.extend(
            [
                {
                    "fit_family": f"gap_{level}", "Ny_min": int(ny[0]), "points": int(len(ny)),
                    "intercept": 0.0, "coefficient": ai, "derived_value": gap["alpha_x_typ"],
                    "derived_name": "alpha_x_typ", "model": "Ai/Ny^2",
                },
                {
                    "fit_family": f"gap_{level}", "Ny_min": int(ny[0]), "points": int(len(ny)),
                    "intercept": gap["intercept"], "coefficient": gap["unconstrained_Ai"],
                    "derived_value": gap["unconstrained_Ai"] / (2 * math.pi),
                    "derived_name": "alpha_x_typ", "model": "intercept+Ai/Ny^2",
                },
                {
                    "fit_family": f"gap_{level}", "Ny_min": int(ny[1]), "points": int(len(ny) - 1),
                    "intercept": 0.0, "coefficient": omit["coefficient"],
                    "derived_value": omit["coefficient"] / (2 * math.pi),
                    "derived_name": "alpha_x_typ", "model": "Ai/Ny^2 omit Ny20",
                },
            ]
        )
    return central, rows


def sample_convergence(
    by_ny: dict[int, dict[str, np.ndarray]], *, counts: Iterable[int], repeats: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    ny = np.asarray(sorted(by_ny), dtype=np.int64)
    rows = []
    for sample_count in counts:
        if any(by_ny[int(value)]["late_slopes"].shape[0] < sample_count for value in ny):
            continue
        draws = 1 if sample_count == 100 else repeats
        estimates = {"A0": [], **{f"Ai{level}": [] for level in range(1, 5)}}
        for _ in range(draws):
            means = []
            for value in ny:
                slopes = by_ny[int(value)]["late_slopes"]
                chosen = rng.choice(slopes.shape[0], size=sample_count, replace=False)
                means.append(slopes[chosen].mean(axis=0))
            fit = fit_finite_size(ny, np.stack(means))
            estimates["A0"].append(fit["primary"]["A0"])
            for level in range(1, 5):
                estimates[f"Ai{level}"].append(fit["gaps"][level - 1]["Ai"])
        for name, values in estimates.items():
            values = np.asarray(values)
            low, high = percentile_interval(values) if values.size > 1 else (float(values[0]), float(values[0]))
            rows.append(
                {
                    "samples_per_Ny": int(sample_count),
                    "metric": name,
                    "mean": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "subset_repeats": int(draws),
                    "selection": "random_without_replacement" if sample_count < 100 else "full_ensemble",
                }
            )
    return rows


def localization_summary(
    by_ny: dict[int, dict[str, np.ndarray]], *, bootstrap_count: int, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    profiles = []
    for ny in sorted(by_ny):
        weights = by_ny[ny]["soft_mode_wall_weights"][:, -1, :SOFT_SUBSPACE_SIZE]
        fraction = weights.sum(axis=-1).mean(axis=-1)
        selected = rng.integers(0, fraction.size, size=(bootstrap_count, fraction.size))
        low, high = percentile_interval(fraction[selected].mean(axis=1))
        rows.append(
            {
                "Ny": ny,
                "samples": int(fraction.size),
                "soft_subspace_modes": SOFT_SUBSPACE_SIZE,
                "combined_wall_fraction_mean": float(fraction.mean()),
                "combined_wall_fraction_ci_low": low,
                "combined_wall_fraction_ci_high": high,
                "passes_boundary_localization": bool(low > 0.80),
            }
        )
        profile = by_ny[ny]["soft_mode_x_profiles"][:, -1, :SOFT_SUBSPACE_SIZE].mean(axis=(0, 1))
        for x, value in enumerate(profile):
            profiles.append({"Ny": ny, "x": x, "soft_subspace_profile": float(value)})
    return rows, profiles


def record_closure_rows(
    by_ny: dict[int, dict[str, np.ndarray]], *, bootstrap_count: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    rows = []
    for ny in sorted(by_ny):
        slopes = by_ny[ny]["late_slopes"][:, 0]
        omega = np.asarray(
            [record_window_slopes(np.arange(2 * ny + 1), row, ny=ny)[1] for row in by_ny[ny]["cumulative_log_probability"]]
        )
        difference = omega - slopes
        chosen = rng.integers(0, difference.size, size=(bootstrap_count, difference.size))
        low, high = percentile_interval(difference[chosen].mean(axis=1))
        relative = abs(float(difference.mean())) / abs(float(slopes.mean()))
        rows.append(
            {
                "Ny": ny,
                "samples": int(slopes.size),
                "late_lambda0_mean": float(slopes.mean()),
                "late_lambda0_sem": sem(slopes),
                "late_omega_slope_mean": float(omega.mean()),
                "late_omega_slope_sem": sem(omega),
                "omega_minus_lambda0": float(difference.mean()),
                "closure_ci_low": low,
                "closure_ci_high": high,
                "relative_closure": relative,
                "same_window_closure_below_one_percent": bool(relative < 0.01),
                "endpoint_record_rate_mean": float((-by_ny[ny]["cumulative_log_probability"][:, -1] / (2 * ny)).mean()),
                "endpoint_is_finite_depth_diagnostic_only": True,
            }
        )
    return rows


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
        }
    )


def _panel_letters(axes: Iterable[Any]) -> None:
    for label, axis in zip("abcdefghijkl", axes):
        axis.text(-0.18, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")


def make_main_figure(
    by_ny: dict[int, dict[str, np.ndarray]],
    convergence: list[dict[str, Any]],
    closure: list[dict[str, Any]],
    finite: dict[str, Any],
    profile_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    configure_plotting()
    sizes = sorted(by_ny)
    colors = dict(zip(sizes, plt.cm.viridis(np.linspace(0.05, 0.92, len(sizes)))))
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.7), constrained_layout=True)
    for ny in sizes:
        cycles = by_ny[ny]["spectrum_cycles"]
        values = by_ny[ny]["levels"][:, :, 0]
        endpoint_x, endpoint_y, endpoint_err = [], [], []
        for endpoint in cycles[cycles >= ny]:
            mask = (cycles >= max(ny, endpoint - ny / 2)) & (cycles <= endpoint)
            if mask.sum() < 2:
                continue
            slopes = np.asarray([linear_slope(cycles[mask], row[mask]) for row in values])
            endpoint_x.append(endpoint / ny)
            endpoint_y.append(-slopes.mean())
            endpoint_err.append(sem(slopes))
        axes[0, 0].errorbar(endpoint_x, endpoint_y, yerr=endpoint_err, color=colors[ny], marker="o", ms=2, lw=0.8, label=fr"${ny}$")
    metric_markers = {"lambda0": "o", "gap1": "s", "gap2": "^", "gap3": "D", "gap4": "v"}
    for metric, marker in metric_markers.items():
        rows = [row for row in convergence if row["metric"] == metric]
        label = r"$\lambda_0$" if metric == "lambda0" else fr"$\Delta_{metric[-1]}$"
        axes[0, 1].plot([row["Ny"] for row in rows], [100 * row["shift"] / abs(row["late_mean"]) for row in rows], marker=marker, lw=0.8, ms=3, label=label)
    axes[0, 1].axhspan(-10, 10, color="0.9", zorder=-5)
    x = [-row["late_lambda0_mean"] for row in closure]
    y = [-row["late_omega_slope_mean"] for row in closure]
    axes[0, 2].errorbar(
        x,
        y,
        xerr=[row["late_lambda0_sem"] for row in closure],
        yerr=[row["late_omega_slope_sem"] for row in closure],
        fmt="o",
        color="tab:blue",
        capsize=2,
    )
    lo, hi = min(x + y), max(x + y)
    axes[0, 2].plot([lo, hi], [lo, hi], "--", color="0.35", lw=0.8)
    inv = np.asarray(finite["inverse_Ny_squared"])
    f0 = np.asarray(finite["f0_tilde"])
    f0_sem = np.asarray(
        [sem(by_ny[int(ny)]["late_slopes"][:, 0]) / ny for ny in finite["Ny"]]
    )
    axes[1, 0].errorbar(inv, f0, yerr=f0_sem, fmt="o", color="tab:blue", capsize=2)
    grid = np.linspace(0, inv.max() * 1.05, 200)
    axes[1, 0].plot(grid, finite["primary"]["intercept"] + finite["primary"]["A0"] * grid, "--", color="black", label=r"$N_y^{-2}$")
    corrected = finite["with_inverse_Ny_fourth"]
    axes[1, 0].plot(grid, corrected["intercept"] + corrected["A0"] * grid + corrected["B0"] * grid**2, ":", color="tab:red", label=r"$N_y^{-2}+N_y^{-4}$")
    ny_array = np.asarray(finite["Ny"], dtype=float)
    gap_colors = ("tab:blue", "tab:green", "tab:purple", "tab:pink")
    for color, gap in zip(gap_colors, finite["gaps"]):
        raw_gap = [by_ny[int(ny)]["late_slopes"][:, 0] - by_ny[int(ny)]["late_slopes"][:, gap["level"]] for ny in ny_array]
        density = np.asarray([values.mean() / ny for values, ny in zip(raw_gap, ny_array)])
        errors = np.asarray([sem(values) / ny for values, ny in zip(raw_gap, ny_array)])
        axes[1, 1].errorbar(inv, density, yerr=errors, fmt="o", ms=3, capsize=1.5, color=color, label=fr"$i={gap['level']}$")
        axes[1, 1].plot(grid, gap["Ai"] * grid, "-", lw=0.7, color=color)
    for ny in (20, 24, 30, 40):
        rows = [row for row in profile_rows if row["Ny"] == ny]
        axes[1, 2].plot([row["x"] for row in rows], [row["soft_subspace_profile"] for row in rows], "o-", ms=2, lw=0.8, color=colors[ny], label=fr"${ny}$")
    axes[0, 0].set(xlabel=r"window endpoint $t/N_y$", ylabel=r"local $-d\ell_0/dt$")
    axes[0, 1].set(xlabel=r"$N_y$", ylabel=r"middle-to-late shift (\%)")
    axes[0, 2].set(xlabel=r"$-\lambda_0$", ylabel=r"$-d\omega/dt$")
    axes[1, 0].set(xlabel=r"$1/N_y^2$", ylabel=r"$-\lambda_0/N_y$")
    axes[1, 1].set(xlabel=r"$1/N_y^2$", ylabel=r"$(\lambda_0-\lambda_i)/N_y$")
    axes[1, 2].set(xlabel=r"transverse coordinate $x$", ylabel="soft-subspace weight/mode")
    axes[0, 0].legend(title=r"$N_y$", ncol=2, fontsize=6, title_fontsize=6)
    axes[0, 1].legend(ncol=2, fontsize=6)
    axes[1, 0].legend(fontsize=6)
    axes[1, 1].legend(ncol=2, fontsize=6)
    axes[1, 2].legend(title=r"$N_y$", fontsize=6, title_fontsize=6)
    _panel_letters(axes.flat)
    fig.savefig(output_root / "many_body_lyapunov_v2_diagnostic.pdf")
    fig.savefig(output_root / "many_body_lyapunov_v2_diagnostic.png", dpi=300)
    plt.close(fig)


def make_supplementary_figure(
    by_ny: dict[int, dict[str, np.ndarray]],
    sample_rows: list[dict[str, Any]],
    numerical_rows: list[dict[str, Any]],
    localization_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    configure_plotting()
    sizes = sorted(by_ny)
    colors = dict(zip(sizes, plt.cm.viridis(np.linspace(0.05, 0.92, len(sizes)))))
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.6), constrained_layout=True)
    for ny in sizes:
        t = by_ny[ny]["spectrum_cycles"] / ny
        axes[0, 0].plot(t, np.nanmean(by_ny[ny]["entropy_nats"], axis=0) / ny, color=colors[ny], lw=0.8, label=fr"${ny}$")
        axes[0, 1].plot(t, np.nanmean(by_ny[ny]["charge_variance"], axis=0) / ny, color=colors[ny], lw=0.8)
        axes[0, 2].plot(t, np.nanmean(by_ny[ny]["cap_fraction"], axis=0), color=colors[ny], lw=0.8)
    for metric, marker in (("A0", "o"), ("Ai1", "s"), ("Ai2", "^"), ("Ai3", "D"), ("Ai4", "v")):
        rows = [row for row in sample_rows if row["metric"] == metric]
        axes[1, 0].errorbar(
            [row["samples_per_Ny"] for row in rows],
            [row["mean"] for row in rows],
            yerr=[
                [row["mean"] - row["ci_low"] for row in rows],
                [row["ci_high"] - row["mean"] for row in rows],
            ],
            marker=marker,
            ms=3,
            capsize=2,
            label=metric,
        )
    names = list(NUMERICAL_LIMITS)
    maxima = [max(float(row.get(name, 0.0)) for row in numerical_rows) for name in names]
    axes[1, 1].bar(np.arange(len(names)), maxima, color="0.4")
    axes[1, 1].set_yscale("symlog", linthresh=1e-18)
    axes[1, 1].set_xticks(np.arange(len(names)), ["Herm.", "eig.", "Gram", "bound", "ext. coup.", "ext. prod."], rotation=35, ha="right")
    axes[1, 2].errorbar(
        [row["Ny"] for row in localization_rows],
        [row["combined_wall_fraction_mean"] for row in localization_rows],
        yerr=[
            [row["combined_wall_fraction_mean"] - row["combined_wall_fraction_ci_low"] for row in localization_rows],
            [row["combined_wall_fraction_ci_high"] - row["combined_wall_fraction_mean"] for row in localization_rows],
        ],
        marker="o", color="tab:blue", capsize=2,
    )
    axes[1, 2].axhline(0.8, ls="--", color="0.4", lw=0.8)
    axes[0, 0].set(xlabel=r"$t/N_y$", ylabel=r"entropy$/N_y$")
    axes[0, 1].set(xlabel=r"$t/N_y$", ylabel=r"charge variance$/N_y$")
    axes[0, 2].set(xlabel=r"$t/N_y$", ylabel="exact-cap fraction")
    axes[1, 0].set(xlabel=r"subset size $S$ per $N_y$", ylabel="finite-size coefficient")
    axes[1, 1].set(ylabel="maximum residual")
    axes[1, 2].set(xlabel=r"$N_y$", ylabel="four-mode wall fraction")
    axes[0, 0].legend(title=r"$N_y$", ncol=2, fontsize=6, title_fontsize=6)
    axes[1, 0].legend(ncol=2, fontsize=6)
    _panel_letters(axes.flat)
    fig.savefig(output_root / "many_body_lyapunov_v2_supplementary.pdf")
    fig.savefig(output_root / "many_body_lyapunov_v2_supplementary.png", dpi=300)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-count", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    results_root = args.results_root.resolve()
    output_root = args.output_root.resolve()
    bootstrap_count = int(args.bootstrap_count or config["analysis"].get("bootstrap_replicates", BOOTSTRAP_DEFAULT))
    seed = int(config["analysis"]["bootstrap_seed"])
    by_ny, trajectory_rows, numerical_rows, provenance = load_and_verify_campaign(
        config=config, results_root=results_root, allow_incomplete=args.allow_incomplete
    )
    if len(by_ny) < 4:
        raise RuntimeError("finite-size analysis needs at least four circumferences")
    convergence_rows = []
    for ny in sorted(by_ny):
        rows = paired_convergence(
            by_ny[ny]["middle_slopes"],
            by_ny[ny]["late_slopes"],
            bootstrap_count=bootstrap_count,
            rng=np.random.default_rng(seed + ny),
            relative_threshold=float(config["analysis"]["relative_shift_threshold"]),
        )
        convergence_rows.extend({"Ny": ny, **row} for row in rows)
    finite, finite_rows = bootstrap_finite_size(
        by_ny, count=bootstrap_count, rng=np.random.default_rng(seed + 1000)
    )
    for gap in finite["gaps"]:
        temporal = all(
            row["passes"] for row in convergence_rows if row["metric"] == f"gap{gap['level']}"
        )
        gap["temporal_stability_passes_all_sizes"] = temporal
        gap["acceptance_passes"] = bool(temporal and gap["finite_size_passes"])
        gap["reported_alpha_x_typ"] = gap["alpha_x_typ"] if gap["acceptance_passes"] else None
        gap["reported_x_typ_over_c_eff"] = (
            gap["x_typ_over_c_eff"]
            if gap["acceptance_passes"] and finite["primary"]["acceptance"]["passes"]
            else None
        )
    finite["primary"]["reported_alpha_c_eff"] = (
        finite["primary"]["alpha_c_eff"] if finite["primary"]["acceptance"]["passes"] else None
    )
    finite["precision_cft_claim_supported"] = bool(
        finite["primary"]["acceptance"]["passes"] and any(gap["acceptance_passes"] for gap in finite["gaps"])
    )
    closure_rows = record_closure_rows(
        by_ny, bootstrap_count=bootstrap_count, rng=np.random.default_rng(seed + 2000)
    )
    sample_rows = sample_convergence(
        by_ny,
        counts=config["analysis"]["sample_prefixes"],
        repeats=bootstrap_count,
        rng=np.random.default_rng(seed + 3000),
    )
    localization_rows, profile_rows = localization_summary(
        by_ny, bootstrap_count=bootstrap_count, rng=np.random.default_rng(seed + 4000)
    )
    size_rows = []
    for ny in sorted(by_ny):
        temporal = [row for row in convergence_rows if row["Ny"] == ny]
        size_rows.append(
            {
                "Ny": ny,
                "samples": int(by_ny[ny]["late_slopes"].shape[0]),
                "T": 2 * ny,
                "T_2Ny_sufficient": bool(all(row["passes"] for row in temporal)),
                "failed_temporal_metrics": ";".join(row["metric"] for row in temporal if not row["passes"]),
                "four_mode_wall_fraction": next(row["combined_wall_fraction_mean"] for row in localization_rows if row["Ny"] == ny),
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "trajectory_slopes_and_record_rates.csv", trajectory_rows)
    write_csv(output_root / "temporal_convergence.csv", convergence_rows)
    write_csv(output_root / "record_weight_closure.csv", closure_rows)
    write_csv(output_root / "finite_size_fits.csv", finite_rows)
    if sample_rows:
        write_csv(output_root / "sample_convergence.csv", sample_rows)
    write_csv(output_root / "localization.csv", localization_rows)
    write_csv(output_root / "localization_profiles.csv", profile_rows)
    write_csv(output_root / "numerical_diagnostics.csv", numerical_rows)
    write_csv(output_root / "size_summary.csv", size_rows)
    current_engine = provenance["current_source_hashes"].get("src/classA_U1FGTN_gpu.py")
    executed_engine = provenance["executed_source_hashes"].get("src/classA_U1FGTN_gpu.py")
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "sampling_revision": config["sampling_revision"],
        "bootstrap_seed": seed,
        "bootstrap_replicates": bootstrap_count,
        "verified_tasks": provenance["verified_tasks"],
        "requested_tasks": provenance["requested_tasks"],
        "verified_files": provenance["verified_files"],
        "verified_trajectories": len(trajectory_rows),
        "requested_trajectories": EXPECTED_TRAJECTORIES,
        "historical_provenance": {
            "download_manifest": provenance["manifest_path"],
            "historical_manifest_used": provenance["historical_manifest_used"],
            "configuration_sha256": config_hash(config),
            "executed_source_hashes": provenance["executed_source_hashes"],
            "current_source_hashes": provenance["current_source_hashes"],
            "source_hash_drift": provenance["source_hash_drift"],
            "executed_engine_sha256": executed_engine,
            "current_canonical_engine_sha256": current_engine,
        },
        "reconstruction": {
            "convention": "ell=log(sigma^2)",
            "leading_levels_reconstructed": 64,
            "exact_caps_preserved_as_infinite_flip_costs": True,
            "negative_infinity_padding_excluded_from_slopes": True,
            "all_products_passed_roundoff_reconstruction": True,
        },
        "temporal_convergence": {
            "windows": {"middle": "[Ny,3Ny/2]", "late": "[3Ny/2,2Ny]"},
            "relative_shift_threshold": RELATIVE_SHIFT_THRESHOLD,
            "sizes": size_rows,
            "all_sizes_pass": bool(all(row["T_2Ny_sufficient"] for row in size_rows)),
        },
        "record_weight_closure": {
            "comparison": "late-window d(omega)/dt versus late-window lambda0",
            "all_relative_discrepancies_below_one_percent": bool(all(row["same_window_closure_below_one_percent"] for row in closure_rows)),
            "endpoint_ratio_retained_as_diagnostic_only": True,
            "rows": closure_rows,
        },
        "finite_size": finite,
        "localization": {
            "method": "aggregate trace weight of first four soft-mode subspace; no individual eigenvector labels",
            "all_sizes_pass_boundary_localization": bool(all(row["passes_boundary_localization"] for row in localization_rows)),
            "rows": localization_rows,
        },
        "acceptance_decisions": {
            "spectral_reconstruction_validated": True,
            "boundary_localization_validated": bool(all(row["passes_boundary_localization"] for row in localization_rows)),
            "T_2Ny_temporally_sufficient": bool(all(row["T_2Ny_sufficient"] for row in size_rows)),
            "alpha_c_eff_reportable": bool(finite["primary"]["acceptance"]["passes"]),
            "absolute_operator_dimensions_reportable": False,
            "x_over_c_eff_reportable": bool(any(gap["reported_x_typ_over_c_eff"] is not None for gap in finite["gaps"])),
            "precision_cft_claim_supported": bool(finite["precision_cft_claim_supported"]),
        },
        "interpretation": (
            "The completed v2 pilot validates the Gaussian many-body-spectrum reconstruction, "
            "same-window record-weight closure, and boundary localization. It does not support "
            "a precision Zabalo-style c_eff or absolute scaling-dimension extraction at T=2Ny."
        ),
        "recommended_follow_up": {
            "launch_authorized": False,
            "proposal": "fresh T=4Ny, S=100 depth study at Ny=20,30,40",
            "must_restart_from_t0": True,
            "reason": "v2 did not save final covariance and RNG state",
        },
    }
    save_json_atomic(output_root / "analysis_summary.json", summary)
    if len(by_ny) == 8 and all(values["late_slopes"].shape[0] == 100 for values in by_ny.values()):
        make_main_figure(by_ny, convergence_rows, closure_rows, finite, profile_rows, output_root)
        make_supplementary_figure(by_ny, sample_rows, numerical_rows, localization_rows, output_root)
    print(
        f"[analysis] verified {provenance['verified_tasks']} tasks / {len(trajectory_rows)} trajectories; wrote {output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
