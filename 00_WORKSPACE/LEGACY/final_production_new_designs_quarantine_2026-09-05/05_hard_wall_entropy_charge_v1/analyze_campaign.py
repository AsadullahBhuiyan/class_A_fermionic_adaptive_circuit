#!/usr/bin/env python3
"""Analyze the two-lane hard-wall entropy and charge campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from entropy_charge_observer import OBSERVER_SCHEMA


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_NY = (30, 35, 40, 45, 50, 55, 60)
EXPECTED_SAMPLES = 100
EXPECTED_SHARDS = 140
SAMPLING_REVISION = "hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v1"
BUNDLE_NAME = "05_hard_wall_entropy_charge"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
RESULT_SCHEMA = "hard_wall_entropy_charge_result_shard_v1"
COMPLETION_SCHEMA = "hard_wall_entropy_charge_completion_v1"
ROOT_SEED = 2026090305
EXECUTION_BATCH_SIZE = {30: 80, 35: 60, 40: 40, 45: 30, 50: 25, 55: 20, 60: 20}
SOURCE_FILES = {
    "run_campaign.py",
    "entropy_charge_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
}
BOOTSTRAP_SEED = 2026090306
DEFAULT_BOOTSTRAP_COUNT = 20_000
WALL_WINDOWS = {"left": (4, 5, 6), "right": (14, 15, 16)}


@dataclass(frozen=True)
class VerifiedResult:
    result_path: Path
    completion_path: Path
    completion: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sibling_runner() -> Any:
    """Load this bundle's runner without colliding with other ``run_campaign`` modules."""

    module_name = "_hard_wall_entropy_charge_analysis_runner"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    runner_path = PACKAGE_ROOT / "run_campaign.py"
    specification = importlib.util.spec_from_file_location(module_name, runner_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load hard-wall runner from {runner_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def log_chord(ay: np.ndarray | Iterable[int], *, ny: int) -> np.ndarray:
    ay = np.asarray(ay, dtype=np.float64)
    if np.any(ay <= 0) or np.any(ay >= int(ny)):
        raise ValueError("log-chord widths must lie strictly between zero and ny")
    return np.log((float(ny) / math.pi) * np.sin(math.pi * ay / float(ny)))


def fit_curve(
    ay: np.ndarray,
    values: np.ndarray,
    *,
    ny: int,
    fit_min: int = 8,
    trim_endpoints: bool = False,
) -> dict[str, float | int]:
    """Fit one mean curve to the locked log-chord window."""

    ay = np.asarray(ay, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if ay.shape != values.shape:
        raise ValueError("ay and values must have the same shape")
    mask = (ay >= int(fit_min)) & (ay <= int(ny) // 2) & np.isfinite(values)
    selected = np.flatnonzero(mask)
    if trim_endpoints and selected.size >= 4:
        selected = selected[1:-1]
    if selected.size < 2:
        raise ValueError("the fit window contains fewer than two finite points")
    x = log_chord(ay[selected], ny=ny)
    y = values[selected]
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float("nan") if total == 0.0 else 1.0 - residual / total,
        "fit_ay_min": int(ay[selected[0]]),
        "fit_ay_max": int(ay[selected[-1]]),
        "fit_point_count": int(selected.size),
    }


def _trajectory_slopes(
    ay: np.ndarray,
    curves: np.ndarray,
    *,
    ny: int,
    fit_min: int = 8,
    trim_endpoints: bool = False,
) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim < 2 or curves.shape[-1] != len(ay):
        raise ValueError("curves must end in the ay axis")
    mask = (np.asarray(ay) >= int(fit_min)) & (np.asarray(ay) <= int(ny) // 2)
    selected = np.flatnonzero(mask)
    if trim_endpoints and selected.size >= 4:
        selected = selected[1:-1]
    if selected.size < 2:
        raise ValueError("the trajectory fit window contains fewer than two points")
    values = curves[..., selected]
    if not np.isfinite(values).all():
        raise FloatingPointError("trajectory curves contain nonfinite fit values")
    x = log_chord(np.asarray(ay)[selected], ny=ny)
    centered = x - x.mean()
    return np.sum(values * centered, axis=-1) / float(np.dot(centered, centered))


def paired_bootstrap_prefactors(
    ay: np.ndarray,
    entropy: np.ndarray,
    charge_variance: np.ndarray,
    *,
    ny: int,
    bootstrap_indices: np.ndarray,
    trim_endpoints: bool = False,
    chunk_size: int = 500,
) -> dict[str, np.ndarray]:
    """Bootstrap whole trajectory IDs, pairing entropy and charge resamples."""

    entropy_slope = _trajectory_slopes(
        ay, entropy, ny=ny, trim_endpoints=trim_endpoints
    )
    variance_slope = _trajectory_slopes(
        ay, charge_variance, ny=ny, trim_endpoints=trim_endpoints
    )
    indices = np.asarray(bootstrap_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != entropy.shape[0]:
        raise ValueError("bootstrap indices do not match the trajectory axis")
    if entropy.shape[0] != charge_variance.shape[0]:
        raise ValueError("entropy and charge must share the trajectory axis")
    context_shape = entropy_slope.shape[1:]
    c_values = np.empty((indices.shape[0], *context_shape), dtype=np.float64)
    k_values = np.empty_like(c_values)
    for start in range(0, len(indices), int(chunk_size)):
        stop = min(len(indices), start + int(chunk_size))
        chosen = indices[start:stop]
        c_values[start:stop] = 3.0 * entropy_slope[chosen].mean(axis=1)
        k_values[start:stop] = math.pi**2 * variance_slope[chosen].mean(axis=1)
    return {"c": c_values, "k": k_values}


def percentile_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.percentile(np.asarray(values), [2.5, 97.5], axis=0)
    return low, high


def _completion_result_name(payload: Mapping[str, Any]) -> str | None:
    for key in ("result_filename", "result_file", "filename"):
        value = payload.get(key)
        if isinstance(value, str) and value.endswith(".npz"):
            return value
    result = payload.get("result")
    if isinstance(result, str) and result.endswith(".npz"):
        return result
    if isinstance(result, Mapping):
        for key in ("filename", "path", "name"):
            value = result.get(key)
            if isinstance(value, str) and value.endswith(".npz"):
                return value
    return None


def _completion_size_sha(payload: Mapping[str, Any]) -> tuple[int | None, str | None]:
    size = payload.get("result_bytes", payload.get("bytes"))
    digest = payload.get("result_sha256", payload.get("sha256"))
    result = payload.get("result")
    if isinstance(result, Mapping):
        size = result.get("bytes", size)
        digest = result.get("sha256", digest)
    return (None if size is None else int(size), None if digest is None else str(digest))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _expected_execution_identity(ny: int, sample_start: int) -> tuple[str, str, int]:
    lane = "A" if int(ny) in (40, 60) else "B"
    batch_size = EXECUTION_BATCH_SIZE[int(ny)]
    batch_index = int(sample_start) // batch_size
    batch_start = batch_index * batch_size
    batch_stop = min(batch_start + batch_size, EXPECTED_SAMPLES)
    task_id = (
        f"lane-{lane}_Ny{int(ny):03d}_execution-{batch_index:03d}_"
        f"samples-{batch_start:03d}-{batch_stop - 1:03d}"
    )
    label = (
        f"{ROOT_SEED}|lane={lane}|Nx=20|Ny={int(ny)}|execution={batch_index}|"
        f"samples={batch_start}:{batch_stop}"
    )
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")
    seed &= (1 << 63) - 1
    return lane, task_id, seed


def discover_verified_results(output_root: Path) -> list[VerifiedResult]:
    """Find result NPZs whose completion JSON verifies bytes and SHA-256."""

    output_root = Path(output_root)
    completions: dict[Path, tuple[Path, dict[str, Any], int, str]] = {}
    for json_path in output_root.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        filename = _completion_result_name(payload)
        size, digest = _completion_size_sha(payload)
        if filename is None or size is None or digest is None:
            continue
        ny = int(payload.get("Ny", -1))
        indices = [int(value) for value in payload.get("global_sample_indices", ())]
        sample_start = int(payload.get("sample_start", -1))
        sample_stop = int(payload.get("sample_stop", -1))
        shard_index = int(payload.get("shard_index", -1))
        lane, execution_batch_id, batch_seed = (
            _expected_execution_identity(ny, sample_start)
            if ny in EXPECTED_NY and 0 <= sample_start < EXPECTED_SAMPLES
            else ("", "", -1)
        )
        expected_task_id = (
            f"Ny{ny:03d}_shard-{shard_index:03d}_"
            f"samples-{sample_start:03d}-{sample_stop - 1:03d}"
        )
        expected_filename = (
            f"shard_{shard_index:03d}_samples_{sample_start:03d}-{sample_stop - 1:03d}.npz"
        )
        source_hashes = payload.get("source_hashes")
        required_identity = (
            payload.get("schema") == COMPLETION_SCHEMA
            and payload.get("status") == "complete"
            and payload.get("bundle") == BUNDLE_NAME
            and payload.get("sampling_revision") == SAMPLING_REVISION
            and payload.get("lane") == lane
            and payload.get("observer_schema") == OBSERVER_SCHEMA
            and payload.get("canonical_entry_point") == CANONICAL_ENTRY_POINT
            and int(payload.get("Nx", -1)) == 20
            and ny in EXPECTED_NY
            and int(payload.get("cycles", -1)) == 2 * ny
            and bool(payload.get("detailed")) == (ny == 40)
            and int(payload.get("sample_count", -1)) == 5
            and sample_stop == sample_start + 5
            and indices == list(range(sample_start, sample_stop))
            and shard_index == sample_start // 5
            and payload.get("task_id") == expected_task_id
            and payload.get("execution_batch_id") == execution_batch_id
            and int(payload.get("batch_seed", -1)) == batch_seed
            and filename == expected_filename
            and size > 0
            and _is_sha256(digest)
            and _is_sha256(payload.get("config_sha256"))
            and isinstance(source_hashes, Mapping)
            and set(source_hashes) == SOURCE_FILES
            and all(_is_sha256(value) for value in source_hashes.values())
        )
        if not required_identity:
            continue
        result_path = Path(filename)
        if not result_path.is_absolute():
            result_path = json_path.parent / result_path
        if json_path.resolve() != result_path.with_suffix(".complete.json").resolve():
            continue
        completions[result_path.resolve()] = (
            json_path.resolve(),
            dict(payload),
            size,
            digest,
        )

    results: list[VerifiedResult] = []
    for result_path, (
        completion_path,
        completion,
        expected_size,
        expected_sha,
    ) in completions.items():
        if not result_path.is_file():
            raise RuntimeError(f"completed result is missing: {result_path}")
        if result_path.stat().st_size != expected_size:
            raise RuntimeError(f"completed result byte count mismatch: {result_path}")
        if sha256_file(result_path) != expected_sha:
            raise RuntimeError(f"completed result checksum mismatch: {result_path}")
        try:
            with np.load(result_path, allow_pickle=False) as data:
                schema = str(np.asarray(data["observer_schema"]).item())
        except (KeyError, ValueError):
            continue
        if schema == OBSERVER_SCHEMA:
            results.append(
                VerifiedResult(
                    result_path=result_path,
                    completion_path=completion_path,
                    completion=completion,
                )
            )
    if not results:
        raise RuntimeError(f"no verified {OBSERVER_SCHEMA} results found under {output_root}")
    return sorted(results, key=lambda item: str(item.result_path))


def load_case_data(results: Iterable[VerifiedResult]) -> dict[int, dict[str, np.ndarray]]:
    """Merge immutable five-trajectory results into one sorted case per Ny."""

    rows: dict[int, list[dict[str, np.ndarray]]] = {}
    campaign_config_sha256: str | None = None
    campaign_source_hashes: dict[str, str] | None = None
    for verified in results:
        path = verified.result_path
        completion_path = verified.completion_path
        completion = verified.completion
        if completion_path != path.with_suffix(".complete.json"):
            raise RuntimeError(f"verified result/completion paths are not siblings: {path}")
        if campaign_config_sha256 is None:
            campaign_config_sha256 = str(completion["config_sha256"])
            campaign_source_hashes = dict(completion["source_hashes"])
        elif completion["config_sha256"] != campaign_config_sha256 or completion[
            "source_hashes"
        ] != campaign_source_hashes:
            raise ValueError("verified shards do not share one config/source identity")
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        if str(payload["observer_schema"].item()) != OBSERVER_SCHEMA:
            raise ValueError(f"unexpected observer schema in {path}")
        if str(payload["schema"].item()) != RESULT_SCHEMA:
            raise ValueError(f"unexpected result schema in {path}")
        ny = int(payload["ny"].item())
        if int(payload["nx"].item()) != 20 or int(payload["physical_cycles"].item()) != 2 * ny:
            raise ValueError(f"geometry/cycle contract mismatch in {path}")
        if bool(payload["detailed"].item()) != (ny == 40):
            raise ValueError(f"detailed-observer contract mismatch in {path}")
        if ny != int(completion["Ny"]) or not np.array_equal(
            payload["sample_ids"],
            np.asarray(completion["global_sample_indices"], dtype=np.int64),
        ):
            raise ValueError(f"result/completion trajectory identity mismatch in {path}")
        scalar_matches = {
            "bundle": completion["bundle"],
            "sampling_revision": completion["sampling_revision"],
            "canonical_entry_point": completion["canonical_entry_point"],
            "lane": completion["lane"],
            "task_id": completion["task_id"],
            "execution_batch_id": completion["execution_batch_id"],
            "config_sha256": completion["config_sha256"],
            "Nx": 20,
            "Ny": ny,
            "cycles_total": 2 * ny,
            "detailed": ny == 40,
            "shard_index": completion["shard_index"],
            "sample_start": completion["sample_start"],
            "sample_stop": completion["sample_stop"],
            "execution_batch_seed": completion["batch_seed"],
        }
        for key, expected_value in scalar_matches.items():
            if key not in payload or payload[key].item() != expected_value:
                raise ValueError(f"result/completion field mismatch for {key} in {path}")
        if not np.array_equal(
            payload["global_sample_indices"],
            np.asarray(completion["global_sample_indices"], dtype=np.int64),
        ):
            raise ValueError(f"result/completion global IDs mismatch in {path}")
        fixed_result_metadata = {
            "wall_locations": np.asarray((5, 15), dtype=np.int64),
            "dtype": np.asarray("complex128"),
            "init_mode": np.asarray("default"),
            "sequence": np.asarray("raster_y"),
            "state_representation": np.asarray("physical_frame"),
            "cycle_zero_semantics": np.asarray(
                "after_born_conditioned_exterior_preparation"
            ),
        }
        for key, expected_value in fixed_result_metadata.items():
            if key not in payload or not np.array_equal(payload[key], expected_value):
                raise ValueError(f"result scientific metadata mismatch for {key} in {path}")
        rows.setdefault(ny, []).append(payload)

    cases: dict[int, dict[str, np.ndarray]] = {}
    for ny, shards in rows.items():
        sample_ids = np.concatenate([row["sample_ids"] for row in shards])
        order = np.argsort(sample_ids)
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise RuntimeError(f"duplicate sample IDs for Ny={ny}")
        shared_keys = {
            "cycles",
            "ay_values",
            "curve_cycles",
            "half_contour_cycles",
            "final_contour_valid",
        }
        merged: dict[str, np.ndarray] = {
            "ny": np.asarray(ny, dtype=np.int64),
            "sample_ids": sample_ids[order],
        }
        trajectory_keys = [
            "global_charge",
            "half_filling_offset",
            "endpoint_entropy",
            "endpoint_charge_mean",
            "endpoint_charge_variance",
            "entropy_curves",
            "charge_mean_curves",
            "charge_variance_curves",
            "half_strip_entropy_contour",
            "half_strip_charge_variance_contour",
            "final_entropy_contour",
            "final_charge_variance_contour",
        ]
        for key in trajectory_keys:
            present = [row[key] for row in shards if key in row]
            if present:
                if len(present) != len(shards):
                    raise ValueError(f"inconsistent shard key {key} for Ny={ny}")
                merged[key] = np.concatenate(present, axis=0)[order]
        for key in shared_keys:
            present = [row[key] for row in shards if key in row]
            if present:
                if any(not np.array_equal(present[0], value) for value in present[1:]):
                    raise ValueError(f"inconsistent shared axis {key} for Ny={ny}")
                merged[key] = present[0]
        cases[ny] = merged
    deployed_source_hashes = {
        relative: sha256_file(PACKAGE_ROOT / relative) for relative in SOURCE_FILES
    }
    if campaign_source_hashes != deployed_source_hashes:
        raise ValueError(
            "completion source hashes do not match the executable analysis bundle"
        )
    runner = _load_sibling_runner()
    if campaign_config_sha256 != runner.config_hash(runner.expected_config()):
        raise ValueError("completion config hash does not match the locked runner config")
    return cases


def validate_complete_cases(cases: Mapping[int, Mapping[str, np.ndarray]]) -> None:
    if tuple(sorted(cases)) != EXPECTED_NY:
        raise RuntimeError(f"expected Ny={EXPECTED_NY}, found {tuple(sorted(cases))}")
    for ny in EXPECTED_NY:
        case = cases[ny]
        if len(case["sample_ids"]) != EXPECTED_SAMPLES:
            raise RuntimeError(f"Ny={ny} has {len(case['sample_ids'])}, expected 100")
        if not np.array_equal(case["sample_ids"], np.arange(EXPECTED_SAMPLES)):
            raise RuntimeError(f"Ny={ny} sample IDs are not exactly 0..99")
        if not np.array_equal(case["cycles"], np.arange(2 * ny + 1)):
            raise ValueError(f"Ny={ny} has an invalid cycle axis")
        if not np.array_equal(case["ay_values"], np.arange(ny // 2 + 1)):
            raise ValueError(f"Ny={ny} has an invalid width axis")
        if case["global_charge"].shape != (100, 2 * ny + 1):
            raise ValueError(f"Ny={ny} has an invalid global-charge shape")
        for key in ("endpoint_entropy", "endpoint_charge_mean", "endpoint_charge_variance"):
            if case[key].shape != (100, ny // 2 + 1) or not np.isfinite(case[key]).all():
                raise ValueError(f"Ny={ny} has invalid {key}")
    detailed = cases[40]
    if not np.array_equal(detailed["curve_cycles"], np.arange(10, 81)):
        raise ValueError("Ny=40 detailed-cycle axis must be 10..80")
    if detailed["entropy_curves"].shape != (100, 71, 21):
        raise ValueError("Ny=40 entropy-curve shape is invalid")
    if detailed["half_strip_entropy_contour"].shape != (100, 81, 20, 20):
        raise ValueError("Ny=40 half-strip contour shape is invalid")
    if detailed["final_entropy_contour"].shape != (100, 21, 20, 20):
        raise ValueError("Ny=40 final-contour shape is invalid")


def final_scaling_analysis(
    cases: Mapping[int, Mapping[str, np.ndarray]],
    *,
    bootstrap_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ny in EXPECTED_NY:
        case = cases[ny]
        ay = case["ay_values"]
        entropy = case["endpoint_entropy"]
        variance = case["endpoint_charge_variance"]
        rng = np.random.default_rng(BOOTSTRAP_SEED + ny)
        indices = rng.integers(0, EXPECTED_SAMPLES, size=(bootstrap_count, EXPECTED_SAMPLES))
        boot = paired_bootstrap_prefactors(
            ay, entropy, variance, ny=ny, bootstrap_indices=indices
        )
        sensitivity = paired_bootstrap_prefactors(
            ay,
            entropy,
            variance,
            ny=ny,
            bootstrap_indices=indices,
            trim_endpoints=True,
        )
        entropy_fit = fit_curve(ay, entropy.mean(0), ny=ny)
        variance_fit = fit_curve(ay, variance.mean(0), ny=ny)
        entropy_trim = fit_curve(ay, entropy.mean(0), ny=ny, trim_endpoints=True)
        variance_trim = fit_curve(ay, variance.mean(0), ny=ny, trim_endpoints=True)
        c_low, c_high = percentile_interval(boot["c"])
        k_low, k_high = percentile_interval(boot["k"])
        c_trim_low, c_trim_high = percentile_interval(sensitivity["c"])
        k_trim_low, k_trim_high = percentile_interval(sensitivity["k"])
        rows.append(
            {
                "Ny": ny,
                "samples": EXPECTED_SAMPLES,
                "entropy_slope": entropy_fit["slope"],
                "entropy_intercept": entropy_fit["intercept"],
                "entropy_r2": entropy_fit["r2"],
                "c": 3.0 * float(entropy_fit["slope"]),
                "c_ci_low": float(c_low),
                "c_ci_high": float(c_high),
                "charge_variance_slope": variance_fit["slope"],
                "charge_variance_intercept": variance_fit["intercept"],
                "charge_variance_r2": variance_fit["r2"],
                "k": math.pi**2 * float(variance_fit["slope"]),
                "k_ci_low": float(k_low),
                "k_ci_high": float(k_high),
                "trimmed_c": 3.0 * float(entropy_trim["slope"]),
                "trimmed_c_ci_low": float(c_trim_low),
                "trimmed_c_ci_high": float(c_trim_high),
                "trimmed_k": math.pi**2 * float(variance_trim["slope"]),
                "trimmed_k_ci_low": float(k_trim_low),
                "trimmed_k_ci_high": float(k_trim_high),
                "fit_ay_min": entropy_fit["fit_ay_min"],
                "fit_ay_max": entropy_fit["fit_ay_max"],
                "trimmed_fit_ay_min": entropy_trim["fit_ay_min"],
                "trimmed_fit_ay_max": entropy_trim["fit_ay_max"],
            }
        )
    return rows


def ny40_dynamics_analysis(
    case: Mapping[str, np.ndarray], *, bootstrap_count: int
) -> list[dict[str, Any]]:
    ay = case["ay_values"]
    entropy = case["entropy_curves"]
    variance = case["charge_variance_curves"]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 4000)
    indices = rng.integers(0, EXPECTED_SAMPLES, size=(bootstrap_count, EXPECTED_SAMPLES))
    boot = paired_bootstrap_prefactors(
        ay, entropy, variance, ny=40, bootstrap_indices=indices
    )
    trimmed_boot = paired_bootstrap_prefactors(
        ay,
        entropy,
        variance,
        ny=40,
        bootstrap_indices=indices,
        trim_endpoints=True,
    )
    delta = boot["c"] - boot["k"]
    c_low, c_high = percentile_interval(boot["c"])
    k_low, k_high = percentile_interval(boot["k"])
    delta_low, delta_high = percentile_interval(delta)
    trimmed_c_low, trimmed_c_high = percentile_interval(trimmed_boot["c"])
    trimmed_k_low, trimmed_k_high = percentile_interval(trimmed_boot["k"])
    rows = []
    for time_index, cycle in enumerate(case["curve_cycles"]):
        entropy_fit = fit_curve(ay, entropy[:, time_index].mean(0), ny=40)
        variance_fit = fit_curve(ay, variance[:, time_index].mean(0), ny=40)
        entropy_trim = fit_curve(
            ay, entropy[:, time_index].mean(0), ny=40, trim_endpoints=True
        )
        variance_trim = fit_curve(
            ay, variance[:, time_index].mean(0), ny=40, trim_endpoints=True
        )
        c_value = 3.0 * float(entropy_fit["slope"])
        k_value = math.pi**2 * float(variance_fit["slope"])
        rows.append(
            {
                "cycle": int(cycle),
                "c": c_value,
                "c_ci_low": float(c_low[time_index]),
                "c_ci_high": float(c_high[time_index]),
                "k": k_value,
                "k_ci_low": float(k_low[time_index]),
                "k_ci_high": float(k_high[time_index]),
                "c_minus_k": c_value - k_value,
                "c_minus_k_ci_low": float(delta_low[time_index]),
                "c_minus_k_ci_high": float(delta_high[time_index]),
                "entropy_r2": entropy_fit["r2"],
                "charge_variance_r2": variance_fit["r2"],
                "trimmed_c": 3.0 * float(entropy_trim["slope"]),
                "trimmed_c_ci_low": float(trimmed_c_low[time_index]),
                "trimmed_c_ci_high": float(trimmed_c_high[time_index]),
                "trimmed_k": math.pi**2 * float(variance_trim["slope"]),
                "trimmed_k_ci_low": float(trimmed_k_low[time_index]),
                "trimmed_k_ci_high": float(trimmed_k_high[time_index]),
            }
        )
    return rows


def wall_integrated_curves(
    contour: np.ndarray, *, wall_x: Iterable[int]
) -> np.ndarray:
    """Integrate a padded all-Ay contour over one three-column wall window."""

    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim != 4:
        raise ValueError("final contour must have shape (samples,Ay,x,dy)")
    samples, width_count, nx, max_ay = contour.shape
    x_values = np.asarray(tuple(wall_x), dtype=np.int64)
    if np.any(x_values < 0) or np.any(x_values >= nx):
        raise ValueError("wall window lies outside the x axis")
    curves = np.zeros((samples, width_count), dtype=np.float64)
    for ay in range(1, min(width_count, max_ay + 1)):
        curves[:, ay] = contour[:, ay, x_values, :ay].sum(axis=(1, 2))
    return curves


def wall_decomposition_analysis(
    case: Mapping[str, np.ndarray], *, bootstrap_count: int
) -> list[dict[str, Any]]:
    ay = case["ay_values"]
    entropy_full = case["endpoint_entropy"]
    variance_full = case["endpoint_charge_variance"]
    entropy_left = wall_integrated_curves(
        case["final_entropy_contour"], wall_x=WALL_WINDOWS["left"]
    )
    entropy_right = wall_integrated_curves(
        case["final_entropy_contour"], wall_x=WALL_WINDOWS["right"]
    )
    variance_left = wall_integrated_curves(
        case["final_charge_variance_contour"], wall_x=WALL_WINDOWS["left"]
    )
    variance_right = wall_integrated_curves(
        case["final_charge_variance_contour"], wall_x=WALL_WINDOWS["right"]
    )
    entropy_leak = entropy_full - entropy_left - entropy_right
    variance_leak = variance_full - variance_left - variance_right

    entropy_closure = np.max(
        np.abs(
            case["final_entropy_contour"].sum(axis=(2, 3)) - entropy_full
        )
    )
    variance_closure = np.max(
        np.abs(
            case["final_charge_variance_contour"].sum(axis=(2, 3))
            - variance_full
        )
    )
    if entropy_closure > 2.0e-8 or variance_closure > 2.0e-8:
        raise FloatingPointError(
            "final contour/scalar closure failed: "
            f"entropy={entropy_closure:.3e}, variance={variance_closure:.3e}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED + 4040)
    indices = rng.integers(0, EXPECTED_SAMPLES, size=(bootstrap_count, EXPECTED_SAMPLES))
    rows: list[dict[str, Any]] = []
    for observable, factor, isolated_factor, full, left, right, leak in (
        ("entropy", 3.0, 6.0, entropy_full, entropy_left, entropy_right, entropy_leak),
        (
            "charge_variance",
            math.pi**2,
            2.0 * math.pi**2,
            variance_full,
            variance_left,
            variance_right,
            variance_leak,
        ),
    ):
        curves = np.stack((full, left, right, leak), axis=1)
        slopes = _trajectory_slopes(ay, curves, ny=40)
        trimmed_slopes = _trajectory_slopes(
            ay, curves, ny=40, trim_endpoints=True
        )
        boot = np.empty((bootstrap_count, 4), dtype=np.float64)
        trimmed_boot = np.empty_like(boot)
        for start in range(0, bootstrap_count, 500):
            stop = min(bootstrap_count, start + 500)
            chosen = indices[start:stop]
            boot[start:stop] = slopes[chosen].mean(axis=1)
            trimmed_boot[start:stop] = trimmed_slopes[chosen].mean(axis=1)
        mean_slopes = slopes.mean(axis=0)
        mean_trimmed_slopes = trimmed_slopes.mean(axis=0)
        scaled = factor * boot
        scaled_trimmed = factor * trimmed_boot
        low, high = percentile_interval(scaled)
        trimmed_low, trimmed_high = percentile_interval(scaled_trimmed)
        left_right = factor * (boot[:, 1] - boot[:, 2])
        lr_low, lr_high = percentile_interval(left_right)
        captured_fraction = (boot[:, 1] + boot[:, 2]) / boot[:, 0]
        captured_low, captured_high = percentile_interval(captured_fraction)
        trimmed_captured_fraction = (
            trimmed_boot[:, 1] + trimmed_boot[:, 2]
        ) / trimmed_boot[:, 0]
        trimmed_captured_low, trimmed_captured_high = percentile_interval(
            trimmed_captured_fraction
        )
        rows.append(
            {
                "observable": observable,
                "full_slope": float(mean_slopes[0]),
                "full_prefactor": float(factor * mean_slopes[0]),
                "full_prefactor_ci_low": float(low[0]),
                "full_prefactor_ci_high": float(high[0]),
                "left_slope": float(mean_slopes[1]),
                "left_share": float(factor * mean_slopes[1]),
                "left_share_ci_low": float(low[1]),
                "left_share_ci_high": float(high[1]),
                "left_isolated_equivalent": float(isolated_factor * mean_slopes[1]),
                "right_slope": float(mean_slopes[2]),
                "right_share": float(factor * mean_slopes[2]),
                "right_share_ci_low": float(low[2]),
                "right_share_ci_high": float(high[2]),
                "right_isolated_equivalent": float(isolated_factor * mean_slopes[2]),
                "left_minus_right_share": float(
                    factor * (mean_slopes[1] - mean_slopes[2])
                ),
                "left_minus_right_ci_low": float(lr_low),
                "left_minus_right_ci_high": float(lr_high),
                "leakage_slope": float(mean_slopes[3]),
                "leakage_prefactor": float(factor * mean_slopes[3]),
                "leakage_prefactor_ci_low": float(low[3]),
                "leakage_prefactor_ci_high": float(high[3]),
                "captured_slope_fraction": float(
                    (mean_slopes[1] + mean_slopes[2]) / mean_slopes[0]
                ),
                "captured_slope_fraction_ci_low": float(captured_low),
                "captured_slope_fraction_ci_high": float(captured_high),
                "contour_closure_max_abs": float(
                    entropy_closure if observable == "entropy" else variance_closure
                ),
                "trimmed_full_prefactor": float(
                    factor * mean_trimmed_slopes[0]
                ),
                "trimmed_full_prefactor_ci_low": float(trimmed_low[0]),
                "trimmed_full_prefactor_ci_high": float(trimmed_high[0]),
                "trimmed_left_share": float(factor * mean_trimmed_slopes[1]),
                "trimmed_left_share_ci_low": float(trimmed_low[1]),
                "trimmed_left_share_ci_high": float(trimmed_high[1]),
                "trimmed_right_share": float(factor * mean_trimmed_slopes[2]),
                "trimmed_right_share_ci_low": float(trimmed_low[2]),
                "trimmed_right_share_ci_high": float(trimmed_high[2]),
                "trimmed_leakage_prefactor": float(
                    factor * mean_trimmed_slopes[3]
                ),
                "trimmed_leakage_prefactor_ci_low": float(trimmed_low[3]),
                "trimmed_leakage_prefactor_ci_high": float(trimmed_high[3]),
                "trimmed_captured_slope_fraction": float(
                    (mean_trimmed_slopes[1] + mean_trimmed_slopes[2])
                    / mean_trimmed_slopes[0]
                ),
                "trimmed_captured_slope_fraction_ci_low": float(
                    trimmed_captured_low
                ),
                "trimmed_captured_slope_fraction_ci_high": float(
                    trimmed_captured_high
                ),
            }
        )
    return rows


def global_charge_analysis(
    cases: Mapping[int, Mapping[str, np.ndarray]]
) -> list[dict[str, Any]]:
    rows = []
    for ny in EXPECTED_NY:
        offsets = np.asarray(cases[ny]["half_filling_offset"], dtype=np.float64)
        for cycle, values in enumerate(offsets.T):
            rows.append(
                {
                    "Ny": ny,
                    "cycle": cycle,
                    "normalized_cycle": cycle / float(ny),
                    "mean_half_filling_offset": float(values.mean()),
                    "sem_half_filling_offset": float(values.std(ddof=1) / math.sqrt(len(values))),
                    "minimum_half_filling_offset": int(values.min()),
                    "maximum_half_filling_offset": int(values.max()),
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    fig.savefig(root / f"{stem}.pdf")
    fig.savefig(root / f"{stem}.png", dpi=300)
    plt.close(fig)


def make_figures(
    cases: Mapping[int, Mapping[str, np.ndarray]],
    scaling_rows: list[Mapping[str, Any]],
    dynamics_rows: list[Mapping[str, Any]],
    wall_rows: list[Mapping[str, Any]],
    analysis_root: Path,
) -> None:
    configure_plotting()
    analysis_root.mkdir(parents=True, exist_ok=True)

    ny = np.asarray([row["Ny"] for row in scaling_rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), constrained_layout=True)
    for axis, key, label in ((axes[0], "c", r"$c$"), (axes[1], "k", r"$k$")):
        mean = np.asarray([row[key] for row in scaling_rows])
        low = np.asarray([row[f"{key}_ci_low"] for row in scaling_rows])
        high = np.asarray([row[f"{key}_ci_high"] for row in scaling_rows])
        axis.errorbar(
            ny,
            mean,
            yerr=(np.maximum(mean - low, 0.0), np.maximum(high - mean, 0.0)),
            color="tab:blue",
            marker="o",
            capsize=2,
        )
        axis.axhline(1.0, color="0.35", ls="--", lw=0.9)
        axis.set(xlabel=r"$N_y$", ylabel=label)
    for label, axis in zip("ab", axes):
        axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    _save_figure(fig, analysis_root, "final_c_k_scaling")

    cycles = np.asarray([row["cycle"] for row in dynamics_rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), constrained_layout=True)
    for key, color in (("c", "tab:blue"), ("k", "tab:orange")):
        mean = np.asarray([row[key] for row in dynamics_rows])
        low = np.asarray([row[f"{key}_ci_low"] for row in dynamics_rows])
        high = np.asarray([row[f"{key}_ci_high"] for row in dynamics_rows])
        axes[0].plot(cycles, mean, color=color, label=fr"${key}$")
        axes[0].fill_between(cycles, low, high, color=color, alpha=0.18, linewidth=0)
    axes[0].axhline(1.0, color="0.35", ls="--", lw=0.9)
    delta = np.asarray([row["c_minus_k"] for row in dynamics_rows])
    delta_low = np.asarray([row["c_minus_k_ci_low"] for row in dynamics_rows])
    delta_high = np.asarray([row["c_minus_k_ci_high"] for row in dynamics_rows])
    axes[1].plot(cycles, delta, color="tab:purple")
    axes[1].fill_between(cycles, delta_low, delta_high, color="tab:purple", alpha=0.18, linewidth=0)
    axes[1].axhline(0.0, color="0.35", ls="--", lw=0.9)
    axes[0].set(xlabel="cycle", ylabel="prefactor")
    axes[0].legend()
    axes[1].set(xlabel="cycle", ylabel=r"$c-k$")
    for label, axis in zip("ab", axes):
        axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    _save_figure(fig, analysis_root, "ny40_c_k_dynamics")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), constrained_layout=True)
    for axis, row in zip(axes, wall_rows):
        values = [row["left_share"], row["right_share"], row["leakage_prefactor"]]
        low = [row["left_share_ci_low"], row["right_share_ci_low"], row["leakage_prefactor_ci_low"]]
        high = [row["left_share_ci_high"], row["right_share_ci_high"], row["leakage_prefactor_ci_high"]]
        axis.bar([0, 1, 2], values, color=["tab:blue", "tab:orange", "0.55"])
        values_array = np.asarray(values)
        axis.errorbar(
            [0, 1, 2],
            values,
            yerr=(
                np.maximum(values_array - np.asarray(low), 0.0),
                np.maximum(np.asarray(high) - values_array, 0.0),
            ),
            fmt="none",
            color="black",
            capsize=2,
        )
        axis.plot([-0.4, 1.4], [0.5, 0.5], color="0.35", ls="--", lw=0.9)
        axis.plot([1.6, 2.4], [0.0, 0.0], color="0.35", ls="--", lw=0.9)
        axis.set_xticks([0, 1, 2], ["left", "right", "outside"])
        axis.set_ylabel("full-system prefactor share")
        axis.set_title(str(row["observable"]).replace("_", " "))
    for label, axis in zip("ab", axes):
        axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    _save_figure(fig, analysis_root, "ny40_wall_decomposition")

    case40 = cases[40]
    entropy_profile = case40["half_strip_entropy_contour"].mean(0).sum(axis=-1).T
    variance_profile = case40["half_strip_charge_variance_contour"].mean(0).sum(axis=-1).T
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), constrained_layout=True)
    for axis, values, title in (
        (axes[0], entropy_profile, "half-strip entropy"),
        (axes[1], variance_profile, "half-strip charge variance"),
    ):
        image = axis.imshow(values, origin="lower", aspect="auto", extent=(0, 80, 0, 19), cmap="viridis")
        axis.set(xlabel="cycle", ylabel=r"$x$", title=title)
        fig.colorbar(image, ax=axis, pad=0.02)
    for label, axis in zip("ab", axes):
        axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    _save_figure(fig, analysis_root, "ny40_half_strip_contour_evolution")

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(EXPECTED_NY)))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), constrained_layout=True)
    for color, size in zip(colors, EXPECTED_NY):
        offsets = cases[size]["half_filling_offset"]
        mean = offsets.mean(0)
        sem = offsets.std(0, ddof=1) / math.sqrt(EXPECTED_SAMPLES)
        x = np.arange(2 * size + 1) / float(size)
        axes[0].plot(x, mean, color=color, label=fr"$N_y={size}$")
        axes[0].fill_between(x, mean - sem, mean + sem, color=color, alpha=0.1, linewidth=0)
        axes[1].plot(size, np.std(offsets[:, -1], ddof=1), "o", color=color)
    axes[0].axhline(0.0, color="0.35", ls="--", lw=0.9)
    axes[0].set(xlabel=r"$t/N_y$", ylabel=r"$\langle Q-N_xN_y\rangle$")
    axes[0].legend(ncol=2, fontsize=6.5)
    axes[1].set(xlabel=r"$N_y$", ylabel=r"final $\operatorname{sd}(Q-N_xN_y)$")
    for label, axis in zip("ab", axes):
        axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    _save_figure(fig, analysis_root, "global_charge_wandering")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--bootstrap-count", type=int, default=DEFAULT_BOOTSTRAP_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_count <= 0:
        raise ValueError("bootstrap-count must be positive")
    output_root = args.output_root.resolve()
    analysis_root = (
        args.analysis_root.resolve()
        if args.analysis_root is not None
        else output_root / "analysis_outputs"
    )
    paths = discover_verified_results(output_root)
    if len(paths) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"analysis requires {EXPECTED_SHARDS} verified five-trajectory shards; "
            f"found {len(paths)}"
        )
    cases = load_case_data(paths)
    validate_complete_cases(cases)
    scaling = final_scaling_analysis(cases, bootstrap_count=args.bootstrap_count)
    dynamics = ny40_dynamics_analysis(cases[40], bootstrap_count=args.bootstrap_count)
    wall = wall_decomposition_analysis(cases[40], bootstrap_count=args.bootstrap_count)
    charge = global_charge_analysis(cases)
    write_csv(analysis_root / "final_c_k_scaling.csv", scaling)
    write_csv(analysis_root / "ny40_c_k_dynamics.csv", dynamics)
    write_csv(analysis_root / "ny40_wall_decomposition.csv", wall)
    write_csv(analysis_root / "global_charge_summary.csv", charge)
    make_figures(cases, scaling, dynamics, wall, analysis_root)
    summary = {
        "schema": "hard_wall_entropy_charge_analysis_v1",
        "observer_schema": OBSERVER_SCHEMA,
        "independent_sampling_unit": "trajectory_after_periodic_y0_average",
        "Nx": 20,
        "Ny": list(EXPECTED_NY),
        "samples_per_size": EXPECTED_SAMPLES,
        "fit_window": "Ay=8..Ny//2",
        "sensitivity_fit": "primary fit with its lower and upper endpoints excluded",
        "bootstrap_replicates": int(args.bootstrap_count),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_interval": "95% percentile whole-trajectory bootstrap",
        "wall_windows": {key: list(value) for key, value in WALL_WINDOWS.items()},
        "verified_result_count": len(paths),
        "outputs": [
            "final_c_k_scaling.csv",
            "ny40_c_k_dynamics.csv",
            "ny40_wall_decomposition.csv",
            "global_charge_summary.csv",
            "final_c_k_scaling.pdf",
            "ny40_c_k_dynamics.pdf",
            "ny40_wall_decomposition.pdf",
            "ny40_half_strip_contour_evolution.pdf",
            "global_charge_wandering.pdf",
        ],
    }
    write_json(analysis_root / "analysis_summary.json", summary)
    print(
        f"[analysis complete] {len(paths)} verified shards, "
        f"{sum(len(case['sample_ids']) for case in cases.values())} trajectories; "
        f"outputs={analysis_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
