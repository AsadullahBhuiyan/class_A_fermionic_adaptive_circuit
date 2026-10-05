#!/usr/bin/env python3
"""Run the aggressively batched domain-wall mutual-information sweep.

The v2 runner writes fixed 50/25-trajectory macro results while importing any
verified five-trajectory v1 results as immutable scientific inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = BUNDLE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from mutual_information_observer import (  # noqa: E402
    OBSERVER_SCHEMA,
    FixedWidthMutualInformationObserver,
)


BUNDLE = "02_domain_wall_bipartite_mutual_information"
RESULT_SCHEMA = "domain_wall_bipartite_mutual_information_task_v2"
COMPLETION_SCHEMA = "domain_wall_bipartite_mutual_information_completion_v2"
AGGREGATE_SCHEMA = "domain_wall_bipartite_mutual_information_aggregate_v2"
AGGREGATE_COMPLETION_SCHEMA = (
    "domain_wall_bipartite_mutual_information_aggregate_completion_v2"
)
LEGACY_RESULT_SCHEMA = "domain_wall_bipartite_mutual_information_task_v1"
LEGACY_COMPLETION_SCHEMA = "domain_wall_bipartite_mutual_information_completion_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
EXPECTED_REVISION = (
    "domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v2_batched_50-25-25"
)
LEGACY_REVISION = "domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v1"
LEGACY_CONFIG_SHA256 = (
    "60943f5f92165a2dde5c6c3dfb17428e0f5ba599c834e874be2426ef510ea485"
)
EXPECTED_ROOT_SEED = 2026090301
EXPECTED_NX = 20
EXPECTED_NY_VALUES = (20, 24, 28)
EXPECTED_WIDTHS = {20: 5, 24: 6, 28: 7}
EXPECTED_ALPHA_VALUES = (
    3.0,
    2.75,
    2.5,
    2.3,
    2.2,
    2.15,
    2.1,
    2.075,
    2.05,
    2.025,
    2.0,
    1.975,
    1.95,
    1.925,
    1.9,
    1.85,
    1.8,
    1.7,
    1.5,
    1.25,
    1.0,
)
EXPECTED_WALLS = ("hard", "soft")
EXPECTED_NSHELL = 1
EXPECTED_ALPHA_2 = 30.0
EXPECTED_SAMPLES = 100
EXPECTED_BATCH_SIZES = {20: 50, 24: 25, 28: 25}
LEGACY_SHARD_SIZE = 5
EXPECTED_CASES = 126
EXPECTED_TASKS = 420
LEGACY_TASKS = 2520
EXPECTED_TRAJECTORIES = 12600
SOURCE_FILES = (
    "run_campaign.py",
    "mutual_information_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)
TRAJECTORY_KEYS = (
    "mutual_information_y0avg",
    "entropy_a_y0avg",
    "entropy_b_y0avg",
    "entropy_union_y0avg",
)


@dataclass(frozen=True)
class Task:
    ny: int
    width: int
    alpha_1: float
    wall: str
    batch_index: int
    sample_start: int
    sample_stop: int
    task_id: str
    seed: int

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def global_sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))

    @property
    def cycles(self) -> int:
        return 2 * self.ny

    @property
    def dw_truncation(self) -> bool:
        return self.wall == "hard"


@dataclass(frozen=True)
class LegacyRecord:
    task: Task
    result_path: Path
    completion_path: Path
    completion: dict[str, Any]


def alpha_tag(alpha: float) -> str:
    text = f"{float(alpha):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(bundle_root: Path = BUNDLE_ROOT) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = bundle_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing required source file: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "sampling_revision",
        "root_seed",
        "Nx",
        "Ny_values",
        "width_rule",
        "alpha_1_values",
        "alpha_2",
        "wall_constructions",
        "nshell",
        "samples_per_case",
        "batch_size_by_Ny",
        "cycles_rule",
        "device",
        "dtype",
        "protocol",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing fields: {missing}")
    normalized = json.loads(json.dumps(config))
    if normalized["sampling_revision"] != EXPECTED_REVISION:
        raise ValueError(f"sampling_revision must be {EXPECTED_REVISION!r}")
    if int(normalized["root_seed"]) != EXPECTED_ROOT_SEED:
        raise ValueError(f"root_seed must be {EXPECTED_ROOT_SEED}")
    if int(normalized["Nx"]) != EXPECTED_NX:
        raise ValueError(f"Nx must be {EXPECTED_NX}")
    if tuple(int(value) for value in normalized["Ny_values"]) != EXPECTED_NY_VALUES:
        raise ValueError(f"Ny_values must be {list(EXPECTED_NY_VALUES)}")
    if normalized["width_rule"] != "Ny//4":
        raise ValueError("width_rule must be 'Ny//4'")
    alpha_values = tuple(float(value) for value in normalized["alpha_1_values"])
    if alpha_values != EXPECTED_ALPHA_VALUES:
        raise ValueError(f"alpha_1_values must be {list(EXPECTED_ALPHA_VALUES)}")
    if float(normalized["alpha_2"]) != EXPECTED_ALPHA_2:
        raise ValueError(f"alpha_2 must be {EXPECTED_ALPHA_2}")
    if (
        tuple(str(value) for value in normalized["wall_constructions"])
        != EXPECTED_WALLS
    ):
        raise ValueError(f"wall_constructions must be {list(EXPECTED_WALLS)}")
    if int(normalized["nshell"]) != EXPECTED_NSHELL:
        raise ValueError(f"nshell must be {EXPECTED_NSHELL}")
    if int(normalized["samples_per_case"]) != EXPECTED_SAMPLES:
        raise ValueError(f"samples_per_case must be {EXPECTED_SAMPLES}")
    batch_sizes = {
        int(ny): int(value) for ny, value in normalized["batch_size_by_Ny"].items()
    }
    if batch_sizes != EXPECTED_BATCH_SIZES:
        raise ValueError(f"batch_size_by_Ny must be {EXPECTED_BATCH_SIZES}")
    if normalized["cycles_rule"] != "2*Ny":
        raise ValueError("cycles_rule must be '2*Ny'")
    if normalized["device"] != "cuda:0" or normalized["dtype"] != "complex128":
        raise ValueError("production requires device='cuda:0' and dtype='complex128'")
    expected_protocol = {
        "DW": True,
        "filling_frac": 0.5,
        "trial_orbitals": "X",
        "init_mode": "default",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "meas_slab_only": True,
        "n_a": 0.5,
        "triv_region_local_mode": False,
    }
    if normalized["protocol"] != expected_protocol:
        raise ValueError(
            "protocol must remain the locked pure-state, perfect-correction, "
            "raster-y domain-wall contract"
        )
    for ny in EXPECTED_NY_VALUES:
        if ny % 4 or ny // 4 != EXPECTED_WIDTHS[ny]:
            raise RuntimeError(f"invalid locked quarter-width geometry for Ny={ny}")
    return normalized


def config_hash(config: dict[str, Any]) -> str:
    return _sha256_bytes(_json_bytes(validate_config(config)))


def task_seed(
    root_seed: int,
    *,
    ny: int,
    alpha_1: float,
    wall: str,
    batch_index: int,
    sample_start: int,
    sample_stop: int,
) -> int:
    label = (
        f"{EXPECTED_REVISION}|{int(root_seed)}|Nx={EXPECTED_NX}|Ny={int(ny)}|"
        f"alpha1={float(alpha_1):.12g}|wall={wall}|batch={int(batch_index)}|"
        f"samples={int(sample_start)}:{int(sample_stop)}"
    )
    return int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    ) & ((1 << 63) - 1)


def legacy_task_seed(
    root_seed: int,
    *,
    ny: int,
    alpha_1: float,
    wall: str,
    batch_index: int,
    sample_start: int,
    sample_stop: int,
) -> int:
    """Reproduce the exact seed identity used by the five-sample v1 runner."""

    label = (
        f"{int(root_seed)}|Nx={EXPECTED_NX}|Ny={int(ny)}|"
        f"alpha1={float(alpha_1):.12g}|wall={wall}|batch={int(batch_index)}|"
        f"samples={int(sample_start)}:{int(sample_stop)}"
    )
    return int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    ) & ((1 << 63) - 1)


def expand_tasks(config: dict[str, Any]) -> list[Task]:
    config = validate_config(config)
    tasks: list[Task] = []
    for ny in config["Ny_values"]:
        ny = int(ny)
        batch_size = EXPECTED_BATCH_SIZES[ny]
        for wall in config["wall_constructions"]:
            wall = str(wall)
            for alpha_1 in config["alpha_1_values"]:
                alpha_1 = float(alpha_1)
                for batch_index, sample_start in enumerate(
                    range(0, EXPECTED_SAMPLES, batch_size)
                ):
                    sample_stop = min(sample_start + batch_size, EXPECTED_SAMPLES)
                    task_id = (
                        f"Ny{ny:02d}_wall-{wall}_alpha1-{alpha_tag(alpha_1)}_"
                        f"macro-{batch_index:03d}_"
                        f"samples-{sample_start:03d}-{sample_stop - 1:03d}"
                    )
                    tasks.append(
                        Task(
                            ny=ny,
                            width=ny // 4,
                            alpha_1=alpha_1,
                            wall=wall,
                            batch_index=batch_index,
                            sample_start=sample_start,
                            sample_stop=sample_stop,
                            task_id=task_id,
                            seed=task_seed(
                                int(config["root_seed"]),
                                ny=ny,
                                alpha_1=alpha_1,
                                wall=wall,
                                batch_index=batch_index,
                                sample_start=sample_start,
                                sample_stop=sample_stop,
                            ),
                        )
                    )
    if (
        len(tasks) != EXPECTED_TASKS
        or len({task.task_id for task in tasks}) != EXPECTED_TASKS
    ):
        raise RuntimeError(f"task expansion must contain {EXPECTED_TASKS} unique tasks")
    if len({task.seed for task in tasks}) != EXPECTED_TASKS:
        raise RuntimeError("independent task seeds must be globally unique")
    if sum(task.sample_count for task in tasks) != EXPECTED_TRAJECTORIES:
        raise RuntimeError(
            f"task expansion must cover {EXPECTED_TRAJECTORIES} trajectories"
        )
    return tasks


def legacy_tasks_for_macro(task: Task) -> tuple[Task, ...]:
    """Return the immutable v1 five-sample tasks contained in one v2 macro."""

    legacy: list[Task] = []
    for sample_start in range(task.sample_start, task.sample_stop, LEGACY_SHARD_SIZE):
        sample_stop = min(sample_start + LEGACY_SHARD_SIZE, task.sample_stop)
        if sample_stop - sample_start != LEGACY_SHARD_SIZE:
            raise RuntimeError("v2 macro ranges must align to five-sample v1 shards")
        batch_index = sample_start // LEGACY_SHARD_SIZE
        task_id = (
            f"Ny{task.ny:02d}_wall-{task.wall}_alpha1-{alpha_tag(task.alpha_1)}_"
            f"batch-{batch_index:03d}_"
            f"samples-{sample_start:03d}-{sample_stop - 1:03d}"
        )
        legacy.append(
            Task(
                ny=task.ny,
                width=task.width,
                alpha_1=task.alpha_1,
                wall=task.wall,
                batch_index=batch_index,
                sample_start=sample_start,
                sample_stop=sample_stop,
                task_id=task_id,
                seed=legacy_task_seed(
                    EXPECTED_ROOT_SEED,
                    ny=task.ny,
                    alpha_1=task.alpha_1,
                    wall=task.wall,
                    batch_index=batch_index,
                    sample_start=sample_start,
                    sample_stop=sample_stop,
                ),
            )
        )
    return tuple(legacy)


def task_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = (
        output_root
        / "results"
        / f"Ny{task.ny:02d}"
        / f"wall-{task.wall}"
        / f"alpha1-{alpha_tag(task.alpha_1)}"
    )
    stem = (
        f"macro_{task.batch_index:03d}_"
        f"samples_{task.sample_start:03d}-{task.sample_stop - 1:03d}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def legacy_task_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = (
        output_root
        / "results"
        / f"Ny{task.ny:02d}"
        / f"wall-{task.wall}"
        / f"alpha1-{alpha_tag(task.alpha_1)}"
    )
    stem = (
        f"batch_{task.batch_index:03d}_"
        f"samples_{task.sample_start:03d}-{task.sample_stop - 1:03d}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def _expected_completion_identity(
    *, task: Task, config_sha256: str, hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "task_id": task.task_id,
        "Nx": EXPECTED_NX,
        "Ny": task.ny,
        "width": task.width,
        "alpha_1": task.alpha_1,
        "alpha_2": EXPECTED_ALPHA_2,
        "wall": task.wall,
        "dw_truncation": task.dw_truncation,
        "nshell": EXPECTED_NSHELL,
        "cycles": task.cycles,
        "batch_index": task.batch_index,
        "sample_start": task.sample_start,
        "sample_stop": task.sample_stop,
        "sample_count": task.sample_count,
        "global_sample_indices": list(task.global_sample_indices),
        "seed": task.seed,
        "config_sha256": config_sha256,
        "source_hashes": hashes,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "observer_schema": OBSERVER_SCHEMA,
    }


def verified_complete(
    *,
    output_root: Path,
    task: Task,
    config_sha256: str,
    hashes: dict[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = task_paths(output_root, task)
    result_exists = result_path.is_file()
    completion_exists = completion_path.is_file()
    if not result_exists and not completion_exists:
        return False, "missing result/completion pair"
    if not result_exists or not completion_exists:
        return False, "incomplete result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion JSON: {exc}"
    expected = _expected_completion_identity(
        task=task, config_sha256=config_sha256, hashes=hashes
    )
    for key, value in expected.items():
        if completion.get(key) != value:
            return False, f"completion identity mismatch: {key}"
    if completion.get("result_filename") != result_path.name:
        return False, "completion result filename mismatch"
    try:
        actual_bytes = result_path.stat().st_size
        actual_sha256 = sha256_file(result_path)
    except OSError as exc:
        return False, f"result readback failed: {exc}"
    if int(completion.get("result_bytes", -1)) != actual_bytes:
        return False, "result byte count mismatch"
    if completion.get("result_sha256") != actual_sha256:
        return False, "result checksum mismatch"
    valid, reason = validate_v2_result_payload(result_path, task, completion)
    if not valid:
        return False, reason
    return True, "verified"


def _validate_observable_payload(
    payload: dict[str, np.ndarray], *, sample_count: int, label: str
) -> str | None:
    arrays: dict[str, np.ndarray] = {}
    for key in TRAJECTORY_KEYS:
        try:
            values = np.asarray(payload[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            return f"{label} has unreadable {key}: {exc}"
        if values.shape != (sample_count,) or not np.isfinite(values).all():
            return f"{label} has invalid {key}"
        arrays[key] = values
    for key in ("entropy_a_y0avg", "entropy_b_y0avg", "entropy_union_y0avg"):
        if float(arrays[key].min()) < -1.0e-10:
            return f"{label} has negative {key}"

    residual = arrays["mutual_information_y0avg"] - (
        arrays["entropy_a_y0avg"]
        + arrays["entropy_b_y0avg"]
        - arrays["entropy_union_y0avg"]
    )
    residual_max = float(np.max(np.abs(residual)))
    if residual_max > 1.0e-8:
        return f"{label} entropy identity residual exceeds tolerance"
    scalar_keys = (
        "full_covariance_max_hermiticity_error",
        "restricted_max_hermiticity_error",
        "restricted_occupation_eigenvalue_min",
        "restricted_occupation_eigenvalue_max",
        "mutual_information_min",
        "entropy_identity_max_abs_residual",
    )
    try:
        scalars = {key: float(np.asarray(payload[key]).item()) for key in scalar_keys}
        eigensolves = int(np.asarray(payload["restricted_eigensolve_count"]).item())
        negative_count = int(np.asarray(payload["materially_negative_mi_count"]).item())
    except (KeyError, TypeError, ValueError) as exc:
        return f"{label} has unreadable diagnostics: {exc}"
    if not np.isfinite(tuple(scalars.values())).all():
        return f"{label} has nonfinite diagnostics"
    if scalars["full_covariance_max_hermiticity_error"] > 1.0e-9:
        return f"{label} full covariance Hermiticity diagnostic exceeds tolerance"
    if scalars["restricted_max_hermiticity_error"] > 1.0e-9:
        return f"{label} restricted Hermiticity diagnostic exceeds tolerance"
    if scalars["restricted_occupation_eigenvalue_min"] < -1.0e-8:
        return f"{label} occupation spectrum falls below tolerance"
    if scalars["restricted_occupation_eigenvalue_max"] > 1.0 + 1.0e-8:
        return f"{label} occupation spectrum exceeds tolerance"
    if eigensolves <= 0:
        return f"{label} eigensolve count is invalid"
    expected_negative = int(
        np.count_nonzero(arrays["mutual_information_y0avg"] < -1.0e-8)
    )
    if negative_count != expected_negative:
        return f"{label} negative-MI diagnostic mismatch"
    if not np.isclose(
        scalars["mutual_information_min"],
        float(arrays["mutual_information_y0avg"].min()),
        rtol=0.0,
        atol=1.0e-12,
    ):
        return f"{label} minimum-MI diagnostic mismatch"
    if not np.isclose(
        scalars["entropy_identity_max_abs_residual"],
        residual_max,
        rtol=0.0,
        atol=1.0e-12,
    ):
        return f"{label} entropy-residual diagnostic mismatch"
    return None


def validate_v2_result_payload(
    result_path: Path, task: Task, completion: dict[str, Any]
) -> tuple[bool, str]:
    try:
        with np.load(result_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
            scalar_expectations = {
                "schema": RESULT_SCHEMA,
                "observer_schema": OBSERVER_SCHEMA,
                "bundle": BUNDLE,
                "sampling_revision": EXPECTED_REVISION,
                "canonical_entry_point": CANONICAL_ENTRY_POINT,
                "task_id": task.task_id,
                "wall": task.wall,
            }
            for key, expected in scalar_expectations.items():
                if key not in payload or str(payload[key].item()) != expected:
                    return False, f"result identity mismatch: {key}"
            integer_expectations = {
                "Nx": EXPECTED_NX,
                "Ny": task.ny,
                "width": task.width,
                "nshell": EXPECTED_NSHELL,
                "batch_index": task.batch_index,
                "sample_start": task.sample_start,
                "sample_stop": task.sample_stop,
                "endpoint_cycle": task.cycles,
                "macro_seed": task.seed,
            }
            for key, expected in integer_expectations.items():
                if key not in payload or int(payload[key].item()) != expected:
                    return False, f"result identity mismatch: {key}"
            float_expectations = {
                "alpha_1": task.alpha_1,
                "alpha_2": EXPECTED_ALPHA_2,
            }
            for key, expected in float_expectations.items():
                if key not in payload or float(payload[key].item()) != expected:
                    return False, f"result identity mismatch: {key}"
            boolean_expectations = {
                "dw_truncation": task.dw_truncation,
                "meas_slab_only_requested": True,
                "meas_slab_only_effective": task.dw_truncation,
            }
            for key, expected in boolean_expectations.items():
                if key not in payload or bool(payload[key].item()) != expected:
                    return False, f"result identity mismatch: {key}"
            expected_indices = np.asarray(task.global_sample_indices, dtype=np.int64)
            if not np.array_equal(
                payload.get("global_sample_indices"), expected_indices
            ):
                return False, "result global sample indices mismatch"
            observable_reason = _validate_observable_payload(
                payload, sample_count=task.sample_count, label="result"
            )
            if observable_reason is not None:
                return False, observable_reason
            origins = np.asarray(payload.get("trajectory_origin"))
            seeds = np.asarray(payload.get("trajectory_rng_group_seed"), dtype=np.int64)
            if origins.shape != (task.sample_count,) or seeds.shape != (
                task.sample_count,
            ):
                return False, "result trajectory provenance shape mismatch"
            legacy_indices = tuple(
                int(value) for value in completion.get("legacy_sample_indices", ())
            )
            computed_indices = tuple(
                int(value) for value in completion.get("computed_sample_indices", ())
            )
            if len(set(legacy_indices)) != len(legacy_indices) or len(
                set(computed_indices)
            ) != len(computed_indices):
                return False, "completion contains duplicate sample indices"
            if legacy_indices != tuple(sorted(legacy_indices)):
                return False, "completion legacy sample indices are not ordered"
            if computed_indices != tuple(sorted(computed_indices)):
                return False, "completion computed sample indices are not ordered"
            if set(legacy_indices) & set(computed_indices):
                return False, "completion legacy/computed coverage overlaps"
            if sorted((*legacy_indices, *computed_indices)) != list(
                task.global_sample_indices
            ):
                return False, "completion sample coverage mismatch"
            if not np.array_equal(
                payload.get("legacy_sample_indices"),
                np.asarray(legacy_indices, dtype=np.int64),
            ):
                return False, "result legacy sample indices mismatch"
            if not np.array_equal(
                payload.get("computed_sample_indices"),
                np.asarray(computed_indices, dtype=np.int64),
            ):
                return False, "result computed sample indices mismatch"
            expected_computed_seed = (
                computed_seed(task, computed_indices) if computed_indices else None
            )
            if completion.get("computed_rng_seed") != expected_computed_seed:
                return False, "completion computed seed mismatch"
            payload_seed = int(payload.get("computed_rng_seed").item())
            if payload_seed != (
                -1 if expected_computed_seed is None else expected_computed_seed
            ):
                return False, "result computed seed mismatch"
            legacy_index_set = set(legacy_indices)
            expected_legacy_tasks: list[Task] = []
            for legacy_task in legacy_tasks_for_macro(task):
                covered = [
                    index in legacy_index_set
                    for index in legacy_task.global_sample_indices
                ]
                if any(covered) and not all(covered):
                    return False, "result contains a partial legacy shard"
                if all(covered):
                    expected_legacy_tasks.append(legacy_task)
                    for index in legacy_task.global_sample_indices:
                        position = index - task.sample_start
                        if (
                            origins[position] != "legacy_v1"
                            or seeds[position] != legacy_task.seed
                        ):
                            return False, "result legacy provenance mismatch"
            for index in computed_indices:
                position = index - task.sample_start
                if (
                    origins[position] != "computed_v2"
                    or seeds[position] != expected_computed_seed
                ):
                    return False, "result computed provenance mismatch"
            legacy_records = completion.get("legacy_inputs")
            if not isinstance(legacy_records, list) or len(legacy_records) != len(
                expected_legacy_tasks
            ):
                return False, "completion legacy input provenance mismatch"
            for record, legacy_task in zip(legacy_records, expected_legacy_tasks):
                if not isinstance(record, dict):
                    return False, "completion legacy input record is invalid"
                if record.get("task_id") != legacy_task.task_id:
                    return False, "completion legacy task identity mismatch"
                if record.get("global_sample_indices") != list(
                    legacy_task.global_sample_indices
                ):
                    return False, "completion legacy task coverage mismatch"
                if int(record.get("result_bytes", -1)) <= 0:
                    return False, "completion legacy result byte count is invalid"
                digest = record.get("result_sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    return False, "completion legacy result checksum is invalid"
                if not _valid_recorded_source_hashes(record.get("source_hashes")):
                    return False, "completion legacy source hashes are invalid"
            expected_legacy_ids = np.asarray(
                [legacy_task.task_id for legacy_task in expected_legacy_tasks]
            )
            if not np.array_equal(payload.get("legacy_task_ids"), expected_legacy_ids):
                return False, "result legacy task IDs mismatch"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return False, f"unreadable result NPZ: {exc}"
    return True, "verified"


def _legacy_completion_identity(task: Task) -> dict[str, Any]:
    return {
        "schema": LEGACY_COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": LEGACY_REVISION,
        "task_id": task.task_id,
        "Nx": EXPECTED_NX,
        "Ny": task.ny,
        "width": task.width,
        "alpha_1": task.alpha_1,
        "alpha_2": EXPECTED_ALPHA_2,
        "wall": task.wall,
        "dw_truncation": task.dw_truncation,
        "nshell": EXPECTED_NSHELL,
        "cycles": task.cycles,
        "batch_index": task.batch_index,
        "sample_start": task.sample_start,
        "sample_stop": task.sample_stop,
        "sample_count": task.sample_count,
        "global_sample_indices": list(task.global_sample_indices),
        "seed": task.seed,
        "config_sha256": LEGACY_CONFIG_SHA256,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "observer_schema": OBSERVER_SCHEMA,
    }


def _valid_recorded_source_hashes(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != set(SOURCE_FILES):
        return False
    hexadecimal = set("0123456789abcdef")
    return all(
        isinstance(digest, str)
        and len(digest) == 64
        and set(digest.lower()) <= hexadecimal
        for digest in value.values()
    )


def verified_legacy_record(
    *, output_root: Path | None, task: Task
) -> tuple[LegacyRecord | None, str]:
    """Verify a v1 pair without comparing historical hashes to v2 sources."""

    if output_root is None:
        return None, "legacy output root disabled"
    result_path, completion_path = legacy_task_paths(Path(output_root), task)
    result_exists, completion_exists = result_path.is_file(), completion_path.is_file()
    if not result_exists and not completion_exists:
        return None, "missing legacy result/completion pair"
    if not result_exists or not completion_exists:
        return None, "incomplete legacy result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable legacy completion JSON: {exc}"
    for key, expected in _legacy_completion_identity(task).items():
        if completion.get(key) != expected:
            return None, f"legacy completion identity mismatch: {key}"
    if not _valid_recorded_source_hashes(completion.get("source_hashes")):
        return None, "legacy completion has invalid recorded source hashes"
    if completion.get("result_filename") != result_path.name:
        return None, "legacy completion result filename mismatch"
    try:
        actual_bytes = result_path.stat().st_size
        actual_sha256 = sha256_file(result_path)
    except OSError as exc:
        return None, f"legacy result readback failed: {exc}"
    if int(completion.get("result_bytes", -1)) != actual_bytes:
        return None, "legacy result byte count mismatch"
    if completion.get("result_sha256") != actual_sha256:
        return None, "legacy result checksum mismatch"
    return (
        LegacyRecord(
            task=task,
            result_path=result_path,
            completion_path=completion_path,
            completion=completion,
        ),
        "verified",
    )


def load_legacy_payload(record: LegacyRecord) -> dict[str, np.ndarray]:
    try:
        with np.load(record.result_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"unreadable legacy NPZ: {exc}") from exc
    task = record.task
    scalar_expectations = {
        "schema": LEGACY_RESULT_SCHEMA,
        "observer_schema": OBSERVER_SCHEMA,
        "bundle": BUNDLE,
        "sampling_revision": LEGACY_REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "task_id": task.task_id,
        "wall": task.wall,
    }
    for key, expected in scalar_expectations.items():
        if key not in payload or str(np.asarray(payload[key]).item()) != expected:
            raise ValueError(f"legacy NPZ identity mismatch: {key}")
    integer_expectations = {
        "Nx": EXPECTED_NX,
        "Ny": task.ny,
        "width": task.width,
        "nshell": EXPECTED_NSHELL,
        "batch_index": task.batch_index,
        "sample_start": task.sample_start,
        "sample_stop": task.sample_stop,
        "endpoint_cycle": task.cycles,
        "batch_seed": task.seed,
    }
    for key, expected in integer_expectations.items():
        if key not in payload or int(np.asarray(payload[key]).item()) != expected:
            raise ValueError(f"legacy NPZ identity mismatch: {key}")
    float_expectations = {
        "alpha_1": task.alpha_1,
        "alpha_2": EXPECTED_ALPHA_2,
    }
    for key, expected in float_expectations.items():
        if key not in payload or float(np.asarray(payload[key]).item()) != expected:
            raise ValueError(f"legacy NPZ identity mismatch: {key}")
    if (
        "dw_truncation" not in payload
        or bool(np.asarray(payload["dw_truncation"]).item()) != task.dw_truncation
    ):
        raise ValueError("legacy NPZ identity mismatch: dw_truncation")
    if not np.array_equal(
        np.asarray(payload.get("global_sample_indices"), dtype=np.int64),
        np.asarray(task.global_sample_indices, dtype=np.int64),
    ):
        raise ValueError("legacy NPZ sample indices mismatch")
    observable_reason = _validate_observable_payload(
        payload, sample_count=task.sample_count, label="legacy NPZ"
    )
    if observable_reason is not None:
        raise ValueError(observable_reason)
    return payload


def validate_a100() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an A100 GPU runtime")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_bytes = int(properties.total_memory)
    name = str(properties.name)
    if "A100" not in name.upper():
        raise RuntimeError(f"production requires an NVIDIA A100, found {name!r}")
    if total_bytes < 38 * 1024**3:
        raise RuntimeError(
            "production requires 40-GB-class GPU memory, found "
            f"{total_bytes / 1024**3:.2f} GiB"
        )
    probe = torch.zeros(1, dtype=torch.complex128, device=device)
    if probe.dtype != torch.complex128:
        raise RuntimeError("complex128 CUDA allocation failed")
    del probe
    return {"name": name, "total_bytes": total_bytes, "device": str(device)}


def _check_space(path: Path, *, required_bytes: int, label: str) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path).free)
    if free < int(required_bytes):
        raise RuntimeError(
            f"insufficient {label} space: free={free / 1024**3:.2f} GiB, "
            f"required={required_bytes / 1024**3:.2f} GiB"
        )
    return free


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    """Copy, DriveFS-readback verify, and atomically publish one stable file."""

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    expected_bytes = local_path.stat().st_size
    expected_sha256 = sha256_file(local_path)
    try:
        shutil.copyfile(local_path, temporary)
        if temporary.stat().st_size != expected_bytes:
            raise OSError(f"Drive temporary byte-count mismatch for {temporary}")
        if sha256_file(temporary) != expected_sha256:
            raise OSError(f"Drive temporary checksum mismatch for {temporary}")
        os.replace(temporary, final_path)
        if final_path.stat().st_size != expected_bytes:
            raise OSError(f"Drive final byte-count mismatch for {final_path}")
        if sha256_file(final_path) != expected_sha256:
            raise OSError(f"Drive final checksum mismatch for {final_path}")
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "filename": final_path.name,
        "bytes": expected_bytes,
        "sha256": expected_sha256,
    }


def build_model(config: dict[str, Any], task: Task) -> Any:
    protocol = config["protocol"]
    model = classA_U1FGTN_gpu(
        Nx=EXPECTED_NX,
        Ny=task.ny,
        DW=True,
        nshell=EXPECTED_NSHELL,
        filling_frac=protocol["filling_frac"],
        alpha_1=task.alpha_1,
        alpha_2=EXPECTED_ALPHA_2,
        trial_orbitals=protocol["trial_orbitals"],
        dw_truncation=task.dw_truncation,
        triv_region_local_mode=protocol["triv_region_local_mode"],
        device=config["device"],
        dtype=config["dtype"],
        backend="local",
    )
    if not model.DW or model.nshell != EXPECTED_NSHELL:
        raise RuntimeError("constructed model violates the DW/nshell contract")
    if bool(model.dw_truncation) != task.dw_truncation:
        raise RuntimeError("constructed model has the wrong wall construction")
    if model.dtype != torch.complex128:
        raise RuntimeError("constructed model is not complex128")
    return model


def computed_seed(task: Task, sample_indices: tuple[int, ...]) -> int:
    label = f"{task.seed}|computed=" + ",".join(str(value) for value in sample_indices)
    return int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    ) & ((1 << 63) - 1)


def run_missing_samples(
    *,
    model: Any,
    config: dict[str, Any],
    task: Task,
    sample_indices: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], float, int]:
    if not sample_indices:
        raise ValueError("run_missing_samples requires at least one sample")
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("computed sample indices must be unique")
    seed = computed_seed(task, sample_indices)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sample_count = len(sample_indices)
    translation_bar = tqdm(
        total=task.ny // 2,
        desc=(
            f"MI Ny={task.ny} {task.wall} alpha={task.alpha_1:g} "
            f"macro={task.batch_index:03d} computed={sample_count} "
            f"reused={task.sample_count - sample_count}"
        ),
        unit="translation",
        dynamic_ncols=True,
        leave=False,
    )
    observer = FixedWidthMutualInformationObserver(
        nx=EXPECTED_NX,
        ny=task.ny,
        width=task.width,
        samples=sample_count,
        expected_cycle=task.cycles,
        progress_bar=translation_bar,
    )
    started = time.monotonic()
    try:
        result = model.run_markov_circuit(
            G_history=False,
            progress=True,
            cycles=task.cycles,
            postselect=config["protocol"]["postselect"],
            postselect_probability=config["protocol"]["postselect_probability"],
            perfect_correction=config["protocol"]["perfect_correction"],
            samples=sample_count,
            init_mode=config["protocol"]["init_mode"],
            save=False,
            n_a=config["protocol"]["n_a"],
            sequence=config["protocol"]["sequence"],
            meas_slab_only=config["protocol"]["meas_slab_only"],
            batch_size=sample_count,
            return_data=False,
            state_representation="auto",
            cycle_observer=observer,
            cycle_observer_cycles=[task.cycles],
            track_choi=False,
            return_native_state=False,
        )
        if torch.device(model.device).type == "cuda":
            torch.cuda.synchronize(model.device)
    finally:
        translation_bar.close()
    elapsed = time.monotonic() - started
    if int(result.get("samples", -1)) != sample_count:
        raise RuntimeError("canonical engine returned the wrong sample count")
    if bool(result.get("choi_tracked", False)):
        raise RuntimeError("canonical engine unexpectedly enabled Choi tracking")
    return observer.payload(), elapsed, seed


def merge_task_payload(
    *,
    task: Task,
    legacy_inputs: list[tuple[LegacyRecord, dict[str, np.ndarray]]],
    computed_indices: tuple[int, ...],
    computed_payload: dict[str, np.ndarray] | None,
    computed_elapsed_seconds: float,
    computed_rng_seed: int | None,
) -> dict[str, np.ndarray]:
    legacy_inputs = sorted(legacy_inputs, key=lambda item: item[0].task.sample_start)
    computed_indices = tuple(int(index) for index in computed_indices)
    if computed_indices != tuple(sorted(computed_indices)):
        raise ValueError("computed sample indices must be in increasing order")
    arrays = {
        key: np.full(task.sample_count, np.nan, dtype=np.float64)
        for key in TRAJECTORY_KEYS
    }
    fill_count = np.zeros(task.sample_count, dtype=np.uint8)
    trajectory_origin = np.full(task.sample_count, "", dtype="U11")
    trajectory_rng_group_seed = np.full(task.sample_count, -1, dtype=np.int64)
    diagnostic_payloads: list[dict[str, np.ndarray]] = []

    for record, payload in legacy_inputs:
        indices = np.asarray(record.task.global_sample_indices, dtype=np.int64)
        positions = indices - task.sample_start
        if np.any(positions < 0) or np.any(positions >= task.sample_count):
            raise ValueError("legacy input lies outside its v2 macro range")
        for key in TRAJECTORY_KEYS:
            arrays[key][positions] = np.asarray(payload[key], dtype=np.float64)
        fill_count[positions] += 1
        trajectory_origin[positions] = "legacy_v1"
        trajectory_rng_group_seed[positions] = record.task.seed
        diagnostic_payloads.append(payload)

    if computed_indices:
        if computed_payload is None or computed_rng_seed is None:
            raise ValueError("computed samples require payload and RNG seed")
        indices = np.asarray(computed_indices, dtype=np.int64)
        positions = indices - task.sample_start
        if np.any(positions < 0) or np.any(positions >= task.sample_count):
            raise ValueError("computed sample lies outside its v2 macro range")
        for key in TRAJECTORY_KEYS:
            values = np.asarray(computed_payload[key], dtype=np.float64)
            if values.shape != (len(computed_indices),):
                raise ValueError(f"computed {key} has the wrong shape")
            arrays[key][positions] = values
        fill_count[positions] += 1
        trajectory_origin[positions] = "computed_v2"
        trajectory_rng_group_seed[positions] = computed_rng_seed
        diagnostic_payloads.append(computed_payload)
    elif computed_payload is not None or computed_rng_seed is not None:
        raise ValueError("empty computed index set must not carry computed data")

    if not np.all(fill_count == 1):
        raise RuntimeError("v2 macro does not contain exactly one value per sample")
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"merged {key} contains nonfinite values")
    for key in ("entropy_a_y0avg", "entropy_b_y0avg", "entropy_union_y0avg"):
        if float(arrays[key].min()) < -1.0e-10:
            raise FloatingPointError(f"merged {key} contains negative entropy")
    if not diagnostic_payloads:
        raise RuntimeError("v2 macro has no legacy or computed input")

    def reduce_scalar(key: str, reducer: Any, cast: Any = float) -> Any:
        values = [
            cast(np.asarray(payload[key]).item()) for payload in diagnostic_payloads
        ]
        return reducer(values)

    mutual_information = arrays["mutual_information_y0avg"]
    entropy_residual = mutual_information - (
        arrays["entropy_a_y0avg"]
        + arrays["entropy_b_y0avg"]
        - arrays["entropy_union_y0avg"]
    )
    legacy_indices = tuple(
        index
        for record, _ in legacy_inputs
        for index in record.task.global_sample_indices
    )
    legacy_elapsed = sum(
        float(record.completion.get("elapsed_seconds", 0.0))
        for record, _ in legacy_inputs
    )
    payload: dict[str, np.ndarray] = {
        **arrays,
        "schema": np.asarray(RESULT_SCHEMA),
        "observer_schema": np.asarray(OBSERVER_SCHEMA),
        "bundle": np.asarray(BUNDLE),
        "sampling_revision": np.asarray(EXPECTED_REVISION),
        "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
        "task_id": np.asarray(task.task_id),
        "Nx": np.asarray(EXPECTED_NX, dtype=np.int64),
        "Ny": np.asarray(task.ny, dtype=np.int64),
        "alpha_1": np.asarray(task.alpha_1, dtype=np.float64),
        "alpha_2": np.asarray(EXPECTED_ALPHA_2, dtype=np.float64),
        "wall": np.asarray(task.wall),
        "dw_truncation": np.asarray(task.dw_truncation, dtype=np.bool_),
        "meas_slab_only_requested": np.asarray(True, dtype=np.bool_),
        "meas_slab_only_effective": np.asarray(task.dw_truncation, dtype=np.bool_),
        "nshell": np.asarray(EXPECTED_NSHELL, dtype=np.int64),
        "batch_index": np.asarray(task.batch_index, dtype=np.int64),
        "sample_start": np.asarray(task.sample_start, dtype=np.int64),
        "sample_stop": np.asarray(task.sample_stop, dtype=np.int64),
        "global_sample_indices": np.asarray(task.global_sample_indices, dtype=np.int64),
        "macro_seed": np.asarray(task.seed, dtype=np.int64),
        "computed_rng_seed": np.asarray(
            -1 if computed_rng_seed is None else computed_rng_seed, dtype=np.int64
        ),
        "trajectory_origin": trajectory_origin,
        "trajectory_rng_group_seed": trajectory_rng_group_seed,
        "legacy_sample_indices": np.asarray(legacy_indices, dtype=np.int64),
        "computed_sample_indices": np.asarray(computed_indices, dtype=np.int64),
        "legacy_task_ids": np.asarray(
            [record.task.task_id for record, _ in legacy_inputs], dtype="U96"
        ),
        "computed_elapsed_seconds": np.asarray(
            computed_elapsed_seconds, dtype=np.float64
        ),
        "imported_legacy_elapsed_seconds": np.asarray(legacy_elapsed, dtype=np.float64),
        "elapsed_seconds": np.asarray(computed_elapsed_seconds, dtype=np.float64),
        "endpoint_cycle": np.asarray(task.cycles, dtype=np.int64),
        "width": np.asarray(task.width, dtype=np.int64),
        "nominal_y0_count": np.asarray(task.ny, dtype=np.int64),
        "unique_y0_count": np.asarray(task.ny // 2, dtype=np.int64),
        "full_covariance_max_hermiticity_error": np.asarray(
            reduce_scalar("full_covariance_max_hermiticity_error", max),
            dtype=np.float64,
        ),
        "restricted_max_hermiticity_error": np.asarray(
            reduce_scalar("restricted_max_hermiticity_error", max),
            dtype=np.float64,
        ),
        "restricted_occupation_eigenvalue_min": np.asarray(
            reduce_scalar("restricted_occupation_eigenvalue_min", min),
            dtype=np.float64,
        ),
        "restricted_occupation_eigenvalue_max": np.asarray(
            reduce_scalar("restricted_occupation_eigenvalue_max", max),
            dtype=np.float64,
        ),
        "restricted_eigensolve_count": np.asarray(
            reduce_scalar("restricted_eigensolve_count", sum, int), dtype=np.int64
        ),
        "materially_negative_mi_count": np.asarray(
            int(np.count_nonzero(mutual_information < -1.0e-8)), dtype=np.int64
        ),
        "mutual_information_min": np.asarray(
            float(mutual_information.min()), dtype=np.float64
        ),
        "entropy_identity_max_abs_residual": np.asarray(
            float(np.max(np.abs(entropy_residual))), dtype=np.float64
        ),
        "entropy_log_base": np.asarray("natural"),
        "y_boundary_condition": np.asarray("periodic"),
    }
    return payload


def _save_task(
    *,
    output_root: Path,
    scratch_root: Path,
    task: Task,
    payload: dict[str, np.ndarray],
    elapsed_seconds: float,
    legacy_inputs: list[tuple[LegacyRecord, dict[str, np.ndarray]]],
    computed_indices: tuple[int, ...],
    computed_rng_seed: int | None,
    config_sha256: str,
    hashes: dict[str, str],
) -> None:
    task_scratch = scratch_root / task.task_id
    if task_scratch.exists():
        shutil.rmtree(task_scratch)
    task_scratch.mkdir(parents=True)
    local_result = task_scratch / "result.npz"
    _write_npz(local_result, payload)
    result_path, completion_path = task_paths(output_root, task)
    published = publish_file(local_result, result_path)
    completion = _expected_completion_identity(
        task=task, config_sha256=config_sha256, hashes=hashes
    )
    completion.update(
        {
            "result_filename": result_path.name,
            "result_bytes": published["bytes"],
            "result_sha256": published["sha256"],
            "samples_saved": task.sample_count,
            "legacy_sample_indices": [
                index
                for record, _ in legacy_inputs
                for index in record.task.global_sample_indices
            ],
            "computed_sample_indices": list(computed_indices),
            "computed_rng_seed": computed_rng_seed,
            "legacy_inputs": [
                {
                    "task_id": record.task.task_id,
                    "global_sample_indices": list(record.task.global_sample_indices),
                    "result_filename": record.result_path.name,
                    "result_bytes": int(record.completion["result_bytes"]),
                    "result_sha256": str(record.completion["result_sha256"]),
                    "source_hashes": dict(record.completion["source_hashes"]),
                }
                for record, _ in legacy_inputs
            ],
            "materially_negative_mi_count": int(
                np.asarray(payload["materially_negative_mi_count"]).item()
            ),
            "elapsed_seconds": float(elapsed_seconds),
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_completion = task_scratch / "completion.json"
    _write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    verified, reason = verified_complete(
        output_root=output_root,
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    if not verified:
        raise OSError(f"published task failed final verification: {reason}")
    shutil.rmtree(task_scratch)


def aggregate_paths(output_root: Path) -> tuple[Path, Path, Path, Path]:
    aggregate_root = output_root / "aggregate"
    return (
        aggregate_root / "domain_wall_bipartite_mutual_information.npz",
        aggregate_root / "domain_wall_bipartite_mutual_information.pdf",
        aggregate_root / "domain_wall_bipartite_mutual_information.png",
        aggregate_root / "aggregate.complete.json",
    )


def _aggregate_completion_identity(
    *, config_sha256: str, hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema": AGGREGATE_COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "config_sha256": config_sha256,
        "source_hashes": hashes,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "observer_schema": OBSERVER_SCHEMA,
        "cases": EXPECTED_CASES,
        "tasks": EXPECTED_TASKS,
        "trajectories": EXPECTED_TRAJECTORIES,
    }


def verified_aggregate(
    *, output_root: Path, config_sha256: str, hashes: dict[str, str]
) -> tuple[bool, str]:
    data_path, pdf_path, png_path, completion_path = aggregate_paths(output_root)
    products = (data_path, pdf_path, png_path)
    if not completion_path.is_file() and not any(path.is_file() for path in products):
        return False, "missing aggregate products"
    if not completion_path.is_file() or not all(path.is_file() for path in products):
        return False, "incomplete aggregate products"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable aggregate completion JSON: {exc}"
    for key, value in _aggregate_completion_identity(
        config_sha256=config_sha256, hashes=hashes
    ).items():
        if completion.get(key) != value:
            return False, f"aggregate identity mismatch: {key}"
    declared = completion.get("products", {})
    for path in products:
        record = declared.get(path.name)
        if not isinstance(record, dict):
            return False, f"missing aggregate product record: {path.name}"
        if int(record.get("bytes", -1)) != path.stat().st_size:
            return False, f"aggregate byte count mismatch: {path.name}"
        if record.get("sha256") != sha256_file(path):
            return False, f"aggregate checksum mismatch: {path.name}"
    return True, "verified"


def collect_aggregate(
    *,
    config: dict[str, Any],
    tasks: list[Task],
    output_root: Path,
    config_sha256: str,
    hashes: dict[str, str],
) -> dict[str, np.ndarray]:
    wall_index = {wall: index for index, wall in enumerate(EXPECTED_WALLS)}
    ny_index = {ny: index for index, ny in enumerate(EXPECTED_NY_VALUES)}
    alpha_index = {alpha: index for index, alpha in enumerate(EXPECTED_ALPHA_VALUES)}
    shape = (
        len(EXPECTED_WALLS),
        len(EXPECTED_NY_VALUES),
        len(EXPECTED_ALPHA_VALUES),
        EXPECTED_SAMPLES,
    )
    arrays = {key: np.full(shape, np.nan, dtype=np.float64) for key in TRAJECTORY_KEYS}
    trajectory_origin = np.full(shape, "", dtype="U11")
    trajectory_rng_group_seed = np.full(shape, -1, dtype=np.int64)
    fill_count = np.zeros(shape, dtype=np.uint8)
    negative_count = 0
    for task in tqdm(tasks, desc="aggregate verified macros", unit="macro"):
        verified, reason = verified_complete(
            output_root=output_root,
            task=task,
            config_sha256=config_sha256,
            hashes=hashes,
        )
        if not verified:
            raise RuntimeError(f"cannot aggregate {task.task_id}: {reason}")
        result_path, _ = task_paths(output_root, task)
        with np.load(result_path, allow_pickle=False) as data:
            indices = np.asarray(data["global_sample_indices"], dtype=np.int64)
            expected_indices = np.asarray(task.global_sample_indices, dtype=np.int64)
            if not np.array_equal(indices, expected_indices):
                raise RuntimeError(f"sample indices mismatch in {result_path}")
            target = (
                wall_index[task.wall],
                ny_index[task.ny],
                alpha_index[task.alpha_1],
                indices,
            )
            for key in TRAJECTORY_KEYS:
                values = np.asarray(data[key], dtype=np.float64)
                if values.shape != (task.sample_count,):
                    raise RuntimeError(f"invalid {key} shape in {result_path}")
                arrays[key][target] = values
            origins = np.asarray(data["trajectory_origin"])
            seeds = np.asarray(data["trajectory_rng_group_seed"], dtype=np.int64)
            if origins.shape != (task.sample_count,) or seeds.shape != (
                task.sample_count,
            ):
                raise RuntimeError(f"invalid trajectory provenance in {result_path}")
            trajectory_origin[target] = origins
            trajectory_rng_group_seed[target] = seeds
            fill_count[target] += 1
            negative_count += int(
                np.asarray(data["materially_negative_mi_count"]).item()
            )
    if not np.all(fill_count == 1):
        raise RuntimeError("aggregate sample coverage is not exactly one")
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"aggregate {key} contains nonfinite values")
    if not np.all(np.isin(trajectory_origin, ("legacy_v1", "computed_v2"))):
        raise RuntimeError("aggregate trajectory provenance is incomplete")
    if np.any(trajectory_rng_group_seed < 0):
        raise RuntimeError("aggregate trajectory RNG provenance is incomplete")
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(AGGREGATE_SCHEMA),
        "bundle": np.asarray(BUNDLE),
        "sampling_revision": np.asarray(EXPECTED_REVISION),
        "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
        "observer_schema": np.asarray(OBSERVER_SCHEMA),
        "wall_labels": np.asarray(EXPECTED_WALLS),
        "Ny_values": np.asarray(EXPECTED_NY_VALUES, dtype=np.int64),
        "width_values": np.asarray(
            [EXPECTED_WIDTHS[ny] for ny in EXPECTED_NY_VALUES], dtype=np.int64
        ),
        "cycles": np.asarray([2 * ny for ny in EXPECTED_NY_VALUES], dtype=np.int64),
        "alpha_1_values": np.asarray(EXPECTED_ALPHA_VALUES, dtype=np.float64),
        "alpha_2": np.asarray(EXPECTED_ALPHA_2, dtype=np.float64),
        "sample_indices": np.arange(EXPECTED_SAMPLES, dtype=np.int64),
        "trajectory_origin": trajectory_origin,
        "trajectory_rng_group_seed": trajectory_rng_group_seed,
        "materially_negative_mi_count": np.asarray(negative_count, dtype=np.int64),
        "config_json": np.asarray(json.dumps(config, sort_keys=True)),
    }
    for key, values in arrays.items():
        payload[key] = values
        payload[f"{key}_mean"] = values.mean(axis=-1)
    return payload


def make_figure(payload: dict[str, np.ndarray], *, pdf: Path, png: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
        }
    )
    alpha_values = np.asarray(payload["alpha_1_values"], dtype=np.float64)
    means = np.asarray(payload["mutual_information_y0avg_mean"], dtype=np.float64)
    styles = (
        {"color": "#d62728", "marker": "^", "linestyle": ":"},
        {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
        {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.0), sharex=True, sharey=True)
    for wall_idx, (wall, axis) in enumerate(zip(EXPECTED_WALLS, axes)):
        for ny_idx, (ny, style) in enumerate(zip(EXPECTED_NY_VALUES, styles)):
            axis.plot(
                alpha_values,
                means[wall_idx, ny_idx],
                label=rf"$N_y={ny}$",
                linewidth=1.2,
                markersize=3.5,
                markeredgewidth=0.6,
                **style,
            )
        axis.axvline(2.0, color="0.25", linestyle="--", linewidth=0.9)
        axis.set_xlabel(r"$\alpha_1$")
        axis.set_title(
            "hard / support-truncated" if wall == "hard" else "soft / untruncated"
        )
        axis.tick_params(direction="in", top=True, right=True)
        for spine in axis.spines.values():
            spine.set_visible(True)
    axes[0].set_ylabel(r"$\overline{I}_{a,b}$")
    axes[0].legend(frameon=False, loc="best")
    axes[0].text(-0.13, 1.03, "(a)", transform=axes[0].transAxes)
    axes[1].text(-0.13, 1.03, "(b)", transform=axes[1].transAxes)
    fig.tight_layout()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def publish_aggregate(
    *,
    config: dict[str, Any],
    tasks: list[Task],
    output_root: Path,
    scratch_root: Path,
    config_sha256: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    verified, reason = verified_aggregate(
        output_root=output_root, config_sha256=config_sha256, hashes=hashes
    )
    if verified:
        print("[aggregate] verified existing aggregate products", flush=True)
        return {"status": "existing", "reason": reason}
    print(f"[aggregate] rebuilding: {reason}", flush=True)
    payload = collect_aggregate(
        config=config,
        tasks=tasks,
        output_root=output_root,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    local_root = scratch_root / "aggregate"
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True)
    local_data = local_root / "domain_wall_bipartite_mutual_information.npz"
    local_pdf = local_root / "domain_wall_bipartite_mutual_information.pdf"
    local_png = local_root / "domain_wall_bipartite_mutual_information.png"
    _write_npz(local_data, payload)
    make_figure(payload, pdf=local_pdf, png=local_png)
    final_data, final_pdf, final_png, completion_path = aggregate_paths(output_root)
    products: dict[str, dict[str, Any]] = {}
    for local_path, final_path in (
        (local_data, final_data),
        (local_pdf, final_pdf),
        (local_png, final_png),
    ):
        record = publish_file(local_path, final_path)
        products[final_path.name] = {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
    completion = _aggregate_completion_identity(
        config_sha256=config_sha256, hashes=hashes
    )
    completion.update(
        {
            "products": products,
            "materially_negative_mi_count": int(
                np.asarray(payload["materially_negative_mi_count"]).item()
            ),
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_completion = local_root / "aggregate.complete.json"
    _write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    verified, reason = verified_aggregate(
        output_root=output_root, config_sha256=config_sha256, hashes=hashes
    )
    if not verified:
        raise OSError(f"published aggregate failed verification: {reason}")
    shutil.rmtree(local_root)
    print("[aggregate] data and figures published and verified", flush=True)
    return {"status": "created", "reason": reason}


def inspect_legacy_inputs(
    *, legacy_output_root: Path | None, pending_tasks: list[Task]
) -> tuple[
    dict[str, list[tuple[LegacyRecord, dict[str, np.ndarray]]]],
    dict[str, int],
]:
    by_macro: dict[str, list[tuple[LegacyRecord, dict[str, np.ndarray]]]] = {}
    verified_pairs = 0
    reusable_samples = 0
    invalid_or_partial = 0
    for task in pending_tasks:
        contributions: list[tuple[LegacyRecord, dict[str, np.ndarray]]] = []
        for legacy_task in legacy_tasks_for_macro(task):
            record, reason = verified_legacy_record(
                output_root=legacy_output_root, task=legacy_task
            )
            if record is None:
                if reason not in (
                    "legacy output root disabled",
                    "missing legacy result/completion pair",
                ):
                    invalid_or_partial += 1
                continue
            try:
                payload = load_legacy_payload(record)
            except ValueError:
                invalid_or_partial += 1
                continue
            contributions.append((record, payload))
            verified_pairs += 1
            reusable_samples += legacy_task.sample_count
        by_macro[task.task_id] = contributions
    return by_macro, {
        "verified_pairs": verified_pairs,
        "reusable_samples": reusable_samples,
        "invalid_or_partial_pairs": invalid_or_partial,
    }


def run_campaign(
    *,
    config: dict[str, Any],
    output_root: Path,
    legacy_output_root: Path | None = None,
    scratch_root: Path,
    report_only: bool = False,
    max_new_tasks: int | None = None,
) -> dict[str, Any]:
    config = validate_config(config)
    tasks = expand_tasks(config)
    hashes = source_hashes()
    config_sha256 = config_hash(config)
    inventory = {
        task.task_id: verified_complete(
            output_root=output_root,
            task=task,
            config_sha256=config_sha256,
            hashes=hashes,
        )
        for task in tasks
    }
    completed = sum(int(value[0]) for value in inventory.values())
    invalid = sum(
        int((not value[0]) and value[1] != "missing result/completion pair")
        for value in inventory.values()
    )
    pending_tasks = [task for task in tasks if not inventory[task.task_id][0]]
    legacy_by_macro, legacy_status = inspect_legacy_inputs(
        legacy_output_root=legacy_output_root, pending_tasks=pending_tasks
    )
    gpu_trajectories_remaining = sum(
        task.sample_count
        - sum(record.task.sample_count for record, _ in legacy_by_macro[task.task_id])
        for task in pending_tasks
    )
    fully_legacy_macros = sum(
        int(
            sum(record.task.sample_count for record, _ in legacy_by_macro[task.task_id])
            == task.sample_count
        )
        for task in pending_tasks
    )
    aggregate_verified, aggregate_reason = verified_aggregate(
        output_root=output_root, config_sha256=config_sha256, hashes=hashes
    )
    workload = {
        "cases": EXPECTED_CASES,
        "samples_per_case": EXPECTED_SAMPLES,
        "trajectories": EXPECTED_TRAJECTORIES,
        "batch_size_by_Ny": EXPECTED_BATCH_SIZES,
        "tasks": len(tasks),
        "completed": completed,
        "pending": len(pending_tasks),
        "invalid_or_partial": invalid,
        "legacy_verified_pairs": legacy_status["verified_pairs"],
        "legacy_reusable_samples": legacy_status["reusable_samples"],
        "legacy_invalid_or_partial_pairs": legacy_status["invalid_or_partial_pairs"],
        "fully_legacy_pending_macros": fully_legacy_macros,
        "gpu_trajectories_remaining": gpu_trajectories_remaining,
        "aggregate_verified": aggregate_verified,
        "aggregate_status": aggregate_reason,
    }
    print(
        json.dumps(
            {
                "bundle": BUNDLE,
                "sampling_revision": config["sampling_revision"],
                "canonical_entry_point": CANONICAL_ENTRY_POINT,
                "output_root": str(output_root),
                "legacy_output_root": (
                    None if legacy_output_root is None else str(legacy_output_root)
                ),
                "scratch_root": str(scratch_root),
                "configuration": config,
                "config_sha256": config_sha256,
                "source_hashes": hashes,
                "workload": workload,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if report_only:
        return {"status": "report_only", **workload}

    gpu = validate_a100()
    local_free = _check_space(scratch_root, required_bytes=2 * 1024**3, label="local")
    drive_free = _check_space(output_root, required_bytes=1024**3, label="Drive")
    print(
        "[preflight] "
        + json.dumps(
            {
                "gpu": gpu,
                "dtype": config["dtype"],
                "local_free_gib": local_free / 1024**3,
                "drive_free_gib": drive_free / 1024**3,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    failed = 0
    new_completed = 0
    legacy_samples_imported = 0
    pending_count = len(pending_tasks)
    bar = tqdm(
        total=len(tasks),
        initial=completed,
        desc="domain-wall BMI campaign",
        unit="macro",
        dynamic_ncols=True,
        leave=True,
    )
    bar.set_postfix(
        completed=completed,
        skipped=completed,
        imported=0,
        pending=pending_count,
        failed=failed,
    )
    grouped: dict[tuple[int, str, float], list[Task]] = {}
    for task in pending_tasks:
        grouped.setdefault((task.ny, task.wall, task.alpha_1), []).append(task)
    try:
        for ny in EXPECTED_NY_VALUES:
            for wall in EXPECTED_WALLS:
                for alpha_1 in EXPECTED_ALPHA_VALUES:
                    group = grouped.get((ny, wall, alpha_1), [])
                    if not group:
                        continue
                    model = None
                    try:
                        for task in group:
                            if (
                                max_new_tasks is not None
                                and new_completed >= max_new_tasks
                            ):
                                summary = {
                                    "status": "partial_limit_reached",
                                    "total": len(tasks),
                                    "completed": completed + new_completed,
                                    "skipped": completed,
                                    "legacy_samples_imported": legacy_samples_imported,
                                    "pending": pending_count,
                                    "failed": failed,
                                }
                                print(
                                    "[campaign summary] " + json.dumps(summary),
                                    flush=True,
                                )
                                return summary
                            bar.set_description(
                                f"Ny={task.ny} {task.wall} a={task.alpha_1:g} "
                                f"samples={task.sample_start:03d}-{task.sample_stop - 1:03d}"
                            )
                            try:
                                legacy_inputs = legacy_by_macro[task.task_id]
                                legacy_indices = {
                                    index
                                    for record, _ in legacy_inputs
                                    for index in record.task.global_sample_indices
                                }
                                missing_indices = tuple(
                                    index
                                    for index in task.global_sample_indices
                                    if index not in legacy_indices
                                )
                                computed_payload = None
                                computed_rng_seed = None
                                elapsed = 0.0
                                if missing_indices:
                                    if model is None:
                                        model = build_model(config, task)
                                    computed_payload, elapsed, computed_rng_seed = (
                                        run_missing_samples(
                                            model=model,
                                            config=config,
                                            task=task,
                                            sample_indices=missing_indices,
                                        )
                                    )
                                payload = merge_task_payload(
                                    task=task,
                                    legacy_inputs=legacy_inputs,
                                    computed_indices=missing_indices,
                                    computed_payload=computed_payload,
                                    computed_elapsed_seconds=elapsed,
                                    computed_rng_seed=computed_rng_seed,
                                )
                                _save_task(
                                    output_root=output_root,
                                    scratch_root=scratch_root,
                                    task=task,
                                    payload=payload,
                                    elapsed_seconds=elapsed,
                                    legacy_inputs=legacy_inputs,
                                    computed_indices=missing_indices,
                                    computed_rng_seed=computed_rng_seed,
                                    config_sha256=config_sha256,
                                    hashes=hashes,
                                )
                            except Exception:
                                failed += 1
                                bar.set_postfix(
                                    completed=completed + new_completed,
                                    skipped=completed,
                                    imported=legacy_samples_imported,
                                    pending=pending_count,
                                    failed=failed,
                                )
                                raise
                            new_completed += 1
                            legacy_samples_imported += sum(
                                record.task.sample_count for record, _ in legacy_inputs
                            )
                            pending_count -= 1
                            bar.update(1)
                            bar.set_postfix(
                                completed=completed + new_completed,
                                skipped=completed,
                                imported=legacy_samples_imported,
                                pending=pending_count,
                                failed=failed,
                            )
                    finally:
                        if model is not None:
                            del model
                            torch.cuda.empty_cache()
    finally:
        bar.close()

    summary = {
        "status": "complete",
        "total": len(tasks),
        "completed": completed + new_completed,
        "skipped": completed,
        "legacy_samples_available": legacy_status["reusable_samples"],
        "legacy_samples_imported": legacy_samples_imported,
        "pending": pending_count,
        "failed": failed,
    }
    if summary["completed"] != len(tasks) or pending_count != 0 or failed != 0:
        raise RuntimeError(f"campaign ended without full completion: {summary}")
    summary["aggregate"] = publish_aggregate(
        config=config,
        tasks=tasks,
        output_root=output_root,
        scratch_root=scratch_root,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    print("[campaign summary] " + json.dumps(summary), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--legacy-output-root", type=Path)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/content/domain_wall_bmi_scratch"),
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-tasks", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        raise ValueError("--max-new-tasks must be nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_campaign(
        config=config,
        output_root=args.output_root,
        legacy_output_root=args.legacy_output_root,
        scratch_root=args.scratch_root,
        report_only=bool(args.report_only),
        max_new_tasks=args.max_new_tasks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
