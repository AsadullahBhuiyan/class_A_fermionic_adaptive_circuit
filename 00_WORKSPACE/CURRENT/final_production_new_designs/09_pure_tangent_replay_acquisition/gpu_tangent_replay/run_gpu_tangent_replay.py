#!/usr/bin/env python3
"""Aggressively batched A100 tangent replay of the completed slot-09 records."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
SRC_ROOT = BUNDLE_ROOT / "src"
for candidate in (BUNDLE_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_campaign as acquisition  # noqa: E402
from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from replay_record_observer import unpack_boolean_record  # noqa: E402


CONFIG_PATH = HERE / "campaign_config.json"
ACQUISITION_REVISION = "pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1"
REPLAY_REVISION = "pure_tangent_gpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1"
RESULT_SCHEMA = "pure_tangent_gpu_replay_result_v1"
COMPLETION_SCHEMA = "pure_tangent_gpu_replay_completion_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
DEFAULT_ACQUISITION_ROOT = Path(
    "/content/drive/MyDrive/classA_final_production_outputs"
) / ACQUISITION_REVISION
DEFAULT_OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/classA_final_production_outputs"
) / REPLAY_REVISION
CHANNELS = ("Ap", "Am", "Bp", "Bm")
EXPECTED_NY = (24, 28, 32)
EXPECTED_CONSTRUCTIONS = ("hard", "soft")
EXPECTED_ALPHA_1 = (1.0, 3.0)
SAMPLES_PER_CASE = 100
SAMPLES_PER_GPU_TASK = 25
EXPECTED_TASKS = 48
EXPECTED_SAMPLES = 1200
SOURCE_PATHS = {
    "runner": Path(__file__).resolve(),
    "config": CONFIG_PATH,
    "acquisition_runner": BUNDLE_ROOT / "run_campaign.py",
    "record_helper": BUNDLE_ROOT / "replay_record_observer.py",
    "gpu_engine": SRC_ROOT / "classA_U1FGTN_gpu.py",
    "occupied_frame": SRC_ROOT / "occupied_frame_gpu.py",
}


@dataclass(frozen=True)
class ReplayTask:
    source: acquisition.Task
    source_row_start: int
    source_row_stop: int

    @property
    def sample_count(self) -> int:
        return self.source_row_stop - self.source_row_start

    @property
    def construction(self) -> str:
        return self.source.construction

    @property
    def ny(self) -> int:
        return self.source.ny

    @property
    def alpha_1(self) -> float:
        return self.source.alpha_1

    @property
    def cycles(self) -> int:
        return self.source.cycles

    @property
    def case_sample_indices(self) -> tuple[int, ...]:
        return self.source.case_sample_indices[
            self.source_row_start : self.source_row_stop
        ]

    @property
    def global_sample_indices(self) -> tuple[int, ...]:
        return self.source.global_sample_indices[
            self.source_row_start : self.source_row_stop
        ]

    @property
    def task_id(self) -> str:
        return (
            f"{self.construction}_Ny{self.ny:03d}_a1-{int(self.alpha_1)}_"
            f"samples-{self.case_sample_indices[0]:03d}-"
            f"{self.case_sample_indices[-1]:03d}"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    expected_bytes = int(local_path.stat().st_size)
    expected_sha = sha256_file(local_path)
    try:
        shutil.copyfile(local_path, temporary)
        if int(temporary.stat().st_size) != expected_bytes:
            raise OSError(f"Drive temporary byte-count mismatch: {temporary}")
        if sha256_file(temporary) != expected_sha:
            raise OSError(f"Drive temporary checksum mismatch: {temporary}")
        os.replace(temporary, final_path)
        if int(final_path.stat().st_size) != expected_bytes:
            raise OSError(f"Drive final byte-count mismatch: {final_path}")
        if sha256_file(final_path) != expected_sha:
            raise OSError(f"Drive final checksum mismatch: {final_path}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"filename": final_path.name, "bytes": expected_bytes, "sha256": expected_sha}


def expected_config() -> dict[str, Any]:
    return {
        "schema": "pure_tangent_gpu_replay_config_v1",
        "sampling_revision": REPLAY_REVISION,
        "acquisition_revision": ACQUISITION_REVISION,
        "Nx": 20,
        "Ny_values": list(EXPECTED_NY),
        "constructions": list(EXPECTED_CONSTRUCTIONS),
        "alpha_1_values": list(EXPECTED_ALPHA_1),
        "alpha_2": 30.0,
        "nshell": 1,
        "samples_per_case": SAMPLES_PER_CASE,
        "samples_per_gpu_task": SAMPLES_PER_GPU_TASK,
        "cycles_multiplier": 2,
        "trial_orbitals": "X",
        "filling_fraction": 0.5,
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "dtype": "complex128",
        "device": "cuda:0",
        "state_representation": "physical_frame",
        "tangent_basis_mode": "batched_canonical_from_occupied_empty_basis",
        "hard_wall_active_slab_only": True,
        "late_window_start_cycle_multiplier": 1,
        "slow_mode_count": 16,
        "singular_tolerance": 1e-14,
        "initial_purity_tolerance": 2e-9,
        "replay_probability_tolerance": 1e-9,
        "endpoint_projector_relative_frobenius_tolerance": 1e-6,
        "cross_replay_projector_relative_frobenius_tolerance": 2e-6,
        "maximum_task_runtime_seconds": 3600.0,
        "maximum_peak_cuda_reserved_gib": 36.0,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "saved_products": {
            "final_scale_separated_one_leg_cocycle": True,
            "per_cycle_qr_log_increments": True,
            "full_window_one_leg_singular_logs": True,
            "late_window_one_leg_singular_logs": True,
            "slowest_particle_hole_rates_and_x_profiles": True,
            "per_sample_occupied_empty_block_sizes": True,
            "per_cycle_dense_jacobians": False,
            "choi_covariance": False,
            "covariance_history": False,
            "intermediate_occupied_frames": False,
        },
    }


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    observed = load_json(path)
    if observed != expected_config():
        raise ValueError("GPU replay configuration differs from its locked v1 contract")
    return observed


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}


def expand_tasks() -> list[ReplayTask]:
    sources = acquisition.expand_tasks(acquisition.expected_config())
    tasks: list[ReplayTask] = []
    for source in sources:
        for start in range(0, source.sample_count, SAMPLES_PER_GPU_TASK):
            tasks.append(
                ReplayTask(
                    source=source,
                    source_row_start=start,
                    source_row_stop=min(source.sample_count, start + SAMPLES_PER_GPU_TASK),
                )
            )
    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"expected {EXPECTED_TASKS} GPU tasks, found {len(tasks)}")
    if sum(task.sample_count for task in tasks) != EXPECTED_SAMPLES:
        raise RuntimeError("GPU task expansion did not produce 1,200 sample rows")
    ids = [task.task_id for task in tasks]
    indices = [index for task in tasks for index in task.global_sample_indices]
    if len(ids) != len(set(ids)) or sorted(indices) != list(range(EXPECTED_SAMPLES)):
        raise RuntimeError("GPU task identities or sample partition are invalid")
    return tasks


def result_paths(output_root: Path, task: ReplayTask) -> tuple[Path, Path]:
    directory = (
        output_root
        / task.construction
        / f"Ny{task.ny:03d}"
        / f"alpha1_{int(task.alpha_1)}"
    )
    stem = f"samples_{task.case_sample_indices[0]:03d}-{task.case_sample_indices[-1]:03d}"
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def task_identity(
    task: ReplayTask,
    *,
    config_sha256: str,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "sampling_revision": REPLAY_REVISION,
        "acquisition_revision": ACQUISITION_REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "task_id": task.task_id,
        "construction": task.construction,
        "Nx": 20,
        "Ny": task.ny,
        "alpha_1": task.alpha_1,
        "alpha_2": 30.0,
        "nshell": 1,
        "cycles": task.cycles,
        "case_sample_indices": list(task.case_sample_indices),
        "global_sample_indices": list(task.global_sample_indices),
        "sample_count": task.sample_count,
        "configuration_sha256": config_sha256,
        "source_hashes": dict(hashes),
        "acquisition_task_id": task.source.task_id,
        "acquisition_batch_seed": task.source.seed,
        "acquisition_source_rows": list(
            range(task.source_row_start, task.source_row_stop)
        ),
    }


def validate_result_npz(
    path: Path,
    task: ReplayTask,
    *,
    config_sha256: str,
    hashes: Mapping[str, str],
) -> None:
    """Validate the closed scientific file before or after Drive publication."""

    dimension = 40 * task.ny
    with np.load(path, allow_pickle=False) as archive:
        scalar_expectations = {
            "schema": RESULT_SCHEMA,
            "sampling_revision": REPLAY_REVISION,
            "acquisition_revision": ACQUISITION_REVISION,
            "canonical_entry_point": CANONICAL_ENTRY_POINT,
            "task_id": task.task_id,
            "configuration_sha256": config_sha256,
            "dtype": "complex128",
            "sample_count": task.sample_count,
            "cycles": task.cycles,
        }
        for key, expected in scalar_expectations.items():
            if archive[key].item() != expected:
                raise ValueError(f"result identity mismatch: {key}")
        if json.loads(str(archive["source_hashes_json"].item())) != dict(hashes):
            raise ValueError("result source hashes mismatch")
        if not np.array_equal(
            archive["global_sample_indices"],
            np.asarray(task.global_sample_indices, dtype=np.int64),
        ):
            raise ValueError("result sample indices mismatch")

        cocycle = archive["final_cocycle_hat"]
        if cocycle.shape != (task.sample_count, dimension, dimension):
            raise ValueError("final cocycle shape mismatch")
        if cocycle.dtype != np.complex128 or not np.isfinite(cocycle).all():
            raise FloatingPointError("final cocycle dtype/finiteness mismatch")
        norms = np.linalg.norm(cocycle.reshape(task.sample_count, -1), axis=1)
        if not np.allclose(norms, 1.0, atol=2e-12, rtol=2e-12):
            raise FloatingPointError("scale-separated final cocycle is not normalized")
        scales = np.asarray(archive["final_cocycle_log_scale"], dtype=np.float64)
        if scales.shape != (task.sample_count,) or not np.all(np.isfinite(scales)):
            raise FloatingPointError("final cocycle scale is invalid")

        active_dimension = int(archive["active_dimension"].item())
        active_indices = np.asarray(archive["active_input_indices"], dtype=np.int64)
        if active_indices.shape != (active_dimension,):
            raise ValueError("active tangent index shape mismatch")
        if np.unique(active_indices).size != active_dimension:
            raise ValueError("active tangent indices are not unique")
        if np.any(active_indices < 0) or np.any(active_indices >= dimension):
            raise ValueError("active tangent indices are out of bounds")

        for prefix, expected_cycles in (("full", task.cycles), ("late", task.ny)):
            block_sizes = np.asarray(archive[f"{prefix}_block_sizes"], dtype=np.int64)
            if block_sizes.shape != (task.sample_count, 2) or np.any(
                block_sizes.sum(axis=1) != active_dimension
            ):
                raise ValueError(f"{prefix} occupied/empty block sizes are invalid")
            if np.any(block_sizes <= 0):
                raise ValueError(f"{prefix} contains an empty tangent block")
            for side, column in (("occupied", 0), ("empty", 1)):
                logs = np.asarray(
                    archive[f"{prefix}_one_leg_logs_{side}"], dtype=np.float64
                )
                valid = np.asarray(
                    archive[f"{prefix}_one_leg_{side}_valid"], dtype=np.bool_
                )
                if logs.shape != (task.sample_count, active_dimension) or valid.shape != logs.shape:
                    raise ValueError(f"{prefix} padded {side} singular-log shape mismatch")
                expected_valid = (
                    np.arange(active_dimension)[None, :] < block_sizes[:, column, None]
                )
                if not np.array_equal(valid, expected_valid):
                    raise ValueError(f"{prefix} padded {side} valid mask mismatch")
                if not np.all(np.isfinite(logs[valid])) or not np.all(np.isneginf(logs[~valid])):
                    raise FloatingPointError(f"{prefix} padded {side} singular logs are invalid")
            expected_history = (task.sample_count, expected_cycles, active_dimension)
            for field in (
                "qr_cumulative_log_diag",
                "qr_log_increments",
                "qr_finite_time_rates",
            ):
                values = np.asarray(archive[f"{prefix}_{field}"], dtype=np.float64)
                if values.shape != expected_history or not np.all(np.isfinite(values)):
                    raise FloatingPointError(f"{prefix} {field} is invalid")
            null_mask = np.asarray(archive[f"{prefix}_cycle_null_mask"], dtype=np.bool_)
            if null_mask.shape != expected_history:
                raise ValueError(f"{prefix} cycle null-mask shape mismatch")
            profiles = np.asarray(archive[f"{prefix}_slow_x_profiles"], dtype=np.float64)
            if profiles.shape != (task.sample_count, 16, 20):
                raise ValueError(f"{prefix} slow x-profile shape mismatch")
            if not np.all(np.isfinite(profiles)) or np.any(profiles < -1e-14):
                raise FloatingPointError(f"{prefix} slow x profiles are invalid")
            slow_rates = np.asarray(archive[f"{prefix}_slow_pair_rates"], dtype=np.float64)
            if slow_rates.shape != (task.sample_count, 16) or not np.all(
                np.isfinite(slow_rates)
            ):
                raise FloatingPointError(f"{prefix} slow tangent rates are invalid")
            for side in ("occupied", "empty"):
                for direction in ("input", "output"):
                    modes = np.asarray(
                        archive[f"{prefix}_slow_{direction}_{side}"],
                        dtype=np.complex128,
                    )
                    if modes.shape != (task.sample_count, dimension, 16):
                        raise ValueError(
                            f"{prefix} slow {direction} {side} shape mismatch"
                        )
                    if not np.all(np.isfinite(modes)):
                        raise FloatingPointError(
                            f"{prefix} slow {direction} {side} is non-finite"
                        )

        for field, tolerance_field in (
            (
                "endpoint_projector_relative_frobenius_error_full",
                "endpoint_projector_relative_frobenius_tolerance",
            ),
            (
                "endpoint_projector_relative_frobenius_error_late",
                "endpoint_projector_relative_frobenius_tolerance",
            ),
            (
                "cross_replay_projector_relative_frobenius_error",
                "cross_replay_projector_relative_frobenius_tolerance",
            ),
            (
                "replay_cycle_log_probability_max_abs_error",
                "replay_probability_tolerance",
            ),
        ):
            values = np.asarray(archive[field], dtype=np.float64)
            tolerance = float(archive[tolerance_field].item())
            if values.shape != (task.sample_count,) or np.any(~np.isfinite(values)):
                raise FloatingPointError(f"invalid replay diagnostic: {field}")
            if np.any(values < 0.0) or np.any(values > tolerance):
                raise FloatingPointError(f"replay gate failed: {field}")

        for field in (
            "choi_covariance_constructed",
            "covariance_history_constructed",
            "intermediate_frames_saved",
            "per_cycle_dense_jacobians_saved",
        ):
            if bool(archive[field].item()):
                raise ValueError(f"forbidden product flag is true: {field}")


def verified_complete(
    output_root: Path,
    task: ReplayTask,
    *,
    config_sha256: str,
    hashes: Mapping[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = result_paths(output_root, task)
    if not result_path.exists() and not completion_path.exists():
        return False, "missing"
    if not result_path.is_file() or not completion_path.is_file():
        return False, "incomplete result/completion pair"
    try:
        completion = load_json(completion_path)
        for key, expected in task_identity(
            task, config_sha256=config_sha256, hashes=hashes
        ).items():
            if completion.get(key) != expected:
                return False, f"completion identity mismatch: {key}"
        if completion.get("result_filename") != result_path.name:
            return False, "result filename mismatch"
        if int(completion.get("result_bytes", -1)) != result_path.stat().st_size:
            return False, "result byte-count mismatch"
        if completion.get("result_sha256") != sha256_file(result_path):
            return False, "result checksum mismatch"
        validate_result_npz(
            result_path,
            task,
            config_sha256=config_sha256,
            hashes=hashes,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        FloatingPointError,
        json.JSONDecodeError,
    ) as exc:
        return False, f"invalid output: {exc}"
    return True, "verified"


def verify_acquisition_sources(
    acquisition_root: Path,
    sources: Sequence[acquisition.Task],
    *,
    verify_checksums: bool,
) -> None:
    for source in tqdm(sources, desc="Verify acquisition inputs", unit="batch"):
        result_path, completion_path = acquisition_paths(acquisition_root, source)
        if not result_path.is_file() or not completion_path.is_file():
            raise FileNotFoundError(f"missing acquisition pair for {source.task_id}")
        completion = load_json(completion_path)
        for key, expected in acquisition._completion_identity(
            task=source,
            configuration_sha256=acquisition.config_sha256(acquisition.expected_config()),
            hashes=acquisition.source_hashes(),
        ).items():
            if completion.get(key) != expected:
                raise ValueError(f"acquisition completion mismatch {source.task_id}: {key}")
        if completion.get("result_filename") != result_path.name:
            raise ValueError(f"acquisition filename mismatch: {source.task_id}")
        if int(completion.get("result_bytes", -1)) != result_path.stat().st_size:
            raise ValueError(f"acquisition byte-count mismatch: {source.task_id}")
        if verify_checksums and completion.get("result_sha256") != sha256_file(result_path):
            raise ValueError(f"acquisition checksum mismatch: {source.task_id}")


def acquisition_paths(
    acquisition_root: Path, source: acquisition.Task
) -> tuple[Path, Path]:
    """Resolve either the live Drive layout or the verified repo import layout."""

    canonical = acquisition.result_paths(acquisition_root, source)
    if canonical[0].is_file() or canonical[1].is_file():
        return canonical
    flattened = tuple(
        acquisition_root / path.relative_to(acquisition_root / "results")
        for path in canonical
    )
    if flattened[0].is_file() or flattened[1].is_file():
        return flattened
    return canonical


def stage_acquisition_source(
    acquisition_root: Path,
    source: acquisition.Task,
    scratch_root: Path,
) -> tuple[Path, Mapping[str, Any]]:
    result_path, completion_path = acquisition_paths(acquisition_root, source)
    completion = load_json(completion_path)
    cache = scratch_root / "input_cache"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{source.task_id}.npz"
    expected_bytes = int(completion["result_bytes"])
    expected_sha = str(completion["result_sha256"])
    if local.is_file() and local.stat().st_size == expected_bytes:
        if sha256_file(local) == expected_sha:
            print(f"[input cache] {source.task_id}", flush=True)
            return local, completion
        local.unlink()
    temporary = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    print(f"[input stage] {source.task_id}: {expected_bytes / 1024**3:.2f} GiB", flush=True)
    try:
        shutil.copyfile(result_path, temporary)
        if temporary.stat().st_size != expected_bytes or sha256_file(temporary) != expected_sha:
            raise OSError(f"staged acquisition checksum mismatch: {source.task_id}")
        os.replace(temporary, local)
    finally:
        temporary.unlink(missing_ok=True)
    return local, completion


def load_task_arrays(local_source: Path, task: ReplayTask) -> dict[str, np.ndarray]:
    rows = slice(task.source_row_start, task.source_row_stop)
    with np.load(local_source, allow_pickle=False) as archive:
        initial_frame = np.asarray(archive["initial_frame"][rows], dtype=np.complex128)
        initial_ranks = np.asarray(archive["initial_ranks"][rows], dtype=np.int64)
        final_frame = np.asarray(archive["final_frame"][rows], dtype=np.complex128)
        final_ranks = np.asarray(archive["final_ranks"][rows], dtype=np.int64)
        schedule = np.asarray(archive["record_schedule"][rows], dtype=np.int32)
        packed = np.asarray(archive["record_outcomes_packed"][rows], dtype=np.uint8)
        source_shape = np.asarray(archive["record_outcomes_shape"], dtype=np.int64)
        measurement_log_probability = np.asarray(
            archive["measurement_log_probability"][rows], dtype=np.float64
        )
    logical_shape = np.asarray(
        (task.sample_count, *tuple(int(v) for v in source_shape[1:])), dtype=np.int64
    )
    outcomes = unpack_boolean_record(packed, logical_shape)
    if not np.all(initial_ranks == initial_ranks[0]):
        raise ValueError(f"prepared ranks differ within {task.task_id}")
    return {
        "initial_frame": initial_frame,
        "initial_ranks": initial_ranks,
        "final_frame": final_frame,
        "final_ranks": final_ranks,
        "schedule": schedule,
        "outcomes": outcomes,
        "measurement_log_probability": measurement_log_probability,
    }


def build_model(task: ReplayTask, *, device: str = "cuda:0") -> classA_U1FGTN_gpu:
    hard = task.construction == "hard"
    model = classA_U1FGTN_gpu(
        Nx=20,
        Ny=task.ny,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=task.alpha_1,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=hard,
        triv_region_local_mode=False,
        device=device,
        dtype="complex128",
        backend="local",
    )
    if tuple(int(value) for value in model.DW_loc) != (5, 15):
        raise RuntimeError(f"unexpected domain-wall positions {model.DW_loc}")
    if model.dtype != torch.complex128:
        raise RuntimeError("GPU tangent replay model is not complex128")
    return model


def occupied_empty_basis(
    model: classA_U1FGTN_gpu,
    frames: Any,
    ranks: Any,
    *,
    hard: bool,
    purity_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame = torch.as_tensor(frames, dtype=torch.complex128, device=model.device)
    rank_values = torch.as_tensor(ranks, dtype=torch.int64, device=model.device)
    if rank_values.shape != (frame.shape[0],):
        raise ValueError("occupied-frame ranks do not match the GPU batch")
    if bool(torch.any(rank_values < 0).item()) or bool(
        torch.any(rank_values > frame.shape[-1]).item()
    ):
        raise ValueError("occupied-frame rank is outside the saved capacity")
    active = model.active_top_layer_indices(meas_slab_only=hard).to(model.device)
    column_mask = (
        torch.arange(frame.shape[-1], device=model.device)[None, :]
        < rank_values[:, None]
    )
    occupied = (frame * column_mask[:, None, :]).index_select(1, active)
    correlation = occupied @ occupied.mH
    correlation = 0.5 * (correlation + correlation.mH)
    occupations, eigenvectors = torch.linalg.eigh(correlation)
    defects = torch.minimum(occupations.abs(), (1.0 - occupations).abs()).amax(dim=1)
    if bool(torch.any(~torch.isfinite(defects)).item()) or float(defects.max().item()) > float(
        purity_tolerance
    ):
        raise FloatingPointError(
            f"active prepared state purity defect {float(defects.max().item()):.3e}"
        )
    occupied_masks = occupations > 0.5
    occupied_counts = torch.count_nonzero(occupied_masks, dim=1)
    empty_counts = occupations.shape[1] - occupied_counts
    block_sizes = torch.stack((occupied_counts, empty_counts), dim=1).to(torch.int64)
    ordered_bases = []
    for row in range(frame.shape[0]):
        mask = occupied_masks[row]
        ordered_bases.append(
            torch.cat(
                (eigenvectors[row, :, mask], eigenvectors[row, :, ~mask]), dim=1
            )
        )
    basis = torch.stack(ordered_bases, dim=0)
    target = torch.eye(basis.shape[-1], dtype=basis.dtype, device=basis.device)
    gram_error = torch.abs(basis.mH @ basis - target).amax()
    if not bool(torch.isfinite(gram_error).item()) or float(gram_error.item()) > 1e-9:
        raise FloatingPointError(f"tangent basis Gram error {float(gram_error.item()):.3e}")
    return basis, active, block_sizes, occupations, defects


class ProbabilityAccumulator:
    def __init__(self, samples: int, cycles: int, device: torch.device) -> None:
        self.values = torch.zeros(
            (samples, cycles + 1), dtype=torch.float64, device=device
        )

    def __call__(
        self,
        *,
        cycle: int,
        sample_offsets: torch.Tensor,
        conditional_log_probability: torch.Tensor,
        **_: Any,
    ) -> None:
        rows = sample_offsets.to(dtype=torch.long, device=self.values.device)
        self.values[rows, int(cycle)] += conditional_log_probability.to(
            torch.float64
        ).sum(dim=1)


class MidpointCapture:
    def __init__(self, target_cycle: int) -> None:
        self.target_cycle = int(target_cycle)
        self.frame: torch.Tensor | None = None
        self.ranks: torch.Tensor | None = None

    def __call__(self, *, cycle: int, state: Any, **_: Any) -> None:
        if int(cycle) != self.target_cycle:
            return
        self.frame = state.frame.detach().clone()
        self.ranks = state.ranks.detach().clone()


class CanonicalTangentCapture:
    def __init__(self, physical_cycles: int, start_cycle: int) -> None:
        self.physical_cycles = int(physical_cycles)
        self.start_cycle = int(start_cycle)
        self.log_diag: list[np.ndarray] = []
        self.rates: list[np.ndarray] = []
        self.null_masks: list[np.ndarray] = []
        self.null_counts: list[np.ndarray] = []
        self.min_probability: list[np.ndarray] = []
        self.min_denominator: list[np.ndarray] = []
        self.invalid_count: list[np.ndarray] = []
        self.core_scales: list[np.ndarray] = []
        self.final_frame: torch.Tensor | None = None
        self.final_core: torch.Tensor | None = None
        self.final_core_scale: torch.Tensor | None = None
        self.final_core_null_count: torch.Tensor | None = None
        self.final_physical_frame: torch.Tensor | None = None
        self.final_physical_ranks: torch.Tensor | None = None

    def __call__(
        self,
        *,
        cycle: int,
        spectra: torch.Tensor,
        G: Any,
        lyapunov_frame: torch.Tensor,
        lyapunov_log_diag: torch.Tensor,
        lyapunov_cycle_null_mask: torch.Tensor,
        lyapunov_null_counts: torch.Tensor,
        lyapunov_min_branch_probability: torch.Tensor,
        lyapunov_min_abs_born_denominator: torch.Tensor,
        lyapunov_invalid_branch_count: torch.Tensor,
        lyapunov_core_hat: torch.Tensor,
        lyapunov_core_log_scale: torch.Tensor,
        lyapunov_core_null_count: torch.Tensor,
        **_: Any,
    ) -> None:
        def cpu(value: torch.Tensor, dtype: Any) -> np.ndarray:
            return np.asarray(value.detach().cpu().numpy(), dtype=dtype)

        self.log_diag.append(cpu(lyapunov_log_diag, np.float64))
        self.rates.append(cpu(spectra, np.float64))
        self.null_masks.append(cpu(lyapunov_cycle_null_mask, np.bool_))
        self.null_counts.append(cpu(lyapunov_null_counts, np.int64))
        self.min_probability.append(cpu(lyapunov_min_branch_probability, np.float64))
        self.min_denominator.append(cpu(lyapunov_min_abs_born_denominator, np.float64))
        self.invalid_count.append(cpu(lyapunov_invalid_branch_count, np.int64))
        self.core_scales.append(cpu(lyapunov_core_log_scale, np.float64))
        if int(cycle) == self.physical_cycles:
            self.final_frame = lyapunov_frame.detach().clone()
            self.final_core = lyapunov_core_hat.detach().clone()
            self.final_core_scale = lyapunov_core_log_scale.detach().clone()
            self.final_core_null_count = lyapunov_core_null_count.detach().clone()
            self.final_physical_frame = G.frame.detach().clone()
            self.final_physical_ranks = G.ranks.detach().clone()

    def finalize(
        self,
        *,
        prefix: str,
        initial_basis: torch.Tensor,
        active_indices: torch.Tensor,
        block_sizes: torch.Tensor | np.ndarray,
        nx: int,
        ny: int,
        slow_mode_count: int,
        singular_tolerance: float,
        materialize_cocycle: bool,
    ) -> dict[str, np.ndarray]:
        expected = self.physical_cycles - self.start_cycle + 1
        if self.final_frame is None or len(self.log_diag) != expected:
            raise RuntimeError(f"{prefix} GPU tangent capture is incomplete")
        assert self.final_core is not None
        assert self.final_core_scale is not None
        assert self.final_core_null_count is not None
        frame = self.final_frame
        image = frame @ self.final_core
        batch, full_dimension, active_dimension = image.shape
        sizes = torch.as_tensor(block_sizes, dtype=torch.int64, device=frame.device)
        if sizes.ndim == 1:
            sizes = sizes.unsqueeze(0).expand(batch, -1)
        if tuple(sizes.shape) != (batch, 2) or bool(
            torch.any(sizes.sum(dim=1) != active_dimension).item()
        ):
            raise RuntimeError("per-sample occupied/empty blocks do not span the active basis")
        scale = self.final_core_scale
        x_coordinates = torch.as_tensor(
            np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny),
            dtype=torch.long,
            device=frame.device,
        )
        if int(x_coordinates.numel()) != full_dimension:
            raise RuntimeError("x-coordinate convention does not match the one-particle space")

        occupied_logs = torch.full(
            (batch, active_dimension), -torch.inf, dtype=torch.float64, device=frame.device
        )
        empty_logs = torch.full_like(occupied_logs, -torch.inf)
        occupied_valid = torch.zeros_like(occupied_logs, dtype=torch.bool)
        empty_valid = torch.zeros_like(empty_logs, dtype=torch.bool)
        pair_indices_rows = []
        pair_rate_rows = []
        occupied_output_rows = []
        empty_output_rows = []
        occupied_input_rows = []
        empty_input_rows = []
        profile_rows = []
        for row in range(batch):
            occupied_dim, empty_dim = (int(value) for value in sizes[row].tolist())
            if min(occupied_dim, empty_dim) <= 0:
                raise FloatingPointError("a tangent occupied/empty block is empty")
            parts: list[dict[str, torch.Tensor]] = []
            start = 0
            for block_index, size in enumerate((occupied_dim, empty_dim)):
                stop = start + size
                block_image = image[row, :, start:stop]
                left, singular, right_h = torch.linalg.svd(
                    block_image, full_matrices=False
                )
                logs = torch.full_like(singular, -torch.inf, dtype=torch.float64)
                finite = torch.isfinite(singular) & (singular > singular_tolerance)
                logs[finite] = torch.log(singular[finite].to(torch.float64))
                logs += scale[row]
                input_active = initial_basis[row, :, start:stop] @ right_h.mH
                input_full = torch.zeros(
                    (full_dimension, size),
                    dtype=torch.complex128,
                    device=frame.device,
                )
                input_full[active_indices, :] = input_active
                parts.append({"logs": logs, "output": left, "input": input_full})
                target_logs = occupied_logs if block_index == 0 else empty_logs
                target_valid = occupied_valid if block_index == 0 else empty_valid
                target_logs[row, :size] = logs
                target_valid[row, :size] = True
                start = stop

            pair_rates = (
                parts[0]["logs"][:, None] + parts[1]["logs"][None, :]
            ) / float(expected)
            finite_rates = torch.where(
                torch.isfinite(pair_rates),
                torch.abs(pair_rates),
                torch.full_like(pair_rates, torch.inf),
            )
            values, flat = torch.topk(
                finite_rates.reshape(-1),
                k=slow_mode_count,
                largest=False,
                sorted=True,
            )
            if bool(torch.any(~torch.isfinite(values)).item()):
                raise FloatingPointError("insufficient finite particle-hole tangent modes")
            occupied_index = torch.div(flat, empty_dim, rounding_mode="floor")
            empty_index = flat % empty_dim
            selected_rates = pair_rates.reshape(-1).index_select(0, flat)
            occupied_output = parts[0]["output"].index_select(1, occupied_index)
            empty_output = parts[1]["output"].index_select(1, empty_index)
            occupied_input = parts[0]["input"].index_select(1, occupied_index)
            empty_input = parts[1]["input"].index_select(1, empty_index)
            density = 0.5 * (
                torch.abs(occupied_output) ** 2 + torch.abs(empty_output) ** 2
            )
            profiles = torch.stack(
                [density[x_coordinates == x, :].sum(dim=0) for x in range(nx)],
                dim=1,
            )
            pair_indices_rows.append(torch.stack((occupied_index, empty_index), dim=1))
            pair_rate_rows.append(selected_rates)
            occupied_output_rows.append(occupied_output)
            empty_output_rows.append(empty_output)
            occupied_input_rows.append(occupied_input)
            empty_input_rows.append(empty_input)
            profile_rows.append(profiles)

        pair_indices = torch.stack(pair_indices_rows, dim=0)
        selected_rates = torch.stack(pair_rate_rows, dim=0)
        occupied_output = torch.stack(occupied_output_rows, dim=0)
        empty_output = torch.stack(empty_output_rows, dim=0)
        occupied_input = torch.stack(occupied_input_rows, dim=0)
        empty_input = torch.stack(empty_input_rows, dim=0)
        profiles = torch.stack(profile_rows, dim=0)

        log_diag = np.stack(self.log_diag, axis=1)
        increments = np.diff(
            np.concatenate(
                (np.zeros((batch, 1, active_dimension), dtype=np.float64), log_diag), axis=1
            ),
            axis=1,
        )
        arrays: dict[str, np.ndarray] = {
            f"{prefix}_window_start_cycle": np.asarray(self.start_cycle, dtype=np.int64),
            f"{prefix}_window_cycle_count": np.asarray(expected, dtype=np.int64),
            f"{prefix}_qr_cumulative_log_diag": log_diag,
            f"{prefix}_qr_log_increments": increments,
            f"{prefix}_qr_finite_time_rates": np.stack(self.rates, axis=1),
            f"{prefix}_cycle_null_mask": np.stack(self.null_masks, axis=1),
            f"{prefix}_cumulative_null_count": np.stack(self.null_counts, axis=1),
            f"{prefix}_cumulative_min_branch_probability": np.stack(
                self.min_probability, axis=1
            ),
            f"{prefix}_cumulative_min_abs_born_denominator": np.stack(
                self.min_denominator, axis=1
            ),
            f"{prefix}_cumulative_invalid_branch_count": np.stack(
                self.invalid_count, axis=1
            ),
            f"{prefix}_core_log_scale_by_cycle": np.stack(self.core_scales, axis=1),
            f"{prefix}_block_sizes": np.asarray(
                sizes.detach().cpu().numpy(), dtype=np.int64
            ),
            f"{prefix}_core_null_count": np.asarray(
                self.final_core_null_count.detach().cpu().numpy(), dtype=np.int64
            ),
            f"{prefix}_one_leg_logs_occupied": np.asarray(
                occupied_logs.detach().cpu().numpy(), dtype=np.float64
            ),
            f"{prefix}_one_leg_logs_empty": np.asarray(
                empty_logs.detach().cpu().numpy(), dtype=np.float64
            ),
            f"{prefix}_one_leg_occupied_valid": np.asarray(
                occupied_valid.detach().cpu().numpy(), dtype=np.bool_
            ),
            f"{prefix}_one_leg_empty_valid": np.asarray(
                empty_valid.detach().cpu().numpy(), dtype=np.bool_
            ),
            f"{prefix}_slow_pair_indices": np.asarray(
                pair_indices.detach().cpu().numpy(), dtype=np.int32
            ),
            f"{prefix}_slow_pair_rates": np.asarray(
                selected_rates.detach().cpu().numpy(), dtype=np.float64
            ),
            f"{prefix}_slow_effective_gaps_per_cycle": np.asarray(
                (-2.0 * selected_rates).detach().cpu().numpy(), dtype=np.float64
            ),
            f"{prefix}_slow_output_occupied": np.asarray(
                occupied_output.detach().cpu().numpy(), dtype=np.complex128
            ),
            f"{prefix}_slow_output_empty": np.asarray(
                empty_output.detach().cpu().numpy(), dtype=np.complex128
            ),
            f"{prefix}_slow_input_occupied": np.asarray(
                occupied_input.detach().cpu().numpy(), dtype=np.complex128
            ),
            f"{prefix}_slow_input_empty": np.asarray(
                empty_input.detach().cpu().numpy(), dtype=np.complex128
            ),
            f"{prefix}_slow_x_profiles": np.asarray(
                profiles.detach().cpu().numpy(), dtype=np.float64
            ),
        }
        if materialize_cocycle:
            product_active = image @ initial_basis.mH
            product = torch.zeros(
                (batch, full_dimension, full_dimension),
                dtype=torch.complex128,
                device=frame.device,
            )
            product[:, :, active_indices] = product_active
            norms = torch.linalg.matrix_norm(product, ord="fro", dim=(-2, -1)).to(
                torch.float64
            )
            if bool(torch.any(~torch.isfinite(norms)).item()) or bool(
                torch.any(norms <= torch.finfo(torch.float64).tiny).item()
            ):
                raise FloatingPointError("final cocycle normalization failed")
            arrays["final_cocycle_hat"] = np.asarray(
                (product / norms[:, None, None]).detach().cpu().numpy(),
                dtype=np.complex128,
            )
            arrays["final_cocycle_log_scale"] = np.asarray(
                (scale + torch.log(norms)).detach().cpu().numpy(), dtype=np.float64
            )
        return arrays


def projector_errors(
    actual_frames: torch.Tensor,
    actual_ranks: torch.Tensor,
    reference_frames: np.ndarray,
    reference_ranks: np.ndarray,
) -> np.ndarray:
    reference = torch.as_tensor(
        reference_frames, dtype=torch.complex128, device=actual_frames.device
    )
    ranks = torch.as_tensor(reference_ranks, dtype=torch.int64, device=actual_frames.device)
    if not bool(torch.all(actual_ranks == ranks).item()):
        raise RuntimeError("replayed endpoint ranks differ from acquisition")
    values = []
    for row in range(actual_frames.shape[0]):
        rank = int(ranks[row].item())
        first = actual_frames[row, :, :rank]
        second = reference[row, :, :rank]
        overlap = first.mH @ second
        squared = 2.0 * rank - 2.0 * torch.linalg.matrix_norm(overlap, ord="fro") ** 2
        values.append(torch.sqrt(torch.clamp(squared.real / max(rank, 1), min=0.0)))
    return np.asarray(torch.stack(values).detach().cpu().numpy(), dtype=np.float64)


def cross_projector_errors(
    first_frames: torch.Tensor,
    first_ranks: torch.Tensor,
    second_frames: torch.Tensor,
    second_ranks: torch.Tensor,
) -> np.ndarray:
    if not bool(torch.all(first_ranks == second_ranks).item()):
        raise RuntimeError("full and late replay ranks differ")
    values = []
    for row in range(first_frames.shape[0]):
        rank = int(first_ranks[row].item())
        overlap = first_frames[row, :, :rank].mH @ second_frames[row, :, :rank]
        squared = 2.0 * rank - 2.0 * torch.linalg.matrix_norm(overlap, ord="fro") ** 2
        values.append(torch.sqrt(torch.clamp(squared.real / max(rank, 1), min=0.0)))
    return np.asarray(torch.stack(values).detach().cpu().numpy(), dtype=np.float64)


def run_window(
    model: classA_U1FGTN_gpu,
    task: ReplayTask,
    loaded: Mapping[str, np.ndarray],
    *,
    start_cycle: int,
    initial_basis: torch.Tensor,
    active_indices: torch.Tensor,
    block_sizes: torch.Tensor | np.ndarray,
    config: Mapping[str, Any],
    stage: str,
    capture_midpoint: bool,
) -> tuple[dict[str, np.ndarray], CanonicalTangentCapture, MidpointCapture | None, np.ndarray]:
    capture = CanonicalTangentCapture(task.cycles, start_cycle)
    midpoint = MidpointCapture(task.ny) if capture_midpoint else None
    probability = ProbabilityAccumulator(task.sample_count, task.cycles, model.device)
    hard = task.construction == "hard"
    print(
        f"[{stage}] {task.task_id}: samples={task.sample_count}, "
        f"cycles={task.cycles}, tangent_start={start_cycle}, active_dim={initial_basis.shape[-1]}",
        flush=True,
    )
    result = model.run_markov_circuit(
        G_history=False,
        progress=True,
        cycles=task.cycles,
        samples=task.sample_count,
        parallelize_samples=False,
        frame_init=loaded["initial_frame"],
        frame_ranks=loaded["initial_ranks"],
        frame_init_prepared=hard,
        initial_purity_tolerance=float(config["initial_purity_tolerance"]),
        save=False,
        return_data=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=hard,
        batch_size=task.sample_count,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        physical_covariance_update="rank1",
        state_representation="physical_frame",
        frozen_schedule=loaded["schedule"],
        frozen_outcomes=loaded["outcomes"],
        record_observer=probability,
        return_native_state=False,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
        native_cycle_observer=None if midpoint is None else midpoint,
        native_cycle_observer_cycles=None if midpoint is None else [task.ny],
        lyapunov_frame_observer=capture,
        lyapunov_initial_frame=initial_basis,
        lyapunov_nvec=initial_basis.shape[-1],
        lyapunov_basis_mode="canonical",
        lyapunov_start_cycle=start_cycle,
        lyapunov_track_restricted_core=True,
        lyapunov_singular_tol=float(config["singular_tolerance"]),
        lyapunov_failure_mode="raise",
    )
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("GPU tangent replay materialized a covariance")
    replay_probability = np.asarray(
        probability.values.detach().cpu().numpy(), dtype=np.float64
    )
    error = float(
        np.max(np.abs(replay_probability - loaded["measurement_log_probability"]))
    )
    if error > float(config["replay_probability_tolerance"]):
        raise FloatingPointError(f"{stage} replay log-probability error {error:.3e}")
    arrays = capture.finalize(
        prefix=stage,
        initial_basis=initial_basis,
        active_indices=active_indices,
        block_sizes=block_sizes,
        nx=20,
        ny=task.ny,
        slow_mode_count=int(config["slow_mode_count"]),
        singular_tolerance=float(config["singular_tolerance"]),
        materialize_cocycle=stage == "full",
    )
    return arrays, capture, midpoint, replay_probability


def run_task(
    task: ReplayTask,
    loaded: Mapping[str, np.ndarray],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    hashes: Mapping[str, str],
    acquisition_completion: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model = build_model(task)
    hard = task.construction == "hard"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(model.device)
    started = time.monotonic()
    print(f"[basis] {task.task_id}: diagonalizing {task.sample_count} prepared states", flush=True)
    full_basis, active, full_blocks, full_occupations, full_defects = occupied_empty_basis(
        model,
        loaded["initial_frame"],
        loaded["initial_ranks"],
        hard=hard,
        purity_tolerance=float(config["initial_purity_tolerance"]),
    )
    full_occupations_cpu = np.asarray(
        full_occupations.detach().cpu().numpy(), dtype=np.float64
    )
    full_defects_cpu = np.asarray(full_defects.detach().cpu().numpy(), dtype=np.float64)
    windows = tqdm(
        total=2, desc=f"{task.task_id} windows", unit="replay", leave=True
    )
    full_arrays, full_capture, midpoint, full_probability = run_window(
        model,
        task,
        loaded,
        start_cycle=1,
        initial_basis=full_basis,
        active_indices=active,
        block_sizes=full_blocks,
        config=config,
        stage="full",
        capture_midpoint=True,
    )
    windows.update(1)
    if midpoint is None or midpoint.frame is None or midpoint.ranks is None:
        raise RuntimeError("full replay did not capture the late-window initial state")
    assert full_capture.final_physical_frame is not None
    assert full_capture.final_physical_ranks is not None
    full_endpoint_frame = np.asarray(
        full_capture.final_physical_frame.detach().cpu().numpy(), dtype=np.complex128
    )
    full_endpoint_ranks = np.asarray(
        full_capture.final_physical_ranks.detach().cpu().numpy(), dtype=np.int64
    )
    full_errors = projector_errors(
        full_capture.final_physical_frame,
        full_capture.final_physical_ranks,
        loaded["final_frame"],
        loaded["final_ranks"],
    )
    late_basis, late_active, late_blocks, late_occupations, late_defects = occupied_empty_basis(
        model,
        midpoint.frame,
        midpoint.ranks,
        hard=hard,
        purity_tolerance=float(config["initial_purity_tolerance"]),
    )
    if not torch.equal(active, late_active):
        raise RuntimeError("full and late tangent spaces use different active indices")
    late_occupations_cpu = np.asarray(
        late_occupations.detach().cpu().numpy(), dtype=np.float64
    )
    late_defects_cpu = np.asarray(late_defects.detach().cpu().numpy(), dtype=np.float64)
    del (
        full_basis,
        full_occupations,
        full_defects,
        full_capture,
        midpoint,
        late_occupations,
        late_defects,
    )
    torch.cuda.empty_cache()
    late_arrays, late_capture, _unused_midpoint, late_probability = run_window(
        model,
        task,
        loaded,
        start_cycle=task.ny + 1,
        initial_basis=late_basis,
        active_indices=late_active,
        block_sizes=late_blocks,
        config=config,
        stage="late",
        capture_midpoint=False,
    )
    windows.update(1)
    windows.close()
    assert late_capture.final_physical_frame is not None
    assert late_capture.final_physical_ranks is not None
    late_errors = projector_errors(
        late_capture.final_physical_frame,
        late_capture.final_physical_ranks,
        loaded["final_frame"],
        loaded["final_ranks"],
    )
    cross_errors = projector_errors(
        late_capture.final_physical_frame,
        late_capture.final_physical_ranks,
        full_endpoint_frame,
        full_endpoint_ranks,
    )
    endpoint_tolerance = float(config["endpoint_projector_relative_frobenius_tolerance"])
    cross_tolerance = float(
        config["cross_replay_projector_relative_frobenius_tolerance"]
    )
    if max(float(full_errors.max()), float(late_errors.max())) > endpoint_tolerance:
        raise FloatingPointError("GPU replay endpoint tolerance exceeded")
    if float(cross_errors.max()) > cross_tolerance:
        raise FloatingPointError("full/late GPU replay consistency tolerance exceeded")
    replay_probability_error = np.maximum(
        np.max(np.abs(full_probability - loaded["measurement_log_probability"]), axis=1),
        np.max(np.abs(late_probability - loaded["measurement_log_probability"]), axis=1),
    )
    elapsed = time.monotonic() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(model.device))
    peak_reserved = int(torch.cuda.max_memory_reserved(model.device))
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(RESULT_SCHEMA),
        "sampling_revision": np.asarray(REPLAY_REVISION),
        "acquisition_revision": np.asarray(ACQUISITION_REVISION),
        "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
        "task_id": np.asarray(task.task_id),
        "construction": np.asarray(task.construction),
        "Nx": np.asarray(20, dtype=np.int64),
        "Ny": np.asarray(task.ny, dtype=np.int64),
        "alpha_1": np.asarray(task.alpha_1, dtype=np.float64),
        "alpha_2": np.asarray(30.0, dtype=np.float64),
        "nshell": np.asarray(1, dtype=np.int64),
        "cycles": np.asarray(task.cycles, dtype=np.int64),
        "sample_count": np.asarray(task.sample_count, dtype=np.int64),
        "case_sample_indices": np.asarray(task.case_sample_indices, dtype=np.int64),
        "global_sample_indices": np.asarray(task.global_sample_indices, dtype=np.int64),
        "configuration_sha256": np.asarray(config_sha256),
        "source_hashes_json": np.asarray(canonical_json(hashes)),
        "acquisition_task_id": np.asarray(task.source.task_id),
        "acquisition_batch_seed": np.asarray(task.source.seed, dtype=np.int64),
        "acquisition_result_sha256": np.asarray(acquisition_completion["result_sha256"]),
        "acquisition_configuration_sha256": np.asarray(
            acquisition_completion["configuration_sha256"]
        ),
        "active_input_indices": np.asarray(active.detach().cpu().numpy(), dtype=np.int64),
        "active_dimension": np.asarray(int(active.numel()), dtype=np.int64),
        "full_initial_active_occupations": np.asarray(
            full_occupations_cpu, dtype=np.float64
        ),
        "late_initial_active_occupations": np.asarray(
            late_occupations_cpu, dtype=np.float64
        ),
        "full_initial_active_purity_defect": np.asarray(
            full_defects_cpu, dtype=np.float64
        ),
        "late_initial_active_purity_defect": np.asarray(
            late_defects_cpu, dtype=np.float64
        ),
        "endpoint_projector_relative_frobenius_error_full": full_errors,
        "endpoint_projector_relative_frobenius_error_late": late_errors,
        "cross_replay_projector_relative_frobenius_error": cross_errors,
        "replay_cycle_log_probability_max_abs_error": replay_probability_error,
        "endpoint_projector_relative_frobenius_tolerance": np.asarray(
            endpoint_tolerance, dtype=np.float64
        ),
        "cross_replay_projector_relative_frobenius_tolerance": np.asarray(
            cross_tolerance, dtype=np.float64
        ),
        "replay_probability_tolerance": np.asarray(
            config["replay_probability_tolerance"], dtype=np.float64
        ),
        "endpoint_replay_verified": np.ones(task.sample_count, dtype=np.bool_),
        "dtype": np.asarray("complex128"),
        "tangent_batch_size": np.asarray(task.sample_count, dtype=np.int64),
        "tangent_stabilization": np.asarray(
            "batched_full_core_qr_from_occupied_empty_initial_basis"
        ),
        "covariance_action_convention": np.asarray("J_T:0[H]=K_T:0^dagger H K_T:0"),
        "choi_covariance_constructed": np.asarray(False, dtype=np.bool_),
        "covariance_history_constructed": np.asarray(False, dtype=np.bool_),
        "intermediate_frames_saved": np.asarray(False, dtype=np.bool_),
        "per_cycle_dense_jacobians_saved": np.asarray(False, dtype=np.bool_),
        **full_arrays,
        **late_arrays,
    }
    performance = {
        "elapsed_seconds": float(elapsed),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }
    print(
        f"[task computed] {task.task_id}: elapsed={elapsed / 60:.2f} min, "
        f"peak_reserved={peak_reserved / 1024**3:.2f} GiB",
        flush=True,
    )
    return arrays, performance


def estimated_result_bytes(task: ReplayTask) -> int:
    dimension = 40 * task.ny
    cocycle = task.sample_count * dimension * dimension * 16
    diagnostics = task.sample_count * task.cycles * dimension * 32
    modes = task.sample_count * 16 * dimension * 16 * 8
    return int(cocycle + diagnostics + modes + 256 * 1024**2)


def require_storage(task: ReplayTask, scratch_root: Path, output_root: Path) -> None:
    estimate = estimated_result_bytes(task)
    local_free = shutil.disk_usage(scratch_root if scratch_root.exists() else scratch_root.parent).free
    drive_parent = output_root
    while not drive_parent.exists() and drive_parent != drive_parent.parent:
        drive_parent = drive_parent.parent
    drive_free = shutil.disk_usage(drive_parent).free
    if local_free < estimate + 4 * 1024**3:
        raise OSError("insufficient local scratch headroom for GPU tangent task")
    if drive_free < 2 * estimate + 4 * 1024**3:
        raise OSError("insufficient Drive headroom for GPU tangent task")
    print(
        f"[storage] estimate={estimate / 1024**3:.2f} GiB, "
        f"local_free={local_free / 1024**3:.1f} GiB, drive_free={drive_free / 1024**3:.1f} GiB",
        flush=True,
    )


def save_task(
    task: ReplayTask,
    arrays: Mapping[str, Any],
    performance: Mapping[str, Any],
    *,
    output_root: Path,
    scratch_root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    hashes: Mapping[str, str],
) -> bool:
    local_dir = scratch_root / "outputs" / task.task_id
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True)
    local_result = local_dir / "result.npz"
    atomic_npz(local_result, arrays)
    validate_result_npz(
        local_result,
        task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    print(f"[validated] {task.task_id}: closed local NPZ passed all gates", flush=True)
    final_result, final_completion = result_paths(output_root, task)
    published = publish_file(local_result, final_result)
    runtime_ok = float(performance["elapsed_seconds"]) <= float(
        config["maximum_task_runtime_seconds"]
    )
    memory_ok = int(performance["peak_cuda_reserved_bytes"]) <= int(
        float(config["maximum_peak_cuda_reserved_gib"]) * 1024**3
    )
    gate_passed = bool(runtime_ok and memory_ok)
    completion = task_identity(task, config_sha256=config_sha256, hashes=hashes)
    completion.update(
        {
            "result_filename": final_result.name,
            "result_bytes": published["bytes"],
            "result_sha256": published["sha256"],
            **dict(performance),
            "performance_gate_passed": gate_passed,
            "maximum_task_runtime_seconds": float(config["maximum_task_runtime_seconds"]),
            "maximum_peak_cuda_reserved_bytes": int(
                float(config["maximum_peak_cuda_reserved_gib"]) * 1024**3
            ),
            "endpoint_projector_relative_frobenius_tolerance": float(
                config["endpoint_projector_relative_frobenius_tolerance"]
            ),
            "cross_replay_projector_relative_frobenius_tolerance": float(
                config["cross_replay_projector_relative_frobenius_tolerance"]
            ),
            "completed_utc": utc_now(),
        }
    )
    local_completion = local_dir / "completion.json"
    atomic_json(local_completion, completion)
    publish_file(local_completion, final_completion)
    shutil.rmtree(local_dir)
    print(
        f"[task durable] {task.task_id}: {published['bytes'] / 1024**3:.2f} GiB, "
        f"sha256={published['sha256'][:16]}...",
        flush=True,
    )
    return gate_passed


def require_a100(config: Mapping[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an A100 GPU runtime")
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name.upper():
        raise RuntimeError(f"production requires an A100, found {properties.name!r}")
    if int(properties.total_memory) < 38 * 1024**3:
        raise RuntimeError(
            f"production requires 40-GB-class memory, found "
            f"{properties.total_memory / 1024**3:.2f} GiB"
        )
    if config["dtype"] != "complex128":
        raise RuntimeError("production dtype must remain complex128")
    print(
        f"[device] {properties.name}, total={properties.total_memory / 1024**3:.2f} GiB, "
        "dtype=complex128",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--acquisition-root", type=Path, default=DEFAULT_ACQUISITION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=Path("/content/pure_tangent_gpu_replay"))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-tasks", type=int)
    parser.add_argument("--skip-input-checksums", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    hashes = source_hashes()
    config_sha256 = canonical_hash(config)
    tasks = expand_tasks()
    acquisition_sources = acquisition.expand_tasks(acquisition.expected_config())
    output_root = args.output_root
    scratch_root = args.scratch_root
    scratch_root.mkdir(parents=True, exist_ok=True)
    print("[configuration] " + json.dumps(config, indent=2, sort_keys=True), flush=True)
    print(f"[bundle] {BUNDLE_ROOT}", flush=True)
    print(f"[acquisition] {args.acquisition_root}", flush=True)
    print(f"[output] {output_root}", flush=True)
    print(f"[scratch] {scratch_root}", flush=True)
    print(
        f"[workload] cases=12, GPU tasks={len(tasks)}, trajectories=1200, "
        f"batch_size={SAMPLES_PER_GPU_TASK}, two tangent replays/task",
        flush=True,
    )
    print(f"[sources] {json.dumps(hashes, sort_keys=True)}", flush=True)

    inventory = [
        (
            task,
            *verified_complete(
                output_root,
                task,
                config_sha256=config_sha256,
                hashes=hashes,
            ),
        )
        for task in tasks
    ]
    complete_count = sum(complete for _, complete, _ in inventory)
    performance_blocks: list[str] = []
    for task, complete, _reason in inventory:
        if not complete:
            continue
        _result_path, completion_path = result_paths(output_root, task)
        completion = load_json(completion_path)
        if completion.get("performance_gate_passed") is not True:
            performance_blocks.append(task.task_id)
    print(
        f"[resume] completed={complete_count}, pending={len(tasks) - complete_count}, "
        f"total={len(tasks)}",
        flush=True,
    )
    for task, complete, reason in inventory:
        if not complete and reason != "missing":
            print(f"[resume warning] {task.task_id}: {reason}", flush=True)
    if args.report_only:
        return 0
    if performance_blocks:
        raise RuntimeError(
            "A completed GPU replay batch exceeded the locked runtime/memory ceiling; "
            "a new smaller-batch revision is required before continuing: "
            + ", ".join(performance_blocks)
        )
    require_a100(config)
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        raise ValueError("--max-new-tasks must be nonnegative")
    verify_acquisition_sources(
        args.acquisition_root,
        acquisition_sources,
        verify_checksums=not args.skip_input_checksums,
    )

    pending = [task for task, complete, _ in inventory if not complete]
    limit = len(pending) if args.max_new_tasks is None else int(args.max_new_tasks)
    completed_new = 0
    current_source_id: str | None = None
    current_local: Path | None = None
    source_completion: Mapping[str, Any] | None = None
    with tqdm(
        total=len(tasks),
        initial=complete_count,
        desc="A100 tangent replay",
        unit="batch",
    ) as bar:
        for task in pending:
            if completed_new >= limit:
                break
            if task.source.task_id != current_source_id:
                if current_local is not None:
                    current_local.unlink(missing_ok=True)
                current_local, source_completion = stage_acquisition_source(
                    args.acquisition_root, task.source, scratch_root
                )
                current_source_id = task.source.task_id
            assert current_local is not None and source_completion is not None
            require_storage(task, scratch_root, output_root)
            loaded = load_task_arrays(current_local, task)
            arrays, performance = run_task(
                task,
                loaded,
                config=config,
                config_sha256=config_sha256,
                hashes=hashes,
                acquisition_completion=source_completion,
            )
            gate_passed = save_task(
                task,
                arrays,
                performance,
                output_root=output_root,
                scratch_root=scratch_root,
                config=config,
                config_sha256=config_sha256,
                hashes=hashes,
            )
            del arrays, loaded
            torch.cuda.empty_cache()
            completed_new += 1
            bar.update(1)
            bar.set_postfix(
                completed=complete_count + completed_new,
                pending=len(tasks) - complete_count - completed_new,
            )
            if not gate_passed:
                raise RuntimeError(
                    f"{task.task_id} is durable but exceeded the one-hour or 36-GiB "
                    "performance gate; create a smaller-batch revision before continuing"
                )
    if current_local is not None:
        current_local.unlink(missing_ok=True)
    print(
        f"[complete] newly_completed={completed_new}, "
        f"remaining={len(tasks) - complete_count - completed_new}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
