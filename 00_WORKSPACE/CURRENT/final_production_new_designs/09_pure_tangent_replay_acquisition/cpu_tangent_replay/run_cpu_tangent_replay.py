#!/usr/bin/env python3
"""Resumable CPU tangent-cocycle replay of the completed slot-09 records."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import struct
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


CONFIG_PATH = HERE / "campaign_config.json"
ACQUISITION_REVISION = "pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1"
REPLAY_REVISION = "pure_tangent_cpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v3"
DEFAULT_ACQUISITION_ROOT = BUNDLE_ROOT / "gpu_data" / ACQUISITION_REVISION
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "cpu_data" / REPLAY_REVISION
ACQUISITION_SCHEMA = "pure_tangent_replay_acquisition_completion_v1"
RESULT_SCHEMA = "pure_tangent_cpu_replay_result_v3"
COMPLETION_SCHEMA = "pure_tangent_cpu_replay_completion_v3"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
CHANNELS = ("Ap", "Am", "Bp", "Bm")
EXPECTED_OCCUPATIONS = np.asarray((False, True, False, True), dtype=np.bool_)
EXPECTED_NY = (24, 28, 32)
EXPECTED_CONSTRUCTIONS = ("hard", "soft")
EXPECTED_ALPHA_1 = (1.0, 3.0)
EXPECTED_CASES = 12
EXPECTED_SAMPLES = 1200
SOURCE_PATHS = {
    "runner": Path(__file__).resolve(),
    "config": CONFIG_PATH,
    "cpu_engine": REPO_ROOT / "src/fgtn/classA_U1FGTN.py",
    "occupied_frame": REPO_ROOT / "src/fgtn/occupied_frame.py",
}


@dataclass(frozen=True)
class BatchSource:
    task_id: str
    construction: str
    ny: int
    alpha_1: float
    cycles: int
    batch_index: int
    batch_seed: int
    case_sample_indices: tuple[int, ...]
    global_sample_indices: tuple[int, ...]
    result_path: str
    completion_path: str
    result_bytes: int
    result_sha256: str
    completion_sha256: str
    acquisition_configuration_sha256: str
    acquisition_source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class SampleTask:
    task_id: str
    construction: str
    ny: int
    alpha_1: float
    cycles: int
    case_sample_index: int
    global_sample_index: int
    batch_row: int
    source: BatchSource


def _utc_now() -> str:
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = load_json(path)
    expected = {
        "schema": "pure_tangent_cpu_replay_config_v3",
        "sampling_revision": REPLAY_REVISION,
        "acquisition_revision": ACQUISITION_REVISION,
        "Nx": 20,
        "Ny_values": list(EXPECTED_NY),
        "constructions": list(EXPECTED_CONSTRUCTIONS),
        "alpha_1_values": list(EXPECTED_ALPHA_1),
        "alpha_2": 30.0,
        "nshell": 1,
        "samples_per_case": 100,
        "cycles_multiplier": 2,
        "trial_orbitals": "X",
        "filling_fraction": 0.5,
        "init_mode": "default",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "dtype": "complex128",
        "state_representation": "physical_frame",
        "tangent_basis_mode": "pure_occupied_empty",
        "hard_wall_active_slab_only": True,
        "late_window_start_cycle_multiplier": 1,
        "slow_mode_count": 16,
        "singular_tolerance": 1e-14,
        "replay_probability_tolerance": 1e-12,
        "endpoint_projector_relative_frobenius_tolerance": 1e-6,
        "endpoint_gate_calibration_revision": "cpu_gpu_projector_replay_pilot_20260910_v1",
        "cross_replay_projector_relative_frobenius_tolerance": 2e-6,
        "cross_replay_gate_calibration_revision": "full_late_cpu_projector_sample018_20260910_v2",
        "initial_purity_tolerance": 2e-9,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "saved_products": {
            "final_scale_separated_one_leg_cocycle": True,
            "per_cycle_qr_log_increments": True,
            "full_window_one_leg_singular_logs": True,
            "late_window_one_leg_singular_logs": True,
            "slowest_particle_hole_rates_and_x_profiles": True,
            "per_cycle_dense_jacobians": False,
            "choi_covariance": False,
            "covariance_history": False,
            "intermediate_occupied_frames": False,
        },
    }
    if value != expected:
        raise ValueError("CPU replay configuration differs from its locked v3 contract")
    return value


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}


def _npy_metadata(npz_path: Path, key: str) -> tuple[tuple[int, ...], np.dtype, bool, int]:
    member_name = f"{key}.npy"
    with zipfile.ZipFile(npz_path, "r") as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            raise ValueError(f"{member_name} is not an uncompressed, unencrypted NPY member")
        header_offset = int(info.header_offset)
    with npz_path.open("rb") as handle:
        handle.seek(header_offset)
        local = handle.read(30)
        if len(local) != 30:
            raise ValueError(f"truncated ZIP local header for {member_name}")
        signature, *_unused, filename_length, extra_length = struct.unpack(
            "<IHHHHHIIIHH", local
        )
        if signature != 0x04034B50:
            raise ValueError(f"invalid ZIP local header for {member_name}")
        handle.seek(filename_length + extra_length, io.SEEK_CUR)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
        data_offset = int(handle.tell())
    return tuple(map(int, shape)), np.dtype(dtype), bool(fortran_order), data_offset


def mmap_npy_member(npz_path: Path, key: str) -> np.memmap:
    shape, dtype, fortran, offset = _npy_metadata(npz_path, key)
    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def _expect_scalar(archive: Any, key: str, expected: Any) -> None:
    actual = archive[key].item()
    if actual != expected:
        raise ValueError(f"acquisition field {key!r} mismatch: {actual!r} != {expected!r}")


def discover_batches(acquisition_root: Path, *, verify_checksums: bool) -> list[BatchSource]:
    completions = sorted(acquisition_root.rglob("*.complete.json"))
    if len(completions) != 32:
        raise RuntimeError(f"expected 32 acquisition completion files, found {len(completions)}")
    sources: list[BatchSource] = []
    shared_acquisition_hashes: Mapping[str, str] | None = None
    for completion_path in tqdm(completions, desc="Verify acquisition batches", unit="batch"):
        completion = load_json(completion_path)
        if completion.get("schema") != ACQUISITION_SCHEMA:
            raise ValueError(f"acquisition completion schema mismatch: {completion_path}")
        if completion.get("sampling_revision") != ACQUISITION_REVISION:
            raise ValueError(f"acquisition revision mismatch: {completion_path}")
        construction = str(completion["construction"])
        ny = int(completion["Ny"])
        alpha_1 = float(completion["alpha_1"])
        cycles = int(completion["cycles"])
        if construction not in EXPECTED_CONSTRUCTIONS or ny not in EXPECTED_NY:
            raise ValueError(f"unregistered acquisition case: {completion_path}")
        if alpha_1 not in EXPECTED_ALPHA_1 or cycles != 2 * ny:
            raise ValueError(f"acquisition science contract mismatch: {completion_path}")
        result_path = completion_path.parent / str(completion["result_filename"])
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result_bytes = int(completion["result_bytes"])
        result_sha256 = str(completion["result_sha256"])
        if result_path.stat().st_size != result_bytes:
            raise ValueError(f"acquisition byte-count mismatch: {result_path}")
        if verify_checksums and sha256_file(result_path) != result_sha256:
            raise ValueError(f"acquisition checksum mismatch: {result_path}")
        acquisition_hashes = dict(completion["source_hashes"])
        if shared_acquisition_hashes is None:
            shared_acquisition_hashes = acquisition_hashes
        elif acquisition_hashes != shared_acquisition_hashes:
            raise ValueError("acquisition source hashes differ across completed batches")
        case_indices = tuple(map(int, completion["case_sample_indices"]))
        global_indices = tuple(map(int, completion["global_sample_indices"]))
        if len(case_indices) != len(global_indices) or not case_indices:
            raise ValueError(f"acquisition sample indices are invalid: {completion_path}")
        with np.load(result_path, allow_pickle=False) as archive:
            _expect_scalar(archive, "schema", "pure_tangent_replay_acquisition_result_v1")
            _expect_scalar(archive, "task_id", str(completion["task_id"]))
            _expect_scalar(archive, "construction", construction)
            _expect_scalar(archive, "Nx", 20)
            _expect_scalar(archive, "Ny", ny)
            _expect_scalar(archive, "alpha_1", alpha_1)
            _expect_scalar(archive, "cycles_total", cycles)
            _expect_scalar(archive, "dtype", "complex128")
            _expect_scalar(archive, "sequence", "raster_y")
            _expect_scalar(archive, "replay_frame_init_prepared", construction == "hard")
            if json.loads(str(archive["source_hashes_json"].item())) != acquisition_hashes:
                raise ValueError(f"acquisition NPZ source hashes mismatch: {result_path}")
            if not np.array_equal(archive["case_sample_indices"], case_indices):
                raise ValueError(f"case sample indices mismatch: {result_path}")
            if not np.array_equal(archive["global_sample_indices"], global_indices):
                raise ValueError(f"global sample indices mismatch: {result_path}")
        count = len(case_indices)
        dimension = 40 * ny
        updates = (11 if construction == "hard" else 20) * ny
        expected_shapes = {
            "initial_ranks": ((count,), np.dtype(np.int64)),
            "record_schedule": ((count, cycles, updates), np.dtype(np.int32)),
            "record_outcomes_packed": (
                (count, (cycles * updates * len(CHANNELS) + 7) // 8),
                np.dtype(np.uint8),
            ),
            "record_targets_packed": (
                (count, (cycles * updates * len(CHANNELS) + 7) // 8),
                np.dtype(np.uint8),
            ),
        }
        for key, (shape, dtype) in expected_shapes.items():
            observed_shape, observed_dtype, _fortran, _offset = _npy_metadata(result_path, key)
            if observed_shape != shape or observed_dtype != dtype:
                raise ValueError(
                    f"acquisition {key} shape/dtype mismatch: {observed_shape}/{observed_dtype}"
                )
        initial_shape, initial_dtype, _fortran, _offset = _npy_metadata(
            result_path, "initial_frame"
        )
        if (
            initial_shape[0:2] != (count, dimension)
            or initial_shape[2] < 1
            or initial_dtype != np.dtype(np.complex128)
        ):
            raise ValueError(
                f"acquisition initial frame shape/dtype mismatch: {result_path}"
            )
        final_shape, final_dtype, _fortran, _offset = _npy_metadata(result_path, "final_frame")
        if final_shape[0:2] != (count, dimension) or final_dtype != np.dtype(np.complex128):
            raise ValueError(f"acquisition final frame shape/dtype mismatch: {result_path}")
        with np.load(result_path, allow_pickle=False) as archive:
            initial_ranks = np.asarray(archive["initial_ranks"], dtype=np.int64)
            final_ranks = np.asarray(archive["final_ranks"], dtype=np.int64)
        if (
            np.any(initial_ranks < 0)
            or np.any(initial_ranks > initial_shape[2])
            or np.any(final_ranks < 0)
            or np.any(final_ranks > final_shape[2])
        ):
            raise ValueError(f"acquisition frame ranks exceed saved capacity: {result_path}")
        sources.append(
            BatchSource(
                task_id=str(completion["task_id"]),
                construction=construction,
                ny=ny,
                alpha_1=alpha_1,
                cycles=cycles,
                batch_index=int(completion["batch_index"]),
                batch_seed=int(completion["seed"]),
                case_sample_indices=case_indices,
                global_sample_indices=global_indices,
                result_path=str(result_path.resolve()),
                completion_path=str(completion_path.resolve()),
                result_bytes=result_bytes,
                result_sha256=result_sha256,
                completion_sha256=sha256_file(completion_path),
                acquisition_configuration_sha256=str(completion["configuration_sha256"]),
                acquisition_source_hashes=acquisition_hashes,
            )
        )
    return sources


def expand_sample_tasks(sources: Sequence[BatchSource]) -> list[SampleTask]:
    tasks: list[SampleTask] = []
    for source in sources:
        for row, (case_index, global_index) in enumerate(
            zip(source.case_sample_indices, source.global_sample_indices)
        ):
            task_id = (
                f"{source.construction}_Ny{source.ny:03d}_a1-{int(source.alpha_1)}_"
                f"sample-{case_index:03d}_global-{global_index:04d}"
            )
            tasks.append(
                SampleTask(
                    task_id=task_id,
                    construction=source.construction,
                    ny=source.ny,
                    alpha_1=source.alpha_1,
                    cycles=source.cycles,
                    case_sample_index=case_index,
                    global_sample_index=global_index,
                    batch_row=row,
                    source=source,
                )
            )
    keys = {
        (task.construction, task.ny, task.alpha_1, task.case_sample_index)
        for task in tasks
    }
    cases = {(task.construction, task.ny, task.alpha_1) for task in tasks}
    if len(tasks) != EXPECTED_SAMPLES or len(keys) != EXPECTED_SAMPLES:
        raise RuntimeError(f"acquisition did not expand to {EXPECTED_SAMPLES} unique samples")
    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(f"acquisition did not contain {EXPECTED_CASES} cases")
    for case in cases:
        observed = sorted(
            task.case_sample_index
            for task in tasks
            if (task.construction, task.ny, task.alpha_1) == case
        )
        if observed != list(range(100)):
            raise RuntimeError(f"case {case} does not contain exactly samples 0..99")
    if sorted(task.global_sample_index for task in tasks) != list(range(EXPECTED_SAMPLES)):
        raise RuntimeError("global acquisition sample indices are not exactly 0..1199")
    return sorted(tasks, key=lambda task: task.global_sample_index)


def result_paths(output_root: Path, task: SampleTask) -> tuple[Path, Path]:
    directory = (
        output_root
        / task.construction
        / f"Ny{task.ny:03d}"
        / f"alpha1_{int(task.alpha_1)}"
    )
    stem = f"sample_{task.case_sample_index:03d}_global_{task.global_sample_index:04d}"
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def task_identity(
    task: SampleTask, *, config_sha256: str, hashes: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
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
        "case_sample_index": task.case_sample_index,
        "global_sample_index": task.global_sample_index,
        "configuration_sha256": config_sha256,
        "source_hashes": dict(hashes),
        "acquisition_task_id": task.source.task_id,
        "acquisition_batch_row": task.batch_row,
        "acquisition_result_filename": Path(task.source.result_path).name,
        "acquisition_result_bytes": task.source.result_bytes,
        "acquisition_result_sha256": task.source.result_sha256,
        "acquisition_completion_sha256": task.source.completion_sha256,
        "acquisition_configuration_sha256": task.source.acquisition_configuration_sha256,
        "acquisition_source_hashes": dict(task.source.acquisition_source_hashes),
    }


def verified_complete(
    output_root: Path,
    task: SampleTask,
    *,
    config_sha256: str,
    hashes: Mapping[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = result_paths(output_root, task)
    if not result_path.exists() and not completion_path.exists():
        return False, "missing"
    if not result_path.is_file() or not completion_path.is_file():
        return False, "incomplete pair"
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
        for key, expected in (
            ("endpoint_projector_relative_frobenius_tolerance", 1e-6),
            (
                "endpoint_gate_calibration_revision",
                "cpu_gpu_projector_replay_pilot_20260910_v1",
            ),
            ("cross_replay_projector_relative_frobenius_tolerance", 2e-6),
            (
                "cross_replay_gate_calibration_revision",
                "full_late_cpu_projector_sample018_20260910_v2",
            ),
        ):
            if completion.get(key) != expected:
                return False, f"completion endpoint-gate mismatch: {key}"
        with np.load(result_path, allow_pickle=False) as archive:
            _expect_scalar(archive, "schema", RESULT_SCHEMA)
            _expect_scalar(archive, "task_id", task.task_id)
            _expect_scalar(archive, "configuration_sha256", config_sha256)
            _expect_scalar(archive, "acquisition_result_sha256", task.source.result_sha256)
            _expect_scalar(archive, "endpoint_replay_verified", True)
            endpoint_tolerance = float(
                archive["endpoint_projector_relative_frobenius_tolerance"].item()
            )
            cross_tolerance = float(
                archive[
                    "cross_replay_projector_relative_frobenius_tolerance"
                ].item()
            )
            if endpoint_tolerance != 1e-6 or cross_tolerance != 2e-6:
                return False, "endpoint tolerance identity mismatch"
            _expect_scalar(
                archive,
                "endpoint_gate_calibration_revision",
                "cpu_gpu_projector_replay_pilot_20260910_v1",
            )
            _expect_scalar(
                archive,
                "cross_replay_gate_calibration_revision",
                "full_late_cpu_projector_sample018_20260910_v2",
            )
            for key, tolerance in (
                ("endpoint_projector_relative_frobenius_error_full", endpoint_tolerance),
                ("endpoint_projector_relative_frobenius_error_late", endpoint_tolerance),
                ("cross_replay_projector_relative_frobenius_error", cross_tolerance),
            ):
                error = float(archive[key].item())
                if not np.isfinite(error) or error < 0.0 or error > tolerance:
                    return False, f"endpoint replay gate failed: {key}"
            active_dimension = int(archive["active_dimension"].item())
        dimension = 40 * task.ny
        shape, dtype, _fortran, _offset = _npy_metadata(result_path, "final_cocycle_hat")
        if shape != (dimension, dimension) or dtype != np.dtype(np.complex128):
            return False, "final cocycle shape/dtype mismatch"
        shape, dtype, _fortran, _offset = _npy_metadata(result_path, "full_qr_log_increments")
        if shape != (task.cycles, active_dimension) or dtype != np.dtype(np.float64):
            return False, "per-cycle QR diagnostic shape/dtype mismatch"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return False, f"invalid output: {exc}"
    return True, "verified"


def _close_memmap(array: np.memmap) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def load_sample(task: SampleTask) -> dict[str, Any]:
    source = Path(task.source.result_path)
    mappings: list[np.memmap] = []
    try:
        for key in (
            "initial_frame",
            "initial_ranks",
            "final_frame",
            "final_ranks",
            "record_schedule",
            "record_outcomes_packed",
            "record_targets_packed",
            "measurement_log_probability",
        ):
            mappings.append(mmap_npy_member(source, key))
        (
            initial_frames,
            initial_ranks,
            final_frames,
            final_ranks,
            schedules,
            outcomes_packed,
            targets_packed,
            log_probability,
        ) = mappings
        row = task.batch_row
        initial_rank = int(initial_ranks[row])
        final_rank = int(final_ranks[row])
        initial = np.array(initial_frames[row, :, :initial_rank], dtype=np.complex128, copy=True)
        final = np.array(final_frames[row, :, :final_rank], dtype=np.complex128, copy=True)
        schedule = np.array(schedules[row], dtype=np.int32, copy=True)
        outcomes_raw = np.array(outcomes_packed[row], dtype=np.uint8, copy=True)
        targets_raw = np.array(targets_packed[row], dtype=np.uint8, copy=True)
        cycle_log_probability = np.array(
            log_probability[row], dtype=np.float64, copy=True
        )
    finally:
        for mapping in mappings:
            _close_memmap(mapping)
    updates = schedule.shape[1]
    logical_count = task.cycles * updates * len(CHANNELS)
    outcomes = np.unpackbits(
        outcomes_raw, count=logical_count, bitorder="little"
    ).astype(np.bool_).reshape(task.cycles, updates, len(CHANNELS))
    targets = np.unpackbits(
        targets_raw, count=logical_count, bitorder="little"
    ).astype(np.bool_).reshape(task.cycles, updates, len(CHANNELS))
    if not np.all(targets == EXPECTED_OCCUPATIONS[None, None, :]):
        raise ValueError(f"saved perfect-correction targets are invalid for {task.task_id}")
    record: list[dict[str, Any]] = []
    for cycle_index in range(task.cycles):
        for update_index in range(updates):
            events: list[dict[str, Any]] = []
            for channel_index, channel in enumerate(CHANNELS):
                outcome = bool(outcomes[cycle_index, update_index, channel_index])
                expected = bool(EXPECTED_OCCUPATIONS[channel_index])
                events.append(
                    {
                        "kind": "measurement",
                        "channel": channel,
                        "outcome_occupied": outcome,
                    }
                )
                if outcome != expected:
                    events.append(
                        {
                            "kind": "correction",
                            "channel": channel,
                            "expected_occupied": expected,
                            "target_occupied": bool(
                                targets[cycle_index, update_index, channel_index]
                            ),
                        }
                    )
            record.append(
                {
                    "cycle": cycle_index + 1,
                    "site_id": int(schedule[cycle_index, update_index]),
                    "branch_events": events,
                }
            )
    return {
        "initial_frame": initial,
        "initial_rank": initial_rank,
        "final_frame": final,
        "final_rank": final_rank,
        "record": record,
        "cycle_log_probability": cycle_log_probability,
        "updates_per_cycle": updates,
    }


def projector_relative_frobenius(first: np.ndarray, second: np.ndarray) -> float:
    overlap = first.conj().T @ second
    squared = float(first.shape[1] + second.shape[1]) - 2.0 * float(
        np.linalg.norm(overlap, ord="fro") ** 2
    )
    squared = max(squared, 0.0)
    return float(np.sqrt(squared / max(first.shape[1], second.shape[1], 1)))


def mode_x_coordinates(nx: int, ny: int) -> np.ndarray:
    return np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)


class ReplayAudit:
    def __init__(self, cycles: int) -> None:
        self.log_probability = np.zeros(cycles + 1, dtype=np.float64)
        self.site_count = np.zeros(cycles + 1, dtype=np.int64)

    def __call__(self, *, cycle: int, branch_log_weight: float, **_: Any) -> None:
        self.log_probability[int(cycle)] += float(branch_log_weight)
        self.site_count[int(cycle)] += 1


class TangentCapture:
    """Capture compact diagnostics and the final scale-separated one-leg product."""

    requires_physical_covariance = False

    def __init__(
        self,
        *,
        physical_cycles: int,
        start_cycle: int,
        progress_callback: Any = None,
    ) -> None:
        self.physical_cycles = int(physical_cycles)
        self.start_cycle = int(start_cycle)
        self.expected_tangent_cycles = self.physical_cycles - self.start_cycle + 1
        self.log_diag: list[np.ndarray] = []
        self.log_increments: list[np.ndarray] = []
        self.qr_rates: list[np.ndarray] = []
        self.null_masks: list[np.ndarray] = []
        self.null_counts: list[int] = []
        self.min_probability: list[float] = []
        self.min_denominator: list[float] = []
        self.invalid_count: list[int] = []
        self.block_scales: list[np.ndarray] = []
        self.final: dict[str, Any] | None = None
        self._previous_log: np.ndarray | None = None
        self.progress_callback = progress_callback

    def __call__(
        self,
        *,
        cycle: int,
        lyapunov_cycle: int,
        G: Any,
        native_state: Any = None,
        spectra: Any,
        lyapunov_frame: Any,
        lyapunov_log_diag: Any,
        lyapunov_cycle_null_mask: Any,
        lyapunov_null_counts: Any,
        lyapunov_min_branch_probability: Any,
        lyapunov_min_abs_born_denominator: Any,
        lyapunov_invalid_branch_count: Any,
        lyapunov_block_sizes: Any,
        lyapunov_block_core_hat: Any,
        lyapunov_block_core_log_scale: Any,
        lyapunov_block_core_null_count: Any,
        lyapunov_initial_block_basis: Any,
        lyapunov_initial_active_occupations: Any,
        lyapunov_initial_active_purity_defect: Any,
        **_: Any,
    ) -> None:
        if G is not None:
            raise RuntimeError("CPU tangent replay unexpectedly materialized a covariance")
        current = np.array(lyapunov_log_diag[0], dtype=np.float64, copy=True)
        previous = np.zeros_like(current) if self._previous_log is None else self._previous_log
        self.log_diag.append(current)
        self.log_increments.append(current - previous)
        self.qr_rates.append(np.array(spectra[0], dtype=np.float64, copy=True))
        self.null_masks.append(
            np.array(lyapunov_cycle_null_mask[0], dtype=np.bool_, copy=True)
        )
        self.null_counts.append(int(lyapunov_null_counts[0]))
        self.min_probability.append(float(lyapunov_min_branch_probability[0]))
        self.min_denominator.append(float(lyapunov_min_abs_born_denominator[0]))
        self.invalid_count.append(int(lyapunov_invalid_branch_count[0]))
        self.block_scales.append(
            np.asarray(
                [float(value[0]) for value in lyapunov_block_core_log_scale],
                dtype=np.float64,
            )
        )
        self._previous_log = current
        if self.progress_callback is not None:
            self.progress_callback(int(cycle), int(lyapunov_cycle))
        if int(cycle) == self.physical_cycles:
            self.final = {
                "frame": np.array(lyapunov_frame[0], dtype=np.complex128, copy=True),
                "block_sizes": tuple(map(int, lyapunov_block_sizes)),
                "core_hat": tuple(
                    np.array(value[0], dtype=np.complex128, copy=True)
                    for value in lyapunov_block_core_hat
                ),
                "core_log_scale": np.asarray(
                    [float(value[0]) for value in lyapunov_block_core_log_scale],
                    dtype=np.float64,
                ),
                "core_null_count": np.asarray(
                    [int(value[0]) for value in lyapunov_block_core_null_count],
                    dtype=np.int64,
                ),
                "initial_basis": tuple(
                    np.array(value[0], dtype=np.complex128, copy=True)
                    for value in lyapunov_initial_block_basis
                ),
                "initial_occupations": np.array(
                    lyapunov_initial_active_occupations, dtype=np.float64, copy=True
                ),
                "initial_purity_defect": float(lyapunov_initial_active_purity_defect),
            }

    def finalize(
        self,
        *,
        prefix: str,
        nx: int,
        ny: int,
        slow_mode_count: int,
        singular_tolerance: float,
        materialize_cocycle: bool,
    ) -> dict[str, Any]:
        if self.final is None or len(self.log_diag) != self.expected_tangent_cycles:
            raise RuntimeError(f"{prefix} tangent capture is incomplete")
        final = self.final
        frame = final["frame"]
        block_sizes = final["block_sizes"]
        if len(block_sizes) != 2 or sum(block_sizes) != frame.shape[1]:
            raise RuntimeError("occupied/empty tangent blocks do not span the active space")
        parts: list[dict[str, np.ndarray]] = []
        block_products: list[np.ndarray] = []
        start = 0
        for block_index, size in enumerate(block_sizes):
            stop = start + size
            q = frame[:, start:stop]
            core = final["core_hat"][block_index]
            basis = final["initial_basis"][block_index]
            left, singular, right_h = np.linalg.svd(core, full_matrices=False)
            logs = np.full(singular.shape, -np.inf, dtype=np.float64)
            finite = np.isfinite(singular) & (singular > singular_tolerance)
            logs[finite] = np.log(singular[finite]) + final["core_log_scale"][block_index]
            output = q @ left
            input_vectors = basis @ right_h.conj().T
            output /= np.maximum(
                np.linalg.norm(output, axis=0), np.finfo(np.float64).tiny
            )
            input_vectors /= np.maximum(
                np.linalg.norm(input_vectors, axis=0), np.finfo(np.float64).tiny
            )
            parts.append(
                {
                    "logs": logs,
                    "output": output,
                    "input": input_vectors,
                }
            )
            if materialize_cocycle:
                block_products.append((q @ core) @ basis.conj().T)
            start = stop
        pair_rates = (
            parts[0]["logs"][:, None] + parts[1]["logs"][None, :]
        ) / float(self.expected_tangent_cycles)
        finite_indices = np.argwhere(np.isfinite(pair_rates))
        if finite_indices.shape[0] < slow_mode_count:
            raise FloatingPointError(
                f"only {finite_indices.shape[0]} finite particle-hole tangent modes"
            )
        finite_values = pair_rates[tuple(finite_indices.T)]
        order = np.lexsort(
            (finite_indices[:, 1], finite_indices[:, 0], np.abs(finite_values))
        )
        selected = finite_indices[order[:slow_mode_count]].astype(np.int32)
        occupied_index, empty_index = selected.T
        occupied_output = parts[0]["output"][:, occupied_index]
        empty_output = parts[1]["output"][:, empty_index]
        occupied_input = parts[0]["input"][:, occupied_index]
        empty_input = parts[1]["input"][:, empty_index]
        x = mode_x_coordinates(nx, ny)
        x_profiles = np.empty((slow_mode_count, nx), dtype=np.float64)
        for mode in range(slow_mode_count):
            density = 0.5 * (
                np.abs(occupied_output[:, mode]) ** 2
                + np.abs(empty_output[:, mode]) ** 2
            )
            x_profiles[mode] = np.bincount(x, weights=density, minlength=nx)
        arrays: dict[str, Any] = {
            f"{prefix}_window_start_cycle": np.asarray(self.start_cycle, dtype=np.int64),
            f"{prefix}_window_cycle_count": np.asarray(
                self.expected_tangent_cycles, dtype=np.int64
            ),
            f"{prefix}_qr_cumulative_log_diag": np.stack(self.log_diag),
            f"{prefix}_qr_log_increments": np.stack(self.log_increments),
            f"{prefix}_qr_finite_time_rates": np.stack(self.qr_rates),
            f"{prefix}_cycle_null_mask": np.stack(self.null_masks),
            f"{prefix}_cumulative_null_count": np.asarray(
                self.null_counts, dtype=np.int64
            ),
            f"{prefix}_cumulative_min_branch_probability": np.asarray(
                self.min_probability, dtype=np.float64
            ),
            f"{prefix}_cumulative_min_abs_born_denominator": np.asarray(
                self.min_denominator, dtype=np.float64
            ),
            f"{prefix}_cumulative_invalid_branch_count": np.asarray(
                self.invalid_count, dtype=np.int64
            ),
            f"{prefix}_block_core_log_scale_by_cycle": np.stack(self.block_scales),
            f"{prefix}_block_sizes": np.asarray(block_sizes, dtype=np.int64),
            f"{prefix}_block_core_null_count": final["core_null_count"],
            f"{prefix}_initial_active_occupations": final["initial_occupations"],
            f"{prefix}_initial_active_purity_defect": np.asarray(
                final["initial_purity_defect"], dtype=np.float64
            ),
            f"{prefix}_one_leg_logs_occupied": parts[0]["logs"],
            f"{prefix}_one_leg_logs_empty": parts[1]["logs"],
            f"{prefix}_slow_pair_indices": selected,
            f"{prefix}_slow_pair_rates": pair_rates[occupied_index, empty_index],
            f"{prefix}_slow_effective_gaps_per_cycle": -2.0
            * pair_rates[occupied_index, empty_index],
            f"{prefix}_slow_output_occupied": occupied_output,
            f"{prefix}_slow_output_empty": empty_output,
            f"{prefix}_slow_input_occupied": occupied_input,
            f"{prefix}_slow_input_empty": empty_input,
            f"{prefix}_slow_x_profiles": x_profiles,
        }
        if materialize_cocycle:
            common_scale = float(np.max(final["core_log_scale"]))
            if not np.isfinite(common_scale):
                raise FloatingPointError("final one-leg cocycle has no finite scale")
            cocycle_hat = np.zeros(
                (frame.shape[0], frame.shape[0]), dtype=np.complex128
            )
            for block, scale in zip(block_products, final["core_log_scale"]):
                cocycle_hat += np.exp(float(scale) - common_scale) * block
            norm = float(np.linalg.norm(cocycle_hat, ord="fro"))
            if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
                raise FloatingPointError("final one-leg cocycle normalization failed")
            cocycle_hat /= norm
            arrays["final_cocycle_hat"] = cocycle_hat
            arrays["final_cocycle_log_scale"] = np.asarray(
                common_scale + np.log(norm), dtype=np.float64
            )
        return arrays


class OneCycleMaterializer:
    """Materialize selected one-cycle maps without making them campaign outputs."""

    requires_physical_covariance = False

    def __init__(
        self,
        *,
        task: SampleTask,
        requested_cycles: set[int],
        root: Path,
        config_sha256: str,
        hashes: Mapping[str, str],
    ) -> None:
        self.task = task
        self.requested_cycles = set(requested_cycles)
        self.root = root
        self.config_sha256 = config_sha256
        self.hashes = dict(hashes)
        self.previous_q: np.ndarray | None = None
        self.written: list[Path] = []

    def __call__(
        self,
        *,
        cycle: int,
        G: Any,
        lyapunov_frame: Any,
        lyapunov_qr_r: Any,
        lyapunov_initial_block_basis: Any,
        **_: Any,
    ) -> None:
        if G is not None:
            raise RuntimeError("one-cycle materialization unexpectedly built a covariance")
        current_q = np.asarray(lyapunov_frame[0], dtype=np.complex128)
        if self.previous_q is None:
            self.previous_q = np.concatenate(
                [
                    np.asarray(value[0], dtype=np.complex128)
                    for value in lyapunov_initial_block_basis
                ],
                axis=1,
            )
        if int(cycle) in self.requested_cycles:
            r = np.asarray(lyapunov_qr_r[0], dtype=np.complex128)
            one_cycle = (current_q @ r) @ self.previous_q.conj().T
            norm = float(np.linalg.norm(one_cycle, ord="fro"))
            if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
                raise FloatingPointError(f"one-cycle map normalization failed at cycle {cycle}")
            path = self.root / self.task.task_id / f"cycle_{int(cycle):03d}.npz"
            atomic_npz(
                path,
                {
                    "schema": np.asarray("pure_tangent_one_cycle_materialization_v1"),
                    "task_id": np.asarray(self.task.task_id),
                    "cycle": np.asarray(cycle, dtype=np.int64),
                    "one_cycle_cocycle_hat": one_cycle / norm,
                    "one_cycle_cocycle_log_scale": np.asarray(
                        np.log(norm), dtype=np.float64
                    ),
                    "configuration_sha256": np.asarray(self.config_sha256),
                    "source_hashes_json": np.asarray(canonical_json(self.hashes)),
                    "acquisition_result_sha256": np.asarray(
                        self.task.source.result_sha256
                    ),
                    "covariance_action_convention": np.asarray(
                        "J_t[H]=K_t^dagger H K_t"
                    ),
                },
            )
            self.written.append(path)
        self.previous_q = np.array(current_q, copy=True)


def materialize_one_cycle_maps(
    task: SampleTask,
    *,
    requested_cycles: set[int],
    root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    hashes: Mapping[str, str],
) -> list[Path]:
    if not requested_cycles or min(requested_cycles) < 1 or max(requested_cycles) > task.cycles:
        raise ValueError(f"materialized cycles must lie in 1..{task.cycles}")
    sample = load_sample(task)
    observer = OneCycleMaterializer(
        task=task,
        requested_cycles=requested_cycles,
        root=root,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    audit = ReplayAudit(task.cycles)
    hard = task.construction == "hard"
    with redirect_stdout(io.StringIO()):
        result = build_model(task).run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=task.cycles,
            samples=1,
            parallelize_samples=False,
            frame_init=sample["initial_frame"],
            frame_init_prepared=hard,
            initial_purity_tolerance=float(config["initial_purity_tolerance"]),
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=hard,
            random_seed=int(task.source.batch_seed),
            postselect=False,
            postselect_probability=0.0,
            perfect_correction=True,
            physical_covariance_update="rank1",
            state_representation="physical_frame",
            trajectory_replay=sample["record"],
            trajectory_weight_observer=audit,
            trajectory_replay_probability_tol=float(
                config["replay_probability_tolerance"]
            ),
            return_native_state=True,
            require_no_covariance_materialization=True,
            frame_reorthonormalize_interval=1,
            lyapunov_frame_observer=observer,
            lyapunov_basis_mode="pure_occupied_empty",
            lyapunov_start_cycle=1,
            lyapunov_full_space=not hard,
            lyapunov_track_restricted_core=True,
            lyapunov_singular_tol=float(config["singular_tolerance"]),
            lyapunov_failure_mode="raise",
        )
    endpoint = np.asarray(result["native_final"]["frame"], dtype=np.complex128)
    error = projector_relative_frobenius(endpoint, sample["final_frame"])
    if error > float(config["endpoint_projector_relative_frobenius_tolerance"]):
        raise FloatingPointError(f"materialization replay endpoint error {error:.3e}")
    if set(path.stem for path in observer.written) != {
        f"cycle_{cycle:03d}" for cycle in requested_cycles
    }:
        raise RuntimeError("not every requested one-cycle map was written")
    return observer.written


def build_model(task: SampleTask) -> classA_U1FGTN:
    hard = task.construction == "hard"
    model = classA_U1FGTN(
        Nx=20,
        Ny=task.ny,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=task.alpha_1,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=hard,
    )
    if tuple(map(int, model.DW_loc)) != (5, 15):
        raise RuntimeError(f"unexpected domain-wall positions {model.DW_loc}")
    model.construct_OW_projectors(
        nshell=1,
        DW=True,
        trial_orbitals="X",
        dw_truncation=hard,
    )
    return model


def run_window(
    task: SampleTask,
    sample: Mapping[str, Any],
    *,
    start_cycle: int,
    config: Mapping[str, Any],
    progress_queue: Any = None,
    stage: str,
) -> tuple[TangentCapture, dict[str, Any], ReplayAudit]:
    progress_callback = (
        None
        if progress_queue is None
        else lambda physical_cycle, tangent_cycle: progress_queue.put(
            {
                "task_id": task.task_id,
                "stage": stage,
                "physical_cycle": physical_cycle,
                "tangent_cycle": tangent_cycle,
                "stage_cycles": task.cycles - start_cycle + 1,
            }
        )
    )
    capture = TangentCapture(
        physical_cycles=task.cycles,
        start_cycle=start_cycle,
        progress_callback=progress_callback,
    )
    audit = ReplayAudit(task.cycles)
    hard = task.construction == "hard"
    if progress_queue is not None:
        progress_queue.put({"task_id": task.task_id, "stage": f"{stage}_setup"})
    with redirect_stdout(io.StringIO()):
        result = build_model(task).run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=task.cycles,
            samples=1,
            parallelize_samples=False,
            init_mode="default",
            frame_init=np.asarray(sample["initial_frame"], dtype=np.complex128),
            frame_init_prepared=hard,
            initial_purity_tolerance=float(config["initial_purity_tolerance"]),
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=hard,
            random_seed=int(task.source.batch_seed),
            postselect=False,
            postselect_probability=0.0,
            perfect_correction=True,
            physical_covariance_update="rank1",
            state_representation="physical_frame",
            trajectory_replay=sample["record"],
            trajectory_weight_observer=audit,
            trajectory_replay_probability_tol=float(
                config["replay_probability_tolerance"]
            ),
            return_native_state=True,
            require_no_covariance_materialization=True,
            frame_reorthonormalize_interval=1,
            lyapunov_frame_observer=capture,
            lyapunov_basis_mode="pure_occupied_empty",
            lyapunov_start_cycle=start_cycle,
            lyapunov_full_space=not hard,
            lyapunov_track_restricted_core=True,
            lyapunov_singular_tol=float(config["singular_tolerance"]),
            lyapunov_failure_mode="raise",
        )
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("canonical CPU replay materialized a covariance")
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("canonical CPU replay did not use the occupied-frame path")
    return capture, result, audit


def run_sample(
    task: SampleTask,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    hashes: Mapping[str, str],
    output_root: Path,
    progress_queue: Any = None,
) -> dict[str, Any]:
    started = time.monotonic()
    if progress_queue is not None:
        progress_queue.put({"task_id": task.task_id, "stage": "load"})
    sample = load_sample(task)
    full, full_result, full_audit = run_window(
        task,
        sample,
        start_cycle=1,
        config=config,
        progress_queue=progress_queue,
        stage="full",
    )
    late_start = int(config["late_window_start_cycle_multiplier"] * task.ny + 1)
    late, late_result, _late_audit = run_window(
        task,
        sample,
        start_cycle=late_start,
        config=config,
        progress_queue=progress_queue,
        stage="late",
    )
    final_reference = np.asarray(sample["final_frame"], dtype=np.complex128)
    replayed_frames = []
    replay_errors = []
    for label, result in (("full", full_result), ("late", late_result)):
        native = result["native_final"]
        replayed = np.asarray(native["frame"], dtype=np.complex128)
        if int(native["rank"]) != int(sample["final_rank"]):
            raise RuntimeError(f"{label} replay final rank mismatch for {task.task_id}")
        error = projector_relative_frobenius(replayed, final_reference)
        if error > float(config["endpoint_projector_relative_frobenius_tolerance"]):
            raise FloatingPointError(
                f"{label} replay endpoint error {error:.3e} exceeds tolerance for {task.task_id}"
            )
        replayed_frames.append(replayed)
        replay_errors.append(error)
    cross_replay_error = projector_relative_frobenius(
        replayed_frames[0], replayed_frames[1]
    )
    if cross_replay_error > float(
        config["cross_replay_projector_relative_frobenius_tolerance"]
    ):
        raise FloatingPointError(
            "full- and late-window physical replays disagree: "
            f"{cross_replay_error:.3e} exceeds "
            f"{float(config['cross_replay_projector_relative_frobenius_tolerance']):.3e}"
        )
    elapsed = time.monotonic() - started
    arrays: dict[str, Any] = {
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
        "case_sample_index": np.asarray(task.case_sample_index, dtype=np.int64),
        "global_sample_index": np.asarray(task.global_sample_index, dtype=np.int64),
        "configuration_sha256": np.asarray(config_sha256),
        "source_hashes_json": np.asarray(canonical_json(hashes)),
        "acquisition_task_id": np.asarray(task.source.task_id),
        "acquisition_batch_row": np.asarray(task.batch_row, dtype=np.int64),
        "acquisition_result_sha256": np.asarray(task.source.result_sha256),
        "acquisition_completion_sha256": np.asarray(task.source.completion_sha256),
        "acquisition_source_hashes_json": np.asarray(
            canonical_json(task.source.acquisition_source_hashes)
        ),
        "prepared_initial_rank": np.asarray(sample["initial_rank"], dtype=np.int64),
        "final_rank": np.asarray(sample["final_rank"], dtype=np.int64),
        "frame_init_prepared": np.asarray(task.construction == "hard", dtype=np.bool_),
        "active_input_indices": np.asarray(
            full_result["active_top_layer_indices"], dtype=np.int64
        ),
        "active_dimension": np.asarray(full.final["frame"].shape[1], dtype=np.int64),
        "endpoint_projector_relative_frobenius_error_full": np.asarray(
            replay_errors[0], dtype=np.float64
        ),
        "endpoint_projector_relative_frobenius_error_late": np.asarray(
            replay_errors[1], dtype=np.float64
        ),
        "endpoint_projector_relative_frobenius_tolerance": np.asarray(
            config["endpoint_projector_relative_frobenius_tolerance"],
            dtype=np.float64,
        ),
        "endpoint_gate_calibration_revision": np.asarray(
            config["endpoint_gate_calibration_revision"]
        ),
        "cross_replay_projector_relative_frobenius_error": np.asarray(
            cross_replay_error, dtype=np.float64
        ),
        "cross_replay_projector_relative_frobenius_tolerance": np.asarray(
            config["cross_replay_projector_relative_frobenius_tolerance"],
            dtype=np.float64,
        ),
        "cross_replay_gate_calibration_revision": np.asarray(
            config["cross_replay_gate_calibration_revision"]
        ),
        "endpoint_replay_verified": np.asarray(True, dtype=np.bool_),
        "saved_cycle_log_probability": np.asarray(
            sample["cycle_log_probability"], dtype=np.float64
        ),
        "replayed_cycle_log_probability": full_audit.log_probability,
        "replay_cycle_log_probability_max_abs_error": np.asarray(
            np.max(
                np.abs(
                    full_audit.log_probability
                    - np.asarray(sample["cycle_log_probability"], dtype=np.float64)
                )
            ),
            dtype=np.float64,
        ),
        "elapsed_seconds": np.asarray(elapsed, dtype=np.float64),
        "computed_utc": np.asarray(_utc_now()),
        "choi_covariance_constructed": np.asarray(False, dtype=np.bool_),
        "covariance_history_constructed": np.asarray(False, dtype=np.bool_),
        "intermediate_frames_saved": np.asarray(False, dtype=np.bool_),
        "per_cycle_dense_jacobians_saved": np.asarray(False, dtype=np.bool_),
    }
    if progress_queue is not None:
        progress_queue.put({"task_id": task.task_id, "stage": "postprocess"})
    arrays.update(
        full.finalize(
            prefix="full",
            nx=20,
            ny=task.ny,
            slow_mode_count=int(config["slow_mode_count"]),
            singular_tolerance=float(config["singular_tolerance"]),
            materialize_cocycle=True,
        )
    )
    arrays.update(
        late.finalize(
            prefix="late",
            nx=20,
            ny=task.ny,
            slow_mode_count=int(config["slow_mode_count"]),
            singular_tolerance=float(config["singular_tolerance"]),
            materialize_cocycle=False,
        )
    )
    if progress_queue is not None:
        progress_queue.put({"task_id": task.task_id, "stage": "write"})
    result_path, completion_path = result_paths(output_root, task)
    atomic_npz(result_path, arrays)
    result_sha = sha256_file(result_path)
    completion = task_identity(task, config_sha256=config_sha256, hashes=hashes)
    completion.update(
        {
            "status": "complete",
            "completed_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "endpoint_projector_relative_frobenius_error": max(replay_errors),
            "endpoint_projector_relative_frobenius_tolerance": float(
                config["endpoint_projector_relative_frobenius_tolerance"]
            ),
            "endpoint_gate_calibration_revision": config[
                "endpoint_gate_calibration_revision"
            ],
            "cross_replay_projector_relative_frobenius_error": cross_replay_error,
            "cross_replay_projector_relative_frobenius_tolerance": float(
                config["cross_replay_projector_relative_frobenius_tolerance"]
            ),
            "cross_replay_gate_calibration_revision": config[
                "cross_replay_gate_calibration_revision"
            ],
            "result_filename": result_path.name,
            "result_bytes": result_path.stat().st_size,
            "result_sha256": result_sha,
        }
    )
    atomic_json(completion_path, completion)
    complete, reason = verified_complete(
        output_root,
        task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    if not complete:
        raise RuntimeError(f"output verification failed for {task.task_id}: {reason}")
    return {
        "task_id": task.task_id,
        "elapsed_seconds": elapsed,
        "result_bytes": result_path.stat().st_size,
        "endpoint_error": max(replay_errors),
    }


_THREADPOOL_LIMIT = None


def _worker_init(cpu_groups: Any, threads: int) -> None:
    global _THREADPOOL_LIMIT
    cpus = list(map(int, cpu_groups.get()))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(threads))
    _THREADPOOL_LIMIT = threadpool_limits(limits=int(threads))
    _THREADPOOL_LIMIT.__enter__()


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = BatchSource(**payload["task"]["source"])
    task_values = dict(payload["task"])
    task_values["source"] = source
    task = SampleTask(**task_values)
    return run_sample(
        task,
        config=payload["config"],
        config_sha256=payload["config_sha256"],
        hashes=payload["hashes"],
        output_root=Path(payload["output_root"]),
        progress_queue=payload.get("progress_queue"),
    )


def parse_cpu_list(text: str) -> list[int]:
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = map(int, token.split("-", 1))
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range {token!r}")
            values.update(range(first, last + 1))
        else:
            values.add(int(token))
    if not values:
        raise ValueError("CPU list is empty")
    return sorted(values)


def cpu_groups(value: str | None, workers: int, threads: int) -> list[list[int]]:
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    if value:
        groups = [parse_cpu_list(group) for group in value.split(";") if group.strip()]
        if len(groups) != workers:
            raise ValueError("--cpu-groups must contain exactly one semicolon-separated group per worker")
        flat = [cpu for group in groups for cpu in group]
        if len(flat) != len(set(flat)):
            raise ValueError("CPU worker groups overlap")
        missing = set(flat) - set(available)
        if missing:
            raise ValueError(f"requested CPUs are unavailable: {sorted(missing)}")
        if any(len(group) < threads for group in groups):
            raise ValueError("each CPU group must contain at least --threads-per-worker CPUs")
        return groups
    if workers * threads > len(available):
        raise ValueError(
            f"workers*threads={workers * threads} exceeds {len(available)} available logical CPUs"
        )
    return [
        available[index * threads : (index + 1) * threads]
        for index in range(workers)
    ]


def _matches_filters(task: SampleTask, args: argparse.Namespace) -> bool:
    return (
        (args.construction is None or task.construction == args.construction)
        and (args.ny is None or task.ny == args.ny)
        and (args.alpha_1 is None or task.alpha_1 == args.alpha_1)
        and (args.sample is None or task.case_sample_index == args.sample)
        and (args.task_id is None or task.task_id == args.task_id)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, default=DEFAULT_ACQUISITION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--cpu-groups", default=None)
    parser.add_argument("--max-new-tasks", type=int, default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-input-checksums", action="store_true")
    parser.add_argument("--construction", choices=EXPECTED_CONSTRUCTIONS)
    parser.add_argument("--ny", type=int, choices=EXPECTED_NY)
    parser.add_argument("--alpha-1", type=float, choices=EXPECTED_ALPHA_1)
    parser.add_argument("--sample", type=int, choices=range(100))
    parser.add_argument("--task-id")
    parser.add_argument("--materialize-task")
    parser.add_argument(
        "--materialize-cycles",
        default="all",
        help="all or a comma-separated list of physical cycles",
    )
    parser.add_argument("--materialize-root", type=Path)
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.threads_per_worker <= 0:
        parser.error("worker and thread counts must be positive")
    config = load_config()
    config_sha = canonical_hash(config)
    hashes = source_hashes()
    acquisition_root = args.acquisition_root.resolve()
    output_root = args.output_root.resolve()
    sources = discover_batches(
        acquisition_root, verify_checksums=not args.skip_input_checksums
    )
    tasks = [task for task in expand_sample_tasks(sources) if _matches_filters(task, args)]
    if not tasks:
        raise RuntimeError("task filters selected no acquisition samples")
    if args.materialize_task is not None:
        matches = [
            task
            for task in expand_sample_tasks(sources)
            if task.task_id == args.materialize_task
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown materialization task {args.materialize_task!r}")
        task = matches[0]
        requested = (
            set(range(1, task.cycles + 1))
            if args.materialize_cycles.strip().lower() == "all"
            else {
                int(value)
                for value in args.materialize_cycles.split(",")
                if value.strip()
            }
        )
        root = (
            args.materialize_root.resolve()
            if args.materialize_root is not None
            else Path(tempfile.gettempdir()) / "pure_tangent_jacobians"
        )
        with threadpool_limits(limits=args.threads_per_worker):
            paths = materialize_one_cycle_maps(
                task,
                requested_cycles=requested,
                root=root,
                config=config,
                config_sha256=config_sha,
                hashes=hashes,
            )
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "materialized_cycles": sorted(requested),
                    "materialize_root": str(root),
                    "files": [str(path) for path in paths],
                },
                indent=2,
            )
        )
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    valid: list[SampleTask] = []
    pending: list[SampleTask] = []
    invalid: list[tuple[str, str]] = []
    for task in tqdm(tasks, desc="Verify CPU replay outputs", unit="sample"):
        complete, reason = verified_complete(
            output_root, task, config_sha256=config_sha, hashes=hashes
        )
        if complete:
            valid.append(task)
        else:
            pending.append(task)
            if reason not in ("missing",):
                invalid.append((task.task_id, reason))
    report = {
        "schema": "pure_tangent_cpu_replay_inventory_v1",
        "sampling_revision": REPLAY_REVISION,
        "acquisition_root": str(acquisition_root),
        "output_root": str(output_root),
        "selected": len(tasks),
        "verified_complete": len(valid),
        "pending": len(pending),
        "invalid_existing_pairs": invalid,
        "configuration_sha256": config_sha,
        "source_hashes": hashes,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.report_only:
        return 0
    if args.max_new_tasks is not None:
        pending = pending[: max(0, int(args.max_new_tasks))]
    if not pending:
        print("[complete] every selected CPU tangent replay is verified", flush=True)
        return 0
    groups = cpu_groups(args.cpu_groups, args.workers, args.threads_per_worker)
    print(
        f"[launch] samples={len(pending)}, workers={args.workers}, "
        f"threads_per_worker={args.threads_per_worker}, cpu_groups={groups}",
        flush=True,
    )
    context = mp.get_context("spawn")
    with context.Manager() as manager:
        group_queue = manager.Queue()
        for group in groups:
            group_queue.put(group)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(group_queue, args.threads_per_worker),
        ) as executor:
            progress_queue = manager.Queue()
            future_map = {}
            task_iterator = iter(pending)

            def submit(task: SampleTask) -> Any:
                values = asdict(task)
                payload = {
                    "task": values,
                    "config": config,
                    "config_sha256": config_sha,
                    "hashes": hashes,
                    "output_root": str(output_root),
                    "progress_queue": progress_queue,
                }
                future = executor.submit(_worker, payload)
                future_map[future] = task
                return future

            for _ in range(min(args.workers, len(pending))):
                submit(next(task_iterator))
            with tqdm(total=len(pending), desc="CPU tangent replay", unit="sample") as bar:
                remaining = set(future_map)
                live: dict[str, str] = {}
                while remaining:
                    while True:
                        try:
                            message = progress_queue.get_nowait()
                        except queue.Empty:
                            break
                        name = str(message["task_id"])
                        stage = str(message["stage"])
                        if stage in ("full", "late"):
                            live[name] = (
                                f"{stage} {int(message['tangent_cycle'])}/"
                                f"{int(message['stage_cycles'])}"
                            )
                        else:
                            live[name] = stage
                    if live:
                        summary = "; ".join(
                            f"{name}: {status}" for name, status in list(live.items())[:4]
                        )
                        bar.set_postfix_str(summary, refresh=True)
                    done, remaining = wait(
                        remaining, timeout=1.0, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        task = future_map[future]
                        result = future.result()
                        live.pop(task.task_id, None)
                        bar.set_postfix_str(
                            f"{task.task_id} {result['elapsed_seconds'] / 60:.1f}m",
                            refresh=False,
                        )
                        bar.update(1)
                        try:
                            next_task = next(task_iterator)
                        except StopIteration:
                            continue
                        remaining.add(submit(next_task))
    print("[done] selected CPU tangent replays completed and verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
