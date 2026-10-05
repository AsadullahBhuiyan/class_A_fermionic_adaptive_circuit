#!/usr/bin/env python3
"""Run the two-lane hard-wall all-Ay entropy-contour campaign on an A100."""

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
from typing import Any, Callable, Mapping

if __name__ == "__main__":
    print("[startup] loading NumPy, Torch, and the canonical GPU engine", flush=True)

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = BUNDLE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from entropy_contour_observer import (  # noqa: E402
    OBSERVER_SCHEMA,
    HardWallAllAyContourObserver,
    frame_window_observables,
    frame_y0_averaged_width,
    periodic_window_indices,
    validate_padded_frame_ranks,
)


BUNDLE = "16_hard_wall_entropy_contour_all_ay"
RESULT_SCHEMA = "hard_wall_entropy_contour_all_ay_result_shard_v3"
COMPLETION_SCHEMA = "hard_wall_entropy_contour_all_ay_completion_v3"
CHECKPOINT_SCHEMA = "hard_wall_entropy_contour_all_ay_checkpoint_v3"
ENDPOINT_PROGRESS_SCHEMA = "hard_wall_entropy_contour_all_ay_endpoint_progress_v3"
BENCHMARK_SCHEMA = "hard_wall_entropy_contour_all_ay_a100_benchmark_v3"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
EXPECTED_REVISION = (
    "hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_raster_v3"
)
EXPECTED_ROOT_SEED = 2026091416
EXPECTED_NX = 20
EXPECTED_NY_VALUES = (30, 35, 40, 45, 50, 55, 60)
LANE_NY_VALUES = {"A": (50, 60), "B": (30, 35, 40, 45, 55)}
EXPECTED_SAMPLES = 100
EXPECTED_RESULT_SHARD_SIZE = 5
EXPECTED_SEGMENT_CYCLES = 5
EXPECTED_EXECUTION_BATCH_SIZES = {
    30: 80,
    35: 60,
    40: 40,
    45: 30,
    50: 25,
    55: 20,
    60: 20,
}
EXPECTED_EXECUTION_BATCH_COUNTS = {30: 2, 35: 2, 40: 3, 45: 4, 50: 4, 55: 5, 60: 5}
SOURCE_FILES = (
    "run_campaign.py",
    "entropy_contour_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)
MATRIX_BATCH_CANDIDATES = (8, 16, 32, 64, 80, 128)
MINIMUM_GPU_HEADROOM_BYTES = 8 * 1024**3
MAXIMUM_PROJECTED_ENDPOINT_SECONDS = 3600.0


@dataclass(frozen=True)
class ExecutionBatch:
    lane: str
    ny: int
    batch_index: int
    sample_start: int
    sample_stop: int
    task_id: str
    seed: int

    @property
    def cycles(self) -> int:
        return 2 * self.ny

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def global_sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))

@dataclass(frozen=True)
class ResultShard:
    execution_batch: ExecutionBatch
    shard_index: int
    sample_start: int
    sample_stop: int

    @property
    def lane(self) -> str:
        return self.execution_batch.lane

    @property
    def ny(self) -> int:
        return self.execution_batch.ny

    @property
    def cycles(self) -> int:
        return self.execution_batch.cycles

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def global_sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))

    @property
    def task_id(self) -> str:
        return (
            f"Ny{self.ny:03d}_shard-{self.shard_index:03d}_"
            f"samples-{self.sample_start:03d}-{self.sample_stop - 1:03d}"
        )

    @property
    def local_slice(self) -> slice:
        start = self.sample_start - self.execution_batch.sample_start
        stop = self.sample_stop - self.execution_batch.sample_start
        return slice(start, stop)


@dataclass
class CheckpointState:
    completed_cycle: int
    elapsed_seconds: float
    frame: np.ndarray
    ranks: np.ndarray
    observer_payload: dict[str, np.ndarray]
    rng_payload: dict[str, np.ndarray]


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


def expected_config() -> dict[str, Any]:
    return {
        "sampling_revision": EXPECTED_REVISION,
        "root_seed": EXPECTED_ROOT_SEED,
        "Nx": EXPECTED_NX,
        "Ny_values": list(EXPECTED_NY_VALUES),
        "lane_Ny_values": {key: list(value) for key, value in LANE_NY_VALUES.items()},
        "samples_per_Ny": EXPECTED_SAMPLES,
        "execution_batch_size_by_Ny": {
            str(key): value for key, value in EXPECTED_EXECUTION_BATCH_SIZES.items()
        },
        "result_shard_size": EXPECTED_RESULT_SHARD_SIZE,
        "cycles_rule": "2*Ny",
        "segment_cycles": EXPECTED_SEGMENT_CYCLES,
        "device": "cuda:0",
        "dtype": "complex128",
        "protocol": {
            "DW": True,
            "domain_wall_interval": [5, 15],
            "dw_truncation": True,
            "meas_slab_only": True,
            "nshell": 1,
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "filling_frac": 0.5,
            "trial_orbitals": "X",
            "init_mode": "default",
            "sequence": "raster_y",
            "perfect_correction": True,
            "postselect": False,
            "postselect_probability": 0.0,
            "n_a": 0.5,
            "state_representation": "physical_frame",
            "triv_region_local_mode": False,
        },
        "observer": {
            "endpoint_only": True,
            "entropy_orders": [1, 2, 3],
            "origin_average": "all_periodic_y0_within_trajectory",
            "origin_coordinate": "relative_dy=(y-y0)_mod_Ny",
            "Ay_values": "0..Ny//2",
            "spatial_contours": ["von_neumann"],
            "contour_retention": "trajectory_resolved_y0_average_all_Ay",
            "contour_padding": "zero_after_valid_dy_count",
            "occupation_tolerance": 1.0e-8,
            "matrix_batch_candidates": list(MATRIX_BATCH_CANDIDATES),
            "minimum_gpu_headroom_gib": 8,
            "benchmark_projection_samples": 20,
            "benchmark_maximum_projected_hours": 1.0,
            "fit_min_Ay": 8,
            "wall_window_half_width": 1,
            "uncertainty": "trajectory_sem_after_within_trajectory_y0_average",
        },
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(config))
    expected = expected_config()
    if normalized != expected:
        differing = sorted(
            key
            for key in set(normalized) | set(expected)
            if normalized.get(key) != expected.get(key)
        )
        raise ValueError(
            "configuration differs from the locked campaign contract in: "
            + ", ".join(differing)
        )
    return normalized


def config_hash(config: dict[str, Any]) -> str:
    return _sha256_bytes(_json_bytes(validate_config(config)))


def execution_seed(
    root_seed: int,
    *,
    lane: str,
    ny: int,
    batch_index: int,
    sample_start: int,
    sample_stop: int,
) -> int:
    lane = str(lane).upper()
    if lane not in LANE_NY_VALUES:
        raise ValueError("lane must be 'A' or 'B'")
    label = (
        f"{int(root_seed)}|lane={lane}|Nx={EXPECTED_NX}|Ny={int(ny)}|"
        f"execution={int(batch_index)}|samples={int(sample_start)}:{int(sample_stop)}"
    )
    return int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    ) & ((1 << 63) - 1)


def expand_execution_batches(config: dict[str, Any], *, lane: str) -> list[ExecutionBatch]:
    config = validate_config(config)
    lane = str(lane).upper()
    if lane not in LANE_NY_VALUES:
        raise ValueError("lane must be 'A' or 'B'")
    tasks: list[ExecutionBatch] = []
    for ny in sorted(LANE_NY_VALUES[lane], reverse=True):
        batch_size = EXPECTED_EXECUTION_BATCH_SIZES[ny]
        for batch_index, sample_start in enumerate(range(0, EXPECTED_SAMPLES, batch_size)):
            sample_stop = min(sample_start + batch_size, EXPECTED_SAMPLES)
            task_id = (
                f"lane-{lane}_Ny{ny:03d}_execution-{batch_index:03d}_"
                f"samples-{sample_start:03d}-{sample_stop - 1:03d}"
            )
            tasks.append(
                ExecutionBatch(
                    lane=lane,
                    ny=ny,
                    batch_index=batch_index,
                    sample_start=sample_start,
                    sample_stop=sample_stop,
                    task_id=task_id,
                    seed=execution_seed(
                        int(config["root_seed"]),
                        lane=lane,
                        ny=ny,
                        batch_index=batch_index,
                        sample_start=sample_start,
                        sample_stop=sample_stop,
                    ),
                )
            )
    expected_count = sum(EXPECTED_EXECUTION_BATCH_COUNTS[ny] for ny in LANE_NY_VALUES[lane])
    if len(tasks) != expected_count or len({task.task_id for task in tasks}) != expected_count:
        raise RuntimeError(f"lane {lane} must contain {expected_count} unique execution batches")
    if len({task.seed for task in tasks}) != len(tasks):
        raise RuntimeError("execution-batch seeds must be unique")
    for ny in LANE_NY_VALUES[lane]:
        indices = [
            index
            for task in tasks
            if task.ny == ny
            for index in task.global_sample_indices
        ]
        if indices != list(range(EXPECTED_SAMPLES)):
            raise RuntimeError(f"lane {lane}, Ny={ny} does not cover exactly 100 samples")
    return tasks


def result_shards(task: ExecutionBatch) -> list[ResultShard]:
    shards = []
    for sample_start in range(task.sample_start, task.sample_stop, EXPECTED_RESULT_SHARD_SIZE):
        sample_stop = min(sample_start + EXPECTED_RESULT_SHARD_SIZE, task.sample_stop)
        shards.append(
            ResultShard(
                execution_batch=task,
                shard_index=sample_start // EXPECTED_RESULT_SHARD_SIZE,
                sample_start=sample_start,
                sample_stop=sample_stop,
            )
        )
    if any(shard.sample_count != EXPECTED_RESULT_SHARD_SIZE for shard in shards):
        raise RuntimeError(f"execution batch {task.task_id} does not split into five-sample shards")
    return shards


def result_paths(output_root: Path, shard: ResultShard) -> tuple[Path, Path]:
    directory = output_root / "results" / f"lane_{shard.lane}" / f"Ny{shard.ny:03d}"
    stem = (
        f"shard_{shard.shard_index:03d}_"
        f"samples_{shard.sample_start:03d}-{shard.sample_stop - 1:03d}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def checkpoint_paths(output_root: Path, task: ExecutionBatch) -> tuple[Path, Path]:
    directory = (
        output_root
        / "checkpoints"
        / f"lane_{task.lane}"
        / f"Ny{task.ny:03d}"
        / (
            f"execution_{task.batch_index:03d}_"
            f"samples_{task.sample_start:03d}-{task.sample_stop - 1:03d}"
        )
    )
    return directory / "checkpoint.npz", directory / "checkpoint.json"


def endpoint_progress_paths(output_root: Path, task: ExecutionBatch) -> tuple[Path, Path]:
    directory = checkpoint_paths(output_root, task)[0].parent
    return directory / "endpoint_progress.npz", directory / "endpoint_progress.json"


def benchmark_path(output_root: Path, lane: str) -> Path:
    return output_root / "benchmarks" / f"lane_{str(lane).upper()}_a100_endpoint.json"


def _completion_identity(
    *, shard: ResultShard, config_sha256: str, hashes: dict[str, str]
) -> dict[str, Any]:
    task = shard.execution_batch
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "lane": shard.lane,
        "task_id": shard.task_id,
        "execution_batch_id": task.task_id,
        "Nx": EXPECTED_NX,
        "Ny": shard.ny,
        "cycles": shard.cycles,
        "endpoint_only": True,
        "shard_index": shard.shard_index,
        "sample_start": shard.sample_start,
        "sample_stop": shard.sample_stop,
        "sample_count": shard.sample_count,
        "global_sample_indices": list(shard.global_sample_indices),
        "batch_seed": task.seed,
        "config_sha256": config_sha256,
        "source_hashes": hashes,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "observer_schema": OBSERVER_SCHEMA,
    }


def verified_complete(
    *,
    output_root: Path,
    shard: ResultShard,
    config_sha256: str,
    hashes: dict[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = result_paths(output_root, shard)
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
    for key, value in _completion_identity(
        shard=shard, config_sha256=config_sha256, hashes=hashes
    ).items():
        if completion.get(key) != value:
            return False, f"completion identity mismatch: {key}"
    if completion.get("result_filename") != result_path.name:
        return False, "completion result filename mismatch"
    try:
        actual_bytes = int(result_path.stat().st_size)
        actual_sha256 = sha256_file(result_path)
    except OSError as exc:
        return False, f"result readback failed: {exc}"
    if int(completion.get("result_bytes", -1)) != actual_bytes:
        return False, "result byte-count mismatch"
    if completion.get("result_sha256") != actual_sha256:
        return False, "result checksum mismatch"
    return True, "verified"


def _checkpoint_identity(
    *, task: ExecutionBatch, config_sha256: str, hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint",
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "lane": task.lane,
        "task_id": task.task_id,
        "Nx": EXPECTED_NX,
        "Ny": task.ny,
        "cycles": task.cycles,
        "sample_start": task.sample_start,
        "sample_stop": task.sample_stop,
        "sample_count": task.sample_count,
        "global_sample_indices": list(task.global_sample_indices),
        "batch_seed": task.seed,
        "segment_cycles": EXPECTED_SEGMENT_CYCLES,
        "config_sha256": config_sha256,
        "source_hashes": hashes,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "observer_schema": OBSERVER_SCHEMA,
    }


def _coerce_npz_payload(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        key = str(key)
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"NPZ payload {key!r} has forbidden object dtype")
        normalized[key] = array
    return normalized


def _write_npz(path: Path, payload: Mapping[str, Any], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    normalized = _coerce_npz_payload(payload)
    with temporary.open("wb") as handle:
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **normalized)
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
    """Copy to DriveFS, read back, checksum, and atomically replace the stable file."""

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    expected_bytes = int(local_path.stat().st_size)
    expected_sha256 = sha256_file(local_path)
    try:
        shutil.copyfile(local_path, temporary)
        if int(temporary.stat().st_size) != expected_bytes:
            raise OSError(f"Drive temporary byte-count mismatch for {temporary}")
        if sha256_file(temporary) != expected_sha256:
            raise OSError(f"Drive temporary checksum mismatch for {temporary}")
        os.replace(temporary, final_path)
        if int(final_path.stat().st_size) != expected_bytes:
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


def _capture_rng_state() -> dict[str, np.ndarray]:
    np_state = np.random.get_state()
    payload: dict[str, np.ndarray] = {
        "numpy_algorithm": np.asarray(np_state[0]),
        "numpy_keys": np.asarray(np_state[1], dtype=np.uint32),
        "numpy_position": np.asarray(np_state[2], dtype=np.int64),
        "numpy_has_gauss": np.asarray(np_state[3], dtype=np.int8),
        "numpy_cached_gaussian": np.asarray(np_state[4], dtype=np.float64),
        "torch_cpu": torch.get_rng_state().detach().cpu().numpy(),
    }
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload["torch_cuda_count"] = np.asarray(len(cuda_states), dtype=np.int64)
    for index, state in enumerate(cuda_states):
        payload[f"torch_cuda_{index}"] = state.detach().cpu().numpy()
    return payload


def _restore_rng_state(payload: Mapping[str, np.ndarray]) -> None:
    algorithm = str(np.asarray(payload["numpy_algorithm"]).item())
    np.random.set_state(
        (
            algorithm,
            np.asarray(payload["numpy_keys"], dtype=np.uint32),
            int(np.asarray(payload["numpy_position"]).item()),
            int(np.asarray(payload["numpy_has_gauss"]).item()),
            float(np.asarray(payload["numpy_cached_gaussian"]).item()),
        )
    )
    torch.set_rng_state(
        torch.as_tensor(np.asarray(payload["torch_cpu"], dtype=np.uint8), dtype=torch.uint8)
    )
    saved_cuda_count = int(np.asarray(payload["torch_cuda_count"]).item())
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_cuda_count != current_cuda_count:
        raise RuntimeError(
            "checkpoint CUDA RNG device count mismatch: "
            f"saved={saved_cuda_count}, current={current_cuda_count}"
        )
    if saved_cuda_count:
        states = [
            torch.as_tensor(
                np.asarray(payload[f"torch_cuda_{index}"], dtype=np.uint8),
                dtype=torch.uint8,
            )
            for index in range(saved_cuda_count)
        ]
        torch.cuda.set_rng_state_all(states)


def _seed_rng(seed: int) -> None:
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _checkpoint_npz_payload(
    *,
    task: ExecutionBatch,
    completed_cycle: int,
    elapsed_seconds: float,
    native_state: Mapping[str, Any],
    observer_payload: Mapping[str, Any],
    rng_payload: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    frame = np.asarray(native_state["frame"])
    ranks = np.asarray(native_state["ranks"], dtype=np.int64)
    if frame.dtype != np.complex128:
        raise RuntimeError(f"checkpoint frame must be complex128, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[:2] != (
        task.sample_count,
        2 * EXPECTED_NX * task.ny,
    ):
        raise RuntimeError(f"unexpected checkpoint frame shape {frame.shape}")
    if ranks.shape != (task.sample_count,):
        raise RuntimeError(f"unexpected checkpoint rank shape {ranks.shape}")
    if np.any(ranks < 0) or np.any(ranks > frame.shape[2]):
        raise RuntimeError("checkpoint ranks lie outside the saved frame capacity")
    payload: dict[str, Any] = {
        "completed_cycle": np.asarray(completed_cycle, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "frame": frame,
        "ranks": ranks,
        "sample_indices": np.asarray(task.global_sample_indices, dtype=np.int64),
    }
    for key, value in observer_payload.items():
        payload[f"observer__{key}"] = value
    for key, value in rng_payload.items():
        payload[f"rng__{key}"] = value
    return _coerce_npz_payload(payload)


def save_checkpoint(
    *,
    output_root: Path,
    scratch_root: Path,
    task: ExecutionBatch,
    completed_cycle: int,
    elapsed_seconds: float,
    native_state: Mapping[str, Any],
    observer: HardWallAllAyContourObserver,
    config_sha256: str,
    hashes: dict[str, str],
) -> dict[str, np.ndarray]:
    observer.validate(
        require_dynamics=completed_cycle == task.cycles,
        require_endpoint=False,
    )
    rng_payload = _capture_rng_state()
    payload = _checkpoint_npz_payload(
        task=task,
        completed_cycle=completed_cycle,
        elapsed_seconds=elapsed_seconds,
        native_state=native_state,
        observer_payload=observer.checkpoint_payload(),
        rng_payload=rng_payload,
    )
    task_scratch = scratch_root / task.task_id
    task_scratch.mkdir(parents=True, exist_ok=True)
    local_npz = task_scratch / "checkpoint.npz"
    _write_npz(local_npz, payload, compressed=False)
    final_npz, final_json = checkpoint_paths(output_root, task)
    published = publish_file(local_npz, final_npz)
    metadata = _checkpoint_identity(
        task=task, config_sha256=config_sha256, hashes=hashes
    )
    metadata.update(
        {
            "completed_cycle": int(completed_cycle),
            "elapsed_seconds": float(elapsed_seconds),
            "checkpoint_filename": final_npz.name,
            "checkpoint_bytes": published["bytes"],
            "checkpoint_sha256": published["sha256"],
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_json = task_scratch / "checkpoint.json"
    _write_json(local_json, metadata)
    publish_file(local_json, final_json)
    return rng_payload


def load_checkpoint(
    *,
    output_root: Path,
    task: ExecutionBatch,
    config_sha256: str,
    hashes: dict[str, str],
) -> tuple[CheckpointState | None, str]:
    npz_path, json_path = checkpoint_paths(output_root, task)
    npz_exists = npz_path.is_file()
    json_exists = json_path.is_file()
    if not npz_exists and not json_exists:
        return None, "no checkpoint"
    if not npz_exists or not json_exists:
        return None, "incomplete checkpoint pair"
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable checkpoint JSON: {exc}"
    for key, value in _checkpoint_identity(
        task=task, config_sha256=config_sha256, hashes=hashes
    ).items():
        if metadata.get(key) != value:
            return None, f"checkpoint identity mismatch: {key}"
    if metadata.get("checkpoint_filename") != npz_path.name:
        return None, "checkpoint filename mismatch"
    try:
        actual_bytes = int(npz_path.stat().st_size)
        actual_sha256 = sha256_file(npz_path)
    except OSError as exc:
        return None, f"checkpoint readback failed: {exc}"
    if int(metadata.get("checkpoint_bytes", -1)) != actual_bytes:
        return None, "checkpoint byte-count mismatch"
    if metadata.get("checkpoint_sha256") != actual_sha256:
        return None, "checkpoint checksum mismatch"
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        return None, f"unreadable checkpoint NPZ: {exc}"
    required = {"completed_cycle", "elapsed_seconds", "frame", "ranks", "sample_indices"}
    missing = sorted(required - set(payload))
    if missing:
        return None, f"checkpoint NPZ missing fields: {missing}"
    completed_cycle = int(payload["completed_cycle"].item())
    if completed_cycle != int(metadata.get("completed_cycle", -1)):
        return None, "checkpoint completed-cycle mismatch"
    if not (0 < completed_cycle <= task.cycles):
        return None, "checkpoint completed cycle is out of range"
    if completed_cycle % EXPECTED_SEGMENT_CYCLES:
        return None, "checkpoint is not on a five-cycle boundary"
    frame = np.asarray(payload["frame"])
    ranks = np.asarray(payload["ranks"], dtype=np.int64)
    if frame.dtype != np.complex128:
        return None, f"checkpoint frame dtype is {frame.dtype}, expected complex128"
    if frame.ndim != 3 or frame.shape[:2] != (
        task.sample_count,
        2 * EXPECTED_NX * task.ny,
    ):
        return None, f"checkpoint frame shape mismatch: {frame.shape}"
    if ranks.shape != (task.sample_count,):
        return None, f"checkpoint rank shape mismatch: {ranks.shape}"
    if np.any(ranks < 0) or np.any(ranks > frame.shape[2]):
        return None, "checkpoint ranks lie outside the saved frame capacity"
    if not np.array_equal(
        payload["sample_indices"], np.asarray(task.global_sample_indices, dtype=np.int64)
    ):
        return None, "checkpoint sample-index mismatch"
    observer_payload = {
        key.removeprefix("observer__"): value
        for key, value in payload.items()
        if key.startswith("observer__")
    }
    rng_payload = {
        key.removeprefix("rng__"): value
        for key, value in payload.items()
        if key.startswith("rng__")
    }
    if not observer_payload or not rng_payload:
        return None, "checkpoint lacks observer or RNG state"
    if "seen_cycles" in observer_payload:
        seen = np.asarray(observer_payload["seen_cycles"], dtype=np.bool_)
        expected_seen = np.arange(task.cycles + 1) <= completed_cycle
        if seen.shape != expected_seen.shape or not np.array_equal(seen, expected_seen):
            return None, "checkpoint observer cycles do not match completed cycle"
    return (
        CheckpointState(
            completed_cycle=completed_cycle,
            elapsed_seconds=float(payload["elapsed_seconds"].item()),
            frame=frame,
            ranks=ranks,
            observer_payload=observer_payload,
            rng_payload=rng_payload,
        ),
        "verified",
    )


def _endpoint_progress_identity(
    *,
    task: ExecutionBatch,
    config_sha256: str,
    hashes: dict[str, str],
    final_checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": ENDPOINT_PROGRESS_SCHEMA,
        "status": "endpoint_progress",
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "lane": task.lane,
        "task_id": task.task_id,
        "Nx": EXPECTED_NX,
        "Ny": task.ny,
        "sample_start": task.sample_start,
        "sample_stop": task.sample_stop,
        "global_sample_indices": list(task.global_sample_indices),
        "config_sha256": config_sha256,
        "source_hashes": hashes,
        "observer_schema": OBSERVER_SCHEMA,
        "final_checkpoint_sha256": final_checkpoint_sha256,
    }


def final_checkpoint_sha256(output_root: Path, task: ExecutionBatch) -> str:
    npz_path, json_path = checkpoint_paths(output_root, task)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    digest = str(metadata.get("checkpoint_sha256", ""))
    if (
        metadata.get("completed_cycle") != task.cycles
        or metadata.get("checkpoint_filename") != npz_path.name
        or len(digest) != 64
        or sha256_file(npz_path) != digest
    ):
        raise RuntimeError("final dynamics checkpoint is not durably verified")
    return digest


def save_endpoint_progress(
    *,
    output_root: Path,
    scratch_root: Path,
    task: ExecutionBatch,
    observer: HardWallAllAyContourObserver,
    config_sha256: str,
    hashes: dict[str, str],
    final_checkpoint_sha256_value: str,
) -> None:
    observer.validate(require_dynamics=True, require_endpoint=False)
    task_scratch = scratch_root / task.task_id
    local_npz = task_scratch / "endpoint_progress.npz"
    _write_npz(local_npz, observer.checkpoint_payload(), compressed=True)
    final_npz, final_json = endpoint_progress_paths(output_root, task)
    published = publish_file(local_npz, final_npz)
    metadata = _endpoint_progress_identity(
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
        final_checkpoint_sha256=final_checkpoint_sha256_value,
    )
    metadata.update(
        {
            "progress_filename": final_npz.name,
            "progress_bytes": published["bytes"],
            "progress_sha256": published["sha256"],
            "completed_widths": int(observer.endpoint_width_seen.sum()),
            "last_durable_Ay": (
                int(observer.ay_values[np.flatnonzero(observer.endpoint_width_seen)[-1]])
                if observer.endpoint_width_seen.any()
                else -1
            ),
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_json = task_scratch / "endpoint_progress.json"
    _write_json(local_json, metadata)
    publish_file(local_json, final_json)


def load_endpoint_progress(
    *,
    output_root: Path,
    task: ExecutionBatch,
    observer: HardWallAllAyContourObserver,
    config_sha256: str,
    hashes: dict[str, str],
    final_checkpoint_sha256_value: str,
) -> tuple[bool, str]:
    npz_path, json_path = endpoint_progress_paths(output_root, task)
    if not npz_path.exists() and not json_path.exists():
        return False, "no endpoint progress"
    if not npz_path.is_file() or not json_path.is_file():
        return False, "incomplete endpoint-progress pair"
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        for key, value in _endpoint_progress_identity(
            task=task,
            config_sha256=config_sha256,
            hashes=hashes,
            final_checkpoint_sha256=final_checkpoint_sha256_value,
        ).items():
            if metadata.get(key) != value:
                return False, f"endpoint-progress identity mismatch: {key}"
        if metadata.get("progress_filename") != npz_path.name:
            return False, "endpoint-progress filename mismatch"
        if int(metadata.get("progress_bytes", -1)) != int(npz_path.stat().st_size):
            return False, "endpoint-progress byte-count mismatch"
        if metadata.get("progress_sha256") != sha256_file(npz_path):
            return False, "endpoint-progress checksum mismatch"
        with np.load(npz_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
        observer.restore_checkpoint(payload)
        observer.validate(require_dynamics=True, require_endpoint=False)
        if int(metadata.get("completed_widths", -1)) != int(
            observer.endpoint_width_seen.sum()
        ):
            return False, "endpoint-progress completed-width count mismatch"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, f"unreadable endpoint progress: {exc}"
    return True, "verified"


def remove_checkpoint(output_root: Path, task: ExecutionBatch) -> None:
    npz_path, json_path = checkpoint_paths(output_root, task)
    endpoint_npz, endpoint_json = endpoint_progress_paths(output_root, task)
    for path in (endpoint_json, endpoint_npz, json_path, npz_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        npz_path.parent.rmdir()
    except OSError:
        pass


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


def _benchmark_identity(
    *, lane: str, config_sha256: str, hashes: dict[str, str], gpu_name: str
) -> dict[str, Any]:
    return {
        "schema": BENCHMARK_SCHEMA,
        "status": "accepted",
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "lane": str(lane).upper(),
        "config_sha256": config_sha256,
        "source_hashes": hashes,
        "gpu_name": gpu_name,
        "Ny": 60,
        "benchmark_samples": 5,
        "projection_samples": 20,
        "candidate_matrix_batch_sizes": list(MATRIX_BATCH_CANDIDATES),
        "minimum_gpu_headroom_bytes": MINIMUM_GPU_HEADROOM_BYTES,
        "maximum_projected_seconds": MAXIMUM_PROJECTED_ENDPOINT_SECONDS,
        "eigensolver_backend": "torch.linalg.eigh on Hermitian C_A=R R^dagger",
    }


def _load_benchmark(
    *,
    output_root: Path,
    lane: str,
    config_sha256: str,
    hashes: dict[str, str],
    gpu_name: str,
) -> tuple[dict[str, Any] | None, str]:
    path = benchmark_path(output_root, lane)
    if not path.is_file():
        return None, "missing benchmark receipt"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable benchmark receipt: {exc}"
    for key, value in _benchmark_identity(
        lane=lane,
        config_sha256=config_sha256,
        hashes=hashes,
        gpu_name=gpu_name,
    ).items():
        if payload.get(key) != value:
            return None, f"benchmark identity mismatch: {key}"
    projected = float(payload.get("projected_20_trajectory_seconds", np.inf))
    batch_map = payload.get("matrix_batch_size_by_Ay")
    if projected > MAXIMUM_PROJECTED_ENDPOINT_SECONDS:
        return None, f"benchmark projection exceeds one hour: {projected / 3600:.3f} h"
    if not isinstance(batch_map, dict) or set(batch_map) != {
        str(ay) for ay in range(31)
    }:
        return None, "benchmark batch-size map is incomplete"
    if any(int(value) not in MATRIX_BATCH_CANDIDATES for value in batch_map.values()):
        return None, "benchmark batch-size map contains an unlocked value"
    return payload, "verified"


def _production_shaped_benchmark_frame(*, samples: int = 5) -> torch.Tensor:
    """Create a deterministic prepared complex128 frame with Ny=60 dimensions."""

    physical = 2 * EXPECTED_NX * 60
    occupied = EXPECTED_NX * 60
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026091499)
    permutation = torch.randperm(physical, generator=generator)[:occupied]
    frame = torch.zeros(
        (int(samples), physical, occupied),
        dtype=torch.complex128,
        device="cuda:0",
    )
    columns = torch.arange(occupied, device=frame.device)
    rows = permutation.to(frame.device)
    frame[:, rows, columns] = 1.0 + 0.0j
    return frame


def _benchmark_svd_equivalence(frame: torch.Tensor) -> dict[str, float]:
    maximum_occupation_error = 0.0
    maximum_scalar_error = 0.0
    for ay in (1, 15, 30):
        indices = periodic_window_indices(
            nx=EXPECTED_NX, ny=60, y0_values=[0], ay=ay, device=frame.device
        )
        rows = frame[:1].index_select(1, indices.reshape(-1)).reshape(
            1, 2 * EXPECTED_NX * ay, int(frame.shape[2])
        )
        singular = torch.linalg.svdvals(rows).square().sort(dim=-1).values
        gram = rows @ rows.mH
        gram = 0.5 * (gram + gram.mH)
        occupations = torch.linalg.eigvalsh(gram).clamp(0.0, 1.0)
        maximum_occupation_error = max(
            maximum_occupation_error,
            float((singular - occupations).abs().amax().detach().cpu()),
        )
        block = frame_window_observables(
            frame[:1],
            indices=indices,
            nx=EXPECTED_NX,
            ay=ay,
            return_contours=True,
            occupation_tolerance=1.0e-8,
        )
        nu = singular
        one_minus = 1.0 - nu
        reference = {
            "entropy_von_neumann": (
                -torch.xlogy(nu, nu) - torch.xlogy(one_minus, one_minus)
            ).sum(-1),
            "entropy_renyi2": -torch.log(nu.square() + one_minus.square()).sum(-1),
            "entropy_renyi3": -0.5 * torch.log(nu.pow(3) + one_minus.pow(3)).sum(-1),
            "charge_mean": nu.sum(-1),
            "charge_variance": (nu * one_minus).sum(-1),
        }
        for key, value in reference.items():
            maximum_scalar_error = max(
                maximum_scalar_error,
                float((block[key][:, 0] - value).abs().amax().detach().cpu()),
            )
        closure = (
            block["contour_von_neumann"].sum((-1, -2))
            - block["entropy_von_neumann"]
        )
        maximum_scalar_error = max(
            maximum_scalar_error,
            float(closure.abs().amax().detach().cpu()),
        )
    if maximum_occupation_error > 2.0e-10 or maximum_scalar_error > 2.0e-9:
        raise RuntimeError(
            "Gram/SVD or contour benchmark equivalence failed: "
            f"occupation={maximum_occupation_error:.3e}, scalar={maximum_scalar_error:.3e}"
        )
    return {
        "maximum_occupation_difference": maximum_occupation_error,
        "maximum_scalar_or_closure_difference": maximum_scalar_error,
    }


def select_fastest_accepted_matrix_batch(
    candidate_rows: list[Mapping[str, Any]],
) -> int:
    """Choose the fastest non-OOM candidate that meets the headroom contract."""

    accepted = [
        row
        for row in candidate_rows
        if bool(row.get("accepted"))
        and row.get("error") is None
        and int(row.get("projected_headroom_bytes", 0))
        >= MINIMUM_GPU_HEADROOM_BYTES
    ]
    if not accepted:
        raise RuntimeError("no candidate matrix batch retains 8 GiB of GPU headroom")
    return int(min(accepted, key=lambda row: float(row["representative_seconds"]))["matrix_batch_size"])


def run_a100_endpoint_benchmark(
    *,
    output_root: Path,
    scratch_root: Path,
    lane: str,
    config_sha256: str,
    hashes: dict[str, str],
    gpu: dict[str, Any],
) -> dict[str, Any]:
    """Autotune the endpoint matrix batch and enforce the one-hour projection."""

    existing, reason = _load_benchmark(
        output_root=output_root,
        lane=lane,
        config_sha256=config_sha256,
        hashes=hashes,
        gpu_name=str(gpu["name"]),
    )
    if existing is not None:
        print(f"[benchmark] reusing server-visible accepted receipt: {reason}", flush=True)
        return existing
    print(f"[benchmark] {reason}; constructing nonproduction Ny=60 prepared frame", flush=True)
    frame = _production_shaped_benchmark_frame(samples=5)
    equivalence = _benchmark_svd_equivalence(frame)
    candidate_rows: list[dict[str, Any]] = []
    representative_widths = (5, 15, 30)
    for candidate in MATRIX_BATCH_CANDIDATES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(frame.device)
        started = time.monotonic()
        error = None
        try:
            for ay in representative_widths:
                frame_y0_averaged_width(
                    frame[:3],
                    nx=EXPECTED_NX,
                    ny=60,
                    ay=ay,
                    matrix_batch_size=candidate,
                    occupation_tolerance=1.0e-8,
                )
            torch.cuda.synchronize(frame.device)
            elapsed = time.monotonic() - started
            peak = int(torch.cuda.max_memory_allocated(frame.device))
            headroom = int(gpu["total_bytes"]) - peak
            accepted = headroom >= MINIMUM_GPU_HEADROOM_BYTES
        except torch.cuda.OutOfMemoryError as exc:
            elapsed, peak, headroom, accepted, error = (
                time.monotonic() - started,
                int(torch.cuda.max_memory_allocated(frame.device)),
                0,
                False,
                f"OOM: {exc}",
            )
            torch.cuda.empty_cache()
        candidate_rows.append(
            {
                "matrix_batch_size": candidate,
                "representative_seconds": elapsed,
                "peak_allocated_bytes": peak,
                "projected_headroom_bytes": headroom,
                "accepted": accepted,
                "error": error,
            }
        )
    selected = select_fastest_accepted_matrix_batch(candidate_rows)
    minimum, maximum = np.inf, -np.inf
    width_timings: dict[str, float] = {}
    started_full = time.monotonic()
    bar = tqdm(range(31), desc="benchmark Ny=60 endpoint", unit="Ay", file=sys.stdout)
    for ay in bar:
        started = time.monotonic()
        block = frame_y0_averaged_width(
            frame,
            nx=EXPECTED_NX,
            ny=60,
            ay=ay,
            matrix_batch_size=selected,
            occupation_tolerance=1.0e-8,
        )
        torch.cuda.synchronize(frame.device)
        width_timings[str(ay)] = time.monotonic() - started
        minimum = min(minimum, float(block["occupation_min"]))
        maximum = max(maximum, float(block["occupation_max"]))
        bar.set_postfix(batch=selected, pairs=5 * 60, last_Ay=ay)
    full_seconds = time.monotonic() - started_full
    projected = 4.0 * full_seconds
    payload = _benchmark_identity(
        lane=lane,
        config_sha256=config_sha256,
        hashes=hashes,
        gpu_name=str(gpu["name"]),
    )
    payload.update(
        {
            "benchmark_seed": 2026091499,
            "matrix_batch_size_by_Ay": {str(ay): selected for ay in range(31)},
            "candidate_results": candidate_rows,
            "width_seconds": width_timings,
            "five_trajectory_endpoint_seconds": full_seconds,
            "projected_20_trajectory_seconds": projected,
            "selected_matrix_batch_size": selected,
            "occupation_minimum": minimum,
            "occupation_maximum": maximum,
            "equivalence": equivalence,
            "contour_products": ["trajectory_resolved_y0_averaged_von_neumann_all_Ay"],
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    if projected > MAXIMUM_PROJECTED_ENDPOINT_SECONDS:
        payload["status"] = "rejected"
        raise RuntimeError(
            "production disabled: Ny=60 endpoint projection is "
            f"{projected / 3600:.3f} h, above the one-hour limit"
        )
    scratch_root.mkdir(parents=True, exist_ok=True)
    local = scratch_root / f"lane_{str(lane).upper()}_a100_endpoint.json"
    _write_json(local, payload)
    publish_file(local, benchmark_path(output_root, lane))
    verified, verify_reason = _load_benchmark(
        output_root=output_root,
        lane=lane,
        config_sha256=config_sha256,
        hashes=hashes,
        gpu_name=str(gpu["name"]),
    )
    if verified is None:
        raise RuntimeError(f"published benchmark failed verification: {verify_reason}")
    print(
        f"[benchmark accepted] batch={selected}; projected 20 trajectories="
        f"{projected / 3600:.3f} h",
        flush=True,
    )
    return verified


def _check_space(path: Path, *, required_bytes: int, label: str) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path).free)
    if free < int(required_bytes):
        raise RuntimeError(
            f"insufficient {label} space: free={free / 1024**3:.2f} GiB, "
            f"required={required_bytes / 1024**3:.2f} GiB"
        )
    return free


def build_model(config: dict[str, Any], *, ny: int) -> Any:
    protocol = config["protocol"]
    print(f"[model] constructing hard wall at Nx={EXPECTED_NX}, Ny={ny}", flush=True)
    model = classA_U1FGTN_gpu(
        Nx=EXPECTED_NX,
        Ny=int(ny),
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=True,
        triv_region_local_mode=False,
        device=config["device"],
        dtype=config["dtype"],
        backend="local",
    )
    if tuple(int(value) for value in model.DW_loc) != (5, 15):
        raise RuntimeError(f"unexpected domain-wall positions: {model.DW_loc}")
    if model.dtype != torch.complex128:
        raise RuntimeError("constructed model is not complex128")
    active = model.active_top_layer_indices(meas_slab_only=True)
    if int(active.numel()) != 22 * int(ny):
        raise RuntimeError("active hard-wall slab does not contain 22*Ny modes")
    return model


def _new_observer(task: ExecutionBatch) -> HardWallAllAyContourObserver:
    print(
        f"[observer] {task.task_id}: dynamics records rank only; "
        "entropy/charge eigensolves begin after the final frame is durable",
        flush=True,
    )
    return HardWallAllAyContourObserver(
        nx=EXPECTED_NX,
        ny=task.ny,
        physical_cycles=task.cycles,
        sample_ids=np.asarray(task.global_sample_indices, dtype=np.int64),
        occupation_tolerance=1.0e-8,
    )


def _run_segment(
    *,
    model: Any,
    config: dict[str, Any],
    task: ExecutionBatch,
    observer: HardWallAllAyContourObserver,
    segment_start: int,
    frame: np.ndarray | None,
    ranks: np.ndarray | None,
    on_cycle_complete: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], float]:
    segment_cycles = min(EXPECTED_SEGMENT_CYCLES, task.cycles - segment_start)
    if segment_cycles <= 0:
        raise ValueError("segment start is already at or beyond the final cycle")

    def observe_global_cycle(*, cycle: int, **payload: Any) -> None:
        local_cycle = int(cycle)
        if segment_start and local_cycle == 0:
            return
        global_cycle = segment_start + local_cycle
        observer(cycle=global_cycle, **payload)
        if local_cycle > 0 and on_cycle_complete is not None:
            on_cycle_complete(global_cycle)

    continuing = frame is not None
    if continuing != (ranks is not None):
        raise RuntimeError("checkpoint frame and ranks must be supplied together")
    protocol = config["protocol"]
    started = time.monotonic()
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=segment_cycles,
        postselect=protocol["postselect"],
        postselect_probability=protocol["postselect_probability"],
        perfect_correction=protocol["perfect_correction"],
        samples=task.sample_count,
        init_mode=protocol["init_mode"],
        frame_init=frame,
        frame_ranks=ranks,
        frame_init_prepared=continuing,
        save=False,
        n_a=protocol["n_a"],
        sequence=protocol["sequence"],
        meas_slab_only=protocol["meas_slab_only"],
        batch_size=task.sample_count,
        return_data=True,
        state_representation=protocol["state_representation"],
        native_cycle_observer=observe_global_cycle,
        track_choi=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
    )
    if torch.device(model.device).type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.monotonic() - started
    if int(result.get("samples", -1)) != task.sample_count:
        raise RuntimeError("canonical engine returned the wrong sample count")
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("canonical engine did not use the physical-frame representation")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("canonical engine materialized a covariance")
    if bool(result.get("choi_tracked", False)):
        raise RuntimeError("canonical engine unexpectedly enabled Choi tracking")
    if bool(result.get("frame_init_prepared", False)) != continuing:
        raise RuntimeError("canonical engine continuation metadata is inconsistent")
    if bool(result.get("exterior_preparation_performed", False)) != (not continuing):
        raise RuntimeError("canonical engine exterior-preparation flag is inconsistent")
    expected_preparation = (
        "skipped_prepared_frame"
        if continuing
        else "born_conditioned_onsite_before_cycle_0"
    )
    if result.get("exterior_preparation") != expected_preparation:
        raise RuntimeError(
            "canonical engine used an unexpected exterior-preparation path: "
            f"{result.get('exterior_preparation')!r}"
        )
    expected_preparation_mode = "already_prepared" if continuing else "born_conditioned"
    if result.get("exterior_preparation_mode") != expected_preparation_mode:
        raise RuntimeError(
            "canonical engine used an unexpected exterior-preparation mode: "
            f"{result.get('exterior_preparation_mode')!r}"
        )
    native = result.get("native_final")
    if not isinstance(native, dict) or "frame" not in native or "ranks" not in native:
        raise RuntimeError("canonical engine did not return a native frame checkpoint")
    return native, elapsed


def _result_payload(
    *,
    observer: HardWallAllAyContourObserver,
    shard: ResultShard,
    config_sha256: str,
    elapsed_seconds: float,
    benchmark: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    task = shard.execution_batch
    payload: dict[str, Any] = dict(observer.result_payload(sample_slice=shard.local_slice))
    payload.update(
        {
            "schema": np.asarray(RESULT_SCHEMA),
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "bundle": np.asarray(BUNDLE),
            "sampling_revision": np.asarray(EXPECTED_REVISION),
            "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
            "lane": np.asarray(shard.lane),
            "task_id": np.asarray(shard.task_id),
            "execution_batch_id": np.asarray(task.task_id),
            "config_sha256": np.asarray(config_sha256),
            "Nx": np.asarray(EXPECTED_NX, dtype=np.int64),
            "Ny": np.asarray(shard.ny, dtype=np.int64),
            "cycles_total": np.asarray(shard.cycles, dtype=np.int64),
            "endpoint_only": np.asarray(True),
            "shard_index": np.asarray(shard.shard_index, dtype=np.int64),
            "sample_start": np.asarray(shard.sample_start, dtype=np.int64),
            "sample_stop": np.asarray(shard.sample_stop, dtype=np.int64),
            "global_sample_indices": np.asarray(
                shard.global_sample_indices, dtype=np.int64
            ),
            "execution_batch_seed": np.asarray(task.seed, dtype=np.int64),
            "elapsed_execution_batch_seconds": np.asarray(
                elapsed_seconds, dtype=np.float64
            ),
            "benchmark_gpu_name": np.asarray(str(benchmark["gpu_name"])),
            "benchmark_selected_matrix_batch_size": np.asarray(
                int(benchmark["selected_matrix_batch_size"]), dtype=np.int64
            ),
            "benchmark_projected_20_trajectory_seconds": np.asarray(
                float(benchmark["projected_20_trajectory_seconds"]), dtype=np.float64
            ),
            "eigensolver_backend": np.asarray(str(benchmark["eigensolver_backend"])),
            "wall_locations": np.asarray((5, 15), dtype=np.int64),
            "dtype": np.asarray("complex128"),
            "init_mode": np.asarray("default"),
            "sequence": np.asarray("raster_y"),
            "state_representation": np.asarray("physical_frame"),
            "cycle_zero_semantics": np.asarray(
                "after_born_conditioned_exterior_preparation"
            ),
        }
    )
    return _coerce_npz_payload(payload)


def save_result_shard(
    *,
    output_root: Path,
    scratch_root: Path,
    shard: ResultShard,
    observer: HardWallAllAyContourObserver,
    elapsed_seconds: float,
    config_sha256: str,
    hashes: dict[str, str],
    benchmark: Mapping[str, Any],
) -> None:
    task_scratch = scratch_root / shard.execution_batch.task_id / "results"
    task_scratch.mkdir(parents=True, exist_ok=True)
    local_result = task_scratch / f"{shard.task_id}.npz"
    payload = _result_payload(
        observer=observer,
        shard=shard,
        config_sha256=config_sha256,
        elapsed_seconds=elapsed_seconds,
        benchmark=benchmark,
    )
    _write_npz(local_result, payload, compressed=True)
    result_path, completion_path = result_paths(output_root, shard)
    published = publish_file(local_result, result_path)
    completion = _completion_identity(
        shard=shard, config_sha256=config_sha256, hashes=hashes
    )
    completion.update(
        {
            "result_filename": result_path.name,
            "result_bytes": published["bytes"],
            "result_sha256": published["sha256"],
            "elapsed_seconds": float(elapsed_seconds),
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_completion = task_scratch / f"{shard.task_id}.complete.json"
    _write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    valid, reason = verified_complete(
        output_root=output_root,
        shard=shard,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    if not valid:
        raise OSError(f"published result shard failed final verification: {reason}")


def execute_batch(
    *,
    model: Any,
    config: dict[str, Any],
    output_root: Path,
    scratch_root: Path,
    task: ExecutionBatch,
    config_sha256: str,
    hashes: dict[str, str],
    benchmark: dict[str, Any],
    on_shard_complete: Callable[[ResultShard], None] | None = None,
) -> float:
    task_scratch = scratch_root / task.task_id
    if task_scratch.exists():
        shutil.rmtree(task_scratch)
    task_scratch.mkdir(parents=True)

    checkpoint, checkpoint_reason = load_checkpoint(
        output_root=output_root,
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    observer = _new_observer(task)
    if checkpoint is None:
        print(f"[checkpoint] {task.task_id}: {checkpoint_reason}; starting at cycle 0", flush=True)
        completed_cycle = 0
        elapsed_seconds = 0.0
        frame = None
        ranks = None
        continuation_rng = None
        _seed_rng(task.seed)
    else:
        completed_cycle = checkpoint.completed_cycle
        elapsed_seconds = checkpoint.elapsed_seconds
        frame = checkpoint.frame
        ranks = checkpoint.ranks
        continuation_rng = checkpoint.rng_payload
        observer.restore_checkpoint(checkpoint.observer_payload)
        observer.validate(
            require_dynamics=completed_cycle == task.cycles,
            require_endpoint=False,
        )
        print(
            f"[checkpoint] {task.task_id}: verified cycle {completed_cycle}/{task.cycles}",
            flush=True,
        )

    cycle_bar = tqdm(
        total=task.cycles,
        initial=completed_cycle,
        desc=f"Ny={task.ny} execution {task.batch_index + 1}",
        unit="cycle",
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
    )
    cycle_bar.set_postfix(durable_cycle=completed_cycle, refresh=True)

    def mark_cycle_complete(_: int) -> None:
        cycle_bar.update(1)

    try:
        while completed_cycle < task.cycles:
            if completed_cycle:
                if continuation_rng is None:
                    raise RuntimeError("continuation is missing its captured RNG state")
                _restore_rng_state(continuation_rng)
            native, elapsed = _run_segment(
                model=model,
                config=config,
                task=task,
                observer=observer,
                segment_start=completed_cycle,
                frame=frame,
                ranks=ranks,
                on_cycle_complete=mark_cycle_complete,
            )
            completed_cycle += min(EXPECTED_SEGMENT_CYCLES, task.cycles - completed_cycle)
            elapsed_seconds += elapsed
            frame = np.asarray(native["frame"])
            ranks = np.asarray(native["ranks"], dtype=np.int64)
            continuation_rng = save_checkpoint(
                output_root=output_root,
                scratch_root=scratch_root,
                task=task,
                completed_cycle=completed_cycle,
                elapsed_seconds=elapsed_seconds,
                native_state=native,
                observer=observer,
                config_sha256=config_sha256,
                hashes=hashes,
            )
            cycle_bar.set_postfix(durable_cycle=completed_cycle, refresh=True)
    finally:
        cycle_bar.close()

    if frame is None or ranks is None or completed_cycle != task.cycles:
        raise RuntimeError("dynamics did not produce a final occupied frame")
    final_digest = final_checkpoint_sha256(output_root, task)
    progress_observer = HardWallAllAyContourObserver(
        nx=EXPECTED_NX,
        ny=task.ny,
        physical_cycles=task.cycles,
        sample_ids=np.asarray(task.global_sample_indices, dtype=np.int64),
        occupation_tolerance=1.0e-8,
    )
    progress_ok, progress_reason = load_endpoint_progress(
        output_root=output_root,
        task=task,
        observer=progress_observer,
        config_sha256=config_sha256,
        hashes=hashes,
        final_checkpoint_sha256_value=final_digest,
    )
    if progress_ok:
        observer = progress_observer
        print(
            f"[endpoint resume] {task.task_id}: {progress_reason}; "
            f"{int(observer.endpoint_width_seen.sum())}/{len(observer.ay_values)} widths durable",
            flush=True,
        )
    else:
        print(
            f"[endpoint resume] {task.task_id}: {progress_reason}; starting Ay=0",
            flush=True,
        )
    endpoint_frame = torch.as_tensor(
        frame,
        dtype=torch.complex128,
        device=torch.device(config["device"]),
    )
    validate_padded_frame_ranks(endpoint_frame, ranks)
    batch_map = benchmark["matrix_batch_size_by_Ay"]
    initial_widths = int(observer.endpoint_width_seen.sum())
    width_bar = tqdm(
        total=len(observer.ay_values),
        initial=initial_widths,
        desc=f"Ny={task.ny} endpoint",
        unit="Ay",
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
    )
    try:
        for ay in observer.ay_values:
            ay = int(ay)
            width_index = int(np.where(observer.ay_values == ay)[0][0])
            if observer.endpoint_width_seen[width_index]:
                continue
            matrix_batch_size = int(batch_map[str(ay)])
            processed_pairs = 0

            def pair_progress(count: int) -> None:
                nonlocal processed_pairs
                processed_pairs += int(count)
                width_bar.set_postfix(
                    pairs=f"{processed_pairs}/{task.sample_count * task.ny if ay else 0}",
                    matrix_batch=matrix_batch_size,
                    last_durable_Ay=(
                        int(observer.ay_values[np.flatnonzero(observer.endpoint_width_seen)[-1]])
                        if observer.endpoint_width_seen.any()
                        else -1
                    ),
                    refresh=False,
                )

            print(
                f"[stage] {task.task_id}: endpoint Ay={ay}/{task.ny // 2}, "
                f"matrix_batch={matrix_batch_size}",
                flush=True,
            )
            started = time.monotonic()
            observer.record_endpoint_width(
                endpoint_frame,
                ay=ay,
                matrix_batch_size=matrix_batch_size,
                elapsed_seconds=0.0,
                pair_progress=pair_progress,
            )
            if endpoint_frame.device.type == "cuda":
                torch.cuda.synchronize(endpoint_frame.device)
            observer.endpoint_seconds_by_ay[width_index] = time.monotonic() - started
            save_endpoint_progress(
                output_root=output_root,
                scratch_root=scratch_root,
                task=task,
                observer=observer,
                config_sha256=config_sha256,
                hashes=hashes,
                final_checkpoint_sha256_value=final_digest,
            )
            width_bar.update(1)
            width_bar.set_postfix(
                pairs=processed_pairs,
                matrix_batch=matrix_batch_size,
                last_durable_Ay=ay,
                refresh=True,
            )
    finally:
        width_bar.close()
    del endpoint_frame
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    observer.validate(require_dynamics=True, require_endpoint=True)
    for shard in result_shards(task):
        already_complete, _ = verified_complete(
            output_root=output_root,
            shard=shard,
            config_sha256=config_sha256,
            hashes=hashes,
        )
        if already_complete:
            continue
        save_result_shard(
            output_root=output_root,
            scratch_root=scratch_root,
            shard=shard,
            observer=observer,
            elapsed_seconds=elapsed_seconds,
            config_sha256=config_sha256,
            hashes=hashes,
            benchmark=benchmark,
        )
        if on_shard_complete is not None:
            on_shard_complete(shard)
    statuses = [
        verified_complete(
            output_root=output_root,
            shard=shard,
            config_sha256=config_sha256,
            hashes=hashes,
        )[0]
        for shard in result_shards(task)
    ]
    if not all(statuses):
        raise OSError("not every five-sample result shard passed final readback")
    remove_checkpoint(output_root, task)
    shutil.rmtree(task_scratch)
    return elapsed_seconds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--lane", choices=("A", "B"), required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--max-new-execution-batches", type=int)
    args = parser.parse_args(argv)
    if args.max_new_execution_batches is not None and args.max_new_execution_batches < 0:
        parser.error("--max-new-execution-batches must be nonnegative")
    return args


def run_campaign(
    *,
    config: dict[str, Any],
    lane: str,
    output_root: Path,
    scratch_root: Path,
    report_only: bool = False,
    benchmark_only: bool = False,
    max_new_execution_batches: int | None = None,
) -> dict[str, Any]:
    config = validate_config(config)
    lane = str(lane).upper()
    tasks = expand_execution_batches(config, lane=lane)
    hashes = source_hashes()
    config_sha256 = config_hash(config)
    shard_status: dict[str, tuple[bool, str]] = {}
    for task in tasks:
        for shard in result_shards(task):
            shard_status[shard.task_id] = verified_complete(
                output_root=output_root,
                shard=shard,
                config_sha256=config_sha256,
                hashes=hashes,
            )
    completed_shards = sum(int(status[0]) for status in shard_status.values())
    invalid_shards = sum(
        int((not status[0]) and status[1] != "missing result/completion pair")
        for status in shard_status.values()
    )
    complete_task_ids = {
        task.task_id
        for task in tasks
        if all(shard_status[shard.task_id][0] for shard in result_shards(task))
    }
    pending_tasks = [task for task in tasks if task.task_id not in complete_task_ids]
    total_shards = sum(len(result_shards(task)) for task in tasks)
    recoverable_dynamics = 0
    recoverable_endpoint = 0
    for task in pending_tasks:
        checkpoint, _ = load_checkpoint(
            output_root=output_root,
            task=task,
            config_sha256=config_sha256,
            hashes=hashes,
        )
        if checkpoint is not None:
            recoverable_dynamics += 1
            if checkpoint.completed_cycle == task.cycles:
                endpoint_npz, endpoint_json = endpoint_progress_paths(output_root, task)
                if endpoint_npz.is_file() and endpoint_json.is_file():
                    recoverable_endpoint += 1
    print(
        json.dumps(
            {
                "bundle": BUNDLE,
                "lane": lane,
                "sampling_revision": EXPECTED_REVISION,
                "canonical_entry_point": CANONICAL_ENTRY_POINT,
                "configuration": config,
                "config_sha256": config_sha256,
                "source_hashes": hashes,
                "output_root": str(output_root),
                "scratch_root": str(scratch_root),
                "workload": {
                    "Ny_values": list(LANE_NY_VALUES[lane]),
                    "trajectories": EXPECTED_SAMPLES * len(LANE_NY_VALUES[lane]),
                    "execution_batches": len(tasks),
                    "completed_execution_batches": len(complete_task_ids),
                    "pending_execution_batches": len(pending_tasks),
                    "result_shards": total_shards,
                    "completed_result_shards": completed_shards,
                    "invalid_or_partial_result_shards": invalid_shards,
                    "recoverable_dynamics_checkpoints": recoverable_dynamics,
                    "recoverable_endpoint_checkpoints": recoverable_endpoint,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if report_only:
        return {
            "status": "report_only",
            "lane": lane,
            "total_execution_batches": len(tasks),
            "completed_execution_batches": len(complete_task_ids),
            "pending_execution_batches": len(pending_tasks),
            "total_result_shards": total_shards,
            "completed_result_shards": completed_shards,
            "invalid_or_partial_result_shards": invalid_shards,
        }
    if not benchmark_only and (max_new_execution_batches == 0 or not pending_tasks):
        status = "partial_limit_reached" if pending_tasks else "complete"
        summary = {
            "status": status,
            "lane": lane,
            "total_execution_batches": len(tasks),
            "completed_execution_batches": len(complete_task_ids),
            "pending_execution_batches": len(pending_tasks),
            "new_execution_batches": 0,
            "failed": 0,
        }
        print("[campaign summary] " + json.dumps(summary, sort_keys=True), flush=True)
        return summary

    gpu = validate_a100()
    local_free = _check_space(scratch_root, required_bytes=12 * 1024**3, label="local")
    drive_free = _check_space(output_root, required_bytes=10 * 1024**3, label="Drive")
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

    benchmark = run_a100_endpoint_benchmark(
        output_root=output_root,
        scratch_root=scratch_root,
        lane=lane,
        config_sha256=config_sha256,
        hashes=hashes,
        gpu=gpu,
    )
    if benchmark_only:
        return {
            "status": "benchmark_accepted",
            "lane": lane,
            "projected_20_trajectory_seconds": benchmark[
                "projected_20_trajectory_seconds"
            ],
            "selected_matrix_batch_size": benchmark["selected_matrix_batch_size"],
        }

    for task in tasks:
        if task.task_id in complete_task_ids:
            remove_checkpoint(output_root, task)

    completed_before = len(complete_task_ids)
    new_completed = 0
    failed = 0
    current_ny: int | None = None
    model: Any | None = None
    elapsed_session: list[float] = []
    durable_shards = completed_shards
    bar = tqdm(
        total=total_shards,
        initial=completed_shards,
        desc=f"hard-wall lane {lane}",
        unit="shard",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )
    bar.set_postfix(
        completed=durable_shards,
        skipped=completed_shards,
        pending=total_shards - durable_shards,
        failed=failed,
    )

    def mark_shard_complete(_: ResultShard) -> None:
        nonlocal durable_shards
        durable_shards += 1
        bar.update(1)
        bar.set_postfix(
            completed=durable_shards,
            skipped=completed_shards,
            pending=total_shards - durable_shards,
            failed=failed,
            refresh=True,
        )

    try:
        for task in pending_tasks:
            if (
                max_new_execution_batches is not None
                and new_completed >= max_new_execution_batches
            ):
                break
            if task.ny != current_ny:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = build_model(config, ny=task.ny)
                current_ny = task.ny
            bar.set_description(
                f"lane {lane} Ny={task.ny} samples "
                f"{task.sample_start:03d}-{task.sample_stop - 1:03d}"
            )
            try:
                elapsed = execute_batch(
                    model=model,
                    config=config,
                    output_root=output_root,
                    scratch_root=scratch_root,
                    task=task,
                    config_sha256=config_sha256,
                    hashes=hashes,
                    benchmark=benchmark,
                    on_shard_complete=mark_shard_complete,
                )
            except Exception:
                failed += 1
                bar.set_postfix(
                    completed=durable_shards,
                    skipped=completed_shards,
                    pending=total_shards - durable_shards,
                    failed=failed,
                    refresh=True,
                )
                raise
            elapsed_session.append(elapsed)
            new_completed += 1
            remaining = len(tasks) - completed_before - new_completed
            bar.set_postfix(
                completed=durable_shards,
                skipped=completed_shards,
                pending=total_shards - durable_shards,
                failed=failed,
                refresh=True,
            )
            projected = remaining * float(np.mean(elapsed_session)) / 3600.0
            print(
                f"[execution complete] {task.task_id}: {elapsed / 60:.1f} min; "
                f"session-mean projected remaining={projected:.1f} A100 h",
                flush=True,
            )
    finally:
        bar.close()

    completed_total = completed_before + new_completed
    pending_total = len(tasks) - completed_total
    status = "complete" if pending_total == 0 else "partial_limit_reached"
    summary = {
        "status": status,
        "lane": lane,
        "total_execution_batches": len(tasks),
        "completed_execution_batches": completed_total,
        "pending_execution_batches": pending_total,
        "new_execution_batches": new_completed,
        "completed_result_shards": durable_shards,
        "pending_result_shards": total_shards - durable_shards,
        "failed": failed,
    }
    print("[campaign summary] " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    print("[startup] hard-wall all-Ay entropy-contour runner entered", flush=True)
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_campaign(
        config=config,
        lane=args.lane,
        output_root=args.output_root.resolve(),
        scratch_root=args.scratch_root.resolve(),
        report_only=args.report_only,
        benchmark_only=args.benchmark_only,
        max_new_execution_batches=args.max_new_execution_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
