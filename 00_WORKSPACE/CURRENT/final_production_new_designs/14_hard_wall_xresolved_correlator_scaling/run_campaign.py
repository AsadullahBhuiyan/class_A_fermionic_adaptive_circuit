"""Run the S100 large-Ny hard-wall x-resolved correlator campaign."""

from __future__ import annotations

import argparse
import gc
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

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BUNDLE_ROOT / "src"
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from endpoint_correlator import (  # noqa: E402
    OBSERVER_SCHEMA,
    endpoint_result_payload,
    validate_endpoint_payload,
)


BUNDLE = "14_hard_wall_xresolved_correlator_scaling"
EXPECTED_REVISION = (
    "hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_raster_"
    "endpoint_frame_halfcov_occupations_v2_30gib_batched"
)
EXPECTED_ROOT_SEED = 2026091001
EXPECTED_NX = 20
EXPECTED_NY_VALUES = (60, 50, 40)
EXPECTED_SAMPLES = 100
EXPECTED_RESULT_SHARD_SIZE = 5
EXPECTED_SEGMENT_CYCLES = 5
EXPECTED_EXECUTION_BATCH_SIZES = {40: 80, 50: 50, 60: 40}
EXPECTED_EXECUTION_BATCH_COUNTS = {40: 2, 50: 2, 60: 3}
RESULT_SCHEMA = "hard_wall_xresolved_endpoint_frame_halfcov_result_v2"
COMPLETION_SCHEMA = "hard_wall_xresolved_endpoint_frame_halfcov_completion_v2"
CHECKPOINT_SCHEMA = "hard_wall_xresolved_dynamics_checkpoint_v2"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
SOURCE_FILES = (
    "run_campaign.py",
    "endpoint_correlator.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)
GPU_MEMORY_HARD_LIMIT_GIB = 30.0
GPU_MEMORY_HARD_LIMIT_BYTES = int(GPU_MEMORY_HARD_LIMIT_GIB * 1024**3)
MINIMUM_DRIVE_HEADROOM_BYTES = 5 * 1024**3
MINIMUM_LOCAL_FREE_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class ExecutionBatch:
    ny: int
    batch_index: int
    sample_start: int
    sample_stop: int
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

    @property
    def task_id(self) -> str:
        return (
            f"Ny{self.ny:03d}_execution-{self.batch_index:03d}_samples-"
            f"{self.sample_start:03d}-{self.sample_stop - 1:03d}"
        )


@dataclass(frozen=True)
class ResultShard:
    execution_batch: ExecutionBatch
    shard_index: int
    sample_start: int
    sample_stop: int

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
    def local_slice(self) -> slice:
        start = self.sample_start - self.execution_batch.sample_start
        stop = self.sample_stop - self.execution_batch.sample_start
        return slice(start, stop)

    @property
    def task_id(self) -> str:
        return (
            f"Ny{self.ny:03d}_shard-{self.shard_index:03d}_samples-"
            f"{self.sample_start:03d}-{self.sample_stop - 1:03d}"
        )


@dataclass(frozen=True)
class Checkpoint:
    completed_cycle: int
    elapsed_seconds: float
    frame: np.ndarray
    ranks: np.ndarray
    rng_payload: dict[str, np.ndarray]


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


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
        "samples_per_Ny": EXPECTED_SAMPLES,
        "execution_batch_size_by_Ny": {
            str(key): value for key, value in EXPECTED_EXECUTION_BATCH_SIZES.items()
        },
        "result_shard_size": EXPECTED_RESULT_SHARD_SIZE,
        "cycles_rule": "2*Ny",
        "segment_cycles": EXPECTED_SEGMENT_CYCLES,
        "device": "cuda:0",
        "dtype": "complex128",
        "gpu_memory_hard_limit_gib": GPU_MEMORY_HARD_LIMIT_GIB,
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
            "frame_reorthonormalize_interval": 1,
        },
        "observables": {
            "endpoint_only": True,
            "cycles": "[2*Ny]",
            "x_resolved_square_correlator": True,
            "legacy_xavg_square_correlator": True,
            "global_charge": True,
            "half_filling_offset": True,
            "occupied_frame": True,
            "occupied_ranks": True,
            "half_system_covariance": True,
            "half_system_occupation_spectrum": True,
            "half_system_region": "[0,Nx)x[0,Ny//2)",
            "half_system_Ay_rule": "Ny//2",
            "half_system_covariance_convention": "C_A=F_A@F_A_dagger",
            "frame_covariance_convention": "C=F@F_dagger;G=2C-I",
            "full_system_covariance_materialization": False,
            "endpoint_reduced_covariance_materialization": True,
        },
    }


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(config)
    expected = expected_config()
    if observed != expected:
        differences = {
            key: {"expected": value, "observed": observed.get(key)}
            for key, value in expected.items()
            if observed.get(key) != value
        }
        differences.update(
            {
                key: {"expected": "absent", "observed": value}
                for key, value in observed.items()
                if key not in expected
            }
        )
        raise ValueError(
            "configuration differs from the locked production contract: "
            + json.dumps(differences, sort_keys=True)
        )
    return observed


def _task_seed(config: Mapping[str, Any], identity: str) -> int:
    digest = hashlib.sha256(
        f"{config['root_seed']}|{config['sampling_revision']}|{identity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def expand_execution_batches(config: Mapping[str, Any]) -> list[ExecutionBatch]:
    config = validate_config(config)
    tasks: list[ExecutionBatch] = []
    for ny in EXPECTED_NY_VALUES:
        batch_size = EXPECTED_EXECUTION_BATCH_SIZES[ny]
        for batch_index, sample_start in enumerate(
            range(0, EXPECTED_SAMPLES, batch_size)
        ):
            sample_stop = min(EXPECTED_SAMPLES, sample_start + batch_size)
            identity = f"hard|{ny}|{batch_index}|{sample_start}|{sample_stop}"
            tasks.append(
                ExecutionBatch(
                    ny=ny,
                    batch_index=batch_index,
                    sample_start=sample_start,
                    sample_stop=sample_stop,
                    seed=_task_seed(config, identity),
                )
            )
    if len(tasks) != 7 or len({task.task_id for task in tasks}) != 7:
        raise RuntimeError("execution expansion did not produce 7 unique batches")
    if len({task.seed for task in tasks}) != len(tasks):
        raise RuntimeError("execution batch seeds are not unique")
    counts = {ny: sum(task.ny == ny for task in tasks) for ny in EXPECTED_NY_VALUES}
    if counts != {ny: EXPECTED_EXECUTION_BATCH_COUNTS[ny] for ny in EXPECTED_NY_VALUES}:
        raise RuntimeError(f"execution batch counts are wrong: {counts}")
    return tasks


def result_shards(task: ExecutionBatch) -> list[ResultShard]:
    shards: list[ResultShard] = []
    for sample_start in range(
        task.sample_start, task.sample_stop, EXPECTED_RESULT_SHARD_SIZE
    ):
        sample_stop = min(task.sample_stop, sample_start + EXPECTED_RESULT_SHARD_SIZE)
        if sample_stop - sample_start != EXPECTED_RESULT_SHARD_SIZE:
            raise RuntimeError("every durable result shard must contain five samples")
        shards.append(
            ResultShard(
                execution_batch=task,
                shard_index=sample_start // EXPECTED_RESULT_SHARD_SIZE,
                sample_start=sample_start,
                sample_stop=sample_stop,
            )
        )
    return shards


def all_result_shards(tasks: list[ExecutionBatch]) -> list[ResultShard]:
    shards = [shard for task in tasks for shard in result_shards(task)]
    slots = {
        (shard.ny, index) for shard in shards for index in shard.global_sample_indices
    }
    if len(shards) != 60 or len(slots) != 300:
        raise RuntimeError(
            "campaign must contain 60 shards and 300 unique sample slots"
        )
    if len({shard.task_id for shard in shards}) != 60:
        raise RuntimeError("result shard task IDs are not unique")
    return shards


def result_paths(output_root: Path, shard: ResultShard) -> tuple[Path, Path]:
    directory = output_root / "results" / f"Ny{shard.ny:03d}"
    stem = (
        f"shard_{shard.shard_index:03d}_samples_"
        f"{shard.sample_start:03d}-{shard.sample_stop - 1:03d}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def checkpoint_paths(output_root: Path, task: ExecutionBatch) -> tuple[Path, Path]:
    directory = output_root / "checkpoints" / task.task_id
    return directory / "checkpoint.npz", directory / "checkpoint.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_npz(path: Path, payload: Mapping[str, Any], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            if compressed:
                np.savez_compressed(handle, **payload)
            else:
                np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    """Copy, read back, atomically rename, and read back through DriveFS."""

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    expected_bytes = int(local_path.stat().st_size)
    expected_sha256 = sha256_file(local_path)
    try:
        shutil.copyfile(local_path, temporary)
        if int(temporary.stat().st_size) != expected_bytes:
            raise OSError(f"Drive temporary byte-count mismatch: {temporary}")
        if sha256_file(temporary) != expected_sha256:
            raise OSError(f"Drive temporary checksum mismatch: {temporary}")
        os.replace(temporary, final_path)
        if int(final_path.stat().st_size) != expected_bytes:
            raise OSError(f"Drive final byte-count mismatch: {final_path}")
        if sha256_file(final_path) != expected_sha256:
            raise OSError(f"Drive final checksum mismatch: {final_path}")
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


def _execution_identity(
    task: ExecutionBatch,
    *,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "execution_batch_id": task.task_id,
        "Nx": EXPECTED_NX,
        "Ny": task.ny,
        "execution_batch_index": task.batch_index,
        "execution_batch_seed": task.seed,
        "execution_sample_indices": list(task.global_sample_indices),
        "cycles": task.cycles,
        "configuration_sha256": configuration_sha256,
        "source_hashes": dict(hashes),
    }


def _shard_identity(
    shard: ResultShard,
    *,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        **_execution_identity(
            shard.execution_batch,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        ),
        "task_id": shard.task_id,
        "shard_index": shard.shard_index,
        "global_sample_indices": list(shard.global_sample_indices),
        "sample_count": shard.sample_count,
    }


def _checkpoint_identity(
    task: ExecutionBatch,
    *,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint",
        "segment_cycles": EXPECTED_SEGMENT_CYCLES,
        **_execution_identity(
            task, configuration_sha256=configuration_sha256, hashes=hashes
        ),
    }


def _completion_identity(
    shard: ResultShard,
    *,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        **_shard_identity(
            shard, configuration_sha256=configuration_sha256, hashes=hashes
        ),
    }


def _validate_result_npz(path: Path, shard: ResultShard) -> None:
    with np.load(path, allow_pickle=False) as archive:
        required_metadata = {
            "schema",
            "bundle",
            "sampling_revision",
            "canonical_entry_point",
            "task_id",
            "execution_batch_id",
            "execution_batch_seed",
            "Nx",
            "Ny",
            "alpha_1",
            "alpha_2",
            "nshell",
            "dtype",
            "sequence",
            "perfect_correction",
            "dw_truncation",
            "meas_slab_only",
        }
        missing = sorted(required_metadata - set(archive.files))
        if missing:
            raise ValueError(f"result NPZ missing metadata: {missing}")
        expected_scalars = {
            "schema": RESULT_SCHEMA,
            "bundle": BUNDLE,
            "sampling_revision": EXPECTED_REVISION,
            "canonical_entry_point": CANONICAL_ENTRY_POINT,
            "task_id": shard.task_id,
            "execution_batch_id": shard.execution_batch.task_id,
            "execution_batch_seed": shard.execution_batch.seed,
            "Nx": EXPECTED_NX,
            "Ny": shard.ny,
            "alpha_1": 1.0,
            "alpha_2": 30.0,
            "nshell": 1,
            "dtype": "complex128",
            "sequence": "raster_y",
            "perfect_correction": True,
            "dw_truncation": True,
            "meas_slab_only": True,
        }
        for key, expected in expected_scalars.items():
            if np.asarray(archive[key]).item() != expected:
                raise ValueError(f"result metadata mismatch: {key}")
        validate_endpoint_payload(
            {key: archive[key] for key in archive.files},
            nx=EXPECTED_NX,
            ny=shard.ny,
            global_sample_indices=np.asarray(shard.global_sample_indices),
        )


def verified_complete(
    *,
    output_root: Path,
    shard: ResultShard,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = result_paths(output_root, shard)
    result_exists = result_path.is_file()
    completion_exists = completion_path.is_file()
    if not result_exists and not completion_exists:
        return False, "missing result and completion"
    if not result_exists or not completion_exists:
        return False, "incomplete result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion JSON: {exc}"
    for key, expected in _completion_identity(
        shard, configuration_sha256=configuration_sha256, hashes=hashes
    ).items():
        if completion.get(key) != expected:
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
    try:
        _validate_result_npz(result_path, shard)
    except (OSError, ValueError, KeyError, FloatingPointError) as exc:
        return False, f"result payload invalid: {exc}"
    return True, "verified"


def _capture_rng_state() -> dict[str, np.ndarray]:
    numpy_state = np.random.get_state()
    payload: dict[str, np.ndarray] = {
        "numpy_algorithm": np.asarray(numpy_state[0]),
        "numpy_keys": np.asarray(numpy_state[1], dtype=np.uint32),
        "numpy_position": np.asarray(numpy_state[2], dtype=np.int64),
        "numpy_has_gauss": np.asarray(numpy_state[3], dtype=np.int8),
        "numpy_cached_gaussian": np.asarray(numpy_state[4], dtype=np.float64),
        "torch_cpu": torch.get_rng_state().detach().cpu().numpy(),
    }
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload["torch_cuda_count"] = np.asarray(len(cuda_states), dtype=np.int64)
    for index, cuda_state in enumerate(cuda_states):
        payload[f"torch_cuda_{index}"] = cuda_state.detach().cpu().numpy()
    return payload


def _restore_rng_state(payload: Mapping[str, np.ndarray]) -> None:
    np.random.set_state(
        (
            str(np.asarray(payload["numpy_algorithm"]).item()),
            np.asarray(payload["numpy_keys"], dtype=np.uint32),
            int(np.asarray(payload["numpy_position"]).item()),
            int(np.asarray(payload["numpy_has_gauss"]).item()),
            float(np.asarray(payload["numpy_cached_gaussian"]).item()),
        )
    )
    torch.set_rng_state(
        torch.as_tensor(np.asarray(payload["torch_cpu"], dtype=np.uint8))
    )
    saved_count = int(np.asarray(payload["torch_cuda_count"]).item())
    current_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_count != current_count:
        raise RuntimeError(
            f"checkpoint CUDA RNG count mismatch: {saved_count} != {current_count}"
        )
    if saved_count:
        torch.cuda.set_rng_state_all(
            [
                torch.as_tensor(
                    np.asarray(payload[f"torch_cuda_{index}"], dtype=np.uint8)
                )
                for index in range(saved_count)
            ]
        )


def _seed_rng(seed: int) -> None:
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _native_arrays(native: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    frame_value, rank_value = native["frame"], native["ranks"]
    frame = (
        frame_value.detach().cpu().numpy()
        if torch.is_tensor(frame_value)
        else np.asarray(frame_value)
    )
    ranks = (
        rank_value.detach().cpu().numpy()
        if torch.is_tensor(rank_value)
        else np.asarray(rank_value)
    )
    return np.asarray(frame, dtype=np.complex128), np.asarray(ranks, dtype=np.int64)


def save_checkpoint(
    *,
    output_root: Path,
    scratch_root: Path,
    task: ExecutionBatch,
    completed_cycle: int,
    elapsed_seconds: float,
    native: Mapping[str, Any],
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> dict[str, np.ndarray]:
    frame, ranks = _native_arrays(native)
    expected_prefix = (task.sample_count, 2 * EXPECTED_NX * task.ny)
    if frame.ndim != 3 or frame.shape[:2] != expected_prefix:
        raise ValueError(f"unexpected checkpoint frame shape {frame.shape}")
    if frame.dtype != np.complex128:
        raise TypeError(f"checkpoint frame must be complex128, got {frame.dtype}")
    if (
        ranks.shape != (task.sample_count,)
        or np.any(ranks < 0)
        or np.any(ranks > frame.shape[2])
    ):
        raise ValueError("unexpected checkpoint ranks")
    rng_payload = _capture_rng_state()
    payload: dict[str, Any] = {
        "completed_cycle": np.asarray(completed_cycle, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "frame": frame,
        "ranks": ranks,
        "sample_indices": np.asarray(task.global_sample_indices, dtype=np.int64),
    }
    payload.update({f"rng__{key}": value for key, value in rng_payload.items()})

    task_scratch = scratch_root / task.task_id
    task_scratch.mkdir(parents=True, exist_ok=True)
    local_npz = task_scratch / "checkpoint.npz"
    _write_npz(local_npz, payload, compressed=False)
    final_npz, final_json = checkpoint_paths(output_root, task)
    published = publish_file(local_npz, final_npz)

    metadata = _checkpoint_identity(
        task, configuration_sha256=configuration_sha256, hashes=hashes
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
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> tuple[Checkpoint | None, str]:
    npz_path, json_path = checkpoint_paths(output_root, task)
    npz_exists, json_exists = npz_path.is_file(), json_path.is_file()
    if not npz_exists and not json_exists:
        return None, "no checkpoint"
    if not npz_exists or not json_exists:
        return None, "incomplete checkpoint pair"
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable checkpoint JSON: {exc}"
    for key, expected in _checkpoint_identity(
        task, configuration_sha256=configuration_sha256, hashes=hashes
    ).items():
        if metadata.get(key) != expected:
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
            payload = {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError) as exc:
        return None, f"unreadable checkpoint NPZ: {exc}"
    required = {
        "completed_cycle",
        "elapsed_seconds",
        "frame",
        "ranks",
        "sample_indices",
    }
    missing = sorted(required - set(payload))
    if missing:
        return None, f"checkpoint NPZ missing fields: {missing}"
    completed_cycle = int(payload["completed_cycle"].item())
    if completed_cycle != int(metadata.get("completed_cycle", -1)):
        return None, "checkpoint completed-cycle mismatch"
    if not 0 < completed_cycle <= task.cycles:
        return None, "checkpoint completed cycle is out of range"
    if completed_cycle != task.cycles and completed_cycle % EXPECTED_SEGMENT_CYCLES:
        return None, "checkpoint is not on a five-cycle boundary"
    frame = np.asarray(payload["frame"])
    ranks = np.asarray(payload["ranks"])
    expected_prefix = (task.sample_count, 2 * EXPECTED_NX * task.ny)
    if frame.ndim != 3 or frame.shape[:2] != expected_prefix:
        return None, "checkpoint frame shape mismatch"
    if frame.dtype != np.complex128 or not np.isfinite(frame).all():
        return None, "checkpoint frame dtype or finiteness mismatch"
    if ranks.shape != (task.sample_count,) or ranks.dtype != np.int64:
        return None, "checkpoint rank shape/dtype mismatch"
    if np.any(ranks < 0) or np.any(ranks > frame.shape[2]):
        return None, "checkpoint ranks lie outside frame capacity"
    if not np.array_equal(
        payload["sample_indices"],
        np.asarray(task.global_sample_indices, dtype=np.int64),
    ):
        return None, "checkpoint sample indices mismatch"
    rng_payload = {
        key.removeprefix("rng__"): value
        for key, value in payload.items()
        if key.startswith("rng__")
    }
    required_rng = {
        "numpy_algorithm",
        "numpy_keys",
        "numpy_position",
        "numpy_has_gauss",
        "numpy_cached_gaussian",
        "torch_cpu",
        "torch_cuda_count",
    }
    if not required_rng.issubset(rng_payload):
        return None, "checkpoint lacks RNG state"
    return (
        Checkpoint(
            completed_cycle=completed_cycle,
            elapsed_seconds=float(payload["elapsed_seconds"].item()),
            frame=frame,
            ranks=ranks,
            rng_payload=rng_payload,
        ),
        "verified",
    )


def remove_checkpoint(output_root: Path, task: ExecutionBatch) -> None:
    npz_path, json_path = checkpoint_paths(output_root, task)
    for path in (json_path, npz_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        npz_path.parent.rmdir()
    except OSError:
        pass


def _check_space(path: Path, *, required_bytes: int, label: str) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path).free)
    if free < int(required_bytes):
        raise RuntimeError(
            f"insufficient {label} space: free={free / 1024**3:.2f} GiB, "
            f"required={required_bytes / 1024**3:.2f} GiB"
        )
    return free


def estimated_frame_result_bytes(shard: ResultShard) -> int:
    """Return uncompressed occupied-frame bytes for one shard."""

    dimension = 2 * EXPECTED_NX * shard.ny
    maximum_rank = EXPECTED_NX * shard.ny
    return (
        shard.sample_count * dimension * maximum_rank * np.dtype(np.complex128).itemsize
    )


def estimated_half_covariance_result_bytes(shard: ResultShard) -> int:
    """Return uncompressed ``C_A`` bytes for ``A=[0,Nx)x[0,Ny//2)``."""

    subsystem_modes = 2 * EXPECTED_NX * (shard.ny // 2)
    return (
        shard.sample_count
        * subsystem_modes
        * subsystem_modes
        * np.dtype(np.complex128).itemsize
    )


def estimated_result_payload_bytes(shard: ResultShard) -> int:
    """Return the dominant frame plus half-system-covariance payload bytes."""

    return estimated_frame_result_bytes(shard) + estimated_half_covariance_result_bytes(
        shard
    )


def required_drive_free_bytes(
    shards: list[ResultShard], valid_shard_ids: set[str]
) -> int:
    """Reserve missing large payloads, modest overhead, and working headroom."""

    missing_result_bytes = sum(
        estimated_result_payload_bytes(shard)
        for shard in shards
        if shard.task_id not in valid_shard_ids
    )
    return MINIMUM_DRIVE_HEADROOM_BYTES + int(np.ceil(1.15 * missing_result_bytes))


def validate_a100() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an A100 GPU runtime")
    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    total = int(properties.total_memory)
    if "A100" not in name.upper():
        raise RuntimeError(f"expected an A100 GPU, found {name!r}")
    if total < 38 * 1024**3:
        raise RuntimeError(
            f"A100 does not have 40-GB-class memory: {total / 1024**3:.2f} GiB"
        )
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, GPU_MEMORY_HARD_LIMIT_BYTES / total), device=0
    )
    return {
        "name": name,
        "total_memory_gib": total / 1024**3,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_memory_hard_limit_gib": GPU_MEMORY_HARD_LIMIT_GIB,
    }


def _gpu_peak_reserved_gib(device: torch.device | str) -> float:
    if torch.device(device).type != "cuda":
        return 0.0
    peak_bytes = int(torch.cuda.max_memory_reserved(torch.device(device)))
    if peak_bytes > GPU_MEMORY_HARD_LIMIT_BYTES:
        raise RuntimeError(
            "GPU peak reserved memory crossed the 30 GiB hard limit: "
            f"{peak_bytes / 1024**3:.2f} GiB"
        )
    return peak_bytes / 1024**3


def build_model(config: Mapping[str, Any], *, ny: int) -> Any:
    protocol = config["protocol"]
    print(f"[model] constructing hard wall at Nx={EXPECTED_NX}, Ny={ny}", flush=True)
    model = classA_U1FGTN_gpu(
        Nx=EXPECTED_NX,
        Ny=int(ny),
        DW=protocol["DW"],
        nshell=protocol["nshell"],
        filling_frac=protocol["filling_frac"],
        alpha_1=protocol["alpha_1"],
        alpha_2=protocol["alpha_2"],
        trial_orbitals=protocol["trial_orbitals"],
        dw_truncation=protocol["dw_truncation"],
        triv_region_local_mode=protocol["triv_region_local_mode"],
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


def _run_segment(
    *,
    model: Any,
    config: Mapping[str, Any],
    task: ExecutionBatch,
    segment_start: int,
    frame: np.ndarray | None,
    ranks: np.ndarray | None,
    on_cycle_complete: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], float]:
    segment_cycles = min(EXPECTED_SEGMENT_CYCLES, task.cycles - segment_start)
    if segment_cycles <= 0:
        raise ValueError("segment start is at or beyond the final cycle")

    def observe_progress(*, cycle: int, **_: Any) -> None:
        local_cycle = int(cycle)
        if local_cycle > 0 and on_cycle_complete is not None:
            on_cycle_complete(segment_start + local_cycle)

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
        native_cycle_observer=observe_progress,
        track_choi=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=protocol["frame_reorthonormalize_interval"],
    )
    if torch.device(model.device).type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.monotonic() - started
    if int(result.get("samples", -1)) != task.sample_count:
        raise RuntimeError("canonical engine returned the wrong sample count")
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("canonical engine did not use physical-frame state")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("canonical engine materialized a covariance")
    if bool(result.get("frame_init_prepared", False)) != continuing:
        raise RuntimeError("canonical continuation metadata is inconsistent")
    if bool(result.get("exterior_preparation_performed", False)) != (not continuing):
        raise RuntimeError("hard-wall exterior preparation count is inconsistent")
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
    native = result.get("native_final")
    if not isinstance(native, dict) or "frame" not in native or "ranks" not in native:
        raise RuntimeError("canonical engine did not return a native frame checkpoint")
    return native, elapsed


def run_dynamics(
    *,
    model: Any,
    config: Mapping[str, Any],
    output_root: Path,
    scratch_root: Path,
    task: ExecutionBatch,
    configuration_sha256: str,
    hashes: Mapping[str, str],
    show_progress: bool = True,
    after_checkpoint: Callable[[int], None] | None = None,
) -> Checkpoint:
    checkpoint, reason = load_checkpoint(
        output_root=output_root,
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    if checkpoint is None:
        print(f"[checkpoint] {task.task_id}: {reason}; starting at cycle 0", flush=True)
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
        print(
            f"[checkpoint] {task.task_id}: verified cycle "
            f"{completed_cycle}/{task.cycles}",
            flush=True,
        )

    if completed_cycle == task.cycles:
        return checkpoint  # type: ignore[return-value]

    cycle_bar = tqdm(
        total=task.cycles,
        initial=completed_cycle,
        desc=f"Ny={task.ny} execution {task.batch_index + 1}",
        unit="cycle",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
        file=sys.stdout,
    )
    cycle_bar.set_postfix(durable_cycle=completed_cycle, refresh=show_progress)

    def mark_cycle_complete(_: int) -> None:
        cycle_bar.update(1)

    try:
        while completed_cycle < task.cycles:
            if completed_cycle:
                if continuation_rng is None:
                    raise RuntimeError("continuation is missing captured RNG state")
                _restore_rng_state(continuation_rng)
            native, elapsed = _run_segment(
                model=model,
                config=config,
                task=task,
                segment_start=completed_cycle,
                frame=frame,
                ranks=ranks,
                on_cycle_complete=mark_cycle_complete,
            )
            completed_cycle += min(
                EXPECTED_SEGMENT_CYCLES, task.cycles - completed_cycle
            )
            elapsed_seconds += elapsed
            frame, ranks = _native_arrays(native)
            peak_gib = _gpu_peak_reserved_gib(model.device)
            continuation_rng = save_checkpoint(
                output_root=output_root,
                scratch_root=scratch_root,
                task=task,
                completed_cycle=completed_cycle,
                elapsed_seconds=elapsed_seconds,
                native=native,
                configuration_sha256=configuration_sha256,
                hashes=hashes,
            )
            cycle_bar.set_postfix(
                durable_cycle=completed_cycle,
                peak_GiB=f"{peak_gib:.2f}",
                refresh=show_progress,
            )
            if after_checkpoint is not None:
                after_checkpoint(completed_cycle)
    finally:
        cycle_bar.close()

    final, reason = load_checkpoint(
        output_root=output_root,
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    if final is None or final.completed_cycle != task.cycles:
        raise RuntimeError(f"final dynamics checkpoint failed verification: {reason}")
    return final


def publish_result_shard(
    *,
    model: Any,
    config: Mapping[str, Any],
    output_root: Path,
    scratch_root: Path,
    shard: ResultShard,
    checkpoint: Checkpoint,
    configuration_sha256: str,
    hashes: Mapping[str, str],
    gpu: Mapping[str, Any],
) -> None:
    local_slice = shard.local_slice
    device = torch.device(model.device)
    frame = torch.as_tensor(checkpoint.frame[local_slice], device=device)
    ranks = torch.as_tensor(checkpoint.ranks[local_slice], device=device)
    payload: dict[str, Any] = endpoint_result_payload(
        frame=frame,
        ranks=ranks,
        nx=EXPECTED_NX,
        ny=shard.ny,
        global_sample_indices=np.asarray(shard.global_sample_indices, dtype=np.int64),
    )
    payload.update(
        {
            "schema": np.asarray(RESULT_SCHEMA),
            "bundle": np.asarray(BUNDLE),
            "sampling_revision": np.asarray(EXPECTED_REVISION),
            "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
            "task_id": np.asarray(shard.task_id),
            "execution_batch_id": np.asarray(shard.execution_batch.task_id),
            "execution_batch_seed": np.asarray(
                shard.execution_batch.seed, dtype=np.int64
            ),
            "Nx": np.asarray(EXPECTED_NX, dtype=np.int64),
            "Ny": np.asarray(shard.ny, dtype=np.int64),
            "alpha_1": np.asarray(1.0, dtype=np.float64),
            "alpha_2": np.asarray(30.0, dtype=np.float64),
            "nshell": np.asarray(1, dtype=np.int64),
            "dtype": np.asarray("complex128"),
            "sequence": np.asarray("raster_y"),
            "perfect_correction": np.asarray(True),
            "dw_truncation": np.asarray(True),
            "meas_slab_only": np.asarray(True),
            "elapsed_execution_batch_seconds": np.asarray(
                checkpoint.elapsed_seconds, dtype=np.float64
            ),
            "gpu_name": np.asarray(str(gpu["name"])),
            "configuration_sha256": np.asarray(configuration_sha256),
        }
    )
    validate_endpoint_payload(
        payload,
        nx=EXPECTED_NX,
        ny=shard.ny,
        global_sample_indices=np.asarray(shard.global_sample_indices),
    )

    result_path, completion_path = result_paths(output_root, shard)
    task_scratch = scratch_root / shard.execution_batch.task_id / "results"
    task_scratch.mkdir(parents=True, exist_ok=True)
    local_npz = task_scratch / result_path.name
    # Numerical frames/covariances are effectively incompressible. ZIP_STORED
    # avoids wasting A100 runtime compressing roughly 14 GiB of scientific state.
    _write_npz(local_npz, payload, compressed=False)
    _validate_result_npz(local_npz, shard)
    published = publish_file(local_npz, result_path)

    completion = _completion_identity(
        shard, configuration_sha256=configuration_sha256, hashes=hashes
    )
    completion.update(
        {
            "result_filename": result_path.name,
            "result_bytes": published["bytes"],
            "result_sha256": published["sha256"],
            "elapsed_execution_batch_seconds": checkpoint.elapsed_seconds,
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_json = task_scratch / completion_path.name
    _write_json(local_json, completion)
    publish_file(local_json, completion_path)
    valid, reason = verified_complete(
        output_root=output_root,
        shard=shard,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    if not valid:
        raise OSError(f"published result shard failed final verification: {reason}")
    for local_path in (local_json, local_npz):
        try:
            local_path.unlink()
        except OSError:
            pass
    del frame, ranks
    if device.type == "cuda":
        torch.cuda.empty_cache()


def execute_batch(
    *,
    model: Any,
    config: Mapping[str, Any],
    output_root: Path,
    scratch_root: Path,
    task: ExecutionBatch,
    configuration_sha256: str,
    hashes: Mapping[str, str],
    gpu: Mapping[str, Any],
    on_shard_complete: Callable[[ResultShard], None] | None = None,
) -> float:
    device = torch.device(model.device)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    task_scratch = scratch_root / task.task_id
    if task_scratch.exists():
        shutil.rmtree(task_scratch)
    task_scratch.mkdir(parents=True)

    checkpoint = run_dynamics(
        model=model,
        config=config,
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    if checkpoint.completed_cycle != task.cycles:
        raise RuntimeError("endpoint reduction requires the final-cycle checkpoint")
    pending: list[ResultShard] = []
    for shard in result_shards(task):
        valid, _ = verified_complete(
            output_root=output_root,
            shard=shard,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
        if not valid:
            pending.append(shard)
    print(
        f"[endpoint] {task.task_id}: {len(pending)} five-sample shards pending",
        flush=True,
    )
    for shard in pending:
        publish_result_shard(
            model=model,
            config=config,
            output_root=output_root,
            scratch_root=scratch_root,
            shard=shard,
            checkpoint=checkpoint,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
            gpu=gpu,
        )
        if on_shard_complete is not None:
            on_shard_complete(shard)
        _gpu_peak_reserved_gib(device)

    for shard in result_shards(task):
        valid, reason = verified_complete(
            output_root=output_root,
            shard=shard,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
        if not valid:
            raise RuntimeError(f"execution batch is missing {shard.task_id}: {reason}")
    remove_checkpoint(output_root, task)
    shutil.rmtree(task_scratch, ignore_errors=True)
    peak_gib = _gpu_peak_reserved_gib(device)
    print(
        f"[gpu memory] {task.task_id}: peak_reserved={peak_gib:.2f} GiB "
        f"(hard_limit={GPU_MEMORY_HARD_LIMIT_GIB:.2f} GiB)",
        flush=True,
    )
    return checkpoint.elapsed_seconds


def run_campaign(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    scratch_root: Path,
    report_only: bool = False,
    max_new_execution_batches: int | None = None,
) -> dict[str, Any]:
    config = validate_config(config)
    configuration_sha256 = config_sha256(config)
    hashes = source_hashes()
    tasks = expand_execution_batches(config)
    shards = all_result_shards(tasks)
    output_root.mkdir(parents=True, exist_ok=True)

    valid_shard_ids: set[str] = set()
    invalid_reasons: dict[str, str] = {}
    for shard in shards:
        valid, reason = verified_complete(
            output_root=output_root,
            shard=shard,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
        if valid:
            valid_shard_ids.add(shard.task_id)
        elif reason != "missing result and completion":
            invalid_reasons[shard.task_id] = reason

    complete_tasks = {
        task.task_id
        for task in tasks
        if all(shard.task_id in valid_shard_ids for shard in result_shards(task))
    }
    recoverable: dict[str, int] = {}
    for task in tasks:
        checkpoint, _ = load_checkpoint(
            output_root=output_root,
            task=task,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
        if checkpoint is not None:
            recoverable[task.task_id] = checkpoint.completed_cycle

    estimated_total_frame_bytes = sum(estimated_frame_result_bytes(s) for s in shards)
    estimated_total_half_covariance_bytes = sum(
        estimated_half_covariance_result_bytes(s) for s in shards
    )
    estimated_total_result_bytes = sum(
        estimated_result_payload_bytes(s) for s in shards
    )
    estimated_remaining_result_bytes = sum(
        estimated_result_payload_bytes(shard)
        for shard in shards
        if shard.task_id not in valid_shard_ids
    )
    required_drive_bytes = required_drive_free_bytes(shards, valid_shard_ids)

    inventory = {
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "configuration_sha256": configuration_sha256,
        "output_root": str(output_root),
        "scratch_root": str(scratch_root),
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "source_paths": {
            relative: str(BUNDLE_ROOT / relative) for relative in SOURCE_FILES
        },
        "dtype": config["dtype"],
        "Ny_order": list(EXPECTED_NY_VALUES),
        "total_trajectories": 300,
        "total_execution_batches": len(tasks),
        "total_result_shards": len(shards),
        "verified_result_shards": len(valid_shard_ids),
        "completed_execution_batches": len(complete_tasks),
        "pending_execution_batches": len(tasks) - len(complete_tasks),
        "recoverable_checkpoints": recoverable,
        "invalid_result_pairs": invalid_reasons,
        "estimated_final_occupied_frames_gib": estimated_total_frame_bytes / 1024**3,
        "estimated_final_half_system_covariances_gib": (
            estimated_total_half_covariance_bytes / 1024**3
        ),
        "estimated_final_large_payloads_gib": estimated_total_result_bytes / 1024**3,
        "estimated_remaining_large_payloads_gib": (
            estimated_remaining_result_bytes / 1024**3
        ),
        "required_drive_free_gib": required_drive_bytes / 1024**3,
        "estimated_total_a100_hours": "21-24",
        "gpu_memory_hard_limit_gib": GPU_MEMORY_HARD_LIMIT_GIB,
        "execution_batch_size_by_Ny": config["execution_batch_size_by_Ny"],
        "source_hashes": hashes,
    }
    print("[campaign inventory]", flush=True)
    print(json.dumps(inventory, indent=2, sort_keys=True), flush=True)
    if report_only:
        return {"status": "report_only", **inventory}

    gpu = validate_a100()
    drive_free = _check_space(
        output_root,
        required_bytes=required_drive_bytes,
        label="Drive",
    )
    local_free = _check_space(
        scratch_root,
        required_bytes=MINIMUM_LOCAL_FREE_BYTES,
        label="local scratch",
    )
    print(
        "[runtime] "
        + json.dumps(
            {
                "gpu": gpu,
                "drive_free_gib": drive_free / 1024**3,
                "local_free_gib": local_free / 1024**3,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for task in tasks:
        if task.task_id in complete_tasks:
            remove_checkpoint(output_root, task)

    pending_tasks = [task for task in tasks if task.task_id not in complete_tasks]
    bar = tqdm(
        total=len(shards),
        initial=len(valid_shard_ids),
        desc="hard-wall x-resolved",
        unit="shard",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )
    failed = 0
    bar.set_postfix(
        verified=len(valid_shard_ids),
        pending=len(shards) - len(valid_shard_ids),
        failed=failed,
        refresh=True,
    )
    new_execution_batches = 0
    current_ny: int | None = None
    model: Any | None = None

    def mark_shard_complete(shard: ResultShard) -> None:
        valid_shard_ids.add(shard.task_id)
        bar.update(1)
        bar.set_postfix(
            verified=len(valid_shard_ids),
            pending=len(shards) - len(valid_shard_ids),
            failed=failed,
            refresh=True,
        )

    try:
        for task in pending_tasks:
            if (
                max_new_execution_batches is not None
                and new_execution_batches >= max_new_execution_batches
            ):
                break
            if task.ny != current_ny:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = build_model(config, ny=task.ny)
                current_ny = task.ny
            bar.set_description(
                f"Ny={task.ny} samples {task.sample_start:03d}-{task.sample_stop - 1:03d}"
            )
            try:
                elapsed = execute_batch(
                    model=model,
                    config=config,
                    output_root=output_root,
                    scratch_root=scratch_root,
                    task=task,
                    configuration_sha256=configuration_sha256,
                    hashes=hashes,
                    gpu=gpu,
                    on_shard_complete=mark_shard_complete,
                )
            except Exception:
                failed += 1
                bar.set_postfix(
                    verified=len(valid_shard_ids),
                    pending=len(shards) - len(valid_shard_ids),
                    failed=failed,
                    refresh=True,
                )
                raise
            new_execution_batches += 1
            print(
                f"[execution complete] {task.task_id}: dynamics={elapsed / 60:.1f} min",
                flush=True,
            )
    finally:
        bar.close()

    verified_final = 0
    for shard in shards:
        valid, _ = verified_complete(
            output_root=output_root,
            shard=shard,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
        verified_final += int(valid)
    completed_final = sum(
        all(
            verified_complete(
                output_root=output_root,
                shard=shard,
                configuration_sha256=configuration_sha256,
                hashes=hashes,
            )[0]
            for shard in result_shards(task)
        )
        for task in tasks
    )
    status = "complete" if verified_final == len(shards) else "partial_limit_reached"
    summary = {
        "status": status,
        "verified_result_shards": verified_final,
        "total_result_shards": len(shards),
        "completed_execution_batches": completed_final,
        "total_execution_batches": len(tasks),
        "new_execution_batches": new_execution_batches,
        "failed_execution_batches": failed,
        "output_root": str(output_root),
    }
    print("[campaign complete]", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-execution-batches", type=int)
    args = parser.parse_args(argv)
    if (
        args.max_new_execution_batches is not None
        and args.max_new_execution_batches < 0
    ):
        parser.error("--max-new-execution-batches must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_campaign(
        config=config,
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        report_only=args.report_only,
        max_new_execution_batches=args.max_new_execution_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
