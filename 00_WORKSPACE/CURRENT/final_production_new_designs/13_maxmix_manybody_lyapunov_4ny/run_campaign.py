"""Simple resumable A100 runner for the hard/soft 4Ny Lyapunov campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BUNDLE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from lyapunov_observer import (  # noqa: E402
    OBSERVER_SCHEMA,
    BatchedActiveSpectrumObserver,
    spectrum_checkpoint_cycles,
)


SAMPLING_REVISION = (
    "maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_"
    "gpu_v4_38gib_memory_scaled"
)
ROOT_SEED = 2026091001
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
RESULT_SCHEMA = "maxmix_manybody_lyapunov_4ny_result_v4"
COMPLETION_SCHEMA = "maxmix_manybody_lyapunov_4ny_completion_v4"
CHECKPOINT_SCHEMA = "maxmix_manybody_lyapunov_4ny_checkpoint_v4"
RESULT_SHARD_SIZE = 5
SEGMENT_CYCLES = 10
RESUME_INITIAL_PURITY_TOLERANCE = 0.50000001
EXECUTION_BATCH_SIZE_BY_NY = {
    20: 100,
    24: 100,
    30: 100,
    36: 90,
    44: 60,
    56: 40,
    60: 35,
}
OBSERVER_CHUNK_BY_NY = {20: 20, 24: 20, 30: 10, 36: 10, 44: 5, 56: 2, 60: 2}
GPU_MEMORY_HARD_LIMIT_GIB = 38.0
GPU_MEMORY_HARD_LIMIT_BYTES = int(GPU_MEMORY_HARD_LIMIT_GIB * 1024**3)
SOURCE_FILES = (
    "run_campaign.py",
    "lyapunov_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)


@dataclass(frozen=True)
class ExecutionBatch:
    construction: str
    ny: int
    execution_index: int
    sample_start: int
    sample_stop: int
    seed: int

    @property
    def nx(self) -> int:
        return 20

    @property
    def cycles(self) -> int:
        return 4 * self.ny

    @property
    def samples(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def sample_indices(self) -> np.ndarray:
        return np.arange(self.sample_start, self.sample_stop, dtype=np.int64)

    @property
    def task_id(self) -> str:
        return (
            f"{self.construction}_Ny{self.ny:03d}_exec{self.execution_index:03d}_"
            f"samples{self.sample_start:03d}-{self.sample_stop - 1:03d}"
        )


@dataclass(frozen=True)
class ResultShard:
    execution: ExecutionBatch
    shard_index: int
    sample_start: int
    sample_stop: int

    @property
    def sample_indices(self) -> np.ndarray:
        return np.arange(self.sample_start, self.sample_stop, dtype=np.int64)

    @property
    def result_id(self) -> str:
        return (
            f"shard_{self.shard_index:03d}_samples_"
            f"{self.sample_start:03d}-{self.sample_stop - 1:03d}"
        )


@dataclass
class Checkpoint:
    completed_cycle: int
    elapsed_seconds: float
    G: np.ndarray
    rng_payload: dict[str, np.ndarray]
    observer_payload: dict[str, np.ndarray]


def expected_config() -> dict[str, Any]:
    return {
        "sampling_revision": SAMPLING_REVISION,
        "root_seed": ROOT_SEED,
        "Nx": 20,
        "Ny_values": [20, 24, 30, 36, 44, 56, 60],
        "samples_per_case": 100,
        "cycles_multiplier": 4,
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "trial_orbitals": "X",
        "filling_fraction": 0.5,
        "n_a": 0.5,
        "init_mode": "maxmix",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "dtype": "complex128",
        "device": "cuda:0",
        "backend": "local",
        "state_representation": "covariance",
        "gpu_memory_hard_limit_gib": GPU_MEMORY_HARD_LIMIT_GIB,
        "result_shard_size": RESULT_SHARD_SIZE,
        "segment_cycles": SEGMENT_CYCLES,
        "resume_initial_purity_tolerance": RESUME_INITIAL_PURITY_TOLERANCE,
        "execution_batch_size_by_Ny": {str(k): v for k, v in EXECUTION_BATCH_SIZE_BY_NY.items()},
        "observer_sample_chunk_by_Ny": {str(k): v for k, v in OBSERVER_CHUNK_BY_NY.items()},
        "constructions": {
            "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
            "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
        },
        "observables": {
            "record_cycles": "0..4Ny inclusive",
            "spectrum_cycles": "stride 4 union Ny,2Ny,3Ny,4Ny",
            "occupation_spectrum": True,
            "leading_log_sigma2_levels": 64,
            "soft_mode_count": 16,
            "transverse_mode_profiles": True,
            "measurement_log_probability": True,
            "cumulative_log_probability": True,
            "save_covariance_history": False,
            "save_final_covariance": False,
            "squared_singular_value_convention": "ell_i=log(sigma_i^2)",
            "cap_tolerance": 1.0e-9,
        },
    }


def validate_config(config: dict[str, Any]) -> None:
    if config != expected_config():
        raise ValueError("configuration does not match the locked bundle-13 contract")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


def source_hashes(bundle_root: Path = BUNDLE_ROOT) -> dict[str, str]:
    return {relative: sha256_file(bundle_root / relative) for relative in SOURCE_FILES}


def execution_seed(construction: str, ny: int, start: int, stop: int) -> int:
    label = f"{ROOT_SEED}|{construction}|Ny={ny}|samples={start}:{stop}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little")


def expand_execution_batches(config: dict[str, Any], construction: str) -> list[ExecutionBatch]:
    validate_config(config)
    if construction not in ("hard", "soft"):
        raise ValueError("construction must be hard or soft")
    tasks: list[ExecutionBatch] = []
    for ny in sorted(config["Ny_values"], reverse=True):
        size = int(config["execution_batch_size_by_Ny"][str(ny)])
        for execution_index, start in enumerate(range(0, 100, size)):
            stop = min(100, start + size)
            tasks.append(
                ExecutionBatch(
                    construction=construction,
                    ny=int(ny),
                    execution_index=execution_index,
                    sample_start=start,
                    sample_stop=stop,
                    seed=execution_seed(construction, int(ny), start, stop),
                )
            )
    expected_batches = sum(
        (100 + int(config["execution_batch_size_by_Ny"][str(ny)]) - 1)
        // int(config["execution_batch_size_by_Ny"][str(ny)])
        for ny in config["Ny_values"]
    )
    if len(tasks) != expected_batches:
        raise AssertionError("execution-batch expansion is incomplete")
    if sum(task.samples for task in tasks) != 700:
        raise AssertionError("each construction must contain 700 trajectories")
    if len({task.seed for task in tasks}) != len(tasks):
        raise AssertionError("execution-batch seeds are not unique")
    return tasks


def result_shards(task: ExecutionBatch) -> list[ResultShard]:
    return [
        ResultShard(
            execution=task,
            shard_index=start // RESULT_SHARD_SIZE,
            sample_start=start,
            sample_stop=min(task.sample_stop, start + RESULT_SHARD_SIZE),
        )
        for start in range(task.sample_start, task.sample_stop, RESULT_SHARD_SIZE)
    ]


def all_result_shards(config: dict[str, Any], construction: str) -> list[ResultShard]:
    return [shard for task in expand_execution_batches(config, construction) for shard in result_shards(task)]


def result_paths(output_root: Path, shard: ResultShard) -> tuple[Path, Path]:
    directory = output_root / "results" / shard.execution.construction / f"Ny{shard.execution.ny:03d}"
    stem = shard.result_id
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def checkpoint_paths(output_root: Path, task: ExecutionBatch) -> tuple[Path, Path]:
    directory = output_root / "checkpoints" / task.construction / f"Ny{task.ny:03d}" / task.task_id
    return directory / "checkpoint.npz", directory / "checkpoint.json"


def _shard_identity(
    shard: ResultShard,
    *,
    cfg_hash: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "sampling_revision": SAMPLING_REVISION,
        "configuration_hash": cfg_hash,
        "source_hashes": hashes,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "construction": shard.execution.construction,
        "Nx": shard.execution.nx,
        "Ny": shard.execution.ny,
        "cycles": shard.execution.cycles,
        "execution_batch_id": shard.execution.task_id,
        "execution_seed": shard.execution.seed,
        "result_id": shard.result_id,
        "sample_indices": shard.sample_indices.tolist(),
        "dtype": "complex128",
    }


def _checkpoint_identity(
    task: ExecutionBatch,
    *,
    cfg_hash: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "sampling_revision": SAMPLING_REVISION,
        "configuration_hash": cfg_hash,
        "source_hashes": hashes,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "construction": task.construction,
        "Nx": task.nx,
        "Ny": task.ny,
        "cycles": task.cycles,
        "execution_batch_id": task.task_id,
        "execution_seed": task.seed,
        "sample_indices": task.sample_indices.tolist(),
        "dtype": "complex128",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(local_path.stat().st_size)
    expected_sha = sha256_file(local_path)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.uploading")
    shutil.copyfile(local_path, temporary)
    if int(temporary.stat().st_size) != expected_bytes or sha256_file(temporary) != expected_sha:
        raise OSError(f"Drive readback mismatch for temporary {temporary}")
    os.replace(temporary, final_path)
    if int(final_path.stat().st_size) != expected_bytes or sha256_file(final_path) != expected_sha:
        raise OSError(f"Drive readback mismatch for final {final_path}")
    return {"bytes": expected_bytes, "sha256": expected_sha}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON payload is not an object")
    return value


def verified_complete(
    output_root: Path,
    shard: ResultShard,
    *,
    cfg_hash: str,
    hashes: dict[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = result_paths(output_root, shard)
    if not result_path.exists() and not completion_path.exists():
        return False, "missing pair"
    if not result_path.is_file() or not completion_path.is_file():
        return False, "incomplete pair"
    try:
        completion = _read_json(completion_path)
    except Exception as exc:
        return False, f"unreadable completion: {exc}"
    for key, expected in _shard_identity(shard, cfg_hash=cfg_hash, hashes=hashes).items():
        if completion.get(key) != expected:
            return False, f"completion identity mismatch: {key}"
    if completion.get("result_filename") != result_path.name:
        return False, "result filename mismatch"
    try:
        actual_bytes = int(result_path.stat().st_size)
        actual_sha = sha256_file(result_path)
    except OSError as exc:
        return False, f"result readback failed: {exc}"
    if completion.get("result_bytes") != actual_bytes or completion.get("result_sha256") != actual_sha:
        return False, "result byte-count/checksum mismatch"
    try:
        with np.load(result_path, allow_pickle=False) as payload:
            if str(payload["result_schema"].item()) != RESULT_SCHEMA:
                return False, "result schema mismatch"
            if not np.array_equal(payload["sample_indices"], shard.sample_indices):
                return False, "result sample IDs mismatch"
            expected_t = shard.execution.cycles + 1
            expected_s = spectrum_checkpoint_cycles(shard.execution.ny).size
            expected_n = (22 if shard.execution.construction == "hard" else 40) * shard.execution.ny
            if payload["cumulative_log_probability"].shape != (
                shard.sample_indices.size,
                expected_t,
            ):
                return False, "result record-weight shape mismatch"
            if payload["occupations"].shape != (
                shard.sample_indices.size,
                expected_s,
                expected_n,
            ):
                return False, "result spectrum shape mismatch"
            if payload["leading_log_sigma2"].shape != (
                shard.sample_indices.size,
                expected_s,
                64,
            ):
                return False, "result many-body-level shape mismatch"
            if "G_final" in payload.files:
                return False, "result unexpectedly saves final covariance"
    except Exception as exc:
        return False, f"result NPZ validation failed: {exc}"
    return True, "verified"


def capture_rng() -> dict[str, np.ndarray]:
    state = np.random.get_state()
    payload = {
        "rng_numpy_algorithm": np.asarray(state[0]),
        "rng_numpy_state": np.asarray(state[1], dtype=np.uint32),
        "rng_numpy_position": np.asarray(state[2], dtype=np.int64),
        "rng_numpy_has_gauss": np.asarray(state[3], dtype=np.int64),
        "rng_numpy_cached_gaussian": np.asarray(state[4], dtype=np.float64),
        "rng_torch_cpu": torch.get_rng_state().cpu().numpy().astype(np.uint8, copy=False),
        "rng_cuda_count": np.asarray(torch.cuda.device_count() if torch.cuda.is_available() else 0, dtype=np.int64),
    }
    if torch.cuda.is_available():
        for index, value in enumerate(torch.cuda.get_rng_state_all()):
            payload[f"rng_cuda_{index}"] = value.cpu().numpy().astype(np.uint8, copy=False)
    return payload


def restore_rng(payload: dict[str, np.ndarray]) -> None:
    np.random.set_state(
        (
            str(np.asarray(payload["rng_numpy_algorithm"]).item()),
            np.asarray(payload["rng_numpy_state"], dtype=np.uint32),
            int(np.asarray(payload["rng_numpy_position"]).item()),
            int(np.asarray(payload["rng_numpy_has_gauss"]).item()),
            float(np.asarray(payload["rng_numpy_cached_gaussian"]).item()),
        )
    )
    torch.set_rng_state(torch.as_tensor(payload["rng_torch_cpu"], dtype=torch.uint8).cpu())
    expected_cuda = int(np.asarray(payload["rng_cuda_count"]).item())
    actual_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if expected_cuda != actual_cuda:
        raise RuntimeError(f"checkpoint CUDA RNG device count changed: {expected_cuda} != {actual_cuda}")
    if expected_cuda:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(payload[f"rng_cuda_{index}"], dtype=torch.uint8).cpu() for index in range(expected_cuda)]
        )


def _split_checkpoint_payload(payload: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    observer = {key: value for key, value in payload.items() if key.startswith("observer_")}
    rng = {key: value for key, value in payload.items() if key.startswith("rng_")}
    return observer, rng


def save_checkpoint(
    output_root: Path,
    scratch_root: Path,
    task: ExecutionBatch,
    *,
    completed_cycle: int,
    elapsed_seconds: float,
    G: np.ndarray,
    observer: BatchedActiveSpectrumObserver,
    cfg_hash: str,
    hashes: dict[str, str],
    rng_payload: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    observer.validate(completed_cycle=completed_cycle)
    G = np.asarray(G)
    n = 2 * task.nx * task.ny
    if G.dtype != np.complex128 or G.shape != (task.samples, n, n):
        raise RuntimeError("checkpoint covariance shape/dtype mismatch")
    rng = capture_rng() if rng_payload is None else {
        key: np.asarray(value).copy() for key, value in rng_payload.items()
    }
    payload: dict[str, np.ndarray] = {
        "checkpoint_schema": np.asarray(CHECKPOINT_SCHEMA),
        "completed_cycle": np.asarray(completed_cycle, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "sample_indices": task.sample_indices,
        "G": G,
        **rng,
        **observer.checkpoint_payload(),
    }
    local_dir = scratch_root / task.task_id
    local_npz = local_dir / "checkpoint.npz"
    local_json = local_dir / "checkpoint.json"
    save_npz(local_npz, payload)
    final_npz, final_json = checkpoint_paths(output_root, task)
    commit = publish_file(local_npz, final_npz)
    metadata = {
        **_checkpoint_identity(task, cfg_hash=cfg_hash, hashes=hashes),
        "completed_cycle": int(completed_cycle),
        "checkpoint_filename": final_npz.name,
        "checkpoint_bytes": commit["bytes"],
        "checkpoint_sha256": commit["sha256"],
    }
    write_json(local_json, metadata)
    publish_file(local_json, final_json)
    return rng


def load_checkpoint(
    output_root: Path,
    task: ExecutionBatch,
    *,
    cfg_hash: str,
    hashes: dict[str, str],
) -> tuple[Checkpoint | None, str]:
    npz_path, json_path = checkpoint_paths(output_root, task)
    if not npz_path.exists() and not json_path.exists():
        return None, "no checkpoint"
    if not npz_path.is_file() or not json_path.is_file():
        return None, "incomplete checkpoint pair"
    try:
        metadata = _read_json(json_path)
        for key, expected in _checkpoint_identity(task, cfg_hash=cfg_hash, hashes=hashes).items():
            if metadata.get(key) != expected:
                return None, f"checkpoint identity mismatch: {key}"
        if metadata.get("checkpoint_filename") != npz_path.name:
            return None, "checkpoint filename mismatch"
        if metadata.get("checkpoint_bytes") != int(npz_path.stat().st_size):
            return None, "checkpoint byte-count mismatch"
        if metadata.get("checkpoint_sha256") != sha256_file(npz_path):
            return None, "checkpoint checksum mismatch"
        with np.load(npz_path, allow_pickle=False) as saved:
            payload = {key: saved[key].copy() for key in saved.files}
        if str(payload["checkpoint_schema"].item()) != CHECKPOINT_SCHEMA:
            return None, "checkpoint schema mismatch"
        completed = int(payload["completed_cycle"].item())
        if completed != int(metadata["completed_cycle"]):
            return None, "checkpoint completed-cycle mismatch"
        if completed < 0 or completed > task.cycles or (completed % SEGMENT_CYCLES and completed != task.cycles):
            return None, "checkpoint cycle is outside the segment contract"
        n = 2 * task.nx * task.ny
        G = payload["G"]
        if G.dtype != np.complex128 or G.shape != (task.samples, n, n):
            return None, "checkpoint covariance shape/dtype mismatch"
        if not np.array_equal(payload["sample_indices"], task.sample_indices):
            return None, "checkpoint sample IDs mismatch"
        observer_payload, rng_payload = _split_checkpoint_payload(payload)
        return Checkpoint(
            completed_cycle=completed,
            elapsed_seconds=float(payload["elapsed_seconds"].item()),
            G=G,
            rng_payload=rng_payload,
            observer_payload=observer_payload,
        ), "verified"
    except Exception as exc:
        return None, f"checkpoint validation failed: {exc}"


def remove_checkpoint(output_root: Path, task: ExecutionBatch) -> None:
    npz_path, json_path = checkpoint_paths(output_root, task)
    for path in (json_path, npz_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def build_model(config: dict[str, Any], construction: str, ny: int) -> classA_U1FGTN_gpu:
    flags = config["constructions"][construction]
    print(
        f"[model] constructing {construction} Nx=20, Ny={ny} projectors",
        flush=True,
    )
    model = classA_U1FGTN_gpu(
        Nx=20,
        Ny=int(ny),
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=bool(flags["dw_truncation"]),
        triv_region_local_mode=False,
        device=config["device"],
        dtype=config["dtype"],
        backend=config["backend"],
    )
    if tuple(int(value) for value in model.DW_loc) != (5, 15):
        raise RuntimeError(f"unexpected domain-wall positions: {model.DW_loc}")
    active = model.active_top_layer_indices(
        meas_slab_only=bool(flags["meas_slab_only"])
    )
    expected_active = (22 if construction == "hard" else 40) * int(ny)
    if int(active.numel()) != expected_active:
        raise RuntimeError(
            f"active transfer space has {int(active.numel())} modes, "
            f"expected {expected_active}"
        )
    if model.dtype != torch.complex128:
        raise RuntimeError("constructed model is not complex128")
    print(
        f"[model] ready: construction={construction}, walls={model.DW_loc}, "
        f"active_modes={expected_active}",
        flush=True,
    )
    return model


def run_segment(
    model: classA_U1FGTN_gpu,
    config: dict[str, Any],
    task: ExecutionBatch,
    observer: BatchedActiveSpectrumObserver,
    *,
    completed_cycle: int,
    segment_cycles: int,
    G_init: np.ndarray | None,
    continuing: bool,
    progress_bar: tqdm,
) -> np.ndarray:
    flags = config["constructions"][task.construction]

    def cycle_observer(*, cycle: int, G: torch.Tensor, **_: Any) -> None:
        local_cycle = int(cycle)
        if continuing and local_cycle == 0:
            return
        global_cycle = int(completed_cycle + local_cycle)
        observer.observe(cycle=global_cycle, G=G)
        if local_cycle > 0:
            progress_bar.update(1)

    def record_observer(*, cycle: int, **payload: Any) -> None:
        observer.record_event(
            cycle=int(completed_cycle + int(cycle)),
            **payload,
        )

    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=int(segment_cycles),
        postselect=False,
        perfect_correction=True,
        samples=task.samples,
        init_mode="maxmix",
        G_init=G_init,
        G_init_prepared=bool(continuing and task.construction == "hard"),
        save=False,
        save_init=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=bool(flags["meas_slab_only"]),
        batch_size=task.samples,
        return_data=True,
        state_representation="covariance",
        initial_purity_tolerance=float(config["resume_initial_purity_tolerance"]),
        cycle_observer=cycle_observer,
        record_observer=record_observer,
    )
    if result.get("state_representation_resolved") != "covariance":
        raise RuntimeError("canonical engine did not use covariance state representation")
    if bool(result.get("G_init_prepared", False)) != bool(continuing and task.construction == "hard"):
        raise RuntimeError("canonical engine did not honor G_init_prepared")
    expected_preparation = bool(task.construction == "hard" and not continuing)
    if bool(result.get("exterior_preparation_performed", False)) != expected_preparation:
        raise RuntimeError("canonical engine exterior-preparation metadata mismatch")
    final = np.asarray(result["G_final"])
    if final.dtype != np.complex128:
        raise RuntimeError(f"canonical engine returned {final.dtype}, expected complex128")
    return final


def publish_result(
    output_root: Path,
    scratch_root: Path,
    shard: ResultShard,
    *,
    observer: BatchedActiveSpectrumObserver,
    elapsed_seconds: float,
    cfg_hash: str,
    hashes: dict[str, str],
) -> None:
    task = shard.execution
    local_start = shard.sample_start - task.sample_start
    local_stop = shard.sample_stop - task.sample_start
    sample_slice = slice(local_start, local_stop)
    payload = {
        "result_schema": np.asarray(RESULT_SCHEMA),
        "sampling_revision": np.asarray(SAMPLING_REVISION),
        "configuration_hash": np.asarray(cfg_hash),
        "canonical_dynamics_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
        "construction": np.asarray(task.construction),
        "Nx": np.asarray(task.nx, dtype=np.int64),
        "Ny": np.asarray(task.ny, dtype=np.int64),
        "execution_batch_id": np.asarray(task.task_id),
        "execution_seed": np.asarray(task.seed, dtype=np.uint64),
        "elapsed_execution_batch_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        **observer.result_payload(sample_slice),
        "centered_covariance_convention": np.asarray("G=2C-I"),
    }
    local_dir = scratch_root / task.task_id / "results"
    local_result = local_dir / f"{shard.result_id}.npz"
    local_completion = local_dir / f"{shard.result_id}.complete.json"
    save_npz(local_result, payload)
    result_path, completion_path = result_paths(output_root, shard)
    commit = publish_file(local_result, result_path)
    completion = {
        **_shard_identity(shard, cfg_hash=cfg_hash, hashes=hashes),
        "result_filename": result_path.name,
        "result_bytes": commit["bytes"],
        "result_sha256": commit["sha256"],
        "completed_unix": time.time(),
    }
    write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    valid, reason = verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)
    if not valid:
        raise OSError(f"published result did not verify: {reason}")


def execute_batch(
    config: dict[str, Any],
    output_root: Path,
    scratch_root: Path,
    task: ExecutionBatch,
    *,
    cfg_hash: str,
    hashes: dict[str, str],
    shard_bar: tqdm,
) -> float:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(config["device"]))
    model = build_model(config, task.construction, task.ny)
    flags = config["constructions"][task.construction]
    active = model.active_top_layer_indices(
        meas_slab_only=bool(flags["meas_slab_only"])
    )
    observer = BatchedActiveSpectrumObserver(
        nx=task.nx,
        ny=task.ny,
        cycles=task.cycles,
        samples=task.samples,
        active_indices=active,
        full_mode_count=model.Nlayer,
        wall_locations=tuple(model.DW_loc),
        construction=task.construction,
        sample_indices=task.sample_indices,
        sample_chunk=int(config["observer_sample_chunk_by_Ny"][str(task.ny)]),
        soft_mode_count=int(config["observables"]["soft_mode_count"]),
        leading_level_count=int(
            config["observables"]["leading_log_sigma2_levels"]
        ),
        cap_tolerance=float(config["observables"]["cap_tolerance"]),
    )
    checkpoint, reason = load_checkpoint(output_root, task, cfg_hash=cfg_hash, hashes=hashes)
    if checkpoint is None:
        completed_cycle = 0
        elapsed_seconds = 0.0
        G = None
        np.random.seed(task.seed)
        torch.manual_seed(task.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task.seed)
        continuation_rng = capture_rng()
        print(f"[checkpoint] {task.task_id}: {reason}; starting fresh", flush=True)
    else:
        completed_cycle = checkpoint.completed_cycle
        elapsed_seconds = checkpoint.elapsed_seconds
        G = checkpoint.G
        continuation_rng = checkpoint.rng_payload
        observer.restore_checkpoint(checkpoint.observer_payload, completed_cycle=completed_cycle)
        print(
            f"[checkpoint] {task.task_id}: {reason}; "
            f"cycle {completed_cycle}/{task.cycles}",
            flush=True,
        )

    with tqdm(
        total=task.cycles,
        initial=completed_cycle,
        desc=f"{task.construction} Ny={task.ny} cycles",
        unit="cycle",
        leave=False,
    ) as cycle_bar:
        while completed_cycle < task.cycles:
            segment = min(SEGMENT_CYCLES, task.cycles - completed_cycle)
            peak_gib = 0.0
            restore_rng(continuation_rng)
            started = time.perf_counter()
            G = run_segment(
                model,
                config,
                task,
                observer,
                completed_cycle=completed_cycle,
                segment_cycles=segment,
                G_init=G,
                continuing=completed_cycle > 0,
                progress_bar=cycle_bar,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_reserved = int(
                    torch.cuda.max_memory_reserved(torch.device(config["device"]))
                )
                if peak_reserved > GPU_MEMORY_HARD_LIMIT_BYTES:
                    raise RuntimeError(
                        "GPU peak reserved memory crossed the 38 GiB hard limit: "
                        f"{peak_reserved / 1024**3:.2f} GiB"
                    )
                peak_gib = peak_reserved / 1024**3
            elapsed_seconds += time.perf_counter() - started
            completed_cycle += segment
            continuation_rng = save_checkpoint(
                output_root,
                scratch_root,
                task,
                completed_cycle=completed_cycle,
                elapsed_seconds=elapsed_seconds,
                G=G,
                observer=observer,
                cfg_hash=cfg_hash,
                hashes=hashes,
            )
            cycle_bar.set_postfix(
                durable_cycle=completed_cycle,
                peak_GiB=f"{peak_gib:.2f}",
            )

    assert G is not None
    observer.validate(completed_cycle=task.cycles)
    for shard in result_shards(task):
        valid, _ = verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)
        if valid:
            continue
        publish_result(
            output_root,
            scratch_root,
            shard,
            observer=observer,
            elapsed_seconds=elapsed_seconds,
            cfg_hash=cfg_hash,
            hashes=hashes,
        )
        shard_bar.update(1)
    for shard in result_shards(task):
        valid, reason = verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)
        if not valid:
            raise RuntimeError(f"batch result failed final verification: {reason}")
    remove_checkpoint(output_root, task)
    shutil.rmtree(scratch_root / task.task_id, ignore_errors=True)
    if torch.cuda.is_available():
        peak_gib = float(
            torch.cuda.max_memory_reserved(torch.device(config["device"])) / 1024**3
        )
    else:
        peak_gib = 0.0
    print(
        f"[gpu memory] {task.task_id}: peak_reserved={peak_gib:.2f} GiB "
        f"(hard_limit={GPU_MEMORY_HARD_LIMIT_GIB:.2f} GiB)",
        flush=True,
    )
    return peak_gib


def require_runtime(config: dict[str, Any]) -> None:
    if config["device"].startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        props = torch.cuda.get_device_properties(torch.device(config["device"]))
        gib = props.total_memory / 1024**3
        name = torch.cuda.get_device_name(torch.device(config["device"]))
        if "A100" not in name or gib < 35.0:
            raise RuntimeError(f"expected A100 40-GB-class GPU, got {name} ({gib:.2f} GiB)")
        fraction = min(1.0, GPU_MEMORY_HARD_LIMIT_BYTES / int(props.total_memory))
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=torch.device(config["device"])
        )
    if torch.complex128 != getattr(torch, config["dtype"]):
        raise RuntimeError("locked dtype is not complex128")


def require_space(output_root: Path, scratch_root: Path) -> None:
    scratch_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    local_free = shutil.disk_usage(scratch_root).free
    drive_free = shutil.disk_usage(output_root).free
    if local_free < 6 * 1024**3:
        raise OSError("less than 6 GiB free under local scratch")
    if drive_free < 8 * 1024**3:
        raise OSError("less than 8 GiB free under the Drive output root")


def run_campaign(
    config: dict[str, Any],
    *,
    construction: str,
    output_root: Path,
    scratch_root: Path,
    report_only: bool,
    max_new_execution_batches: int | None,
) -> dict[str, Any]:
    validate_config(config)
    cfg_hash = config_hash(config)
    hashes = source_hashes()
    tasks = expand_execution_batches(config, construction)
    shards = all_result_shards(config, construction)
    verified = {
        shard.result_id + f"|Ny={shard.execution.ny}": verified_complete(
            output_root, shard, cfg_hash=cfg_hash, hashes=hashes
        )[0]
        for shard in shards
    }
    completed_count = sum(verified.values())
    checkpoints = []
    for task in tasks:
        checkpoint, _ = load_checkpoint(output_root, task, cfg_hash=cfg_hash, hashes=hashes)
        if checkpoint is not None:
            checkpoints.append({"task": task.task_id, "cycle": checkpoint.completed_cycle})
    dashboard = {
        "sampling_revision": SAMPLING_REVISION,
        "construction": construction,
        "configuration_hash": cfg_hash,
        "source_hashes": hashes,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "output_root": str(output_root),
        "scratch_root": str(scratch_root),
        "device": config["device"],
        "dtype": config["dtype"],
        "execution_batches": len(tasks),
        "durable_result_shards": len(shards),
        "completed_shards": completed_count,
        "pending_shards": len(shards) - completed_count,
        "recoverable_checkpoints": checkpoints,
        "workload_sample_cycles": sum(task.samples * task.cycles for task in tasks),
        "gpu_memory_hard_limit_gib": GPU_MEMORY_HARD_LIMIT_GIB,
        "execution_batch_size_by_Ny": config["execution_batch_size_by_Ny"],
    }
    print("[campaign dashboard]", flush=True)
    print(json.dumps(dashboard, indent=2, sort_keys=True), flush=True)
    if report_only:
        return dashboard
    require_runtime(config)
    require_space(output_root, scratch_root)

    completed_batches = 0
    measured_rates: list[float] = []
    remaining_work = sum(
        task.samples * task.cycles
        for task in tasks
        if not all(
            verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)[0]
            for shard in result_shards(task)
        )
    )
    with tqdm(
        total=len(shards),
        initial=completed_count,
        desc=f"{construction} durable shards",
        unit="shard",
    ) as shard_bar:
        for task in tasks:
            task_status = [
                verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)[0]
                for shard in result_shards(task)
            ]
            if all(task_status):
                remove_checkpoint(output_root, task)
                continue
            if max_new_execution_batches is not None and completed_batches >= max_new_execution_batches:
                break
            print(
                f"[task start] {task.task_id}: resident_trajectories={task.samples}, "
                f"cycles={task.cycles}, result_shards={len(result_shards(task))}, "
                f"seed={task.seed}",
                flush=True,
            )
            started = time.perf_counter()
            peak_gib = execute_batch(
                config,
                output_root,
                scratch_root,
                task,
                cfg_hash=cfg_hash,
                hashes=hashes,
                shard_bar=shard_bar,
            )
            batch_elapsed = time.perf_counter() - started
            measured_rates.append(task.samples * task.cycles / batch_elapsed)
            completed_batches += 1
            remaining_work -= task.samples * task.cycles
            mean_rate = float(np.mean(measured_rates))
            print(
                f"[task complete] {task.task_id}: {batch_elapsed / 60.0:.1f} min; "
                f"peak_reserved={peak_gib:.2f} GiB; "
                f"measured projected remaining={remaining_work / mean_rate / 3600.0:.1f} A100 h",
                flush=True,
            )

    completed_final = sum(
        verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)[0]
        for shard in shards
    )
    summary = {**dashboard, "completed_shards": completed_final, "pending_shards": len(shards) - completed_final}
    print("[campaign summary]", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--construction", choices=("hard", "soft"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-execution-batches", type=int)
    args = parser.parse_args(argv)
    if args.max_new_execution_batches is not None and args.max_new_execution_batches < 0:
        parser.error("--max-new-execution-batches must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_campaign(
        config,
        construction=args.construction,
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        report_only=args.report_only,
        max_new_execution_batches=args.max_new_execution_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
