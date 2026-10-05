"""Hard-wall alpha=1,3 full-measurement purification through 2Ny; bundle 21."""

from __future__ import annotations

import argparse
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
from purification_observer import PurificationObserver, extract_slowest_modes, OccupationSpectrumError  # noqa: E402


SAMPLING_REVISION = "hard_wall_full_measurement_nx20_ny30_alpha1-3_s100_2ny_v1"
ROOT_SEED = 2026092521
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
RESULT_SCHEMA = "full_measurement_purification_result_v1"
COMPLETION_SCHEMA = "full_measurement_purification_completion_v1"
CHECKPOINT_SCHEMA = "full_measurement_purification_checkpoint_v1"
RESULT_SHARD_SIZE = 5
SEGMENT_CYCLES = 10
RESUME_INITIAL_PURITY_TOLERANCE = 0.50000001
EXECUTION_BATCH_SIZE_BY_NY = {30: 100}
OBSERVER_CHUNK_BY_NY = {30: 10}
GPU_MEMORY_BUDGET_BYTES = 38_000_000_000  # decimal GB, not GiB
GPU_ALLOCATOR_LIMIT_BYTES = 35_000_000_000  # leave space for CUDA/library workspaces
SOURCE_FILES = (
    "run_campaign.py",
    "purification_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)
# Exact deployed pre-diagnostic-hotfix identity. No wildcard source acceptance:
# only error-path diagnostics/eigensolver recheck changed; dynamics, configuration,
# seeds, accepted-spectrum formulas, schemas and numerical tolerance did not.
PRE_HOTFIX_SOURCE_HASHES = {
    "run_campaign.py": "8f6f5958ad466adbe3e8298c98d1acd906901b12e7e024c4aff0aacd341e3899",
    "purification_observer.py": "16400cc51ea4a074ff4ac64c7ef743fea3b024ba8d942f6ec338f5671f42e472",
    "src/classA_U1FGTN_gpu.py": "53de96bced6839b485afe04fe4aaa15d2e42c249a9cf14f6ca0c931f55409700",
    "src/occupied_frame_gpu.py": "bfc10cefea98ce00184a88b5c375eadfcda8e3dc6954d9f66183d566008951b0",
}


@dataclass(frozen=True)
class ExecutionBatch:
    construction: str
    ny: int
    execution_index: int
    sample_start: int
    sample_stop: int
    seed: int
    alpha_1: float = 1.0

    @property
    def nx(self) -> int:
        return 20

    @property
    def cycles(self) -> int:
        return 2 * self.ny

    @property
    def samples(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def sample_indices(self) -> np.ndarray:
        return np.arange(self.sample_start, self.sample_stop, dtype=np.int64)

    @property
    def task_id(self) -> str:
        return (
            f"{self.construction}_alpha{self.alpha_1:g}_Ny{self.ny:03d}_exec{self.execution_index:03d}_"
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
        "Ny_values": [30],
        "samples_per_case": 100,
        "cycles_multiplier": 2,
        "nshell": 1,
        "alpha_1_values": [1.0, 3.0],
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
        "result_shard_size": RESULT_SHARD_SIZE,
        "segment_cycles": SEGMENT_CYCLES,
        "gpu_memory_budget_bytes": GPU_MEMORY_BUDGET_BYTES,
        "gpu_allocator_limit_bytes": GPU_ALLOCATOR_LIMIT_BYTES,
        "resume_initial_purity_tolerance": RESUME_INITIAL_PURITY_TOLERANCE,
        "execution_batch_size_by_Ny": {str(k): v for k, v in EXECUTION_BATCH_SIZE_BY_NY.items()},
        "observer_sample_chunk_by_Ny": {str(k): v for k, v in OBSERVER_CHUNK_BY_NY.items()},
        "constructions": {
            "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": False},
        },
        "observables": {
            "cycles": "0..2Ny inclusive",
            "occupation_spectrum": True,
            "cell_entropy_contour_alpha1": [1.0],
            "cell_charge_variance_contour_alpha1": [1.0],
            "endpoint_slowest_mode_alpha1": [1.0],
            "slowest_mode_definition": "argmin_j abs(log((1-nu_j)/nu_j)/(2T)) at T=2Ny",
            "slowest_mode_occupation_cutoff": 1e-9,
            "slowest_mode_degeneracy_atol": 1e-10,
            "final_centered_covariance": True,
            "measurement_log_probability": True,
            "cumulative_log_probability": True,
            "log_probability_dtype": "float64",
            "log_probability_origin": "global_maxmix_cycle_zero_no_exterior_preparation",
            "entropy_log_base": "natural",
        },
    }


def validate_config(config: dict[str, Any]) -> None:
    if config != expected_config():
        raise ValueError("configuration does not match the locked bundle-21 full-measurement contract")


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


def source_identity_matches(saved: Any, expected: dict[str, str]) -> bool:
    if saved == expected:
        return True
    diagnostic_sources = {**PRE_HOTFIX_SOURCE_HASHES,
        "run_campaign.py": "aa34cc41f9f7e8751b6a75bcbd2154ac6dd86af2dd0088edc8a6858f3c0ba4f9",
        "purification_observer.py": "35234dfcd7b1f884df37d3a24ab0be4de137457a705ec444162934aa92abdee6"}
    # Repository-wide engine sync adds an opt-in clipping path. This v1 runner
    # never enables it; default-path equivalence is covered by regression tests.
    compatible_default_engines = {
        PRE_HOTFIX_SOURCE_HASHES["src/classA_U1FGTN_gpu.py"],
        "86ad0d5a40aab9cfcc63d9688b024948fed7a891309cd4376745206e2883687c",
    }
    return (saved in (PRE_HOTFIX_SOURCE_HASHES, diagnostic_sources) and expected == source_hashes()
            and expected.get("src/classA_U1FGTN_gpu.py") in compatible_default_engines
            and expected.get("src/occupied_frame_gpu.py") == PRE_HOTFIX_SOURCE_HASHES["src/occupied_frame_gpu.py"])


def execution_seed(construction: str, ny: int, start: int, stop: int, alpha_1: float) -> int:
    label = f"{ROOT_SEED}|{construction}|alpha1={alpha_1:g}|Ny={ny}|samples={start}:{stop}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little")


def expand_execution_batches(config: dict[str, Any], construction: str) -> list[ExecutionBatch]:
    validate_config(config)
    if construction != "hard":
        raise ValueError("only hard-wall construction is allowed")
    tasks: list[ExecutionBatch] = []
    for ny in config["Ny_values"]:
        size = int(config["execution_batch_size_by_Ny"][str(ny)])
        for execution_index, start in enumerate(range(0, 100, size)):
            stop = min(100, start + size)
            for alpha in config["alpha_1_values"]:
                tasks.append(
                    ExecutionBatch(
                        construction=construction,
                        ny=int(ny),
                        execution_index=execution_index,
                        sample_start=start,
                        sample_stop=stop,
                        seed=execution_seed(construction, int(ny), start, stop, alpha),
                        alpha_1=float(alpha),
                    )
                )
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
    directory = output_root / "results" / shard.execution.construction / f"alpha1_{shard.execution.alpha_1:g}" / f"Ny{shard.execution.ny:03d}"
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
        "alpha_1": shard.execution.alpha_1,
        "meas_slab_only": False,
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
        "alpha_1": task.alpha_1,
        "meas_slab_only": False,
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
        writer = np.savez if path.name == "checkpoint.npz" else np.savez_compressed
        writer(handle, **payload)
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
        if key == "source_hashes" and source_identity_matches(completion.get(key), expected):
            continue
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
            expected_n = 2 * shard.execution.nx * shard.execution.ny
            if payload["occupation_spectrum"].shape != (shard.sample_indices.size, expected_t, expected_n):
                return False, "result spectrum shape mismatch"
            if payload["G_final"].shape != (shard.sample_indices.size, expected_n, expected_n):
                return False, "result final covariance shape mismatch"
            if float(payload["alpha_1"].item()) != shard.execution.alpha_1 or bool(payload["meas_slab_only"].item()):
                return False, "result scientific metadata mismatch"
            if not np.array_equal(payload["cycles"], np.arange(expected_t)):
                return False, "result cycle inventory mismatch"
            if shard.execution.alpha_1 == 1:
                if payload["entropy_contour"].shape != (shard.sample_indices.size, expected_t, shard.execution.nx, shard.execution.ny):
                    return False, "result contour shape mismatch"
                if payload["slow_mode_vector"].shape != (shard.sample_indices.size, expected_n):
                    return False, "result slow-mode shape mismatch"
            elif "entropy_contour" in payload.files or "slow_mode_vector" in payload.files:
                return False, "alpha3 result contains unexpected contour/mode products"
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
    observer: PurificationObserver,
    cfg_hash: str,
    hashes: dict[str, str],
    rng_payload: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    observer.validate(completed_cycle=completed_cycle, final=completed_cycle == task.cycles)
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
            if key == "source_hashes" and source_identity_matches(metadata.get(key), expected):
                continue
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


def build_model(config: dict[str, Any], construction: str, ny: int, alpha_1: float) -> classA_U1FGTN_gpu:
    flags = config["constructions"][construction]
    return classA_U1FGTN_gpu(
        Nx=20,
        Ny=int(ny),
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=float(alpha_1),
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=bool(flags["dw_truncation"]),
        triv_region_local_mode=False,
        device=config["device"],
        dtype=config["dtype"],
        backend=config["backend"],
    )


def run_segment(
    model: classA_U1FGTN_gpu,
    config: dict[str, Any],
    task: ExecutionBatch,
    observer: PurificationObserver,
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
        try:
            observer.observe(cycle=global_cycle, G=G)
        except OccupationSpectrumError as exc:
            exc.diagnostics.update(cycle=global_cycle,
                sample_index=int(task.sample_indices[exc.diagnostics["sample_offset"]]))
            raise
        check_gpu_memory(config)
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
        G_init_prepared=bool(continuing),
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
    if bool(result.get("G_init_prepared", False)) != bool(continuing):
        raise RuntimeError("canonical engine did not honor G_init_prepared")
    expected_preparation = False
    if bool(result.get("exterior_preparation_performed", False)) != expected_preparation:
        raise RuntimeError("canonical engine exterior-preparation metadata mismatch")
    if result.get("meas_slab_only_effective") is not False:
        raise RuntimeError("canonical engine did not honor full-system measurement")
    final = np.asarray(result["G_final"])
    if final.dtype != np.complex128:
        raise RuntimeError(f"canonical engine returned {final.dtype}, expected complex128")
    return final


def publish_result(
    output_root: Path,
    scratch_root: Path,
    shard: ResultShard,
    *,
    G_final: np.ndarray,
    observer: PurificationObserver,
    elapsed_seconds: float,
    cfg_hash: str,
    hashes: dict[str, str],
    endpoint_modes: dict[str, np.ndarray] | None = None,
) -> None:
    task = shard.execution
    local_start = shard.sample_start - task.sample_start
    local_stop = shard.sample_stop - task.sample_start
    sample_slice = slice(local_start, local_stop)
    payload = {
        "result_schema": np.asarray(RESULT_SCHEMA),
        "alpha_1": np.asarray(task.alpha_1),
        "alpha_2": np.asarray(30.0),
        "meas_slab_only": np.asarray(False),
        "dw_truncation": np.asarray(True),
        "cycle_zero": np.asarray("full physical layer maximally mixed; no exterior preparation"),
        "configuration_json": np.asarray(canonical_json(expected_config())),
        "source_hashes_json": np.asarray(canonical_json(hashes)),
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
        "G_final": np.asarray(G_final[sample_slice], dtype=np.complex128),
        "centered_covariance_convention": np.asarray("G=2C-I"),
    }
    if task.alpha_1 == 1.0:
        if endpoint_modes is None:
            raise ValueError("alpha1=1 requires the endpoint slow-mode products")
        payload.update({key: value[sample_slice] for key, value in endpoint_modes.items()})
        payload["slowest_mode_definition"] = np.asarray("argmin abs(log((1-nu)/nu)/(2T)); not most negative rate")
        payload["slowest_mode_occupation_cutoff"] = np.asarray(1e-9)
        payload["slowest_mode_cycle"] = np.asarray(task.cycles)
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
) -> None:
    print(f"[task start] {task.task_id}; alpha1={task.alpha_1:g}; samples={task.samples}; cycles={task.cycles}", flush=True)
    if config["device"].startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(config["device"]))
    model = build_model(config, task.construction, task.ny, task.alpha_1)
    observer = PurificationObserver(
        nx=task.nx,
        ny=task.ny,
        cycles=task.cycles,
        sample_indices=task.sample_indices,
        sample_chunk=int(config["observer_sample_chunk_by_Ny"][str(task.ny)]),
        construction=task.construction,
        save_contours=(task.alpha_1 == 1.0),
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
        desc=f"hard alpha1={task.alpha_1:g} Ny={task.ny}",
        unit="cycle",
        leave=False,
    ) as cycle_bar:
        while completed_cycle < task.cycles:
            require_space(output_root, scratch_root)
            segment = min(SEGMENT_CYCLES, task.cycles - completed_cycle)
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
            cycle_bar.set_postfix(durable_cycle=completed_cycle)
            memory = check_gpu_memory(config)
            print(f"[checkpoint] {task.task_id}: verified cycle {completed_cycle}/{task.cycles}; "
                  f"GPU {json.dumps(memory)}", flush=True)
            print(f"[timing] cumulative compute {elapsed_seconds / completed_cycle:.2f} s/cycle; "
                  f"remaining compute in batch ~{elapsed_seconds / completed_cycle * (task.cycles - completed_cycle) / 3600:.2f} h",
                  flush=True)

    assert G is not None
    observer.validate(completed_cycle=task.cycles, final=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    endpoint_modes = None
    if task.alpha_1 == 1.0:
        print("[endpoint] extracting minimum-|lambda| eigenmode per trajectory", flush=True)
        endpoint_modes = extract_slowest_modes(
            G, nx=task.nx, ny=task.ny, cycles=task.cycles,
            device=config["device"], sample_chunk=int(config["observer_sample_chunk_by_Ny"][str(task.ny)]),
        )
    for shard in result_shards(task):
        valid, _ = verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)
        if valid:
            continue
        publish_result(
            output_root,
            scratch_root,
            shard,
            G_final=G,
            observer=observer,
            elapsed_seconds=elapsed_seconds,
            cfg_hash=cfg_hash,
            hashes=hashes,
            endpoint_modes=endpoint_modes,
        )
        shard_bar.update(1)
        if hasattr(shard_bar, "set_postfix"):
            shard_bar.set_postfix(completed=shard_bar.n, pending=shard_bar.total - shard_bar.n, failed=0)
    for shard in result_shards(task):
        valid, reason = verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)
        if not valid:
            raise RuntimeError(f"batch result failed final verification: {reason}")
    remove_checkpoint(output_root, task)
    shutil.rmtree(scratch_root / task.task_id, ignore_errors=True)


def check_gpu_memory(config: dict[str, Any]) -> dict[str, float]:
    if not config["device"].startswith("cuda"):
        return {}
    device = torch.device(config["device"])
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    peak = torch.cuda.max_memory_reserved(device)
    if used > GPU_MEMORY_BUDGET_BYTES or peak > GPU_ALLOCATOR_LIMIT_BYTES:
        raise RuntimeError(
            f"GPU safety budget exceeded: device used={used / 1e9:.2f} GB, "
            f"peak allocator={peak / 1e9:.2f} GB. Resume from the last verified checkpoint; "
            "do not change batch identities."
        )
    return {"device_used_GB": used / 1e9, "peak_reserved_GB": peak / 1e9}


def require_runtime(config: dict[str, Any]) -> None:
    if config["device"].startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        props = torch.cuda.get_device_properties(torch.device(config["device"]))
        gib = props.total_memory / 1024**3
        name = torch.cuda.get_device_name(torch.device(config["device"]))
        if "A100" not in name or gib < 35.0:
            raise RuntimeError(f"expected A100 40-GB-class GPU, got {name} ({gib:.2f} GiB)")
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(torch.device(config["device"]))
        external = max(0, total - free - torch.cuda.memory_reserved(torch.device(config["device"])))
        allocator_bytes = min(GPU_ALLOCATOR_LIMIT_BYTES, GPU_MEMORY_BUDGET_BYTES - external - 1_000_000_000)
        if allocator_bytes < 30_000_000_000:
            raise RuntimeError("Too much GPU memory is already in use; use a fresh A100 runtime")
        torch.cuda.set_per_process_memory_fraction(allocator_bytes / props.total_memory,
                                                  torch.device(config["device"]))
        print(json.dumps({"gpu": name, "total_GiB": gib,
                          "device_budget_GB": GPU_MEMORY_BUDGET_BYTES / 1e9,
                          "allocator_limit_GB": allocator_bytes / 1e9}, indent=2), flush=True)
    if torch.complex128 != getattr(torch, config["dtype"]):
        raise RuntimeError("locked dtype is not complex128")


def require_space(output_root: Path, scratch_root: Path) -> None:
    scratch_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    local_free = shutil.disk_usage(scratch_root).free
    drive_free = shutil.disk_usage(output_root).free
    if local_free < 6 * 1024**3:
        raise OSError("less than 6 GiB free under local scratch")
    if drive_free < 12 * 1024**3:
        raise OSError("less than 12 GiB free under the Drive output root")


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
        shard.execution.task_id + "|" + shard.result_id: verified_complete(
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
    inventory = {
        "config": config,
        "source_root": str(BUNDLE_ROOT),
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
    }
    print("[resume inventory]", flush=True)
    print(json.dumps(inventory, indent=2, sort_keys=True), flush=True)
    if report_only:
        return inventory
    require_runtime(config)
    require_space(output_root, scratch_root)

    completed_batches = 0
    measured_seconds = 0.0
    measured_sample_cycles = 0
    with tqdm(
        total=len(shards),
        initial=completed_count,
        desc=f"{construction} durable shards",
        unit="shard",
    ) as shard_bar:
        shard_bar.set_postfix(completed=completed_count, skipped=completed_count,
                              pending=len(shards) - completed_count, failed=0)
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
            previous, _ = load_checkpoint(output_root, task, cfg_hash=cfg_hash, hashes=hashes)
            cycles_before = previous.completed_cycle if previous is not None else 0
            del previous
            started = time.perf_counter()
            try:
                execute_batch(
                    config,
                    output_root,
                    scratch_root,
                    task,
                    cfg_hash=cfg_hash,
                    hashes=hashes,
                    shard_bar=shard_bar,
                )
            except Exception as exc:
                if isinstance(exc, OccupationSpectrumError):
                    # Diagnostic evidence is NOT a resumable checkpoint or completion.
                    # It never replaces the last good covariance/RNG checkpoint.
                    try:
                        local = scratch_root / task.task_id / "failure_diagnostics"
                        local.mkdir(parents=True, exist_ok=True)
                        report = {**exc.diagnostics, "task": task.task_id,
                                  "configuration_hash": cfg_hash, "source_hashes": hashes,
                                  "message": str(exc)}
                        np.savez(local / "occupation_failure.npz", G=exc.covariance,
                                 occupation_spectrum=exc.occupations)
                        write_json(local / "occupation_failure.json", report)
                        for name in ("occupation_failure.npz", "occupation_failure.json"):
                            publish_file(local / name, output_root / "diagnostics" / task.task_id / name)
                        print(f"[diagnostics] {json.dumps(report, sort_keys=True)}", flush=True)
                    except Exception as diagnostic_error:
                        print(f"[diagnostics save failed] {diagnostic_error}; original error follows", flush=True)
                shard_bar.set_postfix(completed=shard_bar.n, pending=len(shards) - shard_bar.n, failed=1)
                print(f"[failed] {task.task_id}; completed results and any last valid checkpoint are preserved", flush=True)
                raise
            completed_batches += 1
            work = task.samples * (task.cycles - cycles_before)
            if work:
                measured_seconds += time.perf_counter() - started
                measured_sample_cycles += work
                remaining = sum(
                    future.samples * future.cycles for future in tasks
                    if not all(verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)[0]
                               for shard in result_shards(future))
                )
                eta = remaining * measured_seconds / measured_sample_cycles / 3600
                print(f"[ETA] remaining <=~{eta:.2f} hours at measured throughput "
                      "(counts pending batches at full length)", flush=True)

    completed_final = sum(
        verified_complete(output_root, shard, cfg_hash=cfg_hash, hashes=hashes)[0]
        for shard in shards
    )
    summary = {**inventory, "completed_shards": completed_final, "pending_shards": len(shards) - completed_final}
    print("[campaign summary]", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--construction", choices=("hard",), default="hard")
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
