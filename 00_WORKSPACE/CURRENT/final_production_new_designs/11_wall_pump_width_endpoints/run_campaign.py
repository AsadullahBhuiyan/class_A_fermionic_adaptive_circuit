#!/usr/bin/env python3
"""Generate width-controlled monitored-circuit endpoint frames on an A100."""

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
from typing import Any, Mapping

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


BUNDLE = "11_wall_pump_width_endpoints"
REVISION = "wall_pump_width_endpoints_s100_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
RESULT_SCHEMA = "wall_pump_width_endpoint_shard_v1"
COMPLETION_SCHEMA = "wall_pump_width_endpoint_shard_completion_v1"
CHECKPOINT_SCHEMA = "wall_pump_width_endpoint_checkpoint_v1"
BENCHMARK_SCHEMA = "wall_pump_width_endpoint_a100_benchmark_v1"
BACKEND = "gpu"
NY = 24
CYCLES = 48
SHARD_SIZE = 5
CHECKPOINT_CYCLES = 5
PRIMARY_CELLS = (("nsh1", 28), ("nsh1", 32), ("dense", 24), ("dense", 28), ("dense", 32))
BRIDGE_CELLS = (("nsh1", 20), ("nsh1", 24), ("dense", 20))
WALLS = ("soft", "hard")
WALL_FLAGS = {
    "soft": {"dw_truncation": False, "meas_slab_only": False},
    "hard": {"dw_truncation": True, "meas_slab_only": True},
}
SHELLS = {"nsh1": 1, "dense": None}
BENCHMARK_CANDIDATES = (10, 20, 40, 60, 80, 100)
MINIMUM_HEADROOM_BYTES = 8 * 1024**3
MAXIMUM_A100_FORECAST_SECONDS = 40.0 * 3600.0
MINIMUM_PRODUCTION_DRIVE_FREE_BYTES = 25 * 1024**3
SOURCE_FILES = (
    "run_campaign.py",
    "campaign_config.json",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)


@dataclass(frozen=True)
class Case:
    collection: str
    protocol: str
    nx: int
    wall: str
    samples: int

    @property
    def key(self) -> str:
        return f"{self.collection}:{self.protocol}:N{self.nx}x{NY}:{self.wall}"

    @property
    def cell(self) -> str:
        return f"{self.protocol}_N{self.nx}x{NY}"

    @property
    def wall_locations(self) -> tuple[int, int]:
        return self.nx // 4, 3 * self.nx // 4


@dataclass(frozen=True)
class ExecutionBatch:
    case: Case
    batch_index: int
    sample_start: int
    sample_stop: int
    seed: int

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))

    @property
    def task_id(self) -> str:
        return (
            f"{self.case.collection}_{self.case.protocol}_N{self.case.nx}x{NY}_"
            f"{self.case.wall}_batch-{self.batch_index:02d}_"
            f"samples-{self.sample_start:03d}-{self.sample_stop - 1:03d}"
        )


@dataclass(frozen=True)
class ResultShard:
    batch: ExecutionBatch
    shard_index: int
    sample_start: int
    sample_stop: int

    @property
    def case(self) -> Case:
        return self.batch.case

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))

    @property
    def local_slice(self) -> slice:
        return slice(
            self.sample_start - self.batch.sample_start,
            self.sample_stop - self.batch.sample_start,
        )

    @property
    def task_id(self) -> str:
        return (
            f"{self.case.collection}_{self.case.protocol}_N{self.case.nx}x{NY}_"
            f"{self.case.wall}_shard-{self.shard_index:02d}"
        )


@dataclass
class Checkpoint:
    completed_cycle: int
    elapsed_seconds: float
    frame: np.ndarray
    ranks: np.ndarray
    seen_cycles: np.ndarray
    global_charge: np.ndarray
    gram_residual: np.ndarray
    rng: dict[str, np.ndarray]


class ChargeObserver:
    """Small checkpointable rank observer; it never materializes a covariance."""

    def __init__(self, samples: int) -> None:
        self.samples = int(samples)
        self.seen_cycles = np.zeros(CYCLES + 1, dtype=np.bool_)
        self.global_charge = np.full((self.samples, CYCLES + 1), -1, dtype=np.int64)

    def __call__(
        self,
        *,
        cycle: int,
        state: Any,
        batch_start: int = 0,
        batch_count: int | None = None,
        **_: Any,
    ) -> None:
        cycle = int(cycle)
        count = self.samples if batch_count is None else int(batch_count)
        if int(batch_start) != 0 or count != self.samples:
            raise ValueError("the observer requires one complete execution batch")
        if not 0 <= cycle <= CYCLES:
            raise IndexError("cycle lies outside 0..48")
        if self.seen_cycles[cycle]:
            raise RuntimeError(f"duplicate observation at cycle {cycle}")
        ranks = state.ranks.detach().cpu().numpy().astype(np.int64, copy=False)
        if ranks.shape != (self.samples,) or np.any(ranks < 0):
            raise ValueError("invalid native occupied-frame ranks")
        self.global_charge[:, cycle] = ranks
        self.seen_cycles[cycle] = True

    def restore(self, seen_cycles: np.ndarray, global_charge: np.ndarray) -> None:
        seen = np.asarray(seen_cycles, dtype=np.bool_)
        charge = np.asarray(global_charge, dtype=np.int64)
        if seen.shape != self.seen_cycles.shape or charge.shape != self.global_charge.shape:
            raise ValueError("checkpoint observer shape mismatch")
        self.seen_cycles[:] = seen
        self.global_charge[:] = charge
        self.validate(require_complete=False)

    def validate(self, *, require_complete: bool) -> None:
        observed = np.flatnonzero(self.seen_cycles)
        if observed.size and not np.array_equal(observed, np.arange(observed[-1] + 1)):
            raise RuntimeError("observed cycles are not a contiguous prefix")
        if np.any(self.global_charge[:, self.seen_cycles] < 0):
            raise RuntimeError("observed charge is negative")
        if require_complete and not bool(np.all(self.seen_cycles)):
            raise RuntimeError("charge history is incomplete")


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_config(bundle_root: Path = BUNDLE_ROOT) -> dict[str, Any]:
    return json.loads((bundle_root / "campaign_config.json").read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    observed = json.loads(json.dumps(dict(config)))
    expected = expected_config()
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise ValueError(
            "configuration differs from the locked campaign contract in: "
            + ", ".join(differing)
        )
    if observed["sampling_revision"] != REVISION or observed["root_seed"] != 2026090701:
        raise ValueError("sampling identity is not the locked v1 identity")
    if observed["execution_backend"] != BACKEND:
        raise ValueError("this bundle is locked to the GPU backend")
    return observed


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(validate_config(config)).encode("utf-8"))


def source_hashes(bundle_root: Path = BUNDLE_ROOT) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = bundle_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing required source file: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def expand_cases(config: Mapping[str, Any]) -> list[Case]:
    validate_config(config)
    cases: list[Case] = []
    for collection, cells, samples in (
        ("endpoints", PRIMARY_CELLS, 100),
        ("bridge", BRIDGE_CELLS, 25),
    ):
        for protocol, nx in cells:
            for wall in WALLS:
                cases.append(Case(collection, protocol, nx, wall, samples))
    if len(cases) != 16 or sum(case.samples for case in cases) != 1150:
        raise RuntimeError("locked case table must contain 16 cases and 1,150 endpoints")
    if sum(case.samples for case in cases if case.collection == "endpoints") != 1000:
        raise RuntimeError("primary case table must contain exactly 1,000 endpoints")
    if sum(case.samples for case in cases if case.collection == "bridge") != 150:
        raise RuntimeError("bridge case table must contain exactly 150 endpoints")
    return cases


def case_batch_map(cases: list[Case], selected_batch_size: int) -> dict[str, int]:
    selected = int(selected_batch_size)
    if selected not in BENCHMARK_CANDIDATES:
        raise ValueError("selected execution batch is not a benchmark candidate")
    return {case.key: min(selected, case.samples) for case in cases}


def resolved_execution_identity(
    config: Mapping[str, Any], selected_batch_size: int
) -> dict[str, Any]:
    cases = expand_cases(config)
    return {
        "sampling_revision": REVISION,
        "base_config_sha256": config_sha256(config),
        "execution_backend": BACKEND,
        "selected_execution_batch_size": int(selected_batch_size),
        "execution_batch_map": case_batch_map(cases, selected_batch_size),
    }


def resolved_execution_sha256(config: Mapping[str, Any], selected_batch_size: int) -> str:
    return sha256_bytes(
        canonical_json(resolved_execution_identity(config, selected_batch_size)).encode("utf-8")
    )


def _batch_seed(config: Mapping[str, Any], case: Case, start: int, stop: int) -> int:
    label = (
        f"{config['root_seed']}|{REVISION}|backend={BACKEND}|{case.key}|"
        f"samples={start}:{stop}"
    )
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def expand_execution_batches(
    config: Mapping[str, Any], selected_batch_size: int
) -> list[ExecutionBatch]:
    cases = expand_cases(config)
    batch_map = case_batch_map(cases, selected_batch_size)
    tasks: list[ExecutionBatch] = []
    for case in cases:
        size = batch_map[case.key]
        for batch_index, start in enumerate(range(0, case.samples, size)):
            stop = min(case.samples, start + size)
            if start % SHARD_SIZE or stop % SHARD_SIZE:
                raise RuntimeError("execution boundaries must align to five-sample shards")
            tasks.append(
                ExecutionBatch(
                    case=case,
                    batch_index=batch_index,
                    sample_start=start,
                    sample_stop=stop,
                    seed=_batch_seed(config, case, start, stop),
                )
            )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("execution task IDs are not unique")
    if len({task.seed for task in tasks}) != len(tasks):
        raise RuntimeError("execution seeds are not unique")
    return tasks


def result_shards(batch: ExecutionBatch) -> list[ResultShard]:
    shards: list[ResultShard] = []
    for start in range(batch.sample_start, batch.sample_stop, SHARD_SIZE):
        stop = start + SHARD_SIZE
        shards.append(
            ResultShard(
                batch=batch,
                shard_index=start // SHARD_SIZE,
                sample_start=start,
                sample_stop=stop,
            )
        )
    if any(len(shard.sample_ids) != SHARD_SIZE for shard in shards):
        raise RuntimeError("every durable result must contain five trajectories")
    return shards


def all_result_shards(config: Mapping[str, Any], selected_batch_size: int) -> list[ResultShard]:
    shards = [
        shard
        for batch in expand_execution_batches(config, selected_batch_size)
        for shard in result_shards(batch)
    ]
    if len(shards) != 230 or len({shard.task_id for shard in shards}) != 230:
        raise RuntimeError("locked campaign must contain exactly 230 durable shards")
    return shards


def result_paths(output_root: Path, shard: ResultShard) -> tuple[Path, Path]:
    case = shard.case
    directory = (
        output_root
        / case.collection
        / case.protocol
        / f"N{case.nx}x{NY}"
        / case.wall
    )
    stem = f"shard_{shard.shard_index:02d}"
    return directory / f"{stem}.npz", directory / f"{stem}.completion.json"


def checkpoint_paths(output_root: Path, batch: ExecutionBatch) -> tuple[Path, Path]:
    case = batch.case
    directory = (
        output_root
        / "checkpoints"
        / case.collection
        / case.protocol
        / f"N{case.nx}x{NY}"
        / case.wall
        / f"execution_{batch.batch_index:02d}"
    )
    return directory / "checkpoint.npz", directory / "checkpoint.json"


def benchmark_path(output_root: Path) -> Path:
    return output_root / "benchmarks" / "a100_endpoint_benchmark.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    try:
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _coerce_npz(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"NPZ field {key!r} has forbidden object dtype")
        arrays[str(key)] = array
    return arrays


def _write_npz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **_coerce_npz(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    """Publish through DriveFS only after temporary and stable-path readback."""

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
    return {"filename": final_path.name, "bytes": expected_bytes, "sha256": expected_sha256}


def validate_a100() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an A100 GPU runtime")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_bytes = int(properties.total_memory)
    name = str(properties.name)
    if "A100" not in name.upper() or total_bytes < 38 * 1024**3:
        raise RuntimeError(
            f"a 40-GB-class NVIDIA A100 is required; found {name!r}, "
            f"{total_bytes / 1024**3:.2f} GiB"
        )
    probe = torch.zeros(1, dtype=torch.complex128, device=device)
    del probe
    return {"name": name, "total_bytes": total_bytes, "device": str(device)}


def _seed_rng(seed: int) -> None:
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_rng() -> dict[str, np.ndarray]:
    state = np.random.get_state()
    payload: dict[str, np.ndarray] = {
        "numpy_algorithm": np.asarray(state[0]),
        "numpy_keys": np.asarray(state[1], dtype=np.uint32),
        "numpy_position": np.asarray(state[2], dtype=np.int64),
        "numpy_has_gauss": np.asarray(state[3], dtype=np.int8),
        "numpy_cached_gaussian": np.asarray(state[4], dtype=np.float64),
        "torch_cpu": torch.get_rng_state().detach().cpu().numpy(),
    }
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload["torch_cuda_count"] = np.asarray(len(cuda_states), dtype=np.int64)
    for index, state_tensor in enumerate(cuda_states):
        payload[f"torch_cuda_{index}"] = state_tensor.detach().cpu().numpy()
    return payload


def _restore_rng(payload: Mapping[str, np.ndarray]) -> None:
    np.random.set_state(
        (
            str(np.asarray(payload["numpy_algorithm"]).item()),
            np.asarray(payload["numpy_keys"], dtype=np.uint32),
            int(np.asarray(payload["numpy_position"]).item()),
            int(np.asarray(payload["numpy_has_gauss"]).item()),
            float(np.asarray(payload["numpy_cached_gaussian"]).item()),
        )
    )
    torch.set_rng_state(torch.as_tensor(payload["torch_cpu"], dtype=torch.uint8))
    saved_count = int(np.asarray(payload["torch_cuda_count"]).item())
    current_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_count != current_count:
        raise RuntimeError(
            f"checkpoint CUDA device count changed: saved={saved_count}, current={current_count}"
        )
    if saved_count:
        torch.cuda.set_rng_state_all(
            [
                torch.as_tensor(payload[f"torch_cuda_{index}"], dtype=torch.uint8)
                for index in range(saved_count)
            ]
        )


def build_model(config: Mapping[str, Any], case: Case) -> classA_U1FGTN_gpu:
    flags = WALL_FLAGS[case.wall]
    nshell = SHELLS[case.protocol]
    model = classA_U1FGTN_gpu(
        Nx=case.nx,
        Ny=NY,
        DW=True,
        nshell=nshell,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=bool(flags["dw_truncation"]),
        triv_region_local_mode=False,
        device=str(config["device"]),
        dtype=str(config["dtype"]),
        backend="dense" if nshell is None else "local",
    )
    if tuple(int(value) for value in model.DW_loc) != case.wall_locations:
        raise RuntimeError(
            f"unexpected domain-wall locations {model.DW_loc}; expected {case.wall_locations}"
        )
    if model.dtype != torch.complex128:
        raise RuntimeError("constructed model is not complex128")
    return model


def _run_engine(
    *,
    model: classA_U1FGTN_gpu,
    case: Case,
    samples: int,
    cycles: int,
    frame: np.ndarray | None = None,
    ranks: np.ndarray | None = None,
    observer: Any | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    continuing = frame is not None
    if continuing != (ranks is not None):
        raise ValueError("frame and ranks must be supplied together")
    flags = WALL_FLAGS[case.wall]
    result = model.run_markov_circuit(
        G_history=False,
        progress=bool(progress),
        cycles=int(cycles),
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        samples=int(samples),
        init_mode="default",
        frame_init=frame,
        frame_ranks=ranks,
        frame_init_prepared=continuing,
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=bool(flags["meas_slab_only"]),
        batch_size=int(samples),
        return_data=True,
        state_representation="physical_frame",
        native_cycle_observer=observer,
        track_choi=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    if int(result.get("samples", -1)) != int(samples):
        raise RuntimeError("canonical engine returned the wrong sample count")
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("canonical engine did not use the physical-frame representation")
    if bool(result.get("choi_tracked", False)) or bool(result.get("lyapunov_tracked", False)):
        raise RuntimeError("canonical engine unexpectedly tracked Choi or tangent data")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("canonical engine materialized a covariance")
    if bool(result.get("frame_init_prepared", False)) != continuing:
        raise RuntimeError("canonical continuation metadata is inconsistent")
    expected_exterior = case.wall == "hard" and not continuing
    if bool(result.get("exterior_preparation_performed", False)) != expected_exterior:
        raise RuntimeError("canonical exterior-preparation metadata is inconsistent")
    native = result.get("native_final")
    if not isinstance(native, dict) or "frame" not in native or "ranks" not in native:
        raise RuntimeError("canonical engine did not return a native occupied frame")
    native_frame = np.asarray(native["frame"])
    native_ranks = np.asarray(native["ranks"], dtype=np.int64)
    if native_frame.dtype != np.complex128:
        raise RuntimeError(f"native frame dtype is {native_frame.dtype}, expected complex128")
    if native_frame.shape[:2] != (int(samples), 2 * case.nx * NY):
        raise RuntimeError(f"native frame geometry is invalid: {native_frame.shape}")
    if native_ranks.shape != (int(samples),):
        raise RuntimeError("native rank vector is invalid")
    return result


def select_fastest_safe_candidate(rows: list[Mapping[str, Any]]) -> int:
    accepted = [
        row
        for row in rows
        if row.get("error") is None
        and int(row.get("projected_headroom_bytes", 0)) >= MINIMUM_HEADROOM_BYTES
        and float(row.get("trajectory_cycles_per_second", 0.0)) > 0.0
    ]
    if not accepted:
        raise RuntimeError("no execution-batch candidate retains 8 GiB of A100 headroom")
    return int(
        max(
            accepted,
            key=lambda row: (
                float(row["trajectory_cycles_per_second"]),
                int(row["candidate"]),
            ),
        )["candidate"]
    )


def _benchmark_identity(
    *, config: Mapping[str, Any], hashes: Mapping[str, str], gpu_name: str
) -> dict[str, Any]:
    return {
        "schema": BENCHMARK_SCHEMA,
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "base_config_sha256": config_sha256(config),
        "source_hashes": dict(hashes),
        "execution_backend": BACKEND,
        "gpu_name": str(gpu_name),
        "candidate_execution_batch_sizes": list(BENCHMARK_CANDIDATES),
        "short_benchmark_cycles": 5,
        "minimum_gpu_headroom_bytes": MINIMUM_HEADROOM_BYTES,
        "full_timing_samples_per_combination": 5,
        "local_56_core_forecast_seconds": 80.0 * 3600.0,
        "maximum_a100_forecast_seconds": MAXIMUM_A100_FORECAST_SECONDS,
        "worst_case": {"protocol": "dense", "Nx": 32, "Ny": NY, "wall": "soft"},
    }


def load_benchmark(
    *,
    output_root: Path,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    gpu_name: str,
) -> tuple[dict[str, Any] | None, str]:
    path = benchmark_path(output_root)
    if not path.is_file():
        return None, "missing benchmark receipt"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable benchmark receipt: {exc}"
    for key, value in _benchmark_identity(
        config=config, hashes=hashes, gpu_name=gpu_name
    ).items():
        if payload.get(key) != value:
            return None, f"benchmark identity mismatch: {key}"
    selected = int(payload.get("selected_execution_batch_size", -1))
    if selected not in BENCHMARK_CANDIDATES:
        return None, "benchmark selected an invalid execution batch"
    expected_resolved = resolved_execution_identity(config, selected)
    if payload.get("resolved_execution_identity") != expected_resolved:
        return None, "benchmark execution map mismatch"
    if payload.get("resolved_execution_sha256") != resolved_execution_sha256(config, selected):
        return None, "benchmark resolved-config checksum mismatch"
    if payload.get("status") != "accepted":
        return None, "benchmark gate was not accepted"
    projected = float(payload.get("projected_missing_1000_seconds", np.inf))
    if not np.isfinite(projected) or projected > MAXIMUM_A100_FORECAST_SECONDS:
        return None, "benchmark exceeds the locked 40-hour A100 gate"
    full_rows = payload.get("full_48_cycle_timing_rows")
    if not isinstance(full_rows, list) or len(full_rows) != 10:
        return None, "benchmark lacks the ten production-shaped timing rows"
    return payload, "verified"


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmark(
    *,
    output_root: Path,
    scratch_root: Path,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    gpu: Mapping[str, Any],
) -> dict[str, Any]:
    """Benchmark batching, then time one full five-sample shard per primary arm."""

    worst = Case("benchmark", "dense", 32, "soft", 100)
    model = build_model(config, worst)
    rows: list[dict[str, Any]] = []
    print("[benchmark] 5-cycle Nx=32 dense/soft execution-batch sweep", flush=True)
    for candidate in tqdm(
        BENCHMARK_CANDIDATES,
        desc="A100 batch sweep",
        unit="candidate",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        _clear_cuda()
        _seed_rng(2026090790 + candidate)
        torch.cuda.reset_peak_memory_stats(model.device)
        started = time.monotonic()
        error: str | None = None
        result: dict[str, Any] | None = None
        try:
            result = _run_engine(
                model=model,
                case=worst,
                samples=candidate,
                cycles=5,
                progress=False,
            )
            elapsed = time.monotonic() - started
            peak_reserved = int(torch.cuda.max_memory_reserved(model.device))
            throughput = candidate * 5.0 / max(elapsed, np.finfo(float).tiny)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            elapsed = time.monotonic() - started
            peak_reserved = int(torch.cuda.max_memory_reserved(model.device))
            throughput = 0.0
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
        rows.append(
            {
                "candidate": candidate,
                "elapsed_seconds": float(elapsed),
                "peak_cuda_reserved_bytes": peak_reserved,
                "projected_headroom_bytes": int(gpu["total_bytes"]) - peak_reserved,
                "trajectory_cycles_per_second": float(throughput),
                "error": error,
            }
        )
        del result
        _clear_cuda()
    selected = select_fastest_safe_candidate(rows)
    print(f"[benchmark] selected execution batch={selected}", flush=True)

    full_rows: list[dict[str, Any]] = []
    primary_cases = [case for case in expand_cases(config) if case.collection == "endpoints"]
    print("[benchmark] ten full 48-cycle, five-sample production-shaped timings", flush=True)
    for index, case in enumerate(
        tqdm(
            primary_cases,
            desc="full endpoint timings",
            unit="arm",
            dynamic_ncols=True,
            file=sys.stdout,
        )
    ):
        _clear_cuda()
        timing_model = build_model(config, case)
        _seed_rng(2026090800 + index)
        torch.cuda.reset_peak_memory_stats(timing_model.device)
        started = time.monotonic()
        result = _run_engine(
            model=timing_model,
            case=case,
            samples=5,
            cycles=CYCLES,
            progress=False,
        )
        elapsed = time.monotonic() - started
        native = result["native_final"]
        gram = np.asarray(native.get("gram_residual", []), dtype=np.float64)
        if gram.shape != (5,) or not np.isfinite(gram).all() or np.max(gram) > 1.0e-10:
            raise RuntimeError(f"benchmark endpoint Gram validation failed for {case.key}")
        full_rows.append(
            {
                "case": case.key,
                "samples": 5,
                "cycles": CYCLES,
                "selected_execution_batch_size": min(selected, case.samples),
                "elapsed_seconds": float(elapsed),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(timing_model.device)),
                "projected_100_sample_seconds": float(elapsed * 20.0),
                "maximum_gram_residual": float(np.max(gram)),
            }
        )
        del result, native, timing_model
        _clear_cuda()
    projected = float(sum(row["projected_100_sample_seconds"] for row in full_rows))
    status = "accepted" if projected <= MAXIMUM_A100_FORECAST_SECONDS else "rejected"
    payload = {
        **_benchmark_identity(config=config, hashes=hashes, gpu_name=str(gpu["name"])),
        "status": status,
        "selected_execution_batch_size": selected,
        "resolved_execution_identity": resolved_execution_identity(config, selected),
        "resolved_execution_sha256": resolved_execution_sha256(config, selected),
        "short_sweep_rows": rows,
        "full_48_cycle_timing_rows": full_rows,
        "projected_missing_1000_seconds": projected,
        "projected_missing_1000_hours": projected / 3600.0,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    local = scratch_root / "benchmark" / "a100_endpoint_benchmark.json"
    _write_json(local, payload)
    publish_file(local, benchmark_path(output_root))
    verified, reason = load_benchmark(
        output_root=output_root,
        config=config,
        hashes=hashes,
        gpu_name=str(gpu["name"]),
    )
    if status != "accepted":
        raise RuntimeError(
            f"A100 projection {projected / 3600.0:.2f} h exceeds the 40 h gate"
        )
    if verified is None:
        raise RuntimeError(f"published benchmark failed verification: {reason}")
    print(
        f"[benchmark accepted] selected={selected}; projected missing 1000="
        f"{projected / 3600.0:.2f} h <= 40.00 h",
        flush=True,
    )
    return verified


def _shard_identity(
    *,
    shard: ResultShard,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
) -> dict[str, Any]:
    case = shard.case
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "execution_backend": BACKEND,
        "stage": "endpoint_shard",
        "task_id": shard.task_id,
        "execution_batch_id": shard.batch.task_id,
        "collection": case.collection,
        "cell": case.cell,
        "protocol": case.protocol,
        "Nx": case.nx,
        "Ny": NY,
        "wall": case.wall,
        "wall_locations": list(case.wall_locations),
        "samples": SHARD_SIZE,
        "shard_index": shard.shard_index,
        "sample_ids": list(shard.sample_ids),
        "execution_batch_seed": shard.batch.seed,
        "cycles": CYCLES,
        "config_hash": resolved_execution_sha256(config, selected_batch_size),
        "base_config_sha256": config_sha256(config),
        "resolved_execution_sha256": resolved_execution_sha256(config, selected_batch_size),
        "selected_execution_batch_size": int(selected_batch_size),
        "execution_batch_map": case_batch_map(expand_cases(config), selected_batch_size),
        "source_hashes": dict(hashes),
    }


def verified_complete(
    *,
    output_root: Path,
    shard: ResultShard,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
) -> tuple[bool, str]:
    result_path, completion_path = result_paths(output_root, shard)
    if not result_path.exists() and not completion_path.exists():
        return False, "missing result/completion pair"
    if not result_path.is_file() or not completion_path.is_file():
        return False, "incomplete result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion JSON: {exc}"
    for key, value in _shard_identity(
        shard=shard,
        config=config,
        hashes=hashes,
        selected_batch_size=selected_batch_size,
    ).items():
        if completion.get(key) != value:
            return False, f"completion identity mismatch: {key}"
    result_record = completion.get("result", {})
    if completion.get("result_filename") != result_path.name or result_record.get("name") != result_path.name:
        return False, "completion result filename mismatch"
    try:
        size = int(result_path.stat().st_size)
        digest = sha256_file(result_path)
    except OSError as exc:
        return False, f"result readback failed: {exc}"
    if int(completion.get("result_bytes", -1)) != size:
        return False, "result byte-count mismatch"
    if completion.get("result_sha256") != digest:
        return False, "result checksum mismatch"
    if result_record != {"name": result_path.name, "bytes": size, "sha256": digest}:
        return False, "nested result receipt mismatch"
    try:
        with np.load(result_path, allow_pickle=False) as archive:
            if str(archive["schema"].item()) != RESULT_SCHEMA:
                return False, "result schema mismatch"
            if not np.array_equal(archive["sample_ids"], np.asarray(shard.sample_ids)):
                return False, "result sample IDs mismatch"
            frame = archive["frames"]
            ranks = archive["ranks"]
            if frame.dtype != np.complex128 or frame.ndim != 3:
                return False, "result frame dtype/shape mismatch"
            if frame.shape[:2] != (SHARD_SIZE, 2 * shard.case.nx * NY):
                return False, "result frame geometry mismatch"
            if ranks.shape != (SHARD_SIZE,) or np.any(ranks < 0) or np.any(ranks > frame.shape[2]):
                return False, "result ranks mismatch"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"result NPZ validation failed: {exc}"
    return True, "verified"


def _checkpoint_identity(
    *,
    batch: ExecutionBatch,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
) -> dict[str, Any]:
    case = batch.case
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint",
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "execution_backend": BACKEND,
        "task_id": batch.task_id,
        "collection": case.collection,
        "protocol": case.protocol,
        "Nx": case.nx,
        "Ny": NY,
        "wall": case.wall,
        "wall_locations": list(case.wall_locations),
        "sample_start": batch.sample_start,
        "sample_stop": batch.sample_stop,
        "sample_ids": list(batch.sample_ids),
        "execution_batch_seed": batch.seed,
        "cycles": CYCLES,
        "checkpoint_cycles": CHECKPOINT_CYCLES,
        "base_config_sha256": config_sha256(config),
        "resolved_execution_sha256": resolved_execution_sha256(config, selected_batch_size),
        "selected_execution_batch_size": int(selected_batch_size),
        "execution_batch_map": case_batch_map(expand_cases(config), selected_batch_size),
        "source_hashes": dict(hashes),
    }


def save_checkpoint(
    *,
    output_root: Path,
    scratch_root: Path,
    batch: ExecutionBatch,
    completed_cycle: int,
    elapsed_seconds: float,
    native: Mapping[str, Any],
    observer: ChargeObserver,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
) -> dict[str, np.ndarray]:
    frame = np.asarray(native["frame"])
    ranks = np.asarray(native["ranks"], dtype=np.int64)
    if frame.dtype != np.complex128 or frame.shape[:2] != (
        batch.sample_count,
        2 * batch.case.nx * NY,
    ):
        raise RuntimeError("checkpoint frame dtype/shape mismatch")
    observer.validate(require_complete=completed_cycle == CYCLES)
    rng = _capture_rng()
    payload: dict[str, Any] = {
        "completed_cycle": np.asarray(completed_cycle, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "frame": frame,
        "ranks": ranks,
        "sample_ids": np.asarray(batch.sample_ids, dtype=np.int64),
        "seen_cycles": observer.seen_cycles,
        "global_charge": observer.global_charge,
        "gram_residual": np.asarray(native["gram_residual"], dtype=np.float64),
    }
    payload.update({f"rng__{key}": value for key, value in rng.items()})
    local_dir = scratch_root / batch.task_id
    local_npz = local_dir / "checkpoint.npz"
    local_json = local_dir / "checkpoint.json"
    _write_npz(local_npz, payload)
    final_npz, final_json = checkpoint_paths(output_root, batch)
    published = publish_file(local_npz, final_npz)
    metadata = {
        **_checkpoint_identity(
            batch=batch,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
        ),
        "completed_cycle": int(completed_cycle),
        "elapsed_seconds": float(elapsed_seconds),
        "checkpoint_filename": final_npz.name,
        "checkpoint_bytes": published["bytes"],
        "checkpoint_sha256": published["sha256"],
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write_json(local_json, metadata)
    publish_file(local_json, final_json)
    return rng


def load_checkpoint(
    *,
    output_root: Path,
    batch: ExecutionBatch,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
) -> tuple[Checkpoint | None, str]:
    npz_path, json_path = checkpoint_paths(output_root, batch)
    if not npz_path.exists() and not json_path.exists():
        return None, "no checkpoint"
    if not npz_path.is_file() or not json_path.is_file():
        return None, "incomplete checkpoint pair"
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable checkpoint JSON: {exc}"
    for key, value in _checkpoint_identity(
        batch=batch,
        config=config,
        hashes=hashes,
        selected_batch_size=selected_batch_size,
    ).items():
        if metadata.get(key) != value:
            return None, f"checkpoint identity mismatch: {key}"
    if metadata.get("checkpoint_filename") != npz_path.name:
        return None, "checkpoint filename mismatch"
    try:
        if int(metadata.get("checkpoint_bytes", -1)) != int(npz_path.stat().st_size):
            return None, "checkpoint byte-count mismatch"
        if metadata.get("checkpoint_sha256") != sha256_file(npz_path):
            return None, "checkpoint checksum mismatch"
        with np.load(npz_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        return None, f"checkpoint readback failed: {exc}"
    required = {
        "completed_cycle",
        "elapsed_seconds",
        "frame",
        "ranks",
        "sample_ids",
        "seen_cycles",
        "global_charge",
        "gram_residual",
    }
    missing = sorted(required - set(payload))
    if missing:
        return None, f"checkpoint NPZ missing fields: {missing}"
    completed = int(payload["completed_cycle"].item())
    if completed != int(metadata.get("completed_cycle", -1)):
        return None, "checkpoint completed-cycle mismatch"
    if not (0 < completed <= CYCLES) or (completed % CHECKPOINT_CYCLES and completed != CYCLES):
        return None, "checkpoint is not on a valid cycle boundary"
    frame = np.asarray(payload["frame"])
    ranks = np.asarray(payload["ranks"], dtype=np.int64)
    if frame.dtype != np.complex128 or frame.shape[:2] != (
        batch.sample_count,
        2 * batch.case.nx * NY,
    ):
        return None, "checkpoint frame dtype/shape mismatch"
    if ranks.shape != (batch.sample_count,) or np.any(ranks < 0) or np.any(ranks > frame.shape[2]):
        return None, "checkpoint rank mismatch"
    if not np.array_equal(payload["sample_ids"], np.asarray(batch.sample_ids)):
        return None, "checkpoint sample IDs mismatch"
    seen = np.asarray(payload["seen_cycles"], dtype=np.bool_)
    charge = np.asarray(payload["global_charge"], dtype=np.int64)
    gram_residual = np.asarray(payload["gram_residual"], dtype=np.float64)
    if gram_residual.shape != (batch.sample_count,) or not np.isfinite(gram_residual).all():
        return None, "checkpoint Gram diagnostics mismatch"
    expected_seen = np.arange(CYCLES + 1) <= completed
    if seen.shape != expected_seen.shape or not np.array_equal(seen, expected_seen):
        return None, "checkpoint observer cycle mismatch"
    rng = {
        key.removeprefix("rng__"): value
        for key, value in payload.items()
        if key.startswith("rng__")
    }
    if not rng:
        return None, "checkpoint RNG state is missing"
    required_rng = {
        "numpy_algorithm",
        "numpy_keys",
        "numpy_position",
        "numpy_has_gauss",
        "numpy_cached_gaussian",
        "torch_cpu",
        "torch_cuda_count",
    }
    if not required_rng <= set(rng):
        return None, "checkpoint RNG state is incomplete"
    cuda_count = int(np.asarray(rng["torch_cuda_count"]).item())
    if any(f"torch_cuda_{index}" not in rng for index in range(cuda_count)):
        return None, "checkpoint CUDA RNG state is incomplete"
    return (
        Checkpoint(
            completed_cycle=completed,
            elapsed_seconds=float(payload["elapsed_seconds"].item()),
            frame=frame,
            ranks=ranks,
            seen_cycles=seen,
            global_charge=charge,
            gram_residual=gram_residual,
            rng=rng,
        ),
        "verified",
    )


def remove_checkpoint(output_root: Path, batch: ExecutionBatch) -> None:
    npz_path, json_path = checkpoint_paths(output_root, batch)
    for path in (json_path, npz_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        npz_path.parent.rmdir()
    except OSError:
        pass


def _run_segment(
    *,
    model: classA_U1FGTN_gpu,
    batch: ExecutionBatch,
    observer: ChargeObserver,
    segment_start: int,
    frame: np.ndarray | None,
    ranks: np.ndarray | None,
    cycle_bar: tqdm,
) -> tuple[dict[str, Any], float]:
    segment_cycles = min(CHECKPOINT_CYCLES, CYCLES - segment_start)
    if segment_cycles <= 0:
        raise ValueError("segment starts at or beyond the final cycle")

    def observe(*, cycle: int, **payload: Any) -> None:
        local_cycle = int(cycle)
        if segment_start and local_cycle == 0:
            return
        observer(cycle=segment_start + local_cycle, **payload)
        if local_cycle > 0:
            cycle_bar.update(1)

    started = time.monotonic()
    result = _run_engine(
        model=model,
        case=batch.case,
        samples=batch.sample_count,
        cycles=segment_cycles,
        frame=frame,
        ranks=ranks,
        observer=observe,
        progress=False,
    )
    return result["native_final"], time.monotonic() - started


def _frame_diagnostics(
    frame: np.ndarray, ranks: np.ndarray, *, nx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = int(frame.shape[0])
    density_x = np.zeros((samples, nx), dtype=np.float64)
    for sample, rank in enumerate(ranks.tolist()):
        active = frame[sample, :, : int(rank)]
        density = np.sum(np.abs(active) ** 2, axis=1, dtype=np.float64)
        density_x[sample] = density.reshape(NY, nx, 2).sum(axis=(0, 2))
    left = density_x[:, : nx // 2].sum(axis=1)
    right = density_x[:, nx // 2 :].sum(axis=1)
    return density_x, left, right


def _result_payload(
    *,
    shard: ResultShard,
    frame: np.ndarray,
    ranks: np.ndarray,
    gram_residual: np.ndarray,
    observer: ChargeObserver,
    elapsed_seconds: float,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
    benchmark: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    case = shard.case
    local = shard.local_slice
    shard_ranks = np.asarray(ranks[local], dtype=np.int64)
    capacity = int(np.max(shard_ranks, initial=0))
    shard_frame = np.ascontiguousarray(frame[local, :, :capacity], dtype=np.complex128)
    density_x, n_left, n_right = _frame_diagnostics(
        shard_frame, shard_ranks, nx=case.nx
    )
    gram = np.asarray(gram_residual[local], dtype=np.float64)
    if not np.isfinite(shard_frame).all() or not np.isfinite(gram).all():
        raise FloatingPointError("endpoint frame or diagnostics are nonfinite")
    if np.max(gram, initial=0.0) > 1.0e-10:
        raise FloatingPointError(
            f"endpoint frame Gram residual exceeds tolerance: {np.max(gram):.3e}"
        )
    charge = observer.global_charge[local].copy()
    initial_charge = charge[:, 0]
    final_charge = charge[:, -1]
    metadata = {
        **_shard_identity(
            shard=shard,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
        ),
        "result_schema": RESULT_SCHEMA,
        "stage": "endpoint_shard",
        "cell": case.cell,
        "shard_index": shard.shard_index,
        "nshell": SHELLS[case.protocol],
        "wall_flags": WALL_FLAGS[case.wall],
        "dtype": "complex128",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
    }
    return _coerce_npz(
        {
            "schema": np.asarray(RESULT_SCHEMA),
            "bundle": np.asarray(BUNDLE),
            "sampling_revision": np.asarray(REVISION),
            "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
            "execution_backend": np.asarray(BACKEND),
            "stage": np.asarray("endpoint_shard"),
            "collection": np.asarray(case.collection),
            "cell": np.asarray(case.cell),
            "protocol": np.asarray(case.protocol),
            "wall": np.asarray(case.wall),
            "Nx": np.asarray(case.nx, dtype=np.int64),
            "Ny": np.asarray(NY, dtype=np.int64),
            "wall_locations": np.asarray(case.wall_locations, dtype=np.int64),
            "nshell": np.asarray(-1 if SHELLS[case.protocol] is None else 1, dtype=np.int64),
            "nshell_label": np.asarray(case.protocol),
            "alpha_1": np.asarray(1.0, dtype=np.float64),
            "alpha_2": np.asarray(30.0, dtype=np.float64),
            "cycles": np.arange(CYCLES + 1, dtype=np.int64),
            "cycles_total": np.asarray(CYCLES, dtype=np.int64),
            "shard_index": np.asarray(shard.shard_index, dtype=np.int64),
            "sample_ids": np.asarray(shard.sample_ids, dtype=np.int64),
            "frames": shard_frame,
            "ranks": shard_ranks,
            "frame_capacity": np.asarray(capacity, dtype=np.int64),
            "global_charge": charge,
            "half_filling_offset": charge - case.nx * NY,
            "initial_total_charge": initial_charge,
            "final_total_charge": final_charge,
            "net_injected_charge": final_charge - initial_charge,
            "minimum_rank": np.min(charge, axis=1),
            "maximum_rank": np.max(charge, axis=1),
            "density_x": density_x,
            "N_left": n_left,
            "N_right": n_right,
            "charge_partition_residual": n_left + n_right - final_charge,
            "gram_residual": gram,
            "execution_batch_id": np.asarray(shard.batch.task_id),
            "execution_batch_seed": np.asarray(shard.batch.seed, dtype=np.int64),
            "selected_execution_batch_size": np.asarray(selected_batch_size, dtype=np.int64),
            "base_config_sha256": np.asarray(config_sha256(config)),
            "resolved_execution_sha256": np.asarray(
                resolved_execution_sha256(config, selected_batch_size)
            ),
            "source_hashes_json": np.asarray(canonical_json(hashes)),
            "metadata_json": np.asarray(canonical_json(metadata)),
            "elapsed_execution_batch_seconds": np.asarray(
                elapsed_seconds, dtype=np.float64
            ),
            "benchmark_projected_missing_1000_seconds": np.asarray(
                float(benchmark["projected_missing_1000_seconds"]), dtype=np.float64
            ),
        }
    )


def save_result_shard(
    *,
    output_root: Path,
    scratch_root: Path,
    shard: ResultShard,
    payload: Mapping[str, Any],
    elapsed_seconds: float,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
) -> None:
    local_dir = scratch_root / shard.batch.task_id / "results"
    local_result = local_dir / f"{shard.task_id}.npz"
    local_completion = local_dir / f"{shard.task_id}.completion.json"
    _write_npz(local_result, payload)
    result_path, completion_path = result_paths(output_root, shard)
    published = publish_file(local_result, result_path)
    completion = {
        **_shard_identity(
            shard=shard,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
        ),
        "result_filename": result_path.name,
        "result_bytes": published["bytes"],
        "result_sha256": published["sha256"],
        "result": {
            "name": result_path.name,
            "bytes": published["bytes"],
            "sha256": published["sha256"],
        },
        "config_hash": resolved_execution_sha256(config, selected_batch_size),
        "elapsed_execution_batch_seconds": float(elapsed_seconds),
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    valid, reason = verified_complete(
        output_root=output_root,
        shard=shard,
        config=config,
        hashes=hashes,
        selected_batch_size=selected_batch_size,
    )
    if not valid:
        raise OSError(f"published result failed verification: {reason}")


def _space_requirement_bytes(batch: ExecutionBatch) -> int:
    dimension = 2 * batch.case.nx * NY
    approximate_capacity = batch.case.nx * NY + CYCLES
    return int(batch.sample_count * dimension * approximate_capacity * 16)


def _check_space(path: Path, *, required_bytes: int, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path).free)
    if free < int(required_bytes):
        raise RuntimeError(
            f"insufficient {label} space: free={free / 1024**3:.2f} GiB, "
            f"required={required_bytes / 1024**3:.2f} GiB"
        )


def execute_batch(
    *,
    model: classA_U1FGTN_gpu,
    output_root: Path,
    scratch_root: Path,
    batch: ExecutionBatch,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
    benchmark: Mapping[str, Any],
    shard_bar: tqdm,
) -> int:
    pending: list[ResultShard] = []
    for shard in result_shards(batch):
        valid, _ = verified_complete(
            output_root=output_root,
            shard=shard,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
        )
        if not valid:
            pending.append(shard)
    if not pending:
        remove_checkpoint(output_root, batch)
        return 0

    required = _space_requirement_bytes(batch)
    _check_space(scratch_root, required_bytes=2 * required + 1024**3, label="local scratch")
    _check_space(output_root, required_bytes=2 * required + 1024**3, label="Drive")
    local_dir = scratch_root / batch.task_id
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True)

    checkpoint, reason = load_checkpoint(
        output_root=output_root,
        batch=batch,
        config=config,
        hashes=hashes,
        selected_batch_size=selected_batch_size,
    )
    observer = ChargeObserver(batch.sample_count)
    if checkpoint is None:
        print(f"[checkpoint] {batch.task_id}: {reason}; starting at cycle 0", flush=True)
        completed_cycle = 0
        elapsed_seconds = 0.0
        frame = None
        ranks = None
        gram_residual = None
        continuation_rng = None
        _seed_rng(batch.seed)
    else:
        completed_cycle = checkpoint.completed_cycle
        elapsed_seconds = checkpoint.elapsed_seconds
        frame = checkpoint.frame
        ranks = checkpoint.ranks
        gram_residual = checkpoint.gram_residual
        continuation_rng = checkpoint.rng
        observer.restore(checkpoint.seen_cycles, checkpoint.global_charge)
        print(
            f"[checkpoint] {batch.task_id}: verified cycle {completed_cycle}/{CYCLES}",
            flush=True,
        )

    cycle_bar = tqdm(
        total=CYCLES,
        initial=completed_cycle,
        desc=f"N{batch.case.nx} {batch.case.protocol} {batch.case.wall}",
        unit="cycle",
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
    )
    cycle_bar.set_postfix(durable_cycle=completed_cycle, refresh=True)
    try:
        while completed_cycle < CYCLES:
            if completed_cycle:
                if continuation_rng is None:
                    raise RuntimeError("continuation RNG state is missing")
                _restore_rng(continuation_rng)
            native, elapsed = _run_segment(
                model=model,
                batch=batch,
                observer=observer,
                segment_start=completed_cycle,
                frame=frame,
                ranks=ranks,
                cycle_bar=cycle_bar,
            )
            completed_cycle += min(CHECKPOINT_CYCLES, CYCLES - completed_cycle)
            elapsed_seconds += elapsed
            frame = np.asarray(native["frame"])
            ranks = np.asarray(native["ranks"], dtype=np.int64)
            gram_residual = np.asarray(native["gram_residual"], dtype=np.float64)
            continuation_rng = save_checkpoint(
                output_root=output_root,
                scratch_root=scratch_root,
                batch=batch,
                completed_cycle=completed_cycle,
                elapsed_seconds=elapsed_seconds,
                native=native,
                observer=observer,
                config=config,
                hashes=hashes,
                selected_batch_size=selected_batch_size,
            )
            cycle_bar.set_postfix(durable_cycle=completed_cycle, refresh=True)
    finally:
        cycle_bar.close()
    if frame is None or ranks is None or gram_residual is None or completed_cycle != CYCLES:
        raise RuntimeError("execution did not reach the final endpoint")
    observer.validate(require_complete=True)

    published_count = 0
    for shard in result_shards(batch):
        valid, _ = verified_complete(
            output_root=output_root,
            shard=shard,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
        )
        if valid:
            continue
        payload = _result_payload(
            shard=shard,
            frame=frame,
            ranks=ranks,
            gram_residual=gram_residual,
            observer=observer,
            elapsed_seconds=elapsed_seconds,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
            benchmark=benchmark,
        )
        save_result_shard(
            output_root=output_root,
            scratch_root=scratch_root,
            shard=shard,
            payload=payload,
            elapsed_seconds=elapsed_seconds,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
        )
        published_count += 1
        shard_bar.update(1)
        shard_bar.set_postfix(
            completed=shard_bar.n,
            pending=shard_bar.total - shard_bar.n,
            failed=0,
            refresh=True,
        )
    if not all(
        verified_complete(
            output_root=output_root,
            shard=shard,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
        )[0]
        for shard in result_shards(batch)
    ):
        raise OSError("one or more five-sample results failed final verification")
    remove_checkpoint(output_root, batch)
    shutil.rmtree(local_dir)
    return published_count


def _inventory(
    *,
    output_root: Path,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    selected_batch_size: int,
) -> tuple[int, list[tuple[ResultShard, str]]]:
    complete = 0
    pending: list[tuple[ResultShard, str]] = []
    for shard in all_result_shards(config, selected_batch_size):
        valid, reason = verified_complete(
            output_root=output_root,
            shard=shard,
            config=config,
            hashes=hashes,
            selected_batch_size=selected_batch_size,
        )
        if valid:
            complete += 1
        else:
            pending.append((shard, reason))
    return complete, pending


def run_campaign(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    scratch_root: Path,
    report_only: bool,
    benchmark_only: bool,
    max_new_execution_batches: int | None,
) -> dict[str, Any]:
    config = validate_config(config)
    hashes = source_hashes()
    gpu = validate_a100()
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[contract] revision={REVISION}; primary=1000; bridge=150; "
        f"durable shards=230; cycles={CYCLES}; checkpoint every {CHECKPOINT_CYCLES}",
        flush=True,
    )
    print(
        f"[contract] Nx=20,24,28,32; Ny={NY}; nshell=1,dense; walls=soft,hard; "
        "pure half filling; raster_y; perfect correction; no postselection",
        flush=True,
    )
    print(
        f"[runtime] device={gpu['name']}; memory={gpu['total_bytes'] / 1024**3:.2f} GiB; "
        f"dtype=complex128; backend={BACKEND}",
        flush=True,
    )
    print(f"[paths] bundle={BUNDLE_ROOT}; output={output_root}; scratch={scratch_root}", flush=True)
    print(f"[sources] {json.dumps(hashes, sort_keys=True)}", flush=True)

    benchmark, reason = load_benchmark(
        output_root=output_root,
        config=config,
        hashes=hashes,
        gpu_name=str(gpu["name"]),
    )
    if benchmark is None:
        if report_only:
            raise RuntimeError(f"report-only requires an accepted benchmark: {reason}")
        print(f"[benchmark] {reason}; running the locked gate", flush=True)
        benchmark = run_benchmark(
            output_root=output_root,
            scratch_root=scratch_root,
            config=config,
            hashes=hashes,
            gpu=gpu,
        )
    else:
        print(
            f"[benchmark] verified; selected batch={benchmark['selected_execution_batch_size']}; "
            f"forecast={benchmark['projected_missing_1000_hours']:.2f} h",
            flush=True,
        )
    selected = int(benchmark["selected_execution_batch_size"])
    complete, pending = _inventory(
        output_root=output_root,
        config=config,
        hashes=hashes,
        selected_batch_size=selected,
    )
    print(
        f"[resume] verified={complete}/230; pending={len(pending)}; failed=0; "
        f"trajectories durable={complete * SHARD_SIZE}/1150",
        flush=True,
    )
    if pending:
        by_reason: dict[str, int] = {}
        for _, pending_reason in pending:
            by_reason[pending_reason] = by_reason.get(pending_reason, 0) + 1
        print(f"[resume detail] {json.dumps(by_reason, sort_keys=True)}", flush=True)
    if report_only or benchmark_only:
        return {
            "verified_shards": complete,
            "pending_shards": len(pending),
            "selected_execution_batch_size": selected,
            "benchmark_only": bool(benchmark_only),
        }

    _check_space(
        output_root,
        required_bytes=MINIMUM_PRODUCTION_DRIVE_FREE_BYTES,
        label="Drive production",
    )
    print("[space] Drive production free-space gate passed (>=25 GiB)", flush=True)

    executions = expand_execution_batches(config, selected)
    launched = 0
    failures = 0
    shard_bar = tqdm(
        total=230,
        initial=complete,
        desc="endpoint shards",
        unit="shard",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    shard_bar.set_postfix(
        completed=complete,
        pending=230 - complete,
        failed=0,
        refresh=True,
    )
    model_cache: dict[tuple[str, int, str], classA_U1FGTN_gpu] = {}
    try:
        for batch in executions:
            statuses = [
                verified_complete(
                    output_root=output_root,
                    shard=shard,
                    config=config,
                    hashes=hashes,
                    selected_batch_size=selected,
                )[0]
                for shard in result_shards(batch)
            ]
            if all(statuses):
                continue
            if max_new_execution_batches is not None and launched >= max_new_execution_batches:
                break
            key = (batch.case.protocol, batch.case.nx, batch.case.wall)
            if key not in model_cache:
                model_cache[key] = build_model(config, batch.case)
            print(
                f"[execution {launched + 1}] {batch.task_id}; trajectories={batch.sample_count}; "
                f"pending_shards={sum(not value for value in statuses)}; seed={batch.seed}",
                flush=True,
            )
            try:
                execute_batch(
                    model=model_cache[key],
                    output_root=output_root,
                    scratch_root=scratch_root,
                    batch=batch,
                    config=config,
                    hashes=hashes,
                    selected_batch_size=selected,
                    benchmark=benchmark,
                    shard_bar=shard_bar,
                )
            except Exception:
                failures += 1
                shard_bar.set_postfix(
                    completed=shard_bar.n,
                    pending=shard_bar.total - shard_bar.n,
                    failed=failures,
                    refresh=True,
                )
                raise
            launched += 1
    finally:
        shard_bar.close()
        model_cache.clear()
        _clear_cuda()

    final_complete, final_pending = _inventory(
        output_root=output_root,
        config=config,
        hashes=hashes,
        selected_batch_size=selected,
    )
    print(
        f"[completion] verified={final_complete}/230; pending={len(final_pending)}; "
        f"failed={failures}; new_execution_batches={launched}",
        flush=True,
    )
    return {
        "verified_shards": final_complete,
        "pending_shards": len(final_pending),
        "failed": failures,
        "new_execution_batches": launched,
        "selected_execution_batch_size": selected,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--max-new-execution-batches", type=int)
    args = parser.parse_args(argv)
    if args.max_new_execution_batches is not None and args.max_new_execution_batches < 0:
        parser.error("--max-new-execution-batches must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_campaign(
        config=config,
        output_root=args.output_root.resolve(),
        scratch_root=args.scratch_root.resolve(),
        report_only=bool(args.report_only),
        benchmark_only=bool(args.benchmark_only),
        max_new_execution_batches=args.max_new_execution_batches,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
