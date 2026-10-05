#!/usr/bin/env python3
"""Canonical-CPU fallback for the wall-pump endpoint width campaign."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


CONFIG_SCHEMA = "wall_pump_endpoint_cpu_fallback_campaign_v1"
RESULT_SCHEMA = "wall_pump_width_endpoint_shard_v1"
COMPLETION_SCHEMA = "wall_pump_width_endpoint_shard_completion_v1"
CHECKPOINT_SCHEMA = "wall_pump_width_endpoint_cpu_checkpoint_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.wall_pump_endpoint_cpu_fallback_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "imported_endpoints/wall_pump_width_endpoints_s100_v1_cpu_fallback"
SOURCE_PATHS = {
    "cpu_fallback_runner": Path(__file__).resolve(),
    "cpu_fallback_config": DEFAULT_CONFIG,
    "canonical_cpu_engine": REPO_ROOT / "src/fgtn/classA_U1FGTN.py",
    "canonical_occupied_frame": REPO_ROOT / "src/fgtn/occupied_frame.py",
    "gpu_campaign_contract": (
        REPO_ROOT
        / "00_WORKSPACE/CURRENT/final_production_new_designs/11_wall_pump_width_endpoints/campaign_config.json"
    ),
}


@dataclass(frozen=True)
class SampleTask:
    collection: str
    protocol: str
    nx: int
    wall: str
    sample_id: int
    samples_per_wall: int
    seed: int
    lane: int

    @property
    def cell(self) -> str:
        return f"{self.protocol}_N{self.nx}x24"

    @property
    def shard_index(self) -> int:
        return self.sample_id // 5

    @property
    def task_id(self) -> str:
        return (
            f"cpu_endpoint_{self.collection}_{self.protocol}_N{self.nx}x24_"
            f"{self.wall}_sample_{self.sample_id:03d}"
        )

    @property
    def wall_locations(self) -> tuple[int, int]:
        return self.nx // 4, 3 * self.nx // 4


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected schema {CONFIG_SCHEMA!r}")
    if (
        config.get("sampling_revision") != "wall_pump_width_endpoints_s100_v1"
        or config.get("execution_revision") != "wall_pump_width_endpoints_cpu_fallback_v1"
        or int(config.get("root_seed", -1)) != 2026090701
        or int(config.get("Ny", -1)) != 24
        or int(config.get("cycles", -1)) != 48
        or int(config.get("checkpoint_every_cycles", -1)) != 5
        or int(config.get("result_shard_size", -1)) != 5
    ):
        raise ValueError("CPU fallback sampling/checkpoint identity changed")
    expected_primary = {
        ("nsh1", 28, 100), ("nsh1", 32, 100), ("dense", 24, 100),
        ("dense", 28, 100), ("dense", 32, 100),
    }
    primary = {
        (str(row["protocol"]), int(row["Nx"]), int(row["samples_per_wall"]))
        for row in config.get("primary_cells", [])
    }
    if primary != expected_primary or "bridge_cells" in config:
        raise ValueError("CPU fallback cell matrix changed")
    if config.get("walls") != {
        "soft": {"dw_truncation": False, "meas_slab_only": False},
        "hard": {"dw_truncation": True, "meas_slab_only": True},
    }:
        raise ValueError("wall definitions changed")
    if config.get("shells") != {"nsh1": 1, "dense": None}:
        raise ValueError("shell definitions changed")
    expected_dynamics = {
        "DW": True, "wall_locations_rule": ["Nx/4", "3Nx/4"],
        "alpha_1": 1.0, "alpha_2": 30.0, "filling_frac": 0.5,
        "trial_orbitals": "X", "init_mode": "default", "sequence": "raster_y",
        "perfect_correction": True, "postselect": False,
        "postselect_probability": 0.0, "n_a": 0.5,
        "state_representation": "physical_frame",
        "physical_covariance_update": "rank1",
        "frame_reorthonormalize_interval": 1, "dtype": "complex128",
    }
    if config.get("dynamics") != expected_dynamics:
        raise ValueError("CPU fallback dynamics changed")
    if config.get("execution") != {
        "backend": "canonical_cpu",
        "lanes": [
            {"lane": 0, "physical_cpu_set": "0-27", "workers": 28},
            {"lane": 1, "physical_cpu_set": "28-55", "workers": 28},
        ],
        "blas_threads_per_worker": 1,
    }:
        raise ValueError("CPU fallback execution topology changed")


def config_hash(config: Mapping[str, Any]) -> str:
    validate_config(config)
    return hashlib.sha256(canonical_json(dict(config)).encode("utf-8")).hexdigest()


def source_hashes() -> dict[str, str]:
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing CPU fallback source(s): {missing}")
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def _sample_seed(config: Mapping[str, Any], collection: str, protocol: str, nx: int, wall: str, sample_id: int) -> int:
    label = (
        f"{config['root_seed']}|{config['sampling_revision']}|backend=canonical_cpu|"
        f"{collection}|{protocol}|N{nx}x24|{wall}|sample={sample_id}"
    )
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def tasks(config: Mapping[str, Any]) -> list[SampleTask]:
    validate_config(config)
    rows: list[SampleTask] = []
    shard_ordinal = 0
    for collection, key in (("endpoints", "primary_cells"),):
        for cell in config[key]:
            protocol, nx, count = str(cell["protocol"]), int(cell["Nx"]), int(cell["samples_per_wall"])
            for wall in ("soft", "hard"):
                for shard_start in range(0, count, 5):
                    lane = shard_ordinal % 2
                    for sample_id in range(shard_start, shard_start + 5):
                        rows.append(
                            SampleTask(
                                collection, protocol, nx, wall, sample_id, count,
                                _sample_seed(config, collection, protocol, nx, wall, sample_id),
                                lane,
                            )
                        )
                    shard_ordinal += 1
    if len(rows) != 1000 or len({row.task_id for row in rows}) != 1000:
        raise RuntimeError("CPU fallback must contain exactly 1,000 unique trajectories")
    if len({row.seed for row in rows}) != len(rows):
        raise RuntimeError("CPU fallback trajectory seeds are not unique")
    lane_counts = [sum(row.lane == lane for row in rows) for lane in (0, 1)]
    if max(lane_counts) - min(lane_counts) > 5:
        raise RuntimeError(f"CPU fallback lanes are imbalanced: {lane_counts}")
    return rows


def checkpoint_paths(output_root: Path, task: SampleTask) -> tuple[Path, Path]:
    root = (
        Path(output_root) / "checkpoints" / task.collection / task.protocol
        / f"N{task.nx}x24" / task.wall
    )
    data = root / f"sample_{task.sample_id:03d}.checkpoint.npz"
    return data, data.with_suffix(".json")


def shard_paths(output_root: Path, task: SampleTask) -> tuple[Path, Path]:
    root = (
        Path(output_root) / task.collection / task.protocol
        / f"N{task.nx}x24" / task.wall
    )
    data = root / f"shard_{task.shard_index:02d}.npz"
    return data, data.with_suffix(".completion.json")


def failure_path(output_root: Path, task: SampleTask) -> Path:
    return Path(output_root) / "failures" / f"{task.task_id}.json"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pack_tree(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        key = f"state_{len(arrays):04d}"
        arrays[key] = np.asarray(value)
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _pack_tree(item, arrays) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_pack_tree(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_pack_tree(item, arrays) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"checkpoint contains unsupported value {type(value).__name__}")


def _unpack_tree(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, dict) and set(value) == {"__array__"}:
        return np.array(arrays[str(value["__array__"])], copy=True)
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_unpack_tree(item, arrays) for item in value["__tuple__"])
    if isinstance(value, dict):
        return {key: _unpack_tree(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_tree(item, arrays) for item in value]
    return value


def _checkpoint_identity(task: SampleTask, config: Mapping[str, Any], hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": task.task_id,
        "collection": task.collection,
        "cell": task.cell,
        "protocol": task.protocol,
        "Nx": task.nx,
        "Ny": 24,
        "wall": task.wall,
        "sample_id": task.sample_id,
        "seed": task.seed,
        "cycles_total": 48,
        "config_hash": config_hash(config),
        "source_hashes": dict(hashes),
        "execution_backend": "canonical_cpu",
    }


def save_checkpoint(
    output_root: Path,
    task: SampleTask,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    engine_state: Mapping[str, Any],
    charge_history: np.ndarray,
    elapsed_seconds: float,
) -> None:
    cycle = int(engine_state["completed_cycles"])
    arrays: dict[str, np.ndarray] = {}
    tree = _pack_tree(dict(engine_state), arrays)
    arrays["schema"] = np.asarray(CHECKPOINT_SCHEMA)
    arrays["state_tree_json"] = np.asarray(canonical_json(tree))
    arrays["global_charge"] = np.asarray(charge_history, dtype=np.int64)
    arrays["completed_cycle"] = np.asarray(cycle, dtype=np.int64)
    data_path, receipt_path = checkpoint_paths(output_root, task)
    _atomic_npz(data_path, arrays)
    _atomic_json(
        receipt_path,
        {
            **_checkpoint_identity(task, config, hashes),
            "completed_cycle": cycle,
            "elapsed_seconds": float(elapsed_seconds),
            "checkpoint": {
                "name": data_path.name,
                "bytes": data_path.stat().st_size,
                "sha256": sha256_path(data_path),
            },
        },
    )


def load_checkpoint(
    output_root: Path,
    task: SampleTask,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[dict[str, Any] | None, np.ndarray | None, float, str]:
    data_path, receipt_path = checkpoint_paths(output_root, task)
    if not data_path.exists() and not receipt_path.exists():
        return None, None, 0.0, "missing checkpoint"
    if not data_path.is_file() or not receipt_path.is_file():
        return None, None, 0.0, "incomplete checkpoint pair"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for key, value in _checkpoint_identity(task, config, hashes).items():
            if receipt.get(key) != value:
                raise ValueError(f"checkpoint identity mismatch: {key}")
        record = receipt["checkpoint"]
        if (
            record.get("name") != data_path.name
            or int(record.get("bytes", -1)) != data_path.stat().st_size
            or record.get("sha256") != sha256_path(data_path)
        ):
            raise ValueError("checkpoint file receipt mismatch")
        with np.load(data_path, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != CHECKPOINT_SCHEMA:
                raise ValueError("checkpoint NPZ schema mismatch")
            cycle = int(np.asarray(saved["completed_cycle"]).item())
            charge = np.array(saved["global_charge"], dtype=np.int64, copy=True)
            tree = json.loads(str(np.asarray(saved["state_tree_json"]).item()))
            state = _unpack_tree(tree, saved)
        if cycle != int(state["completed_cycles"]) or not 0 <= cycle <= 48:
            raise ValueError("checkpoint cycle mismatch")
        if charge.shape != (49,) or np.any(charge[: cycle + 1] < 0):
            raise ValueError("checkpoint charge prefix is invalid")
        if np.any(charge[cycle + 1 :] != -1):
            raise ValueError("checkpoint charge suffix is populated")
        return state, charge, float(receipt.get("elapsed_seconds", 0.0)), "verified"
    except Exception as exc:
        return None, None, 0.0, f"{type(exc).__name__}: {exc}"


class ChargeObserver:
    def __init__(self, prefix: np.ndarray | None = None) -> None:
        self.values = np.full(49, -1, dtype=np.int64)
        if prefix is not None:
            self.values[:] = np.asarray(prefix, dtype=np.int64)

    def __call__(self, *, cycle: int, state: Any, **_: Any) -> None:
        cycle = int(cycle)
        if self.values[cycle] >= 0:
            raise RuntimeError(f"duplicate CPU endpoint observation at cycle {cycle}")
        self.values[cycle] = int(state.rank)


def _model(task: SampleTask, config: Mapping[str, Any]) -> classA_U1FGTN:
    flags = config["walls"][task.wall]
    return classA_U1FGTN(
        Nx=task.nx,
        Ny=24,
        DW=True,
        nshell=config["shells"][task.protocol],
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=bool(flags["dw_truncation"]),
        twist_y=0.0,
        dw_interval=task.wall_locations,
    )


def _run_one(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, hashes = payload
    output_root = Path(output_text)
    started = time.monotonic()
    checkpoint, charge, elapsed_previous, reason = load_checkpoint(
        output_root, task, config, hashes
    )
    observer = ChargeObserver(charge)

    def checkpoint_observer(*, cycle: int, state: Mapping[str, Any]) -> None:
        if int(cycle) % int(config["checkpoint_every_cycles"]) == 0 or int(cycle) == 48:
            save_checkpoint(
                output_root, task, config, hashes, state, observer.values,
                elapsed_previous + time.monotonic() - started,
            )

    try:
        with threadpool_limits(limits=1):
            model = _model(task, config)
            result = model.run_markov_circuit(
                G_history=False,
                progress=False,
                cycles=48,
                postselect=False,
                postselect_probability=0.0,
                perfect_correction=True,
                samples=1,
                parallelize_samples=False,
                init_mode="default",
                save=False,
                n_a=0.5,
                sequence="raster_y",
                meas_slab_only=bool(config["walls"][task.wall]["meas_slab_only"]),
                random_seed=task.seed,
                physical_covariance_update="rank1",
                state_representation="physical_frame",
                native_cycle_observer=observer,
                return_native_state=True,
                require_no_covariance_materialization=True,
                frame_reorthonormalize_interval=1,
                checkpoint_state=checkpoint,
                checkpoint_observer=checkpoint_observer,
            )
        native = result["native_final"]
        if observer.values.shape != (49,) or np.any(observer.values < 0):
            raise RuntimeError("CPU endpoint charge history is incomplete")
        frame = np.asarray(native["frame"], dtype=np.complex128)
        rank = int(native["rank"])
        if frame.shape != (2 * task.nx * 24, rank) or not np.all(np.isfinite(frame)):
            raise RuntimeError("CPU endpoint frame is invalid")
        if float(native["gram_residual"]) > 1e-10:
            raise FloatingPointError(f"CPU endpoint Gram residual {native['gram_residual']:.3e}")
        state, _, _, cp_reason = load_checkpoint(output_root, task, config, hashes)
        if state is None or int(state["completed_cycles"]) != 48:
            raise RuntimeError(f"final native checkpoint did not verify: {cp_reason}")
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task.task_id, "checkpoint": reason}
    except BaseException as exc:
        path = failure_path(output_root, task)
        _atomic_json(
            path,
            {
                "task_id": task.task_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_unix": time.time(),
            },
        )
        return {"ok": False, "task_id": task.task_id, "error": f"{type(exc).__name__}: {exc}"}


def _shard_members(all_tasks: list[SampleTask], task: SampleTask) -> list[SampleTask]:
    return [
        row for row in all_tasks
        if (
            row.collection, row.protocol, row.nx, row.wall, row.shard_index
        ) == (
            task.collection, task.protocol, task.nx, task.wall, task.shard_index
        )
    ]


def _shard_identity(task: SampleTask, config: Mapping[str, Any], hashes: Mapping[str, str]) -> dict[str, Any]:
    sample_ids = list(range(task.shard_index * 5, task.shard_index * 5 + 5))
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "sampling_revision": config["sampling_revision"],
        "execution_revision": config["execution_revision"],
        "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
        "execution_backend": "canonical_cpu",
        "stage": "endpoint_shard",
        "task_id": (
            f"{task.collection}_{task.protocol}_N{task.nx}x24_"
            f"{task.wall}_shard-{task.shard_index:02d}"
        ),
        "collection": task.collection,
        "cell": task.cell,
        "protocol": task.protocol,
        "Nx": task.nx,
        "Ny": 24,
        "wall": task.wall,
        "wall_locations": list(task.wall_locations),
        "samples": 5,
        "shard_index": task.shard_index,
        "sample_ids": sample_ids,
        "cycles": 48,
        "config_hash": config_hash(config),
        "source_hashes": dict(hashes),
    }


def verify_shard(
    output_root: Path,
    task: SampleTask,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[bool, str]:
    result, completion_path = shard_paths(output_root, task)
    if not result.is_file() or not completion_path.is_file():
        return False, "missing result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        for key, value in _shard_identity(task, config, hashes).items():
            if completion.get(key) != value:
                return False, f"completion identity mismatch: {key}"
        record = completion["result"]
        if (
            record.get("name") != result.name
            or int(record.get("bytes", -1)) != result.stat().st_size
            or record.get("sha256") != sha256_path(result)
        ):
            return False, "result receipt mismatch"
        with np.load(result, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != RESULT_SCHEMA:
                return False, "result schema mismatch"
            if not np.array_equal(saved["sample_ids"], np.arange(task.shard_index * 5, task.shard_index * 5 + 5)):
                return False, "result sample IDs mismatch"
            frames = np.asarray(saved["frames"])
            ranks = np.asarray(saved["ranks"], dtype=np.int64)
            if frames.dtype != np.complex128 or frames.ndim != 3 or frames.shape[:2] != (5, 2 * task.nx * 24):
                return False, "result frame shape/dtype mismatch"
            if ranks.shape != (5,) or np.any(ranks <= 0) or np.any(ranks > frames.shape[2]):
                return False, "result ranks mismatch"
        return True, "verified"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def publish_shard_if_ready(
    output_root: Path,
    all_tasks: list[SampleTask],
    task: SampleTask,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[bool, str]:
    verified, reason = verify_shard(output_root, task, config, hashes)
    if verified:
        return True, reason
    members = sorted(_shard_members(all_tasks, task), key=lambda row: row.sample_id)
    states: list[dict[str, Any]] = []
    charges: list[np.ndarray] = []
    elapsed: list[float] = []
    for member in members:
        state, charge, seconds, cp_reason = load_checkpoint(output_root, member, config, hashes)
        if state is None or charge is None or int(state["completed_cycles"]) != 48:
            return False, f"waiting for {member.task_id}: {cp_reason}"
        states.append(state)
        charges.append(charge)
        elapsed.append(seconds)
    ranks = np.asarray([int(state["native_state"]["rank"]) for state in states], dtype=np.int64)
    capacity = int(np.max(ranks))
    frames = np.zeros((5, 2 * task.nx * 24, capacity), dtype=np.complex128)
    gram = np.empty(5)
    for index, (state, rank) in enumerate(zip(states, ranks.tolist())):
        native = state["native_state"]
        frame = np.asarray(native["frame"], dtype=np.complex128)
        frames[index, :, :rank] = frame
        gram[index] = float(native["gram_residual"])
    if np.max(gram) > 1e-10:
        raise FloatingPointError("completed CPU checkpoint exceeds Gram tolerance")
    density_x = np.empty((5, task.nx))
    for index, rank in enumerate(ranks.tolist()):
        density = np.sum(np.abs(frames[index, :, :rank]) ** 2, axis=1)
        density_x[index] = density.reshape(24, task.nx, 2).sum(axis=(0, 2))
    global_charge = np.stack(charges)
    n_left = density_x[:, : task.nx // 2].sum(axis=1)
    n_right = density_x[:, task.nx // 2 :].sum(axis=1)
    identity = _shard_identity(task, config, hashes)
    metadata = {
        **identity,
        "result_schema": RESULT_SCHEMA,
        "nshell": config["shells"][task.protocol],
        "wall_flags": config["walls"][task.wall],
        "dtype": "complex128",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "trajectory_seeds": [member.seed for member in members],
    }
    arrays = {
        "schema": np.asarray(RESULT_SCHEMA),
        "sampling_revision": np.asarray(config["sampling_revision"]),
        "execution_revision": np.asarray(config["execution_revision"]),
        "canonical_entry_point": np.asarray("classA_U1FGTN.run_markov_circuit"),
        "execution_backend": np.asarray("canonical_cpu"),
        "stage": np.asarray("endpoint_shard"),
        "collection": np.asarray(task.collection),
        "cell": np.asarray(task.cell),
        "protocol": np.asarray(task.protocol),
        "wall": np.asarray(task.wall),
        "Nx": np.asarray(task.nx, dtype=np.int64),
        "Ny": np.asarray(24, dtype=np.int64),
        "wall_locations": np.asarray(task.wall_locations, dtype=np.int64),
        "nshell": np.asarray(-1 if config["shells"][task.protocol] is None else 1, dtype=np.int64),
        "nshell_label": np.asarray(task.protocol),
        "alpha_1": np.asarray(1.0),
        "alpha_2": np.asarray(30.0),
        "cycles": np.arange(49, dtype=np.int64),
        "cycles_total": np.asarray(48, dtype=np.int64),
        "shard_index": np.asarray(task.shard_index, dtype=np.int64),
        "sample_ids": np.asarray([member.sample_id for member in members], dtype=np.int64),
        "trajectory_seeds": np.asarray([member.seed for member in members], dtype=np.int64),
        "frames": frames,
        "ranks": ranks,
        "frame_capacity": np.asarray(capacity, dtype=np.int64),
        "global_charge": global_charge,
        "half_filling_offset": global_charge - task.nx * 24,
        "initial_total_charge": global_charge[:, 0],
        "final_total_charge": global_charge[:, -1],
        "net_injected_charge": global_charge[:, -1] - global_charge[:, 0],
        "minimum_rank": np.min(global_charge, axis=1),
        "maximum_rank": np.max(global_charge, axis=1),
        "density_x": density_x,
        "N_left": n_left,
        "N_right": n_right,
        "charge_partition_residual": n_left + n_right - global_charge[:, -1],
        "gram_residual": gram,
        "config_hash": np.asarray(config_hash(config)),
        "source_hashes_json": np.asarray(canonical_json(hashes)),
        "metadata_json": np.asarray(canonical_json(metadata)),
        "elapsed_sample_seconds": np.asarray(elapsed),
    }
    result, completion_path = shard_paths(output_root, task)
    _atomic_npz(result, arrays)
    record = {"name": result.name, "bytes": result.stat().st_size, "sha256": sha256_path(result)}
    _atomic_json(
        completion_path,
        {
            **identity,
            "result_filename": result.name,
            "result_bytes": result.stat().st_size,
            "result_sha256": record["sha256"],
            "result": record,
            "completed_unix": time.time(),
        },
    )
    verified, reason = verify_shard(output_root, task, config, hashes)
    if not verified:
        raise RuntimeError(f"published CPU shard did not verify: {reason}")
    # A remotely useful shard now supersedes its five large rolling checkpoints.
    for member in members:
        data, receipt = checkpoint_paths(output_root, member)
        data.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
    return True, "published and verified"


def inventory(
    config: Mapping[str, Any],
    output_root: Path,
    hashes: Mapping[str, str],
    *,
    lane: int | None = None,
) -> dict[str, Any]:
    all_rows = tasks(config)
    representatives: dict[tuple[Any, ...], SampleTask] = {}
    for task in all_rows:
        if lane is not None and task.lane != lane:
            continue
        key = (task.collection, task.protocol, task.nx, task.wall, task.shard_index)
        representatives.setdefault(key, task)
    complete = 0
    for task in representatives.values():
        complete += int(verify_shard(output_root, task, config, hashes)[0])
    checkpoints: list[int] = []
    for task in all_rows:
        if lane is not None and task.lane != lane:
            continue
        state, _, _, _ = load_checkpoint(output_root, task, config, hashes)
        if state is not None:
            checkpoints.append(int(state["completed_cycles"]))
    return {
        "schema": "wall_pump_endpoint_cpu_fallback_inventory_v1",
        "lane": "all" if lane is None else lane,
        "expected_shards": len(representatives),
        "verified_shards": complete,
        "pending_shards": len(representatives) - complete,
        "verified_checkpoints": len(checkpoints),
        "checkpoint_cycle_histogram": {
            str(cycle): checkpoints.count(cycle) for cycle in sorted(set(checkpoints))
        },
        "output_root": str(Path(output_root).resolve()),
    }


def run(
    config: Mapping[str, Any],
    output_root: Path,
    hashes: Mapping[str, str],
    *,
    lane: int,
    workers: int,
    max_new_samples: int | None = None,
) -> dict[str, Any]:
    all_rows = tasks(config)
    selected: list[SampleTask] = []
    for task in all_rows:
        if task.lane != lane:
            continue
        if verify_shard(output_root, task, config, hashes)[0]:
            continue
        state, _, _, _ = load_checkpoint(output_root, task, config, hashes)
        if state is not None and int(state["completed_cycles"]) == 48:
            publish_shard_if_ready(output_root, all_rows, task, config, hashes)
            if verify_shard(output_root, task, config, hashes)[0]:
                continue
        selected.append(task)
    if max_new_samples is not None:
        selected = selected[: int(max_new_samples)]
    failures: list[dict[str, Any]] = []
    if selected:
        context = mp.get_context("spawn")
        payloads = [(task, dict(config), str(output_root), dict(hashes)) for task in selected]
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
            pending = {pool.submit(_run_one, payload): payload[0] for payload in payloads}
            with tqdm(total=len(selected), desc=f"CPU endpoint lane {lane}", unit="trajectory", dynamic_ncols=True, file=sys.stdout) as bar:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = pending.pop(future)
                        result = future.result()
                        if result["ok"]:
                            publish_shard_if_ready(output_root, all_rows, task, config, hashes)
                        else:
                            failures.append(result)
                        bar.update(1)
    status = inventory(config, output_root, hashes, lane=lane)
    status["new_sample_failures"] = failures
    print(json.dumps(status, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"CPU endpoint lane {lane} had {len(failures)} trajectory failures")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "run"), nargs="?", default="report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lane", choices=("0", "1", "all"), default="all")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--max-new-samples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    hashes = source_hashes()
    lane = None if args.lane == "all" else int(args.lane)
    if args.stage == "report":
        print(json.dumps(inventory(config, args.output_root, hashes, lane=lane), indent=2, sort_keys=True))
        return 0
    if lane is None:
        raise ValueError("run requires --lane 0 or --lane 1; the launch script owns two-lane orchestration")
    if args.workers < 1 or args.workers > 28:
        raise ValueError("each CPU fallback lane requires 1..28 workers")
    run(
        config,
        args.output_root,
        hashes,
        lane=lane,
        workers=args.workers,
        max_new_samples=args.max_new_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
