#!/usr/bin/env python3
"""Run the parallel, resumable boundary-only Born-record CPU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN


CONFIG_SCHEMA = "boundary_only_record_free_energy_cpu_config_v1"
RESULT_SCHEMA = "boundary_only_record_free_energy_cpu_result_v1"
COMPLETION_SCHEMA = "boundary_only_record_free_energy_cpu_completion_v1"
MANIFEST_SCHEMA = "boundary_only_record_free_energy_cpu_manifest_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
CHANNELS = ("Ap", "Am", "Bp", "Bm")

_SHARED_MODEL: classA_U1FGTN | None = None
_SHARED_GROUND_FRAME: np.ndarray | None = None
_SHARED_GROUND_DIAGNOSTICS: dict[str, float] | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    revision: str
    arm: str
    nx: int
    ny: int
    cycles: int
    sample_index: int
    seed: int
    selected_x: tuple[int, ...]
    config_hash: str
    source_hashes: dict[str, str]
    result_path: str
    completion_path: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("configuration schema mismatch")
    if config["geometry"] != {
        "Nx": 16,
        "Ny_values": [16, 20, 24, 28, 32],
        "domain_wall_interval": [4, 12],
        "twist_y": 1.0e-7,
    }:
        raise ValueError("geometry differs from the locked Nx=16 pilot")
    required_dynamics = {
        "cycles_rule": "4*Ny",
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "trial_orbitals": "X",
        "domain_wall_truncation": True,
        "initialization": "flattened_parent_lowest_half",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "measurement_slab_only": False,
        "state_representation": "physical_frame",
        "dtype": "complex128",
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
    }
    if config["dynamics"] != required_dynamics:
        raise ValueError("dynamics differs from the locked boundary-only pilot")
    if config["arms"] != {
        "exact_wall": {"samples_per_size": 100, "x_offsets_from_walls": [0]},
        "thin_strip": {
            "samples_per_size": 25,
            "x_offsets_from_walls": [-1, 0, 1],
        },
    }:
        raise ValueError("arm table differs from the locked pilot")


def scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": config["schema"],
        "revision": config["revision"],
        "root_seed": int(config["root_seed"]),
        "geometry": config["geometry"],
        "dynamics": config["dynamics"],
        "arms": config["arms"],
    }


def source_hashes(config_path: Path) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        config_path.resolve(),
        REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py",
        REPO_ROOT / "src" / "fgtn" / "occupied_frame.py",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def stable_seed(root_seed: int, revision: str, arm: str, ny: int, sample: int) -> int:
    raw = canonical_json([root_seed, revision, arm, 16, ny, sample]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def selected_x_for_arm(config: Mapping[str, Any], arm: str) -> tuple[int, ...]:
    walls = tuple(int(value) for value in config["geometry"]["domain_wall_interval"])
    offsets = tuple(int(value) for value in config["arms"][arm]["x_offsets_from_walls"])
    return tuple(sorted({wall + offset for wall in walls for offset in offsets}))


def build_tasks(
    config: Mapping[str, Any], output_root: Path, config_path: Path
) -> list[TaskSpec]:
    validate_config(config)
    config_hash = sha256_bytes(canonical_json(scientific_contract(config)).encode("utf-8"))
    hashes = source_hashes(config_path)
    revision = str(config["revision"])
    root_seed = int(config["root_seed"])
    tasks: list[TaskSpec] = []
    for ny in reversed(config["geometry"]["Ny_values"]):
        for arm in ("exact_wall", "thin_strip"):
            count = int(config["arms"][arm]["samples_per_size"])
            selected_x = selected_x_for_arm(config, arm)
            directory = output_root / "results" / arm / f"Ny{int(ny):03d}"
            for sample in range(count):
                task_id = f"{arm}_Nx16_Ny{int(ny):03d}_sample{sample:03d}"
                tasks.append(
                    TaskSpec(
                        task_id=task_id,
                        revision=revision,
                        arm=arm,
                        nx=16,
                        ny=int(ny),
                        cycles=4 * int(ny),
                        sample_index=sample,
                        seed=stable_seed(root_seed, revision, arm, int(ny), sample),
                        selected_x=selected_x,
                        config_hash=config_hash,
                        source_hashes=hashes,
                        result_path=str(directory / f"trajectory_{sample:03d}.npz"),
                        completion_path=str(
                            directory / f"trajectory_{sample:03d}.complete.json"
                        ),
                    )
                )
    expected = 5 * (100 + 25)
    if len(tasks) != expected or len({task.seed for task in tasks}) != expected:
        raise AssertionError("task table must contain 625 globally unique trajectories")
    return tasks


def completion_payload(spec: TaskSpec, result: Path, elapsed: float) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "revision": spec.revision,
        "task_id": spec.task_id,
        "arm": spec.arm,
        "Nx": spec.nx,
        "Ny": spec.ny,
        "cycles": spec.cycles,
        "sample_index": spec.sample_index,
        "seed": spec.seed,
        "selected_x": list(spec.selected_x),
        "config_hash": spec.config_hash,
        "source_hashes": spec.source_hashes,
        "result_filename": result.name,
        "result_bytes": result.stat().st_size,
        "result_sha256": sha256_file(result),
        "elapsed_seconds": float(elapsed),
        "completed_at": utc_now(),
    }


def verify_complete(spec: TaskSpec) -> tuple[bool, str]:
    result = Path(spec.result_path)
    completion = Path(spec.completion_path)
    if not result.is_file() or not completion.is_file():
        return False, "missing result/completion pair"
    try:
        receipt = json.loads(completion.read_text(encoding="utf-8"))
        expected = {
            "schema": COMPLETION_SCHEMA,
            "revision": spec.revision,
            "task_id": spec.task_id,
            "arm": spec.arm,
            "Nx": spec.nx,
            "Ny": spec.ny,
            "cycles": spec.cycles,
            "sample_index": spec.sample_index,
            "seed": spec.seed,
            "selected_x": list(spec.selected_x),
            "config_hash": spec.config_hash,
            "source_hashes": spec.source_hashes,
            "result_filename": result.name,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                return False, f"completion mismatch: {key}"
        if int(receipt["result_bytes"]) != result.stat().st_size:
            return False, "result byte-count mismatch"
        if receipt["result_sha256"] != sha256_file(result):
            return False, "result checksum mismatch"
        with np.load(result, allow_pickle=False) as data:
            if str(data["schema"].item()) != RESULT_SCHEMA:
                return False, "result schema mismatch"
            if str(data["task_id"].item()) != spec.task_id:
                return False, "result task mismatch"
            if not np.array_equal(
                np.asarray(data["cycles"], dtype=np.int64),
                np.arange(spec.cycles + 1, dtype=np.int64),
            ):
                return False, "cycle grid mismatch"
            if int(data["observed_site_count"].item()) != (
                spec.cycles * len(spec.selected_x) * spec.ny
            ):
                return False, "site-count mismatch"
        return True, "verified"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, f"verification error: {exc}"


def build_model_and_ground(
    *, nx: int, ny: int, walls: tuple[int, int], twist_y: float
) -> tuple[classA_U1FGTN, np.ndarray, dict[str, float]]:
    with threadpool_limits(limits=1):
        model = classA_U1FGTN(
            nx,
            ny,
            DW=True,
            nshell=1,
            filling_frac=0.5,
            alpha_1=1.0,
            alpha_2=30.0,
            trial_orbitals="X",
            dw_truncation=True,
            dw_interval=walls,
            twist_y=twist_y,
        )
        model.construct_OW_projectors(
            nshell=1,
            DW=True,
            trial_orbitals="X",
            dw_truncation=True,
            twist_y=twist_y,
        )
        if tuple(int(value) for value in model.DW_loc) != walls:
            raise AssertionError(f"unexpected wall locations {model.DW_loc}")
        modes = 2 * nx * ny
        ow_frames = [
            np.asarray(getattr(model, name), dtype=np.complex128).reshape(modes, -1)
            for name in ("WF_Ap", "WF_Bp", "WF_Am", "WF_Bm")
        ]
        h_flat = (
            ow_frames[0] @ ow_frames[0].conj().T
            + ow_frames[1] @ ow_frames[1].conj().T
            - ow_frames[2] @ ow_frames[2].conj().T
            - ow_frames[3] @ ow_frames[3].conj().T
        )
        h_flat = 0.5 * (h_flat + h_flat.conj().T)
        energies, eigenvectors = np.linalg.eigh(h_flat)
        rank = modes // 2
        occupied = np.ascontiguousarray(eigenvectors[:, :rank], dtype=np.complex128)
        gram = occupied.conj().T @ occupied
        diagnostics = {
            "hamiltonian_hermiticity_residual": float(
                np.max(np.abs(h_flat - h_flat.conj().T))
            ),
            "occupied_gram_residual": float(
                np.linalg.norm(gram - np.eye(rank), ord="fro") / np.sqrt(rank)
            ),
            "half_filling_gap": float(energies[rank] - energies[rank - 1]),
            "highest_occupied_energy": float(energies[rank - 1]),
            "lowest_empty_energy": float(energies[rank]),
        }
    if diagnostics["hamiltonian_hermiticity_residual"] > 1.0e-10:
        raise FloatingPointError("flattened parent is not Hermitian")
    if diagnostics["occupied_gram_residual"] > 1.0e-10:
        raise FloatingPointError("flattened ground frame is not orthonormal")
    return model, occupied, diagnostics


class RecordObserver:
    def __init__(self, *, nx: int, ny: int, cycles: int, selected_x: Sequence[int]):
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycles = int(cycles)
        self.selected_x = tuple(int(value) for value in selected_x)
        self.x_lookup = {value: index for index, value in enumerate(self.selected_x)}
        shape = (self.cycles, len(self.selected_x), self.ny)
        self.site_branch_log_weight = np.full(shape, np.nan, dtype=np.float64)
        self.measurement_log_weight = np.full(shape + (4,), np.nan, dtype=np.float64)
        self.outcomes = np.zeros(shape + (4,), dtype=np.bool_)
        self.site_ids = np.full(
            (self.cycles, len(self.selected_x) * self.ny), -1, dtype=np.int64
        )
        self.site_positions = np.zeros(self.cycles, dtype=np.int64)
        self.correction_log_weight = np.zeros(shape, dtype=np.float64)

    def __call__(self, **payload: Any) -> None:
        cycle = int(payload["cycle"]) - 1
        site_id = int(payload["site_id"])
        x, y = site_id % self.nx, site_id // self.nx
        if x not in self.x_lookup or not (0 <= y < self.ny):
            raise RuntimeError(f"observer received unselected site {site_id}")
        destination = (cycle, self.x_lookup[x], y)
        if np.isfinite(self.site_branch_log_weight[destination]):
            raise RuntimeError(f"duplicate measurement-center record {destination}")
        events = [dict(event) for event in payload["branch_events"]]
        measurements = {
            str(event["channel"]): event
            for event in events
            if str(event.get("kind")) == "measurement"
        }
        if tuple(measurements) != CHANNELS:
            raise RuntimeError(
                f"measurement channel order mismatch: {tuple(measurements)}"
            )
        for index, channel in enumerate(CHANNELS):
            event = measurements[channel]
            self.measurement_log_weight[destination + (index,)] = float(
                event["log_weight"]
            )
            self.outcomes[destination + (index,)] = bool(event["outcome_occupied"])
        branch_log = float(payload["branch_log_weight"])
        measurement_log = float(payload["measurement_log_weight"])
        correction_log = float(payload["correction_log_weight"])
        if not np.isclose(
            measurement_log,
            float(np.sum(self.measurement_log_weight[destination])),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError("site measurement log weight does not close")
        if not np.isclose(
            branch_log, measurement_log + correction_log, rtol=1.0e-12, atol=1.0e-12
        ):
            raise RuntimeError("site branch log weight does not close")
        self.site_branch_log_weight[destination] = branch_log
        self.correction_log_weight[destination] = correction_log
        position = int(self.site_positions[cycle])
        self.site_ids[cycle, position] = site_id
        self.site_positions[cycle] += 1

    def arrays(self) -> dict[str, np.ndarray]:
        if not np.all(np.isfinite(self.site_branch_log_weight)):
            missing = np.argwhere(~np.isfinite(self.site_branch_log_weight))
            raise RuntimeError(f"record is incomplete; first missing index {missing[0]}")
        if not np.all(np.isfinite(self.measurement_log_weight)):
            raise RuntimeError("measurement-event log weights are incomplete")
        expected_sites = len(self.selected_x) * self.ny
        if not np.all(self.site_positions == expected_sites):
            raise RuntimeError("cycle site counts are incomplete")
        if np.max(np.abs(self.correction_log_weight)) > 1.0e-12:
            raise RuntimeError("perfect-correction arm acquired stochastic correction weight")
        increments = np.sum(self.site_branch_log_weight, axis=(1, 2))
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        flattened_outcomes = self.outcomes.reshape(-1)
        return {
            "cycle_log_probability": increments,
            "cumulative_log_probability": cumulative,
            "site_branch_log_probability": self.site_branch_log_weight,
            "measurement_event_log_probability": self.measurement_log_weight,
            "measurement_outcomes_packed": np.packbits(flattened_outcomes),
            "measurement_outcomes_shape": np.asarray(self.outcomes.shape, dtype=np.int64),
            "site_schedule": self.site_ids,
            "observed_site_count": np.asarray(self.site_ids.size, dtype=np.int64),
            "measurement_event_count": np.asarray(flattened_outcomes.size, dtype=np.int64),
            "minimum_measurement_log_probability": np.asarray(
                np.min(self.measurement_log_weight), dtype=np.float64
            ),
            "maximum_measurement_log_probability": np.asarray(
                np.max(self.measurement_log_weight), dtype=np.float64
            ),
        }


def run_task(spec_payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = TaskSpec(**dict(spec_payload))
    valid, _ = verify_complete(spec)
    if valid:
        return {"status": "skipped", "task_id": spec.task_id, "elapsed_seconds": 0.0}
    if _SHARED_MODEL is None or _SHARED_GROUND_FRAME is None or _SHARED_GROUND_DIAGNOSTICS is None:
        raise RuntimeError("worker did not inherit the prepared model and ground state")
    if int(_SHARED_MODEL.Ny) != spec.ny:
        raise RuntimeError("worker inherited the wrong Ny model")
    selected_ids = np.asarray(
        [x + spec.nx * y for x in spec.selected_x for y in range(spec.ny)],
        dtype=np.int64,
    )
    observer = RecordObserver(
        nx=spec.nx, ny=spec.ny, cycles=spec.cycles, selected_x=spec.selected_x
    )
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        result = _SHARED_MODEL.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=spec.cycles,
            postselect=False,
            perfect_correction=True,
            samples=1,
            parallelize_samples=False,
            frame_init=_SHARED_GROUND_FRAME,
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=False,
            measurement_site_ids=selected_ids,
            random_seed=spec.seed,
            state_representation="physical_frame",
            trajectory_weight_observer=observer,
            return_native_state=True,
            require_no_covariance_materialization=True,
        )
    elapsed = time.perf_counter() - started
    native = result["native_final"]
    arrays = observer.arrays()
    cumulative = arrays["cumulative_log_probability"]
    if not np.isclose(
        cumulative[-1], float(native["log_weight"]), rtol=1.0e-11, atol=1.0e-10
    ):
        raise RuntimeError("observer and occupied-frame log weights disagree")
    output = Path(spec.result_path)
    save_npz_atomic(
        output,
        schema=np.asarray(RESULT_SCHEMA),
        revision=np.asarray(spec.revision),
        task_id=np.asarray(spec.task_id),
        arm=np.asarray(spec.arm),
        Nx=np.asarray(spec.nx, dtype=np.int64),
        Ny=np.asarray(spec.ny, dtype=np.int64),
        cycles=np.arange(spec.cycles + 1, dtype=np.int64),
        normalized_cycles=np.arange(spec.cycles + 1, dtype=np.float64) / spec.ny,
        sample_index=np.asarray(spec.sample_index, dtype=np.int64),
        seed=np.asarray(spec.seed, dtype=np.uint64),
        selected_x=np.asarray(spec.selected_x, dtype=np.int64),
        selected_site_ids=selected_ids,
        config_hash=np.asarray(spec.config_hash),
        canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY_POINT),
        dtype=np.asarray("complex128"),
        initialization=np.asarray("flattened_parent_lowest_half"),
        twist_y=np.asarray(float(_SHARED_MODEL.twist_y), dtype=np.float64),
        final_rank=np.asarray(native["rank"], dtype=np.int64),
        minimum_rank=np.asarray(native["min_rank"], dtype=np.int64),
        maximum_rank=np.asarray(native["max_rank"], dtype=np.int64),
        final_gram_residual=np.asarray(native["gram_residual"], dtype=np.float64),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
        **{
            key: np.asarray(value, dtype=np.float64)
            for key, value in _SHARED_GROUND_DIAGNOSTICS.items()
        },
        **arrays,
    )
    receipt = completion_payload(spec, output, elapsed)
    save_json_atomic(Path(spec.completion_path), receipt)
    valid, reason = verify_complete(spec)
    if not valid:
        raise RuntimeError(f"published result failed verification: {reason}")
    return {"status": "completed", "task_id": spec.task_id, "elapsed_seconds": elapsed}


def write_manifest(
    path: Path,
    *,
    config: Mapping[str, Any],
    tasks: Sequence[TaskSpec],
    completed_count: int,
    failures: Sequence[Mapping[str, Any]],
    workers: int,
) -> None:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "revision": config["revision"],
        "status": "failed" if failures else (
            "complete" if completed_count == len(tasks) else "running"
        ),
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "scientific_contract": scientific_contract(config),
        "config_hash": tasks[0].config_hash,
        "source_hashes": tasks[0].source_hashes,
        "task_count": len(tasks),
        "primary_task_count": 500,
        "robustness_task_count": 125,
        "completed_count": int(completed_count),
        "pending_count": int(len(tasks) - completed_count),
        "worker_count": int(workers),
        "one_blas_thread_per_worker": True,
        "parallelization": "independent Born trajectories in a forked process pool",
        "failures": list(failures),
        "updated_at": utc_now(),
    }
    save_json_atomic(path, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "campaign_config.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-new-tasks", type=int)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    if args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    global _SHARED_MODEL, _SHARED_GROUND_FRAME, _SHARED_GROUND_DIAGNOSTICS
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output_root = (
        args.output_root
        if args.output_root is not None
        else PACKAGE_ROOT / "outputs" / str(config["revision"])
    ).resolve()
    tasks = build_tasks(config, output_root, config_path)
    inventory = {task.task_id: verify_complete(task)[0] for task in tasks}
    pending = [task for task in tasks if not inventory[task.task_id]]
    if args.max_new_tasks is not None:
        pending = pending[: args.max_new_tasks]
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    requested = int(args.workers or config["execution"]["default_max_workers"])
    workers = min(requested, available, max(1, len(pending)))
    completed_count = int(sum(inventory.values()))
    manifest = output_root / "campaign_manifest.json"
    failures: list[dict[str, Any]] = []
    print("[campaign] boundary-only record-free-energy CPU pilot", flush=True)
    print(f"[campaign] revision={config['revision']}", flush=True)
    print(f"[campaign] canonical={CANONICAL_ENTRY_POINT}", flush=True)
    print(f"[campaign] output={output_root}", flush=True)
    print(
        "[campaign] Nx=16 Ny=[16,20,24,28,32] T=4Ny "
        f"primary=500 strip=125 complete={completed_count}/625 "
        f"selected={len(pending)} workers={workers}",
        flush=True,
    )
    print(f"[campaign] config_hash={tasks[0].config_hash}", flush=True)
    write_manifest(
        manifest,
        config=config,
        tasks=tasks,
        completed_count=completed_count,
        failures=failures,
        workers=workers,
    )
    if args.report_only or not pending:
        print("[campaign] report only" if args.report_only else "[campaign] already complete")
        return 0
    context = get_context("fork")
    selected_ids = {task.task_id for task in pending}
    progress = tqdm(
        total=len(pending),
        desc="boundary-only Nx16 CPU",
        unit="trajectory",
        dynamic_ncols=True,
    )
    campaign_started = time.perf_counter()
    try:
        for ny in reversed(config["geometry"]["Ny_values"]):
            size_tasks = [
                task for task in tasks if task.task_id in selected_ids and task.ny == int(ny)
            ]
            if not size_tasks:
                continue
            print(f"[prepare] Ny={ny}: constructing OW model and flattened ground state", flush=True)
            _SHARED_MODEL, _SHARED_GROUND_FRAME, _SHARED_GROUND_DIAGNOSTICS = (
                build_model_and_ground(
                    nx=16,
                    ny=int(ny),
                    walls=(4, 12),
                    twist_y=float(config["geometry"]["twist_y"]),
                )
            )
            print(
                f"[prepare] Ny={ny}: frame={_SHARED_GROUND_FRAME.shape} "
                f"gap={_SHARED_GROUND_DIAGNOSTICS['half_filling_gap']:.3e} "
                f"tasks={len(size_tasks)} workers={min(workers, len(size_tasks))}",
                flush=True,
            )
            attempts = {task.task_id: 0 for task in size_tasks}
            remaining = list(size_tasks)
            while remaining:
                retry: list[TaskSpec] = []
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(remaining)), mp_context=context
                ) as executor:
                    futures = {
                        executor.submit(run_task, asdict(task)): task for task in remaining
                    }
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            attempts[task.task_id] += 1
                            if attempts[task.task_id] <= args.max_retries:
                                retry.append(task)
                                progress.write(
                                    f"[retry] {task.task_id}: {exc} "
                                    f"({attempts[task.task_id]}/{args.max_retries})"
                                )
                            else:
                                failures.append(
                                    {
                                        "task_id": task.task_id,
                                        "error": repr(exc),
                                        "traceback": "".join(
                                            traceback.format_exception(exc)
                                        ),
                                    }
                                )
                                progress.write(f"[failed] {task.task_id}: {exc}")
                        else:
                            completed_count += 1
                            progress.update(1)
                            elapsed = time.perf_counter() - campaign_started
                            done_now = max(1, progress.n)
                            eta = elapsed / done_now * (len(pending) - done_now)
                            progress.set_postfix(
                                Ny=ny,
                                last_s=f"{result['elapsed_seconds']:.1f}",
                                eta_h=f"{eta / 3600.0:.1f}",
                            )
                        write_manifest(
                            manifest,
                            config=config,
                            tasks=tasks,
                            completed_count=completed_count,
                            failures=failures,
                            workers=workers,
                        )
                remaining = retry
            _SHARED_MODEL = None
            _SHARED_GROUND_FRAME = None
            _SHARED_GROUND_DIAGNOSTICS = None
    finally:
        progress.close()
    write_manifest(
        manifest,
        config=config,
        tasks=tasks,
        completed_count=completed_count,
        failures=failures,
        workers=workers,
    )
    print(
        f"[campaign] complete={completed_count}/625 failures={len(failures)} "
        f"manifest={manifest}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
