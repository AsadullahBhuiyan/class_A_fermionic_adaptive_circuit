#!/usr/bin/env python3
"""Run the resumable Nx=16 hard/soft many-body Lyapunov CPU campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN
from lyapunov_observer import ActiveSpectrumObserver, spectrum_checkpoint_cycles


CHECKPOINT_SCHEMA = "maxmix_manybody_lyapunov_cpu_checkpoint_v2"
RESULT_SCHEMA = "maxmix_manybody_lyapunov_cpu_result_v2"
COMPLETION_SCHEMA = "maxmix_manybody_lyapunov_cpu_completion_v2"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    revision: str
    nx: int
    ny: int
    cycles: int
    sample_index: int
    seed: int
    config_hash: str
    source_hashes: dict[str, str]
    result_path: str
    completion_path: str
    checkpoint_path: str
    checkpoint_stride: int
    soft_mode_count: int
    leading_level_count: int
    cap_tolerance: float
    construction: str = "hard"
    inner_progress: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(
    root_seed: int,
    revision: str,
    construction: str | int,
    nx: int,
    ny: int,
    sample_index: int | None = None,
) -> int:
    if sample_index is None:
        # Backward-compatible helper form: (root, revision, nx, ny, sample).
        sample_index = int(ny)
        ny = int(nx)
        nx = int(construction)
        construction = "hard"
    raw = canonical_json(
        [int(root_seed), revision, str(construction), int(nx), int(ny), int(sample_index)]
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little", signed=False)


def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_npz_atomic(path: Path, *, compressed: bool = True, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        writer = np.savez_compressed if compressed else np.savez
        writer(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def locked_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": config["schema"],
        "revision": config["revision"],
        "root_seed": int(config["root_seed"]),
        "geometry": config["geometry"],
        "dynamics": config["dynamics"],
        "observer": config["observer"],
        "checkpoint_schema": config["checkpoint"]["schema"],
    }


def validate_config(config: Mapping[str, Any]) -> None:
    geometry, dynamics = config["geometry"], config["dynamics"]
    expected = {
        "Nx": 16,
        "Ny_values": [20, 22, 24, 26, 28, 30],
        "domain_wall": True,
        "domain_wall_interval": [4, 12],
        "constructions": {
            "hard": {
                "domain_wall_truncation": True,
                "measurement_slab_only": True,
            },
            "soft": {
                "domain_wall_truncation": False,
                "measurement_slab_only": False,
            },
        },
    }
    if geometry != expected:
        raise ValueError(f"geometry differs from locked CPU pilot: {geometry}")
    required = {
        "cycles_rule": "4*Ny",
        "samples_per_case": 100,
        "nshell": 1,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "trial_orbitals": "X",
        "initialization": "maxmix",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "n_a": 0.5,
        "state_representation": "covariance",
        "dtype": "complex128",
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
    }
    if dynamics != required:
        raise ValueError(f"dynamics differs from locked CPU pilot: {dynamics}")
    if config["checkpoint"]["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")


def source_hashes() -> dict[str, str]:
    files = (
        Path(__file__).resolve(),
        PACKAGE_ROOT / "lyapunov_observer.py",
        REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py",
        REPO_ROOT / "src" / "fgtn" / "occupied_frame.py",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in files}


def build_tasks(config: Mapping[str, Any], output_root: Path, *, inner_progress: bool = False) -> list[TaskSpec]:
    validate_config(config)
    contract_hash = sha256_bytes(canonical_json(locked_scientific_contract(config)).encode())
    hashes = source_hashes()
    tasks: list[TaskSpec] = []
    nx = int(config["geometry"]["Nx"])
    revision = str(config["revision"])
    root_seed = int(config["root_seed"])
    sample_count = int(config["dynamics"]["samples_per_case"])
    # Interleave constructions and sizes, largest first, so the first completed
    # tasks immediately calibrate the worst-case hard/soft runtime.
    for sample_index in range(sample_count):
        for ny in reversed(config["geometry"]["Ny_values"]):
            for construction in ("hard", "soft"):
                task_id = (
                    f"{construction}_Nx{nx}_Ny{int(ny):03d}_sample{sample_index:03d}"
                )
                directory = output_root / "results" / construction / f"Ny{int(ny):03d}"
                tasks.append(
                    TaskSpec(
                        task_id=task_id,
                        revision=revision,
                        construction=construction,
                        nx=nx,
                        ny=int(ny),
                        cycles=4 * int(ny),
                        sample_index=sample_index,
                        seed=stable_seed(
                            root_seed, revision, construction, nx, int(ny), sample_index
                        ),
                        config_hash=contract_hash,
                        source_hashes=hashes,
                        result_path=str(directory / f"trajectory_{sample_index:03d}.npz"),
                        completion_path=str(
                            directory / f"trajectory_{sample_index:03d}.complete.json"
                        ),
                        checkpoint_path=str(
                            output_root
                            / "checkpoints"
                            / construction
                            / f"Ny{int(ny):03d}"
                            / f"trajectory_{sample_index:03d}.checkpoint.npz"
                        ),
                        checkpoint_stride=int(config["checkpoint"]["stride_cycles"]),
                        soft_mode_count=int(config["observer"]["soft_mode_count"]),
                        leading_level_count=int(config["observer"]["leading_level_count"]),
                        cap_tolerance=float(config["observer"]["cap_tolerance"]),
                        inner_progress=bool(inner_progress),
                    )
                )
    if len(tasks) != 1200 or len({task.seed for task in tasks}) != 1200:
        raise AssertionError("locked task expansion must produce 1,200 unique trajectories")
    return tasks


def completion_payload(spec: TaskSpec, result_path: Path, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "revision": spec.revision,
        "task_id": spec.task_id,
        "construction": spec.construction,
        "Nx": spec.nx,
        "Ny": spec.ny,
        "cycles": spec.cycles,
        "sample_index": spec.sample_index,
        "seed": spec.seed,
        "config_hash": spec.config_hash,
        "source_hashes": spec.source_hashes,
        "result_filename": result_path.name,
        "result_bytes": result_path.stat().st_size,
        "result_sha256": sha256_file(result_path),
        "elapsed_seconds": float(elapsed_seconds),
        "completed_at": utc_now(),
    }


def verify_complete(spec: TaskSpec) -> tuple[bool, str]:
    result_path, completion_path = Path(spec.result_path), Path(spec.completion_path)
    if not result_path.exists() or not completion_path.exists():
        return False, "missing result/completion pair"
    try:
        receipt = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "schema": COMPLETION_SCHEMA,
            "revision": spec.revision,
            "task_id": spec.task_id,
            "construction": spec.construction,
            "Nx": spec.nx,
            "Ny": spec.ny,
            "cycles": spec.cycles,
            "sample_index": spec.sample_index,
            "seed": spec.seed,
            "config_hash": spec.config_hash,
            "source_hashes": spec.source_hashes,
            "result_filename": result_path.name,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                return False, f"completion mismatch: {key}"
        if int(receipt["result_bytes"]) != result_path.stat().st_size:
            return False, "result size mismatch"
        if receipt["result_sha256"] != sha256_file(result_path):
            return False, "result checksum mismatch"
        with np.load(result_path, allow_pickle=False) as data:
            if str(data["schema"].item()) != RESULT_SCHEMA:
                return False, "result schema mismatch"
            if str(data["task_id"].item()) != spec.task_id:
                return False, "result task mismatch"
            if int(data["cycles"][-1]) != spec.cycles:
                return False, "result horizon mismatch"
            if not np.all(np.asarray(data["cycle_seen"], dtype=bool)):
                return False, "result cycles incomplete"
        return True, "verified"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, f"verification error: {exc}"


def _checkpoint_identity(spec: TaskSpec) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "revision": spec.revision,
        "task_id": spec.task_id,
        "construction": spec.construction,
        "Nx": spec.nx,
        "Ny": spec.ny,
        "target_cycles": spec.cycles,
        "sample_index": spec.sample_index,
        "seed": spec.seed,
        "config_hash": spec.config_hash,
        "source_hashes": spec.source_hashes,
    }


def save_checkpoint(
    spec: TaskSpec,
    *,
    engine_state: Mapping[str, Any],
    observer: ActiveSpectrumObserver,
    elapsed_seconds: float,
) -> None:
    path = Path(spec.checkpoint_path)
    completed = int(engine_state["completed_cycles"])
    observer.validate(completed_cycle=completed)
    metadata = {
        **_checkpoint_identity(spec),
        "completed_cycle": completed,
        "elapsed_seconds": float(elapsed_seconds),
        "saved_at": utc_now(),
        "engine_state": {
            key: jsonable(value)
            for key, value in engine_state.items()
            if key not in {"G", "last_ordered_site_ids"}
        },
    }
    save_npz_atomic(
        path,
        compressed=True,
        metadata_json=np.asarray(canonical_json(metadata)),
        engine_G=np.asarray(engine_state["G"], dtype=np.complex128),
        engine_last_ordered_site_ids=np.asarray(engine_state["last_ordered_site_ids"], dtype=np.int64),
        **observer.checkpoint_arrays(),
    )


def load_checkpoint(
    spec: TaskSpec, observer: ActiveSpectrumObserver
) -> tuple[dict[str, Any] | None, float]:
    path = Path(spec.checkpoint_path)
    if not path.exists():
        return None, 0.0
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            for key, value in _checkpoint_identity(spec).items():
                if metadata.get(key) != value:
                    raise ValueError(f"checkpoint identity mismatch: {key}")
            completed = int(metadata["completed_cycle"])
            if completed < 1 or completed > spec.cycles:
                raise ValueError("checkpoint completed cycle is invalid")
            observer.restore_checkpoint_arrays(data, completed_cycle=completed)
            state = dict(metadata["engine_state"])
            state["G"] = np.asarray(data["engine_G"], dtype=np.complex128).copy()
            state["last_ordered_site_ids"] = np.asarray(
                data["engine_last_ordered_site_ids"], dtype=np.int64
            ).copy()
        if int(state["completed_cycles"]) != completed:
            raise ValueError("engine and wrapper checkpoint cycles disagree")
        return state, float(metadata.get("elapsed_seconds", 0.0))
    except Exception as exc:
        raise RuntimeError(f"invalid checkpoint {path}: {exc}") from exc


def engine_state_fingerprint(state: Mapping[str, Any]) -> str:
    """Hash the exact restartable physical state and RNG payload without saving it."""
    digest = hashlib.sha256()
    covariance = np.ascontiguousarray(np.asarray(state["G"], dtype=np.complex128))
    ordered = np.ascontiguousarray(
        np.asarray(state["last_ordered_site_ids"], dtype=np.int64)
    )
    digest.update(str(covariance.dtype).encode())
    digest.update(canonical_json(list(covariance.shape)).encode())
    digest.update(covariance.view(np.uint8))
    digest.update(ordered.view(np.uint8))
    remainder = {
        key: jsonable(value)
        for key, value in state.items()
        if key not in {"G", "last_ordered_site_ids"}
    }
    digest.update(canonical_json(remainder).encode())
    return digest.hexdigest()


def make_model(spec: TaskSpec) -> classA_U1FGTN:
    # Projector construction also calls BLAS/LAPACK. Keep it under the same
    # one-thread contract as evolution so a resumed process reconstructs the
    # checkpoint signature and numerical operators deterministically.
    with threadpool_limits(limits=1):
        wall_locations = (spec.nx // 4, spec.nx - spec.nx // 4)
        model = classA_U1FGTN(
            spec.nx,
            spec.ny,
            DW=True,
            nshell=1,
            alpha_1=1.0,
            alpha_2=30.0,
            trial_orbitals="X",
            dw_truncation=spec.construction == "hard",
            dw_interval=wall_locations,
        )
        model.construct_OW_projectors(
            nshell=1,
            DW=True,
            trial_orbitals="X",
            dw_truncation=spec.construction == "hard",
        )
    if tuple(int(value) for value in model.DW_loc) != wall_locations:
        raise AssertionError(f"unexpected domain walls: {model.DW_loc}")
    active = model.active_top_layer_indices(meas_slab_only=spec.construction == "hard")
    expected_active = (
        2 * (wall_locations[1] - wall_locations[0] + 1)
        if spec.construction == "hard"
        else 2 * spec.nx
    ) * spec.ny
    if active.size != expected_active:
        raise AssertionError(f"active modes {active.size} != {expected_active}")
    return model


def finalize_result(
    spec: TaskSpec,
    observer: ActiveSpectrumObserver,
    *,
    elapsed_seconds: float,
    final_engine_state: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = Path(spec.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = observer.result_arrays()
    save_npz_atomic(
        result_path,
        compressed=True,
        schema=np.asarray(RESULT_SCHEMA),
        revision=np.asarray(spec.revision),
        task_id=np.asarray(spec.task_id),
        construction=np.asarray(spec.construction),
        Nx=np.asarray(spec.nx, dtype=np.int64),
        Ny=np.asarray(spec.ny, dtype=np.int64),
        cycles=np.arange(spec.cycles + 1, dtype=np.int64),
        spectrum_cycles=observer.spectrum_cycles,
        sample_index=np.asarray(spec.sample_index, dtype=np.int64),
        seed=np.asarray(spec.seed, dtype=np.uint32),
        config_hash=np.asarray(spec.config_hash),
        canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY_POINT),
        dtype=np.asarray("complex128"),
        domain_wall_locations=np.asarray(observer.geometry.wall_locations, dtype=np.int64),
        active_mode_count=np.asarray(observer.geometry.active_mode_count, dtype=np.int64),
        final_engine_state_sha256=np.asarray(engine_state_fingerprint(final_engine_state)),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
        **arrays,
    )
    receipt = completion_payload(spec, result_path, elapsed_seconds)
    save_json_atomic(Path(spec.completion_path), receipt)
    valid, reason = verify_complete(spec)
    if not valid:
        raise RuntimeError(f"newly written result failed verification: {reason}")
    Path(spec.checkpoint_path).unlink(missing_ok=True)
    return {"status": "completed", "task_id": spec.task_id, "elapsed_seconds": elapsed_seconds}


def run_task(spec_payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = TaskSpec(**dict(spec_payload))
    valid, _ = verify_complete(spec)
    if valid:
        return {"status": "skipped", "task_id": spec.task_id, "elapsed_seconds": 0.0}
    model = make_model(spec)
    active = model.active_top_layer_indices(meas_slab_only=spec.construction == "hard")
    wall_locations = tuple(int(value) for value in model.DW_loc)
    observer = ActiveSpectrumObserver(
        nx=spec.nx,
        ny=spec.ny,
        cycles=spec.cycles,
        active_indices=active,
        wall_locations=wall_locations,
        construction=spec.construction,
        spectrum_cycles=spectrum_checkpoint_cycles(spec.ny, spec.cycles // spec.ny),
        soft_mode_count=spec.soft_mode_count,
        leading_level_count=spec.leading_level_count,
        cap_tolerance=spec.cap_tolerance,
    )
    checkpoint_state, elapsed_before = load_checkpoint(spec, observer)
    if checkpoint_state is not None and int(checkpoint_state["completed_cycles"]) == spec.cycles:
        return finalize_result(
            spec,
            observer,
            elapsed_seconds=elapsed_before,
            final_engine_state=checkpoint_state,
        )
    started = time.perf_counter()
    final_engine_state: dict[str, Any] | None = None

    def checkpoint_observer(*, cycle: int, state: Mapping[str, Any]) -> None:
        nonlocal final_engine_state
        if int(cycle) == spec.cycles:
            final_engine_state = dict(state)
        if (
            int(cycle) % spec.checkpoint_stride
            and int(cycle) not in set(observer.spectrum_cycles.tolist())
            and int(cycle) != spec.cycles
        ):
            return
        save_checkpoint(
            spec,
            engine_state=state,
            observer=observer,
            elapsed_seconds=elapsed_before + time.perf_counter() - started,
        )
        print(
            f"[checkpoint] {spec.task_id} cycle {int(cycle)}/{spec.cycles}",
            flush=True,
        )

    with threadpool_limits(limits=1):
        model.run_markov_circuit(
            G_history=False,
            progress=spec.inner_progress,
            cycles=spec.cycles,
            postselect=False,
            perfect_correction=True,
            samples=1,
            parallelize_samples=False,
            init_mode="maxmix",
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=spec.construction == "hard",
            random_seed=spec.seed,
            state_representation="covariance",
            cycle_observer=observer.observe,
            trajectory_weight_observer=observer.record_site,
            checkpoint_state=checkpoint_state,
            checkpoint_observer=checkpoint_observer,
        )
    elapsed = elapsed_before + time.perf_counter() - started
    if final_engine_state is None:
        raise RuntimeError("canonical engine did not emit its final checkpoint state")
    return finalize_result(
        spec,
        observer,
        elapsed_seconds=elapsed,
        final_engine_state=final_engine_state,
    )


def write_manifest(
    path: Path,
    *,
    config: Mapping[str, Any],
    tasks: list[TaskSpec],
    failures: list[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    statuses = [verify_complete(task)[0] for task in tasks]
    payload = {
        "schema": "maxmix_manybody_lyapunov_cpu_manifest_v2",
        "revision": config["revision"],
        "status": "failed" if failures else ("complete" if all(statuses) else "incomplete"),
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "scientific_contract": locked_scientific_contract(config),
        "config_hash": tasks[0].config_hash,
        "source_hashes": tasks[0].source_hashes,
        "task_count": len(tasks),
        "completed_count": int(sum(statuses)),
        "pending_count": int(len(tasks) - sum(statuses)),
        "worker_count": int(workers),
        "one_blas_thread_per_worker": True,
        "failures": failures,
        "updated_at": utc_now(),
    }
    save_json_atomic(path, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "campaign_config.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", default="auto", help="auto or a positive integer")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-new-tasks", type=int)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--inner-progress", action="store_true", help="engine site bar; requires one worker")
    parser.add_argument("--max-retries", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    if args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output_root = (
        args.output_root
        if args.output_root is not None
        else PACKAGE_ROOT / "outputs" / str(config["revision"])
    ).resolve()
    tasks = build_tasks(config, output_root, inner_progress=args.inner_progress)
    inventory = [(task, *verify_complete(task)) for task in tasks]
    pending = [task for task, complete, _ in inventory if not complete]
    max_workers = int(args.max_workers or config["execution"]["default_max_workers"])
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    workers = min(max_workers, available, max(1, len(pending)))
    if args.workers != "auto":
        workers = min(int(args.workers), workers)
    if workers < 1:
        raise ValueError("worker count must be positive")
    if args.inner_progress and workers != 1:
        raise ValueError("--inner-progress requires --workers 1")
    if args.max_new_tasks is not None:
        pending = pending[: int(args.max_new_tasks)]
    manifest_path = output_root / "campaign_manifest.json"
    print("[campaign] local Nx=16 hard/soft CPU campaign", flush=True)
    print(f"[campaign] revision={config['revision']}", flush=True)
    print(f"[campaign] canonical={CANONICAL_ENTRY_POINT}", flush=True)
    print(f"[campaign] output={output_root}", flush=True)
    print(
        f"[campaign] Nx=16 hard+soft Ny={config['geometry']['Ny_values']} S=100/case "
        f"T=4Ny tasks=1200 complete={len(tasks)-len([x for x in tasks if not verify_complete(x)[0]])} "
        f"pending={len([x for x in tasks if not verify_complete(x)[0]])} selected={len(pending)} workers={workers}",
        flush=True,
    )
    print(f"[campaign] config_hash={tasks[0].config_hash}", flush=True)
    failures: list[dict[str, Any]] = []
    write_manifest(manifest_path, config=config, tasks=tasks, failures=failures, workers=workers)
    if args.report_only or not pending:
        print("[campaign] report only" if args.report_only else "[campaign] already complete", flush=True)
        return 0
    retries: dict[str, int] = {}
    progress = tqdm(
        total=len(pending),
        desc="Nx16 hard/soft Lyapunov CPU",
        unit="trajectory",
        dynamic_ncols=True,
    )
    context = get_context("spawn")
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(run_task, asdict(task)): task for task in pending}
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    task = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        attempt = retries.get(task.task_id, 0)
                        if attempt < args.max_retries:
                            retries[task.task_id] = attempt + 1
                            futures[executor.submit(run_task, asdict(task))] = task
                            continue
                        failures.append(
                            {"task_id": task.task_id, "error": repr(exc), "traceback": traceback.format_exc()}
                        )
                        progress.write(f"[failed] {task.task_id}: {exc}")
                    else:
                        progress.update(1)
                        elapsed = time.perf_counter() - started
                        done = max(1, progress.n)
                        remaining = elapsed / done * (len(pending) - done)
                        progress.set_postfix(completed=result["task_id"], eta_h=f"{remaining/3600:.1f}")
                    write_manifest(
                        manifest_path, config=config, tasks=tasks, failures=failures, workers=workers
                    )
    finally:
        progress.close()
    manifest = write_manifest(
        manifest_path, config=config, tasks=tasks, failures=failures, workers=workers
    )
    print(
        f"[campaign] status={manifest['status']} complete={manifest['completed_count']}/1200 "
        f"failures={len(failures)} manifest={manifest_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
