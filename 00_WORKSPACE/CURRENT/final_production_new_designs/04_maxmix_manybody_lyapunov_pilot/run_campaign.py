#!/usr/bin/env python3
"""Run the simple batched A100 many-body Lyapunov campaign."""

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
from typing import Any

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
from lyapunov_observer import (  # noqa: E402
    OBSERVER_SCHEMA,
    BatchedActiveSpectrumObserver,
)


BUNDLE = "04_maxmix_manybody_lyapunov_pilot"
RESULT_SCHEMA = "maxmix_manybody_lyapunov_gpu_task_v2"
COMPLETION_SCHEMA = "maxmix_manybody_lyapunov_gpu_completion_v2"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
EXPECTED_REVISION = "maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2"
EXPECTED_ROOT_SEED = 2026090304
EXPECTED_NX = 20
EXPECTED_NY_VALUES = (20, 22, 24, 26, 28, 30, 36, 40)
EXPECTED_SAMPLES = 100
EXPECTED_SHARD_SIZE = 5
EXPECTED_TASKS = 160
EXPECTED_TRAJECTORIES = 800
SOURCE_FILES = (
    "run_campaign.py",
    "lyapunov_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)


@dataclass(frozen=True)
class Task:
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
        "samples_per_Ny": EXPECTED_SAMPLES,
        "shard_size": EXPECTED_SHARD_SIZE,
        "cycles_rule": "2*Ny",
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
            "init_mode": "maxmix",
            "sequence": "raster_y",
            "perfect_correction": True,
            "postselect": False,
            "postselect_probability": 0.0,
            "n_a": 0.5,
            "state_representation": "covariance",
            "backend": "local",
            "triv_region_local_mode": False,
        },
        "observer": {
            "spectrum_cycle_stride": 4,
            "extra_spectrum_cycles": ["Ny", "3*Ny/2", "2*Ny"],
            "soft_mode_count": 16,
            "leading_level_count": 64,
            "cap_tolerance": 1.0e-9,
        },
        "analysis": {
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 2026090305,
            "relative_shift_threshold": 0.1,
            "sample_prefixes": [25, 50, 75, 100],
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


def task_seed(
    root_seed: int,
    *,
    ny: int,
    batch_index: int,
    sample_start: int,
    sample_stop: int,
) -> int:
    label = (
        f"{int(root_seed)}|Nx={EXPECTED_NX}|Ny={int(ny)}|"
        f"batch={int(batch_index)}|samples={int(sample_start)}:{int(sample_stop)}"
    )
    return int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    ) & ((1 << 63) - 1)


def expand_tasks(config: dict[str, Any]) -> list[Task]:
    config = validate_config(config)
    tasks: list[Task] = []
    # Largest circumference first gives an immediate worst-case timing calibration.
    for ny in sorted((int(value) for value in config["Ny_values"]), reverse=True):
        for batch_index, sample_start in enumerate(
            range(0, EXPECTED_SAMPLES, EXPECTED_SHARD_SIZE)
        ):
            sample_stop = min(sample_start + EXPECTED_SHARD_SIZE, EXPECTED_SAMPLES)
            task_id = (
                f"Ny{ny:02d}_batch-{batch_index:03d}_"
                f"samples-{sample_start:03d}-{sample_stop - 1:03d}"
            )
            tasks.append(
                Task(
                    ny=ny,
                    batch_index=batch_index,
                    sample_start=sample_start,
                    sample_stop=sample_stop,
                    task_id=task_id,
                    seed=task_seed(
                        int(config["root_seed"]),
                        ny=ny,
                        batch_index=batch_index,
                        sample_start=sample_start,
                        sample_stop=sample_stop,
                    ),
                )
            )
    if len(tasks) != EXPECTED_TASKS or len({task.task_id for task in tasks}) != EXPECTED_TASKS:
        raise RuntimeError(f"task expansion must contain {EXPECTED_TASKS} unique tasks")
    if len({task.seed for task in tasks}) != EXPECTED_TASKS:
        raise RuntimeError("task seeds must be globally unique")
    if sum(task.sample_count for task in tasks) != EXPECTED_TRAJECTORIES:
        raise RuntimeError(f"task expansion must cover {EXPECTED_TRAJECTORIES} trajectories")
    for ny in EXPECTED_NY_VALUES:
        selected = [task for task in tasks if task.ny == ny]
        indices = [index for task in selected for index in task.global_sample_indices]
        if indices != list(range(EXPECTED_SAMPLES)):
            raise RuntimeError(f"Ny={ny} task table does not cover exactly 100 samples")
    return tasks


def task_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = output_root / "results" / f"Ny{task.ny:02d}"
    stem = (
        f"batch_{task.batch_index:03d}_"
        f"samples_{task.sample_start:03d}-{task.sample_stop - 1:03d}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def _completion_identity(
    *, task: Task, config_sha256: str, hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "task_id": task.task_id,
        "Nx": EXPECTED_NX,
        "Ny": task.ny,
        "cycles": task.cycles,
        "batch_index": task.batch_index,
        "sample_start": task.sample_start,
        "sample_stop": task.sample_stop,
        "sample_count": task.sample_count,
        "global_sample_indices": list(task.global_sample_indices),
        "seed": task.seed,
        "config_sha256": config_sha256,
        "source_hashes": hashes,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "observer_schema": OBSERVER_SCHEMA,
    }


def verified_complete(
    *,
    output_root: Path,
    task: Task,
    config_sha256: str,
    hashes: dict[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = task_paths(output_root, task)
    result_is_file = result_path.is_file()
    completion_is_file = completion_path.is_file()
    if not result_is_file and not completion_is_file:
        return False, "missing result/completion pair"
    if not result_is_file or not completion_is_file:
        return False, "incomplete result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion JSON: {exc}"
    for key, value in _completion_identity(
        task=task, config_sha256=config_sha256, hashes=hashes
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


def scan_resume_inventory(
    *,
    output_root: Path,
    tasks: list[Task],
    config_sha256: str,
    hashes: dict[str, str],
) -> list[tuple[Task, bool, str]]:
    """Verify existing products without issuing a Drive lookup for every absence."""
    print(
        f"[resume scan] checking {len(tasks)} deterministic task slots",
        flush=True,
    )
    progress = tqdm(
        total=len(tasks),
        desc="Resume scan",
        unit="task",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )
    inventory: list[tuple[Task, bool, str]] = []
    results_root = output_root / "results"
    try:
        if not results_root.is_dir():
            inventory = [
                (task, False, "missing result/completion pair") for task in tasks
            ]
            progress.update(len(tasks))
            return inventory

        names_by_ny: dict[int, set[str]] = {}
        for ny in sorted({task.ny for task in tasks}, reverse=True):
            directory = results_root / f"Ny{ny:02d}"
            try:
                names_by_ny[ny] = {entry.name for entry in directory.iterdir()}
            except FileNotFoundError:
                names_by_ny[ny] = set()
            except OSError as exc:
                raise OSError(
                    f"resume scan could not list Drive directory {directory}: {exc}"
                ) from exc

        for task in tasks:
            result_path, completion_path = task_paths(output_root, task)
            names = names_by_ny[task.ny]
            result_listed = result_path.name in names
            completion_listed = completion_path.name in names
            if not result_listed and not completion_listed:
                status = (False, "missing result/completion pair")
            elif not result_listed or not completion_listed:
                status = (False, "incomplete result/completion pair")
            else:
                status = verified_complete(
                    output_root=output_root,
                    task=task,
                    config_sha256=config_sha256,
                    hashes=hashes,
                )
            inventory.append((task, *status))
            progress.update(1)
        return inventory
    finally:
        progress.close()


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


def _check_space(path: Path, *, required_bytes: int, label: str) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path).free)
    if free < int(required_bytes):
        raise RuntimeError(
            f"insufficient {label} space: free={free / 1024**3:.2f} GiB, "
            f"required={required_bytes / 1024**3:.2f} GiB"
        )
    return free


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
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
    """Copy, read back through DriveFS, checksum, and atomically publish."""
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


def build_model(config: dict[str, Any], ny: int) -> Any:
    protocol = config["protocol"]
    print(f"[model] constructing Nx={EXPECTED_NX}, Ny={ny} projectors", flush=True)
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
        backend=protocol["backend"],
    )
    if tuple(int(value) for value in model.DW_loc) != (5, 15):
        raise RuntimeError(f"unexpected domain-wall positions: {model.DW_loc}")
    active = model.active_top_layer_indices(meas_slab_only=True)
    if int(active.numel()) != 22 * int(ny):
        raise RuntimeError("active transfer space does not contain 22*Ny modes")
    if model.dtype != torch.complex128:
        raise RuntimeError("constructed model is not complex128")
    print(
        f"[model] ready: walls={model.DW_loc}, active_modes={int(active.numel())}",
        flush=True,
    )
    return model


def run_task(
    *, model: Any, config: dict[str, Any], task: Task
) -> tuple[dict[str, np.ndarray], float]:
    np.random.seed(task.seed % (2**32))
    torch.manual_seed(task.seed)
    torch.cuda.manual_seed_all(task.seed)
    observer_config = config["observer"]
    active = model.active_top_layer_indices(meas_slab_only=True)
    observer = BatchedActiveSpectrumObserver(
        nx=EXPECTED_NX,
        ny=task.ny,
        cycles=task.cycles,
        samples=task.sample_count,
        active_indices=active,
        full_mode_count=model.Nlayer,
        wall_locations=tuple(model.DW_loc),
        soft_mode_count=int(observer_config["soft_mode_count"]),
        leading_level_count=int(observer_config["leading_level_count"]),
        cap_tolerance=float(observer_config["cap_tolerance"]),
    )
    print(
        f"[task start] {task.task_id}: {task.sample_count} trajectories, "
        f"{task.cycles} cycles, seed={task.seed}",
        flush=True,
    )
    started = time.monotonic()
    result = model.run_markov_circuit(
        G_history=False,
        progress=True,
        cycles=task.cycles,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        samples=task.sample_count,
        init_mode="maxmix",
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=True,
        batch_size=task.sample_count,
        return_data=False,
        state_representation="covariance",
        cycle_observer=observer.observe,
        record_observer=observer.record_event,
        track_choi=False,
        return_native_state=False,
    )
    torch.cuda.synchronize(model.device)
    elapsed = time.monotonic() - started
    if int(result.get("samples", -1)) != task.sample_count:
        raise RuntimeError("canonical engine returned the wrong sample count")
    if bool(result.get("choi_tracked", False)):
        raise RuntimeError("canonical engine unexpectedly enabled Choi tracking")
    payload = observer.result_arrays()
    payload.update(
        {
            "schema": np.asarray(RESULT_SCHEMA),
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "bundle": np.asarray(BUNDLE),
            "sampling_revision": np.asarray(EXPECTED_REVISION),
            "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
            "task_id": np.asarray(task.task_id),
            "Nx": np.asarray(EXPECTED_NX, dtype=np.int64),
            "Ny": np.asarray(task.ny, dtype=np.int64),
            "cycles_total": np.asarray(task.cycles, dtype=np.int64),
            "batch_index": np.asarray(task.batch_index, dtype=np.int64),
            "sample_start": np.asarray(task.sample_start, dtype=np.int64),
            "sample_stop": np.asarray(task.sample_stop, dtype=np.int64),
            "global_sample_indices": np.asarray(
                task.global_sample_indices, dtype=np.int64
            ),
            "batch_seed": np.asarray(task.seed, dtype=np.int64),
            "elapsed_seconds": np.asarray(elapsed, dtype=np.float64),
            "active_mode_count": np.asarray(22 * task.ny, dtype=np.int64),
            "wall_locations": np.asarray((5, 15), dtype=np.int64),
            "dtype": np.asarray("complex128"),
            "init_mode": np.asarray("maxmix"),
            "sequence": np.asarray("raster_y"),
            "normalization_sector": np.asarray(
                "active_slab_after_born_conditioned_exterior_preparation"
            ),
            "singular_value_convention": np.asarray("ell=log(sigma^2)"),
        }
    )
    return payload, elapsed


def save_task(
    *,
    output_root: Path,
    scratch_root: Path,
    task: Task,
    payload: dict[str, np.ndarray],
    elapsed_seconds: float,
    config_sha256: str,
    hashes: dict[str, str],
) -> None:
    task_scratch = scratch_root / task.task_id
    if task_scratch.exists():
        shutil.rmtree(task_scratch)
    task_scratch.mkdir(parents=True)
    local_result = task_scratch / "result.npz"
    _write_npz(local_result, payload)
    result_path, completion_path = task_paths(output_root, task)
    published = publish_file(local_result, result_path)
    completion = _completion_identity(
        task=task, config_sha256=config_sha256, hashes=hashes
    )
    completion.update(
        {
            "result_filename": result_path.name,
            "result_bytes": published["bytes"],
            "result_sha256": published["sha256"],
            "samples_saved": task.sample_count,
            "elapsed_seconds": float(elapsed_seconds),
            "completed_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
    )
    local_completion = task_scratch / "completion.json"
    _write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    valid, reason = verified_complete(
        output_root=output_root,
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    if not valid:
        raise OSError(f"published task failed final verification: {reason}")
    shutil.rmtree(task_scratch)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-tasks", type=int)
    args = parser.parse_args(argv)
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    print("[startup] Lyapunov campaign runner entered", flush=True)
    args = parse_args(argv)
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    output_root = args.output_root.resolve()
    scratch_root = args.scratch_root.resolve()
    tasks = expand_tasks(config)
    hashes = source_hashes()
    config_sha256 = config_hash(config)
    print("[resolved configuration]", flush=True)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)
    print(f"[source] bundle={BUNDLE_ROOT}", flush=True)
    print(f"[source] canonical={CANONICAL_ENTRY_POINT}", flush=True)
    print(f"[output] {output_root}", flush=True)
    print(f"[scratch] {scratch_root}", flush=True)
    print(
        f"[workload] sizes={len(EXPECTED_NY_VALUES)}, tasks={len(tasks)}, "
        f"trajectories={EXPECTED_TRAJECTORIES}, batch_size={EXPECTED_SHARD_SIZE}",
        flush=True,
    )
    inventory = scan_resume_inventory(
        output_root=output_root,
        tasks=tasks,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    completed = sum(valid for _, valid, _ in inventory)
    pending = [task for task, valid, _ in inventory if not valid]
    print(
        f"[resume] completed={completed}, pending={len(pending)}, total={len(tasks)}",
        flush=True,
    )
    if args.report_only:
        for task, valid, reason in inventory:
            if not valid:
                print(f"[pending] {task.task_id}: {reason}", flush=True)
        return 0
    gpu = validate_a100()
    print(
        f"[device] {gpu['name']} ({gpu['total_bytes'] / 1024**3:.2f} GiB), "
        "dtype=complex128",
        flush=True,
    )
    _check_space(scratch_root, required_bytes=3 * 1024**3, label="local scratch")
    _check_space(output_root, required_bytes=1024**3, label="Drive")
    if args.max_new_tasks is not None:
        pending = pending[: args.max_new_tasks]
    if not pending:
        print("[complete] no pending tasks selected", flush=True)
        return 0

    current_ny: int | None = None
    model: Any | None = None
    session_elapsed: list[float] = []
    failures = 0
    progress = tqdm(
        total=len(tasks),
        initial=completed,
        desc="Lyapunov campaign",
        unit="task",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )
    try:
        for task in pending:
            if task.ny != current_ny:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = build_model(config, task.ny)
                current_ny = task.ny
            try:
                payload, elapsed = run_task(model=model, config=config, task=task)
                save_task(
                    output_root=output_root,
                    scratch_root=scratch_root,
                    task=task,
                    payload=payload,
                    elapsed_seconds=elapsed,
                    config_sha256=config_sha256,
                    hashes=hashes,
                )
            except Exception:
                failures += 1
                progress.set_postfix(
                    completed=completed + len(session_elapsed),
                    pending=len(tasks) - completed - len(session_elapsed),
                    failed=failures,
                    refresh=True,
                )
                raise
            session_elapsed.append(elapsed)
            progress.update(1)
            remaining = len(tasks) - completed - len(session_elapsed)
            eta_hours = remaining * float(np.mean(session_elapsed)) / 3600.0
            progress.set_postfix(
                completed=completed + len(session_elapsed),
                pending=remaining,
                failed=failures,
                refresh=True,
            )
            print(
                f"[task complete] {task.task_id}: {elapsed / 60:.1f} min; "
                f"session-mean projected remaining={eta_hours:.1f} A100 h",
                flush=True,
            )
    finally:
        progress.close()
    final_completed = completed + len(session_elapsed)
    print(
        f"[completion summary] completed={final_completed}/{len(tasks)}, "
        f"new={len(session_elapsed)}, pending={len(tasks) - final_completed}, "
        f"failed={failures}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
