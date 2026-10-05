"""Acquire replay-complete pure trajectories without online tangent propagation."""

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
from typing import Any, Mapping

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BUNDLE_ROOT / "src"
for candidate in (BUNDLE_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from replay_record_observer import (  # noqa: E402
    CHANNEL_LABELS,
    OBSERVER_SCHEMA,
    ReplayRecordObserver,
    native_frame_arrays,
    unpack_boolean_record,
)


BUNDLE = "09_pure_tangent_replay_acquisition"
EXPECTED_REVISION = "pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1"
RESULT_SCHEMA = "pure_tangent_replay_acquisition_result_v1"
COMPLETION_SCHEMA = "pure_tangent_replay_acquisition_completion_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
EXPECTED_ROOT_SEED = 2026090409
EXPECTED_NX = 20
EXPECTED_NY = (24, 28, 32)
EXPECTED_ALPHA_1 = (1.0, 3.0)
EXPECTED_SAMPLES = 100
EXPECTED_BATCH_SIZE_BY_NY = {24: 50, 28: 50, 32: 25}
EXPECTED_TASK_COUNT = 32
EXPECTED_TRAJECTORY_COUNT = 1200
CONSTRUCTIONS = {
    "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
    "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
}
SOURCE_FILES = (
    "run_campaign.py",
    "replay_record_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)


@dataclass(frozen=True)
class Task:
    construction: str
    ny: int
    alpha_1: float
    batch_index: int
    sample_start: int
    sample_stop: int
    campaign_sample_start: int
    seed: int

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def cycles(self) -> int:
        return 2 * self.ny

    @property
    def case_sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))

    @property
    def global_sample_indices(self) -> tuple[int, ...]:
        return tuple(
            range(self.campaign_sample_start, self.campaign_sample_start + self.sample_count)
        )

    @property
    def alpha_label(self) -> str:
        return str(int(self.alpha_1))

    @property
    def task_id(self) -> str:
        return (
            f"{self.construction}_Ny{self.ny:03d}_a1-{self.alpha_label}_"
            f"batch-{self.batch_index:03d}_samples-{self.sample_start:03d}-"
            f"{self.sample_stop - 1:03d}"
        )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def source_hashes(bundle_root: Path = BUNDLE_ROOT) -> dict[str, str]:
    return {relative: sha256_file(bundle_root / relative) for relative in SOURCE_FILES}


def expected_config() -> dict[str, Any]:
    return {
        "sampling_revision": EXPECTED_REVISION,
        "root_seed": EXPECTED_ROOT_SEED,
        "Nx": EXPECTED_NX,
        "Ny_values": list(EXPECTED_NY),
        "alpha_1_values": list(EXPECTED_ALPHA_1),
        "alpha_2": 30.0,
        "nshell": 1,
        "samples_per_case": EXPECTED_SAMPLES,
        "batch_size_by_Ny": {str(key): value for key, value in EXPECTED_BATCH_SIZE_BY_NY.items()},
        "cycles_multiplier": 2,
        "trial_orbitals": "X",
        "filling_fraction": 0.5,
        "n_a": 0.5,
        "init_mode": "default",
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "dtype": "complex128",
        "device": "cuda:0",
        "backend": "local",
        "state_representation": "physical_frame",
        "triv_region_local_mode": False,
        "frame_reorthonormalize_interval": 1,
        "max_batch_runtime_seconds": 3600.0,
        "a100_peak_memory_limit_gib": 34.0,
        "constructions": CONSTRUCTIONS,
        "saved_products": {
            "prepared_initial_frame": True,
            "final_frame": True,
            "ordered_schedule": True,
            "bitpacked_outcomes": True,
            "bitpacked_targets": True,
            "cycle_log_probability": True,
            "intermediate_frames": False,
            "tangent_frame": False,
            "choi_covariance": False,
            "covariance_history": False,
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
            "configuration differs from the locked acquisition contract: "
            + json.dumps(differences, sort_keys=True)
        )
    return observed


def _task_seed(config: Mapping[str, Any], identity: str) -> int:
    raw = (
        f"{config['root_seed']}|{config['sampling_revision']}|{identity}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & ((1 << 63) - 1)


def expand_tasks(config: Mapping[str, Any]) -> list[Task]:
    config = validate_config(config)
    tasks: list[Task] = []
    case_index = 0
    for construction in ("hard", "soft"):
        for ny in EXPECTED_NY:
            for alpha_1 in EXPECTED_ALPHA_1:
                size = EXPECTED_BATCH_SIZE_BY_NY[ny]
                for batch_index, sample_start in enumerate(range(0, EXPECTED_SAMPLES, size)):
                    sample_stop = min(EXPECTED_SAMPLES, sample_start + size)
                    identity = (
                        f"{construction}|Ny={ny}|a1={alpha_1:g}|"
                        f"batch={batch_index}|samples={sample_start}:{sample_stop}"
                    )
                    tasks.append(
                        Task(
                            construction=construction,
                            ny=ny,
                            alpha_1=alpha_1,
                            batch_index=batch_index,
                            sample_start=sample_start,
                            sample_stop=sample_stop,
                            campaign_sample_start=case_index * EXPECTED_SAMPLES + sample_start,
                            seed=_task_seed(config, identity),
                        )
                    )
                case_index += 1
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(f"expected {EXPECTED_TASK_COUNT} tasks, found {len(tasks)}")
    if sum(task.sample_count for task in tasks) != EXPECTED_TRAJECTORY_COUNT:
        raise RuntimeError("task expansion did not produce 1,200 trajectories")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("task IDs are not unique")
    if len({task.seed for task in tasks}) != len(tasks):
        raise RuntimeError("task seeds are not unique")
    global_indices = [index for task in tasks for index in task.global_sample_indices]
    if sorted(global_indices) != list(range(EXPECTED_TRAJECTORY_COUNT)):
        raise RuntimeError("campaign-global sample indices are not an exact partition")
    return tasks


def result_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = (
        output_root
        / "results"
        / task.construction
        / f"Ny{task.ny:03d}"
        / f"alpha1_{task.alpha_label}"
    )
    stem = (
        f"batch_{task.batch_index:03d}_samples_"
        f"{task.sample_start:03d}-{task.sample_stop - 1:03d}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def _task_identity(
    *, task: Task, configuration_sha256: str, hashes: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "task_id": task.task_id,
        "construction": task.construction,
        "Nx": EXPECTED_NX,
        "Ny": task.ny,
        "alpha_1": task.alpha_1,
        "alpha_2": 30.0,
        "nshell": 1,
        "batch_index": task.batch_index,
        "case_sample_indices": list(task.case_sample_indices),
        "global_sample_indices": list(task.global_sample_indices),
        "seed": task.seed,
        "cycles": task.cycles,
        "configuration_sha256": configuration_sha256,
        "source_hashes": dict(hashes),
    }


def _completion_identity(
    *, task: Task, configuration_sha256: str, hashes: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        **_task_identity(
            task=task, configuration_sha256=configuration_sha256, hashes=hashes
        ),
    }


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


def _write_npz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
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
    """Copy, read back, and atomically publish one DriveFS file."""

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


def _expected_updates(task: Task) -> int:
    return (11 if task.construction == "hard" else EXPECTED_NX) * task.ny


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise OSError(f"no existing ancestor for storage path {path}")
    return candidate


def estimated_result_upper_bound(task: Task) -> int:
    """Conservative bytes for two full-capacity frames plus the compact record."""

    dimension = 2 * EXPECTED_NX * task.ny
    frame_bytes = 2 * task.sample_count * dimension * dimension * np.dtype(np.complex128).itemsize
    schedule_bytes = task.sample_count * task.cycles * _expected_updates(task) * np.dtype(np.int32).itemsize
    record_bytes = task.sample_count * task.cycles * _expected_updates(task)
    diagnostics_bytes = task.sample_count * (task.cycles + 1) * 32
    return int(frame_bytes + schedule_bytes + record_bytes + diagnostics_bytes + 64 * 1024**2)


def require_storage_headroom(*, task: Task, scratch_root: Path, output_root: Path) -> None:
    estimate = estimated_result_upper_bound(task)
    scratch_required = estimate + 2 * 1024**3
    # Allow one invalid stable result to coexist with the replacement temporary.
    drive_required = 2 * estimate + 2 * 1024**3
    scratch_free = shutil.disk_usage(_existing_ancestor(scratch_root)).free
    drive_free = shutil.disk_usage(_existing_ancestor(output_root)).free
    if scratch_free < scratch_required:
        raise OSError(
            f"insufficient local scratch: free={scratch_free / 1024**3:.2f} GiB, "
            f"required={scratch_required / 1024**3:.2f} GiB"
        )
    if drive_free < drive_required:
        raise OSError(
            f"insufficient DriveFS headroom: free={drive_free / 1024**3:.2f} GiB, "
            f"required={drive_required / 1024**3:.2f} GiB"
        )
    print(
        f"[storage] estimated result <= {estimate / 1024**3:.2f} GiB; "
        f"local free={scratch_free / 1024**3:.2f} GiB; "
        f"DriveFS free={drive_free / 1024**3:.2f} GiB",
        flush=True,
    )


def _validate_result_npz(
    path: Path,
    task: Task,
    *,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> None:
    required = {
        "schema",
        "observer_schema",
        "task_id",
        "construction",
        "Nx",
        "Ny",
        "alpha_1",
        "cycles_total",
        "case_sample_indices",
        "global_sample_indices",
        "batch_seed",
        "initial_frame",
        "initial_ranks",
        "final_frame",
        "final_ranks",
        "record_schedule",
        "record_outcomes_packed",
        "record_outcomes_shape",
        "record_targets_packed",
        "record_targets_shape",
        "measurement_log_probability",
        "cumulative_log_probability",
        "site_event_count",
        "channel_event_count",
        "channel_labels",
        "configuration_sha256",
        "source_hashes_json",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"result NPZ missing fields: {missing}")
        if str(archive["schema"].item()) != RESULT_SCHEMA:
            raise ValueError("result schema mismatch")
        if str(archive["observer_schema"].item()) != OBSERVER_SCHEMA:
            raise ValueError("observer schema mismatch")
        if str(archive["task_id"].item()) != task.task_id:
            raise ValueError("result task ID mismatch")
        if str(archive["construction"].item()) != task.construction:
            raise ValueError("result construction mismatch")
        if int(archive["Nx"].item()) != EXPECTED_NX or int(archive["Ny"].item()) != task.ny:
            raise ValueError("result geometry mismatch")
        if float(archive["alpha_1"].item()) != task.alpha_1:
            raise ValueError("result alpha_1 mismatch")
        if int(archive["cycles_total"].item()) != task.cycles:
            raise ValueError("result cycle count mismatch")
        if str(archive["configuration_sha256"].item()) != configuration_sha256:
            raise ValueError("result configuration hash mismatch")
        if json.loads(str(archive["source_hashes_json"].item())) != dict(hashes):
            raise ValueError("result source hash mismatch")
        if not np.array_equal(
            archive["case_sample_indices"], np.asarray(task.case_sample_indices, dtype=np.int64)
        ):
            raise ValueError("result case sample indices mismatch")
        if not np.array_equal(
            archive["global_sample_indices"], np.asarray(task.global_sample_indices, dtype=np.int64)
        ):
            raise ValueError("result global sample indices mismatch")
        if int(archive["batch_seed"].item()) != task.seed:
            raise ValueError("result batch seed mismatch")

        dimension = 2 * EXPECTED_NX * task.ny
        initial_frame = archive["initial_frame"]
        final_frame = archive["final_frame"]
        initial_ranks = archive["initial_ranks"]
        final_ranks = archive["final_ranks"]
        for label, frame, ranks in (
            ("initial", initial_frame, initial_ranks),
            ("final", final_frame, final_ranks),
        ):
            if frame.dtype != np.complex128 or frame.ndim != 3:
                raise ValueError(f"{label} frame dtype/shape mismatch")
            if frame.shape[:2] != (task.sample_count, dimension):
                raise ValueError(f"{label} frame geometry mismatch")
            if ranks.dtype != np.int64 or ranks.shape != (task.sample_count,):
                raise ValueError(f"{label} rank dtype/shape mismatch")
            if np.any(ranks < 0) or np.any(ranks > frame.shape[2]):
                raise ValueError(f"{label} ranks are outside frame capacity")
            if not np.isfinite(frame).all():
                raise FloatingPointError(f"{label} frame contains nonfinite values")

        record_shape = (task.sample_count, task.cycles, _expected_updates(task))
        schedule = archive["record_schedule"]
        if schedule.dtype != np.int32 or schedule.shape != record_shape:
            raise ValueError("ordered schedule dtype/shape mismatch")
        if np.any(schedule < 0) or np.any(schedule >= EXPECTED_NX * task.ny):
            raise ValueError("ordered schedule contains invalid site IDs")
        outcomes = unpack_boolean_record(
            archive["record_outcomes_packed"], archive["record_outcomes_shape"]
        )
        targets = unpack_boolean_record(
            archive["record_targets_packed"], archive["record_targets_shape"]
        )
        if outcomes.shape != record_shape + (len(CHANNEL_LABELS),):
            raise ValueError("outcome record logical shape mismatch")
        if targets.shape != outcomes.shape:
            raise ValueError("target record logical shape mismatch")
        if tuple(str(value) for value in archive["channel_labels"].tolist()) != CHANNEL_LABELS:
            raise ValueError("record channel labels mismatch")
        for field in ("measurement_log_probability", "cumulative_log_probability"):
            values = archive[field]
            if values.shape != (task.sample_count, task.cycles + 1) or not np.isfinite(values).all():
                raise ValueError(f"{field} is incomplete")
        if not np.allclose(
            archive["cumulative_log_probability"],
            np.cumsum(archive["measurement_log_probability"], axis=1),
            rtol=1e-13,
            atol=1e-13,
        ):
            raise ValueError("cumulative record probability does not match increments")
        if np.any(archive["site_event_count"][:, 1:] != _expected_updates(task)):
            raise ValueError("site-event counts are incomplete")
        if np.any(
            archive["channel_event_count"][:, 1:]
            != len(CHANNEL_LABELS) * _expected_updates(task)
        ):
            raise ValueError("channel-event counts are incomplete")


def verified_complete(
    *,
    output_root: Path,
    task: Task,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = result_paths(output_root, task)
    result_exists, completion_exists = result_path.is_file(), completion_path.is_file()
    if not result_exists and not completion_exists:
        return False, "missing result and completion"
    if not result_exists or not completion_exists:
        return False, "incomplete result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion JSON: {exc}"
    for key, expected in _completion_identity(
        task=task, configuration_sha256=configuration_sha256, hashes=hashes
    ).items():
        if completion.get(key) != expected:
            return False, f"completion identity mismatch: {key}"
    if completion.get("result_filename") != result_path.name:
        return False, "completion result filename mismatch"
    try:
        if int(completion.get("result_bytes", -1)) != int(result_path.stat().st_size):
            return False, "result byte-count mismatch"
        if completion.get("result_sha256") != sha256_file(result_path):
            return False, "result checksum mismatch"
        _validate_result_npz(
            result_path,
            task,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
    except (OSError, ValueError, KeyError, FloatingPointError) as exc:
        return False, f"result payload invalid: {exc}"
    return True, "verified"


def _seed_rng(seed: int) -> None:
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_model(config: Mapping[str, Any], task: Task) -> classA_U1FGTN_gpu:
    flags = CONSTRUCTIONS[task.construction]
    model = classA_U1FGTN_gpu(
        Nx=EXPECTED_NX,
        Ny=task.ny,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=task.alpha_1,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=bool(flags["dw_truncation"]),
        triv_region_local_mode=False,
        device=str(config["device"]),
        dtype=str(config["dtype"]),
        backend="local",
    )
    if tuple(int(value) for value in model.DW_loc) != (5, 15):
        raise RuntimeError(f"unexpected domain-wall locations {model.DW_loc}")
    if model.dtype != torch.complex128:
        raise RuntimeError("constructed model is not complex128")
    return model


def run_task(
    *, model: classA_U1FGTN_gpu, config: Mapping[str, Any], task: Task
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _seed_rng(task.seed)
    observer = ReplayRecordObserver(
        samples=task.sample_count,
        cycles=task.cycles,
        updates_per_cycle=_expected_updates(task),
    )
    flags = CONSTRUCTIONS[task.construction]
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    print(
        f"[task start] {task.task_id}: trajectories={task.sample_count}, "
        f"cycles={task.cycles}, seed={task.seed}",
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
        init_mode="default",
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=bool(flags["meas_slab_only"]),
        batch_size=task.sample_count,
        return_data=True,
        state_representation="physical_frame",
        native_cycle_observer=observer.capture_native_cycle,
        record_observer=observer.record_event,
        track_choi=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.monotonic() - started
    if int(result.get("samples", -1)) != task.sample_count:
        raise RuntimeError("canonical engine returned the wrong sample count")
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("canonical engine did not use physical-frame state")
    if bool(result.get("choi_tracked", False)) or bool(result.get("lyapunov_tracked", False)):
        raise RuntimeError("canonical engine unexpectedly enabled Choi/tangent tracking")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("canonical engine materialized a covariance")
    expected_exterior = task.construction == "hard"
    if bool(result.get("exterior_preparation_performed", False)) != expected_exterior:
        raise RuntimeError("canonical engine exterior-preparation metadata mismatch")

    final_frame, final_ranks = native_frame_arrays(result["native_final"])
    payload = observer.result_arrays()
    payload.update(
        {
            "schema": np.asarray(RESULT_SCHEMA),
            "observer_schema": np.asarray(OBSERVER_SCHEMA),
            "bundle": np.asarray(BUNDLE),
            "sampling_revision": np.asarray(EXPECTED_REVISION),
            "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
            "task_id": np.asarray(task.task_id),
            "construction": np.asarray(task.construction),
            "Nx": np.asarray(EXPECTED_NX, dtype=np.int64),
            "Ny": np.asarray(task.ny, dtype=np.int64),
            "alpha_1": np.asarray(task.alpha_1, dtype=np.float64),
            "alpha_2": np.asarray(30.0, dtype=np.float64),
            "nshell": np.asarray(1, dtype=np.int64),
            "cycles_total": np.asarray(task.cycles, dtype=np.int64),
            "case_sample_indices": np.asarray(task.case_sample_indices, dtype=np.int64),
            "global_sample_indices": np.asarray(task.global_sample_indices, dtype=np.int64),
            "batch_seed": np.asarray(task.seed, dtype=np.int64),
            "final_frame": final_frame,
            "final_ranks": final_ranks,
            "prepared_initial_state": np.asarray(
                "post_exterior_product_frame" if expected_exterior else "pure_half_filled_frame"
            ),
            "replay_frame_init_prepared": np.asarray(expected_exterior, dtype=np.bool_),
            "dtype": np.asarray("complex128"),
            "sequence": np.asarray("raster_y"),
            "configuration_sha256": np.asarray(config_sha256(config)),
            "source_hashes_json": np.asarray(_canonical_json(source_hashes())),
        }
    )
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(model.device))
        if model.device.type == "cuda"
        else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(model.device))
        if model.device.type == "cuda"
        else 0
    )
    performance = {
        "elapsed_seconds": float(elapsed),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }
    print(
        f"[task computed] {task.task_id}: elapsed={elapsed / 60:.2f} min, "
        f"peak_reserved={peak_reserved / 1024**3:.2f} GiB",
        flush=True,
    )
    return payload, performance


def save_task(
    *,
    output_root: Path,
    scratch_root: Path,
    task: Task,
    payload: Mapping[str, Any],
    performance: Mapping[str, Any],
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> bool:
    local_dir = scratch_root / task.task_id
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True)
    local_result = local_dir / "result.npz"
    _write_npz(local_result, payload)
    _validate_result_npz(
        local_result,
        task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    final_result, final_completion = result_paths(output_root, task)
    published_result = publish_file(local_result, final_result)
    runtime_ok = float(performance["elapsed_seconds"]) <= 3600.0
    memory_ok = int(performance["peak_cuda_reserved_bytes"]) <= int(34.0 * 1024**3)
    gate_passed = bool(runtime_ok and memory_ok)
    completion = _completion_identity(
        task=task, configuration_sha256=configuration_sha256, hashes=hashes
    )
    completion.update(
        {
            "result_filename": final_result.name,
            "result_bytes": published_result["bytes"],
            "result_sha256": published_result["sha256"],
            **dict(performance),
            "performance_gate_passed": gate_passed,
            "max_batch_runtime_seconds": 3600.0,
            "a100_peak_memory_limit_bytes": int(34.0 * 1024**3),
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_completion = local_dir / "completion.json"
    _write_json(local_completion, completion)
    publish_file(local_completion, final_completion)
    shutil.rmtree(local_dir)
    print(
        f"[task durable] {task.task_id}: {published_result['bytes'] / 1024**3:.2f} GiB, "
        f"sha256={published_result['sha256'][:16]}...",
        flush=True,
    )
    return gate_passed


def require_a100(config: Mapping[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an A100 GPU runtime")
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name.upper():
        raise RuntimeError(f"production requires an A100, found {properties.name!r}")
    if int(properties.total_memory) < 38 * 1024**3:
        raise RuntimeError(
            f"production requires 40-GB-class memory, found {properties.total_memory / 1024**3:.2f} GiB"
        )
    if str(config["dtype"]) != "complex128":
        raise RuntimeError("production dtype must remain complex128")
    print(
        f"[device] {properties.name}, total={properties.total_memory / 1024**3:.2f} GiB, "
        "dtype=complex128",
        flush=True,
    )


def _load_config(path: Path) -> dict[str, Any]:
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/content/tangent_replay_acquisition"))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-tasks", type=int)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    tasks = expand_tasks(config)
    hashes = source_hashes()
    configuration_sha256 = config_sha256(config)
    output_root = args.output_root.resolve()
    scratch_root = args.scratch_root.resolve()
    print("[configuration] " + json.dumps(config, indent=2, sort_keys=True), flush=True)
    print(f"[bundle] {BUNDLE_ROOT}", flush=True)
    print(f"[output] {output_root}", flush=True)
    print(f"[scratch] {scratch_root}", flush=True)
    print(
        f"[workload] cases=12, batches={len(tasks)}, trajectories="
        f"{sum(task.sample_count for task in tasks)}, batch_sizes={EXPECTED_BATCH_SIZE_BY_NY}",
        flush=True,
    )
    print(f"[sources] {json.dumps(hashes, sort_keys=True)}", flush=True)

    inventory: list[tuple[Task, bool, str]] = []
    performance_blocks: list[str] = []
    for task in tasks:
        complete, reason = verified_complete(
            output_root=output_root,
            task=task,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
        inventory.append((task, complete, reason))
        if complete:
            _, completion_path = result_paths(output_root, task)
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("performance_gate_passed") is not True:
                performance_blocks.append(task.task_id)
    complete_count = sum(complete for _, complete, _ in inventory)
    print(
        f"[resume] completed={complete_count}, pending={len(tasks) - complete_count}, total={len(tasks)}",
        flush=True,
    )
    for task, complete, reason in inventory:
        if not complete and reason != "missing result and completion":
            print(f"[resume warning] {task.task_id}: {reason}", flush=True)
    if args.report_only:
        return 0
    if performance_blocks:
        raise RuntimeError(
            "a completed calibration batch exceeded the locked runtime/memory ceiling; "
            "revise batching under a new campaign revision before continuing: "
            + ", ".join(performance_blocks)
        )
    require_a100(config)
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        raise ValueError("--max-new-tasks must be nonnegative")

    pending = [task for task, complete, _ in inventory if not complete]
    new_limit = len(pending) if args.max_new_tasks is None else int(args.max_new_tasks)
    new_count = 0
    skipped_count = complete_count
    with tqdm(total=len(tasks), initial=complete_count, desc="replay acquisition", unit="batch") as bar:
        for task in pending:
            if new_count >= new_limit:
                break
            require_storage_headroom(
                task=task, scratch_root=scratch_root, output_root=output_root
            )
            model = build_model(config, task)
            payload, performance = run_task(model=model, config=config, task=task)
            gate_passed = save_task(
                output_root=output_root,
                scratch_root=scratch_root,
                task=task,
                payload=payload,
                performance=performance,
                configuration_sha256=configuration_sha256,
                hashes=hashes,
            )
            del payload, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_count += 1
            bar.update(1)
            bar.set_postfix(completed=complete_count + new_count, pending=len(tasks) - complete_count - new_count)
            if not gate_passed:
                raise RuntimeError(
                    f"{task.task_id} is durable but exceeded the one-hour or 34-GiB performance gate; "
                    "the queue stopped before another task"
                )

    remaining = len(tasks) - skipped_count - new_count
    print(
        f"[summary] verified_before={skipped_count}, newly_completed={new_count}, "
        f"remaining={remaining}, total={len(tasks)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
