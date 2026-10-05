"""Run the S100 hard/soft domain-wall correlator campaign on an A100."""

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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from correlator_observer import (  # noqa: E402
    OBSERVER_SCHEMA,
    DomainWallCorrelatorObserver,
)


BUNDLE = "08_domain_wall_correlator_scaling"
EXPECTED_REVISION = (
    "domain_wall_correlator_nx20_ny24-32_a1-1-3_"
    "nsh1-2-dense_s100_2ny_raster_v1"
)
RESULT_SCHEMA = "domain_wall_correlator_result_v1"
COMPLETION_SCHEMA = "domain_wall_correlator_completion_v1"
CHECKPOINT_SCHEMA = "domain_wall_correlator_checkpoint_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
EXPECTED_NX = 20
EXPECTED_NY = (24, 28, 32)
EXPECTED_ALPHA_1 = (1.0, 3.0)
EXPECTED_NSHELL = (1, 2, None)
EXPECTED_SAMPLES = 100
EXPECTED_BATCH_SIZE = 25
EXPECTED_SEGMENT_CYCLES = 16
EXPECTED_ROOT_SEED = 2026090408
CONSTRUCTIONS = {
    "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
    "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
}


@dataclass(frozen=True)
class Task:
    construction: str
    ny: int
    alpha_1: float
    nshell: int | None
    batch_index: int
    sample_start: int
    sample_stop: int
    seed: int

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def cycles(self) -> int:
        return 2 * self.ny

    @property
    def global_sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))

    @property
    def shell_label(self) -> str:
        return "dense" if self.nshell is None else str(self.nshell)

    @property
    def alpha_label(self) -> str:
        return str(int(self.alpha_1))

    @property
    def task_id(self) -> str:
        return (
            f"{self.construction}_Ny{self.ny:03d}_a1-{self.alpha_label}_"
            f"nsh-{self.shell_label}_batch-{self.batch_index:03d}_"
            f"samples-{self.sample_start:03d}-{self.sample_stop - 1:03d}"
        )


@dataclass(frozen=True)
class Checkpoint:
    completed_cycle: int
    elapsed_seconds: float
    frame: np.ndarray
    ranks: np.ndarray
    observer_payload: dict[str, np.ndarray]
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


def source_hashes() -> dict[str, str]:
    relatives = (
        "run_campaign.py",
        "correlator_observer.py",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    )
    return {relative: sha256_file(BUNDLE_ROOT / relative) for relative in relatives}


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
    """Publish through DriveFS only after temporary and final readback."""

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


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(config)
    expected = {
        "sampling_revision": EXPECTED_REVISION,
        "root_seed": EXPECTED_ROOT_SEED,
        "Nx": EXPECTED_NX,
        "Ny_values": list(EXPECTED_NY),
        "alpha_1_values": list(EXPECTED_ALPHA_1),
        "alpha_2": 30.0,
        "nshell_values": [1, 2, None],
        "samples_per_case": EXPECTED_SAMPLES,
        "batch_size": EXPECTED_BATCH_SIZE,
        "cycles_multiplier": 2,
        "segment_cycles": EXPECTED_SEGMENT_CYCLES,
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
        "backend_by_nshell": {"1": "local", "2": "local", "dense": "dense"},
        "state_representation": "physical_frame",
        "triv_region_local_mode": False,
        "frame_reorthonormalize_interval": 1,
        "constructions": CONSTRUCTIONS,
        "observables": {
            "cycles": "0..2Ny inclusive",
            "x_resolved_square_correlator": True,
            "legacy_xavg_square_correlator": True,
            "global_charge": True,
        },
    }
    if config != expected:
        differences = {
            key: {"expected": value, "observed": config.get(key)}
            for key, value in expected.items()
            if config.get(key) != value
        }
        differences.update(
            {
                key: {"expected": "absent", "observed": value}
                for key, value in config.items()
                if key not in expected
            }
        )
        raise ValueError(
            "configuration differs from the locked production contract: "
            + json.dumps(differences, sort_keys=True)
        )
    return config


def _task_seed(config: Mapping[str, Any], identity: str) -> int:
    digest = hashlib.sha256(
        f"{config['root_seed']}|{config['sampling_revision']}|{identity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def expand_tasks(config: Mapping[str, Any], construction: str) -> list[Task]:
    config = validate_config(config)
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"unknown construction {construction!r}")
    tasks: list[Task] = []
    for ny in EXPECTED_NY:
        for alpha_1 in EXPECTED_ALPHA_1:
            for nshell in EXPECTED_NSHELL:
                for batch_index, sample_start in enumerate(
                    range(0, EXPECTED_SAMPLES, EXPECTED_BATCH_SIZE)
                ):
                    sample_stop = min(
                        EXPECTED_SAMPLES, sample_start + EXPECTED_BATCH_SIZE
                    )
                    shell = "dense" if nshell is None else str(nshell)
                    identity = (
                        f"{construction}|{ny}|{alpha_1:g}|{shell}|"
                        f"{batch_index}|{sample_start}|{sample_stop}"
                    )
                    tasks.append(
                        Task(
                            construction=construction,
                            ny=ny,
                            alpha_1=alpha_1,
                            nshell=nshell,
                            batch_index=batch_index,
                            sample_start=sample_start,
                            sample_stop=sample_stop,
                            seed=_task_seed(config, identity),
                        )
                    )
    if len(tasks) != 72 or len({task.task_id for task in tasks}) != 72:
        raise RuntimeError("lane task expansion did not produce 72 unique tasks")
    if len({task.seed for task in tasks}) != len(tasks):
        raise RuntimeError("lane task seeds are not unique")
    return tasks


def result_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = (
        output_root
        / "results"
        / task.construction
        / f"Ny{task.ny:03d}"
        / f"alpha1_{task.alpha_label}"
        / f"nshell_{task.shell_label}"
    )
    stem = (
        f"batch_{task.batch_index:03d}_samples_"
        f"{task.sample_start:03d}-{task.sample_stop - 1:03d}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def checkpoint_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = output_root / "checkpoints" / task.task_id
    return directory / "checkpoint.npz", directory / "checkpoint.json"


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
        "nshell": task.nshell,
        "batch_index": task.batch_index,
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


def _checkpoint_identity(
    *, task: Task, configuration_sha256: str, hashes: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": "checkpoint",
        "segment_cycles": EXPECTED_SEGMENT_CYCLES,
        **_task_identity(
            task=task, configuration_sha256=configuration_sha256, hashes=hashes
        ),
    }


def _validate_result_npz(path: Path, task: Task) -> None:
    expected_shape = (
        task.sample_count,
        task.cycles + 1,
        EXPECTED_NX,
        task.ny // 2 + 1,
    )
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema",
            "observer_schema",
            "task_id",
            "construction",
            "Nx",
            "Ny",
            "alpha_1",
            "alpha_2",
            "nshell",
            "cycles",
            "normalized_cycles",
            "ry_values",
            "x_values",
            "global_sample_indices",
            "x_resolved_square_correlator",
            "xavg_square_correlator_vs_ry",
            "global_charge",
        }
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
        if int(archive["Nx"].item()) != EXPECTED_NX or int(
            archive["Ny"].item()
        ) != task.ny:
            raise ValueError("result geometry mismatch")
        if float(archive["alpha_1"].item()) != task.alpha_1:
            raise ValueError("result alpha_1 mismatch")
        if float(archive["alpha_2"].item()) != 30.0:
            raise ValueError("result alpha_2 mismatch")
        expected_shell = -1 if task.nshell is None else task.nshell
        if int(archive["nshell"].item()) != expected_shell:
            raise ValueError("result nshell mismatch")
        expected_cycles = np.arange(task.cycles + 1, dtype=np.int64)
        if not np.array_equal(archive["cycles"], expected_cycles):
            raise ValueError("result cycle labels mismatch")
        if not np.array_equal(
            archive["normalized_cycles"], expected_cycles / float(task.ny)
        ):
            raise ValueError("result normalized-cycle labels mismatch")
        if not np.array_equal(
            archive["ry_values"], np.arange(task.ny // 2 + 1, dtype=np.int64)
        ):
            raise ValueError("result y-separation labels mismatch")
        if not np.array_equal(
            archive["x_values"], np.arange(EXPECTED_NX, dtype=np.int64)
        ):
            raise ValueError("result x labels mismatch")
        if not np.array_equal(
            archive["global_sample_indices"],
            np.asarray(task.global_sample_indices, dtype=np.int64),
        ):
            raise ValueError("result sample indices mismatch")
        x_resolved = archive["x_resolved_square_correlator"]
        xavg = archive["xavg_square_correlator_vs_ry"]
        charge = archive["global_charge"]
        if x_resolved.shape != expected_shape or x_resolved.dtype != np.float64:
            raise ValueError("result x-resolved correlator shape/dtype mismatch")
        if xavg.shape != expected_shape[:2] + (expected_shape[3],):
            raise ValueError("result x-averaged correlator shape mismatch")
        if charge.shape != expected_shape[:2] or charge.dtype != np.int64:
            raise ValueError("result total-charge shape/dtype mismatch")
        if not np.isfinite(x_resolved).all() or not np.isfinite(xavg).all():
            raise FloatingPointError("result correlator contains nonfinite values")
        if not np.allclose(
            xavg, x_resolved.mean(axis=2), rtol=2.0e-13, atol=2.0e-13
        ):
            raise ValueError("result x average does not match x-resolved correlator")


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
        actual_bytes = int(result_path.stat().st_size)
        actual_sha256 = sha256_file(result_path)
    except OSError as exc:
        return False, f"result readback failed: {exc}"
    if int(completion.get("result_bytes", -1)) != actual_bytes:
        return False, "result byte count mismatch"
    if completion.get("result_sha256") != actual_sha256:
        return False, "result checksum mismatch"
    try:
        _validate_result_npz(result_path, task)
    except (OSError, ValueError, KeyError, FloatingPointError) as exc:
        return False, f"result payload invalid: {exc}"
    return True, "verified"


def _capture_rng_state() -> dict[str, np.ndarray]:
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
    task: Task,
    completed_cycle: int,
    elapsed_seconds: float,
    native: Mapping[str, Any],
    observer: DomainWallCorrelatorObserver,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> dict[str, np.ndarray]:
    frame, ranks = _native_arrays(native)
    expected_prefix = (task.sample_count, 2 * EXPECTED_NX * task.ny)
    if frame.ndim != 3 or frame.shape[:2] != expected_prefix:
        raise ValueError(f"unexpected checkpoint frame shape {frame.shape}")
    if ranks.shape != (task.sample_count,) or np.any(ranks > frame.shape[2]):
        raise ValueError("unexpected checkpoint ranks")
    rng_payload = _capture_rng_state()
    payload: dict[str, Any] = {
        "completed_cycle": np.asarray(completed_cycle, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "frame": frame,
        "ranks": ranks,
        "sample_indices": np.asarray(task.global_sample_indices, dtype=np.int64),
    }
    payload.update(
        {f"observer__{key}": value for key, value in observer.checkpoint_payload().items()}
    )
    payload.update({f"rng__{key}": value for key, value in rng_payload.items()})

    task_scratch = scratch_root / task.task_id
    task_scratch.mkdir(parents=True, exist_ok=True)
    local_npz = task_scratch / "checkpoint.npz"
    _write_npz(local_npz, payload, compressed=False)
    final_npz, final_json = checkpoint_paths(output_root, task)
    published = publish_file(local_npz, final_npz)
    metadata = _checkpoint_identity(
        task=task, configuration_sha256=configuration_sha256, hashes=hashes
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
    task: Task,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> tuple[Checkpoint | None, str]:
    npz_path, json_path = checkpoint_paths(output_root, task)
    npz_exists, json_exists = npz_path.is_file(), json_path.is_file()
    if not npz_exists and not json_exists:
        return None, "no checkpoint"
    if not npz_exists or not json_exists:
        return None, "incomplete checkpoint pair; restarting task"
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable checkpoint JSON ({exc}); restarting task"
    for key, expected in _checkpoint_identity(
        task=task, configuration_sha256=configuration_sha256, hashes=hashes
    ).items():
        if metadata.get(key) != expected:
            return None, f"checkpoint identity mismatch ({key}); restarting task"
    if metadata.get("checkpoint_filename") != npz_path.name:
        return None, "checkpoint filename mismatch; restarting task"
    try:
        if int(metadata.get("checkpoint_bytes", -1)) != int(npz_path.stat().st_size):
            return None, "checkpoint byte-count mismatch; restarting task"
        if metadata.get("checkpoint_sha256") != sha256_file(npz_path):
            return None, "checkpoint checksum mismatch; restarting task"
        with np.load(npz_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        return None, f"unreadable checkpoint NPZ ({exc}); restarting task"
    required = {"completed_cycle", "elapsed_seconds", "frame", "ranks", "sample_indices"}
    if not required.issubset(payload):
        return None, "checkpoint NPZ is incomplete; restarting task"
    completed_cycle = int(payload["completed_cycle"].item())
    if completed_cycle != int(metadata.get("completed_cycle", -1)):
        return None, "checkpoint cycle mismatch; restarting task"
    if not 0 < completed_cycle <= task.cycles:
        return None, "checkpoint cycle out of range; restarting task"
    if completed_cycle != task.cycles and completed_cycle % EXPECTED_SEGMENT_CYCLES:
        return None, "checkpoint cycle is not a segment boundary; restarting task"
    frame = np.asarray(payload["frame"])
    ranks = np.asarray(payload["ranks"])
    if frame.dtype != np.complex128 or frame.ndim != 3:
        return None, "checkpoint frame shape/dtype mismatch; restarting task"
    if frame.shape[:2] != (task.sample_count, 2 * EXPECTED_NX * task.ny):
        return None, "checkpoint frame geometry mismatch; restarting task"
    if ranks.dtype != np.int64 or ranks.shape != (task.sample_count,):
        return None, "checkpoint rank shape/dtype mismatch; restarting task"
    if np.any(ranks < 0) or np.any(ranks > frame.shape[2]):
        return None, "checkpoint ranks out of range; restarting task"
    if not np.array_equal(
        payload["sample_indices"], np.asarray(task.global_sample_indices, dtype=np.int64)
    ):
        return None, "checkpoint sample-index mismatch; restarting task"
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
        return None, "checkpoint lacks observer or RNG state; restarting task"
    return (
        Checkpoint(
            completed_cycle=completed_cycle,
            elapsed_seconds=float(payload["elapsed_seconds"].item()),
            frame=frame,
            ranks=ranks,
            observer_payload=observer_payload,
            rng_payload=rng_payload,
        ),
        "verified",
    )


def remove_checkpoint(output_root: Path, task: Task) -> None:
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


def build_model(config: Mapping[str, Any], task: Task) -> Any:
    construction = CONSTRUCTIONS[task.construction]
    backend_key = "dense" if task.nshell is None else str(task.nshell)
    backend = config["backend_by_nshell"][backend_key]
    model = classA_U1FGTN_gpu(
        Nx=EXPECTED_NX,
        Ny=task.ny,
        DW=True,
        nshell=task.nshell,
        filling_frac=config["filling_fraction"],
        alpha_1=task.alpha_1,
        alpha_2=config["alpha_2"],
        trial_orbitals=config["trial_orbitals"],
        dw_truncation=construction["dw_truncation"],
        triv_region_local_mode=config["triv_region_local_mode"],
        device=config["device"],
        dtype=config["dtype"],
        backend=backend,
    )
    if not model.DW or model.dtype != torch.complex128:
        raise RuntimeError("constructed model violates DW-on/complex128 contract")
    if tuple(int(value) for value in model.DW_loc) != (5, 15):
        raise RuntimeError(f"unexpected domain-wall positions: {model.DW_loc}")
    return model


def _run_segment(
    *,
    config: Mapping[str, Any],
    model: Any,
    task: Task,
    observer: DomainWallCorrelatorObserver,
    segment_start: int,
    frame: np.ndarray | None,
    ranks: np.ndarray | None,
    cycle_bar: tqdm,
) -> tuple[Mapping[str, Any], float]:
    segment_cycles = min(EXPECTED_SEGMENT_CYCLES, task.cycles - segment_start)
    continuing = frame is not None
    if continuing != (ranks is not None):
        raise ValueError("frame and ranks must either both be supplied or both be absent")

    def observe_global_cycle(*, cycle: int, **payload: Any) -> None:
        local_cycle = int(cycle)
        if continuing and local_cycle == 0:
            return
        observer(cycle=segment_start + local_cycle, **payload)
        if local_cycle > 0:
            cycle_bar.update(1)

    construction = CONSTRUCTIONS[task.construction]
    started = time.monotonic()
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=segment_cycles,
        postselect=config["postselect"],
        postselect_probability=config["postselect_probability"],
        perfect_correction=config["perfect_correction"],
        samples=task.sample_count,
        init_mode=config["init_mode"],
        frame_init=frame,
        frame_ranks=ranks,
        frame_init_prepared=continuing,
        save=False,
        n_a=config["n_a"],
        sequence=config["sequence"],
        meas_slab_only=construction["meas_slab_only"],
        batch_size=task.sample_count,
        return_data=True,
        state_representation=config["state_representation"],
        native_cycle_observer=observe_global_cycle,
        track_choi=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=config["frame_reorthonormalize_interval"],
    )
    if torch.device(model.device).type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.monotonic() - started
    if int(result.get("samples", -1)) != task.sample_count:
        raise RuntimeError("canonical engine returned the wrong sample count")
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("canonical engine did not use occupied-frame evolution")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("canonical engine unexpectedly materialized a covariance")
    if bool(result.get("choi_tracked", False)):
        raise RuntimeError("canonical engine unexpectedly enabled Choi tracking")
    if bool(result.get("frame_init_prepared", False)) != continuing:
        raise RuntimeError("canonical engine continuation metadata mismatch")
    if task.construction == "hard":
        expected = "skipped_prepared_frame" if continuing else "born_conditioned_onsite_before_cycle_0"
        expected_mode = "already_prepared" if continuing else "born_conditioned"
        if bool(result.get("exterior_preparation_performed", False)) != (not continuing):
            raise RuntimeError("hard-wall exterior preparation flag mismatch")
        if result.get("exterior_preparation") != expected:
            raise RuntimeError(
                "hard-wall exterior preparation mismatch: "
                f"{result.get('exterior_preparation')!r}"
            )
        if result.get("exterior_preparation_mode") != expected_mode:
            raise RuntimeError("hard-wall exterior preparation mode mismatch")
    elif (
        bool(result.get("exterior_preparation_performed", False))
        or result.get("exterior_preparation") is not None
        or result.get("exterior_preparation_mode") is not None
    ):
        raise RuntimeError("soft-wall run unexpectedly prepared an exterior")
    native = result.get("native_final")
    if not isinstance(native, Mapping) or not {"frame", "ranks"}.issubset(native):
        raise RuntimeError("canonical engine did not return a native final state")
    return native, elapsed


def _result_payload(
    *, task: Task, observer: DomainWallCorrelatorObserver, elapsed_seconds: float
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(observer.result_payload())
    payload.update(
        {
            "schema": np.asarray(RESULT_SCHEMA),
            "bundle": np.asarray(BUNDLE),
            "sampling_revision": np.asarray(EXPECTED_REVISION),
            "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
            "task_id": np.asarray(task.task_id),
            "construction": np.asarray(task.construction),
            "Nx": np.asarray(EXPECTED_NX, dtype=np.int64),
            "Ny": np.asarray(task.ny, dtype=np.int64),
            "alpha_1": np.asarray(task.alpha_1, dtype=np.float64),
            "alpha_2": np.asarray(30.0, dtype=np.float64),
            "nshell": np.asarray(
                -1 if task.nshell is None else task.nshell, dtype=np.int64
            ),
            "batch_index": np.asarray(task.batch_index, dtype=np.int64),
            "sample_start": np.asarray(task.sample_start, dtype=np.int64),
            "sample_stop": np.asarray(task.sample_stop, dtype=np.int64),
            "batch_seed": np.asarray(task.seed, dtype=np.int64),
            "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
            "dw_location": np.asarray([5, 15], dtype=np.int64),
            "dw_truncation": np.asarray(task.construction == "hard"),
            "meas_slab_only": np.asarray(task.construction == "hard"),
            "dtype": np.asarray("complex128"),
            "init_mode": np.asarray("default"),
            "sequence": np.asarray("raster_y"),
            "perfect_correction": np.asarray(True),
            "state_representation": np.asarray("physical_frame"),
        }
    )
    return payload


def save_result(
    *,
    output_root: Path,
    scratch_root: Path,
    task: Task,
    observer: DomainWallCorrelatorObserver,
    elapsed_seconds: float,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> None:
    task_scratch = scratch_root / task.task_id
    task_scratch.mkdir(parents=True, exist_ok=True)
    local_result = task_scratch / "result.npz"
    _write_npz(
        local_result,
        _result_payload(
            task=task, observer=observer, elapsed_seconds=elapsed_seconds
        ),
        compressed=True,
    )
    result_path, completion_path = result_paths(output_root, task)
    published = publish_file(local_result, result_path)
    completion = _completion_identity(
        task=task, configuration_sha256=configuration_sha256, hashes=hashes
    )
    completion.update(
        {
            "observer_schema": OBSERVER_SCHEMA,
            "result_filename": result_path.name,
            "result_bytes": published["bytes"],
            "result_sha256": published["sha256"],
            "elapsed_seconds": float(elapsed_seconds),
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_completion = task_scratch / "result.complete.json"
    _write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    valid, reason = verified_complete(
        output_root=output_root,
        task=task,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    if not valid:
        raise OSError(f"published result failed final verification: {reason}")


def execute_task(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    scratch_root: Path,
    task: Task,
    configuration_sha256: str,
    hashes: Mapping[str, str],
) -> float:
    task_scratch = scratch_root / task.task_id
    if task_scratch.exists():
        shutil.rmtree(task_scratch)
    task_scratch.mkdir(parents=True)
    model = build_model(config, task)
    observer = DomainWallCorrelatorObserver(
        nx=EXPECTED_NX,
        ny=task.ny,
        physical_cycles=task.cycles,
        sample_ids=np.asarray(task.global_sample_indices, dtype=np.int64),
    )
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
        frame = ranks = None
        rng_payload = None
        _seed_rng(task.seed)
    else:
        completed_cycle = checkpoint.completed_cycle
        elapsed_seconds = checkpoint.elapsed_seconds
        frame, ranks = checkpoint.frame, checkpoint.ranks
        rng_payload = checkpoint.rng_payload
        observer.restore_checkpoint(
            checkpoint.observer_payload, completed_cycle=completed_cycle
        )
        print(
            f"[checkpoint] {task.task_id}: verified cycle "
            f"{completed_cycle}/{task.cycles}",
            flush=True,
        )

    cycle_bar = tqdm(
        total=task.cycles,
        initial=completed_cycle,
        desc=f"{task.construction} Ny={task.ny} a1={task.alpha_label} nsh={task.shell_label}",
        unit="cycle",
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
    )
    cycle_bar.set_postfix(durable_cycle=completed_cycle, refresh=True)
    try:
        while completed_cycle < task.cycles:
            if completed_cycle:
                if rng_payload is None:
                    raise RuntimeError("checkpoint continuation lacks RNG state")
                _restore_rng_state(rng_payload)
            native, elapsed = _run_segment(
                config=config,
                model=model,
                task=task,
                observer=observer,
                segment_start=completed_cycle,
                frame=frame,
                ranks=ranks,
                cycle_bar=cycle_bar,
            )
            completed_cycle += min(
                EXPECTED_SEGMENT_CYCLES, task.cycles - completed_cycle
            )
            elapsed_seconds += elapsed
            frame, ranks = _native_arrays(native)
            rng_payload = save_checkpoint(
                output_root=output_root,
                scratch_root=scratch_root,
                task=task,
                completed_cycle=completed_cycle,
                elapsed_seconds=elapsed_seconds,
                native=native,
                observer=observer,
                configuration_sha256=configuration_sha256,
                hashes=hashes,
            )
            cycle_bar.set_postfix(durable_cycle=completed_cycle, refresh=True)
    finally:
        cycle_bar.close()

    observer.validate()
    save_result(
        output_root=output_root,
        scratch_root=scratch_root,
        task=task,
        observer=observer,
        elapsed_seconds=elapsed_seconds,
        configuration_sha256=configuration_sha256,
        hashes=hashes,
    )
    remove_checkpoint(output_root, task)
    shutil.rmtree(task_scratch)
    return elapsed_seconds


def validate_a100() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an A100 GPU runtime")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    name, total_bytes = str(properties.name), int(properties.total_memory)
    if "A100" not in name.upper() or total_bytes < 35 * 1024**3:
        raise RuntimeError(
            f"production requires an A100 40-GB-class GPU, got {name} "
            f"({total_bytes / 1024**3:.2f} GiB)"
        )
    probe = torch.zeros(1, dtype=torch.complex128, device=device)
    del probe
    return {"name": name, "total_bytes": total_bytes, "device": str(device)}


def _check_space(path: Path, *, required_bytes: int, label: str) -> int:
    path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path).free)
    if free < required_bytes:
        raise RuntimeError(
            f"insufficient {label} space: {free / 1024**3:.2f} GiB free, "
            f"{required_bytes / 1024**3:.2f} GiB required"
        )
    return free


def run_campaign(
    *,
    config: Mapping[str, Any],
    construction: str,
    output_root: Path,
    scratch_root: Path,
    report_only: bool = False,
    max_new_tasks: int | None = None,
) -> dict[str, Any]:
    config = validate_config(config)
    tasks = expand_tasks(config, construction)
    hashes = source_hashes()
    configuration_sha256 = config_sha256(config)
    output_root, scratch_root = Path(output_root), Path(scratch_root)
    statuses = {
        task.task_id: verified_complete(
            output_root=output_root,
            task=task,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
        for task in tasks
    }
    complete = sum(valid for valid, _ in statuses.values())
    checkpoint_rows = []
    for task in tasks:
        if statuses[task.task_id][0]:
            continue
        checkpoint, reason = load_checkpoint(
            output_root=output_root,
            task=task,
            configuration_sha256=configuration_sha256,
            hashes=hashes,
        )
        if checkpoint is not None or reason != "no checkpoint":
            checkpoint_rows.append(
                {
                    "task": task.task_id,
                    "cycle": (
                        None if checkpoint is None else checkpoint.completed_cycle
                    ),
                    "status": reason,
                }
            )
    inventory = {
        "bundle": BUNDLE,
        "construction": construction,
        "configuration_sha256": configuration_sha256,
        "tasks_total": len(tasks),
        "tasks_complete": complete,
        "tasks_pending": len(tasks) - complete,
        "trajectories_total": sum(task.sample_count for task in tasks),
        "trajectories_complete": sum(
            task.sample_count for task in tasks if statuses[task.task_id][0]
        ),
        "checkpoints": checkpoint_rows,
        "output_root": str(output_root),
        "scratch_root": str(scratch_root),
        "source_hashes": hashes,
    }
    print("[resolved campaign]", flush=True)
    print(json.dumps(inventory, indent=2, sort_keys=True), flush=True)
    if report_only:
        return inventory
    if max_new_tasks is not None and max_new_tasks < 0:
        raise ValueError("max_new_tasks must be nonnegative")
    gpu = validate_a100()
    local_free = _check_space(
        scratch_root, required_bytes=3 * 1024**3, label="local scratch"
    )
    drive_free = _check_space(
        output_root, required_bytes=3 * 1024**3, label="Drive output"
    )
    print(
        json.dumps(
            {
                "device": gpu,
                "dtype": "complex128",
                "local_free_GiB": local_free / 1024**3,
                "drive_free_GiB": drive_free / 1024**3,
                "new_task_limit": max_new_tasks,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    completed_count, skipped_count, failed_count, new_count = 0, 0, 0, 0
    bar = tqdm(
        tasks,
        total=len(tasks),
        desc=f"{construction} correlator campaign",
        unit="task",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    try:
        for task in bar:
            valid, _ = statuses[task.task_id]
            if valid:
                completed_count += 1
                skipped_count += 1
                remove_checkpoint(output_root, task)
            elif max_new_tasks is not None and new_count >= max_new_tasks:
                continue
            else:
                bar.set_description(f"{construction}: {task.task_id}")
                bar.set_postfix(
                    verified=completed_count,
                    skipped=skipped_count,
                    pending=len(tasks) - completed_count,
                    running=1,
                    failed=failed_count,
                    refresh=True,
                )
                try:
                    execute_task(
                        config=config,
                        output_root=output_root,
                        scratch_root=scratch_root,
                        task=task,
                        configuration_sha256=configuration_sha256,
                        hashes=hashes,
                    )
                except Exception:
                    failed_count += 1
                    bar.set_postfix(
                        verified=completed_count,
                        skipped=skipped_count,
                        pending=len(tasks) - completed_count,
                        running=0,
                        failed=failed_count,
                        refresh=True,
                    )
                    raise
                new_count += 1
                completed_count += 1
                statuses[task.task_id] = (True, "verified")
            bar.set_postfix(
                verified=completed_count,
                skipped=skipped_count,
                pending=len(tasks) - completed_count,
                running=0,
                failed=failed_count,
                refresh=True,
            )
    finally:
        bar.close()

    summary = {
        **inventory,
        "tasks_complete_after_run": completed_count,
        "tasks_newly_completed": new_count,
        "tasks_skipped": skipped_count,
        "tasks_failed": failed_count,
        "tasks_pending_after_run": len(tasks) - completed_count,
    }
    print("[campaign summary]", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--construction", choices=tuple(CONSTRUCTIONS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-tasks", type=int)
    args = parser.parse_args(argv)
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        parser.error("--max-new-tasks must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_campaign(
        config=config,
        construction=args.construction,
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        report_only=args.report_only,
        max_new_tasks=args.max_new_tasks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
