from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import platform
import shutil
import socket
import threading
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu
except ModuleNotFoundError:  # Repository tests import the canonical source package.
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu

try:
    from occupied_frame_gpu import GPU_FRAME_ALGORITHM_VERSION
except ModuleNotFoundError:
    from src.fgtn.occupied_frame_gpu import GPU_FRAME_ALGORITHM_VERSION
from p1_chern_observables import P1ChernObserver
from drive_remote_commit import (
    DEFAULT_DRIVE_ROOT,
    DriveRemoteCommitter,
    RemoteCommitError,
    publish_json,
    read_remote_json,
)


BUNDLE = "01_p1_chern_dynamics"
SAMPLING_REVISION = "production_25sample_p1_chern_v4"
L64_SHARD_SIZE = 1
CONTRACT_AUDIT_SHA256 = (
    "9e44a3c03d201756ca8615d7b10f1488fd1f367afc11a7c05464cc1595c047ae"
)
ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
A100_PREFLIGHT_SCHEMA = "p1_a100_L64_server_verified_cycle_checkpoint_v4"
P1_CHECKPOINT_SCHEMA = "p1_cycle_checkpoint_v2_remote_verified"
P1_CHECKPOINT_POINTER_SCHEMA = "p1_cycle_checkpoint_pointer_v2_remote_verified"
P1_SESSION_CHECKPOINT_PREFIX = "[P1 SESSION CHECKPOINT] "
CHECKPOINT_DIRECTORY = "_cycle_checkpoints"
CHECKPOINT_INTERVAL_CYCLES = 1
P1_CYCLE_RUNTIME_SAFETY_FACTOR = 1.25
P1_CHECKPOINT_WRITE_BUFFER_SECONDS = 120.0
P1_LEASE_SCHEMA = "p1_shard_single_writer_lease_v1"
P1_LEASE_HEARTBEAT_SECONDS = 30.0
P1_LEASE_STALE_SECONDS = 900.0
P1_REMOTE_REQUIRED_HEADROOM_BYTES = 1_342_177_280


def _server_commit_required(path: Path | str) -> bool:
    try:
        Path(path).resolve().relative_to(DEFAULT_DRIVE_ROOT.resolve())
    except ValueError:
        return False
    return True


def _remote_committer(path: Path | str) -> DriveRemoteCommitter:
    if not _server_commit_required(path):
        raise RuntimeError(f"not a production Google Drive path: {path}")
    return DriveRemoteCommitter(drive_root=DEFAULT_DRIVE_ROOT)


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return repr(value)


def write_json_atomic(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def load_config(bundle_root: Path | str) -> dict[str, Any]:
    path = Path(bundle_root) / "production_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    expected_locked = {
        "samples": 25,
        "sample_shard_size_by_L": {"16": 5, "24": 5, "32": 5, "64": 1},
        "checkpoint_unit": "atomic_completed_cycle_occupied_frame_rng_observer",
        "checkpoint_interval_cycles": 1,
        "checkpoint_retention": "one_rolling_checkpoint_per_active_shard_deleted_after_verified_archive",
        "cycles_rule": "L",
        "physical_burn_in_cycles": 0,
        "sequence": "random",
        "dtype": "complex128",
        "canonical_entry_point": ENTRY_POINT,
    }
    if config.get("bundle") != BUNDLE:
        raise ValueError("P1 bundle name is not immutable")
    if config.get("sampling_revision") != SAMPLING_REVISION:
        raise ValueError("P1 sampling revision is not immutable")
    if config.get("audit_sha256") != CONTRACT_AUDIT_SHA256:
        raise ValueError("P1 contract audit hash differs from the approved redesign")
    expected_drive_commit = {
        "schema": "classA_drive_api_commit_v1",
        "authoritative_backend": "google_drive_api_v3",
        "drivefs_role": "non_authoritative_cache",
        "required_headroom_bytes": P1_REMOTE_REQUIRED_HEADROOM_BYTES,
        "checkpoint_publish_order": "generation_remote_verify_then_pointer_remote_verify_then_previous_delete",
        "archive_publish_order": "archive_remote_verify_then_receipt_remote_verify_then_checkpoint_delete",
    }
    if config.get("drive_commit") != expected_drive_commit:
        raise ValueError("P1-v4 Drive durability contract changed")
    if config.get("locked_contract") != expected_locked:
        raise ValueError("P1 locked contract differs from the approved redesign")
    campaign = config.get("P1", {})
    expected = {
        "sizes": [16, 24, 32, 64],
        "nshell_values": [1, 2, None],
        "alpha_1": 1.0,
        "alpha_2": 1.0,
        "initial_state": "random_half_filled_slater",
        "ancilla_occupation": 0.5,
        "center_count": 10,
        "center_sampling": "fresh_per_sample_cycle_shared_across_shells",
        "chern_radius_fraction": 0.4,
        "chern_sign_target": 1.0,
        "declared_observables": ["periodic_trijunction_real_space_chern"],
    }
    mismatches = {
        key: (campaign.get(key), value)
        for key, value in expected.items()
        if campaign.get(key) != value
    }
    if mismatches:
        raise ValueError(f"P1 campaign differs from the approved redesign: {mismatches}")


def expand_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    campaign = config["P1"]
    shard_sizes = config["locked_contract"]["sample_shard_size_by_L"]
    cases: list[dict[str, Any]] = []
    for size in campaign["sizes"]:
        for nshell in campaign["nshell_values"]:
            shell_tag = "none" if nshell is None else str(int(nshell))
            cases.append(
                {
                    "case_id": f"P1_CHERN_L{int(size)}_nsh-{shell_tag}",
                    "campaign": "P1",
                    "kind": "stochastic",
                    "model": {
                        "Nx": int(size),
                        "Ny": int(size),
                        "DW": False,
                        "nshell": nshell,
                        "filling_frac": 0.5,
                        "alpha_1": 1.0,
                        "alpha_2": 1.0,
                        "trial_orbitals": "X",
                        "dw_truncation": False,
                        "device": "cuda:0",
                        "dtype": "complex128",
                        "backend": "dense" if nshell is None else "local",
                        "init_mode": "default",
                    },
                    "run": {
                        "cycles": int(size),
                        "samples": 25,
                        "sequence": "random",
                        "perfect_correction": True,
                        "postselect": False,
                        "postselect_probability": 0.0,
                        "n_a": 0.5,
                    },
                    "execution": {
                        "samples_per_shard": int(shard_sizes[str(int(size))]),
                    },
                    "observer": {
                        "center_count": 10,
                        "radius_fraction": 0.4,
                        "cycles": list(range(int(size) + 1)),
                    },
                }
            )
    if len(cases) != 12:
        raise AssertionError("P1 expansion must contain exactly twelve cases")
    return cases


def case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(case["case_id"]): case for case in cases}
    if len(result) != len(cases):
        raise ValueError("duplicate P1 case ID")
    return result


def samples_per_shard(case: dict[str, Any]) -> int:
    value = int(case.get("execution", {}).get("samples_per_shard", 0))
    total = int(case["run"]["samples"])
    if value <= 0 or total % value:
        raise ValueError(
            f"{case['case_id']}: samples_per_shard must be a positive divisor of {total}"
        )
    return value


def shard_count(case: dict[str, Any]) -> int:
    return int(case["run"]["samples"]) // samples_per_shard(case)


def global_sample_indices(case: dict[str, Any], shard_index: int) -> list[int]:
    count = shard_count(case)
    if not 0 <= int(shard_index) < count:
        raise IndexError(f"shard index must lie in 0..{count - 1}")
    width = samples_per_shard(case)
    start = int(shard_index) * width
    return list(range(start, start + width))


def _source_hashes(src_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(src_dir)): sha256_file(path)
        for path in sorted(src_dir.glob("*.py"))
    }


def _require_a100(*, smoke: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        if smoke:
            return {"device": "cpu", "smoke_override": True}
        raise RuntimeError("P1 production and preflight require an A100 GPU")
    name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if "A100" not in name.upper() and not smoke:
        raise RuntimeError(f"P1 production requires an A100; detected {name!r}")
    if free_bytes / total_bytes < 0.2 and not smoke:
        raise RuntimeError("P1 requires at least 20% free A100 memory")
    return {
        "device": name,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "free_fraction": float(free_bytes / total_bytes),
        "smoke_override": bool(smoke),
    }


def _archive_paths(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any], shard_index: int,
    drive_root: Path, mode: str,
) -> tuple[Path, Path, str, dict[str, Any]]:
    engine_hash = sha256_file(bundle_root / "src" / "classA_U1FGTN_gpu.py")
    source_hashes = _source_hashes(bundle_root / "src")
    run_config = {
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "canonical_engine_sha256": engine_hash,
        "bundle_source_hashes": source_hashes,
        "bundle_source_hashes_sha256": sha256_json(source_hashes),
        "checkpoint_schema": P1_CHECKPOINT_SCHEMA,
        "shard_index": int(shard_index),
        "case": case,
    }
    run_id = f"{BUNDLE}_{sha256_json(run_config)[:16]}"
    scratch_base = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
    scratch = scratch_base / "classA_final_production" / BUNDLE / run_id
    collection_key = "production_output_collection" if mode != "pilot" else "pilot_output_collection"
    output = drive_root.resolve() / str(config[collection_key]) / str(config.get("output_bundle", BUNDLE))
    archive = output / f"{run_id}.tar.gz"
    return scratch, archive, run_id, run_config


def _checkpoint_paths(*, archive: Path, run_id: str) -> tuple[Path, Path]:
    directory = archive.parent / CHECKPOINT_DIRECTORY
    return directory, directory / f"{run_id}.latest.json"


def _npz_scalar(payload: dict[str, np.ndarray], name: str) -> Any:
    if name not in payload:
        raise KeyError(f"P1 checkpoint lacks {name!r}")
    value = np.asarray(payload[name])
    if value.shape != ():
        raise ValueError(f"P1 checkpoint field {name!r} must be scalar")
    return value.item()


def _capture_rng_payload(prefix: str) -> dict[str, np.ndarray]:
    payload = {
        f"{prefix}numpy_state_json": np.asarray(
            json.dumps(json_ready(np.random.get_state()))
        ),
        f"{prefix}torch_cpu": torch.get_rng_state().cpu().numpy(),
        f"{prefix}torch_cuda_count": np.asarray(
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
            dtype=np.int64,
        ),
    }
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            payload[f"{prefix}torch_cuda_{index}"] = state.cpu().numpy()
    return payload


def _restore_rng_payload(payload: dict[str, np.ndarray], prefix: str) -> None:
    cpu = np.asarray(payload[f"{prefix}torch_cpu"])
    if cpu.dtype != np.uint8 or cpu.ndim != 1:
        raise ValueError("P1 checkpoint has an invalid Torch CPU RNG state")
    torch.set_rng_state(torch.as_tensor(cpu, dtype=torch.uint8, device="cpu"))
    numpy_state = json.loads(str(_npz_scalar(payload, f"{prefix}numpy_state_json")))
    if not isinstance(numpy_state, list) or len(numpy_state) != 5:
        raise ValueError("P1 checkpoint has an invalid NumPy RNG state")
    np.random.set_state(
        (
            str(numpy_state[0]),
            np.asarray(numpy_state[1], dtype=np.uint32),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        )
    )
    saved_cuda_count = int(_npz_scalar(payload, f"{prefix}torch_cuda_count"))
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_cuda_count != current_cuda_count:
        raise RuntimeError(
            "P1 checkpoint CUDA RNG device count differs from this runtime"
        )
    for index in range(saved_cuda_count):
        state = np.asarray(payload[f"{prefix}torch_cuda_{index}"])
        if state.dtype != np.uint8 or state.ndim != 1:
            raise ValueError("P1 checkpoint has an invalid CUDA RNG state")
        torch.cuda.set_rng_state(
            torch.as_tensor(state, dtype=torch.uint8, device="cpu"),
            device=index,
        )


def _checkpoint_identity(
    *,
    run_id: str,
    run_config: dict[str, Any],
    case: dict[str, Any],
    shard_index: int,
    global_sample_ids: list[int],
) -> dict[str, Any]:
    return {
        "schema": P1_CHECKPOINT_SCHEMA,
        "bundle": BUNDLE,
        "sampling_revision": SAMPLING_REVISION,
        "audit_sha256": CONTRACT_AUDIT_SHA256,
        "canonical_entry_point": ENTRY_POINT,
        "canonical_engine_sha256": run_config["canonical_engine_sha256"],
        "bundle_source_hashes": run_config["bundle_source_hashes"],
        "bundle_source_hashes_sha256": run_config[
            "bundle_source_hashes_sha256"
        ],
        "run_id": run_id,
        "run_config": run_config,
        "run_config_hash": sha256_json(run_config),
        "case_id": str(case["case_id"]),
        "shard_index": int(shard_index),
        "global_sample_indices": [int(value) for value in global_sample_ids],
        "total_cycles": int(case["run"]["cycles"]),
        "checkpoint_interval_cycles": CHECKPOINT_INTERVAL_CYCLES,
        "state_representation": "physical_frame",
        "frame_algorithm_version": GPU_FRAME_ALGORITHM_VERSION,
        "dtype": "complex128",
    }


def _native_checkpoint_arrays(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    required = {
        "representation",
        "frame",
        "ranks",
        "min_ranks",
        "max_ranks",
        "log_weight",
        "physical_dimension",
        "capacity",
        "frame_algorithm_version",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise KeyError(f"native P1 state lacks checkpoint fields: {missing}")
    return {
        "native_representation": np.asarray(snapshot["representation"]),
        "native_frame": np.asarray(snapshot["frame"]),
        "native_ranks": np.asarray(snapshot["ranks"]),
        "native_min_ranks": np.asarray(snapshot["min_ranks"]),
        "native_max_ranks": np.asarray(snapshot["max_ranks"]),
        "native_log_weight": np.asarray(snapshot["log_weight"]),
        "native_physical_dimension": np.asarray(
            snapshot["physical_dimension"], dtype=np.int64
        ),
        "native_capacity": np.asarray(snapshot["capacity"], dtype=np.int64),
        "native_frame_algorithm_version": np.asarray(
            snapshot["frame_algorithm_version"]
        ),
    }


def _write_cycle_checkpoint(
    *,
    archive: Path,
    run_id: str,
    identity: dict[str, Any],
    completed_cycle: int,
    native_snapshot: dict[str, Any],
    observer: P1ChernObserver,
    initial_rng: dict[str, np.ndarray],
    cumulative_elapsed_seconds: float,
    gpu_peak_allocated_bytes: int,
    gpu_peak_reserved_bytes: int,
) -> dict[str, Any]:
    directory, pointer = _checkpoint_paths(archive=archive, run_id=run_id)
    remote_commit = _server_commit_required(archive)
    if not remote_commit:
        directory.mkdir(parents=True, exist_ok=True)
    identity_json = json.dumps(
        json_ready(identity), sort_keys=True, separators=(",", ":")
    )
    payload: dict[str, np.ndarray] = {
        "checkpoint_schema": np.asarray(P1_CHECKPOINT_SCHEMA),
        "checkpoint_identity_json": np.asarray(identity_json),
        "checkpoint_identity_sha256": np.asarray(
            hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
        ),
        "completed_cycle": np.asarray(completed_cycle, dtype=np.int64),
        "total_cycles": np.asarray(identity["total_cycles"], dtype=np.int64),
        "cumulative_elapsed_seconds": np.asarray(
            cumulative_elapsed_seconds, dtype=np.float64
        ),
        "gpu_peak_allocated_bytes": np.asarray(
            gpu_peak_allocated_bytes, dtype=np.int64
        ),
        "gpu_peak_reserved_bytes": np.asarray(
            gpu_peak_reserved_bytes, dtype=np.int64
        ),
    }
    payload.update(_native_checkpoint_arrays(native_snapshot))
    payload.update(
        {f"observer__{key}": value for key, value in observer.checkpoint_state().items()}
    )
    payload.update(initial_rng)
    payload.update(_capture_rng_payload("rng_current__"))

    temporary: Path | None = None
    try:
        stage_directory = (
            Path("/content/classA_remote_stage/p1_checkpoints")
            if remote_commit
            else directory
        )
        stage_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=stage_directory,
            prefix=f".{run_id}.cycle_{int(completed_cycle):06d}.",
            suffix=".npz.tmp",
            delete=False,
        ) as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        checkpoint_sha256 = sha256_file(temporary)
        generation = directory / (
            f"{run_id}.cycle_{int(completed_cycle):06d}."
            f"{checkpoint_sha256[:16]}.npz"
        )
        previous: dict[str, Any] | None = None
        generation_remote_commit: dict[str, Any] | None = None
        if remote_commit:
            committer = _remote_committer(archive)
            try:
                previous = read_remote_json(committer, pointer)
            except RemoteCommitError as exc:
                if "absent" not in str(exc):
                    raise
            generation_remote_commit = committer.upload_verified(
                temporary,
                generation,
                replace=False,
                required_headroom_bytes=P1_REMOTE_REQUIRED_HEADROOM_BYTES,
            )
        else:
            os.replace(temporary, generation)
            temporary = None
        receipt = {
            "schema": P1_CHECKPOINT_POINTER_SCHEMA,
            "checkpoint_schema": P1_CHECKPOINT_SCHEMA,
            "run_id": run_id,
            "checkpoint": generation.name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": int(
                generation_remote_commit["remote_bytes"]
                if generation_remote_commit is not None
                else generation.stat().st_size
            ),
            "checkpoint_identity_sha256": sha256_json(identity),
            "completed_cycle": int(completed_cycle),
            "total_cycles": int(identity["total_cycles"]),
            "created_unix": time.time(),
        }
        if generation_remote_commit is not None:
            receipt["checkpoint_remote_commit"] = generation_remote_commit
            pointer_remote_commit = publish_json(
                committer,
                receipt,
                pointer,
                replace=True,
                required_headroom_bytes=P1_REMOTE_REQUIRED_HEADROOM_BYTES,
            )
            if previous is not None:
                old = previous.get("checkpoint_remote_commit")
                if (
                    isinstance(old, dict)
                    and old.get("remote_file_id")
                    != generation_remote_commit.get("remote_file_id")
                ):
                    committer.delete_verified(old)
            receipt["pointer_remote_commit"] = pointer_remote_commit
        else:
            write_json_atomic(pointer, receipt)
            _purge_checkpoint_orphans(
                directory=directory, run_id=run_id, keep_generation=generation
            )
        return {**receipt, "pointer_path": str(pointer)}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _purge_checkpoint_orphans(
    *, directory: Path, run_id: str, keep_generation: Path | None
) -> None:
    for temporary in directory.glob(f".{run_id}.cycle_*.npz.tmp"):
        if temporary.is_file():
            temporary.unlink()
    for generation in directory.glob(f"{run_id}.cycle_*.npz"):
        if generation.is_file() and generation != keep_generation:
            generation.unlink()


def _load_cycle_checkpoint(
    *,
    archive: Path,
    run_id: str,
    expected_identity: dict[str, Any],
    observer: P1ChernObserver,
) -> dict[str, Any] | None:
    directory, pointer = _checkpoint_paths(archive=archive, run_id=run_id)
    remote_commit = _server_commit_required(archive)
    if remote_commit:
        committer = _remote_committer(archive)
        try:
            receipt = read_remote_json(committer, pointer)
            committer.path_commit_record(pointer)
        except RemoteCommitError as exc:
            if "absent" in str(exc):
                return None
            raise
    elif not pointer.exists():
        if directory.is_dir():
            _purge_checkpoint_orphans(
                directory=directory, run_id=run_id, keep_generation=None
            )
        return None
    else:
        receipt = json.loads(pointer.read_text(encoding="utf-8"))
    expected_pointer = {
        "schema": P1_CHECKPOINT_POINTER_SCHEMA,
        "checkpoint_schema": P1_CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "checkpoint_identity_sha256": sha256_json(expected_identity),
        "total_cycles": int(expected_identity["total_cycles"]),
    }
    mismatches = {
        key: (receipt.get(key), value)
        for key, value in expected_pointer.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"P1 checkpoint pointer identity mismatch: {mismatches}")
    filename = str(receipt.get("checkpoint", ""))
    if not filename or Path(filename).name != filename:
        raise RuntimeError("P1 checkpoint pointer has an invalid payload filename")
    checkpoint = directory / filename
    checkpoint_for_read = checkpoint
    if remote_commit:
        remote_record = receipt.get("checkpoint_remote_commit")
        if not isinstance(remote_record, dict):
            raise RuntimeError("P1-v4 pointer lacks its remote checkpoint commit")
        committer.verify_commit_record(remote_record)
        if (
            int(remote_record.get("remote_bytes", -1))
            != int(receipt.get("checkpoint_bytes", -2))
            or remote_record.get("remote_sha256")
            != receipt.get("checkpoint_sha256")
        ):
            raise RuntimeError("P1-v4 checkpoint pointer disagrees with Drive metadata")
        cache = Path("/content/classA_remote_cache/p1_checkpoints") / filename
        if (
            not cache.is_file()
            or cache.stat().st_size != int(receipt["checkpoint_bytes"])
            or sha256_file(cache) != receipt["checkpoint_sha256"]
        ):
            committer.download_to(str(remote_record["remote_file_id"]), cache)
        checkpoint_for_read = cache
    else:
        if not checkpoint.is_file():
            raise RuntimeError(f"P1 checkpoint payload is missing: {checkpoint}")
        if checkpoint.stat().st_size != int(receipt.get("checkpoint_bytes", -1)):
            raise RuntimeError("P1 checkpoint payload byte count differs from its pointer")
        if sha256_file(checkpoint) != receipt.get("checkpoint_sha256"):
            raise RuntimeError("P1 checkpoint payload checksum mismatch")
    with np.load(checkpoint_for_read, allow_pickle=False) as stored:
        payload = {key: np.asarray(stored[key]) for key in stored.files}

    if _npz_scalar(payload, "checkpoint_schema") != P1_CHECKPOINT_SCHEMA:
        raise RuntimeError("P1 checkpoint payload has the wrong schema")
    identity_json = str(_npz_scalar(payload, "checkpoint_identity_json"))
    if hashlib.sha256(identity_json.encode("utf-8")).hexdigest() != str(
        _npz_scalar(payload, "checkpoint_identity_sha256")
    ):
        raise RuntimeError("P1 checkpoint embedded identity checksum mismatch")
    embedded_identity = json.loads(identity_json)
    if json_ready(embedded_identity) != json_ready(expected_identity):
        raise RuntimeError("P1 checkpoint embedded identity is stale")
    completed_cycle = int(_npz_scalar(payload, "completed_cycle"))
    total_cycles = int(expected_identity["total_cycles"])
    if (
        completed_cycle != int(receipt.get("completed_cycle", -1))
        or not 0 <= completed_cycle <= total_cycles
        or int(_npz_scalar(payload, "total_cycles")) != total_cycles
    ):
        raise RuntimeError("P1 checkpoint has an invalid completed-cycle index")

    frame = np.asarray(payload["native_frame"])
    ranks = np.asarray(payload["native_ranks"])
    minimum = np.asarray(payload["native_min_ranks"])
    maximum = np.asarray(payload["native_max_ranks"])
    log_weight = np.asarray(payload["native_log_weight"])
    samples = len(expected_identity["global_sample_indices"])
    dimension = 2 * int(expected_identity["run_config"]["case"]["model"]["Nx"]) ** 2
    if frame.dtype != np.complex128 or frame.ndim != 3:
        raise ValueError("P1 checkpoint frame must be rank-3 complex128")
    if frame.shape[0] != samples or frame.shape[1] != dimension:
        raise ValueError("P1 checkpoint frame has the wrong sample/mode axes")
    if ranks.dtype != np.int64 or ranks.shape != (samples,):
        raise ValueError("P1 checkpoint ranks must be int64 with one value per sample")
    for name, values in (("min_ranks", minimum), ("max_ranks", maximum)):
        if values.dtype != np.int64 or values.shape != (samples,):
            raise ValueError(f"P1 checkpoint {name} has the wrong dtype or shape")
    if log_weight.dtype != np.float64 or log_weight.shape != (samples,):
        raise ValueError("P1 checkpoint log weights have the wrong dtype or shape")
    if (
        str(_npz_scalar(payload, "native_representation")) != "physical_frame"
        or str(_npz_scalar(payload, "native_frame_algorithm_version"))
        != GPU_FRAME_ALGORITHM_VERSION
        or int(_npz_scalar(payload, "native_physical_dimension")) != dimension
        or int(_npz_scalar(payload, "native_capacity")) != frame.shape[2]
    ):
        raise ValueError("P1 checkpoint native-state identity differs")
    if (
        np.any(ranks < 0)
        or np.any(ranks > frame.shape[2])
        or np.any(minimum > ranks)
        or np.any(maximum < ranks)
        or not np.isfinite(frame).all()
        or not np.isfinite(log_weight).all()
    ):
        raise ValueError("P1 checkpoint native state is numerically invalid")
    for sample, rank in enumerate(ranks):
        if np.any(frame[sample, :, int(rank) :] != 0.0):
            raise ValueError("P1 checkpoint has nonzero inactive frame padding")

    observer_payload = {
        key.removeprefix("observer__"): value
        for key, value in payload.items()
        if key.startswith("observer__")
    }
    observer.restore_checkpoint_state(
        observer_payload, completed_cycle=completed_cycle
    )
    _restore_rng_payload(payload, "rng_current__")
    _purge_checkpoint_orphans(
        directory=directory, run_id=run_id, keep_generation=checkpoint
    )
    native_state = {
        "representation": "physical_frame",
        "frame": frame,
        "ranks": ranks,
        "min_ranks": minimum,
        "max_ranks": maximum,
        "log_weight": log_weight,
        "physical_dimension": dimension,
        "capacity": frame.shape[2],
        "frame_algorithm_version": GPU_FRAME_ALGORITHM_VERSION,
    }
    initial_rng = {
        key: value for key, value in payload.items() if key.startswith("rng_initial__")
    }
    return {
        "completed_cycle": completed_cycle,
        "native_state": native_state,
        "initial_rng": initial_rng,
        "cumulative_elapsed_seconds": float(
            _npz_scalar(payload, "cumulative_elapsed_seconds")
        ),
        "gpu_peak_allocated_bytes": int(
            _npz_scalar(payload, "gpu_peak_allocated_bytes")
        ),
        "gpu_peak_reserved_bytes": int(
            _npz_scalar(payload, "gpu_peak_reserved_bytes")
        ),
        "receipt": {**receipt, "pointer_path": str(pointer)},
    }


def _cleanup_cycle_checkpoint(*, archive: Path, run_id: str) -> None:
    directory, pointer = _checkpoint_paths(archive=archive, run_id=run_id)
    if _server_commit_required(archive):
        committer = _remote_committer(archive)
        try:
            receipt = read_remote_json(committer, pointer)
        except RemoteCommitError as exc:
            if "absent" in str(exc):
                return
            raise
        generation = receipt.get("checkpoint_remote_commit")
        if not isinstance(generation, dict):
            raise RuntimeError("P1-v4 checkpoint cleanup lacks remote generation metadata")
        pointer_record = committer.path_commit_record(pointer)
        committer.delete_verified_if_present(generation)
        committer.delete_verified_if_present(pointer_record)
        return
    if pointer.exists():
        pointer.unlink()
    if directory.is_dir():
        _purge_checkpoint_orphans(
            directory=directory, run_id=run_id, keep_generation=None
        )
        for path in directory.glob(f"{run_id}.*"):
            if path.is_file():
                path.unlink()
        try:
            directory.rmdir()
        except OSError:
            pass


def _restore_native_diagnostics(state: Any, snapshot: dict[str, Any]) -> None:
    if int(state.capacity) != int(snapshot["capacity"]):
        raise RuntimeError("resumed P1 occupied-frame capacity changed")
    expected_ranks = torch.as_tensor(
        snapshot["ranks"], dtype=torch.long, device=state.device
    )
    if not torch.equal(state.ranks, expected_ranks):
        raise RuntimeError("resumed P1 occupied-frame ranks changed")
    state.min_ranks = torch.as_tensor(
        snapshot["min_ranks"], dtype=torch.long, device=state.device
    ).clone()
    state.max_ranks = torch.as_tensor(
        snapshot["max_ranks"], dtype=torch.long, device=state.device
    ).clone()
    state.log_weight = torch.as_tensor(
        snapshot["log_weight"], dtype=state.real_dtype, device=state.device
    ).clone()
    state.materialization_count = 0
    state.materialization_reasons = []


class _CycleCheckpointObserver:
    def __init__(
        self,
        *,
        archive: Path,
        run_id: str,
        identity: dict[str, Any],
        observer: P1ChernObserver,
        initial_rng: dict[str, np.ndarray],
        resume_cycle: int | None,
        resume_native: dict[str, Any] | None,
        elapsed_before_segment: float,
        segment_started: float,
        peak_allocated_before: int,
        peak_reserved_before: int,
    ) -> None:
        self.archive = archive
        self.run_id = run_id
        self.identity = identity
        self.observer = observer
        self.initial_rng = initial_rng
        self.resume_cycle = resume_cycle
        self.resume_native = resume_native
        self.elapsed_before_segment = float(elapsed_before_segment)
        self.segment_started = float(segment_started)
        self.peak_allocated_before = int(peak_allocated_before)
        self.peak_reserved_before = int(peak_reserved_before)
        self.completed_cycle = resume_cycle
        self.latest_native = resume_native
        self.latest_receipt: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        cycle: int,
        state: Any,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        local_cycle = int(cycle)
        expected_samples = len(self.identity["global_sample_indices"])
        if int(batch_start) != 0 or int(batch_count) != expected_samples:
            raise RuntimeError("P1 cycle checkpoint requires one complete engine batch")
        if self.resume_cycle is not None and local_cycle == 0:
            if self.resume_native is None:
                raise RuntimeError("P1 resume checkpoint lacks native state")
            _restore_native_diagnostics(state, self.resume_native)
            return
        global_cycle = (
            local_cycle
            if self.resume_cycle is None
            else int(self.resume_cycle) + local_cycle
        )
        if not 0 <= global_cycle <= int(self.identity["total_cycles"]):
            raise RuntimeError("P1 checkpoint observer mapped an invalid global cycle")
        self.observer(
            cycle=global_cycle,
            state=state,
            batch_start=batch_start,
            batch_count=batch_count,
        )
        snapshot = state.snapshot(cpu=True)
        peak_allocated = self.peak_allocated_before
        peak_reserved = self.peak_reserved_before
        if torch.cuda.is_available():
            peak_allocated = max(
                peak_allocated, int(torch.cuda.max_memory_allocated(state.device))
            )
            peak_reserved = max(
                peak_reserved, int(torch.cuda.max_memory_reserved(state.device))
            )
        receipt = _write_cycle_checkpoint(
            archive=self.archive,
            run_id=self.run_id,
            identity=self.identity,
            completed_cycle=global_cycle,
            native_snapshot=snapshot,
            observer=self.observer,
            initial_rng=self.initial_rng,
            cumulative_elapsed_seconds=(
                self.elapsed_before_segment
                + time.perf_counter()
                - self.segment_started
            ),
            gpu_peak_allocated_bytes=peak_allocated,
            gpu_peak_reserved_bytes=peak_reserved,
        )
        self.completed_cycle = global_cycle
        self.latest_native = snapshot
        self.latest_receipt = receipt
        print(
            f"[P1 CYCLE CHECKPOINT] cycle={global_cycle}/"
            f"{self.identity['total_cycles']} path={receipt['pointer_path']}",
            flush=True,
        )


def _verify_existing(
    archive: Path,
    *,
    expected_run_id: str | None = None,
    expected_run_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if _server_commit_required(archive):
        committer = _remote_committer(archive)
        try:
            receipt = read_remote_json(committer, receipt_path)
            committer.path_commit_record(receipt_path)
        except RemoteCommitError as exc:
            if "absent" in str(exc):
                try:
                    committer.path_commit_record(archive)
                except RemoteCommitError as archive_exc:
                    if "absent" in str(archive_exc):
                        return None
                    raise
                raise RuntimeError(
                    f"server archive exists without its remotely verified receipt: {archive}"
                ) from exc
            raise
        remote_archive = receipt.get("archive_remote_commit")
        if not isinstance(remote_archive, dict):
            raise RuntimeError("P1-v4 receipt lacks remote archive metadata")
        committer.verify_commit_record(remote_archive)
        if (
            receipt.get("archive") != archive.name
            or receipt.get("archive_sha256") != remote_archive.get("remote_sha256")
            or int(receipt.get("archive_bytes", -1))
            != int(remote_archive.get("remote_bytes", -2))
        ):
            raise RuntimeError("P1-v4 archive receipt disagrees with Drive metadata")
        if expected_run_id is not None and receipt.get("run_id") != expected_run_id:
            raise RuntimeError("archive receipt run ID differs from the requested shard")
        if expected_run_config is not None:
            manifest = _root_manifest_from_archive(archive)
            expected = {
                "status": "complete_local",
                "bundle": BUNDLE,
                "sampling_revision": SAMPLING_REVISION,
                "audit_sha256": CONTRACT_AUDIT_SHA256,
                "canonical_entry_point": ENTRY_POINT,
                "canonical_engine_sha256": expected_run_config[
                    "canonical_engine_sha256"
                ],
                "run_config_hash": sha256_json(expected_run_config),
                "source_hashes": expected_run_config["bundle_source_hashes"],
            }
            mismatches = {
                key: (manifest.get(key), value)
                for key, value in expected.items()
                if json_ready(manifest.get(key)) != json_ready(value)
            }
            if json_ready(manifest.get("run_config")) != json_ready(expected_run_config):
                mismatches["run_config"] = ("archive", "requested")
            if mismatches:
                raise RuntimeError(
                    f"archive scientific/source identity mismatch: {mismatches}"
                )
        return receipt
    if not archive.exists() and not receipt_path.exists():
        return None
    if receipt_path.exists() and not archive.is_file():
        raise RuntimeError(f"archive receipt exists without its archive: {archive}")
    if archive.is_file() and not receipt_path.exists():
        if expected_run_id is None or expected_run_config is None:
            raise RuntimeError(f"archive exists without a recoverable receipt: {archive}")
        manifest = _root_manifest_from_archive(archive)
        if (
            manifest.get("run_config_hash") != sha256_json(expected_run_config)
            or json_ready(manifest.get("run_config")) != json_ready(
                expected_run_config
            )
        ):
            raise RuntimeError(
                "receipt-less P1 archive does not match the requested run identity"
            )
        receipt = {
            "schema_version": 1,
            "run_id": expected_run_id,
            "archive": archive.name,
            "archive_sha256": sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "created_unix": time.time(),
            "recovered_after_interrupted_receipt_commit": True,
        }
        write_json_atomic(receipt_path, receipt)
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("archive") != archive.name:
        raise RuntimeError("archive receipt names a different file")
    if receipt.get("archive_sha256") != sha256_file(archive):
        raise RuntimeError("archive checksum mismatch")
    if expected_run_id is not None and receipt.get("run_id") != expected_run_id:
        raise RuntimeError("archive receipt run ID differs from the requested shard")
    if expected_run_config is not None:
        manifest = _root_manifest_from_archive(archive)
        expected = {
            "status": "complete_local",
            "bundle": BUNDLE,
            "sampling_revision": SAMPLING_REVISION,
            "audit_sha256": CONTRACT_AUDIT_SHA256,
            "canonical_entry_point": ENTRY_POINT,
            "canonical_engine_sha256": expected_run_config[
                "canonical_engine_sha256"
            ],
            "run_config_hash": sha256_json(expected_run_config),
            "source_hashes": expected_run_config["bundle_source_hashes"],
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if json_ready(manifest.get(key)) != json_ready(value)
        }
        if json_ready(manifest.get("run_config")) != json_ready(
            expected_run_config
        ):
            mismatches["run_config"] = ("archive", "requested")
        if mismatches:
            raise RuntimeError(f"archive scientific/source identity mismatch: {mismatches}")
    return receipt


def _root_manifest_from_archive(archive: Path) -> dict[str, Any]:
    archive_for_read = archive
    if _server_commit_required(archive):
        committer = _remote_committer(archive)
        receipt = read_remote_json(
            committer, archive.with_suffix(archive.suffix + ".receipt.json")
        )
        record = receipt.get("archive_remote_commit")
        if not isinstance(record, dict):
            raise RuntimeError("P1-v4 receipt lacks remote archive metadata")
        committer.verify_commit_record(record)
        cache = Path("/content/classA_remote_cache/p1_archives") / archive.name
        if (
            not cache.is_file()
            or cache.stat().st_size != int(record["remote_bytes"])
            or sha256_file(cache) != record["remote_sha256"]
        ):
            committer.download_to(str(record["remote_file_id"]), cache)
        archive_for_read = cache
    with tarfile.open(archive_for_read, "r:gz") as handle:
        members = [
            member
            for member in handle.getmembers()
            if member.name.lstrip("./") == "manifest.json"
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"{archive}: expected one root manifest, found {len(members)}"
            )
        extracted = handle.extractfile(members[0])
        if extracted is None:
            raise RuntimeError(f"{archive}: unreadable root manifest")
        return json.loads(extracted.read().decode("utf-8"))


def _archive(scratch: Path, archive: Path, run_id: str) -> dict[str, Any]:
    remote_commit = _server_commit_required(archive)
    if not remote_commit:
        archive.parent.mkdir(parents=True, exist_ok=True)
    stage_root = (
        Path("/content/classA_remote_stage/p1_archives")
        if remote_commit
        else archive.parent
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    temporary_base = stage_root / f".{run_id}.{os.getpid()}"
    temporary = Path(shutil.make_archive(str(temporary_base), "gztar", root_dir=scratch))
    archive_sha256 = sha256_file(temporary)
    archive_bytes = temporary.stat().st_size
    archive_remote_commit: dict[str, Any] | None = None
    if remote_commit:
        committer = _remote_committer(archive)
        archive_remote_commit = committer.upload_verified(
            temporary,
            archive,
            replace=False,
            required_headroom_bytes=P1_REMOTE_REQUIRED_HEADROOM_BYTES,
        )
    else:
        os.replace(temporary, archive)
    receipt = {
        "schema_version": 2 if remote_commit else 1,
        "run_id": run_id,
        "archive": archive.name,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "created_unix": time.time(),
    }
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if archive_remote_commit is not None:
        receipt["archive_remote_commit"] = archive_remote_commit
        receipt_commit = publish_json(
            committer,
            receipt,
            receipt_path,
            replace=False,
            required_headroom_bytes=P1_REMOTE_REQUIRED_HEADROOM_BYTES,
        )
        receipt["receipt_remote_commit"] = receipt_commit
        temporary.unlink(missing_ok=True)
    else:
        write_json_atomic(receipt_path, receipt)
    return receipt


class _ShardLease:
    """Drive-backed single-writer lease with heartbeat-based crash recovery."""

    def __init__(self, *, archive: Path, run_id: str) -> None:
        directory, _ = _checkpoint_paths(archive=archive, run_id=run_id)
        self.directory = directory
        self.path = directory / f"{run_id}.lease"
        self.run_id = run_id
        self.token = uuid.uuid4().hex
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": P1_LEASE_SCHEMA,
            "run_id": self.run_id,
            "owner_token": self.token,
            "hostname": self.hostname,
            "pid": self.pid,
            "updated_unix": time.time(),
        }

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        return pid > 0 and Path(f"/proc/{pid}").exists()

    @staticmethod
    def _clear_directory(path: Path) -> None:
        if not path.is_dir():
            return
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                raise RuntimeError(f"unexpected nested P1 lease path: {child}")
        path.rmdir()

    def _heartbeat(self) -> None:
        while not self._stop.wait(P1_LEASE_HEARTBEAT_SECONDS):
            try:
                current = json.loads(
                    (self.path / "lease.json").read_text(encoding="utf-8")
                )
                if current.get("owner_token") != self.token:
                    return
                write_json_atomic(self.path / "lease.json", self._payload())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return

    def __enter__(self) -> "_ShardLease":
        self.directory.mkdir(parents=True, exist_ok=True)
        for _ in range(4):
            try:
                self.path.mkdir()
            except FileExistsError:
                lease_json = self.path / "lease.json"
                try:
                    current = json.loads(lease_json.read_text(encoding="utf-8"))
                    age = time.time() - float(current["updated_unix"])
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    age = time.time() - self.path.stat().st_mtime
                    current = {}
                same_host = current.get("hostname") == self.hostname
                same_live_process = (
                    same_host and self._pid_alive(int(current.get("pid", -1)))
                )
                known_dead_same_host = same_host and not same_live_process
                if same_live_process or (
                    not known_dead_same_host and age < P1_LEASE_STALE_SECONDS
                ):
                    raise RuntimeError(
                        "another P1 writer owns this shard lease: "
                        f"{self.path} owner={current}"
                    )
                stale = self.directory / (
                    f".{self.run_id}.stale_lease.{uuid.uuid4().hex}"
                )
                try:
                    os.replace(self.path, stale)
                except FileNotFoundError:
                    continue
                self._clear_directory(stale)
                continue
            write_json_atomic(self.path / "lease.json", self._payload())
            self._thread = threading.Thread(
                target=self._heartbeat,
                name=f"p1-lease-{self.run_id[-8:]}",
                daemon=True,
            )
            self._thread.start()
            return self
        raise RuntimeError(f"could not acquire P1 shard lease: {self.path}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            current = json.loads(
                (self.path / "lease.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            current = {}
        if current.get("owner_token") == self.token:
            self._clear_directory(self.path)
        try:
            self.directory.rmdir()
        except OSError:
            pass


def _single_shard_writer(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("P1 run-case arguments are keyword-only")
        _, archive, run_id, _ = _archive_paths(
            bundle_root=Path(kwargs["bundle_root"]),
            config=kwargs["config"],
            case=kwargs["case"],
            shard_index=int(kwargs["shard_index"]),
            drive_root=Path(kwargs["drive_root"]),
            mode=str(kwargs["mode"]),
        )
        with _ShardLease(archive=archive, run_id=run_id):
            return function(**kwargs)

    return wrapped


@_single_shard_writer
def _run_case(
    *,
    bundle_root: Path,
    config: dict[str, Any],
    case: dict[str, Any],
    shard_index: int,
    drive_root: Path,
    mode: str,
    archive_result: bool = True,
    max_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    if max_runtime_seconds is not None and float(max_runtime_seconds) <= 0.0:
        raise ValueError("max_runtime_seconds must be positive")
    global_sample_ids = global_sample_indices(case, shard_index)
    shard_samples = len(global_sample_ids)
    scratch, archive, run_id, run_config = _archive_paths(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=shard_index,
        drive_root=drive_root,
        mode=mode,
    )
    identity = _checkpoint_identity(
        run_id=run_id,
        run_config=run_config,
        case=case,
        shard_index=shard_index,
        global_sample_ids=global_sample_ids,
    )
    if archive_result:
        existing = _verify_existing(
            archive,
            expected_run_id=run_id,
            expected_run_config=run_config,
        )
        if existing is not None:
            _cleanup_cycle_checkpoint(archive=archive, run_id=run_id)
            return {"status": "already_archived", "receipt": existing}

    smoke = mode == "smoke"
    gpu = _require_a100(smoke=smoke)
    seed = shard_seed(int(config["root_seed"]), str(case["case_id"]), int(shard_index))
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_config = dict(case["model"])
    init_mode = str(model_config.pop("init_mode"))
    if smoke:
        model_config["device"] = "cpu"
    model = classA_U1FGTN_gpu(**model_config)
    observer = P1ChernObserver(
        size=int(case["model"]["Nx"]),
        physical_cycles=int(case["run"]["cycles"]),
        global_sample_ids=global_sample_ids,
        root_seed=int(config["root_seed"]),
        center_count=int(case["observer"]["center_count"]),
        radius_fraction=float(case["observer"]["radius_fraction"]),
    )
    checkpoint = _load_cycle_checkpoint(
        archive=archive,
        run_id=run_id,
        expected_identity=identity,
        observer=observer,
    )
    resumed_from_cycle: int | None = None
    if checkpoint is None:
        completed_cycle = -1
        native_state = None
        initial_rng = _capture_rng_payload("rng_initial__")
        elapsed_before_segment = 0.0
        peak_allocated = 0
        peak_reserved = 0
        latest_checkpoint_receipt = None
    else:
        completed_cycle = int(checkpoint["completed_cycle"])
        resumed_from_cycle = completed_cycle
        native_state = checkpoint["native_state"]
        initial_rng = checkpoint["initial_rng"]
        elapsed_before_segment = float(checkpoint["cumulative_elapsed_seconds"])
        peak_allocated = int(checkpoint["gpu_peak_allocated_bytes"])
        peak_reserved = int(checkpoint["gpu_peak_reserved_bytes"])
        latest_checkpoint_receipt = checkpoint["receipt"]
        print(
            f"[P1 RESUME] {case['case_id']} shard={shard_index} "
            f"from completed cycle {completed_cycle}/{case['run']['cycles']}",
            flush=True,
        )

    if scratch.exists():
        shutil.rmtree(scratch)
    shard_root = scratch / "shards" / f"shard_{int(shard_index):03d}"
    total_cycles = int(case["run"]["cycles"])
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    segment_started = time.perf_counter()
    engine_calls = 0
    cycle_durations: list[float] = []

    def checkpoint_pause(
        *, reason: str, forecast_seconds: float | None
    ) -> dict[str, Any]:
        if latest_checkpoint_receipt is None:
            raise RuntimeError("P1 cannot pause before a durable checkpoint exists")
        return {
            "status": "checkpointed_partial",
            "bundle": BUNDLE,
            "case_id": str(case["case_id"]),
            "shard_index": int(shard_index),
            "completed_cycle": completed_cycle,
            "total_cycles": total_cycles,
            "checkpoint": latest_checkpoint_receipt,
            "checkpoint_path": latest_checkpoint_receipt["pointer_path"],
            "checkpoint_sha256": latest_checkpoint_receipt["checkpoint_sha256"],
            "elapsed_seconds_this_session": time.perf_counter() - segment_started,
            "resumed_from_cycle": resumed_from_cycle,
            "canonical_engine_calls_this_session": engine_calls,
            "forecast_seconds_for_next_cycle": forecast_seconds,
            "stop_reason": reason,
            "resumable": True,
        }

    while completed_cycle < total_cycles:
        if max_runtime_seconds is not None and cycle_durations:
            segment_elapsed = time.perf_counter() - segment_started
            forecast = (
                P1_CYCLE_RUNTIME_SAFETY_FACTOR * max(cycle_durations)
                + P1_CHECKPOINT_WRITE_BUFFER_SECONDS
            )
            if segment_elapsed + forecast >= float(max_runtime_seconds):
                return checkpoint_pause(
                    reason="next_cycle_would_exceed_max_runtime",
                    forecast_seconds=forecast,
                )
        cycle_started = time.perf_counter()
        resume_cycle = completed_cycle if completed_cycle >= 0 else None
        cycle_observer = _CycleCheckpointObserver(
            archive=archive,
            run_id=run_id,
            identity=identity,
            observer=observer,
            initial_rng=initial_rng,
            resume_cycle=resume_cycle,
            resume_native=native_state,
            elapsed_before_segment=elapsed_before_segment,
            segment_started=segment_started,
            peak_allocated_before=peak_allocated,
            peak_reserved_before=peak_reserved,
        )
        result = model.run_markov_circuit(
            cycles=1,
            samples=shard_samples,
            batch_size=shard_samples,
            init_mode=init_mode,
            frame_init=(None if native_state is None else native_state["frame"]),
            frame_ranks=(None if native_state is None else native_state["ranks"]),
            sequence=str(case["run"]["sequence"]),
            perfect_correction=bool(case["run"]["perfect_correction"]),
            postselect=False,
            postselect_probability=0.0,
            n_a=float(case["run"]["n_a"]),
            G_history=False,
            save=False,
            progress=True,
            return_data=False,
            state_representation="auto",
            native_cycle_observer=cycle_observer,
            require_no_covariance_materialization=True,
        )
        engine_calls += 1
        cycle_durations.append(time.perf_counter() - cycle_started)
        if result.get("state_representation_resolved") != "physical_frame":
            raise RuntimeError("random pure P1 did not use the occupied-frame representation")
        if int(result.get("covariance_materialization_count", -1)) != 0:
            raise RuntimeError("P1 unexpectedly materialized a dense covariance")
        if cycle_observer.completed_cycle is None or cycle_observer.latest_native is None:
            raise RuntimeError("P1 canonical one-cycle call produced no durable checkpoint")
        expected_completed = 1 if completed_cycle < 0 else completed_cycle + 1
        if int(cycle_observer.completed_cycle) != expected_completed:
            raise RuntimeError("P1 canonical one-cycle call advanced an unexpected cycle")
        completed_cycle = int(cycle_observer.completed_cycle)
        native_state = cycle_observer.latest_native
        latest_checkpoint_receipt = cycle_observer.latest_receipt
        if torch.cuda.is_available():
            peak_allocated = max(
                peak_allocated, int(torch.cuda.max_memory_allocated(model.device))
            )
            peak_reserved = max(
                peak_reserved, int(torch.cuda.max_memory_reserved(model.device))
            )
        if max_runtime_seconds is not None and completed_cycle < total_cycles:
            segment_elapsed = time.perf_counter() - segment_started
            forecast = (
                P1_CYCLE_RUNTIME_SAFETY_FACTOR * max(cycle_durations)
                + P1_CHECKPOINT_WRITE_BUFFER_SECONDS
            )
            if segment_elapsed + forecast >= float(max_runtime_seconds):
                return checkpoint_pause(
                    reason="next_cycle_would_exceed_max_runtime",
                    forecast_seconds=forecast,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    segment_elapsed = time.perf_counter() - segment_started
    elapsed = elapsed_before_segment + segment_elapsed
    shard_root.mkdir(parents=True, exist_ok=True)

    def rng_for_archive(prefix: str) -> dict[str, np.ndarray]:
        return {
            key.removeprefix(prefix): value
            for key, value in initial_rng.items()
            if key.startswith(prefix)
        }

    with (shard_root / "rng_before.npz").open("wb") as handle:
        np.savez_compressed(handle, **rng_for_archive("rng_initial__"))
    with (shard_root / "rng_after.npz").open("wb") as handle:
        np.savez_compressed(handle, **_capture_rng_payload(""))

    product = observer.save(shard_root / "p1_chern.npz", config=run_config)
    manifest = {
        "schema_version": 2,
        "status": "complete_local",
        "bundle": BUNDLE,
        "sampling_revision": SAMPLING_REVISION,
        "audit_sha256": config["audit_sha256"],
        "canonical_entry_point": ENTRY_POINT,
        "canonical_engine_sha256": run_config["canonical_engine_sha256"],
        "run_config": run_config,
        "run_config_hash": sha256_json(run_config),
        "root_seed": int(config["root_seed"]),
        "case_id": str(case["case_id"]),
        "shard_index": int(shard_index),
        "global_sample_indices": global_sample_ids,
        "shard_generator_seed": seed,
        "gpu_preflight": gpu,
        "elapsed_seconds": elapsed,
        "elapsed_seconds_this_session": segment_elapsed,
        "resumed_from_cycle": resumed_from_cycle,
        "canonical_engine_calls_this_session": engine_calls,
        "checkpoint_contract": {
            "schema": P1_CHECKPOINT_SCHEMA,
            "interval_cycles": CHECKPOINT_INTERVAL_CYCLES,
            "final_completed_cycle": completed_cycle,
            "final_checkpoint_sha256": (
                None
                if latest_checkpoint_receipt is None
                else latest_checkpoint_receipt["checkpoint_sha256"]
            ),
            "cleanup": "only_after_verified_archive",
        },
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "products": {"p1_chern": product},
        "retired_products_absent": [
            "covariance",
            "bott",
            "density",
            "entropy",
            "tangent",
            "convergence",
            "ordered_record",
            "purity",
        ],
        "source_hashes": run_config["bundle_source_hashes"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "created_unix": time.time(),
    }
    write_json_atomic(scratch / "manifest.json", manifest)
    if not archive_result:
        return {
            "status": "preflight_complete",
            "manifest": manifest,
            "scratch": str(scratch),
        }
    receipt = _archive(scratch, archive, run_id)
    verified = _verify_existing(
        archive,
        expected_run_id=run_id,
        expected_run_config=run_config,
    )
    if verified is None or verified.get("archive_sha256") != receipt["archive_sha256"]:
        raise RuntimeError("P1 archive did not pass post-commit verification")
    _cleanup_cycle_checkpoint(archive=archive, run_id=run_id)
    return {"status": "archived_to_drive", "receipt": receipt, "manifest": manifest}



def _a100_preflight(
    *,
    bundle_root: Path,
    config: dict[str, Any],
    drive_root: Path,
    max_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    original = next(
        case for case in expand_cases(config)
        if case["model"]["Nx"] == 64 and case["model"]["nshell"] == 1
    )
    if samples_per_shard(original) != L64_SHARD_SIZE:
        raise RuntimeError("P1 L64 qualification must use a one-trajectory shard")
    _, archive, run_id, run_config = _archive_paths(
        bundle_root=bundle_root,
        config=config,
        case=original,
        shard_index=0,
        drive_root=drive_root,
        mode="production",
    )
    began = time.perf_counter()
    try:
        result = _run_case(
            bundle_root=bundle_root,
            config=config,
            case=original,
            shard_index=0,
            drive_root=drive_root,
            mode="production",
            archive_result=True,
            max_runtime_seconds=max_runtime_seconds,
        )
    except torch.cuda.OutOfMemoryError as exc:
        gpu = _require_a100(smoke=False)
        total_bytes = int(gpu["total_bytes"])
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        payload = {
            "schema": A100_PREFLIGHT_SCHEMA,
            "bundle": BUNDLE,
            "sampling_revision": config["sampling_revision"],
            "audit_sha256": config["audit_sha256"],
            "case_id": original["case_id"],
            "measured_trajectories": L64_SHARD_SIZE,
            "measured_cycles_per_trajectory": 64,
            "measured_centers_per_sample_cycle": 10,
            "elapsed_seconds_before_failure": time.perf_counter() - began,
            "peak_reserved_bytes": peak_reserved,
            "gpu_total_bytes": total_bytes,
            "peak_reserved_fraction": float(peak_reserved / total_bytes),
            "projected_seconds_for_three_L64_cases": None,
            "safe": False,
            "failure": "cuda_out_of_memory",
            "failure_detail": str(exc),
            "scientific_contract_changed": False,
            "execution_contract_changed": True,
            "created_unix": time.time(),
        }
        receipt = _preflight_receipt_path(drive_root=drive_root, config=config)
        _publish_preflight_receipt(receipt, payload)
        payload["receipt_path"] = str(receipt)
        return payload
    if result.get("status") == "checkpointed_partial":
        return {
            **result,
            "preflight": True,
            "safe": None,
            "receipt_written": False,
        }
    archive_receipt = _verify_existing(
        archive,
        expected_run_id=run_id,
        expected_run_config=run_config,
    )
    if archive_receipt is None:
        raise RuntimeError("P1 qualification returned without its production archive")
    manifest = result.get("manifest") or _root_manifest_from_archive(archive)
    if manifest.get("global_sample_indices") != [0]:
        raise RuntimeError("P1 qualification archive is not L64 global sample 0")
    per_shard = float(manifest["elapsed_seconds"])
    per_cycle = per_shard / float(original["run"]["cycles"])
    total_bytes = int(manifest["gpu_preflight"]["total_bytes"])
    peak_reserved = int(manifest["gpu_peak_reserved_bytes"])
    payload = {
        "schema": A100_PREFLIGHT_SCHEMA,
        "bundle": BUNDLE,
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "case_id": original["case_id"],
        "measured_trajectories": L64_SHARD_SIZE,
        "measured_cycles_per_trajectory": 64,
        "measured_centers_per_sample_cycle": 10,
        "elapsed_seconds": per_shard,
        "measured_seconds_per_cycle": per_cycle,
        "checkpoint_interval_cycles": CHECKPOINT_INTERVAL_CYCLES,
        "peak_allocated_bytes": manifest["gpu_peak_allocated_bytes"],
        "peak_reserved_bytes": peak_reserved,
        "gpu_total_bytes": total_bytes,
        "peak_reserved_fraction": float(peak_reserved / total_bytes),
        "projected_seconds_for_three_L64_cases": 75.0 * per_shard,
        "safe": bool(peak_reserved <= 0.8 * total_bytes),
        "safety_rule": "peak_reserved_bytes <= 0.8 * gpu_total_bytes",
        "scientific_contract_changed": False,
        "execution_contract_changed": True,
        "canonical_engine_sha256": manifest["canonical_engine_sha256"],
        "bundle_source_hashes_sha256": run_config["bundle_source_hashes_sha256"],
        "bootstrap_run_config_hash": sha256_json(run_config),
        "bootstrap_shard_index": 0,
        "bootstrap_global_sample_indices": [0],
        "bootstrap_archive": archive.name,
        "bootstrap_archive_sha256": archive_receipt["archive_sha256"],
        "created_unix": time.time(),
    }
    receipt = _preflight_receipt_path(drive_root=drive_root, config=config)
    _publish_preflight_receipt(receipt, payload)
    payload["receipt_path"] = str(receipt)
    return payload

def _preflight_receipt_path(*, drive_root: Path, config: dict[str, Any]) -> Path:
    return (
        drive_root.resolve()
        / str(config["production_output_collection"])
        / str(config.get("output_bundle", BUNDLE))
        / "a100_preflight.json"
    )


def _publish_preflight_receipt(path: Path, payload: dict[str, Any]) -> None:
    if _server_commit_required(path):
        publish_json(
            _remote_committer(path),
            payload,
            path,
            replace=True,
            required_headroom_bytes=P1_REMOTE_REQUIRED_HEADROOM_BYTES,
        )
    else:
        write_json_atomic(path, payload)


def _require_safe_preflight(
    *, drive_root: Path, config: dict[str, Any], bundle_root: Path
) -> dict[str, Any]:
    path = _preflight_receipt_path(drive_root=drive_root, config=config)
    if _server_commit_required(path):
        try:
            committer = _remote_committer(path)
            payload = read_remote_json(committer, path)
            committer.path_commit_record(path)
        except RemoteCommitError as exc:
            raise RuntimeError(
                "P1 production is locked until --a100-preflight succeeds on an A100; "
                f"missing server-verified receipt {path}"
            ) from exc
    else:
        if not path.is_file():
            raise RuntimeError(
                "P1 production is locked until --a100-preflight succeeds on an A100; "
                f"missing {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    source_hashes = _source_hashes(bundle_root / "src")
    bootstrap_case = next(
        case
        for case in expand_cases(config)
        if case["model"]["Nx"] == 64 and case["model"]["nshell"] == 1
    )
    _, expected_archive, expected_run_id, expected_run_config = _archive_paths(
        bundle_root=bundle_root,
        config=config,
        case=bootstrap_case,
        shard_index=0,
        drive_root=drive_root,
        mode="production",
    )
    expected = {
        "schema": A100_PREFLIGHT_SCHEMA,
        "bundle": BUNDLE,
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "scientific_contract_changed": False,
        "execution_contract_changed": True,
        "canonical_engine_sha256": sha256_file(
            bundle_root / "src" / "classA_U1FGTN_gpu.py"
        ),
        "bundle_source_hashes_sha256": expected_run_config[
            "bundle_source_hashes_sha256"
        ],
        "bootstrap_run_config_hash": sha256_json(expected_run_config),
        "checkpoint_interval_cycles": CHECKPOINT_INTERVAL_CYCLES,
        "bootstrap_shard_index": 0,
        "bootstrap_global_sample_indices": [0],
        "safe": True,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if json_ready(payload.get(key)) != json_ready(value)
    }
    if float(payload.get("measured_seconds_per_cycle", 0.0)) <= 0.0:
        mismatches["measured_seconds_per_cycle"] = (
            payload.get("measured_seconds_per_cycle"),
            "positive",
        )
    if mismatches:
        raise RuntimeError(f"P1 A100 preflight receipt is not safe/current: {mismatches}")
    archive_name = str(payload.get("bootstrap_archive", ""))
    if not archive_name or Path(archive_name).name != archive_name:
        raise RuntimeError("P1 A100 receipt has an invalid bootstrap archive name")
    if archive_name != expected_archive.name:
        raise RuntimeError(
            "P1 A100 receipt names a bootstrap archive from a different run identity"
        )
    archive = path.parent / archive_name
    archive_receipt = _verify_existing(
        archive,
        expected_run_id=expected_run_id,
        expected_run_config=expected_run_config,
    )
    if archive_receipt is None:
        raise RuntimeError(f"P1 A100 bootstrap archive is missing: {archive}")
    if archive_receipt.get("archive_sha256") != payload.get(
        "bootstrap_archive_sha256"
    ):
        raise RuntimeError("P1 A100 bootstrap archive hash differs from the receipt")
    manifest = _root_manifest_from_archive(archive)
    manifest_expected = {
        "status": "complete_local",
        "bundle": BUNDLE,
        "sampling_revision": SAMPLING_REVISION,
        "audit_sha256": config["audit_sha256"],
        "case_id": "P1_CHERN_L64_nsh-1",
        "shard_index": 0,
        "global_sample_indices": [0],
        "canonical_engine_sha256": expected["canonical_engine_sha256"],
        "source_hashes": source_hashes,
    }
    manifest_mismatches = {
        key: (manifest.get(key), value)
        for key, value in manifest_expected.items()
        if json_ready(manifest.get(key)) != json_ready(value)
    }
    if manifest_mismatches:
        raise RuntimeError(
            "P1 A100 bootstrap manifest is not safe/current: "
            f"{manifest_mismatches}"
        )
    gpu_name = str(manifest.get("gpu_preflight", {}).get("device", ""))
    if "A100" not in gpu_name.upper():
        raise RuntimeError(
            f"P1 A100 bootstrap manifest names a non-A100 device: {gpu_name!r}"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one lean P1 Chern-dynamics shard")
    parser.add_argument(
        "--bundle-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("production", "pilot", "smoke"), default="production"
    )
    parser.add_argument("--case-id")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-cases-json", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--a100-preflight", action="store_true")
    parser.add_argument("--remote-status", action="store_true")
    parser.add_argument("--pilot-width", type=int, default=20)
    parser.add_argument("--max-runtime-seconds", type=float)
    return parser


def _print_session_checkpoint(result: dict[str, Any]) -> None:
    print(
        P1_SESSION_CHECKPOINT_PREFIX
        + json.dumps(json_ready(result), sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0.0:
        raise ValueError("--max-runtime-seconds must be positive")
    bundle_root = args.bundle_root.resolve()
    config = load_config(bundle_root)
    cases = expand_cases(config)
    if args.list_cases_json:
        print(json.dumps([
            {
                "case_id": case["case_id"],
                "shard_count": shard_count(case),
                "samples_per_shard": samples_per_shard(case),
                "case": case,
            }
            for case in cases
        ]))
        return 0
    if args.list_cases:
        for case in cases:
            print(case["case_id"])
        return 0
    if args.a100_preflight:
        try:
            current = _require_safe_preflight(
                drive_root=args.drive_root.resolve(),
                config=config,
                bundle_root=bundle_root,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            current = None
        if current is not None:
            current = dict(current)
            current["status"] = "reused_current_safe_receipt"
            current["receipt_path"] = str(
                _preflight_receipt_path(
                    drive_root=args.drive_root.resolve(), config=config
                )
            )
            print(json.dumps(current, indent=2, sort_keys=True))
            return 0
        result = _a100_preflight(
            bundle_root=bundle_root,
            config=config,
            drive_root=args.drive_root.resolve(),
            max_runtime_seconds=args.max_runtime_seconds,
        )
        if result.get("status") == "checkpointed_partial":
            _print_session_checkpoint(result)
            return 0
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result.get("safe", False):
            raise RuntimeError("P1 A100 qualification completed but was not safe")
        return 0
    selected = case_index(cases)
    case_id = args.case_id or cases[0]["case_id"]
    if case_id not in selected:
        raise KeyError(f"unknown P1 case {case_id!r}")
    case = selected[case_id]
    if args.remote_status:
        _, archive, run_id, run_config = _archive_paths(
            bundle_root=bundle_root,
            config=config,
            case=case,
            shard_index=int(args.shard_index),
            drive_root=args.drive_root.resolve(),
            mode=str(args.mode),
        )
        receipt = _verify_existing(
            archive,
            expected_run_id=run_id,
            expected_run_config=run_config,
        )
        if receipt is None:
            print(json.dumps({"exists": False, "archive": str(archive)}))
            return 0
        print(json.dumps({
            "exists": True,
            "archive": str(archive),
            "receipt": receipt,
            "manifest": _root_manifest_from_archive(archive),
        }, sort_keys=True))
        return 0
    selected_sample_ids = global_sample_indices(case, int(args.shard_index))
    preflight = {
        "bundle": BUNDLE,
        "case_id": case_id,
        "shard_index": int(args.shard_index),
        "samples": len(selected_sample_ids),
        "global_sample_indices": selected_sample_ids,
        "model": case["model"],
        "run": case["run"],
        "observer": case["observer"],
        "execution": case["execution"],
    }
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        _require_a100(smoke=args.mode == "smoke")
        return 0
    if args.mode == "production":
        _require_safe_preflight(
            drive_root=args.drive_root.resolve(),
            config=config,
            bundle_root=bundle_root,
        )
    result = _run_case(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=int(args.shard_index),
        drive_root=args.drive_root.resolve(),
        mode=str(args.mode),
        archive_result=True,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    if result.get("status") == "checkpointed_partial":
        _print_session_checkpoint(result)
    else:
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
