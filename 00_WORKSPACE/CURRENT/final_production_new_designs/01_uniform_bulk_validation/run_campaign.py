#!/usr/bin/env python3
"""Run the simple S100 every-cycle uniform bulk validation campaign."""

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

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = BUNDLE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from compact_observer import (  # noqa: E402
    OBSERVER_SCHEMA,
    CompactChernChargeObserver,
)


BUNDLE = "01_uniform_bulk_validation"
RESULT_SCHEMA = "uniform_bulk_chern_charge_result_v1"
COMPLETION_SCHEMA = "uniform_bulk_task_completion_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
EXPECTED_REVISION = "uniform_perfect_correction_40cycle_s100_maxl24_batched_v1"
EXPECTED_SIZES = (12, 16, 20, 24)
EXPECTED_NSHELLS = (1, 2, None)
EXPECTED_SAMPLES = 100
EXPECTED_ROOT_SEED = 2026090201
EXPECTED_BATCH_SIZES = {12: 100, 16: 100, 20: 50, 24: 25}
SOURCE_FILES = (
    "run_campaign.py",
    "compact_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)


@dataclass(frozen=True)
class Task:
    size: int
    nshell: int | None
    batch_index: int
    sample_start: int
    sample_stop: int
    task_id: str
    seed: int

    @property
    def shell_tag(self) -> str:
        return shell_tag(self.nshell)

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def global_sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))


def shell_tag(nshell: int | None) -> str:
    return "dense" if nshell is None else str(int(nshell))


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


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "sampling_revision",
        "root_seed",
        "sizes",
        "nshell_values",
        "samples_per_case",
        "batch_size_by_L",
        "cycles",
        "device",
        "dtype",
        "protocol",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing fields: {missing}")

    normalized = json.loads(json.dumps(config))
    if normalized["sampling_revision"] != EXPECTED_REVISION:
        raise ValueError(f"sampling_revision must be {EXPECTED_REVISION!r}")
    if int(normalized["root_seed"]) != EXPECTED_ROOT_SEED:
        raise ValueError(f"root_seed must be {EXPECTED_ROOT_SEED}")
    if tuple(int(value) for value in normalized["sizes"]) != EXPECTED_SIZES:
        raise ValueError(f"sizes must be {list(EXPECTED_SIZES)}")
    nshells = tuple(
        None if value is None else int(value) for value in normalized["nshell_values"]
    )
    if nshells != EXPECTED_NSHELLS:
        raise ValueError("nshell_values must be [1, 2, null]")
    if int(normalized["samples_per_case"]) != EXPECTED_SAMPLES:
        raise ValueError(f"samples_per_case must be {EXPECTED_SAMPLES}")
    batch_sizes = {
        int(size): int(value)
        for size, value in normalized["batch_size_by_L"].items()
    }
    if batch_sizes != EXPECTED_BATCH_SIZES:
        raise ValueError(f"batch_size_by_L must be {EXPECTED_BATCH_SIZES}")
    if int(normalized["cycles"]) != 40:
        raise ValueError("cycles must be 40")
    if normalized["device"] != "cuda:0" or normalized["dtype"] != "complex128":
        raise ValueError("production requires device='cuda:0' and dtype='complex128'")

    protocol = normalized["protocol"]
    expected_protocol = {
        "DW": False,
        "alpha_1": 1.0,
        "alpha_2": 1.0,
        "filling_frac": 0.5,
        "trial_orbitals": "X",
        "init_mode": "default",
        "sequence": "random",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "meas_slab_only": False,
    }
    if protocol != expected_protocol:
        raise ValueError(
            "protocol must remain the locked uniform topological, DW-off, "
            "pure-state perfect-correction contract"
        )
    return normalized


def config_hash(config: dict[str, Any]) -> str:
    return _sha256_bytes(_json_bytes(validate_config(config)))


def task_seed(
    root_seed: int,
    *,
    size: int,
    nshell: int | None,
    batch_index: int,
    sample_start: int,
    sample_stop: int,
) -> int:
    label = (
        f"{int(root_seed)}|L={int(size)}|nshell={shell_tag(nshell)}|"
        f"batch={int(batch_index)}|samples={int(sample_start)}:{int(sample_stop)}"
    )
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def expand_tasks(config: dict[str, Any]) -> list[Task]:
    config = validate_config(config)
    tasks: list[Task] = []
    for size in config["sizes"]:
        batch_size = int(config["batch_size_by_L"][str(int(size))])
        for nshell in config["nshell_values"]:
            for batch_index, sample_start in enumerate(
                range(0, int(config["samples_per_case"]), batch_size)
            ):
                sample_stop = min(
                    sample_start + batch_size, int(config["samples_per_case"])
                )
                tag = shell_tag(nshell)
                task_id = (
                    f"L{int(size):02d}_nsh-{tag}_batch-{batch_index:03d}_"
                    f"samples-{sample_start:03d}-{sample_stop - 1:03d}"
                )
                tasks.append(
                    Task(
                        size=int(size),
                        nshell=None if nshell is None else int(nshell),
                        batch_index=batch_index,
                        sample_start=sample_start,
                        sample_stop=sample_stop,
                        task_id=task_id,
                        seed=task_seed(
                            int(config["root_seed"]),
                            size=int(size),
                            nshell=nshell,
                            batch_index=batch_index,
                            sample_start=sample_start,
                            sample_stop=sample_stop,
                        ),
                    )
                )
    if len(tasks) != 24 or len({task.task_id for task in tasks}) != 24:
        raise RuntimeError("production task expansion must contain 24 unique batches")
    if len({task.seed for task in tasks}) != 24:
        raise RuntimeError("production task seeds must be unique")
    if sum(task.sample_count for task in tasks) != 12 * EXPECTED_SAMPLES:
        raise RuntimeError("production batches must cover exactly 1,200 trajectories")
    return tasks


def task_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = output_root / "results" / f"L{task.size:02d}" / f"nsh-{task.shell_tag}"
    stem = (
        f"batch_{task.batch_index:03d}_"
        f"samples_{task.sample_start:03d}-{task.sample_stop - 1:03d}"
    )
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def _expected_completion_identity(
    *, task: Task, config_sha256: str, hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": EXPECTED_REVISION,
        "task_id": task.task_id,
        "size": task.size,
        "nshell": task.nshell,
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
    result_exists = result_path.is_file()
    completion_exists = completion_path.is_file()
    if not result_exists and not completion_exists:
        return False, "missing result/completion pair"
    if not result_exists or not completion_exists:
        return False, "incomplete result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion JSON: {exc}"
    expected = _expected_completion_identity(
        task=task, config_sha256=config_sha256, hashes=hashes
    )
    for key, value in expected.items():
        if completion.get(key) != value:
            return False, f"completion identity mismatch: {key}"
    if completion.get("result_filename") != result_path.name:
        return False, "completion result filename mismatch"
    try:
        actual_bytes = result_path.stat().st_size
        actual_sha256 = sha256_file(result_path)
    except OSError as exc:
        return False, f"result readback failed: {exc}"
    if int(completion.get("result_bytes", -1)) != actual_bytes:
        return False, "result byte count mismatch"
    if completion.get("result_sha256") != actual_sha256:
        return False, "result checksum mismatch"
    return True, "verified"


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
            f"production requires 40-GB-class GPU memory, found {total_bytes / 1024**3:.2f} GiB"
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
    """Copy, DriveFS-readback verify, and atomically publish one stable file."""

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    expected_bytes = local_path.stat().st_size
    expected_sha256 = sha256_file(local_path)
    try:
        shutil.copyfile(local_path, temporary)
        observed_bytes = temporary.stat().st_size
        observed_sha256 = sha256_file(temporary)
        if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
            raise OSError(
                f"Drive temporary readback mismatch for {temporary}: "
                f"bytes={observed_bytes}/{expected_bytes}, "
                f"sha256={observed_sha256}/{expected_sha256}"
            )
        os.replace(temporary, final_path)
        if final_path.stat().st_size != expected_bytes:
            raise OSError(f"Drive final byte-count readback failed for {final_path}")
        if sha256_file(final_path) != expected_sha256:
            raise OSError(f"Drive final checksum readback failed for {final_path}")
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


def build_model(config: dict[str, Any], *, size: int, nshell: int | None) -> Any:
    protocol = config["protocol"]
    model = classA_U1FGTN_gpu(
        Nx=size,
        Ny=size,
        DW=protocol["DW"],
        nshell=nshell,
        filling_frac=protocol["filling_frac"],
        alpha_1=protocol["alpha_1"],
        alpha_2=protocol["alpha_2"],
        trial_orbitals=protocol["trial_orbitals"],
        dw_truncation=False,
        device=config["device"],
        dtype=config["dtype"],
        backend="dense" if nshell is None else "local",
    )
    if model.DW or model.dtype != torch.complex128:
        raise RuntimeError("constructed model violates DW-off/complex128 contract")
    return model


def run_batch(
    *, model: Any, config: dict[str, Any], task: Task
) -> tuple[dict[str, np.ndarray], float]:
    np.random.seed(task.seed % (2**32))
    torch.manual_seed(task.seed)
    torch.cuda.manual_seed_all(task.seed)
    cycles = int(config["cycles"])
    observer = CompactChernChargeObserver(
        size=task.size, physical_cycles=cycles, samples=task.sample_count
    )
    started = time.monotonic()
    result = model.run_markov_circuit(
        G_history=False,
        progress=True,
        cycles=cycles,
        postselect=config["protocol"]["postselect"],
        postselect_probability=config["protocol"]["postselect_probability"],
        perfect_correction=config["protocol"]["perfect_correction"],
        samples=task.sample_count,
        init_mode=config["protocol"]["init_mode"],
        save=False,
        n_a=0.5,
        sequence=config["protocol"]["sequence"],
        meas_slab_only=config["protocol"]["meas_slab_only"],
        batch_size=task.sample_count,
        return_data=False,
        state_representation="auto",
        native_cycle_observer=observer,
        track_choi=False,
        return_native_state=False,
        require_no_covariance_materialization=True,
    )
    torch.cuda.synchronize(model.device)
    elapsed = time.monotonic() - started
    if bool(result.get("choi_tracked", False)):
        raise RuntimeError("canonical engine unexpectedly enabled Choi tracking")
    payload = observer.payload()
    payload.update(
        {
            "schema": np.asarray(RESULT_SCHEMA),
            "bundle": np.asarray(BUNDLE),
            "sampling_revision": np.asarray(EXPECTED_REVISION),
            "task_id": np.asarray(task.task_id),
            "size": np.asarray(task.size, dtype=np.int64),
            "nshell": np.asarray(-1 if task.nshell is None else task.nshell, dtype=np.int64),
            "batch_index": np.asarray(task.batch_index, dtype=np.int64),
            "sample_start": np.asarray(task.sample_start, dtype=np.int64),
            "sample_stop": np.asarray(task.sample_stop, dtype=np.int64),
            "global_sample_indices": np.asarray(
                task.global_sample_indices, dtype=np.int64
            ),
            "batch_seed": np.asarray(task.seed, dtype=np.int64),
            "elapsed_seconds": np.asarray(elapsed, dtype=np.float64),
        }
    )
    return payload, elapsed


def _save_task(
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

    completion = _expected_completion_identity(
        task=task, config_sha256=config_sha256, hashes=hashes
    )
    completion.update(
        {
            "result_filename": result_path.name,
            "result_bytes": published["bytes"],
            "result_sha256": published["sha256"],
            "cycles_saved": int(np.asarray(payload["cycles"]).size),
            "samples_saved": task.sample_count,
            "elapsed_seconds": float(elapsed_seconds),
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    local_completion = task_scratch / "completion.json"
    _write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    verified, reason = verified_complete(
        output_root=output_root,
        task=task,
        config_sha256=config_sha256,
        hashes=hashes,
    )
    if not verified:
        raise OSError(f"published task failed final verification: {reason}")
    shutil.rmtree(task_scratch)


def run_campaign(
    *,
    config: dict[str, Any],
    output_root: Path,
    scratch_root: Path,
    report_only: bool = False,
    max_new_tasks: int | None = None,
) -> dict[str, Any]:
    config = validate_config(config)
    tasks = expand_tasks(config)
    hashes = source_hashes()
    config_sha256 = config_hash(config)
    inventory: dict[str, tuple[bool, str]] = {
        task.task_id: verified_complete(
            output_root=output_root,
            task=task,
            config_sha256=config_sha256,
            hashes=hashes,
        )
        for task in tasks
    }
    completed = sum(int(value[0]) for value in inventory.values())
    invalid = sum(
        int((not value[0]) and value[1] != "missing result/completion pair")
        for value in inventory.values()
    )
    pending_tasks = [task for task in tasks if not inventory[task.task_id][0]]
    print(
        json.dumps(
            {
                "bundle": BUNDLE,
                "sampling_revision": config["sampling_revision"],
                "canonical_entry_point": CANONICAL_ENTRY_POINT,
                "output_root": str(output_root),
                "scratch_root": str(scratch_root),
                "configuration": config,
                "config_sha256": config_sha256,
                "source_hashes": hashes,
                "workload": {
                    "cases": 18,
                    "samples_per_case": 100,
                    "trajectories": 18 * EXPECTED_SAMPLES,
                    "batch_size_by_L": EXPECTED_BATCH_SIZES,
                    "tasks": len(tasks),
                    "completed": completed,
                    "pending": len(pending_tasks),
                    "invalid_or_partial": invalid,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if report_only:
        return {
            "status": "report_only",
            "total": len(tasks),
            "completed": completed,
            "pending": len(pending_tasks),
            "invalid_or_partial": invalid,
        }

    gpu = validate_a100()
    local_free = _check_space(scratch_root, required_bytes=2 * 1024**3, label="local")
    drive_free = _check_space(output_root, required_bytes=1024**3, label="Drive")
    print(
        "[preflight] "
        + json.dumps(
            {
                "gpu": gpu,
                "dtype": config["dtype"],
                "local_free_gib": local_free / 1024**3,
                "drive_free_gib": drive_free / 1024**3,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    failed = 0
    new_completed = 0
    pending_count = len(pending_tasks)
    bar = tqdm(
        total=len(tasks),
        initial=completed,
        desc="uniform bulk campaign",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    bar.set_postfix(
        completed=completed,
        skipped=completed,
        pending=pending_count,
        failed=failed,
    )
    try:
        grouped: dict[tuple[int, int | None], list[Task]] = {}
        for task in pending_tasks:
            grouped.setdefault((task.size, task.nshell), []).append(task)
        for size in config["sizes"]:
            for nshell in config["nshell_values"]:
                group = grouped.get((int(size), nshell), [])
                if not group:
                    continue
                model = build_model(config, size=int(size), nshell=nshell)
                for task in group:
                    if max_new_tasks is not None and new_completed >= max_new_tasks:
                        summary = {
                            "status": "partial_limit_reached",
                            "total": len(tasks),
                            "completed": completed + new_completed,
                            "skipped": completed,
                            "pending": pending_count,
                            "failed": failed,
                        }
                        print("[campaign summary] " + json.dumps(summary), flush=True)
                        return summary
                    bar.set_description(
                        f"L={task.size} nsh={task.shell_tag} "
                        f"samples={task.sample_start:03d}-{task.sample_stop - 1:03d}"
                    )
                    try:
                        payload, elapsed = run_batch(
                            model=model, config=config, task=task
                        )
                        _save_task(
                            output_root=output_root,
                            scratch_root=scratch_root,
                            task=task,
                            payload=payload,
                            elapsed_seconds=elapsed,
                            config_sha256=config_sha256,
                            hashes=hashes,
                        )
                    except Exception:
                        failed += 1
                        bar.set_postfix(
                            completed=completed + new_completed,
                            skipped=completed,
                            pending=pending_count,
                            failed=failed,
                        )
                        raise
                    new_completed += 1
                    pending_count -= 1
                    bar.update(1)
                    bar.set_postfix(
                        completed=completed + new_completed,
                        skipped=completed,
                        pending=pending_count,
                        failed=failed,
                    )
                del model
                torch.cuda.empty_cache()
    finally:
        bar.close()

    summary = {
        "status": "complete",
        "total": len(tasks),
        "completed": completed + new_completed,
        "skipped": completed,
        "pending": pending_count,
        "failed": failed,
    }
    print("[campaign summary] " + json.dumps(summary), flush=True)
    if summary["completed"] != len(tasks) or pending_count != 0 or failed != 0:
        raise RuntimeError(f"campaign ended without full completion: {summary}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/content/uniform_bulk_validation_scratch"),
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-tasks", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        raise ValueError("--max-new-tasks must be nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_campaign(
        config=config,
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        report_only=bool(args.report_only),
        max_new_tasks=args.max_new_tasks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

