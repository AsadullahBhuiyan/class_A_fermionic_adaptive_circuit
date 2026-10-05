#!/usr/bin/env python3
"""Parallel, resumable one-record frozen-flux endpoint-charge pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


SCHEMA = "frozen_record_flux_charge_pilot_v1"
REFERENCE_SCHEMA = "frozen_record_flux_charge_reference_v1"
TASK_SCHEMA = "frozen_record_flux_charge_task_v1"
COMPLETION_SCHEMA = "frozen_record_flux_charge_completion_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "src/fgtn/classA_U1FGTN.py").is_file()
            and (candidate / "PROJECT_ADMIN/REPO_POLICY.md").is_file()
        ):
            return candidate
    raise RuntimeError("could not locate the repository root")


REPOSITORY_ROOT = repository_root()
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        with temporary.open("wb") as handle:
            handle.write(compressed)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_record(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return list(json.load(handle))


def parse_cpu_list(value: str) -> list[int]:
    selected: set[int] = set()
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if start < 0 or stop < start:
                raise ValueError(f"invalid CPU range {part!r}")
            selected.update(range(start, stop + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU id {cpu}")
            selected.add(cpu)
    if not selected:
        raise ValueError("CPU list is empty")
    available = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else selected
    missing = selected - available
    if missing:
        raise ValueError(f"requested CPUs are unavailable: {sorted(missing)}")
    return sorted(selected)


def set_thread_environment(threads: int) -> None:
    if int(threads) <= 0:
        raise ValueError("BLAS thread count must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(threads))


def scientific_identity(config: dict[str, Any], source_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "root_seed": int(config["root_seed"]),
        "geometry": config["geometry"],
        "arms": config["arms"],
        "dynamics": config["dynamics"],
        "twist": config["twist"],
        "regions": config["regions"],
        "acceptance": config["acceptance"],
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "source_hashes": source_hashes,
    }


def arm_model_config(config: dict[str, Any], arm: dict[str, Any], twist: float) -> dict[str, Any]:
    geometry = config["geometry"]
    return {
        "Nx": int(geometry["Nx"]),
        "Ny": int(geometry["Ny"]),
        "DW": True,
        "nshell": int(geometry["nshell"]),
        "filling_frac": float(geometry["filling_frac"]),
        "alpha_1": float(geometry["alpha_1"]),
        "alpha_2": float(geometry["alpha_2"]),
        "trial_orbitals": str(geometry["trial_orbitals"]),
        "dw_interval": tuple(int(v) for v in geometry["dw_interval"]),
        "dw_truncation": bool(arm["dw_truncation"]),
        "twist_y": float(twist),
    }


def make_model(config: dict[str, Any], arm: dict[str, Any], twist: float) -> classA_U1FGTN:
    model_config = arm_model_config(config, arm, twist)
    model = classA_U1FGTN(**model_config)
    model.construct_OW_projectors(
        nshell=int(model_config["nshell"]),
        DW=True,
        trial_orbitals=str(model_config["trial_orbitals"]),
        dw_truncation=bool(model_config["dw_truncation"]),
        twist_y=float(twist),
    )
    return model


def run_kwargs(config: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    dynamics = config["dynamics"]
    return {
        "G_history": False,
        "progress": False,
        "cycles": int(config["geometry"]["cycles"]),
        "postselect": bool(dynamics["postselect"]),
        "postselect_probability": float(dynamics["postselect_probability"]),
        "perfect_correction": bool(dynamics["perfect_correction"]),
        "samples": 1,
        "parallelize_samples": False,
        "init_mode": str(dynamics["init_mode"]),
        "save": False,
        "sequence": str(dynamics["sequence"]),
        "meas_slab_only": bool(arm["meas_slab_only"]),
        "random_seed": int(config["root_seed"]),
        "physical_covariance_update": str(dynamics["physical_covariance_update"]),
    }


class RecordObserver:
    def __init__(self, *, capture: bool) -> None:
        self.capture = bool(capture)
        self.entries: list[dict[str, Any]] = []
        self.total_log_probability = 0.0
        self.minimum_selected_probability = 1.0

    @staticmethod
    def selected_probability(event: dict[str, Any]) -> float:
        probability = float(event["probability"])
        if event["kind"] == "measurement":
            return probability if bool(event["outcome_occupied"]) else 1.0 - probability
        expected = bool(event["expected_occupied"])
        target = bool(event["target_occupied"])
        if bool(event.get("perfect_correction")):
            return 1.0 if target == expected else 0.0
        occurred = target if expected else not target
        return probability if occurred else 1.0 - probability

    def __call__(
        self,
        *,
        cycle: int,
        site_id: int,
        branch_log_weight: float,
        branch_events: Any,
        **_: Any,
    ) -> None:
        events = [dict(event) for event in branch_events]
        self.total_log_probability += float(branch_log_weight)
        if events:
            self.minimum_selected_probability = min(
                self.minimum_selected_probability,
                *(self.selected_probability(event) for event in events),
            )
        if self.capture:
            self.entries.append({"cycle": int(cycle), "site_id": int(site_id), "branch_events": events})


class InitialStateObserver:
    def __init__(self) -> None:
        self.value: np.ndarray | None = None
        self.source_dtype: np.dtype[Any] | None = None

    def __call__(self, *, cycle: int, G: np.ndarray, **_: Any) -> None:
        if int(cycle) == 0 and self.value is None:
            self.source_dtype = np.asarray(G).dtype
            self.value = np.array(G, dtype=np.complex128, copy=True)


def injected_charge(record: Iterable[dict[str, Any]]) -> int:
    total = 0
    for site in record:
        measurements: dict[str, int] = {}
        for event in site["branch_events"]:
            channel = str(event["channel"])
            if event["kind"] == "measurement":
                measurements[channel] = int(bool(event["outcome_occupied"]))
            elif event["kind"] == "correction":
                if channel not in measurements:
                    raise ValueError(f"correction precedes measurement for {channel!r}")
                total += int(bool(event["target_occupied"])) - measurements[channel]
            else:
                raise ValueError(f"unknown record event kind {event['kind']!r}")
    return int(total)


def mode_x(nx: int, ny: int) -> np.ndarray:
    return np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)


def physical_occupations(G: np.ndarray) -> np.ndarray:
    matrix = np.asarray(G, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected square centered covariance, got {matrix.shape}")
    return 0.5 * (1.0 + np.real(np.diag(matrix)))


def endpoint_charges(G: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    x = mode_x(nx, ny)
    occupation = physical_occupations(G)
    regions = config["regions"]
    left = (x >= int(regions["left_x_start"])) & (x < int(regions["left_x_stop_exclusive"]))
    right = (x >= int(regions["right_x_start"])) & (x < int(regions["right_x_stop_exclusive"]))
    if np.any(left & right) or not np.all(left | right):
        raise ValueError("left/right regions must form a disjoint partition")
    profile = np.asarray([occupation[x == value].sum() for value in range(nx)], dtype=np.float64)
    return {
        "N_left": float(occupation[left].sum()),
        "N_right": float(occupation[right].sum()),
        "N_total": float(occupation.sum()),
        "charge_by_x": profile,
    }


def twist_initial_state(G_init: np.ndarray, twist: float, nx: int, ny: int) -> np.ndarray:
    y = np.repeat(np.arange(ny, dtype=np.float64), 2 * nx)
    phase = np.exp(1j * float(twist) * y / float(ny))
    return phase[:, None] * G_init * phase.conj()[None, :]


def reference_paths(campaign_dir: Path, arm_name: str) -> dict[str, Path]:
    root = campaign_dir / "references" / arm_name
    return {
        "root": root,
        "record": root / "trajectory_record.json.gz",
        "state": root / "parent_state.npz",
        "completion": root / "completion.json",
    }


def task_paths(campaign_dir: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    root = campaign_dir / "tasks" / task["arm"] / task["direction"]
    stem = f"phi_{int(task['twist_index']):03d}"
    return root / f"{stem}.npz", root / f"{stem}.completion.json"


def file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_path(path)}


def verify_reference(
    campaign_dir: Path,
    arm: dict[str, Any],
    reference_hash: str,
) -> dict[str, Any] | None:
    paths = reference_paths(campaign_dir, str(arm["name"]))
    if not all(paths[name].is_file() for name in ("record", "state", "completion")):
        return None
    try:
        completion = load_json(paths["completion"])
        if completion.get("schema") != COMPLETION_SCHEMA:
            return None
        if completion.get("kind") != "reference" or completion.get("reference_hash") != reference_hash:
            return None
        for key, path_key in (("trajectory_record", "record"), ("parent_state", "state")):
            declared = completion["files"][key]
            path = paths[path_key]
            if declared.get("name") != path.name:
                return None
            if int(declared.get("bytes", -1)) != path.stat().st_size:
                return None
            if declared.get("sha256") != sha256_path(path):
                return None
        with np.load(paths["state"], allow_pickle=False) as saved:
            if str(saved["reference_hash"]) != reference_hash:
                return None
            zero_error = float(saved["zero_twist_replay_error"])
            record_hash = str(saved["record_sha256"])
            net_injection = int(saved["net_injected_charge"])
        return {
            "arm": str(arm["name"]),
            "reference_hash": reference_hash,
            "record_sha256": record_hash,
            "net_injected_charge": net_injection,
            "zero_twist_replay_error": zero_error,
            "record_path": str(paths["record"]),
            "state_path": str(paths["state"]),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def verify_task_result(
    result_path: Path,
    completion_path: Path,
    task_hash: str,
    record_sha256: str,
) -> bool:
    if not result_path.is_file() or not completion_path.is_file():
        return False
    try:
        completion = load_json(completion_path)
        if (
            completion.get("schema") != COMPLETION_SCHEMA
            or completion.get("kind") != "twist_replay"
            or completion.get("task_hash") != task_hash
            or completion.get("record_sha256") != record_sha256
            or completion.get("result", {}).get("name") != result_path.name
            or int(completion.get("result", {}).get("bytes", -1)) != result_path.stat().st_size
            or completion.get("result", {}).get("sha256") != sha256_path(result_path)
        ):
            return False
        with np.load(result_path, allow_pickle=False) as saved:
            return (
                str(saved["schema"]) == TASK_SCHEMA
                and str(saved["task_hash"]) == task_hash
                and str(saved["record_sha256"]) == record_sha256
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _generate_reference_worker(payload: dict[str, Any]) -> dict[str, Any]:
    config, arm = payload["config"], payload["arm"]
    campaign_dir = Path(payload["campaign_dir"])
    reference_hash = str(payload["reference_hash"])
    blas_threads = int(payload["blas_threads"])
    set_thread_environment(blas_threads)
    paths = reference_paths(campaign_dir, str(arm["name"]))
    started = time.perf_counter()
    with threadpool_limits(limits=blas_threads):
        recorder = RecordObserver(capture=True)
        initial_observer = InitialStateObserver()
        with redirect_stdout(io.StringIO()):
            parent = make_model(config, arm, 0.0).run_markov_circuit(
                cycle_observer=initial_observer,
                trajectory_weight_observer=recorder,
                **run_kwargs(config, arm),
            )
        if initial_observer.value is None:
            raise RuntimeError("canonical engine did not emit the cycle-zero state")
        if initial_observer.source_dtype != np.dtype(np.complex128):
            raise TypeError(
                f"cycle-zero covariance must be complex128, got {initial_observer.source_dtype}"
            )
        G_init = np.asarray(initial_observer.value, dtype=np.complex128)
        raw_final = np.asarray(parent["G_final"][0])
        if raw_final.dtype != np.dtype(np.complex128):
            raise TypeError(f"final covariance must be complex128, got {raw_final.dtype}")
        G_final = np.asarray(raw_final, dtype=np.complex128)
        net_injection = injected_charge(recorder.entries)
        initial_charge = endpoint_charges(G_init, config)
        final_charge = endpoint_charges(G_final, config)
        continuity = final_charge["N_total"] - initial_charge["N_total"] - net_injection
        tolerance = float(config["acceptance"]["charge_continuity_tolerance"])
        if abs(continuity) > tolerance:
            raise FloatingPointError(
                f"reference charge continuity failed for {arm['name']}: {continuity:.3e}"
            )

        replay_observer = RecordObserver(capture=False)
        with redirect_stdout(io.StringIO()):
            replay = make_model(config, arm, 0.0).run_markov_circuit(
                G_init=G_init,
                trajectory_replay=recorder.entries,
                trajectory_weight_observer=replay_observer,
                trajectory_replay_probability_tol=float(
                    config["acceptance"]["replay_probability_tolerance"]
                ),
                **run_kwargs(config, arm),
            )
        replay_final = np.asarray(replay["G_final"][0], dtype=np.complex128)
        zero_error = float(np.linalg.norm(replay_final - G_final) / G_final.shape[0])
        if zero_error > float(
            config["acceptance"]["zero_twist_replay_frobenius_per_dimension"]
        ):
            raise FloatingPointError(
                f"zero-twist replay parity failed for {arm['name']}: {zero_error:.3e}"
            )

    atomic_gzip_json(paths["record"], recorder.entries)
    record_hash = sha256_path(paths["record"])
    atomic_npz(
        paths["state"],
        schema=np.asarray(REFERENCE_SCHEMA),
        arm=np.asarray(str(arm["name"])),
        reference_hash=np.asarray(reference_hash),
        record_sha256=np.asarray(record_hash),
        G_init=G_init,
        G_final=G_final,
        net_injected_charge=np.asarray(net_injection, dtype=np.int64),
        initial_total_charge=np.asarray(initial_charge["N_total"]),
        final_total_charge=np.asarray(final_charge["N_total"]),
        charge_continuity_residual=np.asarray(continuity),
        zero_twist_replay_error=np.asarray(zero_error),
        zero_twist_replay_minimum_probability=np.asarray(
            replay_observer.minimum_selected_probability
        ),
        elapsed_seconds=np.asarray(time.perf_counter() - started),
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "kind": "reference",
        "arm": str(arm["name"]),
        "reference_hash": reference_hash,
        "record_sha256": record_hash,
        "files": {
            "trajectory_record": file_record(paths["record"]),
            "parent_state": file_record(paths["state"]),
        },
    }
    atomic_json(paths["completion"], completion)
    return {
        "arm": str(arm["name"]),
        "reference_hash": reference_hash,
        "record_sha256": record_hash,
        "net_injected_charge": net_injection,
        "zero_twist_replay_error": zero_error,
        "record_path": str(paths["record"]),
        "state_path": str(paths["state"]),
    }


def expand_tasks(config: dict[str, Any], configuration_hash: str) -> list[dict[str, Any]]:
    twist = config["twist"]
    count = int(twist["points_per_direction"])
    if count < 2:
        raise ValueError("points_per_direction must be at least two")
    offset = float(twist["phi_offset"])
    tasks: list[dict[str, Any]] = []
    for arm in config["arms"]:
        for direction in twist["directions"]:
            sigma = int(direction["sigma"])
            if sigma not in (-1, 1):
                raise ValueError("twist direction sigma must be +1 or -1")
            for index in range(count):
                phi = -sigma * offset + sigma * 2.0 * math.pi * index / (count - 1)
                task_identity = {
                    "schema": TASK_SCHEMA,
                    "configuration_hash": configuration_hash,
                    "arm": str(arm["name"]),
                    "direction": str(direction["name"]),
                    "sigma": sigma,
                    "twist_index": index,
                    "phi": float(phi),
                    "sweep_fraction": float(sigma * index / (count - 1)),
                }
                tasks.append(
                    {
                        **task_identity,
                        "task_id": f"{arm['name']}__{direction['name']}__phi_{index:03d}",
                        "task_hash": canonical_json_sha256(task_identity),
                    }
                )
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("duplicate twist task IDs")
    return tasks


def _twist_worker(payload: dict[str, Any]) -> dict[str, Any]:
    config, arm, task = payload["config"], payload["arm"], payload["task"]
    reference = payload["reference"]
    campaign_dir = Path(payload["campaign_dir"])
    blas_threads = int(payload["blas_threads"])
    set_thread_environment(blas_threads)
    started = time.perf_counter()
    record = load_record(Path(reference["record_path"]))
    with np.load(Path(reference["state_path"]), allow_pickle=False) as parent:
        G_init = np.array(parent["G_init"], dtype=np.complex128, copy=True)
        net_injection = int(parent["net_injected_charge"])
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    phi = float(task["phi"])
    twisted_initial = twist_initial_state(G_init, phi, nx, ny)
    initial_charge = endpoint_charges(twisted_initial, config)
    observer = RecordObserver(capture=False)
    with threadpool_limits(limits=blas_threads):
        with redirect_stdout(io.StringIO()):
            replay = make_model(config, arm, phi).run_markov_circuit(
                G_init=twisted_initial,
                trajectory_replay=record,
                trajectory_weight_observer=observer,
                trajectory_replay_probability_tol=float(
                    config["acceptance"]["replay_probability_tolerance"]
                ),
                **run_kwargs(config, arm),
            )
    raw_final = np.asarray(replay["G_final"][0])
    if raw_final.dtype != np.dtype(np.complex128):
        raise TypeError(f"final covariance must be complex128, got {raw_final.dtype}")
    final = np.asarray(raw_final, dtype=np.complex128)
    final_charge = endpoint_charges(final, config)
    continuity = final_charge["N_total"] - initial_charge["N_total"] - net_injection
    if abs(continuity) > float(config["acceptance"]["charge_continuity_tolerance"]):
        raise FloatingPointError(
            f"charge continuity failed for {task['task_id']}: {continuity:.3e}"
        )
    result_path, completion_path = task_paths(campaign_dir, task)
    endpoint = int(task["twist_index"]) in (0, int(config["twist"]["points_per_direction"]) - 1)
    arrays: dict[str, Any] = {
        "schema": np.asarray(TASK_SCHEMA),
        "task_id": np.asarray(task["task_id"]),
        "task_hash": np.asarray(task["task_hash"]),
        "record_sha256": np.asarray(reference["record_sha256"]),
        "arm": np.asarray(task["arm"]),
        "direction": np.asarray(task["direction"]),
        "sigma": np.asarray(task["sigma"], dtype=np.int8),
        "twist_index": np.asarray(task["twist_index"], dtype=np.int64),
        "phi": np.asarray(phi),
        "sweep_fraction": np.asarray(
            int(task["sigma"]) * int(task["twist_index"])
            / (int(config["twist"]["points_per_direction"]) - 1)
        ),
        "N_left": np.asarray(final_charge["N_left"]),
        "N_right": np.asarray(final_charge["N_right"]),
        "N_total": np.asarray(final_charge["N_total"]),
        "N_initial_total": np.asarray(initial_charge["N_total"]),
        "net_injected_charge": np.asarray(net_injection, dtype=np.int64),
        "charge_continuity_residual": np.asarray(continuity),
        "charge_by_x": final_charge["charge_by_x"],
        "branch_log_probability": np.asarray(observer.total_log_probability),
        "minimum_selected_probability": np.asarray(observer.minimum_selected_probability),
        "elapsed_seconds": np.asarray(time.perf_counter() - started),
        "has_endpoint_covariance": np.asarray(endpoint),
    }
    if endpoint:
        arrays["G_final_endpoint"] = final
    atomic_npz(result_path, **arrays)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "kind": "twist_replay",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "record_sha256": reference["record_sha256"],
        "result": file_record(result_path),
    }
    atomic_json(completion_path, completion)
    return {
        "task_id": task["task_id"],
        "elapsed_seconds": time.perf_counter() - started,
        "minimum_selected_probability": observer.minimum_selected_probability,
    }


def reference_hash_for(
    arm: dict[str, Any], configuration_hash: str, source_hashes: dict[str, str]
) -> str:
    return canonical_json_sha256(
        {
            "schema": REFERENCE_SCHEMA,
            "configuration_hash": configuration_hash,
            "arm": arm,
            "twist": 0.0,
            "source_hashes": source_hashes,
        }
    )


def ensure_references(
    campaign_dir: Path,
    config: dict[str, Any],
    configuration_hash: str,
    source_hashes: dict[str, str],
    blas_threads: int,
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for arm in config["arms"]:
        ref_hash = reference_hash_for(arm, configuration_hash, source_hashes)
        verified = verify_reference(campaign_dir, arm, ref_hash)
        if verified is None:
            pending.append(
                {
                    "campaign_dir": str(campaign_dir),
                    "config": config,
                    "arm": arm,
                    "reference_hash": ref_hash,
                    "blas_threads": blas_threads,
                }
            )
        else:
            references[str(arm["name"])] = verified
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(pending), mp_context=context) as executor:
            futures = [executor.submit(_generate_reference_worker, item) for item in pending]
            with tqdm(total=len(config["arms"]), initial=len(references), desc="Reference records", unit="record") as bar:
                for future in as_completed(futures):
                    result = future.result()
                    references[result["arm"]] = result
                    bar.update(1)
                    bar.set_postfix_str(f"completed={len(references)}")
    if set(references) != {str(arm["name"]) for arm in config["arms"]}:
        raise RuntimeError("reference generation did not complete every arm")
    return references


def arm_by_name(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(arm["name"]): arm for arm in config["arms"]}


def run_twist_tasks(
    campaign_dir: Path,
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    workers: int,
    blas_threads: int,
) -> tuple[int, int]:
    arms = arm_by_name(config)
    pending: list[dict[str, Any]] = []
    complete = 0
    for task in tasks:
        reference = references[task["arm"]]
        result_path, completion_path = task_paths(campaign_dir, task)
        if verify_task_result(
            result_path, completion_path, task["task_hash"], reference["record_sha256"]
        ):
            complete += 1
        else:
            pending.append(
                {
                    "campaign_dir": str(campaign_dir),
                    "config": config,
                    "arm": arms[task["arm"]],
                    "task": task,
                    "reference": reference,
                    "blas_threads": blas_threads,
                }
            )
    print(
        f"[resume inventory] total={len(tasks)} complete={complete} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return complete, 0
    failures: list[tuple[str, str]] = []
    context = mp.get_context("spawn")
    max_workers = max(1, min(int(workers), len(pending)))
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        future_to_task = {
            executor.submit(_twist_worker, payload): payload["task"] for payload in pending
        }
        with tqdm(
            total=len(tasks),
            initial=complete,
            desc="Frozen-record twist replays",
            unit="point",
        ) as bar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    complete += 1
                    bar.set_postfix(
                        completed=complete,
                        failed=len(failures),
                        point_min=f"{result['elapsed_seconds'] / 60.0:.1f}",
                    )
                except Exception as exc:  # surfaced collectively after all independent tasks
                    failures.append((task["task_id"], f"{type(exc).__name__}: {exc}"))
                    bar.set_postfix(completed=complete, failed=len(failures))
                finally:
                    bar.update(1)
    if failures:
        for task_id, message in failures:
            print(f"[failed] {task_id}: {message}", file=sys.stderr, flush=True)
        raise RuntimeError(f"{len(failures)} twist tasks failed; rerun with --resume")
    return complete, len(pending)


def _endpoint_closure(
    G_start: np.ndarray,
    G_stop: np.ndarray,
    sigma: int,
    nx: int,
    ny: int,
) -> float:
    y = np.repeat(np.arange(ny, dtype=np.float64), 2 * nx)
    undo = np.exp(-1j * int(sigma) * 2.0 * math.pi * y / float(ny))
    closed = undo[:, None] * G_stop * undo.conj()[None, :]
    return float(np.linalg.norm(closed - G_start) / G_start.shape[0])


def aggregate_results(
    campaign_dir: Path,
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    configuration_hash: str,
    source_hashes: dict[str, str],
    started_unix: float,
) -> dict[str, Any]:
    raw: dict[tuple[str, str, int], dict[str, Any]] = {}
    endpoint_covariances: dict[tuple[str, str, int], np.ndarray] = {}
    for task in tasks:
        reference = references[task["arm"]]
        result_path, completion_path = task_paths(campaign_dir, task)
        if not verify_task_result(
            result_path, completion_path, task["task_hash"], reference["record_sha256"]
        ):
            raise RuntimeError(f"cannot aggregate unverified task {task['task_id']}")
        with np.load(result_path, allow_pickle=False) as saved:
            key = (task["arm"], task["direction"], int(task["twist_index"]))
            raw[key] = {
                "task_id": task["task_id"],
                "arm": task["arm"],
                "direction": task["direction"],
                "sigma": int(saved["sigma"]),
                "twist_index": int(saved["twist_index"]),
                "phi": float(saved["phi"]),
                "sweep_fraction": float(saved["sweep_fraction"]),
                "N_left": float(saved["N_left"]),
                "N_right": float(saved["N_right"]),
                "N_total": float(saved["N_total"]),
                "N_initial_total": float(saved["N_initial_total"]),
                "net_injected_charge": int(saved["net_injected_charge"]),
                "charge_continuity_residual": float(saved["charge_continuity_residual"]),
                "charge_by_x": np.array(saved["charge_by_x"], copy=True),
                "branch_log_probability": float(saved["branch_log_probability"]),
                "minimum_selected_probability": float(saved["minimum_selected_probability"]),
                "elapsed_seconds": float(saved["elapsed_seconds"]),
                "result_path": str(result_path.relative_to(campaign_dir)),
            }
            if bool(saved["has_endpoint_covariance"]):
                endpoint_covariances[key] = np.array(saved["G_final_endpoint"], copy=True)

    rows: list[dict[str, Any]] = []
    balance_tolerance = float(config["acceptance"]["regional_balance_tolerance"])
    for task in tasks:
        key = (task["arm"], task["direction"], int(task["twist_index"]))
        row = raw[key]
        baseline = raw[(task["arm"], task["direction"], 0)]
        row["delta_N_left"] = row["N_left"] - baseline["N_left"]
        row["delta_N_right"] = row["N_right"] - baseline["N_right"]
        row["q_wall"] = 0.5 * (row["delta_N_right"] - row["delta_N_left"])
        row["regional_balance_residual"] = row["delta_N_left"] + row["delta_N_right"]
        if abs(row["regional_balance_residual"]) > balance_tolerance:
            raise FloatingPointError(
                f"regional balance failed for {row['task_id']}: "
                f"{row['regional_balance_residual']:.3e}"
            )
        rows.append(row)

    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    last_index = int(config["twist"]["points_per_direction"]) - 1
    closures: dict[str, float] = {}
    for arm in config["arms"]:
        for direction in config["twist"]["directions"]:
            prefix = (str(arm["name"]), str(direction["name"]))
            start = endpoint_covariances[(*prefix, 0)]
            stop = endpoint_covariances[(*prefix, last_index)]
            closure = _endpoint_closure(start, stop, int(direction["sigma"]), nx, ny)
            closures[f"{prefix[0]}__{prefix[1]}"] = closure
            if closure > float(
                config["acceptance"]["large_gauge_closure_frobenius_per_dimension"]
            ):
                raise FloatingPointError(
                    f"large-gauge closure failed for {prefix}: {closure:.3e}"
                )

    fieldnames = [
        "task_id", "arm", "direction", "sigma", "twist_index", "phi",
        "sweep_fraction", "N_left", "N_right", "N_total", "N_initial_total",
        "net_injected_charge", "delta_N_left", "delta_N_right", "q_wall",
        "charge_continuity_residual", "regional_balance_residual",
        "branch_log_probability", "minimum_selected_probability", "elapsed_seconds",
        "result_path",
    ]
    lines: list[str] = []
    with tempfile.NamedTemporaryFile(mode="w+", newline="", encoding="utf-8", delete=False) as handle:
        temporary_csv = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
    try:
        atomic_text(campaign_dir / "charge_vs_phi.csv", temporary_csv.read_text(encoding="utf-8"))
    finally:
        temporary_csv.unlink(missing_ok=True)

    arm_names = [str(arm["name"]) for arm in config["arms"]]
    direction_names = [str(direction["name"]) for direction in config["twist"]["directions"]]
    shape = (
        len(arm_names),
        len(direction_names),
        int(config["twist"]["points_per_direction"]),
    )
    def collect(field: str) -> np.ndarray:
        result = np.empty(shape, dtype=np.float64)
        for ia, arm in enumerate(arm_names):
            for idirection, direction in enumerate(direction_names):
                for index in range(shape[2]):
                    result[ia, idirection, index] = raw[(arm, direction, index)][field]
        return result

    profiles = np.empty((*shape, nx), dtype=np.float64)
    for ia, arm in enumerate(arm_names):
        for idirection, direction in enumerate(direction_names):
            for index in range(shape[2]):
                profiles[ia, idirection, index] = raw[(arm, direction, index)]["charge_by_x"]
    atomic_npz(
        campaign_dir / "charge_vs_phi.npz",
        schema=np.asarray(SCHEMA),
        configuration_hash=np.asarray(configuration_hash),
        arms=np.asarray(arm_names),
        directions=np.asarray(direction_names),
        phi=collect("phi"),
        sweep_fraction=collect("sweep_fraction"),
        N_left=collect("N_left"),
        N_right=collect("N_right"),
        N_total=collect("N_total"),
        delta_N_left=np.asarray(
            [[[raw[(a, d, i)]["delta_N_left"] for i in range(shape[2])] for d in direction_names] for a in arm_names]
        ),
        delta_N_right=np.asarray(
            [[[raw[(a, d, i)]["delta_N_right"] for i in range(shape[2])] for d in direction_names] for a in arm_names]
        ),
        q_wall=np.asarray(
            [[[raw[(a, d, i)]["q_wall"] for i in range(shape[2])] for d in direction_names] for a in arm_names]
        ),
        charge_continuity_residual=collect("charge_continuity_residual"),
        regional_balance_residual=np.asarray(
            [[[raw[(a, d, i)]["regional_balance_residual"] for i in range(shape[2])] for d in direction_names] for a in arm_names]
        ),
        minimum_selected_probability=collect("minimum_selected_probability"),
        branch_log_probability=collect("branch_log_probability"),
        charge_by_x=profiles,
        closure_keys=np.asarray(list(closures)),
        closure_values=np.asarray(list(closures.values()), dtype=np.float64),
    )
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "campaign_id": campaign_dir.name,
        "configuration_hash": configuration_hash,
        "source_hashes": source_hashes,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "configuration": config,
        "reference_records": references,
        "task_count": len(tasks),
        "completed_task_count": len(rows),
        "large_gauge_closure_frobenius_per_dimension": closures,
        "maximum_charge_continuity_residual": max(abs(row["charge_continuity_residual"]) for row in rows),
        "maximum_regional_balance_residual": max(abs(row["regional_balance_residual"]) for row in rows),
        "minimum_selected_probability": min(row["minimum_selected_probability"] for row in rows),
        "elapsed_seconds_this_invocation": time.time() - started_unix,
        "twist_offset_rationale": {
            "rule": "phi_initial = -sigma * 1e-7",
            "purpose": "choose the quantization-favoring side of the exponentially small physical-wall avoided crossing at phi=0",
            "exact_N20x40_soft": {
                "phi0_minus_1e-7_q_plus": 0.9999999976,
                "phi0_plus_1e-7_q_minus": -0.9999999924,
                "opposite_direction_magnitude": 0.98397185
            },
            "exact_N20x40_hard_opposite_direction_magnitude": 0.98080969,
            "interpretation": "These near-integer values belong to continuously tracked exact-Hamiltonian branches. Static frozen-record replay must close after 2*pi and has no quantization acceptance gate."
        },
        "artifacts": {
            "aggregate_npz": "charge_vs_phi.npz",
            "aggregate_csv": "charge_vs_phi.csv",
            "figure_pdf": "figures/frozen_record_flux_charge.pdf",
            "figure_png": "figures/frozen_record_flux_charge.png",
        },
    }
    atomic_json(campaign_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "analyze", "report"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "campaign_config.v1.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--campaign-id", default="N16x20_frozen_flux_charge_v1")
    parser.add_argument("--cpu-list")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--blas-threads", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    if config.get("campaign") != "frozen_record_flux_charge_pilot":
        raise ValueError("configuration is not a frozen-record flux-charge pilot")
    engine_path = REPOSITORY_ROOT / "src/fgtn/classA_U1FGTN.py"
    source_hashes = {
        "canonical_cpu_engine_sha256": sha256_path(engine_path),
        "runner_sha256": sha256_path(Path(__file__).resolve()),
    }
    identity = scientific_identity(config, source_hashes)
    configuration_hash = canonical_json_sha256(identity)
    tasks = expand_tasks(config, configuration_hash)
    parallel = config["parallel"]
    cpu_text = args.cpu_list or str(parallel["cpu_list"])
    cpus = parse_cpu_list(cpu_text)
    workers = int(args.workers if args.workers is not None else parallel["workers"])
    blas_threads = int(
        args.blas_threads if args.blas_threads is not None else parallel["blas_threads"]
    )
    if workers <= 0 or workers > len(cpus):
        raise ValueError(f"workers must be in [1, {len(cpus)}]")
    set_thread_environment(blas_threads)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    campaign_dir = args.output_root.resolve() / str(args.campaign_id)
    dashboard = {
        "schema": SCHEMA,
        "command": args.command,
        "campaign_id": args.campaign_id,
        "campaign_dir": str(campaign_dir),
        "configuration_hash": configuration_hash,
        "source_hashes": source_hashes,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "geometry": config["geometry"],
        "arms": config["arms"],
        "twist": config["twist"],
        "reference_task_count": len(config["arms"]),
        "replay_task_count": len(tasks),
        "cpu_list": cpu_text,
        "resolved_cpus": cpus,
        "workers": workers,
        "blas_threads_per_worker": blas_threads,
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
    }
    print("[campaign configuration]", flush=True)
    print(json.dumps(dashboard, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if campaign_dir.exists() and not args.resume:
        material = [path for path in campaign_dir.iterdir() if path.name != "logs"]
        if material:
            raise FileExistsError(
                f"campaign already exists at {campaign_dir}; pass --resume to verify and continue"
            )
    campaign_dir.mkdir(parents=True, exist_ok=True)
    identity_path = campaign_dir / "campaign_identity.json"
    if identity_path.is_file():
        saved_identity = load_json(identity_path)
        if saved_identity.get("configuration_hash") != configuration_hash:
            raise ValueError(
                "saved campaign identity does not match the current scientific "
                "configuration/source; choose a new campaign ID"
            )
    else:
        atomic_json(
            identity_path,
            {
                "schema": SCHEMA,
                "campaign_id": str(args.campaign_id),
                "configuration_hash": configuration_hash,
                "scientific_identity": identity,
            },
        )
    atomic_json(campaign_dir / "resolved_config.json", config)
    started_unix = time.time()

    if args.command in ("all", "report"):
        references: dict[str, dict[str, Any]] = {}
        reference_complete = 0
        for arm in config["arms"]:
            reference_hash = reference_hash_for(arm, configuration_hash, source_hashes)
            verified = verify_reference(campaign_dir, arm, reference_hash)
            if verified is not None:
                references[str(arm["name"])] = verified
                reference_complete += 1
        task_complete = 0
        for task in tasks:
            reference = references.get(task["arm"])
            if reference is None:
                continue
            result_path, completion_path = task_paths(campaign_dir, task)
            if verify_task_result(
                result_path, completion_path, task["task_hash"], reference["record_sha256"]
            ):
                task_complete += 1
        print(
            f"[inventory] references={reference_complete}/{len(config['arms'])} "
            f"replays={task_complete}/{len(tasks)} pending={len(tasks)-task_complete}",
            flush=True,
        )
        if args.command == "report":
            return

    if args.command == "analyze":
        manifest_path = campaign_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing completed manifest: {manifest_path}")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "analyze_results.py"), "--campaign-dir", str(campaign_dir)],
            check=True,
        )
        return

    references = ensure_references(
        campaign_dir, config, configuration_hash, source_hashes, blas_threads
    )
    print("[reference records verified]", flush=True)
    print(json.dumps(references, indent=2, sort_keys=True), flush=True)
    complete, launched = run_twist_tasks(
        campaign_dir, config, tasks, references, workers, blas_threads
    )
    if complete != len(tasks):
        raise RuntimeError(f"only {complete}/{len(tasks)} replay tasks completed")
    manifest = aggregate_results(
        campaign_dir,
        config,
        tasks,
        references,
        configuration_hash,
        source_hashes,
        started_unix,
    )
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "analyze_results.py"), "--campaign-dir", str(campaign_dir)],
        check=True,
    )
    print(
        f"[complete] verified={complete}/{len(tasks)} launched_this_run={launched} "
        f"manifest={campaign_dir / 'manifest.json'}",
        flush=True,
    )
    print(json.dumps({
        "minimum_selected_probability": manifest["minimum_selected_probability"],
        "maximum_charge_continuity_residual": manifest["maximum_charge_continuity_residual"],
        "maximum_regional_balance_residual": manifest["maximum_regional_balance_residual"],
        "large_gauge_closure": manifest["large_gauge_closure_frobenius_per_dimension"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
