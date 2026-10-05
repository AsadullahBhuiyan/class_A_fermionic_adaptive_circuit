#!/usr/bin/env python3
"""Resumable pure-state occupied/empty tangent spectrum on frozen records."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fgtn.classA_U1FGTN import classA_U1FGTN


SCHEMA = "frozen_record_flux_pure_tangent_v1"
REFERENCE_SCHEMA = "frozen_record_flux_pure_tangent_reference_v1"
REFERENCE_COMPLETION_SCHEMA = "frozen_record_flux_pure_tangent_reference_completion_v1"
TASK_SCHEMA = "frozen_record_flux_pure_tangent_task_v1"
COMPLETION_SCHEMA = "frozen_record_flux_pure_tangent_completion_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
DEFAULT_CONFIG = HERE / "campaign_config.pure_tangent_v1.json"
DEFAULT_OUTPUT = HERE / "results" / "N16x20_frozen_flux_pure_tangent_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_record(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return list(json.load(handle))


def source_hashes(config_path: Path) -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "config": config_path.resolve(),
        "cpu_engine": REPO_ROOT / "src/fgtn/classA_U1FGTN.py",
        "occupied_frame": REPO_ROOT / "src/fgtn/occupied_frame.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as base:
            with gzip.GzipFile(filename="", mode="wb", fileobj=base, mtime=0) as zipped:
                zipped.write(raw)
            base.flush()
            os.fsync(base.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def parse_cpu_list(value: str) -> list[int]:
    selected: set[int] = set()
    for token in str(value).split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range {part!r}")
            selected.update(range(first, last + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("CPU list is empty")
    available = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else selected
    missing = selected - available
    if missing:
        raise ValueError(f"requested CPUs are unavailable: {sorted(missing)}")
    return sorted(selected)


def set_thread_environment(threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(threads))


def validate_config(config: dict[str, Any]) -> None:
    geometry = config["geometry"]
    dynamics = config["dynamics"]
    tangent = config["tangent"]
    if (geometry["Nx"], geometry["Ny"], geometry["cycles"]) != (16, 20, 40):
        raise ValueError("v1 geometry is locked to Nx=16, Ny=20, cycles=40")
    if geometry["nshell"] != 1 or geometry["filling_frac"] != 0.5:
        raise ValueError("v1 requires nshell=1 and half filling")
    if (geometry["alpha_1"], geometry["alpha_2"]) != (1.0, 30.0):
        raise ValueError("v1 requires alpha_1=1 and alpha_2=30")
    if not dynamics["perfect_correction"] or dynamics["postselect"]:
        raise ValueError("v1 requires perfect correction without postselection")
    if dynamics["sequence"] != "raster_y" or dynamics["dtype"] != "complex128":
        raise ValueError("v1 requires raster-y order and complex128")
    if dynamics["state_representation"] != "physical_frame":
        raise ValueError("pure tangent tracking requires physical_frame")
    if tangent["basis_mode"] != "pure_occupied_empty" or tangent["start_cycle"] != 1:
        raise ValueError("v1 requires the full pure occupied/empty cocycle from cycle 1")
    if not tangent["full_space"]:
        raise ValueError("v1 requires full-space tangent propagation")
    if (
        tangent["candidate_mode_count"] != 64
        or tangent["minimum_candidate_mode_count"] != 32
        or tangent["tracked_mode_count"] != 32
    ):
        raise ValueError(
            "v1 saves up to 64 finite candidates and requires at least 32 "
            "for the 32 tracked branches"
        )
    directions = {(row["name"], row["sigma"]) for row in config["twist"]["directions"]}
    if directions != {("ccw", 1), ("cw", -1)}:
        raise ValueError("v1 requires clockwise and counterclockwise twist directions")
    if config["twist"]["points_per_direction"] != 17:
        raise ValueError("v1 requires 17 points per direction")
    if config["references"]["mode"] != "generate_zero_twist_raster_y":
        raise ValueError("v1 requires newly generated zero-twist raster-y records")
    if config["canonical_dynamics_entry_point"] != CANONICAL_ENTRY_POINT:
        raise ValueError("canonical CPU dynamics entry point mismatch")


def reference_hash_for(arm: dict[str, Any], config_hash: str) -> str:
    return canonical_hash(
        {
            "schema": REFERENCE_SCHEMA,
            "config_hash": config_hash,
            "arm": arm,
            "twist": 0.0,
            "sequence": "raster_y",
        }
    )


def reference_paths(output_root: Path, config: dict[str, Any], arm_name: str) -> dict[str, Path]:
    root = output_root / str(config["references"]["root"]) / arm_name
    return {
        "root": root,
        "record": root / "trajectory_record.json.gz",
        "state": root / "parent_state.npz",
        "completion": root / "completion.json",
    }


def verify_reference(
    output_root: Path,
    config: dict[str, Any],
    arm: dict[str, Any],
    config_hash: str,
) -> dict[str, Any] | None:
    name = str(arm["name"])
    paths = reference_paths(output_root, config, name)
    if not all(paths[key].is_file() for key in ("record", "state", "completion")):
        return None
    expected_reference_hash = reference_hash_for(arm, config_hash)
    try:
        completion = load_json(paths["completion"])
        if completion.get("schema") != REFERENCE_COMPLETION_SCHEMA:
            return None
        if completion.get("arm") != name or completion.get("reference_hash") != expected_reference_hash:
            return None
        for key, path_key in (("trajectory_record", "record"), ("parent_state", "state")):
            record = completion["files"][key]
            path = paths[path_key]
            if record.get("name") != path.name or int(record.get("bytes", -1)) != path.stat().st_size:
                return None
            if record.get("sha256") != sha256_file(path):
                return None
        dimension = 2 * int(config["geometry"]["Nx"]) * int(config["geometry"]["Ny"])
        with np.load(paths["state"], allow_pickle=False) as saved:
            if str(saved["schema"]) != REFERENCE_SCHEMA:
                return None
            if str(saved["reference_hash"]) != expected_reference_hash:
                return None
            if str(saved["record_sha256"]) != completion["record_sha256"]:
                return None
            for key in ("G_init", "G_final"):
                if saved[key].shape != (dimension, dimension) or saved[key].dtype != np.complex128:
                    return None
            net_injection = int(saved["net_injected_charge"])
            zero_error = float(saved["zero_twist_replay_error"])
        return {
            "root": str(paths["root"]),
            "record_path": str(paths["record"]),
            "state_path": str(paths["state"]),
            "record_sha256": completion["record_sha256"],
            "state_sha256": completion["files"]["parent_state"]["sha256"],
            "reference_hash": expected_reference_hash,
            "net_injected_charge": net_injection,
            "zero_twist_replay_error": zero_error,
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def expand_tasks(config: dict[str, Any], config_hash: str, sources: dict[str, str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    count = int(config["twist"]["points_per_direction"])
    offset = float(config["twist"]["phi_offset"])
    for arm in config["arms"]:
        reference_hash = reference_hash_for(arm, config_hash)
        for direction in config["twist"]["directions"]:
            sigma = int(direction["sigma"])
            for index in range(count):
                phi = -sigma * offset + sigma * 2.0 * np.pi * index / (count - 1)
                task = {
                    "arm": str(arm["name"]),
                    "direction": str(direction["name"]),
                    "sigma": sigma,
                    "twist_index": index,
                    "phi": float(phi),
                }
                task["task_id"] = f"{task['arm']}_{task['direction']}_phi{index:03d}"
                task["task_hash"] = canonical_hash(
                    {
                        "schema": TASK_SCHEMA,
                        "config_hash": config_hash,
                        "source_hashes": sources,
                        "reference_hash": reference_hash,
                        **task,
                    }
                )
                tasks.append(task)
    if len(tasks) != 68 or len({row["task_id"] for row in tasks}) != 68:
        raise RuntimeError("the v1 task table must contain exactly 68 unique tasks")
    return tasks


def task_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    root = output_root / "tasks" / task["arm"] / task["direction"]
    stem = f"phi_{int(task['twist_index']):03d}"
    return root / f"{stem}.npz", root / f"{stem}.completion.json"


def verify_task(output_root: Path, task: dict[str, Any]) -> bool:
    result_path, completion_path = task_paths(output_root, task)
    if not result_path.is_file() or not completion_path.is_file():
        return False
    try:
        completion = load_json(completion_path)
        record = completion["result"]
        if completion.get("schema") != COMPLETION_SCHEMA:
            return False
        if completion.get("task_id") != task["task_id"] or completion.get("task_hash") != task["task_hash"]:
            return False
        if record.get("name") != result_path.name:
            return False
        if int(record.get("bytes", -1)) != result_path.stat().st_size:
            return False
        if record.get("sha256") != sha256_file(result_path):
            return False
        with np.load(result_path, allow_pickle=False) as saved:
            candidate_count = int(saved["candidate_mode_count"])
            return (
                str(saved["schema"]) == TASK_SCHEMA
                and str(saved["task_id"]) == task["task_id"]
                and str(saved["task_hash"]) == task["task_hash"]
                and 32 <= candidate_count <= 64
                and saved["candidate_pair_indices"].shape == (candidate_count, 2)
                and saved["candidate_output_occupied"].shape[1] == candidate_count
                and saved["candidate_output_empty"].shape[1] == candidate_count
                and saved["candidate_output_occupied_physical_occupation"].shape
                == (candidate_count,)
                and saved["candidate_output_empty_physical_occupation"].shape
                == (candidate_count,)
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def arm_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    return next(dict(row) for row in config["arms"] if row["name"] == name)


def make_model(config: dict[str, Any], arm: dict[str, Any], phi: float) -> classA_U1FGTN:
    geometry = config["geometry"]
    model = classA_U1FGTN(
        Nx=int(geometry["Nx"]),
        Ny=int(geometry["Ny"]),
        DW=True,
        nshell=int(geometry["nshell"]),
        filling_frac=float(geometry["filling_frac"]),
        alpha_1=float(geometry["alpha_1"]),
        alpha_2=float(geometry["alpha_2"]),
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_interval=tuple(geometry["dw_interval"]),
        dw_truncation=bool(arm["dw_truncation"]),
        twist_y=float(phi),
    )
    model.construct_OW_projectors(
        nshell=int(geometry["nshell"]),
        DW=True,
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_truncation=bool(arm["dw_truncation"]),
        twist_y=float(phi),
    )
    return model


def mode_coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)
    y = np.repeat(np.arange(ny, dtype=np.float64), 2 * nx)
    return x, y


def twist_initial_state(G_init: np.ndarray, phi: float, nx: int, ny: int) -> np.ndarray:
    _, y = mode_coordinates(nx, ny)
    phase = np.exp(1j * float(phi) * y / float(ny))
    return phase[:, None] * G_init * phase.conj()[None, :]


def physical_occupations(G: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.real(np.diag(np.asarray(G, dtype=np.complex128))))


def endpoint_charges(G: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    x, _ = mode_coordinates(nx, ny)
    occupation = physical_occupations(G)
    regions = config["regions"]
    left = (x >= int(regions["left_x_start"])) & (x < int(regions["left_x_stop_exclusive"]))
    right = (x >= int(regions["right_x_start"])) & (x < int(regions["right_x_stop_exclusive"]))
    if np.any(left & right) or not np.all(left | right):
        raise ValueError("left/right regions must be a disjoint complete partition")
    return {
        "N_left": float(occupation[left].sum()),
        "N_right": float(occupation[right].sum()),
        "N_total": float(occupation.sum()),
        "charge_by_x": np.asarray([occupation[x == value].sum() for value in range(nx)]),
    }


def _reference_run_kwargs(config: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "G_history": False,
        "progress": False,
        "cycles": int(config["geometry"]["cycles"]),
        "postselect": False,
        "postselect_probability": 0.0,
        "perfect_correction": True,
        "samples": 1,
        "parallelize_samples": False,
        "save": False,
        "sequence": "raster_y",
        "meas_slab_only": bool(arm["meas_slab_only"]),
        "random_seed": int(config["root_seed"]),
        "physical_covariance_update": str(config["dynamics"]["physical_covariance_update"]),
        "state_representation": str(config["dynamics"]["state_representation"]),
        "initial_purity_tolerance": float(config["acceptance"]["initial_purity_tolerance"]),
    }


def _generate_reference(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    arm = payload["arm"]
    config_hash = str(payload["config_hash"])
    output_root = Path(payload["output_root"])
    blas_threads = int(payload["blas_threads"])
    set_thread_environment(blas_threads)
    name = str(arm["name"])
    reference_hash = reference_hash_for(arm, config_hash)
    paths = reference_paths(output_root, config, name)
    paths["root"].mkdir(parents=True, exist_ok=True)
    initial_observer = InitialStateObserver()
    record_observer = RecordAudit(capture=True)
    started = time.perf_counter()
    with threadpool_limits(limits=blas_threads), redirect_stdout(io.StringIO()):
        result = make_model(config, arm, 0.0).run_markov_circuit(
            cycle_observer=initial_observer,
            trajectory_weight_observer=record_observer,
            **_reference_run_kwargs(config, arm),
        )
    if initial_observer.value is None:
        raise RuntimeError(f"{name} reference did not emit its cycle-zero state")
    G_init = initial_observer.value
    G_final = np.asarray(result["G_final"][0], dtype=np.complex128)
    record = record_observer.entries
    net_injection = injected_charge(record)
    initial_charge = endpoint_charges(G_init, config)
    final_charge = endpoint_charges(G_final, config)
    continuity = final_charge["N_total"] - initial_charge["N_total"] - net_injection
    if abs(continuity) > float(config["acceptance"]["charge_continuity_tolerance"]):
        raise FloatingPointError(f"{name} reference charge continuity failed: {continuity:.3e}")

    replay_audit = RecordAudit()
    with threadpool_limits(limits=blas_threads), redirect_stdout(io.StringIO()):
        replay = make_model(config, arm, 0.0).run_markov_circuit(
            G_init=G_init,
            trajectory_replay=record,
            trajectory_weight_observer=replay_audit,
            trajectory_replay_probability_tol=float(config["acceptance"]["replay_probability_tolerance"]),
            **_reference_run_kwargs(config, arm),
        )
    replay_final = np.asarray(replay["G_final"][0], dtype=np.complex128)
    zero_error = float(np.linalg.norm(replay_final - G_final, ord="fro") / G_final.shape[0])
    if zero_error > 1e-12:
        raise FloatingPointError(f"{name} zero-twist replay mismatch: {zero_error:.3e}")

    atomic_gzip_json(paths["record"], record)
    record_sha = sha256_file(paths["record"])
    atomic_npz(
        paths["state"],
        schema=np.asarray(REFERENCE_SCHEMA),
        arm=np.asarray(name),
        reference_hash=np.asarray(reference_hash),
        record_sha256=np.asarray(record_sha),
        G_init=G_init,
        G_final=G_final,
        net_injected_charge=np.asarray(net_injection, dtype=np.int64),
        initial_total_charge=np.asarray(initial_charge["N_total"]),
        final_total_charge=np.asarray(final_charge["N_total"]),
        charge_continuity_residual=np.asarray(continuity),
        zero_twist_replay_error=np.asarray(zero_error),
        zero_twist_replay_minimum_probability=np.asarray(replay_audit.minimum_selected_probability),
        elapsed_seconds=np.asarray(time.perf_counter() - started),
    )
    completion = {
        "schema": REFERENCE_COMPLETION_SCHEMA,
        "arm": name,
        "reference_hash": reference_hash,
        "record_sha256": record_sha,
        "files": {
            "trajectory_record": file_record(paths["record"]),
            "parent_state": file_record(paths["state"]),
        },
    }
    atomic_json(paths["completion"], completion)
    verified = verify_reference(output_root, config, arm, config_hash)
    if verified is None:
        raise RuntimeError(f"{name} reference completion readback failed")
    return verified


def ensure_references(
    output_root: Path,
    config: dict[str, Any],
    config_hash: str,
    blas_threads: int,
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for arm in config["arms"]:
        current = verify_reference(output_root, config, arm, config_hash)
        if current is None:
            pending.append(
                {
                    "config": config,
                    "arm": arm,
                    "config_hash": config_hash,
                    "output_root": str(output_root),
                    "blas_threads": blas_threads,
                }
            )
        else:
            verified[str(arm["name"])] = current
    if pending:
        with ProcessPoolExecutor(max_workers=len(pending)) as executor:
            futures = [executor.submit(_generate_reference, payload) for payload in pending]
            with tqdm(total=len(config["arms"]), initial=len(verified), desc="Raster-y references", unit="record") as bar:
                for future in as_completed(futures):
                    row = future.result()
                    verified[Path(row["root"]).name] = row
                    bar.update(1)
    expected = {str(row["name"]) for row in config["arms"]}
    if set(verified) != expected:
        raise RuntimeError("not every raster-y reference is verified")
    return verified


class RecordAudit:
    def __init__(self, *, capture: bool = False) -> None:
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
            self.entries.append(
                {"cycle": int(cycle), "site_id": int(site_id), "branch_events": events}
            )


class InitialStateObserver:
    def __init__(self) -> None:
        self.value: np.ndarray | None = None

    def __call__(self, *, cycle: int, G: np.ndarray, **_: Any) -> None:
        if int(cycle) == 0 and self.value is None:
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


class FinalPureTangentObserver:
    def __init__(self, *, cycles: int, phi: float, config: dict[str, Any]) -> None:
        self.cycles = int(cycles)
        self.phi = float(phi)
        self.config = config
        self.arrays: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        lyapunov_cycle: int,
        lyapunov_frame: Any,
        lyapunov_block_sizes: Any,
        lyapunov_block_core_hat: Any,
        lyapunov_block_core_log_scale: Any,
        lyapunov_block_core_null_count: Any,
        lyapunov_initial_block_basis: Any,
        lyapunov_initial_active_purity_defect: Any,
        lyapunov_min_branch_probability: Any,
        lyapunov_min_abs_born_denominator: Any,
        lyapunov_invalid_branch_count: Any,
        **_: Any,
    ) -> None:
        if int(lyapunov_cycle) != self.cycles:
            return
        frame = np.asarray(lyapunov_frame[0], dtype=np.complex128)
        block_sizes = tuple(map(int, lyapunov_block_sizes))
        if len(block_sizes) != 2 or sum(block_sizes) != frame.shape[1]:
            raise RuntimeError("invalid occupied/empty block payload")
        nx, ny = int(self.config["geometry"]["Nx"]), int(self.config["geometry"]["Ny"])
        _, y = mode_coordinates(nx, ny)
        unwind = np.exp(-1j * self.phi * y / float(ny))[:, None]
        tol = float(self.config["tangent"]["singular_tolerance"])
        parts: list[dict[str, np.ndarray]] = []
        start = 0
        for block_index, size in enumerate(block_sizes):
            stop = start + size
            core = np.asarray(lyapunov_block_core_hat[block_index][0], dtype=np.complex128)
            scale = float(lyapunov_block_core_log_scale[block_index][0])
            left, singular, right_h = np.linalg.svd(core, full_matrices=False)
            logs = np.full(singular.shape, -np.inf, dtype=np.float64)
            positive = np.isfinite(singular) & (singular > tol)
            logs[positive] = np.log(singular[positive]) + scale
            output = frame[:, start:stop] @ left
            initial_basis = np.asarray(
                lyapunov_initial_block_basis[block_index][0], dtype=np.complex128
            )
            input_vectors = initial_basis @ right_h.conj().T
            output /= np.maximum(np.linalg.norm(output, axis=0), np.finfo(float).tiny)
            input_vectors /= np.maximum(np.linalg.norm(input_vectors, axis=0), np.finfo(float).tiny)
            output = unwind * output
            input_vectors = unwind * input_vectors
            residual = np.linalg.norm(
                core @ (core.conj().T @ left) - left * (singular ** 2)[None, :], axis=0
            ) / max(float(singular[0] ** 2), np.finfo(float).tiny)
            parts.append(
                {"logs": logs, "output": output, "input": input_vectors, "residual": residual}
            )
            start = stop

        pair_rates = (parts[0]["logs"][:, None] + parts[1]["logs"][None, :]) / self.cycles
        indices = np.argwhere(np.isfinite(pair_rates))
        values = pair_rates[tuple(indices.T)]
        order = np.lexsort((indices[:, 1], indices[:, 0], np.abs(values)))
        maximum_count = int(self.config["tangent"]["candidate_mode_count"])
        minimum_count = int(self.config["tangent"]["minimum_candidate_mode_count"])
        if len(order) < minimum_count:
            raise RuntimeError(
                f"only {len(order)} finite particle-hole modes; need at least {minimum_count}"
            )
        count = min(maximum_count, len(order))
        chosen = indices[order[:count]].astype(np.int32)
        occupied_index, empty_index = chosen.T
        selected_rates = pair_rates[occupied_index, empty_index]
        occ_out = parts[0]["output"][:, occupied_index]
        emp_out = parts[1]["output"][:, empty_index]
        occ_in = parts[0]["input"][:, occupied_index]
        emp_in = parts[1]["input"][:, empty_index]
        x, _ = mode_coordinates(nx, ny)
        regions = self.config["regions"]
        left = (x >= int(regions["left_x_start"])) & (x < int(regions["left_x_stop_exclusive"]))
        right = ~left
        occ_probability = np.abs(occ_out) ** 2
        emp_probability = np.abs(emp_out) ** 2
        occ_left, occ_right = occ_probability[left].sum(axis=0), occ_probability[right].sum(axis=0)
        emp_left, emp_right = emp_probability[left].sum(axis=0), emp_probability[right].sum(axis=0)
        excitation_charge = 0.5 * ((emp_right - emp_left) - (occ_right - occ_left))
        cell_density = (
            (occ_probability + emp_probability)
            .reshape(ny, nx, 2, count)
            .sum(axis=2)
            .transpose(1, 0, 2)
            / 2.0
        )
        selected_residual = np.maximum(
            parts[0]["residual"][occupied_index], parts[1]["residual"][empty_index]
        )
        maximum_residual = float(np.max(selected_residual))
        if maximum_residual > float(self.config["acceptance"]["maximum_svd_residual"]):
            raise FloatingPointError(f"tangent SVD residual {maximum_residual:.3e} exceeds tolerance")
        physical_gram = (occ_out.conj().T @ occ_out) * (emp_out.conj().T @ emp_out)
        self.arrays = {
            "candidate_mode_count": np.asarray(count, dtype=np.int32),
            "finite_pair_mode_count": np.asarray(len(order), dtype=np.int64),
            "one_leg_logs_occupied": parts[0]["logs"],
            "one_leg_logs_empty": parts[1]["logs"],
            "candidate_pair_indices": chosen,
            "candidate_pair_rates": selected_rates,
            "candidate_effective_gaps_per_cycle": -2.0 * selected_rates,
            "candidate_output_occupied": occ_out,
            "candidate_output_empty": emp_out,
            "candidate_input_occupied": occ_in,
            "candidate_input_empty": emp_in,
            "candidate_cell_density": cell_density,
            "candidate_x_density": cell_density.sum(axis=1).T,
            "candidate_occupied_left_weight": occ_left,
            "candidate_occupied_right_weight": occ_right,
            "candidate_empty_left_weight": emp_left,
            "candidate_empty_right_weight": emp_right,
            "candidate_excitation_charge": excitation_charge,
            "candidate_svd_residual": selected_residual,
            "candidate_particle_hole_gram": physical_gram,
            "candidate_selection_boundary": np.asarray(
                abs(values[order[count]]) - abs(values[order[count - 1]])
                if len(order) > count else np.inf
            ),
            "all_pair_rate_min_abs": np.asarray(np.min(np.abs(values))),
            "block_core_null_count": np.asarray(
                [int(value[0]) for value in lyapunov_block_core_null_count], dtype=np.int64
            ),
            "initial_active_purity_defect": np.asarray(lyapunov_initial_active_purity_defect),
            "tangent_min_branch_probability": np.asarray(lyapunov_min_branch_probability[0]),
            "tangent_min_abs_born_denominator": np.asarray(lyapunov_min_abs_born_denominator[0]),
            "tangent_invalid_branch_count": np.asarray(lyapunov_invalid_branch_count[0]),
        }


def _run_task(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    task = payload["task"]
    reference = payload["reference"]
    output_root = Path(payload["output_root"])
    blas_threads = int(payload["blas_threads"])
    set_thread_environment(blas_threads)
    started = time.perf_counter()
    record_path = Path(reference["record_path"])
    state_path = Path(reference["state_path"])
    if sha256_file(record_path) != reference["record_sha256"] or sha256_file(state_path) != reference["state_sha256"]:
        raise RuntimeError(f"reference changed before {task['task_id']}")
    record = load_record(record_path)
    with np.load(state_path, allow_pickle=False) as parent:
        G_init = np.array(parent["G_init"], dtype=np.complex128, copy=True)
        net_injection = int(parent["net_injected_charge"])
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    cycles = int(config["geometry"]["cycles"])
    phi = float(task["phi"])
    twisted_initial = twist_initial_state(G_init, phi, nx, ny)
    initial_charge = endpoint_charges(twisted_initial, config)
    tangent_observer = FinalPureTangentObserver(cycles=cycles, phi=phi, config=config)
    record_audit = RecordAudit()
    arm = arm_by_name(config, task["arm"])
    with threadpool_limits(limits=blas_threads), redirect_stdout(io.StringIO()):
        result = make_model(config, arm, phi).run_markov_circuit(
            G_init=twisted_initial,
            G_history=False,
            progress=False,
            cycles=cycles,
            postselect=False,
            postselect_probability=0.0,
            perfect_correction=True,
            samples=1,
            parallelize_samples=False,
            save=False,
            sequence=str(config["dynamics"]["sequence"]),
            meas_slab_only=bool(arm["meas_slab_only"]),
            random_seed=int(config["root_seed"]),
            physical_covariance_update=str(config["dynamics"]["physical_covariance_update"]),
            state_representation=str(config["dynamics"]["state_representation"]),
            initial_purity_tolerance=float(config["acceptance"]["initial_purity_tolerance"]),
            trajectory_replay=record,
            trajectory_weight_observer=record_audit,
            trajectory_replay_probability_tol=float(config["acceptance"]["replay_probability_tolerance"]),
            lyapunov_frame_observer=tangent_observer,
            lyapunov_basis_mode=str(config["tangent"]["basis_mode"]),
            lyapunov_start_cycle=int(config["tangent"]["start_cycle"]),
            lyapunov_full_space=bool(config["tangent"]["full_space"]),
            lyapunov_track_restricted_core=True,
            lyapunov_singular_tol=float(config["tangent"]["singular_tolerance"]),
            lyapunov_failure_mode="raise",
        )
    if tangent_observer.arrays is None:
        raise RuntimeError("final tangent payload was not emitted")
    final = np.asarray(result["G_final"][0])
    if final.dtype != np.complex128:
        raise TypeError(f"final covariance must be complex128, got {final.dtype}")
    final_charge = endpoint_charges(final, config)
    continuity = final_charge["N_total"] - initial_charge["N_total"] - net_injection
    if abs(continuity) > float(config["acceptance"]["charge_continuity_tolerance"]):
        raise FloatingPointError(f"charge continuity failed for {task['task_id']}: {continuity:.3e}")
    tangent_arrays = tangent_observer.arrays
    unwound_final = twist_initial_state(final, -phi, nx, ny)
    final_projector = 0.5 * (
        unwound_final + np.eye(unwound_final.shape[0], dtype=np.complex128)
    )
    occ_output = np.asarray(tangent_arrays["candidate_output_occupied"])
    emp_output = np.asarray(tangent_arrays["candidate_output_empty"])
    occ_occupation = np.real(
        np.einsum("ik,ij,jk->k", occ_output.conj(), final_projector, occ_output, optimize=True)
    )
    emp_occupation = np.real(
        np.einsum("ik,ij,jk->k", emp_output.conj(), final_projector, emp_output, optimize=True)
    )
    occ_projector_residual = np.linalg.norm(
        final_projector @ occ_output - occ_output, axis=0
    )
    emp_projector_residual = np.linalg.norm(final_projector @ emp_output, axis=0)
    occupation_diagnostics = np.concatenate((occ_occupation, emp_occupation))
    if not np.all(np.isfinite(occupation_diagnostics)):
        raise FloatingPointError(f"nonfinite physical mode occupation for {task['task_id']}")
    bounds_tolerance = 1e-8
    if np.min(occupation_diagnostics) < -bounds_tolerance or np.max(occupation_diagnostics) > 1.0 + bounds_tolerance:
        raise FloatingPointError(f"physical mode occupation outside [0,1] for {task['task_id']}")
    tangent_arrays = {
        **tangent_arrays,
        "candidate_output_occupied_physical_occupation": occ_occupation,
        "candidate_output_empty_physical_occupation": emp_occupation,
        "candidate_output_occupied_projector_residual": occ_projector_residual,
        "candidate_output_empty_projector_residual": emp_projector_residual,
    }
    arrays: dict[str, Any] = {
        "schema": np.asarray(TASK_SCHEMA),
        "task_id": np.asarray(task["task_id"]),
        "task_hash": np.asarray(task["task_hash"]),
        "arm": np.asarray(task["arm"]),
        "direction": np.asarray(task["direction"]),
        "sigma": np.asarray(task["sigma"], dtype=np.int8),
        "twist_index": np.asarray(task["twist_index"], dtype=np.int32),
        "phi": np.asarray(phi),
        "cycles": np.asarray(cycles, dtype=np.int32),
        "record_sha256": np.asarray(reference["record_sha256"]),
        "reference_hash": np.asarray(reference["reference_hash"]),
        "N_left": np.asarray(final_charge["N_left"]),
        "N_right": np.asarray(final_charge["N_right"]),
        "N_total": np.asarray(final_charge["N_total"]),
        "N_initial_total": np.asarray(initial_charge["N_total"]),
        "net_injected_charge": np.asarray(net_injection, dtype=np.int64),
        "charge_continuity_residual": np.asarray(continuity),
        "charge_by_x": final_charge["charge_by_x"],
        "branch_log_probability": np.asarray(record_audit.total_log_probability),
        "minimum_selected_probability": np.asarray(record_audit.minimum_selected_probability),
        "elapsed_seconds": np.asarray(time.perf_counter() - started),
        **tangent_arrays,
    }
    result_path, completion_path = task_paths(output_root, task)
    atomic_npz(result_path, **arrays)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "reference_hash": reference["reference_hash"],
        "record_sha256": reference["record_sha256"],
        "result": file_record(result_path),
    }
    atomic_json(completion_path, completion)
    if not verify_task(output_root, task):
        raise RuntimeError(f"completion readback failed for {task['task_id']}")
    return {"task_id": task["task_id"], "elapsed_seconds": float(arrays["elapsed_seconds"])}


def write_manifest(
    output_root: Path,
    config_path: Path,
    config: dict[str, Any],
    config_hash: str,
    sources: dict[str, str],
    references: dict[str, dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> None:
    atomic_json(
        output_root / "manifest.json",
        {
            "schema": SCHEMA,
            "created_unix": time.time(),
            "config_path": str(config_path),
            "config_hash": config_hash,
            "source_hashes": sources,
            "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
            "scientific_contract": config,
            "references": references,
            "task_count": len(tasks),
            "claim_boundary": config["claim_boundary"],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--cpu-list")
    parser.add_argument("--blas-threads", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_config(config)
    sources = source_hashes(config_path)
    config_hash = canonical_hash({"schema": SCHEMA, "config": config, "source_hashes": sources})
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    existing_references = {
        str(arm["name"]): current
        for arm in config["arms"]
        if (current := verify_reference(output_root, config, arm, config_hash)) is not None
    }
    tasks = expand_tasks(config, config_hash, sources)
    if args.task_id:
        wanted = set(args.task_id)
        unknown = wanted - {row["task_id"] for row in tasks}
        if unknown:
            raise ValueError(f"unknown task IDs: {sorted(unknown)}")
        tasks = [row for row in tasks if row["task_id"] in wanted]
    complete = [row for row in tasks if verify_task(output_root, row)]
    pending = [row for row in tasks if not verify_task(output_root, row)]
    if args.max_tasks is not None:
        if args.max_tasks < 0:
            raise ValueError("--max-tasks must be nonnegative")
        pending = pending[: args.max_tasks]
    parallel = config["parallel"]
    cpu_text = args.cpu_list or str(parallel["cpu_list"])
    cpus = parse_cpu_list(cpu_text)
    workers = int(args.workers if args.workers is not None else parallel["workers"])
    blas_threads = int(args.blas_threads if args.blas_threads is not None else parallel["blas_threads"])
    if workers <= 0 or workers > len(cpus):
        raise ValueError(f"workers must lie in 1..{len(cpus)}")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    set_thread_environment(blas_threads)
    dashboard = {
        "campaign": config["campaign"],
        "config": str(config_path),
        "config_hash": config_hash,
        "output_root": str(output_root),
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "dtype": config["dynamics"]["dtype"],
        "geometry": config["geometry"],
        "twists": 68,
        "selected_tasks": len(tasks),
        "verified_complete": len(complete),
        "pending_selected": len(tasks) - len(complete),
        "launch_count": len(pending),
        "cpu_list": cpu_text,
        "workers": workers,
        "blas_threads": blas_threads,
        "reference_mode": config["references"]["mode"],
        "verified_references": sorted(existing_references),
        "reference_hashes": {
            str(arm["name"]): reference_hash_for(arm, config_hash)
            for arm in config["arms"]
        },
        "source_hashes": sources,
    }
    print("[pure tangent flux dashboard]", flush=True)
    print(json.dumps(dashboard, indent=2, sort_keys=True), flush=True)
    if args.report_only:
        return 0
    references = ensure_references(
        output_root, config, config_hash, blas_threads
    )
    print("[raster-y references verified]", flush=True)
    print(json.dumps(references, indent=2, sort_keys=True), flush=True)
    write_manifest(output_root, config_path, config, config_hash, sources, references, tasks)
    if not pending:
        print("[complete] every selected task is already verified", flush=True)
        return 0
    failures: list[tuple[str, str]] = []
    payloads = [
        {
            "config": config,
            "task": task,
            "reference": references[task["arm"]],
            "output_root": str(output_root),
            "blas_threads": blas_threads,
        }
        for task in pending
    ]
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
        futures = {executor.submit(_run_task, payload): payload["task"] for payload in payloads}
        with tqdm(total=len(tasks), initial=len(complete), desc="Pure tangent flux", unit="twist", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    bar.set_postfix_str(f"last={result['task_id']} {result['elapsed_seconds'] / 60:.1f}m")
                except Exception as exc:
                    failures.append((task["task_id"], repr(exc)))
                    bar.set_postfix_str(f"failed={len(failures)} last={task['task_id']}")
                bar.update(1)
    if failures:
        print(json.dumps({"failures": failures}, indent=2), flush=True)
        raise RuntimeError(f"{len(failures)} tangent tasks failed; rerun the same command to resume")
    final_complete = sum(verify_task(output_root, row) for row in tasks)
    print(f"[complete] verified={final_complete}/{len(tasks)} output={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
