#!/usr/bin/env python3
"""CPU-only V0/V1 validation campaign and static GPU audit.

All physical trajectories enter through ``classA_U1FGTN.run_markov_circuit``.
The GPU implementation is inspected but never imported or executed here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics.response import unit_cell_reset


ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"
SCHEDULES = ("random", "raster_y", "reverse_raster_y")
CHANNELS = ("Ap", "Am", "Bp", "Bm")
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}
EXPECTED_OCCUPIED = {"Ap": False, "Am": True, "Bp": False, "Bm": True}
FLOAT_EPS = float(np.finfo(np.float64).eps)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v0_prerequisite(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("status") != "CPU_PASS / GPU_NOT_RUN":
        raise ValueError(
            f"V1 prerequisite is not a passing CPU-only V0 summary: {resolved}"
        )
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "status": payload["status"],
        "gpu_status": payload.get("gpu_audit_status"),
    }


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def git_metadata() -> dict[str, Any]:
    def run(*command: str) -> str:
        result = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--short")),
    }


def roundoff_tolerance(dimension: int, operations: int = 1) -> float:
    """Conservative recorded float64 accumulation bound."""
    return float(max(1.0e-12, 2048.0 * FLOAT_EPS * dimension * math.sqrt(max(1, operations))))


def model_for(nx: int, ny: int, *, twist_y: float = 0.0) -> classA_U1FGTN:
    model = classA_U1FGTN(
        nx,
        ny,
        DW=True,
        nshell=1,
        alpha_1=1,
        alpha_2=30,
        trial_orbitals="X",
        dw_truncation=True,
        twist_y=twist_y,
    )
    model.construct_OW_projectors(
        nshell=1,
        DW=True,
        trial_orbitals="X",
        dw_truncation=True,
        twist_y=twist_y,
    )
    return model


def base_run_kwargs(*, cycles: int, seed: int, sequence: str, init_mode: str) -> dict[str, Any]:
    return {
        "G_history": False,
        "progress": False,
        "cycles": int(cycles),
        "samples": 1,
        "parallelize_samples": False,
        "init_mode": init_mode,
        "save": False,
        "n_a": 0.5,
        "sequence": sequence,
        "meas_slab_only": True,
        "random_seed": int(seed),
        "perfect_correction": True,
        "postselect": False,
    }


def charge_map(G: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    diagonal = 0.5 * (1.0 + np.real(np.diag(np.asarray(G))))
    return diagonal.reshape(2, nx, ny, order="F").sum(axis=0)


def covariance_health(G: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(G, dtype=np.complex128)
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    C = 0.5 * (hermitian + np.eye(matrix.shape[0], dtype=np.complex128))
    charge = float(np.trace(C).real)
    charge_variance = float(charge - np.sum(np.abs(C) ** 2).real)
    return {
        "hermiticity_residual": float(np.linalg.norm(matrix - matrix.conj().T, ord="fro")),
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "spectral_bound_violation": float(
            max(0.0, -1.0 - eigenvalues[0], eigenvalues[-1] - 1.0)
        ),
        "total_charge": charge,
        "charge_variance": charge_variance,
    }


class CompactEventRecorder:
    def __init__(self, *, cycles: int, site_ids: Iterable[int], replay: bool = False):
        self.cycles = int(cycles)
        self.site_ids = np.asarray(sorted(int(value) for value in site_ids), dtype=np.int64)
        self.offset = {int(value): index for index, value in enumerate(self.site_ids)}
        shape = (self.cycles, self.site_ids.size)
        self.visit_order = np.full(shape, -1, dtype=np.int32)
        self.outcome_bits = np.zeros(shape, dtype=np.uint8)
        self.correction_bits = np.zeros(shape, dtype=np.uint8)
        self.transfer = np.zeros(shape + (4,), dtype=np.int8)
        self.success_probability = np.full(shape + (4,), np.nan, dtype=np.float32)
        self.branch_log_weight = np.full(shape, np.nan, dtype=np.float64)
        self._visit_count = np.zeros((self.cycles,), dtype=np.int32)
        self.entries: list[dict[str, Any]] | None = [] if replay else None

    def load(self, payload: dict[str, np.ndarray]) -> None:
        old_cycles = min(self.cycles, int(payload["visit_order"].shape[0]))
        if not np.array_equal(payload["site_ids"], self.site_ids):
            raise ValueError("Saved event site_ids do not match this run.")
        for name in (
            "visit_order",
            "outcome_bits",
            "correction_bits",
            "transfer",
            "success_probability",
            "branch_log_weight",
        ):
            getattr(self, name)[:old_cycles] = payload[name][:old_cycles]
        for cycle in range(old_cycles):
            valid = self.visit_order[cycle] >= 0
            self._visit_count[cycle] = int(np.count_nonzero(valid))

    def __call__(self, *, cycle: int, site_id: int, branch_events: Iterable[dict[str, Any]], branch_log_weight: float, **_: Any) -> None:
        cycle_index = int(cycle) - 1
        site_offset = self.offset[int(site_id)]
        visit = int(self._visit_count[cycle_index])
        if self.visit_order[cycle_index, site_offset] >= 0:
            raise RuntimeError(f"Duplicate visit at cycle={cycle}, site_id={site_id}.")
        self.visit_order[cycle_index, site_offset] = visit
        self._visit_count[cycle_index] += 1
        events = [dict(event) for event in branch_events]
        for event in events:
            channel = str(event["channel"])
            channel_index = CHANNEL_INDEX[channel]
            if event["kind"] == "measurement":
                occupied = bool(event["outcome_occupied"])
                if occupied:
                    self.outcome_bits[cycle_index, site_offset] |= np.uint8(1 << channel_index)
                probability = float(event["probability"])
                self.success_probability[cycle_index, site_offset, channel_index] = (
                    probability if occupied else 1.0 - probability
                )
            elif event["kind"] == "correction":
                expected = bool(event["expected_occupied"])
                target = bool(event["target_occupied"])
                if target == expected:
                    self.correction_bits[cycle_index, site_offset] |= np.uint8(1 << channel_index)
                    self.transfer[cycle_index, site_offset, channel_index] = 1 if expected else -1
        self.branch_log_weight[cycle_index, site_offset] = float(branch_log_weight)
        if self.entries is not None:
            self.entries.append(
                {"cycle": int(cycle), "site_id": int(site_id), "branch_events": events}
            )

    def validate_complete(self, completed_cycles: int | None = None) -> None:
        count = self.cycles if completed_cycles is None else int(completed_cycles)
        expected = int(self.site_ids.size)
        actual = self._visit_count[:count]
        if not np.all(actual == expected):
            bad = np.flatnonzero(actual != expected)
            raise RuntimeError(f"Incomplete or duplicate schedule cycles: {bad.tolist()}.")

    def ordered_site_ids(self) -> np.ndarray:
        result = np.full_like(self.visit_order, -1, dtype=np.int64)
        for cycle in range(self.cycles):
            valid = self.visit_order[cycle] >= 0
            result[cycle, self.visit_order[cycle, valid]] = self.site_ids[valid]
        return result

    def payload(self) -> dict[str, np.ndarray]:
        return {
            "site_ids": self.site_ids,
            "visit_order": self.visit_order,
            "ordered_site_ids": self.ordered_site_ids(),
            "outcome_bits": self.outcome_bits,
            "correction_bits": self.correction_bits,
            "transfer": self.transfer,
            "success_probability": self.success_probability,
            "branch_log_weight": self.branch_log_weight,
        }


class CycleRecorder:
    def __init__(self, *, cycles: int, nx: int, ny: int, snapshot_cycles: Iterable[int] = ()):
        self.cycles = int(cycles)
        self.nx = int(nx)
        self.ny = int(ny)
        self.cycle = np.arange(self.cycles + 1, dtype=np.int32)
        self.total_charge = np.full(self.cycles + 1, np.nan)
        self.charge_variance = np.full(self.cycles + 1, np.nan)
        self.successive_delta = np.full(self.cycles + 1, np.nan)
        self.column_charge = np.full((self.cycles + 1, self.nx), np.nan)
        self.snapshot_cycles = {int(value) for value in snapshot_cycles}
        self.snapshots: dict[int, np.ndarray] = {}
        self._previous: np.ndarray | None = None

    def load(self, payload: dict[str, np.ndarray]) -> None:
        old = min(self.cycles + 1, int(payload["total_charge"].shape[0]))
        for name in ("total_charge", "charge_variance", "successive_delta", "column_charge"):
            getattr(self, name)[:old] = payload[name][:old]

    def set_previous(self, G: np.ndarray) -> None:
        self._previous = np.asarray(G, dtype=np.complex128).copy()

    def __call__(self, *, cycle: int, G: np.ndarray, **_: Any) -> None:
        index = int(cycle)
        matrix = np.asarray(G, dtype=np.complex128)
        C = 0.5 * (matrix + np.eye(matrix.shape[0], dtype=np.complex128))
        charge = float(np.trace(C).real)
        self.total_charge[index] = charge
        self.charge_variance[index] = float(charge - np.sum(np.abs(C) ** 2).real)
        self.column_charge[index] = np.sum(charge_map(matrix, nx=self.nx, ny=self.ny), axis=1)
        if self._previous is not None:
            self.successive_delta[index] = float(
                np.linalg.norm(matrix - self._previous, ord="fro") / math.sqrt(matrix.shape[0])
            )
        self._previous = matrix.copy()
        if index in self.snapshot_cycles:
            self.snapshots[index] = matrix.copy()

    def payload(self) -> dict[str, np.ndarray]:
        return {
            "cycle": self.cycle,
            "total_charge": self.total_charge,
            "charge_variance": self.charge_variance,
            "successive_delta": self.successive_delta,
            "column_charge": self.column_charge,
        }


def checkpoint_paths(root: Path) -> tuple[Path, Path]:
    return root / "checkpoint.json", root / "checkpoint_G.npz"


def save_checkpoint(root: Path, state: dict[str, Any]) -> None:
    metadata_path, covariance_path = checkpoint_paths(root)
    state = copy.deepcopy(state)
    covariance = np.asarray(state.pop("G"), dtype=np.complex128)
    state["last_ordered_site_ids"] = np.asarray(
        state["last_ordered_site_ids"], dtype=np.int64
    ).tolist()
    save_npz_atomic(covariance_path, G=covariance)
    state["covariance_sha256"] = file_sha256(covariance_path)
    write_json_atomic(metadata_path, state)


def load_checkpoint(root: Path) -> dict[str, Any] | None:
    metadata_path, covariance_path = checkpoint_paths(root)
    if not metadata_path.exists() or not covariance_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("covariance_sha256") != file_sha256(covariance_path):
        raise RuntimeError(f"Checkpoint covariance hash mismatch in {root}.")
    with np.load(covariance_path, allow_pickle=False) as payload:
        metadata["G"] = payload["G"].astype(np.complex128, copy=True)
    metadata.pop("covariance_sha256", None)
    return metadata


def active_site_ids(model: classA_U1FGTN) -> np.ndarray:
    helper = model._sequence_helper("raster_y", skip_trivial=True)
    return np.asarray(
        [int(x) + model.Nx * int(y) for x, y in helper["coords_for_len"]],
        dtype=np.int64,
    )


def strip_coefficients(model: classA_U1FGTN, G: np.ndarray) -> dict[str, Any]:
    x_min, x_max = sorted(int(value) for value in model.DW_loc)
    if model.Ny < 12:
        ay_values = np.arange(1, model.Ny // 2 + 1, dtype=np.int64)
    else:
        ay_values = np.unique(
            np.clip(np.asarray((4, 8, 12, 16, 20, model.Ny // 2)), 2, model.Ny // 2)
        )
    if ay_values.size < 2:
        raise ValueError("At least two distinct interval sizes are required for log-chord fits.")
    entropy = []
    charge_variance = []
    for ay in ay_values:
        indices = np.asarray(
            [
                mu + 2 * x + 2 * model.Nx * y
                for y in range(int(ay))
                for x in range(x_min, x_max + 1)
                for mu in (0, 1)
            ],
            dtype=np.int64,
        )
        block = np.asarray(G)[np.ix_(indices, indices)]
        C = 0.5 * (block + np.eye(indices.size, dtype=np.complex128))
        eigenvalues = np.clip(np.linalg.eigvalsh(0.5 * (C + C.conj().T)).real, 0.0, 1.0)
        interior = (eigenvalues > 1.0e-14) & (eigenvalues < 1.0 - 1.0e-14)
        entropy.append(
            -float(
                np.sum(
                    eigenvalues[interior] * np.log(eigenvalues[interior])
                    + (1.0 - eigenvalues[interior]) * np.log1p(-eigenvalues[interior])
                )
            )
        )
        charge_variance.append(float(np.sum(eigenvalues * (1.0 - eigenvalues))))
    log_chord = np.log(
        (model.Ny / np.pi) * np.sin(np.pi * ay_values.astype(float) / model.Ny)
    )
    entropy_fit = np.polyfit(log_chord, np.asarray(entropy), 1)
    charge_fit = np.polyfit(log_chord, np.asarray(charge_variance), 1)
    return {
        "ay_values": ay_values,
        "log_chord": log_chord,
        "entropy": np.asarray(entropy),
        "charge_variance": np.asarray(charge_variance),
        "entropy_coefficient": float(entropy_fit[0]),
        "interval_charge_coefficient": float(charge_fit[0]),
    }


def snapshot_metrics(model: classA_U1FGTN, G: np.ndarray) -> dict[str, Any]:
    health = covariance_health(G)
    x_min, x_max = sorted(int(value) for value in model.DW_loc)
    x_center = (x_min + x_max) // 2
    radius = max(1.25, min(3.0, 0.4 * (x_max - x_min + 1)))
    chern = float(
        np.real(
            model.real_space_chern_number(
                G, xref=x_center, yref=model.Ny // 2, radius=radius
            )
        )
    )
    column = np.mean(charge_map(G, nx=model.Nx, ny=model.Ny), axis=1)
    background = float(np.median(column))
    excess = np.abs(column - background)
    walls = np.asarray((x_min, x_max), dtype=np.int64)
    distances = np.min(np.abs(np.arange(model.Nx)[:, None] - walls[None, :]), axis=1)
    localized_weight = float(
        np.sum(excess[distances <= 1]) / max(np.sum(excess), np.finfo(float).tiny)
    )
    return {
        **health,
        "bulk_chern": chern,
        "wall_localized_weight": localized_weight,
        "column_charge_mean": column,
        **strip_coefficients(model, G),
    }


def conditional_response(
    model: classA_U1FGTN,
    checkpoint: dict[str, Any],
    *,
    seed: int,
    sequence: str,
    horizon: int,
) -> dict[str, Any]:
    completed = int(checkpoint["completed_cycles"])
    walls = tuple(int(value) for value in model.DW_loc)
    y0 = model.Ny // 2
    displacement = ((np.arange(model.Ny) - y0 + model.Ny // 2) % model.Ny) - model.Ny // 2
    velocities = []
    profiles = []
    for wall in walls:
        branches = []
        for occupied in (True, False):
            state = copy.deepcopy(checkpoint)
            state["G"] = unit_cell_reset(
                state["G"], nx=model.Nx, ny=model.Ny, x=wall, y=y0, occupied=occupied
            )
            maps = [charge_map(state["G"], nx=model.Nx, ny=model.Ny)]
            model.run_markov_circuit(
                **base_run_kwargs(
                    cycles=completed + horizon,
                    seed=seed,
                    sequence=sequence,
                    init_mode="default",
                ),
                checkpoint_state=state,
                cycle_observer=lambda **payload: maps.append(
                    charge_map(payload["G"], nx=model.Nx, ny=model.Ny)
                ),
            )
            branches.append(np.asarray(maps))
        delta = 0.5 * (branches[0] - branches[1])
        x_values = [(wall - 1) % model.Nx, wall % model.Nx, (wall + 1) % model.Nx]
        profile = np.sum(delta[:, x_values, :], axis=1)
        norm = np.sum(np.abs(profile), axis=1)
        moment = np.divide(
            np.sum(profile * displacement[None, :], axis=1),
            norm,
            out=np.full(norm.shape, np.nan),
            where=norm > 1.0e-14,
        )
        valid = np.isfinite(moment)
        valid[0] = False
        stop = min(horizon, max(2, model.Ny // 4))
        valid[np.arange(valid.size) > stop] = False
        velocity = float(np.polyfit(np.flatnonzero(valid), moment[valid], 1)[0]) if np.count_nonzero(valid) >= 2 else np.nan
        velocities.append(velocity)
        profiles.append(profile)
    return {
        "wall_x": np.asarray(walls, dtype=np.int64),
        "conditional_velocity": np.asarray(velocities, dtype=np.float64),
        "conditional_profile": np.asarray(profiles, dtype=np.float64),
    }


def trajectory_directory(root: Path, sequence: str, sample: int) -> Path:
    return root / sequence / "shards" / f"trajectory_{sample:05d}"


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def completed_shard_is_verified(
    run_dir: Path, summary: dict[str, Any], config_hash: str
) -> bool:
    required = {
        "events.npz",
        "cycle_observables.npz",
        "stationary_snapshots.npz",
        "snapshot_metrics.npz",
    }
    files = summary.get("files", {})
    if (
        summary.get("status") != "complete"
        or summary.get("config_hash") != config_hash
        or not required.issubset(files)
    ):
        return False
    return all(
        (run_dir / name).is_file() and file_sha256(run_dir / name) == digest
        for name, digest in files.items()
    )


def run_v1_trajectory(task: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    nx, ny = int(task["nx"]), int(task["ny"])
    target_cycles = int(task["target_cycles"])
    sample = int(task["sample"])
    seed = int(task["seed"])
    sequence = str(task["sequence"])
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "Nx": nx,
        "Ny": ny,
        "samples": int(task["samples"]),
        "sample": sample,
        "seed": seed,
        "target_cycles": target_cycles,
        "sequence": sequence,
        "init_mode": "default",
        "nshell": 1,
        "alpha_top": 1,
        "alpha_triv": 30,
        "perfect_correction": True,
        "postselect": False,
        "canonical_dynamics_entry_point": ENTRY_POINT,
    }
    config_hash = payload_hash(config)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
        if completed_shard_is_verified(run_dir, summary, config_hash):
            return summary

    model = model_for(nx, ny)
    sites = active_site_ids(model)
    events = CompactEventRecorder(cycles=target_cycles, site_ids=sites)
    snapshot_stride = max(1, ny // 2)
    snapshot_cycles = {
        target_cycles - 2 * snapshot_stride,
        target_cycles - snapshot_stride,
        target_cycles,
    }
    cycles = CycleRecorder(
        cycles=target_cycles, nx=nx, ny=ny, snapshot_cycles=snapshot_cycles
    )
    try:
        checkpoint = load_checkpoint(run_dir) if bool(task.get("resume")) else None
    except (json.JSONDecodeError, RuntimeError, ValueError):
        checkpoint = None
    events_path = run_dir / "events.npz"
    cycles_path = run_dir / "cycle_observables.npz"
    snapshots_path = run_dir / "stationary_snapshots.npz"
    if checkpoint is not None:
        try:
            events.load(load_npz_dict(events_path))
            cycles.load(load_npz_dict(cycles_path))
            if snapshots_path.exists():
                saved_snapshots = load_npz_dict(snapshots_path)
                cycles.snapshots.update(
                    {
                        int(cycle): matrix.copy()
                        for cycle, matrix in zip(
                            saved_snapshots["snapshot_cycles"], saved_snapshots["G"]
                        )
                    }
                )
            cycles.set_previous(checkpoint["G"])
            completed = int(checkpoint["completed_cycles"])
            if completed in snapshot_cycles:
                cycles.snapshots[completed] = np.asarray(checkpoint["G"]).copy()
        except (FileNotFoundError, KeyError, RuntimeError, ValueError):
            checkpoint = None
            events = CompactEventRecorder(cycles=target_cycles, site_ids=sites)
            cycles = CycleRecorder(
                cycles=target_cycles, nx=nx, ny=ny, snapshot_cycles=snapshot_cycles
            )

    def save_available_snapshots() -> None:
        available = {
            int(cycle): matrix
            for cycle, matrix in cycles.snapshots.items()
            if cycle in snapshot_cycles
        }
        if available:
            ordered = np.asarray(sorted(available), dtype=np.int32)
            save_npz_atomic(
                snapshots_path,
                snapshot_cycles=ordered,
                G=np.stack([available[int(cycle)] for cycle in ordered]),
            )

    def checkpoint_callback(*, cycle: int, state: dict[str, Any]) -> None:
        if int(cycle) % snapshot_stride == 0 or int(cycle) == target_cycles:
            save_checkpoint(run_dir, state)
            save_npz_atomic(events_path, **events.payload())
            save_npz_atomic(cycles_path, **cycles.payload())
            save_available_snapshots()

    kwargs = base_run_kwargs(
        cycles=target_cycles, seed=seed, sequence=sequence, init_mode="default"
    )
    if checkpoint is not None:
        kwargs["checkpoint_state"] = checkpoint
    if checkpoint is not None and int(checkpoint["completed_cycles"]) == target_cycles:
        final_checkpoint = checkpoint
    else:
        with threadpool_limits(limits=1):
            result = model.run_markov_circuit(
                **kwargs,
                cycle_observer=cycles,
                trajectory_weight_observer=events,
                checkpoint_observer=checkpoint_callback,
            )
        final_checkpoint = result["checkpoint_state"]
    events.validate_complete(target_cycles)
    save_checkpoint(run_dir, final_checkpoint)
    save_npz_atomic(events_path, **events.payload())
    save_npz_atomic(cycles_path, **cycles.payload())
    snapshots = {
        int(cycle): matrix for cycle, matrix in cycles.snapshots.items() if cycle in snapshot_cycles
    }
    if len(snapshots) != 3:
        raise RuntimeError(
            f"Expected three stationary snapshots at {sorted(snapshot_cycles)}, got {sorted(snapshots)}."
        )
    ordered_snapshot_cycles = np.asarray(sorted(snapshots), dtype=np.int32)
    snapshot_stack = np.stack([snapshots[int(cycle)] for cycle in ordered_snapshot_cycles])
    save_npz_atomic(
        snapshots_path, snapshot_cycles=ordered_snapshot_cycles, G=snapshot_stack
    )
    metrics = [snapshot_metrics(model, G) for G in snapshot_stack]
    response = conditional_response(
        model,
        final_checkpoint,
        seed=seed,
        sequence=sequence,
        horizon=max(2, ny // 2),
    )
    metric_path = run_dir / "snapshot_metrics.npz"
    save_npz_atomic(
        metric_path,
        snapshot_cycles=ordered_snapshot_cycles,
        bulk_chern=np.asarray([item["bulk_chern"] for item in metrics]),
        wall_localized_weight=np.asarray([item["wall_localized_weight"] for item in metrics]),
        entropy_coefficient=np.asarray([item["entropy_coefficient"] for item in metrics]),
        interval_charge_coefficient=np.asarray(
            [item["interval_charge_coefficient"] for item in metrics]
        ),
        conditional_velocity=response["conditional_velocity"],
        conditional_profile=response["conditional_profile"],
        max_hermiticity=np.asarray(
            max(item["hermiticity_residual"] for item in metrics)
        ),
        max_spectral_violation=np.asarray(
            max(item["spectral_bound_violation"] for item in metrics)
        ),
    )
    summary = {
        "status": "complete",
        "created_utc": utc_now(),
        "config": config,
        "config_hash": config_hash,
        "checkpoint_cycle": int(final_checkpoint["completed_cycles"]),
        "checkpoint_signature": final_checkpoint["signature"],
        "files": {
            path.name: file_sha256(path)
            for path in (events_path, cycles_path, snapshots_path, metric_path)
        },
    }
    write_json_atomic(summary_path, summary)
    return summary


def bootstrap_interval(values: np.ndarray, *, rng: np.random.Generator, draws: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    means = np.mean(array[rng.integers(0, array.size, size=(int(draws), array.size))], axis=1)
    return float(np.mean(array)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def merge_v1_shards(root: Path, samples: int) -> dict[str, Any]:
    """Merge verified trajectory scalars in a deterministic schedule/sample order."""
    metric_names = (
        "bulk_chern",
        "wall_localized_weight",
        "entropy_coefficient",
        "interval_charge_coefficient",
        "conditional_velocity",
        "max_hermiticity",
        "max_spectral_violation",
    )
    merged: dict[str, list[np.ndarray]] = {name: [] for name in metric_names}
    shard_hashes = []
    for sequence in SCHEDULES:
        for sample in range(int(samples)):
            run_dir = trajectory_directory(root, sequence, sample)
            summary = json.loads((run_dir / "summary.json").read_text())
            config_hash = str(summary.get("config_hash", ""))
            if not completed_shard_is_verified(run_dir, summary, config_hash):
                raise RuntimeError(f"Refusing to merge unverified shard {run_dir}.")
            config = summary.get("config", {})
            if config.get("sequence") != sequence or int(config.get("sample", -1)) != sample:
                raise RuntimeError(f"Shard identity does not match its path: {run_dir}.")
            payload = load_npz_dict(run_dir / "snapshot_metrics.npz")
            for name in metric_names:
                merged[name].append(np.asarray(payload[name]))
            shard_hashes.append(
                {
                    "sequence": sequence,
                    "sample": sample,
                    "config_hash": config_hash,
                    "metric_sha256": summary["files"]["snapshot_metrics.npz"],
                }
            )
    arrays = {
        name: np.stack(values).reshape((len(SCHEDULES), int(samples)) + values[0].shape)
        for name, values in merged.items()
    }
    merged_path = root / "merged_shards.npz"
    save_npz_atomic(
        merged_path,
        schedules=np.asarray(SCHEDULES),
        sample=np.arange(int(samples), dtype=np.int32),
        **arrays,
    )
    manifest = {
        "created_utc": utc_now(),
        "order": "schedule-major then ascending sample",
        "schedules": SCHEDULES,
        "samples_per_word": int(samples),
        "merged_sha256": file_sha256(merged_path),
        "shards": shard_hashes,
    }
    write_json_atomic(root / "merged_shards_manifest.json", manifest)
    return manifest


def estimate_stationarity(root: Path, sequence: str, samples: int, ny: int) -> dict[str, Any]:
    times = []
    for sample in range(samples):
        payload = load_npz_dict(
            trajectory_directory(root, sequence, sample) / "cycle_observables.npz"
        )
        charge = payload["total_charge"]
        variance = payload["charge_variance"]
        delta = payload["successive_delta"]
        final_slice = slice(max(1, charge.size - ny), charge.size)
        charge_scale = max(1.0, abs(float(np.nanmean(charge[final_slice]))))
        variance_scale = max(1.0, abs(float(np.nanmean(variance[final_slice]))))
        charge_tol = max(0.02 * charge_scale, 3.0 * float(np.nanstd(charge[final_slice])))
        variance_tol = max(0.05 * variance_scale, 3.0 * float(np.nanstd(variance[final_slice])))
        delta_tol = max(1.0e-10, 1.5 * float(np.nanquantile(delta[final_slice], 0.95)))
        reference_charge = float(np.nanmean(charge[final_slice]))
        reference_variance = float(np.nanmean(variance[final_slice]))
        accepted = charge.size - 1
        for start in range(1, max(2, charge.size - ny)):
            stop = min(charge.size, start + ny)
            if stop - start < ny:
                continue
            if (
                np.all(np.abs(charge[start:stop] - reference_charge) <= charge_tol)
                and np.all(np.abs(variance[start:stop] - reference_variance) <= variance_tol)
                and np.all(delta[start:stop] <= delta_tol)
            ):
                accepted = start
                break
        times.append(accepted)
    times_array = np.asarray(times, dtype=np.int32)
    rng = np.random.default_rng(7001 + sum(map(ord, sequence)))
    maxima = np.quantile(
        times_array[rng.integers(0, samples, size=(2000, samples))], 0.95, axis=1
    )
    upper = int(math.ceil(float(np.quantile(maxima, 0.95))))
    return {
        "sequence": sequence,
        "T_epsilon": times_array,
        "upper_95": upper,
        "median": float(np.median(times_array)),
    }


def analyze_v1(root: Path, *, nx: int, ny: int, samples: int, bootstrap_draws: int) -> dict[str, Any]:
    metric_names = (
        "bulk_chern",
        "wall_localized_weight",
        "entropy_coefficient",
        "interval_charge_coefficient",
    )
    raw: dict[str, dict[str, np.ndarray]] = {}
    for sequence in SCHEDULES:
        values = {name: [] for name in metric_names}
        values["conditional_velocity_left"] = []
        values["conditional_velocity_right"] = []
        for sample in range(samples):
            payload = load_npz_dict(
                trajectory_directory(root, sequence, sample) / "snapshot_metrics.npz"
            )
            for name in metric_names:
                values[name].append(float(np.mean(payload[name])))
            velocity = np.asarray(payload["conditional_velocity"])
            values["conditional_velocity_left"].append(float(velocity[0]))
            values["conditional_velocity_right"].append(float(velocity[-1]))
        raw[sequence] = {name: np.asarray(items) for name, items in values.items()}

    rng = np.random.default_rng(99173)
    intervals: dict[str, dict[str, Any]] = {}
    for sequence, values in raw.items():
        intervals[sequence] = {
            name: dict(zip(("mean", "low", "high"), bootstrap_interval(array, rng=rng, draws=bootstrap_draws)))
            for name, array in values.items()
        }
    stable = True
    categorical_change = False
    reasons = []
    for sequence in SCHEDULES:
        item = intervals[sequence]
        if item["bulk_chern"]["low"] * item["bulk_chern"]["high"] <= 0.0:
            stable = False
            reasons.append(f"{sequence}: bulk topology sign is unresolved")
        if item["wall_localized_weight"]["low"] <= 0.5:
            stable = False
            reasons.append(f"{sequence}: less than half of wall excess is localized")
        left = item["conditional_velocity_left"]
        right = item["conditional_velocity_right"]
        if left["low"] * left["high"] <= 0.0 or right["low"] * right["high"] <= 0.0:
            stable = False
            reasons.append(f"{sequence}: conditional chirality sign is unresolved")
        elif left["mean"] * right["mean"] >= 0.0:
            stable = False
            reasons.append(f"{sequence}: opposite walls do not have opposite velocity signs")
    raster = intervals["raster_y"]
    reverse = intervals["reverse_raster_y"]
    for name in metric_names:
        if raster[name]["mean"] * reverse[name]["mean"] < 0.0:
            stable = False
            categorical_change = True
            reasons.append(f"raster reversal changes the sign of {name}")

    for name in ("bulk_chern", "conditional_velocity_left", "conditional_velocity_right"):
        resolved_signs = []
        for sequence in SCHEDULES:
            interval = intervals[sequence][name]
            if interval["low"] > 0.0:
                resolved_signs.append(1)
            elif interval["high"] < 0.0:
                resolved_signs.append(-1)
        if resolved_signs and len(set(resolved_signs)) > 1:
            stable = False
            categorical_change = True
            reasons.append(f"resolved schedules have different signs for {name}")

    localization_categories = {
        sequence: intervals[sequence]["wall_localized_weight"]["low"] > 0.5
        for sequence in SCHEDULES
    }
    if len(set(localization_categories.values())) > 1:
        stable = False
        categorical_change = True
        reasons.append("schedules disagree on transverse wall localization")

    if categorical_change:
        status = "SCHEDULE_DEPENDENT"
    elif stable:
        status = "STAGED_PASS"
    else:
        status = "STAGED_INCONCLUSIVE"

    result = {
        "status": status,
        "categorical_change": categorical_change,
        "created_utc": utc_now(),
        "Nx": nx,
        "Ny": ny,
        "samples_per_word": samples,
        "bootstrap_draws": bootstrap_draws,
        "conditional_chirality_only": True,
        "born_score_response_deferred_to": "H2",
        "intervals": intervals,
        "reasons": reasons,
    }
    write_json_atomic(root / "v1_analysis.json", result)
    plot_v1(result, root / "v1_order_artifact")
    return result


def plot_v1(result: dict[str, Any], stem: Path) -> None:
    names = (
        "bulk_chern",
        "wall_localized_weight",
        "entropy_coefficient",
        "interval_charge_coefficient",
        "conditional_velocity_left",
        "conditional_velocity_right",
    )
    labels = ("bulk C", "wall weight", "entropy", "charge", "v left", "v right")
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.0))
    colors = ("#3366aa", "#dd8844", "#55aa66")
    for axis, name, label in zip(axes.flat, names, labels):
        for index, (sequence, color) in enumerate(zip(SCHEDULES, colors)):
            interval = result["intervals"][sequence][name]
            mean = interval["mean"]
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - interval["low"]], [interval["high"] - mean]],
                marker="o",
                color=color,
                capsize=2,
            )
        axis.axhline(0.0, color="0.75", linewidth=0.7)
        axis.set_xticks(range(3), ("random", "raster", "reverse"), rotation=25)
        axis.set_title(label)
    fig.suptitle(f"V1 staged order-artifact check: {result['status']}")
    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def exhaustive_local_probability(model: classA_U1FGTN) -> dict[str, Any]:
    G = np.zeros((model.Ntot // 2, model.Ntot // 2), dtype=np.complex128)
    weights = []
    for bits in itertools.product((False, True), repeat=4):
        events = []
        for channel, outcome in zip(CHANNELS, bits):
            events.append(
                {"channel": channel, "kind": "measurement", "outcome_occupied": outcome}
            )
            expected = EXPECTED_OCCUPIED[channel]
            if outcome != expected:
                events.append(
                    {
                        "channel": channel,
                        "kind": "correction",
                        "expected_occupied": expected,
                        "target_occupied": expected,
                    }
                )
        _, summary = model.markov_meas_feedback(
            G,
            int(model.DW_loc[0]),
            0,
            perfect_correction=True,
            return_weight_summary=True,
            branch_replay_events=events,
            replay_probability_tol=0.0,
        )
        weights.append(math.exp(float(summary["branch_log_weight"])))
    residual = abs(float(np.sum(weights)) - 1.0)
    return {"branch_count": 16, "probability_sum": float(np.sum(weights)), "residual": residual}


def tangent_five_scale_check() -> dict[str, Any]:
    model = classA_U1FGTN(1, 2, DW=False, nshell=None)
    dimension = model.Ntot // 2
    rng = np.random.default_rng(817)
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    G = 0.2 * (raw + raw.conj().T) / np.linalg.norm(raw)
    direction_raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    direction = 0.5 * (direction_raw + direction_raw.conj().T)
    direction /= np.linalg.norm(direction, ord="fro")
    chi = np.asarray((1.0, 0.2j, -0.3, 0.4 + 0.1j), dtype=np.complex128)
    chi /= np.linalg.norm(chi)
    projector = np.outer(chi, chi.conj())
    state = model._init_lyapunov_state(
        batch_count=1, n_vec=dimension, initial_frame=np.eye(dimension)
    )
    model._lyapunov_apply_dense_channel(
        state, G, np.asarray([0]), projector, particle=True
    )
    factor = state["frame"][0]
    predicted = factor @ direction @ factor.conj().T
    epsilons = np.asarray((1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5))
    errors = []
    for epsilon in epsilons:
        plus = model.measure_only_top_layer(G + epsilon * direction, projector, particle=True, chi=chi)
        minus = model.measure_only_top_layer(G - epsilon * direction, projector, particle=True, chi=chi)
        numerical = (plus - minus) / (2.0 * epsilon)
        errors.append(float(np.linalg.norm(numerical - predicted) / np.linalg.norm(predicted)))
    return {
        "epsilons": epsilons,
        "relative_errors": np.asarray(errors),
        "minimum_relative_error": float(np.min(errors)),
    }


def serial_shard_check() -> dict[str, Any]:
    kwargs = {
        "G_history": False,
        "progress": False,
        "cycles": 3,
        "samples": 2,
        "init_mode": "default",
        "save": False,
        "sequence": "random",
        "meas_slab_only": True,
        "random_seed": 918273,
        "perfect_correction": True,
    }
    serial = model_for(4, 6).run_markov_circuit(
        **kwargs, parallelize_samples=False
    )["G_final"]
    sharded = model_for(4, 6).run_markov_circuit(
        **kwargs,
        parallelize_samples=True,
        n_jobs=2,
        backend="loky",
        throttle=False,
    )["G_final"]
    maximum = float(np.max(np.abs(serial - sharded)))
    tolerance = roundoff_tolerance(serial.shape[-1], 3 * 18 * 4)
    return {
        "bitwise_equal": bool(np.array_equal(serial, sharded)),
        "numerically_equal": bool(maximum <= tolerance),
        "max_abs_diff": maximum,
        "tolerance": tolerance,
    }


def periodic_wrap_check(model: classA_U1FGTN) -> dict[str, Any]:
    x_center = sum(int(value) for value in model.DW_loc) // 2
    payload = model._get_ow_local_support_data(x_center, 0)
    cells = set()
    for index in np.asarray(payload["idx"], dtype=np.int64):
        cell = int(index) // 2
        x = cell % model.Nx
        y = cell // model.Nx
        cells.add((x, y))
    expected = {
        (x % model.Nx, y % model.Ny)
        for x in (x_center - 1, x_center, x_center + 1)
        for y in (-1, 0, 1)
    }
    return {
        "pass": cells == expected,
        "observed_cells": sorted(cells),
        "expected_cells": sorted(expected),
    }


def run_v0_case(nx: int, ny: int, seed: int, cycles: int, output: Path) -> dict[str, Any]:
    model = model_for(nx, ny)
    sites = active_site_ids(model)
    rank_events = CompactEventRecorder(cycles=cycles, site_ids=sites, replay=True)
    rank_cycles = CycleRecorder(cycles=cycles, nx=nx, ny=ny)
    checkpoints = []
    kwargs = base_run_kwargs(cycles=cycles, seed=seed, sequence="random", init_mode="default")
    rank = model.run_markov_circuit(
        **kwargs,
        physical_covariance_update="rank1",
        cycle_observer=rank_cycles,
        trajectory_weight_observer=rank_events,
        checkpoint_observer=lambda **payload: checkpoints.append(payload["state"]),
    )
    rank_events.validate_complete()

    dense_events = CompactEventRecorder(cycles=cycles, site_ids=sites)
    dense = model_for(nx, ny).run_markov_circuit(
        **kwargs,
        physical_covariance_update="dense",
        trajectory_replay=rank_events.entries,
        trajectory_weight_observer=dense_events,
    )
    dense_events.validate_complete()
    split = max(1, cycles // 2)
    split_states = []
    model_for(nx, ny).run_markov_circuit(
        **base_run_kwargs(cycles=split, seed=seed, sequence="random", init_mode="default"),
        checkpoint_observer=lambda **payload: split_states.append(payload["state"]),
    )
    resumed = model_for(nx, ny).run_markov_circuit(
        **kwargs, checkpoint_state=split_states[-1]
    )

    final_rank = rank["G_final"][0]
    final_dense = dense["G_final"][0]
    final_resumed = resumed["G_final"][0]
    health = covariance_health(final_rank)
    dimension = final_rank.shape[0]
    tolerance = roundoff_tolerance(dimension, cycles * max(1, sites.size) * 4)
    rank_dense_diff = float(np.max(np.abs(final_rank - final_dense)))
    resume_diff = float(np.max(np.abs(final_rank - final_resumed)))
    probability_diff = float(
        np.nanmax(
            np.abs(rank_events.success_probability - dense_events.success_probability)
        )
    )
    schedule_equal = bool(
        np.array_equal(rank_events.ordered_site_ids(), dense_events.ordered_site_ids())
    )
    outcome_equal = bool(
        np.array_equal(rank_events.outcome_bits, dense_events.outcome_bits)
    )
    charge_by_cycle = np.sum(rank_events.transfer, axis=(1, 2))
    observed_charge_change = np.diff(rank_cycles.total_charge)
    charge_residual = float(np.max(np.abs(observed_charge_change - charge_by_cycle)))
    direct_charge = float(np.trace(0.5 * (final_rank + np.eye(dimension))).real)
    direct_variance = float(
        direct_charge
        - np.sum(np.abs(0.5 * (final_rank + np.eye(dimension))) ** 2).real
    )
    direct_column_charge = np.sum(charge_map(final_rank, nx=nx, ny=ny), axis=1)
    observer_residuals = {
        "charge": abs(float(rank_cycles.total_charge[-1]) - direct_charge),
        "charge_variance": abs(float(rank_cycles.charge_variance[-1]) - direct_variance),
        "wall_profile": float(
            np.max(np.abs(rank_cycles.column_charge[-1] - direct_column_charge))
        ),
    }
    passed = bool(
        rank_dense_diff <= tolerance
        and resume_diff <= tolerance
        and probability_diff <= tolerance
        and schedule_equal
        and outcome_equal
        and health["hermiticity_residual"] <= tolerance
        and health["spectral_bound_violation"] <= tolerance
        and charge_residual <= tolerance
        and max(observer_residuals.values()) <= tolerance
    )
    result = {
        "status": "pass" if passed else "fail",
        "Nx": nx,
        "Ny": ny,
        "seed": seed,
        "cycles": cycles,
        "tolerance": tolerance,
        "tolerance_formula": "max(1e-12, 2048*eps64*N*sqrt(elementary_blocks))",
        "rank_dense_max_abs_diff": rank_dense_diff,
        "resume_max_abs_diff": resume_diff,
        "probability_max_abs_diff": probability_diff,
        "charge_bookkeeping_max_abs_diff": charge_residual,
        "observer_direct_formula_residuals": observer_residuals,
        "schedule_bitwise_equal": schedule_equal,
        "outcome_bitwise_equal": outcome_equal,
        **health,
    }
    case_dir = output / f"N{nx}x{ny}_seed{seed}_cycles{cycles}"
    write_json_atomic(case_dir / "summary.json", result)
    save_npz_atomic(case_dir / "rank_events.npz", **rank_events.payload())
    save_npz_atomic(
        case_dir / "covariances.npz",
        rank1=final_rank,
        dense=final_dense,
        resumed=final_resumed,
    )
    return result


def gpu_audit(output: Path) -> dict[str, Any]:
    canonical = REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN_gpu.py"
    copies = sorted(
        path
        for path in REPO_ROOT.rglob("classA_U1FGTN_gpu.py")
        if "erroneous_gpu_stuff" not in path.parts
    )
    hashes = {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in copies}
    canonical_hash = file_sha256(canonical)
    source = canonical.read_text()
    forbidden = []
    forbidden_helpers = (
        "_apply_grouped_site_updates(",
        "_run_one_cycle(",
        "_run_markov_cycle(",
    )
    for path in REPO_ROOT.rglob("*.py"):
        if (
            "erroneous_gpu_stuff" in path.parts
            or path.name in ("classA_U1FGTN.py", "classA_U1FGTN_gpu.py")
            or "tests" in path.parts
        ):
            continue
        text = path.read_text(errors="ignore")
        if any(helper in text for helper in forbidden_helpers):
            forbidden.append(str(path.relative_to(REPO_ROOT)))
    torch_installed = importlib.util.find_spec("torch") is not None
    reverse_raster_supported = "reverse_raster_y" in source
    trajectory_replay_supported = "trajectory_replay" in source
    wall_metadata_present = '"DW_loc"' in source or '"domain_walls"' in source
    projector_metadata_present = (
        '"projector_hashes"' in source or '"projector_signature"' in source
    )
    limitations = [
        "GPU code was inspected statically and no CUDA kernels were executed.",
        "CPU/GPU agreement remains untested.",
    ]
    if not torch_installed:
        limitations.append("PyTorch is unavailable in the campaign environment.")
    if not reverse_raster_supported:
        limitations.append("The canonical GPU engine lacks reverse_raster_y support.")
    if not trajectory_replay_supported:
        limitations.append("The canonical GPU engine lacks an exact trajectory replay interface.")
    if not (wall_metadata_present and projector_metadata_present):
        limitations.append(
            "GPU run manifests do not explicitly record both wall and projector signatures."
        )
    result = {
        "status": "AUDIT_ONLY / NOT_VALIDATED",
        "created_utc": utc_now(),
        "torch_installed": torch_installed,
        "cuda_executed": False,
        "canonical_entry_point_present": "def run_markov_circuit(" in source,
        "metadata_entry_point_present": "classA_U1FGTN_gpu.run_markov_circuit" in source,
        "resume_parameter_present": "resume=False" in source,
        "shard_manifest_present": '"shards"' in source,
        "explicit_active_indices_present": '"active_top_layer_indices"' in source,
        "reverse_raster_supported": reverse_raster_supported,
        "trajectory_replay_supported": trajectory_replay_supported,
        "explicit_wall_metadata_present": wall_metadata_present,
        "explicit_projector_signature_present": projector_metadata_present,
        "all_deployable_copies_match": all(value == canonical_hash for value in hashes.values()),
        "canonical_sha256": canonical_hash,
        "copy_hashes": hashes,
        "forbidden_private_cycle_callers": forbidden,
        "limitations": limitations,
    }
    write_json_atomic(output / "gpu_audit.json", result)
    lines = [
        "# GPU Audit (No Runtime Validation)",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- Deployable copies synchronized: {result['all_deployable_copies_match']}",
        f"- Canonical entry point recorded: {result['metadata_entry_point_present']}",
        f"- Resume/shard code present: {result['resume_parameter_present'] and result['shard_manifest_present']}",
        f"- Reverse raster present: {result['reverse_raster_supported']}",
        f"- Exact trajectory replay present: {result['trajectory_replay_supported']}",
        f"- Forbidden private-loop callers: {result['forbidden_private_cycle_callers']}",
        "",
        "This audit does not establish CPU/GPU numerical agreement.",
    ]
    (output / "GPU_AUDIT.md").write_text("\n".join(lines) + "\n")
    return result


def run_v0(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    output = root / "v0"
    output.mkdir(parents=True, exist_ok=True)
    geometries = ((4, 6),) if args.smoke else ((4, 6), (6, 8))
    seeds = (101,) if args.smoke else (101, 202, 303, 404)
    cycle_values = (3,) if args.smoke else (3, 10)
    cases = []
    for nx, ny in geometries:
        local = exhaustive_local_probability(model_for(nx, ny))
        local["Nx"], local["Ny"] = nx, ny
        write_json_atomic(output / f"N{nx}x{ny}_local_probability.json", local)
        for seed in seeds:
            for cycles in cycle_values:
                cases.append(run_v0_case(nx, ny, seed, cycles, output))
    tangent = tangent_five_scale_check()
    save_npz_atomic(output / "tangent_five_scale.npz", **tangent)
    shard = serial_shard_check()
    write_json_atomic(output / "serial_shard_check.json", shard)
    wrapping = periodic_wrap_check(model_for(4, 6))
    write_json_atomic(output / "periodic_wrap_check.json", wrapping)
    zero = model_for(4, 6, twist_y=0.0)
    closed = model_for(4, 6, twist_y=2.0 * np.pi)
    twist_residual = float(np.max(np.abs(closed.Pminus - np.roll(zero.Pminus, -1, axis=1))))
    audit = gpu_audit(output)
    probability_pass = all(
        json.loads(path.read_text())["residual"] <= roundoff_tolerance(2 * 4 * 6, 4)
        for path in output.glob("*_local_probability.json")
    )
    passed = bool(
        all(case["status"] == "pass" for case in cases)
        and probability_pass
        and tangent["minimum_relative_error"] <= 2.0e-6
        and shard["numerically_equal"]
        and wrapping["pass"]
        and twist_residual <= roundoff_tolerance(2 * 4 * 6, 4)
    )
    result = {
        "status": "CPU_PASS / GPU_NOT_RUN" if passed else "CPU_FAIL / GPU_NOT_RUN",
        "created_utc": utc_now(),
        "canonical_dynamics_entry_point": ENTRY_POINT,
        "cases": cases,
        "local_probability_normalization_pass": probability_pass,
        "tangent_minimum_relative_error": tangent["minimum_relative_error"],
        "serial_shard": shard,
        "periodic_wrap": wrapping,
        "twist_2pi_projector_relabel_residual": twist_residual,
        "gpu_audit_status": audit["status"],
        "git": git_metadata(),
    }
    write_json_atomic(output / "validation_summary.json", result)
    marker = output / ("_SUCCESS" if passed else "_FAILED")
    marker.write_text(result["status"] + "\n")
    return result


def run_schedule_preflight(root: Path, *, smoke: bool) -> dict[str, Any]:
    results = []
    geometries = ((4, 6),) if smoke else ((4, 6), (6, 8))
    for nx, ny in geometries:
        for sequence in SCHEDULES:
            model = model_for(nx, ny)
            sites = active_site_ids(model)
            recorder = CompactEventRecorder(cycles=3, site_ids=sites, replay=True)
            expected = model.run_markov_circuit(
                **base_run_kwargs(cycles=3, seed=881, sequence=sequence, init_mode="maxmix"),
                trajectory_weight_observer=recorder,
            )["G_final"][0]
            recorder.validate_complete()
            actual = model_for(nx, ny).run_markov_circuit(
                **base_run_kwargs(cycles=3, seed=881, sequence=sequence, init_mode="maxmix"),
                trajectory_replay=recorder.entries,
            )["G_final"][0]
            results.append(
                {
                    "Nx": nx,
                    "Ny": ny,
                    "sequence": sequence,
                    "replay_max_abs_diff": float(np.max(np.abs(actual - expected))),
                    "one_visit_per_cycle": True,
                }
            )
    payload = {"status": "pass", "results": results}
    write_json_atomic(root / "schedule_preflight.json", payload)
    return payload


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [run_v1_trajectory(task) for task in tasks]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_v1_trajectory, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            result = future.result()
            results.append(result)
            print(
                f"[v1] {task['sequence']} sample {int(task['sample']) + 1}/{task['samples']} complete",
                flush=True,
            )
    return results


def build_v1_tasks(
    output: Path,
    *,
    nx: int,
    ny: int,
    samples: int,
    target_cycles: int,
    seed: int,
    resume: bool,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    tasks = []
    for schedule_index, sequence in enumerate(SCHEDULES):
        root_seed = int(seed) + int(seed_offset) + schedule_index * 10_000_019
        sample_sequences = np.random.SeedSequence(root_seed).spawn(samples)
        for sample, seed_sequence in enumerate(sample_sequences):
            sample_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
            tasks.append(
                {
                    "nx": int(nx),
                    "ny": int(ny),
                    "samples": int(samples),
                    "sample": sample,
                    "seed": sample_seed,
                    "sequence": sequence,
                    "target_cycles": int(target_cycles),
                    "run_dir": str(trajectory_directory(output, sequence, sample)),
                    "resume": bool(resume),
                }
            )
    return tasks


def run_v1_system(
    args: argparse.Namespace,
    output: Path,
    *,
    nx: int,
    ny: int,
    samples: int,
    seed_offset: int = 0,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    burn_limit = 2 * ny
    max_burn = 2 * ny if args.smoke else 16 * ny
    stationarity = {}
    while True:
        target_cycles = burn_limit + 2 * ny
        tasks = build_v1_tasks(
            output,
            nx=nx,
            ny=ny,
            samples=samples,
            target_cycles=target_cycles,
            seed=args.seed,
            resume=args.resume,
            seed_offset=seed_offset,
        )
        run_tasks(tasks, workers=min(args.workers, len(tasks)))
        merge_v1_shards(output, samples)
        stationarity = {
            sequence: estimate_stationarity(output, sequence, samples, ny)
            for sequence in SCHEDULES
        }
        write_json_atomic(output / "stationarity.json", stationarity)
        failing = [
            sequence
            for sequence, result in stationarity.items()
            if int(result["upper_95"]) > burn_limit
        ]
        if not failing or burn_limit >= max_burn:
            break
        burn_limit += ny
        print(f"[v1] extending burn-in limit to {burn_limit} cycles for {failing}", flush=True)

    analysis = analyze_v1(
        output,
        nx=nx,
        ny=ny,
        samples=samples,
        bootstrap_draws=args.bootstrap_samples,
    )
    analysis["stationarity"] = stationarity
    analysis["burn_limit"] = burn_limit
    if any(int(item["upper_95"]) > burn_limit for item in stationarity.values()):
        analysis["status"] = "STATIONARITY_FAIL"
        analysis["reasons"].append("Stationarity failed at the maximum declared burn-in.")
    write_json_atomic(output / "validation_summary.json", analysis)
    (output / ("_SUCCESS" if analysis["status"] == "STAGED_PASS" else "_FAILED")).write_text(
        analysis["status"] + "\n"
    )
    return analysis


def run_v1(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    output = root / "v1"
    output.mkdir(parents=True, exist_ok=True)
    run_schedule_preflight(output, smoke=args.smoke)
    nx, ny = (4, 6) if args.smoke else (int(args.v1_nx), int(args.v1_ny))
    samples = 2 if args.smoke else int(args.v1_samples)
    analysis = run_v1_system(
        args,
        output,
        nx=nx,
        ny=ny,
        samples=samples,
    )
    auto_escalation = not bool(args.no_v1_auto_escalation)
    analysis["campaign_qualification"] = (
        "SMOKE"
        if args.smoke
        else "S10_REDUCED"
        if samples == 10
        else "REDUCED"
        if (nx, ny, samples) != (20, 48, 100)
        else "PRODUCTION"
    )
    analysis["auto_escalation_enabled"] = bool(auto_escalation and not args.smoke)
    if (
        analysis["status"] == "SCHEDULE_DEPENDENT"
        and not args.smoke
        and auto_escalation
    ):
        escalations = {}
        for escalation_ny in (32, 64):
            escalation_output = output / f"escalation_Ny{escalation_ny}"
            print(
                f"[v1] categorical schedule change: escalating all words to "
                f"Nx={nx}, Ny={escalation_ny}, S={samples}",
                flush=True,
            )
            escalations[str(escalation_ny)] = run_v1_system(
                args,
                escalation_output,
                nx=nx,
                ny=escalation_ny,
                samples=samples,
                seed_offset=escalation_ny * 100_000_007,
            )
        analysis["escalations"] = escalations
    elif analysis["status"] == "SCHEDULE_DEPENDENT" and not args.smoke:
        analysis["reasons"].append(
            "Automatic larger-system escalation was disabled for this campaign."
        )
    write_json_atomic(output / "validation_summary.json", analysis)
    return analysis


def run_v1_benchmark(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    nx, ny, samples = int(args.v1_nx), int(args.v1_ny), int(args.v1_samples)
    target_cycles = 4 * ny
    benchmark_config = {
        "Nx": nx,
        "Ny": ny,
        "samples_per_schedule": samples,
        "target_cycles": target_cycles,
        "seed": int(args.seed),
        "workers": int(args.workers),
    }
    config_hash = payload_hash(benchmark_config)
    summary_path = root / "benchmark_summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved.get("status") == "complete" and saved.get("config_hash") == config_hash:
            return saved
    tasks = build_v1_tasks(
        root / "benchmark",
        nx=nx,
        ny=ny,
        samples=samples,
        target_cycles=target_cycles,
        seed=args.seed,
        resume=False,
    )
    representative = tasks[0]
    started = time.monotonic()
    result = run_v1_trajectory(representative)
    elapsed = time.monotonic() - started
    waves = int(math.ceil(len(tasks) / min(args.workers, len(tasks))))
    summary = {
        "status": "complete",
        "created_utc": utc_now(),
        "config": benchmark_config,
        "config_hash": config_hash,
        "representative_schedule": representative["sequence"],
        "representative_sample": representative["sample"],
        "Nx": nx,
        "Ny": ny,
        "samples_per_schedule": samples,
        "total_trajectories": len(tasks),
        "target_cycles": target_cycles,
        "elapsed_seconds": elapsed,
        "concurrent_waves": waves,
        "projected_initial_stage_seconds": elapsed * waves,
        "projection_excludes_stationarity_extensions": True,
        "trajectory_summary": result,
    }
    write_json_atomic(summary_path, summary)
    return summary


def analyze_existing(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    v1 = root / "v1"
    if not v1.exists():
        raise FileNotFoundError(f"No V1 output exists at {v1}.")
    manifest_path = root / "campaign_manifest.json"
    saved_config = {}
    if manifest_path.exists():
        saved_config = json.loads(manifest_path.read_text()).get("v1_config", {})
    nx, ny = (
        (4, 6)
        if args.smoke
        else (
            int(saved_config.get("Nx", args.v1_nx)),
            int(saved_config.get("Ny", args.v1_ny)),
        )
    )
    samples = (
        2
        if args.smoke
        else int(saved_config.get("samples_per_schedule", args.v1_samples))
    )
    return analyze_v1(
        v1,
        nx=nx,
        ny=ny,
        samples=samples,
        bootstrap_draws=args.bootstrap_samples,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("v0", "v1", "all", "analyze", "benchmark")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--cpu-list", default="56-111")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--v1-nx", type=int, default=20)
    parser.add_argument("--v1-ny", type=int, default=48)
    parser.add_argument("--v1-samples", type=int, default=100)
    parser.add_argument("--no-v1-auto-escalation", action="store_true")
    parser.add_argument("--v0-summary", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers <= 0 or args.threads_per_worker != 1:
        raise ValueError("Use a positive worker count and exactly one BLAS thread per worker.")
    if (
        args.v1_nx <= 0
        or args.v1_ny <= 0
        or args.v1_nx % 2
        or args.v1_ny % 2
        or args.v1_samples <= 0
    ):
        raise ValueError("V1 requires positive even Nx/Ny and a positive sample count.")
    v0_prerequisite = None
    if args.v0_summary is not None:
        v0_prerequisite = load_v0_prerequisite(args.v0_summary)
    if args.mode in ("v1", "benchmark") and not args.smoke and v0_prerequisite is None:
        raise ValueError("Production V1-only execution requires --v0-summary.")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    v1_nx, v1_ny = (
        (4, 6) if args.smoke else (int(args.v1_nx), int(args.v1_ny))
    )
    v1_samples = 2 if args.smoke else int(args.v1_samples)
    manifest = {
        "created_utc": utc_now(),
        "mode": args.mode,
        "smoke": bool(args.smoke),
        "resume": bool(args.resume),
        "cpu_list": args.cpu_list,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "canonical_dynamics_entry_point": ENTRY_POINT,
        "gpu_status": "AUDIT_ONLY / NOT_VALIDATED",
        "v0_prerequisite": v0_prerequisite,
        "v1_config": {
            "Nx": v1_nx,
            "Ny": v1_ny,
            "samples_per_schedule": v1_samples,
            "schedules": SCHEDULES,
            "total_trajectories": len(SCHEDULES) * v1_samples,
            "initial_burn_limit": 2 * v1_ny,
            "observation_window": 2 * v1_ny,
            "initial_target_cycles": 4 * v1_ny,
            "max_burn_limit": 2 * v1_ny if args.smoke else 16 * v1_ny,
            "max_target_cycles": 4 * v1_ny if args.smoke else 18 * v1_ny,
            "checkpoint_stride": max(1, v1_ny // 2),
            "auto_escalation_enabled": bool(
                not args.no_v1_auto_escalation and not args.smoke
            ),
            "qualification": (
                "SMOKE"
                if args.smoke
                else "S10_REDUCED"
                if v1_samples == 10
                else "REDUCED"
                if (v1_nx, v1_ny, v1_samples) != (20, 48, 100)
                else "PRODUCTION"
            ),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git": git_metadata(),
    }
    if args.mode != "analyze" or not (root / "campaign_manifest.json").exists():
        write_json_atomic(root / "campaign_manifest.json", manifest)
    if args.mode == "analyze":
        result = analyze_existing(root, args)
    elif args.mode == "benchmark":
        result = run_v1_benchmark(args, root)
    elif args.mode == "v0":
        result = run_v0(args, root)
    elif args.mode == "v1":
        result = run_v1(args, root)
    else:
        v0 = run_v0(args, root)
        if v0["status"] != "CPU_PASS / GPU_NOT_RUN":
            raise SystemExit("V0 failed; V1 was not launched.")
        result = {"v0": v0, "v1": run_v1(args, root)}
    if args.mode != "benchmark":
        write_json_atomic(root / "campaign_summary.json", result)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
