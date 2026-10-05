#!/usr/bin/env python3
"""Completion-resumable CPU campaign for a Born-sampled online flux ramp."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


CAMPAIGN_SCHEMA = "online_flux_ramp_campaign_v1"
BURNIN_SCHEMA = "online_flux_ramp_burnin_v1"
RAMP_SCHEMA = "online_flux_ramp_trajectory_v1"
COMPLETION_SCHEMA = "online_flux_ramp_completion_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.online_ramp_s10_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N16x20_online_flux_ramp_s10_v1"
SOURCE_PATHS = {
    "cpu_engine": REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py",
    "occupied_frame": REPO_ROOT / "src" / "fgtn" / "occupied_frame.py",
    "campaign_runner": Path(__file__).resolve(),
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema",
        "root_seed",
        "geometry",
        "walls",
        "dynamics",
        "twist",
        "regions",
        "ensemble",
        "acceptance",
    )
    return hashlib.sha256(
        canonical_json({key: config[key] for key in keys}).encode("utf-8")
    ).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    geometry, dynamics = config["geometry"], config["dynamics"]
    if (int(geometry["Nx"]), int(geometry["Ny"])) != (16, 20):
        raise ValueError("this versioned pilot is locked to Nx=16, Ny=20")
    if geometry.get("DW") is not True or list(geometry["dw_interval"]) != [4, 12]:
        raise ValueError("the domain-wall geometry must use DW=True and interval [4,12]")
    if (
        int(geometry["nshell"]) != 1
        or float(geometry["filling_frac"]) != 0.5
        or float(geometry["alpha_1"]) != 1.0
        or float(geometry["alpha_2"]) != 30.0
        or str(geometry["trial_orbitals"]) != "X"
    ):
        raise ValueError("geometry/OW parameters differ from the locked pilot")
    if config["walls"] != {
        "soft": {"dw_truncation": False, "meas_slab_only": False},
        "hard": {"dw_truncation": True, "meas_slab_only": True},
    }:
        raise ValueError("soft/hard wall definitions differ from the locked pilot")
    expected_dynamics = {
        "burn_in_cycles": 40,
        "ramp_cycles": 16,
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "init_mode": "default",
        "state_representation": "physical_frame",
        "physical_covariance_update": "rank1",
        "controller_twist_gauge": "uniform",
        "dtype": "complex128",
        "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
    }
    if dynamics != expected_dynamics:
        raise ValueError("dynamics differ from the locked online-ramp contract")
    if config["twist"]["directions"] != {"ccw": 1, "cw": -1}:
        raise ValueError("twist directions must be ccw=+1 and cw=-1")
    if config["twist"].get("spectral_origin_offset") is not None:
        raise ValueError("the online ramp must start at exact zero flux")
    count = int(config["ensemble"]["samples_per_wall"])
    if count not in (1, 10):
        raise ValueError("only the S1 smoke and S10 production ensembles are supported")
    regions = config["regions"]
    if regions != {
        "left_x_start": 0,
        "left_x_stop_exclusive": 8,
        "right_x_start": 8,
        "right_x_stop_exclusive": 16,
    }:
        raise ValueError("left/right regions must be the two x half-systems")


def _seed(root_seed: int, wall: str, sample_id: int, stage: int, direction: str = "") -> int:
    wall_code = {"soft": 0, "hard": 1}[str(wall)]
    direction_code = {"": 0, "ccw": 1, "cw": 2}[str(direction)]
    sequence = np.random.SeedSequence(
        int(root_seed), spawn_key=(wall_code, int(sample_id), int(stage), direction_code)
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def expand_burnin_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    count = int(config["ensemble"]["samples_per_wall"])
    tasks = [
        {
            "stage": "burnin",
            "task_id": f"burnin_{wall}_sample_{sample_id:03d}",
            "wall": wall,
            "sample_id": sample_id,
            "seed": _seed(config["root_seed"], wall, sample_id, 0),
        }
        for wall in ("soft", "hard")
        for sample_id in range(count)
    ]
    _require_unique_tasks(tasks)
    return tasks


def expand_ramp_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    count = int(config["ensemble"]["samples_per_wall"])
    tasks = [
        {
            "stage": "ramp",
            "task_id": f"ramp_{wall}_{direction}_sample_{sample_id:03d}",
            "wall": wall,
            "direction": direction,
            "sigma": int(config["twist"]["directions"][direction]),
            "sample_id": sample_id,
            "seed": _seed(config["root_seed"], wall, sample_id, 1, direction),
            "burnin_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
        }
        for wall in ("soft", "hard")
        for sample_id in range(count)
        for direction in ("ccw", "cw")
    ]
    _require_unique_tasks(tasks)
    return tasks


def _require_unique_tasks(tasks: list[dict[str, Any]]) -> None:
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("task IDs are not unique")
    if len({task["seed"] for task in tasks}) != len(tasks):
        raise RuntimeError("task seeds are not unique")


def result_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    if task["stage"] == "burnin":
        root = Path(output_root) / "burnins" / task["wall"]
        stem = f"sample_{int(task['sample_id']):03d}"
    else:
        root = Path(output_root) / "ramps" / task["wall"] / task["direction"]
        stem = f"sample_{int(task['sample_id']):03d}"
    result = root / f"{stem}.npz"
    return result, result.with_suffix(".completion.json")


def failure_path(output_root: Path, task: dict[str, Any]) -> Path:
    return Path(output_root) / "failures" / f"{task['task_id']}.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata(task: dict[str, Any], config_hash: str, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        **task,
        "config_hash": config_hash,
        "source_hashes": hashes,
        "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
    }


def publish_pair(
    output_root: Path,
    task: dict[str, Any],
    arrays: dict[str, Any],
    config_hash: str,
    hashes: dict[str, str],
    *,
    burnin_sha256: str | None = None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    result, completion = result_paths(output_root, task)
    arrays = dict(arrays)
    arrays["metadata_json"] = np.asarray(canonical_json(_metadata(task, config_hash, hashes)))
    _atomic_npz(result, arrays)
    result_record = {
        "name": result.name,
        "bytes": int(result.stat().st_size),
        "sha256": sha256_path(result),
    }
    payload = {
        "schema": COMPLETION_SCHEMA,
        **_metadata(task, config_hash, hashes),
        "result": result_record,
        "burnin_sha256": burnin_sha256,
        "elapsed_seconds": float(elapsed_seconds),
        "completed_unix": time.time(),
    }
    _atomic_json(completion, payload)
    return result_record


def verify_pair(
    output_root: Path,
    task: dict[str, Any],
    config_hash: str,
    hashes: dict[str, str],
    *,
    burnin_sha256: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    result, completion_path = result_paths(output_root, task)
    if not result.is_file() or not completion_path.is_file():
        return False, "missing result/completion pair", None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "schema": COMPLETION_SCHEMA,
            **_metadata(task, config_hash, hashes),
            "burnin_sha256": burnin_sha256,
        }
        for key, value in expected.items():
            if completion.get(key) != value:
                return False, f"completion {key} mismatch", None
        record = completion["result"]
        if record.get("name") != result.name:
            return False, "result filename mismatch", None
        if int(record.get("bytes", -1)) != result.stat().st_size:
            return False, "result byte-count mismatch", None
        if record.get("sha256") != sha256_path(result):
            return False, "result checksum mismatch", None
        with np.load(result, allow_pickle=False) as saved:
            schema = str(np.asarray(saved["schema"]).item())
            wanted_schema = BURNIN_SCHEMA if task["stage"] == "burnin" else RAMP_SCHEMA
            if schema != wanted_schema:
                return False, "result schema mismatch", None
            metadata = json.loads(str(np.asarray(saved["metadata_json"]).item()))
            if metadata != _metadata(task, config_hash, hashes):
                return False, "result identity mismatch", None
            if task["stage"] == "burnin":
                frame = np.asarray(saved["frame"])
                if frame.dtype != np.complex128 or frame.ndim != 2:
                    return False, "burn-in frame dtype/shape mismatch", None
            else:
                for key in (
                    "phi",
                    "N_left",
                    "N_right",
                    "N_total",
                    "delta_N_left",
                    "delta_N_right",
                    "source_A_left",
                    "source_A_right",
                    "q_x_raw",
                    "q_x_corrected",
                    "rank",
                ):
                    if np.asarray(saved[key]).shape != (17,):
                        return False, f"ramp array {key} has the wrong shape", None
                if str(np.asarray(saved["burnin_sha256"]).item()) != str(burnin_sha256):
                    return False, "ramp burn-in checksum mismatch", None
        return True, "verified", completion
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _model(config: dict[str, Any], wall: str) -> classA_U1FGTN:
    geometry = config["geometry"]
    wall_config = config["walls"][wall]
    return classA_U1FGTN(
        Nx=int(geometry["Nx"]),
        Ny=int(geometry["Ny"]),
        DW=True,
        nshell=int(geometry["nshell"]),
        filling_frac=float(geometry["filling_frac"]),
        alpha_1=float(geometry["alpha_1"]),
        alpha_2=float(geometry["alpha_2"]),
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_interval=tuple(int(value) for value in geometry["dw_interval"]),
        dw_truncation=bool(wall_config["dw_truncation"]),
        twist_y=0.0,
    )


def _engine_kwargs(config: dict[str, Any], wall: str, *, cycles: int, seed: int) -> dict[str, Any]:
    dynamics = config["dynamics"]
    return {
        "G_history": False,
        "progress": False,
        "cycles": int(cycles),
        "postselect": False,
        "postselect_probability": 0.0,
        "perfect_correction": True,
        "samples": 1,
        "parallelize_samples": False,
        "init_mode": "default",
        "save": False,
        "sequence": "raster_y",
        "meas_slab_only": bool(config["walls"][wall]["meas_slab_only"]),
        "random_seed": int(seed),
        "physical_covariance_update": "rank1",
        "state_representation": "physical_frame",
        "return_native_state": True,
        "require_no_covariance_materialization": True,
    }


def _mode_x(nx: int, ny: int) -> np.ndarray:
    return np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)


def _frame_observables(state: Any, nx: int, ny: int) -> tuple[float, float, float, np.ndarray]:
    frame = np.asarray(state.physical_frame)
    if frame.dtype != np.complex128:
        raise TypeError(f"expected complex128 occupied frame, got {frame.dtype}")
    occupation = np.real(np.sum(np.abs(frame) ** 2, axis=1))
    x = _mode_x(nx, ny)
    density_x = np.asarray([occupation[x == value].sum() for value in range(nx)])
    left = float(density_x[: nx // 2].sum())
    right = float(density_x[nx // 2 :].sum())
    return left, right, float(left + right), density_x


class BurninObserver:
    def __init__(self, nx: int, ny: int, cycles: int, progress_queue: Any) -> None:
        self.nx, self.ny, self.cycles = int(nx), int(ny), int(cycles)
        self.progress_queue = progress_queue
        self.initial_total: float | None = None
        self.final_total: float | None = None
        self.net_injected_charge = 0
        self.feedback_event_count = 0

    def event(self, *, outcome_occupied: bool, target_occupied: bool, **_: Any) -> None:
        delta = int(bool(target_occupied)) - int(bool(outcome_occupied))
        self.net_injected_charge += delta
        self.feedback_event_count += abs(delta)

    def cycle(self, *, cycle: int, state: Any, **_: Any) -> None:
        _, _, total, _ = _frame_observables(state, self.nx, self.ny)
        if int(cycle) == 0:
            self.initial_total = total
        if int(cycle) == self.cycles:
            self.final_total = total
        if int(cycle) > 0 and self.progress_queue is not None:
            self.progress_queue.put(("burnin", 1))


class RampObserver:
    def __init__(
        self,
        model: classA_U1FGTN,
        config: dict[str, Any],
        phi: np.ndarray,
        progress_queue: Any,
    ) -> None:
        self.model = model
        self.nx = int(config["geometry"]["Nx"])
        self.ny = int(config["geometry"]["Ny"])
        self.phi = np.asarray(phi, dtype=np.float64)
        self.progress_queue = progress_queue
        shape = (self.phi.size,)
        self.N_left = np.full(shape, np.nan)
        self.N_right = np.full(shape, np.nan)
        self.N_total = np.full(shape, np.nan)
        self.rank = np.full(shape, -1, dtype=np.int64)
        self.source_A_left = np.zeros(shape)
        self.source_A_right = np.zeros(shape)
        self.net_injected_charge = np.zeros(shape, dtype=np.int64)
        self.injection_count = np.zeros(shape, dtype=np.int64)
        self.density_x = np.full((self.phi.size, self.nx), np.nan)
        self.source_A_x = np.zeros((self.phi.size, self.nx))
        self._A_x = np.zeros(self.nx, dtype=np.float64)
        self._net = 0
        self._count = 0
        self._max_source_partition_residual = 0.0
        self._seen = np.zeros(shape, dtype=bool)

    def event(
        self,
        *,
        cycle: int,
        site_id: int,
        channel: str,
        outcome_occupied: bool,
        target_occupied: bool,
        **_: Any,
    ) -> None:
        x_center = int(site_id) % self.nx
        y_center = int(site_id) // self.nx
        attribute = {"Ap": "WF_Ap", "Am": "WF_Am", "Bp": "WF_Bp", "Bm": "WF_Bm"}[
            str(channel)
        ]
        mode = np.asarray(getattr(self.model, attribute)[:, x_center, y_center])
        norm = float(np.real(np.vdot(mode, mode)))
        if not np.isfinite(norm) or norm <= 0.0:
            raise FloatingPointError("encountered a non-normalizable OW source mode")
        weights = np.abs(mode) ** 2 / norm
        mode_x = _mode_x(self.nx, self.ny)
        weight_x = np.asarray([weights[mode_x == value].sum() for value in range(self.nx)])
        residual = abs(float(weight_x.sum()) - 1.0)
        self._max_source_partition_residual = max(
            self._max_source_partition_residual, residual
        )
        delta = int(bool(target_occupied)) - int(bool(outcome_occupied))
        if delta:
            self._A_x += delta * weight_x
            self._net += delta
            self._count += 1

    def cycle(self, *, cycle: int, state: Any, **_: Any) -> None:
        index = int(cycle)
        if not np.isclose(float(self.model.twist_y), float(self.phi[index]), rtol=0.0, atol=1e-14):
            raise RuntimeError(
                f"cycle {index} observed twist {self.model.twist_y}, expected {self.phi[index]}"
            )
        left, right, total, density_x = _frame_observables(state, self.nx, self.ny)
        self.N_left[index] = left
        self.N_right[index] = right
        self.N_total[index] = total
        self.rank[index] = int(state.rank)
        self.source_A_x[index] = self._A_x
        self.source_A_left[index] = float(self._A_x[: self.nx // 2].sum())
        self.source_A_right[index] = float(self._A_x[self.nx // 2 :].sum())
        self.net_injected_charge[index] = self._net
        self.injection_count[index] = self._count
        self.density_x[index] = density_x
        self._seen[index] = True
        if index > 0 and self.progress_queue is not None:
            self.progress_queue.put(("ramp", 1))

    def arrays(self, config: dict[str, Any]) -> dict[str, Any]:
        if not np.all(self._seen):
            raise RuntimeError("the online observer did not receive all 17 cycle boundaries")
        delta_left = self.N_left - self.N_left[0]
        delta_right = self.N_right - self.N_right[0]
        delta_total = self.N_total - self.N_total[0]
        q_x_raw = 0.5 * (delta_right - delta_left)
        q_left = delta_left - self.source_A_left
        q_right = delta_right - self.source_A_right
        q_x_corrected = 0.5 * (q_right - q_left)
        continuity = delta_total - self.net_injected_charge
        corrected_balance = q_left + q_right
        delta_density_x = self.density_x - self.density_x[0]
        corrected_delta_density_x = delta_density_x - self.source_A_x
        tolerance = config["acceptance"]
        if np.max(np.abs(continuity)) > float(tolerance["charge_continuity_tolerance"]):
            raise FloatingPointError("ramp charge continuity exceeded tolerance")
        if np.max(np.abs(corrected_balance)) > float(tolerance["corrected_balance_tolerance"]):
            raise FloatingPointError("source-corrected left/right balance exceeded tolerance")
        if self._max_source_partition_residual > float(tolerance["source_partition_tolerance"]):
            raise FloatingPointError("OW source weights do not partition unity")
        return {
            "schema": np.asarray(RAMP_SCHEMA),
            "cycles": np.arange(self.phi.size, dtype=np.int64),
            "phi": self.phi,
            "N_left": self.N_left,
            "N_right": self.N_right,
            "N_total": self.N_total,
            "delta_N_left": delta_left,
            "delta_N_right": delta_right,
            "delta_N_total": delta_total,
            "q_x_raw": q_x_raw,
            "source_A_left": self.source_A_left,
            "source_A_right": self.source_A_right,
            "source_A_x": self.source_A_x,
            "net_injected_charge": self.net_injected_charge,
            "injection_count": self.injection_count,
            "q_left_corrected": q_left,
            "q_right_corrected": q_right,
            "q_x_corrected": q_x_corrected,
            "charge_continuity_residual": continuity,
            "corrected_balance_residual": corrected_balance,
            "maximum_source_partition_residual": np.asarray(
                self._max_source_partition_residual
            ),
            "rank": self.rank,
            "density_x": self.density_x,
            "delta_density_x": delta_density_x,
            "corrected_delta_density_x": corrected_delta_density_x,
        }


def _record_failure(output_root: Path, task: dict[str, Any], exc: BaseException) -> None:
    _atomic_json(
        failure_path(output_root, task),
        {
            "task_id": task["task_id"],
            "failed_unix": time.time(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _burnin_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    log_path = output_root / "logs" / "tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = _model(config, task["wall"])
                observer = BurninObserver(
                    config["geometry"]["Nx"],
                    config["geometry"]["Ny"],
                    config["dynamics"]["burn_in_cycles"],
                    queue,
                )
                result = model.run_markov_circuit(
                    native_cycle_observer=observer.cycle,
                    native_event_observer=observer.event,
                    **_engine_kwargs(
                        config,
                        task["wall"],
                        cycles=config["dynamics"]["burn_in_cycles"],
                        seed=task["seed"],
                    ),
                )
        native = result["native_final"]
        continuity = float(observer.final_total - observer.initial_total - observer.net_injected_charge)
        if abs(continuity) > float(config["acceptance"]["charge_continuity_tolerance"]):
            raise FloatingPointError(f"burn-in charge continuity failed: {continuity:.3e}")
        arrays = {
            "schema": np.asarray(BURNIN_SCHEMA),
            "frame": np.asarray(native["frame"], dtype=np.complex128),
            "rank": np.asarray(native["rank"], dtype=np.int64),
            "min_rank": np.asarray(native["min_rank"], dtype=np.int64),
            "max_rank": np.asarray(native["max_rank"], dtype=np.int64),
            "log_weight": np.asarray(native["log_weight"], dtype=np.float64),
            "gram_residual": np.asarray(native["gram_residual"], dtype=np.float64),
            "initial_total_charge": np.asarray(observer.initial_total),
            "final_total_charge": np.asarray(observer.final_total),
            "net_injected_charge": np.asarray(observer.net_injected_charge, dtype=np.int64),
            "feedback_event_count": np.asarray(observer.feedback_event_count, dtype=np.int64),
            "charge_continuity_residual": np.asarray(continuity),
        }
        record = publish_pair(
            output_root,
            task,
            arrays,
            config_hash,
            hashes,
            elapsed_seconds=time.perf_counter() - started,
        )
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"], "result": record}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def _load_burnin(output_root: Path, burnin_task: dict[str, Any]) -> tuple[np.ndarray, str]:
    result, completion_path = result_paths(output_root, burnin_task)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    with np.load(result, allow_pickle=False) as saved:
        frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
    return frame, str(completion["result"]["sha256"])


def schedule_for(task: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    cycles = int(config["dynamics"]["ramp_cycles"])
    return np.asarray(
        [int(task["sigma"]) * 2.0 * math.pi * index / cycles for index in range(cycles + 1)],
        dtype=np.float64,
    )


def _ramp_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, burnin_task, config, output_text, config_hash, hashes, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    log_path = output_root / "logs" / "tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame, burnin_sha = _load_burnin(output_root, burnin_task)
        phi = schedule_for(task, config)
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = _model(config, task["wall"])
                observer = RampObserver(model, config, phi, queue)
                result = model.run_markov_circuit(
                    frame_init=frame,
                    frame_init_prepared=True,
                    controller_twist_schedule=phi,
                    controller_twist_gauge="uniform",
                    native_cycle_observer=observer.cycle,
                    native_event_observer=observer.event,
                    **_engine_kwargs(
                        config,
                        task["wall"],
                        cycles=config["dynamics"]["ramp_cycles"],
                        seed=task["seed"],
                    ),
                )
        arrays = observer.arrays(config)
        arrays.update(
            {
                "burnin_sha256": np.asarray(burnin_sha),
                "final_frame_gram_residual": np.asarray(
                    result["native_final"]["gram_residual"]
                ),
            }
        )
        record = publish_pair(
            output_root,
            task,
            arrays,
            config_hash,
            hashes,
            burnin_sha256=burnin_sha,
            elapsed_seconds=time.perf_counter() - started,
        )
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"], "result": record}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def _burnin_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in expand_burnin_tasks(config)}


def inventory(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    burnins = expand_burnin_tasks(config)
    burnin_rows: dict[str, tuple[bool, str, dict[str, Any] | None]] = {
        task["task_id"]: verify_pair(output_root, task, config_hash, hashes)
        for task in burnins
    }
    lookup = _burnin_lookup(config)
    ramp_rows: dict[str, tuple[bool, str, dict[str, Any] | None]] = {}
    for task in expand_ramp_tasks(config):
        parent = burnin_rows[task["burnin_task_id"]]
        parent_sha = None if not parent[0] else str(parent[2]["result"]["sha256"])
        ramp_rows[task["task_id"]] = verify_pair(
            output_root, task, config_hash, hashes, burnin_sha256=parent_sha
        )
    return {
        "config_hash": config_hash,
        "source_hashes": hashes,
        "burnins": burnin_rows,
        "ramps": ramp_rows,
        "burnin_lookup": lookup,
    }


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    burnin_done = sum(row[0] for row in status["burnins"].values())
    ramp_done = sum(row[0] for row in status["ramps"].values())
    print(f"[campaign] {config['campaign_id']}")
    print(f"[contract] Nx=16 Ny=20 raster-y burn-in=40 ramp=16 S={config['ensemble']['samples_per_wall']} per wall")
    print("[contract] soft+hard, pure half filling, nshell=1, alpha=(1,30), perfect correction, complex128")
    print("[twist] phi_j = sigma*2*pi*j/16; exact zero origin; independent CW/CCW continuation RNG")
    print(f"[output] {output_root.resolve()}")
    print(f"[identity] config_sha256={status['config_hash']}")
    for name, digest in status["source_hashes"].items():
        print(f"[source] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] burn-ins verified={burnin_done}/{len(status['burnins'])} pending={len(status['burnins'])-burnin_done}")
    print(f"[resume] ramps verified={ramp_done}/{len(status['ramps'])} pending={len(status['ramps'])-ramp_done}")


def _drain_progress(queue: Any, cycle_bar: Any) -> None:
    while True:
        try:
            _, count = queue.get_nowait()
        except Exception:
            return
        cycle_bar.update(int(count))


def run_stage(
    stage: str,
    config: dict[str, Any],
    output_root: Path,
    workers: int,
    *,
    resume: bool,
) -> None:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    if stage == "burnin":
        tasks = expand_burnin_tasks(config)
        parent_lookup: dict[str, dict[str, Any]] = {}
    else:
        tasks = expand_ramp_tasks(config)
        parent_lookup = _burnin_lookup(config)
    verified: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], str | None]] = []
    for task in tasks:
        burnin_sha = None
        if stage == "ramp":
            parent = parent_lookup[task["burnin_task_id"]]
            parent_ok, parent_reason, parent_completion = verify_pair(
                output_root, parent, config_hash, hashes
            )
            if not parent_ok:
                raise RuntimeError(
                    f"ramp parent {parent['task_id']} is not verified: {parent_reason}"
                )
            burnin_sha = str(parent_completion["result"]["sha256"])
        ok, _, _ = verify_pair(
            output_root, task, config_hash, hashes, burnin_sha256=burnin_sha
        )
        if resume and ok:
            verified.append(task)
        else:
            pending.append((task, burnin_sha))
    cycles = int(
        config["dynamics"]["burn_in_cycles" if stage == "burnin" else "ramp_cycles"]
    )
    print(f"[{stage}] verified={len(verified)} pending={len(pending)} workers={workers}")
    if not pending:
        return
    context = mp.get_context("spawn")
    with mp.Manager() as manager:
        queue = manager.Queue()
        cycle_bar = tqdm(
            total=len(tasks) * cycles,
            initial=len(verified) * cycles,
            desc=f"{stage} cycles",
            unit="cycle",
            position=0,
        )
        task_bar = tqdm(
            total=len(tasks),
            initial=len(verified),
            desc=f"{stage} tasks",
            unit="task",
            position=1,
        )
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=min(int(workers), len(pending)), mp_context=context) as pool:
            futures = set()
            for task, _ in pending:
                if stage == "burnin":
                    payload = (task, config, str(output_root), config_hash, hashes, queue)
                    futures.add(pool.submit(_burnin_worker, payload))
                else:
                    payload = (
                        task,
                        parent_lookup[task["burnin_task_id"]],
                        config,
                        str(output_root),
                        config_hash,
                        hashes,
                        queue,
                    )
                    futures.add(pool.submit(_ramp_worker, payload))
            while futures:
                done, futures = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                _drain_progress(queue, cycle_bar)
                for future in done:
                    row = future.result()
                    task_bar.update(1)
                    if not row["ok"]:
                        failures.append(f"{row['task_id']}: {row['error']}")
                        tqdm.write(f"[{stage} failure] {failures[-1]}")
        _drain_progress(queue, cycle_bar)
        cycle_bar.close()
        task_bar.close()
    if failures:
        raise RuntimeError(f"{stage} failed for {len(failures)} task(s); see failure JSON/logs")


def write_campaign_identity(config: dict[str, Any], output_root: Path) -> None:
    payload = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_hash": scientific_config_hash(config),
        "source_hashes": source_hashes(),
        "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
        "configuration": config,
    }
    _atomic_json(output_root / "campaign_identity.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "burnin", "ramp", "all"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true", help="analyze after all verified ramps exist")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or config["execution"]["workers"])
    if workers <= 0:
        raise ValueError("workers must be positive")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    print(f"[execution] workers={workers} affinity={affinity or 'unavailable'} BLAS_threads={config['execution']['blas_threads']}")
    status = inventory(config, output_root)
    print_inventory(config, output_root, status)
    write_campaign_identity(config, output_root)
    if args.stage == "report":
        return
    if args.stage in ("burnin", "all"):
        run_stage("burnin", config, output_root, workers, resume=args.resume)
    if args.stage in ("ramp", "all"):
        run_stage("ramp", config, output_root, workers, resume=args.resume)
    final = inventory(config, output_root)
    print_inventory(config, output_root, final)
    if args.stage == "all" or args.analyze:
        if not all(row[0] for row in final["ramps"].values()):
            raise RuntimeError("analysis requires every ramp task to be verified")
        import analyze_online_flux_ramp

        analyze_online_flux_ramp.analyze(config, output_root)
    print("[complete] online flux-ramp campaign finished successfully")


if __name__ == "__main__":
    main()
