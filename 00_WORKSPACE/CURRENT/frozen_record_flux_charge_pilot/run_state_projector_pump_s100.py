#!/usr/bin/env python3
"""S100 monitored burn-ins followed by a static state-projector flux pump."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import contextlib
import hashlib
import json
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


CAMPAIGN_SCHEMA = "state_projector_pump_campaign_v1"
BURNIN_SCHEMA = "state_projector_pump_burnin_v1"
PUMP_SCHEMA = "state_projector_pump_path_v1"
COMPLETION_SCHEMA = "state_projector_pump_completion_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.state_projector_pump_n20x24_s100_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N20x24_state_projector_pump_s100_v1"
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


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "root_seed", "geometry", "walls", "dynamics",
        "projector_pump", "regions", "ensemble", "acceptance",
    )
    return hashlib.sha256(
        canonical_json({key: config[key] for key in keys}).encode("utf-8")
    ).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    if config.get("campaign_id") != "N20x24_state_projector_pump_s100_v1":
        raise ValueError("unexpected campaign identity")
    geometry = config["geometry"]
    if geometry != {
        "Nx": 20, "Ny": 24, "DW": True, "dw_interval": [5, 15], "nshell": 1,
        "filling_frac": 0.5, "alpha_1": 1.0, "alpha_2": 30.0,
        "trial_orbitals": "X",
    }:
        raise ValueError("geometry or OW parameters differ from the locked campaign")
    if config["walls"] != {
        "soft": {"dw_truncation": False, "meas_slab_only": False},
        "hard": {"dw_truncation": True, "meas_slab_only": True},
    }:
        raise ValueError("soft/hard wall definitions changed")
    if config["dynamics"] != {
        "burn_in_cycles": 48, "sequence": "raster_y", "perfect_correction": True,
        "postselect": False, "postselect_probability": 0.0, "init_mode": "default",
        "state_representation": "physical_frame", "physical_covariance_update": "rank1",
        "dtype": "complex128", "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
    }:
        raise ValueError("burn-in dynamics differ from the locked contract")
    pump = config["projector_pump"]
    if int(pump["grid_intervals"]) != 64 or float(pump["regulator"]) != 1e-7:
        raise ValueError("the pump requires 64 intervals and the signed 1e-7 regulator")
    if pump["directions"] != {"ccw": 1, "cw": -1}:
        raise ValueError("direction convention changed")
    if int(config["ensemble"]["samples_per_wall"]) != 100:
        raise ValueError("the production ensemble requires 100 trajectories per wall")
    if config["regions"] != {
        "left_x_start": 0, "left_x_stop_exclusive": 10,
        "right_x_start": 10, "right_x_stop_exclusive": 20,
    }:
        raise ValueError("left/right half-system regions changed")


def _seed(root_seed: int, wall: str, sample_id: int) -> int:
    sequence = np.random.SeedSequence(
        int(root_seed), spawn_key=({"soft": 0, "hard": 1}[wall], int(sample_id))
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def burnin_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "stage": "burnin",
            "task_id": f"burnin_{wall}_sample_{sample_id:03d}",
            "wall": wall,
            "sample_id": sample_id,
            "seed": _seed(config["root_seed"], wall, sample_id),
        }
        for wall in ("soft", "hard")
        for sample_id in range(int(config["ensemble"]["samples_per_wall"]))
    ]
    _require_unique(rows, require_seed=True)
    return rows


def pump_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "stage": "pump",
            "task_id": f"pump_{wall}_{direction}_sample_{sample_id:03d}",
            "wall": wall,
            "direction": direction,
            "sigma": int(config["projector_pump"]["directions"][direction]),
            "sample_id": sample_id,
            "burnin_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
        }
        for wall in ("soft", "hard")
        for sample_id in range(int(config["ensemble"]["samples_per_wall"]))
        for direction in ("ccw", "cw")
    ]
    _require_unique(rows, require_seed=False)
    return rows


def _require_unique(rows: list[dict[str, Any]], *, require_seed: bool) -> None:
    if len({row["task_id"] for row in rows}) != len(rows):
        raise RuntimeError("task IDs are not unique")
    if require_seed and len({row["seed"] for row in rows}) != len(rows):
        raise RuntimeError("trajectory seeds are not unique")


def result_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    if task["stage"] == "burnin":
        root = Path(output_root) / "burnins" / task["wall"]
    else:
        root = Path(output_root) / "projector_pump" / task["wall"] / task["direction"]
    result = root / f"sample_{int(task['sample_id']):03d}.npz"
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
    return {**task, "config_hash": config_hash, "source_hashes": hashes}


def publish_pair(
    output_root: Path,
    task: dict[str, Any],
    arrays: dict[str, Any],
    config_hash: str,
    hashes: dict[str, str],
    elapsed_seconds: float,
    burnin_sha256: str | None = None,
) -> None:
    result, completion = result_paths(output_root, task)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(canonical_json(_metadata(task, config_hash, hashes)))
    _atomic_npz(result, payload)
    record = {"name": result.name, "bytes": result.stat().st_size, "sha256": sha256_path(result)}
    _atomic_json(
        completion,
        {
            "schema": COMPLETION_SCHEMA,
            **_metadata(task, config_hash, hashes),
            "burnin_sha256": burnin_sha256,
            "result": record,
            "elapsed_seconds": float(elapsed_seconds),
            "completed_unix": time.time(),
        },
    )


def verify_pair(
    output_root: Path,
    task: dict[str, Any],
    config_hash: str,
    hashes: dict[str, str],
    config: dict[str, Any],
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
        if record.get("name") != result.name or int(record.get("bytes", -1)) != result.stat().st_size:
            return False, "result name or byte count mismatch", None
        if record.get("sha256") != sha256_path(result):
            return False, "result checksum mismatch", None
        with np.load(result, allow_pickle=False) as saved:
            expected_schema = BURNIN_SCHEMA if task["stage"] == "burnin" else PUMP_SCHEMA
            if str(np.asarray(saved["schema"]).item()) != expected_schema:
                return False, "result schema mismatch", None
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != _metadata(task, config_hash, hashes):
                return False, "result identity mismatch", None
            if task["stage"] == "burnin":
                frame = np.asarray(saved["frame"])
                rank = int(np.asarray(saved["rank"]).item())
                if frame.dtype != np.complex128 or frame.shape != (960, rank):
                    return False, "burn-in frame dtype, dimension, or rank mismatch", None
            else:
                if str(np.asarray(saved["burnin_sha256"]).item()) != str(burnin_sha256):
                    return False, "pump burn-in checksum mismatch", None
                count = int(config["projector_pump"]["grid_intervals"]) + 1
                for key in (
                    "phi", "path_fraction", "continued_delta_N_left",
                    "continued_delta_N_right", "continued_delta_N_total", "continued_q_x",
                    "instantaneous_delta_N_left", "instantaneous_delta_N_right",
                    "instantaneous_delta_N_total", "instantaneous_q_x", "instantaneous_rank_gap",
                    "principal_overlap", "selected_weight_floor",
                ):
                    if np.asarray(saved[key]).shape != (count,):
                        return False, f"pump field {key} has the wrong shape", None
                if not np.array_equal(np.asarray(saved["phi"]), flux_grid(config, task["sigma"])):
                    return False, "pump flux grid mismatch", None
                arrays = {
                    key: np.array(saved[key], copy=True)
                    for key in saved.files
                    if key not in {"metadata_json", "burnin_sha256"}
                }
                validate_pump_arrays(arrays, config)
        return True, "verified", completion
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _model(config: dict[str, Any], wall: str) -> classA_U1FGTN:
    geometry, wall_config = config["geometry"], config["walls"][wall]
    return classA_U1FGTN(
        Nx=int(geometry["Nx"]), Ny=int(geometry["Ny"]), DW=True,
        nshell=int(geometry["nshell"]), filling_frac=float(geometry["filling_frac"]),
        alpha_1=float(geometry["alpha_1"]), alpha_2=float(geometry["alpha_2"]),
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_interval=tuple(int(value) for value in geometry["dw_interval"]),
        dw_truncation=bool(wall_config["dw_truncation"]), twist_y=0.0,
    )


def _engine_kwargs(config: dict[str, Any], wall: str, seed: int) -> dict[str, Any]:
    return {
        "G_history": False, "progress": False,
        "cycles": int(config["dynamics"]["burn_in_cycles"]),
        "postselect": False, "postselect_probability": 0.0,
        "perfect_correction": True, "samples": 1, "parallelize_samples": False,
        "init_mode": "default", "save": False, "sequence": "raster_y",
        "meas_slab_only": bool(config["walls"][wall]["meas_slab_only"]),
        "random_seed": int(seed), "physical_covariance_update": "rank1",
        "state_representation": "physical_frame", "return_native_state": True,
        "require_no_covariance_materialization": True,
    }


def _mode_x(nx: int, ny: int) -> np.ndarray:
    return np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)


def _frame_charge(frame: np.ndarray, nx: int, ny: int) -> tuple[float, float, np.ndarray]:
    density = np.real(np.sum(np.abs(frame) ** 2, axis=1))
    mode_x = _mode_x(nx, ny)
    density_x = np.asarray([density[mode_x == x].sum() for x in range(nx)])
    left = float(density_x[: nx // 2].sum())
    right = float(density_x[nx // 2 :].sum())
    return left, right, density_x


class BurninObserver:
    def __init__(self, config: dict[str, Any], queue: Any) -> None:
        self.nx, self.ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
        self.cycles = int(config["dynamics"]["burn_in_cycles"])
        self.queue = queue
        self.initial_total: float | None = None
        self.final_total: float | None = None
        self.net_injected_charge = 0
        self.feedback_event_count = 0

    def event(self, *, outcome_occupied: bool, target_occupied: bool, **_: Any) -> None:
        delta = int(bool(target_occupied)) - int(bool(outcome_occupied))
        self.net_injected_charge += delta
        self.feedback_event_count += abs(delta)

    def cycle(self, *, cycle: int, state: Any, **_: Any) -> None:
        frame = np.asarray(state.physical_frame)
        left, right, _ = _frame_charge(frame, self.nx, self.ny)
        if int(cycle) == 0:
            self.initial_total = left + right
        if int(cycle) == self.cycles:
            self.final_total = left + right
        if int(cycle) > 0 and self.queue is not None:
            self.queue.put(1)


def _record_failure(output_root: Path, task: dict[str, Any], exc: BaseException) -> None:
    _atomic_json(
        failure_path(output_root, task),
        {
            "task_id": task["task_id"], "failed_unix": time.time(),
            "error_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _burnin_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, queue = payload
    output_root, started = Path(output_text), time.perf_counter()
    log_path = output_root / "logs" / "tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = _model(config, task["wall"])
                observer = BurninObserver(config, queue)
                result = model.run_markov_circuit(
                    native_cycle_observer=observer.cycle,
                    native_event_observer=observer.event,
                    **_engine_kwargs(config, task["wall"], task["seed"]),
                )
        native = result["native_final"]
        frame = np.asarray(native["frame"], dtype=np.complex128)
        rank = int(native["rank"])
        if frame.shape != (960, rank) or frame.dtype != np.complex128:
            raise RuntimeError("canonical engine returned an invalid occupied frame")
        continuity = float(observer.final_total - observer.initial_total - observer.net_injected_charge)
        if abs(continuity) > float(config["acceptance"]["burnin_charge_continuity_tolerance"]):
            raise FloatingPointError(f"burn-in charge continuity failed: {continuity:.3e}")
        left, right, density_x = _frame_charge(frame, 20, 24)
        arrays = {
            "schema": np.asarray(BURNIN_SCHEMA), "frame": frame,
            "rank": np.asarray(rank, dtype=np.int64),
            "min_rank": np.asarray(native["min_rank"], dtype=np.int64),
            "max_rank": np.asarray(native["max_rank"], dtype=np.int64),
            "log_weight": np.asarray(native["log_weight"], dtype=np.float64),
            "gram_residual": np.asarray(native["gram_residual"], dtype=np.float64),
            "N_left": np.asarray(left), "N_right": np.asarray(right),
            "density_x": density_x,
            "initial_total_charge": np.asarray(observer.initial_total),
            "final_total_charge": np.asarray(observer.final_total),
            "net_injected_charge": np.asarray(observer.net_injected_charge, dtype=np.int64),
            "feedback_event_count": np.asarray(observer.feedback_event_count, dtype=np.int64),
            "charge_continuity_residual": np.asarray(continuity),
        }
        publish_pair(output_root, task, arrays, config_hash, hashes, time.perf_counter() - started)
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def flux_grid(config: dict[str, Any], sigma: int) -> np.ndarray:
    intervals = int(config["projector_pump"]["grid_intervals"])
    epsilon = float(config["projector_pump"]["regulator"])
    return -int(sigma) * epsilon + int(sigma) * np.linspace(0.0, 2.0 * np.pi, intervals + 1)


def _coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.repeat(np.arange(ny, dtype=np.int64), 2 * nx)
    dy = y[:, None] - y[None, :]
    dy = ((dy + ny // 2) % ny) - ny // 2
    return y, dy


def _select_continued_frame(
    previous: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, rank: int
) -> tuple[np.ndarray, float, float]:
    weights = np.real(np.sum(np.abs(previous.conj().T @ eigenvectors) ** 2, axis=0))
    order = np.lexsort((eigenvalues, -weights))
    selected = np.asarray(eigenvectors[:, order[:rank]], dtype=np.complex128)
    singular = np.linalg.svd(previous.conj().T @ selected, compute_uv=False)
    return selected, float(np.min(singular)), float(np.min(weights[order[:rank]]))


def compute_pump_path(
    frame: np.ndarray, task: dict[str, Any], config: dict[str, Any], queue: Any = None
) -> dict[str, Any]:
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    dimension, rank = frame.shape
    if dimension != 2 * nx * ny or not (0 < rank < dimension):
        raise RuntimeError("burn-in occupied frame has an invalid dimension or rank")
    gram = float(np.max(np.abs(frame.conj().T @ frame - np.eye(rank))))
    if gram > float(config["acceptance"]["frame_gram_tolerance"]):
        raise RuntimeError(f"burn-in frame is not orthonormal: {gram:.3e}")
    projector0 = frame @ frame.conj().T
    projector_residual = float(np.max(np.abs(projector0 @ projector0 - projector0)))
    h0 = np.eye(dimension, dtype=np.complex128) - 2.0 * projector0
    involution_residual = float(np.max(np.abs(h0 @ h0 - np.eye(dimension))))
    y, dy = _coordinates(nx, ny)
    density0_left, density0_right, density0_x = _frame_charge(frame, nx, ny)
    phi = flux_grid(config, task["sigma"])
    count = len(phi)
    c_left, c_right, i_left, i_right = (np.empty(count) for _ in range(4))
    c_density = np.empty((count, nx))
    i_density = np.empty((count, nx))
    gaps, overlaps, weight_floors = (np.empty(count) for _ in range(3))
    previous, h_start, h_end = frame, None, None
    for point, value in enumerate(phi):
        h_phi = np.asarray(h0 * np.exp(1j * float(value) * dy / ny), dtype=np.complex128)
        h_phi = 0.5 * (h_phi + h_phi.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(h_phi)
        continued, overlaps[point], weight_floors[point] = _select_continued_frame(
            previous, eigenvalues, eigenvectors, rank
        )
        instantaneous = np.asarray(eigenvectors[:, :rank], dtype=np.complex128)
        c_left[point], c_right[point], c_density[point] = _frame_charge(continued, nx, ny)
        i_left[point], i_right[point], i_density[point] = _frame_charge(instantaneous, nx, ny)
        gaps[point] = float(eigenvalues[rank] - eigenvalues[rank - 1])
        previous = continued
        h_start = h_phi if point == 0 else h_start
        h_end = h_phi
        if queue is not None:
            queue.put(1)
    c_dl, c_dr = c_left - density0_left, c_right - density0_right
    i_dl, i_dr = i_left - density0_left, i_right - density0_right
    large_gauge = np.exp(1j * int(task["sigma"]) * 2.0 * np.pi * y / ny)
    gauged_start = large_gauge[:, None] * h_start * large_gauge.conj()[None, :]
    arrays = {
        "schema": np.asarray(PUMP_SCHEMA), "phi": phi,
        "path_fraction": np.arange(count, dtype=np.float64) / (count - 1),
        "continued_N_left": c_left, "continued_N_right": c_right,
        "continued_delta_N_left": c_dl, "continued_delta_N_right": c_dr,
        "continued_delta_N_total": c_dl + c_dr,
        "continued_q_x": 0.5 * (c_dr - c_dl),
        "instantaneous_N_left": i_left, "instantaneous_N_right": i_right,
        "instantaneous_delta_N_left": i_dl, "instantaneous_delta_N_right": i_dr,
        "instantaneous_delta_N_total": i_dl + i_dr,
        "instantaneous_q_x": 0.5 * (i_dr - i_dl),
        "continued_density_x": c_density, "instantaneous_density_x": i_density,
        "source_density_x": density0_x, "instantaneous_rank_gap": gaps,
        "principal_overlap": overlaps, "selected_weight_floor": weight_floors,
        "rank": np.asarray(rank, dtype=np.int64),
        "source_N_left": np.asarray(density0_left), "source_N_right": np.asarray(density0_right),
        "input_frame_gram_residual": np.asarray(gram),
        "input_projector_residual": np.asarray(projector_residual),
        "flattened_parent_involution_residual": np.asarray(involution_residual),
        "large_gauge_parent_error": np.asarray(float(np.max(np.abs(h_end - gauged_start)))),
    }
    validate_pump_arrays(arrays, config)
    return arrays


def validate_pump_arrays(arrays: dict[str, Any], config: dict[str, Any]) -> None:
    for key, value in arrays.items():
        if key != "schema" and not np.all(np.isfinite(np.asarray(value))):
            raise RuntimeError(f"nonfinite pump field: {key}")
    acceptance = config["acceptance"]
    if float(arrays["input_projector_residual"]) > float(acceptance["projector_tolerance"]):
        raise RuntimeError("burn-in occupation projector is not idempotent")
    if np.max(np.abs(arrays["continued_delta_N_total"])) > float(acceptance["pump_charge_conservation_tolerance"]):
        raise RuntimeError("continued projector violates charge conservation")
    if np.max(np.abs(arrays["instantaneous_delta_N_total"])) > float(acceptance["pump_charge_conservation_tolerance"]):
        raise RuntimeError("instantaneous projector violates charge conservation")
    if float(arrays["large_gauge_parent_error"]) > float(acceptance["large_gauge_tolerance"]):
        raise RuntimeError("flattened parent does not close by the large gauge transformation")
    if abs(float(arrays["continued_q_x"][0])) > float(acceptance["initial_regulator_charge_tolerance"]):
        raise RuntimeError("regulated continuation does not begin at the burn-in projector")
    if abs(float(arrays["instantaneous_q_x"][-1])) > float(acceptance["instantaneous_endpoint_closure_tolerance"]):
        raise RuntimeError("instantaneous occupied projector does not close after one flux quantum")
    if np.min(arrays["principal_overlap"]) < float(acceptance["minimum_principal_overlap_floor"]):
        raise RuntimeError("flux grid is too coarse for reliable projector continuation")


def _load_burnin(output_root: Path, task: dict[str, Any]) -> tuple[np.ndarray, str]:
    result, completion_path = result_paths(output_root, task)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    with np.load(result, allow_pickle=False) as saved:
        frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
    return frame, str(completion["result"]["sha256"])


def _pump_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, parent, config, output_text, config_hash, hashes, queue = payload
    output_root, started = Path(output_text), time.perf_counter()
    try:
        frame, burnin_sha = _load_burnin(output_root, parent)
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            arrays = compute_pump_path(frame, task, config, queue)
        arrays["burnin_sha256"] = np.asarray(burnin_sha)
        publish_pair(
            output_root, task, arrays, config_hash, hashes,
            time.perf_counter() - started, burnin_sha,
        )
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def inventory(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    parents = burnin_tasks(config)
    burnins = {
        task["task_id"]: verify_pair(output_root, task, config_hash, hashes, config)
        for task in parents
    }
    pumps = {}
    for task in pump_tasks(config):
        row = burnins[task["burnin_task_id"]]
        burnin_sha = str(row[2]["result"]["sha256"]) if row[0] else None
        pumps[task["task_id"]] = verify_pair(
            output_root, task, config_hash, hashes, config, burnin_sha
        )
    return {
        "config_hash": config_hash, "source_hashes": hashes,
        "burnins": burnins, "pumps": pumps,
        "burnin_lookup": {task["task_id"]: task for task in parents},
    }


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    burnin_done = sum(row[0] for row in status["burnins"].values())
    pump_done = sum(row[0] for row in status["pumps"].values())
    print(f"[campaign] {config['campaign_id']}")
    print("[burn-in] Nx=20 Ny=24, 48=2Ny cycles, raster-y, pure half filling")
    print("[dynamics] soft+hard, nshell=1, alpha=(1,30), perfect correction, complex128")
    print("[static pump] h_xi=1-2F_xi F_xi^dagger; 64 flux intervals; overlap continuation")
    print("[observable] Delta N_L, Delta N_R, q_x=(Delta N_R-Delta N_L)/2")
    print(f"[output] {output_root.resolve()}")
    print(f"[identity] config_sha256={status['config_hash']}")
    for name, digest in status["source_hashes"].items():
        print(f"[source] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] burn-ins verified={burnin_done}/200 pending={200-burnin_done}")
    print(f"[resume] pump paths verified={pump_done}/400 pending={400-pump_done}")


def _selected(rows: list[dict[str, Any]], sample_ids: set[int] | None) -> list[dict[str, Any]]:
    return [row for row in rows if sample_ids is None or int(row["sample_id"]) in sample_ids]


def _run_stage(
    stage: str, config: dict[str, Any], output_root: Path, workers: int,
    resume: bool, sample_ids: set[int] | None,
) -> None:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    parents = {task["task_id"]: task for task in burnin_tasks(config)}
    rows = _selected(burnin_tasks(config) if stage == "burnin" else pump_tasks(config), sample_ids)
    verified, pending = [], []
    for task in rows:
        burnin_sha = None
        if stage == "pump":
            parent = parents[task["burnin_task_id"]]
            ok, reason, completion = verify_pair(output_root, parent, config_hash, hashes, config)
            if not ok:
                raise RuntimeError(f"pump parent {parent['task_id']} is not verified: {reason}")
            burnin_sha = str(completion["result"]["sha256"])
        ok, _, _ = verify_pair(output_root, task, config_hash, hashes, config, burnin_sha)
        (verified if resume and ok else pending).append(task)
    units = int(config["dynamics"]["burn_in_cycles"]) if stage == "burnin" else int(config["projector_pump"]["grid_intervals"]) + 1
    print(f"[{stage}] verified={len(verified)}/{len(rows)} pending={len(pending)} workers={workers}")
    if not pending:
        return
    context = mp.get_context("spawn")
    with context.Manager() as manager:
        queue = manager.Queue()
        with tqdm(total=len(rows), initial=len(verified), desc=f"{stage} tasks", unit="task", position=0) as task_bar, \
                tqdm(total=len(rows) * units, initial=len(verified) * units,
                     desc=f"{stage} {'cycles' if stage == 'burnin' else 'flux points'}",
                     unit="cycle" if stage == "burnin" else "point", position=1) as unit_bar, \
                ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=context) as pool:
            futures = set()
            for task in pending:
                if stage == "burnin":
                    futures.add(pool.submit(_burnin_worker, (task, config, str(output_root), config_hash, hashes, queue)))
                else:
                    futures.add(pool.submit(_pump_worker, (
                        task, parents[task["burnin_task_id"]], config, str(output_root),
                        config_hash, hashes, queue,
                    )))
            failures = []
            while futures:
                done, futures = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        unit_bar.update(int(queue.get_nowait()))
                    except Exception:
                        break
                for future in done:
                    result = future.result()
                    task_bar.update(1)
                    if not result["ok"]:
                        failures.append(result)
                        task_bar.write(f"[{stage} failure] {result['task_id']}: {result['error']}")
            while True:
                try:
                    unit_bar.update(int(queue.get_nowait()))
                except Exception:
                    break
        if failures:
            raise RuntimeError(f"{stage} failed for {len(failures)} tasks; inspect failures/")


def write_identity(config: dict[str, Any], output_root: Path) -> None:
    _atomic_json(
        output_root / "campaign_identity.json",
        {
            "schema": CAMPAIGN_SCHEMA, "campaign_id": config["campaign_id"],
            "config_hash": scientific_config_hash(config), "source_hashes": source_hashes(),
            "configuration": config,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "burnin", "pump", "all"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sample-ids", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or config["execution"]["workers"])
    sample_ids = None if args.sample_ids is None else set(args.sample_ids)
    if workers < 1 or (sample_ids is not None and not sample_ids.issubset(set(range(100)))):
        raise ValueError("workers must be positive and sample IDs must lie in 0,...,99")
    print(f"[execution] workers={workers} affinity={sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unavailable'}")
    status = inventory(config, output_root)
    print_inventory(config, output_root, status)
    write_identity(config, output_root)
    if args.stage == "report":
        return 0
    if args.stage in ("burnin", "all"):
        _run_stage("burnin", config, output_root, workers, args.resume, sample_ids)
    if args.stage in ("pump", "all"):
        _run_stage("pump", config, output_root, workers, args.resume, sample_ids)
    final = inventory(config, output_root)
    print_inventory(config, output_root, final)
    if args.analyze:
        if not all(row[0] for row in final["pumps"].values()):
            raise RuntimeError("analysis requires all 400 verified pump paths")
        import analyze_state_projector_pump_s100
        analyze_state_projector_pump_s100.analyze(config, output_root)
    print("[complete] state-projector pump command finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
