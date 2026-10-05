#!/usr/bin/env python3
"""Finite-difference Schrödinger evolution of 50 saved Ny=24 endpoint states."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_state_projector_pump_s100 as source_campaign  # noqa: E402


CAMPAIGN_SCHEMA = "parent_schrodinger_rk4_campaign_v1"
RESULT_SCHEMA = "parent_schrodinger_rk4_path_v1"
COMPLETION_SCHEMA = "parent_schrodinger_rk4_completion_v1"
CHECKPOINT_SCHEMA = "parent_schrodinger_rk4_checkpoint_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.parent_schrodinger_rk4_n20x24_s50_tau1e4_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N20x24_parent_schrodinger_rk4_s50_tau1e4_v1"
SOURCE_PATHS = {
    "source_campaign_runner": Path(source_campaign.__file__).resolve(),
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
        "schema", "campaign_id", "source_campaign", "geometry", "evolution",
        "ensemble", "analysis", "acceptance",
    )
    raw = canonical_json({key: config[key] for key in keys}).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    if config.get("campaign_id") != "N20x24_parent_schrodinger_rk4_s50_tau1e4_v1":
        raise ValueError("unexpected campaign identity")
    source = config["source_campaign"]
    expected_ids = list(range(0, 100, 4))
    if source.get("campaign_id") != "N20x24_state_projector_pump_s100_v1":
        raise ValueError("unexpected source campaign")
    if source.get("sample_ids") != expected_ids:
        raise ValueError("the S50 selection must be sample IDs 0,4,...,96")
    if config["geometry"] != {
        "Nx": 20, "Ny": 24, "left_x_stop_exclusive": 10,
        "periodic_direction": "y",
    }:
        raise ValueError("geometry or subsystem partition changed")
    evolution = config["evolution"]
    if (
        float(evolution["ramp_time"]) != 10000.0
        or int(evolution["flux_intervals"]) != 128
        or int(evolution["steps_per_interval"]) != 320
        or evolution["directions"] != {"ccw": 1, "cw": -1}
        or evolution["dtype"] != "complex128"
    ):
        raise ValueError("long-ramp RK4 discretization changed")
    if config["ensemble"] != {
        "walls": ["soft", "hard"], "endpoint_states_total": 50,
        "paths_total": 100,
        "independent_sampling_unit": "saved monitored endpoint trajectory",
    }:
        raise ValueError("expected 25 endpoints per wall and two directions")
    execution = config["execution"]
    if int(execution["checkpoint_every_intervals"]) < 1:
        raise ValueError("checkpoint interval must be positive")
    if int(execution["progress_update_steps"]) < 1:
        raise ValueError("progress update interval must be positive")


def tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "task_id": f"rk4_{wall}_{direction}_sample_{sample_id:03d}",
            "wall": wall,
            "direction": direction,
            "sigma": int(config["evolution"]["directions"][direction]),
            "sample_id": int(sample_id),
            "source_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
        }
        for wall in config["ensemble"]["walls"]
        for sample_id in config["source_campaign"]["sample_ids"]
        for direction in ("ccw", "cw")
    ]
    if len(rows) != 100 or len({row["task_id"] for row in rows}) != 100:
        raise RuntimeError("expected exactly 100 unique CW/CCW paths from 50 endpoints")
    return rows


def result_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(output_root) / "paths" / task["wall"] / task["direction"]
    result = root / f"sample_{int(task['sample_id']):03d}.npz"
    return result, result.with_suffix(".completion.json")


def checkpoint_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(output_root) / "checkpoints" / task["wall"] / task["direction"]
    checkpoint = root / f"sample_{int(task['sample_id']):03d}.checkpoint.npz"
    return checkpoint, checkpoint.with_suffix(".json")


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


def _atomic_npz(path: Path, arrays: dict[str, Any], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata(
    task: dict[str, Any], config_hash: str, hashes: dict[str, str], source_row: dict[str, Any]
) -> dict[str, Any]:
    return {
        **task,
        "config_hash": config_hash,
        "source_hashes": hashes,
        "source_result": source_row,
    }


def source_context(config: dict[str, Any]) -> dict[str, Any]:
    source_config_path = (PROJECT_ROOT / config["source_campaign"]["config"]).resolve()
    source_root = (PROJECT_ROOT / config["source_campaign"]["output_root"]).resolve()
    source_config = source_campaign.load_config(source_config_path)
    source_campaign.validate_config(source_config)
    if source_config["campaign_id"] != config["source_campaign"]["campaign_id"]:
        raise RuntimeError("source campaign ID mismatch")
    source_hash_map = source_campaign.source_hashes()
    source_config_hash = source_campaign.scientific_config_hash(source_config)
    source_tasks = {task["task_id"]: task for task in source_campaign.burnin_tasks(source_config)}
    selected: dict[str, dict[str, Any]] = {}
    for wall in config["ensemble"]["walls"]:
        for sample_id in config["source_campaign"]["sample_ids"]:
            task_id = f"burnin_{wall}_sample_{int(sample_id):03d}"
            task = source_tasks[task_id]
            ok, reason, completion = source_campaign.verify_pair(
                source_root, task, source_config_hash, source_hash_map, source_config
            )
            if not ok or completion is None:
                raise RuntimeError(f"source endpoint is not verified: {task_id}: {reason}")
            path, _ = source_campaign.result_paths(source_root, task)
            selected[task_id] = {
                "path": str(path),
                "name": path.name,
                "bytes": int(completion["result"]["bytes"]),
                "sha256": str(completion["result"]["sha256"]),
                "source_config_hash": source_config_hash,
            }
    if len(selected) != 50:
        raise RuntimeError(f"expected 50 verified source endpoints, found {len(selected)}")
    return {"rows": selected, "root": source_root}


def _load_frame(source_row: dict[str, Any]) -> np.ndarray:
    path = Path(source_row["path"])
    if path.stat().st_size != int(source_row["bytes"]) or sha256_path(path) != source_row["sha256"]:
        raise RuntimeError(f"source endpoint changed after verification: {path}")
    with np.load(path, allow_pickle=False) as saved:
        frame = np.array(saved["frame"], dtype=np.complex128, order="F", copy=True)
        rank = int(np.asarray(saved["rank"]).item())
    if frame.shape != (960, rank) or frame.dtype != np.complex128:
        raise RuntimeError("source occupied frame has invalid shape, rank, or dtype")
    return frame


def _coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)
    y = np.repeat(np.arange(ny, dtype=np.int64), 2 * nx)
    dy = y[:, None] - y[None, :]
    dy = ((dy + ny // 2) % ny) - ny // 2
    return x, y, np.asarray(dy + ny // 2, dtype=np.int8)


def _density_summary(frame: np.ndarray, x: np.ndarray, nx: int) -> tuple[np.ndarray, float, float]:
    density = np.real(np.sum(np.abs(frame) ** 2, axis=1))
    density_x = np.asarray([density[x == value].sum() for value in range(nx)])
    return density_x, float(density_x[: nx // 2].sum()), float(density_x[nx // 2 :].sum())


def _twisted_parent(
    h0: np.ndarray, dy_index: np.ndarray, dy_values: np.ndarray, phi: float, ny: int
) -> np.ndarray:
    phase = np.exp(1j * float(phi) * dy_values / int(ny))
    twisted = h0 * phase[dy_index]
    return np.asarray(0.5 * (twisted + twisted.conj().T), dtype=np.complex128, order="F")


def rk4_step(
    frame: np.ndarray,
    *,
    step: int,
    dt: float,
    ramp_time: float,
    sigma: int,
    h0: np.ndarray,
    dy_index: np.ndarray,
    dy_values: np.ndarray,
    ny: int,
) -> np.ndarray:
    t0 = float(step) * dt
    coefficient = int(sigma) * 2.0 * np.pi / ramp_time
    h1 = _twisted_parent(h0, dy_index, dy_values, coefficient * t0, ny)
    k1 = -1j * (h1 @ frame)
    hm = _twisted_parent(h0, dy_index, dy_values, coefficient * (t0 + 0.5 * dt), ny)
    k2 = -1j * (hm @ (frame + 0.5 * dt * k1))
    k3 = -1j * (hm @ (frame + 0.5 * dt * k2))
    h4 = _twisted_parent(h0, dy_index, dy_values, coefficient * (t0 + dt), ny)
    k4 = -1j * (h4 @ (frame + dt * k3))
    return np.asarray(
        frame + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
        dtype=np.complex128,
        order="F",
    )


def _fix_qr_gauge(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    pre_gram = float(np.max(np.abs(frame.conj().T @ frame - np.eye(frame.shape[1]))))
    q, r = np.linalg.qr(frame, mode="reduced")
    diagonal = np.diag(r)
    phases = np.ones(diagonal.shape, dtype=np.complex128)
    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    q = np.asarray(q * phases[None, :], dtype=np.complex128, order="F")
    post_gram = float(np.max(np.abs(q.conj().T @ q - np.eye(q.shape[1]))))
    return q, pre_gram, post_gram


def _observe(
    frame: np.ndarray,
    *,
    h0: np.ndarray,
    dy_index: np.ndarray,
    dy_values: np.ndarray,
    phi: float,
    x: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, float, float, float, float]:
    density_x, left, right = _density_summary(frame, x, nx)
    h_phi = _twisted_parent(h0, dy_index, dy_values, phi, ny)
    h_frame = h_phi @ frame
    left_mask = x < nx // 2
    z_left = np.sum(np.conj(frame[left_mask]) * h_frame[left_mask])
    current_left = float(2.0 * np.imag(z_left))
    energy = float(np.real(np.sum(np.conj(frame) * h_frame)))
    return density_x, left, right, current_left, energy


def _checkpoint_payload(
    *,
    frame: np.ndarray,
    completed_step: int,
    completed_interval: int,
    arrays: dict[str, np.ndarray],
    maximum_pre_qr_gram: float,
    maximum_post_qr_gram: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": np.asarray(CHECKPOINT_SCHEMA),
        "metadata_json": np.asarray(canonical_json(metadata)),
        "frame": frame,
        "completed_step": np.asarray(completed_step, dtype=np.int64),
        "completed_interval": np.asarray(completed_interval, dtype=np.int64),
        "maximum_pre_qr_gram": np.asarray(maximum_pre_qr_gram),
        "maximum_post_qr_gram": np.asarray(maximum_post_qr_gram),
        **arrays,
    }


def _write_checkpoint(
    output_root: Path,
    task: dict[str, Any],
    *,
    frame: np.ndarray,
    completed_step: int,
    completed_interval: int,
    arrays: dict[str, np.ndarray],
    maximum_pre_qr_gram: float,
    maximum_post_qr_gram: float,
    metadata: dict[str, Any],
) -> None:
    checkpoint, receipt = checkpoint_paths(output_root, task)
    payload = _checkpoint_payload(
        frame=frame,
        completed_step=completed_step,
        completed_interval=completed_interval,
        arrays=arrays,
        maximum_pre_qr_gram=maximum_pre_qr_gram,
        maximum_post_qr_gram=maximum_post_qr_gram,
        metadata=metadata,
    )
    _atomic_npz(checkpoint, payload, compressed=False)
    _atomic_json(
        receipt,
        {
            "schema": CHECKPOINT_SCHEMA,
            "metadata": metadata,
            "completed_step": int(completed_step),
            "completed_interval": int(completed_interval),
            "checkpoint": {
                "name": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_path(checkpoint),
            },
        },
    )


def _load_checkpoint(
    output_root: Path,
    task: dict[str, Any],
    *,
    metadata: dict[str, Any],
    count: int,
    nx: int,
) -> dict[str, Any] | None:
    checkpoint, receipt_path = checkpoint_paths(output_root, task)
    if not checkpoint.exists() and not receipt_path.exists():
        return None
    try:
        if not checkpoint.is_file() or not receipt_path.is_file():
            raise RuntimeError("incomplete checkpoint pair")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        record = receipt["checkpoint"]
        if receipt.get("schema") != CHECKPOINT_SCHEMA or receipt.get("metadata") != metadata:
            raise RuntimeError("checkpoint identity mismatch")
        if record.get("name") != checkpoint.name or int(record.get("bytes", -1)) != checkpoint.stat().st_size:
            raise RuntimeError("checkpoint name or byte count mismatch")
        if record.get("sha256") != sha256_path(checkpoint):
            raise RuntimeError("checkpoint checksum mismatch")
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != CHECKPOINT_SCHEMA:
                raise RuntimeError("checkpoint schema mismatch")
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != metadata:
                raise RuntimeError("checkpoint metadata mismatch")
            frame = np.array(saved["frame"], dtype=np.complex128, order="F", copy=True)
            completed_step = int(np.asarray(saved["completed_step"]).item())
            completed_interval = int(np.asarray(saved["completed_interval"]).item())
            if completed_step < 0 or completed_interval < 0 or completed_interval >= count:
                raise RuntimeError("checkpoint progress is out of bounds")
            arrays = {
                "phi": np.array(saved["phi"], copy=True),
                "time": np.array(saved["time"], copy=True),
                "N_left": np.array(saved["N_left"], copy=True),
                "N_right": np.array(saved["N_right"], copy=True),
                "density_x": np.array(saved["density_x"], copy=True),
                "current_left": np.array(saved["current_left"], copy=True),
                "energy": np.array(saved["energy"], copy=True),
            }
        if frame.ndim != 2 or frame.shape[0] != 2 * nx * 24:
            raise RuntimeError("checkpoint frame shape mismatch")
        expected_shapes = {
            "phi": (count,), "time": (count,), "N_left": (count,),
            "N_right": (count,), "density_x": (count, nx),
            "current_left": (count,), "energy": (count,),
        }
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape:
                raise RuntimeError(f"checkpoint {key} shape mismatch")
            prefix = arrays[key][: completed_interval + 1]
            if not np.all(np.isfinite(prefix)):
                raise RuntimeError(f"checkpoint {key} prefix is nonfinite")
        if completed_step <= 0 and completed_interval != 0:
            raise RuntimeError("checkpoint step/interval mismatch")
        return {
            "frame": frame,
            "completed_step": completed_step,
            "completed_interval": completed_interval,
            "arrays": arrays,
            "maximum_pre_qr_gram": float(np.asarray(saved_value(checkpoint, "maximum_pre_qr_gram"))),
            "maximum_post_qr_gram": float(np.asarray(saved_value(checkpoint, "maximum_post_qr_gram"))),
        }
    except Exception as exc:
        print(f"[checkpoint reset] {task['task_id']}: {type(exc).__name__}: {exc}", flush=True)
        checkpoint.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        return None


def saved_value(path: Path, key: str) -> Any:
    with np.load(path, allow_pickle=False) as saved:
        return np.array(saved[key], copy=True)


def _delete_checkpoint(output_root: Path, task: dict[str, Any]) -> None:
    checkpoint, receipt = checkpoint_paths(output_root, task)
    checkpoint.unlink(missing_ok=True)
    receipt.unlink(missing_ok=True)


def _validate_result_arrays(arrays: dict[str, Any], config: dict[str, Any]) -> None:
    count = int(config["evolution"]["flux_intervals"]) + 1
    nx = int(config["geometry"]["Nx"])
    required = {
        "phi": (count,), "time": (count,), "N_left": (count,),
        "N_right": (count,), "delta_N_left": (count,), "delta_N_right": (count,),
        "delta_N_total": (count,), "q_x": (count,), "density_x": (count, nx),
        "current_left": (count,), "energy": (count,),
    }
    for key, shape in required.items():
        value = np.asarray(arrays[key])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid result field {key}")
    for key, value in arrays.items():
        if not np.all(np.isfinite(np.asarray(value))):
            raise RuntimeError(f"nonfinite result field {key}")
    if np.max(np.abs(arrays["delta_N_total"])) > float(config["acceptance"]["charge_conservation_tolerance"]):
        raise FloatingPointError("unitary evolution violates total-charge conservation")
    if float(arrays["maximum_post_qr_gram_residual"]) > float(config["acceptance"]["post_qr_gram_tolerance"]):
        raise FloatingPointError("QR-stabilized frame is not orthonormal")


def publish_result(
    output_root: Path,
    task: dict[str, Any],
    arrays: dict[str, Any],
    *,
    config_hash: str,
    hashes: dict[str, str],
    source_row: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    result, completion = result_paths(output_root, task)
    metadata = _metadata(task, config_hash, hashes, source_row)
    payload = dict(arrays)
    payload["schema"] = np.asarray(RESULT_SCHEMA)
    payload["metadata_json"] = np.asarray(canonical_json(metadata))
    _atomic_npz(result, payload, compressed=True)
    _atomic_json(
        completion,
        {
            "schema": COMPLETION_SCHEMA,
            "metadata": metadata,
            "result": {
                "name": result.name,
                "bytes": result.stat().st_size,
                "sha256": sha256_path(result),
            },
            "elapsed_seconds": float(elapsed_seconds),
            "completed_unix": time.time(),
        },
    )


def verify_result(
    output_root: Path,
    task: dict[str, Any],
    *,
    config_hash: str,
    hashes: dict[str, str],
    source_row: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    result, completion_path = result_paths(output_root, task)
    if not result.is_file() or not completion_path.is_file():
        return False, "missing result/completion pair", None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        metadata = _metadata(task, config_hash, hashes, source_row)
        if completion.get("schema") != COMPLETION_SCHEMA or completion.get("metadata") != metadata:
            return False, "completion identity mismatch", None
        record = completion["result"]
        if record.get("name") != result.name or int(record.get("bytes", -1)) != result.stat().st_size:
            return False, "result name or byte count mismatch", None
        if record.get("sha256") != sha256_path(result):
            return False, "result checksum mismatch", None
        with np.load(result, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != RESULT_SCHEMA:
                return False, "result schema mismatch", None
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != metadata:
                return False, "result metadata mismatch", None
            arrays = {key: np.array(saved[key], copy=True) for key in saved.files if key not in {"schema", "metadata_json"}}
        _validate_result_arrays(arrays, config)
        return True, "verified", completion
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _initial_arrays(count: int, nx: int) -> dict[str, np.ndarray]:
    return {
        "phi": np.full(count, np.nan),
        "time": np.full(count, np.nan),
        "N_left": np.full(count, np.nan),
        "N_right": np.full(count, np.nan),
        "density_x": np.full((count, nx), np.nan),
        "current_left": np.full(count, np.nan),
        "energy": np.full(count, np.nan),
    }


def compute_path(
    frame0: np.ndarray,
    task: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    metadata: dict[str, Any],
    progress_queue: Any = None,
) -> dict[str, Any]:
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    intervals = int(config["evolution"]["flux_intervals"])
    steps_per_interval = int(config["evolution"]["steps_per_interval"])
    total_steps = intervals * steps_per_interval
    ramp_time = float(config["evolution"]["ramp_time"])
    dt = ramp_time / total_steps
    count = intervals + 1
    sigma = int(task["sigma"])
    rank = frame0.shape[1]
    input_gram = float(np.max(np.abs(frame0.conj().T @ frame0 - np.eye(rank))))
    if input_gram > float(config["acceptance"]["input_gram_tolerance"]):
        raise RuntimeError(f"source frame is not orthonormal: {input_gram:.3e}")
    projector0 = frame0 @ frame0.conj().T
    h0 = np.asarray(np.eye(frame0.shape[0]) - 2.0 * projector0, dtype=np.complex128, order="F")
    x, y, dy_index = _coordinates(nx, ny)
    dy_values = np.arange(-ny // 2, ny // 2, dtype=np.float64)
    source_density_x, source_left, source_right = _density_summary(frame0, x, nx)
    checkpoint = _load_checkpoint(output_root, task, metadata=metadata, count=count, nx=nx)
    if checkpoint is None:
        frame = np.array(frame0, dtype=np.complex128, order="F", copy=True)
        arrays = _initial_arrays(count, nx)
        density_x, left, right, current_left, energy = _observe(
            frame, h0=h0, dy_index=dy_index, dy_values=dy_values, phi=0.0,
            x=x, nx=nx, ny=ny,
        )
        arrays["phi"][0], arrays["time"][0] = 0.0, 0.0
        arrays["N_left"][0], arrays["N_right"][0] = left, right
        arrays["density_x"][0] = density_x
        arrays["current_left"][0], arrays["energy"][0] = current_left, energy
        completed_step = completed_interval = 0
        maximum_pre_qr_gram = input_gram
        maximum_post_qr_gram = input_gram
    else:
        frame = checkpoint["frame"]
        arrays = checkpoint["arrays"]
        completed_step = int(checkpoint["completed_step"])
        completed_interval = int(checkpoint["completed_interval"])
        maximum_pre_qr_gram = float(checkpoint["maximum_pre_qr_gram"])
        maximum_post_qr_gram = float(checkpoint["maximum_post_qr_gram"])
        expected_step = completed_interval * steps_per_interval
        if completed_step != expected_step:
            raise RuntimeError("checkpoint is not on an observation boundary")
    progress_chunk = int(config["execution"]["progress_update_steps"])
    checkpoint_every = int(config["execution"]["checkpoint_every_intervals"])
    since_progress = 0
    for step in range(completed_step, total_steps):
        frame = rk4_step(
            frame, step=step, dt=dt, ramp_time=ramp_time, sigma=sigma,
            h0=h0, dy_index=dy_index, dy_values=dy_values, ny=ny,
        )
        since_progress += 1
        if progress_queue is not None and since_progress >= progress_chunk:
            progress_queue.put(since_progress)
            since_progress = 0
        finished_step = step + 1
        if finished_step % steps_per_interval != 0:
            continue
        interval = finished_step // steps_per_interval
        frame, pre_gram, post_gram = _fix_qr_gauge(frame)
        maximum_pre_qr_gram = max(maximum_pre_qr_gram, pre_gram)
        maximum_post_qr_gram = max(maximum_post_qr_gram, post_gram)
        t = float(interval) * ramp_time / intervals
        phi = sigma * 2.0 * np.pi * interval / intervals
        density_x, left, right, current_left, energy = _observe(
            frame, h0=h0, dy_index=dy_index, dy_values=dy_values, phi=phi,
            x=x, nx=nx, ny=ny,
        )
        arrays["phi"][interval], arrays["time"][interval] = phi, t
        arrays["N_left"][interval], arrays["N_right"][interval] = left, right
        arrays["density_x"][interval] = density_x
        arrays["current_left"][interval], arrays["energy"][interval] = current_left, energy
        if interval % checkpoint_every == 0 and interval < intervals:
            _write_checkpoint(
                output_root, task, frame=frame, completed_step=finished_step,
                completed_interval=interval, arrays=arrays,
                maximum_pre_qr_gram=maximum_pre_qr_gram,
                maximum_post_qr_gram=maximum_post_qr_gram, metadata=metadata,
            )
    if progress_queue is not None and since_progress:
        progress_queue.put(since_progress)
    delta_left = arrays["N_left"] - source_left
    delta_right = arrays["N_right"] - source_right
    large_gauge = np.exp(1j * sigma * 2.0 * np.pi * y / ny)
    gauged_source = large_gauge[:, None] * frame0
    singular = np.linalg.svd(gauged_source.conj().T @ frame, compute_uv=False)
    final_h = _twisted_parent(h0, dy_index, dy_values, sigma * 2.0 * np.pi, ny)
    eigenvalues, eigenvectors = np.linalg.eigh(final_h)
    instantaneous = np.asarray(eigenvectors[:, :rank], dtype=np.complex128)
    instantaneous_weight = float(np.sum(np.abs(instantaneous.conj().T @ frame) ** 2))
    result = {
        **arrays,
        "delta_N_left": delta_left,
        "delta_N_right": delta_right,
        "delta_N_total": delta_left + delta_right,
        "q_x": 0.5 * (delta_right - delta_left),
        "source_density_x": source_density_x,
        "source_N_left": np.asarray(source_left),
        "source_N_right": np.asarray(source_right),
        "rank": np.asarray(rank, dtype=np.int64),
        "ramp_time": np.asarray(ramp_time),
        "dt": np.asarray(dt),
        "steps_per_interval": np.asarray(steps_per_interval, dtype=np.int64),
        "total_steps": np.asarray(total_steps, dtype=np.int64),
        "input_gram_residual": np.asarray(input_gram),
        "maximum_pre_qr_gram_residual": np.asarray(maximum_pre_qr_gram),
        "maximum_post_qr_gram_residual": np.asarray(maximum_post_qr_gram),
        "large_gauge_minimum_principal_overlap": np.asarray(float(np.min(singular))),
        "large_gauge_mean_principal_overlap": np.asarray(float(np.mean(singular))),
        "instantaneous_occupied_weight": np.asarray(instantaneous_weight),
        "instantaneous_excitation_number": np.asarray(float(rank - instantaneous_weight)),
        "instantaneous_rank_gap": np.asarray(float(eigenvalues[rank] - eigenvalues[rank - 1])),
    }
    _validate_result_arrays(result, config)
    return result


def _record_failure(output_root: Path, task: dict[str, Any], exc: BaseException) -> None:
    _atomic_json(
        failure_path(output_root, task),
        {
            "task_id": task["task_id"], "failed_unix": time.time(),
            "error_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, source_row, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    try:
        frame = _load_frame(source_row)
        metadata = _metadata(task, config_hash, hashes, source_row)
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            arrays = compute_path(frame, task, config, output_root, metadata, queue)
        publish_result(
            output_root, task, arrays, config_hash=config_hash, hashes=hashes,
            source_row=source_row, elapsed_seconds=time.perf_counter() - started,
        )
        ok, reason, _ = verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config,
        )
        if not ok:
            raise RuntimeError(f"published result failed readback: {reason}")
        _delete_checkpoint(output_root, task)
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def inventory(
    config: dict[str, Any], output_root: Path, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    context = source_context(config) if context is None else context
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    rows = {}
    for task in tasks(config):
        source_row = context["rows"][task["source_task_id"]]
        rows[task["task_id"]] = verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config,
        )
    return {"paths": rows, "config_hash": config_hash, "source_hashes": hashes, "source": context}


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    done = sum(row[0] for row in status["paths"].values())
    evolution = config["evolution"]
    dt = float(evolution["ramp_time"]) / (
        int(evolution["flux_intervals"]) * int(evolution["steps_per_interval"])
    )
    print(f"[campaign] {config['campaign_id']}")
    print("[source] 50 verified Ny=24 endpoints: 25 soft + 25 hard; IDs 0,4,...,96")
    print("[Hamiltonian] h0=1-2F0F0^dagger; minimum-image y-flux; no spectral reprojection")
    print(f"[evolution] i dF/dt=h(phi(t))F; T={evolution['ramp_time']}; RK4 dt={dt}; CW+CCW")
    print(f"[workload] 100 paths x {int(evolution['flux_intervals']) * int(evolution['steps_per_interval'])} RK4 steps")
    print(f"[output] {Path(output_root).resolve()}")
    print(f"[identity] config_sha256={status['config_hash']}")
    for name, digest in status["source_hashes"].items():
        print(f"[source code] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] verified={done}/100 pending={100-done}")


def write_identity(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    _atomic_json(
        Path(output_root) / "campaign_identity.json",
        {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": config["campaign_id"],
            "config_hash": status["config_hash"],
            "source_hashes": status["source_hashes"],
            "configuration": config,
            "source_endpoints": status["source"]["rows"],
        },
    )


def run(config: dict[str, Any], output_root: Path, *, workers: int, resume: bool) -> None:
    context = source_context(config)
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    verified, pending, initial_steps = [], [], 0
    total_steps = int(config["evolution"]["flux_intervals"]) * int(config["evolution"]["steps_per_interval"])
    for task in tasks(config):
        source_row = context["rows"][task["source_task_id"]]
        ok, _, _ = verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config,
        )
        if resume and ok:
            verified.append(task)
            initial_steps += total_steps
            continue
        metadata = _metadata(task, config_hash, hashes, source_row)
        checkpoint = _load_checkpoint(
            output_root, task, metadata=metadata,
            count=int(config["evolution"]["flux_intervals"]) + 1,
            nx=int(config["geometry"]["Nx"]),
        )
        initial_steps += 0 if checkpoint is None else int(checkpoint["completed_step"])
        pending.append(task)
    print(f"[run] verified={len(verified)}/100 pending={len(pending)} workers={workers}")
    if not pending:
        return
    context_mp = mp.get_context("spawn")
    with context_mp.Manager() as manager:
        queue = manager.Queue()
        with tqdm(total=100, initial=len(verified), desc="RK4 paths", unit="path", position=0) as task_bar, \
                tqdm(total=100 * total_steps, initial=initial_steps, desc="finite-difference steps", unit="step", position=1) as step_bar, \
                ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=context_mp) as pool:
            futures = {
                pool.submit(
                    _worker,
                    (task, config, str(output_root), config_hash, hashes,
                     context["rows"][task["source_task_id"]], queue),
                )
                for task in pending
            }
            failures = []
            while futures:
                done, futures = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        step_bar.update(int(queue.get_nowait()))
                    except Exception:
                        break
                for future in done:
                    row = future.result()
                    task_bar.update(1)
                    if not row["ok"]:
                        failures.append(row)
                        task_bar.write(f"[failure] {row['task_id']}: {row['error']}")
            while True:
                try:
                    step_bar.update(int(queue.get_nowait()))
                except Exception:
                    break
        if failures:
            raise RuntimeError(f"{len(failures)} paths failed; inspect failures/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "run"), nargs="?", default="report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or config["execution"]["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    print(f"[execution] workers={workers} affinity={sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unavailable'}")
    status = inventory(config, output_root)
    print_inventory(config, output_root, status)
    write_identity(config, output_root, status)
    if args.command == "report":
        return 0
    run(config, output_root, workers=workers, resume=args.resume)
    final = inventory(config, output_root)
    print_inventory(config, output_root, final)
    if not all(row[0] for row in final["paths"].values()):
        raise RuntimeError("campaign ended without 100 verified path pairs")
    if args.analyze:
        import analyze_parent_schrodinger_rk4_s50
        analyze_parent_schrodinger_rk4_s50.analyze(config, output_root)
    print("[complete] long-ramp RK4 campaign finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
