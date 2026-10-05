#!/usr/bin/env python3
"""State-derived adiabatic projector continuation for verified S10 burn-ins."""

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

import run_online_flux_ramp as online  # noqa: E402


CAMPAIGN_SCHEMA = "state_adiabatic_projector_pump_campaign_v1"
RESULT_SCHEMA = "state_adiabatic_projector_pump_result_v1"
COMPLETION_SCHEMA = "state_adiabatic_projector_pump_completion_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.state_adiabatic_projector_pump_s10_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N16x20_state_adiabatic_projector_pump_s10_v1"
SOURCE_PATHS = {
    "source_campaign_runner": Path(online.__file__).resolve(),
    "state_pump_runner": Path(__file__).resolve(),
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "source_campaign", "geometry", "state_parent",
        "continuation", "ensemble", "acceptance",
    )
    return sha256_bytes(canonical_json({key: config[key] for key in keys}).encode("utf-8"))


def _source_path(config: dict[str, Any], key: str) -> Path:
    return (PROJECT_ROOT / str(config["source_campaign"][key])).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    if config.get("campaign_id") != "N16x20_state_adiabatic_projector_pump_s10_v1":
        raise ValueError("unexpected campaign_id")
    source = config["source_campaign"]
    if source.get("campaign_id") != "N16x20_online_flux_ramp_s10_v1":
        raise ValueError("unexpected source campaign")
    source_config = online.load_config(_source_path(config, "config"))
    online.validate_config(source_config)
    if source_config["campaign_id"] != source["campaign_id"]:
        raise ValueError("source campaign identity mismatch")
    if config["geometry"] != {
        "Nx": 16, "Ny": 20, "periodic_direction": "y", "left_x_stop_exclusive": 8,
    }:
        raise ValueError("unexpected geometry or wall partition")
    if config["state_parent"] != {
        "projector": "P_xi = F_xi F_xi^dagger",
        "flattened_parent": "h_xi = 1 - 2 P_xi",
        "twist": "h_rs(phi) = h_rs(0) exp(i phi d_y(r,s)/Ny)",
        "periodic_displacement": "minimum_image",
        "dtype": "complex128",
    }:
        raise ValueError("state-derived parent contract changed")
    continuation = config["continuation"]
    if int(continuation["grid_intervals"]) != 64:
        raise ValueError("the production continuation uses 64 flux intervals")
    if float(continuation["regulator"]) != 1e-7:
        raise ValueError("the branch regulator must equal 1e-7")
    if continuation["directions"] != {"ccw": 1, "cw": -1}:
        raise ValueError("unexpected direction convention")
    if continuation["rank_rule"] != "preserve the saved occupied rank":
        raise ValueError("occupied-rank rule changed")
    ensemble = config["ensemble"]
    if ensemble["walls"] != ["soft", "hard"] or int(ensemble["samples_per_wall"]) != 10:
        raise ValueError("the production ensemble must contain ten soft and ten hard states")


def tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    intervals = int(config["continuation"]["grid_intervals"])
    rows = [
        {
            "task_id": f"state_pump_{wall}_{direction}_sample_{sample_id:03d}",
            "wall": wall,
            "direction": direction,
            "sigma": int(config["continuation"]["directions"][direction]),
            "sample_id": sample_id,
            "grid_intervals": intervals,
            "grid_points": intervals + 1,
            "burnin_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
        }
        for wall in config["ensemble"]["walls"]
        for sample_id in range(int(config["ensemble"]["samples_per_wall"]))
        for direction in config["continuation"]["directions"]
    ]
    if len(rows) != 40 or len({row["task_id"] for row in rows}) != 40:
        raise RuntimeError("expected exactly 40 unique state-pump tasks")
    return rows


def flux_grid(config: dict[str, Any], sigma: int) -> np.ndarray:
    intervals = int(config["continuation"]["grid_intervals"])
    epsilon = float(config["continuation"]["regulator"])
    return -int(sigma) * epsilon + int(sigma) * np.linspace(0.0, 2.0 * np.pi, intervals + 1)


def result_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(output_root) / "trajectories" / task["wall"] / task["direction"]
    result = root / f"sample_{int(task['sample_id']):03d}.npz"
    return result, result.with_suffix(".completion.json")


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


def source_context(config: dict[str, Any]) -> dict[str, Any]:
    source_config = online.load_config(_source_path(config, "config"))
    online.validate_config(source_config)
    source_root = _source_path(config, "output_root")
    inventory = online.inventory(source_config, source_root)
    verified: dict[str, dict[str, Any]] = {}
    for task_id, row in inventory["burnins"].items():
        if not row[0] or row[2] is None:
            raise RuntimeError(f"source burn-in is not verified: {task_id}: {row[1]}")
        task = inventory["burnin_lookup"][task_id]
        result, _ = online.result_paths(source_root, task)
        verified[task_id] = {
            "path": str(result),
            "sha256": str(row[2]["result"]["sha256"]),
            "bytes": int(row[2]["result"]["bytes"]),
        }
    if len(verified) != 20:
        raise RuntimeError(f"expected 20 verified source burn-ins, found {len(verified)}")
    return {
        "config": source_config,
        "root": source_root,
        "burnins": verified,
        "config_hash": inventory["config_hash"],
        "source_hashes": inventory["source_hashes"],
    }


def _load_frame(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    path = Path(row["path"])
    if path.stat().st_size != int(row["bytes"]) or sha256_path(path) != row["sha256"]:
        raise RuntimeError(f"source burn-in changed after verification: {path}")
    with np.load(path, allow_pickle=False) as saved:
        frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
        rank = int(np.asarray(saved["rank"]).item())
    if frame.ndim != 2 or frame.shape[1] != rank or frame.dtype != np.complex128:
        raise RuntimeError("invalid occupied-frame shape, rank, or dtype")
    return frame, rank


def _coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.repeat(np.arange(ny, dtype=np.int64), 2 * nx)
    x = np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)
    dy = y[:, None] - y[None, :]
    dy = ((dy + ny // 2) % ny) - ny // 2
    return x, y, dy


def _density_summary(frame: np.ndarray, x: np.ndarray, nx: int) -> tuple[np.ndarray, float, float]:
    density = np.real(np.sum(np.abs(frame) ** 2, axis=1))
    density_x = np.asarray([density[x == value].sum() for value in range(nx)])
    return density_x, float(density_x[: nx // 2].sum()), float(density_x[nx // 2 :].sum())


def _select_continued_frame(
    previous: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, rank: int
) -> tuple[np.ndarray, float, float]:
    weights = np.real(np.sum(np.abs(previous.conj().T @ eigenvectors) ** 2, axis=0))
    order = np.lexsort((eigenvalues, -weights))
    selected = np.asarray(eigenvectors[:, order[:rank]], dtype=np.complex128)
    singular = np.linalg.svd(previous.conj().T @ selected, compute_uv=False)
    return selected, float(np.min(singular)), float(np.min(weights[order[:rank]]))


def compute_path(
    frame: np.ndarray,
    task: dict[str, Any],
    config: dict[str, Any],
    progress_queue: Any = None,
) -> dict[str, Any]:
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    dimension, rank = frame.shape
    if dimension != 2 * nx * ny:
        raise RuntimeError("occupied frame does not match the locked geometry")
    gram_residual = float(np.max(np.abs(frame.conj().T @ frame - np.eye(rank))))
    if gram_residual > float(config["acceptance"]["input_gram_tolerance"]):
        raise RuntimeError(f"source occupied frame is not orthonormal: {gram_residual:.3e}")
    projector0 = frame @ frame.conj().T
    projector_residual = float(np.max(np.abs(projector0 @ projector0 - projector0)))
    h0 = np.eye(dimension, dtype=np.complex128) - 2.0 * projector0
    involution_residual = float(np.max(np.abs(h0 @ h0 - np.eye(dimension))))
    x, y, dy = _coordinates(nx, ny)
    density0, left0, right0 = _density_summary(frame, x, nx)
    phis = flux_grid(config, int(task["sigma"]))
    count = len(phis)

    continued_left = np.empty(count)
    continued_right = np.empty(count)
    instantaneous_left = np.empty(count)
    instantaneous_right = np.empty(count)
    spectral_gap = np.empty(count)
    principal_overlap = np.empty(count)
    selected_weight_floor = np.empty(count)
    continued_density_x = np.empty((count, nx))
    instantaneous_density_x = np.empty((count, nx))
    eigenvalues_history = np.empty((count, dimension))

    previous = frame
    h_start = h_end = None
    for point, phi in enumerate(phis):
        twist = np.exp(1j * float(phi) * dy / ny)
        h_phi = np.asarray(h0 * twist, dtype=np.complex128)
        h_phi = 0.5 * (h_phi + h_phi.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(h_phi)
        continued, overlap, weight_floor = _select_continued_frame(
            previous, eigenvalues, eigenvectors, rank
        )
        instantaneous = np.asarray(eigenvectors[:, :rank], dtype=np.complex128)
        c_density, c_left, c_right = _density_summary(continued, x, nx)
        i_density, i_left, i_right = _density_summary(instantaneous, x, nx)
        continued_left[point], continued_right[point] = c_left, c_right
        instantaneous_left[point], instantaneous_right[point] = i_left, i_right
        continued_density_x[point] = c_density
        instantaneous_density_x[point] = i_density
        spectral_gap[point] = float(eigenvalues[rank] - eigenvalues[rank - 1])
        principal_overlap[point] = overlap
        selected_weight_floor[point] = weight_floor
        eigenvalues_history[point] = eigenvalues
        previous = continued
        h_start = h_phi if point == 0 else h_start
        h_end = h_phi
        if progress_queue is not None:
            progress_queue.put(1)

    c_delta_left = continued_left - left0
    c_delta_right = continued_right - right0
    i_delta_left = instantaneous_left - left0
    i_delta_right = instantaneous_right - right0
    sigma = int(task["sigma"])
    large_gauge = np.exp(1j * sigma * 2.0 * np.pi * y / ny)
    gauged_start = large_gauge[:, None] * h_start * large_gauge.conj()[None, :]
    large_gauge_error = float(np.max(np.abs(h_end - gauged_start)))

    arrays = {
        "schema": np.asarray(RESULT_SCHEMA),
        "phi": phis,
        "path_fraction": np.arange(count, dtype=np.float64) / (count - 1),
        "continued_N_left": continued_left,
        "continued_N_right": continued_right,
        "continued_delta_N_left": c_delta_left,
        "continued_delta_N_right": c_delta_right,
        "continued_delta_N_total": c_delta_left + c_delta_right,
        "continued_q_x": 0.5 * (c_delta_right - c_delta_left),
        "instantaneous_N_left": instantaneous_left,
        "instantaneous_N_right": instantaneous_right,
        "instantaneous_delta_N_left": i_delta_left,
        "instantaneous_delta_N_right": i_delta_right,
        "instantaneous_delta_N_total": i_delta_left + i_delta_right,
        "instantaneous_q_x": 0.5 * (i_delta_right - i_delta_left),
        "continued_density_x": continued_density_x,
        "instantaneous_density_x": instantaneous_density_x,
        "source_density_x": density0,
        "eigenvalues": eigenvalues_history,
        "instantaneous_rank_gap": spectral_gap,
        "principal_overlap": principal_overlap,
        "selected_weight_floor": selected_weight_floor,
        "rank": np.asarray(rank, dtype=np.int64),
        "source_N_left": np.asarray(left0),
        "source_N_right": np.asarray(right0),
        "input_frame_gram_residual": np.asarray(gram_residual),
        "input_projector_residual": np.asarray(projector_residual),
        "flattened_parent_involution_residual": np.asarray(involution_residual),
        "large_gauge_parent_error": np.asarray(large_gauge_error),
    }
    _validate_science(arrays, task, config)
    return arrays


def _validate_science(arrays: dict[str, Any], task: dict[str, Any], config: dict[str, Any]) -> None:
    acceptance = config["acceptance"]
    for key, value in arrays.items():
        array = np.asarray(value)
        if key not in {"schema"} and not np.all(np.isfinite(array)):
            raise RuntimeError(f"nonfinite result field: {key}")
    if np.max(np.abs(arrays["continued_delta_N_total"])) > float(acceptance["charge_conservation_tolerance"]):
        raise RuntimeError("continued projector violates charge conservation")
    if np.max(np.abs(arrays["instantaneous_delta_N_total"])) > float(acceptance["charge_conservation_tolerance"]):
        raise RuntimeError("instantaneous projector violates charge conservation")
    if float(arrays["large_gauge_parent_error"]) > float(acceptance["large_gauge_tolerance"]):
        raise RuntimeError("state-derived parent does not close by the expected large gauge transformation")
    if abs(float(arrays["continued_q_x"][0])) > float(acceptance["initial_regulator_charge_tolerance"]):
        raise RuntimeError("regulated continuation does not begin at the source projector")
    if abs(float(arrays["instantaneous_q_x"][-1])) > float(acceptance["instantaneous_endpoint_closure_tolerance"]):
        raise RuntimeError("instantaneous occupied projector does not close after one flux quantum")
    if np.min(arrays["principal_overlap"]) < float(acceptance["minimum_principal_overlap_floor"]):
        raise RuntimeError("flux grid is too coarse for reliable projector continuation")


def publish_pair(
    output_root: Path,
    task: dict[str, Any],
    arrays: dict[str, Any],
    config_hash: str,
    hashes: dict[str, str],
    burnin: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    result, completion = result_paths(output_root, task)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(canonical_json(_metadata(task, config_hash, hashes)))
    payload["burnin_sha256"] = np.asarray(burnin["sha256"])
    _atomic_npz(result, payload)
    record = {"name": result.name, "bytes": result.stat().st_size, "sha256": sha256_path(result)}
    _atomic_json(completion, {
        "schema": COMPLETION_SCHEMA,
        **_metadata(task, config_hash, hashes),
        "burnin": {"sha256": burnin["sha256"], "bytes": burnin["bytes"]},
        "result": record,
        "elapsed_seconds": float(elapsed_seconds),
        "completed_unix": time.time(),
    })


def verify_pair(
    output_root: Path,
    task: dict[str, Any],
    config_hash: str,
    hashes: dict[str, str],
    burnin: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    result, completion_path = result_paths(output_root, task)
    if not result.is_file() or not completion_path.is_file():
        return False, "missing result/completion pair", None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "schema": COMPLETION_SCHEMA,
            **_metadata(task, config_hash, hashes),
            "burnin": {"sha256": burnin["sha256"], "bytes": burnin["bytes"]},
        }
        for key, value in expected.items():
            if completion.get(key) != value:
                return False, f"completion {key} mismatch", None
        record = completion["result"]
        if record.get("name") != result.name or int(record.get("bytes", -1)) != result.stat().st_size:
            return False, "result filename or byte-count mismatch", None
        if record.get("sha256") != sha256_path(result):
            return False, "result checksum mismatch", None
        with np.load(result, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != RESULT_SCHEMA:
                return False, "result schema mismatch", None
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != _metadata(task, config_hash, hashes):
                return False, "result identity mismatch", None
            if str(np.asarray(saved["burnin_sha256"]).item()) != burnin["sha256"]:
                return False, "source burn-in checksum mismatch", None
            count = int(task["grid_points"])
            one_dimensional = (
                "phi", "path_fraction", "continued_N_left", "continued_N_right",
                "continued_delta_N_left", "continued_delta_N_right", "continued_delta_N_total",
                "continued_q_x", "instantaneous_N_left", "instantaneous_N_right",
                "instantaneous_delta_N_left", "instantaneous_delta_N_right",
                "instantaneous_delta_N_total", "instantaneous_q_x", "instantaneous_rank_gap",
                "principal_overlap", "selected_weight_floor",
            )
            for key in one_dimensional:
                if np.asarray(saved[key]).shape != (count,):
                    return False, f"invalid shape for {key}", None
            if not np.array_equal(np.asarray(saved["phi"]), flux_grid(config, task["sigma"])):
                return False, "flux grid mismatch", None
            arrays = {key: np.array(saved[key], copy=True) for key in saved.files if key not in {"metadata_json", "burnin_sha256"}}
            _validate_science(arrays, task, config)
        return True, "verified", completion
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, burnin, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    try:
        frame, rank = _load_frame(burnin)
        if rank != frame.shape[1]:
            raise RuntimeError("saved burn-in rank mismatch")
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            arrays = compute_path(frame, task, config, queue)
        publish_pair(output_root, task, arrays, config_hash, hashes, burnin, time.perf_counter() - started)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        failure = output_root / "failures" / f"{task['task_id']}.json"
        _atomic_json(failure, {
            "task_id": task["task_id"], "error_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(), "failed_unix": time.time(),
        })
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def inventory(config: dict[str, Any], output_root: Path, source: dict[str, Any]) -> dict[str, Any]:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    rows = {}
    for task in tasks(config):
        burnin = source["burnins"][task["burnin_task_id"]]
        rows[task["task_id"]] = verify_pair(output_root, task, config_hash, hashes, burnin, config)
    return {"hashes": hashes, "config_hash": config_hash, "rows": rows}


def print_inventory(config: dict[str, Any], output_root: Path, state: dict[str, Any]) -> None:
    done = sum(row[0] for row in state["rows"].values())
    print(f"[campaign] {config['campaign_id']}")
    print("[input] 20 verified complex128 pure-state burn-in frames; no new circuit dynamics")
    print("[parent] P_xi=F_xi F_xi^dagger; h_xi=1-2P_xi; minimum-image y-flux")
    print("[continuation] 64 intervals, signed 1e-7 regulator, overlap-tracked fixed-rank occupied projector")
    print("[control] independently refilled instantaneous projector at every flux")
    print(f"[output] {output_root.resolve()}")
    print(f"[identity] config_sha256={state['config_hash']}")
    for name, digest in state["hashes"].items():
        print(f"[source] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] verified={done}/40 pending={40-done}")


def run(
    config: dict[str, Any], output_root: Path, workers: int, resume: bool,
    sample_ids: set[int] | None,
) -> None:
    source = source_context(config)
    state = inventory(config, output_root, source)
    print_inventory(config, output_root, state)
    selected = [task for task in tasks(config) if sample_ids is None or task["sample_id"] in sample_ids]
    pending = []
    for task in selected:
        row = state["rows"][task["task_id"]]
        if resume and row[0]:
            continue
        pending.append(task)
    if not pending:
        print(f"[complete] all {len(selected)} selected tasks are already verified")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    manager = context.Manager()
    queue = manager.Queue()
    failures: list[dict[str, Any]] = []
    point_total = sum(task["grid_points"] for task in pending)
    with tqdm(total=len(selected), initial=len(selected) - len(pending), desc="state-pump tasks", unit="task") as task_bar, \
            tqdm(total=sum(task["grid_points"] for task in selected),
                 initial=sum(task["grid_points"] for task in selected if task not in pending),
                 desc="flux diagonalizations", unit="point") as point_bar, \
            ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(_worker, (
                task, config, str(output_root), state["config_hash"], state["hashes"],
                source["burnins"][task["burnin_task_id"]], queue,
            ))
            for task in pending
        }
        while futures:
            while True:
                try:
                    point_bar.update(int(queue.get_nowait()))
                except Exception:
                    break
            finished, futures = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in finished:
                row = future.result()
                task_bar.update(1)
                if not row["ok"]:
                    failures.append(row)
                    task_bar.write(f"[failure] {row['task_id']}: {row['error']}")
        while True:
            try:
                point_bar.update(int(queue.get_nowait()))
            except Exception:
                break
    manager.shutdown()
    final = inventory(config, output_root, source)
    missing = [task_id for task_id, row in final["rows"].items()
               if (sample_ids is None or int(task_id.rsplit("_", 1)[-1]) in sample_ids) and not row[0]]
    if failures or missing:
        raise RuntimeError(f"state-pump run incomplete: failures={len(failures)} missing={len(missing)}")
    print(f"[complete] verified {len(selected)}/{len(selected)} selected state-pump tasks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "run"), nargs="?", default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    workers = int(args.workers or config["execution"]["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    sample_ids = None if args.sample_ids is None else set(args.sample_ids)
    if sample_ids is not None and not sample_ids.issubset(set(range(10))):
        raise ValueError("sample IDs must lie in 0,...,9")
    source = source_context(config)
    if args.command == "report":
        print_inventory(config, args.output_root.resolve(), inventory(config, args.output_root.resolve(), source))
        return 0
    run(config, args.output_root.resolve(), workers, args.resume, sample_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
