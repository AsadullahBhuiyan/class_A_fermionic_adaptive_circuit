#!/usr/bin/env python3
"""Locked S100 state-projector pump variants for size and flux-grid tests."""

from __future__ import annotations

import argparse
import contextlib
import csv
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_state_projector_pump_s100 as base  # noqa: E402
from src.fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


CAMPAIGN_SCHEMA = "state_projector_pump_variant_campaign_v1"
BURNIN_SCHEMA = "state_projector_pump_variant_burnin_v1"
PUMP_SCHEMA = base.PUMP_SCHEMA
COMPLETION_SCHEMA = base.COMPLETION_SCHEMA
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.state_projector_pump_n24x24_s100_v1.json"
SOURCE_PATHS = {
    "cpu_engine": REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py",
    "occupied_frame": REPO_ROOT / "src" / "fgtn" / "occupied_frame.py",
    "base_pump_runner": PROJECT_ROOT / "run_state_projector_pump_s100.py",
    "campaign_runner": Path(__file__).resolve(),
}
CAMPAIGNS = {
    "N24x24_state_projector_pump_s100_v1": {
        "root_seed": 2026090502,
        "geometry": {
            "Nx": 24, "Ny": 24, "DW": True, "dw_interval": [6, 18], "nshell": 1,
            "filling_frac": 0.5, "alpha_1": 1.0, "alpha_2": 30.0,
            "trial_orbitals": "X",
        },
        "grid_intervals": 64,
        "bootstrap_seed": 2026090503,
        "independent_sampling_unit": "wall-specific monitored trajectory",
        "regions": {
            "left_x_start": 0, "left_x_stop_exclusive": 12,
            "right_x_start": 12, "right_x_stop_exclusive": 24,
        },
        "endpoint_source": None,
    },
    "N20x24_state_projector_pump_grid128_s100_v1": {
        "root_seed": 2026090304,
        "geometry": {
            "Nx": 20, "Ny": 24, "DW": True, "dw_interval": [5, 15], "nshell": 1,
            "filling_frac": 0.5, "alpha_1": 1.0, "alpha_2": 30.0,
            "trial_orbitals": "X",
        },
        "grid_intervals": 128,
        "bootstrap_seed": 2026090504,
        "independent_sampling_unit": "reused wall-specific monitored endpoint trajectory",
        "regions": {
            "left_x_start": 0, "left_x_stop_exclusive": 10,
            "right_x_start": 10, "right_x_stop_exclusive": 20,
        },
        "endpoint_source": {
            "campaign_id": "N20x24_state_projector_pump_s100_v1",
            "config": "campaign_config.state_projector_pump_n20x24_s100_v1.json",
            "output": "results/N20x24_state_projector_pump_s100_v1",
            "config_sha256": "aafda9bcd52995b486df745762e249ec5b75af12561ca050601ec6bc3dc6245f",
            "source_hashes": {
                "campaign_runner": "8179ee9089b045bec26726d5720c2d87bef0247086a559802ffcd3f3f5866886",
                "cpu_engine": "e8bc0ea58b14f311aa64b4297d100183227a5ed3bcb819b3dd989221264d254e",
                "occupied_frame": "5e689503de89812546c98d4e474511ded13b8ac2b42d57b41c730ed444e2d4c1",
            },
        },
    },
}


canonical_json = base.canonical_json
sha256_path = base.sha256_path
load_config = base.load_config
burnin_tasks = base.burnin_tasks
pump_tasks = base.pump_tasks
result_paths = base.result_paths
failure_path = base.failure_path
publish_pair = base.publish_pair
flux_grid = base.flux_grid
compute_pump_path = base.compute_pump_path
validate_pump_arrays = base.validate_pump_arrays


def source_hashes() -> dict[str, str]:
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "root_seed", "geometry", "walls", "dynamics",
        "endpoint_source", "projector_pump", "regions", "ensemble", "acceptance",
    )
    return hashlib.sha256(
        canonical_json({key: config[key] for key in keys}).encode("utf-8")
    ).hexdigest()


def _expected_pump(intervals: int) -> dict[str, Any]:
    return {
        "grid_intervals": intervals,
        "regulator": 1e-7,
        "directions": {"ccw": 1, "cw": -1},
        "projector": "P_xi = F_xi F_xi^dagger",
        "flattened_parent": "h_xi = 1 - 2 P_xi",
        "twist": "h_rs(phi) = h_rs(0) exp(i phi d_y(r,s)/Ny)",
        "periodic_displacement": "minimum_image",
        "rank_rule": "preserve the burn-in occupied rank",
        "matching": "maximum overlap with the preceding occupied projector",
        "instantaneous_control": "fill the lowest burn-in-rank eigenvectors independently at each phi",
    }


def validate_config(config: dict[str, Any]) -> None:
    campaign_id = str(config.get("campaign_id"))
    if config.get("schema") != CAMPAIGN_SCHEMA or campaign_id not in CAMPAIGNS:
        raise ValueError("unexpected variant campaign schema or identity")
    expected = CAMPAIGNS[campaign_id]
    if int(config.get("root_seed", -1)) != expected["root_seed"]:
        raise ValueError("root seed differs from the locked campaign")
    if config["geometry"] != expected["geometry"]:
        raise ValueError("geometry or OW contract changed")
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
    if config["endpoint_source"] != expected["endpoint_source"]:
        raise ValueError("endpoint-source identity changed")
    if config["projector_pump"] != _expected_pump(int(expected["grid_intervals"])):
        raise ValueError("state-projector pump definition changed")
    if config["regions"] != expected["regions"]:
        raise ValueError("left/right regions changed")
    ensemble = config["ensemble"]
    if (
        int(ensemble["samples_per_wall"]) != 100
        or ensemble["independent_sampling_unit"] != expected["independent_sampling_unit"]
        or int(ensemble["bootstrap_draws"]) != 10000
        or int(ensemble["bootstrap_seed"]) != expected["bootstrap_seed"]
        or float(ensemble["confidence_level"]) != 0.95
    ):
        raise ValueError("ensemble contract changed")
    if config["acceptance"] != {
        "frame_gram_tolerance": 1e-8,
        "burnin_charge_continuity_tolerance": 1e-9,
        "projector_tolerance": 1e-10,
        "pump_charge_conservation_tolerance": 1e-9,
        "large_gauge_tolerance": 1e-9,
        "initial_regulator_charge_tolerance": 1e-5,
        "instantaneous_endpoint_closure_tolerance": 1e-5,
        "minimum_principal_overlap_floor": 0.001,
        "pump_event_threshold": 0.5,
        "quantization_is_acceptance_gate": False,
    }:
        raise ValueError("acceptance contract changed")


def _endpoint_context(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    source = config["endpoint_source"]
    if source is None:
        return {
            "config": config,
            "root": output_root,
            "config_hash": scientific_config_hash(config),
            "source_hashes": source_hashes(),
            "reused": False,
        }
    source_config_path = PROJECT_ROOT / str(source["config"])
    source_root = PROJECT_ROOT / str(source["output"])
    source_config = base.load_config(source_config_path)
    base.validate_config(source_config)
    actual_hash = base.scientific_config_hash(source_config)
    actual_sources = base.source_hashes()
    if actual_hash != source["config_sha256"] or actual_sources != source["source_hashes"]:
        raise RuntimeError("the pinned endpoint-source campaign identity no longer matches")
    return {
        "config": source_config,
        "root": source_root,
        "config_hash": actual_hash,
        "source_hashes": actual_sources,
        "reused": True,
    }


def _verify_own_pair(
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
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "schema": COMPLETION_SCHEMA,
            **base._metadata(task, config_hash, hashes),
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
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != base._metadata(task, config_hash, hashes):
                return False, "result identity mismatch", None
            if task["stage"] == "burnin":
                frame = np.asarray(saved["frame"])
                rank = int(np.asarray(saved["rank"]).item())
                if frame.dtype != np.complex128 or frame.shape != (2 * nx * ny, rank):
                    return False, "burn-in frame dtype, dimension, or rank mismatch", None
                if np.asarray(saved["density_x"]).shape != (nx,):
                    return False, "burn-in x-density shape mismatch", None
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
                for key in ("continued_density_x", "instantaneous_density_x"):
                    if np.asarray(saved[key]).shape != (count, nx):
                        return False, f"pump field {key} has the wrong shape", None
                if np.asarray(saved["source_density_x"]).shape != (nx,):
                    return False, "pump source density has the wrong shape", None
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


def _verify_endpoint(
    context: dict[str, Any], task: dict[str, Any]
) -> tuple[bool, str, dict[str, Any] | None]:
    if context["reused"]:
        return base.verify_pair(
            context["root"], task, context["config_hash"], context["source_hashes"],
            context["config"],
        )
    return _verify_own_pair(
        context["root"], task, context["config_hash"], context["source_hashes"],
        context["config"],
    )


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


def _burnin_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, queue = payload
    output_root, started = Path(output_text), time.perf_counter()
    log_path = output_root / "logs" / "tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    try:
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = _model(config, task["wall"])
                observer = base.BurninObserver(config, queue)
                result = model.run_markov_circuit(
                    native_cycle_observer=observer.cycle,
                    native_event_observer=observer.event,
                    **base._engine_kwargs(config, task["wall"], task["seed"]),
                )
        native = result["native_final"]
        frame = np.asarray(native["frame"], dtype=np.complex128)
        rank = int(native["rank"])
        if frame.shape != (2 * nx * ny, rank) or frame.dtype != np.complex128:
            raise RuntimeError("canonical engine returned an invalid occupied frame")
        continuity = float(observer.final_total - observer.initial_total - observer.net_injected_charge)
        if abs(continuity) > float(config["acceptance"]["burnin_charge_continuity_tolerance"]):
            raise FloatingPointError(f"burn-in charge continuity failed: {continuity:.3e}")
        left, right, density_x = base._frame_charge(frame, nx, ny)
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
        base._record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def _pump_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, parent, endpoint_root_text, burnin_sha, config, output_text, config_hash, hashes, queue = payload
    output_root, started = Path(output_text), time.perf_counter()
    try:
        endpoint_path, _ = result_paths(Path(endpoint_root_text), parent)
        with np.load(endpoint_path, allow_pickle=False) as saved:
            frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
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
        base._record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def inventory(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    endpoint = _endpoint_context(config, output_root)
    parents = burnin_tasks(config)
    burnins = {task["task_id"]: _verify_endpoint(endpoint, task) for task in parents}
    pumps = {}
    for task in pump_tasks(config):
        row = burnins[task["burnin_task_id"]]
        burnin_sha = str(row[2]["result"]["sha256"]) if row[0] else None
        pumps[task["task_id"]] = _verify_own_pair(
            output_root, task, config_hash, hashes, config, burnin_sha,
        )
    return {
        "config_hash": config_hash,
        "source_hashes": hashes,
        "endpoint": endpoint,
        "burnins": burnins,
        "pumps": pumps,
    }


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    burnin_done = sum(row[0] for row in status["burnins"].values())
    pump_done = sum(row[0] for row in status["pumps"].values())
    geometry = config["geometry"]
    source_label = "verified reused Ny=24 endpoints" if status["endpoint"]["reused"] else "new trajectories"
    print(f"[campaign] {config['campaign_id']}")
    print(
        f"[trajectory] Nx={geometry['Nx']} Ny={geometry['Ny']}, 48=2Ny cycles, "
        f"raster-y, pure half filling, source={source_label}"
    )
    print("[dynamics] soft+hard, nshell=1, alpha=(1,30), perfect correction, complex128")
    print(
        f"[static pump] 1-2FF^dagger; {config['projector_pump']['grid_intervals']} "
        "flux intervals; overlap continuation"
    )
    print(f"[output] {output_root.resolve()}")
    print(f"[identity] config_sha256={status['config_hash']}")
    for name, digest in status["source_hashes"].items():
        print(f"[source] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] endpoint trajectories verified={burnin_done}/200 pending={200-burnin_done}")
    print(f"[resume] pump paths verified={pump_done}/400 pending={400-pump_done}")


def _run_stage(
    stage: str,
    config: dict[str, Any],
    output_root: Path,
    workers: int,
    resume: bool,
    sample_ids: set[int] | None,
) -> None:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    endpoint = _endpoint_context(config, output_root)
    parents = {task["task_id"]: task for task in burnin_tasks(config)}
    if stage == "burnin" and endpoint["reused"]:
        print("[burnin] reusing pinned endpoint campaign; no monitored dynamics launched")
        return
    source_rows = burnin_tasks(config) if stage == "burnin" else pump_tasks(config)
    rows = base._selected(source_rows, sample_ids)
    verified, pending, burnin_shas = [], [], {}
    for task in rows:
        burnin_sha = None
        if stage == "burnin":
            ok, _, _ = _verify_own_pair(output_root, task, config_hash, hashes, config)
        else:
            parent = parents[task["burnin_task_id"]]
            ok_parent, reason, completion = _verify_endpoint(endpoint, parent)
            if not ok_parent:
                raise RuntimeError(f"pump parent {parent['task_id']} is not verified: {reason}")
            burnin_sha = str(completion["result"]["sha256"])
            burnin_shas[task["task_id"]] = burnin_sha
            ok, _, _ = _verify_own_pair(output_root, task, config_hash, hashes, config, burnin_sha)
        (verified if resume and ok else pending).append(task)
    units = 48 if stage == "burnin" else int(config["projector_pump"]["grid_intervals"]) + 1
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
                    futures.add(pool.submit(_burnin_worker, (
                        task, config, str(output_root), config_hash, hashes, queue,
                    )))
                else:
                    parent = parents[task["burnin_task_id"]]
                    futures.add(pool.submit(_pump_worker, (
                        task, parent, str(endpoint["root"]), burnin_shas[task["task_id"]],
                        config, str(output_root), config_hash, hashes, queue,
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


def _analyze(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    if not all(row[0] for row in status["burnins"].values()) or not all(
        row[0] for row in status["pumps"].values()
    ):
        raise RuntimeError("analysis requires 200 verified endpoints and 400 verified pump paths")
    rows = []
    for wall in ("soft", "hard"):
        for sample_id in range(100):
            endpoints = {}
            rank = None
            for direction in ("ccw", "cw"):
                task = next(
                    task for task in pump_tasks(config)
                    if task["wall"] == wall
                    and task["direction"] == direction
                    and int(task["sample_id"]) == sample_id
                )
                result, _ = result_paths(output_root, task)
                with np.load(result, allow_pickle=False) as saved:
                    endpoints[direction] = float(saved["continued_q_x"][-1])
                    rank = int(np.asarray(saved["rank"]).item())
            q_odd = 0.5 * (endpoints["ccw"] - endpoints["cw"])
            rows.append(
                {
                    "wall": wall,
                    "sample_id": sample_id,
                    "ccw_q_x": endpoints["ccw"],
                    "cw_q_x": endpoints["cw"],
                    "direction_odd_q_x": q_odd,
                    "pump_event": int(abs(q_odd) > float(config["acceptance"]["pump_event_threshold"])),
                    "rank": rank,
                }
            )
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_root / "state_projector_pump_endpoints.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "state_projector_pump_variant_analysis_v1",
        "campaign_id": config["campaign_id"],
        "config_hash": status["config_hash"],
        "source_hashes": status["source_hashes"],
        "endpoint_csv": str(csv_path),
        "statistics": [
            {
                "wall": wall,
                "samples": 100,
                "pump_events": sum(row["pump_event"] for row in rows if row["wall"] == wall),
                "mean_direction_odd_q_x": float(
                    np.mean([row["direction_odd_q_x"] for row in rows if row["wall"] == wall])
                ),
            }
            for wall in ("soft", "hard")
        ],
    }
    base._atomic_json(analysis_root / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "burnin", "pump", "all"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sample-ids", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else PROJECT_ROOT / "results" / str(config["campaign_id"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or config["execution"]["workers"])
    sample_ids = None if args.sample_ids is None else set(args.sample_ids)
    if workers < 1 or (sample_ids is not None and not sample_ids.issubset(set(range(100)))):
        raise ValueError("workers must be positive and sample IDs must lie in 0,...,99")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "unavailable"
    print(f"[execution] workers={workers} affinity={affinity}")
    status = inventory(config, output_root)
    print_inventory(config, output_root, status)
    base._atomic_json(
        output_root / "campaign_identity.json",
        {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": config["campaign_id"],
            "config_hash": scientific_config_hash(config),
            "source_hashes": source_hashes(),
            "configuration": config,
        },
    )
    if args.stage == "report":
        return 0
    if args.stage in ("burnin", "all"):
        _run_stage("burnin", config, output_root, workers, args.resume, sample_ids)
    if args.stage in ("pump", "all"):
        _run_stage("pump", config, output_root, workers, args.resume, sample_ids)
    final = inventory(config, output_root)
    print_inventory(config, output_root, final)
    if args.analyze:
        _analyze(config, output_root, final)
    print("[complete] state-projector pump variant command finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
