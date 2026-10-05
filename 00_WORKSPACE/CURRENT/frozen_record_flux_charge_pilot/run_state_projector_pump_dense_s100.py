#!/usr/bin/env python3
"""N20x24 S100 state-projector pump with dense (n_shell=infinity) OW modes."""

from __future__ import annotations

import argparse
import contextlib
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


CAMPAIGN_SCHEMA = "state_projector_pump_dense_campaign_v1"
BURNIN_SCHEMA = base.BURNIN_SCHEMA
PUMP_SCHEMA = base.PUMP_SCHEMA
COMPLETION_SCHEMA = base.COMPLETION_SCHEMA
CAMPAIGN_ID = "N20x24_state_projector_pump_s100_dense_v1"
ROOT_SEED = 2026090406
BOOTSTRAP_SEED = 2026090416
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.state_projector_pump_n20x24_s100_dense_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / CAMPAIGN_ID
SOURCE_PATHS = {
    "cpu_engine": REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py",
    "occupied_frame": REPO_ROOT / "src" / "fgtn" / "occupied_frame.py",
    "base_pump_runner": PROJECT_ROOT / "run_state_projector_pump_s100.py",
    "campaign_runner": Path(__file__).resolve(),
}

canonical_json = base.canonical_json
sha256_path = base.sha256_path
load_config = base.load_config
burnin_tasks = base.burnin_tasks
pump_tasks = base.pump_tasks
result_paths = base.result_paths
failure_path = base.failure_path
publish_pair = base.publish_pair
verify_pair = base.verify_pair
flux_grid = base.flux_grid
compute_pump_path = base.compute_pump_path
validate_pump_arrays = base.validate_pump_arrays


def source_hashes() -> dict[str, str]:
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "root_seed", "geometry", "walls", "dynamics",
        "projector_pump", "regions", "ensemble", "acceptance",
    )
    return hashlib.sha256(
        canonical_json({key: config[key] for key in keys}).encode("utf-8")
    ).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA or config.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("unexpected dense campaign schema or identity")
    if int(config.get("root_seed", -1)) != ROOT_SEED:
        raise ValueError("root seed differs from the locked dense campaign")
    if config["geometry"] != {
        "Nx": 20, "Ny": 24, "DW": True, "dw_interval": [5, 15], "nshell": None,
        "filling_frac": 0.5, "alpha_1": 1.0, "alpha_2": 30.0,
        "trial_orbitals": "X",
    }:
        raise ValueError("geometry or dense OW contract changed")
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
    if config["projector_pump"] != {
        "grid_intervals": 64,
        "regulator": 1e-7,
        "directions": {"ccw": 1, "cw": -1},
        "projector": "P_xi = F_xi F_xi^dagger",
        "flattened_parent": "h_xi = 1 - 2 P_xi",
        "twist": "h_rs(phi) = h_rs(0) exp(i phi d_y(r,s)/Ny)",
        "periodic_displacement": "minimum_image",
        "rank_rule": "preserve the burn-in occupied rank",
        "matching": "maximum overlap with the preceding occupied projector",
        "instantaneous_control": "fill the lowest burn-in-rank eigenvectors independently at each phi",
    }:
        raise ValueError("state-projector pump definition changed")
    if config["regions"] != {
        "left_x_start": 0, "left_x_stop_exclusive": 10,
        "right_x_start": 10, "right_x_stop_exclusive": 20,
    }:
        raise ValueError("left/right regions changed")
    ensemble = config["ensemble"]
    if (
        int(ensemble["samples_per_wall"]) != 100
        or ensemble["independent_sampling_unit"] != "wall-specific monitored trajectory"
        or int(ensemble["bootstrap_draws"]) != 10000
        or int(ensemble["bootstrap_seed"]) != BOOTSTRAP_SEED
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


def _model(config: dict[str, Any], wall: str) -> classA_U1FGTN:
    geometry, wall_config = config["geometry"], config["walls"][wall]
    if geometry["nshell"] is not None:
        raise ValueError("dense campaign requires nshell=None")
    return classA_U1FGTN(
        Nx=int(geometry["Nx"]), Ny=int(geometry["Ny"]), DW=True,
        nshell=None, filling_frac=float(geometry["filling_frac"]),
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
        if frame.shape != (960, rank) or frame.dtype != np.complex128:
            raise RuntimeError("canonical engine returned an invalid occupied frame")
        continuity = float(observer.final_total - observer.initial_total - observer.net_injected_charge)
        if abs(continuity) > float(config["acceptance"]["burnin_charge_continuity_tolerance"]):
            raise FloatingPointError(f"burn-in charge continuity failed: {continuity:.3e}")
        left, right, density_x = base._frame_charge(frame, 20, 24)
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


def inventory(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    parents = burnin_tasks(config)
    burnins = {
        task["task_id"]: verify_pair(output_root, task, config_hash, hashes, config)
        for task in parents
    }
    pumps: dict[str, tuple[bool, str, dict[str, Any] | None]] = {}
    for task in pump_tasks(config):
        row = burnins[task["burnin_task_id"]]
        burnin_sha = str(row[2]["result"]["sha256"]) if row[0] else None
        pumps[task["task_id"]] = verify_pair(
            output_root, task, config_hash, hashes, config, burnin_sha
        )
    return {"config_hash": config_hash, "source_hashes": hashes, "burnins": burnins, "pumps": pumps}


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    burnin_done = sum(row[0] for row in status["burnins"].values())
    pump_done = sum(row[0] for row in status["pumps"].values())
    print(f"[campaign] {config['campaign_id']}")
    print("[trajectory] Nx=20 Ny=24, 48=2Ny cycles, raster-y, pure half filling")
    print("[dynamics] soft+hard, nshell=infinity (None/dense), alpha=(1,30), perfect correction, complex128")
    print("[static pump] h_xi=1-2F_xi F_xi^dagger; 64 flux intervals; overlap continuation")
    print("[observable] Delta N_L, Delta N_R, q_x=(Delta N_R-Delta N_L)/2")
    print(f"[output] {output_root.resolve()}")
    print(f"[identity] config_sha256={status['config_hash']}")
    for name, digest in status["source_hashes"].items():
        print(f"[source] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] endpoint trajectories verified={burnin_done}/200 pending={200-burnin_done}")
    print(f"[resume] pump paths verified={pump_done}/400 pending={400-pump_done}")


def _run_stage(
    stage: str, config: dict[str, Any], output_root: Path, workers: int,
    resume: bool, sample_ids: set[int] | None,
) -> None:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    parents = {task["task_id"]: task for task in burnin_tasks(config)}
    source_rows = burnin_tasks(config) if stage == "burnin" else pump_tasks(config)
    rows = base._selected(source_rows, sample_ids)
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
    units = 48 if stage == "burnin" else 65
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
                payload = (task, config, str(output_root), config_hash, hashes, queue)
                if stage == "burnin":
                    futures.add(pool.submit(_burnin_worker, payload))
                else:
                    parent = parents[task["burnin_task_id"]]
                    futures.add(pool.submit(base._pump_worker, (
                        task, parent, config, str(output_root), config_hash, hashes, queue,
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
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "unavailable"
    print(f"[execution] workers={workers} affinity={affinity}")
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
        import analyze_state_projector_pump_s100 as analysis

        analysis.campaign = sys.modules[__name__]
        analysis.analyze(config, output_root)
    print("[complete] dense state-projector pump command finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
