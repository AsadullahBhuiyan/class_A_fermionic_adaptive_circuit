#!/usr/bin/env python3
"""N20 S100 state-projector pumps for the Ny size series."""

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

import run_state_projector_pump_s100 as legacy  # noqa: E402


CAMPAIGN_SCHEMA = "state_projector_pump_campaign_v3"
BURNIN_SCHEMA = "state_projector_pump_endpoint_state_v3"
PUMP_SCHEMA = legacy.PUMP_SCHEMA
COMPLETION_SCHEMA = legacy.COMPLETION_SCHEMA
ALLOWED_NY = (28, 30, 32, 34, 36)
ROOT_SEEDS = {28: 2026090401, 30: 2026090402, 32: 2026090403, 34: 2026090404, 36: 2026090405}
BOOTSTRAP_SEEDS = {28: 2026090411, 30: 2026090412, 32: 2026090413, 34: 2026090414, 36: 2026090415}
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.state_projector_pump_n20x28_s100_v3.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N20x28_state_projector_pump_s100_v3"
SOURCE_PATHS = {
    "cpu_engine": REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py",
    "occupied_frame": REPO_ROOT / "src" / "fgtn" / "occupied_frame.py",
    "campaign_runner": Path(__file__).resolve(),
}

canonical_json = legacy.canonical_json
sha256_path = legacy.sha256_path
load_config = legacy.load_config
burnin_tasks = legacy.burnin_tasks
pump_tasks = legacy.pump_tasks
result_paths = legacy.result_paths
failure_path = legacy.failure_path
publish_pair = legacy.publish_pair
flux_grid = legacy.flux_grid
compute_pump_path = legacy.compute_pump_path
validate_pump_arrays = legacy.validate_pump_arrays


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "root_seed", "geometry", "walls", "dynamics",
        "projector_pump", "regions", "ensemble", "acceptance",
    )
    return hashlib.sha256(
        canonical_json({key: config[key] for key in keys}).encode("utf-8")
    ).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    geometry = config["geometry"]
    ny = int(geometry["Ny"])
    expected_id = f"N20x{ny}_state_projector_pump_s100_v3"
    if ny not in ALLOWED_NY or config.get("campaign_id") != expected_id:
        raise ValueError("unexpected size-series campaign identity")
    if int(config.get("root_seed", -1)) != ROOT_SEEDS[ny]:
        raise ValueError("root seed does not match the locked size")
    if geometry != {
        "Nx": 20, "Ny": ny, "DW": True, "dw_interval": [5, 15], "nshell": 1,
        "filling_frac": 0.5, "alpha_1": 1.0, "alpha_2": 30.0,
        "trial_orbitals": "X",
    }:
        raise ValueError("geometry or OW parameters differ from the locked size series")
    if config["walls"] != {
        "soft": {"dw_truncation": False, "meas_slab_only": False},
        "hard": {"dw_truncation": True, "meas_slab_only": True},
    }:
        raise ValueError("soft/hard wall definitions changed")
    if config["dynamics"] != {
        "burn_in_cycles": 2 * ny, "sequence": "raster_y", "perfect_correction": True,
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
        raise ValueError("projector-pump definition changed")
    if config["regions"] != {
        "left_x_start": 0, "left_x_stop_exclusive": 10,
        "right_x_start": 10, "right_x_stop_exclusive": 20,
    }:
        raise ValueError("left/right half-system regions changed")
    ensemble = config["ensemble"]
    if (
        int(ensemble["samples_per_wall"]) != 100
        or ensemble["independent_sampling_unit"] != "wall-specific monitored trajectory"
        or int(ensemble["bootstrap_draws"]) != 10000
        or int(ensemble["bootstrap_seed"]) != BOOTSTRAP_SEEDS[ny]
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
        raise ValueError("numerical acceptance contract changed")


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
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "schema": COMPLETION_SCHEMA,
            **legacy._metadata(task, config_hash, hashes),
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
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != legacy._metadata(task, config_hash, hashes):
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


def _burnin_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, queue = payload
    output_root, started = Path(output_text), time.perf_counter()
    log_path = output_root / "logs" / "tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    try:
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = legacy._model(config, task["wall"])
                observer = legacy.BurninObserver(config, queue)
                result = model.run_markov_circuit(
                    native_cycle_observer=observer.cycle,
                    native_event_observer=observer.event,
                    **legacy._engine_kwargs(config, task["wall"], task["seed"]),
                )
        native = result["native_final"]
        frame = np.asarray(native["frame"], dtype=np.complex128)
        rank = int(native["rank"])
        if frame.shape != (2 * nx * ny, rank) or frame.dtype != np.complex128:
            raise RuntimeError("canonical engine returned an invalid occupied frame")
        continuity = float(observer.final_total - observer.initial_total - observer.net_injected_charge)
        if abs(continuity) > float(config["acceptance"]["burnin_charge_continuity_tolerance"]):
            raise FloatingPointError(f"burn-in charge continuity failed: {continuity:.3e}")
        left, right, density_x = legacy._frame_charge(frame, nx, ny)
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
        legacy._record_failure(output_root, task, exc)
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
        "config_hash": config_hash,
        "source_hashes": hashes,
        "burnins": burnins,
        "pumps": pumps,
        "burnin_lookup": {task["task_id"]: task for task in parents},
    }


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    burnin_done = sum(row[0] for row in status["burnins"].values())
    pump_done = sum(row[0] for row in status["pumps"].values())
    burnin_total, pump_total = len(status["burnins"]), len(status["pumps"])
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    cycles = int(config["dynamics"]["burn_in_cycles"])
    print(f"[campaign] {config['campaign_id']}")
    print(f"[trajectory] Nx={nx} Ny={ny}, {cycles}=2Ny total cycles, raster-y, pure half filling")
    print("[dynamics] soft+hard, nshell=1, alpha=(1,30), perfect correction, complex128")
    print("[static pump] h_xi=1-2F_xi F_xi^dagger; 64 flux intervals; overlap continuation")
    print("[observable] Delta N_L, Delta N_R, q_x=(Delta N_R-Delta N_L)/2")
    print(f"[output] {output_root.resolve()}")
    print(f"[identity] config_sha256={status['config_hash']}")
    for name, digest in status["source_hashes"].items():
        print(f"[source] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] endpoint trajectories verified={burnin_done}/{burnin_total} pending={burnin_total-burnin_done}")
    print(f"[resume] pump paths verified={pump_done}/{pump_total} pending={pump_total-pump_done}")


def _run_stage(
    stage: str,
    config: dict[str, Any],
    output_root: Path,
    workers: int,
    resume: bool,
    sample_ids: set[int] | None,
) -> None:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    parents = {task["task_id"]: task for task in burnin_tasks(config)}
    source_rows = burnin_tasks(config) if stage == "burnin" else pump_tasks(config)
    rows = legacy._selected(source_rows, sample_ids)
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
    display_stage = "trajectory" if stage == "burnin" else stage
    print(f"[{display_stage}] verified={len(verified)}/{len(rows)} pending={len(pending)} workers={workers}")
    if not pending:
        return
    context = mp.get_context("spawn")
    with context.Manager() as manager:
        queue = manager.Queue()
        with tqdm(total=len(rows), initial=len(verified), desc=f"{display_stage} tasks", unit="task", position=0) as task_bar, \
                tqdm(total=len(rows) * units, initial=len(verified) * units,
                     desc=f"{display_stage} {'cycles' if stage == 'burnin' else 'flux points'}",
                     unit="cycle" if stage == "burnin" else "point", position=1) as unit_bar, \
                ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=context) as pool:
            futures = set()
            for task in pending:
                if stage == "burnin":
                    futures.add(pool.submit(_burnin_worker, (task, config, str(output_root), config_hash, hashes, queue)))
                else:
                    futures.add(pool.submit(legacy._pump_worker, (
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
    legacy._atomic_json(
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
    samples = int(config["ensemble"]["samples_per_wall"])
    if workers < 1 or (sample_ids is not None and not sample_ids.issubset(set(range(samples)))):
        raise ValueError(f"workers must be positive and sample IDs must lie in 0,...,{samples - 1}")
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
            raise RuntimeError("analysis requires all verified pump paths")
        import analyze_state_projector_pump_size_series

        analyze_state_projector_pump_size_series.analyze(config, output_root)
    print("[complete] state-projector pump command finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
