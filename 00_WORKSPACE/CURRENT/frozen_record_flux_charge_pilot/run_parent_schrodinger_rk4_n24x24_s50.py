#!/usr/bin/env python3
"""S25-per-wall N24x24 parent-Hamiltonian RK4 width comparison."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_parent_schrodinger_rk4_s50 as kernel  # noqa: E402
import run_parent_schrodinger_rk4_refinement as refinement  # noqa: E402
import run_state_projector_pump_variants as endpoint_campaign  # noqa: E402


CAMPAIGN_SCHEMA = "parent_schrodinger_rk4_width_campaign_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.parent_schrodinger_rk4_n24x24_s50_tau1e4_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N24x24_parent_schrodinger_rk4_s50_tau1e4_v1"
SOURCE_PATHS = {
    "endpoint_campaign_runner": Path(endpoint_campaign.__file__).resolve(),
    "rk4_numerical_kernel": Path(kernel.__file__).resolve(),
    "campaign_runner": Path(__file__).resolve(),
    "preregistered_baseline": DEFAULT_OUTPUT / "preregistered_comparison_baseline.json",
}

# Expose the unchanged persistence/kernel interface to the read-only monitor.
RESULT_SCHEMA = kernel.RESULT_SCHEMA
COMPLETION_SCHEMA = kernel.COMPLETION_SCHEMA
CHECKPOINT_SCHEMA = kernel.CHECKPOINT_SCHEMA
result_paths = kernel.result_paths
checkpoint_paths = kernel.checkpoint_paths
failure_path = kernel.failure_path
verify_result = kernel.verify_result
_metadata = kernel._metadata


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes() -> dict[str, str]:
    return {name: kernel.sha256_path(path) for name, path in SOURCE_PATHS.items()}


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "source_campaign", "geometry", "evolution",
        "ensemble", "dependency_gate", "analysis", "acceptance",
    )
    raw = kernel.canonical_json({key: config[key] for key in keys}).encode("utf-8")
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    if config.get("campaign_id") != "N24x24_parent_schrodinger_rk4_s50_tau1e4_v1":
        raise ValueError("unexpected campaign identity")
    source = config["source_campaign"]
    expected_ids = list(range(0, 100, 4))
    if source.get("campaign_id") != "N24x24_state_projector_pump_s100_v1":
        raise ValueError("unexpected source campaign")
    if source.get("sample_ids") != expected_ids:
        raise ValueError("the S50 selection must be sample IDs 0,4,...,96")
    if config["geometry"] != {
        "Nx": 24, "Ny": 24, "wall_x": [6, 18], "wall_separation": 12,
        "left_x_stop_exclusive": 12, "periodic_direction": "y",
    }:
        raise ValueError("geometry, walls, or subsystem partition changed")
    evolution = config["evolution"]
    required_evolution = {
        "projector": "P0 = F0 F0^dagger",
        "parent": "h0 = 1 - 2 P0",
        "twist": "h_rs(phi) = h0_rs exp(i phi d_y(r,s)/Ny), Hermitian symmetrized",
        "periodic_displacement": "minimum_image",
        "equation": "i dF/dt = h(phi(t)) F",
        "integrator": "classical explicit RK4 with time-dependent stage Hamiltonians",
        "ramp_time": 10000.0,
        "flux_intervals": 128,
        "steps_per_interval": 320,
        "directions": {"ccw": 1, "cw": -1},
        "schedule": "phi(t) = sigma 2 pi t / ramp_time",
        "frame_gauge": "thin QR at observation boundaries only",
        "dtype": "complex128",
    }
    if evolution != required_evolution:
        raise ValueError("long-ramp RK4 contract changed")
    if config["ensemble"] != {
        "walls": ["soft", "hard"], "samples_per_wall": 25,
        "endpoint_states_total": 50, "paths_total": 100,
        "independent_sampling_unit": "saved monitored endpoint trajectory",
    }:
        raise ValueError("expected 25 endpoints per wall and two directions")
    if int(config["execution"]["checkpoint_every_intervals"]) < 1:
        raise ValueError("checkpoint interval must be positive")
    if int(config["execution"]["progress_update_steps"]) < 1:
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
        raise RuntimeError("expected exactly 100 unique paths")
    return rows


def source_context(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_campaign"]
    source_config_path = (PROJECT_ROOT / source["config"]).resolve()
    source_root = (PROJECT_ROOT / source["output_root"]).resolve()
    source_config = endpoint_campaign.load_config(source_config_path)
    endpoint_campaign.validate_config(source_config)
    if source_config["campaign_id"] != source["campaign_id"]:
        raise RuntimeError("source campaign ID mismatch")
    config_hash = endpoint_campaign.scientific_config_hash(source_config)
    hashes = endpoint_campaign.source_hashes()
    task_map = {row["task_id"]: row for row in endpoint_campaign.burnin_tasks(source_config)}
    selected: dict[str, dict[str, Any]] = {}
    for wall in config["ensemble"]["walls"]:
        for sample_id in source["sample_ids"]:
            task_id = f"burnin_{wall}_sample_{int(sample_id):03d}"
            task = task_map[task_id]
            ok, reason, completion = endpoint_campaign._verify_own_pair(
                source_root, task, config_hash, hashes, source_config
            )
            if not ok or completion is None:
                raise RuntimeError(f"source endpoint is not verified: {task_id}: {reason}")
            path, _ = endpoint_campaign.result_paths(source_root, task)
            selected[task_id] = {
                "path": str(path), "name": path.name,
                "bytes": int(completion["result"]["bytes"]),
                "sha256": str(completion["result"]["sha256"]),
                "source_config_hash": config_hash,
            }
    if len(selected) != 50:
        raise RuntimeError(f"expected 50 verified source endpoints, found {len(selected)}")
    return {"rows": selected, "root": source_root}


def _load_frame(source_row: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    path = Path(source_row["path"])
    if path.stat().st_size != int(source_row["bytes"]) or kernel.sha256_path(path) != source_row["sha256"]:
        raise RuntimeError(f"source endpoint changed after verification: {path}")
    with np.load(path, allow_pickle=False) as saved:
        frame = np.array(saved["frame"], dtype=np.complex128, order="F", copy=True)
        rank = int(np.asarray(saved["rank"]).item())
    expected_rows = 2 * int(config["geometry"]["Nx"]) * int(config["geometry"]["Ny"])
    if frame.shape != (expected_rows, rank) or frame.dtype != np.complex128:
        raise RuntimeError("source occupied frame has invalid shape, rank, or dtype")
    gram = float(np.max(np.abs(frame.conj().T @ frame - np.eye(rank))))
    if gram > float(config["acceptance"]["input_gram_tolerance"]):
        raise RuntimeError(f"source occupied frame Gram residual is {gram:.3e}")
    return frame


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, source_row, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    try:
        frame = _load_frame(source_row, config)
        metadata = kernel._metadata(task, config_hash, hashes, source_row)
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            arrays = kernel.compute_path(frame, task, config, output_root, metadata, queue)
        kernel.publish_result(
            output_root, task, arrays, config_hash=config_hash, hashes=hashes,
            source_row=source_row, elapsed_seconds=time.perf_counter() - started,
        )
        ok, reason, _ = kernel.verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config,
        )
        if not ok:
            raise RuntimeError(f"published result failed readback: {reason}")
        kernel._delete_checkpoint(output_root, task)
        kernel.failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        kernel._record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def inventory(config: dict[str, Any], output_root: Path, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = source_context(config) if context is None else context
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    rows = {}
    for task in tasks(config):
        source_row = context["rows"][task["source_task_id"]]
        rows[task["task_id"]] = kernel.verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config,
        )
    return {"paths": rows, "config_hash": config_hash, "source_hashes": hashes, "source": context}


def verify_dependencies(config: dict[str, Any]) -> dict[str, Any]:
    gate = config["dependency_gate"]
    primary_config = kernel.load_config((PROJECT_ROOT / gate["primary_config"]).resolve())
    kernel.validate_config(primary_config)
    primary_root = (PROJECT_ROOT / gate["primary_output_root"]).resolve()
    primary_status = kernel.inventory(primary_config, primary_root)
    complete = sum(row[0] for row in primary_status["paths"].values())
    if complete != 100:
        raise RuntimeError(f"primary N20x24 campaign is incomplete: {complete}/100")
    refinement_root = (PROJECT_ROOT / gate["refinement_output_root"]).resolve()
    summary_path = refinement_root / "step_halving_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("step-halving summary is absent")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    endpoint_error = float(summary["maximum_endpoint_absolute_difference"])
    path_error = float(summary["maximum_path_absolute_difference"])
    if not np.isfinite(endpoint_error) or not np.isfinite(path_error):
        raise RuntimeError("step-halving summary is nonfinite")
    if endpoint_error > float(gate["maximum_endpoint_absolute_difference"]):
        raise RuntimeError(f"step-halving endpoint error {endpoint_error:.3e} exceeds gate")
    if path_error > float(gate["maximum_path_absolute_difference"]):
        raise RuntimeError(f"step-halving path error {path_error:.3e} exceeds gate")
    return {"primary_complete": complete, "refinement": summary}


def write_identity(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    kernel._atomic_json(
        output_root / "campaign_identity.json",
        {
            "schema": CAMPAIGN_SCHEMA, "campaign_id": config["campaign_id"],
            "config_hash": status["config_hash"], "source_hashes": status["source_hashes"],
            "configuration": config, "source_endpoints": status["source"]["rows"],
        },
    )


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    done = sum(row[0] for row in status["paths"].values())
    total_steps = int(config["evolution"]["flux_intervals"]) * int(config["evolution"]["steps_per_interval"])
    dt = float(config["evolution"]["ramp_time"]) / total_steps
    print(f"[campaign] {config['campaign_id']}")
    print("[source] 50 verified N24x24 endpoints: 25 soft + 25 hard; IDs 0,4,...,96")
    print("[geometry] walls x=6,18; equal separation 12; left/right split x=12")
    print(f"[evolution] T=10000; RK4 dt={dt}; CW+CCW; complex128")
    print(f"[workload] 100 paths x {total_steps} RK4 steps")
    print(f"[resume] verified={done}/100 pending={100-done}")
    print(f"[output] {output_root}")
    print(f"[identity] config_sha256={status['config_hash']}")


def run(config: dict[str, Any], output_root: Path, *, workers: int, resume: bool) -> None:
    context = source_context(config)
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    verified, pending, initial_steps = [], [], 0
    total_steps = int(config["evolution"]["flux_intervals"]) * int(config["evolution"]["steps_per_interval"])
    for task in tasks(config):
        source_row = context["rows"][task["source_task_id"]]
        ok, _, _ = kernel.verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config,
        )
        if resume and ok:
            verified.append(task)
            initial_steps += total_steps
            continue
        metadata = kernel._metadata(task, config_hash, hashes, source_row)
        checkpoint = kernel._load_checkpoint(
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
    failures = []
    with context_mp.Manager() as manager:
        queue = manager.Queue()
        with tqdm(total=100, initial=len(verified), desc="N24x24 RK4 paths", unit="path", position=0) as task_bar, \
                tqdm(total=100 * total_steps, initial=initial_steps, desc="N24x24 finite-difference steps", unit="step", position=1) as step_bar, \
                ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=context_mp) as pool:
            futures = {
                pool.submit(
                    _worker,
                    (task, config, str(output_root), config_hash, hashes,
                     context["rows"][task["source_task_id"]], queue),
                )
                for task in pending
            }
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
    status = inventory(config, output_root)
    print_inventory(config, output_root, status)
    write_identity(config, output_root, status)
    if args.command == "report":
        return 0
    dependency_status = verify_dependencies(config)
    print(f"[dependency gate] primary={dependency_status['primary_complete']}/100; step halving passed")
    run(config, output_root, workers=workers, resume=args.resume)
    final = inventory(config, output_root)
    print_inventory(config, output_root, final)
    if not all(row[0] for row in final["paths"].values()):
        raise RuntimeError("campaign ended without 100 verified path pairs")
    if args.analyze:
        import analyze_parent_schrodinger_rk4_width_comparison
        analyze_parent_schrodinger_rk4_width_comparison.analyze(config, output_root)
    print("[complete] N24x24 width-comparison RK4 campaign finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
