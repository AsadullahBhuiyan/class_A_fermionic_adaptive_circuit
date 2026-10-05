#!/usr/bin/env python3
"""Parallel, deterministic, completion-resumable S100 modular-handedness campaign."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import contextlib
import copy
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modular_packet_analysis import (  # noqa: E402
    ANALYSIS_SCHEMA,
    aggregate_and_plot,
    analyze_frame,
    validate_analysis_arrays,
)
from src.fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402


FRAME_SCHEMA = "modular_handedness_cycle80_frame_v1"
COMPLETION_SCHEMA = "modular_handedness_completion_v1"
CONFIG_PATH = BUNDLE_ROOT / "campaign_config.v1.json"
DEFAULT_OUTPUT = BUNDLE_ROOT / "outputs" / "modular_handedness_nx20_ny40_hard_soft_s100_v1"
SCIENTIFIC_SOURCE_PATHS = {
    "cpu_engine": REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py",
    "occupied_frame": REPO_ROOT / "src" / "fgtn" / "occupied_frame.py",
    "packet_analysis": BUNDLE_ROOT / "modular_packet_analysis.py",
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(stage: str | None = None) -> dict[str, str]:
    names = ("cpu_engine", "occupied_frame") if stage == "simulation" else tuple(SCIENTIFIC_SOURCE_PATHS)
    return {name: sha256_path(SCIENTIFIC_SOURCE_PATHS[name]) for name in names}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scientific_config_hash(config: dict[str, Any], stage: str | None = None) -> str:
    keys = ["schema", "sampling_revision", "root_seed", "geometry", "constructions", "dynamics"]
    if stage != "simulation":
        keys.append("modular_analysis")
    locked = {key: config[key] for key in keys}
    return hashlib.sha256(canonical_json(locked).encode()).hexdigest()


def validate_production_config(config: dict[str, Any]) -> None:
    g, d, a = config["geometry"], config["dynamics"], config["modular_analysis"]
    assert config["root_seed"] == 2026090302
    assert (g["Nx"], g["Ny"], g["nshell"], g["cycles"], g["samples_per_construction"]) == (20, 40, 1, 80, 100)
    assert g["DW"] is True and g["alpha_1"] == 1.0 and g["alpha_2"] == 30.0 and g["filling_frac"] == 0.5
    assert config["constructions"]["hard"]["dw_truncation"] is True
    assert config["constructions"]["hard"]["meas_slab_only"] is True
    assert config["constructions"]["soft"]["dw_truncation"] is False
    assert config["constructions"]["soft"]["meas_slab_only"] is False
    assert d["sequence"] == "raster_y" and d["perfect_correction"] is True
    assert d["postselect"] is False and d["init_mode"] == "default"
    assert d["state_representation"] == "physical_frame" and d["dtype"] == "complex128"
    assert d["samples_per_task"] == 1 and d["parallelize_samples"] is False
    assert a["subsystem_width"] == 20 and a["translated_cuts"] == 40
    assert a["wall_x"] == [5, 15] and a["packet_y_rel"] == [0, 19]


def task_seed(root_seed: int, construction: str, sample_index: int) -> int:
    wall_code = {"hard": 0, "soft": 1}[str(construction)]
    sequence = np.random.SeedSequence(int(root_seed), spawn_key=(wall_code, int(sample_index)))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def expand_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    count = int(config["geometry"]["samples_per_construction"])
    tasks = [
        {
            "task_id": f"{construction}_sample_{sample:03d}",
            "construction": construction,
            "sample_index": sample,
            "seed": task_seed(config["root_seed"], construction, sample),
        }
        for construction in ("hard", "soft")
        for sample in range(count)
    ]
    if len(tasks) != 2 * count or len({task["task_id"] for task in tasks}) != len(tasks) or len({task["seed"] for task in tasks}) != len(tasks):
        raise RuntimeError("Task expansion did not produce unique task IDs and seeds.")
    return tasks


def result_paths(output_root: Path, task: dict[str, Any], stage: str) -> tuple[Path, Path]:
    subdir = "frames" if stage == "simulation" else "packet_products"
    result = Path(output_root) / subdir / task["construction"] / f"sample_{task['sample_index']:03d}.npz"
    return result, result.with_suffix(".completion.json")


def marker_path(output_root: Path, task: dict[str, Any], stage: str, kind: str) -> Path:
    return Path(output_root) / "status" / stage / kind / f"{task['task_id']}.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def _publish_result(
    path: Path,
    arrays: dict[str, Any],
    *,
    task: dict[str, Any],
    stage: str,
    config_hash: str,
    hashes: dict[str, str],
) -> None:
    _atomic_npz(path, arrays)
    checksum, byte_count = sha256_path(path), path.stat().st_size
    completion = {
        "schema": COMPLETION_SCHEMA,
        "stage": stage,
        "task_id": task["task_id"],
        "construction": task["construction"],
        "sample_index": int(task["sample_index"]),
        "seed": int(task["seed"]),
        "config_hash": config_hash,
        "source_hashes": hashes,
        "result": path.name,
        "bytes": int(byte_count),
        "sha256": checksum,
        "completed_unix": time.time(),
    }
    _atomic_json(path.with_suffix(".completion.json"), completion)


def verify_pair(
    output_root: Path,
    task: dict[str, Any],
    stage: str,
    config_hash: str,
    hashes: dict[str, str],
) -> tuple[bool, str]:
    result, completion_path = result_paths(output_root, task, stage)
    if not result.is_file() or not completion_path.is_file():
        return False, "missing pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "schema": COMPLETION_SCHEMA, "stage": stage, "task_id": task["task_id"],
            "construction": task["construction"], "sample_index": int(task["sample_index"]),
            "seed": int(task["seed"]), "config_hash": config_hash,
            "source_hashes": hashes, "result": result.name,
        }
        for key, value in expected.items():
            if completion.get(key) != value:
                return False, f"completion {key} mismatch"
        if int(completion["bytes"]) != result.stat().st_size:
            return False, "byte-count mismatch"
        if str(completion["sha256"]) != sha256_path(result):
            return False, "checksum mismatch"
        with np.load(result, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata["task_id"] != task["task_id"] or metadata["config_hash"] != config_hash or metadata["source_hashes"] != hashes:
                return False, "NPZ identity mismatch"
            if stage == "simulation":
                frame = np.asarray(data["frame"])
                if str(np.asarray(data["schema"]).item()) != FRAME_SCHEMA or frame.dtype != np.complex128:
                    return False, "frame schema/dtype mismatch"
                if frame.shape[0] != 2 * int(metadata["Nx"]) * int(metadata["Ny"]):
                    return False, "frame shape mismatch"
            else:
                validate_analysis_arrays({key: np.asarray(data[key]) for key in data.files}, metadata["config"])
        return True, "verified"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _write_active(output_root: Path, task: dict[str, Any], stage: str) -> Path:
    path = marker_path(output_root, task, stage, "active")
    _atomic_json(path, {"task_id": task["task_id"], "stage": stage, "pid": os.getpid(), "host": os.uname().nodename, "started_unix": time.time()})
    return path


def _record_failure(output_root: Path, task: dict[str, Any], stage: str, exc: BaseException) -> None:
    _atomic_json(marker_path(output_root, task, stage, "failed"), {
        "task_id": task["task_id"], "stage": stage, "pid": os.getpid(), "failed_unix": time.time(),
        "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
    })


def _run_simulation_worker(payload: tuple[dict[str, Any], dict[str, Any], str, str, dict[str, str], Any]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, progress_queue = payload
    output_root = Path(output_text)
    active = _write_active(output_root, task, "simulation")
    failure = marker_path(output_root, task, "simulation", "failed")
    log_path = output_root / "logs" / "simulation" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        geometry = config["geometry"]
        construction = config["constructions"][task["construction"]]
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            model = classA_U1FGTN(
                Nx=int(geometry["Nx"]), Ny=int(geometry["Ny"]), DW=True, nshell=int(geometry["nshell"]),
                filling_frac=float(geometry["filling_frac"]), alpha_1=float(geometry["alpha_1"]),
                alpha_2=float(geometry["alpha_2"]), trial_orbitals=str(geometry["trial_orbitals"]),
                dw_truncation=bool(construction["dw_truncation"]),
            )
            def observe(*, cycle, state, **_kwargs):
                if int(cycle) > 0:
                    progress_queue.put(("simulation_cycle", task["task_id"], 1))
            result = model.run_markov_circuit(
                G_history=False, progress=False, cycles=int(geometry["cycles"]), postselect=False,
                perfect_correction=True, samples=1, parallelize_samples=False, init_mode="default",
                save=False, n_a=0.5, sequence="raster_y", meas_slab_only=bool(construction["meas_slab_only"]),
                random_seed=int(task["seed"]), state_representation="physical_frame",
                native_cycle_observer=observe, return_native_state=True,
                require_no_covariance_materialization=True,
            )
        native = result["native_final"]
        frame = np.asarray(native["frame"], dtype=np.complex128)
        if result["state_representation_resolved"] != "physical_frame" or result["covariance_materialization_count"] != 0:
            raise RuntimeError("Canonical engine did not preserve the locked frame-only execution path.")
        metadata = {
            "task_id": task["task_id"], "construction": task["construction"], "sample_index": task["sample_index"],
            "seed": task["seed"], "config_hash": config_hash, "source_hashes": hashes,
            "canonical_dynamics_entry_point": config["dynamics"]["canonical_entry_point"],
            "Nx": int(geometry["Nx"]), "Ny": int(geometry["Ny"]), "nshell": int(geometry["nshell"]),
            "cycles": int(geometry["cycles"]), "DW": True, "alpha_1": float(geometry["alpha_1"]),
            "alpha_2": float(geometry["alpha_2"]), "dw_truncation": bool(construction["dw_truncation"]),
            "meas_slab_only": bool(construction["meas_slab_only"]), "sequence": "raster_y",
            "perfect_correction": True, "init_mode": "default", "dtype": "complex128",
        }
        arrays = {
            "schema": np.asarray(FRAME_SCHEMA), "frame": frame, "rank": np.asarray(native["rank"], dtype=np.int64),
            "min_rank": np.asarray(native["min_rank"], dtype=np.int64), "max_rank": np.asarray(native["max_rank"], dtype=np.int64),
            "log_weight": np.asarray(native["log_weight"], dtype=np.float64),
            "gram_residual": np.asarray(native["gram_residual"], dtype=np.float64),
            "frame_algorithm_version": np.asarray(native["frame_algorithm_version"]),
            "metadata_json": np.asarray(canonical_json(metadata)),
        }
        result_path, _ = result_paths(output_root, task, "simulation")
        _publish_result(result_path, arrays, task=task, stage="simulation", config_hash=config_hash, hashes=hashes)
        failure.unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        _record_failure(output_root, task, "simulation", exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}
    finally:
        active.unlink(missing_ok=True)


def _run_analysis_worker(payload: tuple[dict[str, Any], dict[str, Any], str, str, dict[str, str], Any]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, progress_queue = payload
    output_root = Path(output_text)
    active = _write_active(output_root, task, "analysis")
    failure = marker_path(output_root, task, "analysis", "failed")
    log_path = output_root / "logs" / "analysis" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        simulation_hash = scientific_config_hash(config, "simulation")
        simulation_sources = source_hashes("simulation")
        valid, reason = verify_pair(output_root, task, "simulation", simulation_hash, simulation_sources)
        if not valid:
            raise RuntimeError(f"Cannot analyze unverified simulation: {reason}")
        frame_path, _ = result_paths(output_root, task, "simulation")
        with np.load(frame_path, allow_pickle=False) as data:
            frame = np.asarray(data["frame"], dtype=np.complex128)
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            arrays = analyze_frame(frame, config, cut_progress=lambda count: progress_queue.put(("analysis_cut", task["task_id"], count)))
        metadata = {
            "task_id": task["task_id"], "construction": task["construction"], "sample_index": task["sample_index"],
            "seed": task["seed"], "config_hash": config_hash, "source_hashes": hashes,
            "source_frame": frame_path.name, "source_frame_sha256": sha256_path(frame_path),
            "config": config,
            "analysis_order": "trajectory_then_translated_cut_then_bootstrap",
            "covariance_pooling_before_propagation": False,
        }
        arrays["metadata_json"] = np.asarray(canonical_json(metadata))
        result_path, _ = result_paths(output_root, task, "analysis")
        _publish_result(result_path, arrays, task=task, stage="analysis", config_hash=config_hash, hashes=hashes)
        failure.unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        _record_failure(output_root, task, "analysis", exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}
    finally:
        active.unlink(missing_ok=True)


def inventory(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    tasks = expand_tasks(config)
    result: dict[str, Any] = {"identities": {}, "stages": {}}
    for stage in ("simulation", "analysis"):
        hashes, config_hash = source_hashes(stage), scientific_config_hash(config, stage)
        result["identities"][stage] = {"config_hash": config_hash, "source_hashes": hashes}
        rows = {}
        for construction in ("hard", "soft"):
            subset = [task for task in tasks if task["construction"] == construction]
            verified = [task for task in subset if verify_pair(output_root, task, stage, config_hash, hashes)[0]]
            active = []
            for task in subset:
                path = marker_path(output_root, task, stage, "active")
                if path.is_file():
                    try:
                        payload = json.loads(path.read_text())
                        if payload.get("host") == os.uname().nodename and Path(f"/proc/{int(payload['pid'])}").exists():
                            active.append(task)
                    except Exception:
                        pass
            failed = [task for task in subset if marker_path(output_root, task, stage, "failed").is_file() and task not in verified]
            rows[construction] = {"verified": len(verified), "pending": len(subset) - len(verified), "active": len(active), "failed": len(failed)}
        result["stages"][stage] = rows
    return result


def print_inventory(status: dict[str, Any]) -> None:
    for stage, identity in status["identities"].items():
        print(f"[identity] {stage} config_hash={identity['config_hash']}")
        for name, digest in identity["source_hashes"].items():
            print(f"[source] {stage} {name} sha256={digest}")
    for stage, constructions in status["stages"].items():
        for construction, counts in constructions.items():
            print(f"[status] {stage:10s} {construction:4s} verified={counts['verified']:3d} pending={counts['pending']:3d} active={counts['active']:2d} failed={counts['failed']:2d}")


def _drain_progress(queue, bars: dict[str, Any]) -> None:
    while True:
        try:
            stage, _task_id, count = queue.get_nowait()
        except Exception:
            break
        if stage in bars:
            bars[stage].update(int(count))


def run_stage(stage: str, config: dict[str, Any], output_root: Path, workers: int, resume: bool) -> None:
    tasks, hashes, config_hash = expand_tasks(config), source_hashes(stage), scientific_config_hash(config, stage)
    if not resume:
        existing = [task for task in tasks if verify_pair(output_root, task, stage, config_hash, hashes)[0]]
        if existing:
            raise RuntimeError(f"{len(existing)} verified {stage} tasks exist; use --resume.")
    pending = [task for task in tasks if not verify_pair(output_root, task, stage, config_hash, hashes)[0]]
    if stage == "analysis":
        simulation_hashes = source_hashes("simulation")
        simulation_config_hash = scientific_config_hash(config, "simulation")
        unavailable = [task for task in pending if not verify_pair(output_root, task, "simulation", simulation_config_hash, simulation_hashes)[0]]
        if unavailable:
            raise RuntimeError(f"{len(unavailable)} analysis tasks lack a verified frame.")
    verified_count = len(tasks) - len(pending)
    print(f"[{stage}] verified={verified_count} pending={len(pending)} workers={workers}")
    if not pending:
        return
    manager = mp.Manager()
    queue = manager.Queue()
    ctx = mp.get_context("fork")
    cycle_total = len(pending) * int(config["geometry"]["cycles"])
    cut_total = len(pending) * int(config["geometry"]["Ny"])
    bars: dict[str, Any] = {}
    if stage == "simulation":
        bars["simulation_cycle"] = tqdm(total=cycle_total, desc="physical cycles", unit="cycle", position=0)
        completion = tqdm(total=len(tasks), initial=verified_count, desc="trajectories", unit="traj", position=1)
        worker = _run_simulation_worker
    else:
        bars["analysis_cut"] = tqdm(total=cut_total, desc="modular cuts", unit="cut", position=0)
        completion = tqdm(total=len(tasks), initial=verified_count, desc="packet products", unit="traj", position=1)
        worker = _run_analysis_worker
    failures: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as pool:
            futures = {
                pool.submit(worker, (task, config, str(output_root), config_hash, hashes, queue)): task
                for task in pending
            }
            outstanding = set(futures)
            while outstanding:
                done, outstanding = wait(outstanding, timeout=0.5, return_when=FIRST_COMPLETED)
                _drain_progress(queue, bars)
                for future in done:
                    row = future.result()
                    if row["ok"]:
                        completion.update(1)
                    else:
                        failures.append(row)
                        tqdm.write(f"[{stage} failure] {row['task_id']}: {row['error']}")
            _drain_progress(queue, bars)
    finally:
        for bar in bars.values():
            bar.close()
        completion.close()
        manager.shutdown()
    if failures:
        raise RuntimeError(f"{len(failures)} {stage} tasks failed; rerun the same command to retry them.")


def aggregate(config: dict[str, Any], output_root: Path) -> None:
    tasks, hashes, config_hash = expand_tasks(config), source_hashes("analysis"), scientific_config_hash(config, "analysis")
    products: dict[str, list[Path]] = {"hard": [], "soft": []}
    for task in tasks:
        valid, reason = verify_pair(output_root, task, "analysis", config_hash, hashes)
        if not valid:
            raise RuntimeError(f"Cannot aggregate {task['task_id']}: {reason}")
        products[task["construction"]].append(result_paths(output_root, task, "analysis")[0])
    summary = aggregate_and_plot(products, config, output_root / "aggregate")
    print(f"[aggregate] wrote figure and summaries to {output_root / 'aggregate'}")
    for construction, row in summary["constructions"].items():
        print(f"[aggregate] {construction}: trajectories={row['trajectories']} H={row['handedness']}")


def smoke() -> None:
    config = copy.deepcopy(load_config())
    config["sampling_revision"] += "_smoke"
    config["root_seed"] += 9000
    config["geometry"].update({"Nx": 4, "Ny": 6, "cycles": 2, "samples_per_construction": 1})
    config["modular_analysis"].update({
        "subsystem_width": 3, "translated_cuts": 6, "wall_x": [1, 3], "packet_y_rel": [0, 2],
        "time_stop": 0.1, "time_step": 0.05, "spectral_cutoffs": [1e-10],
        "wall_window_radii": [1], "primary_wall_window_radius": 1,
        "fit_windows": [[0.0, 0.1]], "primary_fit_window": [0.0, 0.1], "bootstrap_draws": 100,
        "time_chunk": 8,
    })
    output = BUNDLE_ROOT / ".smoke_outputs"
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    print(f"[smoke] output={output}")
    run_stage("simulation", config, output, workers=2, resume=True)
    run_stage("analysis", config, output, workers=2, resume=True)
    status = inventory(config, output)
    print_inventory(status)
    if any(counts["verified"] != 1 for stage in status["stages"].values() for counts in stage.values()):
        raise RuntimeError("Smoke campaign did not verify every task.")
    aggregate(config, output)
    for suffix in ("pdf", "png"):
        if not (output / "aggregate" / f"modular_handedness_s100.{suffix}").is_file():
            raise RuntimeError(f"Smoke aggregation did not create the {suffix.upper()} figure.")
    shutil.rmtree(output)
    print("[smoke] passed: two constructions, exact frame resume, packet reduction, and figure")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "simulate", "analyze", "aggregate", "status", "smoke"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        smoke(); return 0
    config = load_config(args.config)
    validate_production_config(config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or config["execution"]["workers"])
    print(f"[campaign] revision={config['sampling_revision']} output={output_root}")
    print(f"[contract] Nx=20 Ny=40 nshell=1 cycles=80 S=100+100 DW=True alpha=(1,30) raster_y perfect_correction complex128")
    print(f"[execution] workers={workers} affinity={sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unknown'}")
    status = inventory(config, output_root)
    print_inventory(status)
    if args.command == "status":
        return 0
    if args.command in ("all", "simulate"):
        run_stage("simulation", config, output_root, workers, args.resume)
    if args.command in ("all", "analyze"):
        run_stage("analysis", config, output_root, workers, args.resume)
    if args.command in ("all", "aggregate"):
        aggregate(config, output_root)
    print(f"[complete] command={args.command} output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
