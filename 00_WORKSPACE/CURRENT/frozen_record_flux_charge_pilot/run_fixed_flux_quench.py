#!/usr/bin/env python3
"""Completion-resumable S10 fixed-flux frozen-record quenches."""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_frozen_continuous_ramp as frozen  # noqa: E402


CAMPAIGN_SCHEMA = "fixed_flux_quench_campaign_v1"
RESULT_SCHEMA = "fixed_flux_quench_result_v1"
COMPLETION_SCHEMA = "fixed_flux_quench_completion_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.fixed_flux_quench_s10_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N16x20_frozen_record_fixed_flux_quench_s10_v1"
WALLS = frozen.WALLS
DIRECTIONS = frozen.DIRECTIONS
SOURCE_PATHS = {
    "cpu_engine": frozen.SOURCE_PATHS["cpu_engine"],
    "occupied_frame": frozen.SOURCE_PATHS["occupied_frame"],
    "reference_campaign_runner": Path(frozen.__file__).resolve(),
    "fixed_flux_quench_runner": Path(__file__).resolve(),
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "root_seed", "source_campaign", "geometry",
        "walls", "dynamics", "fixed_twists", "near_2pi_offset", "directions",
        "regions", "ensemble", "acceptance",
    )
    return sha256_bytes(canonical_json({key: config[key] for key in keys}).encode("utf-8"))


def _source_path(config: dict[str, Any], key: str) -> Path:
    return (PROJECT_ROOT / str(config["source_campaign"][key])).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    if config.get("campaign_id") != "N16x20_frozen_record_fixed_flux_quench_s10_v1":
        raise ValueError("unexpected campaign_id")
    if int(config.get("root_seed", -1)) != 2026090303:
        raise ValueError("unexpected root seed")
    source = config["source_campaign"]
    if source.get("campaign_id") != "N16x20_frozen_record_continuous_ramp_s10_v1":
        raise ValueError("unexpected source campaign")
    source_config = frozen.load_config(_source_path(config, "config"))
    frozen.validate_config(source_config)
    if source_config["campaign_id"] != source["campaign_id"]:
        raise ValueError("source campaign identity mismatch")
    if config["geometry"] != source_config["geometry"]:
        raise ValueError("geometry differs from the frozen-record source")
    if config["walls"] != source_config["walls"]:
        raise ValueError("wall definitions differ from the frozen-record source")
    expected_dynamics = {
        "burn_in_cycles": 40,
        "replay_cycles": 64,
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "state_representation": "physical_frame",
        "physical_covariance_update": "rank1",
        "controller_twist_gauge": "uniform",
        "dtype": "complex128",
        "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
    }
    if config["dynamics"] != expected_dynamics:
        raise ValueError("dynamics differ from the locked fixed-flux protocol")
    expected_twists = (
        ("pi_over_2", 0.5 * math.pi),
        ("pi", math.pi),
        ("three_pi_over_2", 1.5 * math.pi),
        ("near_2pi", 2.0 * math.pi - 1e-7),
    )
    rows = config["fixed_twists"]
    if len(rows) != len(expected_twists):
        raise ValueError("fixed-twist grid must contain four magnitudes")
    for row, (name, radians) in zip(rows, expected_twists, strict=True):
        if row.get("name") != name or not math.isclose(
            float(row.get("radians", math.nan)), radians, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("fixed-twist grid differs from the locked protocol")
    if float(config["near_2pi_offset"]) != 1e-7:
        raise ValueError("near-2pi offset must equal 1e-7")
    if config["directions"] != {"ccw": 1, "cw": -1}:
        raise ValueError("unexpected direction convention")
    if config["regions"] != source_config["regions"]:
        raise ValueError("regional partition differs from the source campaign")
    if int(config["ensemble"]["samples_per_wall"]) != 10:
        raise ValueError("production ensemble must contain ten samples per wall")


def _seed(root_seed: int, wall: str, sample_id: int, twist_index: int) -> int:
    wall_code = {"soft": 0, "hard": 1}[wall]
    sequence = np.random.SeedSequence(
        int(root_seed), spawn_key=(wall_code, int(sample_id), int(twist_index))
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "stage": "fixed_flux_quench",
            "task_id": f"quench_{wall}_{direction}_{twist['name']}_sample_{sample_id:03d}",
            "wall": wall,
            "direction": direction,
            "sigma": int(config["directions"][direction]),
            "twist_name": str(twist["name"]),
            "twist_magnitude": float(twist["radians"]),
            "fixed_phi": int(config["directions"][direction]) * float(twist["radians"]),
            "twist_index": twist_index,
            "sample_id": sample_id,
            "seed": _seed(config["root_seed"], wall, sample_id, twist_index),
            "burnin_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
            "reference_task_id": f"reference_{wall}_sample_{sample_id:03d}",
            "cycles": int(config["dynamics"]["replay_cycles"]),
        }
        for wall in WALLS
        for sample_id in range(int(config["ensemble"]["samples_per_wall"]))
        for twist_index, twist in enumerate(config["fixed_twists"])
        for direction in DIRECTIONS
    ]
    ids = [row["task_id"] for row in rows]
    if len(rows) != 160 or len(ids) != len(set(ids)):
        raise RuntimeError("expected exactly 160 unique fixed-flux tasks")
    for wall in WALLS:
        for sample_id in range(10):
            for twist_index in range(4):
                pair = [
                    row for row in rows
                    if row["wall"] == wall and row["sample_id"] == sample_id
                    and row["twist_index"] == twist_index
                ]
                if len(pair) != 2 or pair[0]["seed"] != pair[1]["seed"]:
                    raise RuntimeError("CW/CCW fixed-flux seeds must be paired")
    return rows


def result_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    root = (
        Path(output_root) / "quenches" / task["twist_name"]
        / task["wall"] / task["direction"]
    )
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


def _task_metadata(task: dict[str, Any], config_hash: str, hashes: dict[str, str]) -> dict[str, Any]:
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
    dependencies: dict[str, str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    result, completion = result_paths(output_root, task)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(canonical_json(_task_metadata(task, config_hash, hashes)))
    payload["dependencies_json"] = np.asarray(canonical_json(dependencies))
    _atomic_npz(result, payload)
    result_record = {
        "name": result.name,
        "bytes": int(result.stat().st_size),
        "sha256": sha256_path(result),
    }
    _atomic_json(
        completion,
        {
            "schema": COMPLETION_SCHEMA,
            **_task_metadata(task, config_hash, hashes),
            "dependencies": dependencies,
            "result": result_record,
            "elapsed_seconds": float(elapsed_seconds),
            "completed_unix": time.time(),
        },
    )
    return result_record


def source_context(config: dict[str, Any]) -> dict[str, Any]:
    source_config = frozen.load_config(_source_path(config, "config"))
    frozen.validate_config(source_config)
    source_root = _source_path(config, "output_root")
    burnins = frozen.verify_burnins(source_config)
    source_hashes = frozen.source_hashes()
    source_config_hash = frozen.scientific_config_hash(source_config)
    references = frozen.verified_references(
        source_config, source_root, burnins, source_config_hash, source_hashes
    )
    if len(references) != 20:
        raise RuntimeError(f"expected 20 verified frozen records, found {len(references)}")
    return {
        "config": source_config,
        "root": source_root,
        "burnins": burnins,
        "references": references,
        "config_hash": source_config_hash,
        "source_hashes": source_hashes,
    }


def dependencies_for(task: dict[str, Any], source: dict[str, Any]) -> dict[str, str]:
    return {
        "burnin_sha256": source["burnins"][task["burnin_task_id"]]["sha256"],
        "reference_sha256": source["references"][task["reference_task_id"]]["sha256"],
    }


def load_reference(row: dict[str, Any], cycles: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], str, str]:
    path = Path(row["path"])
    if sha256_path(path) != row["sha256"]:
        raise RuntimeError(f"frozen reference changed after verification: {path}")
    with np.load(path, allow_pickle=False) as saved:
        raw = np.asarray(saved["record_json_utf8"], dtype=np.uint8).tobytes()
        record_sha = str(np.asarray(saved["record_sha256"]).item())
        if sha256_bytes(raw) != record_sha:
            raise RuntimeError("embedded frozen-record checksum mismatch")
        baseline = {
            key: np.array(saved[key], dtype=np.float64, copy=True)
            for key in ("N_left", "N_right", "N_total", "density_x")
        }
    record = frozen.record_prefix(frozen.parse_record(raw), int(cycles))
    return record, baseline, str(row["sha256"]), record_sha


def fixed_schedule(task: dict[str, Any]) -> np.ndarray:
    return np.full(int(task["cycles"]) + 1, float(task["fixed_phi"]), dtype=np.float64)


def verify_pair(
    output_root: Path,
    task: dict[str, Any],
    config_hash: str,
    hashes: dict[str, str],
    dependencies: dict[str, str],
) -> tuple[bool, str, dict[str, Any] | None]:
    result, completion_path = result_paths(output_root, task)
    if not result.is_file() or not completion_path.is_file():
        return False, "missing result/completion pair", None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "schema": COMPLETION_SCHEMA,
            **_task_metadata(task, config_hash, hashes),
            "dependencies": dependencies,
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
        length = int(task["cycles"]) + 1
        with np.load(result, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != RESULT_SCHEMA:
                return False, "result schema mismatch", None
            metadata = json.loads(str(np.asarray(saved["metadata_json"]).item()))
            if metadata != _task_metadata(task, config_hash, hashes):
                return False, "result identity mismatch", None
            saved_dependencies = json.loads(str(np.asarray(saved["dependencies_json"]).item()))
            if saved_dependencies != dependencies:
                return False, "result dependency mismatch", None
            one_dimensional = (
                "cycles", "phi", "N_left", "N_right", "N_total",
                "raw_delta_N_left", "raw_delta_N_right", "raw_delta_N_total",
                "raw_q_x", "reference_N_left", "reference_N_right",
                "reference_N_total", "response_delta_N_left",
                "response_delta_N_right", "response_delta_N_total", "response_q_x",
                "net_injected_charge", "injection_count", "charge_continuity_residual",
                "rank", "minimum_selected_probability_by_cycle",
                "branch_log_probability_by_cycle",
            )
            for key in one_dimensional:
                value = np.asarray(saved[key])
                if value.shape != (length,) or not np.all(np.isfinite(value)):
                    return False, f"invalid result field {key}", None
            for key in ("density_x", "reference_density_x", "response_density_x"):
                value = np.asarray(saved[key])
                if value.shape != (length, 16) or not np.all(np.isfinite(value)):
                    return False, f"invalid result field {key}", None
            if not np.array_equal(np.asarray(saved["cycles"]), np.arange(length)):
                return False, "cycle coordinates mismatch", None
            if not np.array_equal(np.asarray(saved["phi"]), fixed_schedule(task)):
                return False, "fixed-flux schedule mismatch", None
            left = np.asarray(saved["N_left"])
            right = np.asarray(saved["N_right"])
            total = np.asarray(saved["N_total"])
            reference_left = np.asarray(saved["reference_N_left"])
            reference_right = np.asarray(saved["reference_N_right"])
            reference_total = np.asarray(saved["reference_N_total"])
            if not np.allclose(saved["raw_delta_N_left"], left - left[0], rtol=0.0, atol=1e-12):
                return False, "raw left-charge identity mismatch", None
            if not np.allclose(saved["raw_delta_N_right"], right - right[0], rtol=0.0, atol=1e-12):
                return False, "raw right-charge identity mismatch", None
            if not np.allclose(saved["raw_delta_N_total"], total - total[0], rtol=0.0, atol=1e-12):
                return False, "raw total-charge identity mismatch", None
            if not np.allclose(saved["raw_q_x"], 0.5 * (saved["raw_delta_N_right"] - saved["raw_delta_N_left"]), rtol=0.0, atol=1e-12):
                return False, "raw q_x identity mismatch", None
            if not np.allclose(saved["response_delta_N_left"], left - reference_left, rtol=0.0, atol=1e-12):
                return False, "response left-charge identity mismatch", None
            if not np.allclose(saved["response_delta_N_right"], right - reference_right, rtol=0.0, atol=1e-12):
                return False, "response right-charge identity mismatch", None
            if not np.allclose(saved["response_delta_N_total"], total - reference_total, rtol=0.0, atol=1e-12):
                return False, "response total-charge identity mismatch", None
            wanted_qx = 0.5 * (saved["response_delta_N_right"] - saved["response_delta_N_left"])
            if not np.allclose(saved["response_q_x"], wanted_qx, rtol=0.0, atol=1e-12):
                return False, "response q_x identity mismatch", None
            if not np.allclose(saved["response_q_x"][0], 0.0, rtol=0.0, atol=1e-12):
                return False, "fixed-flux response does not share the reference origin", None
            if str(np.asarray(saved["reference_sha256"]).item()) != dependencies["reference_sha256"]:
                return False, "reference checksum mismatch", None
            if str(np.asarray(saved["burnin_sha256"]).item()) != dependencies["burnin_sha256"]:
                return False, "burn-in checksum mismatch", None
            gram = float(np.asarray(saved["final_frame_gram_residual"]).item())
            if not np.isfinite(gram) or gram < 0.0:
                return False, "invalid final-frame Gram residual", None
        return True, "verified", completion
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


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


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, source_config, output_text, config_hash, hashes, burnin_row, reference_row, dependencies, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    log_path = output_root / "logs" / "tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame = frozen.load_burnin(burnin_row)
        record, baseline, reference_sha, record_sha = load_reference(reference_row, task["cycles"])
        phi = fixed_schedule(task)
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = frozen._model(source_config, task["wall"])
                audit = frozen.RecordAudit(task["cycles"], capture=False)
                observer = frozen.ChargeObserver(model, source_config, phi, queue, "fixed flux")
                result = model.run_markov_circuit(
                    frame_init=frame,
                    frame_init_prepared=True,
                    controller_twist_schedule=phi,
                    controller_twist_gauge="uniform",
                    trajectory_replay=record,
                    trajectory_replay_probability_tol=float(
                        config["acceptance"]["trajectory_replay_probability_tolerance"]
                    ),
                    native_cycle_observer=observer.cycle,
                    native_event_observer=observer.event,
                    trajectory_weight_observer=audit,
                    **frozen._engine_kwargs(
                        source_config, task["wall"], task["cycles"], task["seed"]
                    ),
                )
        gram = float(result["native_final"]["gram_residual"])
        if not np.isfinite(gram) or gram > float(config["acceptance"]["frame_gram_tolerance"]):
            raise FloatingPointError(f"fixed-flux frame Gram residual failed: {gram:.3e}")
        observed = observer.arrays(source_config, audit)
        response_left = observed["N_left"] - baseline["N_left"]
        response_right = observed["N_right"] - baseline["N_right"]
        response_total = observed["N_total"] - baseline["N_total"]
        response_density = observed["density_x"] - baseline["density_x"]
        arrays = {
            "schema": np.asarray(RESULT_SCHEMA),
            "cycles": observed["cycles"],
            "phi": observed["phi"],
            "N_left": observed["N_left"],
            "N_right": observed["N_right"],
            "N_total": observed["N_total"],
            "raw_delta_N_left": observed["delta_N_left"],
            "raw_delta_N_right": observed["delta_N_right"],
            "raw_delta_N_total": observed["delta_N_total"],
            "raw_q_x": observed["q_x"],
            "reference_N_left": baseline["N_left"],
            "reference_N_right": baseline["N_right"],
            "reference_N_total": baseline["N_total"],
            "response_delta_N_left": response_left,
            "response_delta_N_right": response_right,
            "response_delta_N_total": response_total,
            "response_q_x": 0.5 * (response_right - response_left),
            "net_injected_charge": observed["net_injected_charge"],
            "injection_count": observed["injection_count"],
            "charge_continuity_residual": observed["charge_continuity_residual"],
            "rank": observed["rank"],
            "density_x": observed["density_x"],
            "reference_density_x": baseline["density_x"],
            "response_density_x": response_density,
            "minimum_selected_probability_by_cycle": observed["minimum_selected_probability_by_cycle"],
            "branch_log_probability_by_cycle": observed["branch_log_probability_by_cycle"],
            "burnin_sha256": np.asarray(dependencies["burnin_sha256"]),
            "reference_sha256": np.asarray(reference_sha),
            "record_sha256": np.asarray(record_sha),
            "record_entries": np.asarray(len(record), dtype=np.int64),
            "final_frame_gram_residual": np.asarray(gram),
        }
        if np.max(np.abs(arrays["response_delta_N_total"])) > float(
            config["acceptance"]["charge_continuity_tolerance"]
        ):
            raise FloatingPointError("paired total-charge response exceeded tolerance")
        result_record = publish_pair(
            output_root, task, arrays, config_hash, hashes, dependencies,
            time.perf_counter() - started,
        )
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"], "result": result_record}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def _selected(rows: list[dict[str, Any]], sample_ids: set[int] | None) -> list[dict[str, Any]]:
    if sample_ids is None:
        return rows
    return [row for row in rows if int(row["sample_id"]) in sample_ids]


def _drain_progress(queue: Any, cycle_bar: Any) -> None:
    while True:
        try:
            _, count = queue.get_nowait()
        except Exception:
            return
        cycle_bar.update(int(count))


def run_campaign(
    config: dict[str, Any],
    source: dict[str, Any],
    output_root: Path,
    workers: int,
    sample_ids: set[int] | None,
    resume: bool,
) -> None:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    selected = _selected(tasks(config), sample_ids)
    pending: list[tuple[dict[str, Any], dict[str, str]]] = []
    verified: list[dict[str, Any]] = []
    for task in selected:
        dependencies = dependencies_for(task, source)
        ok, _, _ = verify_pair(output_root, task, config_hash, hashes, dependencies)
        if resume and ok:
            verified.append(task)
        else:
            pending.append((task, dependencies))
    total_cycles = sum(int(task["cycles"]) for task in selected)
    completed_cycles = sum(int(task["cycles"]) for task in verified)
    print(
        f"[fixed flux] verified={len(verified)}/{len(selected)} "
        f"pending={len(pending)} workers={workers}",
        flush=True,
    )
    if not pending:
        return
    context = mp.get_context("spawn")
    with mp.Manager() as manager:
        queue = manager.Queue()
        cycle_bar = tqdm(
            total=total_cycles, initial=completed_cycles,
            desc="fixed-flux cycles", unit="cycle", position=0,
        )
        task_bar = tqdm(
            total=len(selected), initial=len(verified),
            desc="fixed-flux tasks", unit="task", position=1,
        )
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=min(int(workers), len(pending)), mp_context=context) as pool:
            futures = {
                pool.submit(
                    _worker,
                    (
                        task, config, source["config"], str(output_root), config_hash,
                        hashes, source["burnins"][task["burnin_task_id"]],
                        source["references"][task["reference_task_id"]],
                        dependencies, queue,
                    ),
                )
                for task, dependencies in pending
            }
            while futures:
                done, futures = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                _drain_progress(queue, cycle_bar)
                for future in done:
                    row = future.result()
                    task_bar.update(1)
                    if not row["ok"]:
                        failure = f"{row['task_id']}: {row['error']}"
                        failures.append(failure)
                        tqdm.write(f"[fixed-flux failure] {failure}")
        _drain_progress(queue, cycle_bar)
        cycle_bar.close()
        task_bar.close()
    if failures:
        raise RuntimeError(f"fixed-flux campaign failed for {len(failures)} task(s)")


def inventory(config: dict[str, Any], source: dict[str, Any], output_root: Path) -> dict[str, Any]:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    rows = {}
    for task in tasks(config):
        rows[task["task_id"]] = verify_pair(
            output_root, task, config_hash, hashes, dependencies_for(task, source)
        )
    return {"config_hash": config_hash, "source_hashes": hashes, "tasks": rows}


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    complete = sum(row[0] for row in status["tasks"].values())
    print(f"[campaign] {config['campaign_id']}")
    print("[contract] Nx=16 Ny=20 raster-y, soft+hard, S=10, fixed 64-cycle frozen-record replay")
    print("[twists] +/- pi/2, pi, 3pi/2, and (2pi-1e-7); no ramp and no tangent modes")
    print("[observable] direct paired state response relative to the same zero-flux record history")
    print(f"[output] {output_root.resolve()}")
    print(f"[identity] config_sha256={status['config_hash']}")
    for name, digest in status["source_hashes"].items():
        print(f"[source] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] verified={complete}/160 pending={160-complete}")


def validate_pairing(
    config: dict[str, Any], source: dict[str, Any], output_root: Path,
    sample_ids: set[int] | None = None,
) -> None:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    grouped: dict[tuple[str, int, str], dict[str, Path]] = {}
    for task in _selected(tasks(config), sample_ids):
        ok, reason, _ = verify_pair(
            output_root, task, config_hash, hashes, dependencies_for(task, source)
        )
        if not ok:
            raise RuntimeError(f"unverified fixed-flux replay {task['task_id']}: {reason}")
        grouped.setdefault(
            (task["wall"], task["sample_id"], task["twist_name"]), {}
        )[task["direction"]] = result_paths(output_root, task)[0]
    for key, paths in grouped.items():
        if set(paths) != set(DIRECTIONS):
            raise RuntimeError(f"incomplete direction pair for {key}")
        with np.load(paths["ccw"], allow_pickle=False) as positive, np.load(paths["cw"], allow_pickle=False) as negative:
            for field in ("rank", "net_injected_charge", "injection_count"):
                if not np.array_equal(positive[field], negative[field]):
                    raise RuntimeError(f"paired {field} mismatch for {key}")
            if str(positive["record_sha256"]) != str(negative["record_sha256"]):
                raise RuntimeError(f"paired frozen-record mismatch for {key}")
            if not np.array_equal(positive["phi"], -np.asarray(negative["phi"])):
                raise RuntimeError(f"paired fixed twists do not reverse for {key}")
    print(f"[paired validation] verified {len(grouped)} CW/CCW fixed-flux pairs", flush=True)


def write_identity(config: dict[str, Any], source: dict[str, Any], output_root: Path) -> None:
    identity_path = output_root / "campaign_identity.json"
    payload = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_hash": scientific_config_hash(config),
        "source_hashes": source_hashes(),
        "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
        "reused_burnins": {
            key: value["sha256"] for key, value in sorted(source["burnins"].items())
        },
        "reused_references": {
            key: value["sha256"] for key, value in sorted(source["references"].items())
        },
        "configuration": config,
    }
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "campaign identity changed after initialization; use a new versioned output root"
            )
        return
    _atomic_json(identity_path, payload)


def _parse_sample_ids(value: str | None) -> set[int] | None:
    if value is None:
        return None
    parsed = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not parsed or min(parsed) < 0 or max(parsed) >= 10:
        raise ValueError("sample IDs must be a nonempty comma list drawn from 0..9")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "run", "all", "validate"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    source = source_context(config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or config["execution"]["workers"])
    if workers <= 0:
        raise ValueError("workers must be positive")
    sample_ids = _parse_sample_ids(args.sample_ids)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    print(f"[execution] workers={workers} affinity={affinity or 'unavailable'} BLAS_threads={config['execution']['blas_threads']}")
    print("[inputs] verified 20/20 burn-ins and 20/20 frozen 64-cycle records")
    write_identity(config, source, output_root)
    print_inventory(config, output_root, inventory(config, source, output_root))
    if args.stage == "report":
        return
    if args.stage in ("run", "all"):
        run_campaign(config, source, output_root, workers, sample_ids, args.resume)
    if args.stage in ("validate", "all"):
        validate_pairing(config, source, output_root, sample_ids)
    final = inventory(config, source, output_root)
    print_inventory(config, output_root, final)
    if args.analyze:
        if not all(row[0] for row in final["tasks"].values()):
            raise RuntimeError("analysis requires all 160 verified fixed-flux tasks")
        import analyze_fixed_flux_quench

        analyze_fixed_flux_quench.analyze(config, source, output_root)
    print("[complete] fixed-flux frozen-record command finished successfully")


if __name__ == "__main__":
    main()
