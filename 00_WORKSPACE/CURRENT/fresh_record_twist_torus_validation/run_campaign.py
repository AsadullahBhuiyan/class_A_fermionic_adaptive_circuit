#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
FGTN_ROOT = SRC_ROOT / "fgtn"
if str(FGTN_ROOT) not in sys.path:
    sys.path.insert(0, str(FGTN_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from classA_U1FGTN import classA_U1FGTN
from occupied_frame import OccupiedFrameState
from twist_torus import exact_target_frame, gauge_vector, projector_distance

CONFIG_PATH = HERE / "campaign_config.v1.json"
RESULTS_ROOT = HERE / "results"
PROTOCOLS = ("fresh_full_record", "fresh_outcomes")
SOURCE_FILES = (
    Path("src/fgtn/classA_U1FGTN.py"),
    Path("src/fgtn/occupied_frame.py"),
    Path("00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation/twist_torus.py"),
    Path("00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation/run_campaign.py"),
    Path("00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation/analyze_results.py"),
    Path("00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation/campaign_config.v1.json"),
)
CHANNELS = ("Ap", "Am", "Bp", "Bm")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_hashes() -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.exists():
            hashes[str(relative)] = sha256_file(path)
    return hashes


def campaign_root(campaign_id: str) -> Path:
    return RESULTS_ROOT / campaign_id


def update_manifest(root: Path, stage: str, status: str, **details: Any) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["updated_utc"] = utc_now()
    manifest["stages"][stage] = {
        "status": status,
        "recorded_utc": utc_now(),
        **details,
    }
    write_json_atomic(path, manifest)


def initialize_campaign(campaign_id: str) -> Path:
    root = campaign_root(campaign_id)
    for relative in (
        "prepared",
        "raw/target",
        "raw/trajectories",
        "processed",
        "figures",
        "reports",
        "status",
        "logs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config = load_config()
    config_hash = payload_hash(config)
    hashes = source_hashes()
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("config_sha256") != config_hash:
            raise RuntimeError("Resume configuration differs from the locked campaign.")
        if manifest.get("source_sha256") != hashes:
            raise RuntimeError("Resume source hashes differ from the locked campaign.")
        return root
    write_json_atomic(root / "campaign_config.v1.json", config)
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "config_sha256": config_hash,
            "source_sha256": hashes,
            "canonical_dynamics_entry_point": config["canonical_dynamics_entry_point"],
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ["git", "status", "--short"],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                ).stdout.strip()
            ),
            "stages": {},
        },
    )
    return root


def derived_seeds(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config() if config is None else config
    g = config["geometry"]
    points = int(g["twist_grid_x"]) * int(g["twist_grid_y"])
    children = np.random.SeedSequence(int(config["root_seed"])).spawn(2 + 4 * points)

    def value(child: np.random.SeedSequence) -> int:
        return int(child.generate_state(1, dtype=np.uint64)[0])

    cursor = 2
    result: dict[str, Any] = {
        "initial_state": value(children[0]),
        "common_schedule": value(children[1]),
    }
    for protocol in PROTOCOLS:
        result[protocol] = {
            "schedules": [value(item) for item in children[cursor : cursor + points]],
            "outcomes": [value(item) for item in children[cursor + points : cursor + 2 * points]],
        }
        cursor += 2 * points
    return result


def build_model(config: dict[str, Any], tx: float, ty: float, *, nshell: int | None) -> classA_U1FGTN:
    g = config["geometry"]
    model = classA_U1FGTN(
        Nx=int(g["Nx"]),
        Ny=int(g["Ny"]),
        DW=False,
        nshell=nshell,
        filling_frac=float(g["filling_frac"]),
        alpha_1=float(g["alpha_1"]),
        alpha_2=float(g["alpha_2"]),
        trial_orbitals=str(g["trial_orbitals"]),
        dw_truncation=False,
        twist_x=float(tx),
        twist_y=float(ty),
    )
    model.construct_OW_projectors(
        nshell=nshell,
        DW=False,
        trial_orbitals=str(g["trial_orbitals"]),
        dw_truncation=False,
        twist_x=float(tx),
        twist_y=float(ty),
    )
    return model


def twist_values(config: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    config = load_config() if config is None else config
    g = config["geometry"]
    return (
        2 * np.pi * np.arange(int(g["twist_grid_x"])) / int(g["twist_grid_x"]),
        2 * np.pi * np.arange(int(g["twist_grid_y"])) / int(g["twist_grid_y"]),
    )


def schedule_from_seed(config: dict[str, Any], seed: int) -> np.ndarray:
    g = config["geometry"]
    sites = np.arange(int(g["Nx"]) * int(g["Ny"]), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.stack([rng.permutation(sites) for _ in range(int(g["cycles"]))])


def prepare_common_inputs(root: Path) -> None:
    config = load_config()
    seeds = derived_seeds(config)
    initial_path = root / "prepared/initial_state.npz"
    schedule_path = root / "prepared/common_schedule.npz"
    summary_path = root / "prepared/summary.json"
    if summary_path.exists() and initial_path.exists() and schedule_path.exists():
        summary = json.loads(summary_path.read_text())
        if all(
            sha256_file(path) == summary["files"].get(path.name)
            for path in (initial_path, schedule_path)
        ):
            update_manifest(root, "prepare", "complete", resumed=True)
            return
    update_manifest(root, "prepare", "running")
    model = build_model(config, 0.0, 0.0, nshell=int(config["geometry"]["nshell"]))
    dimension = 2 * int(config["geometry"]["Nx"]) * int(config["geometry"]["Ny"])
    initial = model.random_complex_fermion_covariance(
        dimension, rng=np.random.default_rng(int(seeds["initial_state"]))
    )
    initial = 0.5 * (initial + initial.conj().T)
    defect = float(np.linalg.norm(initial @ initial - np.eye(dimension), ord="fro") / np.sqrt(dimension))
    if defect > 1e-10:
        raise RuntimeError(f"Prepared initial covariance is not pure: {defect:.3e}.")
    schedule = schedule_from_seed(config, int(seeds["common_schedule"]))
    save_npz_atomic(initial_path, G0=initial)
    save_npz_atomic(schedule_path, site_schedule=schedule)
    summary = {
        "status": "complete",
        "initial_state_seed": seeds["initial_state"],
        "common_schedule_seed": seeds["common_schedule"],
        "initial_purity_defect": defect,
        "initial_state_content_sha256": model._checkpoint_array_signature(initial),
        "common_schedule_content_sha256": model._checkpoint_array_signature(schedule),
        "files": {path.name: sha256_file(path) for path in (initial_path, schedule_path)},
    }
    write_json_atomic(summary_path, summary)
    update_manifest(root, "prepare", "complete", details=summary)


def chern_partitions(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = config["geometry"]
    nx, ny = int(g["Nx"]), int(g["Ny"])
    radius = 0.4 * min(nx, ny)
    xref, yref = nx // 2, ny // 2
    masks = [np.zeros((nx, ny), dtype=bool) for _ in range(3)]
    for dy in range(-int(radius), int(radius) + 1):
        y = yref + dy
        if not 0 <= y < ny:
            continue
        maximum = int(np.floor(np.sqrt(max(0.0, radius * radius - dy * dy))))
        for x in range(max(0, xref - maximum), min(nx - 1, xref + maximum) + 1):
            angle = float(np.mod(np.arctan2(dy, x - xref), 2 * np.pi))
            sector = 0 if angle < 2 * np.pi / 3 else 1 if angle < 4 * np.pi / 3 else 2
            masks[sector][x, y] = True

    def rows(mask: np.ndarray) -> np.ndarray:
        x, y = np.nonzero(mask)
        return np.sort(np.concatenate((2 * x + 2 * nx * y, 1 + 2 * x + 2 * nx * y))).astype(np.int64)

    return tuple(rows(mask) for mask in masks)  # type: ignore[return-value]


def real_space_chern(frame: np.ndarray, partitions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    ia, ib, ic = partitions

    def block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return frame[left].conj() @ frame[right].T

    value = 12 * np.pi * 1j * (
        np.trace(block(ic, ia) @ block(ia, ib) @ block(ib, ic))
        - np.trace(block(ia, ic) @ block(ic, ib) @ block(ib, ia))
    )
    return float(np.real(value))


_WORKER_CPU_IDS: list[int] = []


def worker_initializer(cpu_ids: list[int]) -> None:
    global _WORKER_CPU_IDS
    _WORKER_CPU_IDS = list(cpu_ids)
    identity = multiprocessing.current_process()._identity
    ordinal = (identity[-1] - 1) if identity else os.getpid()
    cpu = _WORKER_CPU_IDS[ordinal % len(_WORKER_CPU_IDS)]
    os.sched_setaffinity(0, {int(cpu)})


def target_shard_path(root: Path, ix: int, iy: int) -> Path:
    return root / "raw/target" / f"tx_{ix:02d}_ty_{iy:02d}.npz"


def run_target_task(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["root"])
    path = target_shard_path(root, int(task["ix"]), int(task["iy"]))
    if path.exists() and sha256_file(path) == task.get("existing_sha256"):
        return {"ix": task["ix"], "iy": task["iy"], "sha256": task["existing_sha256"], "resumed": True}
    config = load_config()
    model = build_model(config, float(task["tx"]), float(task["ty"]), nshell=None)
    frame = exact_target_frame(model)
    marker = real_space_chern(frame, chern_partitions(config))
    save_npz_atomic(
        path,
        frame=frame,
        twist_x=np.asarray(task["tx"]),
        twist_y=np.asarray(task["ty"]),
        rank=np.asarray(frame.shape[1]),
        real_space_chern=np.asarray(marker),
    )
    return {"ix": task["ix"], "iy": task["iy"], "sha256": sha256_file(path), "resumed": False}


def stage_target(root: Path, cpu_ids: list[int]) -> None:
    config = load_config()
    txs, tys = twist_values(config)
    summary_path = root / "raw/target/summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else {"files": {}}
    tasks = []
    for ix, tx in enumerate(txs):
        for iy, ty in enumerate(tys):
            path = target_shard_path(root, ix, iy)
            tasks.append(
                {
                    "root": str(root),
                    "ix": ix,
                    "iy": iy,
                    "tx": float(tx),
                    "ty": float(ty),
                    "existing_sha256": existing.get("files", {}).get(path.name),
                }
            )
    update_manifest(root, "target_torus", "running", tasks=len(tasks))
    rows = []
    with ProcessPoolExecutor(
        max_workers=len(cpu_ids), initializer=worker_initializer, initargs=(cpu_ids,)
    ) as pool:
        futures = [pool.submit(run_target_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if completed % 32 == 0 or completed == len(tasks):
                print(f"target-torus {completed}/{len(tasks)}", flush=True)

    zero_path = target_shard_path(root, 0, 0)
    with np.load(zero_path) as data:
        zero = np.array(data["frame"], copy=True)
    reference_model = build_model(config, 0.0, 0.0, nshell=int(config["geometry"]["nshell"]))
    alpha = np.ones((reference_model.Nx, reference_model.Ny), dtype=float)
    reference_covariance = reference_model.G_CI_domain_wall(periodic=True, alpha=alpha)
    reference_state = OccupiedFrameState.from_centered_covariance(reference_covariance, representation="physical_frame")
    zero_error = projector_distance(zero, reference_state.frame)

    closure = {}
    for axis in ("x", "y"):
        tx, ty = (2 * np.pi, 0.0) if axis == "x" else (0.0, 2 * np.pi)
        model = build_model(config, tx, ty, nshell=None)
        closed = exact_target_frame(model)
        gauge = gauge_vector(reference_model.Nx, reference_model.Ny, axis)[:, None]
        closure[axis] = projector_distance(closed, gauge * zero)
    files = {target_shard_path(root, row["ix"], row["iy"]).name: row["sha256"] for row in rows}
    summary = {
        "status": "complete",
        "tasks": len(tasks),
        "zero_twist_projector_error": zero_error,
        "closure_projector_error": closure,
        "files": files,
    }
    write_json_atomic(summary_path, summary)
    update_manifest(root, "target_torus", "complete", details=summary)


class FrameCapture:
    def __init__(self, config: dict[str, Any]) -> None:
        self.cycles: list[int] = []
        self.marker: list[float] = []
        self.entropy: list[float] = []
        self.charge: list[float] = []
        self.rank: list[int] = []
        self.gram: list[float] = []
        self.final_frame: np.ndarray | None = None
        g = config["geometry"]
        mask = np.zeros((2, int(g["Nx"]), int(g["Ny"])), dtype=bool)
        region = config["half_region"]
        mask[:, :, int(region["y_start"]) : int(region["y_stop_exclusive"])] = True
        self.region_rows = np.flatnonzero(mask.reshape(-1, order="F"))
        self.partitions = chern_partitions(config)

    def __call__(self, **payload: Any) -> None:
        state: OccupiedFrameState = payload["state"]
        frame = state.physical_frame
        self.cycles.append(int(payload["cycle"]))
        self.marker.append(real_space_chern(frame, self.partitions))
        self.entropy.append(state.regional_entropy(self.region_rows))
        self.charge.append(state.regional_charge(slice(None)))
        self.rank.append(state.rank)
        self.gram.append(state.gram_residual())
        self.final_frame = np.array(frame, copy=True)


class CompactRecord:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def __call__(self, **payload: Any) -> None:
        measurements = [event for event in payload["branch_events"] if event["kind"] == "measurement"]
        if tuple(event["channel"] for event in measurements) != CHANNELS:
            raise RuntimeError("Unexpected measurement-channel order in canonical record.")
        self.entries.append(
            {
                "cycle": int(payload["cycle"]),
                "site_id": int(payload["site_id"]),
                "branch_log_weight": float(payload["branch_log_weight"]),
                "cumulative_log_weight": float(payload["cumulative_log_weight"]),
                "probability": [float(event["probability"]) for event in measurements],
                "outcome": [bool(event["outcome_occupied"]) for event in measurements],
            }
        )

    def arrays(self, cycles: int) -> dict[str, np.ndarray]:
        site_cycle = np.asarray([entry["cycle"] for entry in self.entries], dtype=np.int16)
        site_id = np.asarray([entry["site_id"] for entry in self.entries], dtype=np.int16)
        branch_log = np.asarray([entry["branch_log_weight"] for entry in self.entries], dtype=np.float64)
        cumulative_site = np.asarray([entry["cumulative_log_weight"] for entry in self.entries], dtype=np.float64)
        probability = np.asarray([entry["probability"] for entry in self.entries], dtype=np.float64)
        outcome = np.asarray([entry["outcome"] for entry in self.entries], dtype=np.uint8)
        cumulative_cycle = np.zeros(cycles + 1, dtype=np.float64)
        for cycle in range(1, cycles + 1):
            selected = cumulative_site[site_cycle == cycle]
            if selected.size:
                cumulative_cycle[cycle] = selected[-1]
        return {
            "record_site_cycle": site_cycle,
            "record_site_id": site_id,
            "record_branch_log_weight": branch_log,
            "record_cumulative_log_weight": cumulative_site,
            "measurement_probability": probability,
            "measurement_outcome": outcome,
            "cumulative_log_weight": cumulative_cycle,
        }


def trajectory_paths(root: Path, protocol: str, ix: int, iy: int) -> tuple[Path, Path]:
    directory = root / "raw/trajectories" / protocol / f"tx_{ix:02d}_ty_{iy:02d}"
    return directory / "trajectory.npz", directory / "summary.json"


def run_trajectory_task(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["root"])
    data_path, summary_path = trajectory_paths(root, task["protocol"], task["ix"], task["iy"])
    if summary_path.exists() and data_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("task_sha256") == task["task_sha256"] and sha256_file(data_path) == summary.get("data_sha256"):
            return summary

    config = load_config()
    g = config["geometry"]
    prepared = json.loads((root / "prepared/summary.json").read_text())
    initial_path = root / "prepared/initial_state.npz"
    if sha256_file(initial_path) != prepared["files"][initial_path.name]:
        raise RuntimeError("Prepared initial-state file changed before trajectory execution.")
    with np.load(initial_path) as data:
        initial = np.array(data["G0"], dtype=np.complex128, copy=True)

    if task["protocol"] == "fresh_outcomes":
        schedule_path = root / "prepared/common_schedule.npz"
        with np.load(schedule_path) as data:
            schedule = np.array(data["site_schedule"], dtype=np.int64, copy=True)
    else:
        schedule = schedule_from_seed(config, int(task["schedule_seed"]))

    model = build_model(
        config,
        float(task["twist_x"]),
        float(task["twist_y"]),
        nshell=int(g["nshell"]),
    )
    schedule_hash = model._checkpoint_array_signature(schedule)
    capture = FrameCapture(config)
    record = CompactRecord()
    started = time.perf_counter_ns()
    result = model.run_markov_circuit(
        cycles=int(g["cycles"]),
        samples=1,
        init_mode="default",
        G_init=initial,
        sequence=str(config["dynamics"]["sequence"]),
        perfect_correction=bool(config["dynamics"]["perfect_correction"]),
        postselect=False,
        postselect_probability=0.0,
        meas_slab_only=False,
        physical_covariance_update=str(config["dynamics"]["physical_covariance_update"]),
        random_seed=int(task["outcome_seed"]),
        G_history=False,
        save=False,
        progress=False,
        parallelize_samples=False,
        state_representation="physical_frame",
        return_native_state=True,
        native_cycle_observer=capture,
        trajectory_weight_observer=record,
        site_schedule_replay=schedule,
        trajectory_replay_probability_tol=float(config["acceptance"]["replay_probability_tolerance"]),
        timing_level="off",
    )
    wall_ns = time.perf_counter_ns() - started
    expected_cycles = list(range(int(g["cycles"]) + 1))
    if capture.cycles != expected_cycles or capture.final_frame is None:
        raise RuntimeError("Native observer did not capture every required cycle.")
    if result.get("trajectory_replay"):
        raise RuntimeError("A full trajectory record was unexpectedly replayed.")
    if result.get("site_schedule_replay_sha256") != schedule_hash:
        raise RuntimeError("Canonical dynamics reported a different schedule hash.")
    if not np.isclose(result.get("twist_x"), task["twist_x"], atol=1e-14, rtol=0):
        raise RuntimeError("Canonical dynamics reported a different x twist.")
    if not np.isclose(result.get("twist_y"), task["twist_y"], atol=1e-14, rtol=0):
        raise RuntimeError("Canonical dynamics reported a different y twist.")
    record_arrays = record.arrays(int(g["cycles"]))
    expected_sites = int(g["cycles"]) * int(g["Nx"]) * int(g["Ny"])
    if record_arrays["record_site_id"].size != expected_sites:
        raise RuntimeError("Compact record has the wrong number of visited sites.")
    arrays = {
        "cycles": np.arange(int(g["cycles"]) + 1, dtype=np.int16),
        "real_space_chern": np.asarray(capture.marker),
        "half_entropy": np.asarray(capture.entropy),
        "charge": np.asarray(capture.charge),
        "rank": np.asarray(capture.rank, dtype=np.int16),
        "gram_residual": np.asarray(capture.gram),
        "final_frame": capture.final_frame,
        "site_schedule": schedule,
        **record_arrays,
    }
    save_npz_atomic(data_path, **arrays)
    outcome_digest = hashlib.sha256(record_arrays["measurement_outcome"].tobytes()).hexdigest()
    summary = {
        "status": "complete",
        "task_sha256": task["task_sha256"],
        "protocol": task["protocol"],
        "ix": int(task["ix"]),
        "iy": int(task["iy"]),
        "twist_x": float(task["twist_x"]),
        "twist_y": float(task["twist_y"]),
        "schedule_seed": int(task["schedule_seed"]),
        "outcome_seed": int(task["outcome_seed"]),
        "schedule_content_sha256": schedule_hash,
        "outcome_digest": outcome_digest,
        "trajectory_replay": False,
        "site_schedule_replay": True,
        "final_rank": int(capture.rank[-1]),
        "final_real_space_chern": float(capture.marker[-1]),
        "maximum_gram_residual": float(np.max(capture.gram)),
        "wall_ns": int(wall_ns),
        "peak_rss_kib_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "data_sha256": sha256_file(data_path),
    }
    write_json_atomic(summary_path, summary)
    return summary


def trajectory_tasks(root: Path) -> list[dict[str, Any]]:
    config = load_config()
    txs, tys = twist_values(config)
    seeds = derived_seeds(config)
    tasks = []
    for flat, (ix, iy) in enumerate(np.ndindex(len(txs), len(tys))):
        for protocol in PROTOCOLS:
            schedule_seed = (
                seeds["common_schedule"]
                if protocol == "fresh_outcomes"
                else seeds[protocol]["schedules"][flat]
            )
            payload = {
                "protocol": protocol,
                "ix": ix,
                "iy": iy,
                "twist_x": float(txs[ix]),
                "twist_y": float(tys[iy]),
                "schedule_seed": int(schedule_seed),
                "outcome_seed": int(seeds[protocol]["outcomes"][flat]),
            }
            tasks.append({"root": str(root), **payload, "task_sha256": payload_hash(payload)})
    return tasks


def stage_trajectories(root: Path, cpu_ids: list[int]) -> None:
    tasks = trajectory_tasks(root)
    update_manifest(root, "trajectories", "running", tasks=len(tasks), cpu_ids=cpu_ids)
    completed_rows = []
    with ProcessPoolExecutor(
        max_workers=len(cpu_ids), initializer=worker_initializer, initargs=(cpu_ids,)
    ) as pool:
        futures = [pool.submit(run_trajectory_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            completed_rows.append(row)
            if completed % 16 == 0 or completed == len(tasks):
                print(
                    f"trajectories {completed}/{len(tasks)} latest={row['protocol']} "
                    f"({row['ix']},{row['iy']}) rank={row['final_rank']}",
                    flush=True,
                )
    update_manifest(root, "trajectories", "complete", tasks=len(completed_rows), cpu_ids=cpu_ids)


def select_physical_cpu_ids(limit: int) -> list[int]:
    selected: dict[tuple[int, int], int] = {}
    for cpu_path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda path: int(path.name[3:])):
        cpu = int(cpu_path.name[3:])
        topology = cpu_path / "topology"
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        selected.setdefault((package, core), cpu)
    cpus = list(selected.values())
    if len(cpus) < limit:
        raise RuntimeError(f"Only {len(cpus)} physical cores are available; {limit} requested.")
    return cpus[:limit]


def parse_cpu_list(value: str) -> list[int]:
    result = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = map(int, token.split("-", 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    if len(set(result)) != len(result):
        raise ValueError("CPU_LIST contains duplicate logical CPUs.")
    return result


def stage_preflight(root: Path) -> None:
    update_manifest(root, "preflight", "running")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_flux_cocycle_replay.py",
        "tests/test_fresh_record_twist_torus.py",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    (root / "logs/preflight.log").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        update_manifest(root, "preflight", "failed", returncode=result.returncode)
        raise RuntimeError(f"Preflight tests failed; see {root / 'logs/preflight.log'}.")
    update_manifest(root, "preflight", "complete", command=command)


def run_analysis(root: Path) -> None:
    command = [sys.executable, str(HERE / "analyze_results.py"), "--campaign-root", str(root)]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    (root / "logs/analysis.log").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        update_manifest(root, "analyze", "failed", returncode=result.returncode)
        raise RuntimeError(f"Analysis failed; see {root / 'logs/analysis.log'}.")
    update_manifest(root, "analyze", "complete")


def stage_validate(root: Path) -> bool:
    config = load_config()
    g = config["geometry"]
    acceptance = config["acceptance"]
    errors: list[str] = []
    expected_cycles = np.arange(int(g["cycles"]) + 1)
    common_schedule_hashes: set[str] = set()
    full_schedule_hashes: set[str] = set()
    outcome_seeds: dict[str, set[int]] = {protocol: set() for protocol in PROTOCOLS}
    outcome_digests: dict[str, set[str]] = {protocol: set() for protocol in PROTOCOLS}
    for task in trajectory_tasks(root):
        data_path, summary_path = trajectory_paths(root, task["protocol"], task["ix"], task["iy"])
        if not data_path.exists() or not summary_path.exists():
            errors.append(f"missing trajectory {task['protocol']} ({task['ix']},{task['iy']})")
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("task_sha256") != task["task_sha256"] or sha256_file(data_path) != summary.get("data_sha256"):
            errors.append(f"hash mismatch {task['protocol']} ({task['ix']},{task['iy']})")
            continue
        with np.load(data_path) as data:
            if not np.array_equal(data["cycles"], expected_cycles):
                errors.append(f"incomplete cycles {task['protocol']} ({task['ix']},{task['iy']})")
            for key in ("real_space_chern", "half_entropy", "charge", "rank", "gram_residual", "cumulative_log_weight", "final_frame"):
                if key not in data or not np.all(np.isfinite(data[key])):
                    errors.append(f"missing/nonfinite {key} {task['protocol']} ({task['ix']},{task['iy']})")
            if np.max(data["gram_residual"]) > float(acceptance["gram_hard"]):
                errors.append(f"Gram hard failure {task['protocol']} ({task['ix']},{task['iy']})")
        if summary.get("trajectory_replay"):
            errors.append(f"unexpected trajectory replay {task['protocol']} ({task['ix']},{task['iy']})")
        destination = common_schedule_hashes if task["protocol"] == "fresh_outcomes" else full_schedule_hashes
        destination.add(summary["schedule_content_sha256"])
        outcome_seeds[task["protocol"]].add(int(summary["outcome_seed"]))
        outcome_digests[task["protocol"]].add(summary["outcome_digest"])

    points = int(g["twist_grid_x"]) * int(g["twist_grid_y"])
    if len(common_schedule_hashes) != 1:
        errors.append("fresh_outcomes did not use exactly one common schedule")
    if len(full_schedule_hashes) != points:
        errors.append("fresh_full_record schedules are not unique at all twist points")
    for protocol in PROTOCOLS:
        if len(outcome_seeds[protocol]) != points:
            errors.append(f"{protocol} outcome seeds are not unique")
        if len(outcome_digests[protocol]) != points:
            errors.append(f"{protocol} realized outcome records are not unique")

    target_summary_path = root / "raw/target/summary.json"
    if not target_summary_path.exists():
        errors.append("target summary is missing")
    else:
        target = json.loads(target_summary_path.read_text())
        if target["zero_twist_projector_error"] > float(acceptance["target_zero_projector_hard"]):
            errors.append("zero-twist exact target disagrees with the existing periodic target")
        if max(target["closure_projector_error"].values()) > float(acceptance["target_closure_hard"]):
            errors.append("large-gauge target closure failed")
    analysis_path = root / "processed/analysis_summary.json"
    if not analysis_path.exists():
        errors.append("analysis summary is missing")
    else:
        analysis = json.loads(analysis_path.read_text())
        target_chern = analysis["surfaces"]["target"]["chern"]
        if abs(target_chern - 1.0) > float(acceptance["target_chern_hard"]):
            errors.append("exact target twist torus did not produce C=1")

    summary = {"status": "complete", "passed": not errors, "errors": errors}
    write_json_atomic(root / "status/validation.json", summary)
    update_manifest(root, "validate", "complete" if not errors else "failed", details=summary)
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=("all", "init", "preflight", "target", "trajectories", "analyze", "validate"))
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--cpu-list", default=os.environ.get("CPU_LIST"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config()
    campaign_id = args.campaign_id or f"N16_T16_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    workers = int(config["parallel"]["workers"])
    cpu_ids = parse_cpu_list(args.cpu_list) if args.cpu_list else select_physical_cpu_ids(workers)
    if len(cpu_ids) != workers:
        raise ValueError(f"Exactly {workers} physical CPU IDs are required; received {len(cpu_ids)}.")
    if args.dry_run:
        print(json.dumps({"campaign_id": campaign_id, "trajectory_tasks": 512, "target_tasks": 256, "cpu_ids": cpu_ids}, indent=2))
        return 0

    root = initialize_campaign(campaign_id)
    print(f"campaign_root={root}", flush=True)
    if args.stage in {"all", "init", "preflight", "target", "trajectories"}:
        prepare_common_inputs(root)
    if args.stage in {"all", "preflight"} or args.preflight_only:
        stage_preflight(root)
        if args.preflight_only:
            return 0
    if args.stage in {"all", "target"}:
        stage_target(root, cpu_ids)
    if args.stage in {"all", "trajectories"}:
        stage_trajectories(root, cpu_ids)
    if args.stage in {"all", "analyze"}:
        run_analysis(root)
    passed = True
    if args.stage in {"all", "validate"}:
        passed = stage_validate(root)
    print(f"validation_passed={passed}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
