#!/usr/bin/env python3
"""Run the fixed-schedule, outcome-only trajectory-independence campaign."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CONFIG_PATH = HERE / "campaign_config.v1.json"
RESULTS_ROOT = HERE / "results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402
from src.fgtn.occupied_frame import OccupiedFrameState  # noqa: E402


SOURCE_FILES = (
    Path("src/fgtn/classA_U1FGTN.py"),
    Path("src/fgtn/occupied_frame.py"),
    Path("tests/test_occupied_frame.py"),
    Path("tests/test_trajectory_independence_campaign.py"),
    Path("00_WORKSPACE/CURRENT/trajectory_independence_validation/campaign_config.v1.json"),
    Path("00_WORKSPACE/CURRENT/trajectory_independence_validation/run_campaign.py"),
    Path("00_WORKSPACE/CURRENT/trajectory_independence_validation/analyze_results.py"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_gzip_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(json_ready(payload), handle, separators=(",", ":"))
    temporary.replace(path)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row}) if rows else []
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows([{key: json_ready(row.get(key)) for key in fields} for row in rows])
    temporary.replace(path)


def derived_seeds(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config() if config is None else config
    samples = int(cfg["geometry"]["samples"])
    children = np.random.SeedSequence(int(cfg["root_seed"])).spawn(samples + 2)
    values = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
    return {
        "initial_state": values[0],
        "schedule": values[1],
        "outcomes": values[2:],
    }


def build_model(config: dict[str, Any] | None = None) -> classA_U1FGTN:
    cfg = load_config() if config is None else config
    g = cfg["geometry"]
    model = classA_U1FGTN(
        Nx=int(g["Nx"]),
        Ny=int(g["Ny"]),
        DW=bool(g["DW"]),
        nshell=int(g["nshell"]),
        filling_frac=float(g["filling_frac"]),
        alpha_1=float(g["alpha_1"]),
        alpha_2=float(g["alpha_2"]),
        trial_orbitals=str(g["trial_orbitals"]),
        dw_truncation=bool(g["dw_truncation"]),
    )
    model.construct_OW_projectors(
        nshell=int(g["nshell"]),
        DW=bool(g["DW"]),
        trial_orbitals=str(g["trial_orbitals"]),
        dw_truncation=bool(g["dw_truncation"]),
    )
    return model


def half_region_rows(config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = load_config() if config is None else config
    g = cfg["geometry"]
    mask = np.zeros((2, int(g["Nx"]), int(g["Ny"])), dtype=bool)
    mask[:, :, int(cfg["half_region"]["y_start"]):int(cfg["half_region"]["y_stop_exclusive"])] = True
    return np.flatnonzero(mask.reshape(-1, order="F"))


def chern_partition_indices(config: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = load_config() if config is None else config
    g = cfg["geometry"]
    nx, ny = int(g["Nx"]), int(g["Ny"])
    radius = 0.4 * min(nx, ny)
    xref, yref = nx // 2, ny // 2
    masks = [np.zeros((nx, ny), dtype=bool) for _ in range(3)]
    for dy in range(-int(np.floor(radius)), int(np.floor(radius)) + 1):
        y = yref + dy
        if not 0 <= y < ny:
            continue
        max_dx = int(np.floor(np.sqrt(radius * radius - dy * dy)))
        for x in range(max(0, xref - max_dx), min(nx - 1, xref + max_dx) + 1):
            theta = float(np.mod(np.arctan2(dy, x - xref), 2.0 * np.pi))
            sector = 0 if theta < 2.0 * np.pi / 3.0 else 1 if theta < 4.0 * np.pi / 3.0 else 2
            masks[sector][x, y] = True

    def indices(mask: np.ndarray) -> np.ndarray:
        xs, ys = np.nonzero(mask)
        return np.sort(np.concatenate((2 * xs + 2 * nx * ys, 1 + 2 * xs + 2 * nx * ys))).astype(np.int64)

    return tuple(indices(mask) for mask in masks)  # type: ignore[return-value]


def real_space_chern_from_frame(
    state: OccupiedFrameState,
    partitions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    frame = state.physical_frame
    i_a, i_b, i_c = partitions

    def block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return frame[left].conj() @ frame[right].T

    value = 12.0 * np.pi * 1j * (
        np.trace(block(i_c, i_a) @ block(i_a, i_b) @ block(i_b, i_c))
        - np.trace(block(i_a, i_c) @ block(i_c, i_b) @ block(i_b, i_a))
    )
    return float(np.real(value))


def campaign_root(campaign_id: str) -> Path:
    return RESULTS_ROOT / str(campaign_id)


def source_hashes() -> dict[str, str]:
    return {str(path): sha256_file(REPO_ROOT / path) for path in SOURCE_FILES}


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--short")),
        "status": run("git", "status", "--short"),
    }


def initialize_campaign(campaign_id: str) -> Path:
    root = campaign_root(campaign_id)
    for relative in ("raw/trajectories", "raw/audit", "prepared", "processed/tables", "figures", "reports", "status", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config = load_config()
    config_hash = payload_hash(config)
    hashes = source_hashes()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_sha256") != config_hash:
            raise RuntimeError("Resume config hash does not match the locked campaign.")
        if manifest.get("source_sha256") != hashes:
            raise RuntimeError("Resume source hashes do not match the locked campaign.")
        return root
    write_json_atomic(root / "campaign_config.v1.json", config)
    manifest = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "created_utc": utc_now(),
        "config_sha256": config_hash,
        "source_sha256": hashes,
        "derived_seeds": derived_seeds(config),
        "canonical_dynamics_entry_point": config["canonical_dynamics_entry_point"],
        "git": git_metadata(),
        "stages": {},
    }
    write_json_atomic(manifest_path, manifest)
    return root


def update_manifest(root: Path, stage: str, status: str, **details: Any) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["updated_utc"] = utc_now()
    manifest["stages"][stage] = {"status": status, "recorded_utc": utc_now(), **details}
    write_json_atomic(path, manifest)


def prepared_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "prepared/initial_state.npz",
        root / "prepared/site_schedule.npz",
        root / "prepared/target_state.npz",
        root / "prepared/summary.json",
    )


def stage_prepare(root: Path) -> None:
    initial_path, schedule_path, target_path, summary_path = prepared_paths(root)
    if summary_path.exists() and all(path.exists() for path in (initial_path, schedule_path, target_path)):
        summary = json.loads(summary_path.read_text())
        if all(sha256_file(path) == summary["files"].get(path.name) for path in (initial_path, schedule_path, target_path)):
            update_manifest(root, "prepare", "complete", resumed=True)
            return
    update_manifest(root, "prepare", "running")
    cfg = load_config()
    g = cfg["geometry"]
    seeds = derived_seeds(cfg)
    model = build_model(cfg)
    dimension = 2 * int(g["Nx"]) * int(g["Ny"])
    initial = model.random_complex_fermion_covariance(
        N=dimension, rng=np.random.default_rng(int(seeds["initial_state"]))
    )
    initial = 0.5 * (initial + initial.conj().T)
    purity_defect = float(np.linalg.norm(initial @ initial - np.eye(dimension), ord="fro") / np.sqrt(dimension))
    if purity_defect > 1e-10:
        raise RuntimeError(f"Prepared initial state is not pure: {purity_defect:.3e}.")

    schedule_rng = np.random.default_rng(int(seeds["schedule"]))
    base_sites = np.arange(int(g["Nx"]) * int(g["Ny"]), dtype=np.int64)
    schedule = np.stack([schedule_rng.permutation(base_sites) for _ in range(int(g["cycles"]))])

    alpha = np.full((int(g["Nx"]), int(g["Ny"])), float(g["alpha_1"]), dtype=np.float64)
    target_covariance = model.G_CI_domain_wall(periodic=True, alpha=alpha)
    target_state = OccupiedFrameState.from_centered_covariance(
        target_covariance, representation="physical_frame"
    )
    target_chern = real_space_chern_from_frame(target_state, chern_partition_indices(cfg))

    save_npz_atomic(initial_path, G0=initial)
    save_npz_atomic(schedule_path, site_schedule=schedule)
    save_npz_atomic(
        target_path,
        frame=target_state.frame,
        centered_covariance=target_covariance,
        real_space_chern=np.asarray(target_chern),
    )
    summary = {
        "status": "complete",
        "initial_state_seed": seeds["initial_state"],
        "schedule_seed": seeds["schedule"],
        "initial_purity_defect": purity_defect,
        "target_chern": target_chern,
        "schedule_shape": list(schedule.shape),
        "files": {path.name: sha256_file(path) for path in (initial_path, schedule_path, target_path)},
    }
    write_json_atomic(summary_path, summary)
    update_manifest(root, "prepare", "complete", details=summary)


class TrajectoryRecord:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def __call__(self, **payload: Any) -> None:
        self.entries.append(
            {
                "cycle": int(payload["cycle"]),
                "site_id": int(payload["site_id"]),
                "branch_log_weight": float(payload["branch_log_weight"]),
                "cumulative_log_weight": float(payload["cumulative_log_weight"]),
                "branch_events": [dict(event) for event in payload["branch_events"]],
            }
        )


class FrameCapture:
    def __init__(self, config: dict[str, Any]) -> None:
        self.cycles: list[int] = []
        self.frames: list[np.ndarray] = []
        self.chern: list[float] = []
        self.charge: list[float] = []
        self.rank: list[int] = []
        self.half_entropy: list[float] = []
        self.gram: list[float] = []
        self.occupations: list[np.ndarray] = []
        self.rows = half_region_rows(config)
        self.partitions = chern_partition_indices(config)

    def __call__(self, **payload: Any) -> None:
        state: OccupiedFrameState = payload["state"]
        physical = state.physical_frame
        self.cycles.append(int(payload["cycle"]))
        self.frames.append(np.array(state.frame, copy=True))
        self.chern.append(real_space_chern_from_frame(state, self.partitions))
        self.charge.append(state.regional_charge(slice(None)))
        self.rank.append(state.rank)
        self.half_entropy.append(state.regional_entropy(self.rows))
        self.gram.append(state.gram_residual())
        self.occupations.append(np.sum(np.abs(physical) ** 2, axis=1).real)


def sample_paths(root: Path, sample: int) -> tuple[Path, Path, Path]:
    directory = root / "raw/trajectories" / f"sample_{sample:02d}"
    return directory / "cycle_data.npz", directory / "record.json.gz", directory / "summary.json"


def outcome_digest(entries: list[dict[str, Any]]) -> str:
    values = []
    for entry in entries:
        for event in entry["branch_events"]:
            if event.get("kind") == "measurement":
                values.append(
                    (
                        int(entry["cycle"]),
                        int(entry["site_id"]),
                        str(event["channel"]),
                        bool(event["outcome_occupied"]),
                    )
                )
    return payload_hash(values)


def pin_worker(cpu: int) -> None:
    os.sched_setaffinity(0, {int(cpu)})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _trajectory_worker(task: dict[str, Any]) -> dict[str, Any]:
    pin_worker(int(task["cpu"]))
    root = Path(task["root"])
    sample = int(task["sample"])
    seed = int(task["seed"])
    data_path, record_path, summary_path = sample_paths(root, sample)
    expected_task_hash = str(task["task_hash"])
    if summary_path.exists() and data_path.exists() and record_path.exists():
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("task_hash") == expected_task_hash
            and sha256_file(data_path) == summary.get("files", {}).get(data_path.name)
            and sha256_file(record_path) == summary.get("files", {}).get(record_path.name)
        ):
            return summary

    cfg = load_config()
    g = cfg["geometry"]
    initial_path, schedule_path, _, prepared_summary_path = prepared_paths(root)
    prepared_summary = json.loads(prepared_summary_path.read_text())
    if sha256_file(initial_path) != prepared_summary["files"][initial_path.name]:
        raise RuntimeError("Initial-state hash changed before worker launch.")
    if sha256_file(schedule_path) != prepared_summary["files"][schedule_path.name]:
        raise RuntimeError("Schedule hash changed before worker launch.")
    with np.load(initial_path) as payload:
        initial = np.array(payload["G0"], dtype=np.complex128, copy=True)
    with np.load(schedule_path) as payload:
        schedule = np.array(payload["site_schedule"], dtype=np.int64, copy=True)

    model = build_model(cfg)
    capture = FrameCapture(cfg)
    record = TrajectoryRecord()
    started = time.perf_counter_ns()
    result = model.run_markov_circuit(
        cycles=int(g["cycles"]),
        samples=1,
        init_mode="default",
        G_init=initial,
        sequence=str(cfg["protocol"]["sequence"]),
        perfect_correction=bool(cfg["protocol"]["perfect_correction"]),
        postselect=bool(cfg["protocol"]["postselect"]),
        postselect_probability=float(cfg["protocol"]["postselect_probability"]),
        meas_slab_only=bool(g["meas_slab_only"]),
        physical_covariance_update=str(cfg["protocol"]["physical_covariance_update"]),
        random_seed=seed,
        G_history=False,
        save=False,
        progress=False,
        parallelize_samples=False,
        state_representation=str(cfg["protocol"]["state_representation"]),
        return_native_state=True,
        native_cycle_observer=capture,
        trajectory_weight_observer=record,
        site_schedule_replay=schedule,
        trajectory_replay_probability_tol=float(cfg["acceptance"]["replay_probability_tolerance"]),
        timing_level="off",
    )
    wall_ns = time.perf_counter_ns() - started
    expected_cycles = list(range(int(g["cycles"]) + 1))
    if capture.cycles != expected_cycles:
        raise RuntimeError(f"Sample {sample} did not capture complete cycles 0..T.")
    expected_sites = int(g["cycles"]) * int(g["Nx"]) * int(g["Ny"])
    if len(record.entries) != expected_sites:
        raise RuntimeError(f"Sample {sample} captured {len(record.entries)} of {expected_sites} sites.")
    expected_schedule_hash = model._checkpoint_array_signature(schedule)
    if result.get("site_schedule_replay_sha256") != expected_schedule_hash:
        raise RuntimeError("Canonical result reported a different schedule hash.")

    cumulative = np.zeros(int(g["cycles"]) + 1, dtype=np.float64)
    for entry in record.entries:
        cumulative[int(entry["cycle"])] = float(entry["cumulative_log_weight"])
    arrays: dict[str, Any] = {
        "cycles": np.arange(int(g["cycles"]) + 1, dtype=np.int64),
        "real_space_chern": np.asarray(capture.chern),
        "charge": np.asarray(capture.charge),
        "rank": np.asarray(capture.rank, dtype=np.int64),
        "half_entropy": np.asarray(capture.half_entropy),
        "gram_residual": np.asarray(capture.gram),
        "local_occupations": np.stack(capture.occupations),
        "cumulative_log_weight": cumulative,
    }
    for cycle, frame in zip(capture.cycles, capture.frames):
        arrays[f"frame_cycle_{cycle:03d}"] = frame
    save_npz_atomic(data_path, **arrays)
    write_gzip_json_atomic(
        record_path,
        {
            "schema_version": 1,
            "sample": sample,
            "outcome_seed": seed,
            "initial_state_sha256": prepared_summary["files"][initial_path.name],
            "schedule_sha256": prepared_summary["files"][schedule_path.name],
            "schedule_content_sha256": expected_schedule_hash,
            "entries": record.entries,
        },
    )
    summary = {
        "status": "complete",
        "task_hash": expected_task_hash,
        "sample": sample,
        "outcome_seed": seed,
        "cpu": int(task["cpu"]),
        "wall_ns": int(wall_ns),
        "peak_rss_kib_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "initial_state_sha256": prepared_summary["files"][initial_path.name],
        "schedule_sha256": prepared_summary["files"][schedule_path.name],
        "schedule_content_sha256": expected_schedule_hash,
        "outcome_digest": outcome_digest(record.entries),
        "final_rank": int(capture.rank[-1]),
        "final_charge": float(capture.charge[-1]),
        "final_chern": float(capture.chern[-1]),
        "maximum_gram_residual": float(np.max(capture.gram)),
        "files": {data_path.name: sha256_file(data_path), record_path.name: sha256_file(record_path)},
    }
    write_json_atomic(summary_path, summary)
    return summary


def stage_trajectories(root: Path, cpu_ids: list[int]) -> None:
    update_manifest(root, "trajectories", "running", cpu_list=cpu_ids)
    cfg = load_config()
    seeds = derived_seeds(cfg)["outcomes"]
    initial_path, schedule_path, _, summary_path = prepared_paths(root)
    prepared = json.loads(summary_path.read_text())
    tasks = []
    for sample, seed in enumerate(seeds):
        task_payload = {
            "sample": sample,
            "seed": int(seed),
            "initial": prepared["files"][initial_path.name],
            "schedule": prepared["files"][schedule_path.name],
            "config": payload_hash(cfg),
        }
        tasks.append(
            {
                "root": str(root),
                "sample": sample,
                "seed": int(seed),
                "cpu": int(cpu_ids[sample]),
                "task_hash": payload_hash(task_payload),
            }
        )
    stage_started = time.perf_counter_ns()
    rows = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=context) as pool:
        futures = [pool.submit(_trajectory_worker, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"trajectory sample={row['sample']:02d} rank={row['final_rank']} "
                f"chern={row['final_chern']:.9f}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["sample"]))
    write_csv_atomic(root / "processed/tables/trajectory_runtime.csv", rows)
    update_manifest(
        root,
        "trajectories",
        "complete",
        sample_count=len(rows),
        stage_total_ns=int(time.perf_counter_ns() - stage_started),
    )


def stage_audit(root: Path) -> bool:
    update_manifest(root, "audit", "running")
    cfg = load_config()
    g = cfg["geometry"]
    initial_path, _, _, _ = prepared_paths(root)
    data_path, record_path, _ = sample_paths(root, 0)
    with np.load(initial_path) as payload:
        initial = np.array(payload["G0"], dtype=np.complex128, copy=True)
    with np.load(data_path) as payload:
        frame_arrays = {
            cycle: np.array(payload[f"frame_cycle_{cycle:03d}"], copy=True)
            for cycle in range(int(g["cycles"]) + 1)
        }
    record = read_gzip_json(record_path)["entries"]
    covariance_cycles: dict[int, np.ndarray] = {}

    def observer(**payload: Any) -> None:
        covariance_cycles[int(payload["cycle"])] = np.array(payload["G"], copy=True)

    model = build_model(cfg)
    model.run_markov_circuit(
        cycles=int(g["cycles"]),
        samples=1,
        init_mode="default",
        G_init=initial,
        sequence=str(cfg["protocol"]["sequence"]),
        perfect_correction=bool(cfg["protocol"]["perfect_correction"]),
        random_seed=int(derived_seeds(cfg)["outcomes"][0]),
        G_history=False,
        save=False,
        progress=False,
        parallelize_samples=False,
        physical_covariance_update=str(cfg["protocol"]["physical_covariance_update"]),
        trajectory_replay=record,
        cycle_observer=observer,
        trajectory_replay_probability_tol=float(cfg["acceptance"]["replay_probability_tolerance"]),
    )
    relative = np.empty(int(g["cycles"]) + 1)
    maximum = np.empty_like(relative)
    for cycle in range(int(g["cycles"]) + 1):
        frame = frame_arrays[cycle]
        centered = 2.0 * (frame @ frame.conj().T) - np.eye(frame.shape[0])
        difference = centered - covariance_cycles[cycle]
        relative[cycle] = np.linalg.norm(difference) / max(np.linalg.norm(covariance_cycles[cycle]), 1.0)
        maximum[cycle] = np.max(np.abs(difference))
    audit_path = root / "raw/audit/sample_00_covariance_replay.npz"
    save_npz_atomic(audit_path, cycles=np.arange(relative.size), relative_error=relative, maximum_error=maximum)
    hard = float(cfg["acceptance"]["covariance_audit_hard"])
    passed = bool(np.all(np.isfinite(relative)) and max(np.max(relative), np.max(maximum)) <= hard)
    summary = {
        "status": "complete",
        "passed": passed,
        "maximum_relative_error": float(np.max(relative)),
        "maximum_element_error": float(np.max(maximum)),
        "hard": hard,
        "file_sha256": sha256_file(audit_path),
    }
    write_json_atomic(root / "raw/audit/summary.json", summary)
    update_manifest(root, "audit", "complete" if passed else "failed", details=summary)
    return passed


def _run_analysis(root: Path, mode: str) -> None:
    command = [sys.executable, str(HERE / "analyze_results.py"), mode, "--campaign-root", str(root)]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    (root / f"logs/{mode}.log").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{mode} failed; see {root / f'logs/{mode}.log'}")


def stage_validate(root: Path) -> bool:
    cfg = load_config()
    g = cfg["geometry"]
    errors: list[str] = []
    summaries = []
    expected_cycles = np.arange(int(g["cycles"]) + 1)
    initial_hashes, schedule_hashes, content_hashes, seeds, outcomes = set(), set(), set(), set(), set()
    for sample in range(int(g["samples"])):
        data_path, record_path, summary_path = sample_paths(root, sample)
        if not all(path.exists() for path in (data_path, record_path, summary_path)):
            errors.append(f"missing sample shard {sample}")
            continue
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        if sha256_file(data_path) != summary["files"].get(data_path.name):
            errors.append(f"cycle-data hash mismatch sample {sample}")
        if sha256_file(record_path) != summary["files"].get(record_path.name):
            errors.append(f"record hash mismatch sample {sample}")
        with np.load(data_path) as payload:
            if not np.array_equal(payload["cycles"], expected_cycles):
                errors.append(f"incomplete cycles sample {sample}")
            if any(f"frame_cycle_{cycle:03d}" not in payload for cycle in expected_cycles):
                errors.append(f"missing frame cycle sample {sample}")
            declared = ("real_space_chern", "charge", "rank", "half_entropy", "gram_residual", "local_occupations", "cumulative_log_weight")
            for key in declared:
                if key not in payload or not np.all(np.isfinite(payload[key])):
                    errors.append(f"nonfinite or missing {key} sample {sample}")
            if np.max(payload["gram_residual"]) > float(cfg["acceptance"]["gram_hard"]):
                errors.append(f"Gram residual hard failure sample {sample}")
        initial_hashes.add(summary["initial_state_sha256"])
        schedule_hashes.add(summary["schedule_sha256"])
        content_hashes.add(summary["schedule_content_sha256"])
        seeds.add(int(summary["outcome_seed"]))
        outcomes.add(summary["outcome_digest"])
    if len(initial_hashes) != 1:
        errors.append("samples do not share one initial-state hash")
    if len(schedule_hashes) != 1 or len(content_hashes) != 1:
        errors.append("samples do not share one schedule hash")
    if len(seeds) != int(g["samples"]):
        errors.append("outcome seeds are not distinct")
    if len(outcomes) < 2:
        errors.append("all realized outcome records are identical")
    audit_path = root / "raw/audit/summary.json"
    if not audit_path.exists() or not json.loads(audit_path.read_text()).get("passed"):
        errors.append("covariance replay audit failed or is missing")
    analysis_path = root / "processed/analysis_summary.json"
    if not analysis_path.exists():
        errors.append("analysis summary is missing")
    passed = not errors
    summary = {"status": "complete", "passed": passed, "errors": errors}
    write_json_atomic(root / "status/validation.json", summary)
    update_manifest(root, "validate", "complete" if passed else "failed", details=summary)
    return passed


def _cpu_snapshot() -> dict[int, tuple[int, int]]:
    result = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
            continue
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        result[int(fields[0][3:])] = (sum(values), idle)
    return result


def _cpu_topology(cpu: int) -> tuple[int, int, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    topology = base / "topology"
    package = int((topology / "physical_package_id").read_text())
    core = int((topology / "core_id").read_text())
    nodes = sorted(base.glob("node[0-9]*"))
    node = int(nodes[0].name[4:]) if nodes else package
    return node, package, core


def select_idle_cpu_ids(limit: int, allow_busy: bool = False) -> list[int]:
    first = _cpu_snapshot()
    time.sleep(0.6)
    second = _cpu_snapshot()
    one_per_core: dict[tuple[int, int], tuple[float, int, int]] = {}
    for cpu in sorted(second):
        node, package, core = _cpu_topology(cpu)
        total = second[cpu][0] - first[cpu][0]
        idle = second[cpu][1] - first[cpu][1]
        usage = 100.0 * (1.0 - idle / total) if total > 0 else 100.0
        key = (package, core)
        row = (usage, node, cpu)
        if key not in one_per_core or usage < one_per_core[key][0]:
            one_per_core[key] = row
    threshold = float(load_config()["parallel"]["idle_cpu_threshold_percent"])
    by_node: dict[int, list[tuple[float, int]]] = {}
    for usage, node, cpu in one_per_core.values():
        if allow_busy or usage <= threshold:
            by_node.setdefault(node, []).append((usage, cpu))
    choices = []
    for node, rows in by_node.items():
        rows.sort()
        if len(rows) >= limit:
            choices.append((sum(value for value, _ in rows[:limit]), node, rows))
    if not choices:
        return []
    return [int(cpu) for _, cpu in min(choices, key=lambda item: (item[0], item[1]))[2][:limit]]


def parse_cpu_list(value: str) -> list[int]:
    cpus = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(item) for item in token.split("-", 1))
            cpus.extend(range(start, stop + 1))
        else:
            cpus.append(int(token))
    if len(set(cpus)) != len(cpus):
        raise ValueError("CPU list contains duplicates.")
    return cpus


def validate_cpu_ids(cpu_ids: list[int], required: int) -> list[int]:
    if len(cpu_ids) < required:
        raise ValueError(f"At least {required} CPUs are required.")
    selected = cpu_ids[:required]
    available = os.sched_getaffinity(0)
    if any(cpu not in available for cpu in selected):
        raise ValueError("Requested CPU lies outside the current affinity mask.")
    topology = [_cpu_topology(cpu) for cpu in selected]
    if len({(package, core) for _, package, core in topology}) != required:
        raise ValueError("CPU list contains hyperthread siblings.")
    if len({node for node, _, _ in topology}) != 1:
        raise ValueError("All workers must lie on one NUMA node.")
    return selected


def stage_preflight(root: Path) -> bool:
    update_manifest(root, "preflight", "running")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_occupied_frame.py",
        "tests/test_trajectory_independence_campaign.py",
        "-k",
        "site_schedule_replay or trajectory_independence",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    (root / "logs/preflight_pytest.log").write_text(result.stdout + "\n" + result.stderr)
    passed = result.returncode == 0
    summary = {"status": "complete", "passed": passed, "returncode": result.returncode, "command": command}
    write_json_atomic(root / "status/preflight.json", summary)
    update_manifest(root, "preflight", "complete" if passed else "failed", details=summary)
    return passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("init", "preflight", "prepare", "trajectories", "audit", "analyze", "validate", "report", "all"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--cpu-list", default="")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--select-idle-cpus", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--allow-busy", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.select_idle_cpus:
        print(",".join(map(str, select_idle_cpu_ids(args.limit, args.allow_busy))))
        return 0
    if not args.stage or not args.campaign_id:
        raise SystemExit("stage and --campaign-id are required")
    cfg = load_config()
    workers = int(cfg["parallel"]["workers"])
    if args.max_workers != workers:
        raise SystemExit(f"This locked campaign requires --max-workers={workers}.")
    root = initialize_campaign(args.campaign_id)
    cpu_ids = parse_cpu_list(args.cpu_list)
    if args.stage in ("trajectories", "all"):
        cpu_ids = validate_cpu_ids(cpu_ids, workers)
    if args.stage == "init":
        return 0
    if args.stage == "preflight":
        return 0 if stage_preflight(root) else 1
    if args.stage == "prepare":
        stage_prepare(root)
    elif args.stage == "trajectories":
        stage_prepare(root)
        stage_trajectories(root, cpu_ids)
    elif args.stage == "audit":
        return 0 if stage_audit(root) else 1
    elif args.stage == "analyze":
        _run_analysis(root, "analyze")
        update_manifest(root, "analyze", "complete")
    elif args.stage == "validate":
        return 0 if stage_validate(root) else 1
    elif args.stage == "report":
        _run_analysis(root, "report")
        update_manifest(root, "report", "complete")
    elif args.stage == "all":
        started = time.perf_counter_ns()
        preflight = stage_preflight(root)
        if not preflight:
            update_manifest(root, "campaign", "failed", campaign_total_ns=int(time.perf_counter_ns() - started))
            return 1
        stage_prepare(root)
        stage_trajectories(root, cpu_ids)
        audit = stage_audit(root)
        _run_analysis(root, "analyze")
        update_manifest(root, "analyze", "complete")
        validation = stage_validate(root)
        _run_analysis(root, "report")
        update_manifest(root, "report", "complete")
        passed = bool(audit and validation)
        update_manifest(root, "campaign", "complete" if passed else "failed", campaign_total_ns=int(time.perf_counter_ns() - started))
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
