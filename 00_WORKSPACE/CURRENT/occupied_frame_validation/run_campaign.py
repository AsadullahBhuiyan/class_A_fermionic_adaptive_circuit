#!/usr/bin/env python3
"""Run the paired occupied-frame validation and performance campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    # This is a locked one-process-per-core campaign.  Inheriting (for example)
    # OPENBLAS_NUM_THREADS=32 would invalidate both affinity and the timings.
    os.environ[_name] = "1"

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CONFIG_PATH = HERE / "campaign_config.v1.json"
CAMPAIGN_SIZE: int | None = None
RESULTS_ROOT = HERE / "results"
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402
from src.fgtn.occupied_frame import OccupiedFrameState  # noqa: E402


CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"


def campaign_source_paths() -> tuple[Path, ...]:
    return (
        CONFIG_PATH,
        HERE / "run_campaign.py",
        HERE / "analyze_results.py",
        HERE / "launch_tmux.sh",
        HERE / "tmux_entrypoint.sh",
        HERE / "README.md",
        REPO_ROOT / "src/fgtn/classA_U1FGTN.py",
        REPO_ROOT / "src/fgtn/occupied_frame.py",
        REPO_ROOT / "tests/test_occupied_frame.py",
        REPO_ROOT / "tests/test_occupied_frame_campaign.py",
    )


def current_source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in campaign_source_paths()
        if path.exists()
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            writer.writerows(
                [{key: json_ready(row.get(key)) for key in fields} for row in rows]
            )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    if CAMPAIGN_SIZE is not None:
        size = int(CAMPAIGN_SIZE)
        geometry = config["geometry"]
        geometry["Nx"] = size
        geometry["Ny"] = size
        geometry["cycles"] = 2 * size
        config["checkpoint_cycles"] = [
            0,
            1,
            size // 2,
            size,
            3 * size // 2,
            2 * size,
        ]
        config["half_region"]["y_stop_exclusive"] = size // 2
        config["physics_gate"]["terminal_window_start_cycle"] = 3 * size // 2
    return config


def config_sha256() -> str:
    return payload_hash(load_config())


def campaign_root(campaign_id: str) -> Path:
    return RESULTS_ROOT / str(campaign_id)


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=REPO_ROOT, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--short")),
        "status": run("git", "status", "--short"),
    }


def initialize_campaign(campaign_id: str) -> Path:
    root = campaign_root(campaign_id)
    for relative in (
        "raw/records",
        "raw/correctness",
        "raw/benchmark",
        "processed/tables",
        "processed/arrays",
        "figures",
        "reports",
        "logs",
        "status",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config = load_config()
    config_copy = root / "campaign_config.v1.json"
    if config_copy.exists():
        if payload_hash(json.loads(config_copy.read_text())) != payload_hash(config):
            raise RuntimeError("Resume configuration differs from immutable campaign_config.v1.json.")
    else:
        write_json_atomic(config_copy, config)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("sources") != current_source_hashes():
            raise RuntimeError(
                "Resume source hashes differ from the campaign manifest; start a new campaign ID."
            )
    else:
        seeds = derived_sample_seeds(config)
        write_json_atomic(
            manifest_path,
            {
                "campaign_id": campaign_id,
                "campaign": config["campaign"],
                "schema_version": config["schema_version"],
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
                "config_sha256": config_sha256(),
                "sources": current_source_hashes(),
                "git": git_metadata(),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "thread_environment": {
                        name: os.environ.get(name)
                        for name in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                        )
                    },
                },
                "derived_sample_seeds": seeds,
                "stages": {},
                "resource_history": [],
            },
        )
    return root


def update_manifest(
    root: Path,
    *,
    stage: str,
    status: str,
    cpu_list: str = "",
    details: dict[str, Any] | None = None,
) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_utc": utc_now(),
        **({"details": details} if details else {}),
    }
    if cpu_list:
        manifest.setdefault("resource_history", []).append(
            {"stage": stage, "cpu_list": cpu_list, "recorded_utc": utc_now()}
        )
    manifest["updated_utc"] = utc_now()
    write_json_atomic(path, manifest)


def _cpu_snapshot() -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        token = fields[0]
        if not token.startswith("cpu") or not token[3:].isdigit():
            continue
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        result[int(token[3:])] = (sum(values), idle)
    return result


def _cpu_topology(cpu: int) -> tuple[int, int, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    topology = base / "topology"
    package = int((topology / "physical_package_id").read_text())
    core = int((topology / "core_id").read_text())
    nodes = sorted(base.glob("node[0-9]*"))
    node = int(nodes[0].name[4:]) if nodes else package
    return node, package, core


def select_idle_cpu_ids(limit: int, *, allow_busy: bool = False) -> list[int]:
    config = load_config()
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
        if key not in one_per_core or row[0] < one_per_core[key][0]:
            one_per_core[key] = row
    threshold = float(config["parallel"]["idle_cpu_threshold_percent"])
    eligible = [
        row for row in one_per_core.values() if allow_busy or row[0] <= threshold
    ]
    by_node: dict[int, list[tuple[float, int]]] = {}
    for usage, node, cpu in eligible:
        by_node.setdefault(node, []).append((usage, cpu))
    choices = []
    for node, rows in by_node.items():
        rows.sort()
        if len(rows) >= int(limit):
            choices.append((sum(value[0] for value in rows[:limit]), node, rows))
    if not choices:
        return []
    _, _, selected_rows = min(choices, key=lambda value: (value[0], value[1]))
    return [int(cpu) for _, cpu in selected_rows[:limit]]


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
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
        raise ValueError("CPU list contains duplicate logical CPUs.")
    return cpus


def validate_cpu_ids(cpu_ids: list[int], required: int) -> list[int]:
    """Validate the locked one-physical-core, one-NUMA-node allocation."""

    if len(cpu_ids) < int(required):
        raise ValueError(f"At least {required} physical CPUs are required; got {cpu_ids}.")
    selected = [int(cpu) for cpu in cpu_ids[: int(required)]]
    available = os.sched_getaffinity(0)
    unavailable = [cpu for cpu in selected if cpu not in available]
    if unavailable:
        raise ValueError(f"Requested CPUs are outside this process affinity mask: {unavailable}.")
    topology = [_cpu_topology(cpu) for cpu in selected]
    physical = [(package, core) for _, package, core in topology]
    if len(set(physical)) != len(physical):
        raise ValueError("CPU list contains hyperthread siblings from the same physical core.")
    if load_config()["parallel"]["same_numa_node"]:
        nodes = {node for node, _, _ in topology}
        if len(nodes) != 1:
            raise ValueError(f"Locked campaign CPUs must lie on one NUMA node; got {sorted(nodes)}.")
    return selected


def pin_worker(cpu: int) -> None:
    os.sched_setaffinity(0, {int(cpu)})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def derived_sample_seeds(config: dict[str, Any]) -> dict[str, list[int]]:
    initializations = config["initializations"]
    samples = int(config["geometry"]["samples"])
    children = np.random.SeedSequence(int(config["root_seed"])).spawn(
        len(initializations) * samples
    )
    values = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
    return {
        spec["label"]: values[index * samples : (index + 1) * samples]
        for index, spec in enumerate(initializations)
    }


def initialization_spec(label: str) -> dict[str, Any]:
    for spec in load_config()["initializations"]:
        if spec["label"] == label:
            return dict(spec)
    raise KeyError(label)


def build_model() -> tuple[classA_U1FGTN, dict[str, int]]:
    geometry = load_config()["geometry"]
    started = time.perf_counter_ns()
    model = classA_U1FGTN(
        Nx=int(geometry["Nx"]),
        Ny=int(geometry["Ny"]),
        DW=bool(geometry["DW"]),
        nshell=int(geometry["nshell"]),
        filling_frac=float(geometry["filling_frac"]),
        alpha_1=float(geometry["alpha_1"]),
        alpha_2=float(geometry["alpha_2"]),
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_truncation=bool(geometry["dw_truncation"]),
    )
    model_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    model.construct_OW_projectors(
        nshell=int(geometry["nshell"]),
        DW=bool(geometry["DW"]),
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_truncation=bool(geometry["dw_truncation"]),
    )
    ow_ns = time.perf_counter_ns() - started
    return model, {
        "model_construction": int(model_ns),
        "ow_projector_construction": int(ow_ns),
    }


def common_run_kwargs(spec: dict[str, Any], initial_g: np.ndarray | None = None) -> dict[str, Any]:
    config = load_config()
    geometry = config["geometry"]
    protocol = config["protocol"]
    return {
        "cycles": int(geometry["cycles"]),
        "samples": 1,
        "init_mode": str(spec["init_mode"]),
        "G_init": initial_g,
        "sequence": str(protocol["sequence"]),
        "perfect_correction": bool(protocol["perfect_correction"]),
        "postselect": bool(protocol["postselect"]),
        "postselect_probability": float(protocol["postselect_probability"]),
        "meas_slab_only": bool(geometry["meas_slab_only"]),
        "physical_covariance_update": str(protocol["physical_covariance_update"]),
        "G_history": False,
        "save": False,
        "progress": False,
        "parallelize_samples": False,
        "trajectory_replay_probability_tol": float(
            config["acceptance"]["replay_probability_tolerance"]
        ),
    }


class TrajectoryRecorder:
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


def record_paths(root: Path, label: str, sample: int) -> tuple[Path, Path, Path]:
    directory = root / "raw/records" / label / f"sample_{sample:02d}"
    return directory / "record.json.gz", directory / "initial_state.npz", directory / "summary.json"


def _completed(paths: tuple[Path, ...], summary_path: Path, task_hash: str) -> bool:
    if not summary_path.exists() or not all(path.exists() for path in paths):
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    if summary.get("status") != "complete" or summary.get("task_hash") != task_hash:
        return False
    return all(sha256_file(path) == summary["files"].get(path.name) for path in paths)


def _record_worker(task: dict[str, Any]) -> dict[str, Any]:
    worker_started = time.perf_counter_ns()
    pin_worker(int(task["cpu"]))
    queue_wait_ns = worker_started - int(task["submitted_ns"])
    root = Path(task["root"])
    label = str(task["label"])
    sample = int(task["sample"])
    seed = int(task["seed"])
    spec = initialization_spec(label)
    record_path, state_path, summary_path = record_paths(root, label, sample)
    task_hash = str(task["task_hash"])
    if _completed((record_path, state_path), summary_path, task_hash):
        return json.loads(summary_path.read_text())

    model, setup_timing = build_model()
    recorder = TrajectoryRecorder()
    initial: list[np.ndarray] = []

    def cycle_observer(**payload: Any) -> None:
        if int(payload["cycle"]) == 0:
            initial.append(np.array(payload["G"], dtype=np.complex128, copy=True))

    started = time.perf_counter_ns()
    model.run_markov_circuit(
        **common_run_kwargs(spec),
        random_seed=seed,
        cycle_observer=cycle_observer,
        trajectory_weight_observer=recorder,
        timing_level="off",
    )
    record_generation_ns = time.perf_counter_ns() - started
    if len(initial) != 1:
        raise RuntimeError("Record generation did not capture exactly one cycle-0 state.")
    expected_sites = int(load_config()["geometry"]["cycles"]) * int(
        load_config()["geometry"]["Nx"]
    ) * int(load_config()["geometry"]["Ny"])
    if len(recorder.entries) != expected_sites:
        raise RuntimeError(
            f"Record contains {len(recorder.entries)} sites; expected {expected_sites}."
        )
    write_gzip_json_atomic(
        record_path,
        {
            "schema_version": 1,
            "label": label,
            "sample": sample,
            "seed": seed,
            "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
            "entries": recorder.entries,
        },
    )
    save_npz_atomic(state_path, G0=initial[0])
    summary = {
        "status": "complete",
        "task_hash": task_hash,
        "config_sha256": config_sha256(),
        "label": label,
        "sample": sample,
        "seed": seed,
        "cpu": int(task["cpu"]),
        "queue_wait_ns": int(queue_wait_ns),
        "record_generation_ns": int(record_generation_ns),
        "setup_timing_ns": setup_timing,
        "site_count": len(recorder.entries),
        "event_count": int(
            sum(len(entry["branch_events"]) for entry in recorder.entries)
        ),
        "final_log_weight": float(recorder.entries[-1]["cumulative_log_weight"]),
        "files": {
            record_path.name: sha256_file(record_path),
            state_path.name: sha256_file(state_path),
        },
    }
    write_json_atomic(summary_path, summary)
    return summary


def _binary_entropy(eigenvalues: np.ndarray) -> float:
    values = np.clip(np.real(np.asarray(eigenvalues)), 0.0, 1.0)
    selected = values[(values > 1e-14) & (values < 1.0 - 1e-14)]
    if selected.size == 0:
        return 0.0
    return float(
        -np.sum(selected * np.log(selected) + (1.0 - selected) * np.log1p(-selected))
    )


def half_region_rows() -> np.ndarray:
    geometry = load_config()["geometry"]
    shape = (2, int(geometry["Nx"]), int(geometry["Ny"]))
    mask = np.zeros(shape, dtype=bool)
    start = int(load_config()["half_region"]["y_start"])
    stop = int(load_config()["half_region"]["y_stop_exclusive"])
    mask[:, :, start:stop] = True
    return np.flatnonzero(mask.reshape(-1, order="F"))


def chern_partition_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact disk tri-partition used by ``real_space_chern_number``."""

    geometry = load_config()["geometry"]
    nx, ny = int(geometry["Nx"]), int(geometry["Ny"])
    radius = 0.4 * min(nx, ny)
    xref, yref = nx // 2, ny // 2
    masks = [np.zeros((nx, ny), dtype=bool) for _ in range(3)]
    radius_squared = radius * radius
    for dy in range(-int(np.floor(radius)), int(np.floor(radius)) + 1):
        y = yref + dy
        if y < 0 or y >= ny:
            continue
        max_dx = int(np.floor(np.sqrt(radius_squared - dy * dy)))
        for x in range(max(0, xref - max_dx), min(nx - 1, xref + max_dx) + 1):
            theta = float(np.mod(np.arctan2(dy, x - xref), 2.0 * np.pi))
            sector = 0 if theta < 2.0 * np.pi / 3.0 else 1 if theta < 4.0 * np.pi / 3.0 else 2
            masks[sector][x, y] = True

    def indices(mask: np.ndarray) -> np.ndarray:
        xs, ys = np.nonzero(mask)
        return np.sort(
            np.concatenate((2 * xs + 2 * nx * ys, 1 + 2 * xs + 2 * nx * ys))
        ).astype(np.int64)

    return tuple(indices(mask) for mask in masks)  # type: ignore[return-value]


def real_space_chern_from_frame(
    state: OccupiedFrameState,
    partitions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """Evaluate the disk Chern functional from frame-row overlaps only."""

    physical = state.physical_frame
    i_a, i_b, i_c = partitions

    def projector_block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # The canonical estimator uses P=C.conj().  With C=B B^dagger this is
        # B.conj() B.T, formed only on the requested partition blocks.
        return physical[left, :].conj() @ physical[right, :].T

    p_ca = projector_block(i_c, i_a)
    p_ab = projector_block(i_a, i_b)
    p_bc = projector_block(i_b, i_c)
    p_ac = projector_block(i_a, i_c)
    p_cb = projector_block(i_c, i_b)
    p_ba = projector_block(i_b, i_a)
    value = 12.0 * np.pi * 1j * (
        np.trace(p_ca @ p_ab @ p_bc) - np.trace(p_ac @ p_cb @ p_ba)
    )
    return float(np.real(value))


class CorrectnessCapture:
    def __init__(self, backend: str, checkpoint_cycles: set[int], model: classA_U1FGTN) -> None:
        self.backend = backend
        self.model = model
        self.chern_partitions = chern_partition_indices()
        self.checkpoint_cycles = checkpoint_cycles
        self.cycles: list[int] = []
        self.centered_history: list[np.ndarray] = []
        self.charge: list[float] = []
        self.global_entropy: list[float] = []
        self.regional_entropy: list[float] = []
        self.real_space_chern: list[float] = []
        self.gram_residual: list[float] = []
        self.rank: list[int] = []
        self.checkpoint_covariance: dict[int, np.ndarray] = {}
        self.checkpoint_frame: dict[int, np.ndarray] = {}
        self.observer_timing_ns: dict[str, int] = {
            "charge_observer": 0,
            "global_entropy_eigh_or_svd": 0,
            "regional_entropy_eigh_or_svd": 0,
            "real_space_chern": 0,
            "physical_covariance_reconstruction": 0,
            "checkpoint_copy": 0,
        }
        self.rows = half_region_rows()

    def _append_covariance(self, cycle: int, centered: np.ndarray) -> None:
        centered = np.asarray(centered, dtype=np.complex128)
        dimension = centered.shape[0]
        correlation = 0.5 * (centered + np.eye(dimension, dtype=np.complex128))
        self.cycles.append(int(cycle))
        self.centered_history.append(np.array(centered, copy=True))
        started = time.perf_counter_ns()
        self.charge.append(float(np.real(np.trace(correlation))))
        self.observer_timing_ns["charge_observer"] += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        self.global_entropy.append(_binary_entropy(np.linalg.eigvalsh(correlation)))
        self.observer_timing_ns["global_entropy_eigh_or_svd"] += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        restricted = correlation[np.ix_(self.rows, self.rows)]
        self.regional_entropy.append(_binary_entropy(np.linalg.eigvalsh(restricted)))
        self.observer_timing_ns["regional_entropy_eigh_or_svd"] += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        self.real_space_chern.append(float(np.real(self.model.real_space_chern_number(centered))))
        self.observer_timing_ns["real_space_chern"] += time.perf_counter_ns() - started
        self.gram_residual.append(np.nan)
        self.rank.append(-1)
        if cycle in self.checkpoint_cycles:
            started = time.perf_counter_ns()
            self.checkpoint_covariance[int(cycle)] = np.array(centered, copy=True)
            self.observer_timing_ns["checkpoint_copy"] += time.perf_counter_ns() - started

    def covariance_observer(self, **payload: Any) -> None:
        self._append_covariance(int(payload["cycle"]), payload["G"])

    def frame_observer(self, **payload: Any) -> None:
        state: OccupiedFrameState = payload["state"]
        cycle = int(payload["cycle"])
        started = time.perf_counter_ns()
        centered = state.centered_covariance()
        self.observer_timing_ns["physical_covariance_reconstruction"] += (
            time.perf_counter_ns() - started
        )
        self.cycles.append(cycle)
        self.centered_history.append(centered)
        started = time.perf_counter_ns()
        self.charge.append(state.regional_charge(slice(None)))
        self.observer_timing_ns["charge_observer"] += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        self.global_entropy.append(state.physical_entropy())
        self.observer_timing_ns["global_entropy_eigh_or_svd"] += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        self.regional_entropy.append(state.regional_entropy(self.rows))
        self.observer_timing_ns["regional_entropy_eigh_or_svd"] += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        self.real_space_chern.append(
            real_space_chern_from_frame(state, self.chern_partitions)
        )
        self.observer_timing_ns["real_space_chern"] += time.perf_counter_ns() - started
        self.gram_residual.append(state.gram_residual())
        self.rank.append(state.rank)
        if cycle in self.checkpoint_cycles:
            started = time.perf_counter_ns()
            self.checkpoint_covariance[cycle] = np.array(centered, copy=True)
            self.checkpoint_frame[cycle] = np.array(state.frame, copy=True)
            self.observer_timing_ns["checkpoint_copy"] += time.perf_counter_ns() - started


class EventSketchCapture:
    """Retain cheap deterministic state sketches to localize a failing event."""

    def __init__(self, backend: str, dimension: int, pair_count: int = 8) -> None:
        anchors = np.unique(
            np.linspace(0, int(dimension) - 1, int(pair_count), dtype=np.int64)
        )
        diagonal = [(int(index), int(index)) for index in anchors]
        off_diagonal = [
            (int(index), int((index + 1 + position) % dimension))
            for position, index in enumerate(anchors)
        ]
        self.row_pairs = diagonal + off_diagonal
        self.backend = str(backend)
        self.identities: list[tuple[int, int, str]] = []
        self.values: list[np.ndarray] = []
        self.elapsed_ns = 0

    def __call__(self, **payload: Any) -> None:
        started = time.perf_counter_ns()
        state = payload["state"]
        if isinstance(state, OccupiedFrameState):
            left = np.asarray([pair[0] for pair in self.row_pairs], dtype=np.int64)
            right = np.asarray([pair[1] for pair in self.row_pairs], dtype=np.int64)
            values = 2.0 * state.selected_correlators(left, right) - np.asarray(
                [1.0 if lrow == rrow else 0.0 for lrow, rrow in self.row_pairs],
                dtype=np.float64,
            )
        else:
            covariance = np.asarray(state, dtype=np.complex128)
            values = np.asarray(
                [covariance[left, right] for left, right in self.row_pairs],
                dtype=np.complex128,
            )
        self.identities.append(
            (int(payload["cycle"]), int(payload["site_id"]), str(payload["channel"]))
        )
        self.values.append(np.asarray(values, dtype=np.complex128))
        self.elapsed_ns += time.perf_counter_ns() - started


def _choi_projector_from_payload(payload: dict[str, Any]) -> np.ndarray:
    sigma_ll = np.asarray(payload["sigma_ll"])[0]
    sigma_lr = np.asarray(payload["sigma_lr"])[0]
    sigma_rr = np.asarray(payload["sigma_rr"])[0]
    dimension = sigma_ll.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    return np.block(
        [
            [0.5 * (sigma_ll + identity), 0.5 * sigma_lr],
            [0.5 * sigma_lr.conj().T, 0.5 * (sigma_rr + identity)],
        ]
    )


def _reference_choi_projector(dimension: int) -> np.ndarray:
    identity = np.eye(dimension, dtype=np.complex128) / np.sqrt(2.0)
    frame = np.vstack((identity, identity))
    return frame @ frame.conj().T


def _branch_cycle_metrics(
    reference: list[dict[str, Any]], observed: list[dict[str, Any]], cycles: int
) -> dict[str, np.ndarray | int | dict[str, Any] | None]:
    if len(reference) != len(observed):
        return {
            "branch_disagreement_count": 1,
            "first_disagreement": {"reason": "site_count", "reference": len(reference), "observed": len(observed)},
            "max_probability_error": np.full(cycles + 1, np.inf),
            "minimum_selected_probability": np.zeros(cycles + 1),
            "cumulative_log_weight": np.full(cycles + 1, np.nan),
        }
    maximum = np.zeros(cycles + 1, dtype=np.float64)
    minimum = np.ones(cycles + 1, dtype=np.float64)
    cumulative = np.zeros(cycles + 1, dtype=np.float64)
    disagreements = 0
    first = None
    for site_position, (expected_site, actual_site) in enumerate(zip(reference, observed)):
        cycle = int(expected_site["cycle"])
        if (
            int(actual_site["cycle"]) != cycle
            or int(actual_site["site_id"]) != int(expected_site["site_id"])
        ):
            disagreements += 1
            if first is None:
                first = {"reason": "site_identity", "site_position": site_position}
            continue
        expected_events = expected_site["branch_events"]
        actual_events = actual_site["branch_events"]
        if len(expected_events) != len(actual_events):
            disagreements += 1
            if first is None:
                first = {"reason": "event_count", "site_position": site_position}
            continue
        for event_position, (expected, actual) in enumerate(zip(expected_events, actual_events)):
            identity_fields = ("kind", "channel")
            if any(str(expected.get(key)) != str(actual.get(key)) for key in identity_fields):
                disagreements += 1
                if first is None:
                    first = {
                        "reason": "event_identity",
                        "site_position": site_position,
                        "event_position": event_position,
                    }
            if expected.get("kind") == "measurement":
                if bool(expected["outcome_occupied"]) != bool(actual["outcome_occupied"]):
                    disagreements += 1
                    if first is None:
                        first = {
                            "reason": "measurement_outcome",
                            "site_position": site_position,
                            "event_position": event_position,
                        }
                probability = float(actual["probability"])
                error = abs(probability - float(expected["probability"]))
                maximum[cycle] = max(maximum[cycle], error)
                selected = probability if bool(actual["outcome_occupied"]) else 1.0 - probability
                minimum[cycle] = min(minimum[cycle], selected)
            elif expected.get("kind") == "correction":
                if bool(expected["target_occupied"]) != bool(actual["target_occupied"]):
                    disagreements += 1
                    if first is None:
                        first = {
                            "reason": "correction_target",
                            "site_position": site_position,
                            "event_position": event_position,
                        }
        cumulative[cycle] = float(actual_site["cumulative_log_weight"])
    minimum[0] = 1.0
    return {
        "branch_disagreement_count": int(disagreements),
        "first_disagreement": first,
        "max_probability_error": maximum,
        "minimum_selected_probability": minimum,
        "cumulative_log_weight": cumulative,
    }


def _native_state_bytes(result: dict[str, Any], backend: str) -> int:
    if backend == "frame":
        return int(np.asarray(result["native_final"]["frame"]).nbytes)
    return int(np.asarray(result["G_final"]).nbytes)


def _run_correctness_backend(
    *,
    backend: str,
    spec: dict[str, Any],
    initial_g: np.ndarray,
    record: list[dict[str, Any]],
    seed: int,
    checkpoint_cycles: set[int],
    dense_choi_audit: bool,
) -> dict[str, Any]:
    model, setup_timing = build_model()
    capture = CorrectnessCapture(backend, checkpoint_cycles, model)
    event_sketch = EventSketchCapture(backend, initial_g.shape[0])
    replay = TrajectoryRecorder()
    choi_checkpoints: dict[int, np.ndarray] = {}
    if spec["label"] == "maxmix" and backend == "covariance" and dense_choi_audit:
        choi_checkpoints[0] = _reference_choi_projector(initial_g.shape[0])

    def choi_observer(**payload: Any) -> None:
        choi_checkpoints[int(payload["cycle"])] = _choi_projector_from_payload(payload)

    kwargs = common_run_kwargs(spec, initial_g)
    kwargs.update(
        {
            "random_seed": int(seed),
            "trajectory_replay": record,
            "trajectory_weight_observer": replay,
            "native_event_observer": event_sketch,
            "timing_level": "detailed",
        }
    )
    if backend == "covariance":
        kwargs["cycle_observer"] = capture.covariance_observer
        if spec["label"] == "maxmix" and dense_choi_audit:
            kwargs.update(
                {
                    "track_choi": True,
                    "choi_observer": choi_observer,
                    "choi_observer_cycles": sorted(checkpoint_cycles - {0}),
                }
            )
    else:
        kwargs.update(
            {
                "state_representation": str(spec["frame_representation"]),
                "return_native_state": True,
                "native_cycle_observer": capture.frame_observer,
            }
        )
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    result = model.run_markov_circuit(**kwargs)
    cpu_ns = time.process_time_ns() - cpu_started
    wall_ns = time.perf_counter_ns() - wall_started
    return {
        "backend": backend,
        "capture": capture,
        "event_sketch": event_sketch,
        "replay": replay.entries,
        "choi_checkpoints": choi_checkpoints,
        "setup_timing_ns": setup_timing,
        "wall_ns": int(wall_ns),
        "cpu_ns": int(cpu_ns),
        "timing": result.get("timing"),
        "native_state_bytes": _native_state_bytes(result, backend),
    }


def correctness_paths(root: Path, label: str, sample: int) -> tuple[Path, Path]:
    directory = root / "raw/correctness" / label / f"sample_{sample:02d}"
    return directory / "cycle_data.npz", directory / "summary.json"


def _first_failure_dump(
    output: Path,
    *,
    original: CorrectnessCapture,
    frame: CorrectnessCapture,
    first_cycle: int,
) -> None:
    save_npz_atomic(
        output,
        cycle=np.asarray(first_cycle, dtype=np.int64),
        covariance_state=original.centered_history[first_cycle],
        frame_reconstructed_state=frame.centered_history[first_cycle],
        frame=frame.checkpoint_frame.get(first_cycle, np.empty((0, 0), dtype=np.complex128)),
    )


class _StopAfterCapturedEvent(RuntimeError):
    pass


def _capture_exact_event_state(
    *,
    backend: str,
    spec: dict[str, Any],
    initial_g: np.ndarray,
    record: list[dict[str, Any]],
    seed: int,
    target: tuple[int, int, str],
) -> dict[str, np.ndarray]:
    """Replay one backend until a target channel and copy its exact post-event state."""

    model, _ = build_model()
    captured: dict[str, np.ndarray] = {}

    def observer(**payload: Any) -> None:
        identity = (int(payload["cycle"]), int(payload["site_id"]), str(payload["channel"]))
        if identity != target:
            return
        state = payload["state"]
        if isinstance(state, OccupiedFrameState):
            captured["covariance"] = state.centered_covariance()
            captured["frame"] = np.array(state.frame, copy=True)
        else:
            captured["covariance"] = np.array(state, dtype=np.complex128, copy=True)
        raise _StopAfterCapturedEvent

    kwargs = common_run_kwargs(spec, initial_g)
    kwargs.update(
        {
            "random_seed": int(seed),
            "trajectory_replay": record,
            "native_event_observer": observer,
            "timing_level": "off",
        }
    )
    if backend == "frame":
        kwargs.update(
            {
                "state_representation": str(spec["frame_representation"]),
                "return_native_state": True,
            }
        )
    try:
        model.run_markov_circuit(**kwargs)
    except _StopAfterCapturedEvent:
        pass
    if "covariance" not in captured:
        raise RuntimeError(f"Failure replay did not encounter target event {target!r}.")
    return captured


def _correctness_worker(task: dict[str, Any]) -> dict[str, Any]:
    worker_started = time.perf_counter_ns()
    pin_worker(int(task["cpu"]))
    queue_wait_ns = worker_started - int(task["submitted_ns"])
    root = Path(task["root"])
    label = str(task["label"])
    sample = int(task["sample"])
    seed = int(task["seed"])
    spec = initialization_spec(label)
    data_path, summary_path = correctness_paths(root, label, sample)
    task_hash = str(task["task_hash"])
    if _completed((data_path,), summary_path, task_hash):
        return json.loads(summary_path.read_text())

    record_path, state_path, _ = record_paths(root, label, sample)
    started = time.perf_counter_ns()
    record_payload = read_gzip_json(record_path)
    with np.load(state_path) as payload:
        initial_g = np.array(payload["G0"], dtype=np.complex128, copy=True)
    record_load_decode_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    initial_g = np.array(initial_g, dtype=np.complex128, copy=True, order="C")
    state_copy_for_replay_ns = time.perf_counter_ns() - started
    record = list(record_payload["entries"])
    checkpoint_cycles = set(int(value) for value in load_config()["checkpoint_cycles"])
    dense_choi_audit = label == "maxmix" and sample in set(
        int(value) for value in load_config()["dense_choi_audit_samples"]
    )
    order = ("covariance", "frame") if sample % 2 == 0 else ("frame", "covariance")
    barrier = task.get("barrier")
    outputs: dict[str, dict[str, Any]] = {}
    barrier_wait_ns = 0
    paired_started = time.perf_counter_ns()
    for backend in order:
        if barrier is not None:
            wait_started = time.perf_counter_ns()
            barrier.wait()
            barrier_wait_ns += time.perf_counter_ns() - wait_started
        outputs[backend] = _run_correctness_backend(
            backend=backend,
            spec=spec,
            initial_g=initial_g,
            record=record,
            seed=seed,
            checkpoint_cycles=checkpoint_cycles,
            dense_choi_audit=dense_choi_audit,
        )
    paired_task_ns = time.perf_counter_ns() - paired_started
    covariance = outputs["covariance"]
    frame = outputs["frame"]
    cov_capture: CorrectnessCapture = covariance["capture"]
    frame_capture: CorrectnessCapture = frame["capture"]
    cycles = int(load_config()["geometry"]["cycles"])
    expected_cycles = list(range(cycles + 1))
    if cov_capture.cycles != expected_cycles or frame_capture.cycles != expected_cycles:
        raise RuntimeError("Correctness observers did not retain the complete cycle coordinate 0..T.")

    comparison_started = time.perf_counter_ns()
    covariance_errors = np.zeros(cycles + 1, dtype=np.float64)
    covariance_max_errors = np.zeros(cycles + 1, dtype=np.float64)
    for cycle in expected_cycles:
        left = cov_capture.centered_history[cycle]
        right = frame_capture.centered_history[cycle]
        difference = left - right
        covariance_errors[cycle] = np.linalg.norm(difference) / max(np.linalg.norm(left), 1.0)
        covariance_max_errors[cycle] = np.max(np.abs(difference))
    charge_error = np.abs(np.asarray(cov_capture.charge) - np.asarray(frame_capture.charge))
    global_entropy_error = np.abs(
        np.asarray(cov_capture.global_entropy) - np.asarray(frame_capture.global_entropy)
    )
    regional_entropy_error = np.abs(
        np.asarray(cov_capture.regional_entropy) - np.asarray(frame_capture.regional_entropy)
    )
    chern_error = np.abs(
        np.asarray(cov_capture.real_space_chern)
        - np.asarray(frame_capture.real_space_chern)
    )
    covariance_branch = _branch_cycle_metrics(record, covariance["replay"], cycles)
    frame_branch = _branch_cycle_metrics(record, frame["replay"], cycles)
    log_weight_error = np.abs(
        np.asarray(covariance_branch["cumulative_log_weight"])
        - np.asarray(frame_branch["cumulative_log_weight"])
    )
    branch_probability_error = np.maximum(
        np.asarray(covariance_branch["max_probability_error"]),
        np.asarray(frame_branch["max_probability_error"]),
    )
    covariance_sketch: EventSketchCapture = covariance["event_sketch"]
    frame_sketch: EventSketchCapture = frame["event_sketch"]
    event_identity_disagreement = covariance_sketch.identities != frame_sketch.identities
    if event_identity_disagreement:
        event_sketch_error = np.full(
            max(len(covariance_sketch.values), len(frame_sketch.values)), np.inf
        )
    else:
        event_sketch_error = np.max(
            np.abs(
                np.asarray(covariance_sketch.values, dtype=np.complex128)
                - np.asarray(frame_sketch.values, dtype=np.complex128)
            ),
            axis=1,
        )
    event_sketch_error_by_cycle = np.zeros(cycles + 1, dtype=np.float64)
    if not event_identity_disagreement:
        for identity, error in zip(covariance_sketch.identities, event_sketch_error):
            event_sketch_error_by_cycle[identity[0]] = max(
                event_sketch_error_by_cycle[identity[0]], float(error)
            )
    choi_error = np.full(cycles + 1, np.nan, dtype=np.float64)
    if label == "maxmix" and covariance["choi_checkpoints"]:
        for cycle in checkpoint_cycles:
            dense = covariance["choi_checkpoints"][cycle]
            frame_matrix = frame_capture.checkpoint_frame[cycle]
            projected = frame_matrix @ frame_matrix.conj().T
            choi_error[cycle] = np.linalg.norm(dense - projected) / max(
                np.linalg.norm(dense), 1.0
            )
    acceptance = load_config()["acceptance"]
    failures = []
    checks = {
        "covariance": float(max(np.max(covariance_errors), np.max(covariance_max_errors))),
        "branch_probability": float(np.max(branch_probability_error)),
        "log_weight": float(np.max(log_weight_error)),
        "entropy": float(max(np.max(global_entropy_error), np.max(regional_entropy_error))),
        "chern_observable": float(np.max(chern_error)),
        "gram": float(np.max(frame_capture.gram_residual)),
        "choi": float(np.nanmax(choi_error)) if np.any(np.isfinite(choi_error)) else 0.0,
    }
    for name, observed in checks.items():
        acceptance_name = "covariance" if name == "chern_observable" else name
        if observed > float(acceptance[f"{acceptance_name}_hard"]):
            failures.append(
                {
                    "gate": name,
                    "observed": observed,
                    "hard": acceptance[f"{acceptance_name}_hard"],
                }
            )
    disagreement_count = int(covariance_branch["branch_disagreement_count"]) + int(
        frame_branch["branch_disagreement_count"]
    )
    if disagreement_count:
        failures.append({"gate": "branch_identity", "observed": disagreement_count, "hard": 0})
    if event_identity_disagreement:
        failures.append({"gate": "event_identity", "observed": True, "hard": False})
    fallback_count = int(
        (covariance["timing"] or {}).get("counts", {}).get(
            "regularized_dense_fallback_count", 0
        )
    )
    if fallback_count:
        failures.append({"gate": "dense_fallback", "observed": fallback_count, "hard": 0})
    ranks = np.asarray(frame_capture.rank, dtype=np.int64)
    if np.any(ranks < 0):
        failures.append({"gate": "frame_rank", "observed": ranks.tolist(), "hard": "nonnegative"})
    charge_frame = np.asarray(frame_capture.charge, dtype=np.float64)
    rank_charge_residual = (
        np.abs(ranks.astype(np.float64) - charge_frame)
        if label == "random_pure"
        else np.full(cycles + 1, np.nan, dtype=np.float64)
    )
    if label == "random_pure" and np.max(rank_charge_residual) > 1e-8:
        failures.append(
            {
                "gate": "frame_rank_charge_consistency",
                "observed": float(np.max(rank_charge_residual)),
                "hard": 1e-8,
            }
        )
    frame_timing = frame.get("timing") or {}
    per_cycle_counts = frame_timing.get("per_cycle_counts", {})
    expected_ranks = np.empty(cycles + 1, dtype=np.int64)
    expected_ranks[0] = ranks[0]
    for cycle in range(1, cycles + 1):
        counts = per_cycle_counts.get(str(cycle), {})
        expected_ranks[cycle] = (
            expected_ranks[cycle - 1]
            + int(counts.get("gain_count", 0))
            - int(counts.get("loss_count", 0))
        )
    rank_bookkeeping_residual = np.abs(ranks - expected_ranks)
    if np.any(rank_bookkeeping_residual):
        failures.append(
            {
                "gate": "frame_rank_word_bookkeeping",
                "observed": rank_bookkeeping_residual.tolist(),
                "hard": 0,
            }
        )
    correctness_comparison_ns = time.perf_counter_ns() - comparison_started

    checkpoint_arrays: dict[str, np.ndarray] = {}
    for cycle in sorted(checkpoint_cycles):
        checkpoint_arrays[f"covariance_G_cycle_{cycle}"] = cov_capture.checkpoint_covariance[cycle]
        checkpoint_arrays[f"frame_G_cycle_{cycle}"] = frame_capture.checkpoint_covariance[cycle]
        checkpoint_arrays[f"frame_cycle_{cycle}"] = frame_capture.checkpoint_frame[cycle]
    serialization_started = time.perf_counter_ns()
    save_npz_atomic(
        data_path,
        cycles=np.arange(cycles + 1, dtype=np.int64),
        covariance_relative_error=covariance_errors,
        covariance_max_error=covariance_max_errors,
        charge_covariance=np.asarray(cov_capture.charge),
        charge_frame=np.asarray(frame_capture.charge),
        charge_error=charge_error,
        global_entropy_covariance=np.asarray(cov_capture.global_entropy),
        global_entropy_frame=np.asarray(frame_capture.global_entropy),
        global_entropy_error=global_entropy_error,
        regional_entropy_covariance=np.asarray(cov_capture.regional_entropy),
        regional_entropy_frame=np.asarray(frame_capture.regional_entropy),
        regional_entropy_error=regional_entropy_error,
        real_space_chern_covariance=np.asarray(cov_capture.real_space_chern),
        real_space_chern_frame=np.asarray(frame_capture.real_space_chern),
        real_space_chern_error=chern_error,
        frame_gram_residual=np.asarray(frame_capture.gram_residual),
        frame_rank=ranks,
        frame_rank_expected_from_words=expected_ranks,
        frame_rank_word_residual=rank_bookkeeping_residual,
        frame_rank_charge_residual=rank_charge_residual,
        event_sketch_error_by_cycle=event_sketch_error_by_cycle,
        branch_probability_error=branch_probability_error,
        log_weight_covariance=np.asarray(covariance_branch["cumulative_log_weight"]),
        log_weight_frame=np.asarray(frame_branch["cumulative_log_weight"]),
        log_weight_error=log_weight_error,
        minimum_selected_probability_covariance=np.asarray(
            covariance_branch["minimum_selected_probability"]
        ),
        minimum_selected_probability_frame=np.asarray(
            frame_branch["minimum_selected_probability"]
        ),
        choi_relative_error=choi_error,
        **checkpoint_arrays,
    )
    checkpoint_serialization_ns = time.perf_counter_ns() - serialization_started
    failure_path: Path | None = None
    if failures:
        failing_arrays = {
            "covariance": np.maximum(covariance_errors, covariance_max_errors),
            "branch_probability": branch_probability_error,
            "log_weight": log_weight_error,
            "entropy": np.maximum(global_entropy_error, regional_entropy_error),
            "chern_observable": chern_error,
            "gram": np.asarray(frame_capture.gram_residual),
            "choi": np.nan_to_num(choi_error, nan=0.0),
        }
        hard = {
            name: float(
                acceptance[
                    f"{'covariance' if name == 'chern_observable' else name}_hard"
                ]
            )
            for name in (
                "covariance",
                "branch_probability",
                "log_weight",
                "entropy",
                "chern_observable",
                "gram",
                "choi",
            )
        }
        first_cycles = [
            int(np.flatnonzero(values > hard[name])[0])
            for name, values in failing_arrays.items()
            if np.any(values > hard[name])
        ]
        first_cycle = min(first_cycles) if first_cycles else 0
        target_event: tuple[int, int, str] | None = None
        branch_first = covariance_branch["first_disagreement"] or frame_branch["first_disagreement"]
        if isinstance(branch_first, dict) and "site_position" in branch_first:
            site_entry = record[int(branch_first["site_position"])]
            branch_events = site_entry.get("branch_events", ())
            event_position = min(int(branch_first.get("event_position", 0)), len(branch_events) - 1)
            channel = str(branch_events[event_position]["channel"])
            target_event = (int(site_entry["cycle"]), int(site_entry["site_id"]), channel)
        if target_event is None and not event_identity_disagreement and event_sketch_error.size:
            locations = np.flatnonzero(event_sketch_error > 1e-10)
            if locations.size:
                target_event = covariance_sketch.identities[int(locations[0])]
        if target_event is None:
            cycle_entries = [entry for entry in record if int(entry["cycle"]) == first_cycle]
            if cycle_entries:
                entry = cycle_entries[-1]
                target_event = (
                    int(entry["cycle"]),
                    int(entry["site_id"]),
                    str(entry["branch_events"][-1]["channel"]),
                )
        failure_path = data_path.with_name("first_failure_states.npz")
        if target_event is not None:
            covariance_event = _capture_exact_event_state(
                backend="covariance",
                spec=spec,
                initial_g=initial_g,
                record=record,
                seed=seed,
                target=target_event,
            )
            frame_event = _capture_exact_event_state(
                backend="frame",
                spec=spec,
                initial_g=initial_g,
                record=record,
                seed=seed,
                target=target_event,
            )
            save_npz_atomic(
                failure_path,
                cycle=np.asarray(target_event[0], dtype=np.int64),
                site_id=np.asarray(target_event[1], dtype=np.int64),
                channel=np.asarray(target_event[2]),
                covariance_state=covariance_event["covariance"],
                frame_reconstructed_state=frame_event["covariance"],
                frame=frame_event["frame"],
            )
        else:
            _first_failure_dump(
                failure_path,
                original=cov_capture,
                frame=frame_capture,
                first_cycle=first_cycle,
            )
    summary = {
        "status": "complete",
        "gate_passed": not failures,
        "failures": failures,
        "checks": checks,
        "task_hash": task_hash,
        "config_sha256": config_sha256(),
        "record_sha256": sha256_file(record_path),
        "label": label,
        "sample": sample,
        "seed": seed,
        "cpu": int(task["cpu"]),
        "backend_order": list(order),
        "queue_wait_ns": int(queue_wait_ns),
        "barrier_wait_ns": int(barrier_wait_ns),
        "paired_task_total_ns": int(paired_task_ns),
        "record_load_decode_ns": int(record_load_decode_ns),
        "state_copy_for_replay_ns": int(state_copy_for_replay_ns),
        "correctness_comparison_ns": int(correctness_comparison_ns),
        "checkpoint_serialization_ns": int(checkpoint_serialization_ns),
        "branch_disagreement_count": disagreement_count,
        "first_branch_disagreement": covariance_branch["first_disagreement"] or frame_branch["first_disagreement"],
        "regularized_dense_fallback_count": fallback_count,
        "dense_choi_audit": bool(dense_choi_audit),
        "backends": {
            backend: {
                "wall_ns": int(outputs[backend]["wall_ns"]),
                "cpu_ns": int(outputs[backend]["cpu_ns"]),
                "native_state_bytes": int(outputs[backend]["native_state_bytes"]),
                "setup_timing_ns": outputs[backend]["setup_timing_ns"],
                "observer_timing_ns": outputs[backend]["capture"].observer_timing_ns,
                "event_sketch_observer_ns": int(outputs[backend]["event_sketch"].elapsed_ns),
                "timing": outputs[backend]["timing"],
            }
            for backend in ("covariance", "frame")
        },
        "peak_rss_kib_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "files": {
            data_path.name: sha256_file(data_path),
            **(
                {failure_path.name: sha256_file(failure_path)}
                if failure_path is not None and failure_path.exists()
                else {}
            ),
        },
    }
    output_started = time.perf_counter_ns()
    write_json_atomic(summary_path, summary)
    output_write_ns = time.perf_counter_ns() - output_started
    # The summary itself cannot contain its own final hash, but can retain the
    # measured atomic-write latency by a second atomic replacement.
    summary["output_write_ns"] = int(output_write_ns)
    write_json_atomic(summary_path, summary)
    return summary


def _warmup_kernels() -> None:
    rng = np.random.default_rng(101)
    raw = rng.standard_normal((16, 8)) + 1j * rng.standard_normal((16, 8))
    frame, _ = np.linalg.qr(raw, mode="reduced")
    orbital = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    orbital /= np.linalg.norm(orbital)
    state = OccupiedFrameState(
        frame,
        representation="physical_frame",
        physical_dimension=16,
    )
    state.project_occupied_local(np.arange(16), orbital)
    centered = 2.0 * frame @ frame.conj().T - np.eye(16)
    chi = orbital[:4] / np.linalg.norm(orbital[:4])
    model = classA_U1FGTN(2, 2, DW=False, nshell=1, alpha_1=1, alpha_2=1)
    model._physical_rank1_resolvent_action(centered[:4, :4], chi, centered[:4], 1.0)


def benchmark_paths(root: Path, label: str, sample: int) -> tuple[Path, Path]:
    directory = root / "raw/benchmark" / label / f"sample_{sample:02d}"
    return directory / "timing.npz", directory / "summary.json"


def _run_benchmark_backend(
    *, backend: str, spec: dict[str, Any], initial_g: np.ndarray, record: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    model, setup_timing = build_model()
    kwargs = common_run_kwargs(spec, initial_g)
    kwargs.update(
        {
            "random_seed": int(seed),
            "trajectory_replay": record,
            "timing_level": "coarse",
        }
    )
    if backend == "frame":
        kwargs.update(
            {
                "state_representation": str(spec["frame_representation"]),
                "return_native_state": True,
            }
        )
    cpu_started = time.process_time_ns()
    wall_started = time.perf_counter_ns()
    result = model.run_markov_circuit(**kwargs)
    wall_ns = time.perf_counter_ns() - wall_started
    cpu_ns = time.process_time_ns() - cpu_started
    update_ns = int((result.get("timing") or {}).get("total_ns", {}).get("trajectory_total", wall_ns))
    return {
        "backend": backend,
        "wall_ns": int(wall_ns),
        "cpu_ns": int(cpu_ns),
        "update_ns": update_ns,
        "setup_timing_ns": setup_timing,
        "native_state_bytes": _native_state_bytes(result, backend),
        "timing": result.get("timing"),
    }


def _benchmark_worker(task: dict[str, Any]) -> dict[str, Any]:
    worker_started = time.perf_counter_ns()
    pin_worker(int(task["cpu"]))
    queue_wait_ns = worker_started - int(task["submitted_ns"])
    root = Path(task["root"])
    label = str(task["label"])
    sample = int(task["sample"])
    seed = int(task["seed"])
    spec = initialization_spec(label)
    data_path, summary_path = benchmark_paths(root, label, sample)
    task_hash = str(task["task_hash"])
    if _completed((data_path,), summary_path, task_hash):
        return json.loads(summary_path.read_text())
    record_path, state_path, _ = record_paths(root, label, sample)
    started = time.perf_counter_ns()
    record = list(read_gzip_json(record_path)["entries"])
    with np.load(state_path) as payload:
        initial_g = np.array(payload["G0"], dtype=np.complex128, copy=True)
    record_load_decode_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    initial_g = np.array(initial_g, dtype=np.complex128, copy=True, order="C")
    state_copy_for_replay_ns = time.perf_counter_ns() - started
    warmup_started = time.perf_counter_ns()
    _warmup_kernels()
    worker_import_warmup_ns = time.perf_counter_ns() - warmup_started
    order = ("covariance", "frame") if sample % 2 == 0 else ("frame", "covariance")
    outputs: dict[str, dict[str, Any]] = {}
    barrier = task.get("barrier")
    barrier_wait_ns = 0
    paired_started = time.perf_counter_ns()
    for backend in order:
        if barrier is not None:
            wait_started = time.perf_counter_ns()
            barrier.wait()
            barrier_wait_ns += time.perf_counter_ns() - wait_started
        outputs[backend] = _run_benchmark_backend(
            backend=backend,
            spec=spec,
            initial_g=initial_g,
            record=record,
            seed=seed,
        )
    paired_task_ns = time.perf_counter_ns() - paired_started
    save_npz_atomic(
        data_path,
        backend=np.asarray(["covariance", "frame"]),
        wall_ns=np.asarray([outputs[name]["wall_ns"] for name in ("covariance", "frame")], dtype=np.int64),
        cpu_ns=np.asarray([outputs[name]["cpu_ns"] for name in ("covariance", "frame")], dtype=np.int64),
        update_ns=np.asarray([outputs[name]["update_ns"] for name in ("covariance", "frame")], dtype=np.int64),
        native_state_bytes=np.asarray(
            [outputs[name]["native_state_bytes"] for name in ("covariance", "frame")], dtype=np.int64
        ),
    )
    summary = {
        "status": "complete",
        "task_hash": task_hash,
        "config_sha256": config_sha256(),
        "record_sha256": sha256_file(record_path),
        "label": label,
        "sample": sample,
        "seed": seed,
        "cpu": int(task["cpu"]),
        "backend_order": list(order),
        "queue_wait_ns": int(queue_wait_ns),
        "barrier_wait_ns": int(barrier_wait_ns),
        "paired_task_total_ns": int(paired_task_ns),
        "worker_import_warmup_ns": int(worker_import_warmup_ns),
        "record_load_decode_ns": int(record_load_decode_ns),
        "state_copy_for_replay_ns": int(state_copy_for_replay_ns),
        "backends": outputs,
        "peak_rss_kib_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "files": {data_path.name: sha256_file(data_path)},
    }
    output_started = time.perf_counter_ns()
    write_json_atomic(summary_path, summary)
    output_write_ns = time.perf_counter_ns() - output_started
    summary["output_write_ns"] = int(output_write_ns)
    write_json_atomic(summary_path, summary)
    return summary


def _tasks(root: Path, stage: str, label: str, cpu_ids: list[int]) -> list[dict[str, Any]]:
    config = load_config()
    seeds = derived_sample_seeds(config)[label]
    samples = int(config["geometry"]["samples"])
    tasks = []
    for sample in range(samples):
        task_core = {
            "stage": stage,
            "label": label,
            "sample": sample,
            "seed": int(seeds[sample]),
            "config_sha256": config_sha256(),
        }
        tasks.append(
            {
                **task_core,
                "task_hash": payload_hash(task_core),
                "root": str(root),
                "cpu": int(cpu_ids[sample % len(cpu_ids)]),
            }
        )
    return tasks


def _pending_tasks(root: Path, stage: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = []
    for task in tasks:
        label, sample = str(task["label"]), int(task["sample"])
        if stage == "records":
            record_path, state_path, summary = record_paths(root, label, sample)
            complete = _completed((record_path, state_path), summary, str(task["task_hash"]))
        elif stage == "correctness":
            data, summary = correctness_paths(root, label, sample)
            complete = _completed((data,), summary, str(task["task_hash"]))
        elif stage == "benchmark":
            data, summary = benchmark_paths(root, label, sample)
            complete = _completed((data,), summary, str(task["task_hash"]))
        else:
            raise KeyError(stage)
        if not complete:
            pending.append(task)
    return pending


def _run_parallel_stage(
    tasks: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    use_barrier: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not tasks:
        return [], {"pool_startup": 0, "parent_result_collection": 0, "stage_total": 0}
    manager = multiprocessing.Manager() if use_barrier else None
    barrier = manager.Barrier(len(tasks)) if manager is not None else None
    submission_started = time.perf_counter_ns()
    results = []
    collection_ns = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            futures = []
            for task in tasks:
                payload = dict(task)
                payload["submitted_ns"] = time.perf_counter_ns()
                if barrier is not None:
                    payload["barrier"] = barrier
                futures.append(pool.submit(worker, payload))
            pool_startup_ns = time.perf_counter_ns() - submission_started
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                started = time.perf_counter_ns()
                row = future.result()
                collection_ns += time.perf_counter_ns() - started
                results.append(row)
                print(
                    f"[{row['label']}] {index}/{len(futures)} sample={row['sample']} cpu={row['cpu']}",
                    flush=True,
                )
    finally:
        if manager is not None:
            manager.shutdown()
    return results, {
        "pool_startup": int(pool_startup_ns),
        "parent_result_collection": int(collection_ns),
        "stage_total": int(time.perf_counter_ns() - submission_started),
    }


def stage_preflight(root: Path, cpu_list: str) -> bool:
    update_manifest(root, stage="preflight", status="running", cpu_list=cpu_list)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_occupied_frame.py",
        "tests/test_occupied_frame_campaign.py",
        "tests/test_flux_cocycle_replay.py",
        "tests/test_tangent_cocycle.py",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    (root / "logs/preflight.log").write_text(result.stdout + "\n" + result.stderr)
    passed = result.returncode == 0
    write_json_atomic(
        root / "status/preflight.json",
        {"status": "complete" if passed else "failed", "returncode": result.returncode, "command": command},
    )
    update_manifest(
        root,
        stage="preflight",
        status="complete" if passed else "failed",
        details={"returncode": result.returncode},
    )
    return passed


def stage_records(root: Path, cpu_ids: list[int]) -> None:
    cpu_list = ",".join(map(str, cpu_ids))
    update_manifest(root, stage="records", status="running", cpu_list=cpu_list)
    stage_started = time.perf_counter_ns()
    all_rows = []
    parallel_timings = {}
    for spec in load_config()["initializations"]:
        label = str(spec["label"])
        tasks = _pending_tasks(root, "records", _tasks(root, "records", label, cpu_ids))
        rows, timing = _run_parallel_stage(tasks, _record_worker, use_barrier=False)
        all_rows.extend(rows)
        parallel_timings[label] = timing
    expected = len(load_config()["initializations"]) * int(load_config()["geometry"]["samples"])
    summaries = list((root / "raw/records").glob("*/sample_*/summary.json"))
    if len(summaries) != expected:
        raise RuntimeError(f"Record stage has {len(summaries)} of {expected} completed samples.")
    details = {
        "records": expected,
        "parallel_timing_ns": parallel_timings,
        "stage_total_ns": int(time.perf_counter_ns() - stage_started),
    }
    write_json_atomic(root / "status/records.json", {"status": "complete", **details})
    update_manifest(root, stage="records", status="complete", details=details)


def evaluate_uniform_insulator_gate(root: Path) -> dict[str, Any]:
    """Evaluate topology and purification convergence across all four lanes."""

    config = load_config()
    geometry = config["geometry"]
    gate = config["physics_gate"]
    if bool(geometry["DW"]):
        raise RuntimeError("The uniform-insulator physics gate requires DW=False.")
    model, _ = build_model()
    alpha = np.full(
        (int(geometry["Nx"]), int(geometry["Ny"])),
        float(geometry["alpha_1"]),
        dtype=np.float64,
    )
    target_covariance = model.G_CI_domain_wall(periodic=True, alpha=alpha)
    target_chern = float(np.real(model.real_space_chern_number(target_covariance)))
    samples = int(geometry["samples"])
    dimension = 2 * int(geometry["Nx"]) * int(geometry["Ny"])
    window_start = int(gate["terminal_window_start_cycle"])
    failures: list[dict[str, Any]] = []
    lanes: dict[str, Any] = {}
    for spec in config["initializations"]:
        label = str(spec["label"])
        payloads = []
        for sample in range(samples):
            path, _ = correctness_paths(root, label, sample)
            with np.load(path) as data:
                payloads.append({key: np.asarray(data[key]) for key in data.files})
        for backend in ("covariance", "frame"):
            chern = np.stack(
                [row[f"real_space_chern_{backend}"] for row in payloads], axis=0
            )
            entropy = np.stack(
                [row[f"global_entropy_{backend}"] for row in payloads], axis=0
            )
            final_mean = float(np.mean(chern[:, -1]))
            initial_mean = float(np.mean(chern[:, 0]))
            terminal_error = float(abs(final_mean - target_chern))
            initial_error = float(abs(initial_mean - target_chern))
            drift = float(abs(np.mean(chern[:, -1]) - np.mean(chern[:, window_start])))
            lane_name = f"{label}.{backend}"
            lane = {
                "target_chern_finite_size": target_chern,
                "initial_chern_mean": initial_mean,
                "terminal_chern_mean": final_mean,
                "terminal_chern_standard_deviation": float(np.std(chern[:, -1], ddof=1)),
                "terminal_absolute_error": terminal_error,
                "initial_absolute_error": initial_error,
                "terminal_window_drift": drift,
                "terminal_global_entropy_mean": float(np.mean(entropy[:, -1])),
                "terminal_global_entropy_density_mean": float(
                    np.mean(entropy[:, -1]) / dimension
                ),
            }
            lanes[lane_name] = lane
            if terminal_error > float(gate["terminal_absolute_error_hard"]):
                failures.append(
                    {
                        "lane": lane_name,
                        "gate": "terminal_chern",
                        "observed": terminal_error,
                        "hard": gate["terminal_absolute_error_hard"],
                    }
                )
            if bool(gate["require_improvement_from_cycle_zero"]) and not (
                terminal_error < initial_error
            ):
                failures.append(
                    {
                        "lane": lane_name,
                        "gate": "chern_improvement",
                        "initial_error": initial_error,
                        "terminal_error": terminal_error,
                    }
                )
            if drift > float(gate["terminal_window_drift_hard"]):
                failures.append(
                    {
                        "lane": lane_name,
                        "gate": "terminal_chern_drift",
                        "observed": drift,
                        "hard": gate["terminal_window_drift_hard"],
                    }
                )
            if label == "maxmix":
                entropy_density = lane["terminal_global_entropy_density_mean"]
                if entropy_density > float(gate["maxmix_global_entropy_density_hard"]):
                    failures.append(
                        {
                            "lane": lane_name,
                            "gate": "maxmix_global_entropy_density",
                            "observed": entropy_density,
                            "hard": gate["maxmix_global_entropy_density_hard"],
                        }
                    )
                if bool(gate["maxmix_require_entropy_reduction"]) and not (
                    float(np.mean(entropy[:, -1])) < float(np.mean(entropy[:, 0]))
                ):
                    failures.append(
                        {
                            "lane": lane_name,
                            "gate": "maxmix_entropy_reduction",
                            "initial": float(np.mean(entropy[:, 0])),
                            "terminal": float(np.mean(entropy[:, -1])),
                        }
                    )
    return {
        "passed": not failures,
        "target_chern_finite_size": target_chern,
        "expected_chern_thermodynamic": float(gate["expected_chern"]),
        "lanes": lanes,
        "failures": failures,
    }


def stage_correctness(root: Path, cpu_ids: list[int]) -> bool:
    cpu_list = ",".join(map(str, cpu_ids))
    update_manifest(root, stage="correctness", status="running", cpu_list=cpu_list)
    stage_started = time.perf_counter_ns()
    parallel_timings = {}
    for spec in load_config()["initializations"]:
        label = str(spec["label"])
        tasks = _pending_tasks(root, "correctness", _tasks(root, "correctness", label, cpu_ids))
        rows, timing = _run_parallel_stage(tasks, _correctness_worker, use_barrier=True)
        parallel_timings[label] = timing
    summary_paths = sorted((root / "raw/correctness").glob("*/sample_*/summary.json"))
    expected = len(load_config()["initializations"]) * int(load_config()["geometry"]["samples"])
    if len(summary_paths) != expected:
        raise RuntimeError(f"Correctness stage has {len(summary_paths)} of {expected} completed samples.")
    summaries = [json.loads(path.read_text()) for path in summary_paths]
    failures = [row for row in summaries if not bool(row["gate_passed"])]
    numerical_passed = not failures
    # Physics products remain useful failure-analysis evidence even if a paired
    # numerical comparison misses its hard tolerance.
    physics_gate = evaluate_uniform_insulator_gate(root)
    passed = numerical_passed and bool(physics_gate["passed"])
    details = {
        "samples": expected,
        "passed": passed,
        "numerical_correctness_passed": numerical_passed,
        "uniform_insulator_physics_passed": bool(physics_gate["passed"]),
        "failure_count": len(failures),
        "physics_failure_count": len(physics_gate.get("failures", ())),
        "physics_gate": physics_gate,
        "parallel_timing_ns": parallel_timings,
        "stage_total_ns": int(time.perf_counter_ns() - stage_started),
    }
    write_json_atomic(
        root / "status/correctness_gate.json",
        {"status": "complete", **details, "failures": failures},
    )
    update_manifest(root, stage="correctness", status="complete", details=details)
    return passed


def stage_benchmark(root: Path, cpu_ids: list[int]) -> None:
    gate_path = root / "status/correctness_gate.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text()).get("passed"):
        update_manifest(
            root,
            stage="benchmark",
            status="skipped",
            details={"reason": "correctness gate did not pass"},
        )
        return
    cpu_list = ",".join(map(str, cpu_ids))
    update_manifest(root, stage="benchmark", status="running", cpu_list=cpu_list)
    stage_started = time.perf_counter_ns()
    parallel_timings = {}
    for spec in load_config()["initializations"]:
        label = str(spec["label"])
        tasks = _pending_tasks(root, "benchmark", _tasks(root, "benchmark", label, cpu_ids))
        rows, timing = _run_parallel_stage(tasks, _benchmark_worker, use_barrier=True)
        parallel_timings[label] = timing
    expected = len(load_config()["initializations"]) * int(load_config()["geometry"]["samples"])
    summaries = list((root / "raw/benchmark").glob("*/sample_*/summary.json"))
    if len(summaries) != expected:
        raise RuntimeError(f"Benchmark stage has {len(summaries)} of {expected} completed samples.")
    details = {
        "paired_samples": expected,
        "logical_lanes": expected * 2,
        "parallel_timing_ns": parallel_timings,
        "stage_total_ns": int(time.perf_counter_ns() - stage_started),
    }
    write_json_atomic(root / "status/benchmark.json", {"status": "complete", **details})
    update_manifest(root, stage="benchmark", status="complete", details=details)


def _run_analysis_stage(root: Path, mode: str) -> None:
    command = [sys.executable, str(HERE / "analyze_results.py"), mode, "--campaign-root", str(root)]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    (root / f"logs/{mode}.log").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{mode} failed; see {root / f'logs/{mode}.log'}")


def stage_validate(root: Path) -> bool:
    config = load_config()
    samples = int(config["geometry"]["samples"])
    cycles = int(config["geometry"]["cycles"])
    errors = []
    for spec in config["initializations"]:
        label = spec["label"]
        for sample in range(samples):
            data_path, summary_path = correctness_paths(root, label, sample)
            if not data_path.exists() or not summary_path.exists():
                errors.append(f"missing correctness shard {label}/{sample}")
                continue
            with np.load(data_path) as data:
                if not np.array_equal(data["cycles"], np.arange(cycles + 1)):
                    errors.append(f"incomplete cycle coordinate {label}/{sample}")
            summary = json.loads(summary_path.read_text())
            if not summary.get("gate_passed"):
                errors.append(f"correctness gate failed {label}/{sample}")
        benchmark_count = len(list((root / "raw/benchmark" / label).glob("sample_*/summary.json")))
        gate_passed = json.loads((root / "status/correctness_gate.json").read_text()).get("passed")
        if gate_passed and benchmark_count != samples:
            errors.append(f"benchmark count for {label} is {benchmark_count}, expected {samples}")
    passed = not errors
    write_json_atomic(
        root / "status/validation.json",
        {"status": "complete", "passed": passed, "errors": errors},
    )
    update_manifest(
        root,
        stage="validate",
        status="complete" if passed else "failed",
        details={"passed": passed, "errors": errors},
    )
    return passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=(
            "init",
            "preflight",
            "records",
            "correctness",
            "benchmark",
            "analyze",
            "validate",
            "report",
            "all",
        ),
    )
    parser.add_argument("--campaign-id")
    parser.add_argument(
        "--size",
        type=int,
        default=16,
        help="Square Nx=Ny size; cycles and checkpoints scale as 2*Ny.",
    )
    parser.add_argument("--cpu-list", default="")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--select-idle-cpus", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--allow-busy", action="store_true")
    return parser


def main() -> int:
    global CAMPAIGN_SIZE
    args = build_parser().parse_args()
    if int(args.size) < 4 or int(args.size) % 2:
        raise SystemExit("--size must be an even integer at least 4.")
    CAMPAIGN_SIZE = int(args.size)
    if args.select_idle_cpus:
        print(",".join(map(str, select_idle_cpu_ids(args.limit, allow_busy=args.allow_busy))))
        return 0
    if not args.stage or not args.campaign_id:
        raise SystemExit("stage and --campaign-id are required")
    workers = int(load_config()["parallel"]["workers"])
    if args.max_workers != workers:
        raise SystemExit(f"This locked campaign requires --max-workers={workers}.")
    cpu_ids = parse_cpu_list(args.cpu_list)
    if args.stage not in ("init", "preflight", "analyze", "validate", "report"):
        try:
            cpu_ids = validate_cpu_ids(cpu_ids, workers)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    root = initialize_campaign(args.campaign_id)
    try:
        if args.stage == "init":
            return 0
        if args.stage == "preflight":
            return 0 if stage_preflight(root, args.cpu_list) else 1
        if args.stage == "records":
            stage_records(root, cpu_ids)
        elif args.stage == "correctness":
            return 0 if stage_correctness(root, cpu_ids) else 1
        elif args.stage == "benchmark":
            stage_benchmark(root, cpu_ids)
        elif args.stage == "analyze":
            _run_analysis_stage(root, "analyze")
            update_manifest(root, stage="analyze", status="complete")
        elif args.stage == "validate":
            return 0 if stage_validate(root) else 1
        elif args.stage == "report":
            _run_analysis_stage(root, "report")
            update_manifest(root, stage="report", status="complete")
        elif args.stage == "all":
            campaign_started = time.perf_counter_ns()
            if not stage_preflight(root, args.cpu_list):
                update_manifest(
                    root,
                    stage="campaign",
                    status="failed",
                    details={"campaign_total_ns": int(time.perf_counter_ns() - campaign_started)},
                )
                return 1
            stage_records(root, cpu_ids)
            correctness_passed = stage_correctness(root, cpu_ids)
            if correctness_passed:
                stage_benchmark(root, cpu_ids)
            else:
                update_manifest(
                    root,
                    stage="benchmark",
                    status="skipped",
                    details={"reason": "correctness gate did not pass"},
                )
            _run_analysis_stage(root, "analyze")
            update_manifest(root, stage="analyze", status="complete")
            validation_passed = stage_validate(root)
            _run_analysis_stage(root, "report")
            update_manifest(root, stage="report", status="complete")
            campaign_passed = correctness_passed and validation_passed
            update_manifest(
                root,
                stage="campaign",
                status="complete" if campaign_passed else "failed",
                details={
                    "campaign_total_ns": int(time.perf_counter_ns() - campaign_started),
                    "correctness_passed": bool(correctness_passed),
                    "validation_passed": bool(validation_passed),
                },
            )
            return 0 if campaign_passed else 1
    except Exception as error:
        update_manifest(
            root,
            stage=args.stage,
            status="failed",
            cpu_list=args.cpu_list,
            details={"error": repr(error)},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
