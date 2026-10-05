#!/usr/bin/env python3
"""Completion-resumable continuously replayed frozen-record flux ramps."""

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
from typing import Any, Iterable

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fgtn.classA_U1FGTN import classA_U1FGTN  # noqa: E402
import run_online_flux_ramp as online  # noqa: E402


CAMPAIGN_SCHEMA = "frozen_record_continuous_ramp_campaign_v1"
REFERENCE_SCHEMA = "frozen_record_continuous_reference_v1"
RAMP_SCHEMA = "frozen_record_continuous_ramp_v1"
COMPLETION_SCHEMA = "frozen_record_continuous_completion_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.frozen_continuous_ramp_s10_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "N16x20_frozen_record_continuous_ramp_s10_v1"
SOURCE_PATHS = {
    "cpu_engine": REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py",
    "occupied_frame": REPO_ROOT / "src" / "fgtn" / "occupied_frame.py",
    "campaign_runner": Path(__file__).resolve(),
    "burnin_verifier": Path(online.__file__).resolve(),
}
WALLS = ("soft", "hard")
DIRECTIONS = ("ccw", "cw")


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


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "root_seed", "burnin_source", "geometry",
        "walls", "dynamics", "twist", "regions", "ensemble", "acceptance",
    )
    return sha256_bytes(canonical_json({key: config[key] for key in keys}).encode("utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolved_input_path(config: dict[str, Any], key: str) -> Path:
    return (PROJECT_ROOT / str(config["burnin_source"][key])).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    if config.get("campaign_id") != "N16x20_frozen_record_continuous_ramp_s10_v1":
        raise ValueError("unexpected campaign_id")
    if int(config.get("root_seed", -1)) != 2026090302:
        raise ValueError("unexpected root seed")
    geometry = config["geometry"]
    expected_geometry = {
        "Nx": 16, "Ny": 20, "DW": True, "dw_interval": [4, 12],
        "nshell": 1, "filling_frac": 0.5, "alpha_1": 1.0,
        "alpha_2": 30.0, "trial_orbitals": "X",
    }
    if geometry != expected_geometry:
        raise ValueError("geometry differs from the locked pilot")
    if config["walls"] != {
        "soft": {"dw_truncation": False, "meas_slab_only": False},
        "hard": {"dw_truncation": True, "meas_slab_only": True},
    }:
        raise ValueError("wall definitions differ from the locked pilot")
    expected_dynamics = {
        "burn_in_cycles": 40,
        "reference_cycles": 64,
        "ramp_cycles": [16, 32, 64],
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
        raise ValueError("dynamics differ from the locked pilot")
    if config["twist"] != {
        "directions": {"ccw": 1, "cw": -1},
        "schedule": "phi_j = sigma * 2*pi*j/M, j=0,...,M",
        "spectral_origin_offset": None,
    }:
        raise ValueError("twist contract differs from the locked pilot")
    if config["regions"] != {
        "left_x_start": 0, "left_x_stop_exclusive": 8,
        "right_x_start": 8, "right_x_stop_exclusive": 16,
    }:
        raise ValueError("regions must be the two x half-systems")
    if int(config["ensemble"]["samples_per_wall"]) != 10:
        raise ValueError("production ensemble must contain ten samples per wall")
    source = config["burnin_source"]
    if source.get("campaign_id") != "N16x20_online_flux_ramp_s10_v1":
        raise ValueError("unexpected burn-in campaign")
    old_config = online.load_config(_resolved_input_path(config, "config"))
    online.validate_config(old_config)
    if old_config["campaign_id"] != source["campaign_id"]:
        raise ValueError("burn-in campaign identity mismatch")
    if old_config["geometry"] != geometry or old_config["walls"] != config["walls"]:
        raise ValueError("burn-in geometry/walls do not match the new campaign")


def _seed(root_seed: int, wall: str, sample_id: int, stage: int, ramp_cycles: int = 0) -> int:
    wall_code = {"soft": 0, "hard": 1}[str(wall)]
    sequence = np.random.SeedSequence(
        int(root_seed), spawn_key=(wall_code, int(sample_id), int(stage), int(ramp_cycles))
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def reference_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = [
        {
            "stage": "reference",
            "task_id": f"reference_{wall}_sample_{sample_id:03d}",
            "wall": wall,
            "sample_id": sample_id,
            "seed": _seed(config["root_seed"], wall, sample_id, 0),
            "burnin_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
            "cycles": int(config["dynamics"]["reference_cycles"]),
        }
        for wall in WALLS
        for sample_id in range(int(config["ensemble"]["samples_per_wall"]))
    ]
    _require_unique_ids(tasks)
    return tasks


def replay_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = [
        {
            "stage": "replay",
            "task_id": f"replay_{wall}_{direction}_M{cycles:03d}_sample_{sample_id:03d}",
            "wall": wall,
            "direction": direction,
            "sigma": int(config["twist"]["directions"][direction]),
            "sample_id": sample_id,
            "seed": _seed(config["root_seed"], wall, sample_id, 1, cycles),
            "burnin_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
            "reference_task_id": f"reference_{wall}_sample_{sample_id:03d}",
            "cycles": int(cycles),
        }
        for wall in WALLS
        for sample_id in range(int(config["ensemble"]["samples_per_wall"]))
        for cycles in config["dynamics"]["ramp_cycles"]
        for direction in DIRECTIONS
    ]
    _require_unique_ids(tasks)
    for wall in WALLS:
        for sample_id in range(int(config["ensemble"]["samples_per_wall"])):
            for cycles in config["dynamics"]["ramp_cycles"]:
                pair = [
                    task for task in tasks
                    if task["wall"] == wall and task["sample_id"] == sample_id
                    and task["cycles"] == cycles
                ]
                if len(pair) != 2 or pair[0]["seed"] != pair[1]["seed"]:
                    raise RuntimeError("CW/CCW replay seeds must be paired")
    return tasks


def _require_unique_ids(tasks: list[dict[str, Any]]) -> None:
    ids = [str(task["task_id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("task IDs are not unique")


def result_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    if task["stage"] == "reference":
        root = Path(output_root) / "references" / task["wall"]
    else:
        root = (
            Path(output_root) / "ramps" / f"M{int(task['cycles']):03d}"
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
    arrays = dict(arrays)
    arrays["metadata_json"] = np.asarray(canonical_json(_task_metadata(task, config_hash, hashes)))
    arrays["dependencies_json"] = np.asarray(canonical_json(dependencies))
    _atomic_npz(result, arrays)
    record = {
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
            "result": record,
            "elapsed_seconds": float(elapsed_seconds),
            "completed_unix": time.time(),
        },
    )
    return record


def _required_series(saved: Any, length: int) -> None:
    for key in (
        "cycles", "phi", "N_left", "N_right", "N_total", "delta_N_left",
        "delta_N_right", "delta_N_total", "q_x", "net_injected_charge",
        "injection_count", "rank", "minimum_selected_probability_by_cycle",
        "branch_log_probability_by_cycle",
    ):
        if np.asarray(saved[key]).shape != (length,):
            raise ValueError(f"{key} has the wrong shape")
    if np.asarray(saved["density_x"]).shape != (length, 16):
        raise ValueError("density_x has the wrong shape")


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
        with np.load(result, allow_pickle=False) as saved:
            schema = str(np.asarray(saved["schema"]).item())
            wanted = REFERENCE_SCHEMA if task["stage"] == "reference" else RAMP_SCHEMA
            if schema != wanted:
                return False, "result schema mismatch", None
            metadata = json.loads(str(np.asarray(saved["metadata_json"]).item()))
            if metadata != _task_metadata(task, config_hash, hashes):
                return False, "result identity mismatch", None
            saved_dependencies = json.loads(str(np.asarray(saved["dependencies_json"]).item()))
            if saved_dependencies != dependencies:
                return False, "result dependency mismatch", None
            length = int(task["cycles"]) + 1
            _required_series(saved, length)
            if not np.array_equal(np.asarray(saved["cycles"]), np.arange(length)):
                return False, "cycle coordinates mismatch", None
            expected_phi = (
                np.zeros(length, dtype=np.float64)
                if task["stage"] == "reference" else schedule(task)
            )
            if not np.array_equal(np.asarray(saved["phi"]), expected_phi):
                return False, "twist schedule mismatch", None
            left = np.asarray(saved["N_left"], dtype=np.float64)
            right = np.asarray(saved["N_right"], dtype=np.float64)
            total = np.asarray(saved["N_total"], dtype=np.float64)
            if not np.allclose(np.asarray(saved["delta_N_left"]), left - left[0], rtol=0.0, atol=1e-12):
                return False, "left-charge difference mismatch", None
            if not np.allclose(np.asarray(saved["delta_N_right"]), right - right[0], rtol=0.0, atol=1e-12):
                return False, "right-charge difference mismatch", None
            if not np.allclose(np.asarray(saved["delta_N_total"]), total - total[0], rtol=0.0, atol=1e-12):
                return False, "total-charge difference mismatch", None
            wanted_qx = 0.5 * (
                np.asarray(saved["delta_N_right"]) - np.asarray(saved["delta_N_left"])
            )
            if not np.allclose(np.asarray(saved["q_x"]), wanted_qx, rtol=0.0, atol=1e-12):
                return False, "q_x definition mismatch", None
            wanted_continuity = (
                np.asarray(saved["delta_N_total"]) - np.asarray(saved["net_injected_charge"])
            )
            if not np.allclose(
                np.asarray(saved["charge_continuity_residual"]), wanted_continuity,
                rtol=0.0, atol=1e-12,
            ):
                return False, "charge-continuity identity mismatch", None
            gram_residual = float(np.asarray(saved["final_frame_gram_residual"]).item())
            if not np.isfinite(gram_residual) or gram_residual < 0.0:
                return False, "invalid final-frame Gram residual", None
            if task["stage"] == "reference":
                raw = np.asarray(saved["record_json_utf8"], dtype=np.uint8).tobytes()
                if sha256_bytes(raw) != str(np.asarray(saved["record_sha256"]).item()):
                    return False, "embedded record checksum mismatch", None
                parsed = json.loads(raw.decode("utf-8"))
                if not isinstance(parsed, list) or not parsed:
                    return False, "embedded record is empty or malformed", None
                record_prefix(parsed, int(task["cycles"]))
                frame = np.asarray(saved["final_frame"])
                if frame.dtype != np.complex128 or frame.ndim != 2:
                    return False, "reference final frame dtype/shape mismatch", None
            else:
                if str(np.asarray(saved["reference_sha256"]).item()) != dependencies["reference_sha256"]:
                    return False, "replay reference checksum mismatch", None
            for key in (
                "phi", "N_left", "N_right", "N_total", "delta_N_left",
                "delta_N_right", "delta_N_total", "q_x",
                "charge_continuity_residual", "density_x",
                "minimum_selected_probability_by_cycle",
                "branch_log_probability_by_cycle",
            ):
                if not np.all(np.isfinite(np.asarray(saved[key]))):
                    return False, f"nonfinite result field {key}", None
        return True, "verified", completion
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _old_burnin_context(config: dict[str, Any]) -> tuple[dict[str, Any], Path, str, dict[str, str]]:
    old_config = online.load_config(_resolved_input_path(config, "config"))
    old_root = _resolved_input_path(config, "output_root")
    return old_config, old_root, online.scientific_config_hash(old_config), online.source_hashes()


def verify_burnins(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    old_config, old_root, old_hash, old_sources = _old_burnin_context(config)
    rows: dict[str, dict[str, Any]] = {}
    for task in online.expand_burnin_tasks(old_config):
        ok, reason, completion = online.verify_pair(old_root, task, old_hash, old_sources)
        if not ok or completion is None:
            raise RuntimeError(f"unverified reused burn-in {task['task_id']}: {reason}")
        result, _ = online.result_paths(old_root, task)
        rows[task["task_id"]] = {
            "task": task,
            "result_path": str(result),
            "sha256": str(completion["result"]["sha256"]),
        }
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 verified burn-ins, found {len(rows)}")
    return rows


def load_burnin(row: dict[str, Any]) -> np.ndarray:
    path = Path(row["result_path"])
    if sha256_path(path) != row["sha256"]:
        raise RuntimeError(f"burn-in changed after verification: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return np.array(saved["frame"], dtype=np.complex128, copy=True)


def dependencies_for(
    task: dict[str, Any],
    burnins: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    dependencies = {"burnin_sha256": burnins[task["burnin_task_id"]]["sha256"]}
    if task["stage"] == "replay":
        if references is None or task["reference_task_id"] not in references:
            raise RuntimeError(f"missing reference dependency for {task['task_id']}")
        dependencies["reference_sha256"] = references[task["reference_task_id"]]["sha256"]
    return dependencies


def _model(config: dict[str, Any], wall: str) -> classA_U1FGTN:
    geometry = config["geometry"]
    wall_config = config["walls"][wall]
    return classA_U1FGTN(
        Nx=int(geometry["Nx"]), Ny=int(geometry["Ny"]), DW=True,
        nshell=int(geometry["nshell"]), filling_frac=float(geometry["filling_frac"]),
        alpha_1=float(geometry["alpha_1"]), alpha_2=float(geometry["alpha_2"]),
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_interval=tuple(int(value) for value in geometry["dw_interval"]),
        dw_truncation=bool(wall_config["dw_truncation"]), twist_y=0.0,
    )


def _engine_kwargs(config: dict[str, Any], wall: str, cycles: int, seed: int) -> dict[str, Any]:
    return {
        "G_history": False, "progress": False, "cycles": int(cycles),
        "postselect": False, "postselect_probability": 0.0,
        "perfect_correction": True, "samples": 1, "parallelize_samples": False,
        "init_mode": "default", "save": False, "sequence": "raster_y",
        "meas_slab_only": bool(config["walls"][wall]["meas_slab_only"]),
        "random_seed": int(seed), "physical_covariance_update": "rank1",
        "state_representation": "physical_frame", "return_native_state": True,
        "require_no_covariance_materialization": True,
    }


def _mode_x(nx: int, ny: int) -> np.ndarray:
    return np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)


def _frame_observables(state: Any, nx: int, ny: int) -> tuple[float, float, float, np.ndarray]:
    frame = np.asarray(state.physical_frame)
    if frame.dtype != np.complex128:
        raise TypeError(f"expected complex128 frame, got {frame.dtype}")
    occupation = np.real(np.sum(np.abs(frame) ** 2, axis=1))
    x = _mode_x(nx, ny)
    density_x = np.asarray([occupation[x == value].sum() for value in range(nx)])
    left = float(density_x[: nx // 2].sum())
    right = float(density_x[nx // 2 :].sum())
    return left, right, float(left + right), density_x


class RecordAudit:
    def __init__(self, cycles: int, capture: bool = False) -> None:
        self.capture = bool(capture)
        self.entries: list[dict[str, Any]] = []
        self.log_by_cycle = np.zeros(int(cycles) + 1, dtype=np.float64)
        self.minimum_by_cycle = np.ones(int(cycles) + 1, dtype=np.float64)

    @staticmethod
    def selected_probability(event: dict[str, Any]) -> float:
        probability = float(event["probability"])
        if event["kind"] == "measurement":
            return probability if bool(event["outcome_occupied"]) else 1.0 - probability
        expected = bool(event["expected_occupied"])
        target = bool(event["target_occupied"])
        if bool(event.get("perfect_correction")):
            return 1.0 if target == expected else 0.0
        occurred = target if expected else not target
        return probability if occurred else 1.0 - probability

    def __call__(self, *, cycle: int, site_id: int, branch_log_weight: float, branch_events: Any, **_: Any) -> None:
        index = int(cycle)
        events = [dict(event) for event in branch_events]
        self.log_by_cycle[index] += float(branch_log_weight)
        if events:
            self.minimum_by_cycle[index] = min(
                self.minimum_by_cycle[index],
                *(self.selected_probability(event) for event in events),
            )
        if self.capture:
            self.entries.append({"cycle": index, "site_id": int(site_id), "branch_events": events})


class ChargeObserver:
    def __init__(self, model: classA_U1FGTN, config: dict[str, Any], phi: np.ndarray, queue: Any, stage: str) -> None:
        self.model = model
        self.nx = int(config["geometry"]["Nx"])
        self.ny = int(config["geometry"]["Ny"])
        self.phi = np.asarray(phi, dtype=np.float64)
        self.queue = queue
        self.stage = str(stage)
        length = self.phi.size
        self.N_left = np.full(length, np.nan)
        self.N_right = np.full(length, np.nan)
        self.N_total = np.full(length, np.nan)
        self.rank = np.full(length, -1, dtype=np.int64)
        self.net_injected_charge = np.zeros(length, dtype=np.int64)
        self.injection_count = np.zeros(length, dtype=np.int64)
        self.density_x = np.full((length, self.nx), np.nan)
        self._net = 0
        self._count = 0
        self._seen = np.zeros(length, dtype=bool)

    def event(self, *, outcome_occupied: bool, target_occupied: bool, **_: Any) -> None:
        delta = int(bool(target_occupied)) - int(bool(outcome_occupied))
        self._net += delta
        self._count += abs(delta)

    def cycle(self, *, cycle: int, state: Any, **_: Any) -> None:
        index = int(cycle)
        if not np.isclose(float(self.model.twist_y), float(self.phi[index]), rtol=0.0, atol=1e-14):
            raise RuntimeError(
                f"cycle {index} observed twist {self.model.twist_y}, expected {self.phi[index]}"
            )
        left, right, total, density_x = _frame_observables(state, self.nx, self.ny)
        self.N_left[index], self.N_right[index], self.N_total[index] = left, right, total
        self.rank[index] = int(state.rank)
        self.net_injected_charge[index] = self._net
        self.injection_count[index] = self._count
        self.density_x[index] = density_x
        self._seen[index] = True
        if index > 0 and self.queue is not None:
            self.queue.put((self.stage, 1))

    def arrays(self, config: dict[str, Any], audit: RecordAudit) -> dict[str, Any]:
        if not np.all(self._seen):
            raise RuntimeError("charge observer missed one or more cycle boundaries")
        delta_left = self.N_left - self.N_left[0]
        delta_right = self.N_right - self.N_right[0]
        delta_total = self.N_total - self.N_total[0]
        q_x = 0.5 * (delta_right - delta_left)
        continuity = delta_total - self.net_injected_charge
        for name, value in (
            ("regional charge", np.stack((self.N_left, self.N_right, self.N_total))),
            ("density_x", self.density_x),
            ("record probability", audit.minimum_by_cycle),
            ("record log probability", audit.log_by_cycle),
        ):
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"encountered nonfinite {name}")
        if np.max(np.abs(continuity)) > float(config["acceptance"]["charge_continuity_tolerance"]):
            raise FloatingPointError("charge continuity exceeded tolerance")
        return {
            "cycles": np.arange(self.phi.size, dtype=np.int64),
            "phi": self.phi,
            "N_left": self.N_left,
            "N_right": self.N_right,
            "N_total": self.N_total,
            "delta_N_left": delta_left,
            "delta_N_right": delta_right,
            "delta_N_total": delta_total,
            "q_x": q_x,
            "net_injected_charge": self.net_injected_charge,
            "injection_count": self.injection_count,
            "charge_continuity_residual": continuity,
            "rank": self.rank,
            "density_x": self.density_x,
            "delta_density_x": self.density_x - self.density_x[0],
            "minimum_selected_probability_by_cycle": audit.minimum_by_cycle,
            "branch_log_probability_by_cycle": audit.log_by_cycle,
        }


def record_bytes(record: list[dict[str, Any]]) -> bytes:
    return canonical_json(record).encode("utf-8")


def parse_record(raw: bytes) -> list[dict[str, Any]]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, list):
        raise TypeError("trajectory record must be a list")
    return [dict(entry) for entry in value]


def record_prefix(record: Iterable[dict[str, Any]], cycles: int) -> list[dict[str, Any]]:
    prefix = [dict(entry) for entry in record if 1 <= int(entry["cycle"]) <= int(cycles)]
    seen = {int(entry["cycle"]) for entry in prefix}
    if seen != set(range(1, int(cycles) + 1)):
        raise ValueError(f"record prefix does not cover cycles 1..{cycles}")
    return prefix


def schedule(task: dict[str, Any]) -> np.ndarray:
    cycles = int(task["cycles"])
    sigma = int(task["sigma"])
    return sigma * np.linspace(0.0, 2.0 * math.pi, cycles + 1, dtype=np.float64)


def _reference_row(output_root: Path, task: dict[str, Any], config_hash: str, hashes: dict[str, str], dependencies: dict[str, str]) -> dict[str, Any] | None:
    ok, _, completion = verify_pair(output_root, task, config_hash, hashes, dependencies)
    if not ok or completion is None:
        return None
    result, _ = result_paths(output_root, task)
    return {"task": task, "path": str(result), "sha256": str(completion["result"]["sha256"])}


def load_reference(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    path = Path(row["path"])
    if sha256_path(path) != row["sha256"]:
        raise RuntimeError(f"reference changed after verification: {path}")
    with np.load(path, allow_pickle=False) as saved:
        raw = np.asarray(saved["record_json_utf8"], dtype=np.uint8).tobytes()
        if sha256_bytes(raw) != str(np.asarray(saved["record_sha256"]).item()):
            raise RuntimeError("embedded record checksum mismatch")
    return parse_record(raw), row["sha256"]


def _record_failure(output_root: Path, task: dict[str, Any], exc: BaseException) -> None:
    _atomic_json(
        failure_path(output_root, task),
        {
            "task_id": task["task_id"], "failed_unix": time.time(),
            "error_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _reference_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, burnin_row, dependencies, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    log_path = output_root / "logs" / "tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame = load_burnin(burnin_row)
        phi = np.zeros(int(task["cycles"]) + 1, dtype=np.float64)
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = _model(config, task["wall"])
                audit = RecordAudit(task["cycles"], capture=True)
                observer = ChargeObserver(model, config, phi, queue, "reference")
                result = model.run_markov_circuit(
                    frame_init=frame, frame_init_prepared=True,
                    native_cycle_observer=observer.cycle,
                    native_event_observer=observer.event,
                    trajectory_weight_observer=audit,
                    **_engine_kwargs(config, task["wall"], task["cycles"], task["seed"]),
                )
        gram_residual = float(result["native_final"]["gram_residual"])
        if not np.isfinite(gram_residual) or gram_residual > float(config["acceptance"]["frame_gram_tolerance"]):
            raise FloatingPointError(f"reference frame Gram residual failed: {gram_residual:.3e}")
        raw = record_bytes(audit.entries)
        arrays = observer.arrays(config, audit)
        arrays.update(
            {
                "schema": np.asarray(REFERENCE_SCHEMA),
                "record_json_utf8": np.frombuffer(raw, dtype=np.uint8).copy(),
                "record_sha256": np.asarray(sha256_bytes(raw)),
                "final_frame": np.asarray(result["native_final"]["frame"], dtype=np.complex128),
                "final_frame_gram_residual": np.asarray(gram_residual),
            }
        )
        record = publish_pair(
            output_root, task, arrays, config_hash, hashes, dependencies,
            time.perf_counter() - started,
        )
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"], "result": record}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def _replay_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, burnin_row, reference_row, dependencies, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    log_path = output_root / "logs" / "tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame = load_burnin(burnin_row)
        full_record, reference_sha = load_reference(reference_row)
        prefix = record_prefix(full_record, int(task["cycles"]))
        prefix_raw = record_bytes(prefix)
        phi = schedule(task)
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = _model(config, task["wall"])
                audit = RecordAudit(task["cycles"], capture=False)
                observer = ChargeObserver(model, config, phi, queue, "replay")
                result = model.run_markov_circuit(
                    frame_init=frame, frame_init_prepared=True,
                    controller_twist_schedule=phi,
                    controller_twist_gauge="uniform",
                    trajectory_replay=prefix,
                    trajectory_replay_probability_tol=float(
                        config["acceptance"]["trajectory_replay_probability_tolerance"]
                    ),
                    native_cycle_observer=observer.cycle,
                    native_event_observer=observer.event,
                    trajectory_weight_observer=audit,
                    **_engine_kwargs(config, task["wall"], task["cycles"], task["seed"]),
                )
        gram_residual = float(result["native_final"]["gram_residual"])
        if not np.isfinite(gram_residual) or gram_residual > float(config["acceptance"]["frame_gram_tolerance"]):
            raise FloatingPointError(f"replay frame Gram residual failed: {gram_residual:.3e}")
        arrays = observer.arrays(config, audit)
        arrays.update(
            {
                "schema": np.asarray(RAMP_SCHEMA),
                "burnin_sha256": np.asarray(dependencies["burnin_sha256"]),
                "reference_sha256": np.asarray(reference_sha),
                "record_prefix_sha256": np.asarray(sha256_bytes(prefix_raw)),
                "record_prefix_entries": np.asarray(len(prefix), dtype=np.int64),
                "final_frame_gram_residual": np.asarray(gram_residual),
            }
        )
        record = publish_pair(
            output_root, task, arrays, config_hash, hashes, dependencies,
            time.perf_counter() - started,
        )
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"], "result": record}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def verified_references(
    config: dict[str, Any], output_root: Path, burnins: dict[str, dict[str, Any]],
    config_hash: str, hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for task in reference_tasks(config):
        dependencies = dependencies_for(task, burnins)
        row = _reference_row(output_root, task, config_hash, hashes, dependencies)
        if row is not None:
            rows[task["task_id"]] = row
    return rows


def _selected(tasks: list[dict[str, Any]], sample_ids: set[int] | None) -> list[dict[str, Any]]:
    if sample_ids is None:
        return tasks
    return [task for task in tasks if int(task["sample_id"]) in sample_ids]


def _drain_progress(queue: Any, cycle_bar: Any) -> None:
    while True:
        try:
            _, count = queue.get_nowait()
        except Exception:
            return
        cycle_bar.update(int(count))


def run_stage(
    stage: str, config: dict[str, Any], output_root: Path,
    burnins: dict[str, dict[str, Any]], workers: int,
    sample_ids: set[int] | None, resume: bool,
) -> None:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    all_tasks = reference_tasks(config) if stage == "reference" else replay_tasks(config)
    tasks = _selected(all_tasks, sample_ids)
    references = verified_references(config, output_root, burnins, config_hash, hashes)
    pending: list[tuple[dict[str, Any], dict[str, str]]] = []
    verified: list[dict[str, Any]] = []
    for task in tasks:
        dependencies = dependencies_for(task, burnins, references)
        ok, _, _ = verify_pair(output_root, task, config_hash, hashes, dependencies)
        if resume and ok:
            verified.append(task)
        else:
            pending.append((task, dependencies))
    total_cycles = sum(int(task["cycles"]) for task in tasks)
    completed_cycles = sum(int(task["cycles"]) for task in verified)
    print(f"[{stage}] verified={len(verified)}/{len(tasks)} pending={len(pending)} workers={workers}", flush=True)
    if not pending:
        return
    context = mp.get_context("spawn")
    with mp.Manager() as manager:
        queue = manager.Queue()
        cycle_bar = tqdm(
            total=total_cycles, initial=completed_cycles,
            desc=f"{stage} cycles", unit="cycle", position=0,
        )
        task_bar = tqdm(
            total=len(tasks), initial=len(verified),
            desc=f"{stage} tasks", unit="task", position=1,
        )
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=min(int(workers), len(pending)), mp_context=context) as pool:
            futures = set()
            for task, dependencies in pending:
                if stage == "reference":
                    payload = (
                        task, config, str(output_root), config_hash, hashes,
                        burnins[task["burnin_task_id"]], dependencies, queue,
                    )
                    futures.add(pool.submit(_reference_worker, payload))
                else:
                    reference_row = references[task["reference_task_id"]]
                    payload = (
                        task, config, str(output_root), config_hash, hashes,
                        burnins[task["burnin_task_id"]], reference_row,
                        dependencies, queue,
                    )
                    futures.add(pool.submit(_replay_worker, payload))
            while futures:
                done, futures = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                _drain_progress(queue, cycle_bar)
                for future in done:
                    row = future.result()
                    task_bar.update(1)
                    if not row["ok"]:
                        failure = f"{row['task_id']}: {row['error']}"
                        failures.append(failure)
                        tqdm.write(f"[{stage} failure] {failure}")
        _drain_progress(queue, cycle_bar)
        cycle_bar.close()
        task_bar.close()
    if failures:
        raise RuntimeError(f"{stage} failed for {len(failures)} task(s); see failure JSON/logs")


def inventory(config: dict[str, Any], output_root: Path, burnins: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    references = verified_references(config, output_root, burnins, config_hash, hashes)
    reference_rows = {}
    for task in reference_tasks(config):
        dependencies = dependencies_for(task, burnins)
        reference_rows[task["task_id"]] = verify_pair(
            output_root, task, config_hash, hashes, dependencies
        )
    replay_rows = {}
    for task in replay_tasks(config):
        if task["reference_task_id"] not in references:
            replay_rows[task["task_id"]] = (False, "reference is not verified", None)
            continue
        dependencies = dependencies_for(task, burnins, references)
        replay_rows[task["task_id"]] = verify_pair(
            output_root, task, config_hash, hashes, dependencies
        )
    return {
        "config_hash": config_hash, "source_hashes": hashes,
        "references": reference_rows, "replays": replay_rows,
    }


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    reference_done = sum(row[0] for row in status["references"].values())
    replay_done = sum(row[0] for row in status["replays"].values())
    print(f"[campaign] {config['campaign_id']}")
    print("[contract] Nx=16 Ny=20 raster-y, reused burn-in=40, frozen reference=64")
    print("[contract] soft+hard, S=10, pure, nshell=1, alpha=(1,30), perfect correction, complex128")
    print("[ramps] M=16,32,64; same frozen prefix for CW/CCW; continuous state; no tangent modes")
    print(f"[output] {output_root.resolve()}")
    print(f"[identity] config_sha256={status['config_hash']}")
    for name, digest in status["source_hashes"].items():
        print(f"[source] {name}={SOURCE_PATHS[name]} sha256={digest}")
    print(f"[resume] references verified={reference_done}/20 pending={20-reference_done}")
    print(f"[resume] replays verified={replay_done}/120 pending={120-replay_done}")


def validate_pairing(config: dict[str, Any], output_root: Path, sample_ids: set[int] | None = None) -> None:
    burnins = verify_burnins(config)
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    references = verified_references(config, output_root, burnins, config_hash, hashes)
    tasks = _selected(replay_tasks(config), sample_ids)
    grouped: dict[tuple[str, int, int], dict[str, Path]] = {}
    for task in tasks:
        dependencies = dependencies_for(task, burnins, references)
        ok, reason, _ = verify_pair(output_root, task, config_hash, hashes, dependencies)
        if not ok:
            raise RuntimeError(f"unverified replay during pairing audit: {task['task_id']}: {reason}")
        grouped.setdefault((task["wall"], task["sample_id"], task["cycles"]), {})[task["direction"]] = result_paths(output_root, task)[0]
    for key, paths in grouped.items():
        if set(paths) != set(DIRECTIONS):
            raise RuntimeError(f"incomplete direction pair for {key}")
        with np.load(paths["ccw"], allow_pickle=False) as positive, np.load(paths["cw"], allow_pickle=False) as negative:
            for field in ("rank", "net_injected_charge", "injection_count"):
                if not np.array_equal(positive[field], negative[field]):
                    raise RuntimeError(f"paired {field} mismatch for {key}")
            if str(positive["record_prefix_sha256"]) != str(negative["record_prefix_sha256"]):
                raise RuntimeError(f"paired record-prefix mismatch for {key}")
            if not np.array_equal(positive["phi"], -np.asarray(negative["phi"])):
                raise RuntimeError(f"paired twist schedules do not reverse for {key}")
    print(f"[paired validation] verified {len(grouped)} CW/CCW pairs", flush=True)


def write_identity(config: dict[str, Any], output_root: Path, burnins: dict[str, dict[str, Any]]) -> None:
    identity_path = output_root / "campaign_identity.json"
    payload = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_hash": scientific_config_hash(config),
        "source_hashes": source_hashes(),
        "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
        "reused_burnins": {key: value["sha256"] for key, value in sorted(burnins.items())},
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
    parser.add_argument("stage", choices=("report", "reference", "replay", "all", "validate"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    sample_ids = _parse_sample_ids(args.sample_ids)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or config["execution"]["workers"])
    if workers <= 0:
        raise ValueError("workers must be positive")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    print(f"[execution] workers={workers} affinity={affinity or 'unavailable'} BLAS_threads={config['execution']['blas_threads']}")
    burnins = verify_burnins(config)
    print("[inputs] verified 20/20 existing burn-in result/completion pairs")
    write_identity(config, output_root, burnins)
    print_inventory(config, output_root, inventory(config, output_root, burnins))
    if args.stage == "report":
        return
    if args.stage in ("reference", "all"):
        run_stage("reference", config, output_root, burnins, workers, sample_ids, args.resume)
    if args.stage in ("replay", "all"):
        run_stage("replay", config, output_root, burnins, workers, sample_ids, args.resume)
    if args.stage in ("validate", "all"):
        validate_pairing(config, output_root, sample_ids)
    final = inventory(config, output_root, burnins)
    print_inventory(config, output_root, final)
    if args.analyze:
        if not all(row[0] for row in final["replays"].values()):
            raise RuntimeError("analysis requires all 120 verified replay tasks")
        import analyze_frozen_continuous_ramp

        analyze_frozen_continuous_ramp.analyze(config, output_root)
    print("[complete] frozen-record continuous-ramp command finished successfully")


if __name__ == "__main__":
    main()
