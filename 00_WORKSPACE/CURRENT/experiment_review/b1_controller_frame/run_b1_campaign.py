#!/usr/bin/env python3
"""Resumable CPU campaign for the B1 signed controller-frame pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics import (
    CANONICAL_CPU_ENTRY_POINT,
    CHANNELS,
    CHANNEL_TARGETS,
    TrajectoryActivityRecorder,
    active_cell_coordinates,
)


CONFIG_PATH = HERE / "campaign_config.v2.json"
RESULTS_ROOT = HERE / "results"
FIGURE_WIDTH = 3.375


CONSTRUCTIONS: dict[str, dict[str, Any]] = {
    "explicit_interface": {
        "role": "wall_on",
        "pair": "explicit",
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "dw_truncation": False,
        "meas_slab_only": False,
        "description": "topological--trivial controller interface; all centers active",
    },
    "explicit_wall_off": {
        "role": "wall_off",
        "pair": "explicit",
        "alpha_1": 30.0,
        "alpha_2": 30.0,
        "dw_truncation": False,
        "meas_slab_only": False,
        "description": "all-trivial full-geometry control",
    },
    "support_terminated": {
        "role": "wall_on",
        "pair": "support",
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "dw_truncation": True,
        "meas_slab_only": True,
        "description": "topological slab with hard support termination",
    },
    "support_wall_off": {
        "role": "wall_off",
        "pair": "support",
        "alpha_1": 30.0,
        "alpha_2": 30.0,
        "dw_truncation": True,
        "meas_slab_only": True,
        "description": "all-trivial control retaining the hard support boundary",
    },
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


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
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


def legacy_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(REPO_ROOT)), "exists": False}
    if path.is_file():
        return {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": True,
            "type": "file",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    entries = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = item.stat()
        entries.append((str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns))
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": True,
        "type": "directory",
        "file_count": len(entries),
        "total_bytes": sum(row[1] for row in entries),
        "metadata_tree_sha256": payload_hash(entries),
        "hash_scope": "relative path, byte size, and mtime_ns; individual source documents are content hashed",
    }


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def campaign_dir(campaign_id: str) -> Path:
    return RESULTS_ROOT / str(campaign_id)


def update_manifest(root: Path, *, stage: str | None = None, status: str | None = None,
                    cpu_list: str | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if stage is not None:
        manifest.setdefault("stages", {})[stage] = {
            "status": status,
            "updated_utc": utc_now(),
            **({"details": details} if details else {}),
        }
    if cpu_list is not None:
        manifest.setdefault("resource_history", []).append(
            {"stage": stage, "cpu_list": cpu_list, "recorded_utc": utc_now()}
        )
    manifest["updated_utc"] = utc_now()
    write_json_atomic(path, manifest)
    return manifest


def initialize(campaign_id: str) -> Path:
    root = campaign_dir(campaign_id)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("raw/static", "raw/trajectories", "processed/tables", "processed/sector_static",
                 "figures", "reports", "logs", "status"):
        (root / name).mkdir(parents=True, exist_ok=True)
    config = load_config()
    config_copy = root / CONFIG_PATH.name
    if config_copy.exists():
        if sha256_file(config_copy) != sha256_file(CONFIG_PATH):
            raise RuntimeError("Resume configuration differs from immutable campaign_config.v1.json")
    else:
        shutil.copy2(CONFIG_PATH, config_copy)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        source_paths = [
            CONFIG_PATH,
            HERE / "run_b1_campaign.py",
            HERE / "launch_b1_tmux.sh",
            HERE / "inside_b1_tmux.sh",
            REPO_ROOT / "src/fgtn/classA_U1FGTN.py",
            REPO_ROOT / "src/fgtn/diagnostics/static.py",
            REPO_ROOT / "src/fgtn/diagnostics/activity.py",
        ]
        legacy = [legacy_reference(REPO_ROOT / relative) for relative in config["legacy_paths"]]
        v0_files = sorted((REPO_ROOT / "validation_campaigns/results").glob(
            "v0_v1_cpu_*/v0/validation_summary.json"
        ))
        write_json_atomic(
            manifest_path,
            {
                "campaign_id": campaign_id,
                "campaign": config["campaign"],
                "schema_version": config["schema_version"],
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "canonical_dynamics_entry_point": CANONICAL_CPU_ENTRY_POINT,
                "dtype": config["dtype"],
                "config_sha256": sha256_file(CONFIG_PATH),
                "sources": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_paths},
                "legacy_references": legacy,
                "v0_dependency_candidates": [str(path.relative_to(REPO_ROOT)) for path in v0_files],
                "git": git_metadata(),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                },
                "constructions": CONSTRUCTIONS,
                "stages": {},
                "resource_history": [],
            },
        )
    return root


def _cpu_snapshot() -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        token = line.split()[0]
        if not token.startswith("cpu") or not token[3:].isdigit():
            continue
        fields = [int(value) for value in line.split()[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        result[int(token[3:])] = (sum(fields), idle)
    return result


def select_idle_cpus(limit: int, *, allow_busy: bool = False) -> str:
    first = _cpu_snapshot()
    time.sleep(0.6)
    second = _cpu_snapshot()
    candidates = []
    for cpu in sorted(second):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = int((topology / "core_id").read_text())
        package = int((topology / "physical_package_id").read_text())
        node_links = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
        node = int(node_links[0].name[4:]) if node_links else package
        total = second[cpu][0] - first[cpu][0]
        idle = second[cpu][1] - first[cpu][1]
        usage = 100.0 * (1.0 - idle / total) if total > 0 else 100.0
        candidates.append((usage, node, package, core, cpu))
    one_per_core: dict[tuple[int, int], tuple[float, int, int, int, int]] = {}
    for row in candidates:
        key = (row[2], row[3])
        if key not in one_per_core or row[0] < one_per_core[key][0]:
            one_per_core[key] = row
    threshold = float(load_config()["idle_cpu_threshold_percent"])
    eligible = [row for row in one_per_core.values() if allow_busy or row[0] <= threshold]
    eligible.sort(key=lambda row: (row[0], row[1], row[4]))
    selected = []
    nodes = sorted({row[1] for row in eligible})
    while len(selected) < int(limit):
        progressed = False
        for node in nodes:
            choices = [row for row in eligible if row[1] == node and row not in selected]
            if choices:
                selected.append(choices[0])
                progressed = True
                if len(selected) == int(limit):
                    break
        if not progressed:
            break
    if len(selected) < int(limit):
        return ""
    return ",".join(str(row[4]) for row in selected)


def model_for(label: str, nx: int, ny: int, nshell: int) -> classA_U1FGTN:
    spec = CONSTRUCTIONS[label]
    model = classA_U1FGTN(
        nx,
        ny,
        DW=True,
        nshell=nshell,
        alpha_1=spec["alpha_1"],
        alpha_2=spec["alpha_2"],
        trial_orbitals="X",
        dw_truncation=spec["dw_truncation"],
    )
    model.construct_OW_projectors(
        nshell=nshell,
        DW=True,
        trial_orbitals="X",
        dw_truncation=spec["dw_truncation"],
    )
    return model


def constraint_frame(model: classA_U1FGTN, *, meas_slab_only: bool) -> dict[str, np.ndarray]:
    active = np.asarray(model.active_top_layer_indices(meas_slab_only=meas_slab_only), dtype=np.int64)
    vectors, targets, channels, xs, ys = [], [], [], [], []
    for x, y in active_cell_coordinates(model, meas_slab_only=meas_slab_only):
        for channel_index, channel in enumerate(CHANNELS):
            vector = np.asarray(getattr(model, f"WF_{channel}")[:, x, y], dtype=np.complex128)[active]
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
            targets.append(CHANNEL_TARGETS[channel])
            channels.append(channel_index)
            xs.append(x)
            ys.append(y)
    return {
        "vectors": np.column_stack(vectors),
        "targets": np.asarray(targets, dtype=np.int8),
        "channel_indices": np.asarray(channels, dtype=np.int64),
        "center_x": np.asarray(xs, dtype=np.int64),
        "center_y": np.asarray(ys, dtype=np.int64),
        "active_indices": active,
    }


def frame_residual(mode_weights: np.ndarray, targets: np.ndarray, rank: int) -> np.ndarray:
    occupancy = np.sum(mode_weights[: int(rank)], axis=0)
    return np.where(targets == 1, 1.0 - occupancy, occupancy)


def degenerate_residual_bounds(
    mode_weights: np.ndarray,
    targets: np.ndarray,
    eigenvalues: np.ndarray,
    rank: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Bounds over all rank-r minimizers if the Fermi selection gap closes."""
    rank = int(rank)
    dimension = int(eigenvalues.size)
    residual = frame_residual(mode_weights, targets, rank)
    if rank == 0 or rank == dimension or eigenvalues[rank] - eigenvalues[rank - 1] > tolerance:
        return residual.copy(), residual.copy(), (rank, rank)
    energy = 0.5 * (eigenvalues[rank - 1] + eigenvalues[rank])
    left = rank - 1
    while left > 0 and abs(eigenvalues[left - 1] - energy) <= tolerance:
        left -= 1
    right = rank + 1
    while right < dimension and abs(eigenvalues[right] - energy) <= tolerance:
        right += 1
    base = np.sum(mode_weights[:left], axis=0)
    cluster = np.sum(mode_weights[left:right], axis=0)
    selected = rank - left
    cluster_dimension = right - left
    if selected == 0:
        occupancy_min = occupancy_max = base
    elif selected == cluster_dimension:
        occupancy_min = occupancy_max = base + cluster
    else:
        occupancy_min = base
        occupancy_max = base + cluster
    lower = np.where(targets == 1, 1.0 - occupancy_max, occupancy_min)
    upper = np.where(targets == 1, 1.0 - occupancy_min, occupancy_max)
    return lower, upper, (left, right)


def residual_map(residual: np.ndarray, frame: dict[str, np.ndarray], nx: int, ny: int) -> np.ndarray:
    output = np.full((nx, ny, len(CHANNELS)), np.nan, dtype=np.float64)
    output[frame["center_x"], frame["center_y"], frame["channel_indices"]] = residual
    return output


def _static_worker(task: dict[str, Any]) -> dict[str, Any]:
    label, nx, ny, nshell = task["label"], task["nx"], task["ny"], task["nshell"]
    spec = CONSTRUCTIONS[label]
    with threadpool_limits(limits=1):
        model = model_for(label, nx, ny, nshell)
        frame = constraint_frame(model, meas_slab_only=spec["meas_slab_only"])
        vectors, targets = frame["vectors"], frame["targets"]
        dimension = vectors.shape[0]
        signs = 1.0 - 2.0 * targets.astype(np.float64)
        operator = (vectors * signs[None, :]) @ vectors.conj().T
        operator = 0.5 * (operator + operator.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(operator)
        mode_weights = np.abs(eigenvectors.conj().T @ vectors) ** 2
        rank = dimension // 2
        residual = frame_residual(mode_weights, targets, rank)
        residual_lower, residual_upper, degenerate_cluster = degenerate_residual_bounds(
            mode_weights, targets, eigenvalues, rank, float(task["numerical_tolerance"])
        )
        f_direct = float(np.sum(residual))
        f_formula = float(np.sum(targets) + np.sum(eigenvalues[:rank]))
        wf = vectors[:, targets == 1]
        we = vectors[:, targets == 0]
        qf, sf, _ = np.linalg.svd(wf, full_matrices=False)
        qe, se, _ = np.linalg.svd(we, full_matrices=False)
        rtol = float(task["svd_rtol"])
        rf = int(np.count_nonzero(sf > rtol * sf[0])) if sf.size else 0
        re = int(np.count_nonzero(se > rtol * se[0])) if se.size else 0
        principal = np.linalg.svd(qf[:, :rf].conj().T @ qe[:, :re], compute_uv=False)
        hermiticity = float(np.linalg.norm(operator - operator.conj().T, ord="fro"))
        frame_digest = hashlib.sha256()
        frame_digest.update(np.ascontiguousarray(vectors).view(np.uint8))
        frame_digest.update(np.ascontiguousarray(targets).view(np.uint8))
        output = Path(task["output"])
        save_npz_atomic(
            output,
            dimension=np.asarray(dimension),
            target_rank=np.asarray(rank),
            active_indices=frame["active_indices"],
            operator_eigenvalues=eigenvalues,
            mode_constraint_weights=mode_weights,
            target_occupancies=targets,
            channel_indices=frame["channel_indices"],
            center_x=frame["center_x"],
            center_y=frame["center_y"],
            residuals_half_filling=residual,
            residual_map_half_filling=residual_map(residual, frame, nx, ny),
            residual_lower_half_filling=residual_lower,
            residual_upper_half_filling=residual_upper,
            residual_map_lower_half_filling=residual_map(residual_lower, frame, nx, ny),
            residual_map_upper_half_filling=residual_map(residual_upper, frame, nx, ny),
            degenerate_cluster=np.asarray(degenerate_cluster, dtype=np.int64),
            singular_values_filled=sf,
            singular_values_empty=se,
            principal_cosines=principal,
            selection_gap=np.asarray(eigenvalues[rank] - eigenvalues[rank - 1]),
            f_star=np.asarray(f_direct),
            f_star_formula=np.asarray(f_formula),
        )
        summary = {
            "status": "complete",
            "label": label,
            "Nx": nx,
            "Ny": ny,
            "dimension": dimension,
            "constraints": vectors.shape[1],
            "target_rank": rank,
            "frame_sha256": frame_digest.hexdigest(),
            "f_star": f_direct,
            "f_star_formula": f_formula,
            "cost_residual": abs(f_direct - f_formula),
            "selection_gap": float(eigenvalues[rank] - eigenvalues[rank - 1]),
            "selection_gap_degenerate": bool(
                eigenvalues[rank] - eigenvalues[rank - 1] <= task["numerical_tolerance"]
            ),
            "degenerate_cluster": list(degenerate_cluster),
            "sigma_max": float(principal[0]) if principal.size else 0.0,
            "hermiticity_residual": hermiticity,
            "active_indices_sha256": hashlib.sha256(frame["active_indices"].tobytes()).hexdigest(),
            "output": str(output),
        }
        write_json_atomic(output.with_suffix(".json"), summary)
        return summary


class ChargeObserver:
    def __init__(self, cycles: int, active_indices: np.ndarray) -> None:
        self.charge = np.full((cycles + 1,), np.nan, dtype=np.float64)
        self.active = np.asarray(active_indices, dtype=np.int64)

    def __call__(self, *, cycle: int, G: np.ndarray, **_: Any) -> None:
        matrix = np.asarray(G)
        if matrix.ndim == 3:
            matrix = matrix[0]
        self.charge[int(cycle)] = float(
            0.5 * (np.sum(np.real(np.diag(matrix)[self.active])) + self.active.size)
        )


def _trajectory_worker(task: dict[str, Any]) -> dict[str, Any]:
    output = Path(task["output"])
    summary_path = output.with_suffix(".json")
    if output.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("task_hash") == task["task_hash"]:
            return existing
    label = task["label"]
    spec = CONSTRUCTIONS[label]
    with threadpool_limits(limits=1):
        model = model_for(label, task["nx"], task["ny"], task["nshell"])
        active = np.asarray(
            model.active_top_layer_indices(meas_slab_only=spec["meas_slab_only"]), dtype=np.int64
        )
        recorder = TrajectoryActivityRecorder.from_model(
            model, cycles=task["cycles"], samples=1,
            meas_slab_only=spec["meas_slab_only"],
        )
        charge = ChargeObserver(task["cycles"], active)
        result = model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=task["cycles"],
            samples=1,
            init_mode=task["init_mode"],
            save=False,
            perfect_correction=True,
            sequence=task["sequence"],
            meas_slab_only=spec["meas_slab_only"],
            random_seed=task["seed"],
            cycle_observer=charge,
            trajectory_weight_observer=recorder,
            parallelize_samples=False,
        )
        recorder.assert_complete()
        payload = recorder.payload()
        save_npz_atomic(
            output,
            **payload,
            state_total_charge=charge.charge,
            active_indices=active,
        )
        observed = charge.charge[task["burn_in"] :]
        summary = {
            "status": "complete",
            "task_hash": task["task_hash"],
            "label": label,
            "Nx": task["nx"],
            "Ny": task["ny"],
            "split": task["split"],
            "sample": task["sample"],
            "root_seed": task["seed"],
            "canonical_sample_seed": result["sample_seeds"][0],
            "cycles": task["cycles"],
            "burn_in": task["burn_in"],
            "sequence": task["sequence"],
            "canonical_dynamics_entry_point": CANONICAL_CPU_ENTRY_POINT,
            "active_dimension": int(active.size),
            "charge_min": float(np.min(observed)),
            "charge_max": float(np.max(observed)),
            "max_charge_integer_residual": float(np.max(np.abs(observed - np.rint(observed)))),
            "defect_rate_observed": float(np.mean(recorder.defect[:, task["burn_in"] :])),
            "x_equals_abs_y": bool(np.array_equal(recorder.defect, np.abs(recorder.transfer))),
            "output": str(output),
        }
        write_json_atomic(summary_path, summary)
        return summary


def stage_preflight(root: Path, cpu_list: str) -> None:
    update_manifest(root, stage="preflight", status="running", cpu_list=cpu_list)
    candidates = sorted((REPO_ROOT / "validation_campaigns/results").glob(
        "v0_v1_cpu_*/v0/validation_summary.json"
    ))
    passing = []
    for path in candidates:
        payload = json.loads(path.read_text())
        if str(payload.get("status", "")).startswith("CPU_PASS"):
            passing.append(path)
    if not passing:
        raise RuntimeError("B1 is blocked: no completed CPU_PASS V0 validation was found")
    checks = []
    for label in CONSTRUCTIONS:
        model = model_for(label, 4, 6, 1)
        spec = CONSTRUCTIONS[label]
        active = model.active_top_layer_indices(meas_slab_only=spec["meas_slab_only"])
        norms = []
        for channel in CHANNELS:
            frame = np.asarray(getattr(model, f"WF_{channel}"))
            norms.append(float(np.max(np.abs(np.sum(np.abs(frame) ** 2, axis=0) - 1.0))))
        checks.append({"label": label, "active_dimension": len(active), "max_projector_norm_error": max(norms)})
    tolerance = float(load_config()["numerical_tolerance"])
    if max(row["max_projector_norm_error"] for row in checks) > tolerance:
        raise RuntimeError(f"Controller projector normalization failed: {checks}")
    payload = {
        "status": "pass",
        "v0_dependency": str(passing[-1].relative_to(REPO_ROOT)),
        "v0_sha256": sha256_file(passing[-1]),
        "construction_checks": checks,
        "canonical_dynamics_entry_point": CANONICAL_CPU_ENTRY_POINT,
    }
    write_json_atomic(root / "status/preflight.json", payload)
    update_manifest(root, stage="preflight", status="complete", details=payload)


def stage_static(root: Path, cpu_list: str, workers: int) -> None:
    config = load_config()
    update_manifest(root, stage="static", status="running", cpu_list=cpu_list)
    tasks = []
    for label in config["constructions"]:
        for ny in config["ny_values"]:
            output = root / f"raw/static/{label}/N{config['nx']}x{ny}.npz"
            summary = output.with_suffix(".json")
            if summary.exists() and output.exists() and json.loads(summary.read_text()).get("status") == "complete":
                continue
            tasks.append({
                "label": label,
                "nx": config["nx"],
                "ny": ny,
                "nshell": config["nshell"],
                "svd_rtol": config["svd_rtol"],
                "numerical_tolerance": config["numerical_tolerance"],
                "output": str(output),
            })
    summaries = []
    if tasks:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = [pool.submit(_static_worker, task) for task in tasks]
            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                print(f"[static] complete {summary['label']} N{summary['Nx']}x{summary['Ny']}", flush=True)
    all_summaries = [json.loads(path.read_text()) for path in sorted((root / "raw/static").rglob("*.json"))]
    expected = len(config["constructions"]) * len(config["ny_values"])
    if len(all_summaries) != expected:
        raise RuntimeError(f"Static stage has {len(all_summaries)} of {expected} cases")
    tolerance = float(config["numerical_tolerance"])
    for row in all_summaries:
        if row["cost_residual"] > tolerance or row["hermiticity_residual"] > tolerance:
            raise RuntimeError(f"Static numerical identity failed: {row}")
    write_csv_atomic(root / "processed/tables/static_summary.csv", all_summaries)
    update_manifest(root, stage="static", status="complete", details={"cases": expected})


def trajectory_tasks(root: Path) -> list[dict[str, Any]]:
    config = load_config()
    seed_sequences = np.random.SeedSequence(config["root_seed"]).spawn(
        len(config["constructions"]) * len(config["ny_values"]) * sum(config["trajectory_splits"].values())
    )
    seeds = iter(int(sequence.generate_state(1, dtype=np.uint64)[0]) for sequence in seed_sequences)
    tasks = []
    for label in config["constructions"]:
        for ny in config["ny_values"]:
            for split, samples in config["trajectory_splits"].items():
                for sample in range(samples):
                    output = root / f"raw/trajectories/{label}/N{config['nx']}x{ny}/{split}/trajectory_{sample:03d}.npz"
                    task = {
                        "label": label,
                        "nx": config["nx"],
                        "ny": ny,
                        "nshell": config["nshell"],
                        "split": split,
                        "sample": sample,
                        "seed": next(seeds),
                        "cycles": config["cycles_factor"] * ny,
                        "burn_in": config["burn_in_factor"] * ny,
                        "sequence": config["sequence"],
                        "init_mode": config["init_mode"],
                        "output": str(output),
                    }
                    task["task_hash"] = payload_hash({key: value for key, value in task.items() if key != "output"})
                    tasks.append(task)
    return tasks


def stage_trajectory(root: Path, cpu_list: str, workers: int) -> None:
    config = load_config()
    update_manifest(root, stage="trajectory", status="running", cpu_list=cpu_list)
    tasks = trajectory_tasks(root)
    pending = []
    for task in tasks:
        output = Path(task["output"])
        summary = output.with_suffix(".json")
        if summary.exists() and output.exists():
            payload = json.loads(summary.read_text())
            if payload.get("status") == "complete" and payload.get("task_hash") == task["task_hash"]:
                continue
        pending.append(task)
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = [pool.submit(_trajectory_worker, task) for task in pending]
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                print(
                    f"[trajectory] {index}/{len(pending)} {row['label']} N{row['Nx']}x{row['Ny']} "
                    f"{row['split']}:{row['sample']}", flush=True
                )
    summaries = [json.loads(Path(task["output"]).with_suffix(".json").read_text()) for task in tasks]
    if len({row["root_seed"] for row in summaries}) != len(summaries):
        raise RuntimeError("Training/test or geometry root seeds are not disjoint")
    tolerance = float(config["charge_integer_tolerance"])
    failures = [row for row in summaries if not row["x_equals_abs_y"] or row["max_charge_integer_residual"] > tolerance]
    if failures:
        raise RuntimeError(f"Trajectory bookkeeping failed: {failures}")
    write_csv_atomic(root / "processed/tables/trajectory_summary.csv", summaries)
    update_manifest(root, stage="trajectory", status="complete", details={"trajectories": len(summaries)})


def profile_from_shard(path: Path, burn_in: int, nx: int) -> np.ndarray:
    with np.load(path) as data:
        defect = np.asarray(data["defect_X"])[0, burn_in:]
        valid = np.asarray(data["valid"])[0, burn_in:]
        site_x = np.asarray(data["site_x"])
    profile = np.full((nx,), np.nan, dtype=np.float64)
    for x in range(nx):
        selected = site_x == x
        denominator = int(np.sum(valid[:, selected, :]))
        if denominator:
            profile[x] = float(np.sum(defect[:, selected, :] & valid[:, selected, :]) / denominator)
    return profile


def static_sector_profile(
    static_path: Path,
    sector_counts: dict[int, int],
    nx: int,
    ny: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(static_path) as data:
        weights = np.asarray(data["mode_constraint_weights"])
        targets = np.asarray(data["target_occupancies"])
        xs = np.asarray(data["center_x"])
        ys = np.asarray(data["center_y"])
        channels = np.asarray(data["channel_indices"])
        dimension = int(data["dimension"])
        eigenvalues = np.asarray(data["operator_eigenvalues"])
    ranks = np.asarray(sorted(sector_counts), dtype=np.int64)
    profiles = np.full((ranks.size, nx), np.nan, dtype=np.float64)
    lower_profiles = np.full((ranks.size, nx), np.nan, dtype=np.float64)
    upper_profiles = np.full((ranks.size, nx), np.nan, dtype=np.float64)
    selection_gaps = np.full((ranks.size,), np.inf, dtype=np.float64)
    maps = np.full((ranks.size, nx, ny, len(CHANNELS)), np.nan, dtype=np.float64)
    for index, rank in enumerate(ranks):
        if not 0 <= rank <= dimension:
            raise RuntimeError(f"Observed active charge sector {rank} lies outside 0..{dimension}")
        residual = frame_residual(weights, targets, int(rank))
        lower, upper, _ = degenerate_residual_bounds(
            weights, targets, eigenvalues, int(rank), tolerance
        )
        if 0 < rank < dimension:
            selection_gaps[index] = eigenvalues[rank] - eigenvalues[rank - 1]
        maps[index, xs, ys, channels] = residual
        for x in range(nx):
            values = maps[index, x]
            finite = np.isfinite(values)
            if np.any(finite):
                profiles[index, x] = float(np.mean(values[finite]))
            selected_x = xs == x
            if np.any(selected_x):
                lower_profiles[index, x] = float(np.mean(lower[selected_x]))
                upper_profiles[index, x] = float(np.mean(upper[selected_x]))
    probabilities = np.asarray([sector_counts[int(rank)] for rank in ranks], dtype=np.float64)
    probabilities /= np.sum(probabilities)
    weighted = np.nansum(profiles * probabilities[:, None], axis=0)
    active = np.any(np.isfinite(profiles), axis=0)
    weighted[~active] = np.nan
    return ranks, profiles, weighted, lower_profiles, upper_profiles, selection_gaps, probabilities


def cycle_defect_rates(path: Path, burn_in: int) -> np.ndarray:
    with np.load(path) as data:
        defect = np.asarray(data["defect_X"])[0, burn_in:]
        valid = np.asarray(data["valid"])[0, burn_in:]
    numerator = np.sum(defect & valid, axis=(1, 2))
    denominator = np.sum(valid, axis=(1, 2))
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def pearson_overlap(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return np.nan
    a, b = left[finite], right[finite]
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def wall_masks(nx: int, active: np.ndarray, wall_x: tuple[int, int], width: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx)
    distance = np.minimum.reduce([
        np.abs(x - wall_x[0]), nx - np.abs(x - wall_x[0]),
        np.abs(x - wall_x[1]), nx - np.abs(x - wall_x[1]),
    ])
    wall = active & (distance < width)
    bulk = active & ~wall
    return wall, bulk


def contrast(profile: np.ndarray, wall: np.ndarray, bulk: np.ndarray) -> tuple[float, float]:
    wall_mean = float(np.nanmean(profile[wall]))
    bulk_mean = float(np.nanmean(profile[bulk]))
    bounded = (wall_mean - bulk_mean) / (wall_mean + bulk_mean) if wall_mean + bulk_mean > 0 else np.nan
    ratio = wall_mean / bulk_mean if bulk_mean > 0 else np.inf
    return float(bounded), float(ratio)


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8,
        "mathtext.fontset": "cm",
        "savefig.dpi": 300,
    })


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(root / "figures" / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / "figures" / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def stage_analyze(root: Path, cpu_list: str) -> None:
    config = load_config()
    update_manifest(root, stage="analyze", status="running", cpu_list=cpu_list)
    rows, bootstraps = [], {}
    for label in config["constructions"]:
        for ny in config["ny_values"]:
            base = root / f"raw/trajectories/{label}/N{config['nx']}x{ny}"
            train_paths = sorted((base / "train").glob("trajectory_*.npz"))
            test_paths = sorted((base / "test").glob("trajectory_*.npz"))
            burn_in = config["burn_in_factor"] * ny
            charges = []
            for path in train_paths:
                with np.load(path) as data:
                    values = np.asarray(data["state_total_charge"])[burn_in:]
                residual = np.max(np.abs(values - np.rint(values)))
                if residual > config["charge_integer_tolerance"]:
                    raise RuntimeError(f"Noninteger active charge in {path}: {residual}")
                charges.extend(np.rint(values).astype(int).tolist())
            unique, counts = np.unique(charges, return_counts=True)
            sector_counts = {int(rank): int(count) for rank, count in zip(unique, counts)}
            static_path = root / f"raw/static/{label}/N{config['nx']}x{ny}.npz"
            (
                ranks,
                sector_profiles,
                static_profile,
                sector_lower_profiles,
                sector_upper_profiles,
                selection_gaps,
                sector_probabilities,
            ) = static_sector_profile(
                static_path,
                sector_counts,
                config["nx"],
                ny,
                config["numerical_tolerance"],
            )
            test_profiles = np.stack([
                profile_from_shard(path, burn_in, config["nx"]) for path in test_paths
            ])
            mean_activity = np.nanmean(test_profiles, axis=0)
            cycle_rates = np.stack([cycle_defect_rates(path, burn_in) for path in test_paths])
            mean_cycle_rate = np.nanmean(cycle_rates, axis=0)
            midpoint = mean_cycle_rate.size // 2
            rate_first = float(np.nanmean(mean_cycle_rate[:midpoint]))
            rate_second = float(np.nanmean(mean_cycle_rate[midpoint:]))
            stationarity_delta = rate_second - rate_first
            stationarity_tolerance = max(0.1 * float(np.nanmean(mean_cycle_rate)), 1e-3)
            stationarity_pass = bool(abs(stationarity_delta) <= stationarity_tolerance)
            active = np.isfinite(static_profile) & np.isfinite(mean_activity)
            model = model_for(label, config["nx"], ny, config["nshell"])
            walls = tuple(sorted(int(value) for value in model.DW_loc))
            wall, bulk = wall_masks(
                config["nx"], active, walls, config["wall_window_columns"]
            )
            overlap = pearson_overlap(static_profile, mean_activity)
            static_contrast, static_ratio = contrast(static_profile, wall, bulk)
            activity_contrast, activity_ratio = contrast(mean_activity, wall, bulk)
            rng = np.random.default_rng(config["root_seed"] + 1000 * ny + list(CONSTRUCTIONS).index(label))
            boot_overlap = np.full(config["bootstrap_samples"], np.nan)
            boot_contrast = np.full(config["bootstrap_samples"], np.nan)
            for draw in range(config["bootstrap_samples"]):
                sample = test_profiles[rng.integers(0, len(test_profiles), size=len(test_profiles))]
                profile = np.nanmean(sample, axis=0)
                boot_overlap[draw] = pearson_overlap(static_profile, profile)
                boot_contrast[draw] = contrast(profile, wall, bulk)[0]
            key = (label, ny)
            bootstraps[key] = {"overlap": boot_overlap, "contrast": boot_contrast}
            sector_output = root / f"processed/sector_static/{label}/N{config['nx']}x{ny}.npz"
            save_npz_atomic(
                sector_output,
                charge_sectors=ranks,
                sector_counts=np.asarray([sector_counts[int(rank)] for rank in ranks]),
                sector_residual_profiles=sector_profiles,
                sector_residual_lower_profiles=sector_lower_profiles,
                sector_residual_upper_profiles=sector_upper_profiles,
                sector_selection_gaps=selection_gaps,
                sector_probabilities=sector_probabilities,
                weighted_static_profile=static_profile,
                test_activity_profiles=test_profiles,
                mean_test_activity_profile=mean_activity,
                wall_mask=wall,
                bulk_mask=bulk,
                bootstrap_overlap=boot_overlap,
                bootstrap_activity_contrast=boot_contrast,
                test_cycle_defect_rates=cycle_rates,
                mean_test_cycle_defect_rate=mean_cycle_rate,
            )
            rows.append({
                "label": label,
                "pair": CONSTRUCTIONS[label]["pair"],
                "role": CONSTRUCTIONS[label]["role"],
                "Nx": config["nx"],
                "Ny": ny,
                "charge_sectors": ";".join(str(rank) for rank in ranks),
                "profile_overlap": overlap,
                "profile_overlap_ci_low": float(np.nanquantile(boot_overlap, 0.025)),
                "profile_overlap_ci_high": float(np.nanquantile(boot_overlap, 0.975)),
                "static_wall_bulk_contrast": static_contrast,
                "static_wall_bulk_ratio": static_ratio,
                "activity_wall_bulk_contrast": activity_contrast,
                "activity_wall_bulk_ratio": activity_ratio,
                "activity_contrast_ci_low": float(np.nanquantile(boot_contrast, 0.025)),
                "activity_contrast_ci_high": float(np.nanquantile(boot_contrast, 0.975)),
                "minimum_selection_gap": float(np.min(selection_gaps)),
                "degenerate_sector_count": int(np.count_nonzero(selection_gaps <= config["numerical_tolerance"])),
                "stationarity_first_half_rate": rate_first,
                "stationarity_second_half_rate": rate_second,
                "stationarity_delta": stationarity_delta,
                "stationarity_tolerance": stationarity_tolerance,
                "stationarity_pass": stationarity_pass,
            })
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.45))
            ax.plot(static_profile, marker="o", ms=2, lw=0.9, label="static sector-weighted residual")
            ax.plot(mean_activity, marker="s", ms=2, lw=0.9, label="disjoint test activity")
            ax.axvspan(walls[0] - 0.5, walls[0] + 0.5, color="0.85", zorder=-1)
            ax.axvspan(walls[1] - 0.5, walls[1] + 0.5, color="0.85", zorder=-1)
            ax.set(xlabel=r"$x$", ylabel="constraint defect", title=f"{label.replace('_', ' ')}, $N_y={ny}$")
            ax.legend(frameon=False, fontsize=6)
            save_figure(fig, root, f"profiles_{label}_Ny{ny}")
    comparison_rows = []
    for pair, on, off in (
        ("explicit", "explicit_interface", "explicit_wall_off"),
        ("support", "support_terminated", "support_wall_off"),
    ):
        for ny in config["ny_values"]:
            overlap_diff = bootstraps[(on, ny)]["overlap"] - bootstraps[(off, ny)]["overlap"]
            contrast_diff = bootstraps[(on, ny)]["contrast"] - bootstraps[(off, ny)]["contrast"]
            comparison_rows.append({
                "pair": pair,
                "Nx": config["nx"],
                "Ny": ny,
                "overlap_difference_mean": float(np.nanmean(overlap_diff)),
                "overlap_difference_ci_low": float(np.nanquantile(overlap_diff, 0.025)),
                "overlap_difference_ci_high": float(np.nanquantile(overlap_diff, 0.975)),
                "activity_contrast_difference_mean": float(np.nanmean(contrast_diff)),
                "activity_contrast_difference_ci_low": float(np.nanquantile(contrast_diff, 0.025)),
                "activity_contrast_difference_ci_high": float(np.nanquantile(contrast_diff, 0.975)),
                "descriptive_expansion_signal": bool(
                    np.nanquantile(overlap_diff, 0.025) > 0
                    and np.nanquantile(contrast_diff, 0.025) > 0
                ),
                "decision_role": "descriptive_human_review_only",
            })
    write_csv_atomic(root / "processed/tables/profile_metrics.csv", rows)
    write_csv_atomic(root / "processed/tables/wall_on_vs_control.csv", comparison_rows)
    write_json_atomic(root / "processed/analysis_summary.json", {
        "status": "complete",
        "profile_metric": "Pearson correlation across fixed x origins; no shifts or refits",
        "contrast_metric": "(wall_mean-bulk_mean)/(wall_mean+bulk_mean)",
        "bootstrap_unit": "whole disjoint test trajectory",
        "bootstrap_draws": config["bootstrap_samples"],
        "rows": rows,
        "comparisons": comparison_rows,
        "interpretation": "descriptive pilot; no automatic expansion",
    })
    update_manifest(root, stage="analyze", status="complete", details={"cases": len(rows)})


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def stage_report(root: Path, cpu_list: str) -> None:
    update_manifest(root, stage="report", status="running", cpu_list=cpu_list)
    analysis = json.loads((root / "processed/analysis_summary.json").read_text())
    comparison_lines = []
    for row in analysis["comparisons"]:
        comparison_lines.append(
            f"{row['pair']} & {row['Ny']} & "
            f"{row['overlap_difference_mean']:.3f} "
            f"[{row['overlap_difference_ci_low']:.3f},{row['overlap_difference_ci_high']:.3f}] & "
            f"{row['activity_contrast_difference_mean']:.3f} "
            f"[{row['activity_contrast_difference_ci_low']:.3f},{row['activity_contrast_difference_ci_high']:.3f}] \\\\"
        )
    figures = sorted((root / "figures").glob("profiles_*_Ny40.pdf"))
    figure_tex = "\n".join(
        "\\includegraphics[width=0.48\\textwidth]{"
        + latex_escape(os.path.relpath(path, root / "reports"))
        + "}"
        for path in figures
    )
    tex = rf"""\documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{{revtex4-2}}
\usepackage{{amsmath,amssymb,amsthm,mathtools,bm}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{{hyperref}}
\usepackage{{microtype}}
\setcounter{{tocdepth}}{{2}}
\begin{{document}}
\title{{B1 Signed Controller-Frame Incompatibility Pilot}}
\author{{Numerical campaign review}}
\date{{\today}}
\begin{{abstract}}
We test whether a static fixed-charge controller-frame residual predicts disjoint-test
wrong-outcome activity more specifically than a matched wall-off controller. This is an
exploratory mechanism diagnostic, not a topology, CFT, tangent-spectrum, or mean-channel claim.
\end{{abstract}}
\maketitle
\tableofcontents

\section{{Protocol}}
For normalized controller projectors $P_j$ and targets $s_j$, we diagonalize
$H_{{\rm frame}}=\sum_j(1-2s_j)P_j$ and evaluate the fixed-rank Ky Fan minimizer.
Four independent stationary training trajectories select the encountered charge sectors.
Static residual profiles fixed from training are compared, without shifting or fitting
their origins, with four disjoint test trajectories generated by
\texttt{{classA\_U1FGTN.run\_markov\_circuit}}. Whole trajectories are the bootstrap unit.

\section{{Results}}
The table reports wall-on minus matched-control differences. Intervals are 95\% bootstrap
intervals. They are descriptive and do not automatically authorize a larger sweep.
\begin{{center}}
\begin{{tabular}}{{lrrr}}
\toprule
construction & $N_y$ & profile-overlap difference & activity-contrast difference \\
\midrule
{chr(10).join(comparison_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\begin{{figure}}[p]
\centering
{figure_tex}
\caption{{Sector-weighted static residual and disjoint-test wrong-outcome activity at $N_y=40$.
Gray bands mark the fixed controller-wall origins.}}
\end{{figure}}

\section{{Interpretive limits}}
Nonzero incompatibility is generic for nonorthogonal or overcomplete constraints. The
frame spectrum is not flattened to $\pm1$, is not the conditioned tangent operator, and
is not the ensemble mean channel. No CFT, topology, twist, or chirality analysis is applied
to its minimizer. All raw arrays and exact scalar tables are stored beside this report.
\end{{document}}
"""
    tex_path = root / "reports/b1_controller_frame_report.tex"
    tex_path.write_text(tex)
    compile_result = {"attempted": False, "returncode": None}
    if shutil.which("latexmk"):
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        (root / "logs/report_compile.log").write_text(result.stdout + "\n" + result.stderr)
        compile_result = {"attempted": True, "returncode": result.returncode}
        if result.returncode:
            raise RuntimeError("B1 report failed to compile; see logs/report_compile.log")
    write_json_atomic(root / "status/report.json", {
        "status": "complete",
        "tex": str(tex_path),
        "pdf": str(tex_path.with_suffix(".pdf")),
        "compile": compile_result,
    })
    update_manifest(root, stage="report", status="complete", details=compile_result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("init", "preflight", "static", "trajectory", "analyze", "report"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--cpu-list", default="")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--select-idle-cpus", action="store_true")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--allow-busy", action="store_true")
    return parser


def main() -> int:
    configure_plotting()
    args = build_parser().parse_args()
    if args.select_idle_cpus:
        print(select_idle_cpus(args.limit, allow_busy=args.allow_busy))
        return 0
    if not args.stage or not args.campaign_id:
        raise SystemExit("stage and --campaign-id are required")
    if not 1 <= args.max_workers <= 4:
        raise SystemExit("--max-workers must be 1 through 4")
    root = initialize(args.campaign_id)
    try:
        if args.stage == "init":
            return 0
        if args.stage == "preflight":
            stage_preflight(root, args.cpu_list)
        elif args.stage == "static":
            stage_static(root, args.cpu_list, args.max_workers)
        elif args.stage == "trajectory":
            stage_trajectory(root, args.cpu_list, args.max_workers)
        elif args.stage == "analyze":
            stage_analyze(root, args.cpu_list)
        elif args.stage == "report":
            stage_report(root, args.cpu_list)
    except Exception as error:
        update_manifest(root, stage=args.stage, status="failed", details={"error": repr(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
