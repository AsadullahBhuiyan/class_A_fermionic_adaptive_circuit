#!/usr/bin/env python3
"""Offline radius-stability analysis of saved domain-wall endpoint frames.

No dynamics are run.  Every source result/completion pair is checksum verified,
then the legacy periodic three-wedge Chern estimator is evaluated directly from
the occupied frame V through Gamma=(V V^dagger)^T=V* V^T.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
IMPORTED_ROOT = (
    PROJECT_ROOT
    / "imported_endpoints/wall_pump_width_endpoints_s100_v1"
)
OUTPUT_ROOT = RESULTS_ROOT / "endpoint_chern_radius_stability_alpha1_v1"
ANALYSIS_ROOT = OUTPUT_ROOT / "analysis"
FIGURE_ROOT = ANALYSIS_ROOT / "figures"

NEW_CONFIG = (
    PROJECT_ROOT
    / "../final_production_new_designs/11_wall_pump_width_endpoints/campaign_config.json"
).resolve()
OLD_CAMPAIGNS = {
    24: (
        "N20x24_state_projector_pump_s100_v1",
        PROJECT_ROOT / "campaign_config.state_projector_pump_n20x24_s100_v1.json",
    ),
    28: (
        "N20x28_state_projector_pump_s100_v3",
        PROJECT_ROOT / "campaign_config.state_projector_pump_n20x28_s100_v3.json",
    ),
    30: (
        "N20x30_state_projector_pump_s100_v3",
        PROJECT_ROOT / "campaign_config.state_projector_pump_n20x30_s100_v3.json",
    ),
}
REFERENCE_NPZ = (
    RESULTS_ROOT
    / "N20_state_projector_pump_Ny24_36_s100_series_v3"
    / "analysis/bulk_chern_pump_relation_v1/ny24_ny28_bulk_chern_by_y0.npz"
)

WALLS = ("soft", "hard")
PROTOCOLS = ("nsh1", "dense")
GRAM_TOLERANCE = 1.0e-8
FACTOR_TOLERANCE = 2.0e-11
REFERENCE_TOLERANCE = 2.0e-11
SUMMARY_SCHEMA = "endpoint_chern_radius_stability_summary_v1"


@dataclass(frozen=True)
class SourceTask:
    cohort: str
    campaign: str
    collection: str
    protocol: str
    nshell_label: str
    nx: int
    ny: int
    wall: str
    result_path: str
    completion_path: str
    sample_ids: tuple[int, ...]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_campaign_configs() -> dict[str, Any]:
    new = load_json(NEW_CONFIG)
    require(new.get("sampling_revision") == "wall_pump_width_endpoints_s100_v1", "new revision mismatch")
    require(new.get("Ny") == 24 and new.get("cycles") == 48, "new geometry/cycles mismatch")
    protocol = new.get("protocol", {})
    require(protocol.get("DW") is True, "new campaign is not a domain-wall campaign")
    require(protocol.get("alpha_1") == 1.0 and protocol.get("alpha_2") == 30.0, "new alpha mismatch")
    require(protocol.get("perfect_correction") is True, "new perfect-correction mismatch")
    require(protocol.get("state_representation") == "physical_frame", "new state representation mismatch")
    require(new.get("dtype") == "complex128", "new dtype mismatch")
    require(new.get("shells") == {"nsh1": 1, "dense": None}, "new shell map mismatch")

    old: dict[str, Any] = {}
    for ny, (campaign, path) in OLD_CAMPAIGNS.items():
        config = load_json(path)
        geometry = config.get("geometry", {})
        dynamics = config.get("dynamics", {})
        require(config.get("campaign_id") == campaign, f"old campaign ID mismatch for Ny={ny}")
        require(geometry.get("Nx") == 20 and geometry.get("Ny") == ny, f"old geometry mismatch for Ny={ny}")
        require(geometry.get("DW") is True and geometry.get("dw_interval") == [5, 15], f"old wall mismatch for Ny={ny}")
        require(geometry.get("nshell") == 1, f"old shell mismatch for Ny={ny}")
        require(geometry.get("alpha_1") == 1.0 and geometry.get("alpha_2") == 30.0, f"old alpha mismatch for Ny={ny}")
        require(dynamics.get("burn_in_cycles") == 2 * ny, f"old cycle mismatch for Ny={ny}")
        require(dynamics.get("perfect_correction") is True, f"old correction mismatch for Ny={ny}")
        require(dynamics.get("dtype") == "complex128", f"old dtype mismatch for Ny={ny}")
        require(dynamics.get("state_representation") == "physical_frame", f"old representation mismatch for Ny={ny}")
        old[str(ny)] = {
            "campaign": campaign,
            "path": str(path.relative_to(PROJECT_ROOT.parent.parent.parent)),
            "sha256": sha256_path(path),
            "canonical_json_sha256": canonical_json_sha256(config),
        }
    return {
        "new": {
            "path": str(NEW_CONFIG.relative_to(PROJECT_ROOT.parent.parent.parent)),
            "sha256": sha256_path(NEW_CONFIG),
            "canonical_json_sha256": canonical_json_sha256(new),
        },
        "old": old,
    }


def discover_tasks() -> list[SourceTask]:
    tasks: list[SourceTask] = []
    expected_new_cells = {
        ("bridge", "nsh1", 20): 25,
        ("bridge", "nsh1", 24): 25,
        ("bridge", "dense", 20): 25,
        ("endpoints", "nsh1", 28): 100,
        ("endpoints", "nsh1", 32): 100,
        ("endpoints", "dense", 24): 100,
        ("endpoints", "dense", 28): 100,
        ("endpoints", "dense", 32): 100,
    }
    for (collection, protocol, nx), sample_count in expected_new_cells.items():
        for wall in WALLS:
            directory = IMPORTED_ROOT / collection / protocol / f"N{nx}x24" / wall
            shards = sorted(directory.glob("shard_*.npz"))
            expected_shards = sample_count // 5
            require(len(shards) == expected_shards, f"expected {expected_shards} shards in {directory}, found {len(shards)}")
            observed: list[int] = []
            for shard_index, result_path in enumerate(shards):
                require(result_path.stem == f"shard_{shard_index:02d}", f"noncontiguous shard index: {result_path}")
                sample_ids = tuple(range(5 * shard_index, 5 * shard_index + 5))
                observed.extend(sample_ids)
                tasks.append(
                    SourceTask(
                        cohort="new_width_sweep",
                        campaign="wall_pump_width_endpoints_s100_v1",
                        collection=collection,
                        protocol=protocol,
                        nshell_label="1" if protocol == "nsh1" else "infinity",
                        nx=nx,
                        ny=24,
                        wall=wall,
                        result_path=str(result_path),
                        completion_path=str(result_path.with_suffix(".completion.json")),
                        sample_ids=sample_ids,
                    )
                )
            require(observed == list(range(sample_count)), f"sample coverage mismatch in {directory}")

    for ny, (campaign, _) in OLD_CAMPAIGNS.items():
        for wall in WALLS:
            directory = RESULTS_ROOT / campaign / "burnins" / wall
            results = sorted(directory.glob("sample_*.npz"))
            require(len(results) == 100, f"expected 100 old endpoints in {directory}, found {len(results)}")
            for sample_id, result_path in enumerate(results):
                require(result_path.stem == f"sample_{sample_id:03d}", f"old sample index mismatch: {result_path}")
                tasks.append(
                    SourceTask(
                        cohort="old_fixed_nx",
                        campaign=campaign,
                        collection="burnins",
                        protocol="nsh1",
                        nshell_label="1",
                        nx=20,
                        ny=ny,
                        wall=wall,
                        result_path=str(result_path),
                        completion_path=str(result_path.with_suffix(".completion.json")),
                        sample_ids=(sample_id,),
                    )
                )
    require(len(tasks) == 830, f"expected 830 source tasks, found {len(tasks)}")
    require(sum(len(task.sample_ids) for task in tasks if task.cohort == "new_width_sweep") == 1150, "new trajectory count mismatch")
    require(sum(len(task.sample_ids) for task in tasks if task.cohort == "old_fixed_nx") == 600, "old trajectory count mismatch")
    return tasks


def radii_for(task: SourceTask) -> tuple[float, ...]:
    if task.cohort == "old_fixed_nx":
        return tuple(float(value) for value in range(2, 9))
    legacy = 0.4 * min(task.nx, task.ny)
    values = [float(value) for value in range(2, int(math.ceil(legacy)) + 1)]
    values.append(float(legacy))
    return tuple(sorted(set(round(value, 12) for value in values)))


_PARTITION_CACHE: dict[tuple[int, int, float], tuple[torch.Tensor, ...]] = {}


def partition_indices(
    *, nx: int, ny: int, xref: int, yref: int, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=np.int64)
    y = np.arange(ny, dtype=np.int64)
    dx = (x - int(xref) + nx // 2) % nx - nx // 2
    dy = (y - int(yref) + ny // 2) % ny - ny // 2
    dx_grid, dy_grid = np.meshgrid(dx, dy, indexing="ij")
    inside = dx_grid * dx_grid + dy_grid * dy_grid <= float(radius) ** 2
    theta = np.mod(np.arctan2(dy_grid, dx_grid), 2.0 * np.pi)
    bounds = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 2.0 * np.pi)

    def indices(mask: np.ndarray) -> np.ndarray:
        xx, yy = np.nonzero(mask)
        orbital_zero = 2 * xx + 2 * nx * yy
        return np.sort(np.concatenate((orbital_zero, orbital_zero + 1))).astype(np.int64)

    result = tuple(
        indices(inside & (theta >= bounds[index]) & (theta < bounds[index + 1]))
        for index in range(3)
    )
    require(min(len(item) for item in result) > 0, f"empty Chern sector for Nx={nx}, Ny={ny}, R={radius}")
    return result  # type: ignore[return-value]


def batched_partitions(nx: int, ny: int, radius: float) -> tuple[torch.Tensor, ...]:
    key = (nx, ny, radius)
    cached = _PARTITION_CACHE.get(key)
    if cached is not None:
        return cached
    rows = [
        partition_indices(nx=nx, ny=ny, xref=nx // 2, yref=yref, radius=radius)
        for yref in range(ny)
    ]
    result = tuple(
        torch.as_tensor(np.stack([row[sector] for row in rows]), dtype=torch.long)
        for sector in range(3)
    )
    _PARTITION_CACHE[key] = result
    return result


def chern_by_y0(frame: np.ndarray, partitions: tuple[torch.Tensor, ...]) -> np.ndarray:
    """Evaluate the legacy estimator from Gamma=V* V^T without forming Gamma."""
    require(frame.ndim == 2 and frame.dtype == np.complex128, "frame must be a complex128 matrix")
    occupied = torch.from_numpy(np.ascontiguousarray(frame)).conj()
    a, b, c = partitions
    w_a, w_b, w_c = occupied[a], occupied[b], occupied[c]
    p_ca = w_c @ w_a.mH
    p_ab = w_a @ w_b.mH
    p_bc = w_b @ w_c.mH
    p_ac = w_a @ w_c.mH
    p_cb = w_c @ w_b.mH
    p_ba = w_b @ w_a.mH
    first = torch.diagonal(p_ca @ p_ab @ p_bc, dim1=-2, dim2=-1).sum(-1)
    second = torch.diagonal(p_ac @ p_cb @ p_ba, dim1=-2, dim2=-1).sum(-1)
    values = (12.0 * math.pi * 1j * (first - second)).real
    return values.detach().cpu().numpy().astype(np.float64, copy=False)


def explicit_chern(frame: np.ndarray, sectors: tuple[np.ndarray, ...]) -> float:
    gamma = (frame @ frame.conj().T).T
    a, b, c = sectors
    p_ca = gamma[np.ix_(c, a)]
    p_ab = gamma[np.ix_(a, b)]
    p_bc = gamma[np.ix_(b, c)]
    p_ac = gamma[np.ix_(a, c)]
    p_cb = gamma[np.ix_(c, b)]
    p_ba = gamma[np.ix_(b, a)]
    return float(np.real(12.0 * math.pi * 1j * (np.trace(p_ca @ p_ab @ p_bc) - np.trace(p_ac @ p_cb @ p_ba))))


def verify_result_record(result_path: Path, completion: dict[str, Any]) -> str:
    require(result_path.is_file(), f"missing source result: {result_path}")
    record = completion.get("result", {})
    require(record.get("name") == result_path.name, f"completion filename mismatch: {result_path}")
    require(int(record.get("bytes", -1)) == result_path.stat().st_size, f"completion byte-count mismatch: {result_path}")
    actual = sha256_path(result_path)
    require(record.get("sha256") == actual, f"completion checksum mismatch: {result_path}")
    return actual


def configure_worker() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def analyze_task(task: SourceTask) -> dict[str, Any]:
    configure_worker()
    result_path = Path(task.result_path)
    completion_path = Path(task.completion_path)
    require(completion_path.is_file(), f"missing completion: {completion_path}")
    completion = load_json(completion_path)
    result_sha256 = verify_result_record(result_path, completion)
    radii = radii_for(task)
    records: list[dict[str, Any]] = []

    if task.cohort == "new_width_sweep":
        require(completion.get("schema") == "wall_pump_width_endpoint_shard_completion_v1", f"new completion schema mismatch: {completion_path}")
        require(completion.get("status") == "complete" and completion.get("stage") == "endpoint_shard", f"new completion status mismatch: {completion_path}")
        for key, expected in (
            ("sampling_revision", "wall_pump_width_endpoints_s100_v1"),
            ("collection", task.collection),
            ("protocol", task.protocol),
            ("wall", task.wall),
            ("Nx", task.nx),
            ("Ny", task.ny),
            ("cycles", 48),
            ("execution_backend", "gpu"),
            ("canonical_entry_point", "classA_U1FGTN_gpu.run_markov_circuit"),
        ):
            require(completion.get(key) == expected, f"new completion {key} mismatch: {completion_path}")
        require(tuple(completion.get("sample_ids", [])) == task.sample_ids, f"new completion sample mismatch: {completion_path}")
        require(completion.get("result_filename") == result_path.name, f"duplicate result filename mismatch: {completion_path}")
        require(completion.get("result_bytes") == result_path.stat().st_size, f"duplicate result size mismatch: {completion_path}")
        require(completion.get("result_sha256") == result_sha256, f"duplicate result checksum mismatch: {completion_path}")
        with np.load(result_path, allow_pickle=False) as saved:
            require(str(saved["schema"].item()) == "wall_pump_width_endpoint_shard_v1", f"new result schema mismatch: {result_path}")
            require(str(saved["sampling_revision"].item()) == "wall_pump_width_endpoints_s100_v1", f"new result revision mismatch: {result_path}")
            for key, expected in (("collection", task.collection), ("protocol", task.protocol), ("wall", task.wall)):
                require(str(saved[key].item()) == expected, f"new result {key} mismatch: {result_path}")
            require(int(saved["Nx"].item()) == task.nx and int(saved["Ny"].item()) == task.ny, f"new result geometry mismatch: {result_path}")
            require(float(saved["alpha_1"].item()) == 1.0 and float(saved["alpha_2"].item()) == 30.0, f"new result alpha mismatch: {result_path}")
            require(int(saved["cycles_total"].item()) == 48, f"new result cycle mismatch: {result_path}")
            require(tuple(int(value) for value in saved["sample_ids"]) == task.sample_ids, f"new result sample mismatch: {result_path}")
            frames = np.asarray(saved["frames"])
            ranks = np.asarray(saved["ranks"], dtype=np.int64)
            gram = np.asarray(saved["gram_residual"], dtype=np.float64)
            final_charge = np.asarray(saved["final_total_charge"], dtype=np.int64)
            require(frames.dtype == np.complex128, f"new frame dtype mismatch: {result_path}")
            require(frames.shape[:2] == (len(task.sample_ids), 2 * task.nx * task.ny), f"new frame shape mismatch: {result_path}")
            require(np.array_equal(ranks, final_charge), f"new rank/charge mismatch: {result_path}")
            require(np.all(np.isfinite(gram)) and float(np.max(gram)) <= GRAM_TOLERANCE, f"new Gram residual failure: {result_path}")
            for member, sample_id in enumerate(task.sample_ids):
                rank = int(ranks[member])
                frame = np.array(frames[member, :, :rank], dtype=np.complex128, copy=True)
                require(np.all(np.isfinite(frame)), f"nonfinite new frame: {result_path}, sample={sample_id}")
                records.extend(analyze_frame(task, sample_id, frame, radii, result_sha256))
    else:
        sample_id = task.sample_ids[0]
        require(completion.get("schema") == "state_projector_pump_completion_v1", f"old completion schema mismatch: {completion_path}")
        require(completion.get("stage") == "burnin" and completion.get("wall") == task.wall, f"old completion task mismatch: {completion_path}")
        require(int(completion.get("sample_id", -1)) == sample_id, f"old completion sample mismatch: {completion_path}")
        require(completion.get("task_id") == f"burnin_{task.wall}_sample_{sample_id:03d}", f"old task ID mismatch: {completion_path}")
        with np.load(result_path, allow_pickle=False) as saved:
            require(str(saved["schema"].item()) in {"state_projector_pump_burnin_v1", "state_projector_pump_endpoint_state_v3"}, f"old result schema mismatch: {result_path}")
            frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
            rank = int(saved["rank"].item())
            require(frame.dtype == np.complex128 and frame.shape == (2 * task.nx * task.ny, rank), f"old frame shape/dtype mismatch: {result_path}")
            require(np.all(np.isfinite(frame)), f"nonfinite old frame: {result_path}")
            if "gram_residual" in saved:
                gram = float(saved["gram_residual"].item())
                require(math.isfinite(gram) and gram <= GRAM_TOLERANCE, f"old Gram residual failure: {result_path}")
            if "final_total_charge" in saved:
                final_total_charge = float(saved["final_total_charge"].item())
                require(
                    math.isfinite(final_total_charge)
                    and abs(final_total_charge - rank) <= 1.0e-8,
                    f"old rank/charge mismatch: {result_path}",
                )
        records.extend(analyze_frame(task, sample_id, frame, radii, result_sha256))

    return {
        "source": {
            **asdict(task),
            "result_bytes": result_path.stat().st_size,
            "result_sha256": result_sha256,
            "completion_sha256": sha256_path(completion_path),
            "completion_schema": completion.get("schema"),
        },
        "records": records,
    }


def analyze_frame(
    task: SourceTask,
    sample_id: int,
    frame: np.ndarray,
    radii: Iterable[float],
    source_sha256: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    wall_distance = task.nx / 4.0
    legacy_radius = 0.4 * min(task.nx, task.ny)
    for radius in radii:
        values = chern_by_y0(frame, batched_partitions(task.nx, task.ny, radius))
        require(values.shape == (task.ny,) and np.all(np.isfinite(values)), f"invalid Chern values: {task.result_path}, sample={sample_id}, R={radius}")
        rho = radius / wall_distance
        relation = "contained" if rho < 1.0 - 1e-12 else "crossing" if rho > 1.0 + 1e-12 else "touching"
        result.append(
            {
                "cohort": task.cohort,
                "campaign": task.campaign,
                "collection": task.collection,
                "protocol": task.protocol,
                "nshell_label": task.nshell_label,
                "nx": task.nx,
                "ny": task.ny,
                "wall": task.wall,
                "sample_id": sample_id,
                "source_result": task.result_path,
                "source_sha256": source_sha256,
                "rank": frame.shape[1],
                "radius": float(radius),
                "rho": float(rho),
                "radius_relation": relation,
                "is_legacy_radius": bool(np.isclose(radius, legacy_radius, atol=1e-12, rtol=0.0)),
                "chern_y0_mean": float(np.mean(values)),
                "chern_y0_sd": float(np.std(values, ddof=1)),
                "chern_y0_min": float(np.min(values)),
                "chern_y0_max": float(np.max(values)),
                "chern_by_y0": values,
            }
        )
    return result


def ensemble_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["cohort"], row["campaign"], row["collection"], row["protocol"],
        row["nshell_label"], row["nx"], row["ny"], row["wall"], row["radius"],
    )


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(ensemble_key(row), []).append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda item: item["sample_id"])
        sample_ids = [int(row["sample_id"]) for row in rows]
        require(sample_ids == list(range(len(rows))), f"noncontiguous samples for ensemble {key}")
        values = np.asarray([row["chern_y0_mean"] for row in rows], dtype=np.float64)
        require(len(values) in {25, 100}, f"unexpected ensemble size {len(values)} for {key}")
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        sem = sd / math.sqrt(len(values))
        interval_low = mean - sem
        interval_high = mean + sem
        if interval_low <= 1.0 <= interval_high:
            deviation_low = 0.0
        else:
            deviation_low = min(abs(interval_low - 1.0), abs(interval_high - 1.0))
        summaries.append(
            {
                "cohort": key[0], "campaign": key[1], "collection": key[2],
                "protocol": key[3], "nshell_label": key[4], "nx": key[5],
                "ny": key[6], "wall": key[7], "radius": key[8],
                "rho": rows[0]["rho"],
                "radius_relation": rows[0]["radius_relation"],
                "is_legacy_radius": rows[0]["is_legacy_radius"],
                "samples": len(values),
                "chern_mean": mean,
                "chern_sd": sd,
                "chern_sem": sem,
                "abs_chern_minus_one": abs(mean - 1.0),
                "abs_deviation_sem_low": deviation_low,
                "abs_deviation_sem_high": max(abs(interval_low - 1.0), abs(interval_high - 1.0)),
            }
        )
    return summaries


def stability_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["cohort"], row["campaign"], row["collection"], row["protocol"],
        row["nshell_label"], row["nx"], row["ny"], row["wall"],
    )


def stability_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in summaries:
        groups.setdefault(stability_key(row), []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda item: item["radius"])
        contained = [row for row in rows if row["radius_relation"] == "contained"]
        legacy = [row for row in rows if row["is_legacy_radius"]]
        require(contained and len(legacy) == 1, f"missing stability anchors for {key}")
        largest_contained = max(contained, key=lambda row: row["radius"])
        consecutive = [abs(b["chern_mean"] - a["chern_mean"]) for a, b in zip(rows, rows[1:])]
        output.append(
            {
                "cohort": key[0], "campaign": key[1], "collection": key[2],
                "protocol": key[3], "nshell_label": key[4], "nx": key[5],
                "ny": key[6], "wall": key[7], "samples": rows[0]["samples"],
                "contained_radius_count": len(contained),
                "contained_radius_min": min(row["radius"] for row in contained),
                "contained_radius_max": largest_contained["radius"],
                "contained_mean_spread": max(row["chern_mean"] for row in contained) - min(row["chern_mean"] for row in contained),
                "maximum_abs_successive_change": max(consecutive),
                "legacy_radius": legacy[0]["radius"],
                "legacy_chern_mean": legacy[0]["chern_mean"],
                "largest_contained_chern_mean": largest_contained["chern_mean"],
                "legacy_minus_largest_contained": legacy[0]["chern_mean"] - largest_contained["chern_mean"],
            }
        )
    return output


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value for key, value in row.items()})
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_by_y0_npz(path: Path, records: list[dict[str, Any]]) -> None:
    max_ny = max(int(row["ny"]) for row in records)
    values = np.full((len(records), max_ny), np.nan, dtype=np.float64)
    for index, row in enumerate(records):
        item = np.asarray(row["chern_by_y0"], dtype=np.float64)
        values[index, : len(item)] = item
    text_keys = ("cohort", "campaign", "collection", "protocol", "nshell_label", "wall", "radius_relation", "source_result", "source_sha256")
    int_keys = ("nx", "ny", "sample_id", "rank")
    arrays: dict[str, np.ndarray] = {"chern_by_y0": values}
    for key in text_keys:
        arrays[key] = np.asarray([str(row[key]) for row in records])
    for key in int_keys:
        arrays[key] = np.asarray([int(row[key]) for row in records], dtype=np.int64)
    for key in ("radius", "rho"):
        arrays[key] = np.asarray([float(row[key]) for row in records], dtype=np.float64)
    arrays["is_legacy_radius"] = np.asarray([bool(row["is_legacy_radius"]) for row in records], dtype=np.bool_)
    atomic_npz(path, **arrays)


def trajectory_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    omitted = {"chern_by_y0"}
    return [{key: value for key, value in row.items() if key not in omitted} for row in records]


def plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "Computer Modern Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 300,
        }
    )


def deviation_band(row: dict[str, Any], floor: float) -> tuple[float, float]:
    return max(float(row["abs_deviation_sem_low"]), floor), max(float(row["abs_deviation_sem_high"]), floor)


def plot_width_sweep(summaries: list[dict[str, Any]]) -> tuple[Path, Path]:
    selected = [row for row in summaries if row["cohort"] == "new_width_sweep"]
    require(bool(selected), "new width-sweep summary is empty")
    positive = [float(row["abs_deviation_sem_low"]) for row in selected if float(row["abs_deviation_sem_low"]) > 0]
    floor = max(min(positive) / 2.0, 1.0e-7)
    plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.7), sharex=True)
    colors = {20: "#d95f02", 24: "#1b9e77", 28: "#377eb8", 32: "#984ea3"}
    shell_style = {
        "1": {"marker": "^", "linestyle": ":"},
        "infinity": {"marker": "o", "linestyle": "-"},
    }
    for row_index, wall in enumerate(WALLS):
        for nx in (20, 24, 28, 32):
            for shell in ("1", "infinity"):
                rows = sorted(
                    [row for row in selected if row["wall"] == wall and row["nx"] == nx and row["nshell_label"] == shell],
                    key=lambda row: row["rho"],
                )
                if not rows:
                    continue
                rho = np.asarray([row["rho"] for row in rows])
                mean = np.asarray([row["chern_mean"] for row in rows])
                sem = np.asarray([row["chern_sem"] for row in rows])
                style = shell_style[shell]
                axes[row_index, 0].plot(rho, mean, color=colors[nx], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.0, markersize=3.5)
                axes[row_index, 0].fill_between(rho, mean - sem, mean + sem, color=colors[nx], alpha=0.10, linewidth=0)
                deviation = np.asarray([row["abs_chern_minus_one"] for row in rows])
                low_high = np.asarray([deviation_band(row, floor) for row in rows])
                axes[row_index, 1].plot(rho, np.maximum(deviation, floor), color=colors[nx], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.0, markersize=3.5)
                axes[row_index, 1].fill_between(rho, low_high[:, 0], low_high[:, 1], color=colors[nx], alpha=0.10, linewidth=0)
                for column in range(2):
                    legacy = [row for row in rows if row["is_legacy_radius"]]
                    require(len(legacy) == 1, "width-sweep legacy marker mismatch")
                    yvalue = legacy[0]["chern_mean"] if column == 0 else max(legacy[0]["abs_chern_minus_one"], floor)
                    axes[row_index, column].plot(legacy[0]["rho"], yvalue, marker="*", markersize=7, markerfacecolor="none", markeredgecolor=colors[nx], linestyle="none", zorder=5)
        axes[row_index, 0].axhline(1.0, color="0.35", linestyle="--", linewidth=0.8)
        axes[row_index, 1].set_yscale("log")
        for column in range(2):
            axes[row_index, column].axvline(1.0, color="0.35", linestyle="--", linewidth=0.8)
        axes[row_index, 0].set_ylabel(rf"{wall.capitalize()} $\overline{{C_G}}$")
        axes[row_index, 1].set_ylabel(rf"{wall.capitalize()} $|\overline{{C_G}}-1|$")
    axes[1, 0].set_xlabel(r"normalized radius $\rho=R/(N_x/4)$")
    axes[1, 1].set_xlabel(r"normalized radius $\rho=R/(N_x/4)$")
    color_handles = [Line2D([0], [0], color=colors[nx], linewidth=1.4, label=rf"$N_x={nx}$") for nx in colors]
    shell_handles = [
        Line2D([0], [0], color="black", marker="^", linestyle=":", linewidth=1.0, markersize=4, label=r"$n_{\rm shell}=1$"),
        Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=1.0, markersize=4, label=r"$n_{\rm shell}=\infty$"),
        Line2D([0], [0], color="black", marker="*", markerfacecolor="none", linestyle="none", markersize=7, label=r"legacy $R$"),
    ]
    first_legend = axes[0, 0].legend(handles=color_handles, frameon=False, ncol=2, loc="best", handlelength=1.8, columnspacing=0.8)
    axes[0, 0].add_artist(first_legend)
    axes[0, 1].legend(handles=shell_handles, frameon=False, ncol=1, loc="best", handlelength=1.8)
    for panel, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.04, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=9)
    fig.tight_layout(pad=0.6, w_pad=1.0, h_pad=0.7)
    pdf = FIGURE_ROOT / "new_width_sweep_chern_radius_stability.pdf"
    png = FIGURE_ROOT / "new_width_sweep_chern_radius_stability.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_old_series(summaries: list[dict[str, Any]]) -> tuple[Path, Path]:
    selected = [row for row in summaries if row["cohort"] == "old_fixed_nx"]
    require(bool(selected), "old fixed-Nx summary is empty")
    positive = [float(row["abs_deviation_sem_low"]) for row in selected if float(row["abs_deviation_sem_low"]) > 0]
    floor = max(min(positive) / 2.0, 1.0e-7)
    plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.7), sharex=True)
    styles = {
        24: {"color": "#d95f02", "marker": "^", "linestyle": ":"},
        28: {"color": "#1b9e77", "marker": "s", "linestyle": "--"},
        30: {"color": "#377eb8", "marker": "o", "linestyle": "-"},
    }
    for row_index, wall in enumerate(WALLS):
        for ny, style in styles.items():
            rows = sorted([row for row in selected if row["wall"] == wall and row["ny"] == ny], key=lambda row: row["rho"])
            rho = np.asarray([row["rho"] for row in rows])
            mean = np.asarray([row["chern_mean"] for row in rows])
            sem = np.asarray([row["chern_sem"] for row in rows])
            axes[row_index, 0].plot(rho, mean, **style, linewidth=1.0, markersize=3.5, label=rf"$N_y={ny}$")
            axes[row_index, 0].fill_between(rho, mean - sem, mean + sem, color=style["color"], alpha=0.10, linewidth=0)
            deviation = np.asarray([row["abs_chern_minus_one"] for row in rows])
            low_high = np.asarray([deviation_band(row, floor) for row in rows])
            axes[row_index, 1].plot(rho, np.maximum(deviation, floor), **style, linewidth=1.0, markersize=3.5)
            axes[row_index, 1].fill_between(rho, low_high[:, 0], low_high[:, 1], color=style["color"], alpha=0.10, linewidth=0)
            legacy = [row for row in rows if row["is_legacy_radius"]]
            require(len(legacy) == 1, "old-series legacy marker mismatch")
            axes[row_index, 0].plot(legacy[0]["rho"], legacy[0]["chern_mean"], marker="*", markersize=7, markerfacecolor="none", markeredgecolor=style["color"], linestyle="none", zorder=5)
            axes[row_index, 1].plot(legacy[0]["rho"], max(legacy[0]["abs_chern_minus_one"], floor), marker="*", markersize=7, markerfacecolor="none", markeredgecolor=style["color"], linestyle="none", zorder=5)
        axes[row_index, 0].axhline(1.0, color="0.35", linestyle="--", linewidth=0.8)
        axes[row_index, 1].set_yscale("log")
        for column in range(2):
            axes[row_index, column].axvline(1.0, color="0.35", linestyle="--", linewidth=0.8)
        axes[row_index, 0].set_ylabel(rf"{wall.capitalize()} $\overline{{C_G}}$")
        axes[row_index, 1].set_ylabel(rf"{wall.capitalize()} $|\overline{{C_G}}-1|$")
    axes[1, 0].set_xlabel(r"normalized radius $\rho=R/(N_x/4)$")
    axes[1, 1].set_xlabel(r"normalized radius $\rho=R/(N_x/4)$")
    handles = [Line2D([0], [0], **style, linewidth=1.0, markersize=4, label=rf"$N_y={ny}$") for ny, style in styles.items()]
    handles.append(Line2D([0], [0], color="black", marker="*", markerfacecolor="none", linestyle="none", markersize=7, label=r"legacy $R$"))
    axes[0, 0].legend(handles=handles, frameon=False, ncol=2, loc="best", handlelength=1.8, columnspacing=0.8)
    for panel, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.04, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=9)
    fig.tight_layout(pad=0.6, w_pad=1.0, h_pad=0.7)
    pdf = FIGURE_ROOT / "old_fixed_nx_chern_radius_stability.pdf"
    png = FIGURE_ROOT / "old_fixed_nx_chern_radius_stability.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def validate_factorization(tasks: list[SourceTask]) -> list[dict[str, Any]]:
    chosen = [
        next(task for task in tasks if task.cohort == "old_fixed_nx" and task.ny == 24 and task.wall == "hard" and task.sample_ids == (0,)),
        next(task for task in tasks if task.cohort == "new_width_sweep" and task.nx == 20 and task.protocol == "dense" and task.wall == "soft" and task.sample_ids[0] == 0),
    ]
    checks: list[dict[str, Any]] = []
    for task in chosen:
        with np.load(task.result_path, allow_pickle=False) as saved:
            if task.cohort == "old_fixed_nx":
                frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
            else:
                rank = int(saved["ranks"][0])
                frame = np.array(saved["frames"][0, :, :rank], dtype=np.complex128, copy=True)
        radius = 4.0
        yref = task.ny // 2
        factorized = float(chern_by_y0(frame, batched_partitions(task.nx, task.ny, radius))[yref])
        explicit = explicit_chern(frame, partition_indices(nx=task.nx, ny=task.ny, xref=task.nx // 2, yref=yref, radius=radius))
        error = abs(factorized - explicit)
        require(error <= FACTOR_TOLERANCE, f"factorized/explicit mismatch {error} for {task.result_path}")
        checks.append({"cohort": task.cohort, "source": task.result_path, "radius": radius, "yref": yref, "factorized": factorized, "explicit": explicit, "absolute_error": error})
    return checks


def validate_reference(records: list[dict[str, Any]]) -> dict[str, Any]:
    require(REFERENCE_NPZ.is_file(), f"missing prior radius reference: {REFERENCE_NPZ}")
    lookup = {
        (row["ny"], row["wall"], row["sample_id"], int(row["radius"])): np.asarray(row["chern_by_y0"])
        for row in records
        if row["cohort"] == "old_fixed_nx" and row["ny"] in {24, 28} and row["radius"] in {2.0, 3.0, 4.0}
    }
    maximum = 0.0
    comparisons = 0
    with np.load(REFERENCE_NPZ, allow_pickle=False) as saved:
        for ny in (24, 28):
            for wall in WALLS:
                for radius in (2, 3, 4):
                    reference = np.asarray(saved[f"chern_by_y0_Ny{ny}_{wall}_R{radius}"], dtype=np.float64)
                    current = np.stack([lookup[(ny, wall, sample_id, radius)] for sample_id in range(100)])
                    error = float(np.max(np.abs(reference - current)))
                    maximum = max(maximum, error)
                    comparisons += int(reference.size)
    require(maximum <= REFERENCE_TOLERANCE, f"prior R=2,3,4 regression mismatch: {maximum}")
    return {"reference_path": str(REFERENCE_NPZ), "reference_sha256": sha256_path(REFERENCE_NPZ), "scalar_comparisons": comparisons, "maximum_absolute_error": maximum, "tolerance": REFERENCE_TOLERANCE}


def validate_sems(records: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> float:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in records:
        groups.setdefault(ensemble_key(row), []).append(float(row["chern_y0_mean"]))
    maximum = 0.0
    for row in summaries:
        values = np.asarray(groups[ensemble_key(row)], dtype=np.float64)
        expected = float(values.std(ddof=1) / math.sqrt(len(values)))
        maximum = max(maximum, abs(expected - float(row["chern_sem"])))
    require(maximum <= np.finfo(np.float64).eps * 16, f"SEM validation mismatch: {maximum}")
    return maximum


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.parent.parent.parent.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    require(args.workers >= 1, "--workers must be positive")

    print(f"[config] project={PROJECT_ROOT}", flush=True)
    print(f"[config] output={OUTPUT_ROOT}", flush=True)
    print(f"[config] workers={args.workers}; source verification includes SHA-256 readback", flush=True)
    config_identities = validate_campaign_configs()
    tasks = discover_tasks()
    print("[inventory] 830 result/completion pairs; 1,150 new and 600 old trajectories", flush=True)

    sources: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_task, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="endpoint R sweep", unit="source"):
            payload = future.result()
            sources.append(payload["source"])
            records.extend(payload["records"])
    records.sort(key=lambda row: (row["cohort"], row["campaign"], row["collection"], row["protocol"], row["nx"], row["ny"], WALLS.index(row["wall"]), row["sample_id"], row["radius"]))
    sources.sort(key=lambda row: row["result_path"])
    require(len(sources) == 830 and len({row["result_path"] for row in sources}) == 830, "source result uniqueness failure")
    require(len(records) == 15400, f"expected 15,400 trajectory-radius rows, found {len(records)}")

    print("[validation] checking explicit projector contractions and legacy R=2,3,4 products", flush=True)
    factorization_checks = validate_factorization(tasks)
    reference_check = validate_reference(records)
    summaries = aggregate_records(records)
    sem_error = validate_sems(records, summaries)
    stability = stability_rows(summaries)

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    trajectory_path = ANALYSIS_ROOT / "trajectory_radius_statistics.csv"
    ensemble_path = ANALYSIS_ROOT / "ensemble_radius_statistics.csv"
    stability_path = ANALYSIS_ROOT / "radius_stability_summary.csv"
    values_path = ANALYSIS_ROOT / "chern_by_y0.npz"
    atomic_csv(trajectory_path, trajectory_csv_rows(records))
    atomic_csv(ensemble_path, summaries)
    atomic_csv(stability_path, stability)
    save_by_y0_npz(values_path, records)
    new_pdf, new_png = plot_width_sweep(summaries)
    old_pdf, old_png = plot_old_series(summaries)

    artifacts = [trajectory_path, ensemble_path, stability_path, values_path, new_pdf, new_png, old_pdf, old_png]
    summary_path = ANALYSIS_ROOT / "analysis_summary.json"
    summary = {
        "schema": SUMMARY_SCHEMA,
        "analysis": "offline three-sector real-space Chern radius stability",
        "dynamics_rerun": False,
        "projector": "Gamma=(V V^dagger)^T=V* V^T",
        "trajectory_estimator": "c_xi(R)=mean_y0 C_G,xi(x0=Nx/2,y0;R)",
        "ensemble_estimator": "unweighted mean of trajectory-level c_xi(R)",
        "uncertainty": "sample standard deviation with ddof=1 divided by sqrt(S)",
        "transverse_origins_are_independent_samples": False,
        "cohorts_are_pooled": False,
        "radius_relation": {"contained": "R<Nx/4", "touching": "R=Nx/4", "crossing": "R>Nx/4"},
        "inventory": {
            "source_pairs": len(sources),
            "new_width_sweep_trajectories": 1150,
            "old_fixed_nx_trajectories": 600,
            "trajectory_radius_rows": len(records),
            "ensemble_radius_rows": len(summaries),
            "stability_rows": len(stability),
        },
        "config_identities": config_identities,
        "source_pairs": sources,
        "validation": {
            "all_source_completion_checks_passed": True,
            "required_alpha_1": 1.0,
            "required_alpha_2": 30.0,
            "required_dtype": "complex128",
            "factorized_vs_explicit": factorization_checks,
            "prior_radius_regression": reference_check,
            "maximum_sem_recalculation_error": sem_error,
        },
        "artifacts": [
            {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_path(path)}
            for path in artifacts
        ],
    }
    atomic_json(summary_path, summary)
    print(f"[complete] wrote {len(artifacts) + 1} verified analysis products under {OUTPUT_ROOT}", flush=True)
    print(f"[complete] maximum prior-result error={reference_check['maximum_absolute_error']:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
