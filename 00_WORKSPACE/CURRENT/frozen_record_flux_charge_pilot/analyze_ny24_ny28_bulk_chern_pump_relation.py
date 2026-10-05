#!/usr/bin/env python3
"""Relate bulk-centered real-space Chern markers to the Ny=24,28 pump sectors."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
CAMPAIGNS = {
    24: "N20x24_state_projector_pump_s100_v1",
    28: "N20x28_state_projector_pump_s100_v3",
}
OUTPUT_ROOT = (
    RESULTS_ROOT
    / "N20_state_projector_pump_Ny24_36_s100_series_v3"
    / "analysis/bulk_chern_pump_relation_v1"
)
RADII = (2.0, 3.0, 4.0)
XREF = 10
WALLS = ("soft", "hard")
PUMP_THRESHOLD = 0.5


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_indices(
    *, nx: int, ny: int, xref: int, yref: int, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the repository periodic three-wedge partition convention."""

    x = np.arange(nx, dtype=np.int64)
    y = np.arange(ny, dtype=np.int64)
    dx = (x - int(xref) + nx // 2) % nx - nx // 2
    dy = (y - int(yref) + ny // 2) % ny - ny // 2
    dx_grid, dy_grid = np.meshgrid(dx, dy, indexing="ij")
    inside = dx_grid * dx_grid + dy_grid * dy_grid <= float(radius) ** 2
    theta = np.mod(np.arctan2(dy_grid, dx_grid), 2.0 * np.pi)
    bounds = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 2.0 * np.pi)
    masks = tuple(
        inside & (theta >= bounds[index]) & (theta < bounds[index + 1])
        for index in range(3)
    )

    def indices(mask: np.ndarray) -> np.ndarray:
        xs, ys = np.nonzero(mask)
        first = 2 * xs + 2 * nx * ys
        return np.sort(np.concatenate((first, first + 1))).astype(np.int64)

    result = tuple(indices(mask) for mask in masks)
    if min(len(values) for values in result) == 0:
        raise RuntimeError("Chern partition has an empty wedge")
    return result  # type: ignore[return-value]


def batched_partitions(ny: int, radius: float) -> tuple[torch.Tensor, ...]:
    rows = [
        partition_indices(nx=20, ny=ny, xref=XREF, yref=yref, radius=radius)
        for yref in range(ny)
    ]
    return tuple(
        torch.as_tensor(np.stack([row[wedge] for row in rows]), dtype=torch.long)
        for wedge in range(3)
    )


def chern_by_y0(frame: np.ndarray, partitions: tuple[torch.Tensor, ...]) -> np.ndarray:
    """Evaluate 12*pi*i Tr(P_CA P_AB P_BC - P_AC P_CB P_BA)."""

    occupied = torch.from_numpy(np.asarray(frame, dtype=np.complex128)).conj()
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
    return (12.0 * math.pi * 1j * (first - second)).real.numpy().astype(np.float64)


def verify_endpoint_pair(path: Path, *, ny: int, wall: str, sample_id: int) -> dict[str, Any]:
    completion_path = path.with_suffix(".completion.json")
    if not path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"missing endpoint pair: {path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_task = f"burnin_{wall}_sample_{sample_id:03d}"
    if (
        completion.get("task_id") != expected_task
        or completion.get("wall") != wall
        or int(completion.get("sample_id", -1)) != sample_id
        or completion.get("stage") != "burnin"
    ):
        raise RuntimeError(f"endpoint completion identity mismatch: {path}")
    record = completion.get("result", {})
    if record.get("name") != path.name or int(record.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"endpoint completion size/name mismatch: {path}")
    if record.get("sha256") != sha256_path(path):
        raise RuntimeError(f"endpoint checksum mismatch: {path}")
    return completion


def load_pump_rows(ny: int) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, str]]:
    campaign = RESULTS_ROOT / CAMPAIGNS[ny]
    csv_path = campaign / "analysis/state_projector_pump_endpoints.csv"
    summary_path = campaign / "analysis/analysis_summary.json"
    if not csv_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"Ny={ny} verified pump analysis is missing")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["wall"], int(row["sample_id"])): row for row in rows}
    if len(rows) != 200 or len(lookup) != 200:
        raise RuntimeError(f"Ny={ny} pump table does not contain 200 unique rows")
    return lookup, {
        "pump_csv": str(csv_path),
        "pump_csv_sha256": sha256_path(csv_path),
        "pump_summary": str(summary_path),
        "pump_summary_sha256": sha256_path(summary_path),
    }


def worker(payload: tuple[int, str, int, str]) -> dict[str, Any]:
    ny, wall, sample_id, path_text = payload
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    path = Path(path_text)
    completion = verify_endpoint_pair(path, ny=ny, wall=wall, sample_id=sample_id)
    with np.load(path, allow_pickle=False) as saved:
        frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
        rank = int(np.asarray(saved["rank"]).item())
    if frame.shape != (40 * ny, rank):
        raise RuntimeError(f"endpoint frame shape mismatch: {path}")
    result: dict[str, Any] = {
        "Ny": ny,
        "wall": wall,
        "sample_id": sample_id,
        "rank": rank,
        "endpoint_sha256": completion["result"]["sha256"],
    }
    for radius in RADII:
        values = chern_by_y0(frame, batched_partitions(ny, radius))
        if values.shape != (ny,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid Chern values for {path}, R={radius}")
        key = int(radius)
        result[f"chern_R{key}_mean"] = float(values.mean())
        result[f"chern_R{key}_std_y0"] = float(values.std(ddof=1))
        result[f"chern_R{key}_min"] = float(values.min())
        result[f"chern_R{key}_max"] = float(values.max())
        result[f"chern_R{key}_by_y0"] = values
    return result


def bootstrap_difference(
    values: np.ndarray, event: np.ndarray, *, seed: int, draws: int = 10000
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pumped, closed = values[event], values[~event]
    difference = float(pumped.mean() - closed.mean())
    sampled = (
        pumped[rng.integers(0, len(pumped), size=(draws, len(pumped)))].mean(axis=1)
        - closed[rng.integers(0, len(closed), size=(draws, len(closed)))].mean(axis=1)
    )
    return difference, float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def auc_from_u(values: np.ndarray, event: np.ndarray) -> tuple[float, float]:
    pumped, closed = values[event], values[~event]
    test = mannwhitneyu(pumped, closed, alternative="two-sided")
    return float(test.statistic / (len(pumped) * len(closed))), float(test.pvalue)


def style() -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    pump_lookup: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    inputs: dict[int, dict[str, str]] = {}
    tasks = []
    for ny, campaign_name in CAMPAIGNS.items():
        pump_lookup[ny], inputs[ny] = load_pump_rows(ny)
        for wall in WALLS:
            for sample_id in range(100):
                path = RESULTS_ROOT / campaign_name / f"burnins/{wall}/sample_{sample_id:03d}.npz"
                tasks.append((ny, wall, sample_id, str(path)))

    rows = []
    by_y0: dict[tuple[int, str, int, int], np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="bulk Chern endpoints", unit="trajectory"):
            result = future.result()
            ny, wall, sample_id = result["Ny"], result["wall"], result["sample_id"]
            pump = pump_lookup[ny][(wall, sample_id)]
            result["ccw_q_x"] = float(pump["ccw_q_x"])
            result["cw_q_x"] = float(pump["cw_q_x"])
            result["direction_odd_q_x"] = float(pump["direction_odd_q_x"])
            result["pump_event"] = int(pump["pump_event"])
            for radius in RADII:
                key = int(radius)
                by_y0[(ny, wall, sample_id, key)] = result.pop(f"chern_R{key}_by_y0")
            rows.append(result)
    rows.sort(key=lambda row: (row["Ny"], WALLS.index(row["wall"]), row["sample_id"]))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "ny24_ny28_samplewise_bulk_chern_and_pump.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    arrays: dict[str, np.ndarray] = {}
    for ny in CAMPAIGNS:
        for wall in WALLS:
            for radius in map(int, RADII):
                arrays[f"chern_by_y0_Ny{ny}_{wall}_R{radius}"] = np.stack(
                    [by_y0[(ny, wall, sample_id, radius)] for sample_id in range(100)]
                )
    npz_path = OUTPUT_ROOT / "ny24_ny28_bulk_chern_by_y0.npz"
    np.savez_compressed(npz_path, **arrays)

    statistics = []
    for ny in CAMPAIGNS:
        for wall in WALLS:
            selected = [row for row in rows if row["Ny"] == ny and row["wall"] == wall]
            q_odd = np.asarray([row["direction_odd_q_x"] for row in selected])
            event = np.asarray([bool(row["pump_event"]) for row in selected])
            for radius in map(int, RADII):
                values = np.asarray([row[f"chern_R{radius}_mean"] for row in selected])
                difference, low, high = bootstrap_difference(
                    values, event, seed=2026090400 + ny * 10 + radius + WALLS.index(wall) * 1000
                )
                correlation = pearsonr(values, q_odd)
                auc, mann_p = auc_from_u(values, event)
                statistics.append(
                    {
                        "Ny": ny,
                        "wall": wall,
                        "radius": radius,
                        "samples": len(values),
                        "pump_events": int(event.sum()),
                        "overall_mean": float(values.mean()),
                        "overall_std": float(values.std(ddof=1)),
                        "overall_min": float(values.min()),
                        "overall_max": float(values.max()),
                        "pumped_mean": float(values[event].mean()),
                        "closure_mean": float(values[~event].mean()),
                        "pumped_minus_closure": difference,
                        "difference_bootstrap95_low": low,
                        "difference_bootstrap95_high": high,
                        "pearson_vs_q_odd": float(correlation.statistic),
                        "pearson_pvalue": float(correlation.pvalue),
                        "event_over_closure_auc": auc,
                        "mann_whitney_pvalue": mann_p,
                    }
                )

    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.8), sharex=True, sharey=True)
    for panel, (ax, (ny, wall)) in enumerate(
        zip(axes.flat, ((24, "soft"), (28, "soft"), (24, "hard"), (28, "hard")))
    ):
        selected = [row for row in rows if row["Ny"] == ny and row["wall"] == wall]
        marker = np.asarray([row["chern_R4_mean"] for row in selected])
        q_odd = np.asarray([row["direction_odd_q_x"] for row in selected])
        event = np.asarray([bool(row["pump_event"]) for row in selected])
        ax.scatter(marker[~event], q_odd[~event], s=15, color="0.55", alpha=0.75, label="closure")
        ax.scatter(marker[event], q_odd[event], s=15, color="#1f77b4", alpha=0.75, label="pump")
        stat = next(
            row for row in statistics
            if row["Ny"] == ny and row["wall"] == wall and row["radius"] == 4
        )
        ax.set_title(
            rf"{wall.capitalize()}, $N_y={ny}$; "
            rf"$r={stat['pearson_vs_q_odd']:.2f}$"
        )
        ax.set_xlabel(r"$y_0$-averaged bulk $C_G$ ($R=4$)")
        ax.axhline(PUMP_THRESHOLD, color="0.35", linestyle="--", linewidth=0.7)
        ax.text(-0.14, 1.04, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=9)
    axes[0, 0].set_ylabel(r"endpoint $q_x^{\rm odd}$")
    axes[1, 0].set_ylabel(r"endpoint $q_x^{\rm odd}$")
    axes[0, 1].legend(frameon=False)
    fig.tight_layout()
    scatter_pdf = OUTPUT_ROOT / "bulk_chern_vs_pump_sector.pdf"
    scatter_png = OUTPUT_ROOT / "bulk_chern_vs_pump_sector.png"
    fig.savefig(scatter_pdf, bbox_inches="tight")
    fig.savefig(scatter_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), sharey=True)
    colors = {"soft": "#1f77b4", "hard": "#d62728"}
    for ax, ny in zip(axes, CAMPAIGNS):
        for wall in WALLS:
            selected = [
                row for row in statistics if row["Ny"] == ny and row["wall"] == wall
            ]
            radius = np.asarray([row["radius"] for row in selected])
            difference = np.asarray([row["pumped_minus_closure"] for row in selected])
            low = np.asarray([row["difference_bootstrap95_low"] for row in selected])
            high = np.asarray([row["difference_bootstrap95_high"] for row in selected])
            ax.errorbar(
                radius,
                difference,
                yerr=np.vstack((difference - low, high - difference)),
                marker="o" if wall == "soft" else "s",
                color=colors[wall],
                linewidth=1.1,
                capsize=2,
                label=wall.capitalize(),
            )
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_title(rf"$N_y={ny}$")
        ax.set_xlabel("trijunction radius")
        ax.set_xticks(RADII)
    axes[0].set_ylabel(r"$\langle C_G\rangle_{\rm pump}-\langle C_G\rangle_{\rm closure}$")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    radius_pdf = OUTPUT_ROOT / "bulk_chern_sector_contrast_by_radius.pdf"
    radius_png = OUTPUT_ROOT / "bulk_chern_sector_contrast_by_radius.png"
    fig.savefig(radius_pdf, bbox_inches="tight")
    fig.savefig(radius_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema": "ny24_ny28_bulk_chern_pump_relation_v1",
        "definition": {
            "estimator": "periodic three-sector real-space Chern number from occupied frame",
            "xref": XREF,
            "yref": "all y0 averaged within each trajectory",
            "radii": list(RADII),
            "topological_slab": "x=5,...,15",
            "pump_coordinate": "q_x^odd=(q_x^CCW-q_x^CW)/2",
            "pump_event_threshold": PUMP_THRESHOLD,
        },
        "inputs": inputs,
        "statistics": statistics,
        "samplewise_csv": str(csv_path),
        "by_y0_npz": str(npz_path),
        "figures": {
            "scatter_pdf": str(scatter_pdf),
            "scatter_png": str(scatter_png),
            "radius_pdf": str(radius_pdf),
            "radius_png": str(radius_png),
        },
        "analysis_source_sha256": sha256_path(Path(__file__).resolve()),
    }
    summary_path = OUTPUT_ROOT / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
