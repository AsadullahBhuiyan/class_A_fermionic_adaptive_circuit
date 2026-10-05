#!/usr/bin/env python3
"""Analyze late-time boundary-only Born-record free energies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

import run_campaign as campaign


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_trajectories(
    tasks: list[campaign.TaskSpec], *, allow_partial: bool
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    invalid: list[str] = []
    for task in tasks:
        valid, reason = campaign.verify_complete(task)
        if not valid:
            invalid.append(f"{task.task_id}: {reason}")
            continue
        with np.load(task.result_path, allow_pickle=False) as data:
            cycles = np.asarray(data["cycles"], dtype=np.float64)
            omega = np.asarray(data["cumulative_log_probability"], dtype=np.float64)
            row: dict[str, Any] = {
                "task_id": task.task_id,
                "arm": task.arm,
                "Nx": task.nx,
                "Ny": task.ny,
                "sample_index": task.sample_index,
                "seed": task.seed,
                "endpoint_log_probability": float(omega[-1]),
                "endpoint_rate": float(-omega[-1] / task.cycles),
                "endpoint_rate_per_Ny": float(-omega[-1] / (task.cycles * task.ny)),
                "elapsed_seconds": float(data["elapsed_seconds"]),
                "final_rank": int(data["final_rank"]),
                "minimum_rank": int(data["minimum_rank"]),
                "maximum_rank": int(data["maximum_rank"]),
                "final_gram_residual": float(data["final_gram_residual"]),
            }
            for index, (lower, upper) in enumerate(((1, 2), (2, 3), (3, 4)), 1):
                mask = (cycles >= lower * task.ny) & (cycles <= upper * task.ny)
                lam = slope(cycles[mask], omega[mask])
                row[f"lambda_W{index}"] = lam
                row[f"free_energy_density_W{index}"] = -lam / task.ny
            combined_mask = (cycles >= 2 * task.ny) & (cycles <= 4 * task.ny)
            combined_lambda = slope(cycles[combined_mask], omega[combined_mask])
            row["lambda_W23"] = combined_lambda
            row["free_energy_density_W23"] = -combined_lambda / task.ny
            rows.append(row)
            curves[(task.arm, task.ny)].append(omega)
    if invalid and not allow_partial:
        raise RuntimeError(
            f"campaign is incomplete ({len(invalid)} invalid tasks); first={invalid[0]}"
        )
    if not rows:
        raise RuntimeError("no verified trajectories are available")
    return rows, curves


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(np.asarray(values, float), [0.025, 0.975])
    return float(lower), float(upper)


def temporal_summary(
    rows: list[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), int(row["Ny"]))].append(row)
    output: list[dict[str, Any]] = []
    for (arm, ny), group in sorted(grouped.items()):
        w2 = np.asarray([row["lambda_W2"] for row in group], float)
        w3 = np.asarray([row["lambda_W3"] for row in group], float)
        boot = np.empty(replicates, dtype=float)
        for index in range(replicates):
            draw = rng.integers(0, len(group), size=len(group))
            boot[index] = float(np.mean(w3[draw] - w2[draw]))
        low, high = percentile_interval(boot)
        shift = float(np.mean(w3 - w2))
        output.append(
            {
                "arm": arm,
                "Ny": ny,
                "samples": len(group),
                "mean_lambda_W2": float(np.mean(w2)),
                "mean_lambda_W3": float(np.mean(w3)),
                "mean_shift_W3_minus_W2": shift,
                "shift_ci95_low": low,
                "shift_ci95_high": high,
                "relative_shift": float(abs(shift) / max(abs(float(np.mean(w3))), 1e-300)),
            }
        )
    return output


def finite_size_summary(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    replicates: int,
    seed: int,
    slope_field: str = "lambda_W3",
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    grouped: dict[int, np.ndarray] = {}
    for ny in sorted({int(row["Ny"]) for row in rows if row["arm"] == arm}):
        grouped[ny] = np.asarray(
            [row[slope_field] for row in rows if row["arm"] == arm and int(row["Ny"]) == ny],
            dtype=float,
        )
    if len(grouped) < 3:
        return {"arm": arm, "status": "insufficient_sizes", "sizes": sorted(grouped)}
    nys = np.asarray(sorted(grouped), dtype=float)
    x = 1.0 / nys**2
    y = np.asarray([-np.mean(grouped[int(ny)]) / ny for ny in nys])
    a0, f_inf = np.polyfit(x, y, 1)
    design4 = np.column_stack([np.ones_like(x), x, x**2])
    f4, a4, b4 = np.linalg.lstsq(design4, y, rcond=None)[0]
    omit = nys > np.min(nys)
    a_omit, f_omit = np.polyfit(x[omit], y[omit], 1)
    boot_a = np.empty(replicates, dtype=float)
    for index in range(replicates):
        boot_y = []
        for ny in nys.astype(int):
            values = grouped[int(ny)]
            draw = rng.integers(0, values.size, size=values.size)
            boot_y.append(-float(np.mean(values[draw])) / ny)
        boot_a[index] = np.polyfit(x, np.asarray(boot_y), 1)[0]
    low, high = percentile_interval(boot_a)
    return {
        "arm": arm,
        "slope_field": slope_field,
        "status": "fit",
        "sizes": nys.astype(int).tolist(),
        "samples_per_size": {str(ny): int(grouped[ny].size) for ny in grouped},
        "f_boundary": float(f_inf),
        "A_boundary": float(a0),
        "A_boundary_ci95": [low, high],
        "alpha_c_eff_two_wall": float(-6.0 * a0 / np.pi),
        "alpha_c_eff_two_wall_ci95": [float(-6.0 * high / np.pi), float(-6.0 * low / np.pi)],
        "alpha_c_eff_per_wall_if_independent": float(-3.0 * a0 / np.pi),
        "quartic_sensitivity": {
            "f_boundary": float(f4),
            "A_boundary": float(a4),
            "B_boundary": float(b4),
        },
        "omit_smallest_sensitivity": {
            "omitted_Ny": int(np.min(nys)),
            "f_boundary": float(f_omit),
            "A_boundary": float(a_omit),
        },
        "plot": {"x": x.tolist(), "y": y.tolist()},
    }


def make_figure(
    path_root: Path,
    rows: list[dict[str, Any]],
    curves: dict[tuple[str, int], list[np.ndarray]],
    temporal: list[dict[str, Any]],
    fits: dict[str, dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    for color, ny in zip(colors, (16, 20, 24, 28, 32)):
        values = curves.get(("exact_wall", ny), [])
        if not values:
            continue
        mean = np.mean(np.stack(values), axis=0)
        t = np.arange(mean.size, dtype=float)
        mask = t > 0
        axes[0, 0].plot(t[mask] / ny, -mean[mask] / (t[mask] * ny), color=color, label=f"{ny}")
    axes[0, 0].set(xlabel=r"cycle/$N_y$", ylabel=r"$-\overline{\omega(t)}/(tN_y)$")
    axes[0, 0].legend(title=r"$N_y$", ncol=2, frameon=False)

    exact_temporal = [row for row in temporal if row["arm"] == "exact_wall"]
    if exact_temporal:
        ny = np.asarray([row["Ny"] for row in exact_temporal])
        shift = np.asarray([row["mean_shift_W3_minus_W2"] for row in exact_temporal])
        low = np.asarray([row["shift_ci95_low"] for row in exact_temporal])
        high = np.asarray([row["shift_ci95_high"] for row in exact_temporal])
        axes[0, 1].errorbar(ny, shift, yerr=[shift - low, high - shift], fmt="o-", color="#1f77b4")
    axes[0, 1].axhline(0.0, color="0.3", ls="--", lw=0.8)
    axes[0, 1].set(xlabel=r"$N_y$", ylabel=r"$\overline{\lambda_0^{W_3}-\lambda_0^{W_2}}$")

    for arm, style, color in (("exact_wall", "o-", "#1f77b4"), ("thin_strip", "s--", "#d62728")):
        fit = fits.get(arm, {})
        if fit.get("status") != "fit":
            continue
        x = np.asarray(fit["plot"]["x"])
        y = np.asarray(fit["plot"]["y"])
        axes[1, 0].plot(x, y, style, color=color, label=arm.replace("_", " "))
        grid = np.linspace(0.0, x.max() * 1.05, 100)
        axes[1, 0].plot(grid, fit["f_boundary"] + fit["A_boundary"] * grid, color=color, alpha=0.6)
    axes[1, 0].set(xlabel=r"$1/N_y^2$", ylabel=r"$-\overline{\lambda_0^{W_3}}/N_y$")
    axes[1, 0].legend(frameon=False)

    labels, values, errors = [], [], []
    for arm in ("exact_wall", "thin_strip"):
        fit = fits.get(arm, {})
        if fit.get("status") != "fit":
            continue
        labels.append(arm.replace("_", "\n"))
        values.append(fit["A_boundary"])
        low, high = fit["A_boundary_ci95"]
        errors.append((fit["A_boundary"] - low, high - fit["A_boundary"]))
    if values:
        axes[1, 1].errorbar(
            np.arange(len(values)), values, yerr=np.asarray(errors).T, fmt="o", color="black"
        )
        axes[1, 1].set_xticks(np.arange(len(values)), labels)
    axes[1, 1].axhline(0.0, color="0.3", ls="--", lw=0.8)
    axes[1, 1].set_ylabel(r"$A_{\partial}$")

    for label, axis in zip("abcd", axes.flat):
        axis.text(-0.16, 1.05, f"({label})", transform=axis.transAxes, fontweight="bold")
        axis.tick_params(top=True, right=True)
    fig.savefig(path_root.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_root.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "campaign_config.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    campaign.validate_config(config)
    output_root = (
        args.output_root
        if args.output_root is not None
        else PACKAGE_ROOT / "outputs" / str(config["revision"])
    ).resolve()
    tasks = campaign.build_tasks(config, output_root, config_path)
    rows, curves = load_trajectories(tasks, allow_partial=args.allow_partial)
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    replicates = int(config["analysis"]["bootstrap_replicates"])
    seed = int(config["analysis"]["bootstrap_seed"])
    temporal = temporal_summary(rows, replicates=replicates, seed=seed)
    fits = {
        arm: finite_size_summary(rows, arm=arm, replicates=replicates, seed=seed + index)
        for index, arm in enumerate(("exact_wall", "thin_strip"))
    }
    write_csv(analysis_root / "trajectory_slopes.csv", rows)
    write_csv(analysis_root / "temporal_stability.csv", temporal)
    summary = {
        "schema": "boundary_only_record_free_energy_analysis_v1",
        "revision": config["revision"],
        "verified_trajectory_count": len(rows),
        "complete_campaign": len(rows) == len(tasks),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "finite_size_fits": fits,
        "interpretation": (
            "This is a boundary-restricted circuit. Coefficients are reported directly; "
            "the per-wall value assumes additive independent walls and is not an algebraic "
            "subtraction from the full-slab campaign."
        ),
    }
    (analysis_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(
        analysis_root / "boundary_only_record_free_energy",
        rows,
        curves,
        temporal,
        fits,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
