#!/usr/bin/env python3
"""Compare the original W3 estimator with one slope fit over [2 Ny, 4 Ny]."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import analyze_campaign as base
import run_campaign as campaign


PACKAGE_ROOT = Path(__file__).resolve().parent


def grouped_values(
    rows: list[dict[str, Any]], arm: str, field: str
) -> dict[int, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            grouped[int(row["Ny"])].append(float(row[field]))
    return {ny: np.asarray(values, dtype=float) for ny, values in sorted(grouped.items())}


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in ("exact_wall", "thin_strip"):
        w3 = grouped_values(rows, arm, "lambda_W3")
        w23 = grouped_values(rows, arm, "lambda_W23")
        for ny in sorted(w3):
            density_w3 = -w3[ny] / ny
            density_w23 = -w23[ny] / ny
            sem_w3 = float(np.std(density_w3, ddof=1) / np.sqrt(density_w3.size))
            sem_w23 = float(np.std(density_w23, ddof=1) / np.sqrt(density_w23.size))
            output.append(
                {
                    "arm": arm,
                    "Ny": ny,
                    "samples": int(density_w3.size),
                    "mean_density_W3": float(np.mean(density_w3)),
                    "sem_density_W3": sem_w3,
                    "mean_density_W23": float(np.mean(density_w23)),
                    "sem_density_W23": sem_w23,
                    "mean_density_shift_W23_minus_W3": float(
                        np.mean(density_w23 - density_w3)
                    ),
                    "sem_ratio_W23_over_W3": float(sem_w23 / sem_w3),
                }
            )
    return output


def paired_coefficient_bootstrap(
    rows: list[dict[str, Any]], *, arm: str, replicates: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    arm_rows = [row for row in rows if row["arm"] == arm]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in arm_rows:
        grouped[int(row["Ny"])].append(row)
    nys = np.asarray(sorted(grouped), dtype=float)
    x = 1.0 / nys**2
    boot_w3 = np.empty(replicates, dtype=float)
    boot_w23 = np.empty(replicates, dtype=float)
    for index in range(replicates):
        y_w3: list[float] = []
        y_w23: list[float] = []
        for ny in nys.astype(int):
            group = grouped[int(ny)]
            draw = rng.integers(0, len(group), size=len(group))
            y_w3.append(-float(np.mean([group[item]["lambda_W3"] for item in draw])) / ny)
            y_w23.append(-float(np.mean([group[item]["lambda_W23"] for item in draw])) / ny)
        boot_w3[index] = np.polyfit(x, np.asarray(y_w3), 1)[0]
        boot_w23[index] = np.polyfit(x, np.asarray(y_w23), 1)[0]
    delta = boot_w23 - boot_w3
    return {
        "arm": arm,
        "replicates": replicates,
        "A_W3_ci95": list(base.percentile_interval(boot_w3)),
        "A_W23_ci95": list(base.percentile_interval(boot_w23)),
        "A_W23_minus_W3_mean": float(np.mean(delta)),
        "A_W23_minus_W3_ci95": list(base.percentile_interval(delta)),
        "bootstrap_sd_A_W3": float(np.std(boot_w3, ddof=1)),
        "bootstrap_sd_A_W23": float(np.std(boot_w23, ddof=1)),
        "bootstrap_sd_ratio_W23_over_W3": float(
            np.std(boot_w23, ddof=1) / np.std(boot_w3, ddof=1)
        ),
    }


def make_figure(
    path_root: Path,
    comparisons: list[dict[str, Any]],
    fits: dict[str, dict[str, dict[str, Any]]],
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
    arm_styles = {
        "exact_wall": ("o", "#1f77b4"),
        "thin_strip": ("s", "#d62728"),
    }
    for arm, (marker, color) in arm_styles.items():
        selected = [row for row in comparisons if row["arm"] == arm]
        ny = np.asarray([row["Ny"] for row in selected])
        for field, sem, label, offset, fill in (
            ("mean_density_W3", "sem_density_W3", r"$W_3$", -0.15, "none"),
            ("mean_density_W23", "sem_density_W23", r"$W_{23}$", 0.15, color),
        ):
            axes[0, 0].errorbar(
                ny + offset,
                [row[field] for row in selected],
                yerr=[row[sem] for row in selected],
                fmt=marker,
                mfc=fill,
                mec=color,
                ecolor=color,
                capsize=2,
                label=f"{arm.replace('_', ' ')} {label}",
            )
        axes[0, 1].plot(
            ny,
            [row["sem_ratio_W23_over_W3"] for row in selected],
            marker + "-",
            color=color,
            label=arm.replace("_", " "),
        )
    axes[0, 0].set(xlabel=r"$N_y$", ylabel=r"$-\overline{\lambda_0}/N_y$")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 1].axhline(1.0, color="0.3", ls="--", lw=0.8)
    axes[0, 1].set(xlabel=r"$N_y$", ylabel=r"SEM$(W_{23})$/SEM$(W_3)$")
    axes[0, 1].legend(frameon=False)

    for arm, (_, color) in arm_styles.items():
        fit_w3 = fits[arm]["W3"]
        fit_w23 = fits[arm]["W23"]
        x = np.asarray(fit_w23["plot"]["x"])
        grid = np.linspace(0.0, x.max() * 1.05, 100)
        axes[1, 0].plot(
            x,
            fit_w23["plot"]["y"],
            "o" if arm == "exact_wall" else "s",
            color=color,
            label=arm.replace("_", " "),
        )
        axes[1, 0].plot(
            grid,
            fit_w23["f_boundary"] + fit_w23["A_boundary"] * grid,
            color=color,
        )
        axes[1, 0].plot(
            grid,
            fit_w3["f_boundary"] + fit_w3["A_boundary"] * grid,
            color=color,
            ls="--",
            alpha=0.65,
        )
    axes[1, 0].set(
        xlabel=r"$1/N_y^2$", ylabel=r"$-\overline{\lambda_0^{W_{23}}}/N_y$"
    )
    axes[1, 0].legend(frameon=False)

    positions = np.arange(2)
    width = 0.28
    for offset, window, fill in ((-width / 2, "W3", "none"), (width / 2, "W23", "black")):
        values, lower, upper = [], [], []
        for arm in ("exact_wall", "thin_strip"):
            fit = fits[arm][window]
            value = float(fit["A_boundary"])
            low, high = fit["A_boundary_ci95"]
            values.append(value)
            lower.append(value - low)
            upper.append(high - value)
        axes[1, 1].errorbar(
            positions + offset,
            values,
            yerr=[lower, upper],
            fmt="o",
            mfc=fill,
            mec="black",
            ecolor="black",
            capsize=2,
            label=rf"${window}$",
        )
    axes[1, 1].axhline(0.0, color="0.3", ls="--", lw=0.8)
    axes[1, 1].set_xticks(positions, ["exact wall", "thin strip"])
    axes[1, 1].set_ylabel(r"$A_{\partial}$")
    axes[1, 1].legend(frameon=False)

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
    rows, _ = base.load_trajectories(tasks, allow_partial=args.allow_partial)
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    replicates = int(config["analysis"]["bootstrap_replicates"])
    seed = int(config["analysis"]["bootstrap_seed"]) + 100
    comparisons = comparison_rows(rows)
    fits: dict[str, dict[str, dict[str, Any]]] = {}
    paired: dict[str, dict[str, Any]] = {}
    for index, arm in enumerate(("exact_wall", "thin_strip")):
        fits[arm] = {
            "W3": base.finite_size_summary(
                rows,
                arm=arm,
                replicates=replicates,
                seed=seed + 10 * index,
                slope_field="lambda_W3",
            ),
            "W23": base.finite_size_summary(
                rows,
                arm=arm,
                replicates=replicates,
                seed=seed + 10 * index + 1,
                slope_field="lambda_W23",
            ),
        }
        paired[arm] = paired_coefficient_bootstrap(
            rows, arm=arm, replicates=replicates, seed=seed + 10 * index + 2
        )
    base.write_csv(analysis_root / "trajectory_slopes_combined_window.csv", rows)
    base.write_csv(analysis_root / "combined_window_comparison.csv", comparisons)
    summary = {
        "schema": "boundary_only_record_free_energy_combined_window_v1",
        "revision": config["revision"],
        "verified_trajectory_count": len(rows),
        "complete_campaign": len(rows) == len(tasks),
        "combined_window": "[2 Ny, 4 Ny]",
        "estimator": (
            "One ordinary least-squares slope per trajectory; ensemble uncertainty uses "
            "stratified whole-trajectory bootstrap resampling. Cycles are never treated as "
            "independent samples."
        ),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "finite_size_fits": fits,
        "paired_coefficient_comparison": paired,
    }
    (analysis_root / "analysis_summary_combined_window.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figure(
        analysis_root / "boundary_only_record_free_energy_combined_window",
        comparisons,
        fits,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
