#!/usr/bin/env python3
"""Analyze the trajectory-resolved max-mix many-body Lyapunov CPU pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
from run_campaign import build_tasks, save_json_atomic, validate_config, verify_complete
from lyapunov_observer import leading_log_sigma2_levels


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.shape != x.shape or not np.all(np.isfinite(y)):
        raise ValueError("a slope needs at least two finite aligned observations")
    centered = x - x.mean()
    return float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))


def trajectory_window_slopes(
    cycles: np.ndarray,
    levels: np.ndarray,
    *,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    cycles = np.asarray(cycles, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    middle_mask = (cycles >= ny) & (cycles <= 1.5 * ny)
    late_mask = (cycles >= 1.5 * ny) & (cycles <= 2 * ny)
    middle = np.asarray(
        [linear_slope(cycles[middle_mask], levels[middle_mask, index]) for index in range(levels.shape[1])]
    )
    late = np.asarray(
        [linear_slope(cycles[late_mask], levels[late_mask, index]) for index in range(levels.shape[1])]
    )
    return middle, late


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(low), float(high)


def paired_convergence(
    middle: np.ndarray,
    late: np.ndarray,
    *,
    bootstrap_count: int,
    rng: np.random.Generator,
    relative_threshold: float,
) -> list[dict[str, Any]]:
    middle = np.asarray(middle, dtype=np.float64)
    late = np.asarray(late, dtype=np.float64)
    sample_count = middle.shape[0]
    indices = rng.integers(0, sample_count, size=(bootstrap_count, sample_count))
    middle_boot = middle[indices].mean(axis=1)
    late_boot = late[indices].mean(axis=1)
    metrics = [("lambda0", 0)] + [(f"gap{i}", i) for i in range(1, 5)]
    rows: list[dict[str, Any]] = []
    for name, index in metrics:
        if index == 0:
            middle_values = middle[:, 0]
            late_values = late[:, 0]
            boot_shift = late_boot[:, 0] - middle_boot[:, 0]
            reference = abs(float(late_values.mean()))
        else:
            middle_values = middle[:, 0] - middle[:, index]
            late_values = late[:, 0] - late[:, index]
            boot_shift = (
                late_boot[:, 0] - late_boot[:, index]
                - middle_boot[:, 0]
                + middle_boot[:, index]
            )
            reference = abs(float(late_values.mean()))
        shift = float(late_values.mean() - middle_values.mean())
        low, high = percentile_interval(boot_shift)
        relative = abs(shift) / reference if reference > 0 else math.inf
        rows.append(
            {
                "metric": name,
                "middle_mean": float(middle_values.mean()),
                "late_mean": float(late_values.mean()),
                "shift": shift,
                "shift_ci_low": low,
                "shift_ci_high": high,
                "shift_ci_contains_zero": bool(low <= 0.0 <= high),
                "relative_shift": relative,
                "passes": bool(low <= 0.0 <= high and relative <= relative_threshold),
            }
        )
    return rows


def fit_finite_size(ny_values: np.ndarray, lambda_values: np.ndarray) -> dict[str, Any]:
    ny = np.asarray(ny_values, dtype=np.float64)
    lam = np.asarray(lambda_values, dtype=np.float64)
    x = 1.0 / ny**2
    f0 = -lam[:, 0] / ny
    primary = np.linalg.lstsq(np.column_stack((np.ones(ny.size), x)), f0, rcond=None)[0]
    subleading = np.linalg.lstsq(
        np.column_stack((np.ones(ny.size), x, x**2)), f0, rcond=None
    )[0]
    omit = np.linalg.lstsq(
        np.column_stack((np.ones(ny.size - 1), x[1:])), f0[1:], rcond=None
    )[0]
    a0 = float(primary[1])
    gap_rows = []
    for index in range(1, min(5, lam.shape[1])):
        gap_density = (lam[:, 0] - lam[:, index]) / ny
        fit = np.linalg.lstsq(
            np.column_stack((np.ones(ny.size), x)), gap_density, rcond=None
        )[0]
        ai = float(fit[1])
        gap_rows.append(
            {
                "level": index,
                "intercept": float(fit[0]),
                "Ai": ai,
                "alpha_x_typ": ai / (2.0 * math.pi),
                "x_typ_over_c_eff": -ai / (12.0 * a0),
            }
        )
    return {
        "Ny": ny.astype(int).tolist(),
        "inverse_Ny_squared": x.tolist(),
        "f0_tilde": f0.tolist(),
        "primary": {
            "intercept": float(primary[0]),
            "A0": a0,
            "alpha_c_eff": -6.0 * a0 / math.pi,
        },
        "with_inverse_Ny_fourth": {
            "intercept": float(subleading[0]),
            "A0": float(subleading[1]),
            "B0": float(subleading[2]),
        },
        "omit_Ny20": {"intercept": float(omit[0]), "A0": float(omit[1])},
        "gaps": gap_rows,
    }


def bootstrap_finite_size(
    by_ny: dict[int, dict[str, np.ndarray]], *, count: int, rng: np.random.Generator
) -> dict[str, Any]:
    ny_values = np.asarray(sorted(by_ny), dtype=np.int64)
    alpha_c = np.empty(count, dtype=np.float64)
    alpha_x = np.empty((count, 4), dtype=np.float64)
    ratios = np.empty((count, 4), dtype=np.float64)
    for replicate in range(count):
        sampled = []
        for ny in ny_values:
            values = by_ny[int(ny)]["late_slopes"]
            chosen = rng.integers(0, values.shape[0], size=values.shape[0])
            sampled.append(values[chosen].mean(axis=0))
        fit = fit_finite_size(ny_values, np.stack(sampled))
        alpha_c[replicate] = fit["primary"]["alpha_c_eff"]
        alpha_x[replicate] = [row["alpha_x_typ"] for row in fit["gaps"][:4]]
        ratios[replicate] = [row["x_typ_over_c_eff"] for row in fit["gaps"][:4]]
    return {
        "alpha_c_eff_ci95": percentile_interval(alpha_c),
        "alpha_x_typ_ci95": [percentile_interval(alpha_x[:, index]) for index in range(4)],
        "x_typ_over_c_eff_ci95": [percentile_interval(ratios[:, index]) for index in range(4)],
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )


def make_figure(
    by_ny: dict[int, dict[str, np.ndarray]],
    finite: dict[str, Any],
    output_root: Path,
) -> None:
    configure_plotting()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(by_ny)))
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.45), constrained_layout=True)
    for color, ny in zip(colors, sorted(by_ny)):
        row = by_ny[ny]
        cycles = row["spectrum_cycles"]
        valid = cycles > 0
        mean_levels = row["levels"].mean(axis=0)
        axes[0, 0].plot(cycles[valid] / ny, -mean_levels[valid, 0] / cycles[valid], "o-", ms=2.5, color=color, label=fr"$N_y={ny}$")
        omega = row["cumulative_log_probability"].mean(axis=0)
        all_cycles = np.arange(1, omega.size)
        axes[0, 1].plot(all_cycles / ny, -omega[1:] / all_cycles, "-", lw=1, color=color)
        profile = row["soft_mode_x_profiles"][:, -1, 0].mean(axis=0)
        axes[1, 2].plot(np.arange(profile.size), profile, "o-", ms=2, color=color, label=fr"$N_y={ny}$")
    ny_values = np.asarray(finite["Ny"])
    x = np.asarray(finite["inverse_Ny_squared"])
    f0 = np.asarray(finite["f0_tilde"])
    axes[0, 2].plot(x, f0, "o", color="tab:blue")
    fit_x = np.linspace(0, x.max() * 1.05, 100)
    axes[0, 2].plot(
        fit_x,
        finite["primary"]["intercept"] + finite["primary"]["A0"] * fit_x,
        "--",
        color="black",
    )
    axes[1, 0].plot(ny_values, [by_ny[int(ny)]["free_energy_rate"].mean() for ny in ny_values], "s--", color="tab:green", label=r"$-\omega/T$")
    axes[1, 0].plot(ny_values, [-by_ny[int(ny)]["late_slopes"][:, 0].mean() for ny in ny_values], "o-", color="tab:blue", label=r"$-\lambda_0$")
    gap_rows = finite["gaps"]
    axes[1, 1].bar(
        [row["level"] for row in gap_rows],
        [row["x_typ_over_c_eff"] for row in gap_rows],
        color="tab:purple",
    )
    axes[0, 0].set(xlabel=r"$t/N_y$", ylabel=r"$-\ell_0(t)/t$")
    axes[0, 1].set(xlabel=r"$t/N_y$", ylabel=r"$-\langle\omega(t)\rangle/t$")
    axes[0, 2].set(xlabel=r"$1/N_y^2$", ylabel=r"$-\lambda_0/N_y$")
    axes[1, 0].set(xlabel=r"$N_y$", ylabel="rate")
    axes[1, 1].set(xlabel="level $i$", ylabel=r"$x_i^{\rm typ}/c_{\rm eff}$")
    axes[1, 2].set(xlabel="$x$", ylabel="soft-mode weight")
    axes[0, 0].legend(ncol=2, fontsize=6.5)
    axes[1, 0].legend(fontsize=7)
    for label, axis in zip("abcdef", axes.flat):
        axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / "many_body_lyapunov_pilot.pdf")
    fig.savefig(output_root / "many_body_lyapunov_pilot.png", dpi=300)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "campaign_config.json")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bootstrap-count", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    results_root = (
        args.results_root
        if args.results_root is not None
        else PACKAGE_ROOT / "outputs" / config["revision"]
    ).resolve()
    output_root = (
        args.output_root
        if args.output_root is not None
        else PACKAGE_ROOT / "analysis_outputs" / config["revision"]
    ).resolve()
    tasks = build_tasks(config, results_root)
    verified = [(task, verify_complete(task)) for task in tasks]
    available = [task for task, (valid, _) in verified if valid]
    if len(available) != len(tasks) and not args.allow_incomplete:
        raise RuntimeError(f"campaign is incomplete: {len(available)}/{len(tasks)} verified trajectories")
    bootstrap_count = int(args.bootstrap_count or config["analysis"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["analysis"]["bootstrap_seed"]))
    by_ny: dict[int, dict[str, list[np.ndarray]]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    for task in available:
        with np.load(task.result_path, allow_pickle=False) as data:
            spectrum_cycles = np.asarray(data["spectrum_cycles"], dtype=np.int64)
            occupations = np.asarray(data["occupations"], dtype=np.float64)
            log_z = np.asarray(data["log_z"], dtype=np.float64)
            levels = np.stack(
                [
                    leading_log_sigma2_levels(
                        occupations[position],
                        log_z[position],
                        count=int(config["observer"]["leading_level_count"]),
                        cap_tolerance=float(config["observer"]["cap_tolerance"]),
                    )
                    for position in range(spectrum_cycles.size)
                ]
            )
            stored_levels = np.asarray(data["leading_log_sigma2"], dtype=np.float64)
            if not np.array_equal(levels, stored_levels):
                raise RuntimeError(
                    f"offline level reconstruction disagrees with saved values for {task.task_id}"
                )
            middle, late = trajectory_window_slopes(spectrum_cycles, levels, ny=task.ny)
            omega = np.asarray(data["cumulative_log_probability"], dtype=np.float64)
            store = by_ny.setdefault(task.ny, {})
            for key, value in (
                ("middle_slopes", middle),
                ("late_slopes", late),
                ("levels", levels),
                ("cumulative_log_probability", omega),
                ("free_energy_rate", np.asarray(-omega[-1] / task.cycles)),
                ("soft_mode_x_profiles", np.asarray(data["soft_mode_x_profiles"], dtype=np.float64)),
            ):
                store.setdefault(key, []).append(value)
            store["spectrum_cycles"] = [spectrum_cycles]
        trajectory_rows.append(
            {
                "task_id": task.task_id,
                "Ny": task.ny,
                "sample_index": task.sample_index,
                "seed": task.seed,
                "free_energy_rate": float(-omega[-1] / task.cycles),
                **{f"middle_lambda_{i}": float(middle[i]) for i in range(5)},
                **{f"late_lambda_{i}": float(late[i]) for i in range(5)},
            }
        )
    normalized: dict[int, dict[str, np.ndarray]] = {}
    for ny, rows in by_ny.items():
        normalized[ny] = {
            key: (value[0] if key == "spectrum_cycles" else np.stack(value))
            for key, value in rows.items()
        }
    convergence_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []
    for ny in sorted(normalized):
        values = normalized[ny]
        rows = paired_convergence(
            values["middle_slopes"],
            values["late_slopes"],
            bootstrap_count=bootstrap_count,
            rng=rng,
            relative_threshold=float(config["analysis"]["relative_shift_threshold"]),
        )
        convergence_rows.extend({"Ny": ny, **row} for row in rows)
        lambda0 = values["late_slopes"][:, 0]
        free = values["free_energy_rate"]
        size_rows.append(
            {
                "Ny": ny,
                "samples": int(lambda0.size),
                "late_lambda0_mean": float(lambda0.mean()),
                "late_lambda0_sem": float(lambda0.std(ddof=1) / math.sqrt(lambda0.size)),
                "free_energy_rate_mean": float(free.mean()),
                "free_energy_closure": float(free.mean() + lambda0.mean()),
                "T_2Ny_sufficient": bool(all(row["passes"] for row in rows)),
            }
        )
    means = np.stack([normalized[ny]["late_slopes"].mean(axis=0) for ny in sorted(normalized)])
    finite = fit_finite_size(np.asarray(sorted(normalized)), means)
    finite["bootstrap"] = bootstrap_finite_size(normalized, count=bootstrap_count, rng=rng)
    finite["provisional_only"] = True
    finite["absolute_c_eff_not_claimed"] = True
    sample_rows = []
    for ny in sorted(normalized):
        late = normalized[ny]["late_slopes"]
        for prefix in config["analysis"]["sample_prefixes"]:
            if late.shape[0] < int(prefix):
                continue
            subset = late[: int(prefix)]
            chosen = rng.integers(0, subset.shape[0], size=(bootstrap_count, subset.shape[0]))
            bootstrap_means = subset[chosen].mean(axis=1)
            metrics = [("lambda0", subset[:, 0], bootstrap_means[:, 0])]
            metrics.extend(
                (
                    f"gap{index}",
                    subset[:, 0] - subset[:, index],
                    bootstrap_means[:, 0] - bootstrap_means[:, index],
                )
                for index in range(1, 5)
            )
            for metric, values, boot in metrics:
                low, high = percentile_interval(boot)
                sample_rows.append(
                    {
                        "Ny": ny,
                        "samples": int(prefix),
                        "metric": metric,
                        "mean": float(values.mean()),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "trajectory_slopes.csv", trajectory_rows)
    write_csv(output_root / "size_summary.csv", size_rows)
    write_csv(output_root / "temporal_convergence.csv", convergence_rows)
    write_csv(output_root / "sample_convergence.csv", sample_rows)
    save_json_atomic(
        output_root / "analysis_summary.json",
        {
            "revision": config["revision"],
            "verified_trajectories": len(available),
            "requested_trajectories": len(tasks),
            "bootstrap_replicates": bootstrap_count,
            "finite_size": finite,
            "size_summary": size_rows,
            "temporal_convergence": convergence_rows,
            "interpretation": (
                "Finite-size coefficients are preliminary. The pilot reports alpha*c_eff, "
                "alpha*x_i, and x_i/c_eff; it does not determine alpha or an absolute c_eff."
            ),
        },
    )
    if len(normalized) == 5:
        make_figure(normalized, finite, output_root)
    print(f"[analysis] wrote {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
