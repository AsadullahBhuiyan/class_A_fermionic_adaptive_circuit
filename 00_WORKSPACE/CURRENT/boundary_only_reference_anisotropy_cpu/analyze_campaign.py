#!/usr/bin/env python3
"""Analyze the completed boundary-reference anisotropy calibration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import run_campaign as campaign
from scientific import stable_seed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_size_fit(
    nys: np.ndarray, alpha_values: np.ndarray, bootstrap_alpha: np.ndarray
) -> dict[str, Any]:
    x = 1.0 / nys**2
    slope, intercept = np.polyfit(x, alpha_values, 1)
    omit_slope, omit_intercept = np.polyfit(x[1:], alpha_values[1:], 1)
    design = np.column_stack([np.ones_like(x), x, x**2])
    quartic_intercept, quartic_slope, quartic_coefficient = np.linalg.lstsq(
        design, alpha_values, rcond=None
    )[0]
    draws = bootstrap_alpha.shape[1]
    boot_intercept = np.empty(draws, dtype=np.float64)
    boot_omit = np.empty(draws, dtype=np.float64)
    boot_quartic = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        values = bootstrap_alpha[:, draw]
        boot_intercept[draw] = np.polyfit(x, values, 1)[1]
        boot_omit[draw] = np.polyfit(x[1:], values[1:], 1)[1]
        boot_quartic[draw] = np.linalg.lstsq(design, values, rcond=None)[0][0]
    ci = [float(value) for value in np.quantile(boot_intercept, [0.025, 0.975])]
    return {
        "alpha_infinity": float(intercept),
        "alpha_infinity_ci95": ci,
        "b_over_Ny2": float(slope),
        "omit_Ny16": {
            "alpha_infinity": float(omit_intercept),
            "b_over_Ny2": float(omit_slope),
            "bootstrap_ci95": [
                float(value) for value in np.quantile(boot_omit, [0.025, 0.975])
            ],
        },
        "quartic_sensitivity": {
            "alpha_infinity": float(quartic_intercept),
            "b_over_Ny2": float(quartic_slope),
            "c_over_Ny4": float(quartic_coefficient),
            "bootstrap_ci95": [
                float(value) for value in np.quantile(boot_quartic, [0.025, 0.975])
            ],
        },
        "plot": {"x": x.tolist(), "y": alpha_values.tolist()},
        "bootstrap_alpha_infinity": boot_intercept,
    }


def load_existing_casimir_diagnostic(alpha_infinity: float | None) -> dict[str, Any]:
    source = (
        REPO_ROOT
        / "00_WORKSPACE/CURRENT/boundary_only_record_free_energy_cpu_pilot/outputs"
        / "boundary_only_flattened_ground_nx16_ny16-32_s100_4ny_v1/analysis"
        / "analysis_summary_combined_window.json"
    )
    if not source.exists() or alpha_infinity is None or alpha_infinity <= 0.0:
        return {"status": "unavailable", "source": str(source)}
    payload = json.loads(source.read_text(encoding="utf-8"))
    exact = payload["finite_size_fits"]["exact_wall"]
    output = {
        "status": "diagnostic_only",
        "source": str(source),
        "warning": (
            "Dividing by the calibrated anisotropy does not repair the documented "
            "Casimir fit-window instability."
        ),
    }
    for window in ("W3", "W23"):
        alpha_c = float(exact[window]["alpha_c_eff_per_wall_if_independent"])
        output[window] = {"alpha_c_eff_per_wall": alpha_c, "c_eff_per_wall": alpha_c / alpha_infinity}
    return output


def make_figure(
    root: Path,
    audit: dict[str, Any],
    size_payloads: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    fit: dict[str, Any] | None,
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
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.8), constrained_layout=True)

    gates = audit.get("screen_gates", {})
    for key, gate in gates.items():
        smaller, larger = (int(value) for value in key.split("_vs_"))
        for multiplier, field, marker in (
            (smaller, "smaller", "o"),
            (larger, "larger", "s"),
        ):
            value = gate.get(field, {}).get("alpha")
            if value is not None:
                axes[0, 0].plot(multiplier, value, marker, color="#1f77b4")
    axes[0, 0].set(xlabel=r"burn-in/$N_y$", ylabel=r"audit $\alpha$")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(size_payloads))))
    for color, payload in zip(colors, size_payloads):
        estimate = payload["final"]["estimate"]
        axes[0, 1].plot(
            estimate["temporal_separations"],
            estimate["temporal_means"],
            "o-",
            ms=3,
            color=color,
            label=str(payload["Ny"]),
        )
        axes[0, 1].axhline(estimate["spatial_mean"], color=color, ls="--", alpha=0.55)
    axes[0, 1].set(xlabel=r"$\delta\tau$", ylabel=r"plateau $I(R_1:R_2)$")
    axes[0, 1].legend(title=r"$N_y$", frameon=False, ncol=2, fontsize=6)

    ny = np.asarray([row["Ny"] for row in rows], dtype=float)
    alpha = np.asarray([row["alpha"] for row in rows], dtype=float)
    low = np.asarray([row["alpha_ci_low"] for row in rows], dtype=float)
    high = np.asarray([row["alpha_ci_high"] for row in rows], dtype=float)
    axes[0, 2].errorbar(ny, alpha, yerr=[alpha - low, high - alpha], fmt="o-", capsize=2)
    axes[0, 2].set(xlabel=r"$N_y$", ylabel=r"$\alpha(N_y)$")

    if fit is not None:
        x = np.asarray(fit["plot"]["x"])
        y = np.asarray(fit["plot"]["y"])
        grid = np.linspace(0.0, 1.05 * x.max(), 100)
        axes[1, 0].plot(x, y, "o", color="#1f77b4")
        axes[1, 0].plot(grid, fit["alpha_infinity"] + fit["b_over_Ny2"] * grid, color="#1f77b4")
    axes[1, 0].set(xlabel=r"$1/N_y^2$", ylabel=r"$\alpha(N_y)$")

    for payload, color in zip(size_payloads, colors):
        history = payload["sample_history"]
        axes[1, 1].plot(
            [entry["samples"] for entry in history],
            [entry["relative_ci_half_width"] for entry in history],
            "o-",
            color=color,
            label=str(payload["Ny"]),
        )
    axes[1, 1].axhline(0.10, color="0.3", ls="--")
    axes[1, 1].set(xlabel="trajectories", ylabel="relative 95% half-width")

    wall0, wall1 = [], []
    for payload in size_payloads:
        walls = payload["final"]["wall_gate"].get("walls", [])
        wall0.append(np.nan if len(walls) < 2 or walls[0]["alpha"] is None else walls[0]["alpha"])
        wall1.append(np.nan if len(walls) < 2 or walls[1]["alpha"] is None else walls[1]["alpha"])
    axes[1, 2].plot(ny, wall0, "o-", label="left wall")
    axes[1, 2].plot(ny, wall1, "s--", label="right wall")
    axes[1, 2].set(xlabel=r"$N_y$", ylabel=r"wall-resolved $\alpha$")
    axes[1, 2].legend(frameon=False)

    for label, axis in zip("abcdef", axes.flat):
        axis.text(-0.18, 1.05, f"({label})", transform=axis.transAxes, fontweight="bold")
        axis.tick_params(top=True, right=True)
    fig.savefig(root.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(root.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "campaign_config.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    config = campaign.load_config(args.config.resolve())
    output_root = (
        args.output_root
        if args.output_root is not None
        else PACKAGE_ROOT / "outputs" / str(config["revision"])
    ).resolve()
    analysis_root = output_root / "analysis"
    audit_path = analysis_root / "audit_decision.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"status": "missing"}
    missing, size_payloads = [], []
    for ny in config["geometry"]["Ny_values"]:
        path = analysis_root / f"Ny{int(ny):03d}_result.json"
        if path.exists():
            size_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(int(ny))
    if missing and not args.allow_partial:
        raise RuntimeError(f"production results are missing for Ny={missing}")
    size_payloads.sort(key=lambda payload: int(payload["Ny"]))
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[np.ndarray] = []
    draws = int(config["production"]["bootstrap_draws"])
    for payload in size_payloads:
        final = payload["final"]
        estimate = final["estimate"]
        ci = estimate["alpha_ci95"] or [float("nan"), float("nan")]
        rows.append(
            {
                "Ny": int(payload["Ny"]),
                "status": payload["status"],
                "samples": int(final["samples"]),
                "alpha": float(estimate["alpha"]),
                "alpha_ci_low": float(ci[0]),
                "alpha_ci_high": float(ci[1]),
                "relative_ci_half_width": float(final["relative_ci_half_width"]),
                "bootstrap_resolved_fraction": float(estimate["bootstrap_resolved_fraction"]),
                "wall_gate_passed": bool(final["wall_gate"]["passed"]),
            }
        )
        grid = campaign.grid_values(
            output_root,
            config,
            int(payload["Ny"]),
            int(final["samples"]),
            int(payload["burn_in_multiplier"]),
            int(payload["follow_multiplier"]),
            payload["fine_separations"],
        )
        resampled = campaign.bootstrap_grid(
            grid,
            int(payload["Ny"]),
            draws=draws,
            seed=stable_seed(
                int(config["root_seed"]), int(payload["Ny"]), 20000, "bootstrap"
            ),
        )["bootstrap_alpha"]
        if not np.all(np.isfinite(resampled)):
            raise RuntimeError(f"finite-size bootstrap has unresolved draws for Ny={payload['Ny']}")
        bootstrap_rows.append(np.asarray(resampled, dtype=np.float64))
    fit = None
    finite_size_passed = False
    if len(rows) == len(config["geometry"]["Ny_values"]):
        nys = np.asarray([row["Ny"] for row in rows], dtype=float)
        alphas = np.asarray([row["alpha"] for row in rows], dtype=float)
        fit = finite_size_fit(nys, alphas, np.stack(bootstrap_rows))
        ci = fit["alpha_infinity_ci95"]
        relative_half_width = (ci[1] - ci[0]) / (2.0 * abs(fit["alpha_infinity"]))
        omit_shift = abs(fit["omit_Ny16"]["alpha_infinity"] - fit["alpha_infinity"]) / abs(
            fit["alpha_infinity"]
        )
        quartic_shift = abs(
            fit["quartic_sensitivity"]["alpha_infinity"] - fit["alpha_infinity"]
        ) / abs(fit["alpha_infinity"])
        finite_size_passed = bool(
            all(row["status"] == "calibrated" for row in rows)
            and relative_half_width
            <= float(config["finite_size"]["relative_ci_half_width_target"])
            and omit_shift <= float(config["finite_size"]["sensitivity_relative_tolerance"])
            and quartic_shift <= float(config["finite_size"]["sensitivity_relative_tolerance"])
        )
        fit.update(
            {
                "relative_ci_half_width": float(relative_half_width),
                "omit_Ny16_relative_shift": float(omit_shift),
                "quartic_relative_shift": float(quartic_shift),
                "accepted": finite_size_passed,
            }
        )
        fit.pop("bootstrap_alpha_infinity", None)
    alpha_infinity = None if fit is None else float(fit["alpha_infinity"])
    summary = {
        "schema": "boundary_reference_anisotropy_analysis_v1",
        "revision": config["revision"],
        "status": "calibrated" if finite_size_passed else "unresolved",
        "audit_status": audit.get("status"),
        "missing_sizes": missing,
        "independent_sampling_unit": "complete base Born trajectory",
        "bootstrap_draws": draws,
        "per_size": rows,
        "finite_size_fit": fit,
        "existing_casimir_diagnostic": load_existing_casimir_diagnostic(alpha_infinity),
    }
    analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_root / "anisotropy_by_size.csv", rows)
    (analysis_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if rows:
        make_figure(
            analysis_root / "boundary_reference_anisotropy",
            audit,
            size_payloads,
            rows,
            fit,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if finite_size_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
