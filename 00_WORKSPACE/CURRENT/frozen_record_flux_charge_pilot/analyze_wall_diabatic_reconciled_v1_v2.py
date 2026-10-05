#!/usr/bin/env python3
"""Reconcile the 1,595 immutable v1 pumps with the five verified v2 recoveries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_wall_diabatic_width_sweep as analysis


PROJECT_ROOT = Path(__file__).resolve().parent
V1_ROOT = PROJECT_ROOT / "results/N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1"
V2_ROOT = PROJECT_ROOT / "results/N20_24_28_32x24_wall_diabatic_spectral_pump_numerical_recovery_v2"
OUTPUT_ROOT = PROJECT_ROOT / "results/N20_24_28_32x24_wall_diabatic_spectral_pump_reconciled_v1_v2"
PROTOCOLS = ("nsh1", "dense")
WALLS = ("soft", "hard")
DIRECTIONS = (("ccw", 1), ("cw", -1))
WIDTHS = (20, 24, 28, 32)
COLORS = ("#D55E00", "#009E73", "#0072B2", "#CC79A7")
LINESTYLES = (":", "--", "-", "-.")


def _primary_v1(row: dict[str, Any]) -> bool:
    return (
        row.get("result_collection") == "pump"
        and row.get("variant") in (None, "", "primary")
        and bool(row.get("is_primary", True))
    )


def load_reconciled() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v1, invalid_v1 = analysis.load_path_rows(V1_ROOT)
    v2, invalid_v2 = analysis.load_path_rows(V2_ROOT)
    primary_v1 = [row for row in v1 if _primary_v1(row)]
    recovery_v2 = [row for row in v2 if row.get("result_collection") == "pump"]
    index = {(row["task_id"], row["direction"]): row for row in primary_v1}
    if len(index) != len(primary_v1):
        raise RuntimeError("v1 contains duplicate primary directional keys")
    recovered_keys: list[tuple[str, str]] = []
    for row in recovery_v2:
        key = (row["task_id"], row["direction"])
        if key in index:
            raise RuntimeError(f"v2 recovery overlaps a verified v1 key: {key}")
        index[key] = row
        recovered_keys.append(key)

    expected = {
        (protocol, nx, wall, sample, direction)
        for protocol in PROTOCOLS
        for nx in WIDTHS
        for wall in WALLS
        for sample in range(100)
        for direction, _ in DIRECTIONS
    }
    actual = {
        (
            str(row["protocol"]), int(row["Nx"]), str(row["wall"]),
            int(row["sample_id"]), str(row["direction"]),
        )
        for row in index.values()
    }
    if actual != expected:
        raise RuntimeError(
            f"reconciled matrix mismatch: missing={len(expected - actual)}, "
            f"extra={len(actual - expected)}"
        )
    if len(primary_v1) != 3190 or len(recovery_v2) != 10 or len(index) != 3200:
        raise RuntimeError("expected 3,190 v1 and 10 v2 directional paths")
    return list(index.values()), {
        "v1_directional_paths": len(primary_v1),
        "v1_pairs": len({row["task_id"] for row in primary_v1}),
        "v2_directional_paths": len(recovery_v2),
        "v2_pairs": len({row["task_id"] for row in recovery_v2}),
        "reconciled_directional_paths": len(index),
        "reconciled_pairs": len({row["task_id"] for row in index.values()}),
        "invalid_v1": invalid_v1,
        "invalid_v2": invalid_v2,
        "recovered_directional_keys": [list(key) for key in sorted(recovered_keys)],
    }


def statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        for wall in WALLS:
            for nx in WIDTHS:
                for direction, sigma in DIRECTIONS:
                    selected = [
                        row for row in rows
                        if str(row["protocol"]) == protocol
                        and str(row["wall"]) == wall
                        and int(row["Nx"]) == nx
                        and str(row["direction"]) == direction
                    ]
                    values = np.asarray([row["endpoint_q_x"] for row in selected], dtype=float)
                    resolved = np.asarray([row["resolved"] for row in selected], dtype=bool)
                    if values.size != 100:
                        raise RuntimeError(
                            f"expected S100 for {(protocol, wall, nx, direction)}, "
                            f"found {values.size}"
                        )
                    oriented = sigma * values
                    records.append({
                        "protocol": protocol,
                        "Nx": nx,
                        "Ny": 24,
                        "wall": wall,
                        "direction": direction,
                        "sample_count": int(values.size),
                        "endpoint_mean": float(np.mean(values)),
                        "endpoint_sd": float(np.std(values, ddof=1)),
                        "endpoint_median": float(np.median(values)),
                        "endpoint_minimum": float(np.min(values)),
                        "endpoint_maximum": float(np.max(values)),
                        "correct_sign_abs_qx_gt_0p5_fraction": float(np.mean(oriented > 0.5)),
                        "correct_sign_abs_qx_gt_0p9_fraction": float(np.mean(oriented > 0.9)),
                        "correct_sign_abs_qx_0p95_to_1p05_fraction": float(
                            np.mean((oriented >= 0.95) & (oriented <= 1.05))
                        ),
                        "near_zero_abs_qx_lt_0p1_fraction": float(np.mean(np.abs(values) < 0.1)),
                        "resolved_fraction": float(np.mean(resolved)),
                    })
    return records


def plot_histograms(rows: list[dict[str, Any]], output: Path) -> dict[str, str]:
    analysis._plot_style()
    figure, axes = plt.subplots(2, 4, figsize=(7.05, 4.25), sharex=True, sharey=True)
    bins = np.linspace(-1.05, 1.05, 43)
    columns = (("soft", "ccw"), ("soft", "cw"), ("hard", "ccw"), ("hard", "cw"))
    for row_index, protocol in enumerate(PROTOCOLS):
        for column_index, (wall, direction) in enumerate(columns):
            axis = axes[row_index, column_index]
            for width, color, linestyle in zip(WIDTHS, COLORS, LINESTYLES):
                values = [
                    row["endpoint_q_x"] for row in rows
                    if str(row["protocol"]) == protocol
                    and str(row["wall"]) == wall
                    and int(row["Nx"]) == width
                    and str(row["direction"]) == direction
                ]
                axis.hist(
                    values, bins=bins, histtype="step", linewidth=1.15,
                    color=color, linestyle=linestyle, label=rf"$N_x={width}$",
                )
            axis.axvline(0.0, color="0.35", linestyle=":", linewidth=0.7)
            axis.axvline(
                1.0 if direction == "ccw" else -1.0,
                color="0.55", linestyle="--", linewidth=0.75,
            )
            shell = r"$n_{\rm shell}=1$" if protocol == "nsh1" else r"dense $n_{\rm shell}$"
            axis.set_title(f"{shell}; {wall}, {direction.upper()}")
            axis.set_xlim(-1.05, 1.05)
            axis.set_ylim(0, 105)
            if row_index == 1:
                axis.set_xlabel(r"raw endpoint $q_x$")
            if column_index == 0:
                axis.set_ylabel("samples")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    for label, axis in zip("abcdefgh", axes.ravel()):
        axis.text(-0.19, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
    figure.tight_layout()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pdf = figures / "reconciled_samplewise_wall_response_histograms.pdf"
    png = figures / "reconciled_samplewise_wall_response_histograms.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"pdf": str(pdf), "png": str(png)}


def plot_mean_paths(rows: list[dict[str, Any]], output: Path) -> dict[str, dict[str, str]]:
    analysis._plot_style()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    products: dict[str, dict[str, str]] = {}
    for protocol in PROTOCOLS:
        figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.5), sharex=True, sharey=True)
        for row_index, wall in enumerate(WALLS):
            for column_index, (direction, sigma) in enumerate(DIRECTIONS):
                axis = axes[row_index, column_index]
                for width, color, linestyle in zip(WIDTHS, COLORS, LINESTYLES):
                    selected = [
                        row for row in rows
                        if str(row["protocol"]) == protocol
                        and str(row["wall"]) == wall
                        and int(row["Nx"]) == width
                        and str(row["direction"]) == direction
                    ]
                    if len(selected) != 100:
                        raise RuntimeError(
                            f"expected S100 for {(protocol, wall, width, direction)}"
                        )
                    paths = np.stack([np.asarray(row["q_x"], dtype=float) for row in selected])
                    mean = np.mean(paths, axis=0)
                    sd = np.std(paths, axis=0, ddof=1)
                    phi = np.asarray(selected[0]["phi"], dtype=float)
                    flux = np.abs(phi - float(phi[0]))
                    axis.plot(
                        flux, mean, color=color, linestyle=linestyle, linewidth=1.25,
                        label=rf"$N_x={width}$",
                    )
                    axis.fill_between(
                        flux, mean - sd, mean + sd, color=color, alpha=0.12,
                        linewidth=0,
                    )
                axis.axhline(0.0, color="0.35", linestyle=":", linewidth=0.7)
                axis.axhline(float(sigma), color="0.55", linestyle="--", linewidth=0.75)
                axis.set_title(f"{wall} wall, {direction.upper()}")
                axis.set_xlim(0.0, 2.0 * np.pi)
                axis.set_ylim(-1.25, 1.25)
                axis.set_xticks((0.0, np.pi, 2.0 * np.pi), (r"$0$", r"$\pi$", r"$2\pi$"))
                if row_index == 1:
                    axis.set_xlabel(r"threaded flux $|\phi-\phi_0|$")
                if column_index == 0:
                    axis.set_ylabel(r"raw $q_x$")
        axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
        for label, axis in zip("abcd", axes.ravel()):
            axis.text(-0.16, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
        shell = r"$n_{\rm shell}=1$" if protocol == "nsh1" else r"dense $n_{\rm shell}$"
        figure.suptitle(f"{shell}: monitored-endpoint wall spectral flow", y=1.01, fontsize=9)
        figure.tight_layout()
        stem = f"reconciled_mean_qx_vs_flux_{protocol}"
        pdf, png = figures / f"{stem}.pdf", figures / f"{stem}.png"
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(figure)
        products[protocol] = {"pdf": str(pdf), "png": str(png)}
    return products


def plot_mean_right_charge(rows: list[dict[str, Any]], output: Path) -> dict[str, dict[str, str]]:
    """Plot the raw right-subsystem transfer with trajectory-mean SEM bands."""
    analysis._plot_style()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    products: dict[str, dict[str, str]] = {}
    for protocol in PROTOCOLS:
        figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.5), sharex=True, sharey=True)
        for row_index, wall in enumerate(WALLS):
            for column_index, (direction, sigma) in enumerate(DIRECTIONS):
                axis = axes[row_index, column_index]
                for width, color, linestyle in zip(WIDTHS, COLORS, LINESTYLES):
                    selected = [
                        row for row in rows
                        if str(row["protocol"]) == protocol
                        and str(row["wall"]) == wall
                        and int(row["Nx"]) == width
                        and str(row["direction"]) == direction
                    ]
                    if len(selected) != 100:
                        raise RuntimeError(
                            f"expected S100 for {(protocol, wall, width, direction)}"
                        )
                    paths = np.stack([
                        np.asarray(row["delta_N_right"], dtype=float) for row in selected
                    ])
                    mean = np.mean(paths, axis=0)
                    sem = np.std(paths, axis=0, ddof=1) / np.sqrt(paths.shape[0])
                    phi = np.asarray(selected[0]["phi"], dtype=float)
                    flux = np.abs(phi - float(phi[0]))
                    axis.plot(
                        flux, mean, color=color, linestyle=linestyle, linewidth=1.25,
                        label=rf"$N_x={width}$",
                    )
                    axis.fill_between(
                        flux, mean - sem, mean + sem, color=color, alpha=0.18,
                        linewidth=0,
                    )
                axis.axhline(0.0, color="0.35", linestyle=":", linewidth=0.7)
                axis.axhline(float(sigma), color="0.55", linestyle="--", linewidth=0.75)
                axis.set_title(f"{wall} wall, {direction.upper()}")
                axis.set_xlim(0.0, 2.0 * np.pi)
                axis.set_ylim(-1.12, 1.12)
                axis.set_xticks((0.0, np.pi, 2.0 * np.pi), (r"$0$", r"$\pi$", r"$2\pi$"))
                if row_index == 1:
                    axis.set_xlabel(r"threaded flux $|\phi-\phi_0|$")
                if column_index == 0:
                    axis.set_ylabel(r"mean right-wall transfer $\langle\Delta N_R\rangle$")
        axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
        for label, axis in zip("abcd", axes.ravel()):
            axis.text(-0.18, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
        shell = r"$n_{\rm shell}=1$" if protocol == "nsh1" else r"dense $n_{\rm shell}$"
        figure.suptitle(rf"{shell}: chiral right-wall charge transfer (mean $\pm$ SEM)", y=1.01, fontsize=9)
        figure.tight_layout()
        stem = f"reconciled_mean_delta_N_right_vs_flux_{protocol}_sem"
        pdf, png = figures / f"{stem}.pdf", figures / f"{stem}.png"
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(figure)
        products[protocol] = {"pdf": str(pdf), "png": str(png)}
    return products


def plot_nx32_hard_nsh1_right_charge(rows: list[dict[str, Any]], output: Path) -> dict[str, str]:
    """Make the hard-wall Nx=32, nshell=1 chirality figure with SEM bands."""
    analysis._plot_style()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    styles = {
        "ccw": {
            "label": "CCW", "color": "#0072B2",
            "linestyle": "-.", "marker": "o",
        },
        "cw": {
            "label": "CW", "color": "#D55E00",
            "linestyle": ":", "marker": "^",
        },
    }
    figure, axis = plt.subplots(figsize=(3.375, 2.65))
    fit_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for direction, _ in DIRECTIONS:
        selected = [
            row for row in rows
            if str(row["protocol"]) == "nsh1"
            and str(row["wall"]) == "hard"
            and int(row["Nx"]) == 32
            and str(row["direction"]) == direction
        ]
        if len(selected) != 100:
            raise RuntimeError(f"expected S100 for {('nsh1', 'hard', 32, direction)}")
        paths = np.stack([
            np.asarray(row["delta_N_right"], dtype=float) for row in selected
        ])
        mean = np.mean(paths, axis=0)
        sem = np.std(paths, axis=0, ddof=1) / np.sqrt(paths.shape[0])
        phi = np.asarray(selected[0]["phi"], dtype=float)
        signed_flux = phi - float(phi[0])
        normalized_flux = signed_flux / (2.0 * np.pi)
        slope = float(np.dot(normalized_flux, mean) / np.dot(normalized_flux, normalized_flux))
        fitted = slope * normalized_flux
        residual_sum = float(np.sum((mean - fitted) ** 2))
        origin_total_sum = float(np.sum(mean ** 2))
        r_squared = 1.0 - residual_sum / origin_total_sum
        fit_segments.append((signed_flux, fitted))
        style = styles[direction]
        axis.errorbar(
            signed_flux, mean, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markevery=32, markersize=3.2,
            linewidth=1.35,
            label=rf'{style["label"]} ($m={slope:.4f}$, $R_0^2={r_squared:.6f}$)',
            yerr=sem, errorevery=32,
            elinewidth=0.85, capsize=2.2, capthick=0.85,
        )
    for segment_index, (fit_flux, fit_values) in enumerate(fit_segments):
        axis.plot(
            fit_flux, fit_values, color="0.45", linestyle="--", linewidth=0.9,
            label=(r"zero-intercept linear fits"
                   if segment_index == 0 else None),
            zorder=0,
        )
    axis.axhline(0.0, color="0.35", linestyle=":", linewidth=0.7)
    axis.axvline(0.0, color="0.35", linestyle=":", linewidth=0.7)
    axis.set_xlim(-2.0 * np.pi, 2.0 * np.pi)
    axis.set_ylim(-1.0, 1.0)
    axis.set_xticks(
        (-2.0 * np.pi, -np.pi, 0.0, np.pi, 2.0 * np.pi),
        (r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"),
    )
    axis.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    axis.set_xlabel(r"threaded flux $\phi$")
    axis.set_ylabel(r"$\langle\Delta N_R\rangle$")
    axis.legend(frameon=False, fontsize=5.9, loc="lower right")
    figure.tight_layout()
    stem = "reconciled_Nx32_hard_nsh1_mean_delta_N_right_vs_signed_flux_sem"
    pdf, png = figures / f"{stem}.pdf", figures / f"{stem}.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"pdf": str(pdf), "png": str(png)}


def main() -> int:
    rows, inventory = load_reconciled()
    summary_rows = statistics(rows)
    analysis_root = OUTPUT_ROOT / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_root / "raw_directional_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    figures = {
        "endpoint_histograms": plot_histograms(rows, analysis_root),
        "mean_qx_vs_flux": plot_mean_paths(rows, analysis_root),
        "mean_delta_N_right_vs_flux_sem": plot_mean_right_charge(rows, analysis_root),
        "Nx32_hard_nsh1_mean_delta_N_right_vs_signed_flux_sem": (
            plot_nx32_hard_nsh1_right_charge(rows, analysis_root)
        ),
    }
    summary = {
        "schema": "wall_diabatic_spectral_pump_reconciled_analysis_v1",
        "inventory": inventory,
        "statistics": summary_rows,
        "figures": figures,
        "statistics_csv": str(csv_path),
        "source_hashes": {
            "analysis": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "v1_identity": hashlib.sha256((V1_ROOT / "campaign_identity.json").read_bytes()).hexdigest(),
            "v2_identity": hashlib.sha256((V2_ROOT / "campaign_identity.json").read_bytes()).hexdigest(),
        },
    }
    summary_path = analysis_root / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
