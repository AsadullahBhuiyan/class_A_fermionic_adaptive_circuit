#!/usr/bin/env python3
"""Explain exact and avoided wall crossings using saved equilibrium and circuit data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parent
EQUILIBRIUM_ROOT = (
    REPO_ROOT
    / "notebooks/flattened_hamiltonian_analysis/outputs"
    / "flattened_twist_entanglement_benchmark_v1/run_20260831_233947"
)
CIRCUIT_ROOT = (
    PROJECT_ROOT
    / "results/N20x24_state_projector_pump_s100_v1/analysis/crossing_diagnostics_v1"
)
OUTPUT_ROOT = PROJECT_ROOT / "results/equilibrium_crossing_explainer_v1"


def _near_zero_wall_branch(
    phi: np.ndarray,
    evals: np.ndarray,
    wall_weights: np.ndarray,
    wall: int,
    *,
    window: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the most localized near-zero wall branch across the cyclic seam."""
    x = phi / np.pi
    cyclic_x = np.where(x > 1.0, x - 2.0, x)
    keep_phi = np.abs(cyclic_x) <= window
    flat_e = evals.reshape(len(phi), -1)
    flat_w = wall_weights.reshape(len(phi), -1, 2)
    selected = np.empty(len(phi), dtype=float)
    for point in range(len(phi)):
        candidates = np.flatnonzero(flat_w[point, :, wall] > 0.90)
        if not len(candidates):
            raise RuntimeError(f"no strongly localized equilibrium wall-{wall} mode")
        chosen = candidates[np.argmin(np.abs(flat_e[point, candidates]))]
        selected[point] = flat_e[point, chosen]
    order = np.argsort(cyclic_x[keep_phi])
    return cyclic_x[keep_phi][order], selected[keep_phi][order]


def main() -> None:
    spectrum_path = EQUILIBRIUM_ROOT / "compact_spectra_weights_assignments.npz"
    crossing_path = CIRCUIT_ROOT / "ny24_samplewise_crossing_diagnostics.csv"
    if not spectrum_path.is_file() or not crossing_path.is_file():
        raise FileNotFoundError("required equilibrium or monitored-circuit evidence is missing")

    with np.load(spectrum_path, allow_pickle=False) as saved:
        prefix = "trunc_1_dir_p1"
        phi = np.asarray(saved[f"{prefix}__phi"], dtype=float)
        evals = np.asarray(saved[f"{prefix}__physical_evals"], dtype=float)
        weights = np.asarray(saved[f"{prefix}__physical_wall_weights"], dtype=float)
        instantaneous = np.asarray(
            saved[f"{prefix}__instantaneous__basin_charge"], dtype=float
        )
        continued = np.asarray(saved[f"{prefix}__continued__basin_charge"], dtype=float)

    x0, energy0 = _near_zero_wall_branch(phi, evals, weights, 0)
    x1, energy1 = _near_zero_wall_branch(phi, evals, weights, 1)
    q_instantaneous = 0.5 * (
        instantaneous[:, 1] - instantaneous[0, 1]
        - instantaneous[:, 0] + instantaneous[0, 0]
    )
    q_continued = 0.5 * (
        continued[:, 1] - continued[0, 1]
        - continued[:, 0] + continued[0, 0]
    )
    order = np.argsort(phi)

    circuit = pd.read_csv(crossing_path)
    if len(circuit) != 200 or set(circuit["wall"]) != {"soft", "hard"}:
        raise RuntimeError("unexpected monitored-circuit crossing table")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.65), constrained_layout=True)
    blue, orange, green, purple = "#2474B5", "#E07A2D", "#2A9D66", "#7A5195"

    ax = axes[0, 0]
    ax.plot(x0, energy0, color=blue, lw=2.0, label="left-wall mode")
    ax.plot(x1, energy1, color=orange, lw=2.0, label="right-wall mode")
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.axvline(0.0, color="0.35", lw=0.8, ls=":")
    ax.scatter([0.0], [0.0], s=58, facecolors="white", edgecolors="black", zorder=5)
    ax.annotate(
        "true crossing",
        xy=(0.0, 0.0),
        xytext=(0.035, 0.012),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=8,
    )
    ax.set(xlabel=r"cyclic flux coordinate $\theta/\pi$", ylabel="wall-mode energy")
    ax.set_title("(a) Exact equilibrium crossing", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    ax = axes[0, 1]
    theta = np.linspace(-0.15, 0.15, 501)
    slope = float(np.median(np.abs(np.diff(energy0) / np.diff(x0))))
    gap = 0.012
    diabatic_left = -slope * theta
    diabatic_right = slope * theta
    avoided = np.sqrt((slope * theta) ** 2 + (gap / 2.0) ** 2)
    ax.plot(theta, diabatic_left, color=blue, lw=1.5, ls="--", label="wall character")
    ax.plot(theta, diabatic_right, color=orange, lw=1.5, ls="--")
    ax.plot(theta, -avoided, color="0.15", lw=2.2, label="energy eigenstates")
    ax.plot(theta, avoided, color="0.15", lw=2.2)
    ax.annotate(
        "minimum gap",
        xy=(0.0, gap / 2.0),
        xytext=(0.045, 0.016),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=8,
    )
    ax.text(
        0.02,
        0.97,
        "two-level illustration\n(not equilibrium data)",
        transform=ax.transAxes,
        fontsize=7.3,
        va="top",
    )
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set(xlabel=r"local flux coordinate $(\phi-\phi_\star)/\pi$", ylabel="level")
    ax.set_title("(b) Avoided crossing", loc="left")
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")

    ax = axes[1, 0]
    ax.plot(phi[order] / np.pi, q_instantaneous[order], color="0.4", lw=1.8,
            label="re-fill lowest levels")
    ax.plot(phi[order] / np.pi, q_continued[order], color=green, lw=2.2,
            label="continue occupied modes")
    ax.axhline(1.0, color="0.25", lw=0.8, ls="--")
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.scatter(
        [phi[order][-1] / np.pi, phi[order][-1] / np.pi],
        [q_instantaneous[order][-1], q_continued[order][-1]],
        color=["0.4", green],
        s=28,
        zorder=4,
    )
    ax.set(xlabel=r"flux $\phi/\pi$", ylabel=r"$q_x=(\Delta Q_R-\Delta Q_L)/2$")
    ax.set_title("(c) Exact equilibrium pump", loc="left")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")

    ax = axes[1, 1]
    markers = {"soft": "o", "hard": "s"}
    colors = {"soft": purple, "hard": "#C94C3B"}
    for wall in ("soft", "hard"):
        rows = circuit[circuit["wall"] == wall]
        pumped = rows["pump_event"].astype(bool).to_numpy()
        gap_values = rows["minimum_instantaneous_rank_gap"].to_numpy()
        response = np.abs(rows["direction_odd_q_x"].to_numpy())
        ax.scatter(
            gap_values[~pumped], response[~pumped], s=18, marker=markers[wall],
            facecolors="none", edgecolors=colors[wall], alpha=0.65,
            label=f"{wall}: closing",
        )
        ax.scatter(
            gap_values[pumped], response[pumped], s=22, marker=markers[wall],
            color=colors[wall], alpha=0.72, label=f"{wall}: pumped",
        )
    ax.axhline(0.5, color="0.3", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set(
        xlabel="minimum occupied--empty rank gap",
        ylabel=r"$|q_x^{\rm odd}|$",
    )
    ax.set_title("(d) Monitored ensemble", loc="left")
    ax.text(0.98, 0.96, "100 samples / wall", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="center")

    for ax in axes.flat:
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.16)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_ROOT / "exact_vs_avoided_crossing_explainer.png"
    pdf_path = OUTPUT_ROOT / "exact_vs_avoided_crossing_explainer.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "schema": "equilibrium_crossing_explainer_v1",
        "equilibrium_input": str(spectrum_path.relative_to(REPO_ROOT)),
        "monitored_input": str(crossing_path.relative_to(REPO_ROOT)),
        "equilibrium_case": {
            "Nx": 20,
            "Ny": 40,
            "dw_truncation": True,
            "direction": 1,
            "instantaneous_endpoint_q_x": float(q_instantaneous[order][-1]),
            "continued_endpoint_q_x": float(q_continued[order][-1]),
            "illustrative_avoided_crossing_gap": gap,
            "illustrative_slope_from_equilibrium_branch": slope,
        },
        "monitored_case": {"Nx": 20, "Ny": 24, "samples_per_wall": 100},
        "note": (
            "Panel (b) is an explicitly labeled two-level illustration. "
            "Panels (a), (c), and (d) use saved numerical data."
        ),
    }
    (OUTPUT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
