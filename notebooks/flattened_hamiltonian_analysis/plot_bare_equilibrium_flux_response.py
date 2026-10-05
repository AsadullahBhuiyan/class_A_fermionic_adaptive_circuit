#!/usr/bin/env python3
"""Replot the exact flattened-Hamiltonian charge pump in circuit-style variables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = (
    HERE
    / "outputs/exact_wall_charge_pump_refined_v1/run_20260901_002219"
)


def load_series(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data = data.loc[data["grid"] == "global_257"].copy()
    expected = {(truncation, direction) for truncation in (False, True) for direction in (-1, 1)}
    actual = set(zip(data["dw_truncation"], data["direction"], strict=True))
    if actual != expected:
        raise RuntimeError(f"unexpected equilibrium cases: {sorted(actual)!r}")
    for key, frame in data.groupby(["dw_truncation", "direction"]):
        if len(frame) != 257:
            raise RuntimeError(f"{key} has {len(frame)} points rather than 257")
        s = frame.sort_values("s")["s"].to_numpy(dtype=np.float64)
        if not np.allclose(s, np.linspace(0.0, 1.0, 257), atol=1e-14, rtol=0.0):
            raise RuntimeError(f"{key} does not span one ordered flux quantum")
    return data


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
        }
    )


def select(data: pd.DataFrame, truncation: bool, direction: int) -> pd.DataFrame:
    return data.loc[
        (data["dw_truncation"] == truncation) & (data["direction"] == direction)
    ].sort_values("s")


def common_axis(ax: plt.Axes, *, ylim: tuple[float, float]) -> None:
    ax.axhline(0.0, color="0.25", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"flux path $\phi$")
    ax.tick_params(which="both", direction="in", top=True, right=True)


def save_pair(fig: plt.Figure, output: Path, stem: str) -> tuple[Path, Path]:
    pdf = output / f"{stem}.pdf"
    png = output / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def plot_wall_response(
    data: pd.DataFrame, output: Path, *, family: str
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), sharex=True, sharey=True)
    direction_styles = {
        1: {
            "color": "#1f77b4",
            "linestyle": "-",
            "label": r"positive $2\pi$ path",
        },
        -1: {
            "color": "#d62728",
            "linestyle": "--",
            "label": r"negative $2\pi$ path",
        },
    }
    q_column = f"q_pump_{family}"
    for ax, truncation in zip(axes, (False, True), strict=True):
        for direction in (1, -1):
            frame = select(data, truncation, direction)
            phi = 2.0 * np.pi * frame["s"].to_numpy(dtype=np.float64)
            ax.plot(phi, frame[q_column], linewidth=1.4, **direction_styles[direction])
        common_axis(ax, ylim=(-1.08, 1.08))
        ax.set_title("Untruncated wall" if not truncation else "Wall-sector truncated")
    axes[0].set_ylabel(r"wall response $q_x$")
    axes[1].legend(frameon=False, loc="center right", handlelength=2.5)
    fig.tight_layout(pad=0.35)
    return save_pair(fig, output, f"bare_equilibrium_{family}_wall_response_phi")


def plot_regional_charge(
    data: pd.DataFrame, output: Path, *, family: str
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.85), sharex=True, sharey=True)
    region_styles = {
        "left": {"color": "#9467bd", "region": r"$\Delta Q_L$"},
        "right": {"color": "#ff7f0e", "region": r"$\Delta Q_R$"},
    }
    direction_styles = {
        1: {"linestyle": "-", "direction": r"positive $2\pi$"},
        -1: {"linestyle": "--", "direction": r"negative $2\pi$"},
    }
    for ax, truncation in zip(axes, (False, True), strict=True):
        for direction in (1, -1):
            frame = select(data, truncation, direction)
            phi = 2.0 * np.pi * frame["s"].to_numpy(dtype=np.float64)
            for region in ("left", "right"):
                regional = region_styles[region]
                sense = direction_styles[direction]
                column = f"delta_Q_{region}_basin_{family}"
                ax.plot(
                    phi,
                    frame[column],
                    color=regional["color"],
                    linestyle=sense["linestyle"],
                    linewidth=1.3,
                    label=f'{regional["region"]}, {sense["direction"]}',
                )
        common_axis(ax, ylim=(-1.08, 1.08))
        ax.set_title("Untruncated wall" if not truncation else "Wall-sector truncated")
    axes[0].set_ylabel(r"regional charge change")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        handlelength=2.2,
        fontsize=6.8,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0), pad=0.35)
    return save_pair(fig, output, f"bare_equilibrium_{family}_regional_charge_phi")


def validate_observables(data: pd.DataFrame) -> None:
    for family in ("instantaneous", "continued"):
        left = data[f"delta_Q_left_basin_{family}"].to_numpy(dtype=np.float64)
        right = data[f"delta_Q_right_basin_{family}"].to_numpy(dtype=np.float64)
        q_saved = data[f"q_pump_{family}"].to_numpy(dtype=np.float64)
        q_rebuilt = 0.5 * (right - left)
        if np.max(np.abs(q_saved - q_rebuilt)) > 1e-12:
            raise FloatingPointError(f"saved {family} q_pump does not match regional charge")
        if np.max(np.abs(left + right)) > 1e-12:
            raise FloatingPointError(f"{family} left/right charge is not conserved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output_dir or run_dir / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_series(run_dir / "pump_timeseries.csv")
    validate_observables(data)
    configure_style()

    paths: list[Path] = []
    for family in ("instantaneous", "continued"):
        paths.extend(plot_wall_response(data, output, family=family))
        paths.extend(plot_regional_charge(data, output, family=family))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
