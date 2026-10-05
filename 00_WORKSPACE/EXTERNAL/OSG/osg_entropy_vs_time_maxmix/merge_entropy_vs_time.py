"""
Run locally after all OSG jobs finish.

Globs results/job_*/blocks/N20x{Ny}/block_*.npz, merges all blocks per Ny,
computes mean S(t) +/- SEM, saves CSVs and a figure.

Usage:
    python merge_entropy_vs_time.py [--results-dir ./results] [--output-dir ./merged]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NX       = 20
NY_VALUES = [40, 60, 80, 100, 120, 140, 160, 180, 200]


def merge_ny(Ny: int, results_dir: Path) -> dict | None:
    pattern = results_dir / f"job_*/blocks/N{NX}x{Ny}/block_*.npz"
    paths   = sorted(results_dir.glob(f"job_*/blocks/N{NX}x{Ny}/block_*.npz"))
    if not paths:
        print(f"  Ny={Ny}: no blocks found, skipping")
        return None

    sum_S = sumsq_S = count = cycles = None
    for path in paths:
        d = np.load(path, allow_pickle=False)
        if cycles is None:
            cycles  = d["cycles"].copy()
            sum_S   = d["sum_S"].copy()
            sumsq_S = d["sumsq_S"].copy()
            count   = d["count"].copy()
        else:
            if not np.array_equal(d["cycles"], cycles):
                raise ValueError(f"Cycle mismatch in {path}")
            sum_S   += d["sum_S"]
            sumsq_S += d["sumsq_S"]
            count   += d["count"]

    mean_S = sum_S / count
    var_S  = sumsq_S / count - mean_S ** 2
    sem_S  = np.sqrt(np.maximum(var_S, 0.0) / count)
    n_samples = int(count[0])

    print(f"  Ny={Ny:3d}: {len(paths)} blocks, {n_samples} samples")
    return {
        "Ny":       Ny,
        "cycles":   cycles,
        "mean_S":   mean_S,
        "sem_S":    sem_S,
        "n_samples": n_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--output-dir",  type=Path, default=Path("./merged"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Merging blocks...")
    results = [merge_ny(Ny, args.results_dir) for Ny in NY_VALUES]
    results = [r for r in results if r is not None]

    # Save CSVs
    for r in results:
        Ny = r["Ny"]
        df = pd.DataFrame({
            "cycle":  r["cycles"],
            "mean_S": r["mean_S"],
            "sem_S":  r["sem_S"],
        })
        csv_path = args.output_dir / f"entropy_vs_time_N{NX}x{Ny}.csv"
        df.to_csv(csv_path, index=False)

    # Plot S(t) vs t/Ny for all system sizes
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        Ny    = r["Ny"]
        t_norm = r["cycles"] / Ny
        color  = cmap(i / max(len(results) - 1, 1))
        ax.plot(t_norm, r["mean_S"], color=color, label=f"Ny={Ny}")
        ax.fill_between(t_norm,
                        r["mean_S"] - r["sem_S"],
                        r["mean_S"] + r["sem_S"],
                        color=color, alpha=0.2)

    ax.set_xlabel("t / Ny")
    ax.set_ylabel("S(t)")
    ax.set_title(f"Total entropy vs time, maxmix init, Nx={NX}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = args.output_dir / "entropy_vs_time.pdf"
    fig.savefig(fig_path)
    print(f"Saved {fig_path}")

    # Also plot vs raw cycle on log scale
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        Ny    = r["Ny"]
        color = cmap(i / max(len(results) - 1, 1))
        ax2.plot(r["cycles"], r["mean_S"], color=color, label=f"Ny={Ny}")
        ax2.fill_between(r["cycles"],
                         r["mean_S"] - r["sem_S"],
                         r["mean_S"] + r["sem_S"],
                         color=color, alpha=0.2)

    ax2.set_xlabel("cycle t")
    ax2.set_ylabel("S(t)")
    ax2.set_title(f"Total entropy vs cycle, maxmix init, Nx={NX}")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2_path = args.output_dir / "entropy_vs_cycle.pdf"
    fig2.savefig(fig2_path)
    print(f"Saved {fig2_path}")


if __name__ == "__main__":
    main()
