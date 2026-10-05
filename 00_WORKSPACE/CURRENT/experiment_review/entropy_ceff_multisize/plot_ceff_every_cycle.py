#!/usr/bin/env python3
"""Show every saved physical cycle, without markers or smoothing."""

from __future__ import annotations

import json

import numpy as np

from plot_ceff_absolute_error_vs_normalized_cycle import (
    HERE, NY_VALUES, ROOT, SOURCE_CSV, STYLES,
    configure_matplotlib, load_rows, plt, sha256,
)

PDF_PATH = HERE / "figures/ceff_every_cycle_Nx20_Ny30_40_50_S100.pdf"
PNG_PATH = PDF_PATH.with_suffix(".png")
MANIFEST_PATH = HERE / "ceff_every_cycle_manifest.json"


def make_figure(rows):
    configure_matplotlib()
    plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.85))
    maximum = 1.0
    for ny in NY_VALUES:
        selected = [row for row in rows if row["Ny"] == ny]
        cycles = np.asarray([row["cycle"] for row in selected])
        ceff = np.asarray([row["c_eff"] for row in selected])
        error = np.asarray([row["c_eff_fit_error"] for row in selected])
        if not np.array_equal(cycles, np.arange(1, 2 * ny + 1)):
            raise ValueError(f"Ny={ny}: missing or duplicate physical cycles")
        if not np.all(np.isfinite(ceff)) or not np.all(np.isfinite(error)) or np.any(error < 0):
            raise ValueError(f"Ny={ny}: invalid central charge or fit uncertainty")
        if any(row["Nx"] != 20 or row["samples"] != 100 for row in selected):
            raise ValueError(f"Ny={ny}: unexpected ensemble identity")
        maximum = max(maximum, float(np.max(ceff + error)))
        style = STYLES[ny]
        for ax in axes:
            ax.plot(cycles, ceff, color=style["color"], linestyle=style["linestyle"],
                    marker=None, linewidth=1.25, label=rf"$N_y={ny}$")
            ax.fill_between(cycles, ceff - error, ceff + error,
                            color=style["color"], alpha=0.16, linewidth=0)
    for index, ax in enumerate(axes):
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.8)
        ax.set_xlabel("physical cycle")
        ax.set_ylabel(r"$c_{\mathrm{eff}}$")
        ax.tick_params(which="both", top=True, right=True)
        ax.text(-0.17, 1.04, f"({chr(97 + index)})", transform=ax.transAxes)
    axes[0].set(xlim=(1, 100), ylim=(0, maximum * 1.04), title="All saved cycles")
    axes[0].legend(loc="upper right")
    axes[1].set(xlim=(10, 100), ylim=(0.98, 2.45), title="Convergence detail")
    fig.suptitle(r"$N_x=20$, $S=100$ trajectories per size", fontsize=9, y=0.99)
    fig.text(0.5, 0.02, "Shading: ±1 regression-fit SE (not trajectory SEM)",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=2.0)
    return fig


def main():
    rows = load_rows()
    fig = make_figure(rows)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)
    manifest = {
        "schema": "mean_entropy_ceff_every_cycle_figure_v1",
        "quantity": "c_eff = 3 * slope of trajectory-mean strip entropy",
        "source": {"path": str(SOURCE_CSV.relative_to(ROOT)), "sha256": sha256(SOURCE_CSV)},
        "scientific_contract": {
            "Nx": 20, "Ny": list(NY_VALUES), "independent_trajectories_per_size": 100,
            "initialization": "random pure", "perfect_correction": True,
            "wall": "hard", "nshell": 1, "alpha_1": 1.0, "alpha_2": 30.0,
            "sequence": "raster_y", "dtype": "complex128",
        },
        "estimator_order": "average entropy over trajectories, then OLS log-chord fit",
        "fit_window": "Ay=8..Ny/2 inclusive",
        "cycles": "every integer cycle 1..2Ny; no cycle-zero data",
        "markers": False, "smoothing": False, "subsampling": False,
        "axes": "physical cycle, linear c_eff; full history plus convergence zoom",
        "shading": "plus/minus 3 * OLS slope SE; not trajectory SEM or bootstrap CI",
        "outputs": {
            path.suffix[1:]: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (PDF_PATH, PNG_PATH)
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {PDF_PATH}\n[saved] {PNG_PATH}")
    print("[verified] Ny=30: 60 cycles; Ny=40: 80 cycles; Ny=50: 100 cycles; no markers")


if __name__ == "__main__":
    main()
