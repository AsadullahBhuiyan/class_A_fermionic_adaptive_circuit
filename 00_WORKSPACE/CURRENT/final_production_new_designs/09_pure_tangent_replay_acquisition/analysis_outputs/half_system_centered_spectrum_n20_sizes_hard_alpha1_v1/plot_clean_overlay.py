"""Redraw the saved mixed-mode histograms without titles or inset captions."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    source = root / "mixed_mode_density_overlay.csv"
    table = np.genfromtxt(source, delimiter=",", names=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif"],
                         "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.linewidth": .8, "pdf.fonttype": 42, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(3.375, 2.85))
    styles = [("#c62828", "^", ":"), ("#2e7d32", "s", "--"), ("#1565c0", "o", "-")]
    integrals = {}
    for ny, (color, marker, linestyle) in zip((24, 28, 32), styles):
        rows = table[table["Ny"] == ny]
        edges = np.r_[rows["left_edge"], rows["right_edge"][-1]]
        density = rows["conditional_probability_density"]
        integral = float(density @ np.diff(edges))
        np.testing.assert_allclose(integral, 1, atol=1e-14)
        integrals[str(ny)] = integral
        values = np.where(rows["count"] > 0, density, np.nan)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.stairs(values, edges, color=color, linestyle=linestyle, linewidth=.85,
                  label=rf"$20\times{ny}$")
        ax.plot(centers[::8], values[::8], linestyle="none", marker=marker,
                markersize=2.6, markerfacecolor="none", color=color)
    ax.set(yscale="log", xlim=(-1, 1), xlabel=r"Centered eigenvalue $\lambda$",
           ylabel=r"Conditional density $\rho_{\mathrm{mixed}}(\lambda)$")
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, .97), ncol=3,
              columnspacing=.7, handlelength=1.6, handletextpad=.4)
    fig.tight_layout(pad=.5)
    paths = [root / f"mixed_mode_density_overlay_clean.{ext}" for ext in ("pdf", "png")]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    metadata = {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "change": "Presentation only: remove title and inset caption; preserve histogram values, axes and legend",
                "density_integrals": integrals, "figure_inches": [3.375, 2.85],
                "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (root / "mixed_mode_density_overlay_clean.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
