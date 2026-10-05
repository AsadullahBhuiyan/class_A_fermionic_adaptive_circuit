"""Pool all saved Ny=40 half-system occupations, without endpoint exclusion."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    source = root / "analysis_outputs/half_system_spectra_ny40_50_60_roundoff_filtered/half_system_occupation_and_modular_spectra.npz"
    out = root / "analysis_outputs/half_system_occupation_density_Ny40"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        nu = data["occupations_Ny40"]
        np.testing.assert_array_equal(data["sample_indices_Ny40"], np.arange(100))
    assert nu.shape == (100, 800)
    assert np.isfinite(nu).all() and ((nu >= 0) & (nu <= 1)).all()
    edges = np.linspace(0, 1, 81)
    counts, _ = np.histogram(nu.ravel(), bins=edges)
    density = counts / (nu.size * np.diff(edges))
    assert counts.sum() == nu.size
    area = float(density @ np.diff(edges))
    np.testing.assert_allclose(area, 1, atol=1e-14)
    np.savetxt(out / "histogram.csv", np.column_stack((edges[:-1], edges[1:], counts, density)),
               delimiter=",", header="bin_left,bin_right,count,probability_density", comments="")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif"],
                         "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .8,
                         "pdf.fonttype": 42, "savefig.dpi": 300})
    for scale in ("log", "linear"):
        fig, ax = plt.subplots(figsize=(3.375, 2.7))
        ax.stairs(density, edges, fill=True, color="#2878b5", alpha=.25, linewidth=0)
        ax.stairs(density, edges, color="#1764a0", linewidth=1)
        ax.set(xlabel=r"occupation $\nu$", ylabel=r"probability density $p(\nu)$",
               xlim=(0, 1), yscale=scale)
        if scale == "log":
            ax.set_ylim(density[density > 0].min() * .6, density.max() * 2)
        else:
            ax.set_ylim(0, density.max() * 1.15)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.text(.5, .96, r"$N_x=20,\ N_y=40,\ A_y=20$", ha="center", va="top", transform=ax.transAxes, fontsize=8)
        fig.tight_layout(pad=.5)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"half_system_occupation_density_Ny40_{scale}.{ext}")
        plt.close(fig)
    metadata = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "Nx": 20, "Ny": 40, "samples": 100, "endpoint_cycle": 80,
        "subsystem": "[0,20) x [0,20), both orbitals", "total_levels": nu.size,
        "excluded_levels": 0, "new_clipping": False,
        "ensemble_order": "Pool individual sample eigenvalues; not spectrum of averaged covariance",
        "normalization": "count / (80000 * bin width)", "histogram_integral": area,
        "bins": 80, "endpoint_bin_fraction": float((counts[0] + counts[-1]) / nu.size),
        "figure_inches": [3.375, 2.7], "outputs_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
            if p.suffix in (".pdf", ".png", ".csv")
        },
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
