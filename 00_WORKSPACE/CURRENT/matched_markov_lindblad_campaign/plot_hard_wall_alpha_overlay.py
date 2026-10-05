"""Single-column hard-wall channel spectrum from preserved endpoint data."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "analysis_outputs/alpha1_vs3_hard_soft_full_system_ky"
STEM = "markov_channel_hard_wall_alpha_overlay"


def main():
    source = OUT / "spectra.npz"
    plt.rcParams.update({"font.family": "CMU Sans Serif", "font.size": 8,
                         "xtick.direction": "in", "ytick.direction": "in"})
    with np.load(source, allow_pickle=False) as data:
        ky = data["ky"]
        assert ky.shape == (64,) and np.all(np.isfinite(ky))
        fig, ax = plt.subplots(figsize=(3.375, 2.95))
        for alpha, color, marker, size in [(1, "#2468ad", "o", 10),
                                            (3, "#c0392b", "^", 8)]:
            occ = data[f"markov_channel_alpha{alpha}_hard_occupations"]
            assert occ.shape == (64, 40) and np.all(np.isfinite(occ))
            assert occ.min() >= -1e-10 and occ.max() <= 1 + 1e-10
            ax.scatter(np.repeat(ky / np.pi, 40), occ.ravel(), s=size,
                       marker=marker, facecolors="none", edgecolors=color,
                       linewidths=.55, label=rf"$\alpha_1={alpha}$",
                       zorder=3 if alpha == 3 else 2)
        ax.axhline(.5, color=".5", ls="--", lw=.7, zorder=0)
        ax.set(xlabel=r"$k_y/\pi$", ylabel=r"Occupation $\nu_a(k_y)$",
               xlim=(-1.03, 1.03), ylim=(-.035, 1.035))
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.tick_params(top=True, right=True)
        ax.legend(loc="center left", frameon=False, markerscale=1.3,
                  handletextpad=.3)
        fig.tight_layout(pad=.6)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{STEM}.{ext}", dpi=300)
        plt.close(fig)
    caption = (
        "Hard-wall-only quantum-channel endpoint occupation spectrum. Blue open circles: "
        "alpha_1=1; red open triangles: alpha_1=3; alpha_2=30. Nx=20, Ny=64, "
        "nshell=1, walls x=5,15 inclusive, support truncation enabled, all slabs evolve. "
        "Maximally mixed initialization, perfect correction, complex128, endpoint cycle 128. "
        "Measurement outcomes are averaged analytically for one random schedule per case "
        "(shared seed 2257147520926244099); this is not an S-trajectory Monte Carlo estimate. "
        "The endpoint covariance is spatially y-twirled before its momentum blocks are "
        "diagonalized. All 40 modes per momentum, including exterior modes, are retained. "
        "No temporal averaging, fitting, interpolation, or uncertainty bars. "
        "Dashed line: occupation 1/2. Reuses the preserved spectra without recalculation.\n"
    )
    (OUT / f"{STEM}_caption.txt").write_text(caption)
    files = [source, OUT / "summary.json", Path(__file__),
             OUT / f"{STEM}.png", OUT / f"{STEM}.pdf"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT / f"{STEM}_manifest.json").write_text(
        json.dumps({"sha256": hashes, "family": "markov_channel", "wall": "hard",
                    "alpha_1": [1, 3], "width_inches": 3.375}, indent=2) + "\n")
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
