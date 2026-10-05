"""Figure 16: occupation spectrum above the mean-state squared correlator."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SPECTRA = HERE / "analysis_outputs/raster_y_channel_alpha1_vs3_ky"
CORRELATORS = HERE / "analysis_outputs/raster_y_mean_channel_squared_correlator_v1"
OUT = HERE / "analysis_outputs/raster_y_mean_channel_summary_2x1"
STEM = "hard_wall_mean_channel_summary_2x1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spectrum_manifest = json.loads((SPECTRA / "manifest.json").read_text())["sha256"]
    correlator_manifest = json.loads((CORRELATORS / "manifest.json").read_text())["files_sha256"]
    for folder, manifest, names in (
        (SPECTRA, spectrum_manifest, ("spectra.npz", "summary.json")),
        (CORRELATORS, correlator_manifest, ("curves.npz", "summary.json")),
    ):
        for name in names:
            assert sha(folder / name) == manifest[name]
    spectral_summary = json.loads((SPECTRA / "summary.json").read_text())
    correlator_summary = json.loads((CORRELATORS / "summary.json").read_text())
    for alpha in (1, 3):
        spectrum = spectral_summary[f"alpha{alpha}_hard"]
        correlator = correlator_summary["cases"][str(alpha)]
        assert spectrum["source_sha256"] == correlator["input_sha256"]
        assert spectrum["config"] == correlator["config"]
    plt.rcParams.update({"font.family": "CMU Sans Serif", "font.size": 8,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.375, 4.7),
                                    gridspec_kw={"height_ratios": [1.2, 1]})
    with np.load(SPECTRA / "spectra.npz", allow_pickle=False) as spectra, \
            np.load(CORRELATORS / "curves.npz", allow_pickle=False) as curves:
        r = np.arange(1, 33)
        xx = np.log((64/np.pi)*np.sin(np.pi*r/64))
        for alpha, color, marker, size, ls in (
            (1, "#2468ad", "o", 10, "-"), (3, "#c0392b", "^", 8, ":")
        ):
            occupations = spectra[f"alpha{alpha}_hard"]
            top.scatter(np.repeat(spectra["ky"]/np.pi, 40), occupations.ravel(),
                        s=size, marker=marker, facecolors="none", edgecolors=color,
                        linewidths=.55, label=rf"$\alpha_1={alpha}$",
                        zorder=3 if alpha == 3 else 2)
            values = curves[f"alpha{alpha}_xavg"][1:]
            displayed = values > correlator_summary["display_cutoff"]
            yy = np.full(values.shape, np.nan)
            yy[displayed] = np.log(values[displayed])
            bottom.plot(xx, yy, color=color, marker=marker, ls=ls, ms=3.5,
                        mfc="white", mew=.65, lw=.75, label=rf"$\alpha_1={alpha}$")
    top.axhline(.5, color=".5", ls="--", lw=.7, zorder=0)
    top.set(xlabel=r"$k_y/\pi$", ylabel=r"Occupation $\nu_a(k_y)$",
            xlim=(-1.03, 1.03), ylim=(-.035, 1.035))
    top.set_xticks([-1, -.5, 0, .5, 1])
    top.set_yticks([0, .25, .5, .75, 1])
    top.legend(loc="center left", frameon=False, handletextpad=.3)
    bottom.set(xlabel=r"$\log d_{64}(r_y)$",
               ylabel=r"$\log C_{\overline{G}}^{\mathrm{av}}(r_y)$",
               xlim=(-.06, 3.08), ylim=(np.log(1e-8)-.3, -2.5))
    bottom.set_xticks([0, 1, 2, 3])
    bottom.set_yticks([-15, -10, -5])
    bottom.legend(loc="lower right", frameon=False, handletextpad=.3)
    for ax, letter in zip((top, bottom), "ab"):
        ax.tick_params(top=True, right=True)
        ax.text(-.19, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9)
    fig.subplots_adjust(left=.20, right=.975, top=.965, bottom=.10, hspace=.42)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=300)
    plt.close(fig)
    inputs = [SPECTRA / "spectra.npz", SPECTRA / "summary.json",
              CORRELATORS / "curves.npz", CORRELATORS / "summary.json", Path(__file__)]
    (OUT / "manifest.json").write_text(json.dumps({
        "panels": ["hard-wall raster-y occupation spectrum", "full-x averaged squared mean correlator"],
        "fit": None, "display_cutoff_panel_b": 1e-8,
        "input_sha256": {str(p): sha(p) for p in inputs},
        "output_sha256": {f"{STEM}.{ext}": sha(OUT / f"{STEM}.{ext}") for ext in ("pdf", "png")},
    }, indent=2) + "\n")
    print(OUT / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
