"""Verify preserved raster-y endpoints and plot their occupation spectra."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observables import ky_blocks_from_twirled

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = HERE / "results/raster_y_channel_endpoints_v1_20260915T025739Z"
OUT = HERE / "analysis_outputs/raster_y_channel_alpha1_vs3_ky"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spectra, diagnostics = {}, {}
    for wall in ("hard", "soft"):
        for alpha in (1, 3):
            key = f"alpha{alpha}_{wall}"
            receipt = ROOT / key / "completion.json"
            record = json.loads(receipt.read_text())
            cfg = record["config"]
            assert record["status"] == "complete"
            assert (cfg["Nx"], cfg["Ny"], cfg["cycles"]) == (20, 64, 128)
            assert cfg["site_schedule"] == "raster_y"
            assert cfg["alpha_1"] == alpha and cfg["wall"] == wall
            assert cfg["channel_order"] == ["Ap", "Am", "Bp", "Bm"]
            for relative, digest in record["sources"].items():
                assert sha(REPO / relative) == digest, relative
            source = receipt.parent / record["result_filename"]
            assert source.stat().st_size == record["result_bytes"]
            assert sha(source) == record["result_sha256"]
            with np.load(source, allow_pickle=False) as data:
                assert json.loads(str(data["config_json"])) == cfg
                expected = np.tile([x + 20*y for x in range(20)
                                    for y in range(64)], (128, 1))
                np.testing.assert_array_equal(data["schedule_words"], expected)
                ky = data["ky"]
                occ = data["twirled_ky_occupations"]
                block_ky, blocks = ky_blocks_from_twirled(data["G_final_twirl"][0], 20, 64)
                np.testing.assert_allclose(block_ky, ky, atol=1e-14)
                np.testing.assert_allclose(np.linalg.eigvalsh(blocks), occ, atol=1e-12)
                assert occ.shape == (64, 40) and np.isfinite(occ).all()
                assert occ.min() >= -1e-10 and occ.max() <= 1 + 1e-10
                distance = data["successive_state_distance"].ravel()
                assert np.max(distance[-10:]) < 1e-12
                if "ky" in spectra:
                    np.testing.assert_array_equal(spectra["ky"], ky)
                spectra["ky"] = ky
                spectra[key] = occ
                diagnostics[key] = {
                    "source": str(source), "source_sha256": record["result_sha256"],
                    "receipt_sha256": sha(receipt), "config": cfg,
                    "last_cycle_normalized_frobenius_change": float(distance[-1]),
                    "last_ten_cycles_max_change": float(np.max(distance[-10:])),
                    "ky_zero_central_occupations": occ[np.argmin(abs(ky)), 19:21].tolist(),
                }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "spectra.npz", **spectra)
    plt.rcParams.update({"font.family": "CMU Sans Serif", "font.size": 8,
                         "xtick.direction": "in", "ytick.direction": "in"})
    for wall in ("hard", "soft"):
        stem = f"markov_channel_{wall}_wall_alpha_overlay"
        fig, ax = plt.subplots(figsize=(3.375, 2.95))
        for alpha, color, marker, size in [(1, "#2468ad", "o", 10),
                                           (3, "#c0392b", "^", 8)]:
            ax.scatter(np.repeat(spectra["ky"] / np.pi, 40),
                       spectra[f"alpha{alpha}_{wall}"].ravel(), s=size,
                       marker=marker, facecolors="none", edgecolors=color,
                       linewidths=.55, label=rf"$\alpha_1={alpha}$",
                       zorder=3 if alpha == 3 else 2)
        ax.axhline(.5, color=".5", ls="--", lw=.7, zorder=0)
        ax.set(xlabel=r"$k_y/\pi$", ylabel=r"Occupation $\nu_a(k_y)$",
               xlim=(-1.03, 1.03), ylim=(-.035, 1.035))
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.tick_params(top=True, right=True)
        ax.legend(loc="center left", frameon=False, markerscale=1.3, handletextpad=.3)
        fig.tight_layout(pad=.6)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
        plt.close(fig)
        (OUT / f"{stem}_caption.txt").write_text(
            f"{wall.capitalize()}-wall quantum-channel occupation spectrum: blue open circles "
            "alpha_1=1; red open triangles alpha_1=3; alpha_2=30, Nx=20, Ny=64, nshell=1. "
            "Inclusive slab x=5,...,15, all slabs evolve, maximally mixed initialization, "
            "perfect correction, number dephasing, complex128. Fixed raster_y ordering "
            "(y fast), within-cell Ap,Am,Bp,Bm. Exact Born-averaged two-point evolution, "
            "not a Monte Carlo sample average. Cycle-128 endpoints are stationary at cycle "
            "boundaries to numerical precision (last ten normalized Frobenius increments "
            "below 1e-12). Spatial y-twirl precedes block diagonalization; all 40 modes per "
            "momentum including exterior modes are retained. No temporal or schedule average, "
            "fitting, interpolation, or sampling error bars. Dashed line: occupation 1/2.\n")
    (OUT / "summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    hashes = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.name != "manifest.json"}
    hashes[Path(__file__).name] = sha(Path(__file__))
    (OUT / "manifest.json").write_text(json.dumps({"sha256": hashes}, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
