"""Campaign 21 column entropy: full measurements, raw cycles, trajectory SEM."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from make_figure import DATA1, configure_style, load_data, mean_sem, sha

HERE = Path(__file__).resolve().parent
OUT = HERE / "wall_center_entropy"
STEM = "hard_n20x30_full_measurement_wall_center_entropy_loglog"
POSITIONS = (5, 15, 10)
LABELS = ("Left wall", "Right wall", "Slab center")


def main():
    data = load_data(DATA1, 1)
    # Bind the results and receipts to the original download as well.
    download = DATA1 / "DOWNLOAD_MANIFEST_ALPHA1.json"
    original = {r["path"]: r for r in json.loads(download.read_text())["files"]}
    for record in data["records"]:
        assert record["sha256"] == original[record["path"]]["sha256"]
        assert record["bytes"] == original[record["path"]]["bytes"]
    values = data["spatial"][:, :, POSITIONS]
    assert values.shape == (100, 61, len(POSITIONS))
    assert np.isfinite(values).all() and np.all(values >= 0)
    mean, sem = mean_sem(values)
    cycles = np.arange(61)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "column_entropy.npz", cycles=cycles,
                        x_positions=POSITIONS, sample_indices=np.arange(100),
                        trajectory_column_entropy=values, mean=mean, sem=sem)
    with (OUT / "column_entropy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cycle", "x", "region", "mean_entropy_nats", "sem_entropy_nats"])
        for t in cycles:
            for j, x in enumerate(POSITIONS):
                writer.writerow([t, x, LABELS[j], mean[t, j], sem[t, j]])

    configure_style()
    fig, ax = plt.subplots(figsize=(3.375, 3.0), layout="constrained")
    markers = np.unique(np.rint(np.geomspace(1, 60, 13)).astype(int)) - 1
    for j, (color, marker, style) in enumerate(zip(
            ("#d62728", "#2ca02c", "#1f77b4"),
            ("^", "s", "o"), (":", "--", "-"))):
        lower = mean[1:, j] - sem[1:, j]
        lower = np.where(lower > 0, lower, np.nan)
        ax.fill_between(cycles[1:], lower, mean[1:, j] + sem[1:, j],
                        color=color, alpha=.18, linewidth=0)
        ax.plot(cycles[1:], mean[1:, j], color=color, ls=style, lw=1.25,
                marker=marker, markevery=markers,
                ms=3.4, mfc="white", mew=.85,
                label=rf"{LABELS[j]} ($x={POSITIONS[j]}$)")
    ax.set(xscale="log", yscale="log", xlim=(1, 64), xlabel="Cycle",
           ylabel=r"$\langle\sum_y s(x,y,t)\rangle_\xi$ [nats]")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="lower left", handlelength=2.6)
    ax.text(.97, .96, r"$N_x=20,\ N_y=30$" + "\n" + r"$\alpha_1=1,\ \alpha_2=30$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    fig.savefig(OUT / f"{STEM}.pdf")
    fig.savefig(OUT / f"{STEM}.png", dpi=300)
    plt.close(fig)
    (OUT / f"{STEM}.tex").write_text(r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{hard_n20x30_full_measurement_wall_center_entropy_loglog.pdf}
\caption{Spatially resolved purification with measurements over the full
$20\times30$ system ($\texttt{meas\_slab\_only=False}$), for hard walls,
$\alpha_1=1$ and $\alpha_2=30$. The entropy contour is summed along $y$ at
the left wall ($x=5$), right wall ($x=15$), and slab center ($x=10$), then
averaged over 100 independent trajectories; bands show ordinary trajectory
SEM. Initialization is globally maximally mixed, with no exterior preparation;
the dynamics use perfect correction, $n_{\rm shell}=1$, and raster-$y$ order.
Raw cycles $1$--$60$ are shown on logarithmic axes; cycle zero is omitted.
No fit is imposed.}
\label{fig:full-measurement-wall-center-purification}
\end{figure}
""")
    manifest = dict(
        schema="full_measurement_wall_center_entropy_v1", campaign=21,
        sampling_revision="hard_wall_full_measurement_nx20_ny30_alpha1-3_s100_2ny_v1",
        Nx=20, Ny=30, alpha_1=1, alpha_2=30, meas_slab_only=False,
        initialization="globally maximally mixed; no exterior preparation",
        saved_cycles=[0, 60], displayed_cycles=[1, 60], samples=100,
        x_positions=dict(zip(LABELS, POSITIONS)),
        estimator="sum over y in each trajectory, then mean over trajectories",
        uncertainty="std(ddof=1)/sqrt(100); no bootstrap",
        normalization="nats; no division by Ny",
        entropy_kernel="stored acquisition uses occupation regularization at 1e-12; no new floor added",
        verified_shards=20, inputs=data["records"], executed_sources=data["sources"],
        download_manifest_sha256=sha(download), script_sha256=sha(Path(__file__)),
        cycle60_means=mean[-1].tolist(), cycle60_sems=sem[-1].tolist(),
        outputs={p.name: sha(p) for p in OUT.iterdir()
                 if p.is_file() and p.name != "manifest.json"})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "cycle60_means": mean[-1].tolist()}, indent=2))


if __name__ == "__main__":
    main()
