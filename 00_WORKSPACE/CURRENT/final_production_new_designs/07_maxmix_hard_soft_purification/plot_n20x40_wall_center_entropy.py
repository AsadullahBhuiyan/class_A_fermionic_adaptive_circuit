#!/usr/bin/env python3
"""Plot immutable hard-v2 Ny=40 entropy column sums with trajectory SEM."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import analyze_completed_campaign as campaign
from plot_hard_wall_purification_spatial import configure_plotting

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs" / "hard_n20x40_wall_center_entropy_v1"
NAME = "hard_n20x40_wall_center_entropy_loglog"
POSITIONS = (5, 15, 10)
LABELS = ("Left wall", "Right wall", "Slab center")


def main():
    spec = next(s for s in campaign.CAMPAIGNS if s.key == "purification_hard_v2")
    inventory = campaign._inventory_campaigns(campaign.DEFAULT_REMOTE_INVENTORY)
    records = [r for r in inventory[spec.key]["records"]
               if "Ny040" in r["relative_result_path"]]
    assert len(records) == 20
    parts, ids, sources = [], [], []
    max_closure = 0.0
    for remote in tqdm(records, desc="Verify and load Ny=40", unit="shard"):
        result = spec.data_root / remote["relative_result_path"]
        receipt = spec.data_root / remote["relative_completion_path"]
        for path, prefix in ((result, "result"), (receipt, "completion")):
            assert path.stat().st_size == remote[f"{prefix}_bytes"], path
            assert campaign.sha256_file(path) == remote[f"{prefix}_sha256"], path
        completion = json.loads(receipt.read_text())
        campaign._validate_completion(
            completion, spec, result,
            expected_result_bytes=remote["result_bytes"],
            expected_result_sha256=remote["result_sha256"],
        )
        with np.load(result, allow_pickle=False) as data:
            assert str(data["sampling_revision"].item()) == spec.revision
            assert str(data["configuration_hash"].item()) == spec.configuration_hash
            assert (int(data["Nx"]), int(data["Ny"])) == (20, 40)
            cycles = data["cycles"]
            assert np.array_equal(cycles, np.arange(161))
            sample_ids = data["sample_indices"]
            assert np.array_equal(sample_ids, completion["sample_indices"])
            contour = data["entropy_contour"]
            assert contour.shape == (5, 161, 20, 40)
            assert np.isfinite(contour).all() and contour.min() >= 0
            columns = contour.sum(axis=-1)
            closure = float(np.max(np.abs(columns.sum(axis=-1) - data["total_entropy"])))
            assert closure < 5e-10, closure
            max_closure = max(max_closure, closure)
            parts.append(columns[:, :, POSITIONS])
            ids.append(sample_ids)
        sources.append(remote)
    sample_ids = np.concatenate(ids)
    order = np.argsort(sample_ids)
    assert np.array_equal(sample_ids[order], np.arange(100))
    values = np.concatenate(parts)[order]
    assert values.shape == (100, 161, 3)
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(100)
    assert np.all(mean > 0) and np.isfinite(sem).all()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "column_entropy.npz", cycles=cycles,
                        sample_indices=sample_ids[order], x_positions=POSITIONS,
                        trajectory_column_entropy=values, mean=mean, sem=sem)
    with (OUT / "column_entropy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cycle", "x", "region", "mean_entropy_nats", "sem_entropy_nats"])
        for t in cycles:
            for j, x in enumerate(POSITIONS):
                writer.writerow([t, x, LABELS[j], mean[t, j], sem[t, j]])

    configure_plotting()
    plt.rcParams.update({"font.size": 8, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(3.375, 3.0), layout="constrained")
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    marker_cycles = np.unique(np.rint(np.geomspace(1, 160, 13)).astype(int))
    for j, (color, marker, style) in enumerate(zip(colors, ("^", "s", "o"), (":", "--", "-"))):
        lower = mean[1:, j] - sem[1:, j]
        # Mask nonpositive lower bounds on log axes; never introduce an entropy floor.
        lower = np.where(lower > 0, lower, np.nan)
        ax.fill_between(cycles[1:], lower, mean[1:, j] + sem[1:, j],
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(cycles[1:], mean[1:, j], color=color, ls=style, lw=1.25,
                marker=marker, markevery=marker_cycles - 1, ms=3.4,
                mfc="white", mew=0.85, label=rf"{LABELS[j]} ($x={POSITIONS[j]}$)")
    ax.set(xscale="log", yscale="log", xlim=(1, 170), xlabel="Cycle",
           ylabel=r"$\langle\sum_y s(x,y,t)\rangle_\xi$ [nats]")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="lower left", handlelength=2.6)
    ax.text(0.97, 0.96, r"$N_x=20,\ N_y=40$" + "\n" + r"$\alpha_1=1,\ \alpha_2=30$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    fig.savefig(OUT / f"{NAME}.pdf")
    fig.savefig(OUT / f"{NAME}.png", dpi=300)
    plt.close(fig)
    caption = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{hard_n20x40_wall_center_entropy_loglog.pdf}
\caption{Spatially resolved purification for a $20\times40$ hard-wall system,
$\alpha_1=1$ and $\alpha_2=30$. The entropy contour is summed along $y$ at
the left wall ($x=5$), right wall ($x=15$), and slab center ($x=10$),
then averaged over 100 independent trajectories; bands show ordinary
trajectory SEM. The active slab starts maximally mixed after Born-conditioned
exterior preparation, with slab-only measurements, perfect correction,
$n_{\rm shell}=1$, and raster-$y$ ordering. Raw cycles $1$--$160$ are shown
on logarithmic axes; cycle zero is omitted. No fit is imposed.}
\label{fig:hard-n20x40-wall-center-purification}
\end{figure}
"""
    (OUT / f"{NAME}.tex").write_text(caption)
    manifest = {
        "schema": "hard_n20x40_wall_center_entropy_v1",
        "sampling_revision": spec.revision, "configuration_hash": spec.configuration_hash,
        "source_hashes": spec.source_hashes, "Nx": 20, "Ny": 40,
        "samples": 100, "saved_cycles": [0, 160], "displayed_cycles": [1, 160],
        "meas_slab_only": True, "x_positions": dict(zip(LABELS, POSITIONS)),
        "estimator": "sum over y within each trajectory, then trajectory mean",
        "uncertainty": "std(ddof=1)/sqrt(100); no bootstrap",
        "normalization": "raw column entropy in nats, no division by Ny",
        "max_contour_scalar_closure_error": max_closure,
        "verified_shards": len(records), "inputs": sources,
        "remote_inventory_sha256": campaign.sha256_file(campaign.DEFAULT_REMOTE_INVENTORY),
        "plot_script_sha256": campaign.sha256_file(Path(__file__)),
        "outputs": {p.name: campaign.sha256_file(p) for p in OUT.iterdir()
                    if p.is_file() and p.name != "manifest.json"},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "closure_error": max_closure,
                      "cycle160_means": mean[-1].tolist(),
                      "cycle160_sems": sem[-1].tolist()}, indent=2))


if __name__ == "__main__":
    main()
