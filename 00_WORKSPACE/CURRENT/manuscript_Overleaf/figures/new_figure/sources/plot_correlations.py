#!/usr/bin/env python3
"""Portable manuscript-axis remake using only bundled, preserved correlators.

Run from any directory. No simulation, repository imports, or source-data writes.
Dependencies: numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography

BUNDLE = Path(__file__).resolve().parent.parent
DATA = BUNDLE / "data/correlations"
STEM = "Figure_05_correlations"
SIZES = (24, 28, 32, 40, 50, 60)
SITES = (5, 10, 15)
EXPECTED_BETA = 2.1903330477862717


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified():
    manifest = json.loads((DATA / "import_manifest.json").read_text())
    for name, record in manifest["inputs"].items():
        path = DATA / name
        assert path.stat().st_size == record["bytes"], name
        assert sha256(path) == record["sha256"], name
    summary = json.loads((DATA / "original_summary.json").read_text())
    with (DATA / "original_plotted_data.csv").open(newline="") as stream:
        original = list(csv.DictReader(stream))
    with np.load(DATA / "compact_curves.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    means = {}
    for ny in SIZES:
        values = arrays[f"alpha1_Ny{ny}_xresolved"]
        assert values.shape == (100, 20, ny // 2 + 1)
        assert np.isfinite(values).all() and (values >= 0).all()
        means[ny] = values.mean(axis=1).mean(axis=0)
    assert arrays["alpha3_Ny60_xresolved"].shape == (100, 20, 31)
    alpha3 = arrays["alpha3_Ny60_xresolved"].mean(axis=1).mean(axis=0)
    columns = arrays["alpha1_Ny60_xresolved"].mean(axis=0)
    max_abs_difference = 0.0
    for row in original:
        ny, r = int(row["Ny"]), int(row["ry"])
        if row["panel"] == "a":
            curve = alpha3 if "3" in row["series"] else means[60]
        elif row["panel"] == "b":
            curve = columns[int(row["series"].split("=")[1].rstrip("$"))]
        elif row["panel"] == "c":
            curve = means[ny]
        else:
            continue
        archived = float(row["unmasked_correlator"])
        np.testing.assert_allclose(curve[r], archived, rtol=2e-14, atol=0)
        max_abs_difference = max(max_abs_difference, abs(float(curve[r]) - archived))
        chord = (ny / np.pi) * np.sin(np.pi * r / ny)
        xx = np.log(chord)
        yy = np.log(curve[r])
        if row["panel"] == "c":
            xx = np.log(chord / (ny / np.pi))
            yy = np.log(curve[r] / curve[-1])
            assert (row["in_fit"] == "True") == (r >= 8)
        else:
            assert (row["displayed"] == "True") == (curve[r] > 1e-8)
        np.testing.assert_allclose([float(row["x"]), float(row["y"])], [xx, yy], rtol=2e-14, atol=1e-14)
    # Independently recover the pinned through-origin fit with equal weight per Ny.
    numerator = denominator = 0.0
    for ny in SIZES:
        r = np.arange(8, ny // 2 + 1)
        xx = np.log(np.sin(np.pi * r / ny))
        yy = np.log(means[ny][r] / means[ny][-1])
        numerator += np.mean(xx * yy)
        denominator += np.mean(xx * xx)
    recovered_beta = -numerator / denominator
    assert summary["primary_fit"]["beta"] == EXPECTED_BETA
    np.testing.assert_allclose(recovered_beta, EXPECTED_BETA, rtol=0, atol=2e-14)
    rows = [row for row in original if row["panel"] in ("a", "c") or
            (row["panel"] == "b" and row["series"] in {rf"$x={site}$" for site in SITES})]
    # Plot the exact archived coordinate strings, retaining every A/C point.
    for panel in ("a", "c"):
        assert [r for r in rows if r["panel"] == panel] == [r for r in original if r["panel"] == panel]
    actual_sites = sorted({int(r["series"].split("=")[1].rstrip("$")) for r in rows if r["panel"] == "b"})
    assert actual_sites == list(SITES)
    return rows, summary, {
        "input_hashes_verified": True,
        "panel_a_and_c_rows_exactly_preserved": True,
        "panel_b_columns": actual_sites,
        "removed_panel_b_columns": [6, 14],
        "max_abs_difference_compact_vs_archived_correlator": max_abs_difference,
        "primary_beta_archived": EXPECTED_BETA,
        "primary_beta_independently_recovered": float(recovered_beta),
        "archived_generation_sources": summary["sources"],
        "raw_input_provenance": "data/correlations/original_summary.json:inputs",
    }


def series(rows, panel, label):
    selected = [r for r in rows if r["panel"] == panel and r["series"] == label]
    assert selected
    return (np.array([float(r["x"]) for r in selected]),
            np.array([float(r["y"]) if r["displayed"] == "True" else np.nan for r in selected]))


def render(output):
    rows, summary, checks = load_verified()
    manuscript_style({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'savefig.bbox': None})
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.0))
    for label, color, marker, ls in ((r"$\alpha_1=1$", "#0072B2", "o", "-"),
                                    (r"$\alpha_1=3$", "#D55E00", "^", ":")):
        axes[0].plot(*series(rows, "a", label), color=color, marker=marker, ls=ls,
                     ms=3, mfc="white", mew=.65, lw=.75, label=label)
    axes[0].set_ylabel(r"$\log\overline{C_G^{\mathrm{av}}(r_y)}$")
    axes[0].legend(loc="lower left", frameon=False, handletextpad=.4)
    for site, color, marker, ls in ((5, "#0072B2", "o", "-"),
                                    (10, "#555555", "D", ":"),
                                    (15, "#D55E00", "v", ":")):
        label = rf"$x={site}$"
        axes[1].plot(*series(rows, "b", label), color=color, marker=marker, ls=ls,
                     ms=3, mfc="white", mew=.65, lw=.75, label=label)
    axes[1].set_ylabel(r"$\log\overline{C_G(x,r_y)}$")
    axes[1].text(.97, .95, r"DWs at $x=5,15$", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8)
    axes[1].legend(loc="lower left", frameon=False, ncol=2, columnspacing=.7,
                   handlelength=1.3, handletextpad=.3, labelspacing=.25)
    for ax in axes[:2]:
        ax.set_xlim(-.06, 3.03); ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim(np.log(1e-8) - .3, -2.5)
        ax.set_xlabel(r"$\log D(r_y)$", labelpad=2)
    for span in (summary["fit_shading"]["union"], summary["fit_shading"]["intersection"]):
        axes[2].axvspan(*span, color="0.5", alpha=.12, lw=0, zorder=0)
    for ny, color, marker in zip(SIZES,
            ("#D55E00", "#009E73", "#0072B2", "#CC79A7", "#E69F00", "#333333"),
            ("^", "s", "o", "v", "D", ">")):
        axes[2].plot(*series(rows, "c", str(ny)), marker=marker, ls="none", color=color,
                     ms=3, mfc="white", mew=.65, label=rf"${ny}$")
    grid = np.linspace(-3.02, 0, 200)
    axes[2].plot(grid, summary["primary_fit"]["slope"] * grid, "k--", lw=.9, zorder=5)
    axes[2].text(.04, .09, rf"$\beta={EXPECTED_BETA:.2f}$", transform=axes[2].transAxes, fontsize=8)
    axes[2].legend(title=r"$N_y$", loc="upper right", ncol=3, frameon=False,
                   handlelength=.6, columnspacing=.5, handletextpad=.15,
                   labelspacing=.2, title_fontsize=8)
    axes[2].set_xlabel(r"$\log[D(r_y)/D(N_y/2)]$", labelpad=2)
    axes[2].set_ylabel(r"$\log[\overline{C_G^{\mathrm{av}}(r_y)}/\overline{C_G^{\mathrm{av}}(N_y/2)}]$")
    axes[2].set_xlim(-3.04, .06); axes[2].set_xticks([-3, -2, -1, 0])
    axes[2].set_ylim(-.3, 9.4)
    for ax, letter in zip(axes, "abc"):
        ax.tick_params(direction="in", top=True, right=True)
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.text(-.10, 1.035, f"({letter})", transform=ax.transAxes, fontsize=9)
    prepare_figure(fig, STEM)
    fig.subplots_adjust(left=.205, right=.975, bottom=.075, top=.975, hspace=.53)
    assert all(ax.get_xscale() == ax.get_yscale() == "linear" for ax in axes)
    fig.canvas.draw()
    for axis in axes:
        bounds = axis.get_tightbbox(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    output.mkdir(parents=True, exist_ok=True)
    record_typography(fig, STEM)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{STEM}.{ext}", dpi=300)
    plt.close(fig)
    with (output / "data/correlations/plotted_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    checks.update({"axes": "linear axes showing log chord and log correlator; manuscript unchanged",
                   "canvas_inches": [3.375, 6.0], "PNG_dpi": 300,
                   "renderer_sha256": sha256(__file__),
                   "output_sha256": {f"{STEM}.{ext}": sha256(output / f"{STEM}.{ext}") for ext in ("pdf", "png")}})
    (output / "data/correlations/validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BUNDLE)
    args = parser.parse_args()
    (args.output_dir / "data/correlations").mkdir(parents=True, exist_ok=True)
    render(args.output_dir)
