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
from matplotlib.ticker import FixedLocator, ScalarFormatter, LogFormatterMathtext
import numpy as np
from log_ticks import add_log_minor_ticks
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
    numerator = denominator = residual = anchored_total = 0.0
    nonanchor_fit_points = 0
    for ny in SIZES:
        r = np.arange(8, ny // 2 + 1)
        xx = np.log(np.sin(np.pi * r / ny))
        yy = np.log(means[ny][r] / means[ny][-1])
        numerator += np.mean(xx * yy)
        denominator += np.mean(xx * xx)
        residual += np.mean((yy + EXPECTED_BETA * xx)**2)
        anchored_total += np.mean(yy**2)
        # The normalized antipodal point is fixed at (0,0), not an observation.
        nonanchor_fit_points += int(np.count_nonzero(r < ny // 2))
    recovered_beta = -numerator / denominator
    assert summary["primary_fit"]["beta"] == EXPECTED_BETA
    np.testing.assert_allclose(recovered_beta, EXPECTED_BETA, rtol=0, atol=2e-14)
    recovered_r_squared = 1 - residual / anchored_total
    np.testing.assert_allclose(recovered_r_squared, summary['primary_fit']['R0_squared'], rtol=0, atol=2e-14)
    # Retain original source-panel IDs A/C; displayed panels are (a,b).
    endpoints = {ny: float(next(row['unmasked_correlator'] for row in original
                                if row['panel'] == 'c' and int(row['Ny']) == ny
                                and int(row['ry']) == ny//2)) for ny in SIZES}
    rows = []
    for old in original:
        if old['panel'] not in ('a', 'c') or int(old['ry']) < 2:
            continue
        row = dict(old)
        ny, r, value = int(row['Ny']), int(row['ry']), float(row['unmasked_correlator'])
        row['source_log_x'], row['source_log_y'] = row['x'], row['y']
        row['x'] = str(float(r) if row['panel'] == 'a' else float(np.sin(np.pi*r/ny)))
        row['y'] = str(value if row['panel'] == 'a' else value/endpoints[ny])
        row['displayed'] = str(row['panel'] == 'c' or value > 1e-20)
        rows.append(row)
    with (DATA/'requested_plotted_data.csv').open() as stream:
        requested = list(csv.DictReader(stream))
    assert len(rows) == len(requested)
    for row, target in zip(rows, requested):
        assert int(row['ry']) == int(target['ry']) and int(row['Ny']) == int(target['Ny'])
        assert row['displayed'] == target['displayed'] and row['in_fit'] == target['in_fit']
        np.testing.assert_allclose([float(row['x']),float(row['y'])],
                                   [float(target['x']),float(target['y'])],rtol=2e-14,atol=0)
    return rows, summary, {
        "input_hashes_verified": True,
        "archived_curves_recovered": True,
        "requested_presentation_values_matched": True,
        "display_panel_source_mapping": {"a": "a", "b": "c"},
        "removed_source_panels": ["b"],
        "display_min_ry": 2,
        "panel_a_cutoff": 1e-20,
        "max_abs_difference_compact_vs_archived_correlator": max_abs_difference,
        "primary_beta_archived": EXPECTED_BETA,
        "primary_beta_independently_recovered": float(recovered_beta),
        "beta_uncertainty": {
            "method": "formal weighted least-squares regression standard error in log-normalized coordinates, with intercept fixed to zero",
            "beta_point_estimate": EXPECTED_BETA,
            "beta_standard_error": float(np.sqrt(residual / (nonanchor_fit_points - 1) / denominator)),
            "weighted_residual_sum_squares": float(residual),
            "weighted_design_sum_squares": float(denominator),
            "nonanchor_fit_points": nonanchor_fit_points,
            "residual_degrees_of_freedom": nonanchor_fit_points - 1,
            "weighting": "each original size-specific fit window has total weight one",
            "anchor_handling": "six deterministic antipodal anchors excluded from residual degrees of freedom; original weights and fitted slope unchanged",
            "scope": "conditional regression error; does not account for correlations among separations or fit-window and finite-size systematics",
            "compact_curves_sha256": sha256(DATA / "compact_curves.npz"),
        },
        "displayed_R_squared": summary['primary_fit']['R0_squared'],
        "R_squared_independently_recovered": float(recovered_r_squared),
        "R_squared_definition": "uncentered, in log normalized coordinates; equal total weight per size; archived statistic unchanged",
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
    uncertainty = checks["beta_uncertainty"]
    manuscript_style({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'savefig.bbox': None})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.85))
    for alpha, color, marker, style in ((1, '#0072B2', 'o', '-'), (3, '#D55E00', '^', ':')):
        label = rf'$\alpha_1={alpha}$'
        axes[0].plot(*series(rows, 'a', label), color=color, marker=marker, ls=style,
                     lw=.85, ms=3, mfc='white', mew=.7, label=label)
    axes[0].set(xscale='log', yscale='log', xlim=(1.88, 32), ylim=(5e-21, .001),
                xlabel=r'$r_y$', ylabel=r'$\overline{C_G}(r_y)$')
    axes[0].xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 30]))
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_major_locator(FixedLocator([1e-20, 1e-16, 1e-12, 1e-8, 1e-4]))
    axes[0].legend(loc='lower left', frameon=False, handlelength=1.8, handletextpad=.4)
    # Match the entropy figures: one uniform band for the union of fit windows.
    axes[1].axvspan(*np.exp(summary['fit_shading']['union']),
                   color='0.5', alpha=.15, lw=0, zorder=0)
    for ny, color, marker in zip(SIZES, ('#D55E00','#009E73','#0072B2','#CC79A7','#E69F00','#333333'),
                                 ('^','s','o','v','D','>')):
        axes[1].plot(*series(rows, 'c', str(ny)), marker=marker, ls='none', color=color,
                     ms=3, mfc='white', mew=.65, label=rf'${ny}$')
    grid = np.geomspace(.1, 1, 200)
    size_handles, size_labels = axes[1].get_legend_handles_labels()
    fit_line, = axes[1].plot(grid, grid**(-EXPECTED_BETA), 'k--', lw=.9, zorder=5,
                             label=r'$[\sin(\pi r_y/N_y)]^{-\beta}$')
    fit_legend = axes[1].legend(handles=[fit_line], loc='lower left',
                               bbox_to_anchor=(.04, .27), frameon=False,
                               handlelength=1.3, handletextpad=.35, borderaxespad=0)
    axes[1].add_artist(fit_legend)
    axes[1].text(.04,.09,rf'$\beta={EXPECTED_BETA:.3f}\pm{uncertainty["beta_standard_error"]:.3f}$' + '\n' +
                 rf"$R^2={summary['primary_fit']['R0_squared']:.6f}$",
                 transform=axes[1].transAxes, va='bottom', linespacing=1.7)
    axes[1].set(xscale='log',yscale='log',xlim=(.098,1.06),ylim=(.75,200),
                xlabel=r'$\sin(\pi r_y/N_y)$',
                ylabel=r'$\overline{C_G}(r_y)/\overline{C_G}(N_y/2)$')
    axes[1].xaxis.set_major_locator(FixedLocator([.1,.2,.5,1]))
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].yaxis.set_major_locator(FixedLocator([1,10,100]))
    axes[1].legend(handles=size_handles, labels=size_labels,
                   title=r'$N_y$',loc='upper right',ncol=3,frameon=False,
                   handlelength=.6,columnspacing=.5,handletextpad=.15,labelspacing=.2)
    for ax,letter in zip(axes,'ab'):
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.tick_params(direction='in',top=True,right=True)
        ax.text(-.19,1.045,f'({letter})',transform=ax.transAxes)
        assert ax.get_xscale() == ax.get_yscale() == 'log'
    add_log_minor_ticks(fig)
    prepare_figure(fig, STEM)
    fig.subplots_adjust(left=.10,right=.985,bottom=.20,top=.90,wspace=.32)
    fig.canvas.draw()
    for ax in axes:
        bounds = ax.get_tightbbox(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    output.mkdir(parents=True, exist_ok=True)
    record_typography(fig, STEM)
    for ext in ('pdf','png'):
        fig.savefig(output/f'{STEM}.{ext}',dpi=300)
    plt.close(fig)
    with (output/'data/correlations/plotted_data.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    checks.update({'axes':'log-log axes; raw separation in (a), normalized chord in (b)',
                   'beta_uncertainty': uncertainty,
                   'canvas_inches':[7.05,2.85], 'layout':[1,2],'PNG_dpi':300,
                   'renderer_sha256':sha256(__file__),
                   'output_sha256':{f'{STEM}.{ext}':sha256(output/f'{STEM}.{ext}') for ext in ('pdf','png')}})
    (output/'data/correlations/validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    (output/'data/correlations/beta_uncertainty.json').write_text(json.dumps(uncertainty,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BUNDLE)
    args = parser.parse_args()
    (args.output_dir / "data/correlations").mkdir(parents=True, exist_ok=True)
    render(args.output_dir)
