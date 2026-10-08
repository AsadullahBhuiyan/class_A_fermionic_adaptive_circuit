#!/usr/bin/env python3
"""Render main Figure 10 and appendix Figure A4 from saved channel data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
import numpy as np
from manuscript_palette import ALPHA_COLORS
from log_ticks import add_log_minor_ticks
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/mean_channel'
STEM = 'Figure_11_mean_channel'
SCAN_STEM = 'Figure_A04_channel_gap_scan'
DISPLAY_CUTOFF = 1e-20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(filename):
    return json.loads((DATA / filename).read_text())


def read_csv(filename):
    with (DATA / filename).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in read_json('provenance.json')['compact_inputs_sha256'].items():
        assert sha(DATA / name) == expected, f'Compact input changed: {name}'
    spectral_summary = read_json('spectrum_summary.json')
    correlator_summary = read_json('correlator_summary.json')
    for alpha in (1, 3):
        spectrum = spectral_summary[f'alpha{alpha}_hard']
        correlator = correlator_summary['cases'][str(alpha)]
        assert spectrum['source_sha256'] == correlator['input_sha256']
        assert spectrum['config'] == correlator['config']

    manuscript_style({'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    # Retain the original physical panel dimensions while moving the scan to A4.
    panel_height = 7.15 * (.97 - .06) / (4 + 3 * .45)
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 5.35))
    top, correlator_ax, fit_ax = axes
    spectral_provenance = read_json('untwirled_spectra_provenance.json')
    assert sha(DATA / 'untwirled_spectra.npz') == spectral_provenance['sha256']
    scan_fig, gap_ax = plt.subplots(figsize=(3.375, 2.0))
    displayed_distances = {}
    panel_b_rows = []
    with np.load(DATA / 'untwirled_spectra.npz', allow_pickle=False) as spectra, \
            np.load(DATA / 'curves.npz', allow_pickle=False) as curves:
        r = np.arange(1, 33)
        xx = np.log((64 / np.pi) * np.sin(np.pi * r / 64))
        for alpha, color, marker, size, ls in (
            (1, ALPHA_COLORS[1], 'o', 10, '-'), (3, ALPHA_COLORS[3], '^', 12, ':')
        ):
            occupations = spectra[f'alpha{alpha}']
            assert occupations.shape == (2560,) and np.all(np.diff(occupations) >= 0)
            # Every occupation is displayed with an open marker; no subsampling.
            top.scatter(np.arange(1, 2561), occupations,
                        s=size, marker=marker, facecolors='none', edgecolors=color,
                        linewidths=.55, label=rf'$\alpha_1={alpha}$',
                        zorder=3 if alpha == 3 else 2)
            values = curves[f'alpha{alpha}_xavg'][1:]
            displayed = np.isfinite(values) & (values >= DISPLAY_CUTOFF) & (r >= 2)
            displayed_distances[str(alpha)] = r[displayed].tolist()
            yy = np.full(values.shape, np.nan)
            yy[displayed] = values[displayed]
            correlator_ax.plot(r, yy, color=color, marker=marker, ls=ls, ms=3.5,
                               mfc='white', mew=.65, lw=.75, label=rf'$\alpha_1={alpha}$')
            panel_b_rows += [{'alpha_1': alpha, 'r_y': int(d), 'log_chord': x,
                              'squared_mean_correlator': value, 'log_correlator': np.log(y)}
                             for d, x, value, y in zip(r[displayed], xx[displayed], values[displayed], yy[displayed])]
    top.axhline(.5, color='.5', ls='--', lw=.7, zorder=0)
    top.set(xlabel=r'Ordered mode index $j$', ylabel=r'Occupation $\nu_j$',
            xlim=(-50, 2610), ylim=(-.035, 1.035))
    top.set_xticks([1, 1280, 2560])
    top.get_xticklabels()[-1].set_horizontalalignment('right')
    top.set_yticks([0, .5, 1])
    top.legend(loc='upper left', frameon=False, handletextpad=.3)
    correlator_ax.set(xscale='log', yscale='log', xlabel=r'$r_y$',
                      ylabel=r'$C_{\overline{G}}(r_y)$',
                      xlim=(1.9, 33), ylim=(DISPLAY_CUTOFF, .05))
    correlator_ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 30]))
    correlator_ax.xaxis.set_major_formatter(ScalarFormatter())
    correlator_ax.xaxis.set_minor_formatter(NullFormatter())
    correlator_ax.yaxis.set_major_locator(FixedLocator([1e-20, 1e-14, 1e-8, 1e-2]))
    correlator_ax.legend(loc='upper right', frameon=False, handletextpad=.3)

    rows = read_csv('multiplier_gaps.csv')
    assert len(rows) == 105 and all(r['status'] == 'resolved_positive' and int(r['Nx']) == 20 for r in rows)
    for size, color, marker, style in zip(
        (20, 40, 60, 80, 100), (ALPHA_COLORS[3], '#23934c', ALPHA_COLORS[1], '#8e44ad', '#d17b0f'),
        ('^', 's', 'o', 'D', 'v'), (':', '--', '-', '-.', (0, (3, 1, 1, 1)))
    ):
        selected = sorted((r for r in rows if int(r['Ny']) == size), key=lambda r: float(r['alpha_1']))
        alpha = np.array([float(r['alpha_1']) for r in selected])
        gaps = np.array([float(r['g_C']) for r in selected])
        np.testing.assert_allclose(alpha, np.arange(10, 31) / 10, rtol=0, atol=1e-14)
        np.testing.assert_allclose(gaps, 1 - np.array([float(r['rho_A']) for r in selected])**2, rtol=0, atol=1e-14)
        np.testing.assert_allclose(gaps, -np.expm1(-np.array([float(r['decay_rate']) for r in selected])), rtol=0, atol=1e-14)
        gap_ax.plot(alpha, gaps, color=color, marker=marker, ls=style, mfc='white',
                    ms=3, lw=.8, label=str(size))
    gap_ax.set(xlabel=r'$\alpha_1$', ylabel=r'$g_C=1-\rho(A)^2$')
    gap_ax.set_yticks([.8, .85, .9, .95])
    gap_ax.legend(title=r'$N_y$ ($N_x=20$)', frameon=False, loc='lower right',
                  ncol=2, columnspacing=.7, handlelength=1.4, handletextpad=.4, labelspacing=.1, borderpad=.2)

    fits = read_json('multiplier_fits.json')['results']
    fit_rows = read_csv('multiplier_fit_inputs.csv')
    handles, checked_fits = [], {}
    for alpha, color, marker in ((1, ALPHA_COLORS[1], 'o'), (3, ALPHA_COLORS[3], '^')):
        selected = sorted((r for r in fit_rows if int(r['alpha_1']) == alpha), key=lambda r: int(r['Ny']))
        n = np.array([int(r['Ny']) for r in selected])
        y = np.array([float(r['g_C']) for r in selected])
        np.testing.assert_array_equal(n, [20, 40, 60, 80, 100])
        for row in selected:
            scan_row = next(r for r in rows if float(r['alpha_1']) == alpha and r['Ny'] == row['Ny'])
            np.testing.assert_allclose(float(scan_row['g_C']), float(row['g_C']), rtol=0, atol=1e-14)
        saved = fits[str(alpha)]
        saved_coefficients = [saved['g_infinity'], saved['a']]
        coefficients = np.polynomial.polynomial.polyfit(1 / n, y, 1)
        np.testing.assert_allclose(coefficients, saved_coefficients, rtol=0, atol=1e-12)
        limit, correction = saved_coefficients
        residual = y - (limit + correction / n)
        r2 = 1 - float(residual @ residual) / float((y - y.mean()) @ (y - y.mean()))
        np.testing.assert_allclose(r2, saved['r_squared'], rtol=0, atol=1e-14)
        fit_ax.plot(1 / n, y, ls='none', marker=marker, mfc='white', color=color, ms=4, zorder=3)
        x = np.linspace(0, .05, 200)
        fit_ax.plot(x, limit + correction * x, color=color, ls='-', lw=1)
        fit_ax.plot(0, limit, marker='s', color=color, ms=3.5)
        fit_ax.text(.05, .43 if alpha == 1 else .76,
                    rf'$g_\infty={limit:.5f}$' + '\n' + rf'$R^2={r2:.6f}$',
                    color=color, transform=fit_ax.transAxes, va='top', fontsize=8, linespacing=1.5)
        handles.append(Line2D([], [], color=color, ls='none', marker=marker, mfc='white', ms=4,
                              label=rf'$\alpha_1={alpha}$'))
        checked_fits[str(alpha)] = {'g_infinity': limit, 'a': correction, 'r_squared': r2,
                                    'sizes': n.tolist(), 'maximum_refit_coefficient_difference':
                                    float(np.max(np.abs(coefficients - saved_coefficients)))}
    handles.append(Line2D([], [], color='.3', ls='-', lw=1,
                          label=r'Fit: $g_\infty+a/N_y$'))
    fit_ax.set(xlabel=r'$1/N_y$', ylabel=r'$g_C=1-\rho(A)^2$',
                xlim=(-.001, .052), ylim=(.82, .98))
    fit_ax.set_yticks([.84, .88, .92, .96])
    fit_ax.legend(handles=handles, frameon=False, loc='center right',
                   handlelength=1.2, handletextpad=.35, borderpad=.1, labelspacing=.12)
    for ax, letter in zip(axes, 'abc'):
        ax.tick_params(top=True, right=True)
        ax.text(-.19, 1.04, f'({letter})', transform=ax.transAxes, fontsize=9)
    add_log_minor_ticks(fig)
    prepare_figure(fig, STEM)
    fig.subplots_adjust(left=.20, right=.97, top=.97,
                        bottom=.97-panel_height*(3+2*.45)/5.35, hspace=.45)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in axes]
    assert all(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height for b in boxes), [(b.bounds, fig.bbox.bounds) for b in boxes]
    assert all(boxes[i].y0 > boxes[i+1].y1 for i in range(2)), 'Panels overlap'
    record_typography(fig, STEM)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{STEM}.{ext}', dpi=300)
    plt.close(fig)

    gap_ax.tick_params(top=True, right=True)
    add_log_minor_ticks(scan_fig)
    prepare_figure(scan_fig, SCAN_STEM)
    scan_fig.subplots_adjust(left=.20, right=.89, bottom=.25,
                             top=.25+panel_height/2.0)
    scan_fig.canvas.draw()
    box = gap_ax.get_tightbbox(scan_fig.canvas.get_renderer())
    assert box.x0 >= 0 and box.y0 >= 0
    assert box.x1 <= scan_fig.bbox.width and box.y1 <= scan_fig.bbox.height
    record_typography(scan_fig, SCAN_STEM)
    for ext in ('pdf', 'png'):
        scan_fig.savefig(out / f'{SCAN_STEM}.{ext}', dpi=300)
    plt.close(scan_fig)

    # Keep a readable numerical receipt, including the actual displayed A/B selection.
    if out == ROOT:
        with (DATA / 'panel_b_plotted_data.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(panel_b_rows[0]))
            writer.writeheader()
            writer.writerows(panel_b_rows)
    receipt = {
        'layout': [3, 1], 'figure_inches': [3.375, 5.35], 'dpi': 300,
        'displayed_panels': ['occupation_spectrum', 'mean_correlator', 'gap_size_fit'],
        'original_to_main_panel_mapping': {'a': 'a', 'b': 'b', 'd': 'c'},
        'appendix_scan_stem': SCAN_STEM, 'appendix_scan_layout': [1, 1],
        'appendix_scan_figure_inches': [3.375, 2.0],
        'panels_ab_same_input_arrays_and_estimators': True,
        'panel_a_spectrum': spectral_provenance,
        'panel_a_cycle':128, 'panel_a_markers':'Open marker for every eigenvalue; no subsampling',
        'panel_a_matrix':'Untwirled final trajectory-averaged covariance',
        'panel_b_axes': {'x': 'r_y', 'y': 'C_Gbar(r_y)', 'xscale': 'log', 'yscale': 'log'},
        'panel_b_display_cutoff': DISPLAY_CUTOFF,
        'panel_b_min_ry': 2,
        'panel_b_displayed_separations': displayed_distances,
        'appendix_scan_points': len(rows), 'Nx': 20, 'Ny': [20, 40, 60, 80, 100],
        'gap_definition': 'g_C=1-rho(A)^2', 'gap_units': 'dimensionless',
        'panel_c_fit_objective': 'unweighted least squares on g_C; all five sizes',
        'panel_c_saved_fits_recovered': checked_fits,
        'text_within_canvas_and_panels_separated': True,
        'outputs': {f'{stem}.{ext}': sha(out / f'{stem}.{ext}')
                    for stem in (STEM, SCAN_STEM) for ext in ('pdf', 'png')},
    }
    receipt_path = DATA / 'validation.json' if out == ROOT else out / 'validation.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(out / f'{STEM}.pdf')
    print(out / f'{SCAN_STEM}.pdf')


if __name__ == '__main__':
    main()
