#!/usr/bin/env python3
"""Replot legacy panel (d), with normalized cycles and a semilog inset."""
import json

import matplotlib.pyplot as plt
import numpy as np

from plot_ceff_absolute_error_vs_normalized_cycle import (
    HERE, ROOT, SOURCE_CSV, NY_VALUES, STYLES,
    configure_matplotlib, load_rows, sha256,
)


def main():
    rows = load_rows()
    original = json.loads((HERE / 'figure_manifest.json').read_text())
    for source in original['sources']:
        assert sha256(ROOT / source['path']) == source['sha256']
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.375, 2.75))
    fig.subplots_adjust(left=.19, right=.98, bottom=.18, top=.88)
    inset = ax.inset_axes([.43, .36, .49, .34])
    plotted = {}
    for ny in NY_VALUES:
        selected = [r for r in rows if r['Ny'] == ny
                    and r['cycle'] >= 10 and r['cycle'] % 5 == 0]
        t = np.array([r['cycle'] for r in selected]) / ny
        c = np.array([r['c_eff'] for r in selected])
        e = np.array([r['c_eff_fit_error'] for r in selected])
        np.testing.assert_allclose(t, [r['normalized_cycle'] for r in selected])
        assert np.isfinite(c).all() and np.isfinite(e).all()
        # The absolute-value transform is locally linear here: no interval
        # crosses one, so retaining the original symmetric errors is valid.
        assert np.all(c - e > 1)
        style = STYLES[ny]
        kw = dict(color=style['color'], marker=style['marker'],
                  markerfacecolor='white', linestyle='none',
                  markeredgewidth=.8, elinewidth=.65, capsize=1)
        ax.errorbar(t, c, yerr=e, markersize=3.8,
                    label=rf'$N_y={ny}$', **kw)
        inset.errorbar(t, np.abs(c - 1), yerr=e, markersize=2.5, **kw)
        plotted[str(ny)] = len(selected)
    ax.axhline(1, color='black', linestyle='--', linewidth=.8)
    ax.set(xlim=(0, 2.08), ylim=(.90, 2.45),
           xlabel=r'cycle $t/N_y$', ylabel=r'$c_{\mathrm{eff}}(t)=3m_1(t)$',
           title=r'$N_x=20$, $S=100$')
    ax.set_xticks([0, .5, 1, 1.5, 2])
    ax.legend(loc='upper right', ncol=3, columnspacing=.7,
              handlelength=1, handletextpad=.25, borderaxespad=.3)
    ax.text(-.18, 1.035, '(d)', transform=ax.transAxes,
            ha='left', va='bottom')
    inset.set(xlim=(0, 2.08), ylim=(.025, 1.65), yscale='log',
              xlabel=r'$t/N_y$', ylabel=r'$|c_{\mathrm{eff}}-1|$')
    inset.set_xticks([0, 1, 2])
    inset.xaxis.label.set_size(7)
    inset.yaxis.label.set_size(7)
    inset.xaxis.labelpad = 0
    inset.yaxis.labelpad = 1
    for axis in (ax, inset):
        axis.tick_params(which='both', direction='in', top=True, right=True)
    inset.tick_params(labelsize=6, pad=1, length=2)
    fig.canvas.draw()
    # Labels and tick labels must lie within the fixed-size PDF page.
    renderer = fig.canvas.get_renderer()
    for axis in (ax, inset):
        box = axis.get_tightbbox(renderer)
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    stem = HERE / 'figures/ceff_normalized_cycle_with_log_inset_Nx20_Ny30_40_50_S100'
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ('.pdf', '.png'):
        path = stem.with_suffix(extension)
        fig.savefig(path, dpi=300)
        outputs.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    plt.close(fig)
    manifest = {
        'schema': 'ceff_normalized_cycle_inset_v1',
        'source': {'path': str(SOURCE_CSV.relative_to(ROOT)), 'sha256': sha256(SOURCE_CSV)},
        'scientific_contract': original['scientific_contract'],
        'estimator': original['estimator'],
        'main_axes': {'x': 'cycle/Ny, linear', 'y': 'c_eff, linear'},
        'inset_axes': {'x': 'cycle/Ny, linear', 'y': 'abs(c_eff-1), logarithmic'},
        'points_per_size': plotted,
        'figure_inches': [3.375, 2.75],
        'original_figure_and_data_unchanged': True,
        'outputs': outputs,
    }
    (HERE / 'ceff_normalized_cycle_with_log_inset_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'points_per_size': plotted, 'outputs': outputs}, indent=2))


if __name__ == '__main__':
    main()
