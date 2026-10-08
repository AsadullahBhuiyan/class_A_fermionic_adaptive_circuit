#!/usr/bin/env python3
"""Reproduce the standalone main-text figure: entropy-coefficient convergence with a residual inset."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from central_charge_support import NY_VALUES, STYLES, configure_matplotlib, load_rows, plt
import central_charge_support as base
from log_ticks import add_log_minor_ticks
from manuscript_typography import prepare_figure, record_typography
BUNDLE=Path(__file__).resolve().parent.parent
DATA=BUNDLE/'data/central_charge'
SOURCE_CSV=DATA/'ceff_vs_cycle_Nx20_Ny30_40_50_S100.csv'
STEM='Figure_08_central_charge'
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def draw_convergence(ax):
    """Draw the saved convergence curves and inset on an existing axis."""
    rows = load_rows()
    inset = ax.inset_axes([.55, .39, .40, .34])
    counts = {}
    for ny in NY_VALUES:
        selected = [r for r in rows if r['Ny'] == ny and r['cycle'] >= 10
                    and r['cycle'] % 5 == 0]
        t = np.array([r['cycle'] for r in selected]) / ny
        c = np.array([r['c_eff'] for r in selected])
        e = np.array([r['c_eff_fit_error'] for r in selected])
        np.testing.assert_allclose(t, [r['normalized_cycle'] for r in selected])
        assert np.all(c - e > 1)
        style = STYLES[ny]
        kw = dict(color=style['color'], marker=style['marker'],
                  markerfacecolor='white', linestyle='none', markeredgewidth=.8,
                  elinewidth=.65, capsize=1)
        ax.errorbar(t, c, yerr=e, markersize=3.8, label=rf'$N_y={ny}$', **kw)
        inset.errorbar(t, np.abs(c - 1), yerr=e, markersize=2.5, **kw)
        counts[str(ny)] = len(selected)
    ax.set(xlim=(0, 2.08), ylim=(.90, 2.45), xlabel=r'cycle $t/N_y$',
           ylabel=r'$c_{\mathrm{eff}}(t)$')
    ax.set_xticks([0, .5, 1, 1.5, 2])
    ax.legend(loc='upper right', ncol=3, columnspacing=.7,
              handlelength=1, handletextpad=.25, borderaxespad=.3)
    inset.set(xlim=(0, 2.08), ylim=(.025, 1.65), yscale='log',
              xlabel=r'$t/N_y$', ylabel=r'$|c_{\mathrm{eff}}-1|$')
    inset.set_xticks([0, 1, 2])
    inset.xaxis.label.set_size(8)
    inset.yaxis.label.set_size(8)
    inset.xaxis.labelpad = 0
    inset.yaxis.labelpad = 1
    inset.tick_params(which='both', direction='in', top=True, right=True,
                      labelsize=8, pad=1, length=2)
    ax.axhline(1, color='black', linestyle='--', linewidth=.8)
    ax.set_title(r'$N_x=20$, $S=100$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    return counts, inset

def main(output):
    output.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((DATA/'input_provenance.json').read_text())
    for entry in provenance['sources']:
        path=DATA/Path(entry['path']).name
        if path.suffix=='.csv':
            assert sha256(path)==entry['sha256']
    rows = load_rows()
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.05, 3.10))
    # Use the full text width with a larger, clearly separated residual inset.
    fig.subplots_adjust(left=.09, right=.98, bottom=.18, top=.94)
    counts, inset = draw_convergence(ax)
    ax.set_title('')
    add_log_minor_ticks(fig)
    prepare_figure(fig, "Figure_08_central_charge")
    fig.canvas.draw()
    for axis in (ax, inset):
        box = axis.get_tightbbox(fig.canvas.get_renderer())
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    outputs={}
    record_typography(fig, "Figure_08_central_charge")
    for ext in ('pdf','png'):
        path=output/f'{STEM}.{ext}'
        fig.savefig(path,dpi=300)
        outputs[path.name]=sha256(path)
    plt.close(fig)
    archived=json.loads((DATA/'ceff_cycle_and_endpoint_size_1x2.json').read_text())
    assert counts==archived['panel_a']['points_per_size']
    checks=dict(layout=[1,1],figure_inches=[7.05,3.10],input_csv_hashes_verified=True,
        cycle_points_per_size=counts,fit_values_preserved=True,ensembles_pooled=False,
        panel_letters=[],endpoint_panel_displayed=False,title_removed=True,
        y_label='c_eff(t); extraction defined in caption',
        uncertainty='OLS fit-slope standard error of mean entropy curve; not trajectory SEM',
        estimator='average entropy curves then fit',fit_window='8 <= Ay <= floor(Ny/2)',
        preserved_endpoint_input='ensemble_mean_curve_fits.csv',
        renderer_sha256=sha256(__file__),helper_sha256=sha256(base.__file__),outputs=outputs)
    target=output/'data/central_charge';target.mkdir(parents=True,exist_ok=True)
    (target/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BUNDLE)
    main(parser.parse_args().output_dir)
