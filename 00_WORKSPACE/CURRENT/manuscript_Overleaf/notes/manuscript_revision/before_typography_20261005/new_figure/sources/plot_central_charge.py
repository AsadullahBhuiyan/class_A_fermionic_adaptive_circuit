#!/usr/bin/env python3
"""Reproduce Figure 8 from bundled independent cycle and endpoint ensembles."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
from central_charge_support import NY_VALUES, STYLES, configure_matplotlib, load_rows, plt
import central_charge_support as base
BUNDLE=Path(__file__).resolve().parent.parent
DATA=BUNDLE/'data/central_charge'
SOURCE_CSV=DATA/'ceff_vs_cycle_Nx20_Ny30_40_50_S100.csv'
SIZE_CSV=DATA/'ensemble_mean_curve_fits.csv'
STEM='Figure_08_central_charge'
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main(output):
    output.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((DATA/'input_provenance.json').read_text())
    for entry in provenance['sources']:
        path=DATA/Path(entry['path']).name
        if path.suffix=='.csv':
            assert sha256(path)==entry['sha256']
    rows = load_rows()
    with SIZE_CSV.open(newline='') as handle:
        endpoint = sorted((r for r in csv.DictReader(handle) if r['observable'] == 'c1'),
                          key=lambda r: int(r['Ny']))
    sizes = np.array([int(r['Ny']) for r in endpoint])
    np.testing.assert_array_equal(sizes, [30, 35, 40, 45, 50, 55, 60])
    for row in endpoint:
        assert int(row['Nx']) == 20 and int(row['samples']) == 100
        assert row['estimator_order'] == 'average_curves_then_fit'
        assert int(row['Ay_fit_min']) == 8
        assert int(row['Ay_fit_max']) == int(row['Ny']) // 2
    configure_matplotlib()
    fig, (ax, right) = plt.subplots(2, 1, figsize=(3.375, 5.6))
    fig.subplots_adjust(left=.20, right=.97, bottom=.085, top=.94, hspace=.50)
    inset = ax.inset_axes([.43, .36, .49, .34])
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
           ylabel=r'$c_{\mathrm{eff}}(t)=3m_1(t)$')
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
    c_end = np.array([float(r['converted_coefficient']) for r in endpoint])
    sem = np.array([float(r['converted_covariance_SEM']) for r in endpoint])
    assert np.isfinite(c_end).all() and np.isfinite(sem).all() and (sem >= 0).all()
    np.testing.assert_allclose(c_end, 3 * np.array([float(r['slope']) for r in endpoint]))
    right.errorbar(sizes, c_end, yerr=sem, color='#1F77B4', marker='o',
                   markerfacecolor='white', markeredgewidth=.9, markersize=4,
                   linestyle='--', linewidth=.9, capsize=2, elinewidth=.8)
    right.set(xlim=(28, 62), ylim=(.995, 1.085), xlabel=r'circumference $N_y$',
              ylabel=r'$c_{\mathrm{eff}}(2N_y)=3m_1(2N_y)$')
    right.set_xticks(sizes)
    right.text(.96, .94, r'$t=2N_y$', transform=right.transAxes, ha='right', va='top')
    for label, axis in zip(('(a)', '(b)'), (ax, right)):
        axis.axhline(1, color='black', linestyle='--', linewidth=.8)
        axis.set_title(r'$N_x=20$, $S=100$')
        axis.tick_params(which='both', direction='in', top=True, right=True)
        axis.text(-.18, 1.035, label, transform=axis.transAxes, ha='left', va='bottom', fontsize=9)
    fig.canvas.draw()
    for axis in (ax, right, inset):
        box = axis.get_tightbbox(fig.canvas.get_renderer())
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    assert not ax.get_tightbbox(fig.canvas.get_renderer()).overlaps(
        right.get_tightbbox(fig.canvas.get_renderer()))
    outputs={}
    for ext in ('pdf','png'):
        path=output/f'{STEM}.{ext}'
        fig.savefig(path,dpi=300)
        outputs[path.name]=sha256(path)
    plt.close(fig)
    archived=json.loads((DATA/'ceff_cycle_and_endpoint_size_1x2.json').read_text())
    assert counts==archived['panel_a']['points_per_size']
    assert sizes.tolist()==archived['panel_b']['sizes']
    checks=dict(layout=[2,1],figure_inches=[3.375,5.6],input_csv_hashes_verified=True,cycle_points_per_size=counts,endpoint_sizes=sizes.tolist(),fit_values_preserved=True,ensembles_pooled=False,
        panel_a_uncertainty='OLS fit-slope standard error of mean entropy curve; not trajectory SEM',
        panel_b_uncertainty='trajectory sampling SEM propagated with full width covariance',
        endpoint_estimator='average entropy curves then fit',fit_window='8 <= Ay <= floor(Ny/2)',
        endpoint_central_charge=c_end.tolist(),endpoint_trajectory_SEM=sem.tolist(),
        renderer_sha256=sha256(__file__),helper_sha256=sha256(base.__file__),outputs=outputs)
    target=output/'data/central_charge';target.mkdir(parents=True,exist_ok=True)
    (target/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BUNDLE)
    main(parser.parse_args().output_dir)
