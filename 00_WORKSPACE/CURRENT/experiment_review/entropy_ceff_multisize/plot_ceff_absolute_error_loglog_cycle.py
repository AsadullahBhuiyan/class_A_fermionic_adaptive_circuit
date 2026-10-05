#!/usr/bin/env python3
"""Replot panel (d) as |1-c_eff| versus physical cycle, on log-log axes."""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from plot_ceff_absolute_error_vs_normalized_cycle import (
    HERE, SOURCE_CSV, NY_VALUES, STYLES, load_rows, configure_matplotlib, sha256,
)


def main():
    rows = [r for r in load_rows() if r['cycle'] >= 10 and r['cycle'] % 5 == 0]
    configure_matplotlib()
    plt.rcParams.update({'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
    fig, ax = plt.subplots(figsize=(3.375,2.75))
    for ny in NY_VALUES:
        selected = [r for r in rows if r['Ny']==ny]
        t = np.array([r['cycle'] for r in selected])
        c = np.array([r['c_eff'] for r in selected])
        e = np.array([r['c_eff_fit_error'] for r in selected])
        value = abs(1-c)
        # All displayed fit intervals lie above 1, so the transform preserves SE.
        assert np.all(c-e>1) and np.all(value>0)
        style = STYLES[ny]
        ax.errorbar(t,value,yerr=e,color=style['color'],marker=style['marker'],
                    ls=style['linestyle'],mfc='white',mew=.8,ms=3.5,lw=1,
                    capsize=1.2,label=rf'$N_y={ny}$')
        for row,y in zip(selected,value):
            np.testing.assert_allclose(row['abs_c_eff_minus_one'],y)
    ax.set(xscale='log',yscale='log',xlim=(9,110),
           xlabel='cycle',ylabel=r'$|1-c_{\mathrm{eff}}|$',title=r'$N_x=20$, $S=100$')
    ax.xaxis.set_major_locator(FixedLocator([10,20,30,50,100]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='both',top=True,right=True)
    ax.legend(loc='upper right',frameon=False)
    ax.text(-.18,1.03,'(d)',transform=ax.transAxes,fontsize=9)
    fig.tight_layout(pad=.6)
    stem=HERE/'figures/abs_one_minus_ceff_vs_cycle_loglog_Nx20_Ny30_40_50_S100'
    for ext in ('pdf','png'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=300)
    plt.close(fig)
    csv_path=HERE/'data/abs_one_minus_ceff_vs_cycle_loglog_Nx20_Ny30_40_50_S100.csv'
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'abs_one_minus_ceff_loglog_cycle_manifest.json').write_text(json.dumps(dict(
        source=str(SOURCE_CSV),source_sha256=sha256(SOURCE_CSV),
        estimator='Absolute deviation of c_eff=3*slope of trajectory-averaged strip entropy fit; not average absolute sample deviations.',
        fit_window='Ay=8..Ny/2',uncertainty='Unchanged OLS fit SE, not trajectory SEM; displayed intervals do not cross c_eff=1.',
        plotted_cycles='Same as original panel: 10,15,...,2Ny',
        axes={'x':'physical cycle, logarithmic','y':'abs(1-c_eff), logarithmic'},
        data_points=len(rows),figure_inches=[3.375,2.75],new_fits=False),indent=2)+'\n')
    print(stem.with_suffix('.png'))


if __name__=='__main__':
    main()
