#!/usr/bin/env python3
"""Panel (c), unchanged endpoint means, SEM and fit, with logarithmic axes."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'analysis_outputs/purification_contour_gap_3x1_v3_single_column'
OUT = ROOT/'analysis_outputs/endpoint_gap_panel_c_loglog_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (SOURCE/'gap_summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    fit = json.loads((SOURCE/'analysis_summary.json').read_text())['weighted_log_space_power_law_fit']
    ny = np.array([int(r['Ny']) for r in rows])
    mean = np.array([float(r['mean_gap']) for r in rows])
    sem = np.array([float(r['sample_sem']) for r in rows])
    np.testing.assert_array_equal(ny, [20,24,30,36,44,56,60])
    assert all(r['window']=='endpoint' and int(r['samples'])==100 for r in rows)
    assert np.all(mean-sem>0) and np.all(sem>0)
    # Independently confirm the existing weighted log-space fit is unchanged.
    design = np.column_stack([np.ones(len(ny)), -np.log(ny)])
    weight = mean/sem
    beta = np.linalg.lstsq(design*weight[:,None], np.log(mean)*weight, rcond=None)[0]
    np.testing.assert_allclose(beta, [np.log(fit['amplitude']),fit['exponent']], atol=1e-12)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'mathtext.fontset':'cm','axes.linewidth':.8})
    fig, ax = plt.subplots(figsize=(3.375,2.65))
    dense = np.geomspace(20,60,300)
    ax.plot(dense,fit['amplitude']*dense**(-fit['exponent']),color='.25',ls='--',lw=1.2,
            label=rf'$A N_y^{{-z}},\ z={fit["exponent"]:.2f}\pm{fit["exponent_sem"]:.2f}$')
    ax.errorbar(ny,mean,yerr=sem,ls='none',color='#1f77b4',marker='o',mfc='white',
                ms=4,mew=1,capsize=2,lw=1.1,label=r'$T=4N_y$: mean $\pm$ SEM')
    ax.set(xscale='log',yscale='log',xlim=(19,63),ylim=(.014,.083),
           xlabel=r'circumference $N_y$',ylabel=r'$\Delta$')
    ax.xaxis.set_major_locator(FixedLocator(ny))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    ax.yaxis.set_major_locator(FixedLocator([.02,.03,.04,.06,.08]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.2f}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.legend(loc='upper right',frameon=False,handlelength=1.8,labelspacing=.2)
    ax.text(-.18,1.025,'(c)',transform=ax.transAxes,fontsize=9)
    fig.tight_layout(pad=.6)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'hard_wall_endpoint_gap_loglog.{ext}',dpi=300)
    plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(dict(
        sources=[dict(path=str(SOURCE/name),sha256=hashlib.sha256((SOURCE/name).read_bytes()).hexdigest())
                 for name in ('gap_summary.csv','analysis_summary.json')],
        fit=fit,axes='log-log',samples_per_size=100,
        definition='Per-trajectory min_j |log((1-nu_j)/nu_j)|/(2*T), then sample mean; T=4Ny.',
        uncertainty='Sample-wise SEM, no bootstrap.',
        protocol='Bundle 13: hard wall, Nx=20, alpha1=1, alpha2=30, nshell=1, maxmix initialization, Born sampling, perfect correction.',
        fit_window='All seven sizes, Ny=20 through 60; same saved weighted log-space fit.',
        figure_inches=[3.375,2.65]),indent=2)+'\n')
    print(OUT)


if __name__=='__main__':
    main()
