#!/usr/bin/env python3
"""Plot saved slab-only Ny=30 histories; no simulation, fitting or extrapolation."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from manuscript_typography import configure_style, prepare_figure, record_typography
from manuscript_palette import ALPHA_COLORS
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/gap_convergence'
STEM = 'Figure_A05_gap_convergence'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    provenance = json.loads((DATA/'provenance.json').read_text())
    for name, expected in provenance['inputs'].items():
        assert sha(DATA/name) == expected
    with (DATA/'cycle_gap_summary.csv').open() as stream:
        all_rows = list(csv.DictReader(stream))
        rows = [r for r in all_rows if r['protocol']=='slab_only' and int(r['Ny'])==30]
    with (DATA/'ny30_sample_cycle_gaps.csv').open() as stream:
        sample_rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: int(r['cycle']))
    cycles = np.array([int(r['cycle']) for r in rows])
    assert cycles[-1] == 120 and all(int(r['samples']) == 100 for r in rows)
    columns = {}
    for raw, mean, sem in [('gap','mean_gap','gap_sem'),('modular_gap','mean_modular_gap','modular_gap_sem')]:
        columns[mean] = np.array([float(r[mean]) for r in rows])
        columns[sem] = np.array([float(r[sem]) for r in rows])
        for row in rows:
            selected = [r for r in sample_rows if int(r['cycle'])==int(row['cycle'])]
            assert len(selected)==100 and {int(r['sample']) for r in selected}==set(range(100))
            values = np.array([float(r[raw]) for r in selected])
            np.testing.assert_allclose(values.mean(), float(row[mean]), rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(values.std(ddof=1)/10, float(row[sem]), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(columns['mean_modular_gap'],2*cycles*columns['mean_gap'], rtol=1e-12)
    np.testing.assert_allclose(columns['modular_gap_sem'],2*cycles*columns['gap_sem'], rtol=1e-12)
    configure_style({'axes.linewidth':.8, 'xtick.direction':'in','ytick.direction':'in'})
    fig, axes = plt.subplots(2,1,figsize=(3.375,3.55),sharex=True)
    for ax, mean, sem, label, letter in zip(axes,
            ('mean_gap','mean_modular_gap'),('gap_sem','modular_gap_sem'),
            (r'$\overline{\Delta}(t)$',r'$\overline{g_{\rm mod}}(t)$'),'ab'):
        ax.fill_between(cycles,columns[mean]-columns[sem],columns[mean]+columns[sem], color=ALPHA_COLORS[1],alpha=.18,linewidth=0)
        ax.plot(cycles, columns[mean],color=ALPHA_COLORS[1],lw=1)
        ax.axvline(60,color='.35',ls='--',lw=.8)
        ax.set_ylabel(label,labelpad=2)
        ax.tick_params(which='both',top=True,right=True,pad=2)
        ax.text(-.18,1.04,f'({letter})',transform=ax.transAxes)
        ax.set_xlim(0,120); ax.set_xticks([0,30,60,90,120])
    axes[0].text(.04,.94,r'$N_y=30,\ S=100$',ha='left',va='top',transform=axes[0].transAxes)
    axes[1].set_xlabel('Cycle $t$')
    prepare_figure(fig,STEM)
    fig.subplots_adjust(left=.20,right=.97,bottom=.13,top=.94,hspace=.29)
    record_typography(fig,STEM)
    for ext in ('pdf','png'):
        fig.savefig(ROOT/f'{STEM}.{ext}',dpi=300)
    plt.close(fig)
    changes = {}
    for ny in (20,24,30,36,44,56,60):
        rates = {int(r['cycle']):float(r['mean_gap']) for r in all_rows
                 if r['protocol']=='slab_only' and int(r['Ny'])==ny}
        changes[str(ny)] = {'2Ny_to_4Ny_percent':100*(rates[4*ny]/rates[2*ny]-1),
                            '3Ny_to_4Ny_percent':100*(rates[4*ny]/rates[3*ny]-1)}
    report = dict(seven_size_saved_rate_changes=changes, all_checks_passed=True, protocol='slab_only', Ny=30,Nx=20,samples=100,
                  cycle_range=[int(cycles[0]),int(cycles[-1])], layout=[2,1],
                  sample_means_and_SEMs_verified=True, raw_gap_equals_2t_rate=True,
                  time_fit_or_extrapolation=False, simulation=False,
                  inputs=provenance['inputs'],renderer_sha256=sha(__file__),
                  outputs={f'{STEM}.{ext}':sha(ROOT/f'{STEM}.{ext}') for ext in ('pdf','png')})
    (DATA/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(ROOT/f'{STEM}.png')
if __name__=='__main__': main()
