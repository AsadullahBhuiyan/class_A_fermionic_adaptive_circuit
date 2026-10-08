#!/usr/bin/env python3
"""Saved histories and finite-window growth-rate scaling; no new dynamics."""
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
from log_ticks import add_log_minor_ticks
from analyze_late_gap import power_fit
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
    late=json.loads((DATA/'late_window_analysis.json').read_text())
    trajectories=np.genfromtxt(DATA/'late_window_trajectory_estimators.csv',delimiter=',',names=True)
    assert late['protocol']=='slab_only' and late['samples_per_size']==100
    sizes=np.array(late['sizes'])
    for row in late['summary']:
        sample=trajectories[trajectories['Ny']==row['Ny']]
        np.testing.assert_array_equal(sample['sample'],np.arange(100))
        for column,prefix in [('late_endpoint','late_endpoint'),('late_OLS','late_OLS'),('Delta_2Ny','Delta_2Ny')]:
            np.testing.assert_allclose(sample[column].mean(),row[prefix+'_mean'],atol=1e-14)
            np.testing.assert_allclose(sample[column].std(ddof=1)/10,row[prefix+'_SEM'],atol=1e-14)
    configure_style({'axes.linewidth':.8, 'xtick.direction':'in','ytick.direction':'in','legend.frameon':False})
    fig, axes = plt.subplots(3,1,figsize=(3.375,5.65),gridspec_kw={'height_ratios':[1,1,1.3]})
    axes[0].sharex(axes[1]);axes[0].tick_params(labelbottom=False)
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
    ax=axes[2]
    for key,prefix,color,marker,label in [
            ('Delta_2Ny','Delta_2Ny',ALPHA_COLORS[1],'o',r'$\overline\Delta(2N_y)$'),
            ('late_endpoint','late_endpoint',ALPHA_COLORS[3],'s',r'$\overline a_{\rm late}/2$')]:
        means=np.array([r[prefix+'_mean'] for r in late['summary']])
        sems=np.array([r[prefix+'_SEM'] for r in late['summary']])
        recovered=power_fit(sizes,means,sems)
        for field in ['z','z_regression_error','amplitude']:
            np.testing.assert_allclose(recovered[field],late['fits'][key][field],atol=1e-13)
        ax.errorbar(sizes,means,yerr=sems,color=color,marker=marker,mfc='white',
                    ls='none',ms=3.5,capsize=1,elinewidth=.7,mew=.8,label=label)
        xx=np.geomspace(19,63,200);fit=late['fits'][key]
        ax.plot(xx,fit['amplitude']*xx**(-fit['z']),color=color,ls='--',lw=.85)
    ax.set(xscale='log',yscale='log',xlim=(18,65),ylim=(.01,.10),
           xlabel=r'Circumference $N_y$',ylabel='Finite-window rate')
    ax.set_xticks([20,30,40,60]);ax.set_xticklabels(['20','30','40','60'])
    ax.legend(loc='upper right',handlelength=1.2)
    ax.text(.05,.22,r'$z_{\rm slope}=1.07\pm0.09$',transform=ax.transAxes,va='top')
    ax.text(-.18,1.04,'(c)',transform=ax.transAxes)
    ax.tick_params(which='both',top=True,right=True,pad=2)
    add_log_minor_ticks(fig)
    prepare_figure(fig,STEM)
    fig.subplots_adjust(left=.23,right=.97,bottom=.095,top=.95,hspace=.49)
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
                  cycle_range=[int(cycles[0]),int(cycles[-1])], layout=[3,1],
                  sample_means_and_SEMs_verified=True, raw_gap_equals_2t_rate=True,
                  time_fit_or_extrapolation=True, simulation=False,asymptotic_extrapolation=False,
                  added_panel='seven-size finite-window raw-gap growth rate comparison',
                  slope_label=r'overline a_late / 2',paired_trajectory_SEMs_verified=True,
                  slope_estimate_is_not_assumed_asymptotic_gap=True,
                  late_window_inputs={name:sha(DATA/name) for name in ['late_window_analysis.json','late_window_trajectory_estimators.csv']},
                  inputs=provenance['inputs'],renderer_sha256=sha(__file__),
                  outputs={f'{STEM}.{ext}':sha(ROOT/f'{STEM}.{ext}') for ext in ('pdf','png')})
    (DATA/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(ROOT/f'{STEM}.png')
if __name__=='__main__': main()
