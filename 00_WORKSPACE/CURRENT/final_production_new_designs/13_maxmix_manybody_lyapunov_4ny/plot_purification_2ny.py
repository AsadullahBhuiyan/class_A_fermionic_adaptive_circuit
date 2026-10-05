#!/usr/bin/env python3
"""Shorten time series and recompute finite-time gaps at 2Ny; no simulations."""
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import analyze_campaign as campaign
from plot_purification_gap_summary import ROOT, make_figure, finite_time_gaps
from plot_endpoint_lyapunov_gap import write_csv, weighted_power_law
from plot_purification_gap_density_summary import sha, SOURCE
from plot_purification_control_minmode import minimum_mode_density

CONTROL_FIGURE = ROOT/'analysis_outputs/purification_control_minmode_4x1_v2_single_column'
OUT = ROOT/'analysis_outputs/purification_2ny_v1'


def read_csv(path):
    with path.open() as f:
        return [{k:(int(v) if k in ('Ny','x','cycle','samples','bundle','alpha_1') else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    entropy = [r for r in read_csv(CONTROL_FIGURE/'total_entropy_alpha_comparison.csv') if r['normalized_cycle']<=2]
    spatial = [r for r in read_csv(SOURCE/'spatial_entropy_curves.csv') if r['normalized_cycle']<=2]
    for a in (1,3):
        selected=[r for r in entropy if r['alpha_1']==a]
        np.testing.assert_array_equal([r['cycle'] for r in selected],np.arange(81))
    for ny in (20,30,40):
        for x in (5,15,10):
            np.testing.assert_array_equal([r['cycle'] for r in spatial if r['Ny']==ny and r['x']==x],np.arange(2*ny+1))
    manifest=campaign.verify_manifest()
    paths=sorted(campaign.DATA_ROOT.rglob('*.npz'))
    complete=sorted(campaign.DATA_ROOT.rglob('*.complete.json'))
    assert len(paths)==len(complete)==140
    inventory_digest=campaign.raw_inventory_hash(paths+complete)
    # Same documented historical aggregate-hash discrepancy as analyze_campaign.py.
    # Verify every result hash and receipt identity independently below.
    assert sum(p.stat().st_size for p in paths+complete)==manifest['inventory']['downloaded_bytes']
    samples, profiles, inputs = [], [], []
    for path in tqdm(paths,desc='Verify and extract T=2Ny',unit='shard'):
        receipt=json.loads(path.with_suffix('.complete.json').read_text())
        campaign.validate_completion(receipt,path)
        digest=sha(path)
        assert digest==receipt['result_sha256'] and path.stat().st_size==receipt['result_bytes']
        inputs.append(dict(path=str(path),sha256=digest))
        with np.load(path,allow_pickle=False) as z:
            ny=int(z['Ny']); t=2*ny
            assert str(z['configuration_hash'])==campaign.CONFIGURATION_HASH
            np.testing.assert_array_equal(z['sample_indices'],receipt['sample_indices'])
            idx=np.flatnonzero(z['spectrum_cycles']==t)
            assert len(idx)==1
            k=int(idx[0]); assert z['spectrum_seen'][:,k].all()
            gaps=finite_time_gaps(z['occupations'][:,idx],z['cap_mask'][:,idx],np.array([t]))[:,0]
            costs=z['soft_mode_flip_costs'][:,k]
            np.testing.assert_allclose(gaps,costs.min(axis=1)/(2*t),atol=5e-14)
            for j,s in enumerate(z['sample_indices']):
                samples.append(dict(Ny=ny,sample_index=int(s),T=t,gap=float(gaps[j])))
                if ny==30:
                    m=int(np.argmin(costs[j]))
                    assert np.count_nonzero(np.isclose(costs[j],costs[j,m],atol=1e-12,rtol=1e-10))==1
                    profile=z['soft_mode_x_profiles'][j,k,m]
                    np.testing.assert_allclose(profile.sum(),1,atol=1e-10)
                    for x,p in enumerate(profile):
                        profiles.append(dict(Ny=ny,T=t,sample_index=int(s),x=x,probability=float(p)))
    gaps=[]
    for ny in campaign.NY_VALUES:
        selected=sorted((r for r in samples if r['Ny']==ny),key=lambda r:r['sample_index'])
        np.testing.assert_array_equal([r['sample_index'] for r in selected],np.arange(100))
        values=np.array([r['gap'] for r in selected])
        gaps.append(dict(Ny=ny,window='endpoint',samples=100,T=2*ny,
                         mean_gap=float(values.mean()),sample_sem=float(values.std(ddof=1)/10)))
    fit=weighted_power_law(np.array([r['Ny'] for r in gaps]),np.array([r['mean_gap'] for r in gaps]),np.array([r['sample_sem'] for r in gaps]))
    density,_,_,density_inputs=minimum_mode_density()
    make_figure(entropy,spatial,gaps,fit,output=OUT,alpha_comparison=True,normalized_stop=2,gap_time_multiple=2,
                density=density,density_vmax=float(density.max()),smallest_mode_density=True,density_time_multiple=4)
    import matplotlib.pyplot as plt
    profile_rows=[]
    for x in range(20):
        local=sorted((r for r in profiles if r['x']==x),key=lambda r:r['sample_index'])
        np.testing.assert_array_equal([r['sample_index'] for r in local],np.arange(100))
        values=np.array([r['probability'] for r in local])
        profile_rows.append(dict(x=x,mean_probability=float(values.mean()),sample_sem=float(values.std(ddof=1)/10)))
    np.testing.assert_allclose(sum(r['mean_probability'] for r in profile_rows),1,atol=1e-10)
    fig,ax=plt.subplots(figsize=(3.375,2.5),layout='constrained')
    ax.errorbar(range(20),[r['mean_probability'] for r in profile_rows],
        yerr=[r['sample_sem'] for r in profile_rows],color='#1f77b4',marker='o',mfc='white',
        ms=3.5,lw=1.2,capsize=2,label='100 samples: mean $\\pm$ SEM')
    for x in (5,15): ax.axvline(x,color='.5',ls=':',lw=.7,zorder=0)
    ax.set(xlabel='$x$',ylabel=r'$\overline{p_{\min}(x)}$',xticks=[0,5,10,15,19],
        title=r'Hard wall: $N_y=30$, $T=2N_y=60$',xlim=(-.5,19.5),ylim=(-.015,.8))
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.legend(loc='upper center',frameon=False)
    for ext in ('pdf','png'): fig.savefig(OUT/('Ny030_minimum_mode_x_profile_2ny.'+ext),dpi=300)
    plt.close(fig)
    write_csv(OUT/'Ny030_minimum_mode_x_profile_2ny.csv',profile_rows)
    for name,rows in [('total_entropy_alpha_comparison',entropy),('spatial_entropy_curves',spatial),
                     ('gap_samples',samples),('gap_summary',gaps),('available_Ny030_x_profiles',profiles)]:
        write_csv(OUT/(name+'.csv'),rows)
    record=dict(fit=fit,cycles_cutoff='2Ny',samples_per_size=100,uncertainty='sample SEM; no bootstrap',
        panel_d_status='User requested original heatmap retained, explicitly labeled T=4Ny. Panels a-c use 2Ny.',
        additional_figure='Bundle13 y-summed x profile of unique minimum-magnitude mode at Ny30,T60, mean and sample SEM.',
        raw_inventory_digest=inventory_digest,manifest_inventory_digest=manifest['inventory']['raw_file_inventory_sha256'],
        inventory_note='Historical aggregate mismatch already documented by analyze_campaign.py; all 140 result SHA256 and receipt identities reverified.',
        original_heatmap_Ny=30,original_heatmap_T=120,requested_heatmap_T=60,
        inputs=inputs+density_inputs+[dict(path=str(p),sha256=sha(p)) for p in [CONTROL_FIGURE/'total_entropy_alpha_comparison.csv',SOURCE/'spatial_entropy_curves.csv']],
        new_simulations=False)
    (OUT/'analysis_summary.json').write_text(json.dumps(record,indent=2)+'\n')
    (OUT/'README.md').write_text('# Purification through 2Ny with retained 4Ny heatmap\n\n'
        'Panels (a,b) display the original sample means and SEM through cycle/Ny=2. '
        'Panel (c) uses saved bundle-13 occupations at exactly T=2Ny for each size; '
        'the minimum absolute finite-time exponent is calculated per trajectory before averaging '
        '100 trajectories. The power-law fit uses all seven sizes and SEM-weighted log-space regression. '
        'Panel (d) is deliberately unchanged at T=4Ny, explicitly labeled: the Ny=30 bundle-07 '
        'data retain full covariance only at cycle 120, not cycle 60.\n\n'
        'The separate x-profile figure uses bundle 13 at Ny=30,T=60. For each of 100 trajectories '
        'select the unique mode minimizing absolute lambda, sum its probability over y and orbitals, '
        'then average over trajectories. Error bars are sample SEM. Each profile sums to one. '
        'This is not a reconstructed (x,y) heatmap and not an average over the 16 saved modes. '
        'All ensembles are hard-wall, Nx=20, maxmix initialization, perfect correction, no postselection; '
        'alpha1=1 except the explicitly labeled alpha1=3 control in (a). No new simulations.\n')
    print(json.dumps(fit,indent=2)); print(OUT)


if __name__=='__main__':
    main()
