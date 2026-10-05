#!/usr/bin/env python3
"""Log-log panels a-c at 2Ny; retain the explicitly labeled 4Ny heatmap."""
import csv
import json
from pathlib import Path

import numpy as np
from plot_purification_gap_summary import ROOT,make_figure
from plot_purification_gap_density_summary import sha

SOURCE=ROOT/'analysis_outputs/purification_2ny_v1'
OLD=ROOT/'analysis_outputs/purification_control_minmode_4x1_v2_single_column'
OUT=ROOT/'analysis_outputs/purification_2ny_loglog_v1'


def rows(name):
    with (SOURCE/name).open() as f:
        return [{k:(v if k=='window' else int(v) if k in ('Ny','x','cycle','samples','bundle','alpha_1','T') else float(v))
                 for k,v in r.items()} for r in csv.DictReader(f)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    entropy=rows('total_entropy_alpha_comparison.csv')
    spatial=rows('spatial_entropy_curves.csv')
    gaps=rows('gap_summary.csv')
    fit=json.loads((SOURCE/'analysis_summary.json').read_text())['fit']
    assert all(r['normalized_cycle']<=2 for r in entropy+spatial)
    assert all(r['T']==2*r['Ny'] for r in gaps)
    assert len(gaps)==7 and all(r['samples']==100 for r in gaps)
    with np.load(OLD/'Ny030_minimum_mode_density.npz',allow_pickle=False) as z:
        density=z['mean_density']
        assert z['sample_densities'].shape==(100,30,20)
        np.testing.assert_allclose(density,z['sample_densities'].mean(axis=0),atol=1e-15)
        np.testing.assert_allclose(density.sum(),1,atol=1e-12)
    make_figure(entropy,spatial,gaps,fit,output=OUT,alpha_comparison=True,normalized_stop=2,gap_time_multiple=2,
                density=density,density_vmax=float(density.max()),smallest_mode_density=True,density_time_multiple=4,
                loglog_abc=True)
    names=['total_entropy_alpha_comparison.csv','spatial_entropy_curves.csv','gap_summary.csv','analysis_summary.json']
    (OUT/'analysis_summary.json').write_text(json.dumps(dict(
        fit=fit,panel_a_b='Log-log; omit cycle zero only. Data through 2Ny; unchanged sample means/SEM.',
        panel_c='Log-log gap at T=2Ny; unchanged seven-size weighted power-law fit.',
        panel_d='Retained Ny30 heatmap at T=4Ny, explicitly labeled; no available T=2Ny covariance.',
        inputs=[dict(path=str(SOURCE/n),sha256=sha(SOURCE/n)) for n in names]+
               [dict(path=str(OLD/'Ny030_minimum_mode_density.npz'),sha256=sha(OLD/'Ny030_minimum_mode_density.npz'))],
        figure_inches=[3.375,6.8],new_simulations=False),indent=2)+'\n')
    (OUT/'README.md').write_text('# Log-log purification panels\n\n'
        'Panels a,b omit cycle zero and show the existing means and sample SEM through 2Ny on log-log axes. '
        'Panel c shows the recomputed T=2Ny gaps and unchanged fit z=1.078776 +/- 0.053979. '
        'Panel d remains the original T=4Ny heatmap, explicitly labeled; it has NOT been relabeled as 2Ny. '
        'For a same-time fourth panel, the separate T=2Ny x-profile remains available in ../purification_2ny_v1/. '
        'No new dynamics or fits were run. Original linear-axis figures are preserved.\n')
    print(OUT)


if __name__=='__main__': main()
