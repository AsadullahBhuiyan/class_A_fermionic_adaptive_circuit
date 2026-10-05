"""Restore original T=2Ny size sweep with latest display-only styling."""
import csv
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import remake_gap_t40 as common

figure=common.figure
HERE=Path(__file__).resolve().parent
OUT=HERE/'scaled_T2Ny_gap_raw_cycles_v2'
STEM='purification_ny30_gap_T2Ny_raw_cycles_4x1'


def main():
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    source_path=HERE.parent/'purification_gap_cycle_convergence/analyze.py'
    spec=importlib.util.spec_from_file_location('restored_gap_source',source_path)
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    groups,inputs,sources=source.load_slab()
    summary=[];samples=[]
    for ny,data in sorted(groups.items()):
        k=np.flatnonzero(data['cycles']==2*ny).item()
        values=data['rate'][:,k];raw=data['raw'][:,k]
        mean,sem=source.mean_sem(values)
        summary.append(dict(Ny=ny,T=2*ny,samples=100,mean_gap=float(mean),sample_sem=float(sem)))
        samples.extend(dict(Ny=ny,T=2*ny,sample_index=i,modular_gap=float(raw[i]),gap=float(values[i])) for i in range(100))
    with (figure.OLD/'gap_summary.csv').open() as stream:
        original=list(csv.DictReader(stream))
    assert len(original)==len(summary)
    for old,new in zip(original,summary):
        for key,value in new.items():np.testing.assert_allclose(float(old[key]),value,atol=1e-14,rtol=1e-13)
    fit=common.weighted_power_law(np.array([r['Ny'] for r in summary]),
        np.array([r['mean_gap'] for r in summary]),np.array([r['sample_sem'] for r in summary]))
    OUT.mkdir(exist_ok=True)
    figure.write_csv(OUT/'gap_summary_T2Ny.csv',summary)
    figure.write_csv(OUT/'gap_samples_T2Ny.csv',samples)
    a1=figure.load_data(figure.DATA1,1);a3=figure.load_data(figure.DATA3,3)
    density,_=figure.mean_sem(a1['densities'])
    er,sr=figure.plot(a1['entropy'],a3['entropy'],a1['spatial'],density,summary,fit,
        gap_time_label=r'$T=2N_y$',output_dir=OUT,stem=STEM,
        raw_cycles=True,gap_loglog=True,heatmap_time_annotation=False)
    for name,rows in [('total_entropy_curves.csv',er),('spatial_entropy_curves.csv',sr)]:
        with (HERE/name).open() as stream:prior=list(csv.DictReader(stream))
        assert len(prior)==len(rows)
        for old,new in zip(prior,rows):
            for key,value in new.items():np.testing.assert_equal(float(old[key]),value)
    with np.load(HERE/'Ny030_slowest_mode_density.npz') as z:
        np.testing.assert_array_equal(density,z['mean'])
    caption=(HERE/'caption.tex').read_text().replace('purification_ny30_full_measurement_4x1.pdf',STEM+'.pdf')
    caption=caption.replace('Panels (a,b) use logarithmic axes;',
        'Panels (a,b) show raw cycle number $t$ on logarithmic axes;')
    caption=caption.replace('(c) The previously obtained','(c) On log--log axes, the previously obtained')
    caption=caption.replace('fig:purification-full-measurement-ny30','fig:purification-restored-scaled-time-gap')
    (OUT/'caption.tex').write_text(caption)
    (OUT/'README.md').write_text('''# Restored T=2Ny panel C

Restores the original size sweep at T=2Ny, NOT fixed T40 or T60.
Nx20, Ny20,24,30,36,44,56,60; 100 trajectories per size. Independently
revalidated all 140 Campaign 13 result/receipt pairs and exactly matched
the original seven mean gaps and SEMs. Each sample gap is the minimum
absolute modular energy divided by 2T, before sample averaging.
Fit: SEM-weighted log-space WLS, ordinary propagated sampling errors.
This remains a finite-time scaled-duration fit, not a proven infinite-time law.

Panel C is slab-only with Born-conditioned exterior; A/B/D retain distinct
full-measurement Campaigns 21/22 at Ny30. All are hard-wall alpha2=30,
nshell1, perfect-correction raster-y covariance dynamics. Caption retains
protocol and estimator differences. A/B show raw cycles through60;
B has alpha1=1. C is log-log with clearer ticks. D retains the same T60
mode-density heatmap; its time appears only in the caption.
Earlier fixed-time versions and manuscript are unchanged.
Reproduce: python restore_gap_2ny.py.
''')
    figure.write_json(OUT/'analysis_manifest.json',dict(schema='restored_T2Ny_gap_v2',
        time_schedule='T=2Ny',normalization='2T=4Ny',summary=summary,fit=fit,
        panel_c_meas_slab_only=True,panels_abd_data_unchanged=True,
        exact_match_to_original_summary=True,inputs=inputs+a1['records']+a3['records'],
        campaign13_source_hashes=sources,
        source_hashes={str(p):figure.sha(p) for p in [Path(__file__),HERE/'make_figure.py',
            HERE/'remake_gap_t40.py',source_path,common.BUNDLE/'plot_endpoint_lyapunov_gap.py']},
        outputs=[dict(path=str(p),sha256=figure.sha(p)) for p in sorted(OUT.iterdir())
                 if p.name!='analysis_manifest.json']))
    print(json.dumps(fit));print(OUT)


if __name__=='__main__':main()
