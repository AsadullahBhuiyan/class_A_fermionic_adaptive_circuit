"""All seven saved circumferences at T=60, with the latest four-panel styling."""
import csv
import logging
from pathlib import Path

import numpy as np
import remake_gap_t40 as common

figure = common.figure
HERE = Path(__file__).resolve().parent
OUT = HERE/'fixed_T60_gap_all_sizes_v2'
SIZES = (20,24,30,36,44,56,60)
STEM = 'purification_ny30_gap_T60_all_sizes_4x1'


def main():
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    OUT.mkdir(exist_ok=True)
    samples, summary, inputs, inventory = common.extract(60, SIZES)
    assert len(samples) == 700
    x = np.array(SIZES)
    y = np.array([r['mean_gap'] for r in summary])
    e = np.array([r['sample_sem'] for r in summary])
    fit = common.weighted_power_law(x,y,e)
    np.testing.assert_allclose(fit['exponent'],
        common.weighted_power_law(x,120*y,120*e)['exponent'],atol=1e-12)
    sensitivity = [dict(Ny_min=int(x[start]), Ny_max=60,
        **common.weighted_power_law(x[start:],y[start:],e[start:])) for start in (0,1,2)]
    figure.write_csv(OUT/'gap_samples_T60.csv',samples)
    figure.write_csv(OUT/'gap_summary_T60.csv',summary)
    figure.write_csv(OUT/'fit_window_sensitivity.csv',sensitivity)
    a1 = figure.load_data(figure.DATA1,1)
    a3 = figure.load_data(figure.DATA3,3)
    density, _ = figure.mean_sem(a1['densities'])
    er,sr = figure.plot(a1['entropy'],a3['entropy'],a1['spatial'],density,summary,fit,
        gap_time_label=r'$T=60$',output_dir=OUT,stem=STEM,raw_cycles=True,
        gap_loglog=True,heatmap_time_annotation=False)
    for filename, records in [('total_entropy_curves.csv',er),('spatial_entropy_curves.csv',sr)]:
        with (HERE/filename).open() as stream:
            prior = list(csv.DictReader(stream))
        assert len(prior) == len(records)
        for old,new in zip(prior,records):
            for key,value in new.items():
                np.testing.assert_equal(float(old[key]),value)
    with np.load(HERE/'Ny030_slowest_mode_density.npz') as z:
        np.testing.assert_array_equal(density,z['mean'])
    caption = (HERE/'fixed_T40_gap_raw_cycles_v2/caption.tex').read_text().replace(
        'purification_ny30_gap_T40_raw_cycles_4x1.pdf',STEM+'.pdf').replace(
        'fixed cycle $T=40$', 'fixed cycle $T=60$').replace(
        'fig:purification-fixed-time-gap','fig:purification-fixed-t60-all-sizes')
    (OUT/'caption.tex').write_text(caption)
    table = '\n'.join(f"| {r['Ny']} | {r['mean_gap']:.6f} ± {r['sample_sem']:.6f} |" for r in summary)
    (OUT/'README.md').write_text(f'''# Fixed-T60 gaps, all seven sizes

Panel C: Campaign 13, Nx=20, Ny={SIZES}, S=100 per size, hard walls,
alpha1=1, alpha2=30, nshell=1, raster-y, complex128, slab-only measurement
with Born-conditioned exterior. All 140 result/receipt pairs independently
verified. Extract exact saved cycle 60, no interpolation or simulation.
Compute min absolute log[(1-nu)/nu] per trajectory after excluding saved
pure caps, divide by 120, then average. Uncertainty is sample SD/sqrt(100).

| Ny | Mean gap ± SEM |
|---:|---:|
{table}

SEM-weighted log-space power fit: z={fit['exponent']:.6f} ± {fit['exponent_sem']:.6f},
chi-square={fit['chi_squared']:.4f} for {fit['degrees_of_freedom']} degrees of freedom.
Fit errors propagate absolute sampling SEMs, with no residual rescaling or
bootstrap. This is a descriptive finite-time fit, not an asymptotic exponent;
fit-window sensitivity is saved separately. Raw modular gaps have the same
exponent because the time divisor is constant across sizes.

A/B retain raw cycles 1..60 and log-log axes; C retains log-log axes and
the latest tick styling. D is unchanged at T60, with time in caption only.
A/B/D are the full-measurement Ny30 Campaigns 21/22, not the slab-only
protocol in C. Sharing an observation time does not remove that distinction.
Numerical arrays in A/B/D are checked for exact equality to prior products.
Previous figures and manuscript are untouched. Reproduce with
python remake_gap_t60_all_sizes.py.
''')
    figure.write_json(OUT/'analysis_manifest.json',dict(
        schema='purification_fixed_T60_all_sizes_v2',T=60,normalization=120,
        Nx=20,sizes=list(SIZES),samples_per_size=100,verified_campaign_shards=140,
        panel_c_campaign=common.campaign.REVISION,panel_c_meas_slab_only=True,
        estimator='minimum absolute modular energy per trajectory / 120, then mean and ordinary SEM',
        fit_method='SEM-weighted log-space WLS; absolute sampling uncertainty; no bootstrap',
        fit=fit,fit_window_sensitivity=sensitivity,summary=summary,inventory=inventory,
        panels_abd_data_unchanged=True,
        inputs=inputs+a1['records']+a3['records'],
        source_hashes={str(p):figure.sha(p) for p in [Path(__file__),HERE/'make_figure.py',
            HERE/'remake_gap_t40.py',common.BUNDLE/'plot_endpoint_lyapunov_gap.py',
            common.BUNDLE/'analyze_campaign.py']},
        outputs=[dict(path=str(p),sha256=figure.sha(p)) for p in sorted(OUT.iterdir())
                 if p.name != 'analysis_manifest.json']))
    print(fit)
    print(OUT)


if __name__ == '__main__':
    main()
