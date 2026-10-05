"""Panel C at fixed T=60 for the five user-selected circumferences."""
import csv
import json
import logging
from pathlib import Path

import numpy as np

import remake_gap_t40 as common

figure = common.figure
HERE = Path(__file__).resolve().parent
OUT = HERE/'fixed_T60_gap_ny30-60_v1'
SIZES = (30,36,44,56,60)
T = 60


def main():
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    OUT.mkdir(exist_ok=True)
    protected = [figure.OUT/f'purification_ny30_full_measurement_4x1.{ext}' for ext in ('pdf','png')]
    protected += list((HERE/'fixed_T40_gap_v1').glob('*'))
    protected += [HERE/'analysis_manifest.json', HERE/'caption.tex']
    before = {str(p):figure.sha(p) for p in protected if p.is_file()}
    samples, summary, inputs, inventory = common.extract(T, SIZES)
    assert len(samples) == 500 and [r['Ny'] for r in summary] == list(SIZES)
    x = np.array(SIZES)
    y = np.array([r['mean_gap'] for r in summary])
    e = np.array([r['sample_sem'] for r in summary])
    fit = common.weighted_power_law(x,y,e)
    raw_fit = common.weighted_power_law(x,2*T*y,2*T*e)
    np.testing.assert_allclose(fit['exponent'], raw_fit['exponent'], atol=1e-12)
    sensitivity = []
    for start in (0,1,2):
        f = common.weighted_power_law(x[start:],y[start:],e[start:])
        sensitivity.append(dict(Ny_min=int(x[start]), Ny_max=int(x[-1]), **f))
    figure.write_csv(OUT/'gap_samples_T60.csv', samples)
    figure.write_csv(OUT/'gap_summary_T60.csv', summary)
    figure.write_csv(OUT/'fit_window_sensitivity.csv', sensitivity)
    common.draw_gap(summary, fit, cycle=T, output=OUT)
    print('Fixed-T60 fit:', json.dumps(fit), flush=True)

    a1, a3 = figure.load_data(figure.DATA1,1), figure.load_data(figure.DATA3,3)
    density, _ = figure.mean_sem(a1['densities'])
    er,sr = figure.plot(a1['entropy'],a3['entropy'],a1['spatial'],density,summary,fit,
        gap_time_label=r'$T=60$', output_dir=OUT, stem='purification_ny30_gap_T60_4x1')
    for filename, records in [('total_entropy_curves.csv',er),('spatial_entropy_curves.csv',sr)]:
        with (HERE/filename).open() as handle:
            prior = list(csv.DictReader(handle))
        assert len(prior) == len(records)
        for left,right in zip(prior,records):
            for key,value in right.items():
                np.testing.assert_allclose(float(left[key]),value,rtol=0,atol=0)
    with np.load(HERE/'Ny030_slowest_mode_density.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(density,z['mean'])

    caption = (HERE/'caption.tex').read_text()
    caption = caption.replace('purification_ny30_full_measurement_4x1.pdf','purification_ny30_gap_T60_4x1.pdf')
    caption = caption.replace('at $T=2N_y$,','at the same fixed cycle $T=60$ for every size,')
    caption = caption.replace('$N_y=20,24,30,36,44,56,60$','$N_y=30,36,44,56,60$')
    caption = caption.replace('fig:purification-full-measurement-ny30','fig:purification-fixed-t60-gap')
    (OUT/'caption.tex').write_text(caption)
    table = '\n'.join(f"| {r['Ny']} | {r['mean_gap']:.6f} ± {r['sample_sem']:.6f} |" for r in summary)
    windows = '\n'.join(f"| {r['Ny_min']}–60 | {r['exponent']:.4f} ± {r['exponent_sem']:.4f} |" for r in sensitivity)
    (OUT/'README.md').write_text(f'''# Panel C at fixed T=60

Campaign 13, Nx=20, Ny=30,36,44,56,60, 100 trajectories per size.
The smaller Ny=20,24 cases also contain cycle 60 but are omitted by the
user's explicit size selection. This is not a data-availability limitation.
All 140 campaign result/completion pairs were checksum- and identity-verified;
the selected 100 shards supply exactly 500 trajectories, IDs 0–99 per size.

Hard walls, alpha1=1, alpha2=30, nshell=1, maxmix, Born-conditioned exterior,
perfect correction, raster-y, complex128, meas_slab_only=True.
For each sample take min_j abs(log[(1-nu_j)/nu_j]) at saved cycle 60,
exclude the stored pure caps, divide by 2T=120, then average the 100 gaps.
Errors are ordinary sample SD/sqrt(100), not bootstrap or residual errors.
All gaps are cross-checked against saved soft-mode flip costs.

| Ny | Mean half gap ± SEM |
|---:|---:|
{table}

Preserve panel C's original weighted log-space power-law estimator:
log(mean gap)=log(A)-z*log(Ny), with weights (mean/SEM)^2. The result is
z={fit['exponent']:.6f} ± {fit['exponent_sem']:.6f};
chi-square={fit['chi_squared']:.4f} for {fit['degrees_of_freedom']} degrees
of freedom. The fit uncertainty is propagated from sampling SEMs without
residual rescaling. It does not include finite-time or model uncertainties.

| Fit window | z ± propagated statistical error |
|---:|---:|
{windows}

All sizes share a divisor of 120. Raw modular gaps produce exactly the
same size exponent. This fixed-time result does not establish the
infinite-time limit. No interpolation, new simulation or pooling with
Campaign 26 is performed.

The four-panel version changes only C. A/B/D retain Campaigns 21–22
full-system measurement data at Ny=30, with endpoint T=60. Their means,
SEMs and heatmap are exactly unchanged. Although C and D now share an
observation time, their slab-only versus full-measurement protocols remain
different; C is not a size sweep of the protocol in A/B/D.

Saved as standalone gap_vs_Ny_T60.pdf/png and compound
purification_ny30_gap_T60_4x1.pdf/png, with caption.tex and raw/sample
summary CSVs. Original and T40 figure assets are preserved; the manuscript
has not been changed. Reproduce with python remake_gap_t60.py.
''')
    assert all(figure.sha(Path(p)) == h for p,h in before.items())
    outputs = sorted(p for p in OUT.iterdir() if p.name != 'analysis_manifest.json')
    figure.write_json(OUT/'analysis_manifest.json',dict(schema='purification_fixed_T60_selected_sizes_v1',
        T=T, normalization=2*T, Nx=20, sizes=list(SIZES), samples_per_size=100,
        verified_campaign_shards=140, selected_shards=100, selected_trajectories=500,
        panel_c_campaign=common.campaign.REVISION, panel_c_meas_slab_only=True,
        estimator='trajectory-wise minimum absolute modular energy / 120, then mean and ordinary SEM',
        fit_method='SEM-weighted log-space WLS; absolute propagated sampling errors; no bootstrap',
        fit=fit,fit_window_sensitivity=sensitivity,summary=summary,inventory=inventory,
        panels_abd_unchanged=True, original_assets_sha256=before,
        inputs=inputs+a1['records']+a3['records'],
        source_hashes={str(p):figure.sha(p) for p in [Path(__file__),HERE/'remake_gap_t40.py',HERE/'make_figure.py',common.BUNDLE/'plot_endpoint_lyapunov_gap.py',common.BUNDLE/'analyze_campaign.py']},
        outputs=[dict(path=str(p),sha256=figure.sha(p)) for p in outputs]))
    print(json.dumps(summary,indent=2)); print(OUT)


if __name__ == '__main__':
    main()
