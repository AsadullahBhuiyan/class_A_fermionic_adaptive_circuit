"""Recompute panel C from Campaign 13 at T=40; retain A/B/D and old assets."""
import csv
import importlib.util
import json
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import make_figure as figure

HERE = Path(__file__).resolve().parent
BUNDLE = figure.BUNDLES/'13_maxmix_manybody_lyapunov_4ny'
sys.path.insert(0, str(BUNDLE))
spec = importlib.util.spec_from_file_location('campaign13_analysis', BUNDLE/'analyze_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
from plot_endpoint_lyapunov_gap import weighted_power_law

OUT = HERE/'fixed_T40_gap_v1'
T = 40


def gaps_at_cycle(data, cycle):
    """Exact saved-time selection; exclude stored pure caps, then min per sample."""
    if cycle <= 0:
        raise ValueError('The finite-time rate needs a strictly positive cycle')
    indices = np.flatnonzero(data['spectrum_cycles'] == cycle)
    if len(indices) != 1:
        raise ValueError(f'Cycle {cycle} was not saved exactly once')
    k = int(indices[0])
    if not data['spectrum_seen'][:, k].all():
        raise ValueError('Incomplete spectrum observations')
    nu = data['occupations'][:, k]
    caps = data['cap_mask'][:, k]
    assert nu.ndim == 2 and caps.shape == nu.shape
    assert np.isfinite(nu).all() and np.all((nu >= 0) & (nu <= 1))
    interior = ~caps
    assert np.all(interior.sum(1) > 0)
    assert np.all(nu[interior] > campaign.CAP_TOLERANCE)
    assert np.all(nu[interior] < 1-campaign.CAP_TOLERANCE)
    costs = np.full(nu.shape, np.inf)
    costs[interior] = abs(np.log1p(-nu[interior])-np.log(nu[interior]))
    raw = costs.min(1)
    np.testing.assert_allclose(raw, data['soft_mode_flip_costs'][:, k].min(1), atol=5e-12)
    nearest = np.take_along_axis(nu, np.argmin(costs, axis=1)[:, None], axis=1)[:, 0]
    return raw, raw/(2*cycle), nearest, interior.sum(1)


def extract(cycle=T, sizes=None):
    sizes = tuple(campaign.NY_VALUES if sizes is None else sizes)
    if not sizes or len(set(sizes)) != len(sizes) or not set(sizes) <= set(campaign.NY_VALUES):
        raise ValueError('Invalid or duplicated selected sizes')
    download = campaign.verify_manifest()
    paths = sorted(campaign.DATA_ROOT.rglob('*.npz'))
    assert len(paths) == 140
    rows, inputs = [], []
    for path in tqdm(paths, desc=f'Verify Campaign 13 and extract T={cycle}', unit='shard'):
        receipt_path = path.with_suffix('.complete.json')
        receipt = json.loads(receipt_path.read_text())
        campaign.validate_completion(receipt, path)
        digest = figure.sha(path)
        assert digest == receipt['result_sha256'] and path.stat().st_size == receipt['result_bytes']
        inputs.extend(dict(path=str(p), sha256=h) for p,h in
                      ((path, digest), (receipt_path, figure.sha(receipt_path))))
        with np.load(path, allow_pickle=False) as z:
            assert str(z['configuration_hash']) == campaign.CONFIGURATION_HASH
            assert int(z['Nx']) == 20 and int(z['Ny']) == receipt['Ny']
            np.testing.assert_array_equal(z['sample_indices'], receipt['sample_indices'])
            if int(z['Ny']) not in sizes:
                continue
            raw, gaps, nearest, counts = gaps_at_cycle(z, cycle)
            for i, sample in enumerate(z['sample_indices']):
                rows.append(dict(Nx=20, Ny=int(z['Ny']), sample_index=int(sample), T=cycle,
                                 modular_gap=float(raw[i]), gap=float(gaps[i]),
                                 nearest_occupation=float(nearest[i]), finite_modes=int(counts[i])))
    summary = []
    for ny in sizes:
        case = sorted((r for r in rows if r['Ny'] == ny), key=lambda r:r['sample_index'])
        np.testing.assert_array_equal([r['sample_index'] for r in case], np.arange(100))
        values = np.array([r['gap'] for r in case])
        assert np.isfinite(values).all()
        summary.append(dict(Ny=ny, T=cycle, samples=100, mean_gap=float(values.mean()),
                            sample_sem=float(values.std(ddof=1)/10),
                            mean_modular_gap=float(2*cycle*values.mean()),
                            sem_modular_gap=float(2*cycle*values.std(ddof=1)/10)))
    inventory_hash = campaign.raw_inventory_hash(paths+[p.with_suffix('.complete.json') for p in paths])
    return rows, summary, inputs, dict(computed_inventory_sha256=inventory_hash,
        historical_inventory_sha256=download['inventory']['raw_file_inventory_sha256'],
        note='Historical aggregate inventory mismatch previously documented; all individual receipts and result SHA-256 hashes independently reverified.')


def draw_gap(summary, fit, cycle=T, output=OUT):
    figure.configure_style()
    fig, ax = plt.subplots(figsize=(3.375,2.5), layout='constrained')
    x = np.array([r['Ny'] for r in summary])
    ax.errorbar(x, [r['mean_gap'] for r in summary], yerr=[r['sample_sem'] for r in summary],
                fmt='o', color='#1f77b4', mfc='white', ms=4, capsize=2,
                label=rf'$T={cycle}$: mean $\pm$ SEM')
    xx = np.linspace(x.min(), x.max(), 300)
    ax.plot(xx, fit['amplitude']*xx**(-fit['exponent']), '--', color='.25', lw=1,
            label=rf"$A N_y^{{-z}},\ z={fit['exponent']:.2f}\pm{fit['exponent_sem']:.2f}$")
    ax.set(xlabel=r'circumference $N_y$', ylabel=rf'$\Delta(T={cycle})$', xticks=x,
           ylim=(0, max(r['mean_gap']+r['sample_sem'] for r in summary)*1.3))
    ax.legend(frameon=False, loc='upper right')
    for ext in ('pdf','png'):
        fig.savefig(output/f'gap_vs_Ny_T{cycle}.{ext}', dpi=300)
    plt.close(fig)


def main():
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    OUT.mkdir(exist_ok=True)
    protected = [figure.OUT/f'purification_ny30_full_measurement_4x1.{ext}' for ext in ('pdf','png')]
    protected += [HERE/'analysis_manifest.json', HERE/'caption.tex']
    original = {str(p):figure.sha(p) for p in protected}
    samples, summary, inputs, inventory = extract()
    x = np.array([r['Ny'] for r in summary])
    y = np.array([r['mean_gap'] for r in summary])
    e = np.array([r['sample_sem'] for r in summary])
    fit = weighted_power_law(x, y, e)  # preserve panel C's historical log-space estimator
    raw_fit = weighted_power_law(x, 80*y, 80*e)
    np.testing.assert_allclose(fit['exponent'], raw_fit['exponent'], atol=1e-12)
    sensitivity = []
    for start in (0,1,2):
        f = weighted_power_law(x[start:],y[start:],e[start:])
        sensitivity.append(dict(Ny_min=int(x[start]), Ny_max=int(x[-1]), **f))
    figure.write_csv(OUT/'gap_samples_T40.csv', samples)
    figure.write_csv(OUT/'gap_summary_T40.csv', summary)
    figure.write_csv(OUT/'fit_window_sensitivity.csv', sensitivity)
    draw_gap(summary, fit)
    print('Fixed-T40 fit:', json.dumps(fit), flush=True)

    a1, a3 = figure.load_data(figure.DATA1, 1), figure.load_data(figure.DATA3, 3)
    density, _ = figure.mean_sem(a1['densities'])
    entropy_rows, spatial_rows = figure.plot(a1['entropy'], a3['entropy'], a1['spatial'],
        density, summary, fit, gap_time_label=r'$T=40$', output_dir=OUT,
        stem='purification_ny30_gap_T40_4x1')
    # Assert the unchanged panels use precisely the same means and SEMs.
    for filename, records in [('total_entropy_curves.csv', entropy_rows), ('spatial_entropy_curves.csv', spatial_rows)]:
        with (HERE/filename).open() as handle:
            prior = list(csv.DictReader(handle))
        assert len(prior) == len(records)
        for left,right in zip(prior, records):
            for key,value in right.items():
                np.testing.assert_allclose(float(left[key]), value, atol=0, rtol=0)
    with np.load(HERE/'Ny030_slowest_mode_density.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(density, z['mean'])
    caption = (HERE/'caption.tex').read_text()
    caption = caption.replace('purification_ny30_full_measurement_4x1.pdf','purification_ny30_gap_T40_4x1.pdf')
    caption = caption.replace('at $T=2N_y$,','at the same fixed cycle $T=40$ for every size,')
    caption = caption.replace('fig:purification-full-measurement-ny30','fig:purification-fixed-time-gap')
    (OUT/'caption.tex').write_text(caption)
    table = '\n'.join(f"| {r['Ny']} | {r['mean_gap']:.6f} ± {r['sample_sem']:.6f} |" for r in summary)
    (OUT/'README.md').write_text(f'''# Purification figure with panel C at fixed T=40

Panel C is recomputed from Campaign 13, not Campaign 26. Nx=20,
Ny=20,24,30,36,44,56,60; 100 independent Born trajectories per size.
Hard walls, alpha1=1, alpha2=30, nshell=1, maxmix with Born-conditioned
exterior, perfect correction, raster-y, complex128, meas_slab_only=True.
All 140 result/receipt pairs and all sample IDs are verified.

At exactly the saved cycle T=40, take each trajectory's minimum absolute
modular energy log[(1-nu)/nu], excluding saved pure-mode caps, and divide
by 2T=80. Average these 100 gaps; SEM=sample SD/sqrt(100). The minimum is
not taken after spectral or covariance averaging. Cross-check every gap
against the stored soft-mode flip costs. No bootstrap or new simulation.

| Ny | Mean finite-time half gap ± SEM |
|---:|---:|
{table}

The fit preserves the original panel C estimator: weighted least squares
of log(mean gap) against log(Ny), weights (mean/SEM)^2. Propagated
one-standard-error fit uncertainties use the absolute sampling SEMs,
without residual rescaling. z={fit['exponent']:.6f} ± {fit['exponent_sem']:.6f},
chi-square={fit['chi_squared']:.4f} for {fit['degrees_of_freedom']} degrees
of freedom. This is a descriptive fixed-time fit, not an established
infinite-time Lyapunov exponent. The divisor is constant across all sizes;
fitting raw modular gaps produces the same exponent.

The all-size fit has reduced chi-square {fit['chi_squared']/fit['degrees_of_freedom']:.2f},
so a single power law does not describe all means within their sampling errors
particularly well. Restricting the fit to Ny>=30 changes the estimate to
z={sensitivity[-1]['exponent']:.3f} ± {sensitivity[-1]['exponent_sem']:.3f}
(chi-square={sensitivity[-1]['chi_squared']:.3f} for
{sensitivity[-1]['degrees_of_freedom']} degrees of freedom). The strong size
decrease is robust, but the full-range exponent should not be treated as a
precise universal exponent; small-size corrections and finite observation
time remain possible. Statistical fit errors do not include this model/window
dependence. The original T=2Ny fit and this T=40 fit use correlated observations
from the same trajectories, not independent experiments.

The previous panel C used T=2Ny. Its images and data are preserved. All
other panels use the same data and styling as before: A/B/D remain the
Ny=30 full-measurement Campaigns 21–22 with endpoint T=60. Their means,
SEMs and spatial heatmap were checked to be exactly unchanged. Panel C
is a different slab-only protocol and must not be presented as a size
sweep of A/B/D. No 30x30 simulation has been performed.

Files: gap_vs_Ny_T40.pdf/png is the standalone plot;
purification_ny30_gap_T40_4x1.pdf/png is the revised four-panel plot;
caption.tex explicitly states both protocol and time differences.
CSV tables preserve all 700 sample gaps, means/SEMs and fit-window
sensitivity. analysis_manifest.json binds inputs and output hashes.
The manuscript and original figure assets have not been overwritten.
''')
    assert all(figure.sha(Path(p)) == h for p,h in original.items())
    outputs = sorted(p for p in OUT.iterdir() if p.name != 'analysis_manifest.json')
    figure.write_json(OUT/'analysis_manifest.json', dict(schema='purification_fixed_T40_gap_v1',
        T=40, normalization=80, Nx=20, sizes=list(campaign.NY_VALUES), samples_per_size=100,
        panel_c_campaign=campaign.REVISION, panel_c_meas_slab_only=True,
        estimator='min absolute modular energy per trajectory, divided by 80, then mean and ordinary SEM',
        fit_method='SEM-weighted log-space WLS, same as original panel C; no bootstrap or residual rescaling',
        fit=fit, fit_window_sensitivity=sensitivity, summary=summary, inventory=inventory,
        panels_abd_unchanged=True, original_assets_sha256=original,
        inputs=inputs+a1['records']+a3['records'],
        source_hashes={str(p):figure.sha(p) for p in [Path(__file__),HERE/'make_figure.py',BUNDLE/'plot_endpoint_lyapunov_gap.py',BUNDLE/'analyze_campaign.py']},
        outputs=[dict(path=str(p),sha256=figure.sha(p)) for p in outputs]))
    print(json.dumps(summary, indent=2)); print(OUT)


if __name__ == '__main__':
    main()
