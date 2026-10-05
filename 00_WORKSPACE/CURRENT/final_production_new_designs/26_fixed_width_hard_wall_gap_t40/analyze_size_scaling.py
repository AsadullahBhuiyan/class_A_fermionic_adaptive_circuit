"""Exploratory fixed-T circumference scaling; sample SEM, no bootstrap.

Run the verified base analysis first. Fit ensemble means in linear gap space,
using their ordinary sampling SEM as absolute errors, not residual scatter.
No simulation or change to the immutable endpoint data is performed.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

from analyze_campaign import analyze, write_csv
from endpoint_spectrum import spectral_products
from run_campaign import default_config, identity, result_paths, sha, tasks


def power_law(x, amplitude, exponent):
    return amplitude * (np.asarray(x) / 32.) ** (-exponent)


def fit_models(x, y, sem):
    """Weighted linear-gap fits with absolute, sample-SEM uncertainty."""
    x, y, sem = [np.asarray(a, dtype=float) for a in (x, y, sem)]
    if (x.shape != y.shape or x.shape != sem.shape or len(x) < 3 or
        not all(np.isfinite(a).all() for a in (x, y, sem)) or
        np.any(x <= 0) or np.any(y <= 0) or np.any(sem <= 0)):
        raise ValueError('Need at least three positive finite size/mean/SEM values')
    candidates = {
        'constant': (lambda x, a: a + 0*x, ['constant'], [y.mean()]),
        'inverse_size': (lambda x, a: a*32/x, ['amplitude_at_Ny32'], [y.mean()]),
        'power_law': (power_law, ['amplitude_at_Ny32', 'z'], [y.mean(), 1.]),
        'offset_inverse_size': (lambda x, b, a: b+a*32/x,
                                ['intercept_at_inverse_size_zero', 'amplitude'], [0., y.mean()]),
    }
    jacobians = {
        'constant': lambda x, a: np.ones((len(x), 1)),
        'inverse_size': lambda x, a: (32/x)[:, None],
        'power_law': lambda x, a, z: np.column_stack(((x/32)**(-z),
                                     -a*(x/32)**(-z)*np.log(x/32))),
        'offset_inverse_size': lambda x, b, a: np.column_stack((np.ones(len(x)), 32/x)),
    }
    fits = {}
    for name, (function, labels, initial) in candidates.items():
        parameters, covariance = curve_fit(function, x, y, p0=initial,
                                          sigma=sem, absolute_sigma=True,
                                          jac=jacobians[name], maxfev=10000)
        prediction = function(x, *parameters)
        chi_square = float(np.sum(((y-prediction)/sem)**2))
        dof = len(x) - len(parameters)
        fits[name] = dict(parameters=dict(zip(labels, map(float, parameters))),
                          parameter_sem=dict(zip(labels, map(float, np.sqrt(np.diag(covariance))))),
                          covariance=covariance.tolist(), chi_square=chi_square, dof=dof,
                          reduced_chi_square=chi_square/dof,
                          approximate_chi_square_p=float(chi2.sf(chi_square, dof)),
                          predictions=prediction.tolist(), sizes=x.tolist())
    return fits


def endpoint_contrast(y, sem):
    y, sem = np.asarray(y), np.asarray(sem)
    error = float(np.hypot(sem[0], sem[-1]))
    difference = float(y[0]-y[-1])
    return dict(difference=difference, sem=error, combined_SEMs=difference/error,
                fractional_decrease=float(1-y[-1]/y[0]))


def main(output, destination):
    output, destination = Path(output), Path(destination)
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    analyze(output, destination)  # all 16 source-bound result pairs must pass
    base = json.loads((destination/'analysis_manifest.json').read_text())
    rows = base['summary']
    x = np.array([r['Ny'] for r in rows])
    y = np.array([r['mean_lyapunov_gap'] for r in rows], dtype=float)
    sem = np.array([r['sem_lyapunov_gap'] for r in rows], dtype=float)
    fits = fit_models(x, y, sem)
    sensitivity = []
    for minimum in (20, 24, 28, 32):
        keep = x >= minimum
        f = fit_models(x[keep], y[keep], sem[keep])['power_law']
        sensitivity.append(dict(Ny_min=minimum, Ny_max=48, sizes=int(keep.sum()),
                                z=f['parameters']['z'], z_sem=f['parameter_sem']['z'],
                                chi_square=f['chi_square'], dof=f['dof']))

    # Recompute every spectrum/gap and document that pure-mode caps are irrelevant
    # to the selected minimum; never fit or average a reconstructed mean state.
    diagnostic_rows = []
    config = default_config()
    for task in reversed(tasks(config)):
        ident = identity(task, config)
        for start in (0, 5):
            path, _ = result_paths(output, task, start)
            with np.load(path, allow_pickle=False) as z:
                assert json.loads(str(z['configuration_json'])) == config
                assert json.loads(str(z['source_hashes_json'])) == ident['source_hashes']
                recomputed = spectral_products(z['occupation_spectrum_raw'], task.cycles)
                for key, value in recomputed.items():
                    np.testing.assert_allclose(z[key], value, rtol=1e-13, atol=1e-15)
                for i, sample in enumerate(z['sample_indices']):
                    mode = int(np.argmin(abs(z['modular_energies'][i])))
                    diagnostic_rows.append(dict(Ny=task.Ny, sample_index=int(sample),
                        modular_gap=float(z['modular_gap'][i]), lyapunov_gap=float(z['lyapunov_gap'][i]),
                        minimizing_mode=mode, minimizing_occupation=float(z['occupation_spectrum_raw'][i, mode]),
                        finite_modes=int(z['finite_mode_count'][i]),
                        occupation_bound_excess=float(z['occupation_bound_excess'][i]),
                        hermiticity_residual=float(z['hermiticity_residual'][i])))
    write_csv(destination/'sample_spectral_diagnostics.csv', diagnostic_rows)
    write_csv(destination/'fit_window_sensitivity.csv', sensitivity)

    p = fits['power_law']
    a, exponent = p['parameters']['amplitude_at_Ny32'], p['parameters']['z']
    exp_sem = p['parameter_sem']['z']
    dense_x = np.linspace(x[0], x[-1], 300)
    # Existing base analysis configures the repository's font/tick grammar.
    for name, log_axes, show_samples in [('gap_vs_Ny_powerlaw', False, False),
                                         ('gap_vs_Ny_loglog', True, False),
                                         ('gap_with_individual_samples', False, True)]:
        fig, ax = plt.subplots(figsize=(3.375, 2.65), layout='constrained')
        if show_samples:
            for ny in x:
                vals = [r['lyapunov_gap'] for r in diagnostic_rows if r['Ny'] == ny]
                ax.scatter(ny+np.linspace(-.6, .6, len(vals)), vals, s=9, color='.65',
                           alpha=.65, linewidths=0, zorder=1)
        ax.errorbar(x, y, yerr=sem, fmt='o', color='#1f77b4', mfc='white', ms=4,
                    capsize=2, lw=1, zorder=3, label=r'mean $\pm$ SEM')
        ax.plot(dense_x, power_law(dense_x, a, exponent), '--', color='.2', lw=1.1,
                label=rf'$A(N_y/32)^{{-z}}$, $z={exponent:.2f}\pm{exp_sem:.2f}$')
        ax.plot(dense_x, fits['inverse_size']['parameters']['amplitude_at_Ny32']*32/dense_x,
                ':', color='.5', lw=1, label=r'$B/N_y$ fit')
        ax.set(xlabel=r'circumference $N_y$', ylabel=r'$\overline{\Delta}_{T=40}$',
               xlim=(18.5,49.5), xticks=x)
        if log_axes:
            ax.set_xscale('log'); ax.set_yscale('log')
            from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
            ax.xaxis.set_major_locator(FixedLocator(x)); ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(ScalarFormatter())
        else:
            ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=7, loc='upper right')
        for extension in ('pdf', 'png'):
            fig.savefig(destination/f'{name}.{extension}', dpi=300)
        plt.close(fig)

    contrast = endpoint_contrast(y, sem)
    checks = dict(samples=len(diagnostic_rows),
                  max_occupation_bound_excess=max(r['occupation_bound_excess'] for r in diagnostic_rows),
                  max_hermiticity_residual=max(r['hermiticity_residual'] for r in diagnostic_rows),
                  minimizing_occupation_range=[min(r['minimizing_occupation'] for r in diagnostic_rows),
                                               max(r['minimizing_occupation'] for r in diagnostic_rows)])
    table = '\n'.join(f"| {r['Ny']} | {r['mean_modular_gap']:.4f} ± {r['sem_modular_gap']:.4f} | "
                      f"{r['mean_lyapunov_gap']:.5f} ± {r['sem_lyapunov_gap']:.5f} |" for r in rows)
    sensitivity_table = '\n'.join(f"| {r['Ny_min']}–48 | {r['z']:.2f} ± {r['z_sem']:.2f} |" for r in sensitivity)
    offset = fits['offset_inverse_size']
    note = f'''# Campaign 26: fixed-width, fixed-time gap scaling

## Data and estimator

All 16 result/completion pairs were revalidated against SHA-256, byte count,
task, seed, configuration, source identity and sample coverage. There are 10
independent trajectories at each Ny=20,24,28,32,36,40,44,48, with Nx=20 and
T=40 for every trajectory. Hard walls, maxmix initialization, Born-conditioned
exterior followed by slab-only measurements, perfect correction, raster_y,
alpha1=1, alpha2=30, nshell=1, complex128; no covariance clipping.

For each trajectory, epsilon_j=log[(1-nu_j)/nu_j], g_mod=min_j|epsilon_j|,
and Delta=g_mod/(2T)=g_mod/80. Compute the minimum **before** averaging samples.
This is neither the gap of an averaged spectrum nor the spectrum of an averaged
covariance. It is the active-slab finite-time Lyapunov half gap, not a
half-system entanglement-spectrum gap. The full symmetric gap convention would
double every value without changing the size exponent.

All errors on the means below are sample SD/sqrt(10), with no bootstrap.

| Ny | Raw modular gap ± SEM | Finite-time half gap ± SEM |
|---:|---:|---:|
{table}

## Size trend and exploratory fits

The endpoint contrast Delta(20)-Delta(48) is {contrast['difference']:.6f}
± {contrast['sem']:.6f}, combining independent SEMs in quadrature
({contrast['combined_SEMs']:.2f} combined SEMs). The central decrease is
{100*contrast['fractional_decrease']:.1f}%. It is not pointwise monotonic.
Since 2T=80 is constant, the raw modular gap falls by precisely the same
fraction. The trend therefore cannot be caused solely by a size-dependent
1/(2T) denominator.

Fit the eight ensemble means in **linear gap space**, minimizing
sum_Ny [(mean Delta - A*(Ny/32)^(-z))/SEM]^2. No logarithmic data fit or
sample-wise power-law fitting is used. The result is z={exponent:.4f}
± {exp_sem:.4f}, A={a:.6f} ± {p['parameter_sem']['amplitude_at_Ny32']:.6f}.
Parameter errors use the local weighted-fit covariance with absolute sampling
SEMs, without rescaling by residual chi-square. These are approximate
linearized one-standard-error uncertainties, not bootstrap intervals.

Power law: chi-square={p['chi_square']:.3f} for {p['dof']} degrees of freedom.
Fixed inverse-size model: chi-square={fits['inverse_size']['chi_square']:.3f}
for {fits['inverse_size']['dof']} degrees of freedom. A constant gap gives
chi-square={fits['constant']['chi_square']:.3f} for {fits['constant']['dof']}
degrees of freedom. Thus this range favors a decreasing gap and is compatible
with 1/Ny; a precise universal exponent is not established. The chi-square
probabilities in the JSON are only approximate because SEMs are themselves
estimated from ten samples and sample gaps need not be Gaussian.

The alternative Delta=b+a*(32/Ny) gives b=
{offset['parameters']['intercept_at_inverse_size_zero']:.6f} ±
{offset['parameter_sem']['intercept_at_inverse_size_zero']:.6f}.
This diagnostic fit leaves b unconstrained; a negative central b is not a
physical negative gap. Zero is compatible, but finite positive intercepts
are not excluded by these data. No thermodynamic extrapolation is adopted.

| Power-law fit window | z ± propagated fit SEM |
|---:|---:|
{sensitivity_table}

## Scope and cautions

This is a fixed-T=40, fixed-Nx=20 finite-time trend. Only endpoint data were
saved, so time convergence cannot be established. Increasing Ny does not
increase wall separation here. This is not a two-dimensional thermodynamic
limit or a demonstrated infinite-time Lyapunov gap closure.

Campaign 25 used square Nx=Ny systems at T=10 with independent samples.
Those data are neither pooled nor used in this fit. Differences from that
pilot confound observation time with geometry except at Nx=Ny=20. This
result also does not prove that the earlier variable-T scaling was artificial.

All 80 gaps are finite. Largest occupation-bound excess:
{checks['max_occupation_bound_excess']:.3g}; largest saved covariance Hermiticity
residual: {checks['max_hermiticity_residual']:.3g}. Occupations selecting the
minimum range from {checks['minimizing_occupation_range'][0]:.5f} to
{checks['minimizing_occupation_range'][1]:.5f}, far from the 1e-9 pure-mode caps.
Every saved modular energy, cap, rate and gap was independently recomputed.

## Outputs

- `gap_vs_Ny_powerlaw.pdf/png`: primary linear-axis plot with mean ± SEM,
  exploratory power law, and fixed inverse-size fit. No extrapolation.
- `gap_vs_Ny_loglog.pdf/png`: the same fits and sampling errors on log axes.
- `gap_with_individual_samples.pdf/png`: all 80 individual gaps; horizontal
  offsets only separate points and are not physical size changes.
- `lyapunov_gap_vs_Ny.pdf/png`, `modular_gap_vs_Ny.pdf/png`: data-only plots.
- `gap_summary.csv`, `sample_gaps.csv`, `sample_spectral_diagnostics.csv`,
  `fit_window_sensitivity.csv`: means, SEMs, samples and checks.
- `size_scaling_manifest.json`: fits, covariances, definitions, hashes, provenance.

Reproduce with `python analyze_size_scaling.py --output-root <downloaded campaign>
--analysis-root <analysis_outputs>`. The original data, Drive deployment and
manuscript remain unchanged.

Suggested caption: Finite-time Lyapunov half gap versus circumference at fixed
Nx=20 and T=40 physical cycles. Hard-wall, slab-only adaptive dynamics begin
from a maximally mixed state with the canonical Born-conditioned exterior.
Points are the mean of ten trajectory-wise minimum absolute modular energies
divided by 2T; bars are ordinary trajectory SEM. The dashed curve fits
A(Ny/32)^(-z) over Ny=20–48 using inverse-SEM-squared weights, with
z={exponent:.2f}±{exp_sem:.2f}; the dotted curve is the fixed 1/Ny fit.
The fit is descriptive of this finite-time window, not an infinite-time
extrapolation.
'''
    (destination/'README.md').write_text(note)
    products = sorted(p for p in destination.iterdir() if p.suffix in ('.pdf', '.png', '.csv') or p.name == 'README.md')
    manifest = dict(sampling_revision=config['sampling_revision'], config=config,
        estimator=base['estimator'], uncertainty=base['uncertainty'],
        fit_method='SEM-weighted least squares in linear gap space; absolute_sigma=True; no residual rescaling',
        interpretation='Exploratory circumference scaling at fixed time, not established infinite-time gap closure',
        fits=fits, fit_window_sensitivity=sensitivity, endpoint_contrast=contrast, numerical_checks=checks,
        inputs=base['inputs'], source_sha256=sha(Path(__file__)),
        outputs=[dict(path=p.name, sha256=sha(p)) for p in products])
    (destination/'size_scaling_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(fits=fits, endpoint_contrast=contrast, checks=checks), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--analysis-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.output_root, args.analysis_root)
