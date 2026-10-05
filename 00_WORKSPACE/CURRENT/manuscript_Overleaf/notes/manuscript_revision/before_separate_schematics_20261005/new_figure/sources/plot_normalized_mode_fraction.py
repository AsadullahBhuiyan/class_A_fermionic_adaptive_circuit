#!/usr/bin/env python3
"""Standalone Figure-7(c) variant normalized by subsystem dimension.

Average fractions over cut origins inside each trajectory, then trajectories.
Preserve the existing raw-count fit and divide its prediction by 2*Nx*Ay.
No fitting of a straight line to the fractions and no dynamics are performed.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/entanglement_spectrum'
DATA = ROOT / 'data/normalized_mode_fraction'
STEM = ROOT / 'diagnostics/normalized_mode_fraction_vs_log_chord'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    originals = {p.name: digest(p) for p in ROOT.glob('Figure_*.*') if p.suffix in ('.pdf', '.png')}
    provenance = json.loads((SOURCE / 'provenance.json').read_text())
    assert digest(SOURCE / 'inputs.npz') == provenance['compact_input_sha256']
    with np.load(SOURCE / 'inputs.npz', allow_pickle=False) as data:
        counts = data['counts_by_sample_width_origin'].copy()
        widths = data['widths'].copy()
        np.testing.assert_array_equal(widths, np.arange(1, 17))
        np.testing.assert_array_equal(data['origins'], np.arange(32))
        np.testing.assert_array_equal(data['sample_ids'], np.arange(100))
        for key, value in dict(Nx=20, Ny=32, Ay=16, cycle=64, L=.99).items():
            assert data[key].item() == value
    assert counts.shape == (100, 16, 32)
    dimensions = 2 * 20 * widths
    assert np.all((counts >= 0) & (counts <= dimensions[None, :, None]))
    per_origin_fraction = counts / dimensions[None, :, None]
    trajectory_fraction = per_origin_fraction.mean(axis=2)
    mean = trajectory_fraction.mean(axis=0)
    sem = trajectory_fraction.std(axis=0, ddof=1) / 10
    trajectory_count = counts.mean(axis=2)
    raw_mean = trajectory_count.mean(axis=0)
    raw_sem = trajectory_count.std(axis=0, ddof=1) / 10
    np.testing.assert_allclose(mean, raw_mean/dimensions, rtol=0, atol=1e-14)
    np.testing.assert_allclose(sem, raw_sem/dimensions, rtol=0, atol=1e-14)
    np.testing.assert_allclose(mean[-1], 29.811875/640, rtol=0, atol=1e-14)
    log_chord = np.log(32 / np.pi * np.sin(np.pi * widths / 32))
    fit_mask = widths >= 5
    design = np.column_stack([np.ones(fit_mask.sum()), log_chord[fit_mask]])
    trajectory_coefficients = trajectory_count[:, fit_mask] @ np.linalg.pinv(design).T
    coefficients = trajectory_coefficients.mean(axis=0)
    coefficient_sem = trajectory_coefficients.std(axis=0, ddof=1) / 10
    expected = provenance['expected_fit']
    for value, name in zip([*coefficients, *coefficient_sem], ['intercept', 'slope', 'intercept_sem', 'slope_sem']):
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=1e-12)
    smooth_width = np.linspace(1, 16, 500)
    smooth_x = np.log(32/np.pi * np.sin(np.pi*smooth_width/32))
    prediction = (coefficients[0]+coefficients[1]*smooth_x)/(40*smooth_width)

    DATA.mkdir(parents=True, exist_ok=True)
    STEM.parent.mkdir(parents=True, exist_ok=True)
    os.environ['TEXINPUTS'] = str(SOURCE/'latex_support') + os.pathsep + os.environ.get('TEXINPUTS', '')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['CMU Sans Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'text.usetex': True, 'pdf.fonttype': 42, 'savefig.dpi': 300,
        'axes.linewidth': .8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    })
    fig, ax = plt.subplots(figsize=(3.375, 2.9))
    ax.axvspan(log_chord[fit_mask].min(), log_chord.max(), color='0.93', linewidth=0, zorder=0)
    line, = ax.plot(smooth_x, prediction, color='black', linestyle='--', linewidth=.9,
                    label='Rescaled count fit', zorder=1)
    dots = ax.errorbar(log_chord, mean, yerr=sem, color='#1565c0', marker='o',
                       markerfacecolor='white', markeredgewidth=.8, linestyle='none',
                       markersize=3.5, capsize=2, elinewidth=.7, zorder=3,
                       label='Mean over samples and cuts')
    ax.set(xlabel=r'$\log d(A_y)$', ylabel=r'Mean fraction $\overline{N}_{0.99}/(2N_xA_y)$',
           ylim=(0, .75))
    ax.tick_params(which='both', top=True, right=True)
    ax.text(.96, .97, r'$N_x=20$, $N_y=32$, $\alpha_1=1$',
            transform=ax.transAxes, ha='right', va='top')
    ax.legend(handles=[dots, line], loc='upper right', bbox_to_anchor=(1, .89),
              frameon=False, handlelength=1.6)
    ax.text(.96, .62, r'$S=100$, all $32$ origins'+'\n'+r'$|\lambda|\leq0.99$',
            transform=ax.transAxes, ha='right', va='top', linespacing=1.35)
    fig.tight_layout(pad=.85)
    bounds = ax.get_tightbbox(fig.canvas.get_renderer())
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    fig.savefig(STEM.with_suffix('.pdf'))
    plt.close(fig)
    subprocess.run(['pdftoppm', '-r', '300', '-singlefile', '-png',
                    str(STEM.with_suffix('.pdf')), str(STEM)], check=True)
    np.savetxt(DATA/'fraction_by_width.csv', np.column_stack([
        widths, dimensions, log_chord, raw_mean, raw_sem, mean, sem]), delimiter=',',
        header='Ay,subsystem_modes,log_chord,mean_count,count_trajectory_SEM,mean_fraction,fraction_trajectory_SEM', comments='')
    np.savetxt(DATA/'rescaled_fit.csv', np.column_stack([smooth_width, smooth_x, prediction]),
               delimiter=',', header='Ay,log_chord,predicted_fraction', comments='')
    np.savez_compressed(DATA/'trajectory_fractions.npz', sample_ids=np.arange(100), widths=widths,
                        trajectory_fractions=trajectory_fraction)
    assert originals == {p.name: digest(p) for p in ROOT.glob('Figure_*.*') if p.suffix in ('.pdf', '.png')}
    receipt = dict(all_checks_passed=True, input='data/entanglement_spectrum/inputs.npz',
                   input_sha256=digest(SOURCE/'inputs.npz'), source_sha256=digest(Path(__file__)),
                   protocol=provenance['protocol'], alpha_1=1, geometry='all x, both orbitals, every translated y strip',
                   samples=100, origins=32, independent_unit='trajectory',
                   normalization='N_window/(2*Nx*Ay), with Nx=20; denominator includes all subsystem modes',
                   estimator='normalize each cut; average 32 origins per trajectory; average 100 trajectories',
                   uncertainty='sample standard deviation of 100 trajectory-level origin averages / sqrt(100)',
                   display_widths=[1,16], raw_count_fit_widths=[5,16], raw_count_fit_intercept=float(coefficients[0]),
                   raw_count_fit_slope=float(coefficients[1]), raw_count_fit_coefficient_SEM=coefficient_sem.tolist(),
                   fit='existing raw-count log-chord fit divided by 2*Nx*Ay; no fit to fraction',
                   half_strip_mean_fraction=float(mean[-1]), half_strip_fraction_SEM=float(sem[-1]),
                   figures_preserved=originals, no_dynamics_rerun=True, figure_inches=[3.375,2.9],
                   outputs={str(STEM.with_suffix(s).relative_to(ROOT)):digest(STEM.with_suffix(s)) for s in ('.pdf','.png')})
    (DATA/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['half_strip_mean_fraction','half_strip_fraction_SEM','outputs']},indent=2))


if __name__ == '__main__':
    main()
