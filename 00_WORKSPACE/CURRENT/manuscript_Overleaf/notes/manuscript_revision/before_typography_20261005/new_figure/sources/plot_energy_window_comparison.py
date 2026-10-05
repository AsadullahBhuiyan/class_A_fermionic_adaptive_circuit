#!/usr/bin/env python3
"""Separate, unit-area energy histograms for the matched fixed-cut spectra.

Uses the portable occupation_inputs.npz already delivered with Figure 7.
Does not run dynamics or alter any numbered manuscript figure.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/entanglement_spectrum'
DATA = ROOT / 'data/energy_window_comparison'
STEM = ROOT / 'diagnostics/normalized_entanglement_energy_alpha1_1_vs_3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    figure7 = {p.name: digest(p) for p in ROOT.glob('Figure_07*') if p.is_file()}
    provenance = json.loads((SOURCE / 'occupation_provenance.json').read_text())
    input_path = SOURCE / 'occupation_inputs.npz'
    assert digest(input_path) == provenance['compact_input_sha256']
    with np.load(input_path, allow_pickle=False) as data:
        spectra = data['centered_occupations'].copy()
        np.testing.assert_array_equal(data['alpha_1'], [1, 3])
        np.testing.assert_array_equal(data['sample_ids'], np.arange(100))
        np.testing.assert_array_equal(data['subsystem_indices'], np.arange(640))
        for key, value in dict(Nx=20, Ny=32, Ay=16, cycle=64, origin=0).items():
            assert data[key].item() == value
    assert spectra.shape == (2, 100, 640) and np.isfinite(spectra).all()
    energy_limit = float(np.log(199))
    edges = np.linspace(-energy_limit, energy_limit, 102)
    histograms, densities, summary, mode_rows, trajectory_rows = [], [], [], [], []
    for alpha, centered in zip((1, 3), spectra):
        selected = np.abs(centered) <= .99
        samples, modes = np.nonzero(selected)
        retained = centered[selected]
        energy = np.log1p(-retained) - np.log1p(retained)
        assert np.isfinite(energy).all() and np.max(np.abs(energy)) <= energy_limit
        np.testing.assert_allclose(-np.tanh(energy/2), retained, rtol=0, atol=5e-16)
        counts, _ = np.histogram(energy, bins=edges)
        total = len(energy)
        assert counts.sum() == total and total > 0
        density = counts / (total * np.diff(edges))
        np.testing.assert_allclose(np.sum(density * np.diff(edges)), 1, rtol=0, atol=1e-14)
        by_sample = selected.sum(axis=1)
        central = np.bincount(samples[np.abs(energy) < 1], minlength=100)
        histograms.append(counts)
        densities.append(density)
        mode_rows.append(np.column_stack([np.full(total, alpha), samples, modes, retained, energy]))
        trajectory_rows.append(np.column_stack([np.full(100, alpha), np.arange(100), by_sample, central]))
        summary.append(dict(alpha_1=alpha, retained_modes=total,
                            retained_fraction_of_full_spectrum=total/64000,
                            mean_modes_per_cut=float(by_sample.mean()),
                            trajectory_SEM=float(by_sample.std(ddof=1)/10),
                            nonempty_trajectories=int(np.count_nonzero(by_sample)),
                            count_range=[int(by_sample.min()), int(by_sample.max())],
                            abs_energy_below_one=int(central.sum()),
                            trajectories_with_abs_energy_below_one=int(np.count_nonzero(central)),
                            conditional_fraction_abs_energy_below_one=float(central.sum()/total),
                            min_abs_energy=float(np.min(np.abs(energy))),
                            density_integral=float(np.sum(density*np.diff(edges)))))

    DATA.mkdir(parents=True, exist_ok=True)
    STEM.parent.mkdir(parents=True, exist_ok=True)
    os.environ['TEXINPUTS'] = str(SOURCE / 'latex_support') + os.pathsep + os.environ.get('TEXINPUTS', '')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['CMU Sans Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'text.usetex': True, 'pdf.fonttype': 42, 'savefig.dpi': 300,
        'axes.linewidth': .8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    })
    fig, ax = plt.subplots(figsize=(3.375, 2.9))
    for density, alpha, style, color in zip(densities, (1, 3), ('-', ':'), ('#1565c0', '#d73027')):
        ax.stairs(density, edges, fill=True, color=color, alpha=.08)
        ax.stairs(density, edges, color=color, linestyle=style, linewidth=1.1,
                  label=rf'$\alpha_1={alpha}$')
    ax.set(xlabel=r'Entanglement energy $\varepsilon$',
           ylabel=r'Conditional probability density $p_W(\varepsilon)$',
           xlim=(-energy_limit, energy_limit), ylim=(0, np.max(densities)*1.32))
    ax.tick_params(which='both', top=True, right=True)
    ax.text(.5, .97, r'$N_y=32$, $A_y=16$, cycle $64$',
            transform=ax.transAxes, ha='center', va='top')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, .89), ncol=2, frameon=False,
              columnspacing=1, handlelength=1.8)
    ax.text(.5, .72, r'$W:\ |\lambda|\leq0.99$'+'\n'+r'Unit area within $W$',
            transform=ax.transAxes, ha='center', va='top', linespacing=1.35)
    fig.tight_layout(pad=.85)
    bounds = ax.get_tightbbox(fig.canvas.get_renderer())
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    assert ax.get_xscale() == ax.get_yscale() == 'linear'
    fig.savefig(STEM.with_suffix('.pdf'))
    plt.close(fig)
    subprocess.run(['pdftoppm', '-r', '300', '-singlefile', '-png',
                    str(STEM.with_suffix('.pdf')), str(STEM)], check=True)

    np.savetxt(DATA / 'energy_histogram.csv', np.column_stack([
        edges[:-1], edges[1:], np.array(histograms).T, np.array(densities).T]), delimiter=',',
        header='energy_left,energy_right,alpha1_1_count,alpha1_3_count,alpha1_1_density,alpha1_3_density', comments='')
    np.savetxt(DATA / 'retained_modes.csv', np.concatenate(mode_rows), delimiter=',',
               fmt=['%d', '%d', '%d', '%.17g', '%.17g'],
               header='alpha_1,sample_id,mode_index,centered_occupation,entanglement_energy', comments='')
    np.savetxt(DATA / 'trajectory_counts.csv', np.concatenate(trajectory_rows), delimiter=',', fmt='%d',
               header='alpha_1,sample_id,retained_modes,abs_energy_below_one', comments='')
    assert figure7 == {p.name: digest(p) for p in ROOT.glob('Figure_07*') if p.is_file()}
    result = dict(all_checks_passed=True, input=str(input_path.relative_to(ROOT)), input_sha256=digest(input_path),
                  source_sha256=digest(Path(__file__)), bins=101, energy_range=[-energy_limit, energy_limit],
                  centered_occupation_window=[-.99, .99], normalization='pooled count / (total retained count * bin width), separately for each alpha',
                  sampling='100 independent trajectories per alpha, fixed origin y0=0; no origin pooling',
                  protocol=provenance['protocol'], axes='linear', figure_inches=[3.375, 2.9],
                  histogram_uncertainty='none; modes are correlated within each trajectory',
                  fits='none', no_dynamics_rerun=True, ensembles=summary,
                  figure_7_unchanged=figure7,
                  outputs={str(STEM.with_suffix(s).relative_to(ROOT)): digest(STEM.with_suffix(s)) for s in ('.pdf', '.png')})
    (DATA / 'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'ensembles': summary, 'outputs': result['outputs']}, indent=2))


if __name__ == '__main__':
    main()
