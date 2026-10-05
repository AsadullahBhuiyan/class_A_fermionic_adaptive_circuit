#!/usr/bin/env python3
"""Sample-averaged slot-07 hard-wall finite-mode probability densities."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from analyze_profiles import sha256

HERE = Path(__file__).resolve().parent
OUT = HERE / 'hard_wall_mean_heatmaps'


def sample_density(vectors, indices, ny):
    if not vectors.shape[1]: raise ValueError('Empty finite-mode subspace')
    full = np.zeros((40 * ny, vectors.shape[1]))
    full[indices] = abs(vectors)**2
    per_mode = full.reshape(ny, 20, 2, -1).sum(axis=2)
    np.testing.assert_allclose(per_mode.sum(axis=(0, 1)), 1, atol=1e-10)
    # Each sample has equal total weight even when its finite-mode count varies.
    return per_mode.mean(axis=2)


def main():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    manifest = json.loads((HERE / 'analysis_manifest.json').read_text())
    expected = {d['result_file']: d['result_sha256'] for d in manifest['outputs']}
    OUT.mkdir(exist_ok=True)
    means, summaries, inputs = {}, [], []
    for ny in [20, 30, 40]:
        paths = sorted((HERE / 'endpoint_modes/hard' / f'Ny{ny:03d}').glob('sample_*.npz'))
        if len(paths) != 100: raise ValueError(f'Expected 100 hard-wall samples for Ny={ny}')
        densities, counts, ids = [], [], []
        for path in paths:
            key = str(path.relative_to(HERE))
            digest = sha256(path)
            if digest != expected[key]: raise ValueError(f'Input checksum mismatch: {path}')
            with np.load(path, allow_pickle=False) as z:
                if str(z['construction']) != 'hard' or int(z['Ny']) != ny or int(z['cycles']) != 4*ny:
                    raise ValueError('Scientific identity mismatch')
                if not bool(z['finite_subspace_separated_from_caps']): raise ValueError('Unresolved subspace')
                v = z['finite_eigenvectors']
                density = sample_density(v, z['active_indices'], ny)
                np.testing.assert_allclose(density.sum(axis=0), z['finite_subspace_mean_x_profile'], atol=1e-12)
                densities.append(density); counts.append(v.shape[1]); ids.append(int(z['sample_index']))
            inputs.append(dict(file=key, sha256=digest))
        np.testing.assert_array_equal(sorted(ids), np.arange(100))
        samples = np.stack(densities)
        mean, sem = samples.mean(axis=0), samples.std(axis=0, ddof=1)/10
        np.testing.assert_allclose(samples.sum(axis=(1,2)), 1, atol=1e-10)
        np.testing.assert_allclose(mean.sum(), 1, atol=1e-10)
        np.testing.assert_array_equal(mean[:, :5], 0)
        np.testing.assert_array_equal(mean[:, 16:], 0)
        np.savez_compressed(OUT / f'hard_Ny{ny:03d}_mean_density.npz', mean_density=mean,
            sem_density=sem, sample_densities=samples, sample_indices=ids, finite_mode_counts=counts,
            Nx=20, Ny=ny, cycles=4*ny, samples=100, cap_tolerance=1e-9)
        means[ny] = mean
        summaries.append(dict(Ny=ny, samples=100, cycles=4*ny, finite_modes_total=sum(counts),
            finite_modes_min=min(counts), finite_modes_max=max(counts), probability_sum=float(mean.sum()),
            wall_weight=float(mean[:, [4,5,6,14,15,16]].sum())))
    os.environ['TEXINPUTS'] = str(HERE.parent / 'hard_wall_tangent_gap_analysis/latex_support') + os.pathsep + os.environ.get('TEXINPUTS', '')
    mpl.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['CMU Sans Serif'], 'font.size':8,
        'text.usetex':True, 'text.latex.preamble':r'\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}',
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.4), layout='constrained')
    vmax = max(float(m.max()) for m in means.values())
    for ax, (ny, mean), label in zip(axes, means.items(), ['(a)','(b)','(c)']):
        im = ax.imshow(mean, origin='lower', interpolation='nearest', aspect='auto',
            extent=(-.5,19.5,-.5,ny-.5), cmap='magma', vmin=0, vmax=vmax)
        ax.set(xlabel='$x$', ylabel='$y$', title=rf'$N_y={ny},\quad S=100$', xticks=[0,5,10,15,19])
        ax.text(-.07,1.035,label,transform=ax.transAxes,ha='right')
        # Do not obscure the brightest columns with wall-location overlays.
    fig.colorbar(im, ax=axes, label=r'Sample-averaged probability $\overline{p(x,y)}$', shrink=.92)
    pdf = OUT / 'slot07_hard_wall_sample_averaged_density.pdf'
    fig.savefig(pdf); plt.close(fig)
    subprocess.run(['pdftoppm','-png','-r','300','-singlefile',str(pdf),str(pdf.with_suffix(''))],check=True)
    (OUT / 'manifest.json').write_text(json.dumps(dict(script_sha256=sha256(__file__),
        estimator='Equal mean of orbital-summed resolved-mode densities within each sample, then equal mean of 100 samples',
        color_scale='Common linear scale, unrescaled probability per unit cell', cases=summaries, inputs=inputs,
        outputs=[dict(file=p.name,sha256=sha256(p)) for p in sorted(OUT.glob('*.npz'))]),indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__': main()
