"""Spatial weights of saved dominant right eigenvectors at alpha_1=1 and 3."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ny', type=int, default=100)
    args = parser.parse_args()
    root = args.root.resolve()
    records, inputs, arrays = [], {}, {}
    for alpha, index in [(1, 0), (3, 20)]:
        folder = root / f'Ny{args.ny:03d}_a{index:02d}_{alpha:.1f}'
        path, receipt = folder / 'spectrum.npz', folder / 'completion.json'
        completion = json.loads(receipt.read_text())
        assert completion['status'] == 'complete'
        assert completion['result_filename'] == path.name
        assert completion['result_bytes'] == path.stat().st_size
        assert completion['result_sha256'] == sha(path)
        inputs.update({str(p): sha(p) for p in (path, receipt)})
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(data['config_json'].item())
            diagnostics = json.loads(data['diagnostics_json'].item())
            nx, ny = config['Nx'], config['Ny']
            assert (nx, ny, config['alpha_1']) == (20, args.ny, alpha)
            assert config == completion['config']
            vector = data['dominant_eigenvector'].copy()
            eigenvalue = data['dominant_eigenvalue'].item()
            radius = float(data['spectral_radius'])
            np.testing.assert_allclose(np.linalg.norm(vector), 1, atol=1e-12)
            np.testing.assert_allclose(abs(eigenvalue), radius, atol=1e-14)
            np.testing.assert_allclose(np.max(abs(data['eigenvalues'])), radius, atol=1e-14)
            # Canonical flattening i=mu+2*x+2*Nx*y; map is [y,x].
            weight = np.sum(abs(vector.reshape(ny, nx, 2))**2, axis=2)
            direct = np.zeros((ny, nx))
            for i, value in enumerate(vector):
                direct[i // (2*nx), (i // 2) % nx] += abs(value)**2
            np.testing.assert_allclose(weight, direct, atol=1e-16)
            np.testing.assert_allclose(weight.sum(), 1, atol=1e-12)
            left, right = config['walls']
            expected = np.flatnonzero(((np.arange(vector.size)//2) % nx >= left) &
                                      ((np.arange(vector.size)//2) % nx <= right))
            np.testing.assert_array_equal(data['interior_indices'], expected)
            near = np.isclose(abs(data['eigenvalues']), radius, atol=1e-10, rtol=0)
        arrays[f'weight_alpha{alpha}'] = weight
        arrays[f'x_marginal_alpha{alpha}'] = weight.sum(axis=0)
        arrays[f'y_marginal_alpha{alpha}'] = weight.sum(axis=1)
        arrays[f'right_vector_alpha{alpha}'] = vector
        records.append(dict(alpha_1=alpha, Nx=nx, Ny=ny, walls=[left, right],
            config=config, eigenvalue=[eigenvalue.real, eigenvalue.imag], rho_A=radius,
            g_C=1-radius**2, dominant_sector=diagnostics['dominant_sector'],
            eigenpair_residual=diagnostics['dominant_residual'],
            left_right_overlap=diagnostics['dominant_left_right_overlap'],
            eigenvalues_within_1e_minus_10_of_dominant_modulus=int(near.sum()),
            interior_weight=float(weight[:,left:right+1].sum()),
            boundary_site_weight=float(weight[:,[left,right]].sum()),
            peak_y_x=list(map(int,np.unravel_index(weight.argmax(),weight.shape))),
            max_site_weight=float(weight.max())))
    out = root / 'slow_mode_profiles' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.65), sharex=True, sharey=True,
                             layout='constrained')
    vmax = max(r['max_site_weight'] for r in records)
    for ax, row, letter in zip(axes, records, ['a','b']):
        weight = arrays[f"weight_alpha{row['alpha_1']}"]
        im = ax.imshow(weight, origin='lower', extent=(-.5,nx-.5,-.5,ny-.5),
                       vmin=0, vmax=vmax, cmap='magma', interpolation='nearest', aspect='auto')
        for wall in row['walls']:
            ax.axvline(wall, color='cyan', ls='--', lw=.7, alpha=.9)
        ax.set(xlabel=r'$x$', title=rf"$\alpha_1={row['alpha_1']}$", xticks=[0,5,10,15,19])
        ax.tick_params(top=True, right=True)
        ax.text(-.06,1.04,f'({letter})',transform=ax.transAxes,ha='right',va='bottom')
    axes[0].set_ylabel(r'$y$')
    fig.colorbar(im, ax=axes, fraction=.045, pad=.025,
                 label=r'$w(x,y)=\sum_\mu|v_{x,y,\mu}|^2$')
    for extension in ('png','pdf'):
        fig.savefig(out/f'slow_mode_alpha1_alpha3.{extension}',dpi=300)
    plt.close(fig)
    np.savez_compressed(out/'spatial_weights.npz',**arrays)
    (out/'diagnostics.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'caption.txt').write_text(
        f'Saved slow covariance mode for Nx=20, Ny={ny}; (a) alpha_1=1, '
        '(b) alpha_1=3. The normalized dominant RIGHT eigenvector Av=lambda*v gives '
        'delta G=v*v^dagger with multiplier |lambda|^2=rho(A)^2. Color shows its '
        'orbital-summed diagonal weight, with sum over all sites equal to one. '
        'Both panels use the same linear color scale; the site axes are displayed '
        'with unequal aspect ratio. Cyan dashed lines mark inclusive wall sites '
        'x=5 and 15 (interior x=5,...,15; interfaces between x=4/5 and 15/16). '
        'Canonical indexing i=mu+2*x+2*Nx*y. This is one saved right eigenvector, '
        'not a sum over a degenerate slow subspace, not the left eigenvector, '
        'and not a biorthogonal density. Near-degenerate eigenvalue counts and '
        'left/right overlaps are recorded in diagnostics.json. The profile is '
        'at the raster-cycle boundary; spatial asymmetry can depend on the ordered '
        'schedule. Hard-wall support truncation, all slabs active, alpha_2=30, '
        'nshell=1, periodic, zero twist, X orbitals, raster-y Ap/Am/Bp/Bm, '
        'complex128, perfect correction with measurement dephasing. '
        'No sampled trajectories, initial state, time evolution, fit, or error bars.\n')
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        input_sha256=inputs,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),
        indent=2)+'\n')
    print(json.dumps(dict(output=str(out),summary=[{k:v for k,v in r.items() if k!='config'}
                                                  for r in records]),indent=2))


if __name__=='__main__':
    main()
