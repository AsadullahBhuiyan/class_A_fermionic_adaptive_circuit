#!/usr/bin/env python3
"""Locate every cycle-60 covariance mode using saved Figure 3(b) trajectories."""
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
BUNDLE = REPO / '00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    provenance = json.loads((BUNDLE / 'data/purification/occupation_source_provenance.json').read_text())
    inputs = [row for row in provenance['inputs'] if '/alpha1_1/' in row['path']]
    assert len(inputs) == 20
    nx, ny, n = 20, 30, 1200
    occupations = np.empty((100, n))
    mean_x = np.empty((100, n))
    std_x = np.empty((100, n))
    probabilities_x = np.empty((100, nx, n))
    seen = set()
    validation = []
    x = np.arange(nx, dtype=float)
    print('Saved full-system maximally mixed ensemble: alpha1=1, Nx=20, Ny=30, cycle=60, S=100.')
    print('Diagonalize C=(G_final+I)/2; basis order is (y,x,orbital). Four BLAS threads; no simulation.')
    started = time.monotonic()
    with threadpool_limits(limits=4):
        progress = tqdm(total=100, desc='Cycle-60 eigenmodes', unit='trajectory')
        for item in inputs:
            path = Path(item['path'])
            assert sha256(path) == item['sha256'], path
            with np.load(path, allow_pickle=False) as z:
                assert int(z['cycles'][-1]) == 60 and int(z['alpha_1']) == 1
                assert str(z['centered_covariance_convention']) == 'G=2C-I'
                matrices = z['G_final']
                ids = z['sample_indices']
                spectra = z['occupation_spectrum'][:, -1, :]
                saved_vectors = z['slow_mode_vector']
                saved_indices = z['slow_mode_spectrum_index']
                saved_densities = z['slow_mode_density']
            for local, sample in enumerate(ids):
                sample = int(sample)
                assert sample not in seen
                seen.add(sample)
                g = matrices[local]
                hermiticity = float(np.max(np.abs(g - g.conj().T)))
                assert hermiticity < 1e-9 and np.isfinite(g).all()
                c = (g + g.conj().T) / 4 + np.eye(n) / 2
                nu, u = eigh(c, driver='evd', check_finite=False)
                assert nu.min() > -1e-9 and nu.max() < 1 + 1e-9
                spectral_error = float(np.max(np.abs(nu - spectra[local])))
                assert spectral_error < 1e-10
                px = (np.abs(u) ** 2).reshape(ny, nx, 2, n).sum(axis=(0, 2))
                normalization_error = float(np.max(np.abs(px.sum(axis=0) - 1)))
                completeness_error = float(np.max(np.abs(px.sum(axis=1) - 2 * ny)))
                assert normalization_error < 1e-11 and completeness_error < 1e-10
                mx = x @ px
                sx = np.sqrt(np.maximum(x ** 2 @ px - mx ** 2, 0))
                # Residuals for distributed ranks and every substantially mixed mode.
                check_indices = np.unique(np.r_[np.linspace(0, n - 1, 20, dtype=int),
                    np.flatnonzero((nu > .005) & (nu < .995))])
                residual = float(np.max(np.linalg.norm(
                    c @ u[:, check_indices] - u[:, check_indices] * nu[check_indices], axis=0)))
                assert residual < 1e-10
                j = int(saved_indices[local])
                slow_overlap = float(abs(np.vdot(saved_vectors[local], u[:, j])) ** 2)
                density_error = float(np.max(abs(px[:, j] - saved_densities[local].sum(axis=1))))
                assert slow_overlap > 1 - 1e-7 and density_error < 1e-7
                occupations[sample], mean_x[sample], std_x[sample] = nu, mx, sx
                probabilities_x[sample] = px
                validation.append(dict(sample=sample, hermiticity=hermiticity,
                    spectral_error=spectral_error, normalization_error=normalization_error,
                    completeness_error=completeness_error, eigenvector_residual=residual,
                    saved_slow_mode_overlap=slow_overlap, saved_slow_mode_density_error=density_error))
                progress.update(1)
        progress.close()
    assert seen == set(range(100))
    # Histogram is a trace of the binned spectral projector against each column.
    # Unlike individual near-degenerate eigenvectors, it is invariant to rotations
    # within any eigenspace lying wholly inside an occupation bin.
    edges = np.linspace(0, 1, 41)
    spatial_weight = np.zeros((nx, 40))
    centers_weight = np.zeros((nx, 40))
    for sample in range(100):
        bins = np.minimum(np.maximum(np.searchsorted(edges, occupations[sample], side='right') - 1, 0), 39)
        for col in range(nx):
            np.add.at(spatial_weight[col], bins, probabilities_x[sample, col] / 100)
        center_bins = np.minimum(np.maximum(np.floor(mean_x[sample] + .5).astype(int), 0), nx - 1)
        np.add.at(centers_weight, (center_bins, bins), 1 / 100)
    np.testing.assert_allclose(spatial_weight.sum(axis=1), 60, atol=1e-10)
    np.testing.assert_allclose(spatial_weight.sum(), 1200, atol=1e-9)
    mixed = (occupations > .005) & (occupations < .995)
    near_half = (occupations > .1) & (occupations < .9)
    mixed_profile = (probabilities_x * mixed[:, None, :]).sum(axis=2).mean(axis=0)
    wall_columns = np.array([5, 6, 14, 15])
    wall_weight = probabilities_x[:, wall_columns, :].sum(axis=1)
    nearest_wall_distance = np.minimum(abs(mean_x - 5), abs(mean_x - 15))
    summary = dict(alpha_1=1, Nx=nx, Ny=ny, cycle=60, samples=100,
        modes_per_trajectory=n, total_modes=100 * n,
        initialization='Full system maximally mixed; full-measurement protocol matching Figure 3(b)',
        source_covariance='G_final=2C-I', coordinate_order='y,x,orbital',
        mode_position='sum_(y,orbital,x) x |u_(y,x,orbital),j|^2',
        spatial_histogram='Mean sum_j p_j(x) indicator(nu_j in bin); 40 uniform occupation bins',
        mixed_window='0.005 < nu < 0.995; same as abs(2nu-1)<0.99',
        mixed_modes_per_trajectory=dict(mean=float(mixed.sum(1).mean()),
            min=int(mixed.sum(1).min()), max=int(mixed.sum(1).max()), total=int(mixed.sum())),
        near_half_window='0.1 < nu < 0.9', near_half_modes_total=int(near_half.sum()),
        mixed_mode_mean_wall_window_weight=float(wall_weight[mixed].mean()),
        mixed_mode_weight_fraction_in_wall_windows=float(mixed_profile[wall_columns].sum() / mixed_profile.sum()),
        mixed_mode_fraction_with_center_within_one_site_of_wall=float((nearest_wall_distance[mixed] < 1).mean()),
        mixed_mode_mean_x_range=[float(mean_x[mixed].min()),float(mean_x[mixed].max())],
        mixed_mode_median_std_x=float(np.median(std_x[mixed])),
        mixed_mode_mean_x_histogram=np.histogram(mean_x[mixed],bins=np.arange(-.5,20.5))[0].tolist(),
        mixed_mode_mean_spatial_weight=mixed_profile.tolist(),
        caveats=['Linear mean x uses the torus coordinate cut x=0..19; a mean position can hide split support.',
                 'Near-pure numerically degenerate modes have basis-dependent individual centers.',
                 'The binned spatial spectral weight avoids rank averaging and is invariant within degenerate eigenspaces.'],
        validation_maxima={key:max(row[key] for row in validation) for key in
            ('hermiticity','spectral_error','normalization_error','completeness_error','eigenvector_residual','saved_slow_mode_density_error')},
        minimum_saved_slow_mode_overlap=min(row['saved_slow_mode_overlap'] for row in validation),
        source_inputs=inputs, runtime_seconds=time.monotonic()-started)
    np.savez_compressed(OUT/'mode_positions.npz', sample_ids=np.arange(100),
        occupations=occupations, mean_x=mean_x, std_x=std_x, p_x=probabilities_x,
        wall_weight=wall_weight, occupation_bin_edges=edges, spatial_spectral_weight=spatial_weight,
        center_histogram=centers_weight, mixed_mode_spatial_profile=mixed_profile)
    with (OUT/'mode_positions.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample','sorted_rank_zero_based','occupation','mean_x','std_x','wall_window_weight'])
        for sample in range(100):
            writer.writerows((sample,j,occupations[sample,j],mean_x[sample,j],std_x[sample,j],wall_weight[sample,j]) for j in range(n))
    summary['outputs_sha256'] = {p.name:sha256(p) for p in (OUT/'mode_positions.npz',OUT/'mode_positions.csv')}
    summary['analysis_sha256'] = sha256(__file__)
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_inputs','mixed_mode_mean_spatial_weight')},indent=2))


if __name__ == '__main__':
    main()
