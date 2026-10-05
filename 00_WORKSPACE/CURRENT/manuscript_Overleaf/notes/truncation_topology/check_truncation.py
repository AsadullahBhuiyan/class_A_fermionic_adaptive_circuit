"""Reproduce the deterministic OW-band comparison; no circuit simulation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'PROJECT_ADMIN/REPO_POLICY.md').exists())
SOURCE = ROOT / 'technical_report/analyze_ow_truncation.py'


def main():
    spec = importlib.util.spec_from_file_location('native_ow_analysis', SOURCE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    minus, plus = native.projector_fourier_coefficients(1.)
    rows = []
    for size in (201, 401):
        print(f'Checking alpha=1, w=1 on {size} x {size} momenta', flush=True)
        k = 2 * np.pi * np.arange(size) / size
        kx, ky = np.meshgrid(k, k, indexing='ij')
        target = native.band_projector(kx, ky, 1., -1)
        truncated = np.stack([
            native.truncated_spinor(kx, ky, minus, native.IDENTITY[:, j], 1)
            for j in (0, 1)], axis=-1)
        hermiticity = float(np.max(np.abs(truncated - truncated.conj().swapaxes(-1, -2))))
        error = float(np.max(np.abs(np.linalg.eigvalsh(truncated - target))))
        values, vectors = np.linalg.eigh(truncated)
        occupied = vectors[..., 1]
        overlap = np.einsum('...a,...ab,...b->...', occupied.conj(), target, occupied).real
        weights = [native.support_weight(coefficients, trial, 1)
                   for coefficients in (minus, plus) for trial in native.TRIAL_SPINORS]
        hamiltonian = np.zeros_like(target)
        for sign, coefficients in ((-1, minus), (1, plus)):
            for trial in native.TRIAL_SPINORS:
                vector = native.truncated_spinor(kx, ky, coefficients, trial, 1, normalize=True)
                hamiltonian += sign * vector[..., :, None] * vector.conj()[..., None, :]
        identity_error = float(np.max(np.abs(
            hamiltonian - (native.IDENTITY - 2 * truncated) / weights[0])))
        np.testing.assert_allclose(weights, weights[0], rtol=0, atol=1e-14)
        assert hermiticity < 1e-13 and identity_error < 1e-13
        assert np.all(values[..., 0] < .5) and np.all(values[..., 1] > .5)
        rows.append(dict(momentum_grid=size,
            sampled_max_operator_error=error,
            sampled_min_target_band_overlap=float(overlap.min()),
            sampled_min_absolute_frame_energy=float(np.min(np.abs(np.linalg.eigvalsh(hamiltonian)))),
            normalization_weights=weights,
            hermiticity_residual=hermiticity,
            quadratic_frame_identity_residual=identity_error,
            sampled_condition_delta_less_than_half=error < .5))
    print('Checking the single-cell w=0 limit and its interpolation crossing', flush=True)
    f0 = minus[0, 0]
    d0 = float((f0[1, 1] - f0[0, 0]).real)
    z0 = (1 + d0**2) / 4
    weights0 = [native.support_weight(c, trial, 0)
                for c in (minus, plus) for trial in native.TRIAL_SPINORS]
    h0 = np.zeros((2, 2), dtype=complex)
    for sign, coefficients in ((-1, minus), (1, plus)):
        for trial in native.TRIAL_SPINORS:
            vector = native.truncated_spinor(
                np.asarray(0.), np.asarray(0.), coefficients, trial, 0, normalize=True)
            h0 += sign * np.outer(vector, vector.conj())
    energy0, vectors0 = np.linalg.eigh(h0)
    r0 = np.outer(vectors0[:, 0], vectors0[:, 0].conj())
    target_gamma = native.band_projector(np.asarray(0.), np.asarray(0.), 1., -1)
    s_star = 1 / (1 + d0)
    crossing = (1 - s_star) * target_gamma + s_star * f0
    np.testing.assert_allclose(f0, np.diag([(1-d0)/2, (1+d0)/2]), atol=1e-14)
    np.testing.assert_allclose(weights0, z0, atol=1e-14)
    np.testing.assert_allclose(h0, (native.IDENTITY - 2*f0)/z0, atol=1e-14)
    np.testing.assert_allclose(r0, np.diag([0., 1.]), atol=1e-14)
    np.testing.assert_allclose(crossing, .5*native.IDENTITY, atol=1e-14)
    onsite = dict(window_half_width=0, d0=d0, normalization=z0,
        truncated_matrix_real=f0.real.tolist(), frame_energies=energy0.tolist(),
        negative_band_projector_real=r0.real.tolist(),
        target_overlap_at_gamma=float(np.trace(target_gamma @ r0).real),
        operator_error_at_gamma=float(np.max(np.abs(np.linalg.eigvalsh(f0-target_gamma)))),
        interpolation_crossing=s_star,
        crossing_residual=float(np.max(np.abs(crossing-.5*native.IDENTITY))),
        scope='One unit cell with two orbitals; constant auxiliary projector has Chern number zero')
    result = dict(alpha=1., window_half_width=1,
        window='square |r_x|, |r_y| <= 1',
        fourier_coefficient_grid=native.FOURIER_GRID,
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        numpy_version=np.__version__, results=rows, single_site_check=onsite,
        new_circuit_simulations=False, continuum_bound_certified=False,
        scope='Auxiliary Bloch Hamiltonian, not individual trajectory projectors',
        limitation='Sampled extrema and numerical Fourier coefficients; no certified quadrature or inter-grid bound')
    output = HERE / 'truncation_check.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
