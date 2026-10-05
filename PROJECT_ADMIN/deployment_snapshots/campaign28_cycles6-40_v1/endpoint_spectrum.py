"""Full-system cycle occupations and finite-time Lyapunov half gap."""
import numpy as np
import torch
from tqdm.auto import tqdm

CAP_TOLERANCE = 1e-9


def spectral_products(raw, cycles, tolerance=CAP_TOLERANCE):
    raw = np.asarray(raw, dtype=np.float64)
    if cycles <= 0 or raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Need finite sample-by-mode occupations and positive cycles')
    excess = np.maximum(np.maximum(-raw.min(1), raw.max(1)-1), 0)
    if np.any(excess > tolerance):
        raise FloatingPointError(f'Occupation bounds exceeded: min={raw.min():.17g}, max={raw.max():.17g}, tolerance={tolerance}')
    empty, full = raw <= tolerance, raw >= 1-tolerance
    capped = raw.copy()
    capped[empty], capped[full] = 0., 1.
    interior = ~(empty|full)
    epsilon = np.full(raw.shape, np.nan)
    epsilon[empty], epsilon[full] = np.inf, -np.inf
    epsilon[interior] = np.log1p(-raw[interior])-np.log(raw[interior])
    modular_gap = np.min(abs(epsilon), axis=1)
    return dict(occupation_spectrum_raw=raw, occupation_spectrum=capped,
                cap_mask=empty|full, modular_energies=epsilon, lyapunov_rates=epsilon/(2*cycles),
                modular_gap=modular_gap, lyapunov_gap=modular_gap/(2*cycles),
                finite_gap=np.isfinite(modular_gap), occupation_bound_excess=excess)


def validate_finite_modes(products):
    """Check the padded, sample-resolved finite-mode result contract."""
    spectra = products['occupation_spectrum_raw']
    count = products['finite_mode_count']
    indices = products['finite_mode_indices']
    vectors = products['finite_mode_vectors']
    if (count.shape != (spectra.shape[0],) or count.dtype.kind not in 'iu' or
        indices.dtype.kind not in 'iu' or vectors.dtype != np.complex128 or
        vectors.ndim != 3 or vectors.shape[:2] != spectra.shape or
        indices.shape != (spectra.shape[0], vectors.shape[2]) or not np.isfinite(vectors).all()):
        raise ValueError('Invalid finite-mode array shape/dtype')
    for i, n in enumerate(count):
        wanted = np.flatnonzero(~products['cap_mask'][i])
        if n != len(wanted) or n > vectors.shape[2] or not np.array_equal(indices[i,:n], wanted):
            raise ValueError('Finite-mode count or spectrum indices do not match caps')
        if not np.all(indices[i,n:] == -1) or np.any(vectors[i,:,n:] != 0):
            raise ValueError('Finite-mode padding must be -1 indices and zero vectors')
        np.testing.assert_allclose(np.sum(abs(vectors[i,:,:n])**2, axis=0), 1., atol=1e-8, rtol=0)
    for key in ('finite_mode_eigen_residual', 'finite_mode_orthogonality_residual'):
        residual = products[key]
        if residual.shape != count.shape or not np.isfinite(residual).all() or np.any(residual < 0) or np.any(residual > 1e-8):
            raise ValueError(f'Invalid {key}')


def extract_endpoint(G, active_indices, cycles, device='cuda:0', progress=True):
    """One sample per eigh; retain every uncapped eigenvector on the active basis."""
    if np.asarray(G).dtype != np.complex128:
        raise TypeError('Checkpoint/endpoint covariance must be complex128')
    active = torch.as_tensor(active_indices, dtype=torch.long, device=device)
    spectra, residuals, modes, mode_indices, eigen_errors, orth_errors = [], [], [], [], [], []
    for sample in tqdm(G, desc=f'cycle {cycles} occupations', unit='sample', disable=not progress):
        full = torch.as_tensor(sample, dtype=torch.complex128, device=device)
        if not bool(torch.isfinite(full).all()):
            raise FloatingPointError('Nonfinite endpoint covariance')
        residual = float((full-full.mH).abs().max().item())
        if residual > 1e-8:
            raise FloatingPointError(f'Endpoint Hermiticity residual: {residual}')
        block = full.index_select(0, active).index_select(1, active)
        block = .25*(block+block.mH)
        block.diagonal().add_(.5)
        values, vectors = torch.linalg.eigh(block)
        # Validate bounds before applying the same cap rule used by gap extraction.
        one = spectral_products(values.detach().cpu().numpy()[None], cycles)
        idx = np.flatnonzero(~one['cap_mask'][0])
        selected_indices = torch.as_tensor(idx, device=device)
        selected = vectors[:, selected_indices]
        if len(idx):
            eigen_error = float(torch.linalg.vector_norm(block@selected-selected*values[selected_indices], dim=0).max().item())
            orth_error = float((selected.mH@selected-torch.eye(len(idx), device=device, dtype=selected.dtype)).abs().max().item())
        else:
            eigen_error = orth_error = 0.
        eigen_errors.append(eigen_error); orth_errors.append(orth_error)
        modes.append(selected.detach().cpu().numpy())
        mode_indices.append(idx)
        spectra.append(values.detach().cpu().numpy())
        residuals.append(residual)
        del full, block, values, vectors, selected
    result = spectral_products(np.array(spectra), cycles)
    result['hermiticity_residual'] = np.array(residuals)
    count = np.array([len(idx) for idx in mode_indices], dtype=np.int64)
    maximum = int(count.max(initial=0))
    padded_vectors = np.zeros((len(G), len(active_indices), maximum), dtype=np.complex128)
    padded_indices = np.full((len(G), maximum), -1, dtype=np.int64)
    for i, n in enumerate(count):
        padded_vectors[i,:,:n] = modes[i]
        padded_indices[i,:n] = mode_indices[i]
    result.update(finite_mode_count=count, finite_mode_indices=padded_indices,
                  finite_mode_vectors=padded_vectors,
                  finite_mode_eigen_residual=np.array(eigen_errors),
                  finite_mode_orthogonality_residual=np.array(orth_errors))
    validate_finite_modes(result)
    return result

