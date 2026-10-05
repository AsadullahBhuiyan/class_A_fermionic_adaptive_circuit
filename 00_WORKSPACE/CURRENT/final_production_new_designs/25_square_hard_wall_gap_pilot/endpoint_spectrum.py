"""Endpoint-only active-slab occupations and finite-time Lyapunov half gap."""
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


def extract_endpoint(G, active_indices, cycles, device='cuda:0', progress=True):
    """One sample per eigvalsh; never retain eigenvectors or covariance history."""
    if np.asarray(G).dtype != np.complex128:
        raise TypeError('Checkpoint/endpoint covariance must be complex128')
    active = torch.as_tensor(active_indices, dtype=torch.long, device=device)
    spectra, residuals = [], []
    for sample in tqdm(G, desc='endpoint occupations', unit='sample', disable=not progress):
        full = torch.as_tensor(sample, dtype=torch.complex128, device=device)
        if not bool(torch.isfinite(full).all()):
            raise FloatingPointError('Nonfinite endpoint covariance')
        residual = float((full-full.mH).abs().max().item())
        if residual > 1e-8:
            raise FloatingPointError(f'Endpoint Hermiticity residual: {residual}')
        block = full.index_select(0, active).index_select(1, active)
        block = .25*(block+block.mH)
        block.diagonal().add_(.5)
        values = torch.linalg.eigvalsh(block)
        spectra.append(values.detach().cpu().numpy())
        residuals.append(residual)
        del full, block, values
    result = spectral_products(np.array(spectra), cycles)
    result['hermiticity_residual'] = np.array(residuals)
    return result
