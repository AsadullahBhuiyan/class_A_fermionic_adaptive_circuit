"""Sample-resolved endpoint gaps and one gap mode, with explicit saturation flags."""
import numpy as np
import torch
from tqdm.auto import tqdm

CAP_TOLERANCE = 1e-9  # in centered a, NOT in occupation nu
TIE_TOLERANCE = 1e-12
PLACEHOLDER = 100.0


def gap_fields(rates):
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] == 0 or np.isnan(rates).any():
        raise ValueError('Expected nonempty sample-by-mode rates without NaNs')
    raw = np.min(abs(rates), axis=1)
    infinite = np.isposinf(raw)
    return dict(gap_raw=raw, gap_value=np.where(infinite, PLACEHOLDER, raw),
                gap_is_infinite=infinite, gap_mode_valid=~infinite)


def spectral_products(centered, cycles):
    a = np.asarray(centered, dtype=np.float64)
    if cycles <= 0 or a.ndim != 2 or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError('Need finite sample-by-mode centered spectra and positive T')
    excess = np.maximum(np.max(abs(a), axis=1)-1, 0)
    if np.any(excess > CAP_TOLERANCE):
        raise FloatingPointError(f'Centered spectral bounds exceeded: {excess.max()}')
    if np.any(np.diff(a, axis=1) < 0):
        raise ValueError('Spectra must be in ascending centered-eigenvalue order')
    low, high = a <= -1+CAP_TOLERANCE, a >= 1-CAP_TOLERANCE
    cap = low | high
    rates = np.empty_like(a)
    rates[low], rates[high] = np.inf, -np.inf
    rates[~cap] = -np.arctanh(a[~cap])/cycles
    fields = gap_fields(rates)
    ties = np.zeros_like(cap)
    index = np.full(len(a), -1, dtype=np.int64)
    for i in range(len(a)):
        if fields['gap_mode_valid'][i]:
            ties[i] = np.isfinite(rates[i]) & (abs(abs(rates[i])-fields['gap_raw'][i]) <= TIE_TOLERANCE)
            index[i] = np.flatnonzero(ties[i])[0]
    return dict(centered_spectrum_raw=a, occupation_spectrum_raw=(1+a)/2,
                lyapunov_rates=rates, cap_mask=cap, cap_low_count=low.sum(1),
                cap_high_count=high.sum(1), finite_mode_count=(~cap).sum(1),
                spectral_bound_excess=excess, gap_mode_index=index,
                gap_tie_mask=ties, gap_tie_count=ties.sum(1), **fields)


def extract_endpoint(G, active_indices, cycles, device='cuda:0', progress=True):
    if np.asarray(G).dtype != np.complex128:
        raise TypeError('Endpoint covariance must be complex128')
    active = torch.as_tensor(active_indices, dtype=torch.long, device=device)
    rows = []
    for sample in tqdm(G, desc='endpoint spectra / gap modes', unit='sample', disable=not progress):
        full = torch.as_tensor(sample, dtype=torch.complex128, device=device)
        if not bool(torch.isfinite(full).all()):
            raise FloatingPointError('Nonfinite endpoint covariance')
        herm = float((full-full.mH).abs().max().item())
        if herm > 1e-8:
            raise FloatingPointError(f'Hermiticity residual {herm}')
        block = full.index_select(0, active).index_select(1, active)
        block = (block+block.mH)/2
        values, vectors = torch.linalg.eigh(block)
        # Canonicalize returned eigenpairs, including numerical ordering within
        # near-degenerate cap clusters. Never sort eigenvalues without columns.
        order = torch.argsort(values, stable=True)
        values, vectors = values[order], vectors[:, order]
        p = spectral_products(values.detach().cpu().numpy()[None], cycles)
        mode = np.zeros(len(active_indices), dtype=np.complex128)
        occupation, rate, residual = 0., 0., 0.
        eig_error = float(torch.linalg.vector_norm(block@vectors-vectors*values, dim=0).max().item())
        if eig_error > 1e-8:
            raise FloatingPointError(f'Eigen-equation residual {eig_error}')
        if p['gap_mode_valid'][0]:
            j = int(p['gap_mode_index'][0])
            v = vectors[:,j]
            mode = v.detach().cpu().numpy().copy()
            pivot = int(np.argmax(abs(mode)))
            mode *= np.conj(mode[pivot])/abs(mode[pivot])
            occupation = p['occupation_spectrum_raw'][0,j]
            rate = p['lyapunov_rates'][0,j]
            residual = float(torch.linalg.vector_norm(block@v-values[j]*v).item())
        p.update(gap_mode_vector=mode[None], gap_mode_occupation=np.array([occupation]),
                 gap_mode_rate=np.array([rate]), gap_mode_residual=np.array([residual]),
                 eigensolver_residual=np.array([eig_error]), hermiticity_residual=np.array([herm]))
        rows.append(p)
    result = {k: np.concatenate([p[k] for p in rows], axis=0) for k in rows[0]}
    validate_products(result, cycles)
    return result


def validate_products(p, cycles):
    expected = spectral_products(p['centered_spectrum_raw'], cycles)
    for k,v in expected.items():
        np.testing.assert_array_equal(p[k], v)
    vectors = p['gap_mode_vector']
    if vectors.shape != expected['centered_spectrum_raw'].shape or vectors.dtype != np.complex128 or not np.isfinite(vectors).all():
        raise ValueError('Invalid gap-mode vector shape/dtype')
    for i, valid in enumerate(expected['gap_mode_valid']):
        if valid:
            v = vectors[i]
            np.testing.assert_allclose(np.vdot(v,v), 1, atol=1e-10, rtol=0)
            pivot = v[np.argmax(abs(v))]
            if abs(pivot.imag) > 1e-12 or pivot.real < 0:
                raise ValueError('Gap mode is not phase fixed')
            j = int(expected['gap_mode_index'][i])
            np.testing.assert_equal(p['gap_mode_occupation'][i], expected['occupation_spectrum_raw'][i,j])
            np.testing.assert_equal(p['gap_mode_rate'][i], expected['lyapunov_rates'][i,j])
        elif np.any(vectors[i]) or p['gap_mode_occupation'][i] != 0 or p['gap_mode_rate'][i] != 0:
            raise ValueError('Invalid mode must have zero padding')
    for k in ('gap_mode_residual','eigensolver_residual','hermiticity_residual'):
        v = p[k]
        if v.shape != (len(vectors),) or not np.isfinite(v).all() or np.any(v < 0) or np.any(v > 1e-8):
            raise ValueError(f'Invalid residual {k}')
