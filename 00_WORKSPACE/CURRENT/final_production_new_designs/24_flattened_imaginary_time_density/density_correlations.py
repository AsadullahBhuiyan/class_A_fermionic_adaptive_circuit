"""Equilibrium density autocorrelations; no stochastic or time-stepping dynamics."""
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from classA_U1FGTN import classA_U1FGTN


def parent_blocks(config):
    """Existing spatial benchmark's B0 convention, retaining untwisted Bloch labels.

    Regulate representative OW columns at k+phi/Ny, then translate those columns
    using the periodic Bloch basis. This is an occupation regulator, not a claim
    that a flux-threaded finite Hamiltonian was constructed.
    """
    nx, ny = config['Nx'], config['Ny']
    model = classA_U1FGTN(Nx=nx, Ny=ny, DW=True, nshell=config['nshell'],
                         filling_frac=.5, alpha_1=config['alpha_1'],
                         alpha_2=config['alpha_2'], trial_orbitals='X',
                         dw_truncation=True)
    model.construct_OW_projectors(nshell=config['nshell'], DW=True,
                                 trial_orbitals='X', dw_truncation=True)
    if tuple(model.DW_loc) != (nx//4, 3*nx//4):
        raise ValueError(f'Unexpected wall locations: {model.DW_loc}')
    k = 2*np.pi*np.arange(ny)/ny
    transform = np.exp(-1j*(k[:, None]+config['occupation_twist']/ny)*np.arange(ny))/np.sqrt(ny)
    blocks = np.zeros((ny, 2*nx, 2*nx), dtype=np.complex128)
    for name, sign in [('WF_Ap', 1), ('WF_Bp', 1), ('WF_Am', -1), ('WF_Bm', -1)]:
        frame = np.asarray(getattr(model, name), dtype=np.complex128).reshape(ny, 2*nx, nx, ny)
        representative = np.einsum('ky,yar->kar', transform, frame[:, :, :, 0], optimize=True)
        blocks += sign*ny*np.einsum('kar,kbr->kab', representative, representative.conj(), optimize=True)
    error = float(np.max(abs(blocks-blocks.conj().transpose(0, 2, 1))))
    if error > 1e-10:
        raise ValueError(f'Non-Hermitian parent: {error}')
    return .5*(blocks+blocks.conj().transpose(0, 2, 1)), error


def eigensystem(blocks):
    energies, vectors = np.linalg.eigh(blocks)
    ny, q = energies.shape
    order = np.argsort(energies.ravel(), kind='stable')
    occupied = np.zeros(ny*q, dtype=bool)
    occupied[order[:ny*q//2]] = True
    occupied = occupied.reshape(ny, q)
    gap = float(energies.ravel()[order[ny*q//2]] - energies.ravel()[order[ny*q//2-1]])
    if gap < -1e-12:
        raise ValueError('Invalid half-filling gap')
    return energies, vectors, occupied, gap


def transition_weights(energies, vectors, occupied):
    """Rows label x; columns label all (occupied,empty) particle-hole pairs."""
    ny, q = energies.shape
    ki, ai = np.nonzero(occupied)
    kj, bj = np.nonzero(~occupied)
    vo = vectors[ki, :, ai].T  # (2Nx, number occupied), Bloch components only
    ve = vectors[kj, :, bj].T
    gaps = energies[kj, bj][None, :] - energies[ki, ai][:, None]
    if gaps.min() < -1e-12:
        raise ValueError('Negative particle-hole excitation energy')
    same_k = ki[:, None] == kj[None, :]
    local, column = [], []
    for x in range(q//2):
        overlap = vo[2*x:2*x+2].conj().T @ ve[2*x:2*x+2]
        weights = np.abs(overlap)**2
        local.append((weights/ny**2).ravel())
        column.append((weights*same_k/ny).ravel())
    # Total charge has no occupied-to-empty matrix elements.
    number_weight = float(np.sum(abs(vo.conj().T @ ve)**2 * same_k))
    return np.maximum(gaps, 0).ravel(), np.array(local), np.array(column), number_weight


def evaluate(gaps, weights, tau, chunk=8, progress=True):
    out = np.empty((len(tau), len(weights)), dtype=float)
    with tqdm(total=len(tau), desc='imaginary-time points', unit='tau', disable=not progress) as bar:
        for start in range(0, len(tau), chunk):
            times = tau[start:start+chunk]
            # All arguments are <=0, even at very long imaginary times.
            decay = np.exp(-times[:, None]*gaps[None, :])
            out[start:start+len(times)] = decay @ weights.T
            bar.update(len(times))
    return out


def covariance_checks(vectors, occupied):
    ny, q, _ = vectors.shape
    projector = (vectors*occupied[:, None, :]) @ vectors.conj().transpose(0, 2, 1)
    delta = np.fft.ifft(projector, axis=0)
    local_var, column_var, means, spatial, legacy = [], [], [], [], []
    for x in range(q//2):
        sl = slice(2*x, 2*x+2)
        p = delta[0, sl, sl]
        means.append(np.trace(p).real)
        local_var.append(np.trace(p-p@p).real)
        pk = projector[:, sl, sl]
        column_var.append(np.mean(np.trace(pk-pk@pk, axis1=1, axis2=2)).real)
        positive = .5*np.sum(abs(delta[:, sl, sl])**2, axis=(1, 2))
        connected = -2*positive
        connected[0] += np.trace(p).real
        spatial.append(connected)
        legacy.append(positive)
    diagnostics = {
        'projector_idempotency_max_abs': float(np.max(abs(projector@projector-projector))),
        'occupied_rank': int(occupied.sum()),
        'occupied_ranks_by_momentum': occupied.sum(axis=1).tolist(),
    }
    return dict(mean_cell_density=np.array(means), mean_column_charge=ny*np.array(means),
                equal_time_local_variance=np.array(local_var),
                equal_time_column_variance_per_Ny=np.array(column_var),
                spatial_connected=np.array(spatial), spatial_legacy_positive=np.array(legacy)), diagnostics


def normalized(values, tolerance):
    valid = values[0] > tolerance
    out = np.full_like(values, np.nan)
    out[:, valid] = values[:, valid]/values[0, valid]
    return out, valid


def calculate(config, progress=True):
    import time
    started = time.perf_counter()
    blocks, hermitian_error = parent_blocks(config)
    constructed = time.perf_counter()
    energies, vectors, occupied, gap = eigensystem(blocks)
    if not np.all(occupied.sum(1) == config['Nx']):
        raise ValueError('Regulated filling differs from spatial benchmark: not Nx per momentum')
    print(f'[parent] {len(blocks)} blocks; {occupied.sum()} occupied modes; gap={gap:.6g}', flush=True)
    gaps, local_w, column_w, number_weight = transition_weights(energies, vectors, occupied)
    prepared = time.perf_counter()
    tau = np.r_[0., np.geomspace(config['tau_min'], config['tau_max'], config['tau_points'])]
    combined = evaluate(gaps, np.concatenate((local_w, column_w)), tau,
                        config['tau_chunk'], progress)
    local, column = np.split(combined, 2, axis=1)
    static, diagnostics = covariance_checks(vectors, occupied)
    for values, target in [(local, static['equal_time_local_variance']),
                           (column, static['equal_time_column_variance_per_Ny'])]:
        np.testing.assert_allclose(values[0], target, atol=1e-10, rtol=1e-9)
        if not np.isfinite(values).all() or values.min() < 0 or np.max(np.diff(values, axis=0)) > 1e-12:
            raise ValueError('Autocorrelation positivity/monotonicity failed')
    if number_weight > 1e-20 or diagnostics['projector_idempotency_max_abs'] > 1e-10:
        raise ValueError('Projector/number conservation check failed')
    local_n, local_valid = normalized(local, config['normalization_tolerance'])
    column_n, column_valid = normalized(column, config['normalization_tolerance'])
    diagnostics.update(half_filling_gap=gap, parent_hermiticity_max_abs=hermitian_error,
                       total_number_transition_weight=number_weight,
                       minimum_transition_energy=float(gaps.min()),
                       minimum_transition_local_weight=local_w[:, np.argmin(gaps)].tolist(),
                       minimum_transition_column_weight=column_w[:, np.argmin(gaps)].tolist(),
                       parent_construction_seconds=constructed-started,
                       eigensystem_and_weights_seconds=prepared-constructed,
                       correlators_and_validation_seconds=time.perf_counter()-prepared,
                       computation_seconds=time.perf_counter()-started,
                       imaginary_time_units='inverse unrescaled signed OW-parent energy; not circuit cycles')
    return dict(tau=tau, x=np.arange(config['Nx']), energies=energies, eigenvectors=vectors,
                occupied=occupied, parent_blocks=blocks, local=local, column=column,
                local_normalized=local_n, column_normalized=column_n,
                local_normalization_valid=local_valid, column_normalization_valid=column_valid,
                **static), diagnostics
