"""All-origin endpoint entropy, intrinsic charge variance, and entropy contour.

Basis order is i=2*(y*Nx+x)+mu. Contours are stored as (sample,x,dy).
No function here evolves the circuit or averages independent trajectories.
"""
import numpy as np
import torch

CONTOUR = 'endpoint__contour_von_neumann_y0avg'
ENTROPY = 'endpoint__entropy_von_neumann'
VARIANCE = 'endpoint__charge_variance'
KEYS = (CONTOUR, ENTROPY, VARIANCE)


def periodic_indices(nx, ny, origins, ay, device):
    y = (torch.as_tensor(origins, device=device)[:, None] + torch.arange(ay, device=device)) % ny
    x = torch.arange(nx, device=device)
    mu = torch.arange(2, device=device)
    return (2 * (y[:, :, None, None] * nx + x[None, None, :, None])
            + mu[None, None, None, :]).reshape(len(origins), 2 * nx * ay)


def solve_pairs(frame, sample_ids, origins, *, nx, ny, ay, tolerance=1e-8):
    """One bounded matrix batch; all results remain on the frame's device."""
    indices = periodic_indices(nx, ny, origins, ay, frame.device)
    rows = frame[sample_ids[:, None], indices]
    restricted = rows @ rows.mH
    restricted = (restricted + restricted.mH) * 0.5
    occupations, vectors = torch.linalg.eigh(restricted)
    lo, hi = float(occupations.min()), float(occupations.max())
    if lo < -tolerance or hi > 1 + tolerance:
        raise FloatingPointError(f'Restricted occupations outside [0,1]: {lo}, {hi}')
    nu = occupations.clamp(0, 1)
    h = -torch.xlogy(nu, nu) - torch.xlogy(1 - nu, 1 - nu)
    contour = (vectors.abs().square() @ h.unsqueeze(-1)).squeeze(-1)
    contour = contour.reshape(len(origins), ay, nx, 2).sum(-1).transpose(1, 2).contiguous()
    return {CONTOUR: contour, ENTROPY: h.sum(-1), VARIANCE: (nu * (1 - nu)).sum(-1)}, lo, hi


@torch.inference_mode()
def averaged_width(frame, *, nx, ny, ay, matrix_batch, tolerance=1e-8, progress=None):
    if frame.dtype != torch.complex128 or frame.ndim != 3 or frame.shape[1] != 2 * nx * ny:
        raise ValueError('Endpoint requires complex128 native physical frames')
    if not 0 <= ay <= ny // 2 or matrix_batch < 1:
        raise ValueError('Invalid endpoint width or matrix batch')
    count = frame.shape[0]
    result = {CONTOUR: torch.zeros((count, nx, ay), dtype=torch.float64, device=frame.device),
              ENTROPY: torch.zeros(count, dtype=torch.float64, device=frame.device),
              VARIANCE: torch.zeros(count, dtype=torch.float64, device=frame.device)}
    lo, hi = 0., 1.
    if ay:
        samples = torch.arange(count, device=frame.device).repeat_interleave(ny)
        origins = torch.arange(ny, device=frame.device).repeat(count)
        for first in range(0, count * ny, matrix_batch):
            stop = min(first + matrix_batch, count * ny)
            ids = samples[first:stop]
            values, block_lo, block_hi = solve_pairs(frame, ids, origins[first:stop],
                nx=nx, ny=ny, ay=ay, tolerance=tolerance)
            for key in KEYS:
                result[key].index_add_(0, ids, values[key])
            lo, hi = min(lo, block_lo), max(hi, block_hi)
            if progress is not None:
                progress(stop - first)
        result = {key: value / ny for key, value in result.items()}
    return {key: value.cpu().numpy() for key, value in result.items()}, dict(occupation_min=lo, occupation_max=hi)


def empty_endpoint(samples, nx, ny):
    half = ny // 2
    return {CONTOUR: np.zeros((samples, half + 1, nx, half), dtype=np.float64),
            ENTROPY: np.zeros((samples, half + 1), dtype=np.float64),
            VARIANCE: np.zeros((samples, half + 1), dtype=np.float64)}


def validate_endpoint(values, *, nx, ny, samples, widths=None, closure_tolerance=2e-8):
    template = empty_endpoint(samples, nx, ny)
    if set(values) != set(template):
        raise ValueError('Endpoint keys differ from saved-data contract')
    widths = list(range(ny // 2 + 1)) if widths is None else list(widths)
    for key in KEYS:
        value = values[key]
        if value.shape != template[key].shape or value.dtype != np.float64:
            raise ValueError(f'Wrong shape/dtype: {key}')
        if not np.isfinite(value[:, widths]).all() or np.min(value[:, widths], initial=0) < -1e-10:
            raise FloatingPointError(f'Nonfinite or negative endpoint values: {key}')
        if not np.all(value[:, 0] == 0):
            raise ValueError('Width-zero values must be exactly zero')
    maximum = 0.
    for ay in widths:
        contour = values[CONTOUR][:, ay]
        if not np.all(contour[:, :, ay:] == 0):
            raise ValueError(f'Nonzero contour padding at Ay={ay}')
        error = abs(contour[:, :, :ay].sum((1, 2)) - values[ENTROPY][:, ay]).max(initial=0)
        maximum = max(maximum, float(error))
        if np.max(values[ENTROPY][:, ay], initial=0) > 2 * nx * ay * np.log(2) + closure_tolerance:
            raise FloatingPointError('Entropy exceeds subsystem bound')
        if np.max(values[VARIANCE][:, ay], initial=0) > nx * ay / 2 + closure_tolerance:
            raise FloatingPointError('Charge variance exceeds subsystem bound')
    if maximum > closure_tolerance:
        raise FloatingPointError(f'Contour closure error {maximum}')
    return maximum
