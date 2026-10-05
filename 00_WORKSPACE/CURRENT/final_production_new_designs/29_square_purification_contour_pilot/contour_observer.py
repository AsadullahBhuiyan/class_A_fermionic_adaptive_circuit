"""Full-system mixed-state entropy, not subsystem entanglement entropy."""
import numpy as np
import torch


def observe_covariance(G, L, cycle, tolerance=1e-9, entropy_eps=1e-12):
    if G.shape != (1, 2*L*L, 2*L*L) or G.dtype != torch.complex128:
        raise ValueError('Expected one full complex128 covariance')
    with torch.inference_mode():
        if not bool(torch.isfinite(G).all()):
            raise FloatingPointError('Nonfinite covariance')
        herm = float((G-G.mH).abs().max())
        if herm > 1e-9:
            raise FloatingPointError(f'Hermiticity residual {herm}')
        C = (G[0]+G[0].mH)/4 + torch.eye(2*L*L,device=G.device,dtype=G.dtype)/2
        nu, vectors = torch.linalg.eigh(C)
        excess = max(0., -float(nu.min()), float(nu.max())-1)
        if excess > tolerance:
            # Independent recheck, never clip/project the evolving covariance.
            from scipy.linalg import eigh
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                values, vec = eigh(C.cpu().numpy(), driver='evr')
            excess = max(0., -float(values.min()), float(values.max())-1)
            if not np.isfinite(values).all() or excess > tolerance:
                raise FloatingPointError(f'CPU-confirmed occupation excess {excess:.9e}')
            print('[observer] using independently rechecked CPU eigenpairs', flush=True)
            nu = torch.as_tensor(values,device=G.device)
            vectors = torch.as_tensor(vec,device=G.device)
        if not bool(torch.isfinite(nu).all()):
            raise FloatingPointError('Nonfinite occupations')
        safe = nu.clamp(entropy_eps,1-entropy_eps)
        weights = -safe*safe.log()-(1-safe)*torch.log1p(-safe)
        cell = (vectors.abs().square()@weights).reshape(L,L,2).sum(-1).T
        entropy = float(weights.sum())
        closure = abs(float(cell.sum())-entropy)
        if closure > 1e-10*max(1.,entropy):
            raise FloatingPointError('Entropy contour closure failed')
        raw = nu.cpu().numpy()
        interior = (raw>1e-9)&(raw<1-1e-9)
        gap = float(np.min(abs(np.log1p(-raw[interior])-np.log(raw[interior])))) if interior.any() else np.nan
        return dict(occupation_spectrum=raw, entropy_contour=cell.cpu().numpy(),
            total_entropy=np.array(entropy), global_charge=np.array(float(nu.sum())),
            modular_gap=np.array(gap), lyapunov_gap=np.array(gap/(2*cycle) if cycle else np.nan),
            finite_mode_count=np.array(interior.sum()), hermiticity_residual=np.array(herm),
            occupation_bound_excess=np.array(excess), entropy_closure_error=np.array(closure))


def allocate(L, cycles):
    shapes = dict(occupation_spectrum=(2*L*L,),entropy_contour=(L,L))
    scalars = ('total_entropy','global_charge','modular_gap','lyapunov_gap',
               'finite_mode_count','hermiticity_residual','occupation_bound_excess','entropy_closure_error')
    return {key:np.full((cycles+1,)+shape,np.nan) for key,shape in
            (list(shapes.items())+[(key,()) for key in scalars])}
