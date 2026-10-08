"""Clean, strictly periodic OW-parent benchmarks. No circuit simulation.

P denotes the spectral projector in the canonical stored OW-column convention;
the manuscript two-point function is G=P.T. All lengths count unit cells.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit, xlogy
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'PROJECT_ADMIN/REPO_POLICY.md').exists())
MANUSCRIPT = HERE.parents[1]
sys.path.insert(0, str(REPO / 'src/fgtn'))
from classA_U1FGTN import classA_U1FGTN

ALPHAS = (1., 1.25, 1.5, 1.7, 1.8, 1.85, 1.9, 1.925, 1.95, 1.975,
          2., 2.025, 2.05, 2.075, 2.1, 2.15, 2.2, 2.3, 2.5, 2.75, 3.)
GAP_SIZES = (20, 24, 30, 36, 44, 56, 60)
CORR_SIZES = (24, 28, 32, 40, 50, 60)
ENTROPY_SIZES = (30, 35, 40, 45, 50, 55, 60)
WALL_SIZES = (30, 35, 40, 45, 55)
CHANNELS = (('WF_Ap', 1), ('WF_Bp', 1), ('WF_Am', -1), ('WF_Bm', -1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def npz_write(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def csv_write(path, rows):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_identity():
    files = [Path(__file__), REPO/'src/fgtn/classA_U1FGTN.py', REPO/'src/fgtn/occupied_frame.py']
    return {str(p.relative_to(REPO)): sha(p) for p in files}


def interfaces(nx):
    return nx//2-max(1, nx//4), nx//2+max(1, nx//4)


def configuration(nx, ny, alpha):
    return dict(Nx=nx, Ny=ny, alpha_1=float(alpha), alpha_2=30., nshell=1,
                trial_orbitals='X', dw_truncation=True, interfaces=list(interfaces(nx)),
                periodic_x=True, periodic_y=True, twist_x=0., twist_y=0., filling=.5,
                dtype='complex128', degeneracy_tolerance_relative=1e-12,
                construction='classA_U1FGTN.construct_OW_projectors',
                parent='WF_Ap WF_Ap^dag + WF_Bp WF_Bp^dag - WF_Am WF_Am^dag - WF_Bm WF_Bm^dag',
                correlation_convention='G = P.T, P = V_occ V_occ^dag',
                revision='clean_periodic_parent_v1')


def case_id(nx, ny, alpha):
    return f'nx{nx}_ny{ny}_a{alpha:g}'.replace('.', 'p')


def task_table():
    cases = {(20, n, a) for n in (20, 24, 28) for a in ALPHAS}
    cases |= {(20, n, 1.) for n in GAP_SIZES+CORR_SIZES+ENTROPY_SIZES}
    cases |= {(20, n, 3.) for n in (30, 32, 60)}
    cases |= {(n, n, 1.) for n in (20, 30, 40)}
    return sorted(cases)


def build_parent(nx, ny, alpha, *, dense_check=False, occupation_regulator=0.):
    """Use canonical OW construction, then Fourier transform exact translations.

    occupation_regulator is ONLY for the saved prior-protocol reconciliation.
    Production tasks always pass zero. It duplicates the legacy representative-
    column regulator, not a physical boundary-flux construction.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        model = classA_U1FGTN(Nx=nx, Ny=ny, DW=True, nshell=1, filling_frac=.5,
                             alpha_1=alpha, alpha_2=30, trial_orbitals='X',
                             dw_truncation=True, twist_x=0., twist_y=0.)
        model.construct_OW_projectors(nshell=1, DW=True, trial_orbitals='X', dw_truncation=True)
    assert tuple(model.DW_loc) == interfaces(nx)
    q = 2*nx
    blocks = np.zeros((ny, q, q), complex)
    full = np.zeros((ny*q, ny*q), complex) if dense_check else None
    translation = normalization = support = 0.
    region = np.zeros(nx, bool)
    left, right = interfaces(nx)
    region[left:right+1] = True
    crossing = np.repeat(region, 2)[:, None] != region[None, :]
    phase = np.exp(-1j*(2*np.pi*np.arange(ny)[:, None]+occupation_regulator)
                   * np.arange(ny)[None, :]/ny)
    for name, sign in CHANNELS:
        w = getattr(model, name).reshape(ny, q, nx, ny)
        representative = w[:, :, :, 0]
        # Check every translated center, without relying only on a Fourier identity.
        for y in range(ny):
            translation = max(translation, float(abs(w[:, :, :, y]-np.roll(representative, y, axis=0)).max()))
        normalization = max(normalization, float(abs(np.sum(abs(w)**2, axis=(0,1))-1).max()))
        support = max(support, float(abs(representative[:, crossing]).max(initial=0)))
        wk = np.einsum('ky,yar->kar', phase, representative, optimize=True)
        blocks += sign*np.einsum('kar,kbr->kab', wk, wk.conj(), optimize=True)
        if dense_check:
            flat = w.reshape(ny*q, nx*ny)
            full += sign*(flat @ flat.conj().T)
    hermiticity = float(abs(blocks-blocks.conj().transpose(0,2,1)).max())
    diagnostics = dict(ow_translation_residual=translation, ow_normalization_residual=normalization,
                       hard_wall_support_residual=support, parent_hermiticity_residual=hermiticity)
    assert translation < 2e-12 and normalization < 2e-12 and support == 0 and hermiticity < 2e-12
    blocks = (blocks+blocks.conj().transpose(0,2,1))/2
    if dense_check:
        reconstructed = dense_from_delta(np.fft.ifft(blocks, axis=0), ny)
        diagnostics['dense_parent_residual'] = float(abs(full-reconstructed).max())
        assert diagnostics['dense_parent_residual'] < 2e-12
    return blocks, diagnostics


def groups(values, tolerance):
    start = 0
    for j in range(1, len(values)+1):
        if j == len(values) or values[j]-values[start] > tolerance:
            yield slice(start, j)
            start = j


def site_basis(v):
    """Deterministic Gram--Schmidt of projected coordinate basis vectors."""
    p = v @ v.conj().T
    out = []
    for i in range(len(p)):
        candidate = p[:, i].copy()
        for _ in range(2):
            for u in out:
                candidate -= u*np.vdot(u, candidate)
        norm = np.linalg.norm(candidate)
        if norm > 1e-9:
            out.append(candidate/norm)
        if len(out) == v.shape[1]:
            break
    assert len(out) == v.shape[1]
    return np.column_stack(out)


def diagonalize(blocks):
    ny, q, _ = blocks.shape
    raw, vectors = np.linalg.eigh(blocks)
    energies = raw.copy()
    tol = 1e-12*max(1., float(abs(raw).max()))
    x = np.repeat(np.arange(q//2), 2)
    for k in range(ny):
        for g in groups(raw[k], tol):
            if g.stop-g.start == 1:
                continue
            v = vectors[k, :, g]
            xx = v.conj().T @ (x[:, None]*v)
            xv, rotation = eigh((xx+xx.conj().T)/2)
            v = v @ rotation
            for a in groups(xv, 1e-10):
                v[:, a] = site_basis(v[:, a])
            vectors[k, :, g] = v
            energies[k, g] = raw[k, g].mean()
    rank = ny*q//2
    order = np.argsort(energies.ravel(), kind='stable')
    lo, hi = energies.ravel()[order[[rank-1, rank]]]
    mu = (lo+hi)/2
    occupied = np.zeros(ny*q, bool)
    alternative = occupied.copy()
    # A stable momentum-first, position-second tie rule at the Fermi level.
    tie = np.flatnonzero(abs(energies.ravel()-mu) <= tol) if hi-lo <= tol else np.array([], int)
    if len(tie):
        below = np.flatnonzero(energies.ravel() < mu-tol)
        occupied[below] = alternative[below] = True
        need = rank-len(below)
        occupied[tie[:need]] = True
        alternative[tie[-need:]] = True
    else:
        occupied[order[:rank]] = True
        alternative = occupied.copy()
        need = 0
    occupied = occupied.reshape(ny, q)
    alternative = alternative.reshape(ny, q)
    p = np.einsum('kai,ki,kbi->kab', vectors, occupied, vectors.conj(), optimize=True)
    alt = np.einsum('kai,ki,kbi->kab', vectors, alternative, vectors.conj(), optimize=True)
    raw_sorted = np.sort(raw.ravel())
    residual = float(abs(blocks @ vectors-vectors*energies[:, None, :]).max())
    diag = dict(degeneracy_tolerance=tol, fermi_degenerate_count=len(tie),
                fermi_degenerate_occupied=need, rank=int(occupied.sum()),
                occupied_rank_by_momentum=occupied.sum(1).tolist(),
                minimum_absolute_energy=float(abs(raw).min()),
                half_filling_gap=float(raw_sorted[rank]-raw_sorted[rank-1]),
                negative_energy_count=int((raw < 0).sum()),
                particle_hole_pairing_residual=float(abs(raw_sorted+raw_sorted[::-1]).max()),
                eigensystem_residual=residual,
                projector_idempotency_residual=float(abs(p@p-p).max()),
                alternative_projector_difference=float(abs(p-alt).max()))
    assert diag['rank'] == rank and residual < 1e-10 and diag['projector_idempotency_residual'] < 1e-12
    return dict(parent_blocks=blocks, energies=energies, raw_energies=raw, vectors=vectors,
                occupied=occupied, projector_delta=np.fft.ifft(p,axis=0),
                alternative_projector_delta=np.fft.ifft(alt,axis=0)), diag


def dense_from_delta(delta, width, origin=0):
    ny, q, _ = delta.shape
    rows = (origin+np.arange(width)) % ny
    blocks = delta[(rows[:,None]-rows[None,:]) % ny]
    return blocks.transpose(0,2,1,3).reshape(width*q, width*q)


def entropy_values(nu):
    assert np.min(nu) > -2e-10 and np.max(nu) < 1+2e-10
    nu = np.clip(nu, 0., 1.)
    return -xlogy(nu, nu)-xlogy(1-nu, 1-nu)


def restricted(delta, width, contour=False):
    p = dense_from_delta(delta, width)
    if contour:
        nu, v = eigh(p, check_finite=False, driver='evd')
        s = (abs(v)**2) @ entropy_values(nu)
    else:
        nu = eigh(p, eigvals_only=True, check_finite=False, driver='evd')
        s = None
    return nu, s


def correlations(delta):
    ny, q, _ = delta.shape
    nx = q//2
    out = np.zeros((nx, ny//2+1))
    for x in range(nx):
        a = slice(2*x,2*x+2)
        out[x] = .5*np.sum(abs(delta[:ny//2+1,a,a])**2,axis=(1,2))
    return out.mean(0), out


def strip_curves(delta, *, with_contour):
    ny, q, _ = delta.shape
    rows = np.arange(1,ny//2+1)
    ent, variance, counts, left, right = [], [], [], [], []
    half_nu = half_contour = None
    for width in rows:
        nu, contour = restricted(delta, width, with_contour)
        ent.append(float(entropy_values(nu).sum()))
        nu = np.clip(nu,0,1)
        variance.append(float(np.sum(nu*(1-nu))))
        counts.append(int((abs(2*nu-1)<.99).sum()))
        if contour is not None:
            contour = contour.reshape(width,q//2,2).sum(2)
            assert abs(contour.sum()-ent[-1]) < 2e-10
            left.append(float(contour[:,5:7].sum()))
            right.append(float(contour[:,14:16].sum()))
        if width == ny//2:
            half_nu, half_contour = nu, contour
    out = dict(widths=rows, entropy=np.array(ent), variance=np.array(variance),
               mode_count=np.array(counts), half_nu=half_nu)
    if with_contour:
        out.update(wall_left=np.array(left), wall_right=np.array(right), half_contour=half_contour)
    return out


def mutual_information(delta):
    ny, q, _ = delta.shape
    width = ny//4
    a = np.arange(width*q)
    b = a+q*(ny//2)
    ab = np.r_[a,b]
    p = dense_from_delta(delta, ny)
    s_a = float(entropy_values(eigh(p[np.ix_(a,a)],eigvals_only=True)).sum())
    s_b = float(entropy_values(eigh(p[np.ix_(b,b)],eigvals_only=True)).sum())
    s_ab = float(entropy_values(eigh(p[np.ix_(ab,ab)],eigvals_only=True)).sum())
    value = s_a+s_b-s_ab
    assert value > -1e-9
    return dict(mutual_information=value, strip_entropy=s_a, union_entropy=s_ab,
                opposite_strip_entropy_difference=abs(s_a-s_b))


def minimal_mode_density(data):
    e, v = data['raw_energies'], data['vectors']
    ny,q = e.shape
    tol = 1e-12*max(1.,float(abs(e).max()))
    mask = abs(abs(e)-abs(e).min()) <= tol
    density = np.einsum('kai,ki->a',abs(v)**2,mask)/(ny*mask.sum())
    result = np.broadcast_to(density.reshape(q//2,2).sum(1), (ny,q//2)).copy()
    assert abs(result.sum()-1)<1e-12
    return result, int(mask.sum())


def compute_case(nx,ny,alpha):
    blocks, diag = build_parent(nx,ny,alpha)
    data, spectral = diagonalize(blocks)
    diag.update(spectral)
    delta = data['projector_delta']
    data['correlation'], data['column_correlation'] = correlations(delta)
    if nx==20 and ((alpha==1 and ny in ENTROPY_SIZES+(32,)) or (alpha==3 and ny==32)):
        data.update(strip_curves(delta, with_contour=(ny in WALL_SIZES+(32,))))
    if nx==20 and ny in (20,24,28):
        diag.update(mutual_information(delta))
    if nx==20 and alpha==1 and ny==30:
        data['minimal_mode_density'], diag['minimal_absolute_energy_multiplicity'] = minimal_mode_density(data)
    if nx==20 and ny==30 and alpha in (1,3):
        data['filter_times'] = np.array([1.,5.,60.])
        data['filter_occupations'] = np.sort(expit(-2*data['filter_times'][:,None]*data['energies'].ravel()),axis=1)
        diag['filter_max_trace_deviation_from_half'] = float(abs(data['filter_occupations'].sum(1)-nx*ny).max())
    if diag['fermi_degenerate_count']:
        altcorr, _ = correlations(data['alternative_projector_delta'])
        diag['alternative_correlator_max_difference'] = float(abs(altcorr-data['correlation']).max())
        s1 = float(entropy_values(restricted(delta,ny//2)[0]).sum())
        s2 = float(entropy_values(restricted(data['alternative_projector_delta'],ny//2)[0]).sum())
        diag['alternative_half_entropy_difference'] = s2-s1
    return data,diag


def load_case(nx,ny,alpha,root=HERE/'data'):
    path = root/'cases'/f'{case_id(nx,ny,alpha)}.npz'
    with np.load(path,allow_pickle=False) as z:
        data = {k:z[k] for k in z.files}
    return data,json.loads(path.with_suffix('.json').read_text())['diagnostics']


def verified_case(path, config, sources):
    try:
        record = json.loads(path.with_suffix('.json').read_text())
        return (record['config']==config and record['sources']==sources and
                record['bytes']==path.stat().st_size and record['sha256']==sha(path))
    except (OSError,ValueError,KeyError):
        return False


def run(threads=4):
    sources = source_identity()
    output = HERE/'data/cases'
    output.mkdir(parents=True,exist_ok=True)
    tasks = task_table()
    print(f'CPU canonical OW parent; {len(tasks)} deterministic cases; {threads} BLAS threads; output {output}',flush=True)
    start=time.monotonic()
    with threadpool_limits(limits=threads):
        for nx,ny,alpha in tqdm(tasks,desc='Clean periodic parent',unit='case'):
            config=configuration(nx,ny,alpha)
            path=output/f'{case_id(nx,ny,alpha)}.npz'
            if verified_case(path,config,sources):
                continue
            t0=time.monotonic()
            data,diag=compute_case(nx,ny,alpha)
            npz_write(path,**data)
            json_write(path.with_suffix('.json'),dict(config=config,sources=sources,diagnostics=diag,
                       bytes=path.stat().st_size,sha256=sha(path),elapsed_seconds=time.monotonic()-t0))
    json_write(HERE/'data/run.json',dict(cases=[case_id(*t) for t in tasks],sources=sources,
               manuscript_sha256=sha(MANUSCRIPT/'manuscript.tex'), manuscript_pdf_sha256=sha(MANUSCRIPT/'manuscript.pdf'),
               threads=threads,elapsed_seconds=time.monotonic()-start,
               scientific_uncertainty='none; deterministic equilibrium states'))
    print(f'Completed and verified {len(tasks)} cases in {time.monotonic()-start:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.threads<1: parser.error('threads must be positive')
    run(args.threads)
