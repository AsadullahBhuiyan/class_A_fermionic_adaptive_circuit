"""Exact square hard-wall spectra at L=60,80,100; both sectors retained."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
from scipy.linalg import eig
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'spectral_gap_v1'))
import run_sweep as spectral

SIZES = (60, 80, 100)
sha = spectral.sha
atomic_json = spectral.atomic_json


def config(alpha, size):
    cfg = spectral.config(alpha, size)
    cfg.update(revision='hard_wall_square_large_spectral_v1', Nx=size, Ny=size,
               walls=[size//4, 3*size//4],
               wall_scaling='floor(Nx/4),floor(3Nx/4),inclusive',
               solver='complete_dense_spectrum_of_both_exact_hard_wall_blocks')
    return cfg


def sources():
    result = spectral.sources()
    result[str(Path(__file__).relative_to(spectral.REPO))] = sha(__file__)
    return result


def folder(root, alpha, size):
    return root / f'alpha{alpha}_L{size:03d}'


def verified(root, alpha, size):
    return spectral.verified_complete(folder(root, alpha, size), config(alpha, size), sources())


def construct_blocks(model, progress=True):
    """Restrict canonical modes, never regenerate them on a smaller lattice.

    The periodic exterior is ONE block (connected across the x seam). No slab
    is frozen or discarded. Nonzero inter-sector support is a fatal error.
    """
    dimension = 2 * model.Nx * model.Ny
    left, right = model.DW_loc
    x_coordinates = (np.arange(dimension) // 2) % model.Nx
    interior = (x_coordinates >= left) & (x_coordinates <= right)
    indices = [np.flatnonzero(interior), np.flatnonzero(~interior)]
    block_id = np.where(interior, 0, 1)
    local_id = np.empty(dimension, dtype=int)
    for idx in indices:
        local_id[idx] = np.arange(len(idx))
    matrices = [np.eye(len(idx), dtype=np.complex128) for idx in indices]
    max_error = 0.
    for x, y in tqdm(spectral.raster_word(model), desc='construct both Q blocks',
                     unit='site', disable=not progress):
        payload = model._get_ow_local_support_data(x, y)
        sector = 0 if left <= x <= right else 1
        support = payload['idx']
        mask = block_id[support] == sector
        mapped = local_id[support[mask]]
        matrix = matrices[sector]
        for name in spectral.ORDER:
            vector = payload[name]
            if np.any(vector[~mask] != 0):
                raise ValueError('Nonzero inter-sector OW support: block decomposition invalid')
            vector = vector[mask]
            error = float(abs(np.vdot(vector, vector).real - 1))
            max_error = max(max_error, error)
            if error > 1e-12:
                raise ValueError('OW mode lost normalization')
            rows = matrix[mapped, :]
            matrix[mapped, :] = rows - np.outer(vector, vector.conj() @ rows)
    return indices, matrices, dict(maximum_mode_norm_error=max_error,
                                  inter_sector_support_max=0., block_dimensions=[len(i) for i in indices])


def block_action(indices, matrices, vectors):
    out = np.empty_like(vectors)
    for idx, matrix in zip(indices, matrices):
        out[idx] = matrix @ vectors[idx]
    return out


def solve_blocks(model, indices, matrices):
    spectra = []
    dominant = None
    diagnostics = []
    for sector, (idx, matrix) in enumerate(zip(indices, matrices)):
        print(f'[eigensolver start] sector={sector}, dimension={len(idx)}', flush=True)
        start = time.monotonic()
        values, left, right = eig(matrix, left=True, right=True, check_finite=True)
        k = int(np.argmax(abs(values)))
        value = values[k]
        radius = float(abs(value))
        spectral.classify_radius(radius)
        rvec = right[:, k] / np.linalg.norm(right[:, k])
        lvec = left[:, k] / np.linalg.norm(left[:, k])
        rr = float(np.linalg.norm(matrix @ rvec - value * rvec))
        lr = float(np.linalg.norm(matrix.conj().T @ lvec - value.conjugate() * lvec))
        if max(rr, lr) > 1e-10:
            raise ValueError('Block dominant eigenpair residual failed')
        seconds = time.monotonic() - start
        diagnostics.append(dict(sector=sector, dimension=len(idx), spectral_radius=radius,
                                dominant_residual=rr, dominant_left_residual=lr,
                                eigensolver_seconds=seconds))
        if dominant is None or radius > dominant['radius']:
            full_right = np.zeros(2*model.Nx*model.Ny, dtype=np.complex128)
            full_left = np.zeros_like(full_right)
            full_right[idx], full_left[idx] = rvec, lvec
            dominant = dict(radius=radius, value=value, right=full_right, left=full_left,
                            sector=sector, residual=rr, left_residual=lr)
        spectra.append(values.copy())
        print(f'[eigensolver done] sector={sector}, rho={radius:.12g}, seconds={seconds:.1f}', flush=True)
        del left, right
        gc.collect()
    return np.concatenate(spectra), dominant, diagnostics


def run_case(root, alpha, size):
    path = folder(root, alpha, size)
    path.mkdir(parents=True, exist_ok=True)
    cfg, hashes = config(alpha, size), sources()
    if verified(root, alpha, size):
        print('[skip verified]', path, flush=True)
        return
    # Includes canonical OW construction's dense intermediate arrays, not only A.
    required_gib = 100 * (size/100)**4 + 8
    available_gib = psutil.virtual_memory().available / 2**30
    if available_gib < required_gib:
        raise MemoryError(f'Need conservative {required_gib:.1f} GiB headroom; available {available_gib:.1f}')
    start = time.monotonic()
    print('[configuration]', json.dumps(cfg), flush=True)
    print('[resources]', json.dumps(dict(available_gib=available_gib, required_gib=required_gib,
                                        cpu_affinity=sorted(os.sched_getaffinity(0)))), flush=True)
    model = spectral.make_model(cfg)
    constructed = time.monotonic()
    indices, matrices, checks = construct_blocks(model)
    product_done = time.monotonic()
    rng = np.random.default_rng(2026093001 + alpha*1000 + size)
    probes = rng.normal(size=(2*size*size, 3)) + 1j*rng.normal(size=(2*size*size, 3))
    probes /= np.linalg.norm(probes, axis=0)
    error = float(np.linalg.norm(block_action(indices, matrices, probes)
                                 - spectral.independent_action(model, probes)))
    if error > 1e-11:
        raise ValueError('Independent full-system projector sweep disagrees with blocks')
    values, dominant, block_details = solve_blocks(model, indices, matrices)
    residual = float(np.linalg.norm(spectral.independent_action(model, dominant['right'][:, None])[:, 0]
                                   - dominant['value'] * dominant['right']))
    if residual > 1e-10:
        raise ValueError('Independent full-system dominant residual failed')
    radius = dominant['radius']
    gap = float(-2*np.log(radius)) if radius else None
    details = dict(**checks, blocks=block_details, dominant_sector=dominant['sector'],
        independent_product_action_error=error, independent_dominant_residual=residual,
        dominant_residual=dominant['residual'], dominant_left_residual=dominant['left_residual'],
        spectral_radius=radius, covariance_gap_raw=gap, gap_status=spectral.classify_radius(radius),
        covariance_multiplier_gap=float(1-radius**2), unit_modulus_tolerance=spectral.UNIT_TOL,
        ow_build_seconds=constructed-start, product_seconds=product_done-constructed,
        elapsed_seconds=time.monotonic()-start,
        eigensolver_seconds=sum(d['eigensolver_seconds'] for d in block_details),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        cpu_affinity=sorted(os.sched_getaffinity(0)))
    if sources() != hashes:
        raise RuntimeError('Scientific sources changed during calculation; refusing completion')
    temporary = path / 'spectrum.tmp.npz'
    np.savez_compressed(temporary, eigenvalues=values, dominant_eigenvector=dominant['right'],
        dominant_left_eigenvector=dominant['left'], dominant_eigenvalue=dominant['value'],
        spectral_radius=radius, covariance_gap_raw=gap if gap is not None else np.inf,
        covariance_multiplier_gap=1-radius**2, interior_indices=indices[0], exterior_indices=indices[1],
        config_json=np.asarray(json.dumps(cfg, sort_keys=True)),
        diagnostics_json=np.asarray(json.dumps(details)))
    with np.load(temporary, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['eigenvalues'], values)
    temporary.replace(path / 'spectrum.npz')
    result = path / 'spectrum.npz'
    atomic_json(path / 'completion.json', dict(status='complete', config=cfg, sources=hashes,
        diagnostics=details, result_filename=result.name, result_bytes=result.stat().st_size,
        result_sha256=sha(result)))
    assert verified(root, alpha, size)
    print('[complete]', path.name, json.dumps(details), flush=True)


def inventory(root):
    return [dict(alpha=a, size=n, complete=verified(root,a,n)) for n in SIZES for a in (1,3)]


def queue(root, cpus):
    failures = []
    with tqdm(total=6, desc='large-square spectra', unit='case') as bar:
        for size in SIZES:
            # L100 canonical OW construction can have substantial peak memory.
            # Serialize those two cases; smaller pairs safely run concurrently.
            groups = [[1], [3]] if size == 100 else [[1, 3]]
            for group in groups:
                processes = []
                for alpha in group:
                    if verified(root, alpha, size):
                        bar.update(1)
                        continue
                    cpu = cpus[0 if alpha == 1 else 1]
                    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'case',
                           '--root', str(root), '--alpha', str(alpha), '--size', str(size), '--cpu', str(cpu)]
                    handle = (root / f'alpha{alpha}_L{size:03d}.log').open('a', buffering=1)
                    processes.append((alpha, subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT), handle))
                for alpha, process, handle in processes:
                    code = process.wait()
                    handle.close()
                    if code != 0 or not verified(root, alpha, size):
                        failures.append(dict(alpha=alpha, size=size, returncode=code))
                    bar.update(1)
                    atomic_json(root/'queue_status.json', dict(status='running', cases=inventory(root), failures=failures))
    if failures:
        atomic_json(root/'queue_status.json', dict(status='failed', cases=inventory(root), failures=failures))
        return 1
    result = subprocess.run([sys.executable, str(HERE/'analyze_large.py'), '--root', str(root)])
    atomic_json(root/'queue_status.json', dict(status='complete' if result.returncode == 0 else 'analysis_failed',
                                              cases=inventory(root), analysis_returncode=result.returncode))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch','queue','case','report'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--cpus', default='0,7')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--alpha', type=int, choices=[1,3])
    parser.add_argument('--size', type=int, choices=SIZES)
    args = parser.parse_args()
    if args.mode in ('launch','queue'):
        cpus = [int(c) for c in args.cpus.split(',')]
        if len(cpus) != 2 or len(set(cpus)) != 2 or not set(cpus) <= os.sched_getaffinity(0):
            parser.error('Require two distinct allowed CPUs')
    if args.mode == 'launch':
        root = (args.root or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
        root.mkdir(parents=True, exist_ok=True)
        session = 'square_large_gap_' + root.name
        cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'queue','--root',str(root),'--cpus',args.cpus]
        atomic_json(root/'launch.json', dict(session=session, command=cmd, sources=sources(),
            cases=[config(a,n) for n in SIZES for a in (1,3)]))
        atomic_json(root/'queue_status.json', dict(status='launched', cases=inventory(root)))
        shell = shlex.join(cmd) + ' > ' + shlex.quote(str(root/'queue.log')) + ' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(HERE),'bash','-lc',shell], check=True)
        print(json.dumps(dict(root=str(root), session=session), indent=2))
        return 0
    if args.root is None:
        parser.error('--root required')
    root = args.root.resolve()
    if args.mode == 'report':
        rows = inventory(root)
        print(json.dumps(dict(complete=sum(r['complete'] for r in rows), total=6, cases=rows), indent=2))
        return 0
    if args.mode == 'queue':
        return queue(root, cpus)
    if args.cpu is None or args.cpu not in os.sched_getaffinity(0) or args.alpha is None or args.size is None:
        parser.error('case needs allowed --cpu, --alpha and --size')
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    try:
        run_case(root, args.alpha, args.size)
    except Exception:
        path = folder(root, args.alpha, args.size)
        path.mkdir(parents=True, exist_ok=True)
        atomic_json(path/'failure.json', dict(traceback=traceback.format_exc()))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
