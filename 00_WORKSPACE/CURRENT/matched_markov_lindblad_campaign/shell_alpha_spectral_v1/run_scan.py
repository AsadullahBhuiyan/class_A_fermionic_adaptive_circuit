"""Hard-wall channel spectra for shells 1, 2, infinity and 21 alpha values."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'square_large_spectral_v1'))
import run_large as backend

SIZES = (20, 40, 60, 80, 100)
ALPHAS = tuple(i / 10 for i in range(10, 31))
SHELLS = (1, 2, None)
sha, atomic_json = backend.sha, backend.atomic_json


def config(index, ny, shell):
    if index not in range(21) or shell not in SHELLS:
        raise ValueError('Invalid alpha index or shell')
    cfg = backend.spectral.config(ALPHAS[index], ny)
    cfg.update(revision='hard_wall_shell_alpha_spectral_v1', alpha_index=index,
               nshell=shell, shell_label='infinity' if shell is None else str(shell),
               infinite_shell_meaning='no shell cutoff; hard-wall region mask retained',
               zero_bloch_norm_convention='canonical nmag=1e-15 at exact zeros; alpha=2 not shifted',
               solver='complete_dense_spectrum_of_both_exact_hard_wall_blocks')
    return cfg


def make_model(cfg):
    # The older spectral.make_model deliberately hard-codes nshell=1: do not use it.
    return backend.spectral.build_model({'model': dict(
        Nx=cfg['Nx'], Ny=cfg['Ny'], domain_wall=True, wall_locations=cfg['walls'],
        alpha_run_in=cfg['alpha_1'], alpha_run_out=cfg['alpha_2'],
        nshell=cfg['nshell'], trial_orbitals=cfg['trial_orbitals'],
        dw_truncation=cfg['dw_truncation'])})


def sources():
    result = backend.sources()
    result[str(Path(__file__).relative_to(backend.spectral.REPO))] = sha(__file__)
    return result


def tasks():
    return [(i, ny, shell) for shell in SHELLS for ny in SIZES for i in range(21)]


def folder(root, index, ny, shell):
    label = 'inf' if shell is None else str(shell)
    return root / f'shell{label}_Ny{ny:03d}_a{index:02d}_{ALPHAS[index]:.1f}'


def verified(root, index, ny, shell):
    return backend.spectral.verified_complete(folder(root, index, ny, shell),
                                              config(index, ny, shell), sources())


def run_case(root, index, ny, shell):
    path = folder(root, index, ny, shell)
    path.mkdir(parents=True, exist_ok=True)
    if verified(root, index, ny, shell):
        print('[skip verified]', path.name, flush=True)
        return
    cfg, hashes = config(index, ny, shell), sources()
    required = 100 * (20 * ny / 10000) ** 2 + 2
    if psutil.virtual_memory().available / 2**30 < required:
        raise MemoryError(f'Need {required:.1f} GiB available headroom')
    print('[configuration]', json.dumps(cfg), flush=True)
    start = time.monotonic()
    model = make_model(cfg)
    assert model.nshell == shell and model.dw_truncation
    built = time.monotonic()
    indices, matrices, checks = backend.construct_blocks(model)
    product_done = time.monotonic()
    rng = np.random.default_rng(2026100700 + 100 * index + ny)
    probes = rng.normal(size=(2 * 20 * ny, 3)) + 1j * rng.normal(size=(2 * 20 * ny, 3))
    probes /= np.linalg.norm(probes, axis=0)
    action_error = float(np.linalg.norm(backend.block_action(indices, matrices, probes)
                         - backend.spectral.independent_action(model, probes)))
    if action_error > 1e-11:
        raise ValueError('Block action disagrees with full-length OW sweep')
    values, dominant, blocks = backend.solve_blocks(model, indices, matrices)
    residual = float(np.linalg.norm(backend.spectral.independent_action(
        model, dominant['right'][:, None])[:, 0] - dominant['value'] * dominant['right']))
    if residual > 1e-10:
        raise ValueError('Independent dominant-eigenpair residual failed')
    radius = dominant['radius']
    rate = float(-2 * np.log(radius)) if radius else None
    details = dict(**checks, blocks=blocks, dominant_sector=dominant['sector'],
        independent_product_action_error=action_error, independent_dominant_residual=residual,
        spectral_radius=radius, covariance_gap_raw=rate, covariance_multiplier_gap=float(1-radius**2),
        gap_status=backend.spectral.classify_radius(radius),
        unit_modulus_tolerance=backend.spectral.UNIT_TOL,
        ow_build_seconds=built-start, product_seconds=product_done-built,
        eigensolver_seconds=sum(b['eigensolver_seconds'] for b in blocks),
        elapsed_seconds=time.monotonic()-start,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        numpy_version=np.__version__, scipy_version=backend.spectral.scipy.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        completed_utc=datetime.now(timezone.utc).isoformat())
    if sources() != hashes:
        raise RuntimeError('Scientific sources changed during computation')
    temporary = path / 'spectrum.tmp.npz'
    np.savez_compressed(temporary, eigenvalues=values,
        dominant_eigenvector=dominant['right'], dominant_left_eigenvector=dominant['left'],
        dominant_eigenvalue=dominant['value'], spectral_radius=radius,
        covariance_multiplier_gap=1-radius**2, covariance_gap_raw=np.inf if rate is None else rate,
        interior_indices=indices[0], exterior_indices=indices[1],
        config_json=np.asarray(json.dumps(cfg, sort_keys=True)),
        diagnostics_json=np.asarray(json.dumps(details)))
    with np.load(temporary, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['eigenvalues'], values)
    result = path / 'spectrum.npz'
    temporary.replace(result)
    atomic_json(path / 'completion.json', dict(status='complete', config=cfg, sources=hashes,
        diagnostics=details, result_filename=result.name, result_bytes=result.stat().st_size,
        result_sha256=sha(result)))
    assert verified(root, index, ny, shell)
    print('[complete]', path.name, json.dumps(details), flush=True)


def inventory(root):
    return [dict(alpha_1=ALPHAS[i], Ny=ny, nshell=shell, complete=verified(root, i, ny, shell))
            for i, ny, shell in tasks()]


def analyze(root):
    rows, inputs = [], {}
    for i, ny, shell in tasks():
        if not verified(root, i, ny, shell):
            raise RuntimeError('Analysis requires all 315 verified cases')
        path = folder(root, i, ny, shell)
        d = json.loads((path / 'completion.json').read_text())['diagnostics']
        rows.append(dict(alpha_1=ALPHAS[i], Nx=20, Ny=ny, nshell='inf' if shell is None else shell,
                         rho_A=d['spectral_radius'], g_C=d['covariance_multiplier_gap'],
                         kappa_C=d['covariance_gap_raw'], status=d['gap_status'],
                         residual=d['independent_dominant_residual'], seconds=d['elapsed_seconds']))
        for name in ('completion.json', 'spectrum.npz'):
            inputs[str((path / name).relative_to(root))] = sha(path / name)
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    with (out / 'gaps.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8,
                         'text.usetex': True, 'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.6), sharex=True)
    styles = zip(SIZES, ('#c0392b', '#23934c', '#2468ad', '#8e44ad', '#d17b0f'),
                 ('^', 's', 'o', 'D', 'v'), (':', '--', '-', '-.', (0, (3, 1, 1, 1))))
    styles = list(styles)
    for panel, (ax, shell) in enumerate(zip(axes, (1, 2, 'inf'))):
        for ny, color, marker, line in styles:
            selected = [r for r in rows if r['nshell'] == shell and r['Ny'] == ny]
            ax.plot(ALPHAS, [r['g_C'] for r in selected], color=color, marker=marker,
                    ls=line, mfc='white', ms=2.5, lw=.8, label=str(ny))
        ax.axvline(2, color='gray', ls='--', lw=.6)
        ax.tick_params(top=True, right=True)
        ax.set_ylabel(r'$g_C=1-\rho(A)^2$')
        label = r'\infty' if shell == 'inf' else str(shell)
        ax.text(.03, .08, rf'$n_{{\mathrm{{shell}}}}={label}$', transform=ax.transAxes)
        ax.text(-.15, 1.02, f'({chr(97+panel)})', transform=ax.transAxes)
    axes[0].legend(title=r'$N_y$ ($N_x=20$)', ncol=3, frameon=False, fontsize=8)
    axes[-1].set_xlabel(r'$\alpha_1$')
    fig.tight_layout(pad=.8)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'channel_gap_shell_comparison.{ext}', dpi=300)
    plt.close(fig)
    atomic_json(out / 'provenance.json', dict(input_sha256=inputs, sources=sources(),
        meaning='Exact spectral channel gaps; no time evolution, trajectory sampling, fits or SEM. '
                'Nx=20 fixed width, both hard-wall blocks active. Infinite shell retains wall mask. '
                'Unit-modulus modes are retained, not discarded; consult status in CSV.',
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.name != 'provenance.json'}))


def worker(root, number, cpus):
    failures = []
    for i, ny, shell in tqdm(tasks()[number::len(cpus)], desc=f'worker {number}', unit='case'):
        if verified(root, i, ny, shell):
            continue
        name = folder(root, i, ny, shell).name
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'case', '--root', str(root),
               '--index', str(i), '--ny', str(ny), '--shell', 'inf' if shell is None else str(shell),
               '--cpu', str(cpus[number])]
        with (root / f'{name}.log').open('a', buffering=1) as handle:
            code = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT).returncode
        if code or not verified(root, i, ny, shell):
            failures.append(dict(case=name, returncode=code))
    return failures


def queue(root, cpus):
    if psutil.virtual_memory().available / 2**30 < 6 * len(cpus) + 12:
        raise MemoryError('Insufficient headroom for six GiB per worker plus 12 GiB reserve')
    atomic_json(root / 'queue_status.json', dict(status='running', workers=len(cpus), total=315))
    failures = []
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures = [pool.submit(worker, root, i, cpus) for i in range(len(cpus))]
        for future in as_completed(futures):
            failures.extend(future.result())
    rows = inventory(root)
    if failures or not all(r['complete'] for r in rows):
        atomic_json(root / 'queue_status.json', dict(status='failed', failures=failures, cases=rows))
        return 1
    try:
        analyze(root)
    except Exception:
        atomic_json(root / 'queue_status.json', dict(status='analysis_failed', complete=315,
                                                    traceback=traceback.format_exc()))
        raise
    atomic_json(root / 'queue_status.json', dict(status='complete', complete=315, total=315))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'queue', 'case', 'report', 'analyze'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--cpus', default='1,2,3,4,5,6,8,9,10,11,12,13,17,18,19,20,21,22,23,24,25')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--index', type=int, choices=range(21))
    parser.add_argument('--ny', type=int, choices=SIZES)
    parser.add_argument('--shell', choices=('1', '2', 'inf'))
    args = parser.parse_args()
    cpus = [int(c) for c in args.cpus.split(',')]
    if args.mode in ('launch', 'queue') and (not 1 <= len(cpus) < 56 or len(set(cpus)) != len(cpus)
                                            or not set(cpus) <= os.sched_getaffinity(0)):
        parser.error('Require 1..55 distinct allowed CPUs')
    if args.root is None and args.mode != 'launch':
        parser.error('--root required')
    root = (args.root or HERE / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if args.mode == 'launch':
        root.mkdir(parents=True, exist_ok=True)
        session = 'shell_gap_' + root.name
        if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
            parser.error('Queue already running')
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'queue', '--root', str(root),
               '--cpus', args.cpus]
        atomic_json(root / 'launch.json', dict(session=session, command=cmd, cpus=cpus,
            sources=sources(), cases=[config(*task) for task in tasks()]))
        shell_command = shlex.join(cmd) + ' > ' + shlex.quote(str(root / 'queue.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(HERE),
                        'bash', '-lc', shell_command], check=True)
        print(json.dumps(dict(root=str(root), session=session, workers=len(cpus), total=315), indent=2))
    elif args.mode == 'report':
        rows = inventory(root)
        print(json.dumps(dict(complete=sum(r['complete'] for r in rows), total=315, cases=rows), indent=2))
    elif args.mode == 'queue':
        return queue(root, cpus)
    elif args.mode == 'analyze':
        analyze(root)
    else:
        if args.cpu not in os.sched_getaffinity(0) or args.index is None or args.ny is None or args.shell is None:
            parser.error('case requires --cpu, --index, --ny, --shell')
        os.sched_setaffinity(0, {args.cpu})
        os.nice(10)
        shell = None if args.shell == 'inf' else int(args.shell)
        try:
            run_case(root, args.index, args.ny, shell)
        except Exception:
            path = folder(root, args.index, args.ny, shell)
            path.mkdir(parents=True, exist_ok=True)
            atomic_json(path / 'failure.json', dict(traceback=traceback.format_exc()))
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
