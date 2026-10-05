"""Square hard-wall stationary occupation spectra; canonical CPU channel only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'steady_purity_gap_v1'))
import run_purity as parent

SIZES = (20, 30, 40, 50, 60, 70, 80)
ALPHAS = (1, 3)
sha, atomic_json = parent.sha, parent.atomic_json


def config(alpha, size):
    cfg = parent.config(alpha, size)
    cfg.update(revision='hard_wall_square_stationary_purity_gap_v1', Nx=size, Ny=size,
               walls=[size//4, 3*size//4],
               wall_scaling='floor(Nx/4),floor(3Nx/4),inclusive',
               solver='canonical_covariance_dynamics_then_full_and_twirled_occupation_spectra')
    return cfg


def sources():
    result = parent.sources()
    result[str(Path(__file__).relative_to(parent.spectral.REPO))] = sha(Path(__file__))
    return result


def folder(root, alpha, size):
    return root / f'alpha{alpha}_L{size:03d}'


def verified(root, alpha, size):
    directory = folder(root, alpha, size)
    try:
        rec = json.loads((directory/'completion.json').read_text())
        result = directory/'dynamics.npz'
        return (rec['status'] == 'complete' and rec['diagnostics']['converged']
                and rec['config'] == config(alpha, size) and rec['sources'] == sources()
                and rec['result_filename'] == result.name
                and rec['result_bytes'] == result.stat().st_size
                and rec['result_sha256'] == sha(result))
    except (OSError, KeyError, ValueError):
        return False


def run_case(root, alpha, size):
    if verified(root, alpha, size):
        print('[skip verified]', alpha, size, flush=True)
        return
    cfg, hashes = config(alpha, size), sources()
    target = folder(root, alpha, size)
    target.mkdir(parents=True, exist_ok=True)
    print('[configuration]', json.dumps(cfg), flush=True)
    print('[resources]', json.dumps(dict(cpu_affinity=sorted(os.sched_getaffinity(0)),
                                         numerical_threads=1, output=str(target))), flush=True)
    arrays, diagnostics = parent.collect(parent.spectral.make_model(cfg), cfg)
    if sources() != hashes:
        raise RuntimeError('Source changed during calculation')
    tmp = target/'dynamics.tmp.npz'
    np.savez_compressed(tmp, **arrays, config_json=np.asarray(json.dumps(cfg, sort_keys=True)))
    with np.load(tmp, allow_pickle=False) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    path = target/'dynamics.npz'
    tmp.replace(path)
    rec = dict(status='complete' if diagnostics['converged'] else 'not_converged',
               config=cfg, sources=hashes, diagnostics=diagnostics,
               result_filename=path.name, result_bytes=path.stat().st_size,
               result_sha256=sha(path))
    atomic_json(target/'completion.json', rec)
    print('[result]', json.dumps(diagnostics), flush=True)
    if not diagnostics['converged']:
        raise RuntimeError('Saved finite-time data but failed the stationarity criterion')
    assert verified(root, alpha, size)


def report(root):
    rows = []
    for size in SIZES:
        for alpha in ALPHAS:
            if verified(root, alpha, size):
                rec = json.loads((folder(root, alpha, size)/'completion.json').read_text())
                d = rec['diagnostics']
                rows.append(dict(alpha_1=alpha, L=size, Nx=size, Ny=size,
                    purity_gap_full=d['purity_gap_full'], purity_gap_twirl=d['purity_gap_twirl'],
                    half_filling_distance_full=d['half_filling_distance_full'],
                    half_filling_distance_twirl=d['half_filling_distance_twirl'],
                    final_frobenius_change=d['final_frobenius_change'],
                    elapsed_seconds=d['elapsed_seconds']))
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    atomic_json(out/'status.json', dict(complete=len(rows), total=14, cases=rows))
    if not rows:
        return
    with (out/'gaps.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8,
                         'mathtext.fontset': 'cm', 'axes.labelsize': 9,
                         'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7))
    for ax, key, label, letter in zip(axes, ('purity_gap_full', 'purity_gap_twirl'),
                                     ('Full state', 'Translation twirl'), 'ab'):
        for alpha, color, marker, ls in ((1, '#2468ad', 'o', '-'), (3, '#c0392b', '^', ':')):
            selected = [r for r in rows if r['alpha_1'] == alpha]
            ax.plot([r['L'] for r in selected], [r[key] for r in selected],
                    color=color, marker=marker, ls=ls, ms=4, mfc='white', label=rf'$\alpha_1={alpha}$')
        ax.set(xlabel=r'$L=N_x=N_y$', ylabel=r'$\Delta_{\mathrm{pur}}$', ylim=(-.025, 1.08))
        ax.tick_params(top=True, right=True)
        ax.text(.04, .83, label, transform=ax.transAxes)
        ax.text(-.12, 1.025, f'({letter})', transform=ax.transAxes, fontsize=9)
        ax.legend(frameon=False, loc='center right')
    fig.tight_layout(pad=.8)
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'square_stationary_gaps.{ext}', dpi=300)
    plt.close(fig)
    (out/'caption.txt').write_text(
        'Square Nx=Ny=L hard-wall canonical outcome-averaged channel; alpha_1=1,3; '
        'alpha_2=30; nshell=1; walls floor(L/4),floor(3L/4), inclusive; all slabs '
        'active, periodic, X trials, zero twist, complex128, perfect correction '
        'with measurement dephasing, raster-y Ap/Am/Bp/Bm. Maxmix initial state, '
        '60 cycles plus cycle 61 stationarity check. Delta_pur=min|1-2n|, measured '
        'from the final full correlation matrix or after discrete y twirling. '
        'Exact two-point evolution; no trajectory samples or error bars. No fit. '
        'Only checksum-verified cases passing stationarity are plotted. '
        f'{len(rows)}/14 accepted cases.\n')


def queue(root, cpus):
    started = time.monotonic()
    def worker(alpha, cpu):
        failed = []
        for size in tqdm(SIZES, desc=f'alpha={alpha}', unit='case'):
            if verified(root, alpha, size):
                continue
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'case',
                       '--root', str(root), '--alpha', str(alpha), '--size', str(size), '--cpu', str(cpu)]
            print('[launch case]', shlex.join(command), flush=True)
            with (root/f'alpha{alpha}_L{size:03d}.log').open('a', buffering=1) as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            accepted = child.returncode == 0 and verified(root, alpha, size)
            if not accepted:
                failed.append(size)
            print('[case finished]', alpha, size, 'accepted=', accepted, flush=True)
            atomic_json(root/f'worker_alpha{alpha}.json', dict(failures=failed,
                complete=sum(verified(root, alpha, n) for n in SIZES), total=len(SIZES)))
        return failed
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, a, cpu) for a, cpu in zip(ALPHAS, cpus)]
        failures = [job.result() for job in jobs]
    report(root)
    atomic_json(root/'queue_status.json', dict(status='failed' if any(failures) else 'complete',
                                              failures=failures, elapsed_seconds=time.monotonic()-started))
    return int(any(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch', 'queue', 'case', 'report'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--cpus', default='9,10')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--alpha', type=int, choices=ALPHAS)
    parser.add_argument('--size', type=int, choices=SIZES)
    args = parser.parse_args()
    root = (args.root or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cpus = [int(c) for c in args.cpus.split(',')]
    if args.mode in ('queue', 'launch'):
        if len(cpus) != 2 or len(set(cpus)) != 2 or not set(cpus) <= os.sched_getaffinity(0):
            parser.error('Specify two distinct allowed CPUs')
    if args.mode == 'launch':
        session = 'square_purity_'+root.name
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'queue', '--root', str(root), '--cpus', args.cpus]
        atomic_json(root/'launch.json', dict(session=session, cpus=cpus, command=cmd, sources=sources(),
                                             cases=[config(a, n) for a in ALPHAS for n in SIZES]))
        shell = 'set -o pipefail; '+shlex.join(cmd)+' 2>&1 | tee -a '+shlex.quote(str(root/'queue.log'))
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash', '-lc', shell], check=True)
        print(json.dumps(dict(session=session, root=str(root), cpus=cpus, cases=14), indent=2))
    elif args.mode == 'queue':
        return queue(root, cpus)
    elif args.mode == 'report':
        report(root)
        print((root/'analysis/status.json').read_text())
    else:
        if args.alpha is None or args.size is None or args.cpu not in os.sched_getaffinity(0):
            parser.error('case requires alpha, size, and an allowed CPU')
        os.sched_setaffinity(0, {args.cpu})
        os.nice(10)
        run_case(root, args.alpha, args.size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
