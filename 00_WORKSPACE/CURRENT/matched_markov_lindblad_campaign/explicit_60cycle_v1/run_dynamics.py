"""Canonical CPU raster-y channel evolution: 60 cycles, all sixteen gap cases."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'spectral_gap_v1'))
import run_sweep as spectral

SIZES = spectral.SIZES
REFERENCE = PROJECT / 'spectral_gap_v1/results/20260929T191448Z'
sha = spectral.sha
atomic_json = spectral.atomic_json
FLOOR = 1e-13  # diagnostic censoring threshold, not a stopping criterion


def config(alpha, ny):
    cfg = spectral.config(alpha, ny)
    cfg.update(revision='hard_wall_raster_y_explicit_60cycle_v1', cycles=60,
               init_mode='maxmix', initial_correlation='identity/2',
               explicit_dynamics=True, canonical_source='classA_U1FGTN.run_markov_channel',
               n_a=0.5, sampling=False, numerical_floor=FLOOR)
    return cfg


def sources():
    result = spectral.sources()
    result[str(Path(__file__).resolve().relative_to(spectral.REPO))] = sha(__file__)
    return result


def collect(model, cycles=60, progress=True):
    """Observe the canonical channel without modifying its state or stopping early."""
    n = 2*model.Nx*model.Ny
    word = np.asarray([x+model.Nx*y for x,y in spectral.raster_word(model)])
    seen, increments, charges, norms, times, hermiticity = [], [], [], [], [], []
    previous = None
    start = time.monotonic()
    with tqdm(total=cycles, desc='canonical channel cycles', unit='cycle', disable=not progress) as bar:
        def observer(**payload):
            nonlocal previous
            t, raw = int(payload['cycle']), payload['G']
            if t != len(seen) or raw.dtype != np.complex128 or not np.isfinite(raw).all():
                raise ValueError('Invalid cycle coordinate, dtype, or state')
            if t:
                np.testing.assert_array_equal(payload['ordered_site_ids'], word)
            elif np.count_nonzero(raw):
                raise ValueError('Initial state is not maximally mixed')
            seen.append(t)
            # Engine uses raw=2*C-I. RMS Frobenius changes are in physical C units.
            increments.append(np.nan if previous is None else np.linalg.norm(raw-previous)/(2*np.sqrt(n)))
            trace = float(np.trace(raw).real)
            charges.append((n+trace)/2)
            norms.append(np.sqrt(max(float(np.vdot(raw, raw).real)+n+2*trace, 0))/(2*np.sqrt(n)))
            hermiticity.append(float(np.linalg.norm(raw-raw.conj().T)/np.sqrt(n)))
            times.append(time.monotonic()-start)
            previous = raw.copy()
            if t:
                bar.update(1)
        result = model.run_markov_channel(G_history=False, progress=False, cycles=cycles,
            init_mode='maxmix', save=False, n_a=0.5, sequence='raster_y', decoh=True,
            perfect_correction=True, cycle_observer=observer)
    np.testing.assert_array_equal(seen, np.arange(cycles+1))
    np.testing.assert_array_equal(previous, result['G_final'])
    if result['cycles'] != cycles or max(hermiticity) > 1e-12:
        raise ValueError('Cycle horizon or Hermiticity check failed')
    endpoint = result['G_final']*0.5
    endpoint[np.diag_indices(n)] += 0.5
    inc = np.asarray(increments)
    rate = np.full(cycles+1, np.nan)
    valid = (inc[1:-1] > FLOOR) & (inc[2:] > FLOOR)
    rate[2:][valid] = -np.log(inc[2:][valid]/inc[1:-1][valid])
    return dict(cycle=np.asarray(seen), successive_covariance_rms=inc,
                global_charge=np.asarray(charges), correlation_rms=np.asarray(norms),
                elapsed_seconds=np.asarray(times), hermiticity_rms=np.asarray(hermiticity),
                above_numerical_floor=np.isfinite(inc) & (inc > FLOOR),
                local_log_decay_rate=rate, C_final=endpoint, raster_site_word=word)


def verified_complete(folder, cfg, hashes):
    try:
        receipt = json.loads((folder/'completion.json').read_text())
        path = folder/'dynamics.npz'
        if (receipt['status'] != 'complete' or receipt['config'] != cfg
            or receipt['sources'] != hashes or receipt['result_filename'] != path.name
            or receipt['result_bytes'] != path.stat().st_size or receipt['result_sha256'] != sha(path)):
            return False
        with np.load(path, allow_pickle=False) as data:
            return bool(json.loads(str(data['config_json'])) == cfg
                and np.array_equal(data['cycle'], np.arange(cfg['cycles']+1))
                and data['C_final'].shape == (2*cfg['Nx']*cfg['Ny'],)*2
                and np.isfinite(data['C_final']).all())
    except (OSError, KeyError, ValueError, EOFError):
        return False


def publish(folder, cfg, hashes, arrays, diagnostics, reference):
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder/'dynamics.tmp.npz'
    np.savez_compressed(tmp, **arrays, config_json=np.asarray(json.dumps(cfg, sort_keys=True)))
    with np.load(tmp, allow_pickle=False) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    tmp.replace(folder/'dynamics.npz')
    path = folder/'dynamics.npz'
    atomic_json(folder/'completion.json', dict(status='complete', config=cfg, sources=hashes,
        diagnostics=diagnostics, spectral_reference=reference, result_filename=path.name,
        result_bytes=path.stat().st_size, result_sha256=sha(path)))
    assert verified_complete(folder, cfg, hashes)


def run_case(root, alpha, ny):
    cfg, hashes = config(alpha, ny), sources()
    folder = root/f'alpha{alpha}_Ny{ny:03d}'
    if verified_complete(folder, cfg, hashes):
        print('[skip verified]', folder.name, flush=True)
        return
    ref = REFERENCE/folder.name
    if not spectral.verified_complete(ref, spectral.config(alpha, ny), spectral.sources()):
        raise RuntimeError('Spectral reference no longer matches its source/configuration/checksum')
    receipt = json.loads((ref/'completion.json').read_text())
    reference = dict(path=str(ref), receipt_sha256=sha(ref/'completion.json'),
        result_sha256=sha(ref/'spectrum.npz'), covariance_gap=receipt['diagnostics']['covariance_gap_raw'])
    print('[configuration]', json.dumps(cfg), flush=True)
    print('[output]', folder, '[spectral gap]', reference['covariance_gap'], flush=True)
    start = time.monotonic()
    model = spectral.make_model(cfg)
    arrays = collect(model, cycles=cfg['cycles'])
    print('[endpoint eigensolver start]', flush=True)
    occupations = np.linalg.eigvalsh(arrays['C_final'])
    if occupations[0] < -1e-10 or occupations[-1] > 1+1e-10:
        raise ValueError('Endpoint occupations are outside [0,1]')
    arrays['endpoint_occupations'] = occupations
    arrays['spectral_covariance_gap'] = np.asarray(reference['covariance_gap'])
    arrays['spectral_reference_decay'] = np.exp(-reference['covariance_gap']*np.arange(cfg['cycles']+1))
    diagnostics = dict(elapsed_seconds=time.monotonic()-start, cpu_affinity=sorted(os.sched_getaffinity(0)),
        final_successive_covariance_rms=float(arrays['successive_covariance_rms'][-1]),
        minimum_occupation=float(occupations[0]), maximum_occupation=float(occupations[-1]),
        completed_utc=datetime.now(timezone.utc).isoformat(), source_entry_point=cfg['canonical_source'])
    if sources() != hashes:
        raise RuntimeError('Source files changed during the case')
    publish(folder, cfg, hashes, arrays, diagnostics, reference)
    print('[complete]', folder.name, json.dumps(diagnostics), flush=True)


def inventory(root):
    hashes = sources()
    return [dict(alpha=a, Ny=n, complete=verified_complete(root/f'alpha{a}_Ny{n:03d}', config(a,n), hashes))
            for a in (1,3) for n in SIZES]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['launch','worker','case','report'])
    p.add_argument('--root', type=Path)
    p.add_argument('--cpus', default='0,7')
    p.add_argument('--cpu', type=int)
    p.add_argument('--alpha', type=int, choices=[1,3])
    p.add_argument('--ny', type=int, choices=SIZES)
    args=p.parse_args()
    if args.mode == 'launch':
        cpus=[int(x) for x in args.cpus.split(',')]
        if len(cpus)!=2 or len(set(cpus))!=2 or not set(cpus)<=os.sched_getaffinity(0):
            p.error('Specify two distinct allowed CPUs after checking utilization')
        root=(args.root or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
        root.mkdir(parents=True, exist_ok=True)
        jobs=[]
        for a,cpu in zip((1,3),cpus):
            session=f'channel60_a{a}_{root.name}'
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--root',str(root),'--alpha',str(a),'--cpu',str(cpu)]
            shell='set -o pipefail; '+shlex.join(cmd)+' 2>&1 | tee -a '+shlex.quote(str(root/f'worker_alpha{a}.log'))
            subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(HERE),'bash','-lc',shell],check=True)
            jobs.append(dict(alpha=a,cpu=cpu,session=session,command=cmd))
        atomic_json(root/'launch.json',dict(jobs=jobs,sources=sources(),cycles=60,root=str(root)))
        print(json.dumps(dict(root=str(root),jobs=jobs),indent=2))
        return 0
    if args.root is None:
        p.error('--root is required')
    args.root=args.root.resolve()
    if args.mode == 'report':
        rows=inventory(args.root)
        print(json.dumps(dict(complete=sum(r['complete'] for r in rows),total=16,cases=rows),indent=2))
        return 0
    if args.cpu is None or args.alpha is None:
        p.error('--cpu and --alpha are required')
    os.sched_setaffinity(0,{args.cpu})
    os.nice(10)
    args.root.mkdir(parents=True,exist_ok=True)
    if args.mode == 'case':
        if args.ny is None:
            p.error('--ny is required')
        run_case(args.root,args.alpha,args.ny)
        return 0
    completed=0
    failures=[]
    hashes=sources()
    for ny in tqdm(SIZES,desc=f'alpha={args.alpha} cases',unit='case'):
        label=f'alpha{args.alpha}_Ny{ny:03d}'
        if verified_complete(args.root/label,config(args.alpha,ny),hashes):
            print('[skip verified]',label,flush=True)
            completed+=1
            continue
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'case','--root',str(args.root),
             '--alpha',str(args.alpha),'--ny',str(ny),'--cpu',str(args.cpu)]
        with (args.root/f'{label}.log').open('a',buffering=1) as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode==0 and verified_complete(args.root/label,config(args.alpha,ny),hashes):
            completed+=1
        else:
            failures.append(label)
        print(f'[inventory] complete={completed}, failed={len(failures)}, unvisited={8-completed-len(failures)}',flush=True)
    atomic_json(args.root/f'worker_alpha{args.alpha}.json',dict(complete=completed,failures=failures))
    return int(bool(failures))


if __name__=='__main__':
    raise SystemExit(main())
