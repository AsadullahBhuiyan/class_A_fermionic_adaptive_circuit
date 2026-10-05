"""Exact raster-y channel dynamics and stationary occupation gaps; no sampling."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import eigvalsh
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'fixed_width_alpha_spectral_v1'))
import run_scan as reference
spectral = reference.backend.spectral
sha, atomic_json = spectral.sha, spectral.atomic_json
REFERENCE = PROJECT / 'fixed_width_alpha_spectral_v1/results/20261001T220546Z'
SIZES = (20, 40, 60, 80, 100)
ALPHAS = (1, 3)
TOL = 1e-10


def config(alpha, ny):
    cfg = reference.config(0 if alpha == 1 else 20, ny)
    cfg.update(revision='hard_wall_stationary_purity_gap_v1', explicit_dynamics=True,
               canonical_source='classA_U1FGTN.run_markov_channel',
               init_mode='maxmix', n_a=0.5, cycles=61,
               observation_cycles=[0, 1, 5, 10, 20, 40, 60, 61],
               stationarity_tolerance=TOL,
               purity_gap_definition='min(abs(1-2*eigenvalues(C)))',
               half_filling_distance_definition='min(abs(eigenvalues(C)-0.5))',
               twirl='separate postprocessed y-translation average, not the actual fixed point')
    return cfg


def sources():
    result = reference.sources()
    result[str(Path(__file__).relative_to(spectral.REPO))] = sha(__file__)
    return result


def case_dir(root, alpha, ny):
    return root / f'alpha{alpha}_Ny{ny:03d}'


def occupation_spectra(correlation, nx, ny, walls):
    """Full spectrum via exact disconnected hard-wall blocks, plus twirl spectrum.

    The momentum diagonal of F C F^dagger equals the exact translation twirl;
    off-diagonal momentum blocks are discarded only in this SECOND estimator.
    """
    n = 2*nx*ny
    if correlation.shape != (n, n) or not np.isfinite(correlation).all():
        raise ValueError('Invalid correlation matrix')
    herm = float(np.linalg.norm(correlation-correlation.conj().T))
    if herm > TOL:
        raise ValueError(f'Non-Hermitian correlation matrix: {herm}')
    x = (np.arange(n)//2) % nx
    inside = np.flatnonzero((x >= walls[0]) & (x <= walls[1]))
    outside = np.flatnonzero((x < walls[0]) | (x > walls[1]))
    cross = float(np.linalg.norm(correlation[np.ix_(inside, outside)]))
    if cross > TOL:
        raise ValueError(f'Hard-wall block decomposition not valid: {cross}')
    interior = eigvalsh(correlation[np.ix_(inside, inside)], check_finite=False)
    exterior = eigvalsh(correlation[np.ix_(outside, outside)], check_finite=False)
    full = np.sort(np.concatenate([interior, exterior]))
    block = correlation.reshape(ny, 2*nx, ny, 2*nx)
    fourier = np.fft.ifft(np.fft.fft(block, axis=0, norm='ortho'), axis=2, norm='ortho')
    ky_blocks = np.asarray([fourier[k, :, k, :] for k in range(ny)])
    ky_blocks = (ky_blocks+ky_blocks.conj().transpose(0, 2, 1))*0.5
    ky_values = np.linalg.eigvalsh(ky_blocks)
    if min(full.min(), ky_values.min()) < -TOL or max(full.max(), ky_values.max()) > 1+TOL:
        raise ValueError('Occupation outside physical interval [0,1]')
    # Discrete y-translation mismatch, without changing the input matrix.
    translated = np.roll(np.roll(block, 1, axis=0), 1, axis=2).reshape(n, n)
    trans = float(np.linalg.norm(translated-correlation)/max(np.linalg.norm(correlation), 1e-300))
    return dict(full=full, interior=interior, exterior=exterior, ky=ky_values,
                translation_residual=trans, cross_block_residual=cross, hermiticity=herm)


def collect(model, cfg, progress=True):
    n = 2*cfg['Nx']*cfg['Ny']
    word = np.asarray([x+cfg['Nx']*y for x, y in spectral.raster_word(model)])
    cycles, changes, charges, obs_cycles, spectra, trans = [], [], [], [], [], []
    previous = None
    endpoint60 = None
    start = time.monotonic()
    with tqdm(total=cfg['cycles'], desc='canonical channel cycles', unit='cycle', disable=not progress) as bar:
        def observer(**payload):
            nonlocal previous, endpoint60
            t, raw = int(payload['cycle']), payload['G']
            if t != len(cycles) or raw.dtype != np.complex128 or not np.isfinite(raw).all():
                raise ValueError('Invalid cycle, dtype, or state')
            if t:
                np.testing.assert_array_equal(payload['ordered_site_ids'], word)
            elif np.count_nonzero(raw):
                raise ValueError('Expected maximally mixed initialization')
            cycles.append(t)
            changes.append(np.nan if previous is None else float(np.linalg.norm(raw-previous)/2))
            charges.append(float((n+np.trace(raw).real)/2))
            previous = raw.copy()
            if t in cfg['observation_cycles']:
                print(f'[occupation eigensolvers] cycle={t}', flush=True)
                c = raw*0.5 + np.eye(n)*0.5
                result = occupation_spectra(c, cfg['Nx'], cfg['Ny'], cfg['walls'])
                obs_cycles.append(t)
                spectra.append(result)
                trans.append(result['translation_residual'])
                print('[gaps]', json.dumps(dict(cycle=t,
                    purity_full=float(np.min(abs(1-2*result['full']))),
                    purity_twirl=float(np.min(abs(1-2*result['ky']))))), flush=True)
                if t == cfg['cycles']-1:
                    endpoint60 = c.copy()
            if t:
                bar.update()
        result = model.run_markov_channel(G_history=False, progress=False, cycles=cfg['cycles'],
            init_mode='maxmix', save=False, n_a=0.5, sequence='raster_y', decoh=True,
            perfect_correction=True, cycle_observer=observer)
    np.testing.assert_array_equal(previous, result['G_final'])
    full = np.asarray([s['full'] for s in spectra])
    ky = np.asarray([s['ky'] for s in spectra])
    pure = np.min(abs(1-2*full), axis=1)
    twirl = np.min(abs(1-2*ky), axis=(1, 2))
    converged = bool(max(changes[-5:]) < cfg['stationarity_tolerance']
                     and abs(pure[-1]-pure[-2]) < 2*cfg['stationarity_tolerance']
                     and abs(twirl[-1]-twirl[-2]) < 2*cfg['stationarity_tolerance'])
    if endpoint60 is None:
        raise ValueError('Missing penultimate convergence-check snapshot')
    arrays = dict(cycles=np.asarray(cycles), successive_frobenius_change=np.asarray(changes),
        global_charge=np.asarray(charges), observation_cycles=np.asarray(obs_cycles),
        occupations_full=full, occupations_interior=np.asarray([s['interior'] for s in spectra]),
        occupations_exterior=np.asarray([s['exterior'] for s in spectra]), occupations_ky_twirl=ky,
        ky=2*np.pi*np.fft.fftfreq(cfg['Ny']), purity_gap_full=pure, purity_gap_twirl=twirl,
        half_filling_distance_full=pure/2, half_filling_distance_twirl=twirl/2,
        translation_residual=np.asarray(trans), C_penultimate=endpoint60,
        C_final=previous*0.5+np.eye(n)*0.5)
    diagnostics = dict(converged=converged, final_frobenius_change=changes[-1],
        last_five_max_change=max(changes[-5:]), purity_gap_full=float(pure[-1]),
        purity_gap_twirl=float(twirl[-1]), half_filling_distance_full=float(pure[-1]/2),
        half_filling_distance_twirl=float(twirl[-1]/2),
        final_translation_residual=trans[-1], elapsed_seconds=time.monotonic()-start,
        purity_full_below_resolution=bool(pure[-1] < 2*TOL),
        purity_twirl_below_resolution=bool(twirl[-1] < 2*TOL))
    return arrays, diagnostics


def verified(root, alpha, ny):
    folder = case_dir(root, alpha, ny)
    try:
        receipt = json.loads((folder/'completion.json').read_text())
        result = folder/'dynamics.npz'
        return (receipt['status'] == 'complete' and receipt['config'] == config(alpha, ny)
                and receipt['sources'] == sources() and receipt['result_filename'] == result.name
                and receipt['result_bytes'] == result.stat().st_size
                and receipt['result_sha256'] == sha(result))
    except (OSError, KeyError, ValueError):
        return False


def run_case(root, alpha, ny):
    if verified(root, alpha, ny):
        print('[skip verified]', alpha, ny, flush=True)
        return
    index = 0 if alpha == 1 else 20
    if not reference.verified(REFERENCE, index, ny):
        raise ValueError('Spectral reference source/configuration/checksum mismatch')
    ref_folder = reference.folder(REFERENCE, index, ny)
    ref_receipt = json.loads((ref_folder/'completion.json').read_text())
    ref = dict(folder=str(ref_folder), receipt_sha256=sha(ref_folder/'completion.json'),
               spectrum_sha256=sha(ref_folder/'spectrum.npz'),
               channel_gap=ref_receipt['diagnostics']['covariance_multiplier_gap'])
    cfg, hashes = config(alpha, ny), sources()
    print('[configuration]', json.dumps(cfg), flush=True)
    print('[reference]', json.dumps(ref), flush=True)
    folder = case_dir(root, alpha, ny)
    folder.mkdir(parents=True, exist_ok=True)
    model = spectral.make_model(cfg)
    arrays, diagnostics = collect(model, cfg)
    if sources() != hashes:
        raise ValueError('Source changed during calculation')
    tmp = folder/'dynamics.tmp.npz'
    np.savez_compressed(tmp, **arrays, config_json=np.asarray(json.dumps(cfg, sort_keys=True)))
    with np.load(tmp, allow_pickle=False) as data:
        for k, v in arrays.items():
            np.testing.assert_array_equal(data[k], v)
    tmp.replace(folder/'dynamics.npz')
    path = folder/'dynamics.npz'
    receipt = dict(status='complete' if diagnostics['converged'] else 'not_converged',
        config=cfg, sources=hashes, diagnostics=diagnostics, spectral_reference=ref,
        result_filename=path.name, result_bytes=path.stat().st_size, result_sha256=sha(path))
    atomic_json(folder/'completion.json', receipt)
    print('[result]', json.dumps(receipt['diagnostics']), flush=True)
    if not diagnostics['converged']:
        raise RuntimeError('Finite-time results saved, but NOT accepted as a stationary state')
    assert verified(root, alpha, ny)


def analyze(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, inputs = [], {}
    for alpha in ALPHAS:
        for ny in SIZES:
            if not verified(root, alpha, ny):
                continue
            folder = case_dir(root, alpha, ny)
            record = json.loads((folder/'completion.json').read_text())
            d = record['diagnostics']
            rows.append(dict(alpha_1=alpha, Nx=20, Ny=ny,
                channel_gap=record['spectral_reference']['channel_gap'],
                purity_gap_full=d['purity_gap_full'], purity_gap_twirl=d['purity_gap_twirl'],
                half_filling_distance_full=d['half_filling_distance_full'],
                half_filling_distance_twirl=d['half_filling_distance_twirl'],
                stationarity_residual=d['final_frobenius_change']))
            inputs[str(folder/'completion.json')] = sha(folder/'completion.json')
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    atomic_json(out/'status.json', dict(complete=len(rows), total=10, cases=rows))
    if not rows:
        return
    with (out/'gaps.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.family':'CMU Sans Serif', 'font.size':8,
                         'xtick.direction':'in', 'ytick.direction':'in'})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.65))
    for ax, key, label, letter in zip(axes,
            ('channel_gap', 'purity_gap_full', 'purity_gap_twirl'),
            (r'$g_C$', r'$\Delta_{\mathrm{pur}}$ (actual state)',
             r'$\Delta_{\mathrm{pur}}$ ($y$-twirled)'), 'abc'):
        for alpha, color, marker, ls in ((1,'#2468ad','o','-'), (3,'#c0392b','^',':')):
            selected = [r for r in rows if r['alpha_1'] == alpha]
            ax.plot([r['Ny'] for r in selected], [r[key] for r in selected],
                color=color, marker=marker, ls=ls, ms=3, mfc='white', label=rf'$\alpha_1={alpha}$')
        ax.set(xlabel=r'$N_y$', ylabel=label)
        ax.tick_params(top=True, right=True)
        ax.text(-.12,1.04,f'({letter})',transform=ax.transAxes)
        ax.axhline(0, color='gray', ls='--', lw=.5)
    axes[0].legend(frameon=False)
    fig.tight_layout(pad=.8)
    for ext in ('png','pdf'):
        fig.savefig(out/f'channel_and_stationary_purity_gaps.{ext}', dpi=300)
    plt.close(fig)
    (out/'caption.txt').write_text(
        'Exact outcome-averaged channel, no trajectory samples or error bars. Nx=20; '
        'Ny=20,40,60,80,100; alpha_1=1,3; alpha_2=30; nshell=1; hard-wall support '
        'truncation at inclusive x=5,15; all slabs active; periodic; X trials; zero twist; '
        'complex128; perfect correction plus measurement dephasing; raster-y Ap/Am/Bp/Bm. '
        'Maximally mixed initial state, 60 cycles plus a 61st convergence check. '
        'Accepted cases have last five absolute Frobenius increments <1e-10 and stable '
        'endpoint occupation gaps. Purity gap=min|1-2n|, twice distance to half filling. '
        'The y-twirl is postprocessing, not assumed stationary under the ordered channel. '
        'All orbitals/slabs included, no covariance-history averaging, no fits. '
        'Finite-size nonzero minima do not establish a thermodynamic gap. '
        f'This plot contains {len(rows)}/10 verified cases.\n')
    atomic_json(out/'manifest.json', dict(inputs=inputs, sources=sources(),
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['case','queue','launch','report'])
    p.add_argument('--root', type=Path)
    p.add_argument('--alpha', type=int, choices=ALPHAS)
    p.add_argument('--ny', type=int, choices=SIZES)
    p.add_argument('--cpu', type=int)
    p.add_argument('--cpus', default='9,10')
    args = p.parse_args()
    root = (args.root or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'launch':
        cpus = [int(c) for c in args.cpus.split(',')]
        if len(cpus)!=2 or len(set(cpus))!=2 or not set(cpus)<=os.sched_getaffinity(0):
            p.error('Need two distinct available CPUs')
        session = 'stationary_purity_'+root.name
        cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'queue','--root',str(root),'--cpus',args.cpus]
        shell = 'set -o pipefail; '+shlex.join(cmd)+' 2>&1 | tee -a '+shlex.quote(str(root/'queue.log'))
        atomic_json(root/'launch.json', dict(session=session,command=cmd,sources=sources(),
                    cpus=cpus,cases=[config(a,n) for a in ALPHAS for n in SIZES]))
        subprocess.run(['tmux','new-session','-d','-s',session,'bash','-lc',shell], check=True)
        print(json.dumps(dict(root=str(root),session=session), indent=2))
    elif args.mode == 'queue':
        from concurrent.futures import ThreadPoolExecutor
        def worker(alpha, cpu):
            failures = []
            for ny in tqdm(SIZES, desc=f'alpha={alpha}', unit='case'):
                if verified(root, alpha, ny):
                    continue
                cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'case','--root',str(root),
                       '--alpha',str(alpha),'--ny',str(ny),'--cpu',str(cpu)]
                with (root/f'alpha{alpha}_Ny{ny:03d}.log').open('a', buffering=1) as log:
                    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if r.returncode or not verified(root,alpha,ny):
                    failures.append(ny)
                atomic_json(root/f'worker_alpha{alpha}.json', dict(failures=failures,
                    complete=sum(verified(root,alpha,n) for n in SIZES), total=5))
            return failures
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(worker,a,c) for a,c in zip(ALPHAS,map(int,args.cpus.split(',')))]
            failures = [j.result() for j in jobs]
        analyze(root)
        print('[queue finished]', json.dumps(dict(failures=failures)), flush=True)
        return int(any(failures))
    elif args.mode == 'case':
        if args.alpha is None or args.ny is None or args.cpu is None:
            p.error('case requires alpha, ny, cpu')
        os.sched_setaffinity(0,{args.cpu})
        os.nice(10)
        run_case(root,args.alpha,args.ny)
    else:
        analyze(root)
        print((root/'analysis/status.json').read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
