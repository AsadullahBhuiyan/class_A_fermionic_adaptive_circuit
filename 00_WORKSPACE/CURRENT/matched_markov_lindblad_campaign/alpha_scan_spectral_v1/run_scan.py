"""CPU spectral gaps: alpha_1=1.0:0.1:3.0, square L=20,40,60."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
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
import traceback
import numpy as np
import psutil
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'square_large_spectral_v1'))
import run_large as backend

SIZES = (20,40,60)
ALPHAS = tuple((10+i)/10 for i in range(21))
sha, atomic_json = backend.sha, backend.atomic_json


def config(index, size):
    if index not in range(21):
        raise ValueError('alpha index must be 0..20')
    cfg = backend.config(ALPHAS[index], size)
    cfg.update(revision='hard_wall_square_alpha_scan_spectral_v1', alpha_index=index,
        zero_bloch_norm_convention='canonical nmag=1e-15 at exact zeros; alpha=2 not shifted')
    return cfg


def sources():
    hashes = backend.sources()
    hashes[str(Path(__file__).relative_to(backend.spectral.REPO))] = sha(__file__)
    return hashes


def folder(root, index, size):
    return root/f'L{size:03d}_a{index:02d}_{ALPHAS[index]:.1f}'


def verified(root, index, size):
    return backend.spectral.verified_complete(folder(root,index,size),config(index,size),sources())


def tasks():
    # Striding this list across workers balances the expensive L60 cases.
    return [(i,n) for n in reversed(SIZES) for i in range(21)]


def run_case(root, index, size):
    path = folder(root,index,size)
    path.mkdir(parents=True, exist_ok=True)
    if verified(root,index,size):
        print('[skip verified]',path.name,flush=True)
        return
    cfg, hashes = config(index,size), sources()
    available = psutil.virtual_memory().available/2**30
    required = 100*(size/100)**4+8
    if available < required:
        raise MemoryError(f'Required headroom {required:.1f} GiB; available {available:.1f}')
    print('[configuration]',json.dumps(cfg),flush=True)
    print('[resources]',json.dumps(dict(cpu_affinity=sorted(os.sched_getaffinity(0)),
                                      available_gib=available,required_gib=required)),flush=True)
    start = time.monotonic()
    model = backend.spectral.make_model(cfg)
    ow_done = time.monotonic()
    indices, matrices, checks = backend.construct_blocks(model)
    product_done = time.monotonic()
    # Integer diagnostic seed, including noninteger alpha values. Not sampling.
    rng = np.random.default_rng(2026100100 + 100*index + size)
    probes = rng.normal(size=(2*size*size,3))+1j*rng.normal(size=(2*size*size,3))
    probes /= np.linalg.norm(probes,axis=0)
    action_error = float(np.linalg.norm(backend.block_action(indices,matrices,probes)
                                       - backend.spectral.independent_action(model,probes)))
    if action_error > 1e-11:
        raise ValueError('Block product differs from full-system projector action')
    values, dominant, block_details = backend.solve_blocks(model,indices,matrices)
    residual = float(np.linalg.norm(backend.spectral.independent_action(model,dominant['right'][:,None])[:,0]
                                    - dominant['value']*dominant['right']))
    if residual > 1e-10:
        raise ValueError('Independent dominant eigenpair check failed')
    radius = dominant['radius']
    gap = float(-2*np.log(radius)) if radius else None
    details = dict(**checks,blocks=block_details,dominant_sector=dominant['sector'],
        independent_product_action_error=action_error,independent_dominant_residual=residual,
        dominant_residual=dominant['residual'],dominant_left_residual=dominant['left_residual'],
        dominant_left_right_overlap=float(abs(np.vdot(dominant['left'],dominant['right']))),
        spectral_radius=radius,covariance_gap_raw=gap,covariance_multiplier_gap=float(1-radius**2),
        gap_status=backend.spectral.classify_radius(radius),unit_modulus_tolerance=backend.spectral.UNIT_TOL,
        ow_build_seconds=ow_done-start,product_seconds=product_done-ow_done,
        eigensolver_seconds=sum(b['eigensolver_seconds'] for b in block_details),
        elapsed_seconds=time.monotonic()-start,numpy_version=np.__version__,
        scipy_version=backend.spectral.scipy.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)),completed_utc=datetime.now(timezone.utc).isoformat())
    if sources() != hashes:
        raise RuntimeError('Source files changed during calculation')
    temporary = path/'spectrum.tmp.npz'
    np.savez_compressed(temporary,eigenvalues=values,dominant_eigenvector=dominant['right'],
        dominant_left_eigenvector=dominant['left'],dominant_eigenvalue=dominant['value'],
        spectral_radius=radius,covariance_gap_raw=gap if gap is not None else np.inf,
        covariance_multiplier_gap=1-radius**2,interior_indices=indices[0],exterior_indices=indices[1],
        config_json=np.asarray(json.dumps(cfg,sort_keys=True)),diagnostics_json=np.asarray(json.dumps(details)))
    with np.load(temporary,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['eigenvalues'],values)
    result = path/'spectrum.npz'
    temporary.replace(result)
    atomic_json(path/'completion.json',dict(status='complete',config=cfg,sources=hashes,diagnostics=details,
        result_filename=result.name,result_bytes=result.stat().st_size,result_sha256=sha(result)))
    assert verified(root,index,size)
    print('[complete]',path.name,json.dumps(details),flush=True)


def inventory(root):
    return [dict(alpha_index=i,alpha_1=ALPHAS[i],size=n,complete=verified(root,i,n)) for i,n in tasks()]


def worker(root, worker_index, count, cpu):
    os.sched_setaffinity(0,{cpu})
    os.nice(10)
    failures = []
    assigned = tasks()[worker_index::count]
    completed = 0
    for index,size in tqdm(assigned,desc=f'alpha scan worker {worker_index}',unit='case'):
        if not verified(root,index,size):
            cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'case','--root',str(root),
                   '--index',str(index),'--size',str(size),'--cpu',str(cpu)]
            with (root/(folder(root,index,size).name+'.log')).open('a',buffering=1) as log:
                result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode or not verified(root,index,size):
                failures.append(dict(index=index,size=size,returncode=result.returncode))
                atomic_json(root/f'worker_{worker_index}.json',dict(complete=completed,total=len(assigned),failures=failures))
                continue
        completed += 1
        atomic_json(root/f'worker_{worker_index}.json',dict(complete=completed,total=len(assigned),failures=failures))
    return int(bool(failures))


def analyze(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, inputs = [], {}
    for size in SIZES:
        for index,alpha in enumerate(ALPHAS):
            if not verified(root,index,size):
                raise RuntimeError(f'Incomplete alpha={alpha}, L={size}')
            path = folder(root,index,size)
            details = json.loads((path/'completion.json').read_text())['diagnostics']
            rows.append(dict(alpha_1=alpha,L=size,rho=details['spectral_radius'],
                gap=details['covariance_gap_raw'],status=details['gap_status'],
                residual=details['independent_dominant_residual'],elapsed_seconds=details['elapsed_seconds']))
            inputs[str(path/'completion.json')] = sha(path/'completion.json')
            inputs[str(path/'spectrum.npz')] = sha(path/'spectrum.npz')
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,ax = plt.subplots(figsize=(3.375,2.8))
    for size,color,marker,style in zip(SIZES,('#c0392b','#23934c','#2468ad'),('^','s','o'),(':','--','-')):
        selected = [r for r in rows if r['L']==size]
        ax.plot(ALPHAS,[r['gap'] if r['status']=='resolved_positive' else np.nan for r in selected],
                color=color,marker=marker,ls=style,mfc='white',ms=3,lw=.8,label=str(size))
        for row in selected:
            if row['status'] != 'resolved_positive':
                ax.plot(row['alpha_1'],0,'x',color=color)
    ax.set(xlabel=r'$\alpha_1$',ylabel=r'$\Delta_C=-2\log\rho(A)$ (cycle$^{-1}$)')
    ax.tick_params(top=True,right=True)
    ax.legend(title=r'$N_x=N_y$',frameon=False)
    fig.tight_layout(pad=.7)
    for ext in ('png','pdf'):
        fig.savefig(out/f'channel_gap_vs_alpha.{ext}',dpi=300)
    plt.close(fig)
    with (out/'gaps.csv').open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out/'caption.txt').write_text('Exact covariance-channel spectral gap for square L=20,40,60; '
        'alpha_1=1.0,1.1,...,3.0, alpha_2=30, nshell=1, hard-wall support truncation, '
        'inclusive walls floor(L/4),floor(3L/4), all slabs active, X orbitals, periodic, zero twist, '
        'raster-y Ap/Am/Bp/Bm, complex128, perfect correction and measurement dephasing. '
        'Full spectra of both invariant hard-wall blocks, not trajectory sampling or relaxation fits. '
        'Alpha=2 uses the canonical zero-Bloch-norm prescription without a parameter offset. '
        'Unresolved modes are marked at zero and retain raw rates in the CSV/NPZ. '
        'Lines guide the eye. No full many-body or thermodynamic gap claim.\n')
    atomic_json(out/'manifest.json',dict(input_sha256=inputs,source_sha256=sha(__file__),
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print('[analysis complete]',out,flush=True)


def queue(root,cpus):
    handles,processes = [],[]
    atomic_json(root/'queue_status.json',dict(status='running',cpus=cpus,total=63))
    try:
        for i,cpu in enumerate(cpus):
            handle = (root/f'worker_{i}.log').open('a',buffering=1)
            handles.append(handle)
            cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'worker','--root',str(root),
                   '--worker-index',str(i),'--workers',str(len(cpus)),'--cpu',str(cpu)]
            processes.append(subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
    finally:
        for handle in handles:
            handle.close()
    rows = inventory(root)
    if any(codes) or not all(row['complete'] for row in rows):
        atomic_json(root/'queue_status.json',dict(status='failed',returncodes=codes,cases=rows))
        return 1
    analyze(root)
    atomic_json(root/'queue_status.json',dict(status='complete',complete=63,total=63,cases=rows))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['launch','queue','worker','case','report','analyze'])
    parser.add_argument('--root',type=Path)
    parser.add_argument('--cpus',default='0,7,40')
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--worker-index',type=int)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--index',type=int,choices=range(21))
    parser.add_argument('--size',type=int,choices=SIZES)
    args = parser.parse_args()
    if args.mode in ('launch','queue'):
        cpus = [int(c) for c in args.cpus.split(',')]
        if len(cpus)!=3 or len(set(cpus))!=3 or not set(cpus)<=os.sched_getaffinity(0):
            parser.error('Require three distinct allowed CPUs')
    if args.mode=='launch':
        root = (args.root or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
        root.mkdir(parents=True,exist_ok=True)
        session = 'alpha_gap_'+root.name
        if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:
            parser.error('This queue is already running')
        cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'queue','--root',str(root),'--cpus',args.cpus]
        atomic_json(root/'launch.json',dict(session=session,cpus=cpus,command=cmd,sources=sources(),
                                          cases=[config(i,n) for i,n in tasks()]))
        shell = shlex.join(cmd)+' > '+shlex.quote(str(root/'queue.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(HERE),'bash','-lc',shell],check=True)
        print(json.dumps(dict(root=str(root),session=session,cpus=cpus,total=63),indent=2))
        return 0
    if args.root is None:
        parser.error('--root required')
    root = args.root.resolve()
    if args.mode=='report':
        rows = inventory(root)
        print(json.dumps(dict(complete=sum(row['complete'] for row in rows),total=63,cases=rows),indent=2))
        return 0
    if args.mode=='analyze':
        analyze(root)
        return 0
    if args.mode=='queue':
        try:
            return queue(root,cpus)
        except Exception:
            atomic_json(root/'queue_status.json',dict(status='failed',traceback=traceback.format_exc()))
            raise
    if args.cpu is None or args.cpu not in os.sched_getaffinity(0):
        parser.error('Require an allowed --cpu')
    if args.mode=='worker':
        if args.worker_index is None or not 0<=args.worker_index<args.workers:
            parser.error('Require valid --worker-index and --workers')
        return worker(root,args.worker_index,args.workers,args.cpu)
    if args.index is None or args.size is None:
        parser.error('Require --index and --size')
    os.sched_setaffinity(0,{args.cpu})
    os.nice(10)
    try:
        run_case(root,args.index,args.size)
    except Exception:
        path = folder(root,args.index,args.size)
        path.mkdir(parents=True,exist_ok=True)
        atomic_json(path/'failure.json',dict(traceback=traceback.format_exc()))
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
