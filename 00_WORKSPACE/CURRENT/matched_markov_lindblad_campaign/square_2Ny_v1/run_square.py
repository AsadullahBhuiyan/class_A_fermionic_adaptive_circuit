"""Queued square hard-wall sweep, Nx=Ny and exactly 2Ny canonical cycles."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from tqdm.auto import tqdm

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT/'explicit_60cycle_v1'))
import run_dynamics as dynamics

spectral=dynamics.spectral
SIZES=dynamics.SIZES
sha=dynamics.sha
atomic_json=dynamics.atomic_json
AFTER=PROJECT/'explicit_60cycle_v1/results/20260929T193023Z'


def config(alpha, size):
    cfg=dynamics.config(alpha,size)
    cfg.update(revision='hard_wall_square_2Ny_v1',Nx=size,Ny=size,cycles=2*size,
               walls=[size//4,(3*size)//4],wall_scaling='floor(Nx/4),floor(3Nx/4),inclusive')
    return cfg


def spectral_config(alpha,size):
    cfg=spectral.config(alpha,size)
    cfg.update(revision='hard_wall_square_spectral_v1',Nx=size,Ny=size,
               walls=[size//4,(3*size)//4],wall_scaling='floor(Nx/4),floor(3Nx/4),inclusive')
    return cfg


def sources():
    result=dynamics.sources()
    result[str(Path(__file__).relative_to(spectral.REPO))]=sha(__file__)
    return result


def case_dir(root,alpha,size):
    return root/f'alpha{alpha}_L{size:03d}'


def verified(root,alpha,size):
    folder=case_dir(root,alpha,size)
    return (dynamics.verified_complete(folder,config(alpha,size),sources())
            and spectral.verified_complete(folder/'spectral',spectral_config(alpha,size),sources()))


def run_case(root,alpha,size):
    folder=case_dir(root,alpha,size)
    if verified(root,alpha,size):
        print('[skip verified]',folder.name,flush=True)
        return
    cfg=config(alpha,size)
    hashes=sources()
    print('[configuration]',json.dumps(cfg),flush=True)
    print('[output]',folder,flush=True)
    ref=folder/'spectral'
    if not spectral.verified_complete(ref,spectral_config(alpha,size),hashes):
        print('[square-geometry spectral reference]',flush=True)
        spectral.run_case(ref,spectral_config(alpha,size),hashes)
    if dynamics.verified_complete(folder,cfg,hashes):
        print('[skip verified dynamics]',folder.name,flush=True)
        return
    spec=json.loads((ref/'completion.json').read_text())
    reference=dict(path=str(ref),receipt_sha256=sha(ref/'completion.json'),
        result_sha256=sha(ref/'spectrum.npz'),covariance_gap=spec['diagnostics']['covariance_gap_raw'])
    start=time.monotonic()
    model=spectral.make_model(cfg)
    arrays=dynamics.collect(model,cycles=cfg['cycles'])
    print('[endpoint occupation eigensolver start]',flush=True)
    occupations=np.linalg.eigvalsh(arrays['C_final'])
    if occupations[0]<-1e-10 or occupations[-1]>1+1e-10:
        raise ValueError('Endpoint covariance is outside the occupation interval')
    arrays['endpoint_occupations']=occupations
    gap=reference['covariance_gap']
    arrays['spectral_covariance_gap']=np.asarray(np.inf if gap is None else gap)
    if gap is None:
        arrays['spectral_reference_decay']=np.r_[1.,np.zeros(cfg['cycles'])]
    else:
        arrays['spectral_reference_decay']=np.exp(-gap*np.arange(cfg['cycles']+1))
    diagnostics=dict(elapsed_seconds=time.monotonic()-start,cpu_affinity=sorted(os.sched_getaffinity(0)),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        final_successive_covariance_rms=float(arrays['successive_covariance_rms'][-1]),
        spectral_gap_status=spec['diagnostics']['gap_status'],
        minimum_occupation=float(occupations[0]),maximum_occupation=float(occupations[-1]))
    if sources()!=hashes:
        raise RuntimeError('Sources changed during the calculation')
    dynamics.publish(folder,cfg,hashes,arrays,diagnostics,reference)
    print('[complete]',folder.name,json.dumps(diagnostics),flush=True)


def predecessor_complete(root):
    paths=[root/f'worker_alpha{a}.json' for a in (1,3)]
    if not all(p.exists() for p in paths):
        return False
    states=[json.loads(p.read_text()) for p in paths]
    if any(s['complete']!=8 or s['failures'] for s in states):
        raise RuntimeError('Preceding 60-cycle sweep has failed/incomplete cases; queue stopped')
    if not all(row['complete'] for row in dynamics.inventory(root)):
        raise RuntimeError('Preceding outputs failed verification; queue stopped')
    return True


def worker(root,alpha,cpu):
    os.sched_setaffinity(0,{cpu})
    os.nice(10)
    completed=0
    failures=[]
    for size in tqdm(SIZES,desc=f'square alpha={alpha}',unit='case'):
        folder=case_dir(root,alpha,size)
        if verified(root,alpha,size):
            print('[skip verified]',folder.name,flush=True)
            completed+=1
            continue
        cmd=[sys.executable,'-u',str(Path(__file__)),'case','--root',str(root),
             '--alpha',str(alpha),'--size',str(size),'--cpu',str(cpu)]
        with (root/f'{folder.name}.log').open('a',buffering=1) as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode==0 and verified(root,alpha,size):
            completed+=1
        else:
            failures.append(folder.name)
        print(f'[inventory] completed={completed}, failed={len(failures)}, unvisited={8-completed-len(failures)}',flush=True)
    atomic_json(root/f'worker_alpha{alpha}.json',dict(complete=completed,failures=failures))
    return int(bool(failures))


def queue(root,after,cpus):
    print('[waiting for preceding sweep]',after,flush=True)
    atomic_json(root/'queue_status.json',dict(status='waiting',after=str(after)))
    deadline=time.monotonic()+24*3600
    while not predecessor_complete(after):
        if time.monotonic()>deadline:
            raise TimeoutError('Preceding sweep did not complete in 24 hours')
        time.sleep(15)
    print('[preceding sweep verified; starting square workers]',flush=True)
    atomic_json(root/'queue_status.json',dict(status='running',cpus=cpus))
    handles=[]
    processes=[]
    try:
        for alpha,cpu in zip((1,3),cpus):
            handle=(root/f'worker_alpha{alpha}.log').open('a',buffering=1)
            handles.append(handle)
            cmd=[sys.executable,'-u',str(Path(__file__)),'worker','--root',str(root),
                 '--alpha',str(alpha),'--cpu',str(cpu)]
            processes.append(subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT))
        codes=[process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    if any(codes):
        atomic_json(root/'queue_status.json',dict(status='failed',returncodes=codes))
        return 1
    result=subprocess.run([sys.executable,str(HERE/'analyze_square.py'),'--root',str(root)])
    atomic_json(root/'queue_status.json',dict(status='complete' if result.returncode==0 else 'analysis_failed',
                                            analysis_returncode=result.returncode))
    return result.returncode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['launch','queue','worker','case','report'])
    p.add_argument('--root',type=Path)
    p.add_argument('--after',type=Path,default=AFTER)
    p.add_argument('--cpus',default='0,7')
    p.add_argument('--cpu',type=int)
    p.add_argument('--alpha',type=int,choices=[1,3])
    p.add_argument('--size',type=int,choices=SIZES)
    args=p.parse_args()
    # Only the coordinator needs two cores. Children inherit the worker's
    # single-core affinity and must validate only their requested --cpu.
    cpus=None
    if args.mode in ('launch','queue'):
        try:
            cpus=[int(c) for c in args.cpus.split(',')]
        except ValueError:
            p.error('--cpus must contain two integer CPU IDs')
        if len(cpus)!=2 or len(set(cpus))!=2 or not set(cpus)<=os.sched_getaffinity(0):
            p.error('Two distinct allowed CPU IDs required')
    if args.mode=='launch':
        root=(args.root or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
        root.mkdir(parents=True,exist_ok=True)
        session=f'square2Ny_{root.name}'
        cmd=[sys.executable,'-u',str(Path(__file__)),'queue','--root',str(root),
             '--after',str(args.after.resolve()),'--cpus',args.cpus]
        shell=shlex.join(cmd)+' > '+shlex.quote(str(root/'queue.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(HERE),'bash','-lc',shell],check=True)
        atomic_json(root/'launch.json',dict(session=session,command=cmd,after=str(args.after.resolve()),
            cpus=cpus,sources=sources(),cases=[config(a,n) for a in (1,3) for n in SIZES]))
        print(json.dumps(dict(root=str(root),session=session,after=str(args.after.resolve()),cpus=cpus),indent=2))
        return 0
    if args.root is None:
        p.error('--root is required')
    root=args.root.resolve()
    if args.mode=='report':
        rows=[dict(alpha=a,size=n,complete=verified(root,a,n)) for a in (1,3) for n in SIZES]
        print(json.dumps(dict(complete=sum(r['complete'] for r in rows),total=16,cases=rows),indent=2))
        return 0
    if args.mode=='queue':
        try:
            return queue(root,args.after.resolve(),cpus)
        except Exception as exc:
            atomic_json(root/'queue_status.json',dict(status='failed',error=str(exc)))
            raise
    if args.cpu is None or args.alpha is None:
        p.error('--cpu and --alpha required')
    if args.cpu not in os.sched_getaffinity(0):
        p.error('--cpu is outside the inherited allowed CPU set')
    if args.mode=='worker':
        return worker(root,args.alpha,args.cpu)
    if args.size is None:
        p.error('--size required')
    os.sched_setaffinity(0,{args.cpu})
    os.nice(10)
    run_case(root,args.alpha,args.size)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
