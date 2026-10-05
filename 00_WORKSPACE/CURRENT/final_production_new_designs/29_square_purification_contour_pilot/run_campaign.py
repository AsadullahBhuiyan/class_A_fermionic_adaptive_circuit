"""Two one-trajectory square purification tasks, exact five-cycle resume."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
from classA_U1FGTN_gpu import classA_U1FGTN_gpu
from contour_observer import observe_covariance, allocate
from io_utils import sha, pair_verified, publish_pair, capture_rng, restore_rng

SOURCE_FILES = ('run_campaign.py','contour_observer.py','io_utils.py',
                'src/classA_U1FGTN_gpu.py','src/occupied_frame_gpu.py')


def default_config():
    return dict(sampling_revision='square_full_measurement_purification_l30-40_s1_t60_v1',
        root_seed=2026092929,sizes=[30,40],samples=1,cycles=60,segment_cycles=5,
        DW=True,dw_truncation=True,meas_slab_only=False,triv_region_local_mode=False,
        alpha_1=1.,alpha_2=30.,nshell=1,trial_orbitals='X',filling_fraction=.5,
        init_mode='maxmix',n_a=.5,sequence='raster_y',perfect_correction=True,
        postselect=False,state_representation='covariance',dtype='complex128',
        device='cuda:0',backend='local',covariance_spectral_clip=False,
        occupation_tolerance=1e-9,entropy_eps=1e-12,initial_purity_tolerance=.50000001)


def validate_config(c):
    base = default_config()
    if set(c) != set(base):
        raise ValueError('Unexpected or missing configuration fields')
    # Expose the complete contract but forbid accidentally changing its physics.
    for key in set(base)-{'sampling_revision','root_seed','sizes','cycles','segment_cycles','device'}:
        if c[key] != base[key]:
            raise ValueError(f'Unsupported scientific change: {key}')
    if not c['sizes'] or len(set(c['sizes'])) != len(c['sizes']):
        raise ValueError('Sizes must be nonempty and unique')
    if any(type(L) is not int or L<4 for L in c['sizes']):
        raise ValueError('Invalid square sizes')
    if type(c['cycles']) is not int or c['cycles']<1 or type(c['segment_cycles']) is not int or c['segment_cycles']<1:
        raise ValueError('Positive integer cycles required')


def identity(c,L):
    # Selection of the other square does not affect this task's identity/seed.
    scientific = dict(c); scientific.pop('sizes')
    seed = int.from_bytes(hashlib.sha256(f"{c['root_seed']}|L{L}|sample0".encode()).digest()[:4],'little')
    return dict(schema='square_contour_v1',task=f'L{L:03d}_sample000',seed=seed,L=L,
        config=scientific,config_sha256=hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest(),
        source_hashes={name:sha(HERE/name) for name in SOURCE_FILES},
        canonical_dynamics_entry_point='classA_U1FGTN_gpu.run_markov_circuit')


def paths(root,L,kind):
    folder=Path(root)/f'L{L:03d}_sample000'
    return folder/f'{kind}.npz',folder/f'{kind}.json'


def load_pair(root,L,kind,ident):
    path,receipt=paths(root,L,kind)
    if not pair_verified(path,receipt,dict(ident,kind=kind)):
        return None
    try:
        with np.load(path,allow_pickle=False) as z:
            p={k:z[k] for k in z.files}
        t=int(p['completed_cycle']); T=ident['config']['cycles']
        if not 0<=t<=T or (kind=='result' and t!=T): return None
        expected=allocate(L,T)
        for key,array in expected.items():
            if p[key].shape!=array.shape: return None
            if key not in ('modular_gap','lyapunov_gap') and not np.isfinite(p[key][:t+1]).all(): return None
        np.testing.assert_allclose(p['entropy_contour'][:t+1].sum((1,2)),p['total_entropy'][:t+1],atol=1e-9)
        if kind=='checkpoint':
            if p['G'].shape!=(1,2*L*L,2*L*L) or p['G'].dtype!=np.complex128 or not np.isfinite(p['G']).all(): return None
            for key in ('rng_np_algorithm','rng_np_state','rng_np_position','rng_np_has_gauss','rng_np_gauss','rng_torch','rng_cuda_count'):
                if key not in p: return None
            if any(f'rng_cuda_{i}' not in p for i in range(int(p['rng_cuda_count']))): return None
        return p
    except (OSError,ValueError,KeyError,AssertionError):
        return None


def build_model(c,L):
    return classA_U1FGTN_gpu(Nx=L,Ny=L,DW=c['DW'],nshell=c['nshell'],
        filling_frac=c['filling_fraction'],alpha_1=c['alpha_1'],alpha_2=c['alpha_2'],
        trial_orbitals=c['trial_orbitals'],dw_truncation=c['dw_truncation'],
        triv_region_local_mode=c['triv_region_local_mode'],device=c['device'],
        dtype=c['dtype'],backend=c['backend'])


def run_segment(model,c,L,G,completed,count,rng,arrays,bar=None):
    def observe(cycle,G,**unused):
        if completed and cycle==0: return
        t=completed+cycle
        values=observe_covariance(G,L,t,c['occupation_tolerance'],c['entropy_eps'])
        for key,value in values.items(): arrays[key][t]=value
        if cycle and bar is not None:
            bar.update(1);bar.set_postfix(entropy=f"{float(values['total_entropy']):.4g}")
    if rng is not None: restore_rng(rng)
    result=model.run_markov_circuit(G_history=False,progress=False,cycles=count,samples=1,
        init_mode=c['init_mode'],G_init=G,G_init_prepared=bool(completed),
        perfect_correction=c['perfect_correction'],postselect=c['postselect'],
        save=False,save_init=False,n_a=c['n_a'],sequence=c['sequence'],
        meas_slab_only=c['meas_slab_only'],batch_size=1,return_data=True,
        state_representation=c['state_representation'],initial_purity_tolerance=c['initial_purity_tolerance'],
        cycle_observer=observe)
    if result.get('exterior_preparation_performed'):
        raise RuntimeError('Full-measurement global-maxmix run must not prepare exterior')
    if result['state_representation_resolved']!='covariance':
        raise RuntimeError('Unexpected engine state representation')
    return np.asarray(result['G_final']),capture_rng()


def cleanup(root,L,ident):
    if load_pair(root,L,'result',ident) is None:
        raise RuntimeError('Refuse checkpoint cleanup before result verification')
    for path in paths(root,L,'checkpoint'): path.unlink(missing_ok=True)


def run_task(c,L,root,scratch,max_segments=None):
    ident=identity(c,L)
    if load_pair(root,L,'result',ident) is not None:
        cleanup(root,L,ident);print(f'[skip] L={L}: verified complete',flush=True);return True
    checkpoint=load_pair(root,L,'checkpoint',ident)
    if checkpoint is None and any(p.exists() for p in paths(root,L,'checkpoint')):
        print(f'[warning] L={L}: invalid/partial checkpoint; deterministic restart',flush=True)
    t=0 if checkpoint is None else int(checkpoint['completed_cycle'])
    arrays=allocate(L,c['cycles'])
    if checkpoint is not None:
        arrays={key:checkpoint[key] for key in arrays}
    G=None if checkpoint is None else checkpoint['G']
    rng=checkpoint
    elapsed=0. if checkpoint is None else float(checkpoint['elapsed_seconds'])
    model=build_model(c,L)
    print(f'[task] L={L}, walls={model.DW_loc}, seed={ident["seed"]}, resume={t}/{c["cycles"]}',flush=True)
    if checkpoint is None:
        np.random.seed(ident['seed']);torch.manual_seed(ident['seed'])
        if torch.cuda.is_available():torch.cuda.manual_seed_all(ident['seed'])
    segments=0
    with tqdm(total=c['cycles'],initial=t,desc=f'L={L} dynamics + entropy contour',unit='cycle') as bar:
        while t<c['cycles']:
            if max_segments is not None and segments>=max_segments:return False
            begin=time.perf_counter();count=min(c['segment_cycles'],c['cycles']-t)
            G,rng=run_segment(model,c,L,G,t,count,rng,arrays,bar)
            elapsed+=time.perf_counter()-begin;t+=count;segments+=1
            payload=dict(arrays,G=G,completed_cycle=np.array(t),elapsed_seconds=np.array(elapsed),**rng)
            path,receipt=paths(root,L,'checkpoint')
            publish_pair(payload,path,receipt,dict(ident,kind='checkpoint'),Path(scratch)/f'L{L}',compressed=False)
            print(f'[checkpoint] L={L}: verified {t}/{c["cycles"]}; compute {elapsed:.1f}s',flush=True)
    payload=dict(arrays,cycles=np.arange(c['cycles']+1),sample_indices=np.array([0]),
        Nx=np.array(L),Ny=np.array(L),walls=np.array(model.DW_loc),
        completed_cycle=np.array(t),elapsed_seconds=np.array(elapsed),
        configuration_json=np.array(json.dumps(c,sort_keys=True)))
    path,receipt=paths(root,L,'result')
    publish_pair(payload,path,receipt,dict(ident,kind='result'),Path(scratch)/f'L{L}',compressed=True)
    cleanup(root,L,ident)
    print(f'[complete] L={L}; compact result verified; rolling checkpoint removed',flush=True)
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--scratch-root',required=True)
    p.add_argument('--report-only',action='store_true');p.add_argument('--max-new-tasks',type=int)
    args=p.parse_args();c=json.loads(Path(args.config).read_text());validate_config(c)
    inventory=[]
    for L in c['sizes']:
        ident=identity(c,L);checkpoint=load_pair(args.output_root,L,'checkpoint',ident)
        inventory.append(dict(L=L,complete=load_pair(args.output_root,L,'result',ident) is not None,
            recoverable_cycle=None if checkpoint is None else int(checkpoint['completed_cycle'])))
    print(json.dumps(dict(config=c,source_root=str(HERE),source_hashes=identity(c,c['sizes'][0])['source_hashes'],
        output_root=args.output_root,scratch_root=args.scratch_root,inventory=inventory,
        workload_sample_cycles=len(c['sizes'])*c['cycles']),indent=2),flush=True)
    if args.report_only:return
    if not torch.cuda.is_available() or not str(c['device']).startswith('cuda'):
        raise RuntimeError('Select an A100 40-GB-class Colab runtime')
    device=torch.device(c['device']);props=torch.cuda.get_device_properties(device)
    print(f'[device] {props.name}; {props.total_memory/2**30:.2f} GiB; complex128',flush=True)
    if 'A100' not in props.name or props.total_memory<38_000_000_000:
        raise RuntimeError('Select an A100 40-GB-class Colab runtime')
    for folder in (args.output_root,args.scratch_root):
        Path(folder).mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(folder).free<2*2**30:raise OSError(f'Need at least 2 GiB free: {folder}')
    new=0;done=sum(row['complete'] for row in inventory)
    with tqdm(total=len(inventory),initial=done,desc='Verified trajectories',unit='task') as outer:
        for row in inventory:
            if row['complete']:continue
            if args.max_new_tasks is not None and new>=args.max_new_tasks:break
            run_task(c,row['L'],args.output_root,args.scratch_root);new+=1;done+=1;outer.update(1)
    print(f'[summary] {done}/{len(inventory)} verified trajectories; {new} completed this session',flush=True)


if __name__=='__main__':main()
